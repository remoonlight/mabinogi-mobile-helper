"""A short idle during travel is not an arrival or a failed claim."""
import copy
from types import SimpleNamespace

import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import medicine_process as process, ui_act
from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.action_finish import StpSample
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem
from mm_assist.memory.readers.facility_ui import select_facility_view
from mm_assist.safety import autofarm_gates as gates
from test_facility_memory import sample


@pytest.fixture
def travel(monkeypatch):
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()
    bar_ui.reset_recovery_for_tests()
    gates.reset_stage_gates_for_tests()
    away = select_facility_view(sample()['render'], 0)
    image = cv2.imread(str(ROOT/'tests/fixtures/facility_memory/travel_pending_20261021.png'))
    state = dict(travel=False, idle_seen=False, reads=0, clock=1000., moving=1)
    clicks = []
    def stp():
        if not state['travel']:return 0
        state['reads'] += 1
        if state['reads'] == 1:return 1
        if state['reads'] == 2:
            state['idle_seen'] = True
            return 0
        return state['moving']
    def click(*xy):
        clicks.append(xy)
        assert xy == (223,305), 'travel can never authorize claim'
        state['travel'] = True
        return True
    monkeypatch.setattr(process.time, 'monotonic', lambda:state['clock'])
    monkeypatch.setattr(process.time, 'time', lambda:state['clock'])
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda:False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_recipe', lambda:False)
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **_:False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda:False)
    monkeypatch.setattr(ui_act, 'confirm_go_facility_popup', lambda *a, **kw:None)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot',lambda *a, **kw:'offline-fixture')
    def run():
        return process._run_process(None, facility='metal', recipe='', mode='claim', items={},
            do_click=click, read_items=None,
            grab=lambda:(state.get('arrival_image',image) if state['idle_seen'] else image).copy(), on_progress=None,
            root=None, sleep_fn=lambda n:state.__setitem__('clock',state['clock']+n),
            read_stp=stp, defer_completion=True,
            memory_reader=lambda:None if state['idle_seen'] else copy.deepcopy(away))
    yield state, clicks, run
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()
    bar_ui.reset_recovery_for_tests()
    gates.reset_stage_gates_for_tests()


def test_renewed_movement_after_idle_candidate_is_readonly_wait(travel):
    state, clicks, run = travel
    result = run()
    assert result.status == 'stp_busy' and not result.ok and not result.action_issued
    assert clicks == [(223,305)] and state['clock'] <= 1002.5


@pytest.mark.parametrize('value', [0, None])
def test_unknown_or_idle_is_not_used_as_busy_permission(travel, value):
    state, clicks, run = travel
    state['moving'] = value
    result = run()
    assert result.status == 'facility_memory_pending' and not result.action_issued
    assert clicks == [(223,305)]


@pytest.mark.parametrize('value', [0, None])
@pytest.mark.parametrize('stack', ['ready_hud', 'not_ready', 'unknown_cover'])
def test_real_loading_title_after_travel_waits_without_claim_or_repeating_go(travel, monkeypatch, value, stack):
    import time
    import numpy as np
    from mm_assist.life_execute import ui_logic
    from mm_assist.memory.readers.ui_stack import UiStackSnap

    state, clicks, run = travel
    image = cv2.imdecode(np.fromfile(ROOT/'tests/fixtures/life_faults_20261024/loading_to_dugald_claim.png',
                                    dtype=np.uint8), 1)
    assert image is not None
    names = ['HudUIStepState', 'TutorialTooltipUIState']
    if stack == 'unknown_cover': names.append('UnknownPopup')
    reader = SimpleNamespace(read=lambda **_: UiStackSnap(ready=stack != 'not_ready',
        names=names, sampled_ts=time.time()))
    native = ui_logic.resolve_navigation
    resolve = lambda frame: native(frame, reader=reader)
    if stack == 'ready_hud': assert resolve(image).node == 'overlay'
    monkeypatch.setattr(ui_logic, 'resolve_navigation', resolve)
    state.update(arrival_image=image, moving=value)
    result = run()
    expected = 'input_layer_wait' if stack == 'ready_hud' else 'facility_memory_pending'
    assert result.status == expected and not result.ok and not result.action_issued
    assert clicks == [(223,305)] and state['clock'] <= 1002.5


@pytest.mark.parametrize('value,fresh,expected', [(1,True,True),(1,False,False),
                                               (0,True,False),(None,True,False)])
def test_production_busy_classification_requires_current_watch_sample(monkeypatch,value,fresh,expected):
    monkeypatch.setattr('mm_assist.life_plan.action_finish.read_completion_stp',
        lambda:StpSample(value,1000,1,fresh))
    assert process._travel_still_busy() is expected


def test_real_claim_result_keeps_debug_plan_and_waits_for_newer_idle(travel, monkeypatch):
    state, clicks, run = travel
    item = PlanItem('p1','claim','领取 金屬',facility='metal')
    monkeypatch.setattr(controller,'plan_life',lambda *a, **kw:[item])
    controller.start_life(snapshot=LifeSnapshot(ok=True),now=1000,pause_on_failure=True)
    observations = [StpSample(0,1000,1,True)]
    calls = []
    def execute(*a, **kw):
        calls.append(True)
        result = run()
        observations[0] = StpSample(1,state['clock'],1,True)
        return result
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',execute)
    monkeypatch.setattr('mm_assist.life_plan.action_finish.read_completion_stp',lambda:observations[0])
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=0))
    current = controller.get_controller()
    assert current.running and not current.paused and current.current == item
    assert current.pending_action is None and clicks == [(223,305)]
    for obs in [StpSample(1,1003,1,True),StpSample(None,1004,1,False),
                StpSample(0,1000,1,True)]:
        observations[0] = obs
        bar_ui.pump_life_once(None,SimpleNamespace(hwnd=0))
    assert len(calls)==1 and clicks == [(223,305)]
    observations[0] = StpSample(0,1005,1,True)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
        lambda *a, **kw:ExecuteResult(False,'stock_read_wait','arrived re-observation; no input'))
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=0))
    assert controller.get_controller().running and not controller.get_controller().paused
    assert clicks == [(223,305)]


def test_claim_loading_wait_returns_to_real_page_gate_without_reexecuting_travel(travel, monkeypatch):
    import time
    import numpy as np
    from mm_assist.life_execute import ui_logic
    from mm_assist.memory.readers.ui_stack import UiStackSnap
    state, clicks, run = travel
    image = cv2.imdecode(np.fromfile(ROOT/'tests/fixtures/life_faults_20261024/loading_to_dugald_claim.png',
                                    dtype=np.uint8), 1)
    state.update(arrival_image=image, moving=0)
    reader = SimpleNamespace(read=lambda **_: UiStackSnap(ready=True,
        names=['HudUIStepState', 'TutorialTooltipUIState'], sampled_ts=time.time()))
    native_resolve = ui_logic.resolve_navigation
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda frame: native_resolve(frame, reader=reader))
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda _: image)
    item = PlanItem('p1','claim','领取 金屬',facility='metal')
    monkeypatch.setattr(controller,'plan_life',lambda *a, **kw:[item])
    controller.start_life(snapshot=LifeSnapshot(ok=True),now=1000,pause_on_failure=True)
    monkeypatch.setattr('mm_assist.life_plan.action_finish.read_completion_stp',
        lambda: StpSample(0,state['clock'],1,True))
    real_gate = bar_ui._execution_page_ready
    monkeypatch.setattr(bar_ui, '_execution_page_ready',
        lambda _: real_gate(SimpleNamespace(hwnd=7)) if state['travel'] else True)
    calls = []
    def execute(*a, **kw):
        calls.append(True)
        return run()
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',
        lambda *a, **kw: pytest.fail('loading cannot authorize recovery'))
    for _ in range(3):
        bar_ui.pump_life_once(None,SimpleNamespace(hwnd=0))
        state['clock'] += .2
        current = controller.get_controller()
        assert current.running and not current.paused and current.current == item
        assert current.pending_action is None
    assert calls == [True] and clicks == [(223,305)]


@pytest.mark.parametrize('interruption', ['pause', 'new_start', 'exception'])
def test_late_loading_observation_cannot_advance_or_replace_the_current_run(travel, monkeypatch, interruption):
    from mm_assist.life_execute import ui_logic
    state, clicks, run = travel
    state['moving'] = 0
    item = PlanItem('p1','claim','领取 金屬',facility='metal')
    monkeypatch.setattr(controller,'plan_life',lambda *a, **kw:[item])
    controller.start_life(snapshot=LifeSnapshot(ok=True),now=1000,pause_on_failure=True)
    monkeypatch.setattr('mm_assist.life_plan.action_finish.read_completion_stp',
        lambda: StpSample(0,state['clock'],1,True))
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda _: True)
    def resolve(_):
        if interruption == 'pause': controller.pause_life()
        elif interruption == 'new_start':
            controller.start_life(snapshot=LifeSnapshot(ok=True),now=2000,pause_on_failure=True)
        else: raise RuntimeError('offline stack read failed')
        return ui_logic.UiWhere('overlay',1,0,0,'fixture',stack_ready=True,
            stack_names=('HudUIStepState','TutorialTooltipUIState'))
    monkeypatch.setattr(ui_logic,'resolve_navigation',resolve)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',lambda *a, **kw:run())
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=0))
    current = controller.get_controller()
    assert current.current == item and current.queue == [item] and current.pending_action is None
    assert current.life_started_at == (2000 if interruption == 'new_start' else 1000)
    assert current.running if interruption == 'new_start' else current.paused
    assert clicks == [(223,305)]
