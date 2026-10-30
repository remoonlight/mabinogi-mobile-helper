"""User-designated ingredient menu interrupts gathering before other plans."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import time

import pytest

from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.action_finish import StpSample
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem
from mm_assist.memory.readers import ui_stack


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()
    bar_ui.reset_recovery_for_tests()
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda _: True)
    monkeypatch.setattr('mm_assist.safety.input.foreground_input_scope', lambda _, **kwargs: nullcontext())
    monkeypatch.setattr('mm_assist.life_plan.action_finish.read_completion_stp',
                        lambda: StpSample(0, time.time(), 123, True))
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: '')
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',
                        lambda *a, **k: pytest.fail('repair menu must bypass recovery'))
    yield
    controller.reset_controller_for_tests()
    bar_ui.reset_recovery_for_tests()


def start(monkeypatch, *, mode='skill', existing=False, tracking=False, kind='gather'):
    current = PlanItem('p1', kind, '採集 蜘蛛網', skill='日常採集',
                       detail='蜘蛛網', gather_mode=mode)
    tail = [replace(current, id='p2'), PlanItem('p3', 'feed', '投料 木材',
                                             facility='wood', recipe='木材')]
    if existing:
        tail.append(PlanItem('p4', 'repair', '一键修理全部工具'))
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [current, *tail])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000, pause_on_failure=tracking)
    return current, tail


def stack(monkeypatch, *, names=('IngredientGuideUIInstance',), ready=True, age=0, read=None):
    def sample(**kwargs):
        assert kwargs == {'force': True}
        if read:
            read()
        return ui_stack.UiStackSnap(names=list(names), ready=ready, sampled_ts=time.time()-age)
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: SimpleNamespace(read=sample))


@pytest.mark.parametrize('mode', ['skill', 'process_mat'])
@pytest.mark.parametrize('pending', [False, True])
@pytest.mark.parametrize('existing', [False, True])
def test_menu_preempts_gather_and_runs_unique_repair_first(monkeypatch, mode, pending, existing):
    item, tail = start(monkeypatch, mode=mode, existing=existing)
    if pending:
        controller.mark_gather_pending(item.id, 1000, saw_busy=True)
        controller.wait_action_completion(item.id, 1000,
            ExecuteResult(False, 'action_pending', action_issued=True), actions_complete=True)
    stack(monkeypatch)
    executed = []
    def execute(p, **kwargs):
        executed.append(p.kind)
        return ExecuteResult(True, 'done')
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    win = SimpleNamespace(hwnd=123)
    bar_ui.pump_life_once(None, win)
    st = controller.get_controller()
    assert st.current.kind == 'repair' and executed == []
    assert len([p for p in st.queue if p.kind == 'repair']) == 1
    assert st.pending_action is None and not st.gather_pending_id
    assert st.training_counts == {}  # Interrupted collection is not success.
    assert [(p.kind, p.title) for p in st.queue[1:]] == [(p.kind, p.title) for p in tail if p.kind != 'repair']
    assert [int(p.id[1:]) for p in st.queue] == sorted(int(p.id[1:]) for p in st.queue)
    bar_ui.pump_life_once(None, win)
    assert executed == ['repair']
    assert controller.get_controller().current.title == tail[0].title


def test_menu_appearing_inside_click_chain_stops_before_next_input(monkeypatch):
    from mm_assist.life_execute import ui_act
    from mm_assist.safety import autofarm_gates
    start(monkeypatch)
    visible = [False]
    stack(monkeypatch, read=lambda: None)
    reader = SimpleNamespace(read=lambda **k: ui_stack.UiStackSnap(
        names=['IngredientGuideUIInstance'] if visible[0] else [], ready=True, sampled_ts=time.time()))
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: reader)
    clicks = []
    def execute(item, **kwargs):
        visible[0] = True
        return ui_act.click_spot(lambda *xy: clicks.append(xy) or True,
            (10, 10), '采集入口', wait_refresh=False) or ExecuteResult(True, 'done')
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    autofarm_gates._stage_counts['另一步'] = 2
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    assert clicks == [] and controller.get_controller().current.kind == 'repair'
    assert autofarm_gates._stage_counts['另一步'] == 2
    assert not ui_act.pump_aborted()  # Scope does not cancel repair or other threads.


def test_menu_appearing_during_real_navigation_blocks_its_click(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import ui_nav
    from mm_assist.life_execute.ui_logic import UiWhere
    start(monkeypatch)
    visible = [False]
    reader = SimpleNamespace(read=lambda **k: ui_stack.UiStackSnap(
        names=['IngredientGuideUIInstance'] if visible[0] else [], ready=True, sampled_ts=time.time()))
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: reader)
    monkeypatch.setattr(ui_nav, 'locate_edge', lambda *a, **k: (10, 10))
    monkeypatch.setattr(ui_nav, '_hud_xy', lambda *a, **k: (10, 10))
    clicks = []
    def resolve(frame):
        visible[0] = True
        return UiWhere('main_hud', 0, 0, 0, 'fixture', stack_ready=True)
    def execute(item, **kwargs):
        return ui_nav.ensure_node('system_menu', grab=lambda: np.zeros((100, 100, 3), dtype=np.uint8),
            resolve=resolve, click=lambda *xy: clicks.append(xy) or True, sleep_fn=lambda _: None)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    assert clicks == [] and controller.get_controller().current.kind == 'repair'


def test_menu_after_primary_input_is_not_counted_done_or_replayed(monkeypatch):
    start(monkeypatch)
    visible = [False]
    reader = SimpleNamespace(read=lambda **k: ui_stack.UiStackSnap(
        names=['IngredientGuideUIInstance'] if visible[0] else [], ready=True, sampled_ts=time.time()))
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: reader)
    calls = []
    def execute(item, **kwargs):
        calls.append(item.kind)
        kwargs['on_action_issued']()
        visible[0] = True
        return ExecuteResult(False, 'action_pending', action_issued=True)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    st = controller.get_controller()
    assert calls == ['gather'] and st.current.kind == 'repair'
    assert st.pending_action is None and st.training_counts == {}


@pytest.mark.parametrize('interruption', ['pause', 'start'])
def test_interruption_while_saving_evidence_keeps_queue(monkeypatch, interruption):
    item, tail = start(monkeypatch)
    stack(monkeypatch)
    def shot(*a, **k):
        if interruption == 'pause':
            controller.pause_life()
        else:
            controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000)
        return ''
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', shot)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda *a, **k: pytest.fail('no input'))
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    assert controller.get_controller().queue == [item, *tail]


@pytest.mark.parametrize('names,ready,age', [
    (('IngredientGuideUIInstance',), False, 0),
    (('IngredientGuideUIInstance',), True, 100),
    (('IngredientGuideUIInstance',), True, -100),
    (('IngredientGuideUIInstance',), True, float('nan')),
    (('OtherIngredientGuideUIInstance',), True, 0),
    ((), True, 0),
])
def test_only_exact_fresh_ready_menu_triggers(monkeypatch, names, ready, age):
    item, _ = start(monkeypatch)
    stack(monkeypatch, names=names, ready=ready, age=age)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
                        lambda *a, **k: ExecuteResult(False, 'stp_busy'))
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    assert controller.get_controller().current == item
    assert not any(p.kind == 'repair' for p in controller.get_controller().queue)


@pytest.mark.parametrize('interruption', ['pause', 'start', 'exception'])
def test_late_or_failed_observation_cannot_insert_repair(monkeypatch, interruption):
    item, _ = start(monkeypatch)
    def read():
        if interruption == 'pause':
            controller.pause_life()
        elif interruption == 'start':
            controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000)
        else:
            raise RuntimeError('read failed')
    stack(monkeypatch, read=read)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
                        lambda *a, **k: ExecuteResult(False, 'stp_busy'))
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    assert controller.get_controller().current == item
    assert not any(p.kind == 'repair' for p in controller.get_controller().queue)


def test_tracking_preserves_interrupted_gather_for_review(monkeypatch):
    item, _ = start(monkeypatch, tracking=True)
    stack(monkeypatch)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda *a, **k: pytest.fail('no input'))
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    st = controller.get_controller()
    assert st.paused and st.current == item and not any(p.kind == 'repair' for p in st.queue)


def test_menu_outside_gather_does_not_request_repair(monkeypatch):
    start(monkeypatch, kind='feed')
    stack(monkeypatch, read=lambda: pytest.fail('non-gather must not check repair menu'))
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda *a, **k: ExecuteResult(True, 'done'))
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    assert not any(p.kind == 'repair' for p in controller.get_controller().queue)
