"""Recovery must observe movement and each dispatched Esc before advancing."""
import time
from types import SimpleNamespace

import pytest

from mm_assist.life_execute import ui_logic, ui_recover
from mm_assist.life_plan import action_finish, bar_ui, controller
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem

DIRECT_RESOLVER = ui_logic.resolve_navigation


@pytest.fixture
def scene(monkeypatch):
    controller.reset_controller_for_tests()
    queue = [PlanItem('p1', 'repair', '修理'), PlanItem('p2', 'repair', '下一项')]
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: list(queue))
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    events = []
    state = dict(node='inventory', stp=0, ready=True, stamp=None)
    win = SimpleNamespace(hwnd=7)
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda _: True)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: '')
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr('mm_assist.safety.input.check_abort', lambda: None)
    monkeypatch.setattr('mm_assist.life_plan.live_test.inspect_nav_watch_ready', lambda **k: ({}, ''))
    monkeypatch.setattr('mm_assist.life_execute.ui_nav._hud_xy', lambda *a, **k: (10, 20))
    def grab(_):
        events.append('frame')
        return object()
    def resolve(_):
        events.append('recognize')
        # The actual resolve_navigation contract deliberately leaves stp unset.
        return ui_logic.UiWhere(state['node'], 0, 0, None, 'direct',
            stack_ready=state['ready'], stack_sampled_ts=state['stamp'] or time.time())
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', grab)
    monkeypatch.setattr(ui_logic, 'resolve_navigation', resolve)
    monkeypatch.setattr(action_finish, 'read_completion_stp', lambda: action_finish.StpSample(
        state['stp'], state['stamp'] or time.time(), 7, state['ready']))
    monkeypatch.setattr('mm_assist.safety.input.press_escape', lambda **_: events.append('esc'))
    monkeypatch.setattr(ui_recover.time, 'sleep', lambda sec: events.append(sec))
    yield state, events, queue, win
    controller.reset_controller_for_tests()


def test_real_direct_page_with_unset_stp_can_dispatch_and_verify_esc(scene, monkeypatch):
    state, events, _, win = scene
    def press(**_):
        events.append('esc')
        state['node'] = 'main_hud'
    monkeypatch.setattr('mm_assist.safety.input.press_escape', press)
    result = ui_recover.recover_once('wrong_page', win=win)
    assert result['ok'] and result['esc_sent']
    assert events.count('esc') == 1 and events.count('recognize') == 2
    assert sum(v for v in events if isinstance(v, float)) == pytest.approx(2)


def test_actual_navigation_resolver_and_independent_action_reader_work_together(scene, monkeypatch):
    state, events, _, win = scene
    def read(**_):
        return SimpleNamespace(ready=True, names=('EquipmentInventoryUIInstance',),
            empty_confirmed=False, craft_category=None, sampled_ts=time.time())
    reader = SimpleNamespace(read=read)
    monkeypatch.setattr('mm_assist.memory.readers.ui_stack.shared_ui_stack_reader', lambda: reader)
    # Use the actual immediate resolver, rather than an idealized UiWhere stp=0.
    monkeypatch.setattr(ui_logic, 'resolve_navigation', DIRECT_RESOLVER)
    monkeypatch.setattr('mm_assist.life_plan.node_inspect.compute_inspect',
                        lambda *a, **k: {'node': 'inventory'})
    result = ui_recover.recover_once('wrong_page', win=win)
    assert result['esc_sent'] and result['status'] == 'escape_unverified'
    assert not result['waiting'] and events.count('esc') == 1


def test_final_esc_cannot_skip_on_hud_without_independent_menu(scene, monkeypatch):
    state, events, queue, win = scene
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
        lambda *a, **k: events.append('execute') or ExecuteResult(False, 'wrong_page'))
    visible = [False]
    monkeypatch.setattr('mm_assist.life_execute.ui_nav._hud_xy',
                        lambda *a, **k: (10, 20) if visible[0] else None)
    def press(**_):
        events.append('esc')
        if events.count('esc') == 2:
            state['node'] = 'main_hud'
    monkeypatch.setattr('mm_assist.safety.input.press_escape', press)
    bar_ui.pump_life_once(None, win)
    assert events.count('esc') == 2 and controller.get_controller().current == queue[0]
    bar_ui.pump_life_once(None, win)
    assert events.count('esc') == 2 and events.count('execute') == 1
    visible[0] = True
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
                        lambda *a, **k: ExecuteResult(True, 'done'))
    bar_ui.pump_life_once(None, win)
    assert events.count('esc') == 2 and controller.get_controller().current == queue[1]


def test_movement_does_not_show_recovery_or_skip_current_plan(scene, monkeypatch):
    state, events, queue, win = scene
    state['stp'] = 1
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
                        lambda *a, **k: pytest.fail('movement accepts no new action'))
    for _ in range(3):
        bar_ui.pump_life_once(None, win)
        assert controller.get_controller().current == queue[0]
        assert '脱困' not in controller.get_controller().current_action
    assert 'esc' not in events and bar_ui._RECOVERY_COUNT == 0


@pytest.mark.parametrize('unready', ['busy', 'unknown', 'stale'])
def test_unissued_wait_does_not_consume_esc_budget_or_replay_executor(scene, monkeypatch, unready):
    state, events, queue, win = scene
    def execute(*a, **k):
        events.append('execute')
        if unready == 'busy':
            state['stp'] = 1
        elif unready == 'unknown':
            state['node'] = 'unknown'
        else:
            state['stamp'] = 1
        return ExecuteResult(False, 'wrong_page')
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    for _ in range(3):
        bar_ui.pump_life_once(None, win)
        assert controller.get_controller().current == queue[0]
        assert bar_ui._RECOVERY_COUNT == 0
    assert events.count('execute') == 1 and 'esc' not in events


def test_two_actual_escs_each_wait_and_recognize_before_skip(scene, monkeypatch):
    state, events, queue, win = scene
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
        lambda *a, **k: events.append('execute') or ExecuteResult(False, 'wrong_page'))
    bar_ui.pump_life_once(None, win)
    assert events.count('esc') == 2 and events.count('execute') == 1
    presses = [i for i, e in enumerate(events) if e == 'esc']
    for start, end in zip(presses, presses[1:] + [len(events)]):
        part = events[start + 1:end]
        assert sum(v for v in part if isinstance(v, float)) == pytest.approx(2)
        assert 'recognize' in part
    assert controller.get_controller().current == queue[1]


@pytest.mark.parametrize('interrupt', ['stale', 'pause', 'exception', 'new_start'])
def test_post_input_verification_survives_wait_pause_and_exception(scene, monkeypatch, interrupt):
    state, events, queue, win = scene
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
        lambda *a, **k: events.append('execute') or ExecuteResult(False, 'wrong_page'))
    def sleep(sec):
        events.append(sec)
        state['stamp'] = 1
        if interrupt == 'pause':
            controller.pause_life()
        elif interrupt == 'new_start':
            controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000)
        elif interrupt == 'exception':
            raise RuntimeError('post-input observation failed')
    monkeypatch.setattr(ui_recover.time, 'sleep', sleep)
    bar_ui.pump_life_once(None, win)
    assert events.count('esc') == 1
    assert controller.get_controller().current == queue[0]
    if interrupt == 'new_start':
        assert controller.get_controller().life_started_at == 2000
        return
    if interrupt == 'pause':
        controller.resume_life()
    # Still stale: no second Esc, no executor replay, no skip.
    monkeypatch.setattr(ui_recover.time, 'sleep', lambda sec: events.append(sec))
    bar_ui.pump_life_once(None, win)
    assert events.count('esc') == 1 and events.count('execute') == 1
    state['stamp'] = None
    bar_ui.pump_life_once(None, win)
    assert events.count('esc') == 2 and controller.get_controller().current == queue[1]
