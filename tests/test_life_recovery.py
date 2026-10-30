"""正常备料恢复同一计划两次；旧代、已发动作及人工接管不重发。"""
import pytest

from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem


def setup(monkeypatch, *, tracking=False):
    controller.reset_controller_for_tests()
    monkeypatch.setattr(bar_ui, '_RECOVERY_KEY', None)
    monkeypatch.setattr(bar_ui, '_RECOVERY_COUNT', 0)
    queue = [PlanItem('p1', 'repair', '一键修理'), PlanItem('p2', 'repair', '下一项')]
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: list(queue))
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda win: True)
    st = controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000, pause_on_failure=tracking)
    return st, queue


@pytest.mark.parametrize('recover_ok,expected_exec', [(True, 3), (False, 1)])
def test_two_recoveries_then_skip_same_plan(monkeypatch, recover_ok, expected_exec):
    _, queue = setup(monkeypatch)
    events = []
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: events.append('shot') or 'first.png')
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', lambda *a, **k: events.append('recover') or {'ok': recover_ok})
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda item, **k: events.append(item.id) or ExecuteResult(False, 'continent_atlas_miss', '原始原因'))
    bar_ui.pump_life_once(None, None)
    assert events.count('recover') == 2
    assert events.count('p1') == expected_exec and 'p2' not in events
    assert events.index('shot') < events.index('recover')
    assert controller.get_controller().current == queue[1]


def test_recovery_success_continues_original_plan(monkeypatch):
    _, queue = setup(monkeypatch)
    calls = []
    results = iter([ExecuteResult(False, 'nav_verify_timeout'), ExecuteResult(True, 'done')])
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda item, **k: calls.append(item.id) or next(results))
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', lambda *a, **k: calls.append('recover') or {'ok': True})
    bar_ui.pump_life_once(None, None)
    assert calls == ['p1', 'recover', 'p1']
    assert controller.get_controller().current == queue[1]


@pytest.mark.parametrize('event', ['pause', 'new_start', 'exception'])
def test_recovery_interruption_and_exception_are_bounded(monkeypatch, event):
    st, queue = setup(monkeypatch)
    calls = []
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda item, **k: calls.append(item.id) or ExecuteResult(False, 'nav_verify_timeout'))
    def recover(*a, **k):
        calls.append('recover')
        if event == 'pause':
            controller.pause_life()
        elif event == 'new_start':
            controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000)
        else:
            raise RuntimeError('recovery failure')
        return {'ok': True}
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', recover)
    bar_ui.pump_life_once(None, None)
    assert calls == (['p1', 'recover', 'recover'] if event == 'exception' else ['p1', 'recover'])
    latest = controller.get_controller()
    if event == 'pause':
        assert latest.paused and latest.current == queue[0]
    elif event == 'new_start':
        assert latest.life_started_at == 2000 and latest.current == queue[0]
    else:
        assert latest.current == queue[1]


@pytest.mark.parametrize('result,tracking', [
    (ExecuteResult(False, 'no_mats'), False),
    (ExecuteResult(False, 'error', 'click_focus_lost'), False),
    (ExecuteResult(False, 'error', 'click_pointer_moved'), False),
    (ExecuteResult(False, 'settle_fail', action_issued=True), False),
    (ExecuteResult(False, 'nav_timeout'), True),
])
def test_no_replay_after_dispatch_or_safety_yield(monkeypatch, result, tracking):
    setup(monkeypatch, tracking=tracking)
    calls = []
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda *a, **k: calls.append('execute') or result)
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', lambda *a, **k: pytest.fail('unsafe recovery'))
    bar_ui.pump_life_once(None, None)
    assert calls == ['execute']


@pytest.mark.parametrize('raises', [False, True])
def test_repair_confirmation_remains_issued_after_exit_failure(monkeypatch, raises):
    from mm_assist.life_execute.repair import execute_repair
    from mm_assist.config import AppConfig
    calls = []
    def dialog(cfg, win, **kwargs):
        kwargs['on_action_issued']()
        if raises:
            raise RuntimeError('告别采证异常')
        return win, {'ok': False, 'stage': 'leave_dialog', 'error': '告别未关'}
    monkeypatch.setattr('mm_assist.life_execute.repair_dialog.run_repair_dialog_chain', dialog)
    monkeypatch.setattr('mm_assist.life_execute.repair.one_click_repair',
                        lambda cfg, win, **k: k['dialog_fn'](cfg, win))
    result = execute_repair(AppConfig(), None, on_action_issued=lambda: calls.append('issued'))
    assert calls == ['issued']
    assert not result.ok and result.action_issued


@pytest.mark.parametrize('node,menu,ready,expected', [
    ('main_hud', (10, 20), True, True), ('main_hud', None, True, False),
    ('unknown', (10, 20), True, False), ('main_hud', (10, 20), False, False),
])
def test_production_recovery_needs_independent_fresh_hud(monkeypatch, node, menu, ready, expected):
    from types import SimpleNamespace
    from mm_assist.life_execute.ui_recover import recover_once
    from mm_assist.life_execute.ui_logic import UiWhere
    monkeypatch.setattr('mm_assist.life_execute.ui_act.prepare_to', lambda *a, **k: ExecuteResult(True, 'at_node'))
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda win: object())
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation',
        lambda frame: UiWhere(node, 0, 0, 0, 'fixture', stack_ready=ready,
                             stack_sampled_ts=__import__('time').time()))
    monkeypatch.setattr('mm_assist.life_execute.ui_nav._hud_xy', lambda *a, **k: menu)
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    assert recover_once('nav_timeout', win=SimpleNamespace(hwnd=7))['ok'] == expected


def test_pause_resume_keeps_two_recovery_budget(monkeypatch):
    _, queue = setup(monkeypatch)
    calls = []
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda *a, **k: ExecuteResult(False, 'nav_timeout'))
    def recover(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            controller.pause_life()
        return {'ok': False}
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', recover)
    bar_ui.pump_life_once(None, None)
    assert controller.get_controller().current == queue[0]
    controller.resume_life()
    bar_ui.pump_life_once(None, None)
    assert calls == [1, 1] and controller.get_controller().current == queue[1]


def test_dispatched_callback_survives_exception_without_recovery(monkeypatch):
    _, queue = setup(monkeypatch)
    def execute(*a, **k):
        k['on_action_issued']()
        raise RuntimeError('结果采证失败')
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',
        lambda *a, **k: pytest.fail('issued action must not be replayed'))
    bar_ui.pump_life_once(None, None)
    assert controller.get_controller().current == queue[1]


def test_recovery_focus_yield_does_not_try_again(monkeypatch):
    _, queue = setup(monkeypatch)
    calls = []
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
        lambda *a, **k: ExecuteResult(False, 'nav_timeout'))
    def recover(*a, **k):
        calls.append(1)
        raise RuntimeError('click_focus_lost')
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', recover)
    bar_ui.pump_life_once(None, None)
    assert calls == [1] and controller.get_controller().current == queue[1]


def test_missing_window_cannot_prove_recovery_from_cached_inspect(monkeypatch):
    from mm_assist.life_execute.ui_recover import recover_once
    assert not recover_once('nav_timeout')['ok']


def test_idle_stall_preserves_current_item_for_recovery(monkeypatch):
    _, queue = setup(monkeypatch)
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.ui_param_key', lambda: ('stp=0', 'main_hud'))
    monkeypatch.setattr(bar_ui, 'idle_stall_due', lambda **k: True)
    failure = bar_ui._maybe_idle_stall(None)
    assert failure.status == 'idle_stall'
    assert controller.get_controller().current == queue[0]


def test_initial_readiness_failure_uses_same_recovery_budget(monkeypatch):
    _, queue = setup(monkeypatch)
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda win: pytest.fail('failed gate must not repeat'))
    calls = []
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda *a, **k: pytest.fail('unverified page'))
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',
        lambda *a, **k: calls.append(1) or {'ok': False})
    bar_ui.pump_life_once(None, None, initial_result=ExecuteResult(False, 'exec_exception'))
    assert calls == [1, 1] and controller.get_controller().current == queue[1]


def test_outer_exception_from_old_generation_cannot_recover_new_plan(monkeypatch):
    setup(monkeypatch)
    calls = []
    def pump(*a, **k):
        calls.append(1)
        controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000)
        raise RuntimeError('旧代迟到异常')
    monkeypatch.setattr(bar_ui, 'pump_life_once', pump)
    monkeypatch.setattr(controller, 'tick_facility_progress', lambda: None)
    monkeypatch.setattr(bar_ui, '_resolve_life_win', lambda *a: None)
    monkeypatch.setattr(bar_ui, '_maybe_idle_stall', lambda win: None)
    monkeypatch.setattr(bar_ui, '_note_fail_and_continue', lambda **k: pytest.fail('new generation must survive'))
    monkeypatch.setattr(bar_ui.time, 'sleep', lambda _: controller.pause_life())
    bar_ui._pump_loop(None, None)
    assert calls == [1] and controller.get_controller().life_started_at == 2000
