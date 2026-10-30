"""Resident pointer dispatch with simulated windows; never sends desktop input."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mm_assist.platform.window import WindowInfo
from mm_assist.safety import input as inputs


@pytest.mark.parametrize('desktop,point,expected', [
    ((0, 0, 1920, 1080), (0, 0), (0, 0)),
    ((0, 0, 1920, 1080), (1919, 1079), (65535, 65535)),
    ((-1920, -1080, 3840, 2160), (-1920, -1080), (0, 0)),
    ((-1920, -1080, 3840, 2160), (1919, 1079), (65535, 65535)),
    ((-1920, 0, 3840, 1080), (-960, 540), (16388, 32798)),
])
def test_mouse_absolute_uses_entire_virtual_desktop(monkeypatch, desktop, point, expected):
    events = []
    metrics = dict(zip((76, 77, 78, 79), desktop))
    metrics.update({0: 1920, 1: 1080})
    def send(count, ptr, size):
        mi = ptr._obj.mi
        events.append((mi.dx, mi.dy, mi.dwFlags))
        return 1
    monkeypatch.setattr(inputs.ctypes.windll, 'user32', SimpleNamespace(
        GetSystemMetrics=metrics.__getitem__, SendInput=send))
    inputs._send_mouse(inputs.MOUSEEVENTF_MOVE | inputs.MOUSEEVENTF_ABSOLUTE, *point)
    assert events == [(*expected, 0x0001 | 0x8000 | 0x4000)]


def test_mouse_release_does_not_depend_on_desktop_geometry(monkeypatch):
    events = []
    def metrics(_):
        pytest.fail('A button release must not query desktop coordinates')
    monkeypatch.setattr(inputs.ctypes.windll, 'user32', SimpleNamespace(
        GetSystemMetrics=metrics, SendInput=lambda *args: events.append('up') or 1))
    inputs._send_mouse(inputs.MOUSEEVENTF_LEFTUP)
    assert events == ['up']


@pytest.mark.parametrize('desktop,point', [
    ((0, 0, 0, 1080), (0, 20)),
    ((0, 0, 1920, 1080), (-1, 20)),
    ((-1920, 0, 3840, 1080), (1920, 20)),
])
def test_invalid_desktop_point_never_dispatches(monkeypatch, desktop, point):
    metrics = dict(zip((76, 77, 78, 79), desktop))
    monkeypatch.setattr(inputs.ctypes.windll, 'user32', SimpleNamespace(
        GetSystemMetrics=metrics.__getitem__,
        SendInput=lambda *a: pytest.fail('invalid coordinates must not send')))
    with pytest.raises(RuntimeError, match='click_desktop_geometry'):
        inputs._send_mouse(inputs.MOUSEEVENTF_MOVE | inputs.MOUSEEVENTF_ABSOLUTE, *point)


@pytest.fixture
def dispatch(monkeypatch):
    import win32gui
    win = WindowInfo(11, 'offline', -900, 100, 500, 400)
    state = SimpleNamespace(win=win, clock=0., events=[], pointer=(-880, 120), fail_at=None,
                            change='move')
    monkeypatch.setattr('mm_assist.safety.readonly.deny_input', lambda *a: None)
    monkeypatch.setattr('mm_assist.life_plan.live_test.record_input', lambda *a: None)
    monkeypatch.setattr('mm_assist.life_plan.live_test.check_input_deadline', lambda *a: None)
    monkeypatch.setattr('mm_assist.safety.lease.in_test_mode', lambda: True)
    monkeypatch.setattr(inputs, '_guard_before_input', lambda *a, **k: None)
    monkeypatch.setattr(inputs, 'check_abort', lambda: None)
    monkeypatch.setattr(inputs, 'install_mouse_safety', lambda: None)
    monkeypatch.setattr(inputs, 'release_mouse_buttons', lambda: state.events.append('release'))
    monkeypatch.setattr(inputs, '_sendinput_yield_unfocused', lambda *a: False)
    monkeypatch.setattr(inputs, 'is_foreground', lambda _: True)
    monkeypatch.setattr(inputs, 'require_game_point', lambda *a: None)
    monkeypatch.setattr(inputs, '_stamp_sendinput_heartbeat', lambda: None)
    monkeypatch.setattr(inputs, 'hud_suspend_for_input', lambda **k: None)
    monkeypatch.setattr(inputs, 'hud_resume_after_input', lambda: state.events.append('restore'))
    monkeypatch.setattr(inputs.time, 'monotonic', lambda: state.clock)
    def sleep(seconds):
        state.clock += seconds
    monkeypatch.setattr(inputs.time, 'sleep', sleep)
    def geometry(_):
        if state.fail_at is not None and state.clock >= state.fail_at:
            if state.change == 'resize':
                return replace(win, width=480)
            return replace(win, left=win.left+30)
        return win
    monkeypatch.setattr('mm_assist.platform.window.refresh_window', geometry)
    def move(x, y):
        state.pointer = (x, y)
        state.events.append(('move', state.clock, (x, y)))
        return 1
    def send(flags, *xy):
        state.events.append(('mouse', state.clock, flags))
        if flags & inputs.MOUSEEVENTF_MOVE:
            state.pointer = tuple(xy)
    monkeypatch.setattr(inputs.ctypes.windll, 'user32', SimpleNamespace(SetCursorPos=move))
    monkeypatch.setattr(inputs, '_send_mouse', send)
    monkeypatch.setattr(win32gui, 'GetCursorPos', lambda: state.pointer)
    return state


@pytest.mark.parametrize('change', ['move', 'resize'])
@pytest.mark.parametrize('fail_at', [.04, .10, .23, .55])
def test_drag_window_change_stops_movement_and_releases(dispatch, change, fail_at):
    dispatch.fail_at, dispatch.change = fail_at, change
    with inputs.foreground_input_scope(11), pytest.raises(RuntimeError, match='window_(moved|resized)'):
        inputs.drag_client(dispatch.win, 20, 300, 20, 100, end_hold_s=.4)
    moves = [event for event in dispatch.events if isinstance(event, tuple) and event[0] == 'move']
    assert not any(event[1] >= fail_at for event in moves)
    downs = [event for event in dispatch.events if isinstance(event, tuple)
             and event[0] == 'mouse' and event[2] & inputs.MOUSEEVENTF_LEFTDOWN]
    assert len(downs) == (0 if fail_at == .04 else 1)
    assert dispatch.events[-2:] == ['restore', 'release']
    assert not inputs.foreground_input_enabled(11)


def test_click_window_move_during_settle_never_presses(dispatch):
    dispatch.fail_at = .04
    with inputs.foreground_input_scope(11), pytest.raises(RuntimeError, match='window_moved'):
        inputs.click_client(dispatch.win, 20, 20)
    assert not any(isinstance(event, tuple) and event[0] == 'mouse'
                   and event[2] & inputs.MOUSEEVENTF_LEFTDOWN for event in dispatch.events)
    assert dispatch.events[-1] == 'release'


@pytest.mark.parametrize('change', ['focus', 'pointer', 'geometry'])
def test_click_revalidates_after_slow_final_stack_check(dispatch, monkeypatch, change):
    """Already-foreground work has no activation timer, but cannot reuse old input facts."""
    checks = []
    def guard(*a, **kw):
        checks.append(1)
        if len(checks) != 2:
            return
        dispatch.clock += 1.2
        if change == 'focus':
            monkeypatch.setattr(inputs, 'is_foreground', lambda _: False)
            monkeypatch.setattr(inputs, 'require_game_point',
                lambda *a: (_ for _ in ()).throw(RuntimeError('click_focus_lost')))
        elif change == 'pointer':
            dispatch.pointer = (0, 0)
        else:
            dispatch.fail_at = dispatch.clock
    monkeypatch.setattr(inputs, '_guard_before_input', guard)
    with inputs.foreground_input_scope(11), pytest.raises(RuntimeError):
        inputs.click_client(dispatch.win, 20, 20)
    assert not any(isinstance(e, tuple) and e[0] == 'mouse'
                   and e[2] & inputs.MOUSEEVENTF_LEFTDOWN for e in dispatch.events)
    assert dispatch.events[-1] == 'release'


@pytest.mark.parametrize('operation', ['click', 'drag'])
def test_resident_focus_loss_after_preparation_never_refocuses(dispatch, monkeypatch, operation):
    foreground = [True]
    monkeypatch.setattr(inputs, 'is_foreground', lambda _: foreground[0])
    def prepared(_):
        foreground[0] = False
        return None
    monkeypatch.setattr(inputs, '_focus_for_dispatch', prepared)
    monkeypatch.setattr('mm_assist.safety.lease.in_test_mode', lambda: False)
    monkeypatch.setattr(inputs, 'focus_window', lambda *_: pytest.fail('must yield to lost focus'))
    with inputs.foreground_input_scope(11), pytest.raises(RuntimeError, match='click_focus_lost'):
        if operation == 'click':
            inputs.click_client(dispatch.win, 20, 20)
        else:
            inputs.drag_client(dispatch.win, 20, 300, 20, 100)
    assert not any(isinstance(e, tuple) and e[0] in {'mouse', 'move'} for e in dispatch.events)
    assert dispatch.events[-1] == 'release'


@pytest.mark.parametrize('enabled', [False, True])
def test_optional_trace_records_actual_client_coordinates(dispatch, monkeypatch, enabled):
    traces = []
    monkeypatch.setattr('mm_assist.life_plan.execution_records.pointer_trace',
                        lambda **row: traces.append(row), raising=False)
    with inputs.pointer_trace_scope(enabled):
        assert inputs.drag_client(dispatch.win, 20, 300, 20, 100, steps=2, duration_s=.10)
    assert bool(traces) is enabled
    if enabled:
        row = traces[0]
        assert row['kind'] == 'drag' and row['hwnd'] == 11
        assert row['samples'][0]['actual_client'] == [20, 300]
        assert row['samples'][-1]['actual_client'] == [20, 100]
        assert row['samples'][-1]['requested_client'] == [20, 100]
        assert row['elapsed_s'] > 0 and row['error'] == ''
    assert dispatch.events[-1] == 'release'


def test_trace_query_failure_never_prevents_release(dispatch, monkeypatch):
    import win32gui
    traces = []
    monkeypatch.setattr('mm_assist.life_plan.execution_records.pointer_trace',
                        lambda **row: traces.append(row), raising=False)
    def broken():
        raise OSError('pointer unavailable')
    monkeypatch.setattr(win32gui, 'GetCursorPos', broken)
    with inputs.pointer_trace_scope(True):
        assert inputs.drag_client(dispatch.win, 20, 300, 20, 100, steps=2)
    assert traces[0]['samples'][0]['actual_client'] is None
    assert dispatch.events[-1] == 'release'


@pytest.mark.parametrize('operation', ['click', 'drag'])
def test_slow_trace_cannot_bypass_focus_deadline(dispatch, monkeypatch, operation):
    import win32gui
    def slow_pointer():
        dispatch.clock += .6
        return dispatch.pointer
    monkeypatch.setattr(win32gui, 'GetCursorPos', slow_pointer)
    monkeypatch.setattr(inputs, '_focus_for_dispatch', lambda _: 1.)
    with inputs.pointer_trace_scope(True), inputs.foreground_input_scope(11):
        if operation == 'click':
            with pytest.raises(RuntimeError, match='focus_click_timeout'):
                inputs.click_client(dispatch.win, 20, 20)
        else:
            # Drag has one diagnostic read; make its initial focus budget smaller.
            monkeypatch.setattr(inputs, '_focus_for_dispatch', lambda _: .5)
            with pytest.raises(RuntimeError, match='focus_click_timeout'):
                inputs.drag_client(dispatch.win, 20, 300, 20, 100)
    assert not any(isinstance(event, tuple) and event[0] == 'mouse'
                   and event[2] & inputs.MOUSEEVENTF_LEFTDOWN for event in dispatch.events)
    assert dispatch.events[-1] == 'release'


@pytest.mark.parametrize('failure', ['pause', 'focus', 'deadline', 'resize'])
def test_click_hold_interrupts_and_releases_without_replay(dispatch, monkeypatch, failure):
    if failure == 'pause':
        monkeypatch.setattr('mm_assist.tool_repair.op_cancelled', lambda: dispatch.clock >= .13)
    elif failure == 'focus':
        monkeypatch.setattr(inputs, 'is_foreground', lambda _: dispatch.clock < .13)
    elif failure == 'resize':
        dispatch.fail_at, dispatch.change = .13, 'resize'
    else:
        def deadline(_):
            if dispatch.clock >= .13:
                raise RuntimeError('expired')
        monkeypatch.setattr('mm_assist.life_plan.live_test.check_input_deadline', deadline)
    with inputs.foreground_input_scope(11), pytest.raises(RuntimeError):
        inputs.click_client(dispatch.win, 20, 20, hold_s=.3)
    assert dispatch.clock < .18
    downs = [event for event in dispatch.events if isinstance(event, tuple)
             and event[0] == 'mouse' and event[2] & inputs.MOUSEEVENTF_LEFTDOWN]
    assert len(downs) == 1 and dispatch.events[-1] == 'release'


@pytest.mark.parametrize('operation', ['click', 'drag'])
@pytest.mark.parametrize('primary_failure', [False, True])
def test_rejected_release_is_a_failure_and_preserves_primary_reason(dispatch, monkeypatch,
                                                                  operation, primary_failure):
    traces = []
    monkeypatch.setattr('mm_assist.life_plan.execution_records.pointer_trace',
                        lambda **row: traces.append(row))
    send = inputs._send_mouse
    def fail_release(flags, *xy):
        if flags == inputs.MOUSEEVENTF_LEFTUP:
            raise RuntimeError('release_rejected')
        send(flags, *xy)
    monkeypatch.setattr(inputs, '_send_mouse', fail_release)
    if primary_failure:
        dispatch.fail_at = .10
    with inputs.pointer_trace_scope(True), inputs.foreground_input_scope(11):
        with pytest.raises(RuntimeError, match='window_moved' if primary_failure else 'release_rejected'):
            if operation == 'click':
                inputs.click_client(dispatch.win, 20, 20, hold_s=.3)
            else:
                inputs.drag_client(dispatch.win, 20, 300, 20, 100)
    assert 'release_rejected' in traces[0]['error']
    assert dispatch.events[-1] == 'release'


def test_drag_dispatch_uses_the_same_integer_path_as_occlusion_and_trace(dispatch, monkeypatch):
    paths, traces = [], []
    monkeypatch.setattr(inputs, 'hud_suspend_for_input', lambda **k: paths.append(k['points']))
    monkeypatch.setattr('mm_assist.life_plan.execution_records.pointer_trace', lambda **row: traces.append(row))
    with inputs.pointer_trace_scope(True):
        assert inputs.drag_client(dispatch.win, 20, 300, 21, 100, steps=3)
    moves = [event[2] for event in dispatch.events if isinstance(event, tuple) and event[0] == 'move']
    assert moves == list(paths[0])
    assert traces[0]['planned_client_points'] == [[20, 300], [20, 233], [20, 166], [21, 100]]


@pytest.mark.parametrize('failure', ['focus', 'stack', 'geometry', 'wrong_hwnd'])
def test_resident_windowless_escape_cannot_bypass_key_guards(dispatch, monkeypatch, failure):
    monkeypatch.setattr(inputs, '_send_key', lambda *a: pytest.fail('refused Esc must not dispatch'))
    if failure == 'focus':
        monkeypatch.setattr(inputs, 'is_foreground', lambda _: False)
    elif failure == 'geometry':
        dispatch.fail_at, dispatch.change = 0., 'resize'
    elif failure == 'stack':
        def guard(*a, **k):
            raise RuntimeError('click_ui_not_ready')
        monkeypatch.setattr(inputs, '_guard_before_input', guard)
    with inputs.foreground_input_scope(11, win=dispatch.win), pytest.raises(RuntimeError):
        inputs.press_escape(**({'hwnd': 12} if failure == 'wrong_hwnd' else {}))
    assert inputs._FOREGROUND_INPUT_WINDOW.get() is None


@pytest.mark.parametrize('kind,executor', [
    ('gather', 'mm_assist.life_execute.skill_nearby.execute_skill_nearby'),
    ('feed', 'mm_assist.life_plan.execute.execute_facility_feed'),
    ('claim', 'mm_assist.life_plan.execute.execute_facility_claim'),
    ('repair', 'mm_assist.life_execute.repair.execute_repair'),
])
@pytest.mark.parametrize('enabled', [False, True])
def test_resident_pump_routes_all_plan_kinds_through_shared_input(dispatch, monkeypatch, tmp_path,
                                                                 kind, executor, enabled):
    from mm_assist.config import AppConfig
    from mm_assist.life_plan import action_finish, bar_ui, controller, execution_records as records
    from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()
    bar_ui.reset_recovery_for_tests()
    item = PlanItem('p1', kind, 'offline plan', recipe='木材' if kind == 'feed' else '',
                    facility='wood', skill='伐木', detail='樹木')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item])
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda _: True)
    monkeypatch.setattr('mm_assist.life_execute.ui_act.pump_aborted', lambda: False)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly', lambda **k: None)
    monkeypatch.setattr('mm_assist.life_plan.game_stop.GameStopWatch.check', lambda *a, **k: False)
    monkeypatch.setattr('mm_assist.life_plan.gather_repair.GatherRepairWatch.check', lambda *a: False)
    monkeypatch.setattr(action_finish, 'read_completion_stp',
                        lambda: action_finish.StpSample(0, 1000., 42, True))
    rec = records.ExecutionRecorder(tmp_path/'records.json', wall=lambda: 1000.,
                                    mono=lambda: dispatch.clock, pid=42)
    monkeypatch.setattr(records, '_RECORDER', rec)
    calls = []
    def execute(*a, **k):
        calls.append(kind)
        assert inputs.foreground_input_enabled(11)
        assert inputs.click_client(dispatch.win, 20, 20)
        assert inputs.drag_client(dispatch.win, 20, 300, 20, 100, steps=2)
        # Legacy map branches omit win; the resident pump must still bind Esc.
        inputs.press_escape()
        return ExecuteResult(False, 'cancel')
    monkeypatch.setattr(executor, execute)
    monkeypatch.setattr(inputs, '_send_key', lambda vk: calls.append(vk))
    cfg = AppConfig.model_validate({'window': {'input_trace': enabled}})
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000.)
    try:
        bar_ui.pump_life_once(cfg, dispatch.win)
        assert calls == [kind, 0x1B]
        row = rec.read()[0]
        traces = [e for e in row['events'] if e['event'] == 'pointer_trace']
        assert len(traces) == (2 if enabled else 0)
        assert row['pid'] == 42 and row['run_started_at'] == 1000. and row['plan_id'] == 'p1'
        assert not inputs.foreground_input_enabled(11)
        # The formal scope must not leak into unrelated later input.
        assert inputs.drag_client(dispatch.win, 20, 300, 20, 100, steps=2)
        assert len([e for e in rec.read()[0]['events'] if e['event'] == 'pointer_trace']) == len(traces)
    finally:
        controller.reset_controller_for_tests()
        bar_ui.reset_pump_settle()
        bar_ui.reset_recovery_for_tests()
