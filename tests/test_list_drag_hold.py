"""Observe actual input ordering without sending any Windows input."""
from types import SimpleNamespace

import pytest

from mm_assist.safety import input as inputs


@pytest.mark.parametrize('failure', ['', 'cancel', 'focus', 'deadline'])
def test_endpoint_is_held_before_release_and_interrupt_releases(monkeypatch, failure):
    win = SimpleNamespace(hwnd=7, left=0, top=0, width=400, height=400)
    events, clock = [], [0.0]
    monkeypatch.setattr(inputs.sys, 'platform', 'win32')
    monkeypatch.setattr(inputs, 'require_valid_client_xy', lambda w, x, y: (x, y))
    monkeypatch.setattr('mm_assist.life_plan.live_test.record_input', lambda *a: 30)
    def deadline(_):
        if failure == 'deadline' and clock[0] >= .55:
            raise RuntimeError('expired')
    monkeypatch.setattr('mm_assist.life_plan.live_test.check_input_deadline', deadline)
    monkeypatch.setattr(inputs, '_guard_before_input', lambda *a, **kw: None)
    monkeypatch.setattr(inputs, 'install_mouse_safety', lambda: None)
    monkeypatch.setattr(inputs, '_focus_for_dispatch', lambda _: None)
    monkeypatch.setattr(inputs, '_sendinput_yield_unfocused', lambda *a: False)
    monkeypatch.setattr(inputs, 'is_foreground', lambda _: failure != 'focus' or clock[0] < .55)
    monkeypatch.setattr(inputs, '_refresh_same_client_size', lambda _: win)
    monkeypatch.setattr(inputs, 'hud_suspend_for_input', lambda **kw: None)
    monkeypatch.setattr(inputs, 'hud_resume_after_input', lambda: None)
    monkeypatch.setattr(inputs, '_stamp_sendinput_heartbeat', lambda: None)
    monkeypatch.setattr(inputs, 'release_mouse_buttons', lambda: events.append(('released', clock[0])))
    monkeypatch.setattr(inputs, '_send_mouse', lambda flags: events.append((flags, clock[0])))
    monkeypatch.setattr(inputs.ctypes.windll, 'user32', SimpleNamespace(
        SetCursorPos=lambda x, y: events.append(((x, y), clock[0]))))
    monkeypatch.setattr(inputs.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(inputs.time, 'sleep', lambda n: clock.__setitem__(0, clock[0] + n))
    def abort():
        if failure == 'cancel' and clock[0] >= .55:
            raise RuntimeError('cancel')
    monkeypatch.setattr(inputs, 'check_abort', abort)
    monkeypatch.setattr('mm_assist.tool_repair.op_cancelled', lambda: False)
    if failure:
        with pytest.raises(RuntimeError):
            inputs.drag_client(win, 200, 300, 200, 100, end_hold_s=.4)
    else:
        assert inputs.drag_client(win, 200, 300, 200, 100, end_hold_s=.4)
        arrived = next(ts for event, ts in events if event == (200, 100))
        lifted = next(ts for event, ts in events if event == inputs.MOUSEEVENTF_LEFTUP)
        assert lifted - arrived >= .4
    assert events[-1][0] == 'released'
    assert any(event == inputs.MOUSEEVENTF_LEFTUP for event, _ in events)
