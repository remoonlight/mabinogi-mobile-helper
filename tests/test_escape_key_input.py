"""Esc must use guarded foreground input and the full Windows INPUT ABI."""
import ctypes
import sys
from types import SimpleNamespace

import pytest

from mm_assist.safety import input as inputs


@pytest.mark.skipif(sys.platform != 'win32', reason='Win32 input ABI')
@pytest.mark.parametrize('mode', ['ok', 'focus', 'stack', 'owner', 'lease', 'geometry', 'cancel'])
def test_client_escape_checks_guards_without_focusing(monkeypatch, mode):
    events = []
    win = SimpleNamespace(hwnd=7)
    monkeypatch.setattr(inputs, 'check_abort', lambda: None)
    monkeypatch.setattr('mm_assist.tool_repair.op_cancelled', lambda: mode == 'cancel')
    monkeypatch.setattr('mm_assist.life_plan.live_test.record_input', lambda *a: None)
    monkeypatch.setattr(inputs, 'focus_window', lambda _: pytest.fail('Esc must not steal focus'))
    monkeypatch.setattr(inputs, 'is_foreground', lambda _: mode != 'focus')
    def guard(*args, **kwargs):
        assert args[0] == 'key' and kwargs['method'] == 'sendinput'
        events.append('guard')
        if mode in {'stack', 'owner', 'lease'}:
            raise RuntimeError(mode)
    monkeypatch.setattr(inputs, '_guard_before_input', guard)
    def geometry(_):
        events.append('geometry')
        if mode == 'geometry':
            raise RuntimeError(mode)
        return win
    monkeypatch.setattr(inputs, '_refresh_same_client_size', geometry)
    monkeypatch.setattr(inputs, '_send_key', lambda vk: events.append(vk))
    monkeypatch.setattr(inputs, '_stamp_sendinput_heartbeat', lambda: events.append('heartbeat'))
    if mode != 'ok':
        with pytest.raises(RuntimeError):
            inputs.press_escape(win=win)
        assert 0x1B not in events and 'heartbeat' not in events
    else:
        inputs.press_escape(win=win)
        assert events == ['guard', 'geometry', 0x1B, 'heartbeat']


@pytest.mark.skipif(sys.platform != 'win32', reason='Win32 input ABI')
@pytest.mark.parametrize('mode', ['ok', 'down_rejected', 'wait_error', 'up_rejected'])
def test_key_abi_and_keyup_even_when_wait_raises(monkeypatch, mode):
    events = []
    def send(count, ptr, size):
        assert count == 1 and size == ctypes.sizeof(inputs.INPUT)
        assert ptr._obj.type == 1 and ptr._obj.u.ki.wVk == 0x1B
        events.append(int(ptr._obj.u.ki.dwFlags))
        return 0 if mode == 'down_rejected' or (mode == 'up_rejected' and events[-1] == 2) else 1
    def sleep(_):
        if mode == 'wait_error':
            raise RuntimeError('stop while held')
    monkeypatch.setattr(ctypes.windll, 'user32', SimpleNamespace(SendInput=send))
    monkeypatch.setattr(inputs.time, 'sleep', sleep)
    if mode == 'ok':
        inputs._send_key(0x1B)
    else:
        with pytest.raises(RuntimeError):
            inputs._send_key(0x1B)
    assert events == ([0] if mode == 'down_rejected' else [0, 2])
