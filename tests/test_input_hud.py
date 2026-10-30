"""状态栏输入让行：仅当本程序 HUD 挡住落点/拖拽路径才隐藏。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _game_win(**kw):
    base = dict(hwnd=11, left=100, top=200, width=500, height=400)
    base.update(kw)
    return SimpleNamespace(**base)


def _patch_click_guards(monkeypatch, inputs, events):
    monkeypatch.setattr("mm_assist.safety.readonly.deny_input", lambda *a: None)
    monkeypatch.setattr("mm_assist.life_plan.live_test.record_input", lambda *a: None)
    monkeypatch.setattr("mm_assist.safety.lease.in_test_mode", lambda: True)
    monkeypatch.setattr(inputs, "_guard_before_input", lambda *a, **k: None)
    monkeypatch.setattr(inputs, "install_mouse_safety", lambda: None)
    monkeypatch.setattr(inputs, "release_mouse_buttons", lambda: events.append("release"))
    monkeypatch.setattr(inputs, "_sendinput_yield_unfocused", lambda *a: False)
    monkeypatch.setattr(inputs, "is_foreground", lambda _: True)
    monkeypatch.setattr(inputs, "_stamp_sendinput_heartbeat", lambda: None)
    monkeypatch.setattr(inputs, "_send_mouse", lambda flags, *xy: events.append(("mouse", flags, xy)))
    monkeypatch.setattr(inputs.time, "sleep", lambda _t: None)


@pytest.mark.parametrize('focus_delay,focused,error',[(.1,True,None),(1.1,True,None),(.1,False,'click_focus_lost')])
@pytest.mark.parametrize('operation',['click','drag'])
def test_life_background_click_focuses_after_evidence_and_obeys_one_second(monkeypatch,focus_delay,focused,error,operation):
    import win32gui
    from mm_assist.safety import input as inputs
    events, clock, foreground = [], [0.0], [False]
    win = _game_win()
    _patch_click_guards(monkeypatch,inputs,events)
    monkeypatch.setattr(inputs.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(inputs,'is_foreground',lambda _:foreground[0])
    monkeypatch.setattr('mm_assist.life_plan.live_test.record_input',lambda *a:events.append('evidence'))
    monkeypatch.setattr(inputs,'_guard_before_input',lambda *a,**k:events.append('authorized'))
    monkeypatch.setattr(inputs,'_refresh_same_client_size',lambda _:win)
    monkeypatch.setattr(inputs,'hud_suspend_for_input',lambda **k:None)
    monkeypatch.setattr(inputs,'hud_resume_after_input',lambda:None)
    monkeypatch.setattr(inputs,'require_game_point',lambda *a:None)
    monkeypatch.setattr(win32gui,'GetCursorPos',lambda:(win.left+20,win.top+20))
    if operation == 'drag':
        import ctypes
        from types import SimpleNamespace
        monkeypatch.setattr(ctypes.windll,'user32',SimpleNamespace(SetCursorPos=lambda *a:None))
    dispatch = (lambda:inputs.click_client(win,20,20)) if operation == 'click' else (
        lambda:inputs.drag_client(win,20,20,40,40,steps=2,duration_s=.02))
    def focus(*a,**k):
        events.append('focus')
        clock[0] += focus_delay
        foreground[0] = focused
        return focused
    monkeypatch.setattr(inputs,'focus_window',focus)
    with inputs.foreground_input_scope(win.hwnd):
        if error:
            with pytest.raises(RuntimeError,match=error):
                dispatch()
        else:
            assert dispatch()
    assert events[:3] == ['evidence','authorized','focus']
    downs = [e for e in events if isinstance(e,tuple) and e[0]=='mouse' and e[1]&inputs.MOUSEEVENTF_LEFTDOWN]
    assert len(downs) == (0 if error else 1)
    assert events[-1] == 'release'
    assert not inputs.foreground_input_enabled(win.hwnd)


def test_life_focus_scope_does_not_bypass_authorization(monkeypatch):
    from mm_assist.safety import input as inputs
    from mm_assist.safety.lease import NotAuthorized
    events = []
    win = _game_win()
    _patch_click_guards(monkeypatch,inputs,events)
    def refused(*a,**k):
        raise NotAuthorized('no lease')
    monkeypatch.setattr(inputs,'_guard_before_input',refused)
    monkeypatch.setattr(inputs,'focus_window',lambda *a,**k:pytest.fail('must authorize before focus'))
    with inputs.foreground_input_scope(win.hwnd),pytest.raises(NotAuthorized):
        inputs.click_client(win,20,20)
    assert not events


def test_focus_budget_starts_when_game_foreground_is_confirmed(monkeypatch):
    from mm_assist.safety import input as inputs
    clock, foreground = [0.], [False]
    monkeypatch.setattr(inputs, 'check_abort', lambda: None)
    monkeypatch.setattr(inputs.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(inputs, 'is_foreground', lambda _: foreground[0])
    def focus(_):
        clock[0] += 1.1  # Preparation before the window actually activates.
        foreground[0] = True
    monkeypatch.setattr(inputs, 'focus_window', focus)
    with inputs.foreground_input_scope(11):
        deadline = inputs._focus_for_dispatch(_game_win())
        assert deadline == pytest.approx(2.1)
        clock[0] += .9
        inputs._check_focus_deadline(deadline)
        clock[0] += .1
        with pytest.raises(RuntimeError, match='focus_click_timeout'):
            inputs._check_focus_deadline(deadline)


def test_already_foreground_slow_recheck_is_not_a_new_focus_timeout(monkeypatch):
    """The overnight log timed ordinary foreground clicks as fresh activation."""
    import win32gui
    from mm_assist.safety import input as inputs
    events, clock, checks = [], [0.0], []
    win = _game_win()
    _patch_click_guards(monkeypatch, inputs, events)
    monkeypatch.setattr(inputs.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(inputs, '_refresh_same_client_size', lambda _: win)
    monkeypatch.setattr(inputs, 'hud_suspend_for_input', lambda **_: None)
    monkeypatch.setattr(inputs, 'hud_resume_after_input', lambda: None)
    monkeypatch.setattr(inputs, 'require_game_point', lambda *a: None)
    monkeypatch.setattr(inputs, 'focus_window', lambda *_: pytest.fail('already foreground'))
    monkeypatch.setattr(win32gui, 'GetCursorPos', lambda: (win.left+20, win.top+20))
    def guard(*a, **kw):
        checks.append(clock[0])
        clock[0] += 1.2
    monkeypatch.setattr(inputs, '_guard_before_input', guard)
    with inputs.foreground_input_scope(win.hwnd):
        assert inputs.click_client(win, 20, 20)
    assert len(checks) == 2, 'current stack must still be rechecked before down'
    assert len([e for e in events if isinstance(e, tuple) and e[1]&inputs.MOUSEEVENTF_LEFTDOWN]) == 1


@pytest.mark.parametrize('operation', ['click', 'drag'])
def test_background_slow_final_check_keeps_original_focus_deadline(monkeypatch, operation):
    import win32gui
    from mm_assist.safety import input as inputs
    events, clock, foreground, checks = [], [0.0], [False], []
    win = _game_win()
    _patch_click_guards(monkeypatch, inputs, events)
    monkeypatch.setattr(inputs.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(inputs, 'is_foreground', lambda _: foreground[0])
    monkeypatch.setattr(inputs, 'focus_window', lambda *_: foreground.__setitem__(0, True))
    monkeypatch.setattr(inputs, '_refresh_same_client_size', lambda _: win)
    monkeypatch.setattr(inputs, 'hud_suspend_for_input', lambda **_: None)
    monkeypatch.setattr(inputs, 'hud_resume_after_input', lambda: None)
    monkeypatch.setattr(inputs, 'require_game_point', lambda *a: None)
    monkeypatch.setattr(win32gui, 'GetCursorPos', lambda: (win.left+20, win.top+20))
    if operation == 'drag':
        import ctypes
        monkeypatch.setattr(ctypes.windll, 'user32', SimpleNamespace(SetCursorPos=lambda *a: None))
    def guard(*a, **kw):
        checks.append(clock[0])
        if len(checks) == 2:
            clock[0] += 1.2
    monkeypatch.setattr(inputs, '_guard_before_input', guard)
    with inputs.foreground_input_scope(win.hwnd), pytest.raises(RuntimeError, match='focus_click_timeout'):
        if operation == 'click':
            inputs.click_client(win, 20, 20)
        else:
            inputs.drag_client(win, 20, 20, 40, 40, steps=2, duration_s=.02)
    assert not any(isinstance(e, tuple) and e[1]&inputs.MOUSEEVENTF_LEFTDOWN for e in events)
    assert events[-1] == 'release'


def test_visible_hud_not_blocking_point_is_not_hidden(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)
    monkeypatch.setattr(win32gui, "WindowFromPoint", lambda xy: 11)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    inputs.hud_suspend_for_input(points=[(120, 220)])
    assert seen == []
    assert inputs._INPUT_HUD_RESTORE is False


def test_suspend_without_points_does_not_hide(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    inputs.hud_suspend_for_input()
    assert seen == []


@pytest.mark.parametrize('room', [True, False])
def test_resident_blocking_hud_moves_clear_or_refuses_without_hiding(monkeypatch, room):
    import win32gui
    import win32api
    from mm_assist.safety import input as inputs
    hud = [550, 540, 1348, 2160]
    game = (949, 360, 2861, 1802)
    moves, shows = [], []
    monkeypatch.setattr(inputs, '_INPUT_HUD_HWND', 42)
    monkeypatch.setattr(inputs, '_INPUT_HUD_RESTORE', False)
    monkeypatch.setattr(win32gui, 'IsWindow', lambda _: True)
    monkeypatch.setattr(win32gui, 'IsWindowVisible', lambda _: True)
    monkeypatch.setattr(win32gui, 'IsChild', lambda *a: False)
    monkeypatch.setattr(win32gui, 'GetWindowRect', lambda hwnd: tuple(hud) if hwnd == 42 else game)
    monkeypatch.setattr(win32api, 'MonitorFromWindow', lambda *a: 1)
    monkeypatch.setattr(win32api, 'GetMonitorInfo', lambda _: {'Work': (0, 0, 3840, 2160) if room else game})
    monkeypatch.setattr(win32gui, 'WindowFromPoint', lambda pt: 42 if (
        hud[0] <= pt[0] < hud[2] and hud[1] <= pt[1] < hud[3]) else 11)
    def move(hwnd, after, x, y, width, height, flags):
        moves.append((x, y))
        hud[:] = [x, y, x + 798, y + 1620]
    monkeypatch.setattr(win32gui, 'SetWindowPos', move)
    monkeypatch.setattr(win32gui, 'ShowWindow', lambda *a: shows.append(a))
    with inputs.foreground_input_scope(11):
        if room:
            inputs.hud_suspend_for_input(points=[(1045, 1152)])
            assert moves and not inputs._hud_blocks_polyline([(1045, 1152)])
        else:
            with pytest.raises(RuntimeError, match='click_hud_occluded'):
                inputs.hud_suspend_for_input(points=[(1045, 1152)])
            assert not moves
    inputs.hud_resume_after_input()
    assert shows == [] and not inputs._INPUT_HUD_RESTORE


def test_hud_hwnd_blocking_point_hides_and_restores_once(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)
    monkeypatch.setattr(win32gui, "WindowFromPoint", lambda xy: 42)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    inputs.hud_suspend_for_input(points=[(10, 10)])
    inputs.hud_resume_after_input()
    assert seen == [(42, 0), (42, 4)]
    assert inputs._INPUT_HUD_RESTORE is False


def test_hud_child_blocking_point_hides(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)
    monkeypatch.setattr(win32gui, "WindowFromPoint", lambda xy: 99)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: parent == 42 and child == 99)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    inputs.hud_suspend_for_input(points=[(10, 10)])
    assert seen == [(42, 0)]


def test_third_party_occlusion_still_click_occluded(monkeypatch):
    import win32gui
    from mm_assist.safety.input import require_game_point

    monkeypatch.setattr("mm_assist.safety.input.is_foreground", lambda h: True)
    monkeypatch.setattr(win32gui, "WindowFromPoint", lambda xy: 77)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    with pytest.raises(RuntimeError, match="click_occluded"):
        require_game_point(11, 100, 200)


def test_query_failure_uses_safe_hide(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)

    def boom(_xy):
        raise OSError("WindowFromPoint failed")

    monkeypatch.setattr(win32gui, "WindowFromPoint", boom)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    inputs.hud_suspend_for_input(points=[(10, 10)])
    assert seen == [(42, 0)]


def test_drag_midpoint_hud_hides_even_if_start_is_clear(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)

    def wfp(xy):
        sx, sy = xy
        if sx == 150 and sy == 250:
            return 42
        return 11

    monkeypatch.setattr(win32gui, "WindowFromPoint", wfp)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    inputs.hud_suspend_for_input(points=[(100, 200), (150, 250), (200, 300)])
    assert seen == [(42, 0)]
    inputs.hud_resume_after_input()
    assert seen == [(42, 0), (42, 4)]


def test_click_client_skips_hide_when_hud_clear_uses_refreshed_coords(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    events = []
    win = _game_win()
    refreshed = _game_win(left=140, top=260)
    _patch_click_guards(monkeypatch, inputs, events)
    monkeypatch.setattr("mm_assist.platform.window.refresh_window", lambda w: refreshed)
    monkeypatch.setattr(inputs, "require_game_point", lambda hwnd, sx, sy: events.append(("point", hwnd, sx, sy)))
    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)
    queried = []

    def wfp(xy):
        queried.append(xy)
        return 11

    monkeypatch.setattr(win32gui, "WindowFromPoint", wfp)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    monkeypatch.setattr(win32gui, "GetCursorPos", lambda: (160, 280))
    assert inputs.click_client(win, 20, 20)
    assert queried == [(160, 280)]
    assert seen == []
    assert events[-1] == "release"
    assert ("point", 11, 160, 280) in events


def test_click_client_hides_when_hud_blocks_then_restores_on_occlusion_error(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    events = []
    win = _game_win()
    _patch_click_guards(monkeypatch, inputs, events)
    monkeypatch.setattr("mm_assist.platform.window.refresh_window", lambda w: w)
    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)
    monkeypatch.setattr(win32gui, "WindowFromPoint", lambda xy: 42)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    monkeypatch.setattr(win32gui, "GetCursorPos", lambda: (120, 220))

    def occlude(*_a):
        raise RuntimeError("click_occluded: third party")

    monkeypatch.setattr(inputs, "require_game_point", occlude)
    with pytest.raises(RuntimeError, match="click_occluded"):
        inputs.click_client(win, 20, 20)
    assert seen == [(42, 0), (42, 4)]
    assert events[-1] == "release"


def test_unity_hook_click_does_not_suspend_hud(monkeypatch):
    import sys
    import types
    from mm_assist.safety import input as inputs

    events = []
    win = _game_win()
    hook_mod = types.ModuleType("mm_assist.unity_input_hook")
    hook_mod.click_client_unity_hook = lambda *a, **k: events.append("hook")
    monkeypatch.setitem(sys.modules, "mm_assist.unity_input_hook", hook_mod)
    monkeypatch.setattr("mm_assist.safety.readonly.deny_input", lambda *a: None)
    monkeypatch.setattr("mm_assist.life_plan.live_test.record_input", lambda *a: None)
    monkeypatch.setattr(inputs, "_guard_before_input", lambda *a, **k: None)
    monkeypatch.setattr(inputs, "release_mouse_buttons", lambda: events.append("release"))
    monkeypatch.setattr(inputs, "hud_suspend_for_input", lambda *a, **k: events.append("hide"))
    assert inputs.click_client(win, 20, 20, method="unity_hook")
    assert events == ["hook", "release"]


def test_drag_client_hides_for_path_not_only_start(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    events = []
    win = _game_win()
    monkeypatch.setattr("mm_assist.safety.readonly.deny_input", lambda *a: None)
    monkeypatch.setattr("mm_assist.life_plan.live_test.record_input", lambda *a: None)
    monkeypatch.setattr("mm_assist.safety.lease.in_test_mode", lambda: True)
    monkeypatch.setattr(inputs, "_guard_before_input", lambda *a, **k: None)
    monkeypatch.setattr(inputs, "install_mouse_safety", lambda: None)
    monkeypatch.setattr(inputs, "release_mouse_buttons", lambda: events.append("release"))
    monkeypatch.setattr(inputs, "_sendinput_yield_unfocused", lambda *a: False)
    monkeypatch.setattr(inputs, "is_foreground", lambda _: True)
    monkeypatch.setattr(inputs, "_stamp_sendinput_heartbeat", lambda: None)
    monkeypatch.setattr(inputs, "_send_mouse", lambda flags, *xy: None)
    monkeypatch.setattr(inputs.time, "sleep", lambda _t: None)
    monkeypatch.setattr("mm_assist.platform.window.refresh_window", lambda w: w)
    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)

    def wfp(xy):
        sx, sy = xy
        if sx == 180 and sy == 290:
            return 42
        return 11

    monkeypatch.setattr(win32gui, "WindowFromPoint", wfp)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    user32 = SimpleNamespace(SetCursorPos=lambda *a: None)
    import ctypes as real_ctypes

    monkeypatch.setattr(real_ctypes.windll, "user32", user32)

    assert inputs.drag_client(win, 30, 40, 130, 140, steps=2, duration_s=0.02)
    assert seen[0] == (42, 0)
    assert seen[-1] == (42, 4)
    assert events[-1] == "release"


def test_abort_during_drag_restores_hud_and_releases(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    events = []
    win = _game_win()
    monkeypatch.setattr("mm_assist.safety.readonly.deny_input", lambda *a: None)
    monkeypatch.setattr("mm_assist.life_plan.live_test.record_input", lambda *a: None)
    monkeypatch.setattr("mm_assist.safety.lease.in_test_mode", lambda: True)
    monkeypatch.setattr(inputs, "_guard_before_input", lambda *a, **k: None)
    monkeypatch.setattr(inputs, "install_mouse_safety", lambda: None)
    monkeypatch.setattr(inputs, "release_mouse_buttons", lambda: events.append("release"))
    monkeypatch.setattr(inputs, "_sendinput_yield_unfocused", lambda *a: False)
    monkeypatch.setattr(inputs, "is_foreground", lambda _: True)
    monkeypatch.setattr(inputs, "_send_mouse", lambda flags, *xy: None)
    monkeypatch.setattr(inputs.time, "sleep", lambda _t: None)
    monkeypatch.setattr("mm_assist.platform.window.refresh_window", lambda w: w)
    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)
    monkeypatch.setattr(win32gui, "WindowFromPoint", lambda xy: 42)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    user32 = SimpleNamespace(SetCursorPos=lambda *a: None)
    import ctypes as real_ctypes

    monkeypatch.setattr(real_ctypes.windll, "user32", user32)

    n = {"i": 0}

    def abort_after_move():
        n["i"] += 1
        if n["i"] > 1:
            raise inputs.AbortRequested("user abort")

    monkeypatch.setattr(inputs, "check_abort", abort_after_move)
    with pytest.raises(inputs.AbortRequested):
        inputs.drag_client(win, 30, 40, 80, 80, steps=4, duration_s=0.02)
    assert seen == [(42, 0), (42, 4)]
    assert events[-1] == "release"


def test_polyline_crosses_narrow_hud_between_samples_hides(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)
    monkeypatch.setattr(win32gui, "GetWindowRect", lambda h: (100, 100, 200, 110))
    monkeypatch.setattr(win32gui, "WindowFromPoint", lambda xy: 11)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    inputs.hud_suspend_for_input(points=[(150, 50), (150, 200)])
    assert seen == [(42, 0)]


def test_polyline_near_hud_without_intersect_does_not_hide(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs

    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)
    monkeypatch.setattr(win32gui, "GetWindowRect", lambda h: (100, 100, 200, 110))
    monkeypatch.setattr(win32gui, "WindowFromPoint", lambda xy: 11)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    seen = []
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    inputs.hud_suspend_for_input(points=[(150, 50), (250, 50)])
    assert seen == []
    assert inputs._INPUT_HUD_RESTORE is False


def test_click_client_refresh_size_change_does_not_press_and_releases(monkeypatch):
    from mm_assist.safety import input as inputs

    events = []
    win = _game_win()
    resized = _game_win(width=480, height=380)
    _patch_click_guards(monkeypatch, inputs, events)
    monkeypatch.setattr("mm_assist.platform.window.refresh_window", lambda w: resized)
    with pytest.raises(RuntimeError, match="click_window_resized"):
        inputs.click_client(win, 20, 20)
    assert not any(
        isinstance(e, tuple) and e[0] == "mouse" and (e[1] & inputs.MOUSEEVENTF_LEFTDOWN)
        for e in events
    )
    assert events[-1] == "release"


def test_drag_client_refresh_size_change_does_not_press_and_releases(monkeypatch):
    from mm_assist.safety import input as inputs

    events = []
    win = _game_win()
    resized = _game_win(width=480, height=380)
    monkeypatch.setattr("mm_assist.safety.readonly.deny_input", lambda *a: None)
    monkeypatch.setattr("mm_assist.life_plan.live_test.record_input", lambda *a: None)
    monkeypatch.setattr("mm_assist.safety.lease.in_test_mode", lambda: True)
    monkeypatch.setattr(inputs, "_guard_before_input", lambda *a, **k: None)
    monkeypatch.setattr(inputs, "install_mouse_safety", lambda: None)
    monkeypatch.setattr(inputs, "release_mouse_buttons", lambda: events.append("release"))
    monkeypatch.setattr(inputs, "_sendinput_yield_unfocused", lambda *a: False)
    monkeypatch.setattr(inputs, "_stamp_sendinput_heartbeat", lambda: None)
    monkeypatch.setattr(
        inputs, "_send_mouse", lambda flags, *xy: events.append(("mouse", flags, xy))
    )
    monkeypatch.setattr(inputs.time, "sleep", lambda _t: None)
    monkeypatch.setattr("mm_assist.platform.window.refresh_window", lambda w: resized)
    monkeypatch.setattr(inputs, "hud_suspend_for_input", lambda *a, **k: events.append("hide"))
    user32 = SimpleNamespace(SetCursorPos=lambda *a: events.append(("move", a)))
    import ctypes as real_ctypes

    monkeypatch.setattr(real_ctypes.windll, "user32", user32)
    with pytest.raises(RuntimeError, match="click_window_resized"):
        inputs.drag_client(win, 30, 40, 80, 80, steps=2, duration_s=0.02)
    assert "hide" not in events
    assert not any(e[0] == "move" for e in events if isinstance(e, tuple))
    assert not any(
        isinstance(e, tuple) and e[0] == "mouse" and (e[1] & inputs.MOUSEEVENTF_LEFTDOWN)
        for e in events
    )
    assert events[-1] == "release"
