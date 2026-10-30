"""Explicit Esc recovery: bounded, fresh evidence, no business action replay."""
import time
from types import SimpleNamespace

import pytest

from mm_assist.life_execute.ui_logic import UiWhere
from mm_assist.life_execute.ui_recover import recover_once


def where(node='craft_hub', *, ready=True, sampled=None, names=(), stp=0):
    return UiWhere(node, 0, 0, stp, 'fixture', stack_ready=ready,
                   stack_sampled_ts=time.time() if sampled is None else sampled,
                   stack_names=names)


@pytest.fixture(autouse=True)
def no_live_input(monkeypatch):
    monkeypatch.setattr('mm_assist.safety.input.press_escape',
                        lambda **_: pytest.fail('tests must inject Esc'))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr('mm_assist.tool_repair.op_cancelled', lambda: False)
    monkeypatch.setattr('mm_assist.safety.input.check_abort', lambda: None)
    monkeypatch.setattr('mm_assist.life_plan.live_test.inspect_nav_watch_ready', lambda **_: ({}, ''))
    monkeypatch.setattr('mm_assist.life_execute.ui_nav._hud_xy', lambda *a, **k: (10, 20))


def test_one_esc_waits_two_seconds_then_new_frame(monkeypatch):
    events = []
    frames = iter([object(), object()])
    def grab():
        events.append('frame')
        return next(frames)
    observations = iter(['craft_hub', 'main_hud'])
    result = recover_once('need_verify', grab=grab,
        resolve=lambda _: where(next(observations)), press_esc=lambda: events.append('esc'),
        sleep_fn=lambda sec: events.append(sec))
    assert result['ok'] and result['action'] == 'press_escape'
    assert events[0:2] == ['frame', 'esc'] and events[-1] == 'frame'
    assert sum(v for v in events if isinstance(v, float)) == pytest.approx(2.0)


@pytest.mark.parametrize('observation', [where('unknown'), where(ready=False),
    where(sampled=1), where(names=('ToyDimLayerUIInstance',)), where(stp=1)])
def test_unready_unknown_blocked_busy_refuse_esc(observation):
    result = recover_once('need_verify', grab=lambda: object(), resolve=lambda _: observation,
        press_esc=lambda: pytest.fail('unsafe Esc'), sleep_fn=lambda _: None)
    assert not result['ok']


@pytest.mark.parametrize('node,ready,post_sample,menu,ok', [
    ('main_hud', True, True, (10, 20), True),
    ('main_hud', True, True, None, False),
    ('main_hud', False, True, (10, 20), False),
    ('main_hud', True, False, (10, 20), False),
    ('craft_hub', True, True, (10, 20), False),
])
def test_post_esc_requires_independent_post_input_hud(monkeypatch, node, ready, post_sample, menu, ok):
    sent = []
    def resolve(_):
        return where(node, ready=ready, sampled=time.time() if post_sample else 1) if sent else where()
    monkeypatch.setattr('mm_assist.life_execute.ui_nav._hud_xy', lambda *a, **k: menu)
    result = recover_once('need_verify', grab=lambda: object(), resolve=resolve,
        press_esc=lambda: sent.append(1), sleep_fn=lambda _: None)
    assert result['ok'] == ok and sent == [1]


def test_already_hud_never_opens_menu_with_esc():
    assert recover_once('need_verify', grab=lambda: object(), resolve=lambda _: where('main_hud'),
        press_esc=lambda: pytest.fail('already HUD'), sleep_fn=lambda _: None)['ok']


def test_cancellation_during_wait_stops_verification():
    stopped = []
    events = []
    result = recover_once('need_verify', grab=lambda: events.append('frame') or object(),
        resolve=lambda _: where(), press_esc=lambda: events.append('esc'),
        sleep_fn=lambda _: stopped.append(1), cancelled=lambda: bool(stopped))
    assert result['status'] == 'cancel' and events == ['frame', 'esc']


def test_default_recovery_uses_guarded_esc_not_navigation(monkeypatch):
    sent = []
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda _: object())
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation',
        lambda _: where('main_hud') if sent else where())
    monkeypatch.setattr('mm_assist.safety.input.press_escape', lambda **k: sent.append(k))
    monkeypatch.setattr('mm_assist.life_execute.ui_act.prepare_to',
        lambda *a, **k: pytest.fail('recovery must bypass stuck click path'))
    monkeypatch.setattr('time.sleep', lambda _: None)
    win = SimpleNamespace(hwnd=7)
    assert recover_once('need_verify', win=win)['ok'] and sent == [{'win': win}]


def test_wrong_watch_generation_prevents_default_esc(monkeypatch):
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda _: object())
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation', lambda _: where())
    monkeypatch.setattr('mm_assist.life_plan.live_test.inspect_nav_watch_ready',
                        lambda **_: ({}, 'watch_ready_wrong_pid'))
    result = recover_once('need_verify', win=SimpleNamespace(hwnd=7))
    assert not result['ok'] and 'click_watch_not_ready' in result['status']


@pytest.mark.parametrize('mode', ['success', 'failure', 'pause', 'new_start', 'exception'])
def test_manual_recovery_uses_two_attempts_and_exits_thread(monkeypatch, mode):
    from mm_assist.life_plan import bar_ui, controller
    from mm_assist.life_plan.types import LifeSnapshot
    controller.reset_controller_for_tests()
    monkeypatch.setattr(bar_ui, '_PUMP', None)
    monkeypatch.setattr(bar_ui, 'repair_busy', lambda: False)
    monkeypatch.setattr(bar_ui, '_resolve_life_win', lambda *a: SimpleNamespace(hwnd=7))
    events = []
    hints = []
    monkeypatch.setattr(bar_ui, '_clear_pump_if_self', lambda: events.append('exit'))
    def recover(*a, **k):
        events.append('esc')
        if mode == 'pause':
            controller.pause_life()
            assert k['cancelled']()
        if mode == 'new_start':
            controller.start_life(snapshot=LifeSnapshot(ok=True), now=100)
            assert k['cancelled']()
        if mode == 'exception':
            raise RuntimeError('click_focus_lost')
        return {'ok': mode == 'success'}
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', recover)
    # This test exercises the real cancel flag for the manual worker.
    from mm_assist import tool_repair
    monkeypatch.setattr(tool_repair, 'op_cancelled', tool_repair._OP_CANCEL.is_set)
    assert bar_ui.start_escape_recovery(None, None, hints.append)
    thread = bar_ui._PUMP
    thread.join(timeout=2)
    assert not thread.is_alive() and events[-1] == 'exit'
    assert events.count('esc') == (2 if mode == 'failure' else 1)
    assert not controller.get_controller().running or mode == 'new_start'
