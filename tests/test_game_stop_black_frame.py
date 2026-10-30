"""A transient invalid capture blocks input without losing the pending action."""
from types import SimpleNamespace
import importlib

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.action_finish import StpSample
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem


@pytest.fixture
def pending(monkeypatch):
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()
    bar_ui.reset_recovery_for_tests()
    first = PlanItem('p11', 'gather', '採集 鐵礦石', recipe='鐵礦石')
    tail = PlanItem('p12', 'gather', '採集 鐵礦石', recipe='鐵礦石')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [first, tail])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    controller.wait_action_completion(first.id, 1000,
        ExecuteResult(False, 'action_pending', action_issued=True), actions_complete=True)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
        lambda *a, **k: pytest.fail('never resend an issued gather'))
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',
        lambda *a, **k: pytest.fail('invalid pixels never authorize recovery'))
    m = importlib.import_module('mm_assist.life_plan.game_stop')
    monkeypatch.setattr(m.GameStopWatch, '_evidence', lambda *a: None)
    clock = [10.0]
    monkeypatch.setattr(m.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(m.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    monkeypatch.setattr('mm_assist.life_plan.action_finish.read_completion_stp',
        lambda: StpSample(0, clock[0]+100, 1, True))
    yield m, clock, first, tail
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()


def black():
    # Exact first failed frame, retained as a lossless fixture.
    frame = cv2.imread(str(ROOT/'tests/fixtures/game_stop/black_20261014.png'))
    assert frame.shape == (1440, 1920, 3) and np.max(frame) == 0
    return frame


def hud():
    return cv2.imread(str(ROOT/'tests/fixtures/process_stance/repair_hud_failure_20261011.png'))


@pytest.mark.parametrize('tracking', [False, True])
def test_transient_black_rechecks_then_observes_same_action_without_dispatch(pending, monkeypatch, tracking):
    m, clock, first, tail = pending
    controller._STATE.pause_on_failure = tracking
    frames = iter([black(), hud()])
    seen = []
    def capture(*a):
        assert controller.get_controller().pending_action is not None
        frame = next(frames, hud())
        seen.append(frame)
        return frame
    monkeypatch.setattr(m, '_capture', capture)
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    state = controller.get_controller()
    assert len(seen) >= 2 and state.running and not state.paused
    assert state.pending_action and state.current == first and state.queue == [first, tail]
    assert state.pause_on_failure is tracking
    # Four seconds of fresh independent stp observations are still required.
    for elapsed in [2, 4]:
        clock[0] += 2
        bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    assert controller.get_controller().current == tail


@pytest.mark.parametrize('tracking', [False, True])
def test_persistent_black_is_bounded_and_preserves_action(pending, monkeypatch, tracking):
    m, clock, first, tail = pending
    controller._STATE.pause_on_failure = tracking
    frame = black()
    calls = []
    monkeypatch.setattr(m, '_capture', lambda *a: calls.append(clock[0]) or frame)
    start = clock[0]
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    state = controller.get_controller()
    assert 1 < len(calls) <= 24 and clock[0]-start <= 2.01
    assert state.paused is tracking and state.pending_action and state.current == first
    assert state.running is not tracking
    assert state.queue == [first, tail]
    monkeypatch.setattr(m, '_capture', lambda *a: hud())
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    assert controller.get_controller().paused is tracking  # Debug requires manual resume.


def test_black_followed_by_stop_dialog_still_pauses(pending, monkeypatch):
    m, _, first, tail = pending
    dialog = cv2.imread(str(ROOT/'tests/fixtures/game_stop/both_1920.png'))
    frames = iter([black(), dialog])
    monkeypatch.setattr(m, '_capture', lambda *a: next(frames, dialog))
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    state = controller.get_controller()
    assert state.paused and state.pending_action and state.queue == [first, tail]
    assert '遊戲已執行過久' in state.current_action and '連線已中斷' in state.current_action


@pytest.mark.parametrize('interrupt', ['pause', 'start', 'exception'])
def test_readonly_wait_cancellation_new_generation_and_exception(pending, monkeypatch, interrupt):
    m, _, first, tail = pending
    seen = []
    def capture(*a):
        seen.append(1)
        if len(seen) == 1:
            return black()
        if interrupt == 'pause':
            controller.pause_life()
        elif interrupt == 'start':
            controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000)
        else:
            raise RuntimeError('capture failed')
        return hud()
    monkeypatch.setattr(m, '_capture', capture)
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=123))
    assert len(seen) == 2
    state = controller.get_controller()
    if interrupt == 'start':
        assert state.life_started_at == 2000 and state.running and state.pending_action is None
    else:
        assert state.paused and state.pending_action and state.queue == [first, tail]


def test_black_at_input_boundary_never_allows_input_before_valid_pixels(pending, monkeypatch):
    m, _, _, _ = pending
    controller._STATE.pause_on_failure = True  # Explicit debug still pauses on invalid pixels.
    from mm_assist.safety.input import _guard_before_input, AbortRequested
    watch = m.GameStopWatch(None, SimpleNamespace(hwnd=123), controller.get_controller())
    calls = []
    monkeypatch.setattr(m, '_capture', lambda *a: calls.append(1) or black())
    token = m._WATCH.set(watch)
    try:
        with pytest.raises(AbortRequested, match='game_stop_paused'):
            _guard_before_input('click', SimpleNamespace(hwnd=123), method='sendinput', client_xy=(10, 10))
    finally:
        m._WATCH.reset(token)
    assert len(calls) == 1 and controller.get_controller().paused
