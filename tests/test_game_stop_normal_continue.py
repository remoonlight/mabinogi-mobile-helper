"""Invalid capture is not a termination dialog or a business success."""
from types import SimpleNamespace
import importlib

import pytest

from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem


@pytest.fixture
def running(monkeypatch):
    m = importlib.import_module('mm_assist.life_plan.game_stop')
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()
    first = PlanItem('p1', 'claim', '领取 木材', facility='wood')
    tail = PlanItem('p2', 'claim', '领取 金屬', facility='metal')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [first, tail])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    monkeypatch.setattr(m.GameStopWatch, '_evidence', lambda *a: None)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: '')
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly', lambda **k: None)
    # Retained current failure: no stop dialog could be identified from black pixels.
    from mm_assist.config import ROOT
    import cv2
    fixture = cv2.imread(str(ROOT/'tests/fixtures/game_stop/black_20261014.png'))
    import numpy as np
    assert fixture.shape == (1440, 1920, 3) and np.max(fixture) == 0
    clock = [10.0]
    monkeypatch.setattr(m.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(m.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    monkeypatch.setattr(m, '_capture', lambda *a: fixture)
    valid = np.full((720, 960, 3), 30, np.uint8)
    yield m, first, tail, valid
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()


def test_normal_unissued_failure_waits_for_pixels_then_skips_and_executes_tail(running, monkeypatch):
    m, first, tail, valid = running
    calls = []
    @m.guard_life_pump
    def pump(cfg, win):
        calls.append(controller.get_controller().current)
    win = SimpleNamespace(hwnd=123)
    pump(None, win)
    state = controller.get_controller()
    assert state.running and not state.paused and state.current == first and calls == []
    pump(None, win)  # Still black: no input and no drain of the queue.
    assert controller.get_controller().queue == [first, tail] and calls == []
    monkeypatch.setattr(m, '_capture', lambda *a: valid)
    pump(None, win)
    state = controller.get_controller()
    assert state.running and not state.paused and state.current == tail and calls == []
    assert state.skipped_claim_facilities == {'wood'} and not state.feed_spend
    pump(None, win)
    assert calls == [tail]


def test_normal_issued_completed_chain_keeps_observation_without_resending(running, monkeypatch):
    m, first, tail, valid = running
    controller.wait_action_completion(first.id, 1000,
        ExecuteResult(False, 'action_pending', action_issued=True), actions_complete=True)
    calls = []
    @m.guard_life_pump
    def observe(cfg, win):
        calls.append(controller.get_controller().pending_action)
    observe(None, SimpleNamespace(hwnd=123))
    state = controller.get_controller()
    assert state.running and not state.paused and state.pending_action and calls == []
    monkeypatch.setattr(m, '_capture', lambda *a: valid)
    observe(None, SimpleNamespace(hwnd=123))
    assert len(calls) == 1 and calls[0].actions_complete
    assert controller.get_controller().queue == [first, tail]


@pytest.mark.parametrize('change', ['pause', 'start'])
def test_blocked_old_item_does_not_skip_after_manual_pause_or_new_start(running, monkeypatch, change):
    m, first, tail, valid = running
    calls = []
    @m.guard_life_pump
    def pump(cfg, win):
        calls.append(1)
    pump(None, SimpleNamespace(hwnd=123))
    if change == 'pause':
        controller.pause_life()
    else:
        controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000)
    monkeypatch.setattr(m, '_capture', lambda *a: valid)
    pump(None, SimpleNamespace(hwnd=123))
    state = controller.get_controller()
    assert state.current == first and state.queue == [first, tail]
    assert state.paused if change == 'pause' else state.life_started_at == 2000
    assert calls == ([] if change == 'pause' else [1])


def test_normal_input_boundary_aborts_and_releases_without_permanent_pause(running, monkeypatch):
    m, _, _, _ = running
    from mm_assist.safety.input import check_abort, AbortRequested
    released = []
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: released.append(1))
    watch = m.GameStopWatch(None, SimpleNamespace(hwnd=123), controller.get_controller())
    token = m._WATCH.set(watch)
    try:
        with pytest.raises(AbortRequested):
            check_abort()
    finally:
        m._WATCH.reset(token)
    assert released and controller.get_controller().running and not controller.get_controller().paused


def test_real_pump_continues_to_tail_after_capture_recovers(running, monkeypatch):
    m, first, tail, valid = running
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda _: True)
    monkeypatch.setattr('mm_assist.life_plan.action_finish.read_completion_stp',
                        lambda: SimpleNamespace(fresh=True, stp=0))
    calls = []
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
        lambda item, **kwargs: calls.append(item) or ExecuteResult(False, 'cancel'))
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',
                        lambda *a, **k: pytest.fail('no Esc authorized by missing pixels'))
    win = SimpleNamespace(hwnd=123)
    bar_ui.pump_life_once(None, win)
    assert controller.get_controller().current == first and calls == []
    monkeypatch.setattr(m, '_capture', lambda *a: valid)
    bar_ui.pump_life_once(None, win)
    assert controller.get_controller().current == tail and calls == []
    bar_ui.pump_life_once(None, win)
    assert calls == [tail] and controller.get_controller().running


def test_old_frame_failure_handoff_cannot_skip_new_run_with_same_item(running):
    _, first, tail, _ = running
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000)
    bar_ui._note_fail_and_continue(win=None, item=first,
        reason='game_frame_unavailable', run_started_at=1000)
    assert controller.get_controller().queue == [first, tail]


def test_repeated_invalid_capture_only_saves_first_evidence(running, monkeypatch):
    m, _, _, _ = running
    saved = []
    monkeypatch.setattr(m.GameStopWatch, '_evidence', lambda *a: saved.append(1))
    @m.guard_life_pump
    def pump(cfg, win):
        pytest.fail('invalid frame cannot run business')
    for _ in range(3):
        pump(None, SimpleNamespace(hwnd=123))
    assert saved == [1] and controller.get_controller().running
