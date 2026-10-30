"""A slow watch UI publication must not starve the existing stop sampler."""
import os
import threading
from types import SimpleNamespace

import pytest

from mm_assist.life_plan import action_finish, bar_ui, controller
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem
from mm_assist.perception import scene


def monitor(monkeypatch, wall):
    mon = object.__new__(scene.SceneMonitor)
    mon._icon_lock = threading.Lock()
    mon._stop_mem = None
    mon._stop_mem_last_ts = 0.
    mon._stop_image = False
    mon._stop_image_ts = wall[0]
    mon._stop_pixel_generation = 1
    mon._stop_pixel_closed = False
    mon._stop_pixel_target = SimpleNamespace(hwnd=123)
    mon.last_ms = SimpleNamespace(ui_stack_ready=True)
    monkeypatch.setattr(scene.time, 'time', lambda: wall[0])
    return mon


def test_real_pump_finishes_with_current_sampler_while_ui_feed_is_stale(monkeypatch):
    wall = [100.]
    mon = monitor(monkeypatch, wall)
    controller.reset_controller_for_tests()
    first = PlanItem('p1', 'gather', '采集 高級原木')
    next_item = PlanItem('p2', 'gather', '下一项')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [first, next_item])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1.)
    calls = []
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
                        lambda item, **k: calls.append(item.id) or ExecuteResult(True, 'done', action_issued=True))
    monkeypatch.setattr(bar_ui.time, 'monotonic', lambda: wall[0])
    # The observed failure: full state file publishes only every ~6 seconds.
    monkeypatch.setattr('mm_assist.perception.state_ui.read_state_ui', lambda: SimpleNamespace(
        stp=0, fresh=False, pid_alive=True, pid=os.getpid(), raw={'ts_unix': 94.}))
    with action_finish.completion_stp_source(mon.read_stop_sample):
        for elapsed in range(5):
            wall[0] = 100. + elapsed
            mon._stop_image_ts = wall[0]
            bar_ui.pump_life_once(None, None)
            assert controller.get_controller().current == (first if elapsed < 4 else next_item)
    assert calls == ['p1']


@pytest.mark.parametrize('fault', ['expired', 'future', 'unknown', 'closed', 'no_window', 'unready', 'exception'])
def test_bound_source_failure_never_falls_back_to_old_zero(monkeypatch, fault):
    wall = [100.]
    mon = monitor(monkeypatch, wall)
    monkeypatch.setattr('mm_assist.perception.state_ui.read_state_ui',
                        lambda: pytest.fail('bound source must be authoritative'))
    if fault == 'expired': mon._stop_image_ts = 97.
    if fault == 'future': mon._stop_image_ts = 101.
    if fault == 'unknown': mon._stop_image = None
    if fault == 'closed': mon._stop_pixel_closed = True
    if fault == 'no_window': mon._stop_pixel_target = None
    if fault == 'unready': mon.last_ms.ui_stack_ready = False
    def read():
        if fault == 'exception': raise RuntimeError('source unavailable')
        return mon.read_stop_sample()
    with action_finish.completion_stp_source(read):
        sample = action_finish.read_completion_stp()
        assert not sample.fresh and sample.stp is None


def test_source_uses_memory_priority_and_real_source_timestamp(monkeypatch):
    wall = [100.]
    mon = monitor(monkeypatch, wall)
    mon._stop_mem, mon._stop_mem_last_ts = True, 99.
    with action_finish.completion_stp_source(mon.read_stop_sample):
        sample = action_finish.read_completion_stp()
        assert sample.stp == 1 and sample.timestamp == 99. and sample.fresh
        wall[0] = 101.6
        mon._stop_image_ts = 101.5
        sample = action_finish.read_completion_stp()
        assert sample.stp == 0 and sample.timestamp == 101.5 and sample.fresh


def test_source_generation_and_duplicate_sample_cannot_complete_old_streak(monkeypatch):
    wall = [100.]
    mon = monitor(monkeypatch, wall)
    idle = action_finish.IdleSegments()
    with action_finish.completion_stp_source(mon.read_stop_sample):
        assert not idle.observe(action_finish.read_completion_stp(), 0.)
        wall[0] = mon._stop_image_ts = 102.
        assert not idle.observe(action_finish.read_completion_stp(), 2.)
        mon._stop_pixel_generation += 1
        wall[0] = mon._stop_image_ts = 104.
        assert not idle.observe(action_finish.read_completion_stp(), 4.)
        assert idle.zero_since == 4.
        assert not idle.observe(action_finish.read_completion_stp(), 6.)


def test_source_unbound_on_exception_and_file_fallback_remains(monkeypatch):
    wall = [100.]
    mon = monitor(monkeypatch, wall)
    monkeypatch.setattr('mm_assist.perception.state_ui.read_state_ui', lambda: None)
    with pytest.raises(RuntimeError):
        with action_finish.completion_stp_source(mon.read_stop_sample):
            assert action_finish.read_completion_stp().fresh
            raise RuntimeError('watch closed')
    assert not action_finish.read_completion_stp().fresh


@pytest.mark.parametrize('change', ['pause', 'new_start', 'exception'])
def test_bound_sample_cannot_advance_paused_new_or_failed_observation(monkeypatch, change):
    wall = [100.]
    mon = monitor(monkeypatch, wall)
    controller.reset_controller_for_tests()
    item = PlanItem('p1', 'gather', '采集 高級原木')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1.)
    controller.wait_action_completion(item.id, 1., ExecuteResult(True, 'done'), actions_complete=True)
    with action_finish.completion_stp_source(mon.read_stop_sample):
        for t in (0., 2.):
            wall[0] = mon._stop_image_ts = 100. + t
            assert not controller.observe_action_completion(action_finish.read_completion_stp(),
                now=t, item_id=item.id, run_started_at=1.)
    def read():
        if change == 'pause': controller.pause_life()
        elif change == 'new_start': controller.start_life(snapshot=LifeSnapshot(ok=True), now=2.)
        else: raise RuntimeError('sampler failed')
        return mon.read_stop_sample()
    wall[0] = mon._stop_image_ts = 104.
    with action_finish.completion_stp_source(read):
        assert not controller.observe_action_completion(action_finish.read_completion_stp(),
            now=4., item_id=item.id, run_started_at=1.)
    state = controller.get_controller()
    assert state.current == item
    if change == 'new_start':
        assert state.life_started_at == 2. and state.pending_action is None
        return
    assert state.pending_action.idle.zero_since is None
    if change == 'pause': controller.resume_life(now=2.)
    with action_finish.completion_stp_source(mon.read_stop_sample):
        for t in (5., 7., 9.):
            wall[0] = mon._stop_image_ts = 100. + t
            assert controller.observe_action_completion(action_finish.read_completion_stp(),
                now=t, item_id=item.id, run_started_at=1.) == (t == 9.)
