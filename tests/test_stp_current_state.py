"""The stop bit is independent of panel identity and expires with its source."""
import threading
from types import SimpleNamespace

import pytest

from mm_assist.memory.readers.scene import MemSceneInfo
from mm_assist.perception import scene
from mm_assist.perception.ui_fine import HudActionSummary


@pytest.mark.parametrize('names', [(), ('HudUIStepState',), ('SideNavigationUIState',), ('AlteringFacilityPopupUIInstance',)])
def test_visible_stop_not_forced_zero_by_menu_or_facility(names):
    ms = MemSceneInfo(menu_open=True, ui_stack_ready=True, ui_stack_names=names)
    assert scene.format_ui_params(ms, action=HudActionSummary(stop_visible=True)) == 'stp=1'


def monitor(monkeypatch):
    # Exercise production fusion without initializing readers or game handles.
    mon = object.__new__(scene.SceneMonitor)
    mon._icon_lock = threading.Lock()
    mon._stop_mem = False
    mon._stop_mem_last_ts = 90.
    mon._stop_image = True
    mon._stop_image_ts = 99.9
    mon._action_summary = HudActionSummary(stop_visible=False)
    mon._action_last_ts = 90.
    mon._icon_summary = None
    mon._menu_nav_line = ''
    monkeypatch.setattr(scene.time, 'time', lambda: 100.)
    return mon


def test_old_memory_zero_cannot_hide_current_stop_image(monkeypatch):
    mon = monitor(monkeypatch)
    ms = MemSceneInfo(menu_open=False, ui_stack_ready=True)
    assert mon._params_line(ms) == 'stp=1'


def test_all_sources_expired_remain_unknown_instead_of_idle(monkeypatch):
    mon = monitor(monkeypatch)
    mon._stop_image_ts = 90.
    ms = MemSceneInfo(menu_open=False, ui_stack_ready=True)
    assert mon._params_line(ms) == 'stp=?'


def test_fresh_memory_still_has_priority_and_future_source_is_rejected(monkeypatch):
    mon = monitor(monkeypatch)
    ms = MemSceneInfo(menu_open=False, ui_stack_ready=True)
    mon._stop_mem_last_ts = 99.9
    assert mon._params_line(ms) == 'stp=0'
    mon._stop_mem_last_ts = 101.
    assert mon._params_line(ms) == 'stp=1'


def test_late_pixel_result_cannot_replace_newer_reading(monkeypatch):
    mon = monitor(monkeypatch)
    mon._apply_stop_pixel(False, sampled_at=90.)
    assert mon._stop_image is True and mon._stop_image_ts == 99.9


def test_black_capture_does_not_establish_idle(monkeypatch):
    import numpy as np
    mon = monitor(monkeypatch)
    monkeypatch.setattr('mm_assist.platform.window.refresh_window', lambda win: win)
    monkeypatch.setattr('mm_assist.memory.process.pid_from_hwnd', lambda hwnd: 101)
    monkeypatch.setattr('mm_assist.platform.capture.grab_client',
                        lambda *a, **k: np.zeros((1440, 1920, 3), np.uint8))
    assert mon._sample_stop_pixel_now(SimpleNamespace(hwnd=123, width=1920, height=1440)) is None


@pytest.mark.parametrize('error', [False, True])
def test_memory_refresh_failure_does_not_renew_old_zero(monkeypatch, error):
    mon = monitor(monkeypatch)
    mon._icon_client = (1920, 1440)
    def capture(**kwargs):
        if error:
            raise RuntimeError('reader unavailable')
        return None
    mon._action_reader = SimpleNamespace(capture=capture)
    mon._action_busy = True
    mon._refresh_action_light()
    assert mon._stop_mem_last_ts == 90. and not mon._action_busy
    assert mon._params_line(MemSceneInfo(menu_open=False, ui_stack_ready=True)) == 'stp=1'


def test_pixel_worker_runs_even_with_known_memory_and_exits(monkeypatch):
    mon = monitor(monkeypatch)
    mon._stop_pixel_min_s = .5
    mon._stop_pixel_busy = False
    sampled = threading.Event()
    completed = threading.Event()
    def sample(win):
        sampled.set()
        return True
    monkeypatch.setattr(mon, '_sample_stop_pixel_now', sample)
    original_apply = mon._apply_stop_pixel
    def apply(*args, **kwargs):
        original_apply(*args, **kwargs)
        completed.set()
    monkeypatch.setattr(mon, '_apply_stop_pixel', apply)
    original = threading.Thread
    workers = []
    def thread(*args, **kwargs):
        worker = original(*args, **kwargs)
        workers.append(worker)
        return worker
    monkeypatch.setattr(threading, 'Thread', thread)
    mon._maybe_kick_stop_pixel(SimpleNamespace(hwnd=123))
    try:
        assert sampled.wait(1) and completed.wait(1)
        assert mon._stop_image_ts == 100.
    finally:
        # Sampling now continues between ticks; explicitly close its lifetime.
        mon._close_stop_pixel()
    for worker in workers:
        worker.join(1)
        assert not worker.is_alive()
    assert len(workers) == 1 and not mon._stop_pixel_busy


def test_expired_state_payload_cannot_complete_running_gather(monkeypatch):
    from mm_assist.perception.state_ui import build_state_ui_payload
    from mm_assist.life_plan.action_finish import IdleSegments, StpSample
    mon = monitor(monkeypatch)
    ms = MemSceneInfo(menu_open=False, ui_stack_ready=True)
    mon._stop_image_ts = 90.
    payload = build_state_ui_payload(SimpleNamespace(ui='hud', ui_params=mon._params_line(ms)),
                                     ms, pid=123, now=100.)
    assert payload['stp'] is None
    idle = IdleSegments()
    assert not idle.observe(StpSample(0, 99., 123, True), now=10.)
    assert not idle.observe(StpSample(payload['stp'], 100., 123, True), now=15.)
    assert idle.zero_since is None
