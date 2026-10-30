"""One pixel reader must keep fresh samples while the state-bar tick is slow."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from mm_assist.memory.readers.scene import MemSceneInfo
from mm_assist.perception import scene
from test_stp_current_state import monitor


def prepared(monkeypatch):
    mon = monitor(monkeypatch)
    mon._stop_pixel_min_s = .01
    mon._stop_pixel_busy = False
    return mon


def finish(mon):
    # Also clean up the pre-fix one-shot worker when recording red tests.
    close = getattr(mon, '_close_stop_pixel', None)
    if close:
        close()
    worker = getattr(mon, '_stop_pixel_thread', None)
    if worker:
        worker.join(1)
        assert not worker.is_alive()


def test_slow_main_tick_does_not_expire_the_only_pixel_sample(monkeypatch):
    mon = prepared(monkeypatch)
    two = threading.Event()
    calls = []
    def sample(win):
        calls.append(win.hwnd)
        if len(calls) == 2:
            monkeypatch.setattr(scene.time, 'time', lambda: 104.)
            two.set()
        return True
    monkeypatch.setattr(mon, '_sample_stop_pixel_now', sample)
    try:
        mon._maybe_kick_stop_pixel(SimpleNamespace(hwnd=123))
        assert two.wait(.5), 'Pixel refresh depended on another main tick'
        # The second capture began at 100 and is rightly stale; a subsequent
        # independent capture must renew it rather than treating unknown as 0.
        done = threading.Event()
        original = mon._apply_stop_pixel
        def apply(*args, **kwargs):
            original(*args, **kwargs)
            if mon._stop_image_ts == 104.:
                done.set()
        monkeypatch.setattr(mon, '_apply_stop_pixel', apply)
        assert done.wait(.5)
        assert mon._params_line(MemSceneInfo(ui_stack_ready=True)) == 'stp=1'
    finally:
        finish(mon)


def test_repeated_tick_keeps_one_capture_worker(monkeypatch):
    mon = prepared(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def sample(win):
        calls.append(win.hwnd)
        entered.set()
        release.wait(1)
        return True
    monkeypatch.setattr(mon, '_sample_stop_pixel_now', sample)
    try:
        mon._maybe_kick_stop_pixel(SimpleNamespace(hwnd=123))
        assert entered.wait(.5)
        worker = getattr(mon, '_stop_pixel_thread', None)
        for _ in range(5):
            mon._maybe_kick_stop_pixel(SimpleNamespace(hwnd=123))
        assert len(calls) == 1
        assert worker is not None and mon._stop_pixel_thread is worker
    finally:
        release.set()
        finish(mon)


def test_changed_target_discards_old_inflight_capture(monkeypatch):
    mon = prepared(monkeypatch)
    entered, release, current = threading.Event(), threading.Event(), threading.Event()
    def sample(win):
        if win.hwnd == 123:
            entered.set()
            release.wait(1)
            return True
        current.set()
        return False
    monkeypatch.setattr(mon, '_sample_stop_pixel_now', sample)
    try:
        mon._maybe_kick_stop_pixel(SimpleNamespace(hwnd=123))
        assert entered.wait(.5)
        mon._maybe_kick_stop_pixel(SimpleNamespace(hwnd=456))
        release.set()
        assert current.wait(.5)
        finish(mon)
        assert mon._stop_image is not True
    finally:
        release.set()
        finish(mon)


def test_missing_window_pauses_sampling_and_clears_old_fact(monkeypatch):
    mon = prepared(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def sample(win):
        entered.set()
        release.wait(1)
        return True
    monkeypatch.setattr(mon, '_sample_stop_pixel_now', sample)
    try:
        mon._maybe_kick_stop_pixel(SimpleNamespace(hwnd=123))
        assert entered.wait(.5)
        mon._maybe_kick_stop_pixel(None)
        release.set()
        finish(mon)
        assert mon._stop_image is None
        assert mon._params_line(MemSceneInfo(ui_stack_ready=True)) == 'stp=?'
    finally:
        release.set()
        finish(mon)


def test_capture_exception_does_not_strand_the_sampler(monkeypatch):
    mon = prepared(monkeypatch)
    recovered = threading.Event()
    calls = []
    def sample(win):
        calls.append(win.hwnd)
        if len(calls) == 1:
            raise RuntimeError('transient capture failure')
        recovered.set()
        return False
    monkeypatch.setattr(mon, '_sample_stop_pixel_now', sample)
    try:
        mon._maybe_kick_stop_pixel(SimpleNamespace(hwnd=123))
        assert recovered.wait(.5)
    finally:
        finish(mon)


def test_close_rejects_late_result_and_cannot_start_another_worker(monkeypatch):
    mon = prepared(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def sample(win):
        entered.set()
        release.wait(1)
        return True
    monkeypatch.setattr(mon, '_sample_stop_pixel_now', sample)
    try:
        mon._maybe_kick_stop_pixel(SimpleNamespace(hwnd=123))
        assert entered.wait(.5)
        close = getattr(mon, '_close_stop_pixel', None)
        assert close is not None
        close()
        release.set()
        finish(mon)
        assert mon._stop_image is None
        mon._maybe_kick_stop_pixel(SimpleNamespace(hwnd=123))
        assert not mon._stop_pixel_busy
    finally:
        release.set()
        finish(mon)


@pytest.mark.parametrize('fault', ['none', 'pid_changed', 'resized', 'image_size', 'stable'])
def test_pixel_sample_checks_window_association_before_adoption(monkeypatch, fault):
    mon = prepared(monkeypatch)
    win = SimpleNamespace(hwnd=123, left=0, top=0, width=320, height=240)
    windows = iter([win, SimpleNamespace(**{**vars(win), 'width': 321}) if fault == 'resized' else win])
    pids = iter([None if fault == 'none' else 101, 102 if fault == 'pid_changed' else 101])
    monkeypatch.setattr('mm_assist.platform.window.refresh_window', lambda w: next(windows))
    monkeypatch.setattr('mm_assist.memory.process.pid_from_hwnd', lambda hwnd: next(pids))
    image = np.full((240, 321 if fault == 'image_size' else 320, 3), 30, np.uint8)
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda *a, **k: image)
    monkeypatch.setattr('mm_assist.perception.hud_sense.detect_stop_icon', lambda *a, **k: False)
    assert mon._sample_stop_pixel_now(win) is (False if fault == 'stable' else None)


def test_previous_target_memory_result_cannot_override_current_pixels(monkeypatch):
    mon = prepared(monkeypatch)
    mon._icon_client = (1920, 1440)
    mon._stop_pixel_generation = 1
    def capture(**kwargs):
        mon._stop_pixel_generation = 2
        return SimpleNamespace()
    mon._action_reader = SimpleNamespace(capture=capture)
    monkeypatch.setattr('mm_assist.perception.ui_fine.summarize_hud_action',
                        lambda f: SimpleNamespace(stop_visible=False))
    mon._refresh_action_light()
    assert mon._stop_mem_last_ts == 90.
    assert mon._params_line(MemSceneInfo(ui_stack_ready=True)) == 'stp=1'
