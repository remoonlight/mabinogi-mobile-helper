"""An already-open map has no OCR in the lean path; pixels still prove its tier."""
import hashlib
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT, AppConfig
from mm_assist.nav import map_atlas as maps, map_localize as localize
from mm_assist.platform.window import WindowInfo


def frame(name):
    image = cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/process_stance' / name, np.uint8), 1)
    assert image is not None
    return image


def window(image):
    return WindowInfo(0, 'offline', 0, 0, image.shape[1], image.shape[0])


def forbid_input(monkeypatch):
    monkeypatch.setattr(maps, '_click', lambda *a, **k: pytest.fail('no breadcrumb input on proven overview'))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr(localize, 'drag_until_near',
        lambda **k: (_ for _ in ()).throw(RuntimeError('independent atlas navigation reached')))


def test_registered_failure_frames_are_exact_originals():
    manifest = json.loads((ROOT / 'tests/fixtures/process_stance/repair_map_empty_ocr_20261011.json').read_text('utf-8'))
    for row in manifest['frames']:
        assert hashlib.sha256((ROOT / row['fixture']).read_bytes()).hexdigest() == row['sha256']


@pytest.mark.parametrize('name', ['continent_empty_ocr_20261011.png', 'continent_bankor_upper.png'])
def test_empty_ocr_overview_uses_independent_pixels_before_any_ascent(monkeypatch, name):
    image = frame(name)
    forbid_input(monkeypatch)
    monkeypatch.setattr(maps, 'click_ulara_text_and_ocr', lambda *a: pytest.fail('OCR absence must not trigger ascent'))
    with pytest.raises(RuntimeError, match='independent atlas navigation reached'):
        maps.continent_pixel_enter_dubarton(AppConfig(), window(image), lines=[], img=image)


def test_real_lean_open_with_empty_ocr_reaches_continent_pixels(monkeypatch):
    image = frame('continent_empty_ocr_20261011.png')
    win = window(image)
    forbid_input(monkeypatch)
    monkeypatch.setattr(maps, 'dismiss_pre_map_blockers', lambda cfg, w: w)
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **k: (win, image))
    monkeypatch.setattr(maps, '_foreground_grab', lambda *a, **k: (win, image))
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation',
        lambda _: SimpleNamespace(node='map', stack_ready=True, stack_sampled_ts=time.time()))
    monkeypatch.setattr('mm_assist.nav.compat.in_production_mode', lambda: True)
    monkeypatch.setattr(maps, '_focus_for_map_open', lambda *a: pytest.fail('map already open'))
    monkeypatch.setattr(maps, 'ocr_image', lambda *a, **k: pytest.fail('lean path has no OCR'))
    with pytest.raises(RuntimeError, match='independent atlas navigation reached'):
        maps._map_goto_once(AppConfig(), win, region='杜巴頓', npc='奈麗絲',
            timeout_s=1, idle_abort_s=1, npc_only=True, ban_smith_outside_town=True)


def test_ascent_with_missing_text_checks_new_pixels_before_failing(monkeypatch):
    before = frame('bankor_town_map.png')
    after = frame('continent_empty_ocr_20261011.png')
    win = window(after)
    forbid_input(monkeypatch)
    calls = []
    def ascent(*args):
        calls.append(True)
        return win, [], 'unknown'
    monkeypatch.setattr(maps, 'click_ulara_text_and_ocr', ascent)
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **k: (win, after))
    with pytest.raises(RuntimeError, match='independent atlas navigation reached'):
        maps.continent_pixel_enter_dubarton(AppConfig(), window(before), lines=[], img=before)
    assert calls == [True]


@pytest.mark.parametrize('name', ['repair_hud_failure_20261011.png', 'wood_standalone.png',
    'bankor_town_map.png', 'cloud_region_map_20261007.png', 'laerte_region_split_title.png'])
def test_empty_ocr_wrong_page_and_detail_never_allow_continent_drag(monkeypatch, name):
    image = frame(name)
    win = window(image)
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr(maps, 'click_ulara_text_and_ocr', lambda *a: (win, [], 'unknown'))
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **k: (win, image))
    monkeypatch.setattr(localize, 'drag_until_near', lambda **k: pytest.fail('no independent overview proof'))
    result = maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=[], img=image)
    assert not result['ok']
    assert result['stage'] in {'continent_open', 'continent_region_stuck'}


@pytest.mark.parametrize('locked_frame', [True, False])
def test_map_auto_capture_bypasses_stale_shared_bitblt(monkeypatch, locked_frame):
    fresh = frame('continent_empty_ocr_20261011.png')
    stale = frame('repair_hud_failure_20261011.png')
    win = window(fresh)
    used = []
    class PrivateGrabber:
        def __init__(self, method):
            used.append(method)
        def grab_client(self, current):
            assert current == win
            return fresh
        def close(self):
            used.append('closed')
    fake_frame = SimpleNamespace(refresh=lambda: None, to_window=lambda: win, grab=lambda _: stale)
    monkeypatch.setattr('mm_assist.perception.game_frame.get_frame', lambda: fake_frame if locked_frame else None)
    monkeypatch.setattr('mm_assist.platform.capture.Grabber', PrivateGrabber)
    monkeypatch.setattr(maps, 'prepare_grabber', lambda _: None)
    monkeypatch.setattr(maps, 'grab_client', lambda *a, **k: stale)
    monkeypatch.setattr(maps, 'refresh_window', lambda _: win)
    monkeypatch.setattr(maps, '_steal_focus_allowed', lambda: False)
    current, image = maps._foreground_grab(AppConfig(), win, settle_s=0)
    assert current == win and image is fresh
    assert used == ['printwindow', 'closed']


@pytest.mark.parametrize('method', ['bitblt', 'mss', 'printwindow'])
def test_explicit_map_capture_method_is_preserved(monkeypatch, method):
    win = WindowInfo(0, 'offline', 0, 0, 1920, 1440)
    used = []
    class PrivateGrabber:
        def __init__(self, method): used.append(method)
        def grab_client(self, current): return 'fresh'
        def close(self): used.append('closed')
    monkeypatch.setattr('mm_assist.platform.capture.Grabber', PrivateGrabber)
    monkeypatch.setattr('mm_assist.platform.capture.configure_grabber', lambda *a: pytest.fail('do not change shared backend'))
    cfg = AppConfig()
    cfg.window.capture_method = method
    assert maps._grab_navigation_client(cfg, win) == 'fresh'
    assert used == [method, 'closed']


def test_map_capture_failure_closes_private_backend_without_stale_fallback(monkeypatch):
    used = []
    class PrivateGrabber:
        def __init__(self, method): used.append(method)
        def grab_client(self, current): raise RuntimeError('capture unavailable')
        def close(self): used.append('closed')
    monkeypatch.setattr('mm_assist.platform.capture.Grabber', PrivateGrabber)
    monkeypatch.setattr(maps, 'grab_client', lambda *a, **k: pytest.fail('no stale fallback'))
    with pytest.raises(RuntimeError, match='capture unavailable'):
        maps._grab_navigation_client(AppConfig(), WindowInfo(0, 'offline', 0, 0, 1920, 1440))
    assert used == ['printwindow', 'closed']
