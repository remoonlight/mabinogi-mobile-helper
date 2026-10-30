"""User's cloud-region map must ascend through the left breadcrumb."""
import hashlib
import json

import cv2
import numpy as np
import pytest

from mm_assist.config import AppConfig, ROOT
from mm_assist.nav import map_atlas as maps
from mm_assist.perception.ocr import OcrLine
from mm_assist.platform.window import WindowInfo


@pytest.fixture(autouse=True)
def current_map_parameters(monkeypatch):
    # User's 2026-10-09 fixed-click requirement replaces header localization;
    # each simulated map observation still needs independent current parameters.
    from mm_assist.life_execute.ui_logic import UiWhere
    import time
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation',
        lambda _: UiWhere('map', None, None, None, 'direct', stack_ready=True,
                          stack_sampled_ts=time.time()))


def frame():
    return cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/process_stance/cloud_region_map_20261007.png', dtype=np.uint8), cv2.IMREAD_COLOR)


def header_lines():
    # Actual v5 box merges the continent, minimap hint, and region; no slash.
    return [OcrLine('歐拉太陸地下雲海曠野', .94785,
                    [[81.5, 34.5], [425., 35.5], [425., 80.5], [81.5, 79.5]])]


def test_merged_cloud_header_is_region_in_navigation_and_scene():
    from mm_assist.perception import map_classify
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    lines = header_lines()
    assert maps._map_level(maps._blob(lines), lines, win) == 'region'
    assert not maps.is_continent_overview(lines, win)
    assert map_classify._map_level(maps._blob(lines), lines, win) == 'region'


def test_merged_no_slash_title_clicks_continent_segment_not_region(monkeypatch):
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    clicks = []
    monkeypatch.setattr(maps, '_click', lambda window, x, y, **kwargs: clicks.append((x, y)))
    assert maps._click_ulara_breadcrumb(header_lines(), win, frame=frame())
    assert len(clicks) == 1
    assert 88 <= clicks[0][0] < 174 and 38 <= clicks[0][1] < 80


def test_original_region_pixels_override_incomplete_continent_ocr(monkeypatch):
    from mm_assist.life_plan import node_inspect
    monkeypatch.setattr(node_inspect, '_ocr_blob', lambda *a, **k: '歐拉大陸')
    assert node_inspect.inspect_map_from_frame(frame()) == 'map'


def test_original_pixels_locate_breadcrumb_without_ocr(monkeypatch):
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    clicks = []
    monkeypatch.setattr(maps, '_click', lambda window, x, y, **kwargs: clicks.append((x, y)))
    assert maps._click_ulara_breadcrumb([], win, frame=frame())
    assert clicks == [(116, 59)]


def test_missing_header_has_no_blind_fallback(monkeypatch):
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    monkeypatch.setattr(maps, '_click_cal', lambda *a, **k: pytest.fail('no calibrated blind click'))
    monkeypatch.setattr(maps, '_click', lambda *a, **k: pytest.fail('no visible target'))
    assert not maps._click_ulara_breadcrumb([], win)


def test_real_ascent_uses_new_frame_and_stops_on_unchanged_region(monkeypatch):
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    image = frame()
    lines = header_lines()
    clicks = []
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **k: (win, image))
    monkeypatch.setattr(maps, '_click', lambda window, x, y, **k: clicks.append((x, y)))
    monkeypatch.setattr(maps, '_wait_action_effect', lambda *a, **k: (win, image, False, 0))
    monkeypatch.setattr(maps, 'snap', lambda *a, **k: (win, image, lines, None))
    monkeypatch.setattr('mm_assist.nav.map_localize.drag_until_near', lambda *a, **k: pytest.fail('region cannot be dragged with continent atlas'))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    out = maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=lines, img=image)
    assert clicks == [(116, 59)]
    assert not out['ok'] and out['stage'] == 'continent_region_stuck'


def test_current_map_parameters_allow_fixed_click_when_label_is_missing(monkeypatch):
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    monkeypatch.setattr(maps, 'ocr_image', lambda *a, **k: [])
    clicks = []
    monkeypatch.setattr(maps, '_click', lambda window, x, y, **k: clicks.append((x, y)))
    assert maps._click_ulara_breadcrumb(header_lines(), win, frame=np.zeros_like(frame()))
    assert clicks == [(116, 59)]


def test_verified_cloud_to_overview_uses_one_ascent_then_allows_atlas(monkeypatch):
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    initial = frame()
    overview = cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/process_stance/continent_bankor_upper.png', dtype=np.uint8), cv2.IMREAD_COLOR)
    after = [OcrLine('歐拉大陸', .99, [[80, 35], [230, 35], [230, 65], [80, 65]])]
    clicks = []
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **k: (win, overview if clicks else initial))
    monkeypatch.setattr(maps, '_click', lambda window, x, y, **k: clicks.append((x, y)))
    monkeypatch.setattr(maps, '_wait_action_effect', lambda *a, **k: (win, overview, True, 10))
    monkeypatch.setattr(maps, 'snap', lambda *a, **k: (win, overview, after, None))
    monkeypatch.setattr('mm_assist.nav.map_localize.load_continent_atlas', lambda *a: (None, None))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    out = maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=header_lines(), img=initial)
    assert clicks == [(116, 59)]
    assert out['stage'] == 'continent_atlas'  # Overview accepted before lookup.


def test_registered_header_source_crops_and_wrong_page_negatives():
    from mm_assist.perception.map_header import cloud_region_visible, locate_ulara_breadcrumb
    original = frame()
    for name in ('ulara_breadcrumb_label', 'cloud_region_title'):
        spec = json.loads((ROOT / f'knowledge/templates/map/{name}.json').read_text(encoding='utf-8'))
        for key, hash_key in (('source', 'source_sha256'), ('client_source', 'client_sha256'), ('template', 'template_sha256')):
            assert hashlib.sha256((ROOT / spec[key]).read_bytes()).hexdigest() == spec[hash_key]
        x, y, w, h = spec['client_px']
        template = cv2.imdecode(np.fromfile(ROOT / spec['template'], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(template, original[y:y+h, x:x+w])
    assert cloud_region_visible(original)
    for rel in ('continent_bankor_upper.png', 'laerte_region_split_title.png'):
        other = cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/process_stance' / rel, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert not cloud_region_visible(other)
    missing_close = original.copy()
    missing_close[:173, int(1912 * .90):] = 0
    assert not cloud_region_visible(missing_close)
    assert locate_ulara_breadcrumb(missing_close) is None
