"""Repair map ascent: current map parameters authorize one configured click."""
from types import SimpleNamespace
import time

import numpy as np
import pytest

from mm_assist.config import AppConfig
from mm_assist.nav import map_atlas as maps
from mm_assist.platform.window import WindowInfo


def where(node='map', ready=True):
    return SimpleNamespace(node=node, stack_ready=ready, stack_sampled_ts=time.time())


@pytest.mark.parametrize('width,height,expected', [(1912, 1442, (116, 59)), (956, 721, (58, 29))])
def test_new_map_parameters_click_fixed_point_without_header_ocr(monkeypatch, width, height, expected):
    win = WindowInfo(0, 'offline', 0, 0, width, height)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    observed, clicks = [], []
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation',
                        lambda image: observed.append(image) or where())
    monkeypatch.setattr(maps, 'ocr_image', lambda *a, **kw: pytest.fail('header OCR must not gate fixed click'))
    monkeypatch.setattr('mm_assist.perception.map_header.locate_ulara_breadcrumb',
                        lambda *a, **kw: pytest.fail('header pixels must not gate fixed click'))
    monkeypatch.setattr(maps, '_click', lambda window, x, y, **kw: clicks.append((x, y)))
    assert maps._click_ulara_breadcrumb([], win, frame=frame)
    assert len(observed) == 1 and observed[0] is frame
    assert clicks == [expected]


@pytest.mark.parametrize('node,ready,age', [('main_hud', True, 0), ('dialog', True, 0),
    ('overlay', True, 0), ('map', False, 0), ('unknown', False, 0), ('map', True, 10)])
def test_other_unready_or_stale_parameters_cannot_click(monkeypatch, node, ready, age):
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    state = where(node, ready)
    state.stack_sampled_ts -= age
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation', lambda _: state)
    monkeypatch.setattr(maps, '_click', lambda *a, **kw: pytest.fail('no map authorization'))
    assert not maps._click_ulara_breadcrumb([], win, frame=np.zeros((1442, 1912, 3), np.uint8))


def test_real_ascent_observes_change_after_parameter_fixed_click(monkeypatch):
    from mm_assist.perception.ocr import OcrLine
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    before = np.zeros((1442, 1912, 3), np.uint8)
    after = np.full_like(before, 100)
    clicks, observed = [], []
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **kw: (win, before))
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation', lambda _: where())
    monkeypatch.setattr(maps, 'ocr_image', lambda *a, **kw: [])
    monkeypatch.setattr(maps, '_click', lambda window, x, y, **kw: clicks.append((x, y)))
    def effect(cfg, window, image, **kwargs):
        assert clicks == [(116, 59)] and image is before
        observed.append(True)
        return win, after, True, 10
    monkeypatch.setattr(maps, '_wait_action_effect', effect)
    lines = [OcrLine('歐拉大陸', .99, [[80, 35], [230, 35], [230, 65], [80, 65]])]
    monkeypatch.setattr(maps, 'snap', lambda *a, **kw: (win, after, lines, None))
    _, _, level = maps.click_ulara_text_and_ocr(AppConfig(), win, [])
    assert observed == [True] and maps.click_ulara_text_and_ocr.last_changed
    assert level == 'continent'


@pytest.mark.parametrize('node', ['map_continent', 'map_dunbarton'])
def test_map_detail_parameters_use_same_fixed_position(monkeypatch, node):
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation', lambda _: where(node))
    clicks = []
    monkeypatch.setattr(maps, '_click', lambda window, x, y, **kw: clicks.append((x, y)))
    assert maps._click_ulara_breadcrumb([], win, frame=np.zeros((1442, 1912, 3), np.uint8))
    assert clicks == [(116, 59)]


def test_missing_config_or_changed_client_size_has_no_fallback(monkeypatch, tmp_path):
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation', lambda _: where())
    monkeypatch.setattr(maps, 'ROOT', tmp_path)
    monkeypatch.setattr(maps, '_click', lambda *a, **kw: pytest.fail('no fixed coordinate fallback'))
    assert not maps._click_ulara_breadcrumb([], win, frame=np.zeros((1442, 1912, 3), np.uint8))
    assert not maps._click_ulara_breadcrumb([], win, frame=np.zeros((721, 956, 3), np.uint8))


def test_config_point_is_registered_original_label_center():
    import hashlib
    import json
    from mm_assist.config import ROOT
    spec = json.loads((ROOT / maps.MAP_NAVIGATION_CONFIG_REL).read_text(encoding='utf-8'))['ulara_breadcrumb']
    label = json.loads((ROOT / 'knowledge/templates/map/ulara_breadcrumb_label.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT / spec['source_frame']).read_bytes()).hexdigest() == spec['source_sha256']
    assert spec['design_resolution'] == label['window_to_client'][2:]
    x, y, w, h = label['client_px']
    assert spec['client_xy'] == [x + w // 2, y + h // 2]


def test_already_map_parameters_do_not_reopen_minimap(monkeypatch):
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    image = np.zeros((1442, 1912, 3), np.uint8)
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **kw: (win, image))
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation', lambda _: where())
    monkeypatch.setattr(maps, '_focus_for_map_open', lambda *a: pytest.fail('map is already open'))
    monkeypatch.setattr(maps, '_click_map_open', lambda *a, **kw: pytest.fail('do not toggle map'))
    result = maps.open_map(AppConfig(), win, skip_precheck=True)
    assert result.state == 'map' and result.lines == []


@pytest.mark.parametrize('blob', ['', 'HomeESC每週目標'])
def test_repair_map_entry_reaches_fixed_click_when_ocr_cannot_confirm_open(monkeypatch, blob):
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    image = np.zeros((1442, 1912, 3), np.uint8)
    clicks = []
    monkeypatch.setattr(maps, 'dismiss_pre_map_blockers', lambda cfg, window: window)
    monkeypatch.setattr(maps, 'load_whitelist', lambda cfg: {})
    monkeypatch.setattr('mm_assist.nav.compat.in_production_mode', lambda: True)
    monkeypatch.setattr(maps, 'open_map', lambda *a, **kw: maps.MapSnapshot(state='unknown', lines=[], blob=blob))
    monkeypatch.setattr(maps, '_foreground_grab', lambda *a, **kw: (win, image))
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **kw: (win, image))
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation', lambda _: where())
    monkeypatch.setattr(maps, '_click', lambda window, x, y, **kw: clicks.append((x, y)))
    monkeypatch.setattr(maps, '_wait_action_effect', lambda *a, **kw: (win, image, False, 0))
    monkeypatch.setattr(maps, 'snap', lambda *a, **kw: (win, image, [], None))
    monkeypatch.setattr('mm_assist.nav.map_localize.drag_until_near',
                        lambda *a, **kw: pytest.fail('map parameters cannot authorize atlas dragging'))
    result = maps._map_goto_once(AppConfig(), win, region='杜巴頓', npc='奈麗絲', timeout_s=1,
                                idle_abort_s=1, npc_only=True, ban_smith_outside_town=True)
    assert clicks == [(116, 59)]
    assert not result['ok'] and result['stage'] == 'continent_open'
