"""从 assist 迁入的地图纯函数测试。"""

from __future__ import annotations

import numpy as np
import pytest

from mm_assist.nav.map_atlas import (
    MapHit,
    _card_has_repair_npc,
    _is_smith_only,
    _looks_region_name,
    _map_level,
    _repair_hit_preference,
    classify_state,
    is_repair_facility_text,
    load_whitelist,
    merge_region_hits,
    prefer_town_avatar_click,
    scan_region_text_hits,
)
from mm_assist.config import AppConfig
from mm_assist.perception.ocr import OcrLine
from mm_assist.platform.window import WindowInfo


def _line(text: str, x: int, y: int, w: int = 80, h: int = 24) -> OcrLine:
    return OcrLine(
        text,
        0.9,
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
    )


def test_split_laerte_breadcrumb_is_region_not_overview(monkeypatch):
    import mm_assist.nav.map_atlas as maps
    win = WindowInfo(0, 'offline', 0, 0, 1912, 1442)
    # Original failure 20261004T042844Z: OCR can split at the slash/icon.
    lines = [_line('歐拉大陸', 80, 35, 140), _line('萊爾特丘陵', 270, 35, 190),
             _line('深層地下城', 1600, 210), _line('巴里1礦區', 100, 1190)]
    assert _map_level(maps._blob(lines), lines, win) == 'region'
    assert not maps.is_continent_overview(lines, win)


def test_looks_region_name():
    assert _looks_region_name("杜巴頓")
    assert not _looks_region_name("地圖")
    assert not _looks_region_name("120m")
    assert not _looks_region_name("前往此處")


def test_classify_continent():
    lines = [
        _line("地圖", 40, 20),
        _line("歐拉大陸", 200, 30),
        _line("杜巴頓", 400, 400),
        _line("佛格斯", 600, 500),
        _line("艾明馬夏", 800, 600),
    ]
    assert classify_state(lines) == "continent"


def test_classify_traveling():
    lines = [_line("停止", 900, 1200)]
    assert classify_state(lines) == "traveling"


def test_classify_npc_card():
    lines = [_line("奈麗絲", 900, 400), _line("前往此處", 1450, 1350)]
    assert classify_state(lines) == "npc_card"


def test_scan_region_text_hits():
    win = WindowInfo(0, "t", 0, 0, 1920, 1440)
    lines = [
        _line("地圖", 40, 20),
        _line("杜巴頓", 500, 500),
        _line("120m", 700, 500),
    ]
    hits = scan_region_text_hits(lines, win)
    assert any("杜巴" in h.text for h in hits)
    assert not any("120" in h.text for h in hits)


def test_merge_keeps_distant_icon():
    texts = [MapHit("text", "杜巴頓", 0.3, 0.4, 8)]
    icons = [
        MapHit("icon", "icon_0", 0.31, 0.41, 2),
        MapHit("icon", "icon_1", 0.7, 0.6, 2),
    ]
    m = merge_region_hits(texts, icons)
    assert len(m) == 2
    assert any(h.kind == "icon" and h.nx > 0.5 for h in m)


def test_classify_hud_blank():
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    img[10:100, 1700:1900] = 40
    assert classify_state([], img) in ("hud", "unknown")


def test_repair_facility_text():
    assert is_repair_facility_text("截匠铺")
    assert is_repair_facility_text("鐵匠鋪")
    assert is_repair_facility_text("奈麗絲")
    assert not is_repair_facility_text("武器")


def test_repair_hit_prefers_nerys_over_smith():
    assert _repair_hit_preference("奈麗絲") > _repair_hit_preference("鐵匠鋪")


def test_prefer_town_avatar_atlas_over_smith_ocr():
    ocr = MapHit(kind="avatar", text="鐵匠鋪", nx=0.47, ny=0.47, score=9.0)
    atlas = MapHit(kind="avatar", text="atlas:奈麗絲", nx=0.48, ny=0.52, score=0.57)
    chosen = prefer_town_avatar_click(ocr_hit=ocr, atlas_hit=atlas)
    assert chosen is atlas


def test_smith_only_and_card_npc():
    assert _is_smith_only("鐵匠鋪")
    assert not _is_smith_only("奈麗絲")
    assert _card_has_repair_npc("奈麗絲 前往此處", ("奈麗絲",))
    assert not _card_has_repair_npc("鐵匠鋪 前往此處", ("奈麗絲",))


def test_map_level_continent_when_duba_on_overview():
    win = WindowInfo(0, "t", 0, 0, 1920, 1440)
    lines = [
        _line("地圖", 40, 20),
        _line("歐拉大陸", 200, 30),
        _line("杜巴頓", 500, 500),
    ]
    b = "".join(L.text for L in lines)
    assert _map_level(b, lines, win) == "continent"
    from mm_assist.nav.map_atlas import is_continent_overview

    assert is_continent_overview(lines, win) is True


def test_map_level_region_when_breadcrumb_has_duba():
    win = WindowInfo(0, "t", 0, 0, 1920, 1440)
    lines = [
        _line("歐拉大陸／杜巴頓", 200, 40),
        _line("杜巴頓北門", 500, 500),
    ]
    b = "".join(L.text for L in lines)
    assert _map_level(b, lines, win) == "region"


@pytest.mark.parametrize("city", ["庫漢", "库漢", "佛格斯", "艾明馬夏"])
def test_map_level_secondary_city_breadcrumb_is_not_overview(city):
    from mm_assist.nav.map_atlas import is_continent_overview

    win = WindowInfo(0, "t", 0, 0, 1912, 1442)
    lines = [
        _line(f"歐拉大陸／{city}", 80, 40, w=290),
        _line("地下城", 600, 600),
        _line("希德狩場", 800, 800),
    ]
    assert is_continent_overview(lines, win) is False
    assert _map_level("".join(line.text for line in lines), lines, win) == "region"


@pytest.mark.parametrize("changed", [True, False])
def test_continent_entry_stops_when_breadcrumb_does_not_ascend(monkeypatch, changed):
    import mm_assist.nav.map_atlas as maps
    import mm_assist.nav.map_localize as localize

    win = WindowInfo(0, "t", 0, 0, 1912, 1442)
    img = np.zeros((1442, 1912, 3), dtype=np.uint8)
    lines = [_line("歐拉大陸／狼之森林", 80, 40, w=290)]
    clicks = []

    def breadcrumb(cfg, window, current):
        clicks.append(current)
        return window, lines, "region"

    breadcrumb.last_changed = changed
    monkeypatch.setattr(maps, "click_ulara_text_and_ocr", breadcrumb)
    monkeypatch.setattr(maps, "snap", lambda *a, **k: (win, img, lines, None))
    monkeypatch.setattr(maps, "_grab_map_frame", lambda *a, **k: (win, img))
    monkeypatch.setattr(maps, "open_map", lambda *a, **k: pytest.fail("must not reopen"))
    monkeypatch.setattr(maps, "ensure_map_closed", lambda *a, **k: pytest.fail("must not close/reopen"))
    monkeypatch.setattr(localize, "load_continent_atlas", lambda *a, **k: pytest.fail("detail is not atlas input"))
    monkeypatch.setattr("mm_assist.safety.input.release_mouse_buttons", lambda: None)

    out = maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=lines, img=img)
    assert len(clicks) == 1
    assert out["ok"] is False
    assert out["stage"] == "continent_region_stuck"


def test_city_detail_ascends_before_loading_continent_atlas(monkeypatch):
    import mm_assist.nav.map_atlas as maps

    win = WindowInfo(0, "t", 0, 0, 1912, 1442)
    img = np.zeros((1442, 1912, 3), dtype=np.uint8)
    detail = [_line("歐拉大陸／庫漢", 80, 40, w=290), _line("地下城", 600, 600)]
    overview = [_line("歐拉大陸", 80, 40), _line("杜巴頓", 600, 600)]
    clicks = []
    from mm_assist.life_execute.ui_logic import UiWhere
    import time
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation',
        lambda _: UiWhere('map', None, None, None, 'direct', stack_ready=True,
                          stack_sampled_ts=time.time()))
    monkeypatch.setattr(maps, "_grab_map_frame", lambda *a, **k: (win, img))
    monkeypatch.setattr(maps, "ocr_image", lambda *a, **k: detail)
    monkeypatch.setattr(maps, "_click", lambda window, x, y, **k: clicks.append((x, y)))
    monkeypatch.setattr(maps, "_wait_action_effect", lambda *a, **k: (win, img, True, 10))
    monkeypatch.setattr(maps, "snap", lambda *a, **k: (win, img, overview, None))

    def atlas(*args):
        assert len(clicks) == 1  # Real breadcrumb click precedes atlas lookup.
        return None, None

    monkeypatch.setattr("mm_assist.nav.map_localize.load_continent_atlas", atlas)
    monkeypatch.setattr("mm_assist.safety.input.release_mouse_buttons", lambda: None)
    out = maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=detail, img=img)
    assert clicks == [(116, 59)]
    assert out["stage"] == "continent_atlas"


def test_verified_town_to_region_can_ascend_next_layer(monkeypatch):
    import mm_assist.nav.map_atlas as maps

    win = WindowInfo(0, "t", 0, 0, 1912, 1442)
    img = np.zeros((1442, 1912, 3), dtype=np.uint8)
    town = [_line("城鎮地圖", 80, 40), _line("旅館", 600, 600)]
    region = [_line("歐拉大陸／狼之森林", 80, 40, w=290)]
    overview = [_line("歐拉大陸", 80, 40), _line("杜巴頓", 600, 600)]
    pages = iter([(region, "region"), (overview, "continent")])
    clicks = []

    def breadcrumb(cfg, window, current):
        clicks.append(current)
        after, level = next(pages)
        return window, after, level

    breadcrumb.last_changed = True
    monkeypatch.setattr(maps, "click_ulara_text_and_ocr", breadcrumb)
    monkeypatch.setattr(maps, "_grab_map_frame", lambda *a, **k: (win, img))
    monkeypatch.setattr("mm_assist.nav.map_localize.load_continent_atlas", lambda *a: (None, None))
    monkeypatch.setattr("mm_assist.safety.input.release_mouse_buttons", lambda: None)
    assert _map_level("".join(line.text for line in town), town, win) == "town"
    out = maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=town, img=img)
    assert len(clicks) == 2
    assert out["stage"] == "continent_atlas"


def test_map_level_facilities_beat_messy_continent_title():
    win = WindowInfo(0, "t", 0, 0, 1920, 1440)
    lines = [
        _line("欧拉大睦／杜巴頓", 200, 40),
        _line("鐵匠铺", 500, 120),
        _line("旅館", 700, 120),
    ]
    b = "".join(L.text for L in lines)
    assert _map_level(b, lines, win) == "town"


def test_town_atlas_files_load():
    from mm_assist.nav.map_localize import load_continent_atlas, load_town_atlas

    cfg = AppConfig()
    from mm_assist.nav.map_atlas import maps_root

    root = maps_root(cfg)
    atlas, meta = load_town_atlas(root, "dunbarton")
    assert atlas is not None
    assert meta is not None
    cont, cmeta = load_continent_atlas(root)
    assert cont is not None
    assert cmeta is not None


def test_town_pixel_click_npc_dry_run_no_nameerror(monkeypatch):
    import numpy as np

    from mm_assist.nav.map_atlas import town_pixel_click_npc
    from mm_assist.nav.map_localize import DragNearResult

    cfg = AppConfig()
    win = WindowInfo(0, "t", 0, 0, 1920, 1440)
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    lines = [
        _line("地圖", 40, 20),
        _line("杜巴頓", 200, 30),
        _line("鐵匠铺", 580, 900),
    ]

    monkeypatch.setattr(
        "mm_assist.perception.game_frame.ensure_game_frame", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "mm_assist.nav.map_atlas._grab_map_frame", lambda *a, **k: (win, img)
    )
    monkeypatch.setattr(
        "mm_assist.nav.map_atlas._snap_map_light", lambda *a, **k: (win, img, lines)
    )
    monkeypatch.setattr(
        "mm_assist.nav.map_atlas.snap", lambda *a, **k: (win, img, lines, None)
    )
    monkeypatch.setattr("mm_assist.nav.map_atlas._drag_map", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.nav.map_atlas._click", lambda *a, **k: None)
    monkeypatch.setattr(
        "mm_assist.nav.map_atlas._wait_frame_stable", lambda *a, **k: (win, img, True)
    )
    monkeypatch.setattr(
        "mm_assist.nav.map_localize.drag_until_near",
        lambda **k: DragNearResult(ok=False, loc=None, steps=0, note="dry"),
    )
    monkeypatch.setattr(
        "mm_assist.nav.map_atlas.find_npc_avatar", lambda *a, **k: None
    )

    out = town_pixel_click_npc(cfg, win)
    assert isinstance(out, dict)
    assert "ok" in out
    assert out.get("ok") is False


def test_walking_quest_is_not_dialog():
    from mm_assist.nav.map_atlas import _is_walking_to_npc_blob, _looks_dialog
    from mm_assist.platform.window import WindowInfo

    b = "前往奈麗絲處前往與奈麗絲交談中[每週目標]"
    win = WindowInfo(0, "t", 0, 0, 1920, 1440)
    assert _is_walking_to_npc_blob(b) is True
    assert _looks_dialog(b, [], win) is False
    mixed = "前往與奈麗絲交談中修理特化結束對話"
    assert _is_walking_to_npc_blob(mixed) is True
    assert _looks_dialog(mixed, [], win) is True
    assert _is_walking_to_npc_blob("跳過對話呵歡迎") is False
    cfg = AppConfig()
    wl = load_whitelist(cfg)
    names = [n.get("name") for n in wl.get("npcs") or []]
    assert "奈麗絲" in names
