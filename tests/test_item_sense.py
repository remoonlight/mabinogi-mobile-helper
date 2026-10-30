# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

from mm_assist.perception.item_sense import (
    crop_right_of_icon,
    items_for_skill,
    line_matches_name,
    locate_skill_item,
    ocr_name_matches,
    scroll_plan_for_item,
)
from mm_assist.platform.match import MatchHit


def test_ocr_name_matches_compact():
    assert ocr_name_matches(" 樹木 ", "樹木")
    assert ocr_name_matches("甲胄樹Lv", "甲胄樹")
    assert not ocr_name_matches("尖葉樹", "樹木")


@pytest.mark.parametrize('sid,name,fixture,y0,y1,mask0,mask1', [
    ('harvest', '黃豆', 'harvest_list_soybean.png', 1180, 1270, 1160, 1300),
    ('hoe', '防風草', 'hoe_list_parsnip.png', 1290, 1390, 1290, 1410),
])
def test_crop_live_frame_locates_correct_row_and_rejects_missing_target(
        sid, name, fixture, y0, y1, mask0, mask1):
    import cv2
    from mm_assist.config import ROOT

    path = ROOT / 'tests/fixtures/process_stance' / fixture
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None
    out = locate_skill_item(frame, skill_id=sid, name=name)
    assert out['ok'] and out['src'] == 'template+ocr'
    assert out['name'] == name
    x, y = out['click_client']
    assert 1190 <= x <= 1330 and y0 <= y <= y1
    # Leave other rows intact: none may stand in for the requested crop.
    absent = frame.copy()
    absent[mask0:mask1, 1000:1820] = 0
    missing = locate_skill_item(absent, skill_id=sid, name=name)
    assert not missing['ok']
    assert 'click_client' not in missing


def test_line_matches_name_traditional_variant():
    catalog = items_for_skill("woodcut")
    assert line_matches_name("粗壯樹", "粗壮樹", catalog)
    assert line_matches_name("粗壯樹 Lv.5", "粗壮樹", catalog)
    assert line_matches_name("甲冑樹", "甲胄樹", catalog)
    assert line_matches_name("甲冑樹 Lv.20", "甲胄樹", catalog)
    catalog = items_for_skill("herb")
    assert not line_matches_name("採集藥草", "藥草", catalog)
    assert line_matches_name("藥草", "藥草", catalog)
    assert line_matches_name("净净蘑菇", "淨淨蘑菇", catalog)
    assert line_matches_name("净净蘑菇 Lv.30", "淨淨蘑菇", catalog)
    assert line_matches_name("凈凈蘑菇", "淨淨蘑菇", catalog)
    assert line_matches_name("凈凈蘑菇 Lv.20", "淨淨蘑菇", catalog)
    catalog = items_for_skill("mine")
    assert "冰" in catalog
    assert "冰冷的光群" not in catalog
    assert line_matches_name("冰", "冰", catalog)
    assert not line_matches_name("冰冷的光群", "冰", catalog)


def test_line_matches_name_disambiguates_insect():
    catalog = items_for_skill("insect")
    assert line_matches_name("冰冷的光群", "冰冷的光群", catalog)
    assert not line_matches_name("冰冷的光群", "光群", catalog)


def test_line_matches_medicine_facility_names():
    catalog = ("淨淨蘑菇汁液", "淨淨蘑菇孢子")
    assert line_matches_name("净净蘑菇孢子", "淨淨蘑菇孢子", catalog)
    assert line_matches_name("凈淨蘑菇汁液", "淨淨蘑菇汁液", catalog)
    assert line_matches_name("15淨淨蘑菇孢子", "净净蘑菇孢子", catalog)
    assert not line_matches_name("净净蘑菇孢子", "淨淨蘑菇汁液", catalog)
    assert not line_matches_name("新芽蘑菇汁液", "淨淨蘑菇汁液", catalog)


def test_scroll_plan_down_for_later_item():
    catalog = items_for_skill("herb")
    visible = list(catalog[:2])
    target = catalog[5]
    plan = scroll_plan_for_item("herb", target, visible)
    assert plan.get("direction") == "down"
    assert int(plan.get("needs_scroll") or 0) > 0


def test_scroll_plan_forbidden_up():
    catalog = items_for_skill("woodcut")
    visible = [catalog[-1]]
    plan = scroll_plan_for_item("woodcut", catalog[0], visible)
    assert plan.get("error") == "scroll_up_forbidden"


def test_crop_right_of_icon_stays_inside():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    hit = MatchHit(score=0.9, cx=40, cy=20, x=10, y=10, w=20, h=20)
    strip = crop_right_of_icon(img, hit)
    assert strip.size > 0
    assert strip.shape[0] <= 100 and strip.shape[1] <= 200


def test_locate_missing_template(tmp_path):
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    out = locate_skill_item(img, skill_id="woodcut", name="樹木", root=tmp_path)
    assert out["ok"] is False
    assert out["error"] == "missing_template"


def test_locate_rejects_template_without_ocr(monkeypatch, tmp_path):
    from mm_assist.perception import item_sense as mod

    rel = mod.item_template_path("woodcut", "樹木", root=tmp_path)
    rel.parent.mkdir(parents=True, exist_ok=True)
    rel.write_bytes(b"fake")

    img = np.zeros((400, 800, 3), dtype=np.uint8)
    hit = MatchHit(score=0.9, cx=500, cy=200, x=480, y=180, w=40, h=40)
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: hit)
    monkeypatch.setattr(mod, "ocr_image", lambda *a, **k: [])
    monkeypatch.setattr(mod, "_ocr_lines_in_item_list", lambda *a, **k: [])
    monkeypatch.setattr(mod, "items_for_skill", lambda *a, **k: ("樹木",))

    out = locate_skill_item(img, skill_id="woodcut", name="樹木", root=tmp_path)
    assert out["ok"] is False
    assert out["error"] == "not_found"


def test_geometry_locate_row(monkeypatch):
    from mm_assist.perception import item_sense as mod

    catalog = items_for_skill("woodcut")
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)

    class _Line:
        def __init__(self, text: str):
            self.text = text

    def _fake_ocr(strip, **k):
        return [_Line("成材樹")]

    monkeypatch.setattr(mod, "ocr_image", _fake_ocr)
    out = mod._locate_by_row_geometry(
        img, item="成材樹", sid="woodcut", catalog=catalog
    )
    assert out is not None
    assert out.get("ok") is True
    assert out.get("src") == "geometry+ocr"


def test_enumerate_ocrs_item_list_once(monkeypatch):
    from mm_assist.perception import item_sense as mod
    from mm_assist.perception.ocr import OcrLine

    n = {"ocr": 0}
    lines = [OcrLine("藥草", 0.9, [[60, 40], [90, 40], [90, 55], [60, 55]])]

    def fake_ocr(_frame):
        n["ocr"] += 1
        return lines

    monkeypatch.setattr(mod, "items_for_skill", lambda *a, **k: ("藥草", "血紅藥草", "箭花"))
    monkeypatch.setattr(mod, "_ocr_lines_in_item_list", fake_ocr)
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: pytest.fail("no per-item template"))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    rows = mod.enumerate_visible_skill_items(img, skill_id="herb")
    assert n["ocr"] == 1
    assert any(r.get("ok") and r.get("name") == "藥草" for r in rows)


def test_locate_full_row_ocr_clickable(monkeypatch):
    from mm_assist.perception.ocr import OcrLine
    from mm_assist.perception import item_sense as mod

    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    rx, ry, _rw, _rh = mod.ITEM_LIST_ROI
    inset = mod.ITEM_LIST_EDGE_INSET
    y0 = int(h * (ry + inset + 0.08))
    y1 = y0 + 36
    x0 = int(w * (rx + 0.05))
    x1 = x0 + 120
    line = OcrLine("淨淨蘑菇", 0.9, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.platform.match.match_icon_fuzzy", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_ocr_lines_in_item_list", lambda *a, **k: [line])
    monkeypatch.setattr(mod, "_locate_by_row_geometry", lambda *a, **k: None)
    out = locate_skill_item(img, skill_id="herb", name="淨淨蘑菇")
    assert out.get("ok") is True
    assert out.get("click_client") == [int((x0 + x1) / 2), int((y0 + y1) / 2)]


def test_locate_clipped_row_not_found_scrolls_down(monkeypatch):
    from mm_assist.perception.ocr import OcrLine
    from mm_assist.perception import item_sense as mod

    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    rx, ry, rw, rh = mod.ITEM_LIST_ROI
    y1 = int(h * (ry + rh) - 4)
    y0 = y1 - 28
    x0 = int(w * (rx + 0.05))
    x1 = x0 + 120
    line = OcrLine("淨淨蘑菇", 0.9, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.platform.match.match_icon_fuzzy", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_ocr_lines_in_item_list", lambda *a, **k: [line])
    monkeypatch.setattr(mod, "_locate_by_row_geometry", lambda *a, **k: None)
    monkeypatch.setattr(
        mod,
        "enumerate_visible_skill_items",
        lambda *a, **k: [{"ok": True, "name": "躲躲花"}],
    )
    out = locate_skill_item(img, skill_id="herb", name="淨淨蘑菇")
    assert out.get("ok") is False
    assert out.get("reason") == "clipped_row"
    plan = out.get("scroll_plan") or {}
    assert plan.get("direction") == "down"


def test_locate_rejects_scar_flower_as_mushroom(monkeypatch, tmp_path):
    from mm_assist.perception.ocr import OcrLine
    from mm_assist.perception import item_sense as mod

    img = np.zeros((400, 800, 3), dtype=np.uint8)
    hit = MatchHit(score=0.9, cx=500, cy=200, x=480, y=180, w=40, h=40)
    rel = mod.item_template_path("herb", "淨淨蘑菇", root=tmp_path)
    rel.parent.mkdir(parents=True, exist_ok=True)
    rel.write_bytes(b"fake")
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: hit)
    monkeypatch.setattr("mm_assist.platform.match.match_icon_fuzzy", lambda *a, **k: None)
    monkeypatch.setattr(mod, "ocr_image", lambda *a, **k: [OcrLine("傷痕花", 0.9, None)])
    monkeypatch.setattr(mod, "_ocr_lines_in_item_list", lambda *a, **k: [])
    monkeypatch.setattr(mod, "_locate_by_row_geometry", lambda *a, **k: None)
    monkeypatch.setattr(mod, "items_for_skill", lambda *a, **k: ("淨淨蘑菇", "躲躲花"))
    monkeypatch.setattr(mod, "enumerate_visible_skill_items", lambda *a, **k: [])
    out = locate_skill_item(img, skill_id="herb", name="淨淨蘑菇", root=tmp_path)
    assert out.get("ok") is False
    assert out.get("error") in {"not_found", "missing_template"}


def test_scar_flower_detail_fixture_clicks_full_mushroom_not_clipped_flower(monkeypatch):
    import cv2
    from pathlib import Path

    from mm_assist.perception import item_sense as mod

    frame = cv2.imread(str(Path(__file__).parent / "fixtures/ui_benchmark/herb_detail_scar_flower.png"))
    assert frame is not None
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.platform.match.match_icon_fuzzy", lambda *a, **k: None)
    out = locate_skill_item(frame, skill_id="herb", name="淨淨蘑菇")
    assert out.get("ok") is True
    cy = int((out.get("click_client") or [0, 0])[1])
    h = int(frame.shape[0])
    _x0, iy0, _x1, iy1 = mod._list_inner_bounds(frame)
    assert iy0 <= cy <= iy1
    roi_bottom = h * (mod.ITEM_LIST_ROI[1] + mod.ITEM_LIST_ROI[3])
    assert cy < roi_bottom - h * mod.ITEM_LIST_EDGE_INSET


def test_live_170020_after_scroll_clicks_mushroom_not_scar():
    import cv2
    from pathlib import Path

    from mm_assist.perception import item_sense as mod
    from mm_assist.perception.item_sense import enumerate_visible_skill_items, locate_skill_item

    frame = cv2.imread(
        str(
            Path(__file__).resolve().parents[1]
            / "logs/life_tests/20260915T170020_0a7b9e04/011_20260915T090111Z.png"
        )
    )
    if frame is None:
        pytest.skip("170020 fixture missing")
    vis = {
        row["name"]: int((row.get("click_client") or [0, 0])[1])
        for row in enumerate_visible_skill_items(frame, skill_id="herb")
        if row.get("ok")
    }
    out = locate_skill_item(frame, skill_id="herb", name="淨淨蘑菇")
    assert out.get("ok") is True
    cy = int((out.get("click_client") or [0, 0])[1])
    mush_y = vis.get("淨淨蘑菇")
    scar_y = vis.get("傷痕花")
    assert mush_y is not None
    assert abs(cy - mush_y) <= 24
    if scar_y is not None:
        assert abs(cy - mush_y) < abs(cy - scar_y)
    _x0, iy0, _x1, iy1 = mod._list_inner_bounds(frame)
    assert iy0 <= cy <= iy1


def test_mushroom_below_fold_fixture_plans_down_scroll(monkeypatch):
    import cv2
    from pathlib import Path

    from mm_assist.perception import item_sense as mod
    from mm_assist.perception.item_sense import item_list_scroll_drag_points, locate_skill_item

    frame = cv2.imread(str(Path(__file__).parent / "fixtures/ui_benchmark/herb_list_mushroom_below.png"))
    assert frame is not None
    visible = ("藥草", "血紅藥草", "箭花", "魔力藥草")
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.platform.match.match_icon_fuzzy", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_ocr_lines_in_item_list", lambda *a, **k: [])
    monkeypatch.setattr(mod, "_locate_by_row_geometry", lambda *a, **k: None)
    monkeypatch.setattr(
        mod,
        "enumerate_visible_skill_items",
        lambda *a, **k: [{"ok": True, "name": n} for n in visible],
    )
    out = locate_skill_item(frame, skill_id="herb", name="淨淨蘑菇")
    assert out["ok"] is False
    plan = out.get("scroll_plan") or {}
    assert plan.get("direction") == "down"
    assert int(plan.get("needs_scroll") or 0) > 0
    (p0, p1) = item_list_scroll_drag_points(frame)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    rx, ry, rw, rh = mod.ITEM_LIST_ROI
    x0, x1 = int(w * rx), int(w * (rx + rw))
    y0, y1 = int(h * ry), int(h * (ry + rh))
    for x, y in (p0, p1):
        assert x0 <= x <= x1
        assert y0 <= y <= y1
    (s0, e0) = item_list_scroll_drag_points(frame, step=0)
    (s1, e1) = item_list_scroll_drag_points(frame, step=1)
    dy0 = abs(s0[1] - e0[1])
    dy1 = abs(s1[1] - e1[1])
    assert dy0 > 0 and dy1 > 0
    assert dy1 < dy0
    assert dy0 >= int(h * 0.24864)  # unchanged minimum distance after moving coordinates into config


def test_locate_copper_vein_not_white_copper_or_level_col():
    """下翻后的採礦列表：铜矿行不得点成白铜或右侧等级。"""
    import cv2
    from pathlib import Path

    from mm_assist.perception.item_sense import locate_skill_item

    frame = cv2.imdecode(
        np.fromfile(
            str(Path(__file__).parent / "fixtures" / "ui_benchmark" / "mine_list_copper.png"),
            dtype=np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )
    assert frame is not None
    copper = locate_skill_item(frame, skill_id="mine", name="銅礦脈")
    white = locate_skill_item(frame, skill_id="mine", name="白銅礦脈")
    assert copper.get("ok") is True
    cx, cy = copper["click_client"]
    h, w = frame.shape[:2]
    assert cx / w < 0.78
    if white.get("ok"):
        wy = int(white["click_client"][1])
        assert cy < wy - 20
    assert 900 <= cy <= 1100
