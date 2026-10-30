# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.perception.ocr import OcrLine
from mm_assist.perception import process_sense as mod
from mm_assist.platform.match import MatchHit, clear_template_cache

# 独立于产品配置的预期文案；入口与设施页共用，保留全部错设施反例。
FACILITY_LABELS = {
    "metal": "金屬加工", "wood": "木材加工", "leather": "皮革加工",
    "cloth": "布料加工", "medicine": "藥品加工", "food": "食材加工",
}


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))


def _line(text: str, x: int = 80, y: int = 160) -> OcrLine:
    return OcrLine(
        text,
        0.92,
        [[x, y], [x + 80, y], [x + 80, y + 20], [x, y + 20]],
    )


def _copy_catalog(tmp_path: Path) -> None:
    src = ROOT / "knowledge" / "templates" / "process" / "catalog.json"
    dest = tmp_path / "knowledge" / "templates" / "process" / "catalog.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    mod.clear_catalog_cache()


def _tiny_pattern() -> np.ndarray:
    tmpl = np.zeros((28, 52, 3), dtype=np.uint8)
    tmpl[:, :] = (12, 190, 70)
    tmpl[3:25, 6:46] = (210, 40, 170)
    tmpl[10:18, 20:32] = (40, 40, 240)
    return tmpl


def test_catalog_targets_and_chains():
    metal = {r["name"] + "|" + str(r.get("variant") or "") for r in mod.recipes_for_facility("metal")}
    wood = {r["name"] for r in mod.recipes_for_facility("wood")}
    cloth = {r["name"] for r in mod.recipes_for_facility("cloth")}
    assert "合金鋼錠|鋼錠" in metal
    assert "鋼錠|" in metal
    assert "鐵錠|iron_ore" in metal
    assert "鐵錠|ore" not in metal
    assert wood == {"高級木材", "木材+", "木材"}
    assert cloth == {"高級布料", "布料+", "布料"}
    assert any(r.get("target") for r in mod.recipes_for_facility("metal") if r["name"] == "合金鋼錠")
    assert any(r.get("target") for r in mod.recipes_for_facility("wood") if r["name"] == "高級木材")
    assert any(r.get("target") for r in mod.recipes_for_facility("cloth") if r["name"] == "高級布料")
    for fid in mod.HUB_CARD_IDS:
        assert fid in ("metal", "wood", "leather", "cloth", "medicine", "food")
    assert mod.RECIPE_FACILITIES == ("metal", "wood", "cloth", "medicine")
    med = {r["name"] for r in mod.recipes_for_facility("medicine")}
    assert med == {"淨淨蘑菇汁液", "淨淨蘑菇孢子", "增幅蘑菇汁液", "增幅蘑菇孢子",
                   "咻咻蘑菇汁液", "咻咻蘑菇孢子"}
    assert any(r.get("target") for r in mod.recipes_for_facility("medicine") if r["name"] == "淨淨蘑菇汁液")
    forbidden = json.dumps(
        json.loads((ROOT / "knowledge/templates/process/catalog.json").read_text(encoding="utf-8")),
        ensure_ascii=False,
    )
    assert "特殊鋼錠" not in forbidden
    assert "高級木材+" not in forbidden
    assert "process_metal_鐵錠_ore" not in {
        r["id"] for r in mod.recipes_for_facility("metal")
    }


def test_hub_card_ocr_fallback_all_six(monkeypatch):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    monkeypatch.setattr(mod, "_match_card", lambda *a, **k: None)
    for cid, text in FACILITY_LABELS.items():
        monkeypatch.setattr(mod, "_ocr_lines", lambda frame, roi, t=text: [_line(t, 200, 180)])
        hit = mod.locate_hub_card(img, cid)
        assert hit["ok"] is True, cid
        assert hit["src"] == "ocr_cht"
        assert hit["card_id"] == cid
        for other in FACILITY_LABELS:
            if other == cid:
                continue
            miss = mod.locate_hub_card(img, other)
            assert miss["ok"] is False, (cid, other)


def test_hub_card_ocr_rejects_short_medicine_glyph(monkeypatch):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    monkeypatch.setattr(mod, "_match_card", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_ocr_lines", lambda frame, roi: [_line("藥", 200, 180)])
    out = mod.locate_hub_card(img, "medicine")
    assert out["ok"] is False


def test_detect_facility_ocr_all_six_titles(monkeypatch):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    for fid, text in FACILITY_LABELS.items():
        def ocr(frame, roi, t=text):
            if roi == mod.TITLE_ROI:
                return [_line(t, 20, 10)]
            if roi == mod.FACILITY_LIST_ROI:
                return [_line("列表", 80, 160)]
            return []

        monkeypatch.setattr(mod, "_ocr_lines", ocr)
        page = mod.detect_facility_page(img, expected=fid)
        assert page["ok"] is True, fid
        assert page["facility"] == fid


def test_detect_facility_overview_go_equip_without_recipe_list(monkeypatch):
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)

    def ocr(frame, roi):
        if roi == mod.TITLE_ROI:
            return [_line("木材加工", 40, 20)]
        if roi == mod.PROCESS_BUTTON_ROI:
            return [_line("前往設備", 960, 700)]
        return []

    monkeypatch.setattr(mod, "_ocr_lines", ocr)
    monkeypatch.setattr(mod, "locate_claim_all", lambda *a, **k: {"ok": False})
    page = mod.detect_facility_page(img, expected="wood")
    assert page["ok"] is True
    assert page["facility"] == "wood"


def test_locate_go_equip_uses_title_side_roi(monkeypatch):
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)

    def ocr(frame, roi):
        if roi == mod.GO_EQUIP_ROI:
            return [_line("前往設備", 220, 320)]
        return []

    monkeypatch.setattr(mod, "_ocr_lines", ocr)
    hit = mod.locate_go_equip(img)
    assert hit["ok"] is True
    assert hit["click_client"] == [260, 330]


def test_detect_facility_overview_claim_all_without_recipe_list(monkeypatch):
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)

    def ocr(frame, roi):
        if roi == mod.TITLE_ROI:
            return [_line("木材加工", 40, 20)]
        return []

    monkeypatch.setattr(mod, "_ocr_lines", ocr)
    monkeypatch.setattr(
        mod, "locate_claim_all", lambda *a, **k: {"ok": True, "click_client": [160, 1040]}
    )
    page = mod.detect_facility_page(img, expected="wood")
    assert page["ok"] is True
    assert page["facility"] == "wood"


def test_locate_hub_card_real_wood_paste():
    card_path = ROOT / "knowledge" / "templates" / "craft_hub" / "card_wood.png"
    raw = np.fromfile(str(card_path), dtype=np.uint8)
    card = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    assert card is not None
    frame = np.full((720, 1280, 3), 36, dtype=np.uint8)
    ch, cw = card.shape[:2]
    x, y = 420, 160
    frame[y : y + ch, x : x + cw] = card
    out = mod.locate_hub_card(frame, "wood")
    assert out["ok"] is True
    assert out["src"] == "template"
    assert out["found"] is True
    assert out["click_client"]


def test_locate_hub_card_ambiguous(monkeypatch):
    img = np.zeros((400, 800, 3), dtype=np.uint8)

    def fake_match(frame, card_id, *, root=None):
        if card_id == "wood":
            return MatchHit(score=0.90, cx=10, cy=10, x=1, y=1, w=8, h=8, margin=0.01)
        if card_id == "metal":
            return MatchHit(score=0.89, cx=20, cy=10, x=11, y=1, w=8, h=8, margin=0.01)
        return None

    monkeypatch.setattr(mod, "_match_card", fake_match)
    out = mod.locate_hub_card(img, "wood")
    assert out["ok"] is False
    assert out["error"] == "ambiguous_card"


def test_locate_hub_card_far_apart_not_ambiguous(monkeypatch):
    img = np.zeros((400, 800, 3), dtype=np.uint8)

    def fake_match(frame, card_id, *, root=None):
        if card_id == "medicine":
            return MatchHit(score=0.90, cx=80, cy=200, x=70, y=190, w=20, h=20, margin=0.2)
        if card_id == "food":
            return MatchHit(score=0.89, cx=400, cy=200, x=390, y=190, w=20, h=20, margin=0.2)
        return None

    monkeypatch.setattr(mod, "_match_card", fake_match)
    out = mod.locate_hub_card(img, "medicine")
    assert out["ok"] is True
    assert out["src"] == "template"
    assert out["click_client"] == [80, 200]


def test_any_hub_card_needs_two_templates(monkeypatch):
    img = np.zeros((400, 800, 3), dtype=np.uint8)

    def one(frame, card_id, *, root=None):
        if card_id == "wood":
            return MatchHit(score=0.90, cx=80, cy=80, x=70, y=70, w=20, h=20, margin=0.2)
        return None

    monkeypatch.setattr(mod, "_match_card", one)
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [])
    assert mod._any_hub_card(img) is False

    def two(frame, card_id, *, root=None):
        if card_id in ("wood", "metal"):
            return MatchHit(score=0.90, cx=80, cy=80, x=70, y=70, w=20, h=20, margin=0.2)
        return None

    monkeypatch.setattr(mod, "_match_card", two)
    assert mod._any_hub_card(img) is True


def test_locate_hub_card_grid_when_others_hit(monkeypatch):
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)

    def fake_match(frame, card_id, *, root=None):
        if card_id == "metal":
            return MatchHit(score=0.90, cx=418, cy=403, x=400, y=380, w=40, h=40, margin=0.2)
        if card_id == "wood":
            return MatchHit(score=0.91, cx=830, cy=403, x=810, y=380, w=40, h=40, margin=0.2)
        return None

    monkeypatch.setattr(mod, "_match_card", fake_match)
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [])
    out = mod.locate_hub_card(img, "medicine")
    assert out["ok"] is True
    assert out["src"] == "hub_grid"
    assert out["click_client"] == [int(1920 * 0.218), int(1440 * 0.580)]


def test_locate_hub_card_grid_from_process_title(monkeypatch):
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    monkeypatch.setattr(mod, "_match_card", lambda *a, **k: None)

    def fake_ocr(frame, roi, **k):
        if roi == mod.TITLE_ROI:
            return [_line("加工", 80, 20)]
        return []

    monkeypatch.setattr(mod, "_ocr_lines", fake_ocr)
    out = mod.locate_hub_card(img, "medicine")
    assert out["ok"] is True
    assert out["src"] == "hub_grid"


def test_locate_process_recipe_missing_page():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = mod.locate_process_recipe(img, facility="wood", name="高級木材")
    assert out["ok"] is False
    assert out["error"] in ("k_assist_or_hud", "unknown_page", "missing_list_or_process")


def test_locate_process_recipe_template_priority(tmp_path, monkeypatch):
    _copy_catalog(tmp_path)
    tmpl = _tiny_pattern()
    png = tmp_path / "knowledge" / "templates" / "process" / "metal" / "合金鋼錠.png"
    _write_png(png, tmpl)
    clear_template_cache()

    frame = np.full((600, 800, 3), 20, dtype=np.uint8)
    th, tw = tmpl.shape[:2]
    x, y = 420, 90
    frame[y : y + th, x : x + tw] = tmpl

    monkeypatch.setattr(mod, "detect_facility_page", lambda *a, **k: {"ok": True, "facility": "metal"})
    monkeypatch.setattr(mod, "_gray_from_ocr", lambda *a, **k: False)

    out = mod.locate_process_recipe(
        frame, facility="metal", name="合金鋼錠", variant="鋼錠", root=tmp_path
    )
    assert out["ok"] is True
    assert out["src"] == "template"
    assert out["found"] is True


def test_locate_process_recipe_ocr_fallback(monkeypatch):
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    monkeypatch.setattr(mod, "detect_facility_page", lambda *a, **k: {"ok": True, "facility": "wood"})
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_ocr_lines", lambda frame, roi: [_line("高級木材")])
    out = mod.locate_process_recipe(img, facility="wood", name="高級木材")
    assert out["ok"] is True
    assert out["src"] == "ocr_cht"


def test_iron_need_variant():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    out = mod.locate_process_recipe(img, facility="metal", name="鐵錠")
    assert out["ok"] is False
    assert out["error"] == "need_variant"
    assert mod.recipe_click_variant("鐵錠") == "iron_ore"
    assert mod.recipe_click_variant("鋼錠") is None


def test_alloy_ingot_locates_without_catalog_variant(monkeypatch):
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    monkeypatch.setattr(mod, "detect_facility_page", lambda *a, **k: {"ok": True, "facility": "metal"})
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_ocr_lines", lambda frame, roi: [_line("合金鋼錠")])
    row = mod._find_recipe("metal", "合金鋼錠")
    assert row is not None
    out = mod.locate_process_recipe(img, facility="metal", name="合金鋼錠")
    assert out["ok"] is True
    assert out["src"] == "ocr_cht"
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    out = mod.locate_process_recipe(img, facility="metal", name="鐵錠")
    assert out["ok"] is False
    assert out["error"] == "need_variant"


def test_iron_variant_ocr_disambiguate(monkeypatch):
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    monkeypatch.setattr(mod, "detect_facility_page", lambda *a, **k: {"ok": True, "facility": "metal"})
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: None)

    monkeypatch.setattr(mod, "_ocr_lines", lambda frame, roi: [_line("鐵錠(鐵礦石)")])
    a = mod.locate_process_recipe(img, facility="metal", name="鐵錠", variant="iron_ore")
    b = mod.locate_process_recipe(img, facility="metal", name="鐵錠", variant="ore")
    assert a["ok"] is True
    assert b["ok"] is False
    assert b["error"] == "unknown_recipe"

    monkeypatch.setattr(mod, "_ocr_lines", lambda frame, roi: [_line("鐵錠(礦石)")])
    d = mod.locate_process_recipe(img, facility="metal", name="鐵錠", variant="iron_ore")
    assert d["ok"] is False


def test_detect_facility_unknown_and_half_and_mismatch(monkeypatch):
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    unk = mod.detect_facility_page(blank)
    assert unk["ok"] is False
    assert unk["error"] in ("k_assist_or_hud", "unknown_page")

    narrow = np.zeros((400, 48, 3), dtype=np.uint8)
    half = mod.detect_facility_page(narrow)
    assert half["ok"] is False
    assert half["error"] == "half_window"

    def ocr_assist(frame, roi):
        if roi == mod.TITLE_ROI:
            return [_line("協助", 20, 10)]
        return []

    monkeypatch.setattr(mod, "_ocr_lines", ocr_assist)
    ka = mod.detect_facility_page(blank)
    assert ka["ok"] is False
    assert ka["error"] == "k_assist_or_hud"

    def ocr_wood(frame, roi):
        if roi == mod.TITLE_ROI:
            return [_line("木材加工", 20, 10)]
        if roi == mod.FACILITY_LIST_ROI:
            return [_line("高級木材")]
        return []

    monkeypatch.setattr(mod, "_ocr_lines", ocr_wood)
    wrong = mod.detect_facility_page(blank, expected="metal")
    assert wrong["ok"] is False
    assert wrong["error"] == "wrong_facility"


def test_recipe_template_outside_roi_not_found(tmp_path, monkeypatch):
    _copy_catalog(tmp_path)
    tmpl = _tiny_pattern()
    png = tmp_path / "knowledge" / "templates" / "process" / "wood" / "高級木材.png"
    _write_png(png, tmpl)
    clear_template_cache()

    frame = np.full((600, 800, 3), 18, dtype=np.uint8)
    th, tw = tmpl.shape[:2]
    # FACILITY_LIST_ROI 从 x=0.50 起；放到左侧应 miss
    x, y = 40, 140
    frame[y : y + th, x : x + tw] = tmpl

    monkeypatch.setattr(mod, "detect_facility_page", lambda *a, **k: {"ok": True, "facility": "wood"})
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [])
    out = mod.locate_process_recipe(frame, facility="wood", name="高級木材", root=tmp_path)
    assert out["ok"] is False
    assert out["error"] == "not_found"


def test_ocr_disambiguates_wood_and_cloth_short_names(monkeypatch):
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    monkeypatch.setattr(mod, "detect_facility_page", lambda *a, **k: {"ok": True, "facility": "wood"})
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_ocr_lines", lambda frame, roi: [_line("高級木材")])
    short = mod.locate_process_recipe(img, facility="wood", name="木材")
    long = mod.locate_process_recipe(img, facility="wood", name="高級木材")
    assert short["ok"] is False
    assert long["ok"] is True

    monkeypatch.setattr(mod, "detect_facility_page", lambda *a, **k: {"ok": True, "facility": "cloth"})
    monkeypatch.setattr(mod, "_ocr_lines", lambda frame, roi: [_line("高級布料")])
    cloth_short = mod.locate_process_recipe(img, facility="cloth", name="布料")
    cloth_long = mod.locate_process_recipe(img, facility="cloth", name="高級布料")
    assert cloth_short["ok"] is False
    assert cloth_long["ok"] is True


def test_btn_here_away_helpers():
    from mm_assist.benchmarks.process_sense_bench import _btn_is_away_go, _btn_is_here

    assert _btn_is_here({"at_facility": True, "ok": True})
    assert _btn_is_here({"ok": True, "label": "加工", "at_facility": True})
    assert _btn_is_away_go({"ok": True, "at_facility": False, "label": "前往加工"})
    assert not _btn_is_away_go({"ok": False})
    here = {"ok": True, "label": "加工", "at_facility": True, "variant_id": "process_btn_full_here"}
    assert _btn_is_here(here)
    assert not _btn_is_away_go(here)


def test_auto_nav_requires_live_env(monkeypatch):
    from mm_assist.benchmarks.process_sense_bench import run_auto_nav_and_crop

    monkeypatch.delenv("MM_LIVE_BENCHMARK", raising=False)
    with pytest.raises(RuntimeError, match="MM_LIVE_BENCHMARK"):
        run_auto_nav_and_crop(None, None)  # type: ignore[arg-type]


def test_hub_visible_false_on_facility_page(monkeypatch):
    from mm_assist.benchmarks import process_sense_bench as bench

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(bench, "detect_facility_page", lambda *a, **k: {"ok": True, "facility": "wood"})
    monkeypatch.setattr(bench, "locate_hub_card", lambda *a, **k: {"ok": True})
    assert bench._hub_visible(img) is False
    monkeypatch.setattr(bench, "detect_facility_page", lambda *a, **k: {"ok": False})
    assert bench._hub_visible(img) is True


def test_expand_row_box_stays_in_list_roi():
    from mm_assist.benchmarks.process_sense_bench import expand_row_box
    from mm_assist.perception.process_sense import FACILITY_LIST_ROI

    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    box = expand_row_box(frame, {"x": 420, "y": 80, "w": 40, "h": 20}, FACILITY_LIST_ROI)
    x, y, w, h = box
    assert x >= int(0.38 * 800)
    assert y >= int(0.06 * 600)
    assert x + w <= int((0.38 + 0.60) * 800) + 1
    assert y + h <= int((0.06 + 0.55) * 600) + 1
    assert w >= 4 and h >= 4


def test_classify_process_btn_four_states():
    short_away = mod.classify_process_btn_state("前往加工", "所需材料 0/3")
    assert short_away["variant_id"] == "process_btn_short_away"
    assert short_away["at_facility"] is False
    full_away = mod.classify_process_btn_state("前往加工", "所需材料 3/3 1/1")
    assert full_away["variant_id"] == "process_btn_full_away"
    short_here = mod.classify_process_btn_state("加工", "所需材料 1/5")
    assert short_here["variant_id"] == "process_btn_short_here"
    assert short_here["at_facility"] is True
    full_here = mod.classify_process_btn_state("加工", "所需材料 5/5")
    assert full_here["variant_id"] == "process_btn_full_here"
    # 明细「加工時間 / 金屬加工設備」不得把设施前「加工」钮判成未知
    noisy = mod.classify_process_btn_state(
        "加工時間00:30金屬加工設備加工",
        "所需材料 5/5",
    )
    assert noisy["label"] == "加工"
    assert noisy["variant_id"] == "process_btn_full_here"
    assert noisy["at_facility"] is True


def test_materials_ok_splits_glued_have_need():
    blob = "所需材料16/316/4鐵錠煤炭"
    assert mod.infer_materials_ok(blob) is True
    glued_ten = "所需材料 24/1020/1"
    assert mod.infer_materials_ok(glued_ten) is True
    # 药师明细 37/30 11/3 粘成 37/30113，不能当成 37/30113 缺料
    spore = "15@26所需材料封印的遺忘碎片封37/30113浮淨蘑菇紙大成功機率15.0%"
    assert mod.infer_materials_ok(spore) is True


def test_claim_all_roi_covers_slot_label():
    # 1920x1440 上「全部領取」单排约 y=1204；五槽两行时上移约 y=1037
    h, w = 1440, 1920
    x, y, rw, rh = mod.CLAIM_ALL_ROI
    y0, y1 = int(y * h), int((y + rh) * h)
    x0, x1 = int(x * w), int((x + rw) * w)
    assert y0 <= 1037 <= y1
    assert y0 <= 1204 <= y1
    assert x0 <= 138 <= x1
    assert x0 <= 236 <= x1


def test_locate_claim_all_ocr(monkeypatch):
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    monkeypatch.setattr(mod, "claim_all_path", lambda **k: Path("no_such.png"))
    ln = _line("全部領取", 80, 1100)
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [ln])
    out = mod.locate_claim_all(img, prefer_ocr=True)
    assert out["ok"] is True
    assert out["at_facility_ui_same"] is True
    assert out["click_client"] == [120, 1110]


def test_locate_result_confirm_ocr(monkeypatch):
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    ln = _line("確認", 900, 1280)
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [ln])
    out = mod.locate_result_confirm(img)
    assert out["ok"] is True
    assert out["click_client"] == [940, 1290]


def test_locate_result_confirm_miss(monkeypatch):
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [])
    out = mod.locate_result_confirm(img)
    assert out["ok"] is False
    assert out.get("error") == "not_found"


def test_locate_common_popup_confirm_ocr(monkeypatch):
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    ln = _line("確認", 1080, 960)
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [ln])
    out = mod.locate_common_popup_confirm(img)
    assert out["ok"] is True
    assert out["click_client"] == [1120, 970]


def test_locate_process_button_template(tmp_path, monkeypatch):
    tmpl = _tiny_pattern()
    png = tmp_path / "knowledge" / "templates" / "process" / "process_btn.png"
    _write_png(png, tmpl)
    clear_template_cache()
    frame = np.full((600, 800, 3), 22, dtype=np.uint8)
    th, tw = tmpl.shape[:2]
    # PROCESS_BUTTON_ROI = (0.55, 0.78, 0.40, 0.18) → x>=440, y>=468
    x, y = 460, 480
    frame[y : y + th, x : x + tw] = tmpl
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [])
    out = mod.locate_process_button(frame, root=tmp_path)
    assert out["ok"] is True
    assert out["src"] == "template"


def test_locate_process_button_prefers_goto_over_time(monkeypatch):
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    monkeypatch.setattr(mod, "process_btn_path", lambda **k: Path("no_such.png"))
    time_ln = _line("加工時間需要30秒", 1271, 901)
    go_ln = _line("前往加工", 1400, 1338)
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [time_ln, go_ln])
    out = mod.locate_process_button(img, prefer_ocr=True)
    assert out["ok"] is True
    assert out["click_client"] == [1440, 1348]
    assert out["label"] == "前往加工"


def test_locate_process_button_prefers_short_over_time(monkeypatch):
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    monkeypatch.setattr(mod, "process_btn_path", lambda **k: Path("no_such.png"))
    time_ln = _line("加工時間需要30秒", 1271, 901)
    here_ln = _line("加工", 1400, 1338)
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [time_ln, here_ln])
    out = mod.locate_process_button(img, prefer_ocr=True)
    assert out["ok"] is True
    assert out["click_client"] == [1440, 1348]
    assert out["label"] == "加工"
    assert out["at_facility"] is True
    assert str(out.get("variant_id") or "").endswith("_here")


def _process_stance_bgr(name: str) -> np.ndarray:
    path = ROOT / "tests" / "fixtures" / "process_stance" / name
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def test_pick_template_prefers_away_when_substring_here_also_matches():
    away = MatchHit(score=0.97891, cx=1487, cy=1356, x=0, y=0, w=1, h=1, margin=0.9)
    here = MatchHit(score=0.92323, cx=1520, cy=1355, x=0, y=0, w=1, h=1, margin=0.9)
    picked = mod._pick_template_process_hit(
        [
            (here, Path("process_btn_short_here.png")),
            (away, Path("process_btn_short_away.png")),
        ]
    )
    assert picked is not None
    assert picked[0].cx == 1487
    assert picked[1].name.endswith("_away.png")


def test_locate_away_fixture_is_not_here(monkeypatch):
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [])
    out = mod.locate_process_button(_process_stance_bgr("away_stable.png"))
    assert out["ok"] is True
    assert out["label"] == "前往加工"
    assert out["at_facility"] is False
    assert out["click_client"] == [1487, 1356]
    assert Path(str(out["path"])).stem.endswith("_away")


def test_locate_here_fixtures_stay_here(monkeypatch):
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [])
    for name in ("here_stable.png", "here_formal.png"):
        out = mod.locate_process_button(_process_stance_bgr(name))
        assert out["ok"] is True, name
        assert out["label"] == "加工", name
        assert out["at_facility"] is True, name
        assert Path(str(out["path"])).stem.endswith("_here"), name
        assert out["click_client"][0] != 1520, name


def test_locate_facility_no_detail_has_no_process_btn(monkeypatch):
    monkeypatch.setattr(mod, "_ocr_lines", lambda *a, **k: [])
    out = mod.locate_process_button(_process_stance_bgr("facility_no_detail.png"))
    assert out["ok"] is False


@pytest.mark.parametrize('name',['wood_away_anim_first.png','wood_away_anim_fresh.png'])
def test_live_animated_away_label_is_not_its_here_suffix(monkeypatch,name):
    monkeypatch.setattr(mod,'_ocr_lines',lambda *a,**k:[])
    out=mod.locate_process_button(_process_stance_bgr(name))
    assert out['ok'] and out['at_facility'] is False
    assert out['label'] == '前往加工' and out['src'] == 'template'
    assert Path(out['path']).stem.endswith('_away')


def test_gather_material_ref_gap_vs_reuse():
    reuse = mod.gather_material_ref("鐵礦石")
    assert reuse["kind"] == "reuse"
    assert reuse["ok"] is True
    assert "鐵礦脈" in reuse["path"]

    gap = mod.gather_material_ref("原木")
    assert gap["ok"] is False
    assert gap["error"] == "gap"
    assert gap["kind"] == "gap"

    wool = mod.gather_material_ref("高級羊毛")
    assert wool["error"] == "gap"
