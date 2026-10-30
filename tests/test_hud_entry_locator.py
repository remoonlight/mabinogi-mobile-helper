from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mm_assist.perception.hud_entry_locator import (
    locate_hud_entry,
    safe_capture_id,
)


def _textured_square(size: int = 24) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(image, (2, 2), (size - 3, size - 3), (240, 240, 240), 2)
    cv2.line(image, (4, size // 2), (size - 5, size // 2), (40, 180, 250), 3)
    cv2.circle(image, (size // 2, size // 2), 4, (255, 80, 30), -1)
    return image


def test_template_locator_returns_client_normalized_and_screen_coordinates(tmp_path: Path):
    template = _textured_square()
    template_path = tmp_path / "entry.png"
    assert cv2.imwrite(str(template_path), template)
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    frame[40:64, 90:114] = template
    entry = {
        "id": "menu",
        "locator": {
            "detector": "template",
            "template": str(template_path),
            "threshold": 0.8,
            "min_margin": 0.02,
            "maturity": "accepted",
        },
    }

    result = locate_hud_entry(entry, frame, screen_origin=(300, 200))

    assert result["feature_ok"] is True
    assert result["position_ok"] is True
    assert result["accepted"] is True
    assert result["click"]["client"] == [102, 52]
    assert result["click"]["screen"] == [402, 252]
    assert result["click"]["norm"] == pytest.approx([0.51, 52 / 120])


def test_missing_template_exposes_only_provisional_roi_point(tmp_path: Path):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    entry = {
        "id": "menu",
        "locator": {
            "detector": "template",
            "template": "missing.png",
            "roi": [0.8, 0.0, 0.2, 0.2],
            "maturity": "calibration",
        },
    }

    result = locate_hud_entry(entry, frame, root=tmp_path)

    assert result["status"] == "calibration_required"
    assert result["feature_ok"] is False
    assert result["position_ok"] is True
    assert result["provisional"] is True
    assert result["accepted"] is False
    assert result["click"]["client"] == [180, 10]


def test_stop_shape_locator_returns_white_square_center():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.circle(frame, (260, 170), 24, (90, 190, 110), -1)
    cv2.rectangle(frame, (251, 161), (269, 179), (245, 245, 245), -1)
    entry = {
        "id": "stop",
        "locator": {
            "detector": "stop_shape",
            "roi": [0.7, 0.7, 0.3, 0.3],
            "maturity": "accepted",
        },
    }

    result = locate_hud_entry(entry, frame)

    assert result["feature_ok"] is True
    assert result["accepted"] is True
    assert result["click"]["client"] == pytest.approx([260, 170], abs=1)


@pytest.mark.parametrize(
    "value",
    ["../escape", "a/b", "a\\b", "", "CON", "name.", "name ", "x" * 65],
)
def test_capture_id_rejects_unsafe_paths(value: str):
    with pytest.raises(ValueError):
        safe_capture_id(value)


def test_capture_id_accepts_safe_slug():
    assert safe_capture_id("hud-menu_20260823.1") == "hud-menu_20260823.1"


def test_prefer_roi_center_uses_roi_midpoint_not_template():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    entry = {
        "id": "menu",
        "locator": {
            "detector": "template",
            "template": "missing.png",
            "roi": [0.5, 0.0, 0.2, 0.2],
            "maturity": "calibration",
        },
    }

    result = locate_hud_entry(entry, frame, prefer_roi_center=True, screen_origin=(10, 20))

    assert result["reason"] == "roi_center"
    assert result["provisional"] is True
    assert result["feature_ok"] is False
    assert result["click"]["client"] == [120, 10]
    assert result["click"]["screen"] == [130, 30]


def _staggered_menu(size: int = 48) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 2 - 2, (42, 42, 42), -1)
    lengths = (int(size * 0.62), int(size * 0.48), int(size * 0.34))
    ys = (int(size * 0.32), int(size * 0.50), int(size * 0.68))
    x0 = int(size * 0.22)
    for length, y in zip(lengths, ys, strict=True):
        cv2.line(img, (x0, y), (x0 + length, y), (240, 240, 240), 3, cv2.LINE_AA)
    return img


def _equal_menu_with_badge(size: int = 48) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 2 - 2, (42, 42, 42), -1)
    x0, x1 = int(size * 0.22), int(size * 0.78)
    for y in (int(size * 0.32), int(size * 0.50), int(size * 0.68)):
        cv2.line(img, (x0, y), (x1, y), (240, 240, 240), 3, cv2.LINE_AA)
    cv2.circle(img, (size - 7, 7), 6, (0, 0, 255), -1)
    return img


def test_menu_locator_accepts_equal_bars_and_red_badge(tmp_path: Path):
    template_path = tmp_path / "menu.png"
    assert cv2.imwrite(str(template_path), _staggered_menu())
    icon = _equal_menu_with_badge()
    frame = np.zeros((200, 360, 3), dtype=np.uint8)
    frame[8:56, 300:348] = icon
    entry = {
        "id": "menu",
        "locator": {
            "detector": "template",
            "template": str(template_path),
            "roi": [0.78, 0.0, 0.22, 0.16],
            "threshold": 0.58,
            "match": "fuzzy",
            "maturity": "accepted",
        },
    }

    result = locate_hud_entry(entry, frame)

    assert result["feature_ok"] is True
    assert result["status"] == "matched"
    cx, cy = result["click"]["client"]
    assert 300 <= cx <= 348
    assert 8 <= cy <= 56


def test_menu_locator_prefers_hamburger_over_fuzzy_false_positive(tmp_path: Path):
    template_path = tmp_path / "menu.png"
    assert cv2.imwrite(str(template_path), _staggered_menu())
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[10:58, 10:58] = (180, 180, 40)
    icon = _equal_menu_with_badge()
    frame[8:56, 330:378] = icon
    entry = {
        "id": "menu",
        "locator": {
            "detector": "template",
            "template": str(template_path),
            "roi": [0.0, 0.0, 1.0, 0.4],
            "threshold": 0.35,
            "match": "fuzzy",
            "maturity": "accepted",
        },
    }
    result = locate_hud_entry(entry, frame)
    assert result["feature_ok"] is True
    cx, cy = result["click"]["client"]
    assert 330 <= cx <= 378
    assert 8 <= cy <= 56


def test_menu_locator_rejects_hit_left_of_esc_band(tmp_path: Path, monkeypatch):
    """实机 (1620,55) 左侧假命中不得当菜单；cx 须 ≥ 83% 客户区宽。"""
    from types import SimpleNamespace

    template_path = tmp_path / "menu.png"
    assert cv2.imwrite(str(template_path), _equal_menu_with_badge())
    frame = np.zeros((1440, 1920, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "mm_assist.perception.hud_entry_locator.find_hamburger_bars",
        lambda *a, **k: SimpleNamespace(
            x=1580, y=30, w=80, h=50, cx=1620, cy=55, score=0.9, margin=0.1
        ),
    )
    entry = {
        "id": "menu",
        "locator": {
            "detector": "template",
            "template": str(template_path),
            "roi": [0.83, 0.0, 0.12, 0.12],
            "threshold": 0.58,
            "match": "fuzzy",
            "maturity": "accepted",
        },
    }
    ok = locate_hud_entry(entry, frame)
    assert ok["feature_ok"] is True
    assert ok["click"]["client"][0] >= int(0.83 * 1920)

    monkeypatch.setattr(
        "mm_assist.perception.hud_entry_locator.find_hamburger_bars",
        lambda *a, **k: SimpleNamespace(
            x=1480, y=30, w=80, h=50, cx=1520, cy=55, score=0.9, margin=0.1
        ),
    )
    monkeypatch.setattr(
        "mm_assist.perception.hud_entry_locator.match_icon_fuzzy",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mm_assist.perception.hud_entry_locator.match_template",
        lambda *a, **k: None,
    )
    bad = locate_hud_entry(entry, frame)
    assert bad["feature_ok"] is False
    assert bad["status"] == "no_match"

    monkeypatch.setattr(
        "mm_assist.perception.hud_entry_locator.find_hamburger_bars",
        lambda *a, **k: SimpleNamespace(
            x=1679, y=90, w=80, h=50, cx=1719, cy=122, score=0.9, margin=0.1
        ),
    )
    mapish = locate_hud_entry(entry, frame)
    assert mapish["feature_ok"] is False
    assert mapish["status"] == "no_match"


def _bag_entry(template_path: Path, roi: list[float]) -> dict:
    return {
        "id": "bag",
        "locator": {
            "detector": "template",
            "template": str(template_path),
            "roi": roi,
            "threshold": 0.62,
            "match": "fuzzy",
            "maturity": "calibration",
        },
    }


def test_bag_locator_hits_bottom_right_icon(tmp_path: Path):
    template = _textured_square(24)
    template_path = tmp_path / "bag.png"
    assert cv2.imwrite(str(template_path), template)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    # 归一化中心约 (0.845, 0.93)，与实机右下背包带一致
    frame[267:291, 326:350] = template
    result = locate_hud_entry(
        _bag_entry(template_path, [0.78, 0.86, 0.14, 0.14]), frame
    )
    assert result["feature_ok"] is True
    assert result["status"] == "matched"
    cx, cy = result["click"]["client"]
    assert 326 <= cx <= 350
    assert 267 <= cy <= 291


def test_bag_locator_rejects_stop_and_top_right_false_hits(tmp_path: Path):
    template = _textured_square(24)
    template_path = tmp_path / "bag.png"
    assert cv2.imwrite(str(template_path), template)
    wide_roi = [0.70, 0.0, 0.30, 1.0]

    top = np.zeros((300, 400, 3), dtype=np.uint8)
    top[8:32, 326:350] = template
    top_hit = locate_hud_entry(_bag_entry(template_path, wide_roi), top)
    assert top_hit["feature_ok"] is False
    assert top_hit["status"] == "no_match"

    stop = np.zeros((300, 400, 3), dtype=np.uint8)
    stop[264:288, 364:388] = template
    stop_hit = locate_hud_entry(_bag_entry(template_path, wide_roi), stop)
    assert stop_hit["feature_ok"] is False
    assert stop_hit["status"] == "no_match"


def test_menu_locator_hits_lairt_claim_fail_frame():
    """claim:metal 萊爾特丘陵失败帧：菜单在 ESC 左侧，相关低于旧阈值 0.58。"""
    from mm_assist.config import ROOT
    from mm_assist.jobs.hud_entries import entry_by_id, load_main_hud_catalog

    frame_path = ROOT / "tests" / "fixtures" / "ui_benchmark" / "hud_menu_lairt_claim.png"
    raw = np.fromfile(str(frame_path), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    assert frame is not None
    entry = dict(entry_by_id(load_main_hud_catalog(), "menu"))
    result = locate_hud_entry(entry, frame, root=ROOT)
    assert result["feature_ok"] is True
    nx, ny = result["click"]["norm"]
    assert 0.83 <= nx < 0.87
    assert ny < 0.08


def test_bag_locator_matches_20260824_live_source_frame():
    from mm_assist.config import ROOT
    from mm_assist.jobs.hud_entries import entry_by_id, load_main_hud_catalog

    frame_path = (
        ROOT
        / "knowledge"
        / "captures"
        / "transitions"
        / "20260824T084921Z"
        / "main_hud_bag_inventory"
        / "source.png"
    )
    if not frame_path.is_file():
        pytest.skip("missing archived bag source frame")
    frame = cv2.imread(str(frame_path))
    assert frame is not None
    entry = dict(entry_by_id(load_main_hud_catalog(), "bag"))
    result = locate_hud_entry(entry, frame, root=ROOT)
    assert result["feature_ok"] is True
    nx, ny = result["click"]["norm"]
    assert 0.80 <= nx <= 0.89
    assert 0.86 <= ny <= 0.98


@pytest.mark.parametrize("reject_fuzzy", [False, True])
def test_menu_rejected_map_texture_does_not_mask_real_icon(monkeypatch, reject_fuzzy):
    from mm_assist.config import ROOT
    from mm_assist.jobs.hud_entries import entry_by_id, load_main_hud_catalog
    from mm_assist.platform.match import find_hamburger_bars
    from mm_assist.perception import hud_entry_locator as locator

    path = ROOT / "tests/fixtures/ui_benchmark/hud_menu_river_20260930.png"
    frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    entry = dict(entry_by_id(load_main_hud_catalog(), "menu"))
    false_hit = find_hamburger_bars(frame, roi=tuple(entry["locator"]["roi"]))
    assert false_hit is not None and false_hit.cx / frame.shape[1] >= 0.87
    if reject_fuzzy:
        monkeypatch.setattr(locator, "match_icon_fuzzy", lambda *a, **k: false_hit)

    result = locate_hud_entry(entry, frame, root=ROOT)

    assert result["feature_ok"] is True
    x, y = result["click"]["client"]
    assert 1586 <= x <= 1642 and 25 <= y <= 85


def test_menu_all_out_of_position_candidates_refuse_click(monkeypatch):
    from mm_assist.config import ROOT
    from mm_assist.jobs.hud_entries import entry_by_id, load_main_hud_catalog
    from mm_assist.platform.match import MatchHit
    from mm_assist.perception import hud_entry_locator as locator

    entry = dict(entry_by_id(load_main_hud_catalog(), "menu"))
    outside = MatchHit(0.99, 1745, 13, 1678, 5, 134, 16, 0.12)
    for name in ("find_hamburger_bars", "match_icon_fuzzy", "match_template"):
        monkeypatch.setattr(locator, name, lambda *a, **k: outside)
    result = locate_hud_entry(entry, np.zeros((1442, 1912, 3), dtype=np.uint8), root=ROOT)
    assert result["feature_ok"] is False
    assert result["click"] is None
