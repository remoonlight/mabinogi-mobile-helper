from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mm_assist.jobs.state_watch import format_mem_panel
from mm_assist.memory.readers.scene import MemSceneInfo
from mm_assist.perception.menu_icons import (
    CHARACTER,
    MENU_ENTRY,
    format_menu_nav,
    probe_menu_icons_fast,
)
from mm_assist.perception.ocr import OcrLine
from mm_assist.perception.scene import format_ui_params, format_ui_relation
from mm_assist.platform.match import MatchHit, find_hamburger_bars, match_icon_fuzzy


def _blank(h: int = 200, w: int = 320) -> np.ndarray:
    return np.full((h, w, 3), 30, dtype=np.uint8)


def test_dialog_does_not_emit_inv_menu_dlg():
    ms = MemSceneInfo(
        menu_open=False,
        dialog_active=True,
        ui_stack_ready=True,
    )
    text = format_ui_params(ms)
    assert text == "stp=?"
    assert "menu=" not in text
    assert "dlg=" not in text


def test_format_ui_params_never_emits_nav():
    ms = MemSceneInfo(
        menu_open=False,
        dialog_active=False,
        ui_stack_ready=True,
    )
    text = format_ui_params(ms, menu_nav="nav=伐木,角色")
    assert "nav=" not in text
    assert "cbt=" not in text
    assert "stp=?" in text  # 未读取停止态不代表已停止。
    assert text.startswith("stp=")
    assert "inv=" not in text
    assert "menu=" not in text
    assert "dlg=" not in text
    assert "stp=0" not in text


def test_format_ui_params_keeps_independent_stop_when_menu_open():
    from mm_assist.perception.ui_fine import HudActionSummary

    ms = MemSceneInfo(
        menu_open=False,
        dialog_active=False,
        ui_stack_ready=True,
    )
    act = HudActionSummary(stop_visible=True)
    text = format_ui_params(ms, action=act)
    assert "stp=1" in text
    assert "cbt=" not in text
    open_menu = MemSceneInfo(
        menu_open=True,
        dialog_active=False,
        ui_stack_ready=True,
    )
    assert "stp=1" in format_ui_params(open_menu, action=act)
    assert "stp=0" not in format_ui_params(open_menu, action=act)


def test_menu_param_is_memory_only():
    hub = MemSceneInfo(
        menu_open=False,
        dialog_active=False,
        map_open=True,
        profile_open=True,
        craft_hub_category=1,
        ui_stack_ready=True,
    )
    assert "stp=?" == format_ui_params(hub)
    open_m = MemSceneInfo(
        menu_open=True,
        dialog_active=False,
        craft_hub_category=None,
        ui_stack_ready=True,
    )
    assert format_ui_params(open_m) == "stp=?"


def test_format_ui_params_needs_stack_before_stp():
    from mm_assist.perception.ui_fine import HudActionSummary

    ms = MemSceneInfo(menu_open=False, ui_stack_ready=False)
    walking = HudActionSummary(stop_visible=True)
    assert format_ui_params(ms) == "—"
    assert format_ui_params(ms, action=walking) == "—"
    left = MemSceneInfo(menu_open=True, ui_stack_ready=False)
    assert format_ui_params(left, action=walking) == "—"
    ready = MemSceneInfo(menu_open=False, dialog_active=False, ui_stack_ready=True)
    assert format_ui_params(ready, action=walking) == "stp=1"


def test_format_ui_relation_still_uses_internal_nav():
    ms = MemSceneInfo(menu_open=True, profile_open=False)
    assert format_ui_relation(ms, "menu", menu_nav="nav=伐木") == "角色-生活技能"
    assert format_ui_relation(ms, "menu", menu_nav="nav=角色") == "角色"


def test_format_ui_relation_stack_beats_hud_menu_bit():
    bag = MemSceneInfo(
        menu_open=True,
        ui_stack_ready=True,
        ui_stack_names=("TutorialTooltipUIState", "EquipmentInventoryUIState", "HudUIStepState"),
    )
    assert format_ui_relation(bag, "menu") == "背包"
    side = MemSceneInfo(
        menu_open=True,
        ui_stack_ready=True,
        ui_stack_names=("TutorialTooltipUIState", "SideNavigationUIState", "HudUIStepState"),
    )
    assert format_ui_relation(side, "menu", menu_nav="nav=伐木") == "菜单"
    main = MemSceneInfo(
        menu_open=False,
        ui_stack_ready=True,
        ui_stack_names=("TutorialTooltipUIState",),
        combat_mode_ui=False,
    )
    assert format_ui_relation(main, "hud") == "主界面非战斗"
    overlay = MemSceneInfo(
        menu_open=True,
        ui_stack_ready=True,
        ui_stack_names=("HudUIStepState",),
    )
    assert format_ui_relation(overlay, "menu") == "离开主界面"
    life = MemSceneInfo(
        menu_open=True,
        ui_stack_ready=True,
        ui_stack_names=("LivingSkillUIState", "HudUIStepState"),
    )
    assert format_ui_relation(life, "menu") == "角色-生活技能"


def test_family_from_stack_names_matches_relation():
    from mm_assist.perception.state_ui import (
        FAMILY_INVENTORY,
        FAMILY_MAIN_HUD,
        FAMILY_MENU,
        FAMILY_UNKNOWN,
        family_from_scene,
    )

    assert (
        family_from_scene(
            "menu",
            ui_stack_ready=True,
            ui_stack_names=["EquipmentInventoryUIState", "HudUIStepState"],
        )
        == FAMILY_INVENTORY
    )
    assert (
        family_from_scene(
            "menu",
            ui_stack_ready=True,
            ui_stack_names=["SideNavigationUIState", "HudUIStepState"],
        )
        == FAMILY_MENU
    )
    assert (
        family_from_scene(
            "hud",
            ui_stack_ready=True,
            ui_stack_names=["TutorialTooltipUIState"],
        )
        == FAMILY_MAIN_HUD
    )
    assert (
        family_from_scene(
            "menu",
            ui_stack_ready=True,
            ui_stack_names=["HudUIStepState"],
        )
        == FAMILY_UNKNOWN
    )


def test_format_mem_panel_strips_nav_token():
    snap = SimpleNamespace(
        mem_status_label="正常",
        mem_xy=(1.0, 2.0),
        ui_label="主界面",
        geo="",
        ui_params="inv=0 menu=0 dlg=0 stp=1 nav=伐木,角色",
        ui_relation="",
    )
    text = format_mem_panel(snap, None)
    assert "nav=" not in text
    assert "inv=" not in text
    assert "stp=1" in text


def test_match_icon_fuzzy_ignores_red_badge(tmp_path):
    tmpl = np.full((24, 24, 3), 80, dtype=np.uint8)
    tmpl[6:18, 6:18] = (200, 180, 40)
    path = tmp_path / "icon.png"
    import cv2

    cv2.imwrite(str(path), tmpl)
    frame = _blank()
    frame[40:64, 50:74] = tmpl
    frame[40:48, 66:74] = (0, 0, 255)
    hit = match_icon_fuzzy(frame, path, threshold=0.55)
    assert hit is not None
    assert abs(hit.cx - 62) <= 4
    assert abs(hit.cy - 52) <= 4


def test_find_hamburger_bars_on_synthetic_icon():
    frame = _blank(h=160, w=240)
    x0, y0 = 190, 8
    roi = frame[y0 : y0 + 36, x0 : x0 + 36]
    roi[:] = (40, 40, 40)
    for i, yy in enumerate((6, 16, 26)):
        roi[yy : yy + 3, 6:30] = 230
    hit = find_hamburger_bars(frame, roi=(0.72, 0.0, 0.28, 0.28))
    assert hit is not None
    assert hit.cx > 180


def test_probe_icon_preferred_over_ocr(monkeypatch):
    icon_hit = MatchHit(score=0.91, cx=100, cy=40, x=90, y=30, w=20, h=20, margin=0.1)

    def fake_fuzzy(frame, path, roi=None, threshold=0.58, scales=()):
        if "icon_character" in str(path).replace("\\", "/"):
            return icon_hit
        return None

    ocr_calls: list[tuple] = []

    def fake_ocr(frame, roi, min_confidence=0.5, **kwargs):
        ocr_calls.append(roi)
        return [OcrLine(text="角色", confidence=0.99)]

    monkeypatch.setattr("mm_assist.perception.menu_icons.match_icon_fuzzy", fake_fuzzy)
    monkeypatch.setattr("mm_assist.perception.menu_icons.find_hamburger_bars", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.perception.ocr.ocr_roi", fake_ocr)
    out = probe_menu_icons_fast(_blank(), menu_open=True)
    hit = (out.get(CHARACTER) or {}).get("hit")
    assert hit is not None
    assert hit.get("source") == "fuzzy"
    assert hit.get("ocr") is True
    assert hit.get("score") == pytest.approx(0.91)
    assert ocr_calls


def test_probe_ocr_fallback_and_aliases(monkeypatch):
    monkeypatch.setattr("mm_assist.perception.menu_icons.match_icon_fuzzy", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.perception.menu_icons.find_hamburger_bars", lambda *a, **k: None)

    def fake_ocr(frame, roi, min_confidence=0.5, **kwargs):
        return [OcrLine(text="人物", confidence=0.88)]

    monkeypatch.setattr("mm_assist.perception.ocr.ocr_roi", fake_ocr)
    out = probe_menu_icons_fast(_blank(), menu_open=True)
    hit = (out.get(CHARACTER) or {}).get("hit")
    assert hit is not None
    assert hit.get("source") == "ocr"
    assert "nav=角色" in format_menu_nav(out) or "角色" in format_menu_nav(out)


def test_probe_skips_subtree_when_menu_closed(monkeypatch):
    seen: list[str] = []

    def fake_fuzzy(frame, path, roi=None, threshold=0.58, scales=()):
        seen.append(str(path))
        return None

    monkeypatch.setattr("mm_assist.perception.menu_icons.match_icon_fuzzy", fake_fuzzy)
    monkeypatch.setattr("mm_assist.perception.menu_icons.find_hamburger_bars", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.perception.ocr.ocr_roi", lambda *a, **k: [])
    out = probe_menu_icons_fast(_blank(), menu_open=False)
    assert all("icon_character" not in p.replace("\\", "/") for p in seen)
    assert (out.get(CHARACTER) or {}).get("hit") is None
    assert MENU_ENTRY in out
