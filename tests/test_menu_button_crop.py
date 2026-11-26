# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mm_assist.config import ROOT
from mm_assist.jobs.menu_buttons import (
    MENU_BUTTONS,
    PAGE_BUTTONS,
    MenuButtonSpec,
    clamp_box,
    crop_bgr,
    process_page_buttons,
    save_menu_button_crop,
    select_page_buttons,
)


def test_four_button_filenames():
    names = [s.filename for s in MENU_BUTTONS]
    assert names == [
        "icon_character.png",
        "icon_inventory.png",
        "icon_crafting.png",
        "icon_close_x.png",
    ]


def test_crop_and_overwrite(tmp_path: Path):
    img = np.zeros((80, 100, 3), dtype=np.uint8)
    img[10:30, 20:50] = (0, 0, 255)
    spec = MENU_BUTTONS[0]
    first = save_menu_button_crop(img, spec, (20, 10, 30, 20), root=tmp_path)
    png = Path(first["png"])
    assert png.is_file()
    assert png.name == "icon_character.png"
    side = json.loads(Path(first["json"]).read_text(encoding="utf-8"))
    assert side["client_px"] == {"x": 20, "y": 10, "w": 30, "h": 20}
    img2 = np.zeros((80, 100, 3), dtype=np.uint8)
    img2[10:30, 20:50] = (255, 0, 0)
    second = save_menu_button_crop(img2, spec, (20, 10, 30, 20), root=tmp_path)
    assert Path(second["png"]) == png
    crop = crop_bgr(img2, 20, 10, 30, 20)
    assert int(crop[0, 0, 0]) == 255


def test_clamp_box_stays_inside():
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    x, y, w, h = clamp_box(img, -8, -3, 400, 400)
    assert x == 0 and y == 0
    assert w == 50 and h == 40


def test_page_button_writes_rel_dir(tmp_path: Path):
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    spec = PAGE_BUTTONS[0]
    assert spec.id == "inventory_close"
    out = save_menu_button_crop(img, spec, (2, 2, 10, 10), root=tmp_path)
    png = Path(out["png"])
    assert png.as_posix().endswith("knowledge/templates/inventory/close.png")
    assert png.is_file()


def test_process_queue_from_catalog():
    buttons = select_page_buttons(["process"])
    assert buttons == process_page_buttons()
    ids = [s.id for s in buttons]
    assert len(ids) == 14
    assert len(set(ids)) == 14
    assert ids[-5:] == [
        "process_btn_short_away",
        "process_btn_full_away",
        "process_btn_short_here",
        "process_btn_full_here",
        "claim_all",
    ]
    assert buttons[-1].filename == "claim_all.png"
    assert buttons[-5].filename == "process_btn_short_away.png"
    assert "process_metal_合金鋼錠" in ids
    assert "process_metal_鐵錠_iron_ore" in ids
    assert "process_metal_鐵錠_ore" not in ids
    assert "process_wood_高級木材" in ids
    assert "process_cloth_高級布料" in ids
    iron = next(s for s in buttons if s.id == "process_metal_鐵錠_iron_ore")
    assert "鐵錠(鐵礦石)" in iron.need_ui


def test_gather_anchor_queue():
    buttons = select_page_buttons(["gather-anchor"])
    assert len(buttons) == 18
    assert {'gather_skill_harvest_黃豆', 'gather_skill_hoe_防風草'} <= {s.id for s in buttons}
    assert all("daily" not in s.id and "daily" not in s.rel_dir for s in buttons)
    assert buttons[0].id == "gather_skill_woodcut_樹木"
    assert buttons[0].label == "樹木"
    assert buttons[-1].id == "gather_skill_fish_南側城牆釣魚場"
    assert buttons[-1].label == "南側城牆釣魚場"
    assert buttons[-1].filename == "南側城牆釣魚場.png"


def test_gather_anchor_pngs_on_disk():
    from mm_assist.perception.item_sense import GATHER_SKILL_IDS

    for spec in select_page_buttons(["gather-anchor"]):
        png = ROOT / spec.rel_dir / spec.filename
        assert png.is_file(), png
    daily = ROOT / "knowledge" / "templates" / "gather_skill" / "daily"
    assert {p.stem for p in daily.glob("*.png")} == {"巢穴", "蜘蛛網", "水井", "蜘蛛絲團"}
    for sid in GATHER_SKILL_IDS:
        folder = ROOT / "knowledge" / "templates" / "gather_skill" / sid
        assert len(tuple(folder.glob("*.png"))) >= 2


def test_gather_anchor_json_writes_role(tmp_path: Path):
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    spec = select_page_buttons(["gather-anchor"])[0]
    out = save_menu_button_crop(img, spec, (1, 1, 8, 8), root=tmp_path)
    side = json.loads(Path(out["json"]).read_text(encoding="utf-8"))
    assert side["role"] == "page_anchor"
    assert "gather_skill/woodcut" in side["rel_dir"].replace("\\", "/")


def test_craft_tabs_queue():
    buttons = select_page_buttons(["craft-tabs"])
    assert len(buttons) == 8
    assert [s.id for s in buttons] == [
        "craft_tab_quick_on",
        "craft_tab_quick_off",
        "craft_tab_process_on",
        "craft_tab_process_off",
        "craft_tab_make_on",
        "craft_tab_make_off",
        "craft_tab_alchemy_on",
        "craft_tab_alchemy_off",
    ]
    assert [s.filename for s in buttons] == [
        "tab_quick_on.png",
        "tab_quick_off.png",
        "tab_process_on.png",
        "tab_process_off.png",
        "tab_make_on.png",
        "tab_make_off.png",
        "tab_alchemy_on.png",
        "tab_alchemy_off.png",
    ]
    assert [s.role for s in buttons] == [
        "tab_on",
        "tab_off",
        "tab_on",
        "tab_off",
        "tab_on",
        "tab_off",
        "tab_on",
        "tab_off",
    ]
    assert all(s.rel_dir.replace("\\", "/") == "knowledge/templates/craft_hub" for s in buttons)


def test_craft_tabs_json_writes_role(tmp_path: Path):
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    spec = select_page_buttons(["craft-tabs"])[0]
    out = save_menu_button_crop(img, spec, (1, 1, 8, 8), root=tmp_path)
    side = json.loads(Path(out["json"]).read_text(encoding="utf-8"))
    assert spec.id == "craft_tab_quick_on"
    assert side["role"] == "tab_on"
    assert "craft_hub" in side["rel_dir"].replace("\\", "/")


def test_save_unicode_filename(tmp_path: Path):
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    spec = MenuButtonSpec(
        "item_woodcut_jian",
        "尖葉樹",
        "尖葉樹.png",
        "knowledge/templates/items/woodcut",
        "test",
    )
    out = save_menu_button_crop(img, spec, (1, 1, 8, 8), root=tmp_path)
    assert Path(out["png"]).is_file()
    assert Path(out["png"]).name == "尖葉樹.png"
