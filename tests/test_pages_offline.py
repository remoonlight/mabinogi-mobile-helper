# -*- coding: utf-8 -*-
from __future__ import annotations

import json

import cv2
import numpy as np

from pathlib import Path

from mm_assist.benchmarks.pages_offline import (
    format_pages_table,
    load_graph,
    node_matches,
    parse_pair_dir,
    run_pages_offline,
    sn_for_expected,
)


def test_parse_pair_dir_uses_graph_edge():
    g = load_graph()
    assert parse_pair_dir("main_hud_menu_system_menu", g) == ("main_hud", "system_menu")
    assert parse_pair_dir("life_grid_skill_gather_detail", g) == ("life_grid", "gather_skill")
    assert parse_pair_dir("system_menu_role_profile", g) == ("system_menu", "profile")
    assert parse_pair_dir("map_dunbarton_nerys_avatar", g) == (
        "map_dunbarton",
        "map_dunbarton_nerys",
    )
    assert parse_pair_dir("main_hud_map_dunbarton", g) == ("main_hud", "map_dunbarton")
    assert parse_pair_dir("map_dunbarton_nerys", g) == ("", "")


def test_sn_and_match():
    assert sn_for_expected("craft_hub") == "fine:craft_hub"
    assert sn_for_expected("facility:metal") == "fine:facility"
    assert sn_for_expected("gather_skill:daily") == "fine:life_grid"
    assert sn_for_expected("inventory") == ""
    assert sn_for_expected("facility_detail") == "fine:facility_detail"
    assert sn_for_expected("item_info") == "item_info"
    assert node_matches("gather_skill:daily", "gather_skill")
    assert not node_matches("life_grid", "inventory")


def _write_graph(tmp_path):
    g = load_graph()
    dest = tmp_path / "knowledge" / "ui" / "contracts"
    dest.mkdir(parents=True)
    (dest / "ui_transition_graph.json").write_text(json.dumps(g), encoding="utf-8")
    return g


def test_pages_offline_empty_and_missing_png(tmp_path):
    _write_graph(tmp_path)
    empty = run_pages_offline(warmup=0, repeats=1, root=tmp_path, out_dir=tmp_path / "led0")
    assert empty["ok"] is False
    assert empty["enough"] is False
    assert empty["frames"] == 0

    pair = tmp_path / "knowledge" / "captures" / "transitions" / "unit" / "main_hud_menu_system_menu"
    pair.mkdir(parents=True)
    missing = run_pages_offline(warmup=0, repeats=1, root=tmp_path, out_dir=tmp_path / "led1")
    assert missing["ok"] is False
    assert missing["enough"] is False
    assert missing["frames"] == 0


def test_pages_offline_tmp_corpus(tmp_path, monkeypatch):
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.perception import ocr as ocr_mod

    _write_graph(tmp_path)
    pair = (
        tmp_path
        / "knowledge"
        / "captures"
        / "transitions"
        / "unit"
        / "life_grid_skill_gather_detail"
    )
    pair.mkdir(parents=True)
    img = np.full((32, 32, 3), 40, dtype=np.uint8)
    cv2.imwrite(str(pair / "source.png"), img)
    cv2.imwrite(str(pair / "target.png"), img)

    seen_sn: list[str] = []
    orig_ocr = ocr_mod.ocr_roi

    def fake_refine(_frame, category=None, *, sn=""):
        seen_sn.append(sn)
        return "life_grid" if sn == "fine:life_grid" else "gather_detail"

    monkeypatch.setattr(ni, "refine_inspect_node", fake_refine)

    out = run_pages_offline(warmup=0, repeats=1, root=tmp_path, out_dir=tmp_path / "led")
    assert ocr_mod.ocr_roi is orig_ocr
    assert out["schema"] == "pages_offline.v1"
    assert out["ok"] is True
    assert out["enough"] is False
    assert out["frames"] == 2
    assert out["ocr_stub"] is True
    assert "craft_hub" in out["families_missing"]
    assert Path(out["ledger"]).is_file()
    assert Path(out["ledger"]).name.startswith("pages_")
    assert (tmp_path / "led" / "pages.jsonl").is_file()
    assert "" in seen_sn
    assert "fine:life_grid" in seen_sn
    table = format_pages_table(out)
    assert "life_grid_skill_gather_detail" in table
    assert "enough=False" in table
