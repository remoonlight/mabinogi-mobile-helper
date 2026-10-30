"""核心界面关系图：schema、禁止边、只读成对校验。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mm_assist.jobs.transitions import (
    TRIGGER_WHOS,
    assess_transition_captures,
    edge_by_id,
    load_ui_transition_graph,
    open_transition_frame,
    require_allowed_edge,
    validate_transition,
    validate_ui_transition_graph,
)


def test_load_graph_schema():
    g = load_ui_transition_graph()
    assert g["schema"] == "ui_transition_graph.v1"
    ids = {n["id"] for n in g["nodes"]}
    assert "main_hud" in ids and "facility" in ids


def test_load_graph_whitelist_aligned():
    g = load_ui_transition_graph()
    ids = {n["id"] for n in g["nodes"]}
    for nid in (
        "item_info",
        "obtain_method",
        "dialog_skip",
        "craft_quick",
        "craft_make",
        "craft_alchemy",
        "facility_recipe",
        "craft_item_recipe",
        "gather_detail",
    ):
        assert nid in ids
    kinds = {e["kind"] for e in g["edges"]}
    assert kinds == {"allowed", "forbidden"}
    for e in g["edges"]:
        assert e["trigger"]["who"] in TRIGGER_WHOS
    agent_edges = {e["id"] for e in g["edges"] if e["trigger"]["who"] == "agent"}
    # Current items child covers shell tabs: verified back-to-make is required.
    assert agent_edges == {
        "craft_item_recipe_back_craft_items",
        "craft_items_back_craft_make",
        "craft_shell_tab_craft_hub",
        "craft_shell_tab_craft_make",
        "craft_shell_x_main_hud",
        "cash_shop_close_main_hud",
        "craft_alchemy_tab_craft_hub",
        "craft_make_tab_craft_hub",
        "craft_quick_tab_craft_hub",
    }


def test_item_info_open_obtain_method_user_manual():
    g = load_ui_transition_graph()
    e = edge_by_id(g, "item_info_open_obtain_method")
    assert e["from"] == "item_info"
    assert e["to"] == "obtain_method"
    assert e["kind"] == "allowed"
    assert e["trigger"]["who"] == "user_manual"
    assert e["trigger"]["label"] == "获得方法"
    assert e["evidence"]["status"] == "unverified"


def test_unknown_edge_rejected():
    g = load_ui_transition_graph()
    with pytest.raises(KeyError, match="未知 edge"):
        edge_by_id(g, "not_a_real_edge")


def test_forbidden_edge_cannot_open_job():
    g = load_ui_transition_graph()
    with pytest.raises(ValueError, match="禁止边"):
        require_allowed_edge(g, "facility_avatar_profile_forbidden")
    with pytest.raises(ValueError, match="禁止边"):
        require_allowed_edge(g, "main_hud_avatar_profile")
    with pytest.raises(ValueError, match="禁止边"):
        open_transition_frame("facility_avatar_profile_forbidden", "source")
    with pytest.raises(ValueError, match="禁止边"):
        open_transition_frame("main_hud_avatar_profile", "source")


def test_open_transition_frame_method_b(tmp_path: Path):
    job = open_transition_frame(
        "system_menu_role_profile",
        "source",
        transition_id="unit_tid",
        path=tmp_path / "j.json",
    )
    assert job["status"] == "await_start"
    assert job["method"] == "B"
    assert job["action"] == "readonly_ui_fine"
    assert job["args"]["edge_id"] == "system_menu_role_profile"
    assert job["args"]["pair_role"] == "source"
    assert job["args"]["node_id"] == "system_menu"


def test_validate_transition_fixture_pair_ok_enough_false():
    root = Path(__file__).resolve().parent / "fixtures" / "transition_menu_role_profile"
    out = validate_transition(
        "system_menu_role_profile",
        source_path=root / "source.json",
        target_path=root / "target.json",
    )
    assert out["ok"] is True
    assert out["pair_ok"] is True
    assert out["enough"] is False
    assert out["assess"]["params_ok"] is True
    assert out["assess"]["icon_ok"] is True


def test_validate_transition_main_hud_avatar_forbidden():
    root = Path(__file__).resolve().parent / "fixtures" / "transition_menu_role_profile"
    out = validate_transition(
        "main_hud_avatar_profile",
        source_path=root / "source.json",
        target_path=root / "target.json",
    )
    assert out["ok"] is False
    assert "禁止边" in str(out["error"])


def test_validate_transition_missing_file(tmp_path: Path):
    out = validate_transition(
        "system_menu_role_profile",
        source_path=tmp_path / "missing.json",
        target_path=tmp_path / "missing2.json",
    )
    assert out["ok"] is False
    assert "FileNotFoundError" in str(out["error"])


def test_validate_transition_forbidden_edge():
    root = Path(__file__).resolve().parent / "fixtures" / "transition_main_hud_profile"
    out = validate_transition(
        "facility_avatar_profile_forbidden",
        source_path=root / "source.json",
        target_path=root / "target.json",
    )
    assert out["ok"] is False
    assert "禁止边" in str(out["error"])


def test_dunbarton_map_requires_nerys_avatar_signal():
    graph = load_ui_transition_graph()
    map_node = next(node for node in graph["nodes"] if node["id"] == "map_dunbarton")
    assert "nerys_avatar_visible" not in map_node["signals"]
    edge = edge_by_id(graph, "map_dunbarton_nerys_avatar")
    assert edge["to"] == "map_dunbarton_nerys"
    ordinary_map_frame = {
        "result": {
            "icons": [{"name": "Map", "x": 100, "y": 100}],
            "icon_errors": [],
            "panel_probe": {
                "map_open": True,
                "menu_open": True,
                "profile_open": False,
            },
        }
    }
    out = assess_transition_captures(
        edge, ordinary_map_frame, ordinary_map_frame, graph=graph
    )
    assert out["pair_ok"] is False
    assert "nerys_avatar_visible:want=True got=None" in out["target_miss"]


def test_invalid_graph_unknown_node():
    with pytest.raises(ValueError, match="未知 node"):
        validate_ui_transition_graph(
            {
                "schema": "ui_transition_graph.v1",
                "nodes": [
                    {
                        "id": "not_core",
                        "name": "x",
                        "need_ui": "y",
                        "signals": {"menu_open": False},
                    }
                ],
                "edges": [
                    {
                        "id": "e",
                        "from": "not_core",
                        "to": "not_core",
                        "kind": "allowed",
                        "trigger": {
                            "type": "visible_button",
                            "label": "x",
                            "who": "user_manual",
                        },
                        "evidence": {"status": "unverified"},
                    }
                ],
            }
        )


def test_cli_has_transition_commands():
    from mm_assist.cli import ALLOWED_CMDS

    assert "capture" in ALLOWED_CMDS
    assert "validate" in ALLOWED_CMDS
    forbidden = {"gather", "craft", "map-goto", "map-town-click"}
    assert not (forbidden & set(ALLOWED_CMDS))
