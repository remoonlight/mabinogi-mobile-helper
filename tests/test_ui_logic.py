from __future__ import annotations

import pytest

from mm_assist.life_execute.ui_logic import UiWhere, skip_click


@pytest.fixture(autouse=True)
def _ignore_live_inspect(monkeypatch):
    from mm_assist.life_execute.ui_nav import reset_edge_memo_for_tests
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_page", lambda **k: ""
    )
    reset_edge_memo_for_tests()
    reset_stage_gates_for_tests()
    yield
    reset_edge_memo_for_tests()
    reset_stage_gates_for_tests()


def test_skip_open_menu_when_already_in_menu():
    assert skip_click("system_menu", "打开菜单") is True
    assert skip_click("system_menu", "菜单") is True
    assert skip_click("main_hud", "打开菜单") is False
    assert skip_click("main_hud", "菜单") is False
    assert skip_click("main_hud", "打开菜单", menu=1) is True
    assert skip_click("main_hud", "菜单", menu=1) is True
    assert skip_click("life_grid", "打开角色") is True
    assert skip_click("profile", "角色") is True
    assert skip_click("system_menu", "角色") is False
    assert skip_click("craft_hub", "打开製作") is True
    assert skip_click("craft_hub", "製作") is True
    assert skip_click("craft_hub", "制作") is False
    assert skip_click("craft_alchemy", "打开製作") is False
    assert skip_click("craft_make", "製作") is False
    assert skip_click("craft_quick", "打开制作") is False
    assert skip_click("gather_detail", "技能大格") is True
    assert skip_click("gather_skill", "点採礦") is True


def test_close_template_paths_exist():
    from mm_assist.config import ROOT
    from mm_assist.life_execute.ui_logic import _close_rels_for

    nodes = (
        "craft_hub",
        "craft_make",
        "craft_alchemy",
        "facility",
        "facility:cloth",
        "inventory",
        "profile",
        "life_grid",
        "gather_detail",
        "map_dunbarton",
        "system_menu",
        "overlay",
        "unknown",
    )
    for node in nodes:
        rels = _close_rels_for(node)
        assert rels, node
        for rel in rels:
            assert (ROOT / rel).is_file(), f"{node} -> {rel}"
    hub = _close_rels_for("craft_hub")[0].as_posix()
    assert "facility" not in hub
    assert hub.endswith("craft_hub/close_x.png")
    assert skip_click("life_grid", "技能大格") is False
    assert skip_click("life_grid", "点採礦") is False


def test_graph_node_keeps_craft_tabs():
    from mm_assist.life_execute.ui_logic import graph_node, is_craft_shell

    assert graph_node("craft_alchemy") == "craft_alchemy"
    assert graph_node("craft_quick") == "craft_quick"
    assert graph_node("craft_make") == "craft_make"
    assert graph_node("craft_hub") == "craft_hub"
    assert is_craft_shell("craft_alchemy") is True
    assert is_craft_shell("facility") is False
    assert is_craft_shell("facility:metal") is False
    assert graph_node("facility:metal") == "facility"
    assert graph_node("facility:metal:鋼錠") == "facility_recipe"
    assert graph_node("gather_detail:淨淨蘑菇") == "gather_detail"


def test_menu_craft_label_exact():
    from mm_assist.perception.menu_icons import _is_menu_craft_label

    assert _is_menu_craft_label("製作") is True
    assert _is_menu_craft_label("製作(K)") is True
    assert _is_menu_craft_label("特製") is False
    assert _is_menu_craft_label("重甲製作") is False


def test_esc_craft_grid_if_char_cell(monkeypatch):
    import numpy as np
    from mm_assist.perception import menu_icons as mi
    from mm_assist.platform.match import MatchHit

    img = np.zeros((1440, 1920, 3), dtype=np.uint8)

    def fake_find(name, frame, *, roi=None, threshold=0.72, min_margin=0.02):
        if name == mi.CHARACTER:
            return MatchHit(score=0.8, cx=1400, cy=280, x=1390, y=270, w=20, h=20, margin=0.2)
        return None

    monkeypatch.setattr(mi, "find_template", fake_find)
    monkeypatch.setattr("mm_assist.perception.ocr.ocr_roi", lambda *a, **k: [])
    xy = mi.locate_esc_cell_by_label(img, ("製作",))
    assert xy == (int(1920 * mi.ESC_CRAFT_NX), int(1440 * mi.ESC_GRID_NY))


def test_esc_craft_label_offset_lands_on_icon(monkeypatch):
    import numpy as np
    from mm_assist.perception import menu_icons as mi

    img = np.zeros((1440, 1920, 3), dtype=np.uint8)

    class Ln:
        text = "製作(K)"
        box = [[1700, 330], [1820, 330], [1820, 380], [1700, 380]]

    monkeypatch.setattr(mi, "find_template", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.perception.ocr.ocr_roi", lambda *a, **k: [Ln()])
    xy = mi.locate_esc_cell_by_label(img, ("製作",))
    ly = 355
    assert xy == (1760, ly - int(mi.ESC_CRAFT_LABEL_UP_NY * 1440))
    assert xy[1] > ly - int(0.055 * 1440)


def test_resolve_prefers_mem_inventory(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(
        ui_logic,
        "read_state_ui",
        lambda **k: _menu1_view(family="inventory", params="inv=1 menu=1 dlg=0 stp=0"),
    )
    where = ui_logic.resolve_ui_node(frame=None)
    assert where.node == "inventory"
    assert where.source == "mem"


def test_resolve_empty_stack_inspect_idle_is_main_hud(monkeypatch):
    from mm_assist.life_execute import ui_logic

    class V:
        params = "stp=0"
        family = "unknown"
        dialog_active = False
        menu_open = False
        map_open = False
        profile_open = False
        stp = 0
        craft_hub_category = None
        craft_tab = None
        relation = ""
        ui_stack_names = []
        ui_stack_ready = True
        raw = {
            "ui_stack": "主界面",
            "ui_stack_names": [],
            "ui_stack_ready": True,
            "ui_stack_empty_ok": False,
        }

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: V())
    monkeypatch.setattr(
        ui_logic,
        "read_inspect_record",
        lambda **k: {"zh": "主界面-待机", "node": "main_hud", "stack_names": []},
    )
    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: "main_hud")
    where = ui_logic.resolve_ui_node(frame=None)
    assert where.node == "main_hud"


@pytest.mark.parametrize('frame_present,refined,hud_icon', [
    (True, 'facility', False),
    (False, None, None),
    (True, 'unknown', False),
    (True, 'profile', True),
], ids=['wrong-facility-refinement', 'no-frame', 'missing-hud-icon', 'wrong-profile-refinement'])
def test_confirmed_hud_stack_keeps_identity(monkeypatch, frame_present, refined, hud_icon):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, 'read_state_ui', lambda **k: _hud_view())
    if refined is not None:
        monkeypatch.setattr(ui_logic, '_refine_menu1', lambda *a, **k: refined)
        monkeypatch.setattr(ui_logic, '_main_hud_confirmed', lambda *a, **k: hud_icon)
    where = ui_logic.resolve_ui_node(frame=object() if frame_present else None)
    assert where.node == 'main_hud'









def test_profile_stack_beats_stale_hud_hint(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(
        ui_logic, "read_state_ui", lambda **k: _menu1_view(profile_open=True)
    )
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "profile"


def test_resolve_inv1_is_inventory_even_if_hud(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(
        ui_logic,
        "read_state_ui",
        lambda **k: _menu1_view(family="inventory", params="inv=1 menu=1 dlg=0 stp=0"),
    )
    monkeypatch.setattr(ui_logic, "_refine_menu1", lambda *a, **k: "unknown")
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: True)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "inventory"


def test_resolve_hud_beats_false_craft_hub(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _hud_view(family="unknown"))
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "facility_page_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "system_menu_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "recipe_action_state", lambda *a, **k: None)
    monkeypatch.setattr(ui_logic, "craft_hub_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: True)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "main_hud"


def test_resolve_craft_hub_when_not_hud(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view())
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "facility_page_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "system_menu_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "recipe_action_state", lambda *a, **k: None)
    monkeypatch.setattr(ui_logic, "craft_hub_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: False)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "system_menu"


def test_resolve_facility_beats_craft_hub(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view())
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "facility_page_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "craft_hub_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: True)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "system_menu"


def test_resolve_menu0_ignores_facility_ocr(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _hud_view())
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "system_menu_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "recipe_action_state", lambda *a, **k: None)
    monkeypatch.setattr(ui_logic, "facility_page_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: True)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "main_hud"


def test_resolve_params_pending_hud_is_main_hud(monkeypatch):
    from mm_assist.life_execute import ui_logic

    class V:
        params = "—"
        family = "unknown"
        dialog_active = False
        menu_open = None
        map_open = False
        profile_open = False
        stp = None

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: V())
    monkeypatch.setattr(ui_logic, "_refine_menu1", lambda *a, **k: "unknown")
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: True)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "unknown"


def test_recover_wrong_tab_uses_esc_and_verifies_result(monkeypatch):
    from mm_assist.life_execute.ui_recover import recover_once
    import time

    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.life_plan.node_inspect.craft_tab_xy",
        lambda *_a, **_k: (9, 8),
    )
    keys = []
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr('mm_assist.life_execute.ui_nav._hud_xy', lambda *a, **k: (10, 20))
    r = recover_once(
        "wrong_tab",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        press_esc=lambda: keys.append(1), sleep_fn=lambda _: None,
        resolve=lambda _: UiWhere('main_hud' if keys else 'craft_make', 0, 0, 0,
                                 'fixture', stack_ready=True, stack_sampled_ts=time.time()),
    )
    assert r["ok"] is True
    assert r["action"] == "press_escape"
    assert clicks == [] and keys == [1]


def test_path_follows_review_not_avatar():
    from mm_assist.life_execute.ui_nav import path_edges

    to_profile = [e["id"] for e in path_edges("main_hud", "profile") or []]
    assert to_profile[0] == "main_hud_menu_system_menu"
    assert "main_hud_avatar_profile" not in to_profile
    from_fac = [e["id"] for e in path_edges("facility", "gather_skill") or []]
    assert from_fac[0] == "facility_back_craft_hub"
    assert "facility_avatar_gather_forbidden" not in from_fac
    assert path_edges("facility", "facility") == []
    inv = [e["id"] for e in path_edges("main_hud", "inventory") or []]
    assert inv == ["main_hud_menu_system_menu", "system_menu_bag_inventory"]
    craft = [e["id"] for e in path_edges("main_hud", "craft_hub") or []]
    assert craft == ["main_hud_menu_system_menu", "system_menu_craft_craft_hub"]
    tab = path_edges("craft_alchemy", "craft_hub") or []
    assert tab
    assert tab[0]["id"] == "craft_alchemy_tab_craft_hub"
    assert tab[0]["trigger"]["label"] == "加工"


def test_refine_gather_detail_uses_find_nearby(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: True)
    assert ui_logic._refine_menu1(object(), root=None) == "gather_detail"


def test_gather_detail_visible_forwards_roi_to_match_template(tmp_path):
    from pathlib import Path
    from unittest.mock import patch

    from mm_assist.life_execute.ui_logic import _match, gather_detail_visible
    from mm_assist.platform.match import MatchHit

    template = tmp_path / "knowledge" / "templates" / "gather_detail" / "find_nearby.png"
    template.parent.mkdir(parents=True)
    template.write_bytes(b"fixture: low-level matcher is mocked")
    hit = MatchHit(score=0.9, cx=110, cy=220, x=100, y=200, w=20, h=40)
    frame = object()
    with patch("mm_assist.platform.match.match_template", return_value=hit) as matcher:
        visible = gather_detail_visible(frame, root=tmp_path)
        _match(frame, Path("knowledge") / "templates" / "gather_detail" / "find_nearby.png", root=tmp_path)
    assert visible is True
    assert matcher.call_count == 2
    detail_kwargs = matcher.call_args_list[0].kwargs
    assert detail_kwargs["roi"] == (0.0, 0.38, 1.0, 0.60)
    assert detail_kwargs["threshold"] == 0.70
    assert matcher.call_args_list[0].args[0] is frame
    assert matcher.call_args_list[0].args[1] == template
    default_kwargs = matcher.call_args_list[1].kwargs
    assert default_kwargs["threshold"] == 0.72
    assert default_kwargs.get("roi") is None


def test_gather_detail_visible_missing_file_or_none_frame_does_not_match(tmp_path):
    from unittest.mock import patch

    from mm_assist.life_execute.ui_logic import gather_detail_visible

    with patch("mm_assist.platform.match.match_template") as matcher:
        assert gather_detail_visible(None, root=tmp_path) is False
        assert matcher.call_count == 0
        assert gather_detail_visible(object(), root=tmp_path) is False
        assert matcher.call_count == 0


def test_gather_detail_visible_match_error_fail_closed(tmp_path):
    from unittest.mock import patch

    from mm_assist.life_execute.ui_logic import gather_detail_visible

    template = tmp_path / "knowledge" / "templates" / "gather_detail" / "find_nearby.png"
    template.parent.mkdir(parents=True)
    template.write_bytes(b"fixture")
    with patch("mm_assist.platform.match.match_template", side_effect=RuntimeError("cv")):
        assert gather_detail_visible(object(), root=tmp_path) is False






def test_no_click_without_recognition():
    from mm_assist.life_execute.ui_logic import UiWhere, recognized_click_xy
    from mm_assist.life_execute.ui_nav import ensure_node, locate_edge

    assert recognized_click_xy({"click": {"client": [10, 20]}, "feature_ok": False}) is None
    assert recognized_click_xy({"ok": False, "cx": 10, "cy": 20}) is None
    assert recognized_click_xy({"ok": True, "cx": 10, "cy": 20}) == (10, 20)
    assert locate_edge({"id": "map_dunbarton_nerys_avatar"}, None) is None
    clicks: list[tuple[int, int]] = []
    r = ensure_node(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("unknown", None, None, None, "icon_only"),
    )
    assert r.ok is False
    assert r.status == "unknown_ui"
    assert clicks == []










def _hud_view(**extra):
    class V:
        params = extra.get("params", "inv=0 menu=0 dlg=0 stp=0")
        family = extra.get("family", "main_hud")
        dialog_active = extra.get("dialog_active", False)
        menu_open = extra.get("menu_open", False)
        map_open = extra.get("map_open", False)
        profile_open = extra.get("profile_open", False)
        stp = extra.get("stp", 0)
        craft_hub_category = extra.get("craft_hub_category", None)
        craft_tab = extra.get("craft_tab", None)
        relation = extra.get("relation", "")
        ui_stack_names = extra.get("ui_stack_names", ["TutorialTooltipUIState"])
        ui_stack_ready = extra.get("ui_stack_ready", True)

    return V()


def _menu1_view(**extra):
    p = extra.get("params", "inv=0 menu=1 dlg=0 stp=0")
    fam = extra.get("family", "unknown")
    if extra.get("ui_stack_names") is not None:
        names = extra["ui_stack_names"]
    elif "inv=1" in p or fam == "inventory":
        names = ["TutorialTooltipUIState", "EquipmentInventoryUIState", "HudUIStepState"]
    elif "dlg=1" in p or fam in ("dialog", "repair"):
        names = ["TutorialTooltipUIState", "DialogueInteractionUIState", "HudUIStepState"]
    elif fam == "craft_hub":
        names = ["TutorialTooltipUIState", "CraftingHubUIState", "HudUIStepState"]
    elif fam == "prf" or extra.get("profile_open"):
        names = ["NewProfileUIState", "HudUIStepState"]
    elif extra.get("map_open") or fam == "map":
        names = ["MapUIState", "HudUIStepState"]
    elif fam == "menu" or "menu=1" in p:
        names = ["TutorialTooltipUIState", "SideNavigationUIState", "HudUIStepState"]
    else:
        names = ["HudUIStepState"]

    class V:
        params = p
        family = extra.get("family", "unknown")
        dialog_active = extra.get("dialog_active", False)
        menu_open = extra.get("menu_open", True)
        map_open = extra.get("map_open", False)
        profile_open = extra.get("profile_open", False)
        stp = extra.get("stp", 0)
        craft_hub_category = extra.get("craft_hub_category", None)
        craft_tab = extra.get("craft_tab", None)
        relation = extra.get("relation", "")
        ui_stack_names = names
        ui_stack_ready = extra.get("ui_stack_ready", True)

    return V()


def test_resolve_dlg1_is_dialog(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view(params="inv=0 menu=1 dlg=1 stp=0"))
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "dialog"


def test_resolve_gather_skill_from_item_art(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view())
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "recipe_action_state", lambda *a, **k: None)
    monkeypatch.setattr(ui_logic, "facility_page_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "detect_gather_skill_id", lambda *a, **k: "daily")
    monkeypatch.setattr(ui_logic, "system_menu_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "detect_craft_tab", lambda *a, **k: "")
    monkeypatch.setattr(ui_logic, "craft_hub_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "life_grid_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: False)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "system_menu"


def test_resolve_life_grid_beats_system_menu(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view())
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "life_grid_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "profile_shell_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "system_menu_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: False)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "system_menu"


def test_resolve_life_relation_beats_false_hub(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(
        ui_logic,
        "read_state_ui",
        lambda **k: _menu1_view(
            relation="生活技能界面",
            ui_stack_names=[
                "NewProfileUIState",
                "LivingSkillUIState",
                "HudUIStepState",
            ],
        ),
    )
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "life_grid_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "profile_shell_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "system_menu_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "recipe_action_state", lambda *a, **k: None)
    monkeypatch.setattr(ui_logic, "facility_page_visible", lambda *a, **k: False)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "life_grid"


def test_resolve_character_relation_is_profile(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(
        ui_logic,
        "read_state_ui",
        lambda **k: _menu1_view(
            relation="人物界面",
            ui_stack_names=["NewProfileUIState", "HudUIStepState"],
        ),
    )
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "life_grid_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "profile_shell_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "system_menu_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "recipe_action_state", lambda *a, **k: None)
    monkeypatch.setattr(ui_logic, "facility_page_visible", lambda *a, **k: False)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "profile"


def test_resolve_system_menu_skips_recipe_ocr(monkeypatch):
    from mm_assist.life_execute import ui_logic

    def boom(*a, **k):
        raise AssertionError("菜单上不得跑配方 OCR")

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view())
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "life_grid_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "system_menu_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "gather_detail_visible", boom)
    monkeypatch.setattr(ui_logic, "recipe_action_state", boom)
    monkeypatch.setattr(ui_logic, "facility_page_visible", boom)
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: False)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "system_menu"


def test_resolve_facility_beats_stale_life_relation(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(
        ui_logic,
        "read_state_ui",
        lambda **k: _menu1_view(
            relation="生活技能界面",
            ui_stack_names=["NewProfileUIState", "HudUIStepState"],
        ),
    )
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "recipe_action_state", lambda *a, **k: None)
    monkeypatch.setattr(ui_logic, "facility_page_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "life_grid_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: False)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "profile"


def test_resolve_menu_family_not_craft_hub(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(
        ui_logic, "read_state_ui", lambda **k: _menu1_view(family="menu", relation="菜单")
    )
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "life_grid_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "profile_shell_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "craft_hub_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "_match", lambda *a, **k: type("H", (), {"score": 0.9, "cx": 1, "cy": 1})())
    monkeypatch.setattr(ui_logic, "_hit_xy", lambda *a, **k: (1, 1))
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "system_menu"


def test_resolve_facility_recipe_away(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view())
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(
        ui_logic, "recipe_action_state", lambda *a, **k: {"label": "前往加工", "stance": "away"}
    )
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: False)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "system_menu"


def test_resolve_opening_craft_category_does_not_prove_current_tab(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(
        ui_logic,
        "read_state_ui",
        lambda **k: _menu1_view(
            family="craft_hub", craft_hub_category=3, craft_tab="炼金"
        ),
    )
    where = ui_logic.resolve_ui_node(frame=None)
    assert where.node == "fine:craft_hub"
    assert where.craft_tab == ""


def test_detect_craft_tab_uses_left_tabs_not_ocr(monkeypatch):
    from mm_assist.life_execute import ui_logic
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_selected_craft_tab", lambda _f: "craft_alchemy")
    monkeypatch.setattr(
        ui_logic,
        "_ocr_blob",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("ocr should skip")),
    )
    assert ui_logic.detect_craft_tab(object()) == "alchemy"


def _blank_menu_frame(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "gather_detail_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "recipe_action_state", lambda *a, **k: None)
    monkeypatch.setattr(ui_logic, "facility_page_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "detect_gather_skill_id", lambda *a, **k: "")
    monkeypatch.setattr(ui_logic, "life_grid_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "craft_hub_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "_ocr_blob", lambda *a, **k: "")
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, "profile_shell_visible", lambda *a, **k: False)


def test_resolve_system_menu_not_stale_craft_hub(monkeypatch):
    from mm_assist.life_execute import ui_logic

    _blank_menu_frame(monkeypatch)
    monkeypatch.setattr(
        ui_logic,
        "read_state_ui",
        lambda **k: _menu1_view(family="craft_hub", craft_hub_category=1, craft_tab="加工"),
    )
    monkeypatch.setattr(ui_logic, "system_menu_visible", lambda *a, **k: True)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "fine:craft_hub"


def test_resolve_frame_ignores_stale_hub_mem(monkeypatch):
    from mm_assist.life_execute import ui_logic

    _blank_menu_frame(monkeypatch)
    monkeypatch.setattr(
        ui_logic,
        "read_state_ui",
        lambda **k: _menu1_view(family="craft_hub", craft_hub_category=1),
    )
    monkeypatch.setattr(ui_logic, "system_menu_visible", lambda *a, **k: False)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "fine:craft_hub"


def test_locate_edge_craft_falls_back_to_icon(monkeypatch):
    from mm_assist.life_execute import ui_nav

    class Hit:
        ok = True
        cx = 91
        cy = 42

    monkeypatch.setattr(
        "mm_assist.perception.menu_icons.locate_esc_cell_by_label",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(ui_nav, "_match", lambda *a, **k: Hit())
    xy = ui_nav.locate_edge({"id": "system_menu_craft_craft_hub"}, object())
    assert xy == (91, 42)


def test_locate_edge_craft_missing_evidence_returns_none(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import ui_nav

    monkeypatch.setattr(
        "mm_assist.perception.menu_icons.locate_esc_cell_by_label",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(ui_nav, "_match", lambda *a, **k: None)
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    xy = ui_nav.locate_edge({"id": "system_menu_craft_craft_hub"}, img)
    # The page-only navigation contract forbids guessing a cell on a blank image.
    assert xy is None


def test_locate_edge_herb_cell_accepts_mid_score(monkeypatch):
    from mm_assist.life_execute import ui_nav
    from mm_assist.platform.match import MatchHit

    monkeypatch.setattr(
        ui_nav,
        "_match",
        lambda *a, **k: MatchHit(score=0.62, cx=855, cy=621, x=840, y=600, w=30, h=30, margin=0.1),
    )
    xy = ui_nav.locate_edge(
        {"id": "life_grid_skill_gather_detail"}, object(), skill_id="herb"
    )
    assert xy == (855, 621)


def test_resolve_wallet_overlay(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view())
    monkeypatch.setattr(ui_logic, "currency_wallet_visible", lambda *a, **k: True)
    monkeypatch.setattr(ui_logic, "_main_hud_confirmed", lambda *a, **k: False)
    where = ui_logic.resolve_ui_node(frame=object())
    assert where.node == "system_menu"


def test_resolve_map_from_mem(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view(map_open=True))
    where = ui_logic.resolve_ui_node(frame=None)
    assert where.node == "map"


def test_path_aliases_new_nodes():
    from mm_assist.life_execute.ui_nav import path_edges

    assert path_edges("facility_recipe", "facility") == []
    close_list = [e["id"] for e in path_edges("gather_skill", "life_grid") or []]
    assert close_list == ["gather_skill_close_life_grid"]
    back = [e["id"] for e in path_edges("facility_recipe", "craft_hub") or []]
    assert back == ["facility_back_craft_hub"]


def test_skip_click_new_nodes():
    assert skip_click("gather_skill", "打开生活技能") is True
    assert skip_click("facility_recipe", "打开製作") is True
    assert skip_click("currency_wallet", "打开製作") is True


def test_resolve_reads_inspect_metal_ingot(monkeypatch):
    from mm_assist.life_execute import ui_logic
    from mm_assist.life_plan.node_inspect import FACILITY_ZH, format_node_zh

    zh = format_node_zh("facility:metal:鋼錠")
    assert FACILITY_ZH["metal"] in zh
    names = ["HudUIStepState", "CraftingHubUIState"]
    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: "facility:metal:鋼錠")
    monkeypatch.setattr(
        ui_logic,
        "read_inspect_record",
        lambda **k: {"node": "facility:metal:鋼錠", "stack_names": names},
    )
    monkeypatch.setattr(
        ui_logic,
        "read_state_ui",
        lambda **k: _menu1_view(family="craft_hub", ui_stack_names=names),
    )
    where = ui_logic.resolve_ui_node(frame=None)
    assert where.node == "facility:metal:鋼錠"
    assert where.facility_id == "metal"
    assert ui_logic.graph_node(where.node) == "facility_recipe"


def test_wait_ui_params_no_files(tmp_path, monkeypatch):
    from mm_assist.life_execute import ui_logic

    assert ui_logic.WAIT_PARAM_S >= 30.0
    monkeypatch.setattr(ui_logic, "skip_wait_ui_params", lambda: False)
    assert ui_logic.wait_ui_params_refresh(root=tmp_path, pre_wait_s=0) is True


def test_wait_ui_params_until_stamps_bump(tmp_path, monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "skip_wait_ui_params", lambda: False)
    logs = tmp_path / "logs"
    logs.mkdir()
    su = logs / "state_watch_ui.json"
    insp = logs / "ui_inspect.json"
    payload = '{"ts_unix": 1.0, "ui_params": "inv=0 menu=1 dlg=0 stp=0", "ui_stack_ts": 10.0}\n'
    su.write_text(payload, encoding="utf-8")
    insp.write_text('{"ts_unix": 2.0, "node": "profile"}\n', encoding="utf-8")
    n = {"i": 0}

    def bump(_s: float) -> None:
        n["i"] += 1
        t = 1.0 + n["i"]
        su.write_text(
            f'{{"ts_unix": {t}, "ui_params": "inv=0 menu=1 dlg=0 stp=0", "ui_stack_ts": {t + 10}}}\n',
            encoding="utf-8",
        )
        insp.write_text(
            f'{{"ts_unix": {t + 1}, "node": "profile"}}\n',
            encoding="utf-8",
        )

    assert ui_logic.wait_ui_params_refresh(
        root=tmp_path, sleep_fn=bump, timeout_s=2.0, pre_wait_s=0
    ) is True
    assert n["i"] >= 2


def test_wait_ui_params_does_not_wait_on_inspect_lag(tmp_path, monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "skip_wait_ui_params", lambda: False)
    logs = tmp_path / "logs"
    logs.mkdir()
    su = logs / "state_watch_ui.json"
    insp = logs / "ui_inspect.json"
    su.write_text(
        '{"ts_unix": 1.0, "ui_params": "inv=0 menu=1 dlg=0 stp=0", "ui_stack_ts": 10.0}\n',
        encoding="utf-8",
    )
    insp.write_text('{"ts_unix": 2.0, "node": "profile"}\n', encoding="utf-8")
    n = {"i": 0}

    def bump_su_only(_s: float) -> None:
        n["i"] += 1
        t = 1.0 + n["i"]
        su.write_text(
            f'{{"ts_unix": {t}, "ui_params": "inv=0 menu=1 dlg=0 stp=0", "ui_stack_ts": {t + 10}}}\n',
            encoding="utf-8",
        )

    assert ui_logic.wait_ui_params_refresh(
        root=tmp_path, sleep_fn=bump_su_only, timeout_s=2.0, pre_wait_s=0
    ) is True
    assert n["i"] >= 2


def test_wait_ui_params_two_ticks_must_agree(tmp_path, monkeypatch):
    import json
    import os
    import time

    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "skip_wait_ui_params", lambda: False)
    logs = tmp_path / "logs"
    logs.mkdir()
    su = logs / "state_watch_ui.json"
    insp = logs / "ui_inspect.json"
    base = time.time()
    pid = os.getpid()
    su.write_text(
        json.dumps(
            {
                "ts_unix": base,
                "ui_params": "inv=0 menu=1 dlg=0 stp=0",
                "ui_stack_ts": base,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    insp.write_text(
        json.dumps({"ts_unix": base, "node": "craft_quick", "pid": pid}) + "\n",
        encoding="utf-8",
    )
    n = {"i": 0}

    def bump(_s: float) -> None:
        n["i"] += 1
        t = base + n["i"]
        node = "craft_quick" if n["i"] == 1 else "profile"
        su.write_text(
            json.dumps(
                {
                    "ts_unix": t,
                    "ui_params": "inv=0 menu=1 dlg=0 stp=0",
                    "ui_stack_ts": t,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        insp.write_text(
            json.dumps({"ts_unix": t + 0.01, "node": node, "pid": pid}) + "\n",
            encoding="utf-8",
        )

    assert ui_logic.wait_ui_params_refresh(
        root=tmp_path, sleep_fn=bump, timeout_s=2.0, pre_wait_s=0
    ) is True
    assert n["i"] >= 3


def test_wait_ui_params_timeout_stale(tmp_path, monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "skip_wait_ui_params", lambda: False)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "state_watch_ui.json").write_text('{"ts_unix": 1.0}\n', encoding="utf-8")
    now = {"t": 0.0}
    monkeypatch.setattr("time.time", lambda: now["t"])

    def jump(_s: float) -> None:
        now["t"] += 10.0

    assert ui_logic.wait_ui_params_refresh(
        root=tmp_path, sleep_fn=jump, timeout_s=0.2, pre_wait_s=0
    ) is False


def test_wait_ui_params_same_stack_ts_not_stable(tmp_path, monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "skip_wait_ui_params", lambda: False)
    logs = tmp_path / "logs"
    logs.mkdir()
    su = logs / "state_watch_ui.json"
    su.write_text(
        '{"ts_unix": 1.0, "ui_params": "stp=0", "ui_stack_ts": 5.0}\n',
        encoding="utf-8",
    )
    n = {"i": 0}
    now = {"t": 0.0}
    monkeypatch.setattr("time.time", lambda: now["t"])

    def bump(_s: float) -> None:
        n["i"] += 1
        now["t"] += 1.0
        su.write_text(
            f'{{"ts_unix": {1.0 + n["i"]}, "ui_params": "stp=0", "ui_stack_ts": 5.0}}\n',
            encoding="utf-8",
        )

    assert (
        ui_logic.wait_ui_params_refresh(
            root=tmp_path, sleep_fn=bump, timeout_s=2.0, pre_wait_s=0
        )
        is False
    )
    assert n["i"] >= 2


def test_wait_ui_params_distinct_stack_ts_stable(tmp_path, monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "skip_wait_ui_params", lambda: False)
    logs = tmp_path / "logs"
    logs.mkdir()
    su = logs / "state_watch_ui.json"
    su.write_text(
        '{"ts_unix": 1.0, "ui_params": "stp=0", "ui_stack_ts": 5.0}\n',
        encoding="utf-8",
    )
    n = {"i": 0}

    def bump(_s: float) -> None:
        n["i"] += 1
        t = 1.0 + n["i"]
        su.write_text(
            f'{{"ts_unix": {t}, "ui_params": "stp=0", "ui_stack_ts": {5.0 + n["i"]}}}\n',
            encoding="utf-8",
        )

    assert (
        ui_logic.wait_ui_params_refresh(
            root=tmp_path, sleep_fn=bump, timeout_s=2.0, pre_wait_s=0
        )
        is True
    )
    assert n["i"] >= 2


def test_resolve_menu0_ignores_inspect_facility(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: "facility:metal:鋼錠")
    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _hud_view())
    where = ui_logic.resolve_ui_node(frame=None)
    assert where.node == "main_hud"


def test_resolve_menu1_ignores_inspect_main_hud(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: "main_hud")
    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view(family="menu"))
    where = ui_logic.resolve_ui_node(frame=None)
    assert where.node == "system_menu"
    assert where.menu == 1


def test_resolve_ignores_forged_inspect_wallet(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: "currency_wallet")
    monkeypatch.setattr(
        ui_logic,
        "read_state_ui",
        lambda **k: _menu1_view(family="craft_hub", craft_hub_category=1),
    )
    where = ui_logic.resolve_ui_node(frame=None)
    assert where.node == "fine:craft_hub"
    assert where.overlay != "currency_wallet"


@pytest.mark.parametrize("refined", ["life_grid", "gather_skill:daily", "gather_detail:淨淨蘑菇"])
def test_resolve_profile_stack_accepts_life_page_refinement(monkeypatch, refined):
    from mm_assist.life_execute import ui_logic

    names = ["UILayerHideState", "NewProfileUIInstance"]
    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view(profile_open=True, ui_stack_names=names))
    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: refined)
    monkeypatch.setattr(ui_logic, "read_inspect_record", lambda **k: {"node": refined, "stack_names": names})
    assert ui_logic.resolve_ui_node().node == refined


def test_navigation_uses_live_stack_and_frame_without_display_files(monkeypatch):
    import cv2
    from pathlib import Path
    from mm_assist.life_execute import ui_logic
    from mm_assist.memory.readers.ui_stack import UiStackSnap

    class Reader:
        def read(self, **kwargs):
            return UiStackSnap(names=["NewProfileUIInstance"], ready=True)
    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: pytest.fail("display file used"))
    monkeypatch.setattr(ui_logic, "read_inspect_record", lambda **k: pytest.fail("display file used"))
    frame = cv2.imread(str(Path(__file__).parent / "fixtures/ui_benchmark/life_grid_real.png"))
    assert ui_logic.resolve_navigation(frame, reader=Reader()).node == "life_grid"


def test_navigation_skill_popup_is_not_its_dimmed_background_grid():
    import cv2
    from pathlib import Path
    from mm_assist.life_execute.ui_logic import resolve_navigation
    from mm_assist.memory.readers.ui_stack import UiStackSnap
    class Reader:
        def read(self, **kwargs):
            return UiStackSnap(names=["NewProfileUIInstance"], ready=True)
    frame = cv2.imread(str(Path(__file__).parent / "fixtures/ui_benchmark/herb_skill_popup.png"))
    assert resolve_navigation(frame, reader=Reader()).node == "gather_skill:herb"


def test_navigation_discards_stack_change_during_recognition(monkeypatch):
    from mm_assist.life_execute import ui_logic
    from mm_assist.life_plan import node_inspect
    from mm_assist.memory.readers.ui_stack import UiStackSnap

    rows = iter([UiStackSnap(names=["NewProfileUIInstance"], ready=True),
                 UiStackSnap(names=["SideNavigationUIState"], ready=True)])
    class Reader:
        def read(self, **kwargs):
            return next(rows)
    monkeypatch.setattr(node_inspect, "compute_inspect", lambda *a, **k: {"node": "life_grid"})
    assert ui_logic.resolve_navigation(object(), reader=Reader()).node == "unknown"


def test_navigation_confirmed_empty_hud_without_icon_is_main_hud(monkeypatch):
    from mm_assist.life_execute import ui_logic
    from mm_assist.memory.readers.ui_stack import UiStackSnap
    class Reader:
        def read(self, **kwargs):
            return UiStackSnap(names=[], ready=True, empty_confirmed=True)
    monkeypatch.setattr("mm_assist.life_plan.node_inspect.compute_inspect", lambda *a, **k: {"node": "main_hud"})
    monkeypatch.setattr(ui_logic, "_hud_entry_ok", lambda *a, **k: False)
    assert ui_logic.resolve_navigation(object(), reader=Reader()).node == "main_hud"


def test_navigation_fake_empty_stays_unknown(monkeypatch):
    from mm_assist.life_execute import ui_logic
    from mm_assist.memory.readers.ui_stack import UiStackSnap
    class Reader:
        def read(self, **kwargs):
            return UiStackSnap(names=[], ready=False, empty_confirmed=False)
    monkeypatch.setattr("mm_assist.life_plan.node_inspect.compute_inspect", lambda *a, **k: {"node": "main_hud"})
    monkeypatch.setattr(ui_logic, "_hud_entry_ok", lambda *a, **k: True)
    assert ui_logic.resolve_navigation(object(), reader=Reader()).node == "unknown"


def test_navigation_accepts_bag_when_menu_icon_misses(monkeypatch):
    from mm_assist.life_execute import ui_logic
    from mm_assist.memory.readers.ui_stack import UiStackSnap

    class Reader:
        def read(self, **kwargs):
            return UiStackSnap(names=[], ready=True, empty_confirmed=True)

    monkeypatch.setattr(
        "mm_assist.life_plan.node_inspect.compute_inspect",
        lambda *a, **k: {"node": "main_hud"},
    )
    monkeypatch.setattr(
        ui_logic,
        "_hud_entry_ok",
        lambda _f, eid, **k: eid == "bag",
    )
    assert ui_logic.resolve_navigation(object(), reader=Reader()).node == "main_hud"


def test_profile_tab_identity_maps_to_profile_navigation():
    from mm_assist.life_execute.ui_logic import graph_node
    assert graph_node("profile:info") == "profile"
    assert graph_node("profile:combat") == "profile"


def test_navigation_plain_inventory_skips_item_ocr_but_keeps_overlay_recognition(monkeypatch):
    from mm_assist.life_execute.ui_logic import resolve_navigation
    from mm_assist.memory.readers.ui_stack import UiStackSnap
    names = ["EquipmentInventoryUIInstance", "SizeAdjustmentSender"]
    class Reader:
        def read(self, **kwargs):
            return UiStackSnap(names=list(names), ready=True)
    calls = []
    monkeypatch.setattr("mm_assist.life_plan.node_inspect.compute_inspect",
                        lambda *a, **k: calls.append(True) or {"node": "item_info"})
    assert resolve_navigation(object(), reader=Reader()).node == "inventory"
    assert not calls
    names.append("DetailCardUIInstance")
    assert resolve_navigation(object(), reader=Reader()).node == "item_info"
    assert calls


@pytest.mark.parametrize("refined,names", [
    ("facility:metal:鋼錠", ["UILayerHideState", "NewProfileUIInstance"]),
    ("life_grid", ["CraftingHubUIState"]),
])
def test_resolve_profile_rejects_unrelated_or_stale_refinement(monkeypatch, refined, names):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: _menu1_view(profile_open=True, ui_stack_names=["UILayerHideState", "NewProfileUIInstance"]))
    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: refined)
    monkeypatch.setattr(ui_logic, "read_inspect_record", lambda **k: {"node": refined, "stack_names": names})
    assert ui_logic.graph_node(ui_logic.resolve_ui_node().node) == "profile"


def test_resolve_inspect_override_same_compare_key(monkeypatch):
    from mm_assist.life_execute import ui_logic
    from mm_assist.memory.readers.ui_stack import stack_compare_key

    names_mem = ["HudUIStepState", "CraftingHubUIState"]
    names_insp = ["CraftingHubUIState", "HudUIStepState"]
    assert list(names_mem) != list(names_insp)
    assert stack_compare_key(names_mem) == stack_compare_key(names_insp)

    class V:
        params = "inv=0 menu=1 dlg=0 stp=0"
        family = "craft_hub"
        dialog_active = False
        menu_open = True
        map_open = False
        profile_open = False
        stp = 0
        craft_hub_category = None
        craft_tab = ""
        relation = "加工"
        ui_stack_names = names_mem
        ui_stack_ready = True

    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: "facility:metal:鋼錠")
    monkeypatch.setattr(
        ui_logic,
        "read_inspect_record",
        lambda **k: {"node": "facility:metal:鋼錠", "stack_names": names_insp},
    )
    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: V())
    where = ui_logic.resolve_ui_node(frame=None)
    assert where.node == "facility:metal:鋼錠"
    assert where.source == "inspect"


def test_ui_param_key_frozen_inspect_node_empty(tmp_path):
    import json
    import os
    import time

    from mm_assist.life_execute import ui_logic

    logs = tmp_path / "logs"
    logs.mkdir()
    now = time.time()
    stale = now - ui_logic.INSPECT_MAX_AGE_S - 1.0
    (logs / "ui_inspect.json").write_text(
        json.dumps(
            {"ts_unix": stale, "node": "craft_hub", "pid": os.getpid()},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    su = logs / "state_watch_ui.json"
    last_node = None
    for i in range(3):
        su.write_text(
            json.dumps(
                {
                    "ts_unix": now + i,
                    "ui_params": "stp=0",
                    "stp": 0,
                    "ui_stack_ready": True,
                    "ui_stack_names": ["TutorialTooltipUIState"],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _key, node = ui_logic.ui_param_key(root=tmp_path)
        last_node = node
        assert node == ""
    assert last_node == ""


def test_ui_param_key_foreign_pid_node_empty(tmp_path):
    import json
    import time

    from mm_assist.life_execute import ui_logic

    logs = tmp_path / "logs"
    logs.mkdir()
    now = time.time()
    (logs / "ui_inspect.json").write_text(
        json.dumps(
            {"ts_unix": now, "node": "craft_hub", "pid": 1},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / "state_watch_ui.json").write_text(
        json.dumps(
            {
                "ts_unix": now,
                "ui_params": "stp=0",
                "stp": 0,
                "ui_stack_ready": True,
                "ui_stack_names": ["TutorialTooltipUIState"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _key, node = ui_logic.ui_param_key(root=tmp_path)
    assert node == ""


def test_write_inspect_page_includes_pid(tmp_path):
    import os

    from mm_assist.life_execute import ui_logic

    ui_logic.write_inspect_page("main_hud", root=tmp_path, stack_ready=True)
    rec = ui_logic.read_inspect_record(root=tmp_path)
    assert rec.get("pid") == os.getpid()
    assert rec.get("node") == "main_hud"


def test_node_from_mem_not_ready_is_unknown():
    from types import SimpleNamespace

    from mm_assist.life_execute import ui_logic

    v = SimpleNamespace(
        params="inv=0 menu=0 dlg=0 stp=0",
        family="main_hud",
        relation="加工",
        ui_stack_names=[],
        ui_stack_ready=False,
        craft_hub_category=1,
        menu_open=False,
        dialog_active=False,
        map_open=False,
        profile_open=False,
        stp=0,
    )
    node, _tab = ui_logic.node_from_mem(v)
    assert node == "unknown"


def test_resolve_inspect_requires_stack_ready(monkeypatch):
    from mm_assist.life_execute import ui_logic

    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: "facility:metal:鋼錠")
    monkeypatch.setattr(
        ui_logic,
        "read_inspect_record",
        lambda **k: {"node": "facility:metal:鋼錠", "stack_names": ["HudUIStepState"]},
    )
    monkeypatch.setattr(
        ui_logic,
        "read_state_ui",
        lambda **k: _menu1_view(ui_stack_ready=False, ui_stack_names=[]),
    )
    where = ui_logic.resolve_ui_node(frame=None)
    assert where.node == "unknown"
    assert where.stack_ready is False


def test_ui_settled_requires_stack_ready_not_menu_bit():
    from mm_assist.life_execute import ui_logic

    ready = ui_logic.UiWhere(
        "main_hud", None, None, 0, "mem", stack_ready=True
    )
    assert ui_logic.ui_settled(resolve=lambda _f=None: ready) is True
    pending = ui_logic.UiWhere("main_hud", 0, 0, 0, "mem", stack_ready=False)
    assert ui_logic.ui_settled(resolve=lambda _f=None: pending) is False
    unknown = ui_logic.UiWhere("unknown", None, None, 0, "mem", stack_ready=True)
    assert ui_logic.ui_settled(resolve=lambda _f=None: unknown) is False


def test_write_inspect_page_refreshes_ts_when_node_same(tmp_path):
    import time

    from mm_assist.life_execute import ui_logic

    ui_logic.write_inspect_page(
        "main_hud", zh="主界面", root=tmp_path, stack_ready=True, source="stack"
    )
    rec1 = ui_logic.read_inspect_record(root=tmp_path)
    t1 = float(rec1["ts_unix"])
    time.sleep(0.02)
    ui_logic.write_inspect_page(
        "main_hud", zh="主界面", root=tmp_path, stack_ready=True, source="stack"
    )
    rec2 = ui_logic.read_inspect_record(root=tmp_path)
    assert rec2.get("node") == "main_hud"
    assert float(rec2["ts_unix"]) > t1


def test_write_inspect_page_cost_does_not_block_same(tmp_path):
    import time

    from mm_assist.life_execute import ui_logic

    ui_logic.write_inspect_page(
        "main_hud",
        zh="主界面",
        root=tmp_path,
        stack_ready=True,
        source="stack",
        cost={"tpl_n": 1, "ocr_n": 0, "ms": 10.0},
    )
    rec1 = ui_logic.read_inspect_record(root=tmp_path)
    t1 = float(rec1["ts_unix"])
    time.sleep(0.02)
    ui_logic.write_inspect_page(
        "main_hud",
        zh="主界面",
        root=tmp_path,
        stack_ready=True,
        source="stack",
        cost={"tpl_n": 9, "ocr_n": 2, "ms": 99.0},
    )
    rec2 = ui_logic.read_inspect_record(root=tmp_path)
    assert rec2.get("node") == "main_hud"
    assert float(rec2["ts_unix"]) > t1
    assert rec2.get("cost") == {"tpl_n": 9, "ocr_n": 2, "ms": 99.0}


def test_node_from_mem_matches_stack_node_captured():
    from types import SimpleNamespace

    from mm_assist.life_execute import ui_logic
    from mm_assist.memory.readers.ui_stack import stack_node

    cases = [
        (["TutorialTooltipUIState"], True, None),
        (
            ["TutorialTooltipUIState", "SideNavigationUIState", "HudUIStepState"],
            True,
            None,
        ),
        (
            [
                "TutorialTooltipUIState",
                "EquipmentInventoryUIState",
                "HudUIStepState",
            ],
            True,
            None,
        ),
        (
            [
                "TutorialTooltipUIState",
                "DialogueInteractionUIState",
                "HudUIStepState",
            ],
            True,
            None,
        ),
        (
            [
                "SkipButtonUIState",
                "DialogueInteractionUIState",
                "HudUIStepState",
            ],
            True,
            None,
        ),
        (
            [
                "DetailCardUIState",
                "EquipmentInventoryUIState",
                "HudUIStepState",
            ],
            True,
            None,
        ),
        (["CraftingHubUIState", "HudUIStepState"], True, 1),
        (["HudUIStepState"], True, None),
        (["CashShopUIState", "HudUIStepState"], True, None),
        ([], True, None),
        (["TutorialTooltipUIState"], False, None),
    ]
    for names, ready, cat in cases:
        sn = stack_node(names, ready=ready, craft_category=cat)
        v = SimpleNamespace(
            ui_stack_names=names,
            ui_stack_ready=ready,
            craft_hub_category=cat,
            craft_tab="",
        )
        node, _tab = ui_logic.node_from_mem(v)
        if not ready or not sn:
            assert node == "unknown", (names, ready, sn, node)
        else:
            assert node == sn, (names, ready, sn, node)


def test_node_from_mem_overlay_not_guessed():
    from types import SimpleNamespace

    from mm_assist.life_execute import ui_logic

    v = SimpleNamespace(
        params="inv=0 menu=1 dlg=0 stp=0",
        family="craft_hub",
        relation="加工",
        ui_stack_names=["HudUIStepState"],
        ui_stack_ready=True,
        craft_hub_category=None,
        menu_open=True,
        dialog_active=False,
        map_open=False,
        profile_open=False,
        craft_tab="加工",
        stp=0,
    )
    node, tab = ui_logic.node_from_mem(v)
    assert node == "overlay"
    assert tab == ""  # A retained label without a current shell is not page evidence.
