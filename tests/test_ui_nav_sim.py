from __future__ import annotations

import pytest
import time
from types import SimpleNamespace

from mm_assist.life_execute.ui_logic import UiWhere
from mm_assist.life_execute import ui_nav
from mm_assist.life_plan.execute import is_nav_result, is_sense_result
from mm_assist.safety.autofarm_gates import allow_stage_action, reset_stage_gates_for_tests


def test_direct_navigation_never_uses_display_files(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("navigation read a diagnostic display file")
    monkeypatch.setattr("mm_assist.life_execute.ui_logic.ui_param_stamps", forbidden)
    monkeypatch.setattr(ui_nav, "ui_param_key", forbidden)
    monkeypatch.setattr(ui_nav, "read_inspect_record", forbidden)
    monkeypatch.setattr(ui_nav, "resolve_navigation", lambda f: UiWhere("unknown", None, None, None, "direct_pending"))
    clicks = []
    result = ui_nav.ensure_node("profile", grab=lambda: object(), click=lambda *xy: clicks.append(xy), sleep_fn=lambda s: None)
    assert result.status == "unknown_ui"
    assert not clicks


def test_direct_transient_unknown_waits_for_fresh_page_without_input(monkeypatch):
    rows = iter([UiWhere("unknown", None, None, None, "direct_pending"),
                 UiWhere("system_menu", 1, 0, None, "direct")])
    monkeypatch.setattr(ui_nav, "resolve_navigation", lambda f: next(rows))
    result = ui_nav.ensure_node("system_menu", grab=lambda: object(),
        click=lambda *xy: pytest.fail("already on menu"), sleep_fn=lambda s: None)
    assert result.ok


def test_ensure_node_common_popup_clicks_confirm_then_arrives(monkeypatch):
    import numpy as np
    names = (
        "HudUIStepState",
        "AlteringFacilityPopupUIInstance",
        "ButtonCommonPopupUIState",
        "ButtonCommonPopupInstance",
        "ClickShieldUIState",
    )
    pages = iter(
        [
            UiWhere("overlay", 1, 0, None, "direct", stack_names=names, stack_ready=True),
            UiWhere("profile", 1, 0, None, "direct"),
        ]
    )
    present = {"n": True}
    monkeypatch.setattr('mm_assist.life_execute.ui_act.read_state_ui', lambda **_: SimpleNamespace(
        fresh=True, pid_alive=True, raw={'ui_stack_ready': True,
        'ui_stack_ts': time.time(), 'ui_stack_names': list(names) if present['n'] else ['PlayerProfilePopupUIInstance']}))

    def _has(*_a, **k):
        if k.get("names"):
            return True
        return present["n"]

    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_common_popup", _has)
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_common_popup_confirm",
        lambda *a, **k: {"ok": True, "click_client": [1100, 980]},
    )
    monkeypatch.setattr(ui_nav, "resolve_navigation", lambda f: next(pages))
    clicks: list[tuple[int, int]] = []

    def click(x, y):
        clicks.append((x, y))
        present["n"] = False
        return True

    result = ui_nav.ensure_node(
        "profile", grab=lambda: np.zeros((1442, 1912, 3), dtype=np.uint8), click=click,
        sleep_fn=lambda s: pytest.fail('recognized popup must be handled before waiting on its shield')
    )
    assert result.ok
    assert clicks == [(1100, 980)]


def test_ensure_node_facility_result_clicks_confirm_then_arrives(monkeypatch):
    names = (
        "CraftingHubUIState",
        "FacilityResultUIState",
        "AlteringFacilityPopupUIInstance",
        "FacilityResultUIInstance",
    )
    pages = iter(
        [
            UiWhere("facility:cloth", 1, 0, None, "direct", stack_names=names, stack_ready=True),
            UiWhere("facility:wood", 1, 0, None, "direct"),
        ]
    )
    present = {"n": True}
    monkeypatch.setattr('mm_assist.life_execute.ui_act.read_state_ui', lambda **_: SimpleNamespace(
        fresh=True, pid_alive=True, raw={'ui_stack_ready': True,
        'ui_stack_ts': time.time(), 'ui_stack_names': list(names) if present['n'] else ['AlteringFacilityPopupUIInstance']}))
    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_common_popup", lambda *a, **k: False)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_facility_result", lambda *a, **k: present["n"])
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": True, "click_client": [1100, 980]},
    )
    monkeypatch.setattr(ui_nav, "resolve_navigation", lambda f: next(pages))
    clicks: list[tuple[int, int]] = []

    def click(x, y):
        clicks.append((x, y))
        present["n"] = False
        return True

    result = ui_nav.ensure_node(
        "facility", facility="wood", grab=lambda: object(), click=click, sleep_fn=lambda s: None
    )
    assert result.ok
    assert clicks == [(1100, 980)]


def test_real_prepare_uses_direct_navigation_without_display_alignment(monkeypatch):
    from mm_assist.life_execute import ui_act
    where = UiWhere("life_grid", 1, 0, None, "direct")
    monkeypatch.setattr("mm_assist.life_execute.ui_logic.resolve_navigation", lambda f: where)
    monkeypatch.setattr(ui_nav, "resolve_navigation", lambda f: where)
    monkeypatch.setattr(ui_act, "inspect_is_menu", lambda: pytest.fail("legacy display alignment"))
    result = ui_act.prepare_to("life_grid", grab=lambda: object(), click=lambda *xy: pytest.fail("already at target"))
    assert result.ok


def test_prepare_confirmed_hud_unknown_node_is_not_unknown_ui(monkeypatch):
    from mm_assist.life_execute import ui_act
    from mm_assist.life_plan.types import ExecuteResult

    where = UiWhere("unknown", 0, 0, 0, "direct", stack_names=(), stack_ready=True)
    monkeypatch.setattr("mm_assist.life_execute.ui_logic.resolve_navigation", lambda f: where)
    monkeypatch.setattr(ui_nav, "resolve_navigation", lambda f: where)
    monkeypatch.setattr(ui_act, "inspect_is_hud_idle", lambda: False)
    monkeypatch.setattr(ui_act, "inspect_is_true_hud", lambda: False)
    monkeypatch.setattr(
        ui_nav,
        "ensure_node",
        lambda *a, **k: ExecuteResult(True, "at_node", "gather_skill"),
    )
    result = ui_act.prepare_to(
        "gather_skill",
        grab=lambda: object(),
        click=lambda *xy: pytest.fail("true hud must not fail-closed"),
        sleep_fn=lambda _s: None,
        read_stp=lambda: 0,
        skill_id="daily",
    )
    assert result.ok
    assert result.status != "unknown_ui"


def test_direct_waits_for_menu_control_animation_without_reclick(monkeypatch):
    page = ["system_menu"]
    hits = iter([None, (90, 22), (11, 22), (11, 22)])
    monkeypatch.setattr(ui_nav, "resolve_navigation", lambda f: UiWhere(page[0], 1, 0, None, "direct"))
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: next(hits))
    clicks = []
    def click(*xy):
        clicks.append(xy)
        page[0] = "profile"
        return True
    result = ui_nav.ensure_node("profile", grab=lambda: object(), click=click, sleep_fn=lambda s: None)
    assert result.ok
    assert clicks == [(11, 22)]


def test_unnamed_skill_page_does_not_confirm_requested_skill():
    result = ui_nav.ensure_node("gather_skill", skill_id="herb", grab=lambda: object(),
        click=lambda *xy: pytest.fail("unidentified skill"), sleep_fn=lambda s: None,
        resolve=lambda f: UiWhere("gather_skill", 1, 0, None, "mem"))
    assert not result.ok
    assert result.status == "unknown_skill"


def test_craft_button_not_found_never_returns_fixed_coordinate(monkeypatch):
    import numpy as np
    monkeypatch.setattr(ui_nav, "_match", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.perception.menu_icons.locate_esc_cell_by_label", lambda *a, **k: None)
    assert ui_nav.locate_edge({"id": "system_menu_craft_craft_hub"}, np.zeros((900, 1600, 3), dtype=np.uint8)) is None


@pytest.mark.parametrize("edge_id,ocr", [
    ("system_menu_role_profile", "locate_esc_char_by_label"),
    ("system_menu_craft_craft_hub", "locate_esc_cell_by_label"),
])
def test_menu_icon_match_avoids_ocr(monkeypatch, edge_id, ocr):
    monkeypatch.setattr(ui_nav, "_match", lambda *a, **k: {"ok": True, "click_client": [1250, 220]})
    monkeypatch.setattr("mm_assist.perception.menu_icons." + ocr, lambda *a, **k: pytest.fail("unnecessary OCR"))
    assert ui_nav.locate_edge({"id": edge_id}, object()) == (1250, 220)


@pytest.mark.parametrize("edge_id,expected_x", [
    ("system_menu_role_profile", 1327),
    ("system_menu_bag_inventory", 1447),
    ("system_menu_craft_craft_hub", 1567),
])
def test_real_tall_menu_buttons_match_the_visible_cell(edge_id, expected_x):
    import cv2
    from pathlib import Path
    frame = cv2.imread(str(Path(__file__).parent / "fixtures/ui_benchmark/system_menu_real.png"))
    xy = ui_nav.locate_edge({"id": edge_id}, frame)
    assert xy is not None
    assert abs(xy[0] - expected_x) < 25
    assert 210 < xy[1] < 265


@pytest.mark.parametrize("edge_id", [
    "system_menu_role_profile", "system_menu_bag_inventory", "system_menu_craft_craft_hub",
])
def test_menu_animation_without_buttons_never_produces_click(edge_id):
    import cv2
    from pathlib import Path
    frame = cv2.imread(str(Path(__file__).parent / "fixtures/ui_benchmark/system_menu_transition.png"))
    assert frame is not None
    assert ui_nav.locate_edge({"id": edge_id}, frame) is None


@pytest.fixture(autouse=True)
def _reset_nav_sim(monkeypatch):
    reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: False,
    )
    yield
    reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()


def test_stale_param_content_skip_click(monkeypatch):
    monkeypatch.setattr(ui_nav, "skip_wait_ui_params", lambda: False)
    n = {"i": 0}

    def keys(*, root=None):
        n["i"] += 1
        return (f"inv={n['i']}", "")

    monkeypatch.setattr("mm_assist.life_execute.ui_logic.ui_param_key", keys)
    monkeypatch.setattr(ui_nav, "ui_param_key", keys)
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (11, 22))
    clicks: list[tuple[int, int]] = []
    r = ui_nav.ensure_node(
        "profile",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("system_menu", 1, 0, 0, "mem"),
    )
    assert clicks == []
    assert r.ok is False


def test_param_file_ts_bump_still_clicks(monkeypatch):
    monkeypatch.setattr(ui_nav, "skip_wait_ui_params", lambda: False)
    n = {"i": 0}

    def stamps(*, root=None):
        n["i"] += 1
        return (float(n["i"]), 0.0)

    monkeypatch.setattr("mm_assist.life_execute.ui_logic.ui_param_stamps", stamps)
    monkeypatch.setattr(ui_nav, "ui_param_stamps", stamps)
    monkeypatch.setattr(ui_nav, "ui_param_key", lambda **k: ("inv=0 menu=1", "system_menu"))
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (11, 22))
    clicks: list[tuple[int, int]] = []
    r = ui_nav.ensure_node(
        "profile",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("system_menu", 1, 0, 0, "mem"),
    )
    assert clicks == [(11, 22)]
    assert r.status == "nav_verify_timeout"


def test_gather_chain_one_click_per_edge(monkeypatch):
    page = {"n": "system_menu"}
    seen: list[str] = []

    def do_click(x, y):
        cur = page["n"]
        seen.append(cur)
        nxt = {
            "system_menu": "profile",
            "profile": "life_grid",
            "life_grid": "gather_skill:mine",
        }.get(cur)
        if nxt:
            page["n"] = nxt
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (1, 1))
    r = ui_nav.ensure_node(
        "gather_skill",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        skill_id="mine",
    )
    assert r.ok is True
    assert seen == ["system_menu", "profile", "life_grid"]
    assert page["n"] == "gather_skill:mine"


def test_skill_list_is_not_item_detail(monkeypatch):
    clicks = []
    result = ui_nav.ensure_node(
        "gather_detail", grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("gather_skill", 1, 0, 0, "mem"),
    )
    assert not result.ok
    assert clicks == []


def test_wrong_skill_list_cannot_pass_real_prepare(monkeypatch):
    from mm_assist.life_execute import ui_act

    monkeypatch.setattr(ui_act, "inspect_is_menu", lambda: False)
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: None)
    clicks = []
    result = ui_act.prepare_to(
        "gather_skill", skill_id="herb", grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("gather_skill:mine", 1, 0, 0, "mem"),
    )
    assert not result.ok
    assert clicks == []


def test_cross_skill_returns_to_grid_before_selecting_target(monkeypatch):
    page = ["gather_skill:mine"]
    edges = []
    def locate(edge, *a, **k):
        edges.append(edge["id"])
        return (10, 20)
    def click(*xy):
        page[0] = "life_grid" if page[0] == "gather_skill:mine" else "gather_skill:herb"
        return True
    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    result = ui_nav.ensure_node("gather_skill", skill_id="herb", grab=lambda: object(), click=click,
        sleep_fn=lambda s: None, resolve=lambda f: UiWhere(page[0], 1, 0, None, "mem"))
    assert result.ok
    assert edges == ["gather_skill_close_life_grid", "life_grid_skill_gather_detail"]


def test_late_page_change_two_frames_one_click(monkeypatch):
    grabs = {"n": 0}

    def grab():
        grabs["n"] += 1
        return grabs["n"]

    def resolve(_f=None):
        n = grabs["n"]
        node = "profile" if n >= 4 else "system_menu"
        return UiWhere(node, 1, 0, 0, "mem")

    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (11, 22))
    clicks: list[tuple[int, int]] = []
    r = ui_nav.ensure_node(
        "profile",
        grab=grab,
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=resolve,
    )
    assert r.ok is True
    assert clicks == [(11, 22)]


@pytest.mark.parametrize("already_ok", [None, False], ids=["default-arrival", "explicit-arrival-rejected"])
def test_never_leave_page_verify_timeout(monkeypatch, already_ok):
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (11, 22))
    clicks: list[tuple[int, int]] = []
    r = ui_nav.ensure_node(
        "profile",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("system_menu", 1, 0, 0, "mem"),
        already_ok=None if already_ok is None else lambda _f: False,
    )
    assert r.ok is False
    assert r.status == "nav_verify_timeout"
    assert clicks == [(11, 22)]
    assert is_nav_result(r) is True
    assert is_sense_result(r) is False


def test_second_ensure_does_not_reclick(monkeypatch):
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (11, 22))
    clicks: list[tuple[int, int]] = []

    def run():
        return ui_nav.ensure_node(
            "profile",
            grab=lambda: object(),
            click=lambda x, y: clicks.append((x, y)) or True,
            sleep_fn=lambda s: None,
            resolve=lambda _f=None: UiWhere("system_menu", 1, 0, 0, "mem"),
        )

    r1 = run()
    r2 = run()
    assert r1.status == "nav_verify_timeout"
    assert r2.status == "nav_no_move"
    assert clicks == [(11, 22)]
    assert "已点过" in r2.message


def test_old_page_stage_full_chain_still_runs(monkeypatch):
    assert allow_stage_action("切页") is True
    assert allow_stage_action("切页") is True
    assert allow_stage_action("切页") is False
    page = {"n": "system_menu"}

    def do_click(x, y):
        nxt = {
            "system_menu": "profile",
            "profile": "life_grid",
            "life_grid": "gather_skill:mine",
        }.get(page["n"])
        if nxt:
            page["n"] = nxt
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (1, 1))
    r = ui_nav.ensure_node(
        "gather_skill",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        skill_id="mine",
    )
    assert r.ok is True
    assert page["n"] == "gather_skill:mine"


def test_wait_progress_is_human_not_edge_id(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (11, 22))
    ui_nav.ensure_node(
        "profile",
        grab=lambda: object(),
        click=lambda x, y: True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("system_menu", 1, 0, 0, "mem"),
        on_progress=seen.append,
    )
    assert "等新页" in seen
    assert all("system_menu_role_profile" not in s for s in seen)


def test_locate_unknown_template_edge_no_full_frame():
    assert ui_nav.locate_edge({"id": "not_a_real_edge"}, object()) is None


def test_edge_stage_uses_edge_id():
    assert (
        ui_nav.edge_stage({"id": "system_menu_role_profile", "trigger": {"label": "角色"}})
        == "system_menu_role_profile"
    )
    assert (
        ui_nav.edge_stage({"id": "main_hud_menu_system_menu", "trigger": {"label": "打开菜单"}})
        == "main_hud_menu_system_menu"
    )
    assert ui_nav.edge_stage({"id": "profile_close_main_hud", "trigger": {"label": "关闭"}}) == "清界面"


def test_path_edges_alchemy_to_craft_hub_has_tab():
    hops = ui_nav.path_edges("craft_alchemy", "craft_hub") or []
    assert hops
    assert hops[0]["id"] == "craft_alchemy_tab_craft_hub"


def test_path_edges_craft_hub_to_quick_tab():
    hops = ui_nav.path_edges("craft_hub", "craft_quick") or []
    assert hops
    assert hops[0]["id"] == "craft_hub_tab_craft_quick"


def test_path_edges_skill_item_and_recipe():
    hops = ui_nav.path_edges("gather_skill", "item_info") or []
    assert hops and hops[0]["id"] == "gather_skill_item_item_info"
    hops = ui_nav.path_edges("item_info", "obtain_method") or []
    assert hops and hops[0]["id"] == "item_info_open_obtain_method"
    hops = ui_nav.path_edges("gather_detail", "obtain_method") or []
    assert hops and hops[0]["id"] == "gather_detail_open_obtain_method"
    hops = ui_nav.path_edges("gather_skill", "gather_detail") or []
    assert hops and hops[0]["id"] == "gather_skill_item_gather_detail"
    hops = ui_nav.path_edges("facility", "facility_recipe") or []
    assert hops and hops[0]["id"] == "facility_open_recipe"
    hops = ui_nav.path_edges("facility:metal", "facility:wood:木材") or []
    assert hops and hops[0]["id"] == "facility_back_craft_hub"
    hops = ui_nav.path_edges("facility:metal", "facility:wood") or []
    assert hops and hops[0]["id"] == "facility_back_craft_hub"
    assert ui_nav.path_edges("facility_recipe", "facility") == []
    hops = ui_nav.path_edges("gather_detail:鐵礦脈", "life_grid") or []
    assert hops and hops[0]["id"] == "gather_detail_close_life_grid"


def test_ensure_wrong_skill_detail_leaves_before_item(monkeypatch):
    page = {"n": "gather_detail:鐵礦脈"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        ids.append(str(edge.get("id") or ""))
        return (1, 1)

    def do_click(x, y):
        nxt = {
            "gather_detail:鐵礦脈": "life_grid",
            "life_grid": "gather_skill:herb",
            "gather_skill:herb": "gather_detail:淨淨蘑菇",
        }.get(page["n"])
        if nxt:
            page["n"] = nxt
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "gather_detail",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        skill_id="herb",
        item_name="淨淨蘑菇",
    )
    assert r.ok is True
    assert page["n"] == "gather_detail:淨淨蘑菇"
    assert "gather_detail_close_life_grid" in ids
    assert "gather_skill_item_gather_detail" in ids


def test_ensure_sheep_from_unlabeled_detail_closes_not_obtain(monkeypatch):
    page = {"n": "gather_detail"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        ids.append(str(edge.get("id") or ""))
        return (1, 1)

    def do_click(x, y):
        nxt = {
            "gather_detail": "life_grid",
            "life_grid": "gather_skill:shear",
            "gather_skill:shear": "gather_detail:羊",
        }.get(page["n"])
        if nxt:
            page["n"] = nxt
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "item_info",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        skill_id="shear",
        item_name="羊",
    )
    assert r.ok is True
    assert ids[0] != "gather_detail_open_obtain_method"
    assert "gather_detail_open_obtain_method" not in ids
    assert page["n"] == "gather_detail:羊"


def test_ensure_daily_list_to_woodcut_item_info_closes_first(monkeypatch):
    page = {"n": "gather_skill:daily"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        eid = str(edge.get("id") or "")
        ids.append(eid)
        skill = str(k.get("skill_id") or "")
        if eid in {"gather_skill_item_item_info", "gather_skill_item_gather_detail"}:
            if page["n"] != f"gather_skill:{skill}":
                return None
        return (1, 1)

    def do_click(x, y):
        nxt = {
            "gather_skill:daily": "life_grid",
            "life_grid": "gather_skill:woodcut",
            "gather_skill:woodcut": "item_info:樹木",
        }.get(page["n"])
        if nxt:
            page["n"] = nxt
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "item_info",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        skill_id="woodcut",
        item_name="樹木",
    )
    assert r.ok is True
    assert "找不到物名" not in str(r.message or "")
    assert r.status != "物名"
    assert ids[0] == "gather_skill_close_life_grid"
    assert "life_grid_skill_gather_detail" in ids
    assert "gather_skill_item_item_info" in ids
    assert page["n"] == "item_info:樹木"


def test_ensure_nest_detail_to_web_item_info_closes_before_obtain(monkeypatch):
    page = {"n": "gather_detail:巢穴"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        eid = str(edge.get("id") or "")
        ids.append(eid)
        item = str(k.get("item_name") or "")
        if eid in {"item_info_open_obtain_method", "gather_detail_open_obtain_method"}:
            if item and item not in page["n"]:
                return None
        if eid in {"gather_skill_item_item_info", "gather_skill_item_gather_detail"}:
            skill = str(k.get("skill_id") or "")
            if page["n"] != f"gather_skill:{skill}":
                return None
        return (1, 1)

    def do_click(x, y):
        nxt = {
            "gather_detail:巢穴": "life_grid",
            "life_grid": "gather_skill:daily",
            "gather_skill:daily": "item_info:蜘蛛網",
        }.get(page["n"])
        if nxt:
            page["n"] = nxt
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "item_info",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        skill_id="daily",
        item_name="蜘蛛網",
    )
    assert r.ok is True
    assert ids
    assert ids[0] != "gather_detail_open_obtain_method"
    assert ids[0] == "gather_detail_close_life_grid"
    assert "gather_detail_open_obtain_method" not in ids
    assert "找不到获得方法" not in str(r.message or "")
    assert page["n"] == "item_info:蜘蛛網"


def test_ensure_wrong_detail_close_landing_on_skill_list_continues(monkeypatch):
    page = {"n": "gather_detail:蜘蛛網"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        eid = str(edge.get("id") or "")
        ids.append(eid)
        if eid in {"item_info_open_obtain_method", "gather_detail_open_obtain_method"}:
            return None
        return (1, 1)

    def do_click(x, y):
        nxt = {
            "gather_detail:蜘蛛網": "gather_skill:daily",
            "gather_skill:daily": "item_info:水井",
        }.get(page["n"])
        if nxt:
            page["n"] = nxt
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "item_info",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        skill_id="daily",
        item_name="水井",
    )
    assert r.ok is True
    assert ids[0] == "gather_detail_close_life_grid"
    assert r.status != "wrong_page"
    assert page["n"] == "item_info:水井"


def test_ensure_craft_hub_from_alchemy_not_fake_ok(monkeypatch):
    monkeypatch.setattr(ui_nav, "craft_hub_visible", lambda *a, **k: False)
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: None)
    clicks: list[tuple[int, int]] = []
    r = ui_nav.ensure_node(
        "craft_hub",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("craft_alchemy", 1, 0, 0, "mem"),
    )
    assert r.ok is False
    assert r.status != "at_node"
    assert clicks == []


def test_craft_hub_tab_craft_make_stale_quick_not_wrong_page(monkeypatch):
    page = {"n": "craft_hub", "polls": 0}

    def locate(edge, *a, **k):
        return (1, 1)

    def do_click(x, y):
        page["n"] = "craft_quick"
        return True

    def resolve(_f=None):
        if page["n"] == "craft_quick":
            page["polls"] += 1
            if page["polls"] >= 2:
                page["n"] = "craft_make"
        return UiWhere(page["n"], 1, 0, 0, "mem")

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "craft_make",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=resolve,
    )
    assert r.ok is True
    assert r.status != "wrong_page"
    assert page["n"] == "craft_make"


def test_craft_hub_tab_craft_make_clicks_not_skip(monkeypatch):
    page = {"n": "craft_hub"}
    labels: list[str] = []
    located: list[str] = []

    def locate(edge, *a, **k):
        eid = str(edge.get("id") or "")
        located.append(eid)
        labels.append(str((edge.get("trigger") or {}).get("label") or ""))
        return (1, 1)

    def do_click(x, y):
        if page["n"] == "craft_hub":
            page["n"] = "craft_make"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "craft_make",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
    )
    assert r.ok is True
    assert "craft_hub_tab_craft_make" in located
    assert labels == ["制作"]
    assert page["n"] == "craft_make"


def test_ensure_craft_hub_from_alchemy_clicks_process_tab(monkeypatch):
    page = {"n": "craft_alchemy"}
    seen: list[str] = []
    labels: list[str] = []

    def locate(edge, *a, **k):
        labels.append(str((edge.get("trigger") or {}).get("label") or ""))
        return (1, 1)

    def do_click(x, y):
        seen.append(page["n"])
        if page["n"] == "craft_alchemy":
            page["n"] = "craft_hub"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    monkeypatch.setattr(ui_nav, "craft_hub_visible", lambda *a, **k: page["n"] == "craft_hub")
    r = ui_nav.ensure_node(
        "craft_hub",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
    )
    assert r.ok is True
    assert seen == ["craft_alchemy"]
    assert labels == ["加工"]
    assert page["n"] == "craft_hub"


def test_menu_opens_quick_then_process_tab_to_facility(monkeypatch):
    page = {"n": "system_menu"}
    labels: list[str] = []

    def locate(edge, *a, **k):
        labels.append(str((edge.get("trigger") or {}).get("label") or ""))
        return (1, 1)

    def do_click(x, y):
        nxt = {
            "system_menu": "craft_quick",
            "craft_quick": "craft_hub",
            "craft_hub": "facility",
        }.get(page["n"])
        if nxt:
            page["n"] = nxt
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    monkeypatch.setattr(ui_nav, "craft_hub_visible", lambda *a, **k: page["n"] == "craft_hub")
    r = ui_nav.ensure_node(
        "facility",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        already_ok=lambda _f: page["n"] == "facility",
        facility="metal",
    )
    assert r.ok is True
    assert page["n"] == "facility"
    assert "加工" in labels
    assert r.status != "wrong_page"


def test_locate_craft_tab_edge_uses_node_inspect(monkeypatch):
    seen: list[str] = []

    def fake_xy(frame, spot):
        seen.append(spot)
        return (12, 34)

    monkeypatch.setattr("mm_assist.life_plan.node_inspect.craft_tab_xy", fake_xy)
    xy = ui_nav.locate_edge(
        {
            "id": "craft_alchemy_tab_craft_hub",
            "from": "craft_alchemy",
            "to": "craft_hub",
            "trigger": {"label": "加工"},
        },
        object(),
    )
    assert xy == (12, 34)
    assert seen == ["加工"]


def test_edge_stage_craft_tab_is_edge_id():
    assert (
        ui_nav.edge_stage(
            {"id": "craft_alchemy_tab_craft_hub", "trigger": {"label": "加工"}}
        )
        == "craft_alchemy_tab_craft_hub"
    )


def test_ensure_node_facility_unverified_does_not_click_back(monkeypatch):
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (7, 8))
    clicks: list[tuple[int, int]] = []
    r = ui_nav.ensure_node(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("facility", 1, 0, 0, "mem"),
        already_ok=lambda _f: False,
    )
    assert clicks == []
    assert r.ok is False
    assert r.status == "need_verify"
    assert "不退页" in str(r.message or "")


def test_ensure_node_unknown_facility_does_not_leave_when_target_named(monkeypatch):
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (7, 8))
    clicks: list[tuple[int, int]] = []
    pages = iter(
        [
            UiWhere("facility", 1, 0, 0, "mem"),
            UiWhere("craft_hub", 1, 0, 0, "mem"),
            UiWhere("facility:wood", 1, 0, 0, "mem"),
        ]
    )
    r = ui_nav.ensure_node(
        "facility",
        facility="wood",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: next(pages, UiWhere("facility:wood", 1, 0, 0, "mem")),
        already_ok=lambda _f: False,
    )
    assert clicks == []
    assert r.status == "need_verify"
    assert "不退页" in r.message


def test_ensure_node_leaves_only_when_wrong_facility_confirmed(monkeypatch):
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (7, 8))
    clicks: list[tuple[int, int]] = []
    r = ui_nav.ensure_node(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("facility", 1, 0, 0, "mem"),
        already_ok=lambda _f: False,
        leave_wrong_page=lambda _f: True,
        facility="metal",
    )
    assert clicks[0] == (7, 8)
    assert r.status != "at_node" or r.ok is True


def test_ensure_node_drags_item_list_before_missing_name(monkeypatch):
    import numpy as np
    from mm_assist.perception import item_sense as mod

    frame = np.zeros((1440, 1920, 3), dtype=np.uint8)
    loc_n = {"i": 0}
    drags: list[tuple[int, int, int, int]] = []
    clicks: list[tuple[int, int]] = []
    page = {"n": "gather_skill:herb"}

    def fake_locate(frame, *, skill_id, name, root=None):
        loc_n["i"] += 1
        if loc_n["i"] == 1:
            return {
                "ok": False,
                "error": "not_found",
                "scroll_plan": {"needs_scroll": 2, "direction": "down"},
            }
        return {"ok": True, "click_client": [900, 400]}

    monkeypatch.setattr(mod, "locate_skill_item", fake_locate)

    def click(x, y):
        clicks.append((x, y))
        page["n"] = "item_info:淨淨蘑菇"
        return True

    r = ui_nav.ensure_node(
        "item_info",
        grab=lambda: frame,
        click=click,
        drag_fn=lambda x0, y0, x1, y1: drags.append((x0, y0, x1, y1)),
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        skill_id="herb",
        item_name="淨淨蘑菇",
    )
    assert drags
    assert len(drags) <= 2
    assert clicks == [(900, 400)]
    assert r.ok is True
    assert "找不到物名" not in str(r.message or "")


def test_ensure_node_item_list_scroll_caps_at_two(monkeypatch):
    import numpy as np
    from mm_assist.perception import item_sense as mod

    frame = np.zeros((1440, 1920, 3), dtype=np.uint8)
    drags: list[tuple[int, int, int, int]] = []
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        mod,
        "locate_skill_item",
        lambda *a, **k: {
            "ok": False,
            "error": "not_found",
            "scroll_plan": {"needs_scroll": 8, "direction": "down"},
        },
    )
    r = ui_nav.ensure_node(
        "item_info",
        grab=lambda: frame,
        click=lambda x, y: clicks.append((x, y)) or True,
        drag_fn=lambda x0, y0, x1, y1: drags.append((x0, y0, x1, y1)),
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("gather_skill:herb", 1, 0, 0, "mem"),
        skill_id="herb",
        item_name="淨淨蘑菇",
    )
    assert drags == drags[:2] and len(drags) == 2
    assert clicks == []
    assert r.ok is False
    assert r.status == "物名"
    assert "找不到物名" in str(r.message or "")


def test_ensure_node_does_not_click_clipped_row(monkeypatch):
    import numpy as np
    from mm_assist.perception import item_sense as mod

    frame = np.zeros((1440, 1920, 3), dtype=np.uint8)
    drags: list[tuple[int, int, int, int]] = []
    clicks: list[tuple[int, int]] = []
    n = {"i": 0}

    def fake_locate(*a, **k):
        n["i"] += 1
        return {
            "ok": False,
            "error": "not_found",
            "reason": "clipped_row",
            "scroll_plan": {"needs_scroll": 1, "direction": "down", "hint": "clipped_row"},
        }

    monkeypatch.setattr(mod, "locate_skill_item", fake_locate)
    r = ui_nav.ensure_node(
        "item_info",
        grab=lambda: frame,
        click=lambda x, y: clicks.append((x, y)) or True,
        drag_fn=lambda x0, y0, x1, y1: drags.append((x0, y0, x1, y1)),
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("gather_skill:herb", 1, 0, 0, "mem"),
        skill_id="herb",
        item_name="淨淨蘑菇",
    )
    assert clicks == []
    assert len(drags) == 2
    assert r.ok is False
    assert r.status == "clipped_row"


def test_ensure_sap_item_info_from_inventory_not_skill_list(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere

    page = {"n": "inventory"}
    ids: list[str] = []
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr("mm_assist.life_plan.node_inspect.locate_spot", lambda *a, **k: (11, 12))

    def locate(edge, *a, **k):
        eid = str(edge.get("id") or "")
        ids.append(eid)
        if eid == "inventory_item_item_info" and str(k.get("item_name") or "") == "樹液":
            return (11, 12)
        if eid == "gather_skill_item_item_info":
            return (99, 99)
        return (1, 1)

    def do_click(x, y):
        clicks.append((x, y))
        if page["n"] == "inventory":
            page["n"] = "item_info:樹液"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "item_info",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        skill_id="wood",
        item_name="樹液",
    )
    assert r.ok is True
    assert clicks == [(11, 12)]
    assert "gather_skill_item_item_info" not in ids
    assert ids[0] == "inventory_item_item_info"
    assert page["n"] == "item_info:樹液"


def test_sap_hops_from_main_hud_goes_facility_not_bag():
    hops = ui_nav._sap_hops("main_hud", "item_info", root=None)
    assert hops
    ids = [str(h.get("id") or "") for h in hops]
    assert "system_menu_bag_inventory" not in ids
    assert "inventory_item_item_info" not in ids
    assert ids[0] == "main_hud_menu_system_menu"
    assert "craft_hub_card_facility" in ids or any("craft" in i for i in ids)


def test_sap_hops_inventory_without_card_skips_guess_click(monkeypatch):
    monkeypatch.setattr("mm_assist.life_plan.node_inspect.locate_spot", lambda *a, **k: None)
    hops = ui_nav._sap_hops("inventory", "item_info", root=None, frame=object())
    assert hops
    assert hops[0].get("id") != "inventory_item_item_info"


def test_sap_hops_wood_facility_opens_high_wood_then_recipe_clicks_sap():
    hops = ui_nav._sap_hops("facility:wood", "item_info", root=None)
    assert hops[0]["id"] == "facility_open_recipe"
    assert (hops[0].get("trigger") or {}).get("label") == "高級木材"
    hops2 = ui_nav._sap_hops("facility:wood:高級木材", "item_info", root=None)
    assert hops2[0]["id"] == "facility_item_item_info"


def test_ensure_facility_wood_from_confirmed_wrong_generic_x_closes_to_hud(monkeypatch):
    page = {"n": "facility"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        ids.append(str(edge.get("id") or ""))
        return (1, 1)

    def do_click(x, y):
        eid = ids[-1] if ids else ""
        if page["n"] == "facility":
            page["n"] = "main_hud"
        elif page["n"] == "main_hud":
            page["n"] = "system_menu" if eid == "main_hud_menu_system_menu" else "craft_hub"
        elif page["n"] == "system_menu":
            page["n"] = "craft_hub"
        elif page["n"] == "craft_hub":
            page["n"] = "facility:wood"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "facility",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        facility="wood",
        leave_wrong_page=lambda _f: page["n"] == "facility",
    )
    assert r.ok is True
    assert ids[0] == "facility_back_craft_hub"
    assert "system_menu_craft_craft_hub" in ids or "craft_hub_card_facility" in ids
    assert page["n"] == "facility:wood"


def test_ensure_facility_wood_from_metal_goes_hub(monkeypatch):
    page = {"n": "facility:metal"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        ids.append(str(edge.get("id") or ""))
        return (1, 1)

    def do_click(x, y):
        if page["n"] == "facility:metal":
            page["n"] = "craft_hub"
        elif page["n"] == "craft_hub":
            page["n"] = "facility:wood"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "facility",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        facility="wood",
    )
    assert r.ok is True
    assert ids[0] == "facility_back_craft_hub"
    assert page["n"] == "facility:wood"


def test_ensure_inventory_from_skill_list_closes_not_item(monkeypatch):
    page = {"n": "gather_skill:fish"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        eid = str(edge.get("id") or "")
        ids.append(eid)
        if eid in {"gather_skill_item_item_info", "gather_skill_item_gather_detail"}:
            return (99, 99)
        return (1, 1)

    def do_click(x, y):
        eid = ids[-1] if ids else ""
        if page["n"] == "gather_skill:fish":
            page["n"] = "life_grid"
        elif page["n"] == "life_grid":
            page["n"] = "main_hud"
        elif page["n"] == "main_hud":
            page["n"] = "system_menu" if eid == "main_hud_menu_system_menu" else "inventory"
        elif page["n"] == "system_menu":
            page["n"] = "inventory"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "inventory",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
    )
    assert r.ok is True
    assert "gather_skill_item_item_info" not in ids
    assert page["n"] == "inventory"


def test_ensure_inventory_from_sheep_detail_accepts_skill_list(monkeypatch):
    page = {"n": "gather_detail:羊"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        ids.append(str(edge.get("id") or ""))
        return (1, 1)

    def do_click(x, y):
        eid = ids[-1] if ids else ""
        if page["n"].startswith("gather_detail"):
            page["n"] = "gather_skill:shear"
        elif page["n"].startswith("gather_skill"):
            page["n"] = "life_grid"
        elif page["n"] == "life_grid":
            page["n"] = "main_hud"
        elif page["n"] == "main_hud":
            page["n"] = "system_menu" if eid == "main_hud_menu_system_menu" else "inventory"
        elif page["n"] == "system_menu":
            page["n"] = "inventory"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "inventory",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
    )
    assert r.ok is True
    assert r.status != "wrong_page"
    assert page["n"] == "inventory"


def test_ensure_inventory_from_facility_backs_hub_not_recipe(monkeypatch):
    page = {"n": "facility:wood"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        ids.append(str(edge.get("id") or ""))
        return (1, 1)

    def do_click(x, y):
        eid = ids[-1] if ids else ""
        if page["n"] == "facility:wood":
            page["n"] = "craft_hub"
        elif page["n"] == "craft_hub":
            page["n"] = "main_hud"
        elif page["n"] == "main_hud":
            page["n"] = "system_menu" if eid == "main_hud_menu_system_menu" else "inventory"
        elif page["n"] == "system_menu":
            page["n"] = "inventory"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "inventory",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
    )
    assert r.ok is True
    assert ids[0] == "facility_back_craft_hub"
    assert "facility_open_recipe" not in ids
    assert page["n"] == "inventory"


def test_ensure_medicine_recipe_from_wood_backs_hub(monkeypatch):
    page = {"n": "facility:wood:高級木材"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        ids.append(str(edge.get("id") or ""))
        return (1, 1)

    def do_click(x, y):
        if page["n"].startswith("facility:wood"):
            page["n"] = "craft_hub"
        elif page["n"] == "craft_hub":
            page["n"] = "facility:medicine"
        elif page["n"] == "facility:medicine":
            page["n"] = "facility:medicine:淨淨蘑菇孢子"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "facility_recipe",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        facility="medicine",
        item_name="淨淨蘑菇孢子",
    )
    assert r.ok is True
    assert ids[0] == "facility_back_craft_hub"
    assert page["n"] == "facility:medicine:淨淨蘑菇孢子"


def test_ensure_craft_hub_from_recipe_backs_facility_first(monkeypatch):
    ui_nav.reset_edge_memo_for_tests()
    page = {"n": "facility:medicine:淨淨蘑菇孢子"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        ids.append(str(edge.get("id") or ""))
        return (1, 1)

    def do_click(x, y):
        if page["n"] == "facility:medicine:淨淨蘑菇孢子":
            page["n"] = "facility:medicine"
        elif page["n"] == "facility:medicine":
            page["n"] = "craft_hub"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    monkeypatch.setattr(ui_nav, "craft_hub_visible", lambda *a, **k: True)
    r = ui_nav.ensure_node(
        "craft_hub",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
    )
    assert r.ok is True
    assert ids == ["facility_back_craft_hub", "facility_back_craft_hub"]
    assert page["n"] == "craft_hub"


def test_ensure_sap_from_herb_list_closes_not_click_item(monkeypatch):
    page = {"n": "gather_skill:herb"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        eid = str(edge.get("id") or "")
        ids.append(eid)
        if eid in {"gather_skill_item_item_info", "gather_skill_item_gather_detail"}:
            return (99, 99)
        return (1, 1)

    def do_click(x, y):
        if page["n"] == "gather_skill:herb":
            page["n"] = "life_grid"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "item_info",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 1, 0, 0, "mem"),
        skill_id="wood",
        item_name="樹液",
    )
    assert ids
    assert ids[0] != "gather_skill_item_item_info"
    assert "gather_skill_item_item_info" not in ids
    assert page["n"] != "gather_skill:herb"


def test_locate_obtain_method_edge_none_without_spot(monkeypatch):
    monkeypatch.setattr("mm_assist.life_plan.node_inspect.locate_spot", lambda *a, **k: None)
    assert (
        ui_nav.locate_edge(
            {"id": "gather_detail_open_obtain_method", "from": "gather_detail", "to": "obtain_method"},
            object(),
        )
        is None
    )
    assert (
        ui_nav.locate_edge(
            {"id": "item_info_open_obtain_method", "from": "item_info", "to": "obtain_method"},
            object(),
        )
        is None
    )


def test_locate_obtain_method_edge_on_gather_detail(monkeypatch):
    seen: list[tuple[str, str]] = []

    def loc(node, spot, frame, **k):
        seen.append((str(node), str(spot)))
        if node == "gather_detail" and spot == "获得方法":
            return (40, 50)
        return None

    monkeypatch.setattr("mm_assist.life_plan.node_inspect.locate_spot", loc)
    xy = ui_nav.locate_edge(
        {"id": "gather_detail_open_obtain_method", "from": "gather_detail", "to": "obtain_method"},
        object(),
    )
    assert xy == (40, 50)
    assert ("item_info", "获得方法") in seen
    assert ("gather_detail", "获得方法") in seen


def test_path_edges_overlay_closes_to_hud():
    hops = ui_nav.path_edges("overlay", "main_hud")
    assert hops and hops[0]["id"] == "overlay_x_main_hud"


def test_ensure_overlay_closes_to_main_hud(monkeypatch):
    page = {"n": "overlay"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        ids.append(str(edge.get("id") or ""))
        return (1800, 40)

    def do_click(x, y):
        if page["n"] == "overlay":
            page["n"] = "main_hud"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "main_hud",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(page["n"], 0, 0, 0, "mem"),
    )
    assert r.ok is True
    assert ids == ["overlay_x_main_hud"]
    assert page["n"] == "main_hud"


def test_ensure_dungeon_overlay_closes_before_gather_skill(monkeypatch):
    page = {"n": "stack:DungeonListUIInstance"}
    ids: list[str] = []

    def locate(edge, *a, **k):
        ids.append(str(edge.get("id") or ""))
        if str(edge.get("id")) == "overlay_x_main_hud":
            return (1800, 40)
        return None

    def do_click(x, y):
        if "Dungeon" in page["n"] or page["n"] == "overlay":
            page["n"] = "main_hud"
        return True

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    r = ui_nav.ensure_node(
        "gather_skill",
        grab=lambda: object(),
        click=do_click,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere(
            page["n"],
            0,
            0,
            0,
            "mem",
            stack_names=("DungeonListUIInstance", "DungeonEntryUIInstance")
            if "Dungeon" in page["n"]
            else (),
        ),
    )
    assert ids[:1] == ["overlay_x_main_hud"]
    assert page["n"] == "main_hud"
    assert r.ok is False


@pytest.mark.parametrize('visible', [False, True])
def test_craft_hub_destination_is_recognized_once_per_observation(monkeypatch, visible):
    from mm_assist.life_execute import ui_nav
    from mm_assist.life_execute.ui_logic import UiWhere
    frame, queries, clicks = object(), [], []
    def recognize(image, **kw):
        assert image is frame
        queries.append(True)
        return visible
    monkeypatch.setattr(ui_nav, 'craft_hub_visible', recognize)
    result = ui_nav.ensure_node('craft_hub', grab=lambda: frame,
        click=lambda *xy: clicks.append(xy) or True, sleep_fn=lambda _: None,
        resolve=lambda _: UiWhere('craft_hub', 1, 0, 0, 'fixture'))
    assert not clicks and result.ok is visible
    assert result.status == ('at_node' if visible else 'wrong_tab')
    assert queries == [True], 'rejecting this same frame must not rerun craft-hub recognition'


def test_ensure_already_on_target_zero_clicks():
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_execute.ui_nav import ensure_node

    clicks: list[tuple[int, int]] = []
    r = ensure_node(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("facility", 1, 0, 0, "mem+icon"),
        already_ok=lambda _f: True,
    )
    assert r.ok is True
    assert clicks == []


def test_ensure_stops_on_cancel(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_execute.ui_nav import ensure_node

    monkeypatch.setattr("mm_assist.tool_repair.op_cancelled", lambda: True)
    clicks: list[tuple[int, int]] = []
    r = ensure_node(
        "system_menu",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("main_hud", 0, 0, 0, "mem"),
        already_ok=lambda _f: False,
    )
    assert r.ok is False
    assert r.status == "cancel"
    assert clicks == []


def test_navigation_waits_after_one_click_without_reclicking(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_execute import ui_nav

    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (11, 22))
    clicks: list[tuple[int, int]] = []
    sleeps: list[float] = []
    r = ui_nav.ensure_node(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: sleeps.append(s),
        resolve=lambda _f=None: UiWhere("main_hud", 0, 0, 0, "mem"),
    )
    assert r.ok is False
    assert r.status == "nav_verify_timeout"
    assert clicks == [(11, 22)]
    assert sleeps[0] == pytest.approx(2.0)
    assert all(s <= 0.151 for s in sleeps[1:])
    assert sum(sleeps) == pytest.approx(6.0, abs=1e-6)


def test_ensure_menu1_stale_hud_does_not_click_menu(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_execute import ui_nav

    labels: list[str] = []

    def locate(edge, *a, **k):
        labels.append(str((edge.get("trigger") or {}).get("label") or edge.get("id") or ""))
        return (11, 22)

    monkeypatch.setattr(ui_nav, "locate_edge", locate)
    clicks: list[tuple[int, int]] = []
    ui_nav.ensure_node(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: None,
        resolve=lambda _f=None: UiWhere("main_hud", 1, 0, 0, "inspect"),
        already_ok=lambda _f: False,
    )
    assert "打开菜单" not in labels
    assert clicks
