from __future__ import annotations

from types import SimpleNamespace

import pytest

from mm_assist.life_plan.node_inspect import (
    NODE_ZH,
    SPOTS,
    compute_inspect,
    format_node_zh,
    inspect_needs_grab,
    inspect_page_zh,
    locate_spot,
    mem_inspect_node,
    spots_for_node,
)


@pytest.fixture(autouse=True)
def _reset_inspect_shot_memo():
    from mm_assist.life_plan.node_inspect import _reset_facility_fid_cache, _reset_inspect_cost

    _reset_inspect_cost()
    _reset_facility_fid_cache()
    yield
    _reset_inspect_cost()
    _reset_facility_fid_cache()


def test_format_node_zh_known_and_skip():
    assert format_node_zh("main_hud") == "主界面"
    assert format_node_zh("main_hud", stp=0) == "主界面-待机"
    assert format_node_zh("main_hud", stp=1) == "主界面-动作中"
    assert format_node_zh("system_menu", stp=1) == "菜单"
    assert inspect_page_zh("main_hud", stack_ready=False, stp=1) == "—"
    assert inspect_page_zh("main_hud", stack_ready=True, stp=0) == "主界面-待机"
    assert inspect_page_zh("main_hud", stack_ready=True, stp=1) == "主界面-动作中"
    assert inspect_page_zh("system_menu", stack_ready=True, stp=1) == "菜单"
    assert format_node_zh("craft_hub") == "制作-加工"
    assert format_node_zh("craft_quick") == "制作-快速制作"
    assert format_node_zh("craft_make") == "制作-制作"
    assert format_node_zh("craft_alchemy") == "制作-炼金术"
    assert format_node_zh("facility") == "制作-加工-设施"
    assert format_node_zh("facility:metal") == "制作-加工-金属加工"
    assert format_node_zh("facility:wood") == "制作-加工-木材加工"
    assert format_node_zh("facility:leather") == "制作-加工-皮革加工"
    assert format_node_zh("facility:cloth") == "制作-加工-布料加工"
    assert format_node_zh("facility:medicine") == "制作-加工-药品加工"
    assert format_node_zh("facility:food") == "制作-加工-食材加工"
    assert format_node_zh("facility:metal:鐵錠") == "制作-加工-金属加工-铁锭"
    assert format_node_zh("facility:wood:高級木材") == "制作-加工-木材加工-高级木材"
    assert format_node_zh("facility:medicine:淨淨蘑菇汁液") == "制作-加工-药品加工-净净蘑菇汁液"
    assert format_node_zh("currency_wallet") == "持有的货币"
    assert format_node_zh("dialog") == "对话"
    assert format_node_zh("dialog_skip") == "对话-跳过对话"
    assert format_node_zh("item_info") == "物品详情"
    assert format_node_zh("obtain_method") == "获得方法"
    assert format_node_zh("gather_detail") == "角色-生活技能-技能详情"
    assert format_node_zh("gather_detail:蜘蛛網") == "角色-生活技能-技能详情-蜘蛛網"
    assert format_node_zh("craft_shell") == "制作"
    assert format_node_zh("fine:craft_hub") == "—"
    assert inspect_page_zh("craft_shell", stack_ready=True) == "—"
    assert inspect_page_zh("fine:craft_hub", stack_ready=True) == "—"
    assert inspect_page_zh("craft_hub", stack_ready=True) == "制作-加工"
    assert format_node_zh("map_continent") == "地图-欧拉大陆"
    assert format_node_zh("map_kuhan") == "地图-库汉"
    assert format_node_zh("map_dunbarton") == "地图-杜巴顿"
    assert format_node_zh("stack:CashShopUIState") == "商城"
    assert format_node_zh("unknown") == "未知界面"
    assert format_node_zh("overlay") == "离开主界面"
    assert format_node_zh("stack:NoSuchUIState") == "未知界面"
    assert format_node_zh("not_a_page") == "未知界面"
    assert format_node_zh("") == "—"
    assert spots_for_node("unknown") == ()
    assert spots_for_node("overlay") == ()
    assert spots_for_node("stack:NoSuchUIState") == ()


def test_main_hud_stp_zh_from_stack():
    from mm_assist.life_plan import node_inspect as ni

    idle = ni.compute_inspect(
        None,
        view=SimpleNamespace(
            ui_stack_names=["TutorialTooltipUIState"],
            ui_stack_ready=True,
            stp=0,
        ),
    )
    assert idle["node"] == "main_hud"
    assert idle["stp"] == 0
    assert format_node_zh(idle["node"], stp=idle["stp"]) == "主界面-待机"
    busy = ni.compute_inspect(
        None,
        view=SimpleNamespace(
            ui_stack_names=["TutorialTooltipUIState"],
            ui_stack_ready=True,
            stp=1,
        ),
    )
    assert busy["node"] == "main_hud"
    assert busy["stp"] == 1
    assert format_node_zh(busy["node"], stp=busy["stp"]) == "主界面-动作中"
    hud_only = ni.compute_inspect(
        None,
        view=SimpleNamespace(
            ui_stack_names=["TutorialTooltipUIState", "HudUIStepState"],
            ui_stack_ready=True,
            stp=0,
        ),
    )
    assert hud_only["node"] == "overlay"
    assert hud_only["stp"] is None
    assert format_node_zh(hud_only["node"]) == "离开主界面"
    left = ni.compute_inspect(
        None,
        view=SimpleNamespace(
            ui_stack_names=["SideNavigationUIState", "HudUIStepState"],
            ui_stack_ready=True,
            stp=1,
        ),
    )
    assert left["node"] == "system_menu"
    assert left["stp"] is None
    assert format_node_zh(left["node"], stp=1) == "菜单"
    pending = ni.compute_inspect(
        None,
        view=SimpleNamespace(
            ui_stack_names=[],
            ui_stack_ready=False,
            stp=1,
        ),
    )
    assert pending["node"] == ""
    assert pending["stp"] is None


def test_hold_inspect_freezes_until_stack_confirms():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    hud = {
        "node": "main_hud",
        "stack_ready": True,
        "stp": 0,
        "stack_names": ["TutorialTooltipUIState"],
        "source": "stack",
    }
    pending = {
        "node": "",
        "stack_ready": False,
        "stp": None,
        "stack_names": [],
        "source": "pending",
    }
    frozen, streak = hold_inspect_payload(hud, pending)
    assert frozen["node"] == "main_hud"
    assert frozen["stp"] == 0
    assert streak == 0
    menu = {
        "node": "system_menu",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["SideNavigationUIState", "HudUIStepState"],
        "source": "stack",
    }
    shown, _st = hold_inspect_payload(frozen, menu)
    assert shown["node"] == "system_menu"
    fake_hud = {
        "node": "main_hud",
        "stack_ready": True,
        "stp": 0,
        "stack_names": ["TutorialTooltipUIState"],
        "source": "stack",
    }
    still_menu, st1 = hold_inspect_payload(shown, fake_hud, hud_streak=0)
    assert still_menu["node"] == "system_menu"
    assert st1 == 1
    now_hud, st2 = hold_inspect_payload(still_menu, fake_hud, hud_streak=st1)
    assert now_hud["node"] == "main_hud"
    assert st2 == 0
    overlay = {
        "node": "",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["TutorialTooltipUIState", "HudUIStepState"],
        "source": "pending",
    }
    left, _st = hold_inspect_payload(hud, overlay)
    assert left["node"] == "overlay"
    skew = dict(overlay)
    skew["skew"] = True
    frozen2, _ = hold_inspect_payload(hud, skew)
    assert frozen2["node"] == "main_hud"


def test_hold_empty_stack_main_hud_needs_menu_icon():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    menu = {
        "node": "system_menu",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["SideNavigationUIState", "HudUIStepState"],
        "source": "stack",
    }
    no_icon = {
        "node": "main_hud",
        "stack_ready": True,
        "stp": None,
        "stack_names": [],
        "source": "pending",
        "hud_menu_icon": False,
    }
    frozen, st = hold_inspect_payload(menu, no_icon, hud_streak=0)
    assert frozen["node"] == "system_menu"
    assert st == 0
    no_icon_stp = dict(no_icon)
    no_icon_stp["stp"] = 0
    still_empty, st_a = hold_inspect_payload(menu, no_icon_stp, hud_streak=0)
    assert still_empty["node"] == "system_menu"
    assert st_a == 1
    now_empty, st_b = hold_inspect_payload(still_empty, no_icon_stp, hud_streak=st_a)
    assert now_empty["node"] == "main_hud"
    assert st_b == 0
    with_icon = {
        "node": "main_hud",
        "stack_ready": True,
        "stp": 0,
        "stack_names": [],
        "source": "stack+tpl",
        "hud_menu_icon": True,
    }
    still, st1 = hold_inspect_payload(menu, with_icon, hud_streak=0)
    assert still["node"] == "system_menu"
    assert st1 == 1
    now_hud, st2 = hold_inspect_payload(still, with_icon, hud_streak=st1)
    assert now_hud["node"] == "main_hud"
    assert st2 == 0


def test_stack_sn_from_mem_empty_ok():
    from mm_assist.life_plan.node_inspect import _stack_sn_from_mem

    hud = SimpleNamespace(
        ui_stack_names=[],
        ui_stack_ready=True,
        ui_stack_empty_ok=True,
    )
    assert _stack_sn_from_mem(hud) == "main_hud"
    pending = SimpleNamespace(
        ui_stack_names=[],
        ui_stack_ready=True,
        ui_stack_empty_ok=False,
    )
    assert _stack_sn_from_mem(pending) == ""


def test_compute_inspect_empty_ok_no_frame():
    view = SimpleNamespace(
        ui_stack_names=[],
        ui_stack_ready=True,
        ui_stack_empty_ok=True,
        stp=1,
    )
    payload = compute_inspect(None, view=view)
    assert payload["node"] == "main_hud"
    assert payload["empty_ok"] is True
    assert payload["stp"] == 1


def test_compute_inspect_empty_stack_menu_icon(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    view = SimpleNamespace(
        ui_stack_names=[],
        ui_stack_ready=True,
        ui_stack_empty_ok=True,
        stp=0,
    )
    monkeypatch.setattr(ni, "locate_spot", lambda *_a, **_k: None)
    miss = ni.compute_inspect(object(), view=view)
    assert miss["node"] == "main_hud"
    assert miss["empty_ok"] is True
    assert miss["hud_menu_icon"] is False
    assert miss["stp"] == 0
    menu = {
        "node": "system_menu",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["SideNavigationUIInstance"],
        "source": "stack",
    }
    held, st_m = ni.hold_inspect_payload(menu, miss, hud_streak=0)
    assert held["node"] == "system_menu"
    assert st_m == 1
    released, st_r = ni.hold_inspect_payload(held, miss, hud_streak=st_m)
    assert released["node"] == "main_hud"
    monkeypatch.setattr(ni, "locate_spot", lambda *_a, **_k: (10, 20))
    hit = ni.compute_inspect(object(), view=view)
    assert hit["node"] == "main_hud"
    assert hit["hud_menu_icon"] is True
    assert hit["stp"] == 0


def test_hold_inspect_keeps_life_tab_on_coarse_profile():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    life = {
        "node": "life_grid",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["HudUIStepState", "NewProfileUIState"],
        "source": "stack+tpl",
    }
    coarse = {
        "node": "profile",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["HudUIStepState", "NewProfileUIState", "TutorialTooltipUIState"],
        "source": "stack",
    }
    kept, _st = hold_inspect_payload(life, coarse)
    assert kept["node"] == "life_grid"
    looked = dict(coarse)
    looked["profile_looked"] = True
    left, _ = hold_inspect_payload(life, looked)
    assert left["node"] == "profile"
    combat = dict(coarse)
    combat["node"] = "profile:combat"
    combat["profile_looked"] = True
    switched, _ = hold_inspect_payload(life, combat)
    assert switched["node"] == "profile:combat"


def test_hold_inspect_keeps_gather_skill_on_coarse_profile():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    skill = {
        "node": "gather_skill:daily",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["HudUIStepState", "NewProfileUIState"],
        "source": "stack+tpl",
    }
    coarse = {
        "node": "profile",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["HudUIStepState", "NewProfileUIState"],
        "source": "stack",
        "profile_looked": True,
    }
    kept, _ = hold_inspect_payload(skill, coarse)
    assert kept["node"] == "gather_skill:daily"
    info = dict(coarse)
    info["node"] = "profile:info"
    left, _ = hold_inspect_payload(skill, info)
    assert left["node"] == "profile:info"
    grid = {
        "node": "life_grid",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["HudUIStepState", "NewProfileUIState"],
        "source": "stack+tpl",
        "life_looked": True,
        "profile_looked": True,
    }
    backed, _ = hold_inspect_payload(skill, grid)
    assert backed["node"] == "life_grid"


def test_hold_inspect_keeps_gather_detail_on_coarse_life_grid():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    detail = {
        "node": "gather_detail",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["LivingSkillUIState", "HudUIStepState"],
        "source": "stack+tpl",
    }
    skill = {
        "node": "gather_skill:daily",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["LivingSkillUIState", "HudUIStepState"],
        "source": "stack+tpl",
    }
    coarse = {
        "node": "life_grid",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["LivingSkillUIState", "HudUIStepState"],
        "source": "stack",
    }
    kept_d, _st = hold_inspect_payload(detail, coarse)
    assert kept_d["node"] == "gather_detail"
    kept_s, _ = hold_inspect_payload(skill, coarse)
    assert kept_s["node"] == "gather_skill:daily"
    looked = dict(coarse)
    looked["life_looked"] = True
    left, _ = hold_inspect_payload(detail, looked)
    assert left["node"] == "life_grid"


def test_hold_inspect_keeps_craft_fine_on_shell():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    make = {
        "node": "craft_make",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["CraftingHubUIState", "HudUIStepState"],
        "source": "stack+tpl",
        "craft_how": "tpl",
    }
    shell = {
        "node": "craft_shell",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["CraftingHubUIState", "HudUIStepState"],
        "source": "stack",
    }
    kept, _ = hold_inspect_payload(make, shell)
    assert kept["node"] == "craft_make"
    fine_sn = dict(shell)
    fine_sn["node"] = "fine:craft_hub"
    kept2, _ = hold_inspect_payload(make, fine_sn)
    assert kept2["node"] == "craft_make"


def test_hold_inspect_craft_weak_needs_three():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    process = {
        "node": "craft_hub",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["CraftingHubUIState", "HudUIStepState"],
        "source": "stack+tpl",
        "craft_how": "tpl",
    }
    weak = {
        "node": "craft_alchemy",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["CraftingHubUIState", "HudUIStepState"],
        "source": "stack+mem",
        "craft_how": "mem",
    }
    a, _ = hold_inspect_payload(process, weak)
    assert a["node"] == "craft_hub"
    assert a["craft_weak"] == ("craft_alchemy", 1)
    b, _ = hold_inspect_payload(a, weak)
    assert b["node"] == "craft_hub"
    assert b["craft_weak"] == ("craft_alchemy", 2)
    c, _ = hold_inspect_payload(b, weak)
    assert c["node"] == "craft_alchemy"
    assert "craft_weak" not in c


def test_hold_inspect_craft_tpl_switches_now():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    process = {
        "node": "craft_hub",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["CraftingHubUIState", "HudUIStepState"],
        "source": "stack+tpl",
        "craft_how": "tpl",
    }
    alchemy = {
        "node": "craft_alchemy",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["CraftingHubUIState", "HudUIStepState"],
        "source": "stack+tpl",
        "craft_how": "tpl",
    }
    shown, _ = hold_inspect_payload(process, alchemy)
    assert shown["node"] == "craft_alchemy"


def test_hold_inspect_keeps_facility_fid_off_craft_hub():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    metal = {
        "node": "facility:metal",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["CraftingHubUIState", "HudUIStepState"],
        "source": "stack+tpl",
    }
    fac_stack = [
        "AlteringFacilityPopupUIInstance",
        "CraftingHubPopupUIInstance",
        "HudUIStepState",
    ]
    weak_hub = {
        "node": "craft_hub",
        "stack_ready": True,
        "stp": None,
        "stack_names": fac_stack,
        "source": "stack+tpl",
        "hub_cards": False,
    }
    kept, _ = hold_inspect_payload(metal, weak_hub)
    assert kept["node"] == "facility:metal"
    shell = {
        "node": "craft_shell",
        "stack_ready": True,
        "stp": None,
        "stack_names": fac_stack,
        "source": "stack",
        "hub_cards": False,
    }
    kept_shell, _ = hold_inspect_payload(metal, shell)
    assert kept_shell["node"] == "facility:metal"
    fine_hub = dict(shell)
    fine_hub["node"] = "fine:craft_hub"
    kept_fine, _ = hold_inspect_payload(metal, fine_hub)
    assert kept_fine["node"] == "facility:metal"
    cards = dict(weak_hub)
    cards["hub_cards"] = True
    shown, _ = hold_inspect_payload(metal, cards)
    assert shown["node"] == "craft_hub"
    alchemy = {
        "node": "craft_alchemy",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["CraftingHubUIState", "HudUIStepState"],
        "source": "stack+tpl",
        "hub_cards": False,
    }
    shown_alchemy, _ = hold_inspect_payload(metal, alchemy)
    assert shown_alchemy["node"] == "craft_alchemy"


def test_hold_inspect_releases_facility_when_stack_back_to_shell():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    wood = {
        "node": "facility:wood",
        "stack_ready": True,
        "stp": None,
        "stack_names": [
            "AlteringFacilityPopupUIInstance",
            "CraftingHubPopupUIInstance",
            "HudUIStepState",
        ],
        "source": "stack+tpl",
    }
    hub = {
        "node": "fine:craft_hub",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["CraftingHubPopupUIInstance", "HudUIStepState"],
        "source": "stack",
        "hub_cards": False,
    }
    shown, _ = hold_inspect_payload(wood, hub)
    assert shown["node"] == "fine:craft_hub"
    coarse = dict(hub)
    coarse["node"] = "craft_hub"
    shown_coarse, _ = hold_inspect_payload(wood, coarse)
    assert shown_coarse["node"] == "craft_hub"
    recipe = dict(wood)
    recipe["node"] = "facility_recipe"
    shown_recipe, _ = hold_inspect_payload(recipe, hub)
    assert shown_recipe["node"] == "fine:craft_hub"
    named = dict(wood)
    named["node"] = "facility:wood:高級木材"
    shown_named, _ = hold_inspect_payload(named, hub)
    assert shown_named["node"] == "fine:craft_hub"


def test_hold_inspect_keeps_facility_when_stack_not_ready():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    wood = {
        "node": "facility:wood",
        "stack_ready": True,
        "stp": None,
        "stack_names": [
            "AlteringFacilityPopupUIInstance",
            "CraftingHubPopupUIInstance",
            "HudUIStepState",
        ],
        "source": "stack+tpl",
    }
    pending = {
        "node": "fine:craft_hub",
        "stack_ready": False,
        "stp": None,
        "stack_names": ["CraftingHubPopupUIInstance", "HudUIStepState"],
        "source": "stack",
        "hub_cards": False,
    }
    kept, _ = hold_inspect_payload(wood, pending)
    assert kept["node"] == "facility:wood"


def test_inspect_map_from_frame_not_mem_geo(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    view = SimpleNamespace(
        ui_stack_names=["MapUIState", "HudUIStepState"],
        ui_stack_ready=True,
        map_type=2,
        geo="庫漢",
    )
    out = compute_inspect(None, view=view)
    # MapType.Town also describes 萊爾特丘陵; the enum alone is insufficient.
    assert out["node"] == "map"
    assert out["node"] != "map_kuhan"
    assert ni.inspect_needs_grab("map", view=view) is True
    assert ni.inspect_needs_grab("map", view=SimpleNamespace(map_type=1)) is True
    view_no_type = SimpleNamespace(
        ui_stack_names=["MapUIState", "HudUIStepState"],
        ui_stack_ready=True,
        geo="庫漢",
    )
    assert compute_inspect(None, view=view_no_type)["node"] == "map"
    assert ni.inspect_needs_grab("map") is True
    assert ni.inspect_needs_grab("map", view=view_no_type) is True
    monkeypatch.setattr(ni, "_tmpl", lambda *_a, **_k: False)
    monkeypatch.setattr(ni, "_ocr_blob", lambda *_a, **_k: "歐拉大陸")
    assert ni.inspect_map_from_frame(object()) == "map_continent"
    monkeypatch.setattr(ni, "_ocr_blob", lambda *_a, **_k: "歐拉大陸／杜巴頓")
    assert ni.inspect_map_from_frame(object()) == "map_dunbarton"
    monkeypatch.setattr(ni, "_ocr_blob", lambda *_a, **_k: "庫漢")
    assert ni.inspect_map_from_frame(object()) == "map"
    monkeypatch.setattr(ni, "_tmpl", lambda *_a, **_k: True)
    assert ni.inspect_map_from_frame(object()) == "map_dunbarton"
    calls = {"n": 0}

    def boom(_f):
        calls["n"] += 1
        return "map_continent", "ocr"

    monkeypatch.setattr(ni, "_inspect_map_how", boom)
    with_frame = compute_inspect(object(), view=view)
    assert with_frame["node"] == "map_continent"
    assert calls["n"] == 1
    monkeypatch.setattr(
        ni, "_inspect_map_how", lambda _f: ("map_dunbarton", "tpl")
    )
    coarse_type = SimpleNamespace(
        ui_stack_names=["MapUIState", "HudUIStepState"],
        ui_stack_ready=True,
        map_type=4,
        geo="庫漢",
    )
    assert compute_inspect(object(), view=coarse_type)["node"] == "map_dunbarton"
    assert compute_inspect(None, view=coarse_type)["node"] == "map"


def test_spots_omit_forbidden():
    names = [name for vals in SPOTS.values() for name in vals]
    for bad in ("头像", "皮革", "食材"):
        assert bad not in names
    assert "製作" not in SPOTS["main_hud"]
    assert "背包" not in SPOTS["main_hud"]
    assert "奈丽丝" not in " ".join(names)
    assert spots_for_node("dialog") == ()
    assert "菜单" in spots_for_node("main_hud")
    assert "藥品加工" in spots_for_node("craft_hub")
    assert "皮革加工" in spots_for_node("craft_hub")
    assert "食材加工" in spots_for_node("craft_hub")
    for node in ("craft_quick", "craft_hub", "craft_make", "craft_alchemy"):
        for tab in ("加工", "制作", "炼金术"):
            assert tab in spots_for_node(node)
    assert spots_for_node("life_grid") == (
        "日常採集",
        "伐木",
        "採礦",
        "採集藥草",
        "剪羊毛",
        "收割",
        "鋤地",
        "昆蟲採集",
        "釣魚",
        "关闭/X",
    )
    assert "鍛造" not in spots_for_node("life_grid")
    assert spots_for_node("currency_wallet") == ("右上 X",)


def test_all_zh_nodes_have_spot_tuple():
    for key in NODE_ZH:
        assert key in SPOTS


def test_locate_spot_none_on_empty():
    assert locate_spot("main_hud", "菜单", None) is None
    assert locate_spot("main_hud", "", object()) is None


def test_locate_craft_side_tabs():
    from mm_assist.life_plan import node_inspect as ni

    assert ni.locate_spot("craft_quick", "加工", object()) is None
    assert ni.locate_spot("craft_make", "炼金术", object()) is None
    assert ni.locate_spot("craft_alchemy", "制作", None) is None
    assert ni._craft_side_tab_xy(object(), "加工") is None
    assert ni.craft_tab_xy(None, "加工") is None


def test_craft_side_tab_xy_uses_template_hit_only(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(
        ni,
        "_craft_tab_scores",
        lambda _f: {
            "craft_hub": {"on": 0.0, "off": 0.81, "on_xy": None, "off_xy": (96, 670)},
            "craft_quick": {"on": 0.0, "off": 0.0, "on_xy": None, "off_xy": None},
            "craft_make": {"on": 0.0, "off": 0.0, "on_xy": None, "off_xy": None},
            "craft_alchemy": {"on": 0.0, "off": 0.0, "on_xy": None, "off_xy": None},
        },
    )
    assert ni._craft_side_tab_xy(object(), "加工") == (96, 670)
    monkeypatch.setattr(ni, "_craft_tab_scores", lambda _f: ni._empty_craft_tab_scores())
    assert ni._craft_side_tab_xy(object(), "加工") is None


def test_compute_inspect_no_frame_has_node_key():
    view = SimpleNamespace(ui_stack_names=[], ui_stack_ready=False)
    out = compute_inspect(None, spot="", view=view)
    assert "node" in out
    assert out["xy"] is None
    out2 = compute_inspect(None, spot="菜单", view=view)
    assert out2["xy"] is None
    assert out["node"] == ""


def test_mem_inspect_node_hard_family_only():
    assert mem_inspect_node(
        SimpleNamespace(
            family="menu",
            ui_stack_ready=True,
            ui_stack_names=["SideNavigationUIState", "HudUIStepState"],
        )
    ) == "system_menu"
    assert mem_inspect_node(
        SimpleNamespace(
            family="craft_hub",
            ui_stack_ready=True,
            ui_stack_names=["CraftingHubUIState", "HudUIStepState"],
            craft_hub_category=1,
        )
    ) == ""
    assert mem_inspect_node(
        SimpleNamespace(
            family="craft_hub",
            ui_stack_ready=True,
            ui_stack_names=["HudUIStepState"],
        )
    ) == "overlay"
    assert mem_inspect_node(
        SimpleNamespace(
            family="prf",
            ui_stack_ready=True,
            ui_stack_names=["NewProfileUIState", "HudUIStepState"],
        )
    ) == "profile"
    assert mem_inspect_node(
        SimpleNamespace(
            family="main_hud",
            ui_stack_ready=True,
            ui_stack_names=["TutorialTooltipUIState"],
        )
    ) == "main_hud"
    assert format_node_zh("system_menu") == "菜单"


def test_compute_inspect_inv_menu_before_ocr(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(
        ni, "refine_inspect_node", lambda _f, category=None, **_k: "facility:metal"
    )
    monkeypatch.setattr(ni, "locate_spot", lambda *_a, **_k: (1, 1))
    bag = SimpleNamespace(
        ui_stack_names=["EquipmentInventoryUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    assert ni.compute_inspect(object(), spot="", view=bag)["node"] == "inventory"
    main = SimpleNamespace(
        ui_stack_names=["TutorialTooltipUIState"],
        ui_stack_ready=True,
    )
    assert ni.compute_inspect(object(), spot="", view=main)["node"] == "main_hud"
    hub = SimpleNamespace(
        ui_stack_names=["CraftingHubUIState", "HudUIStepState"],
        ui_stack_ready=True,
        craft_hub_category=1,
    )
    assert ni.compute_inspect(None, spot="", view=hub)["node"] == "craft_shell"
    hub_quick = SimpleNamespace(
        ui_stack_names=["CraftingHubUIState", "HudUIStepState"],
        ui_stack_ready=True,
        craft_hub_category=0,
    )
    assert ni.compute_inspect(None, spot="", view=hub_quick)["node"] == "craft_shell"


def test_inventory_bag_mats_and_obtain_overlay(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    spots = ni.spots_for_node("inventory")
    assert "樹液" in spots
    assert "煤炭" in spots
    assert "关闭背包" in spots

    class _Ln:
        def __init__(self, text: str, box: list[tuple[int, int]]):
            self.text = text
            self.box = box

    monkeypatch.setattr(
        ni,
        "_ocr_lines_roi",
        lambda *_a, **_k: [
            _Ln("10/煤炭", [(800, 120), (920, 120), (920, 160), (800, 160)])
        ],
    )
    assert ni._bag_item_xy(object(), "煤炭") == (860, 140)

    monkeypatch.setattr(ni, "_inventory_overlay_node", lambda _f: "item_info:樹液")
    monkeypatch.setattr(ni, "locate_spot", lambda *_a, **_k: None)
    bag = SimpleNamespace(
        ui_stack_names=["EquipmentInventoryUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    assert ni.compute_inspect(object(), spot="", view=bag)["node"] == "item_info:樹液"
    detail = SimpleNamespace(
        ui_stack_names=["DetailCardUIState", "EquipmentInventoryUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    assert ni.compute_inspect(object(), spot="", view=detail)["node"] == "item_info"
    howto = SimpleNamespace(
        ui_stack_names=["HowToGetSelectUIState", "EquipmentInventoryUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    assert ni.compute_inspect(object(), spot="", view=howto)["node"] == "obtain_method"


def test_compute_inspect_life_grid_needs_spot(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "locate_spot", lambda *_a, **_k: (325, 245))
    monkeypatch.setattr(ni, "_gather_detail_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    view = SimpleNamespace(
        ui_stack_names=["LivingSkillUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    no_frame = ni.compute_inspect(None, spot="", view=view)
    assert no_frame["node"] == "life_grid"
    empty = ni.compute_inspect(object(), spot="", view=view)
    assert empty["node"] == "life_grid"
    assert empty["xy"] is None
    picked = ni.compute_inspect(object(), spot="日常採集", view=view)
    assert picked["xy"] == (325, 245)


def test_compute_inspect_life_grid_refines_from_frame(monkeypatch):
    import numpy as np
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    view = SimpleNamespace(
        ui_stack_names=["LivingSkillUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    monkeypatch.setattr(ni, "_gather_detail_visible", lambda _f: True)
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "daily")
    assert ni.compute_inspect(object(), spot="", view=view)["node"] == "gather_detail"
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    monkeypatch.setattr(ni, "_ocr_blob", lambda *_a, **_k: "蜘蛛網")
    assert ni.compute_inspect(frame, spot="", view=view)["node"] == "gather_detail:蜘蛛網"
    monkeypatch.setattr(ni, "_ocr_blob", lambda *_a, **_k: "不是物名")
    assert ni.compute_inspect(frame, spot="", view=view)["node"] == "gather_detail"
    prev = {"node": "gather_detail:蜘蛛網", "stack_ready": True, "stack_names": []}
    kept, _ = hold_inspect_payload(
        prev,
        {"node": "gather_detail", "stack_ready": True, "stack_names": ["LivingSkillUIState"]},
    )
    assert kept["node"] == "gather_detail:蜘蛛網"
    assert ni.spots_for_node("gather_detail:蜘蛛網") == ni.spots_for_node("gather_detail")
    assert ni.mem_should_apply_stack("gather_detail", "gather_detail:蜘蛛網", inflight=False) is False
    monkeypatch.setattr(ni, "_gather_detail_visible", lambda _f: False)
    assert ni.compute_inspect(object(), spot="", view=view)["node"] == "gather_skill:daily"
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    assert ni.compute_inspect(object(), spot="", view=view)["node"] == "life_grid"
    assert ni.inspect_needs_grab("life_grid") is True


def test_refine_inspect_node_order(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        return "facility/back.png" in rel.replace("\\", "/")

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(
        "mm_assist.perception.process_sense._match_card",
        lambda *a, **k: None,
    )
    assert ni.refine_inspect_node(object()) == "facility"


def test_refine_facility_kind(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        return "facility/back.png" in rel.replace("\\", "/")

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(ni, "_detect_facility_fid", lambda _f: "metal")
    monkeypatch.setattr(ni, "_detect_facility_recipe", lambda *_a, **_k: "")
    assert ni.refine_inspect_node(object()) == "facility:metal"
    assert "合金鋼錠" in ni.spots_for_node("facility:metal")
    assert "鐵錠(礦石)" in ni.spots_for_node("facility:metal")
    assert "左上 ←" in ni.spots_for_node("facility:metal")
    assert "前往加工" not in ni.spots_for_node("facility:metal")
    rec_spots = ni.spots_for_node("facility:metal:鐵錠")
    assert "前往加工" in rec_spots
    assert "加工" in rec_spots
    assert "合金鋼錠" in rec_spots


def test_refine_overlay_and_empty_skip_hub_scan(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    calls = {"n": 0}

    def count_hub(_f):
        calls["n"] += 1
        return False

    monkeypatch.setattr(ni, "_hub_overview_visible", count_hub)
    monkeypatch.setattr(ni, "_gather_detail_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "")
    monkeypatch.setattr(ni, "_tmpl", lambda *_a, **_k: False)
    monkeypatch.setattr(ni, "_facility_if_claim_all", lambda _f: "")
    monkeypatch.setattr(ni, "_system_menu_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.currency_wallet_visible",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "")
    monkeypatch.setattr(ni, "_process_hub_visible", lambda _f: False)
    ni.refine_inspect_node(object(), sn="overlay")
    ni.refine_inspect_node(object(), sn="")
    assert calls["n"] == 0


def test_hub_overview_visible_templates_only(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.perception import process_sense as ps

    def fake_match(_frame, card_id, *, root=None):
        _ = root
        if card_id in ("metal", "wood"):
            return object()
        return None

    def boom(*_a, **_k):
        raise AssertionError("hub overview must not OCR")

    monkeypatch.setattr(ps, "_match_card", fake_match)
    monkeypatch.setattr(ps, "_ocr_lines", boom)
    monkeypatch.setattr(ps, "_any_hub_card", boom)
    assert ni._hub_overview_visible(object()) is True


def test_refine_hub_cards_stay_craft_hub(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        return "facility/back.png" in rel.replace("\\", "/")

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(ni, "_hub_overview_visible", lambda _f: True)
    monkeypatch.setattr(ni, "_detect_facility_fid", lambda _f: "metal")
    monkeypatch.setattr(ni, "_facility_if_claim_all", lambda _f: "facility:metal")
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "craft_hub")
    node = ni.refine_inspect_node(object(), sn="fine:craft_hub")
    assert node == "craft_hub"
    assert ni.format_node_zh(node) == "制作-加工"
    assert ni.inspect_page_zh(node, stack_ready=True) == "制作-加工"


def test_refine_facility_title_without_hub_cards(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        return "facility/back.png" in str(rel).replace("\\", "/")

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(ni, "_hub_overview_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_facility_if_claim_all", lambda _f: "")
    monkeypatch.setattr(ni, "_detect_facility_fid", lambda _f: "metal")
    monkeypatch.setattr(ni, "_detect_facility_recipe", lambda *_a, **_k: "")
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "craft_hub")
    node = ni.refine_inspect_node(object(), sn="fine:facility")
    assert node == "facility:metal"
    assert ni.format_node_zh(node) == "制作-加工-金属加工"
    assert ni.inspect_page_zh(node, stack_ready=True) == "制作-加工-金属加工"


def test_refine_claim_all_keeps_facility_fid(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_tmpl", lambda *_a, **_k: False)
    monkeypatch.setattr(ni, "_hub_overview_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "")
    monkeypatch.setattr(ni, "_detect_item_info", lambda _f: "")
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda _f: {"ok": True},
    )
    monkeypatch.setattr(ni, "_detect_facility_fid", lambda _f: "metal")
    monkeypatch.setattr(ni, "_detect_facility_recipe", lambda *_a, **_k: "")
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "craft_hub")
    node = ni.refine_inspect_node(object(), sn="fine:facility")
    assert node == "facility:metal"
    assert ni.format_node_zh(node) == "制作-加工-金属加工"
    assert ni.inspect_page_zh(node, stack_ready=True) == "制作-加工-金属加工"


def test_refine_craft_hub_sn_ignores_facility_title(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_tmpl", lambda *_a, **_k: False)
    monkeypatch.setattr(ni, "_hub_overview_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_facility_if_claim_all", lambda _f: "")
    monkeypatch.setattr(ni, "_detect_facility_fid", lambda _f: "wood")
    monkeypatch.setattr(ni, "_detect_facility_recipe", lambda *_a, **_k: "")
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "craft_hub")
    monkeypatch.setattr(ni, "_process_hub_visible", lambda _f: False)
    for sn in ("fine:craft_hub", "craft_shell"):
        node = ni.refine_inspect_node(object(), sn=sn)
        assert node == "craft_hub"
        assert not str(node).startswith("facility")
        assert ni.format_node_zh(node) == "制作-加工"


def test_refine_empty_sn_still_facility_from_title(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_tmpl", lambda *_a, **_k: False)
    monkeypatch.setattr(ni, "_hub_overview_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_gather_detail_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "")
    monkeypatch.setattr(ni, "_facility_if_claim_all", lambda _f: "")
    monkeypatch.setattr(ni, "_system_menu_visible", lambda _f: False)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.currency_wallet_visible",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    monkeypatch.setattr(ni, "_detect_facility_fid", lambda _f: "wood")
    monkeypatch.setattr(ni, "_detect_facility_recipe", lambda *_a, **_k: "")
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "craft_hub")
    monkeypatch.setattr(ni, "_process_hub_visible", lambda _f: False)
    node = ni.refine_inspect_node(object(), sn="")
    assert node == "facility:wood"


def test_refine_facility_recipe(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        return "facility/back.png" in rel.replace("\\", "/")

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(ni, "_detect_facility_fid", lambda _f: "metal")
    monkeypatch.setattr(ni, "_detect_facility_recipe", lambda *_a, **_k: "鐵錠")
    assert ni.refine_inspect_node(object()) == "facility:metal"
    assert ni.refine_inspect_node(object(), sn="fine:facility_recipe") == "facility:metal:鐵錠"
    assert ni.format_node_zh("facility:metal:鐵錠") == "制作-加工-金属加工-铁锭"


def test_coal_obtain_chain_inspect(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    assert ni.format_node_zh("item_info:煤炭") == "物品详情-煤炭"
    assert ni.format_node_zh("obtain_method:煤炭") == "获得方法-煤炭"
    alloy = ni.spots_for_node("facility:metal:合金鋼錠")
    assert "煤炭" in alloy
    assert "銅礦石" in alloy
    assert ni.spots_for_node("item_info:煤炭") == ("获得方法", "附近位置")
    assert ni.spots_for_node("obtain_method:煤炭") == ("煤炭礦脈",)

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        return "facility/back.png" in rel.replace("\\", "/")

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(ni, "_detect_facility_fid", lambda _f: "metal")
    monkeypatch.setattr(ni, "_detect_facility_recipe", lambda *_a, **_k: "合金鋼錠")
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "obtain_method:煤炭")
    monkeypatch.setattr(ni, "_detect_item_info", lambda _f: "item_info:煤炭")
    assert ni.refine_inspect_node(object()) == "obtain_method:煤炭"
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "")
    assert ni.refine_inspect_node(object()) == "item_info:煤炭"
    monkeypatch.setattr(ni, "_detect_item_info", lambda _f: "")
    assert ni.refine_inspect_node(object()) == "facility:metal"

    class _Ln:
        def __init__(self, text: str, box: list[tuple[int, int]]):
            self.text = text
            self.box = box

    lines = [
        _Ln("✦ 推薦 ✦", [(80, 40), (200, 40), (200, 70), (80, 70)]),
        _Ln("煤炭礦脈", [(120, 90), (400, 90), (400, 130), (120, 130)]),
        _Ln("煤炭礦脈", [(120, 210), (400, 210), (400, 250), (120, 250)]),
    ]
    monkeypatch.setattr(ni, "_ocr_lines_roi", lambda *_a, **_k: lines)
    assert ni._named_line_xy(
        object(), ni._OBTAIN_LIST_ROI, "煤炭礦脈", contains=True
    ) == (260, 110)
    simp = [_Ln("煤炭矿脉", [(120, 90), (400, 90), (400, 130), (120, 130)])]
    monkeypatch.setattr(ni, "_ocr_lines_roi", lambda *_a, **_k: simp)
    assert ni._obtain_vein_xy(object()) == (260, 110)
    monkeypatch.setattr(ni, "_ocr_lines_roi", lambda *_a, **_k: [])
    import numpy as np

    blank = np.zeros((1000, 1600, 3), dtype=np.uint8)
    assert ni._obtain_vein_xy(blank) == (int(1600 * 0.64), int(1000 * 0.26))
    assert ni.locate_spot("obtain_method:煤炭", "煤炭矿脉", blank) == (
        int(1600 * 0.64),
        int(1000 * 0.26),
    )


def test_inspect_skips_popup_ocr_on_recipe_page(monkeypatch):
    import numpy as np
    from mm_assist.life_plan import node_inspect as ni

    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    called: list[str] = []

    def fake_ocr(frm, roi, *, fallback=False):
        called.append("ocr")
        return []

    monkeypatch.setattr(ni, "_ocr_lines_roi", fake_ocr)
    monkeypatch.setattr(ni, "_process_btn_hit", lambda _f: True)
    assert ni._detect_obtain_method(frame) == ""
    assert "ocr" not in called
    assert ni._detect_item_info(frame) == ""


def test_detect_item_info_coal(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    class _Ln:
        def __init__(self, text: str):
            self.text = text
            self.box = [(0, 0), (1, 0), (1, 1), (0, 1)]

    def fake_ocr(_f, roi, *, fallback=False):
        _ = fallback
        if roi == ni._ITEM_INFO_FOOT_ROI:
            return [_Ln("獲得方法")]
        if roi == ni._ITEM_INFO_HDR_ROI:
            return [_Ln("煤炭"), _Ln("材料")]
        return []

    monkeypatch.setattr(ni, "_ocr_lines_roi", fake_ocr)
    assert ni._detect_item_info(object()) == "item_info:煤炭"


def test_detect_bag_item_card_sap(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    class _Ln:
        def __init__(self, text: str):
            self.text = text
            self.box = [(0, 0), (1, 0), (1, 1), (0, 1)]

    monkeypatch.setattr(
        ni,
        "_ocr_lines_roi",
        lambda _f, roi, *, fallback=False: (
            [_Ln("樹液"), _Ln("材料"), _Ln("獲得方法")]
            if roi == ni._BAG_CARD_ROI
            else []
        ),
    )
    assert ni._detect_bag_item_card(object()) == "item_info:樹液"
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "")
    monkeypatch.setattr(ni, "_detect_item_info", lambda _f, **k: "")
    assert ni._inventory_overlay_node(object()) == "item_info:樹液"


def test_sap_obtain_chain_inspect(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    assert ni.format_node_zh("facility:wood:高級木材") == "制作-加工-木材加工-高级木材"
    assert ni.format_node_zh("item_info:樹液") == "物品详情-树液"
    assert ni.format_node_zh("obtain_method:樹液") == "获得方法-树液"
    assert "樹液" in ni.spots_for_node("facility:wood:高級木材")
    assert ni.spots_for_node("item_info:樹液") == ("获得方法", "附近位置")
    assert ni.spots_for_node("obtain_method:樹液") == ("伐木",)
    assert ni._pick_material_from_blob("树液材料可販售") == "樹液"

    class _Ln:
        def __init__(self, text: str, box: list[tuple[int, int]]):
            self.text = text
            self.box = box

    lines = [
        _Ln("伐木", [(100, 80), (280, 80), (280, 120), (100, 120)]),
        _Ln("伐木(道具效果)", [(100, 160), (420, 160), (420, 200), (100, 200)]),
    ]
    monkeypatch.setattr(ni, "_ocr_lines_roi", lambda *_a, **_k: lines)
    assert ni._obtain_first_named_xy(
        object(), "伐木", exclude=("道具", "道具效果")
    ) == (190, 100)
    assert ni.locate_spot("obtain_method:樹液", "伐木", object()) == (190, 100)


def test_pick_recipe_from_detail_blob():
    from mm_assist.life_plan.node_inspect import _pick_recipe_from_blob

    names = ("鐵錠", "鐵錠(礦石)", "鐵錠(鐵礦石)", "合金鋼錠")
    assert _pick_recipe_from_blob("鐵錠", names) == "鐵錠"
    assert _pick_recipe_from_blob("合金鋼錠", names) == "合金鋼錠"
    assert _pick_recipe_from_blob("铁锭", names) == "鐵錠"


def test_compute_inspect_facility_needs_spot(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(
        ni, "refine_inspect_node", lambda _f, category=None, **_k: "facility:metal"
    )
    monkeypatch.setattr(ni, "locate_spot", lambda *_a, **_k: (900, 400))
    view = SimpleNamespace(
        ui_stack_names=["CraftingHubAlteringFacilityState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    empty = ni.compute_inspect(object(), spot="", view=view)
    assert empty["node"] == "facility:metal"
    assert empty["xy"] is None
    picked = ni.compute_inspect(object(), spot="合金鋼錠", view=view)
    assert picked["xy"] == (900, 400)


def test_refine_menu_skips_gather_scan(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        p = rel.replace("\\", "/")
        return "icon_character.png" in p or "icon_inventory.png" in p

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(
        ni,
        "_detect_gather_skill_sid",
        lambda _f: (_ for _ in ()).throw(RuntimeError("gather should skip")),
    )
    monkeypatch.setattr(
        ni,
        "_process_hub_visible",
        lambda _f: (_ for _ in ()).throw(RuntimeError("hub should skip")),
    )
    assert ni.refine_inspect_node(object()) == "system_menu"


def test_compute_inspect_passes_sn_to_refine(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    seen: dict[str, str] = {}

    def fake(_f, category=None, *, sn=""):
        _ = category
        seen["sn"] = sn
        return "craft_hub"

    monkeypatch.setattr(ni, "refine_inspect_node", fake)
    view = SimpleNamespace(
        ui_stack_names=["CraftingHubUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    out = ni.compute_inspect(object(), spot="", view=view)
    assert out["node"] == "craft_hub"
    assert seen.get("sn") == "fine:craft_hub"


def test_refine_craft_hub_skips_menu_and_gather(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def boom(_f):
        raise RuntimeError("craft_hub route should skip")

    monkeypatch.setattr(ni, "_system_menu_visible", boom)
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", boom)

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        p = str(rel).replace("\\", "/")
        if "/life_grid/" in p:
            raise RuntimeError("life_grid should skip")
        return False

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(ni, "_facility_if_claim_all", lambda _f: "")
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "craft_hub")
    monkeypatch.setattr(ni, "_process_hub_visible", lambda _f: False)
    assert ni.refine_inspect_node(object(), sn="fine:craft_hub") == "craft_hub"


def test_refine_menu_icon_and_life_tab_pass_roi(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    seen: list[tuple[str, object]] = []

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold
        p = str(rel).replace("\\", "/")
        seen.append((p, roi))
        return "icon_menu_52x44.png" in p

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(ni, "_gather_detail_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "")
    monkeypatch.setattr(ni, "_facility_if_claim_all", lambda _f: "")
    monkeypatch.setattr(ni, "_system_menu_visible", lambda _f: False)
    assert ni.refine_inspect_node(object(), sn="overlay") == "main_hud"
    menu_rois = [r for p, r in seen if "icon_menu_52x44.png" in p]
    assert menu_rois and menu_rois[0] is not None

    seen.clear()

    def fake_life(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold
        p = str(rel).replace("\\", "/")
        seen.append((p, roi))
        return "profile/life_tab.png" in p

    monkeypatch.setattr(ni, "_tmpl", fake_life)
    monkeypatch.setattr(ni, "_life_skills_page_visible", lambda _f: True)
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.currency_wallet_visible",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "")
    monkeypatch.setattr(ni, "_process_hub_visible", lambda _f: False)
    assert ni.refine_inspect_node(object(), sn="") == "life_grid"
    tab_rois = [r for p, r in seen if "profile/life_tab.png" in p]
    assert tab_rois and tab_rois[0] is not None


def test_system_menu_visible_uses_locate_spot_roi(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    rois: dict[str, object] = {}

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold
        rois[str(rel).replace("\\", "/").rsplit("/", 1)[-1]] = roi
        return True

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    assert ni._system_menu_visible(object()) is True
    assert rois["icon_character.png"] == (0.66, 0.12, 0.08, 0.10)
    assert rois["icon_crafting.png"] == (0.79, 0.12, 0.08, 0.10)
    assert rois["icon_inventory.png"] == (0.73, 0.12, 0.07, 0.10)


def test_menu_icon_xy_requires_template_in_box(monkeypatch):
    import numpy as np
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.platform import match as match_mod
    from mm_assist.platform.match import MatchHit

    frame = np.zeros((1440, 1920, 3), dtype=np.uint8)
    monkeypatch.setattr(match_mod, "match_template", lambda *a, **k: None)
    assert ni._menu_icon_xy(
        frame,
        "knowledge/templates/menu/icon_inventory.png",
        (0.73, 0.12, 0.07, 0.10),
    ) is None
    stray = MatchHit(score=0.9, cx=1865, cy=91, x=1800, y=50, w=80, h=80, margin=0.1)
    monkeypatch.setattr(match_mod, "match_template", lambda *a, **k: stray)
    assert ni._menu_icon_xy(
        frame,
        "knowledge/templates/menu/icon_inventory.png",
        (0.73, 0.12, 0.07, 0.10),
    ) is None
    inside = MatchHit(score=0.9, cx=1450, cy=240, x=1423, y=202, w=59, h=70, margin=0.1)
    monkeypatch.setattr(match_mod, "match_template", lambda *a, **k: inside)
    assert ni._menu_icon_xy(
        frame,
        "knowledge/templates/menu/icon_inventory.png",
        (0.73, 0.12, 0.07, 0.10),
    ) == (1450, 240)
    assert ni._menu_icon_xy(None, "knowledge/templates/menu/icon_inventory.png", None) is None


def test_tmpl_reuses_gray_cache():
    import numpy as np
    from mm_assist.config import ROOT
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.platform import match as match_mod

    rel = "knowledge/templates/menu/icon_menu_52x44.png"
    path = ROOT / rel
    if not path.is_file():
        import pytest

        pytest.skip("missing menu template")
    match_mod.clear_template_cache()
    ni._template_exists.cache_clear()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    ni._tmpl(frame, rel, threshold=0.99, roi=ni._ROI_MENU_ENTRY)
    mid = match_mod._load_template_gray.cache_info()
    ni._tmpl(frame, rel, threshold=0.99, roi=ni._ROI_MENU_ENTRY)
    after = match_mod._load_template_gray.cache_info()
    assert after.misses == mid.misses


def test_gather_skill_anchor_pngs_cached():
    from mm_assist.life_plan import node_inspect as ni

    ni._gather_skill_page_anchor_pngs.cache_clear()
    ni._gather_skill_page_anchor_pngs("daily")
    mid = ni._gather_skill_page_anchor_pngs.cache_info()
    ni._gather_skill_page_anchor_pngs("daily")
    after = ni._gather_skill_page_anchor_pngs.cache_info()
    assert after.hits == mid.hits + 1
    assert after.misses == mid.misses


def test_refine_currency_wallet_before_life_grid(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_tmpl", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.currency_wallet_visible",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    monkeypatch.setattr(
        "mm_assist.perception.process_sense._match_card",
        lambda *a, **k: None,
    )
    assert ni.refine_inspect_node(object()) == "currency_wallet"


def test_refine_process_hub_before_gather_scan(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_tmpl", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.currency_wallet_visible",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    monkeypatch.setattr(ni, "_process_hub_visible", lambda _f: True)
    assert ni.refine_inspect_node(object()) == "craft_hub"


def test_craft_shell_from_title(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_craft_chrome", lambda _f: True)
    monkeypatch.setattr(ni, "_selected_craft_tab", lambda _f: "")
    monkeypatch.setattr(ni, "_craft_title_blob", lambda _f: "快速製作")
    assert ni._craft_shell_node(object()) == "craft_quick"
    monkeypatch.setattr(ni, "_craft_title_blob", lambda _f: "加工")
    assert ni._craft_shell_node(object()) == "craft_hub"
    monkeypatch.setattr(ni, "_craft_title_blob", lambda _f: "製作")
    assert ni._craft_shell_node(object()) == "craft_make"
    monkeypatch.setattr(ni, "_craft_title_blob", lambda _f: "制作")
    assert ni._craft_shell_node(object()) == "craft_make"
    monkeypatch.setattr(ni, "_craft_title_blob", lambda _f: "鍊金術")
    assert ni._craft_shell_node(object()) == "craft_alchemy"


def test_craft_make_shot_004_if_present():
    from pathlib import Path

    import cv2
    from mm_assist.life_plan import node_inspect as ni

    root = Path("logs/life_tests/20260914T160708_e8d84a54")
    if not root.is_dir():
        pytest.skip("live shot dir absent")
    shots = sorted(root.glob("*004*.png")) + sorted(root.glob("**/*004*.png"))
    if not shots:
        pytest.skip("004 png absent")
    frame = cv2.imread(str(shots[0]))
    if frame is None:
        pytest.skip("004 unreadable")
    assert ni._selected_craft_tab(frame) == "craft_make"
    assert ni._craft_shell_node(frame) == "craft_make"


def test_craft_shell_category_before_ocr(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_craft_chrome", lambda _f: True)
    monkeypatch.setattr(ni, "_selected_craft_tab", lambda _f: "")
    monkeypatch.setattr(
        ni,
        "_craft_title_blob",
        lambda _f: (_ for _ in ()).throw(RuntimeError("ocr should skip")),
    )
    ni._reset_inspect_cost()
    assert ni._craft_shell_how(object(), category=3) == ("craft_alchemy", "mem")


def test_craft_shell_icon_beats_category(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_craft_chrome", lambda _f: True)
    monkeypatch.setattr(ni, "_selected_craft_tab", lambda _f: "craft_make")
    monkeypatch.setattr(
        ni,
        "_craft_title_blob",
        lambda _f: (_ for _ in ()).throw(RuntimeError("ocr should skip")),
    )
    assert ni._craft_shell_node(object(), category=3) == "craft_make"


def test_craft_shell_how_memoized_same_frame(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    ni._reset_inspect_cost()
    frame = object()
    n = {"chrome": 0}

    def chrome(_f):
        n["chrome"] += 1
        return True

    monkeypatch.setattr(ni, "_craft_chrome", chrome)
    monkeypatch.setattr(ni, "_selected_craft_tab", lambda _f: "craft_hub")
    a = ni._craft_shell_how(frame, category=1)
    b = ni._craft_shell_how(frame, category=1)
    assert a == b == ("craft_hub", "tpl")
    assert n["chrome"] == 1


def test_craft_title_icon_does_not_override_mem(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    assert [n for n, _rel in ni._CRAFT_TITLE_ICONS] == ["craft_quick"]
    monkeypatch.setattr(ni, "_craft_chrome", lambda _f: True)
    monkeypatch.setattr(ni, "_selected_craft_tab", lambda _f: "")
    monkeypatch.setattr(ni, "_best_title_icon", lambda _f: "craft_quick")
    monkeypatch.setattr(
        ni,
        "_craft_title_blob",
        lambda _f: (_ for _ in ()).throw(RuntimeError("ocr should skip")),
    )
    assert ni._craft_shell_how(object(), category=1) == ("craft_hub", "mem")


def test_craft_shell_icon_before_ocr(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_craft_chrome", lambda _f: True)
    monkeypatch.setattr(ni, "_selected_craft_tab", lambda _f: "craft_make")
    monkeypatch.setattr(
        ni,
        "_craft_title_blob",
        lambda _f: (_ for _ in ()).throw(RuntimeError("ocr should skip")),
    )
    assert ni._craft_shell_node(object()) == "craft_make"


def test_craft_shell_without_left_tabs_not_quick(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_craft_chrome", lambda _f: False)
    monkeypatch.setattr(ni, "_best_title_icon", lambda _f: "craft_quick")
    monkeypatch.setattr(ni, "_craft_title_blob", lambda _f: "快速製作")
    assert ni._craft_shell_node(object()) == ""


def _craft_slot_px(ni, node: str) -> tuple[int, int]:
    for slot in ni._craft_tab_slots():
        if slot[0] == node:
            return int(slot[3] * 1920), int(slot[4] * 1440)
    raise AssertionError(node)


def test_craft_selected_tab_cyan(monkeypatch):
    import numpy as np
    from mm_assist.life_plan import node_inspect as ni

    frame = np.zeros((1440, 1920, 3), np.uint8)
    for slot in ni._craft_tab_slots():
        x, y = int(slot[3] * 1920), int(slot[4] * 1440)
        frame[y - 12 : y + 12, x - 12 : x + 12] = (160, 160, 160)
    mx, my = _craft_slot_px(ni, "craft_make")
    frame[my - 12 : my + 12, mx - 12 : mx + 12] = (220, 200, 40)
    monkeypatch.setattr(ni, "_match_rel", lambda *_a, **_k: None)
    monkeypatch.setattr(ni, "_craft_chrome", lambda _f: True)
    monkeypatch.setattr(ni, "_craft_title_blob", lambda _f: "")
    assert ni._craft_shell_node(frame) == "craft_make"


def test_craft_selected_tab_on_template(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    cx, cy = _craft_slot_px(ni, "craft_alchemy")

    hit = SimpleNamespace(score=0.91, cx=cx, cy=cy)

    def fake_match(_frame, rel, *, roi=None, threshold=0.0):
        if rel.endswith("tab_alchemy_on.png"):
            return hit
        return None

    monkeypatch.setattr(ni, "_match_rel", fake_match)
    monkeypatch.setattr(ni, "_craft_chrome", lambda _f: True)
    monkeypatch.setattr(ni, "_craft_title_blob", lambda _f: "")
    import numpy as np

    frame = np.zeros((1440, 1920, 3), np.uint8)
    assert ni._selected_craft_tab(frame) == "craft_alchemy"
    assert ni._craft_shell_node(frame) == "craft_alchemy"


def test_craft_tab_geometry_gate(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    qx, qy = _craft_slot_px(ni, "craft_quick")

    hit = SimpleNamespace(score=0.95, cx=qx, cy=qy)

    def fake_match(_frame, rel, *, roi=None, threshold=0.0):
        if rel.endswith("tab_alchemy_on.png"):
            return hit
        return None

    monkeypatch.setattr(ni, "_match_rel", fake_match)
    monkeypatch.setattr(ni, "_craft_chrome", lambda _f: True)
    import numpy as np

    frame = np.zeros((1440, 1920, 3), np.uint8)
    scores = ni._craft_tab_scores(frame)
    assert scores["craft_alchemy"]["on"] == 0.0
    assert ni._selected_craft_tab(frame) == ""


def test_craft_tab_on_must_beat_off(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    cx, cy = _craft_slot_px(ni, "craft_hub")

    def fake_match(_frame, rel, *, roi=None, threshold=0.0):
        if rel.endswith("tab_process_on.png"):
            return SimpleNamespace(score=0.72, cx=cx, cy=cy)
        if rel.endswith("tab_process_off.png"):
            return SimpleNamespace(score=0.88, cx=cx, cy=cy)
        return None

    monkeypatch.setattr(ni, "_match_rel", fake_match)
    monkeypatch.setattr(ni, "_craft_chrome", lambda _f: True)
    import numpy as np

    frame = np.zeros((1440, 1920, 3), np.uint8)
    assert ni._selected_craft_tab(frame) == ""


def test_craft_tab_scores_one_pass(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    n = {"n": 0}

    def fake_match(*_a, **_k):
        n["n"] += 1
        return None

    monkeypatch.setattr(ni, "_match_rel", fake_match)
    import numpy as np

    frame = np.zeros((1440, 1920, 3), np.uint8)
    ni._reset_inspect_cost()
    ni._craft_tab_scores(frame)
    ni._craft_chrome(frame)
    ni._selected_craft_tab(frame)
    assert n["n"] == 8


def test_craft_tab_slots_from_json():
    from mm_assist.life_plan import node_inspect as ni

    slots = {s[0]: s for s in ni._craft_tab_slots()}
    assert set(slots) == {"craft_quick", "craft_hub", "craft_make", "craft_alchemy"}
    quick = slots["craft_quick"]
    assert abs(quick[3] - ((34 + 164) / 2) / 1920) < 0.002
    assert abs(quick[4] - ((430 + 558) / 2) / 1440) < 0.002
    alch = slots["craft_alchemy"]
    assert abs(alch[3] - ((41 + 156) / 2) / 1920) < 0.002
    assert abs(alch[4] - ((968 + 1091) / 2) / 1440) < 0.002


def test_process_hub_visible_by_title(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_craft_chrome", lambda _f: True)
    monkeypatch.setattr(ni, "_craft_title_blob", lambda _f: "加工")
    assert ni._process_hub_visible(object()) is True
    monkeypatch.setattr(ni, "_craft_title_blob", lambda _f: "快速製作")
    monkeypatch.setattr("mm_assist.perception.process_sense._match_card", lambda *_a, **_k: None)
    assert ni._process_hub_visible(object()) is False
    assert ni._quick_craft_visible(object()) is True


def test_refine_quick_craft(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_tmpl", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.currency_wallet_visible",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "craft_quick")
    monkeypatch.setattr(ni, "_process_hub_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    monkeypatch.setattr(
        "mm_assist.perception.process_sense._match_card",
        lambda *a, **k: None,
    )
    assert ni.refine_inspect_node(object()) == "craft_quick"


def test_refine_profile_before_craft_quick(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        p = str(rel).replace("\\", "/")
        return "profile/life_tab.png" in p or "/life_grid/daily.png" in p

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(ni, "_system_menu_visible", lambda _f: False)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.currency_wallet_visible",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "craft_quick")
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "")
    monkeypatch.setattr(ni, "_detect_item_info", lambda _f: "")
    assert ni.refine_inspect_node(object()) == "life_grid"


def test_refine_claim_all_not_profile(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        return "profile/life_tab.png" in str(rel).replace("\\", "/")

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(ni, "_system_menu_visible", lambda _f: False)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.currency_wallet_visible",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "")
    monkeypatch.setattr(ni, "_detect_item_info", lambda _f: "")
    monkeypatch.setattr(
        ni,
        "_facility_if_claim_all",
        lambda _f: "facility",
    )
    assert ni.refine_inspect_node(object()) == "facility"


def test_refine_gather_skill_before_life_grid(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        return "life_grid/" in str(rel).replace("\\", "/")

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(ni, "_system_menu_visible", lambda _f: False)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.currency_wallet_visible",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "daily")
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "")
    assert ni.refine_inspect_node(object()) == "gather_skill:daily"
    assert ni.format_node_zh("gather_skill:daily") == "角色-生活技能-日常採集"
    assert ni.format_node_zh("gather_skill:woodcut") == "角色-生活技能-伐木"
    assert ni.format_node_zh("gather_skill:shear") == "角色-生活技能-剪羊毛"
    assert ni.format_node_zh("gather_skill:insect") == "角色-生活技能-昆蟲採集"


def test_compute_inspect_not_ready_skips_refine(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    calls = {"n": 0}

    def boom(_f):
        calls["n"] += 1
        return "main_hud"

    monkeypatch.setattr(ni, "refine_inspect_node", boom)
    view = SimpleNamespace(ui_stack_names=[], ui_stack_ready=False)
    out = ni.compute_inspect(object(), spot="", view=view)
    assert out["node"] == ""
    assert out["node"] != "main_hud"
    assert calls["n"] == 0


def test_compute_inspect_grab_nodes_call_refine(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    calls = {"n": 0}

    def count(_f, category=None):
        calls["n"] += 1
        return ""

    monkeypatch.setattr(ni, "refine_inspect_node", count)
    monkeypatch.setattr(ni, "_inventory_overlay_node", lambda _f: "")
    bag = SimpleNamespace(
        ui_stack_names=["EquipmentInventoryUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    assert ni.compute_inspect(object(), spot="", view=bag)["node"] == "inventory"
    assert calls["n"] == 0
    calls["n"] = 0
    main = SimpleNamespace(
        ui_stack_names=["TutorialTooltipUIState"],
        ui_stack_ready=True,
    )
    assert ni.compute_inspect(object(), spot="", view=main)["node"] == "main_hud"
    assert calls["n"] == 0
    assert ni.inspect_needs_grab("main_hud") is False
    assert ni.inspect_needs_grab("overlay") is False
    assert ni.inspect_needs_grab("item_info") is True
    assert ni.inspect_needs_grab("obtain_method") is True
    assert ni.inspect_needs_grab("profile") is True
    assert ni.inspect_needs_grab("profile:info") is False
    assert ni.inspect_needs_grab("inventory") is True
    assert ni.inspect_needs_grab("fine:facility") is True
    assert ni.inspect_needs_grab("fine:craft_hub") is True
    assert ni.inspect_needs_grab("craft_shell") is True
    assert ni.inspect_needs_grab("life_grid") is True
    assert ni.inspect_needs_grab("map") is True
    assert ni.inspect_needs_grab("craft_hub") is False
    assert ni.mem_should_apply_stack("profile", "", inflight=True) is True
    assert ni.mem_should_apply_stack("profile", "profile", inflight=True) is False
    assert ni.mem_should_apply_stack("profile", "life_grid", inflight=False) is False
    assert ni.mem_should_apply_stack("profile", "profile:info", inflight=False) is False
    assert ni.mem_should_apply_stack("profile", "gather_skill:daily", inflight=False) is False
    assert ni.mem_should_apply_stack("profile", "gather_detail", inflight=False) is False
    assert ni.mem_should_apply_stack("system_menu", "", inflight=False) is True
    assert ni.mem_should_apply_stack("life_grid", "", inflight=True) is True
    assert ni.mem_should_apply_stack("life_grid", "gather_detail", inflight=True) is False
    assert ni.mem_should_apply_stack(
        "life_grid", "gather_skill:daily", inflight=True
    ) is False
    assert ni.mem_should_apply_stack("life_grid", "gather_detail", inflight=False) is False
    assert ni.mem_should_apply_stack("fine:craft_hub", "craft_make", inflight=False) is False
    assert ni.mem_should_apply_stack("fine:craft_hub", "", inflight=False) is True
    assert ni.mem_should_apply_stack("fine:craft_hub", "facility:metal", inflight=False) is True
    assert ni.mem_should_apply_stack("fine:craft_hub", "facility", inflight=False) is True
    calls["n"] = 0
    dlg = SimpleNamespace(
        ui_stack_names=["DialogueInteractionUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    assert ni.compute_inspect(object(), spot="", view=dlg)["node"] == "dialog"
    assert calls["n"] == 0
    calls["n"] = 0
    menu = SimpleNamespace(
        ui_stack_names=["SideNavigationUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    assert ni.compute_inspect(object(), spot="", view=menu)["node"] == "system_menu"
    assert calls["n"] == 0


def test_gather_skill_spots_are_complete():
    from mm_assist.perception.item_sense import items_for_skill

    herb = spots_for_node("gather_skill:herb")
    assert herb == items_for_skill("herb")
    # 台服 S1 新增柔柔蘑菇，保留原有十二项。
    assert len(herb) == 13
    assert {'傷痕花', '增幅蘑菇', '柔柔蘑菇'} <= set(herb)
    assert "淨淨蘑菇" in herb
    insect = spots_for_node("gather_skill:insect")
    assert insect == items_for_skill("insect")
    assert {"冰冷的光群", "荒蕪的昆蟲群", "荒涼的昆蟲群"} <= set(insect)
    daily = spots_for_node("gather_skill:daily")
    assert daily == ("巢穴", "蜘蛛網", "水", "水井", "蜘蛛絲團")


def test_detect_gather_skill_picks_peak(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def fake_match(_frame, path, **kwargs):
        p = str(path).replace("\\", "/")
        from mm_assist.platform.match import MatchHit

        if "/gather_skill/daily/" in p or "/items/daily/巢穴.png" in p:
            return MatchHit(score=0.97, cx=1, cy=1, x=0, y=0, w=8, h=8)
        if "/insect/" in p:
            return MatchHit(score=0.62, cx=1, cy=1, x=0, y=0, w=8, h=8)
        return None

    monkeypatch.setattr("mm_assist.platform.match.match_template", fake_match)
    assert ni._detect_gather_skill_sid(object()) == "daily"


def test_unique_item_xy_rejects_lookalike_and_offscreen(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.platform.match import MatchHit

    def fake_match(_frame, path, **kwargs):
        p = str(path).replace("\\", "/")
        table = {
            "粗壮樹.png": MatchHit(score=0.99, cx=1100, cy=500, x=1000, y=460, w=80, h=80),
            "成材樹.png": MatchHit(score=0.80, cx=1100, cy=505, x=1000, y=465, w=80, h=80),
            "甲胄樹.png": MatchHit(score=0.79, cx=1100, cy=302, x=1000, y=262, w=80, h=80),
        }
        for key, hit in table.items():
            if p.endswith(key) or f"/{key}" in p:
                return hit
        return None

    monkeypatch.setattr("mm_assist.platform.match.match_template", fake_match)
    assert ni._unique_item_xy(object(), "woodcut", "粗壮樹") == (1100, 500)
    assert ni._unique_item_xy(object(), "woodcut", "成材樹") is None
    assert ni._unique_item_xy(object(), "woodcut", "甲胄樹") is None
    assert ni.locate_spot("gather_skill:woodcut", "甲胄樹", object()) is None


def test_unique_item_xy_last_row_visible(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.platform.match import MatchHit

    def fake_match(_frame, path, **kwargs):
        p = str(path).replace("\\", "/")
        roi = kwargs.get("roi") or (0.0, 0.0, 1.0, 1.0)
        y0 = float(roi[1])
        last = y0 >= 0.80
        if "甲胄樹" in p and last:
            return MatchHit(score=0.95, cx=1100, cy=1320, x=1000, y=1280, w=80, h=80)
        if p.endswith("樹木.png") and y0 < 0.70:
            return MatchHit(score=0.96, cx=1100, cy=920, x=1000, y=880, w=80, h=80)
        return None

    monkeypatch.setattr("mm_assist.platform.match.match_template", fake_match)
    assert ni._unique_item_xy(object(), "woodcut", "甲胄樹") == (1100, 1320)


def test_ocr_item_xy_reads_last_row_name(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.perception.ocr import OcrLine

    monkeypatch.setattr(ni, "_unique_item_xy", lambda *_a, **_k: None)

    def fake_ocr(_frame, _roi, **_k):
        return [
            OcrLine(
                text="甲胄樹",
                confidence=0.9,
                box=[[1200.0, 1300.0], [1400.0, 1300.0], [1400.0, 1360.0], [1200.0, 1360.0]],
            )
        ]

    monkeypatch.setattr("mm_assist.perception.ocr.ocr_roi", fake_ocr)
    assert ni._item_list_xy(object(), "woodcut", "甲胄樹") == (1300, 1330)


def test_item_list_xy_offscreen_cold_swarm_not_other(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.perception.ocr import OcrLine
    from mm_assist.platform.match import MatchHit

    def fake_match(_frame, path, **kwargs):
        p = str(path).replace("\\", "/")
        if "冰冷的光群" in p:
            return MatchHit(score=0.88, cx=1100, cy=700, x=1000, y=660, w=80, h=80)
        if "寧靜的光群" in p:
            return MatchHit(score=0.86, cx=1100, cy=700, x=1000, y=660, w=80, h=80)
        return None

    def fake_ocr(_frame, _roi, **_k):
        return [
            OcrLine(text="光群", confidence=0.9, box=[[1, 1], [2, 1], [2, 2], [1, 2]]),
            OcrLine(text="寧靜的光群", confidence=0.9, box=[[1100, 680], [1300, 680], [1300, 720], [1100, 720]]),
            OcrLine(text="溫暖的光群", confidence=0.9, box=[[1100, 800], [1300, 800], [1300, 840], [1100, 840]]),
        ]

    monkeypatch.setattr("mm_assist.platform.match.match_template", fake_match)
    monkeypatch.setattr("mm_assist.perception.ocr.ocr_roi", fake_ocr)
    assert ni._item_list_xy(object(), "insect", "冰冷的光群") is None
    assert ni.locate_spot("gather_skill:insect", "冰冷的光群", object()) is None


def test_compute_inspect_stack_after_skew_pending():
    from mm_assist.life_plan import node_inspect as ni

    bag = SimpleNamespace(
        ui_stack_names=["EquipmentInventoryUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    out = ni.compute_inspect(
        object(),
        spot="",
        view=bag,
        stack_after=["MapUIState", "HudUIStepState"],
    )
    assert out["node"] == ""
    assert out["source"] == "pending"
    assert out["xy"] is None
    assert out.get("skew") is True


def test_compute_inspect_stack_after_same_keeps_grab_nodes(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_inventory_overlay_node", lambda _f: "")
    monkeypatch.setattr(
        ni, "refine_inspect_node", lambda _f, category=None, **_k: "facility:metal"
    )
    bag = SimpleNamespace(
        ui_stack_names=["EquipmentInventoryUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    inv = ni.compute_inspect(
        object(),
        spot="",
        view=bag,
        stack_after=["HudUIStepState", "EquipmentInventoryUIState"],
    )
    assert inv["node"] == "inventory"
    assert inv["source"] != "pending"
    assert inv.get("skew") is not True
    fac = SimpleNamespace(
        ui_stack_names=["CraftingHubAlteringFacilityState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    fine = ni.compute_inspect(
        object(),
        spot="",
        view=fac,
        stack_after=["CraftingHubAlteringFacilityState", "HudUIStepState"],
    )
    assert fine["node"] == "facility:metal"
    assert fine["source"] != "pending"


def test_compute_inspect_direct_no_stack_after_unchanged():
    from mm_assist.life_plan import node_inspect as ni

    dlg = SimpleNamespace(
        ui_stack_names=["DialogueInteractionUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    out = ni.compute_inspect(None, spot="", view=dlg)
    assert out["node"] == "dialog"
    assert out["source"] == "stack"
    assert "skew" not in out
    menu = SimpleNamespace(
        ui_stack_names=["SideNavigationUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    assert ni.compute_inspect(None, spot="", view=menu)["node"] == "system_menu"
    main = SimpleNamespace(
        ui_stack_names=["TutorialTooltipUIState"],
        ui_stack_ready=True,
    )
    assert ni.compute_inspect(None, spot="", view=main)["node"] == "main_hud"


def test_format_node_zh_profile_tabs():
    assert format_node_zh("profile") == "角色"
    assert format_node_zh("profile:info") == "角色-我的资讯"
    assert format_node_zh("profile:combat") == "角色-战斗技能"
    assert format_node_zh("life_grid") == "角色-生活技能"
    assert format_node_zh("profile:career") == "角色-职业"
    assert spots_for_node("profile:info") == ("关闭/X",)
    assert spots_for_node("profile:combat") == ("关闭/X",)
    assert spots_for_node("profile:career") == ("关闭/X",)
    assert "生活技能 Tab" not in spots_for_node("profile:info")
    assert inspect_needs_grab("profile") is True
    assert inspect_needs_grab("profile:info") is False
    assert inspect_needs_grab("life_grid") is True


def test_compute_inspect_profile_tabs_from_detect(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    view = SimpleNamespace(
        ui_stack_names=["NewProfileUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    none_frame = ni.compute_inspect(None, spot="", view=view)
    assert none_frame["node"] == "profile"
    assert format_node_zh(none_frame["node"]) == "角色"
    assert none_frame["source"] == "stack"

    monkeypatch.setattr(ni, "detect_profile_top_tab", lambda _f: "")
    monkeypatch.setattr(ni, "_life_skills_page_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    monkeypatch.setattr(ni, "_gather_detail_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_profile_non_life_tab", lambda _f: "")
    miss = ni.compute_inspect(object(), spot="", view=view)
    assert miss["node"] == "profile"
    assert format_node_zh(miss["node"]) == "角色"
    assert miss["source"] == "stack"

    monkeypatch.setattr(ni, "detect_profile_top_tab", lambda _f: "info")
    info = ni.compute_inspect(object(), spot="", view=view)
    assert info["node"] == "profile:info"
    assert info["source"] == "stack+tpl"
    monkeypatch.setattr(ni, "detect_profile_top_tab", lambda _f: "combat")
    monkeypatch.setattr(ni, "_life_skills_page_visible", lambda _f: True)
    combat = ni.compute_inspect(object(), spot="", view=view)
    assert combat["node"] == "profile:combat"
    monkeypatch.setattr(ni, "_life_skills_page_visible", lambda _f: False)
    monkeypatch.setattr(ni, "detect_profile_top_tab", lambda _f: "life")
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    life = ni.compute_inspect(object(), spot="", view=view)
    assert life["node"] == "profile"
    assert life["source"] == "stack"
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "daily")
    daily = ni.compute_inspect(object(), spot="", view=view)
    assert daily["node"] == "gather_skill:daily"
    assert format_node_zh(daily["node"]) == "角色-生活技能-日常採集"
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    monkeypatch.setattr(ni, "detect_profile_top_tab", lambda _f: "career")
    career = ni.compute_inspect(object(), spot="", view=view)
    assert career["node"] == "profile:career"


def test_compute_inspect_profile_life_from_skill_grid(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    view = SimpleNamespace(
        ui_stack_names=["NewProfileUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    monkeypatch.setattr(ni, "detect_profile_top_tab", lambda _f: "")
    monkeypatch.setattr(ni, "_life_skills_page_visible", lambda _f: True)
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    monkeypatch.setattr(ni, "_gather_detail_visible", lambda _f: False)
    hit = ni.compute_inspect(object(), spot="", view=view)
    assert hit["node"] == "life_grid"
    assert format_node_zh(hit["node"]) == "角色-生活技能"


def test_profile_grid_fast_path_skips_all_skill_searches(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    view = SimpleNamespace(ui_stack_names=["NewProfileUIInstance"], ui_stack_ready=True)
    monkeypatch.setattr(ni, "_gather_detail_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_life_grid_confirmed", lambda _f: True)
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: pytest.fail("grid must not search every skill"))
    assert ni.compute_inspect(object(), spot="", view=view)["node"] == "life_grid"
    # An open detail panel wins even if the grid remains visible behind it.
    monkeypatch.setattr(ni, "_gather_detail_visible", lambda _f: True)
    monkeypatch.setattr(ni, "_gather_detail_named", lambda _f: "gather_detail:淨淨蘑菇")
    assert ni.compute_inspect(object(), spot="", view=view)["node"] == "gather_detail:淨淨蘑菇"


def test_grid_fast_path_requires_two_independent_cells(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_tmpl", lambda f, p, **k: p.endswith("daily.png"))
    assert not ni._life_grid_confirmed(object())
    monkeypatch.setattr(ni, "_tmpl", lambda f, p, **k: p.endswith(("daily.png", "woodcut.png")))
    assert ni._life_grid_confirmed(object())


def test_real_life_grid_reaches_navigation_without_exhaustive_search(monkeypatch):
    import cv2
    from pathlib import Path
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.life_execute import ui_logic

    frame = cv2.imread(str(Path(__file__).parent / "fixtures/ui_benchmark/life_grid_real.png"))
    assert frame is not None
    view = SimpleNamespace(ui_stack_names=["UILayerHideState", "NewProfileUIInstance"], ui_stack_ready=True)
    record = ni.compute_inspect(frame, spot="", view=view)
    assert record["node"] == "life_grid"
    assert record["cost"]["tpl_n"] <= 4
    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: record["node"])
    monkeypatch.setattr(ui_logic, "read_inspect_record", lambda **k: record)
    monkeypatch.setattr(ui_logic, "read_state_ui", lambda **k: view)
    assert ui_logic.resolve_ui_node().node == "life_grid"


def test_detect_profile_tab_slots_picks_saturated():
    import numpy as np

    from mm_assist.life_plan import node_inspect as ni

    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    img[:] = (36, 36, 36)
    img[671:751, 56:136] = (200, 90, 210)
    assert ni.detect_profile_top_tab_slots(img) == "combat"
    assert ni.detect_profile_top_tab(img) == "combat"


def test_detect_profile_tab_slots_life():
    import numpy as np

    from mm_assist.life_plan import node_inspect as ni

    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    img[:] = (36, 36, 36)
    img[751:831, 56:136] = (210, 200, 40)
    assert ni.detect_profile_top_tab_slots(img) == "life"


def test_grab_client_norm_box_bitblt_roi(monkeypatch):
    import numpy as np

    from mm_assist.platform.capture import Grabber
    from mm_assist.platform.window import WindowInfo

    g = Grabber(method="bitblt")
    seen: dict[str, int] = {}

    def fake_bitblt(_win, *, ox=0, oy=0, width=None, height=None):
        seen["ox"] = ox
        seen["oy"] = oy
        seen["w"] = width
        seen["h"] = height
        return np.zeros((height, width, 3), dtype=np.uint8) + 40

    monkeypatch.setattr(g, "_bitblt_client", fake_bitblt)
    monkeypatch.setattr(g, "grab_client", lambda _w: (_ for _ in ()).throw(AssertionError("整窗")))
    win = WindowInfo(hwnd=1, title="t", left=0, top=0, width=1920, height=1440)
    patch = g.grab_client_norm_box(
        win, (0.00, 0.38, 0.10, 0.76), prefer_mss=False, fallback_full=False
    )
    assert patch is not None
    assert seen["ox"] == 0
    assert seen["oy"] == int(1440 * 0.38)
    assert seen["w"] == int(1920 * 0.10)
    assert seen["h"] == int(1440 * (0.76 - 0.38))
    assert patch.shape[1] == seen["w"]
    assert patch.shape[0] == seen["h"]
    g.close()


def test_page_ocr_latch_cht_only_no_v5_fallback(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    calls: list[dict] = []

    def fake_ocr(_frame, _roi, **kwargs):
        calls.append(dict(kwargs))
        return []

    monkeypatch.setattr("mm_assist.perception.ocr.ocr_roi", fake_ocr)
    assert ni._ocr_lines_roi(object(), (0.0, 0.0, 1.0, 1.0), fallback=True) == []
    assert len(calls) == 1
    assert calls[0].get("backend") == "cht"

    calls.clear()
    assert ni._craft_title_blob(object()) == ""
    assert len(calls) == 1
    assert calls[0].get("backend") == "cht"

    calls.clear()
    monkeypatch.setattr("mm_assist.platform.match.match_template", lambda *_a, **_k: None)
    assert ni._ocr_item_xy(object(), "daily", "煤炭") is None
    assert len(calls) == 1
    assert calls[0].get("backend") == "cht"

    calls.clear()
    assert ni._facility_item_xy(object(), "metal", "合金鋼錠") is None
    assert len(calls) == 1
    assert calls[0].get("backend") == "cht"


def test_refine_facility_stack_skips_item_info(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    n = {"n": 0}

    def count(_f, **_k):
        n["n"] += 1
        return "item_info:x"

    monkeypatch.setattr(ni, "_detect_item_info", count)
    monkeypatch.setattr(ni, "_tmpl", lambda *_a, **_k: True)
    monkeypatch.setattr(ni, "_hub_overview_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_detect_facility_fid", lambda _f: "metal")
    monkeypatch.setattr(ni, "_facility_if_claim_all", lambda _f: "")
    for sn in ("fine:facility",):
        n["n"] = 0
        assert ni.refine_inspect_node(object(), sn=sn) == "facility:metal"
        assert n["n"] == 0
    n["n"] = 0
    hub_n = ni.refine_inspect_node(object(), sn="fine:craft_hub")
    assert n["n"] == 0
    assert hub_n != "item_info:x"
    assert not str(hub_n).startswith("item_info")


def test_facility_fid_cache_and_missing_png_ocr(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.memory.readers.ui_stack import stack_compare_key

    calls = {"n": 0}

    def blob(*_a, **_k):
        calls["n"] += 1
        return "金屬加工"

    monkeypatch.setattr(ni, "_tmpl", lambda *_a, **_k: False)
    monkeypatch.setattr(ni, "_ocr_blob", blob)
    monkeypatch.setattr(
        "mm_assist.perception.process_sense._title_facility",
        lambda _b: "metal",
    )
    ni._reset_facility_fid_cache()
    ni._inspect_stack_key.set(stack_compare_key(["CraftingHubUIState", "HudUIStepState"]))
    frame = object()
    assert ni._detect_facility_fid(frame) == "metal"
    assert calls["n"] == 1
    assert ni._detect_facility_fid(frame) == "metal"
    assert calls["n"] == 1
    assert ni._detect_facility_fid(object()) == "metal"
    assert calls["n"] == 2
    view = SimpleNamespace(
        ui_stack_names=["CraftingHubUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    assert ni.inspect_needs_grab("fine:facility", view=view) is True
    assert ni.inspect_needs_grab("fine:facility") is True


def test_inspect_needs_grab_hud_only_overlay():
    from mm_assist.life_plan.node_inspect import inspect_needs_grab

    hud = SimpleNamespace(ui_stack_names=["HudUIStepState"], ui_stack_ready=True)
    assert inspect_needs_grab("overlay", view=hud) is True
    mixed = SimpleNamespace(
        ui_stack_names=["HudUIStepState", "CraftingHubUIState"],
        ui_stack_ready=True,
    )
    assert inspect_needs_grab("overlay", view=mixed) is False
    assert inspect_needs_grab("overlay") is False


def test_refine_overlay_icon_menu_is_main_hud(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        return "icon_menu_52x44.png" in str(rel).replace("\\", "/")

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    assert ni.refine_inspect_node(object(), sn="overlay") == "main_hud"


def test_hold_not_ready_keeps_hud_streak():
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    menu = {
        "node": "system_menu",
        "stack_ready": True,
        "stp": None,
        "stack_names": ["SideNavigationUIState", "HudUIStepState"],
        "source": "stack",
    }
    hud = {
        "node": "main_hud",
        "stack_ready": True,
        "stp": 0,
        "stack_names": ["TutorialTooltipUIState"],
        "source": "stack",
    }
    frozen, st1 = hold_inspect_payload(menu, hud, hud_streak=0)
    assert frozen["node"] == "system_menu"
    assert st1 == 1
    pending = {
        "node": "",
        "stack_ready": False,
        "stp": None,
        "stack_names": [],
        "source": "pending",
    }
    still, st2 = hold_inspect_payload(frozen, pending, hud_streak=st1)
    assert still["node"] == "system_menu"
    assert st2 == 1
    skew = dict(hud)
    skew["skew"] = True
    still2, st3 = hold_inspect_payload(still, skew, hud_streak=st2)
    assert still2["node"] == "system_menu"
    assert st3 == 1
    now, st4 = hold_inspect_payload(still2, hud, hud_streak=st3)
    assert now["node"] == "main_hud"
    assert st4 == 0


def test_item_info_named_grab_spots_and_hold(monkeypatch):
    import numpy as np
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.life_plan.node_inspect import hold_inspect_payload

    assert ni.inspect_needs_grab("item_info") is True
    assert ni.inspect_needs_grab("obtain_method") is True
    assert ni.inspect_needs_grab("item_info:樹液") is True
    assert ni.inspect_needs_grab("obtain_method:煤炭") is True
    assert "gather_detail" not in ni._refine_steps("item_info")
    assert "gather_detail" not in ni._refine_steps("obtain_method")
    assert ni.format_node_zh("item_info") == "物品详情"
    assert ni.format_node_zh("item_info") != "物品详情-材料"
    assert ni._pick_material_from_blob("銅礦石一堆") == "銅礦石"
    assert ni._pick_material_from_blob("淨淨蘑菇新鲜") == "淨淨蘑菇"
    assert ni._pick_material_from_blob("没有闭集名") == ""
    assert ni._pick_material_from_blob("採集藥草") == ""
    assert ni._pick_material_from_blob("採集藥草等级20以上") == ""
    assert ni.spots_for_node("item_info") == ("获得方法", "附近位置")
    assert ni.spots_for_node("item_info:煤炭") == ("获得方法", "附近位置")
    assert ni.spots_for_node("obtain_method:煤炭") == ("煤炭礦脈",)

    class _Ln:
        def __init__(self, text: str, box: list[tuple[int, int]] | None = None):
            self.text = text
            self.box = box or [(0, 0), (1, 0), (1, 1), (0, 1)]

    def unnamed_ocr(_f, roi, *, fallback=False):
        _ = fallback
        if roi == ni._ITEM_INFO_FOOT_ROI:
            return [_Ln("獲得方法"), _Ln("銅礦石")]
        if roi == ni._ITEM_INFO_HDR_ROI:
            return [_Ln("材料")]
        return []

    monkeypatch.setattr(ni, "_ocr_lines_roi", unnamed_ocr)
    assert ni._detect_item_info(object()) == "item_info"
    assert ni.format_node_zh(ni._detect_item_info(object())) == "物品详情"

    monkeypatch.setattr(
        ni,
        "_ocr_lines_roi",
        lambda _f, roi, *, fallback=False: (
            [_Ln("材料"), _Ln("獲得方法")] if roi == ni._BAG_CARD_ROI else []
        ),
    )
    assert ni._detect_bag_item_card(object()) == ""

    detail = SimpleNamespace(
        ui_stack_names=["DetailCardUIState", "EquipmentInventoryUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    howto = SimpleNamespace(
        ui_stack_names=["HowToGetSelectUIState", "EquipmentInventoryUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    refine_n = {"n": 0}

    def count_refine(*_a, **_k):
        refine_n["n"] += 1
        return "gather_detail"

    monkeypatch.setattr(ni, "refine_inspect_node", count_refine)
    dummy = ni.compute_inspect(object(), spot="", view=detail)
    assert dummy["node"] == "item_info"
    assert refine_n["n"] == 0
    assert ni.compute_inspect(object(), spot="", view=howto)["node"] == "obtain_method"
    assert refine_n["n"] == 0

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    monkeypatch.setattr(ni, "_detect_item_info", lambda _f: "item_info:樹液")
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "obtain_method:樹液")
    named = ni.compute_inspect(frame, spot="", view=detail)
    assert named["node"] == "item_info:樹液"
    assert refine_n["n"] == 0
    om = ni.compute_inspect(frame, spot="", view=howto)
    assert om["node"] == "obtain_method:樹液"
    assert refine_n["n"] == 0

    assert ni.mem_should_apply_stack("item_info", "item_info:樹液", inflight=False) is False
    assert ni.mem_should_apply_stack("obtain_method", "obtain_method:樹液", inflight=False) is False
    assert ni.mem_should_apply_stack("item_info", "", inflight=False) is True

    prev = {"node": "item_info:樹液", "stack_ready": True, "stack_names": []}
    coarse = {"node": "item_info", "stack_ready": True, "stack_names": ["DetailCardUIState"]}
    kept, _ = hold_inspect_payload(prev, coarse)
    assert kept["node"] == "item_info:樹液"
    coal = {"node": "item_info:煤炭", "stack_ready": True, "stack_names": ["DetailCardUIState"]}
    updated, _ = hold_inspect_payload(kept, coal)
    assert updated["node"] == "item_info:煤炭"
    howto = {
        "node": "obtain_method",
        "stack_ready": True,
        "stack_names": ["HowToGetSelectUIState"],
    }
    carried, _ = hold_inspect_payload(prev, howto)
    assert carried["node"] == "obtain_method:樹液"
    assert carried["stack_names"] == ["HowToGetSelectUIState"]
    back, _ = hold_inspect_payload(
        carried,
        {"node": "item_info", "stack_ready": True, "stack_names": ["DetailCardUIState"]},
    )
    assert back["node"] == "item_info:樹液"

    hits = [_Ln("尋找附近位置", [(40, 80), (200, 80), (200, 120), (40, 120)])]
    monkeypatch.setattr(ni, "_ocr_lines_roi", lambda *_a, **_k: hits)
    assert ni.locate_spot("item_info:煤炭", "附近位置", object()) == (120, 100)
    assert ni.locate_spot("item_info", "获得方法", object()) is None
    monkeypatch.setattr(ni, "_ocr_lines_roi", lambda *_a, **_k: [])
    assert ni.locate_spot("item_info:煤炭", "附近位置", object()) is None


def test_locate_item_info_obtain_on_coarse_node(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    class _Ln:
        def __init__(self, text: str, box: list[tuple[int, int]]):
            self.text = text
            self.box = box

    lines = [_Ln("獲得方法", [(10, 20), (90, 20), (90, 50), (10, 50)])]
    monkeypatch.setattr(ni, "_ocr_lines_roi", lambda *_a, **_k: lines)
    assert ni.locate_spot("item_info", "获得方法", object()) == (50, 35)


def test_craft_stack_layers_refine_route():
    from mm_assist.life_plan import node_inspect as ni

    hub = ni._refine_steps("fine:craft_hub")
    assert "facility" not in hub
    assert "claim_all" not in hub
    assert "craft_shell" in hub
    rec = ni._refine_steps("fine:facility_recipe")
    assert "craft_shell" not in rec
    assert "process_hub" not in rec
    assert "facility_recipe" in rec
    assert ni._hub_stack_sn("fine:facility") is False
    assert ni._hub_stack_sn("fine:craft_hub") is True
    assert ni._fine_fallback("fine:facility_recipe") == "facility_recipe"
    assert ni.format_node_zh("fine:facility_recipe") == "制作-加工-加工详情"
    assert ni.inspect_page_zh("fine:facility_recipe", stack_ready=True) == "制作-加工-加工详情"
    assert ni.inspect_page_zh("fine:facility_recipe", stack_ready=True) != "制作"
    assert ni.format_node_zh("fine:facility_detail") == "制作-加工-加工详情"
    assert ni.mem_should_apply_stack("fine:facility", "facility:metal", inflight=False) is False
    view = SimpleNamespace(
        ui_stack_names=["AlteringFacilityPopupUIInstance", "HudUIStepState"],
        ui_stack_ready=True,
    )
    out = ni.compute_inspect(None, spot="", view=view)
    assert out["node"] == "facility"
    assert out["source"] == "stack"


def test_craft_hub_refine_skips_facility_back(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    seen: list[str] = []
    hub_n = {"n": 0}

    def fake_tmpl(_frame, rel, *, threshold=0.70, roi=None):
        _ = threshold, roi
        seen.append(str(rel).replace("\\", "/"))
        return False

    monkeypatch.setattr(ni, "_tmpl", fake_tmpl)
    monkeypatch.setattr(ni, "_hub_overview_visible", lambda _f: hub_n.__setitem__("n", hub_n["n"] + 1) or False)
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "craft_quick")
    monkeypatch.setattr(ni, "_process_hub_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_facility_if_claim_all", lambda _f: "")
    ni.refine_inspect_node(object(), sn="fine:craft_hub")
    assert not any("facility/back.png" in p for p in seen)
    hub_n["n"] = 0
    monkeypatch.setattr(ni, "_detect_facility_fid", lambda _f: "metal")
    ni.refine_inspect_node(object(), sn="fine:facility")
    assert hub_n["n"] == 0


def test_refine_facility_detail_sap(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_detect_item_info", lambda _f, **k: "item_info:樹液")
    assert ni.refine_inspect_node(object(), sn="fine:facility_detail") == "item_info:樹液"
    assert ni.inspect_needs_grab("fine:facility_detail") is True
    assert ni.mem_should_apply_stack(
        "fine:facility_detail", "item_info:樹液", inflight=False
    ) is False


def test_refine_facility_detail_falls_to_recipe(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(ni, "_detect_item_info", lambda _f, **k: "")
    monkeypatch.setattr(ni, "_facility_kind_node", lambda _f, **k: "facility:wood")
    monkeypatch.setattr(ni, "_detect_facility_recipe", lambda *_a, **_k: "木材+")
    monkeypatch.setattr(ni, "_hub_overview_visible", lambda _f: False)
    assert ni.refine_inspect_node(object(), sn="fine:facility_detail") == "facility:wood:木材+"
    monkeypatch.setattr(ni, "_detect_item_info", lambda _f, **k: "item_info:高級木材")
    assert ni.refine_inspect_node(object(), sn="fine:facility_detail") == "facility:wood:木材+"


def test_transition_frame_is_not_life_grid():
    import cv2
    from pathlib import Path

    from mm_assist.life_plan import node_inspect as ni

    frame = cv2.imread(str(Path(__file__).parent / "fixtures/ui_benchmark/system_menu_transition.png"))
    assert frame is not None
    view = SimpleNamespace(
        ui_stack_names=["NewProfileUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    out = ni.compute_inspect(frame, spot="", view=view)
    assert out["node"] != "life_grid"
    assert out["node"] in ("profile", "profile:info", "profile:combat", "profile:career")
    assert ni.refine_inspect_node(frame, sn="") != "life_grid"
    assert ni.refine_inspect_node(frame, sn="fine:life_grid") != "life_grid"
    assert ni._profile_non_life_tab(frame) == ""
    assert ni.detect_profile_top_tab(frame) == "life"


def test_profile_info_fixture_short_circuits_gather_scan(monkeypatch):
    import cv2
    from pathlib import Path

    from mm_assist.life_plan import node_inspect as ni

    frame = cv2.imread(str(Path(__file__).parent / "fixtures/ui_benchmark/profile_info_real.png"))
    assert frame is not None
    monkeypatch.setattr(
        ni,
        "_detect_gather_skill_sid",
        lambda _f: pytest.fail("info tab must not scan gather items"),
    )
    view = SimpleNamespace(
        ui_stack_names=["NewProfileUIState", "HudUIStepState"],
        ui_stack_ready=True,
    )
    out = ni.compute_inspect(frame, spot="", view=view)
    assert out["node"] == "profile:info"
    assert out["source"] == "stack+tpl"
    assert out["cost"]["tpl_n"] <= 8
    assert ni._profile_non_life_tab(frame) == "info"


def test_life_tab_template_alone_is_not_life_grid(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni

    monkeypatch.setattr(
        ni,
        "_tmpl",
        lambda _f, rel, **k: "profile/life_tab.png" in str(rel).replace("\\", "/"),
    )
    monkeypatch.setattr(ni, "_gather_detail_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_detect_gather_skill_sid", lambda _f: "")
    monkeypatch.setattr(ni, "_system_menu_visible", lambda _f: False)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.currency_wallet_visible",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(ni, "_craft_shell_node", lambda _f, category=None: "")
    monkeypatch.setattr(ni, "_process_hub_visible", lambda _f: False)
    monkeypatch.setattr(ni, "_detect_obtain_method", lambda _f: "")
    assert ni.refine_inspect_node(object(), sn="") != "life_grid"
    assert ni.refine_inspect_node(object(), sn="fine:life_grid") != "life_grid"
