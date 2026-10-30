from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mm_assist.knowledge_paths import ui_config
from mm_assist.memory.readers.ui_stack import (
    Il2CppUiStackReader,
    _mmlist_count,
    format_stack_line,
    label_for,
    left_main_hud,
    pick_stack_addrs,
    stack_compare_key,
    stack_dlg,
    stack_hud_only,
    stack_inv,
    stack_kind,
    stack_left_overlay,
    stack_menu,
    stack_node,
    stack_common_popup,
)


def test_config_labels_zh():
    labels = json.loads(ui_config("il2cpp_ui_stack.json").read_text(encoding="utf-8"))["labels"]
    assert labels["CashShopUIState"] == "商城"
    assert labels["EventUIState"] == "活动"
    assert labels["ToyHudZeroLayerUIState"] == "公告事项"
    assert labels["PostBoxUIState"] == "信箱"
    assert labels["LivingSkillUIState"] == "生活技能详情"
    assert labels["LivingSkillUIInstance"] == "生活技能详情"
    assert labels["DetailCardUIState"] == "物品详情"
    assert labels["DetailCardUIInstance"] == "物品详情"
    assert labels["MapUIState"] == "地图"
    assert labels["CraftingHubUIState"] == "制作"
    assert labels["CraftingHubUIInstance"] == "制作壳"
    assert labels["CraftingHubPopupUIInstance"] == "制作壳"
    assert labels["AlteringFacilityPopupUIInstance"] == "加工设施"
    assert labels["AlteringPanelPopupUIInstance"] == "加工配方"
    assert labels["CurrencyUIState"] == "持有的货币"
    assert labels["NewDailyMissionUIState"] == "挑战"
    assert labels["SideNavigationUIState"] == "菜单"
    assert labels["SideNavigationUIInstance"] == "菜单"
    assert labels["EquipmentInventoryUIInstance"] == "背包"
    assert labels["MapLocationPanelUIState"] == "地图标记"
    assert labels["HowToGetSelectUIState"] == "获得方法"
    assert labels["SkipButtonUIState"] == "对话-跳过对话"
    assert labels["ButtonCommonPopupInstance"] == "确认弹窗"
    assert labels["ButtonCommonPopupUIState"] == "确认弹窗"


def test_mmlist_count_system_list_and_mmlist(monkeypatch):
    from mm_assist.memory.readers import ui_stack as mod

    off = {"MMList._size": 0x18, "MMList.size": 0x1C}

    def ru32(_h, addr):
        return {0x118: 1365, 0x11C: 2, 0x218: 3, 0x21C: 9}[addr]

    monkeypatch.setattr(mod, "read_u32", ru32)
    assert _mmlist_count(None, 0x100, off, 8) == 2
    assert _mmlist_count(None, 0x200, off, 8) == 3


def test_format_stack_line_empty_and_unknown():
    assert format_stack_line([]) == "主界面"
    assert format_stack_line(["UnknownThingUIState"], {}) == "UnknownThingUIState"
    assert "names=" not in format_stack_line(["LivingSkillUIState"], {"LivingSkillUIState": "生活技能详情"})
    assert (
        format_stack_line(
            [
                "DialogueInteractionUIState",
                "SkipCircleUIState",
                "SkipButtonUIState",
                "HudUIStepState",
                "DialoguePlayerComponent",
            ],
            {"DialoguePlayerComponent": "对话"},
        )
        == "对话 > HudUIStepState > SkipButtonUIState > SkipCircleUIState > DialogueInteractionUIState"
    )


def test_stack_common_popup_names():
    names = [
        "HudUIStepState",
        "CraftingHubUIState",
        "ButtonCommonPopupUIState",
        "ButtonCommonPopupInstance",
    ]
    assert stack_common_popup(names) is True
    assert stack_common_popup(["AlteringFacilityPopupUIInstance", "HudUIStepState"]) is False
    labs = {"ButtonCommonPopupInstance": "确认弹窗", "ButtonCommonPopupUIState": "确认弹窗"}
    assert format_stack_line(names, labs).startswith("确认弹窗")


def test_stack_top_left_and_labels():
    labels = {
        "CraftingHubUIInstance": "制作",
        "CraftingHubAlteringFacilityState": "加工六卡",
        "LivingSkillUIState": "生活技能详情",
    }
    line = format_stack_line(
        ["CraftingHubUIInstance", "CraftingHubAlteringFacilityState"],
        labels,
    )
    assert line == "加工六卡 > 制作"
    assert format_stack_line(["LivingSkillUIState"], labels) == "生活技能详情"
    assert format_stack_line(["CraftingHubUIState"], labels) == "CraftingHubUIState"


def test_label_keeps_full_class_name():
    assert label_for("FooBarUIInstance", {}) == "FooBarUIInstance"
    assert label_for("CraftingHubAlteringFacilityState", {"CraftingHubAlteringFacilityState": "加工六卡"}) == "加工六卡"






def test_read_ready_from_players_without_state_stacks(monkeypatch):
    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = []
    r._labels = {}
    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(
        r,
        "_players_for_read",
        lambda _state_names, force=False: (
            ["SideNavigationUIState", "HudUIStepState"],
            0x11,
        ),
    )
    snap = r.read()
    assert snap.ready is True
    assert snap.error == ""
    assert snap.names == ["SideNavigationUIState", "HudUIStepState"]
    assert stack_node(snap.names, ready=snap.ready) == "system_menu"


def test_read_pending_without_stacks_or_player_names(monkeypatch):
    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = []
    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(r, "_players_for_read", lambda _state_names, force=False: ([], 0))
    snap = r.read()
    assert snap.ready is False
    assert snap.names == []
    assert snap.error == "stack_scan_pending"
    assert stack_node(snap.names, ready=snap.ready) == ""


def test_read_unready_when_stacks_but_empty_names(monkeypatch):
    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = [1]
    r._scan_done_n = 1
    r._scan_busy = False
    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(r, "_names_from_state_lists", lambda: [])
    monkeypatch.setattr(r, "_players_for_read", lambda _state_names, force=False: ([], 0))
    snap = r.read()
    assert snap.ready is False
    assert snap.names == []
    assert snap.error == "stack_names_empty"
    assert stack_node(snap.names, ready=snap.ready) == ""


def test_read_hot_path_caches_line(monkeypatch):
    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = [1]
    r._labels = {}
    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(r, "_names_from_state_lists", lambda: ["CraftingHubUIState"])
    monkeypatch.setattr(r, "_names_from_players", lambda: ([], 0))
    snap = r.read()
    assert snap.line == "CraftingHubUIState"
    assert r.read() is snap


def test_ensure_reopens_on_pid_change(monkeypatch):
    closed: list[int] = []
    r = Il2CppUiStackReader()
    r._h = 99
    r._ga = 1
    r._pid = 111
    r._state_stacks = [1, 2]
    monkeypatch.setattr("mm_assist.memory.readers.ui_stack.find_game_pid", lambda: 222)
    monkeypatch.setattr(
        "mm_assist.memory.readers.ui_stack.close_handle", lambda h: closed.append(h)
    )
    monkeypatch.setattr("mm_assist.memory.readers.ui_stack.open_process", lambda pid: 1000)
    monkeypatch.setattr(
        "mm_assist.memory.readers.ui_stack.module_base", lambda pid, _m: 0x400000
    )
    assert r._ensure() is True
    assert closed == [99]
    assert r._pid == 222
    assert r._h == 1000
    assert r._ga == 0x400000
    assert r._state_stacks == []


def test_ensure_state_stacks_does_not_scan_on_tick(monkeypatch):
    from mm_assist.memory.readers import ui_stack as mod

    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._next_stack_scan = 0.0
    scanned = {"n": 0}

    def fake_scan(*_a, **_k):
        scanned["n"] += 1
        return []

    started: list[str] = []

    class FakeThread:
        def __init__(self, target=None, args=(), name="", daemon=False):
            started.append(name)

        def start(self) -> None:
            return None

    monkeypatch.setattr(mod, "scan_klass_objects", fake_scan)
    monkeypatch.setattr(mod, "resolve_klass", lambda *_a, **_k: 0xABC)
    monkeypatch.setattr(mod, "read_ptr", lambda *_a, **_k: 0)
    monkeypatch.setattr(mod.threading, "Thread", FakeThread)
    r._ensure_state_stacks()
    assert scanned["n"] == 0
    assert started == ["ui-stack-scan"]
    assert r._scan_busy is True


def test_pick_stack_addrs_prefers_leave_layer():
    assert pick_stack_addrs([((0, 1, 8), 10), ((1, 2, 3), 20)], cap=1) == [20]
    assert pick_stack_addrs([(1, 10), (5, 20), (3, 30)], cap=2) == [20, 30]
    assert pick_stack_addrs([(1, 0), (2, 8)]) == [8]
    bag = ((1, 4, 4), 99)
    bits = ((1, 1, 1), 10), ((1, 1, 1), 20), ((0, 1, 1), 30), ((1, 1, 1), 40)
    picked = pick_stack_addrs([*bits, bag], cap=12)
    assert picked[0] == 99
    assert 40 in picked


def test_ensure_state_stacks_rescans_short_list(monkeypatch):
    import time

    from mm_assist.memory.readers import ui_stack as mod

    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._state_stacks = [0x50]
    r._next_stack_scan = time.perf_counter() + 30
    started: list[str] = []

    class FakeThread:
        def __init__(self, target=None, args=(), name="", daemon=False):
            started.append(name)

        def start(self) -> None:
            return None

    monkeypatch.setattr(mod, "resolve_klass", lambda *_a, **_k: 0xABC)
    monkeypatch.setattr(mod, "read_ptr", lambda *_a, **_k: 0xABC)
    monkeypatch.setattr(r, "_stack_list_size", lambda *_a, **_k: 1)
    monkeypatch.setattr(mod.threading, "Thread", FakeThread)
    r._ensure_state_stacks()
    # size==1 不再每拍狂扫；节流期内与 size>=2 一样跳过。
    assert started == []


def test_ensure_state_stacks_skips_when_list_ok(monkeypatch):
    import time

    from mm_assist.memory.readers import ui_stack as mod

    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._state_stacks = [0x50]
    r._next_stack_scan = time.perf_counter() + 30
    started: list[str] = []

    class FakeThread:
        def __init__(self, target=None, args=(), name="", daemon=False):
            started.append(name)

        def start(self) -> None:
            return None

    monkeypatch.setattr(mod, "resolve_klass", lambda *_a, **_k: 0xABC)
    monkeypatch.setattr(mod, "read_ptr", lambda *_a, **_k: 0xABC)
    monkeypatch.setattr(r, "_stack_list_size", lambda *_a, **_k: 3)
    monkeypatch.setattr(mod.threading, "Thread", FakeThread)
    r._ensure_state_stacks()
    assert started == []


def test_stack_bits_match_2026_09_04_capture():
    main = ["TutorialTooltipUIState"]
    menu = ["TutorialTooltipUIState", "SideNavigationUIState", "HudUIStepState"]
    bag = [
        "TutorialTooltipUIState",
        "EquipmentInventoryUIState",
        "HudUIStepState",
        "UILayerHideState",
    ]
    dlg = ["TutorialTooltipUIState", "DialogueInteractionUIState", "HudUIStepState"]
    hub = ["TutorialTooltipUIState", "CraftingHubUIState", "HudUIStepState"]
    assert stack_kind(["MapLocationPanelUIState", "HudUIStepState"], ready=True) == "map"
    assert stack_kind(["MapUIState", "HudUIStepState"], ready=True) == "map"
    assert left_main_hud(main) is False
    assert left_main_hud(menu) is True
    assert left_main_hud(["TutorialTooltipUIState", "SideNavigationUIState"]) is True
    assert stack_menu(menu) is True
    assert stack_inv(bag) is True
    assert stack_dlg(dlg) is True
    assert "HudUIStepState" not in stack_compare_key(hub)
    assert stack_compare_key(menu) != stack_compare_key(hub)


@pytest.mark.parametrize(
    "names,ready,want",
    [
        (["TutorialTooltipUIState"], True, "main_hud"),
        (
            [
                "TutorialTooltipUIState",
                "EquipmentInventoryUIState",
                "HudUIStepState",
            ],
            True,
            "inventory",
        ),
        (
            ["TutorialTooltipUIState", "SideNavigationUIState", "HudUIStepState"],
            True,
            "system_menu",
        ),
        (["CraftingHubUIState", "HudUIStepState"], True, "fine:craft_hub"),
        (["CashShopUIState", "HudUIStepState"], True, "stack:CashShopUIState"),
        (["CashShopUIInstance"], True, "stack:CashShopUIState"),
        (["DungeonListUIInstance", "DungeonEntryUIInstance"], True, "stack:DungeonListUIInstance"),
        (["TutorialTooltipUIState", "SideNavigationUIState"], True, "system_menu"),
        (["TutorialTooltipUIState", "EquipmentInventoryUIState"], True, "inventory"),
        (
            [
                "HudUIStepState",
                "SideNavigationUIState",
                "EquipmentInventoryUIInstance",
            ],
            True,
            "inventory",
        ),
        (["HudUIStepState"], True, "overlay"),
        (["TutorialTooltipUIState"], False, ""),
        ([], False, ""),
        ([], True, ""),
    ],
)
def test_stack_node_parametrized(names, ready, want):
    assert stack_node(names, ready=ready) == want


def test_stack_node_fine_over_inventory():
    skip_dlg = [
        "SkipButtonUIState",
        "DialogueInteractionUIState",
        "HudUIStepState",
    ]
    assert stack_node(skip_dlg, ready=True) == "dialog_skip"
    detail_bag = [
        "DetailCardUIState",
        "EquipmentInventoryUIState",
        "HudUIStepState",
    ]
    assert stack_node(detail_bag, ready=True) == "item_info"
    detail_bag_instance = [
        "EquipmentInventoryUIInstance",
        "SizeAdjustmentSender",
        "DetailCardUIInstance",
    ]
    assert stack_node(detail_bag_instance, ready=True) == "item_info"
    assert stack_left_overlay(detail_bag_instance) is True
    assert stack_node(["EquipmentInventoryUIInstance"], ready=True) == "inventory"
    howto_bag = [
        "HowToGetSelectUIState",
        "EquipmentInventoryUIState",
        "HudUIStepState",
    ]
    assert stack_node(howto_bag, ready=True) == "obtain_method"
    howto_craft = [
        "HudUIStepState",
        "CraftingHubPopupUIInstance",
        "AlteringFacilityPopupUIInstance",
        "AlteringPanelPopupUIInstance",
        "DetailCardUIInstance",
        "HowToGetSelectUIInstance",
    ]
    assert stack_node(howto_craft, ready=True) == "obtain_method"


def test_stack_node_map_life_craft_tabs():
    assert stack_node(["MapUIState", "HudUIStepState"], ready=True) == "map"
    both = [
        "NewProfileUIState",
        "LivingSkillUIState",
        "HudUIStepState",
    ]
    assert stack_node(both, ready=True) == "life_grid"
    skill_detail = [
        "NewProfileUIInstance",
        "LivingSkillUIInstance",
        "SizeAdjustmentSender",
        "DetailCardUIInstance",
    ]
    assert stack_node(skill_detail, ready=True) == "life_grid"
    assert stack_node(["NewProfileUIState", "HudUIStepState"], ready=True) == "profile"
    assert (
        stack_node(
            ["CraftingHubUIState", "HudUIStepState"],
            ready=True,
            craft_category=3,
        )
        == "fine:craft_hub"
    )
    assert (
        stack_node(
            ["CraftingHubUIState", "HudUIStepState"],
            ready=True,
            craft_category=0,
        )
        == "fine:craft_hub"
    )
    assert (
        stack_node(
            [
                "CraftingHubAlteringFocusState",
                "CraftingHubUIState",
                "HudUIStepState",
            ],
            ready=True,
        )
        == "item_info"
    )
    assert (
        stack_node(
            [
                "CraftingHubCraftingFocusState",
                "CraftingHubUIState",
                "HudUIStepState",
            ],
            ready=True,
        )
        == "item_info"
    )
    assert (
        stack_node(
            [
                "CraftingHubAlteringSelectState",
                "CraftingHubUIState",
                "HudUIStepState",
            ],
            ready=True,
        )
        == "fine:craft_hub"
    )


def test_reuse_stack_snap_expires_after_limit():
    from types import SimpleNamespace

    from mm_assist.perception.scene import (
        STACK_REUSE_MAX_S,
        STACK_REUSE_MAX_TICKS,
        SceneMonitor,
        apply_stack_bits,
    )
    from mm_assist.memory.readers.scene import MemSceneInfo

    mon = object.__new__(SceneMonitor)
    mon.last_ms = SimpleNamespace(
        ui_stack_names=("HudUIStepState",),
        ui_stack_ready=True,
    )
    mon.last = SimpleNamespace(ui_stack="菜单")
    mon._stack_reuse_started = None
    mon._stack_reuse_n = 0
    t0 = 1000.0
    first = mon._reuse_stack_snap(now=t0)
    assert first is not None
    assert first.error == "stack_read_reused"
    assert first.ready is True
    second = mon._reuse_stack_snap(now=t0 + 0.2)
    assert second.error == "stack_read_reused"
    assert STACK_REUSE_MAX_TICKS == 2
    expired_tick = mon._reuse_stack_snap(now=t0 + 0.4)
    assert expired_tick.error == "stack_reuse_expired"
    assert expired_tick.ready is False
    assert list(expired_tick.names) == []

    mon._stack_reuse_started = None
    mon._stack_reuse_n = 0
    mon._reuse_stack_snap(now=t0)
    expired_time = mon._reuse_stack_snap(now=t0 + STACK_REUSE_MAX_S + 0.01)
    assert expired_time.error == "stack_reuse_expired"
    assert expired_time.ready is False

    ms = MemSceneInfo()
    apply_stack_bits(ms, expired_time)
    assert ms.ui_stack_ready is False
    assert ms.ui_stack_error == "stack_reuse_expired"


def test_build_state_ui_payload_ui_stack_error():
    from types import SimpleNamespace

    from mm_assist.memory.readers.scene import MemSceneInfo
    from mm_assist.perception.state_ui import build_state_ui_payload

    snap = SimpleNamespace(
        ui="",
        ui_detail="",
        ui_source="",
        ui_params="stp=0",
        ui_stack="—",
        ui_relation="",
        geo="",
        mem_status="ok",
    )
    reused = MemSceneInfo(ui_stack_ready=True, ui_stack_error="stack_read_reused")
    payload = build_state_ui_payload(snap, reused)
    assert payload["ui_stack_error"] == "stack_read_reused"
    pending = MemSceneInfo(ui_stack_ready=False, ui_stack_error="stack_scan_pending")
    assert build_state_ui_payload(snap, pending)["ui_stack_error"] == "stack_scan_pending"
    ok = MemSceneInfo(ui_stack_ready=True)
    assert "ui_stack_error" not in build_state_ui_payload(snap, ok)
    assert "ui_stack_ts" not in build_state_ui_payload(snap, ok)
    stamped = MemSceneInfo(ui_stack_ready=True, ui_stack_ts=12.5)
    assert build_state_ui_payload(snap, stamped)["ui_stack_ts"] == 12.5
    assert payload["ui_stack_empty_ok"] is False
    confirmed = MemSceneInfo(ui_stack_ready=True, ui_stack_empty_ok=True)
    assert build_state_ui_payload(snap, confirmed)["ui_stack_empty_ok"] is True
    assert build_state_ui_payload(snap, ok)["ui_stack_empty_ok"] is False


def test_apply_stack_bits_copies_sampled_ts():
    from mm_assist.memory.readers.scene import MemSceneInfo
    from mm_assist.memory.readers.ui_stack import UiStackSnap
    from mm_assist.perception.scene import apply_stack_bits

    ms = MemSceneInfo()
    apply_stack_bits(
        ms,
        UiStackSnap(
            names=["HudUIStepState"],
            line="主界面",
            n=1,
            ready=True,
            sampled_ts=42.25,
        ),
    )
    assert ms.ui_stack_ts == 42.25


def test_line_cache_keeps_sampled_ts(monkeypatch):
    import threading

    from mm_assist.memory.readers import ui_stack as us

    reader = object.__new__(us.Il2CppUiStackReader)
    reader._h = 1
    reader._pid = 1
    reader._ga = 1
    reader._lock = threading.Lock()
    reader._state_stacks = [1]
    reader._labels = {}
    reader.last_error = None
    cached = us.UiStackSnap(
        names=["HudUIStepState"],
        line="主界面",
        n=1,
        ready=True,
        sampled_ts=12.5,
    )
    reader._last_snap = cached
    reader._last_read_ts = 100.0
    monkeypatch.setattr(reader, "_ensure", lambda: True)
    monkeypatch.setattr(us.time, "perf_counter", lambda: 100.15)
    monkeypatch.setattr(us.time, "time", lambda: 99.0)
    out = reader.read()
    assert out is cached
    assert out.sampled_ts == 12.5


def test_line_cache_miss_assigns_new_sampled_ts(monkeypatch):
    import threading

    from mm_assist.memory.readers import ui_stack as us

    assert us._LINE_CACHE_S == 0.2
    assert us._SCAN_OK_S <= 1.5
    assert us._SCAN_WALL_S <= 2.0
    assert us._PLAYERS_THROTTLE_S <= 0.4
    reader = object.__new__(us.Il2CppUiStackReader)
    reader._h = 1
    reader._pid = 1
    reader._ga = 1
    reader._lock = threading.Lock()
    reader._state_stacks = [1]
    reader._labels = {}
    reader.last_error = None
    reader._last_snap = us.UiStackSnap(
        names=["HudUIStepState"],
        line="主界面",
        n=1,
        ready=True,
        sampled_ts=1.0,
    )
    reader._last_read_ts = 0.0
    wall = {"t": 11.0}
    monkeypatch.setattr(reader, "_ensure", lambda: True)
    monkeypatch.setattr(reader, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(
        reader, "_names_from_state_lists", lambda: (["HudUIStepState"], None)
    )
    monkeypatch.setattr(reader, "_players_for_read", lambda *a, **k: ([], 0))
    monkeypatch.setattr(us.time, "perf_counter", lambda: 1.0)
    monkeypatch.setattr(us.time, "time", lambda: wall["t"])
    first = reader.read()
    assert first.sampled_ts == 11.0
    assert first is reader._last_snap

    wall["t"] = 99.0
    monkeypatch.setattr(us.time, "perf_counter", lambda: 1.05)
    hit = reader.read()
    assert hit.sampled_ts == 11.0
    assert hit is first

    wall["t"] = 12.0
    monkeypatch.setattr(us.time, "perf_counter", lambda: 1.5)
    fresh = reader.read()
    assert fresh.sampled_ts == 12.0
    assert fresh.sampled_ts != first.sampled_ts


def test_stack_hud_only():
    assert stack_hud_only(["HudUIStepState"]) is True
    assert stack_hud_only(["HudUIStepState", "TutorialTooltipUIState"]) is True
    assert stack_hud_only(["HudUIStepState", "CraftingHubUIState"]) is False
    assert stack_hud_only([]) is False


def test_stack_left_overlay_instance_not_global():
    hud_tip = ["HudUIStepState", "TutorialTooltipUIState"]
    assert stack_hud_only(hud_tip) is True
    assert stack_left_overlay(hud_tip) is True
    assert stack_left_overlay(["TutorialTooltipUIState"]) is False
    assert stack_left_overlay(["DetailCardUIInstance"]) is False
    assert stack_left_overlay(["DetailCardUIState"]) is True
    bag_instance = ["EquipmentInventoryUIInstance", "DetailCardUIInstance"]
    assert stack_left_overlay(bag_instance) is True
    craft_instance = [
        "CraftingHubPopupUIInstance",
        "AlteringFacilityPopupUIInstance",
        "AlteringPanelPopupUIInstance",
        "DetailCardUIInstance",
    ]
    assert stack_left_overlay(craft_instance) is True
    assert stack_left_overlay(
        [
            "CraftingHubPopupUIInstance",
            "AlteringFacilityPopupUIInstance",
            "AlteringPanelPopupUIInstance",
        ]
    ) is True
    assert stack_left_overlay(["DungeonListUIInstance", "DungeonEntryUIInstance"]) is True


def test_craft_popup_stack_layers():
    hub = ["CraftingHubPopupUIInstance", "HudUIStepState"]
    fac = [
        "AlteringFacilityPopupUIInstance",
        "CraftingHubPopupUIInstance",
        "HudUIStepState",
    ]
    panel = [
        "AlteringPanelPopupUIInstance",
        "AlteringFacilityPopupUIInstance",
        "CraftingHubPopupUIInstance",
        "HudUIStepState",
    ]
    assert stack_node(hub, ready=True) == "fine:craft_hub"
    assert stack_node(fac, ready=True) == "fine:facility"
    assert stack_node(panel, ready=True) == "fine:facility_recipe"
    panel_detail = [
        "AlteringPanelPopupUIInstance",
        "AlteringFacilityPopupUIInstance",
        "CraftingHubPopupUIInstance",
        "DetailCardUIInstance",
        "HudUIStepState",
    ]
    assert stack_node(panel_detail, ready=True) == "fine:facility_detail"
    assert stack_kind(panel_detail, ready=True) == "craft_hub"
    assert (
        stack_node(
            [
                "CraftingHubAlteringFocusState",
                "AlteringPanelPopupUIInstance",
                "HudUIStepState",
            ],
            ready=True,
        )
        == "item_info"
    )
    only_fac = ["AlteringFacilityPopupUIInstance"]
    assert stack_left_overlay(only_fac) is True
    assert stack_kind(only_fac, ready=True) == "craft_hub"
    assert stack_kind(hub, ready=True) == "craft_hub"
    assert stack_kind(fac, ready=True) == "craft_hub"
    assert stack_kind(panel, ready=True) == "craft_hub"
    assert (
        stack_node(
            ["CraftingHubAlteringFacilityPanelState", "HudUIStepState"],
            ready=True,
        )
        == "fine:facility"
    )


def test_stack_facility_detail_vs_bag_and_life():
    craft = [
        "CraftingHubPopupUIInstance",
        "AlteringFacilityPopupUIInstance",
        "AlteringPanelPopupUIInstance",
        "DetailCardUIInstance",
    ]
    assert stack_node(craft, ready=True) == "fine:facility_detail"
    assert stack_node(craft, ready=True) != "fine:facility_recipe"
    assert stack_node(craft, ready=True) != "item_info"
    no_card = [
        "CraftingHubPopupUIInstance",
        "AlteringFacilityPopupUIInstance",
        "AlteringPanelPopupUIInstance",
    ]
    assert stack_node(no_card, ready=True) == "fine:facility_recipe"
    bag = [
        "EquipmentInventoryUIInstance",
        "DetailCardUIInstance",
    ]
    assert stack_node(bag, ready=True) == "item_info"
    skill = [
        "NewProfileUIInstance",
        "LivingSkillUIInstance",
        "SizeAdjustmentSender",
        "DetailCardUIInstance",
    ]
    assert stack_node(skill, ready=True) == "life_grid"


def test_craft_panel_state_with_hub_shell_is_hub():
    assert (
        stack_node(
            [
                "CraftingHubAlteringFacilityPanelState",
                "CraftingHubPopupUIInstance",
                "HudUIStepState",
            ],
            ready=True,
        )
        == "fine:craft_hub"
    )
    assert (
        stack_node(
            ["CraftingHubAlteringFacilityPanelState", "HudUIStepState"],
            ready=True,
        )
        == "fine:facility"
    )


def test_read_scan_busy_or_never_scanned_pending(monkeypatch):
    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = []
    r._empty_stacks = []
    r._scan_done_n = 0
    r._scan_busy = True
    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(r, "_players_for_read", lambda _state_names, force=False: ([], 0))
    snap = r.read()
    assert snap.ready is False
    assert snap.empty_confirmed is False
    assert snap.error == "stack_scan_pending"
    r._scan_busy = False
    r._scan_done_n = 0
    snap2 = r.read()
    assert snap2.ready is False
    assert snap2.error == "stack_scan_pending"


def test_force_read_scan_busy_does_not_reuse_old_ready_page(monkeypatch):
    from mm_assist.memory.readers.ui_stack import UiStackSnap

    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = []
    r._empty_stacks = []
    r._scan_done_n = 0
    r._scan_busy = True
    r._last_snap = UiStackSnap(
        names=["HudUIStepState"],
        line="主界面",
        n=1,
        ready=True,
        sampled_ts=1.0,
    )
    r._last_read_ts = 0.0
    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(r, "_players_for_read", lambda _state_names, force=False: ([], 0))
    snap = r.read(force=True)
    assert snap.ready is False
    assert snap.names == []
    assert snap.error == "stack_scan_pending"


def test_force_read_scan_busy_keeps_confirmed_empty_hud(monkeypatch):
    from mm_assist.memory.readers.ui_stack import UiStackSnap

    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = []
    r._empty_stacks = []
    r._scan_done_n = 1
    r._scan_busy = True
    kept = UiStackSnap(
        names=[],
        line="",
        n=0,
        ready=True,
        empty_confirmed=True,
        sampled_ts=1.0,
    )
    r._last_snap = kept
    r._last_read_ts = 0.0
    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(r, "_players_for_read", lambda _state_names, force=False: ([], 0))
    snap = r.read(force=True)
    assert snap is kept
    assert snap.ready is True
    assert snap.empty_confirmed is True
    assert snap.error == ""


def test_read_empty_names_two_ticks_confirm(monkeypatch):
    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = [1]
    r._scan_done_n = 1
    r._scan_busy = False
    r._empty_streak = 0
    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(r, "_names_from_state_lists", lambda: [])
    monkeypatch.setattr(r, "_players_for_read", lambda _state_names, force=False: ([], 0))
    first = r.read(force=True)
    assert first.ready is False
    assert first.empty_confirmed is False
    assert first.error == "stack_names_empty"
    second = r.read(force=True)
    assert second.ready is True
    assert second.empty_confirmed is True
    assert second.error == ""
    assert second.names == []
    assert second.to_dict()["empty_confirmed"] is True


def test_read_scan_done_no_candidates_two_ticks_confirm(monkeypatch):
    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = []
    r._empty_stacks = []
    r._scan_done_n = 1
    r._scan_busy = False
    r._empty_streak = 0
    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(r, "_players_for_read", lambda _state_names, force=False: ([], 0))
    first = r.read(force=True)
    assert first.ready is False
    assert first.empty_confirmed is False
    assert first.error == "stack_names_empty"
    second = r.read(force=True)
    assert second.ready is True
    assert second.empty_confirmed is True
    assert second.error == ""
    assert second.names == []
    assert second.to_dict()["empty_confirmed"] is True


def test_read_players_hud_not_empty_main(monkeypatch):
    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = []
    r._scan_done_n = 2
    r._scan_busy = False
    r._empty_stacks = [0x10]
    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(
        r,
        "_players_for_read",
        lambda _state_names, force=False: (["HudUIStepState"], 0x11),
    )
    snap = r.read(force=True)
    assert snap.empty_confirmed is False
    assert "HudUIStepState" in snap.names
    assert stack_node(snap.names, ready=snap.ready, empty_ok=True) != "main_hud"
    assert stack_node(snap.names, ready=True) == "overlay"


def test_stack_node_empty_ok_gate():
    assert stack_node([], ready=True, empty_ok=True) == "main_hud"
    assert stack_kind([], ready=True, empty_ok=True) == "main"
    assert stack_node([], ready=True) == ""
    assert stack_kind([], ready=True) == "overlay"


def test_scan_weak_empty_not_in_names_then_upgrade(monkeypatch):
    from mm_assist.memory.readers import ui_stack as mod

    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._scan_busy = True
    monkeypatch.setattr(
        mod,
        "scan_klass_objects",
        lambda *_a, **_k: [0xA0, 0xB0],
    )
    monkeypatch.setattr(r, "_plausible_state_stack", lambda *_a, **_k: False)
    monkeypatch.setattr(r, "_empty_state_stack", lambda _h, addr, _k: addr == 0xA0)
    monkeypatch.setattr(r, "_rank_stack_addrs", lambda *_a, **_k: [])
    r._scan_state_stacks(1, 1, 0xABC)
    assert r._empty_stacks == [0xA0]
    assert r._state_stacks == []
    assert r._scan_done_n == 1
    assert r._scan_busy is False
    names, _cat = r._names_from_state_lists()
    assert names == []

    r._empty_stacks = [0x50]
    r._state_stacks = []
    r._next_stack_scan = 1e18
    started: list[str] = []

    class FakeThread:
        def __init__(self, target=None, args=(), name="", daemon=False):
            started.append(name)

        def start(self) -> None:
            return None

    monkeypatch.setattr(mod, "resolve_klass", lambda *_a, **_k: 0xABC)
    monkeypatch.setattr(mod, "read_ptr", lambda *_a, **_k: 0xABC)
    monkeypatch.setattr(r, "_plausible_state_stack", lambda *_a, **_k: True)
    monkeypatch.setattr(r, "_stack_list_size", lambda *_a, **_k: 2)
    monkeypatch.setattr(mod.threading, "Thread", FakeThread)
    r._ensure_state_stacks()
    assert 0x50 in r._state_stacks
    assert r._empty_stacks == []
    assert started == []


def test_bind0_player_overlay_does_not_zero_scan_throttle(monkeypatch):
    import time

    from mm_assist.memory.readers import ui_stack as mod

    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = []
    r._scan_done_n = 2
    r._scan_busy = False
    due = time.perf_counter() + 30
    r._next_stack_scan = due
    started: list[str] = []

    class FakeThread:
        def __init__(self, target=None, args=(), name="", daemon=False):
            started.append(name)

        def start(self) -> None:
            return None

    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(mod, "resolve_klass", lambda *_a, **_k: 0xABC)
    monkeypatch.setattr(mod, "read_ptr", lambda *_a, **_k: 0)
    monkeypatch.setattr(mod.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        r,
        "_players_for_read",
        lambda _state_names, force=False: (
            ["LivingSkillUIState", "HudUIStepState"],
            0x11,
        ),
    )
    snap = r.read(force=True)
    assert snap.ready is True
    assert "LivingSkillUIState" in snap.names
    assert started == []
    assert r._next_stack_scan == due
    assert "lock_wait_ms" in snap.cost_ms
    assert "ensure_ms" in snap.cost_ms
    assert "ensure_stacks_ms" in snap.cost_ms
    assert "state_ms" in snap.cost_ms
    assert "players_ms" in snap.cost_ms
    assert "overlay_rescan_ms" in snap.cost_ms
    assert snap.cost_ms["overlay_rescan_ms"] < 20


def test_bound_player_overlay_mismatch_still_forces_rescan(monkeypatch):
    import time

    from mm_assist.memory.readers import ui_stack as mod

    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = [0x50]
    r._scan_done_n = 2
    r._scan_busy = False
    r._next_stack_scan = time.perf_counter() + 30
    started: list[str] = []

    class FakeThread:
        def __init__(self, target=None, args=(), name="", daemon=False):
            started.append(name)

        def start(self) -> None:
            return None

    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_names_from_state_lists", lambda: [])
    monkeypatch.setattr(mod, "resolve_klass", lambda *_a, **_k: 0xABC)
    monkeypatch.setattr(mod, "read_ptr", lambda *_a, **_k: 0xABC)
    monkeypatch.setattr(r, "_stack_list_size", lambda *_a, **_k: 3)
    monkeypatch.setattr(mod.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        r,
        "_players_for_read",
        lambda _state_names, force=False: (
            ["LivingSkillUIState", "HudUIStepState"],
            0x11,
        ),
    )
    snap = r.read(force=True)
    assert snap.ready is True
    assert started == ["ui-stack-scan"]
    assert r._scan_busy is True


def test_rank_stack_addrs_not_under_lock(monkeypatch):
    from mm_assist.memory.readers import ui_stack as mod

    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._scan_busy = True
    unlocked = {"ok": False}

    def fake_rank(*_a, **_k):
        unlocked["ok"] = r._lock.acquire(blocking=False)
        if unlocked["ok"]:
            r._lock.release()
        return [0xA0]

    monkeypatch.setattr(mod, "scan_klass_objects", lambda *_a, **_k: [0xA0])
    monkeypatch.setattr(r, "_plausible_state_stack", lambda *_a, **_k: True)
    monkeypatch.setattr(r, "_empty_state_stack", lambda *_a, **_k: False)
    monkeypatch.setattr(r, "_rank_stack_addrs", fake_rank)
    r._scan_state_stacks(1, 1, 0xABC)
    assert unlocked["ok"] is True
    assert r._state_stacks == [0xA0]


def test_warm_reader_checks_handle_without_enumerating_all_processes(monkeypatch):
    from mm_assist.memory.readers import ui_stack as mod
    r = Il2CppUiStackReader()
    r._h, r._ga, r._pid = 123, 456, 789
    monkeypatch.setattr(mod, "process_handle_alive", lambda handle: handle == 123, raising=False)
    monkeypatch.setattr(mod, "find_game_pid", lambda: pytest.fail("warm read enumerated processes"))
    assert r._ensure()


def test_dead_handle_rebinds_even_if_new_process_has_same_pid(monkeypatch):
    from mm_assist.memory.readers import ui_stack as mod
    r = Il2CppUiStackReader()
    r._h, r._ga, r._pid = 123, 456, 789
    closed = []
    monkeypatch.setattr(mod, "process_handle_alive", lambda handle: False, raising=False)
    monkeypatch.setattr(mod, "find_game_pid", lambda: 789)
    monkeypatch.setattr(mod, "close_handle", lambda handle: closed.append(handle))
    monkeypatch.setattr(mod, "open_process", lambda pid: 124)
    monkeypatch.setattr(mod, "module_base", lambda *a: 457)
    assert r._ensure()
    assert (r._h, r._ga, r._pid) == (124, 457, 789)
    assert closed == [123]


def test_read_gate_lock_wait_ms(monkeypatch):
    import threading
    import time

    r = Il2CppUiStackReader()
    r._h = 1
    r._ga = 1
    r._pid = 1
    r._state_stacks = [1]
    r._labels = {}
    monkeypatch.setattr(r, "_ensure", lambda: True)
    monkeypatch.setattr(r, "_ensure_state_stacks", lambda: None)
    monkeypatch.setattr(r, "_names_from_state_lists", lambda: ["HudUIStepState"])
    monkeypatch.setattr(r, "_names_from_players", lambda: ([], 0))
    held = threading.Event()
    released = threading.Event()

    def holder() -> None:
        r._read_gate.acquire()
        held.set()
        time.sleep(0.06)
        r._read_gate.release()
        released.set()

    threading.Thread(target=holder, daemon=True).start()
    assert held.wait(1.0)
    snap = r.read(force=True)
    assert snap.cost_ms["lock_wait_ms"] >= 40
    assert released.wait(1.0)
