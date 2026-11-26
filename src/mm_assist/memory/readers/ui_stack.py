"""IL2CPP 只读：当前打开的 UI 状态栈。

主链：UIStateStack.stateList → 各 IUIState 的 Il2CppClass.name。
旁路：UIAPI.Stacking.players 与 listeners 中的当前有效地图。
探针：``python -m mm_assist.tools.il2cpp_ui_stack_probe``
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from mm_assist.knowledge_paths import ui_config
from mm_assist.memory.il2cpp import module_base, parse_int, resolve_klass, scan_klass_objects, read_il2cpp_string
from mm_assist.memory.process import (
    close_handle,
    find_game_pid,
    open_process,
    process_handle_alive,
    read_identity,
    read_mem,
    read_ptr,
    read_u32,
    read_u8,
)

log = logging.getLogger(__name__)

_LINE_CACHE_S = 0.2
_PLAYERS_THROTTLE_S = 0.35
_SCAN_OK_S = 1.2
_SCAN_FAIL_MIN_S = 1.2
_SCAN_FAIL_MAX_S = 8.0
_SCAN_WALL_S = 2.0
_SCAN_MAX_HITS = 48
_ULARA_WORLD_RESOURCES = frozenset({"WorldMap@v1_5", "WorldMap@v2"})


def selected_map_view(fact: Any) -> dict[str, Any] | None:
    """Names come from the active map UI, never the player's current location."""
    if not isinstance(fact, dict):
        return None
    kind = fact.get("map_type")
    resource = fact.get("resource_name")
    titles = fact.get("titles")
    if type(kind) is not int or kind not in (1, 2, 3, 4) or not isinstance(resource, str) or not resource:
        return None
    key = {1: "world", 2: "town", 3: "dungeon"}.get(kind)
    title = re.sub(r"<[^>]*>", "", str((titles or {}).get(key, ""))).strip() if isinstance(titles, dict) else ""
    node = "map"
    if kind == 1 and resource in _ULARA_WORLD_RESOURCES and title == "歐拉大陸":
        node = "map_continent"
    elif kind == 2 and resource == "Dunbarton" and title == "杜巴頓":
        node = "map_dunbarton"
    return {**fact, "selected_title": title, "node": node}


def _cfg() -> dict[str, Any]:
    return json.loads(ui_config("il2cpp_ui_stack.json").read_text(encoding="utf-8"))


def read_c_string(hproc, addr: int, *, max_len: int = 160) -> str:
    if not addr or addr < 0x10000:
        return ""
    raw = read_mem(hproc, addr, max_len)
    if not raw:
        return ""
    i = raw.find(b"\x00")
    blob = raw if i < 0 else raw[:i]
    try:
        return blob.decode("ascii")
    except UnicodeDecodeError:
        return blob.decode("utf-8", errors="replace")


def klass_simple_name(hproc, obj: int, name_off: int) -> str:
    if not obj:
        return ""
    klass = read_ptr(hproc, obj)
    if not klass or klass < 0x10000:
        return ""
    name_ptr = read_ptr(hproc, klass + name_off)
    raw = read_c_string(hproc, name_ptr or 0)
    if not raw:
        return ""
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    return raw


def label_for(class_name: str, labels: dict[str, str]) -> str:
    name = (class_name or "").strip()
    if not name:
        return ""
    return str(labels.get(name) or name)


def format_stack_line(
    class_names: list[str],
    labels: dict[str, str] | None = None,
) -> str:
    """栈顶在左：有中文映射用中文，否则完整类名。"""
    if not class_names:
        return "主界面"
    labs = labels or {}
    shown: list[str] = []
    for raw in reversed(class_names):
        lab = label_for(raw, labs)
        if lab and lab not in shown:
            shown.append(lab)
    return " > ".join(shown) if shown else "主界面"


HUD_STEP_NAME = "HudUIStepState"
TUTORIAL_TOOLTIP_NAME = "TutorialTooltipUIState"
# Observed current result/rank layers cover the game even when the stateList
# is temporarily absent and only the active player instance is available.
RESULT_OVERLAY_NAMES = frozenset({'ClearResultUIInstance', 'ClearResultUIState', 'ClearRankUIState'})
# Dump types own ClickShieldHandle; an underlying page is not input-ready
# while these instances/states remain in the active UI stack.
INPUT_BLOCKING_NAMES = frozenset({"ToyTransparentLayerUIInstance", "ToyTransparentLayerUIState",
    "ToyDimLayerUIInstance", "ToyDimLayerUIState", "ClickShieldInstance", "ClickShieldUIState"})


def input_blocking_layers(names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    return tuple(n for n in (names or ()) if n in INPUT_BLOCKING_NAMES)


def facility_travel_popup_has_only_own_shield(names: list[str] | tuple[str, ...] | None) -> bool:
    """The identified common popup may receive input; its underlying page may not."""
    ordered = list(names or ())
    active = set(ordered)
    return ("ButtonCommonPopupInstance" in active
            and "AlteringFacilityPopupUIInstance" in active
            and all(ordered.index(n) < ordered.index("ButtonCommonPopupInstance")
                    for n in ("AlteringFacilityPopupUIInstance", "AlteringPanelPopupUIInstance",
                              "FacilityResultUIInstance") if n in active)
            and set(input_blocking_layers(names)) <= {"ClickShieldUIState", "ClickShieldInstance"})


INVENTORY_NAMES = frozenset(
    {
        "EquipmentInventoryUIState",
        "EquipmentInventoryUIInstance",
        "InventoryUIInstance",
    }
)
DIALOGUE_NAMES = frozenset(
    {"DialogueInteractionUIState", "DialoguePlayerComponent"}
)
SIDE_NAV_NAMES = frozenset(
    {"SideNavigationUIState", "SideNavigationUIInstance"}
)
PROFILE_NAMES = frozenset({"NewProfileUIState", "NewProfileUIInstance"})
_PROFILE_SHELL_NAMES = PROFILE_NAMES | frozenset({
    "HudUIStepState", "TutorialTooltipUIState", "UILayerHideState", "SizeAdjustmentSender",
})
_PROFILE_CATEGORY_NODES = {0: "profile:info", 1: "profile:combat", 2: "life_grid", 3: "profile:career"}


def profile_node_from_category(names, *, ready: bool, category: int | None) -> str:
    """Only the verified four-tab shell; skill lists/details and unknown layers fall back."""
    seq = set(names or [])
    if (not ready or type(category) is not int or "NewProfileUIInstance" not in seq
            or not seq <= _PROFILE_SHELL_NAMES):
        return ""
    return _PROFILE_CATEGORY_NODES.get(category, "")


LIFE_SKILL_NAMES = frozenset(
    {
        "LivingSkillUIState",
        "LivingSkillUIInstance",
        "LivingSkillLevelUpUIState",
    }
)
REPAIR_NAMES = frozenset(
    {
        "EquipmentRepairListUIInstance",
        "EquipmentRepairUIInstance",
        "EquipmentRepairListUIState",
    }
)
MAP_STACK_NAMES = frozenset({"MapUIInstance", "MapUIState", "MapLocationPanelUIState",
                             "MapLocationPanelUIInstance"})
MAP_LISTENER_COMPANIONS = MAP_STACK_NAMES | {"HudUIStepState", "TutorialTooltipUIState"}
SKIP_BUTTON_NAME = "SkipButtonUIState"
HOWTO_NAME = "HowToGetSelectUIState"
HOWTO_NAMES = frozenset({"HowToGetSelectUIState", "HowToGetSelectUIInstance"})
DETAIL_CARD_NAME = "DetailCardUIState"
DETAIL_CARD_INSTANCE_NAME = "DetailCardUIInstance"
CURRENCY_NAME = "CurrencyUIState"
CURRENCY_NAMES = frozenset({CURRENCY_NAME, "CurrencyUIInstance"})
_DISPLAY_STACK = (
    "CashShopUIState",
    "CashShopUIInstance",
    "EventUIState",
    "ToyHudZeroLayerUIState",
    "PostBoxUIState",
    "NewDailyMissionUIState",
    "DungeonListUIInstance",
    "DungeonEntryUIInstance",
    "DungeonListUIState",
    "DungeonEntryUIState",
)
_CRAFT_PREFIXES = ("CraftingHub", "Facility", "Altering")
CRAFT_CAT_VALID = frozenset({0, 1, 2, 3})
CRAFT_CURRENT_SHELL_NAMES = frozenset({"CraftingHubUIInstance", "CraftingHubPopupUIInstance",
    "CraftingHubUIState", "HudUIStepState", "TutorialTooltipUIState", "UILayerHideState"})


def craft_node_from_current(names, *, ready: bool, category: int | None) -> str:
    seq = list(names or [])
    if (not ready or type(category) is not int or category not in CRAFT_CAT_VALID
            or seq.count("CraftingHubPopupUIInstance") != 1
            or not set(seq) <= CRAFT_CURRENT_SHELL_NAMES):
        return ""
    return {0:"craft_quick", 1:"craft_hub", 2:"craft_make", 3:"craft_alchemy"}[category]

CRAFT_PANEL_NAMES = frozenset({"AlteringPanelPopupUIInstance"})
CRAFT_FACILITY_NAMES = frozenset({"AlteringFacilityPopupUIInstance"})
CRAFT_FACILITY_STATE_FALLBACK = frozenset({"CraftingHubAlteringFacilityPanelState"})
CRAFT_HUB_SHELL_NAMES = frozenset(
    {"CraftingHubPopupUIInstance", "CraftingHubUIInstance"}
)
COMMON_POPUP_NAMES = frozenset(
    {"ButtonCommonPopupInstance", "ButtonCommonPopupUIState"}
)


def stack_common_popup(names: list[str] | None) -> bool:
    """领取后「附近没有设施、要前往吗」这类通用确认弹窗。"""
    seq = _stack_seq(names)
    return any(n in COMMON_POPUP_NAMES for n in seq)


def stack_craft_hub(names: list[str]) -> bool:
    return any(n.startswith(_CRAFT_PREFIXES) for n in names)


def _stack_craft_item_focus(names: list[str]) -> bool:
    """制作物品明细：CraftingHub*Focus*。Select / 六卡 / FacilityPanel 不是明细。"""
    for n in names:
        if not n.startswith("CraftingHub") or "Focus" not in n:
            continue
        if "Select" in n or "FacilityPanel" in n:
            continue
        return True
    return False


def _stack_facility_detail(names: list[str]) -> bool:
    """加工层上的 DetailCardUIInstance：配方原料卡，不是背包物品详情。"""
    seq = _stack_seq(names)
    if DETAIL_CARD_INSTANCE_NAME not in seq:
        return False
    if stack_inv(seq) or _stack_craft_item_focus(seq):
        return False
    if stack_craft_hub(seq):
        return True
    if any(n in CRAFT_PANEL_NAMES or n in CRAFT_FACILITY_NAMES for n in seq):
        return True
    return any(n.startswith("Altering") or n.startswith("Facility") for n in seq)


def _i32(h, addr: int) -> int | None:
    raw = read_u32(h, addr)
    if raw is None:
        return None
    return int(raw if raw < 0x80000000 else raw - 0x100000000)


def craft_node_from_stack(
    names: list[str],
    *,
    craft_category: int | None = None,
) -> str:
    """制作栈且无物品明细：配方 Popup→fine:facility_recipe，设施 Popup→fine:facility。

    有 Hub 壳 Popup/实例且无设施 Popup 时，即使残留 PanelState 也是 fine:craft_hub。
    仅 PanelState、无 Hub 壳实例才弱证据兜底 fine:facility。CraftingHubUIState 不算壳。
    fine:craft_hub 是「制作壳待分 Tab」，不是终态「制作-加工」。category 不进认页。
    """
    del craft_category
    seq = _stack_seq(names)
    if _stack_craft_item_focus(seq):
        return ""
    if any(n in CRAFT_PANEL_NAMES for n in seq):
        return "fine:facility_recipe"
    if any(n in CRAFT_FACILITY_NAMES for n in seq):
        return "fine:facility"
    if any(n in CRAFT_HUB_SHELL_NAMES for n in seq):
        return "fine:craft_hub"
    if any(n in CRAFT_FACILITY_STATE_FALLBACK for n in seq):
        return "fine:facility"
    if stack_craft_hub(seq):
        return "fine:craft_hub"
    return ""


def _stack_seq(names: list[str] | None) -> list[str]:
    return [str(n) for n in (names or []) if n]


def stack_left_overlay(names: list[str] | None) -> bool:
    """已离开主界面：有 Hud，或已有菜单/背包/对话/制作等离开层（堆扫未带到 Hud 时仍算离开）。"""
    seq = _stack_seq(names)
    if input_blocking_layers(seq) or any(n in RESULT_OVERLAY_NAMES for n in seq):
        return True
    if not seq:
        return False
    if HUD_STEP_NAME in seq:
        return True
    if SKIP_BUTTON_NAME in seq or any(n in HOWTO_NAMES for n in seq) or DETAIL_CARD_NAME in seq:
        return True
    if any(n in CURRENCY_NAMES or n in REPAIR_NAMES for n in seq):
        return True
    if stack_dlg(seq) or any(n.startswith("DialogueInteraction") for n in seq):
        return True
    if stack_inv(seq) or stack_menu(seq) or stack_craft_hub(seq):
        return True
    if "MapLocationPanelUIState" in seq or any(n in MAP_STACK_NAMES for n in seq):
        return True
    if any(n in LIFE_SKILL_NAMES or n in PROFILE_NAMES for n in seq):
        return True
    if any(n in _DISPLAY_STACK for n in seq):
        return True
    if _stack_craft_item_focus(seq):
        return True
    return False


def stack_node(
    names: list[str] | None,
    *,
    ready: bool,
    craft_category: int | None = None,
    empty_ok: bool = False,
) -> str:
    """栈就绪后的栏上节点。未就绪返回空串。离开层先于「无 Hud=主界面」。

    empty_ok 仅在调用方已确认空栈（连续两拍、无玩家离开层）时为 True，才把空串认成 main_hud。
    """
    if not ready:
        return ""
    seq = _stack_seq(names)
    if not seq:
        return "main_hud" if empty_ok else ""
    if input_blocking_layers(seq) or any(n in RESULT_OVERLAY_NAMES for n in seq):
        return "overlay"
    if SKIP_BUTTON_NAME in seq:
        return "dialog_skip"
    if stack_dlg(seq) or any(n.startswith("DialogueInteraction") for n in seq):
        return "dialog"
    if any(n in HOWTO_NAMES for n in seq):
        return "obtain_method"
    if DETAIL_CARD_NAME in seq or _stack_craft_item_focus(seq):
        return "item_info"
    if "MapLocationPanelUIState" in seq or any(n in MAP_STACK_NAMES for n in seq):
        return "map"
    if DETAIL_CARD_INSTANCE_NAME in seq and stack_inv(seq):
        return "item_info"
    if stack_inv(seq):
        return "inventory"
    if stack_menu(seq):
        return "system_menu"
    if any(n in LIFE_SKILL_NAMES for n in seq):
        return "life_grid"
    if any(n in PROFILE_NAMES for n in seq):
        return "profile"
    if any(n in CURRENCY_NAMES for n in seq):
        return "currency_wallet"
    if any(n in REPAIR_NAMES for n in seq):
        return "overlay"
    if _stack_facility_detail(seq):
        return "fine:facility_detail"
    crafted = craft_node_from_stack(seq, craft_category=craft_category)
    if crafted:
        return crafted
    for cls in _DISPLAY_STACK:
        if cls in seq:
            return "stack:CashShopUIState" if cls == "CashShopUIInstance" else f"stack:{cls}"
    if HUD_STEP_NAME in seq:
        return "overlay"
    return "main_hud"


def stack_kind(
    names: list[str] | None, *, ready: bool, empty_ok: bool = False
) -> str | None:
    """栈就绪后的粗类。未就绪返回 None，禁止把空栈当主界面。

    overlay = 已离开主界面但还没有更细类名（未采页 fail-closed）。
    empty_ok 默认 False，现有调用行为不变。
    """
    if not ready:
        return None
    seq = _stack_seq(names)
    node = stack_node(seq, ready=True, empty_ok=empty_ok)
    if node in ("dialog", "dialog_skip"):
        return "dialog"
    if node in ("inventory", "item_info", "obtain_method"):
        return "inventory"
    if any(n in REPAIR_NAMES for n in seq):
        return "repair"
    if node in ("fine:life_grid", "life_grid"):
        return "life_skill"
    if node == "profile":
        return "profile"
    if node in (
        "fine:craft_hub",
        "fine:facility",
        "fine:facility_recipe",
        "fine:facility_detail",
        "craft_hub",
        "craft_shell",
        "craft_quick",
        "craft_make",
        "craft_alchemy",
        "facility_recipe",
    ) or str(node).startswith("facility"):
        return "craft_hub"
    if node == "map_dunbarton" or str(node).startswith("map"):
        return "map"
    if node == "system_menu":
        return "side_nav"
    if node == "main_hud":
        return "main"
    return "overlay"


def stack_relation_label(kind: str | None) -> str | None:
    if kind == "dialog":
        return "对话界面"
    if kind == "inventory":
        return "物品界面"
    if kind == "repair":
        return "修理界面"
    if kind == "life_skill":
        return "生活技能界面"
    if kind == "profile":
        return "人物界面"
    if kind == "craft_hub":
        return "制作界面"
    if kind == "map":
        return "地图"
    if kind == "side_nav":
        return "菜单"
    if kind == "overlay":
        return "离开主界面"
    if kind == "main":
        return None
    return None


def pick_stack_addrs(sized: list[tuple[Any, int]], *, cap: int = 3) -> list[int]:
    """(排序键, 地址) 大的优先，最多 cap 个。键可以是长度，或 (离开层, 类名数, 长度)。"""
    ranked = sorted(
        ((key, int(addr)) for key, addr in sized if addr),
        reverse=True,
    )
    out: list[int] = []
    for _sz, addr in ranked:
        if addr not in out:
            out.append(addr)
        if len(out) >= cap:
            break
    return out


def left_main_hud(names: list[str] | None) -> bool | None:
    """已离开主界面。names 为 None 表示未就绪。"""
    if names is None:
        return None
    return stack_left_overlay(names)


def stack_inv(names: list[str]) -> bool:
    return any(n in INVENTORY_NAMES for n in names)


def stack_dlg(names: list[str]) -> bool:
    return any(n in DIALOGUE_NAMES for n in names)


def stack_menu(names: list[str]) -> bool:
    """系统侧栏。不当「离开主界面」用。"""
    return any(n in SIDE_NAV_NAMES for n in names)


def stack_compare_key(names: list[str]) -> str:
    """换页对拍：去掉 HudUIStepState 后排序拼接。"""
    bits = sorted(n for n in names if n and n != HUD_STEP_NAME)
    return "|".join(bits)


def stack_hud_only(names: list[str] | None) -> bool:
    """去掉 Hud 与教程提示后栈为空：主界面候选（仍须菜单图标+两拍）。"""
    seq = _stack_seq(names)
    rest = [n for n in seq if n not in (HUD_STEP_NAME, TUTORIAL_TOOLTIP_NAME)]
    return bool(seq) and not rest


_LIST_COUNT_MAX = 32


def _mmlist_count(handle, list_ptr: int, off: dict[str, int], n_arr: int) -> int:
    """System.List 的 _size 在 0x18；MMList 同槽是 version，真 size 在 0x1C。"""
    size_list = int(read_u32(handle, list_ptr + off["MMList._size"]) or 0)
    size_mm = int(read_u32(handle, list_ptr + (off.get("MMList.size") or 0x1C)) or 0)
    cap_ok = n_arr if n_arr > 0 else _LIST_COUNT_MAX

    def _ok(n: int) -> bool:
        return 0 < n <= _LIST_COUNT_MAX and n <= cap_ok

    if _ok(size_list):
        return size_list
    if _ok(size_mm):
        return size_mm
    return 0


@dataclass
class UiStackSnap:
    names: list[str] = field(default_factory=list)
    line: str = "—"
    n: int = 0
    error: str = ""
    stacking: int = 0
    ready: bool = False
    craft_category: int | None = None
    craft_current_category: int | None = None
    craft_identity: tuple[int, ...] | None = None
    profile_category: int | None = None
    profile_identity: tuple[int, int] | None = None
    map_identity: tuple[int, int, int] | None = None
    map_view: dict[str, Any] | None = None
    sampled_ts: float = 0.0
    empty_confirmed: bool = False
    cost_ms: dict[str, float] = field(default_factory=dict)
    process_identity: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "n": self.n,
            "names": list(self.names),
            "error": self.error,
            "stacking": hex(self.stacking) if self.stacking else "0",
            "ready": self.ready,
            "craft_category": self.craft_category,
            "craft_current_category": self.craft_current_category,
            "craft_identity": self.craft_identity,
            "profile_category": self.profile_category,
            "profile_identity": self.profile_identity,
            "map_identity": self.map_identity,
            "map_view": self.map_view,
            "sampled_ts": self.sampled_ts,
            "empty_confirmed": self.empty_confirmed,
            "cost_ms": dict(self.cost_ms),
            "process_identity": dict(self.process_identity),
        }


_SHARED_READER = None
_SHARED_READER_LOCK = threading.Lock()


def shared_ui_stack_reader():
    """显示与导航共用定位结果；read(force=True) 仍读取当前游戏状态。"""
    global _SHARED_READER
    with _SHARED_READER_LOCK:
        if _SHARED_READER is None:
            _SHARED_READER = Il2CppUiStackReader()
        return _SHARED_READER


def peek_ui_stack(pid: int, *, max_age_s: float = 1.5) -> UiStackSnap | None:
    """Scheduling hint only: no attach, read, wait or reader creation."""
    reader = _SHARED_READER
    if reader is None or getattr(reader, '_pid', None) != pid:
        return None
    snap = getattr(reader, '_last_snap', None)
    if snap is None or not snap.ready or snap.error:
        return None
    if not -0.05 <= time.time() - float(snap.sampled_ts or 0) <= max_age_s:
        return None
    return snap


class Il2CppUiStackReader:
    def __init__(self) -> None:
        raw = _cfg()
        self._module = str(raw.get("module") or "GameAssembly.dll")
        self._rva = {k: parse_int(v) for k, v in (raw.get("typeinfo_rva") or {}).items()}
        self._off = {k: parse_int(v) for k, v in (raw.get("offsets") or {}).items()}
        self._labels = {
            str(k): str(v) for k, v in (raw.get("labels") or {}).items() if str(v).strip()
        }
        self._skip = {str(x) for x in (raw.get("skip_names") or []) if str(x).strip()}
        self._active_only = {
            str(x) for x in (raw.get("active_only_names") or []) if str(x).strip()
        }
        self._h = None
        self._pid: int | None = None
        self._ga: int | None = None
        self._state_stacks: list[int] = []
        self._empty_stacks: list[int] = []
        self._scan_done_n = 0
        self._empty_streak = 0
        self._next_stack_scan = 0.0
        self._scan_fail_s = _SCAN_FAIL_MIN_S
        self._scan_busy = False
        self._scan_busy_since = 0.0
        self._lock = threading.Lock()
        self._read_gate = threading.RLock()
        self._cached_players: tuple[list[str], int] = ([], 0)
        self._next_players = 0.0
        self._last_snap: UiStackSnap | None = None
        self._last_read_ts = 0.0
        self._log_key: tuple[Any, ...] | None = None
        self.last_error: str | None = None
        self.last_timing: dict[str, float] = {}

    def close(self) -> None:
        with self._lock:
            h = self._h
            self._h = None
            self._ga = None
            self._pid = None
            self._state_stacks = []
            self._empty_stacks = []
            self._scan_done_n = 0
            self._empty_streak = 0
            self._scan_busy = False
            self._scan_busy_since = 0.0
        if h:
            close_handle(h)

    def _ensure(self) -> bool:
        if self._h and self._ga and self._pid and process_handle_alive(self._h):
            try:
                read_identity(self._h, self._pid)
                return True
            except OSError as exc:
                self.close()
                self.last_error = str(exc)
                return False
        if self._h:
            self.close()
        pid = find_game_pid()
        if not pid:
            self.last_error = "process_not_found"
            return False
        self.close()
        try:
            h = open_process(pid)
        except OSError as exc:
            self.last_error = str(exc)
            return False
        if not h:
            self.last_error = "open_process_failed"
            return False
        ga = module_base(pid, self._module)
        if not ga:
            close_handle(h)
            self.last_error = "gameassembly_not_found"
            return False
        self._pid = pid
        self._h = h
        self._ga = ga
        self._last_snap = None
        return True

    def _read_ptr_list(self, list_ptr: int, *, cap: int = 12, h=None) -> list[int]:
        handle = h if h is not None else self._h
        assert handle is not None
        off = self._off
        if not list_ptr:
            return []
        items = read_ptr(handle, list_ptr + off["MMList._items"])
        if not items:
            return []
        n_arr = int(read_u32(handle, items + off["Il2CppArray.max_length"]) or 0)
        size = _mmlist_count(handle, list_ptr, off, n_arr)
        if size <= 0:
            return []
        n = min(int(size), int(n_arr or size), cap)
        vec = items + off["Il2CppArray.vector"]
        out: list[int] = []
        for i in range(n):
            p = read_ptr(handle, vec + i * 8)
            if p and p > 0x10000:
                out.append(int(p))
        return out

    def _ensure_state_stacks(self) -> None:
        assert self._h is not None and self._ga is not None
        now = time.perf_counter()
        klass = resolve_klass(self._h, self._ga, self._rva.get("UIStateStack") or 0)
        with self._lock:
            stacks = list(self._state_stacks)
            empty = list(self._empty_stacks)
        alive: list[int] = []
        empty_alive: list[int] = []
        upgraded: list[int] = []
        if klass:
            for addr in stacks:
                if read_ptr(self._h, addr) == klass:
                    alive.append(addr)
            for addr in empty:
                if read_ptr(self._h, addr) != klass:
                    continue
                if self._plausible_state_stack(self._h, addr, klass):
                    upgraded.append(addr)
                else:
                    empty_alive.append(addr)
        if upgraded:
            seen = set(alive)
            for addr in upgraded:
                if addr not in seen:
                    alive.append(addr)
                    seen.add(addr)
        with self._lock:
            self._state_stacks = alive
            self._empty_stacks = empty_alive[:4]
            max_sz = 0
            if klass and alive:
                max_sz = max(
                    (self._stack_list_size(self._h, addr) for addr in alive),
                    default=0,
                )
            # 节流期内：有活栈且 size>=1 则跳过；size==0 或无 alive 也按节流，不每拍狂扫 size==1。
            if now < self._next_stack_scan:
                if alive and max_sz >= 1:
                    return
                return
            if self._scan_busy:
                since = float(self._scan_busy_since or 0.0)
                if since and (now - since) > (_SCAN_WALL_S + 1.0):
                    self._scan_busy = False
                    self._scan_busy_since = 0.0
                else:
                    return
            if not klass:
                self._scan_fail_s = min(_SCAN_FAIL_MAX_S, self._scan_fail_s * 2)
                self._next_stack_scan = now + self._scan_fail_s
                return
            self._scan_busy = True
            self._scan_busy_since = now
            h, ga = self._h, self._ga
        threading.Thread(
            target=self._scan_state_stacks,
            args=(h, ga, klass),
            name="ui-stack-scan",
            daemon=True,
        ).start()

    def _plausible_state_stack(self, h, addr: int, klass: int) -> bool:
        """堆上 klass 指针假阳极多；只计 stateList 像 MMList 的对象。"""
        if not addr or read_ptr(h, addr) != klass:
            return False
        lst = read_ptr(h, addr + self._off["UIStateStack.stateList"]) or 0
        if lst < 0x10000:
            return False
        size = int(read_u32(h, lst + self._off["MMList._size"]) or 0)
        return 1 <= size <= 24

    def _empty_state_stack(self, h, addr: int, klass: int) -> bool:
        """size==0 弱候选：klass 匹配 + stateList/_items 有效 + 数组长 1..32。不进正式 names。"""
        if not addr or read_ptr(h, addr) != klass:
            return False
        lst = read_ptr(h, addr + self._off["UIStateStack.stateList"]) or 0
        if lst < 0x10000:
            return False
        items = read_ptr(h, lst + self._off["MMList._items"])
        if not items or items < 0x10000:
            return False
        n_arr = int(read_u32(h, items + self._off["Il2CppArray.max_length"]) or 0)
        if not (1 <= n_arr <= 32):
            return False
        size = int(read_u32(h, lst + self._off["MMList._size"]) or 0)
        return size == 0

    def _stack_list_size(self, h, addr: int) -> int:
        lst = read_ptr(h, addr + self._off["UIStateStack.stateList"]) or 0
        if lst < 0x10000:
            return 0
        return int(read_u32(h, lst + self._off["MMList._size"]) or 0)

    def _rank_stack_addrs(self, h, addrs: list[int], klass: int) -> list[int]:
        sized: list[tuple[Any, int]] = []
        for addr in addrs:
            if not self._plausible_state_stack(h, addr, klass):
                continue
            names, _cat = self._names_from_one_stack(h, addr)
            leave = 1 if stack_left_overlay(names) else 0
            key = (leave, len(names), self._stack_list_size(h, addr))
            sized.append((key, int(addr)))
        return pick_stack_addrs(sized, cap=12)

    def _scan_state_stacks(self, h, ga, klass: int) -> None:
        _ = ga
        try:
            hits = scan_klass_objects(
                h,
                klass,
                max_hits=_SCAN_MAX_HITS,
                max_wall_s=_SCAN_WALL_S,
                hit_filter=lambda _n, addr: (
                    self._plausible_state_stack(h, addr, klass)
                    or self._empty_state_stack(h, addr, klass)
                ),
            )
            good = [int(addr) for addr in hits if self._plausible_state_stack(h, addr, klass)]
            weak = [
                int(addr)
                for addr in hits
                if self._empty_state_stack(h, addr, klass)
                and not self._plausible_state_stack(h, addr, klass)
            ]
            now = time.perf_counter()
            ranked = self._rank_stack_addrs(h, good, klass) if good else []
            with self._lock:
                if self._h is not h:
                    return
                self._empty_stacks = weak[:4]
                if good:
                    self._state_stacks = ranked
                    self._scan_fail_s = _SCAN_FAIL_MIN_S
                    self._next_stack_scan = now + _SCAN_OK_S
                else:
                    self._scan_fail_s = min(
                        _SCAN_FAIL_MAX_S, max(_SCAN_FAIL_MIN_S, self._scan_fail_s * 2)
                    )
                    self._next_stack_scan = now + self._scan_fail_s
        except Exception:
            log.debug("ui stack scan failed", exc_info=True)
            now = time.perf_counter()
            with self._lock:
                self._scan_fail_s = min(
                    _SCAN_FAIL_MAX_S, max(_SCAN_FAIL_MIN_S, self._scan_fail_s * 2)
                )
                self._next_stack_scan = now + self._scan_fail_s
        finally:
            with self._lock:
                if self._h is h:
                    self._scan_done_n += 1
                self._scan_busy = False
                self._scan_busy_since = 0.0

    def _names_from_one_stack(self, h, addr: int) -> tuple[list[str], int | None]:
        name_off = self._off["Il2CppClass.name"]
        list_off = self._off["UIStateStack.stateList"]
        cat_off = self._off.get("CraftingHubUIState.CategoryType") or 0
        local: list[str] = []
        local_cat: int | None = None
        lst = read_ptr(h, addr + list_off)
        ptrs = self._read_ptr_list(lst or 0, h=h)
        for st in ptrs:
            nm = klass_simple_name(h, st, name_off)
            if not nm or nm in self._skip:
                continue
            if nm in self._active_only:
                flag = read_u8(h, st + 0x10)
                if flag != 1:
                    continue
            local.append(nm)
            if nm == "CraftingHubUIState" and cat_off:
                cat = _i32(h, st + cat_off)
                if cat in CRAFT_CAT_VALID:
                    local_cat = cat
        return local, local_cat

    def _names_from_state_lists(self) -> tuple[list[str], int | None]:
        assert self._h is not None
        with self._lock:
            stacks = list(self._state_stacks)
        names: list[str] = []
        seen: set[str] = set()
        craft_cat: int | None = None
        for stack in stacks:
            local, local_cat = self._names_from_one_stack(self._h, stack)
            if local_cat is not None:
                craft_cat = local_cat
            for nm in local:
                if nm in seen:
                    continue
                seen.add(nm)
                names.append(nm)
        return names, craft_cat

    def _names_from_players(self) -> tuple[list[str], int]:
        assert self._h is not None and self._ga is not None
        api_rva = self._rva.get("UIAPI") or 0
        api_klass = resolve_klass(self._h, self._ga, api_rva)
        if not api_klass:
            return [], 0
        sf = read_ptr(self._h, api_klass + self._off["Il2CppClass.static_fields"])
        if not sf or sf < 0x10000:
            return [], 0
        stacking = read_ptr(self._h, sf + self._off["UIAPI.static.Stacking"]) or 0
        svc_rva = self._rva.get("UIStackingService") or 0
        svc_klass = resolve_klass(self._h, self._ga, svc_rva) if svc_rva else 0
        if not stacking:
            return [], 0
        name_off = self._off["Il2CppClass.name"]
        if svc_klass and read_ptr(self._h, stacking) != svc_klass:
            svc_nm = klass_simple_name(self._h, stacking, name_off)
            if svc_nm and svc_nm != "UIStackingService":
                return [], int(stacking)
        inst_off = self._off["UIStackingPlayer.instance"]
        names: list[str] = []
        players = self._read_ptr_list(
            read_ptr(self._h, stacking + self._off["UIStackingService.players"]) or 0
        )
        for player in players:
            inst = read_ptr(self._h, player + inst_off)
            nm = klass_simple_name(self._h, inst or 0, name_off)
            if not nm:
                nm = klass_simple_name(self._h, player, name_off)
            if nm:
                names.append(nm)
        return names, int(stacking)

    def _active_map_listener(self, stacking: int) -> tuple[int, int, int] | None:
        """Read only the current bounded listener list and its live map scope."""
        if not stacking or not self._h or not self._ga:
            return None
        h, off = self._h, self._off
        try:
            def name(p):
                return klass_simple_name(h, p or 0, off["Il2CppClass.name"])

            api = resolve_klass(h, self._ga, self._rva["UIAPI"])
            sf = read_ptr(h, api + off["Il2CppClass.static_fields"]) if api else 0
            if (not sf or read_ptr(h, sf + off["UIAPI.static.Stacking"]) != stacking
                    or name(stacking) != "UIStackingService"):
                return None
            lst = read_ptr(h, stacking + off["UIStackingService.listeners"]) or 0
            items = read_ptr(h, lst + off["MMList._items"]) if lst else 0
            if not items:
                return None
            version = read_u32(h, lst + off["MMList._size"])
            size = read_u32(h, lst + off["MMList.size"])
            cap = read_u32(h, items + off["Il2CppArray.max_length"])
            if version is None or size is None or cap is None or not 0 < size <= cap <= 128:
                return None
            entries = [read_ptr(h, items + off["Il2CppArray.vector"] + 8*i) or 0
                       for i in range(size)]
            maps = [p for p in entries if name(p) == "MapUIInstance"]
            if len(maps) != 1:
                return None
            inst = maps[0]
            hud = read_ptr(h, inst + off["MapUIInstance.hudUIStepState"]) or 0
            scope = read_ptr(h, inst + off["MapUIInstance.mapShowHideResourceScope"]) or 0
            resource = read_ptr(h, scope + off["MapResourceScope.ResourceScope"]) if scope else 0
            if (name(hud) != "HudUIStepState" or name(scope) != "MapResourceScope"
                    or name(resource) != "UIResourceScope"
                    or read_u8(h, inst + off["MapUIInstance.isComponentCreated"]) != 1
                    or read_u8(h, inst + off["MapUIInstance.isRegisteredAsStackingListener"]) != 1
                    or read_u8(h, resource + off["UIResourceScope.isDestroyed"]) != 0):
                return None
            # Closing/reopening and concurrent list mutations invalidate this sample.
            if (resolve_klass(h, self._ga, self._rva["UIAPI"]) != api
                    or read_ptr(h, api + off["Il2CppClass.static_fields"]) != sf
                    or read_ptr(h, sf + off["UIAPI.static.Stacking"]) != stacking
                    or read_ptr(h, stacking + off["UIStackingService.listeners"]) != lst
                    or read_ptr(h, lst + off["MMList._items"]) != items
                    or read_u32(h, lst + off["MMList._size"]) != version
                    or read_u32(h, lst + off["MMList.size"]) != size
                    or read_u32(h, items + off["Il2CppArray.max_length"]) != cap
                    or any(read_ptr(h, items + off["Il2CppArray.vector"] + 8*i) != p
                           for i, p in enumerate(entries))
                    or read_ptr(h, inst + off["MapUIInstance.hudUIStepState"]) != hud
                    or read_ptr(h, inst + off["MapUIInstance.mapShowHideResourceScope"]) != scope
                    or read_ptr(h, scope + off["MapResourceScope.ResourceScope"]) != resource
                    or read_u8(h, inst + off["MapUIInstance.isComponentCreated"]) != 1
                    or read_u8(h, inst + off["MapUIInstance.isRegisteredAsStackingListener"]) != 1
                    or read_u8(h, resource + off["UIResourceScope.isDestroyed"]) != 0):
                return None
            return inst, scope, resource
        except Exception:
            return None

    def _active_map_view(self, stacking: int, identity: tuple[int, int, int]) -> dict[str, Any] | None:
        """Double-read the displayed map through its active instance; no heap candidates."""
        h, off = self._h, self._off
        try:
            def typed(p, expected):
                if not p or p < 0x10000 or klass_simple_name(h, p, off["Il2CppClass.name"]) != expected:
                    raise ValueError("map association unavailable")
                return p

            def sample():
                links = []
                def link(parent, field, expected):
                    p = typed(read_ptr(h, parent + off[field]), expected)
                    links.append((parent, field, p))
                    return p
                inst = identity[0]
                ctx = link(inst, "MapUIInstance.ctx", "UIInstanceContext`1")
                builder = link(ctx, "MapUIContext.ComponentBuilder", "MapUIInstanceComponentBuilder")
                stage = link(builder, "MapUIInstanceComponentBuilder.gameStageEntityComponents", "GameStageEntityComponents")
                current = link(stage, "GameStageEntityComponents.CurrentViewingMapUIComponent", "CurrentViewingMapUIComponent")
                image = link(inst, "MapUIInstance.mapImageUI", "MapImageUI")
                if read_u8(h, image + off["MapImageUI.IsLoadingMap"]) != 0:
                    return None
                meta = link(image, "MapImageUI.MapModel.Meta", "MapModelMeta")
                button = link(inst, "MapUIInstance.mapButtonUI", "MapButtonUI")
                slot = link(button, "MapButtonUI.UISlot", "MapButtonUISlot")
                fact = {key: read_u32(h, current + off["CurrentViewingMapUIComponent." + field])
                        for key, field in (("map_type", "MapType"), ("geo_key", "GeoDataKey"),
                            ("world_key", "WorldDataKey"), ("viewing_channel", "ViewingChannelSheetId"),
                            ("game_space", "GameSpaceSheetId"), ("additional_channel", "AdditionalViewingChannelSheetId"),
                            ("moon_gate", "CurrentMoonGateSheetId"), ("current_channel", "CurrentChannelSheetId"))}
                resource = read_ptr(h, meta + off["MapModelMeta.ResourceName"])
                fact["resource_name"] = read_il2cpp_string(h, resource, max_chars=128) if resource else None
                fact["titles"] = {}
                for key, field in (("world", "TitleWorldText"), ("town", "TitleTownText"), ("dungeon", "TitleDungeonText")):
                    if "MarqueeTextLabelSlot.SourceText" in off:
                        label_field = {"world": "WorldMapMarqueeLabel", "town": "TownMapMarqueeLabel",
                                       "dungeon": "DungeonMapMarqueeLabel"}[key]
                        label = link(slot, "MapButtonUISlot." + label_field, "MarqueeTextLabelSlot")
                        text = link(label, "MarqueeTextLabelSlot.SourceText", "UIText")
                    else:
                        text = link(slot, "MapButtonUISlot." + field, "UIText")
                    raw = read_ptr(h, text + off["UIText.rawText"])
                    fact["titles"][key] = read_il2cpp_string(h, raw, max_chars=128) if raw else ""
                if any(v is None for k, v in fact.items() if k != "titles") or not selected_map_view(fact):
                    return None
                return links, fact

            first = sample()
            if not first or sample() != first or self._active_map_listener(stacking) != identity:
                return None
            return first[1]
        except Exception:
            return None

    def _active_profile(self, stacking: int) -> tuple[int, tuple[int, int]] | None:
        """Bounded active association, never heap candidates or readonly opening TopCategory."""
        if not stacking or not self._h or not self._ga:
            return None
        h, off = self._h, self._off
        try:
            api = resolve_klass(h, self._ga, self._rva["UIAPI"])
            sf = read_ptr(h, api + off["Il2CppClass.static_fields"]) if api else 0
            if not sf or read_ptr(h, sf + off["UIAPI.static.Stacking"]) != stacking:
                return None
            def name(p):
                return klass_simple_name(h, p or 0, off["Il2CppClass.name"])
            if name(stacking) != "UIStackingService":
                return None
            lst = read_ptr(h, stacking + off["UIStackingService.players"]) or 0
            items = read_ptr(h, lst + off["MMList._items"]) if lst else 0
            if not items:
                return None
            version = read_u32(h, lst + off["MMList._size"])
            size = read_u32(h, lst + off["MMList.size"])
            cap = read_u32(h, items + off["Il2CppArray.max_length"])
            if version is None or size is None or cap is None or not 0 < size <= cap <= 128:
                return None
            pairs, names = [], []
            for i in range(size):
                player = read_ptr(h, items + off["Il2CppArray.vector"] + 8*i) or 0
                if name(player) != "UIStackingPlayer":
                    return None
                inst = read_ptr(h, player + off["UIStackingPlayer.instance"]) or 0
                pairs.append((player, inst))
                names.append(name(inst))
            profiles = [inst for (_, inst), nm in zip(pairs, names) if nm == "NewProfileUIInstance"]
            if len(profiles) != 1:
                return None
            inst = profiles[0]
            vm = read_ptr(h, inst + off["NewProfileUIInstance.viewModelComponent"]) or 0
            if name(vm) != "ProfileUIViewModelComponent":
                return None
            category = _i32(h, vm + off["ProfileUIViewModelComponent.currentTopCategory"])
            if not profile_node_from_category(names, ready=True, category=category):
                return None
            # Recheck roots and contents; closing, reopening or switching in this read is unknown.
            if (read_ptr(h, sf + off["UIAPI.static.Stacking"]) != stacking
                    or read_ptr(h, stacking + off["UIStackingService.players"]) != lst
                    or read_ptr(h, lst + off["MMList._items"]) != items
                    or read_u32(h, lst + off["MMList._size"]) != version
                    or read_u32(h, lst + off["MMList.size"]) != size
                    or any(read_ptr(h, p + off["UIStackingPlayer.instance"]) != obj for p, obj in pairs)
                    or read_ptr(h, inst + off["NewProfileUIInstance.viewModelComponent"]) != vm
                    or _i32(h, vm + off["ProfileUIViewModelComponent.currentTopCategory"]) != category):
                return None
            return category, (inst, vm)
        except Exception:
            return None

    def _active_craft(self, stacking: int) -> tuple[int, tuple[int, ...]] | None:
        """Current live shell keeper; no opening-state parameter, heap scan or fact cache.

        Field sources: dump.cs CraftingHubPopupUIInstance/UIViewModelComponent,
        CraftingHubUIComponent/UIKeeper and MMMLUIHandle. Covering facilities,
        cards and dialogs are deliberately outside this shell-only fact.
        """
        if not stacking or not self._h or not self._ga:
            return None
        h, off = self._h, self._off
        deadline = time.perf_counter() + 0.12
        checked, named = {}, {}
        def value(fn, addr):
            if time.perf_counter() > deadline:
                raise ValueError("craft read budget")
            result = fn(h, addr)
            if result is None:
                raise ValueError("craft unreadable field")
            old = checked.setdefault((fn, addr), result)
            if old != result:
                raise ValueError("craft changed while reading")
            return result
        def ptr(addr):
            p = value(read_ptr, addr)
            if type(p) is not int or p < 0x10000:
                raise ValueError("craft pointer")
            return p
        def name(p):
            if time.perf_counter() > deadline:
                raise ValueError("craft read budget")
            result = klass_simple_name(h, p, off["Il2CppClass.name"])
            if named.setdefault(p, result) != result:
                raise ValueError("craft class changed while reading")
            return result
        try:
            api = resolve_klass(h, self._ga, self._rva["UIAPI"])
            sf = ptr(api + off["Il2CppClass.static_fields"])
            if ptr(sf + off["UIAPI.static.Stacking"]) != stacking or name(stacking) != "UIStackingService":
                return None
            players = ptr(stacking + off["UIStackingService.players"])
            items = ptr(players + off["MMList._items"])
            version = value(read_u32, players + off["MMList._size"])
            size = value(read_u32, players + off["MMList.size"])
            cap = value(read_u32, items + off["Il2CppArray.max_length"])
            if (version is None or size is None or cap is None
                    or not 0 < size <= 16 or not size <= cap <= 128):
                return None
            instances = []
            for i in range(size):
                player = ptr(items + off["Il2CppArray.vector"] + 8*i)
                if name(player) != "UIStackingPlayer":
                    return None
                inst = ptr(player + off["UIStackingPlayer.instance"])
                instances.append((inst, name(inst)))
            if not craft_node_from_current([nm for _, nm in instances], ready=True, category=0):
                return None
            popups = [obj for obj, nm in instances if nm == "CraftingHubPopupUIInstance"]
            if len(popups) != 1:
                return None
            popup = popups[0]
            component = ptr(popup + off["CraftingHubPopupUIInstance.component"])
            vmc = ptr(popup + off["CraftingHubPopupUIInstance.viewModelComponent"])
            handle = ptr(popup + off["CraftingHubPopupUIInstance.handle"])
            vm = ptr(popup + off["CraftingHubPopupUIInstance.viewModel"])
            if (name(component) != "CraftingHubUIComponent" or name(vmc) != "CraftingHubPopupUIViewModelComponent"
                    or name(handle) != "MMMLUIHandle" or name(vm) != "MMMLUIViewModel`2"
                    or ptr(popup + off["CraftingHubPopupUIInstance.base_handle"]) != handle
                    or ptr(vmc + off["CraftingHubPopupUIViewModelComponent.handle"]) != handle
                    or ptr(vmc + off["CraftingHubPopupUIViewModelComponent.component"]) != component
                    or ptr(handle + off["MMMLUIHandle.viewModel"]) != vm
                    or value(read_u8, handle + off["MMMLUIHandle.active"]) != 1):
                return None
            hubs = [obj for obj, nm in instances if nm == "CraftingHubUIInstance"]
            if len(hubs) > 1 or any(ptr(obj + off["CraftingHubUIInstance.component"]) != component for obj in hubs):
                return None
            root = ptr(handle + off["MMMLUIHandle.root"])
            root_id = value(read_u32, handle + off["MMMLUIHandle.root_id"])
            service = ptr(handle + off["MMMLUIHandle.service"])
            nullable = handle + off["MMMLUIHandle.window_id"]
            if name(root) != "XForest" or name(service) != "MMMLService" or value(read_u8, nullable) != 1:
                return None
            window_id = value(read_u32, nullable + 4)
            windows = ptr(service + off["MMMLService.windows"])
            count = value(read_u32, windows + off["MMMLWindows.count"])
            keys = ptr(windows + off["MMMLWindows.keys"])
            values = ptr(windows + off["MMMLWindows.values"])
            if (count is None or not 0 < count <= 128 or
                    any((value(read_u32, arr + off["Il2CppArray.max_length"]) or 0) < count for arr in (keys, values))):
                return None
            matches = []
            for i in range(count):
                key = value(read_u32, keys + off["Il2CppArray.vector"] + 4*i)
                obj = ptr(values + off["Il2CppArray.vector"] + 8*i)
                if key == window_id:
                    matches.append(obj)
            if len(matches) != 1:
                return None
            window = matches[0]
            scope = ptr(window + off["UnityMMMLWindow.scope"])
            if (name(window) != "UnityMMMLWindow" or name(scope) != "UIResourceScope"
                    or value(read_u8, window + off["UnityMMMLWindow.closing"]) != 0
                    or value(read_u8, scope + off["UIResourceScope.isDestroyed"]) != 0
                    or ptr(window + off["UnityMMMLWindow.root"]) != root
                    or value(read_u32, window + off["UnityMMMLWindow.root_id"]) != root_id
                    or value(read_u32, window + off["UnityMMMLWindow.id"]) != window_id):
                return None
            keeper = ptr(component + off["CraftingHubUIComponent.keeper"])
            if name(keeper) != "CraftingHubUIKeeper":
                return None
            category = value(read_u32, keeper + off["CraftingHubUIKeeper.currentTopCategory"])
            if type(category) is not int or category not in CRAFT_CAT_VALID:
                return None
            for (fn, addr), observed in checked.items():
                if time.perf_counter() > deadline or fn(h, addr) != observed:
                    return None
            if (resolve_klass(h, self._ga, self._rva["UIAPI"]) != api
                    or any(name(addr) != nm for addr, nm in named.items())):
                return None
            return category, (popup, component, keeper, vmc, handle)
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _players_for_read(self, state_names: list[str], *, force: bool = False) -> tuple[list[str], int]:
        now = time.perf_counter()
        if state_names and not force:
            if now < self._next_players and self._cached_players[0]:
                return self._cached_players
            self._cached_players = self._names_from_players()
            self._next_players = now + _PLAYERS_THROTTLE_S
            return self._cached_players
        self._cached_players = self._names_from_players()
        if force:
            self._next_players = now + _PLAYERS_THROTTLE_S
        return self._cached_players

    def read_facility_ui(self, *, diagnostics: dict | None = None) -> dict[str, Any] | None:
        """Current widgets under the same serial gate; never start a heap scan."""
        return self._read_mmml_window(diagnostics=diagnostics, craft_render=False)

    def read_craft_render(self, *, diagnostics: dict | None = None) -> dict[str, Any] | None:
        """Bounded diagnostic only, with the same shared locks and no heap scan."""
        return self._read_mmml_window(diagnostics=diagnostics, craft_render=True)

    def _read_mmml_window(self, *, diagnostics, craft_render):
        from mm_assist.memory.il2cpp import ui_memory_read_slot
        deadline = time.perf_counter() + 0.25
        # Announce before waiting for the stack gate so its active scan can yield.
        with ui_memory_read_slot(timeout_s=0.25) as acquired:
            if not acquired:
                if diagnostics is not None:diagnostics.update(error='heap_read_busy')
                return None
            if not self._read_gate.acquire(timeout=max(0.0, deadline-time.perf_counter())):
                if diagnostics is not None:diagnostics.update(error='reader_busy')
                return None
            try:
                if not self._ensure():
                    if diagnostics is not None:diagnostics.update(error=self.last_error or 'attach')
                    return None
                binding=(self._h,self._pid,self._ga)
                before=read_identity(self._h,self._pid)
                if find_game_pid()!=self._pid:
                    if diagnostics is not None:diagnostics.update(error='selected_process_changed')
                    return None
                from mm_assist.memory.readers.facility_ui import read_active_facility, read_active_craft_render
                read_window = read_active_craft_render if craft_render else read_active_facility
                fact = read_window(self._h, self._ga, self._rva['UIAPI'],diagnostics=diagnostics)
                if fact is None:return None
                if (binding!=(self._h,self._pid,self._ga)
                        or read_identity(self._h,self._pid)!=before
                        or find_game_pid()!=self._pid):
                    if diagnostics is not None:diagnostics.update(error='process_binding_changed')
                    return None
                return {**fact, 'game_pid': self._pid, 'process_identity': before}
            except OSError as exc:
                self.last_error=str(exc)
                if diagnostics is not None:diagnostics.update(error=str(exc))
                return None
            finally:
                self._read_gate.release()

    def read(self, *, force: bool = False) -> UiStackSnap:
        # 同一进程的显示与导航不能交错更新读缓存。
        gate = getattr(self, "_read_gate", None)
        if gate is None:
            gate = self._read_gate = threading.RLock()
        t_gate = time.perf_counter()
        gate.acquire()
        lock_wait_ms = (time.perf_counter() - t_gate) * 1000.0
        try:
            binding = (self._h, self._pid, self._ga)
            before = read_identity(self._h, self._pid) if self._h and self._pid else None
            # The serialized read already verified this exact binding. Do not
            # repeat the same expensive identity I/O inside _ensure; the final
            # independent verification below still brackets the memory read.
            snap = self._read_current(force=force,
                verified_binding=binding if before is not None and all(binding) else None)
            if self._h and self._pid:
                after = read_identity(self._h, self._pid)
                if before is not None and before != after:
                    self.close()
                    return UiStackSnap(error='process_binding_changed', sampled_ts=time.time())
                snap.process_identity = after
            timing = dict(self.last_timing)
            timing["lock_wait_ms"] = round(lock_wait_ms, 1)
            timing["total_ms"] = round((time.perf_counter() - t_gate) * 1000.0, 1)
            self.last_timing = timing
            snap.cost_ms = timing
            if lock_wait_ms >= 50.0 or float(timing.get("inner_ms") or 0.0) >= 200.0:
                log.info(
                    "ui_stack_timing wait=%.1f ensure=%.1f stacks=%.1f state=%.1f "
                    "players=%.1f rescan=%.1f inner=%.1f total=%.1f force=%s cached=%s",
                    lock_wait_ms,
                    float(timing.get("ensure_ms") or 0.0),
                    float(timing.get("ensure_stacks_ms") or 0.0),
                    float(timing.get("state_ms") or 0.0),
                    float(timing.get("players_ms") or 0.0),
                    float(timing.get("overlay_rescan_ms") or 0.0),
                    float(timing.get("inner_ms") or 0.0),
                    float(timing.get("total_ms") or 0.0),
                    int(bool(force)),
                    int(timing.get("cached") or 0),
                )
            return snap
        except OSError as exc:
            self.close()
            self.last_error = str(exc)
            return UiStackSnap(error=str(exc), sampled_ts=time.time())
        finally:
            gate.release()

    def _read_current(self, *, force: bool = False, verified_binding=None) -> UiStackSnap:
        t0 = time.perf_counter()
        timing: dict[str, float] = {"cached": 0.0}
        if verified_binding is not None and verified_binding != (self._h, self._pid, self._ga):
            return UiStackSnap(error='process_binding_changed', sampled_ts=time.time())
        if verified_binding is None and not self._ensure():
            timing["ensure_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            timing["inner_ms"] = timing["ensure_ms"]
            self.last_timing = timing
            return UiStackSnap(
                line="—",
                error=self.last_error or "ensure_failed",
                ready=False,
                sampled_ts=time.time(),
                cost_ms=timing,
            )
        t1 = time.perf_counter()
        timing["ensure_ms"] = round((t1 - t0) * 1000.0, 1)
        now = t1
        if (
            not force
            and self._last_snap
            and self._last_snap.ready
            and self._last_snap.names
            and not self._last_snap.error
            and (now - self._last_read_ts) < _LINE_CACHE_S
        ):
            timing["cached"] = 1.0
            timing["inner_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            self.last_timing = timing
            return self._last_snap
        sampled_ts = time.time()
        assert self._h is not None and self._ga is not None
        self._ensure_state_stacks()
        t2 = time.perf_counter()
        timing["ensure_stacks_ms"] = round((t2 - t1) * 1000.0, 1)
        with self._lock:
            has_stacks = bool(self._state_stacks)
        packed = self._names_from_state_lists() if has_stacks else ([], None)
        if isinstance(packed, tuple) and len(packed) == 2:
            state_names, craft_cat = packed
        else:
            state_names, craft_cat = list(packed or []), None
        t3 = time.perf_counter()
        timing["state_ms"] = round((t3 - t2) * 1000.0, 1)
        player_names, stacking = self._players_for_read(state_names, force=force)
        t4 = time.perf_counter()
        timing["players_ms"] = round((t4 - t3) * 1000.0, 1)
        names = state_names + [n for n in player_names if n not in state_names]
        # bind=0 时 names 来自 players：禁止每拍把节流清零，否则堆扫(_SCAN_WALL_S=2s)
        # 与共享句柄 RPM / _read_gate 叠在导航 force 读上，表现为 ~2.3s。
        if (
            stack_left_overlay(player_names)
            and not stack_left_overlay(state_names)
        ):
            with self._lock:
                bound = bool(self._state_stacks)
            if bound:
                with self._lock:
                    self._next_stack_scan = 0.0
                self._ensure_state_stacks()
        timing["overlay_rescan_ms"] = round((time.perf_counter() - t4) * 1000.0, 1)
        map_identity = (self._active_map_listener(stacking)
                        if set(names) <= MAP_LISTENER_COMPANIONS else None)
        if map_identity and "MapUIInstance" not in names:
            names.append("MapUIInstance")

        def _finish(out: UiStackSnap) -> UiStackSnap:
            timing["inner_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            self.last_timing = timing
            out.cost_ms = timing
            return out

        if names:
            with self._lock:
                self._empty_streak = 0
            line = format_stack_line(names, self._labels)
            if self._last_snap and self._last_snap.ready and line == self._last_snap.line:
                line = self._last_snap.line
            self.last_error = None
            profile = self._active_profile(stacking) if "NewProfileUIInstance" in names else None
            craft = self._active_craft(stacking) if "CraftingHubPopupUIInstance" in names else None
            snap = UiStackSnap(
                names=names,
                line=line,
                n=len(names),
                stacking=stacking,
                ready=True,
                craft_category=craft_cat,
                craft_current_category=craft[0] if craft else None,
                craft_identity=craft[1] if craft else None,
                profile_category=profile[0] if profile else None,
                profile_identity=profile[1] if profile else None,
                map_identity=map_identity,
                map_view=self._active_map_view(stacking, map_identity) if map_identity else None,
                sampled_ts=sampled_ts,
                empty_confirmed=False,
            )
            self._last_snap = snap
            self._last_read_ts = now
            self._emit_stack_log(
                snap, state_names=state_names, player_names=player_names
            )
            return _finish(snap)
        with self._lock:
            busy = self._scan_busy
            done_n = int(self._scan_done_n)
        overlay = stack_left_overlay(player_names)
        players_empty = (not player_names) and (not overlay)
        still_scanning = bool(busy) or done_n < 1
        can_count = (
            (not busy)
            and done_n >= 1
            and not state_names
            and players_empty
        )
        if still_scanning:
            prev = self._last_snap
            if (
                prev is not None
                and prev.ready
                and prev.empty_confirmed
                and not prev.names
                and done_n >= 1
                and not overlay
            ):
                return _finish(prev)
            with self._lock:
                self._empty_streak = 0
            snap = UiStackSnap(
                names=[],
                line="扫描中…",
                n=0,
                error="stack_scan_pending",
                stacking=stacking,
                ready=False,
                sampled_ts=sampled_ts,
                empty_confirmed=False,
            )
        elif can_count:
            with self._lock:
                self._empty_streak = int(self._empty_streak) + 1
                streak = self._empty_streak
            if streak >= 2:
                self.last_error = None
                snap = UiStackSnap(
                    names=[],
                    line=format_stack_line([], self._labels),
                    n=0,
                    error="",
                    stacking=stacking,
                    ready=True,
                    sampled_ts=sampled_ts,
                    empty_confirmed=True,
                )
            else:
                snap = UiStackSnap(
                    names=[],
                    line="—",
                    n=0,
                    error="stack_names_empty",
                    stacking=stacking,
                    ready=False,
                    sampled_ts=sampled_ts,
                    empty_confirmed=False,
                )
        else:
            with self._lock:
                self._empty_streak = 0
            snap = UiStackSnap(
                names=[],
                line="—",
                n=0,
                error="stack_names_empty",
                stacking=stacking,
                ready=False,
                sampled_ts=sampled_ts,
                empty_confirmed=False,
            )
        self._last_snap = snap
        self._last_read_ts = now
        self._emit_stack_log(
            snap, state_names=state_names, player_names=player_names
        )
        return _finish(snap)

    def _emit_stack_log(
        self,
        snap: UiStackSnap,
        *,
        state_names: list[str],
        player_names: list[str],
    ) -> None:
        with self._lock:
            nbind = len(self._state_stacks)
        key = (tuple(snap.names), snap.ready, str(snap.error or ""), nbind)
        if key == getattr(self, "_log_key", None):
            return
        self._log_key = key
        log.info(
            "ui_stack ready=%s err=%s bind=%s names=%s state=%s players=%s",
            snap.ready,
            snap.error or "-",
            nbind,
            ",".join(snap.names) or "-",
            ",".join(state_names) or "-",
            ",".join(player_names) or "-",
        )
