"""IL2CPP 只读：界面开关 + 频道/地区 DisplayName + 战斗 UI 模式。

配置：`knowledge/ui/config/il2cpp_scene.json`
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm_assist.memory.il2cpp import (
    module_base,
    parse_int,
    read_il2cpp_string,
    resolve_klass,
    scan_klass_objects,
    scan_klass_objects_multi,
)
from mm_assist.memory.process import close_handle, find_game_pid, open_process, read_ptr, read_u8, read_u32, read_identity

log = logging.getLogger(__name__)

from mm_assist.knowledge_paths import SCENE_JSON

DEFAULT_REL = SCENE_JSON
from mm_assist.config import ROOT as _REPO_ROOT

_WARM_SEED_FILE = _REPO_ROOT / "logs" / "il2cpp_warm_seeds.json"

# 状态栏预热目标类型（已加载/总数 右上角口径）
# 栏上暖启只保地点 + 开图细分（A2）。inv/menu/dlg/Hud/战斗态不进满格。
WARM_BASE_LIMITS: dict[str, int] = {
    "GameSpaceAPI": 64,
    "ChannelIdentifierComponent": 4,
    "CurrentViewingMapUIComponent": 8,
}

# 开面板才创建的组件：后台周期性补扫（限额须盖过伪命中，真实例靠 DirtyRecord 筛）
_UI_REFRESH_LIMITS: dict[str, int] = {
    "MapUIVisibleComponent": 96,
    "MinimapUIVisibleComponent": 16,
    "InventoryVisibleDataComponent": 48,
    "DialoguePlayerComponent": 64,
    "CurrentViewingMapUIComponent": 48,
    "HudUIVisibleComponent": 48,
    "CombatUIComponent": 16,
    "ChattingUIStoredDataComponent": 16,
    "ChatDisplayDataComponent": 16,
    "EquipmentRepairListUIComponent": 32,
    # B4：菜单细分 UIState（开面板才新建；限额盖伪命中）
    "CraftingHubUIState": 16,
    "SideNavigationUIState": 16,
    "NewProfileUIState": 16,
}
_UI_REFRESH_S = 120.0  # 全量 UI 补扫极重；仅 allow_scan 路径调度，且拉长间隔
# lean/状态栏：不再堆扫 Hud/Inv/Dlg；仅地名空时后台补 GameSpace。

# 地图/人物成对采证：同步补扫（比 lean 更宽；墙钟见 _PAIR_UI_SYNC_WALL_S）
_PAIR_UI_SYNC_LIMITS: dict[str, int] = {
    "MapUIVisibleComponent": 192,
    "CurrentViewingMapUIComponent": 96,
    "NewProfileUIState": 96,
    "SideNavigationUIState": 48,
    "CraftingHubUIState": 48,
    "HudUIVisibleComponent": 64,
    "InventoryVisibleDataComponent": 64,
}
_PAIR_UI_SYNC_WALL_S = 8.0

# 种子复用时必须过活体筛；仅 klass 匹配会吃到幽灵 → lean 永不补扫 → 地区/界面拖几十秒
_LIVE_SEED_TYPES = frozenset(
    {
        "MapUIVisibleComponent",
        "CurrentViewingMapUIComponent",
        "EquipmentRepairListUIComponent",
        "NewProfileUIState",
    }
)

# 这些类型曾要求 falses>0，单实例真开面板会被判假
_PANEL_FLAG_TYPES = frozenset(
    {
        "DialoguePlayerComponent",
        "InventoryVisibleDataComponent",
        "MapUIVisibleComponent",
        "CombatUIComponent",
        "EquipmentRepairListUIComponent",
    }
)

# 堆扫会命中 TypeInfo 邻区 / 裸指针；真组件 DirtyRecord.frameCount 会随主循环涨
_KLASS_NEAR = 0x10000
_DIRTY_MIN_FRAME = 1000
_DIRTY_UPDATE_TYPES = frozenset(
    {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23, 24, 25, 26, 100, 101}
)
# DialoguePlayState: None=0 Loading=1 Loaded=2 Playing=3 Ended=4 Unloading=5 …
# dlg：IsActive + PlayState∈{1..4}。Ended 时选项 UI 仍常在（修理链）；None/Unload 不算。
# 手操 2026-07-30：对话开着 dlg=0 → Dirty 过严滤掉真实例；同 map 真值优先。
DIALOG_ACTIVE_PLAY_STATES = frozenset({1, 2, 3, 4})


def is_combat_hud_state(combat_state: int | None) -> bool:
    """T1/T2：战斗 HUD 以 CombatState∈{CombatReady,InCombat} 为准；勿单独信 cbt。"""
    return combat_state in (1, 2)


# MapType: Invalid=0 / World=1 / Town=2 / Dungeon=3 / MoonGate=4
MAP_TYPE_VALID = frozenset({1, 2, 3, 4})


def is_valid_viewing_map_type(mt: int | None) -> bool:
    """mt∈{World,Town,Dungeon,MoonGate} 视为正在看大地图层。"""
    return mt in MAP_TYPE_VALID


# B3 冻结：rpr/rmd（见 knowledge/ui/ui_mode_findings/B3_repair.md）
EQUIPMENT_REPAIR_LIST_IS_ACTIVE_OFF = 0x49  # EquipmentRepairList.isActive
EQUIPMENT_REPAIR_LIST_IS_MODAL_OFF = 0x4A  # EquipmentRepairList.isModalOpened


def is_repair_list_active_signal(*, is_active: int | None) -> bool:
    """rpr：EquipmentRepairList.isActive==1。"""
    return is_active == 1


def is_repair_modal_open_signal(*, is_modal_opened: int | None) -> bool:
    """rmd：EquipmentRepairList.isModalOpened==1。"""
    return is_modal_opened == 1


def is_repair_ui_signal(
    *,
    is_active: int | None = None,
    is_modal_opened: int | None = None,
) -> bool:
    """原始字节侧：rpr 或 rmd → 修理 UI 开。"""
    return is_repair_list_active_signal(is_active=is_active) or is_repair_modal_open_signal(
        is_modal_opened=is_modal_opened
    )


def is_repair_ui_open(
    *,
    repair_list_active: bool | None = None,
    repair_modal_open: bool | None = None,
) -> bool:
    """MemSceneInfo 侧：rpr 或 rmd 为真 → classify ui=repair。"""
    return bool(repair_list_active) or bool(repair_modal_open)


# B4 冻结：T7–T12 菜单细分（见 knowledge/ui/ui_mode_findings/B4_menu_subdivide.md）
CRAFTING_HUB_CATEGORY_TYPE_OFF = 0x14  # CraftingHubUIState.CategoryType
NEW_PROFILE_TOP_CATEGORY_OFF = 0x10  # NewProfileUIState.TopCategory
CRAFT_HUB_CATEGORY_VALID = frozenset({0, 1, 2, 3})
CRAFT_HUB_CATEGORY_SLUG = {
    0: "quick",  # QuickCrafting → T9
    1: "altering",  # Altering → T10
    2: "skill",  # SkillCrafting → T11
    3: "alchemy",  # Alchemy → T12
}
CRAFT_HUB_TAB_LABELS = {
    0: "快速制作",
    1: "加工",
    2: "制作",
    3: "炼金术",
}
PROFILE_TOP_CATEGORY_VALID = frozenset({0, 1, 2, 3, 4})


def is_valid_craft_hub_category(category: int | None) -> bool:
    """CraftingHubCategoryType∈{QuickCrafting,Altering,SkillCrafting,Alchemy}。"""
    return category in CRAFT_HUB_CATEGORY_VALID


def resolve_craft_hub_category_from_live(
    cats: list[int] | tuple[int, ...],
) -> tuple[int | None, str]:
    """从活实例 CategoryType 列表决议业务 hub（只认 1..3）。

    开製作时常多活同值；旧逻辑 ``len==1`` 会把同值共识打成 null（r33 成对帧）。
    返回 ``(category|None, tag)``：
    - ``ok_single`` / ``ok_consensus``：可读
    - ``hub_no_live`` / ``hub_no_altering`` / ``hub_conflict``：明确失败标签
    """
    live = [int(c) for c in cats]
    if not live:
        return None, "hub_no_live"
    altering = [c for c in live if c in (1, 2, 3)]
    if not altering:
        return None, "hub_no_altering"
    uniq = sorted(set(altering))
    if len(uniq) > 1:
        return None, "hub_conflict"
    cat = uniq[0]
    tag = "ok_single" if altering.count(cat) == 1 else "ok_consensus"
    return cat, tag


def craft_hub_tab_label(category: int | None) -> str | None:
    """製作 hub CategoryType → 状态栏 craft_tab 中文名。"""
    if category is None:
        return None
    try:
        cat = int(category)
    except (TypeError, ValueError):
        return None
    return CRAFT_HUB_TAB_LABELS.get(cat)


def craft_hub_detail_slug(category: int | None) -> str | None:
    """menu 细分：hub∈{1,2,3} → craft；hub=0 是地图（不算製作）。"""
    if category == 0:
        return None
    slug = CRAFT_HUB_CATEGORY_SLUG.get(category) if category is not None else None
    return f"craft_hub:{slug}" if slug else None


def resolve_profile_open_from_live_n(live_n: int | None) -> bool | None:
    """人物界面开态（2026-08-02 ui-discern 复检）。

    关态可残留 1 个幽灵实例；开态常见 live≥2（相对关态新增 addr）。
    - live≥2 → 开
    - live==0 → 关
    - live==1 → 不定（幽灵或干净开态）
    """
    if live_n is None:
        return None
    if live_n >= 2:
        return True
    if live_n <= 0:
        return False
    return None


def resolve_menu_subdivide_detail(
    *,
    craft_hub_category: int | None = None,
    profile_open: bool | None = None,
    side_nav_open: bool | None = None,
) -> str:
    """menu_open 且非地图时：Hub(1..3) 细分；nav/prf 已停用。"""
    slug = craft_hub_detail_slug(craft_hub_category)
    if slug:
        return slug
    return ""


def reconcile_map_open_signal(
    *,
    map_vis: bool | None,
    viewing_mt: int | None,
) -> tuple[bool | None, int | None]:
    """合入 MapUIVisible 与 CurrentViewing.MapType → (map_open, map_type)。

    2026-07-30 错绑回退（对齐 07-29 假阳规则）：
    - 须 map_vis=True 才开图；mt 只填类型，**不单独开图**
    - 粘住的 CurrentViewing.mt=1 曾经 map_vis=False 仍开 → map_align 钉死参数行
    """
    mt_out = viewing_mt if is_valid_viewing_map_type(viewing_mt) else None
    if map_vis is True:
        return True, mt_out
    if map_vis is False:
        return False, None
    return None, None


def dirty_record_ok(frame: int | None, update_type: int | None) -> bool:
    """DirtySource/DirtyRecord 是否像活组件（拒零帧与垃圾 updateType）。"""
    if frame is None or update_type is None:
        return False
    if frame < _DIRTY_MIN_FRAME:
        return False
    sb = update_type & 0xFF
    if sb > 127:
        sb -= 256
    return sb in _DIRTY_UPDATE_TYPES


def evaluate_panel_flag(
    trues: int, falses: int, *, panel: bool, majority: bool = False
) -> bool | None:
    """从活实例 0/1 计数判定开关。

    panel=True：任一为真即开（真值过多视为垃圾）。
    majority=True：须 trues>falses（修理等易假阳位）。
    """
    if trues == 0 and falses == 0:
        return None
    if majority:
        if trues > falses and trues <= 2:
            return True
        return False
    if panel:
        if trues > 0 and trues <= 2:
            return True
        if trues == 0:
            return False
        return False
    return bool(trues > 0)

MAP_TYPE_CN = {
    0: "无效",
    1: "大陆/世界",
    2: "城镇",
    3: "地下城",
    4: "月亮门",
}

COMBAT_STATE_CN = {
    0: "NonCombat",
    1: "CombatReady",
    2: "InCombat",
    3: "CombatEnd",
}

_DEFAULT = {
    "module": "GameAssembly.dll",
    "typeinfo_rva": {
        "CurrentViewingMapUIComponent": 0x794D9F8,
        "ChannelIdentifierComponent": 0x79884C8,
        "MapUIVisibleComponent": 0x7938680,
        "MinimapUIVisibleComponent": 0x799B640,
        "InventoryVisibleDataComponent": 0x79FE1E8,
        "DialoguePlayerComponent": 0x7A64BE8,
        "RadioDialoguePlayerComponent": 0x7A8FDB8,
        "EquipmentRepairUIComponent": 0x79514F8,
        "EquipmentRepairListUIComponent": 0x794F9B8,
        "CombatUIComponent": 0x7ABCC38,
        "ControlCharacterStatesComponent": 0x7B35D40,
        "ChannelSheetData": 0x798D320,
        "GameSpaceSheetData": 0x7AA0E48,
        "GameSpaceAPI": 0x7AA0770,
        "SkillButtonsUI": 0x7A97D78,
        "SkillButtonsUIComponent": 0x7A98450,
        "HudUIVisibleComponent": 0x79FA650,
        "SystemContainers": 0x79AC6C8,
        "ChattingUIStoredDataComponent": 0x79F9888,
        "ChatDisplayDataComponent": 0x79ECDB8,
        "CraftingHubUIState": 0x7931BD0,
        "SideNavigationUIState": 0x7A59D90,
        "NewProfileUIState": 0x7ACC4F0,
    },
    "offsets": {
        "CurrentViewingMapUI.MapType": 0x20,
        "CurrentViewingMapUI.GeoDataKey": 0x24,
        "CurrentViewingMapUI.ViewingChannelSheetId": 0x2C,
        "CurrentViewingMapUI.CurrentChannelSheetId": 0x3C,
        "ChannelIdentifier.ChannelSheetId": 0x10,
        "ChannelIdentifier.IsChangingChannel": 0x1F,
        "ChannelIdentifier.IsPlayerInNormalField": 0x18,
        "ChannelIdentifier.IsPlayerInDungeon": 0x14,
        "MapUIVisible.IsVisible": 0x10,
        "MinimapUIVisible.IsVisible": 0x28,
        "InventoryVisible.IsInventoryVisible": 0x10,
        "DialoguePlayer.IsActive": 0x48,
        "DialoguePlayer.IsRequested": 0x49,
        "DialoguePlayer.DialoguePlayState": 0x44,
        "RadioDialogue.isPlayingRadioDialogue": 0x24,
        "EquipmentRepairList.isShowReserved": 0x48,
        "EquipmentRepairList.isActive": 0x49,
        "EquipmentRepairList.isModalOpened": 0x4A,
        "CombatUI.IsUIVisible": 0x38,
        "ControlCharacterStates.CombatState": 0x4C,
        "ChannelSheetData.Id": 0x10,
        "ChannelSheetData.NexonServiceLogDisplayName": 0x18,
        "ChannelSheetData.GameSpace": 0x30,
        "GameSpaceSheetData.DisplayName": 0x18,
        "GameSpaceAPI.CurrentGameSpaceSheetId": 0x10,
        "GameSpaceAPI.CurrentGameSpaceSheetData": 0x18,
        "SkillButtonsUI.IsCombatModeLast": 0x28,
        "SkillButtonsUIComponent.UI": 0x18,
        "SkillButtonsUIComponent.IsHudUIVisible": 0x30,
        "SkillButtonsUIComponent.IsCombatUIVisible": 0x31,
        "HudUIVisible.HideAll": 0x28,
        "SystemContainers.ControlCharacterStatesComponent": 0x28C8,
        "ChattingUIStored.IsFocus": 0x40,
        "ChatDisplay.IsFullScreen": 0x18,
        "ChatDisplay.IsMiddleScreen": 0x24,
        "CraftingHubUIState.StateKey": 0x10,
        "CraftingHubUIState.CategoryType": 0x14,
        "NewProfileUIState.TopCategory": 0x10,
    },
}


def default_path(knowledge_root: Path | None = None) -> Path:
    if knowledge_root is None:
        return SCENE_JSON
    return Path(knowledge_root) / "ui" / "config" / "il2cpp_scene.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or default_path()
    cfg = {
        "module": _DEFAULT["module"],
        "typeinfo_rva": dict(_DEFAULT["typeinfo_rva"]),
        "offsets": dict(_DEFAULT["offsets"]),
    }
    if not p.is_file():
        return cfg
    raw = json.loads(p.read_text(encoding="utf-8-sig"))
    cfg["module"] = str(raw.get("module") or cfg["module"])
    for k, v in (raw.get("typeinfo_rva") or {}).items():
        cfg["typeinfo_rva"][k] = parse_int(v)
    for k, v in (raw.get("offsets") or {}).items():
        cfg["offsets"][k] = parse_int(v)
    if isinstance(raw.get('framework_type_tokens'), dict):
        cfg['framework_type_tokens'] = raw['framework_type_tokens']
    return cfg


def plausible_channel_sheet_id(cid: int | None) -> bool:
    """ChannelSheetId 应为小正整数；0 / 指针量级视为脏读。"""
    if cid is None:
        return False
    return 1 <= int(cid) <= 100_000


@dataclass
class MemSceneInfo:
    map_open: bool | None = None
    minimap_visible: bool | None = None
    dialog_active: bool | None = None  # 由 UI 栈派生
    menu_open: bool | None = None  # 由 UI 栈：含 HudUIStepState = 离开主界面
    ui_stack_names: tuple[str, ...] = field(default_factory=tuple)
    ui_stack_ready: bool = False
    ui_stack_error: str = ""
    memory_process: dict = field(default_factory=dict)
    ui_stack_ts: float = 0.0
    ui_stack_empty_ok: bool = False
    profile_category: int | None = None  # Current active profile VM; not readonly profile_tops.
    profile_identity: tuple[int, int] | None = None
    craft_current_category: int | None = None
    craft_identity: tuple[int, ...] | None = None
    map_identity: tuple[int, int, int] | None = None
    map_view: dict[str, Any] | None = None
    quest_open: bool | None = None  # 任务簿（暂弱信号，可空）
    chat_open: bool | None = None  # 聊天焦点或展开
    chat_focus: bool | None = None  # ChattingUIStored.IsFocus
    chat_fullscreen: bool | None = None  # ChatDisplay full/middle
    repair_list_active: bool | None = None  # EquipmentRepairList.isActive
    repair_modal_open: bool | None = None  # EquipmentRepairList.isModalOpened
    # B4：T7–T12 菜单细分（实机可标未验）
    craft_hub_category: int | None = None  # CraftingHubUIState.CategoryType（0..3；-1=None）
    side_nav_open: bool | None = None  # SideNavigationUIState 实例存活 → T7
    profile_open: bool | None = None  # NewProfile live 启发式；有效才上栏 prf=
    profile_live_n: int | None = None  # 活体数旁证（prf_n）
    # 对照用扩展读（不上四参；不改 menu/inv/dlg/stp）
    hud_ui_step: int | None = None
    hud_hide_all: int | None = None
    profile_tops: tuple[int, ...] = field(default_factory=tuple)
    hub_cats: tuple[int, ...] = field(default_factory=tuple)
    sn_live_n: int | None = None
    hub_live_n: int | None = None
    hud_live_n: int | None = None
    np_obj_n: int | None = None
    sn_obj_n: int | None = None
    hub_obj_n: int | None = None
    hud_obj_n: int | None = None
    in_combat: bool | None = None  # 兼容旧字段：= combat_mode_ui
    combat_mode_ui: bool | None = None  # 技能栏战斗/非战斗模式
    combat_state: int | None = None  # CombatState 枚举 0..3
    display_name: str = ""  # GameSpace.DisplayName 权威中文
    map_type: int | None = None
    map_type_name: str = ""
    channel_sheet_id: int | None = None
    viewing_channel_sheet_id: int | None = None
    geo_data_key: int | None = None
    in_dungeon: bool | None = None
    in_normal_field: bool | None = None
    changing_channel: bool | None = None
    source_hits: list[str] = field(default_factory=list)
    age_ms: float = 0.0
    status: str = "pending"

    @property
    def combat_state_name(self) -> str:
        if self.combat_state is None:
            return ""
        return COMBAT_STATE_CN.get(self.combat_state, str(self.combat_state))

    @property
    def ui_hint(self) -> str | None:
        """业务主标签线索（2026-07-30）：dlg → 背包+menu? → menu+hub0地图 → menu → hud。"""
        from mm_assist.memory.readers.ui_stack import stack_inv

        if self.dialog_active:
            return "dialog"
        inv = stack_inv(list(self.ui_stack_names or ()))
        if inv and self.menu_open is None:
            return "inventory"
        if self.menu_open is True and self.craft_hub_category == 0:
            mt = self.map_type
            if mt == 1:
                return "map_continent"
            if mt == 2:
                return "map_town"
            return "map_region"
        if self.menu_open:
            return "menu"
        if self.menu_open is False:
            return "hud"
        if inv:
            return "inventory"
        return None

    @property
    def geo_hint(self) -> str | None:
        """调试摘要；面板地理主文案用 display_name。"""
        parts: list[str] = []
        if self.display_name:
            parts.append(self.display_name)
        if self.map_type_name:
            parts.append(f"图:{self.map_type_name}")
        if self.channel_sheet_id is not None:
            parts.append(f"ch={self.channel_sheet_id}")
        if self.in_dungeon:
            parts.append("地下城")
        elif self.in_normal_field:
            parts.append("野外")
        if self.changing_channel:
            parts.append("换频中")
        return " · ".join(parts) if parts else None


class Il2CppSceneReader:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_path()
        self._cfg = load_config(self.path)
        self._pid: int | None = None
        self._h = None
        self._ga: int | None = None
        self._klasses: dict[str, int] = {}
        self._objs: dict[str, list[int]] = {}
        self._last_err = ""
        self._last_craft_hub_diag: dict[str, Any] = {
            "live_cats": [],
            "tag": "hub_no_live",
            "category": None,
        }
        self._lock = threading.RLock()
        self._resolving = False
        self._pause_resolve = False  # 预热期间禁止并发堆扫（否则易 OOM/空异常）
        self._bg: threading.Thread | None = None
        self._closed = False
        self._cache: MemSceneInfo | None = None
        self._resolved = False
        self._sheet_cache_ch: int | None = None
        self._sheet_obj: int | None = None
        self._display_name_cache: str = ""
        self._sheet_miss_ch: int | None = None  # 已扫过且未命中，避免每帧全堆扫
        self._next_ui_refresh = 0.0
        self._ui_refreshing = False
        self._ui_refresh_n = 0
        self._next_map_sync = 0.0
        self._map_rescanning = False
        self._on_warm_progress: Any | None = None

    @property
    def last_error(self) -> str:
        return self._last_err

    @property
    def resolving(self) -> bool:
        return self._resolving

    def set_warm_progress(self, fn: Any | None) -> None:
        self._on_warm_progress = fn

    def _emit_warm_progress(self) -> None:
        fn = self._on_warm_progress
        if fn is None:
            return
        try:
            fn(*self.warm_load_progress())
        except Exception as exc:
            log.debug("warm progress: %s", exc)

    def close(self) -> None:
        self._closed = True
        bg = self._bg
        if bg is not None and bg.is_alive() and threading.current_thread() is not bg:
            bg.join(timeout=2.0)
        with self._lock:
            if self._h is not None:
                try:
                    close_handle(self._h)
                except Exception:
                    pass
            self._h = None
            self._pid = None
            self._ga = None
            self._klasses.clear()
            self._objs.clear()
            self._bg = None
            self._cache = None
            self._sheet_cache_ch = None
            self._sheet_obj = None
            self._display_name_cache = ""
            self._sheet_miss_ch = None
            self._resolved = False

    def _ensure_proc(self, hwnd: int | None = None) -> bool:
        if self._h is not None and self._pid:
            try:
                read_identity(self._h, self._pid)
            except OSError as exc:
                self.close()
                self._last_err = str(exc)
                return False
        pid = find_game_pid(hwnd=hwnd)
        if not pid:
            self._last_err = "process_not_found"
            return False
        if self._h is None or self._pid != pid:
            with self._lock:
                if self._h is not None:
                    try:
                        close_handle(self._h)
                    except Exception:
                        pass
                try:
                    self._h = open_process(pid)
                    self._pid = pid
                    self._closed = False
                    self._ga = None
                    self._klasses.clear()
                    self._objs.clear()
                    self._resolved = False
                    self._sheet_cache_ch = None
                    self._sheet_obj = None
                    self._display_name_cache = ""
                    self._sheet_miss_ch = None
                except OSError as e:
                    self._last_err = f"open_failed:{e}"
                    return False
        return True

    def _resolve_klasses(self) -> bool:
        assert self._h is not None and self._pid is not None
        ga = module_base(self._pid, str(self._cfg["module"]))
        if not ga:
            self._last_err = "gameassembly_not_found"
            return False
        self._ga = ga
        self._klasses.clear()
        for name, rva in (self._cfg.get("typeinfo_rva") or {}).items():
            k = resolve_klass(self._h, ga, int(rva))
            if k:
                self._klasses[name] = k
        return bool(self._klasses)

    def _off(self, key: str) -> int:
        return int(self._cfg["offsets"][key])

    def _collect(self, type_name: str, *, max_hits: int = 20) -> list[int]:
        assert self._h is not None
        klass = self._klasses.get(type_name)
        if not klass:
            return []
        out: list[int] = []
        for obj in scan_klass_objects(
            self._h, klass, max_hits=max_hits, should_stop=lambda: self._closed
        ):
            if read_ptr(self._h, obj) == klass:
                out.append(obj)
        self._objs[type_name] = out
        return out

    def _collect_many(self, wanted: dict[str, int], *, merge: bool = False) -> None:
        """一次堆遍历取多类实例（页只读一次）。merge=True 时并入仍存活的旧指针。"""
        assert self._h is not None
        klasses = {n: self._klasses[n] for n in wanted if n in self._klasses}
        if not klasses or self._closed:
            return

        def _on_first(name: str, objs: list[int]) -> None:
            klass = klasses.get(name)
            if not klass or self._h is None:
                return
            fresh = [o for o in objs if read_ptr(self._h, o) == klass]
            if not fresh:
                return
            with self._lock:
                if not merge:
                    self._objs[name] = fresh
                else:
                    seen: set[int] = set(self._objs.get(name) or [])
                    merged = list(self._objs.get(name) or [])
                    for o in fresh:
                        if o in seen:
                            continue
                        seen.add(o)
                        merged.append(o)
                    self._objs[name] = merged
            self._emit_warm_progress()

        res = scan_klass_objects_multi(
            self._h,
            klasses,
            max_hits={n: wanted[n] for n in klasses},
            should_stop=lambda: self._closed,
            on_first_hit=_on_first,
        )
        for name, objs in res.items():
            klass = klasses[name]
            fresh = [o for o in objs if read_ptr(self._h, o) == klass]
            if not merge:
                self._objs[name] = fresh
                continue
            seen: set[int] = set(fresh)
            merged = list(fresh)
            for o in self._objs.get(name) or []:
                if o in seen:
                    continue
                if read_ptr(self._h, o) != klass:
                    continue
                seen.add(o)
                merged.append(o)
            self._objs[name] = merged

    def _flag_or(self, type_name: str, off_key: str) -> bool | None:
        assert self._h is not None
        klass = self._klasses.get(type_name)
        objs = self._objs.get(type_name) or []
        if not klass or not objs:
            return None
        off = self._off(off_key)
        trues = falses = 0
        alive: list[int] = []
        for obj in objs:
            if read_ptr(self._h, obj) != klass:
                continue
            alive.append(obj)
            if abs(obj - klass) < _KLASS_NEAR:
                continue
            if not self._is_live_component(type_name, obj):
                continue
            v = read_u8(self._h, obj + off)
            if v is None or v not in (0, 1):
                continue
            if v:
                trues += 1
            else:
                falses += 1
        self._objs[type_name] = alive
        return evaluate_panel_flag(
            trues,
            falses,
            panel=type_name in _PANEL_FLAG_TYPES,
            majority=type_name
            in (
                "EquipmentRepairListUIComponent",
            ),
        )

    def _is_live_component(self, type_name: str, obj: int) -> bool:
        """拒 TypeInfo 邻区伪命中；Map/Dlg/Hud 靠 DirtyRecord；开图 Dirty 可放宽。"""
        assert self._h is not None
        if type_name == "MapUIVisibleComponent":
            # dump: IsVisible@0x10 + 过渡/堆叠 bool@0x11..0x14；DirtySource@0x18
            vis = read_u8(self._h, obj + self._off("MapUIVisible.IsVisible"))
            if vis is None or vis not in (0, 1):
                return False
            for o in range(0x11, 0x15):
                v = read_u8(self._h, obj + o)
                if v is None or v not in (0, 1):
                    return False
            fr = read_u32(self._h, obj + 0x18)
            ut = read_u32(self._h, obj + 0x1C)
            # 拒未初始化幽灵（实机：vis=1 但 fr=ut=0）
            if (fr or 0) == 0 and (ut or 0) == 0:
                return False
            dirty_ok = dirty_record_ok(fr, ut)
            # B2：开图时常 IsVisible=1 但 Dirty 未涨；真值优先防漏
            if vis == 1:
                return True
            return dirty_ok
        if type_name == "DialoguePlayerComponent":
            for o in (0x48, 0x49, 0x4A):
                v = read_u8(self._h, obj + o)
                if v is None or v not in (0, 1):
                    return False
            st = read_u32(self._h, obj + self._off("DialoguePlayer.DialoguePlayState"))
            if st is None or st > 7:
                return False
            a = read_u8(self._h, obj + self._off("DialoguePlayer.IsActive"))
            fr = read_u32(self._h, obj + 0x68)
            ut = read_u32(self._h, obj + 0x6C)
            if (fr or 0) == 0 and (ut or 0) == 0 and a != 1:
                return False
            dirty_ok = dirty_record_ok(fr, ut)
            # 对话中 Dirty 常不涨；真值优先防漏（对齐 MapUIVisible IsVisible=1）
            if a == 1 or st in DIALOG_ACTIVE_PLAY_STATES:
                return True
            return dirty_ok
        if type_name == "HudUIVisibleComponent":
            hide = read_u8(self._h, obj + self._off("HudUIVisible.HideAll"))
            if hide is None or hide not in (0, 1):
                return False
            step = read_u32(self._h, obj + 0x10)
            if step is None:
                return False
            # HudUIStep as i32
            si = step if step < 0x80000000 else step - 0x100000000
            if si < -1 or si > 23:
                return False
            return dirty_record_ok(
                read_u32(self._h, obj + 0x20), read_u32(self._h, obj + 0x24)
            )
        if type_name == "CurrentViewingMapUIComponent":
            # dump: mapTypeDirtySource@0x10；MapType@0x20
            mt = read_u32(self._h, obj + self._off("CurrentViewingMapUI.MapType"))
            if mt is None or mt > 4:
                return False
            vc = read_u32(
                self._h, obj + self._off("CurrentViewingMapUI.ViewingChannelSheetId")
            )
            cc = read_u32(
                self._h, obj + self._off("CurrentViewingMapUI.CurrentChannelSheetId")
            )
            # 频道 SheetId 实机通常 < 几千；拒 32760 一类垃圾
            if vc is not None and vc > 10000:
                return False
            if cc is not None and cc > 10000:
                return False
            dirty_ok = dirty_record_ok(
                read_u32(self._h, obj + 0x10), read_u32(self._h, obj + 0x14)
            )
            # 粘住幽灵：mt=1 但 vc=cc=0 且 Dirty 失败（曾单独开图钉死参数行）
            if is_valid_viewing_map_type(mt):
                if (vc or 0) == 0 and (cc or 0) == 0 and not dirty_ok:
                    return False
                return True
            # Invalid(0) 仍靠 Dirty，避免裸指针伪命中
            return dirty_ok
        if type_name == "InventoryVisibleDataComponent":
            v = read_u8(
                self._h, obj + self._off("InventoryVisible.IsInventoryVisible")
            )
            # T5 缓：无 DirtySource；闪烁 1→? 勿当真（见 B2_inv_map.md）
            return v in (0, 1)
        if type_name == "MinimapUIVisibleComponent":
            v = read_u8(self._h, obj + self._off("MinimapUIVisible.IsVisible"))
            return v in (0, 1)
        if type_name == "CombatUIComponent":
            v = read_u8(self._h, obj + self._off("CombatUI.IsUIVisible"))
            return v in (0, 1)
        if type_name == "EquipmentRepairListUIComponent":
            for o in (
                self._off("EquipmentRepairList.isShowReserved"),
                self._off("EquipmentRepairList.isActive"),
                self._off("EquipmentRepairList.isModalOpened"),
            ):
                v = read_u8(self._h, obj + o)
                if v is None or v not in (0, 1):
                    return False
            return True
        # B4 UIState：无 DirtyRecord；靠字段合法域 + 非 TypeInfo 邻区
        if type_name == "CraftingHubUIState":
            cat = read_u32(self._h, obj + self._off("CraftingHubUIState.CategoryType"))
            if cat is None:
                return False
            # i32：None=-1 … Alchemy=3
            si = cat if cat < 0x80000000 else cat - 0x100000000
            return -1 <= si <= 3
        if type_name == "NewProfileUIState":
            top = read_u32(self._h, obj + self._off("NewProfileUIState.TopCategory"))
            if top is None:
                return False
            return top in PROFILE_TOP_CATEGORY_VALID
        if type_name == "SideNavigationUIState":
            # dump：空 payload；开态 = State 对象存活（见 B4_menu_subdivide.md）
            return True
        return True

    def _read_ui_state_present(self, type_name: str) -> bool | None:
        """IUIState 实例存活启发式：klass 匹配且非 TypeInfo 邻区。"""
        assert self._h is not None
        klass = self._klasses.get(type_name)
        if not klass:
            return None
        if type_name not in self._objs:
            return None
        objs = self._objs.get(type_name) or []
        alive: list[int] = []
        for obj in objs:
            if read_ptr(self._h, obj) != klass:
                continue
            alive.append(obj)
            if abs(obj - klass) < _KLASS_NEAR:
                continue
            if not self._is_live_component(type_name, obj):
                continue
            self._objs[type_name] = alive
            return True
        self._objs[type_name] = alive
        return False

    def _count_ui_state_live(self, type_name: str) -> int | None:
        """合法活实例数；类型未解析/未扫 → None。"""
        assert self._h is not None
        klass = self._klasses.get(type_name)
        if not klass:
            return None
        if type_name not in self._objs:
            return None
        objs = self._objs.get(type_name) or []
        alive: list[int] = []
        live_n = 0
        for obj in objs:
            if read_ptr(self._h, obj) != klass:
                continue
            alive.append(obj)
            if abs(obj - klass) < _KLASS_NEAR:
                continue
            if not self._is_live_component(type_name, obj):
                continue
            live_n += 1
        self._objs[type_name] = alive
        return live_n

    def _read_craft_hub_category(self) -> int | None:
        """CraftingHubUIState.CategoryType；无业务 hub 则 None。

        cat=0 与地图 hub=0 冲突，业务只认 1..3。
        多活同值取共识（开製作常堆多实例）；异值冲突 → None + tag。
        """
        assert self._h is not None
        empty = {
            "live_cats": [],
            "tag": "hub_no_live",
            "category": None,
        }
        klass = self._klasses.get("CraftingHubUIState")
        if not klass:
            self._last_craft_hub_diag = dict(empty)
            return None
        if "CraftingHubUIState" not in self._objs:
            self._last_craft_hub_diag = dict(empty)
            return None
        objs = self._objs.get("CraftingHubUIState") or []
        off = self._off("CraftingHubUIState.CategoryType")
        alive: list[int] = []
        live_cats: list[int] = []
        for obj in objs:
            if read_ptr(self._h, obj) != klass:
                continue
            alive.append(obj)
            if abs(obj - klass) < _KLASS_NEAR:
                continue
            if not self._is_live_component("CraftingHubUIState", obj):
                continue
            cat = read_u32(self._h, obj + off)
            if cat is None:
                continue
            si = cat if cat < 0x80000000 else cat - 0x100000000
            if -1 <= si <= 3:
                live_cats.append(int(si))
        self._objs["CraftingHubUIState"] = alive
        resolved, tag = resolve_craft_hub_category_from_live(live_cats)
        self._last_craft_hub_diag = {
            "live_cats": list(live_cats),
            "tag": tag,
            "category": resolved,
        }
        return resolved

    def _read_viewing_map_type(self) -> int | None:
        assert self._h is not None
        klass = self._klasses.get("CurrentViewingMapUIComponent")
        objs = self._objs.get("CurrentViewingMapUIComponent") or []
        if not klass or not objs:
            return None
        off = self._off("CurrentViewingMapUI.MapType")
        best: int | None = None
        for obj in objs:
            if read_ptr(self._h, obj) != klass:
                continue
            if abs(obj - klass) < _KLASS_NEAR:
                continue
            if not self._is_live_component("CurrentViewingMapUIComponent", obj):
                continue
            mt = read_u32(self._h, obj + off)
            if mt is None or mt > 4:
                continue
            if best is None or mt > best:
                best = mt
        return best

    def _maybe_refresh_ui_objs(self, *, lean: bool = False) -> None:
        """后台补扫。lean=只补地点；非 lean 才走 allow_scan 全量 UI 限额。"""
        if (not self._resolved or self._closed or self._ui_refreshing
                or self._pause_resolve or self._resolving or self._map_rescanning):
            return
        now = time.monotonic()
        if now < self._next_ui_refresh:
            return
        if lean:
            prev = self._cache
            dn_cache = (self._display_name_cache or "").strip()
            if prev is not None and not (prev.display_name or "").strip() and dn_cache:
                prev.display_name = dn_cache
                if "display_name" not in (prev.source_hits or []):
                    prev.source_hits = list(prev.source_hits or []) + ["display_name*"]
                self._next_ui_refresh = now + 4.0
                return
            if (prev is None or not (prev.display_name or "").strip()) and not dn_cache:
                self._ui_refreshing = True

                def _geo_work() -> None:
                    try:
                        with self._lock:
                            if self._closed or self._h is None:
                                return
                        self._ensure_display_name(max_wall_s=6.0)
                        with self._lock:
                            if self._cache is not None and (
                                self._display_name_cache or ""
                            ).strip():
                                self._cache.display_name = self._display_name_cache
                                if "display_name" not in (
                                    self._cache.source_hits or []
                                ):
                                    self._cache.source_hits = list(
                                        self._cache.source_hits or []
                                    ) + ["display_name*"]
                    except Exception as exc:
                        log.debug("lean geo rescue: %s", exc)
                    finally:
                        self._ui_refreshing = False
                        self._next_ui_refresh = time.monotonic() + 4.0

                threading.Thread(
                    target=_geo_work, name="il2cpp-lean-geo", daemon=True
                ).start()
            else:
                self._next_ui_refresh = now + 4.0
            return
        interval = _UI_REFRESH_S
        limits = _UI_REFRESH_LIMITS
        self._next_ui_refresh = now + interval
        self._ui_refreshing = True

        def _work() -> None:
            t0 = time.perf_counter()
            try:
                with self._lock:
                    if self._closed or self._h is None:
                        return
                    h = self._h
                    klasses = {
                        n: self._klasses[n]
                        for n in limits
                        if n in self._klasses
                    }
                    old_objs = {n: list(self._objs.get(n) or []) for n in klasses}
                if not klasses or self._closed:
                    return
                res = scan_klass_objects_multi(
                    h,
                    klasses,
                    max_hits={n: limits[n] for n in klasses},
                    should_stop=lambda: self._closed,
                    max_wall_s=None,
                )
                merged_map: dict[str, list[int]] = {}
                for name, objs in res.items():
                    klass = klasses[name]
                    fresh = [o for o in objs if read_ptr(h, o) == klass]
                    seen: set[int] = set(fresh)
                    merged = list(fresh)
                    for o in old_objs.get(name) or []:
                        if o in seen:
                            continue
                        if read_ptr(h, o) != klass:
                            continue
                        seen.add(o)
                        merged.append(o)
                    merged_map[name] = merged
                with self._lock:
                    if self._closed:
                        return
                    for name, merged in merged_map.items():
                        self._objs[name] = merged
                    self._ui_refresh_n += 1
                self._emit_warm_progress()
                log.info(
                    "ui objs refresh #%d lean=%s in %.0fms",
                    self._ui_refresh_n,
                    lean,
                    (time.perf_counter() - t0) * 1000.0,
                )
            except Exception as exc:
                log.debug("ui objs refresh: %s", exc)
            finally:
                self._ui_refreshing = False
                self._next_ui_refresh = time.monotonic() + interval

        threading.Thread(target=_work, name="il2cpp-ui-refresh", daemon=True).start()

    def _flag_first(self, type_name: str, off_key: str) -> bool | None:
        """只读存活实例上的 0/1 标志；多实例取多数（忽略非布尔垃圾值）。"""
        assert self._h is not None
        klass = self._klasses.get(type_name)
        objs = self._objs.get(type_name) or []
        if not klass or not objs:
            return None
        off = self._off(off_key)
        alive: list[int] = []
        zeros = ones = 0
        for obj in objs:
            if read_ptr(self._h, obj) != klass:
                continue
            alive.append(obj)
            if abs(obj - klass) < _KLASS_NEAR:
                continue
            v = read_u8(self._h, obj + off)
            if v is None or v not in (0, 1):
                continue
            if v:
                ones += 1
            else:
                zeros += 1
        self._objs[type_name] = alive
        if zeros == 0 and ones == 0:
            return None
        return ones > zeros

    def _first_obj(self, type_name: str) -> int | None:
        assert self._h is not None
        klass = self._klasses.get(type_name)
        objs = self._objs.get(type_name) or []
        if not klass:
            return None
        for obj in objs:
            if read_ptr(self._h, obj) == klass:
                return obj
        return None

    def _best_channel_obj(self) -> tuple[int | None, int | None]:
        """在多个 ChannelIdentifier 候选里挑可信 ChannelSheetId。

        堆扫常扫到死实例：SheetId=0 或指针量级脏值。返回 (obj, sheet_id)。
        """
        assert self._h is not None
        klass = self._klasses.get("ChannelIdentifierComponent")
        objs = self._objs.get("ChannelIdentifierComponent") or []
        if not klass:
            return None, None
        off = self._off("ChannelIdentifier.ChannelSheetId")
        best_obj: int | None = None
        best_cid: int | None = None
        for obj in objs:
            if read_ptr(self._h, obj) != klass:
                continue
            cid = read_u32(self._h, obj + off)
            if not plausible_channel_sheet_id(cid):
                continue
            # 优先沿用上次成功的 sheet，避免候选抖动
            if (
                self._cache is not None
                and self._cache.channel_sheet_id is not None
                and int(cid) == int(self._cache.channel_sheet_id)
            ):
                return obj, int(cid)
            if best_cid is None:
                best_obj, best_cid = obj, int(cid)
        return best_obj, best_cid

    def _resolve_all_objs(self) -> None:
        wanted = {
            "MapUIVisibleComponent": 96,
            "MinimapUIVisibleComponent": 16,
            "InventoryVisibleDataComponent": 48,
            "DialoguePlayerComponent": 64,
            "ChannelIdentifierComponent": 8,
            "CurrentViewingMapUIComponent": 48,
            "SkillButtonsUI": 8,
            "SkillButtonsUIComponent": 12,
            "HudUIVisibleComponent": 48,
            "ControlCharacterStatesComponent": 8,
            "SystemContainers": 4,
            "CombatUIComponent": 16,
            "ChattingUIStoredDataComponent": 16,
            "ChatDisplayDataComponent": 16,
            "EquipmentRepairListUIComponent": 32,
            "GameSpaceAPI": 64,
            "CraftingHubUIState": 16,
            "SideNavigationUIState": 16,
            "NewProfileUIState": 16,
        }
        self._emit_warm_progress()
        self._collect_many(wanted)
        # GameSpaceAPI：从合并扫结果里筛带合法 DisplayName 的实例（不再二次全堆扫）
        if "GameSpaceAPI" in self._klasses and not self._closed:
            self._prime_gamespace_api(rescan=False)
        self._resolved = True
        self._next_ui_refresh = time.monotonic() + _UI_REFRESH_S

    def _has_live_obj(self, name: str) -> bool:
        """进度用：有活实例才算扫到。光有 klass 指针（幽灵/种子假阳）不算。"""
        if self._h is None:
            return False
        klass = self._klasses.get(name)
        objs = self._objs.get(name) or []
        if not klass:
            return False
        # 开图才有实例；暖启扫过且 0 命中也算这一格完成，避免主界面永远 2/3
        if name == "CurrentViewingMapUIComponent" and self._resolved and not objs:
            return True
        if not objs:
            return False
        for obj in objs:
            if read_ptr(self._h, obj) != klass:
                continue
            if abs(obj - klass) < _KLASS_NEAR:
                continue
            if name == "GameSpaceAPI":
                if self._gamespace_seed_ok(obj):
                    return True
                continue
            if name in _LIVE_SEED_TYPES:
                try:
                    if self._is_live_component(name, obj):
                        return True
                except Exception:
                    continue
                continue
            return True
        return False

    def _params_incomplete(self) -> bool:
        """地名未就绪时右上角不得报满格。"""
        cache = self._cache
        if cache is None:
            return True
        if not (cache.display_name or "").strip():
            return True
        return False

    def warm_load_progress(
        self, *, extra_names: tuple[str, ...] | list[str] = ()
    ) -> tuple[int, int]:
        """预热进度：活体类型数 / 目标数。参数未齐时不满格。"""
        names = list(WARM_BASE_LIMITS.keys())
        for n in extra_names:
            if n and n not in names:
                names.append(str(n))
        total = len(names)
        if total <= 0:
            return 0, 0
        loaded = sum(1 for n in names if self._has_live_obj(n))
        if self._params_incomplete() and loaded >= total:
            loaded = max(0, total - 1)
        return loaded, total

    def warm_bundle(
        self,
        hwnd: int | None = None,
        *,
        extra_limits: dict[str, int] | None = None,
        extra_klasses: dict[str, int] | None = None,
        max_wall_s: float = 4.5,
        max_regions: int = 0,
        on_progress: Any | None = None,
    ) -> tuple[MemSceneInfo | None, dict[str, list[int]]]:
        """一次堆遍历：场景 UI 类型 + 调用方附加类型。返回 (场景快照, 全部命中)。

        默认墙钟 ≤4.5s。优先复用上次成功的对象种子（同 PID+GA），再补扫缺失类型。
        每类收集多个候选再 klass 校验，避免 max_hits=1 吃到假阳就停。
        on_progress(loaded, total)：可选，供状态栏刷新已加载/总数。
        """
        if not self._ensure_proc(hwnd=hwnd):
            return None, {}
        if not self._resolve_klasses():
            return None, {}
        assert self._h is not None and self._pid is not None and self._ga is not None
        wanted = dict(WARM_BASE_LIMITS)
        if extra_limits:
            wanted.update(extra_limits)
        if extra_klasses:
            for name, klass in extra_klasses.items():
                if klass and name not in self._klasses:
                    self._klasses[name] = klass
        klasses = {n: self._klasses[n] for n in wanted if n in self._klasses}
        if not klasses:
            return None, {}

        extra_names = tuple(n for n in wanted if n not in WARM_BASE_LIMITS)

        def _prog() -> None:
            self._emit_warm_progress()
            if on_progress is None:
                return
            try:
                on_progress(*self.warm_load_progress(extra_names=extra_names))
            except Exception as exc:
                log.debug("warm on_progress: %s", exc)

        wall = max(0.5, float(max_wall_s))
        t0 = time.perf_counter()
        seeded = self._apply_warm_seeds(klasses)
        # An old saved place name must not suppress current root discovery.
        if 'GameSpaceAPI' in klasses:
            fixed = self._framework_gamespace_api()
            if fixed and self._gamespace_seed_ok(fixed):
                klasses['GameSpaceAPI'] = self._klasses['GameSpaceAPI']
                if not self._objs.get('GameSpaceAPI'):
                    seeded += 1
                self._objs['GameSpaceAPI'] = [fixed]
        _prog()
        missing = {n: wanted[n] for n in klasses if not self._objs.get(n)}
        res: dict[str, list[int]] = {n: list(self._objs.get(n) or []) for n in klasses}

        def _adopt(scanned: dict[str, list[int]]) -> None:
            for name, objs in scanned.items():
                klass = klasses[name]
                fresh = [
                    o
                    for o in objs
                    if read_ptr(self._h, o) == klass and abs(o - klass) >= _KLASS_NEAR
                ]
                if name == "GameSpaceAPI":
                    fresh = [o for o in fresh if self._gamespace_seed_ok(o)]
                elif name in _LIVE_SEED_TYPES:
                    fresh = [o for o in fresh if self._is_live_component(name, o)]
                if fresh:
                    self._objs[name] = fresh
                    res[name] = fresh
            _prog()

        def _publish_early() -> None:
            """关键类型到手就写 cache，状态栏不必等整轮 warm 结束。"""
            self._resolved = True
            info0 = self._read_once()
            if not (info0.display_name or "").strip() and self._display_name_cache:
                info0.display_name = self._display_name_cache
                if "display_name" not in (info0.source_hits or []):
                    info0.source_hits = list(info0.source_hits or []) + ["display_name*"]
            info0 = self._sticky_merge(info0)
            self._cache = info0
            self._last_err = "" if info0.status == "ok" else info0.status

        # 冷启动无种子：默认 6s（原 12s 太慢）；MM_STATE_WARM_COLD_S 可调
        if seeded == 0 and missing:
            try:
                cold = float(__import__("os").environ.get("MM_STATE_WARM_COLD_S", "6"))
            except ValueError:
                cold = 6.0
            wall = max(wall, max(4.0, cold))
            log.info("warm cold-start wall=%.1fs (no seeds)", wall)

        if missing:
            # 一轮扫齐全部缺失类型（multi-needle 同遍堆）；勿再拆 Hud/其余两轮串行
            remain = max(0.0, wall - (time.perf_counter() - t0))
            # 地区缺失时为单类型续扫留预算，不在每轮都重扫同一批大区。
            if "GameSpaceAPI" in missing:
                remain = min(remain, wall * 0.4)

            def _warm_hit_ok(name: str, obj: int) -> bool:
                # Hud：只有活体计入 max_hits，否则热区幽灵占满永远扫不到冷区真例
                if name == "HudUIVisibleComponent":
                    k = klasses.get(name)
                    if not k or read_ptr(self._h, obj) != k:
                        return False
                    if abs(obj - k) < _KLASS_NEAR:
                        return False
                    return self._is_live_component(name, obj)
                # GameSpace：须能读出 DisplayName，否则幽灵占满限额 → 地区永久「未知」
                if name == "GameSpaceAPI":
                    return self._gamespace_seed_ok(obj)
                return True

            scanned = scan_klass_objects_multi(
                self._h,
                {n: klasses[n] for n in missing},
                max_hits={n: missing[n] for n in missing},
                chunk=4 * 1024 * 1024,
                should_stop=lambda: self._closed,
                max_wall_s=remain,
                max_regions=(int(max_regions) if max_regions and max_regions > 0 else None),
                hit_filter=_warm_hit_ok,
                on_first_hit=lambda name, objs: _adopt({name: objs}),
            )
            _adopt(scanned)
            _prog()
            # 地点或开图类型到手就上栏，不必等整轮收尾
            if any(
                self._objs.get(n)
                for n in (
                    "GameSpaceAPI",
                    "ChannelIdentifierComponent",
                    "CurrentViewingMapUIComponent",
                )
            ):
                _publish_early()
        else:
            log.info("warm seeds hit all %d types (skip heap)", len(klasses))
            _publish_early()

        remain = wall - (time.perf_counter() - t0)
        # 种子指针常含幽灵：强制补 DisplayName（界面改信 UI 栈，不再压活 Hud）
        need_geo = not (self._display_name_cache or "").strip()
        if not self._closed and need_geo and remain > 0:
            self._ensure_display_name(max_wall_s=remain)
            _prog()
        self._resolved = True
        info = self._read_once()
        if not (info.display_name or "").strip() and self._display_name_cache:
            info.display_name = self._display_name_cache
            if "display_name" not in (info.source_hits or []):
                info.source_hits = list(info.source_hits or []) + ["display_name*"]
        info = self._sticky_merge(info)
        self._cache = info
        weak = not (info.display_name or "").strip()
        self._last_err = "" if info.status == "ok" or info.source_hits else (
            info.status or self._last_err
        )
        # 弱信号：尽快再补扫，勿等 120s 全量 refresh
        self._next_ui_refresh = time.monotonic() + (2.0 if weak else _UI_REFRESH_S)
        _prog()
        self._save_warm_seeds(klasses, display_name=info.display_name or self._display_name_cache)
        n_ok = sum(1 for n in klasses if self._objs.get(n))
        log.info(
            "warm_bundle types=%d seeded=%d ok=%d geo=%r menu=%s wall=%.2fs",
            len(klasses),
            seeded,
            n_ok,
            (info.display_name or "")[:24],
            info.menu_open,
            time.perf_counter() - t0,
        )
        return info, res

    def _ensure_display_name(self, *, max_wall_s: float = 2.0) -> None:
        """保证有地区 DisplayName；必要时限时重扫 GameSpaceAPI。"""
        # 缓存可能是旧种子地名；仍尝试读活指针，失败再 rescan
        name = self._read_display_from_gamespace_api(allow_rescan=False)
        if name:
            self._display_name_cache = name
            return
        self._prime_gamespace_api(rescan=True, max_wall_s=max_wall_s)
        name = self._read_display_from_gamespace_api(allow_rescan=False)
        if name:
            self._display_name_cache = name

    def _rescue_critical_ui(self, *, max_wall_s: float = 2.0) -> None:
        """预热后仍弱：只压 GameSpace 地区名。"""
        need_geo = not (self._display_name_cache or "").strip()
        if need_geo:
            self._ensure_display_name(max_wall_s=max(1.0, float(max_wall_s)))

    def _warm_seed_key(self) -> str:
        return f"{int(self._pid)}:{hex(int(self._ga or 0))}"

    def _gamespace_seed_ok(self, api: int) -> bool:
        """GameSpaceAPI 种子：须能读出非空 DisplayName。"""
        assert self._h is not None
        api_klass = self._klasses.get("GameSpaceAPI")
        gs_klass = self._klasses.get("GameSpaceSheetData")
        if not api_klass or read_ptr(self._h, api) != api_klass:
            return False
        off_data = self._off("GameSpaceAPI.CurrentGameSpaceSheetData")
        gs = read_ptr(self._h, api + off_data)
        if not gs:
            return False
        if gs_klass and read_ptr(self._h, gs) != gs_klass:
            return False
        name = self._read_display_from_gamespace(gs)
        if name:
            self._display_name_cache = name
            return True
        return False

    def _apply_warm_seeds(self, klasses: dict[str, int]) -> int:
        """复用上次成功对象指针；返回命中类型数。幽灵不计入 → 触发补扫。"""
        assert self._h is not None
        try:
            if not _WARM_SEED_FILE.is_file():
                return 0
            raw = json.loads(_WARM_SEED_FILE.read_text(encoding="utf-8"))
            entry = raw.get(self._warm_seed_key()) if isinstance(raw, dict) else None
            if not isinstance(entry, dict):
                return 0
        except Exception as exc:
            log.debug("warm seed load: %s", exc)
            return 0
        # 新格式 {objs:..., display_name:...}；旧格式直接 name->addrs
        objs_map = entry.get("objs") if isinstance(entry.get("objs"), dict) else entry
        dn = entry.get("display_name")
        if isinstance(dn, str) and dn.strip():
            self._display_name_cache = dn.strip()
        hit = 0
        if not isinstance(objs_map, dict):
            return 0
        for name, klass in klasses.items():
            addrs = objs_map.get(name)
            if not isinstance(addrs, list):
                continue
            good: list[int] = []
            for a in addrs[:16]:
                try:
                    addr = int(a, 0) if isinstance(a, str) else int(a)
                except (TypeError, ValueError):
                    continue
                if not addr or read_ptr(self._h, addr) != klass:
                    continue
                if abs(addr - klass) < _KLASS_NEAR:
                    continue
                if name == "GameSpaceAPI":
                    if not self._gamespace_seed_ok(addr):
                        continue
                elif name in _LIVE_SEED_TYPES:
                    if not self._is_live_component(name, addr):
                        continue
                good.append(addr)
            if good:
                self._objs[name] = good
                hit += 1
                self._emit_warm_progress()
        return hit

    def _save_warm_seeds(
        self, klasses: dict[str, int], *, display_name: str = ""
    ) -> None:
        try:
            _WARM_SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
            raw: dict = {}
            if _WARM_SEED_FILE.is_file():
                try:
                    raw = json.loads(_WARM_SEED_FILE.read_text(encoding="utf-8"))
                except Exception:
                    raw = {}
            if not isinstance(raw, dict):
                raw = {}
            objs: dict[str, list[str]] = {}
            for name in klasses:
                addrs = list(self._objs.get(name) or [])[:8]
                if addrs:
                    objs[name] = [hex(int(a)) for a in addrs]
            if not objs and not (display_name or "").strip():
                return
            payload: dict[str, Any] = {"objs": objs}
            name = (display_name or self._display_name_cache or "").strip()
            if name:
                payload["display_name"] = name
            else:
                # 勿用空地名覆盖上次成功种子（否则同 PID 重启后地区一直「未知」）
                prev = raw.get(self._warm_seed_key())
                if isinstance(prev, dict):
                    prev_dn = prev.get("display_name")
                    if isinstance(prev_dn, str) and prev_dn.strip():
                        payload["display_name"] = prev_dn.strip()
                    prev_objs = prev.get("objs") if isinstance(prev.get("objs"), dict) else None
                    if prev_objs and "GameSpaceAPI" not in objs and prev_objs.get(
                        "GameSpaceAPI"
                    ):
                        objs["GameSpaceAPI"] = list(prev_objs["GameSpaceAPI"])[:8]
                        payload["objs"] = objs
            raw[self._warm_seed_key()] = payload
            if len(raw) > 8:
                keys = list(raw.keys())[-8:]
                raw = {k: raw[k] for k in keys}
            tmp = _WARM_SEED_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(_WARM_SEED_FILE)
        except Exception as exc:
            log.debug("warm seed save: %s", exc)

    def _framework_gamespace_api(self) -> int:
        """Resolve the current framework singleton with stable, typed links only."""
        h, ga = self._h, self._ga
        rva = self._cfg.get('typeinfo_rva', {}).get('FrameworkSingleton')
        required = ('Il2CppClass.static_fields', 'FrameworkSingleton.static.instance',
                    'FrameworkSingleton.GameSpaceAPI')
        if not h or not ga or not rva or any(k not in self._cfg['offsets'] for k in required):
            return 0
        try:
            def matches(key, actual):
                known = self._klasses.get(key)
                if known == actual:
                    return True
                spec = self._cfg.get('framework_type_tokens', {}).get(key, {})
                token = spec.get('token')
                if not token or known != parse_int(token) or parse_int(token) >> 29 != 1:
                    return False
                from mm_assist.memory.readers.ui_stack import read_c_string
                return (read_c_string(h, read_ptr(h, actual + 0x10) or 0) == spec.get('name')
                        and read_c_string(h, read_ptr(h, actual + 0x18) or 0) == spec.get('namespace'))
            klass = resolve_klass(h, ga, parse_int(rva))
            if not klass:
                return 0
            sf = read_ptr(h, klass + self._off(required[0]))
            root = read_ptr(h, sf + self._off(required[1])) if sf else 0
            if not root or read_ptr(h, root) != klass:
                return 0
            api = read_ptr(h, root + self._off(required[2]))
            api_class = read_ptr(h, api) if api else 0
            if not api_class or not matches('GameSpaceAPI', api_class):
                return 0
            data_off = self._off('GameSpaceAPI.CurrentGameSpaceSheetData')
            data = read_ptr(h, api + data_off)
            data_class = read_ptr(h, data) if data else 0
            if not data_class or not matches('GameSpaceSheetData', data_class):
                return 0
            if not self._read_display_from_gamespace(data):
                return 0
            if (self._h != h or self._ga != ga or self._closed
                    or read_ptr(h, klass + self._off(required[0])) != sf
                    or read_ptr(h, sf + self._off(required[1])) != root
                    or read_ptr(h, root) != klass
                    or read_ptr(h, root + self._off(required[2])) != api
                    or read_ptr(h, api) != api_class
                    or read_ptr(h, api + data_off) != data
                    or read_ptr(h, data) != data_class):
                return 0
            self._klasses['GameSpaceAPI'] = api_class
            self._klasses['GameSpaceSheetData'] = data_class
            return int(api)
        except (OSError, TypeError, ValueError):
            return 0

    def _prime_gamespace_api(
        self, *, rescan: bool = True, max_wall_s: float | None = 1.0
    ) -> None:
        """保留带合法 GameSpaceSheetData 的 GameSpaceAPI 实例。"""
        assert self._h is not None
        api_klass = self._klasses.get("GameSpaceAPI")
        gs_klass = self._klasses.get("GameSpaceSheetData")
        if not api_klass:
            return
        off_data = self._off("GameSpaceAPI.CurrentGameSpaceSheetData")
        good: list[int] = []
        t0 = time.perf_counter()

        def _stop() -> bool:
            if self._closed or bool(good):
                return True
            if max_wall_s is not None and (time.perf_counter() - t0) >= float(max_wall_s):
                return True
            return False

        fixed = self._framework_gamespace_api()
        candidates = [fixed] if fixed else list(self._objs.get("GameSpaceAPI") or [])
        if not fixed and (rescan or not candidates):
            candidates = scan_klass_objects(
                self._h,
                api_klass,
                max_hits=1,
                should_stop=_stop,
                max_wall_s=max_wall_s,
                hit_filter=lambda _name, obj: abs(obj - api_klass) >= _KLASS_NEAR
                and self._gamespace_seed_ok(obj),
                resume=True,
            )
        for api in candidates:
            if self._closed:
                break
            if read_ptr(self._h, api) != api_klass:
                continue
            gs = read_ptr(self._h, api + off_data)
            if not gs:
                continue
            if gs_klass and read_ptr(self._h, gs) != gs_klass:
                continue
            name = self._read_display_from_gamespace(gs)
            if name:
                good.append(api)
                self._display_name_cache = name
                break
        self._objs["GameSpaceAPI"] = good

    def _schedule_map_rescan(self) -> None:
        """后台补扫 MapUIVisible（不堵 lean 读）。"""
        if (self._closed or self._map_rescanning or self._h is None
                or self._pause_resolve or self._resolving or self._ui_refreshing
                or not self._display_name_cache):
            return
        from mm_assist.memory.readers.ui_stack import (
            peek_ui_stack, CRAFT_CURRENT_SHELL_NAMES, CRAFT_FACILITY_NAMES, CRAFT_PANEL_NAMES,
        )
        hint = peek_ui_stack(self._pid)
        if (hint is not None and 'CraftingHubPopupUIInstance' in hint.names
                and set(hint.names) <= (CRAFT_CURRENT_SHELL_NAMES | CRAFT_FACILITY_NAMES | CRAFT_PANEL_NAMES)):
            # The current crafting shell has its own authoritative UI chain.
            # Defer optional legacy map discovery; do not report map=False.
            return
        now = time.monotonic()
        if now < self._next_map_sync:
            return
        self._next_map_sync = now + 1.0
        self._map_rescanning = True

        def _work() -> None:
            t0 = time.perf_counter()
            try:
                with self._lock:
                    if self._closed or self._h is None:
                        return
                    h = self._h
                    klass = self._klasses.get("MapUIVisibleComponent")
                    old = list(self._objs.get("MapUIVisibleComponent") or [])
                if not klass:
                    return
                res = scan_klass_objects_multi(
                    h,
                    {"MapUIVisibleComponent": klass},
                    max_hits={"MapUIVisibleComponent": 192},
                    should_stop=lambda: self._closed,
                    max_wall_s=2.0,
                )
                objs = res.get("MapUIVisibleComponent") or []
                fresh = [o for o in objs if read_ptr(h, o) == klass]
                seen: set[int] = set(fresh)
                merged = list(fresh)
                for o in old:
                    if o in seen:
                        continue
                    if read_ptr(h, o) != klass:
                        continue
                    seen.add(o)
                    merged.append(o)
                with self._lock:
                    if not self._closed:
                        self._objs["MapUIVisibleComponent"] = merged
                log.debug(
                    "map rescan n=%d in %.0fms",
                    len(merged),
                    (time.perf_counter() - t0) * 1000.0,
                )
            except Exception as exc:
                log.debug("map rescan: %s", exc)
            finally:
                self._map_rescanning = False
                self._next_map_sync = time.monotonic() + 3.0

        threading.Thread(target=_work, name="il2cpp-map-rescan", daemon=True).start()

    def _sync_rescan_types(
        self, limits: dict[str, int], *, max_wall_s: float | None = None
    ) -> None:
        """同步合并扫类型（开图/开人物新建实例须尽快进表）。"""
        assert self._h is not None
        if self._closed or not limits:
            return
        klasses = {
            n: self._klasses[n] for n in limits if n in self._klasses and self._klasses[n]
        }
        if not klasses:
            return
        old_objs = {n: list(self._objs.get(n) or []) for n in klasses}
        res = scan_klass_objects_multi(
            self._h,
            klasses,
            max_hits={n: limits[n] for n in klasses},
            should_stop=lambda: self._closed,
            max_wall_s=max_wall_s,
        )
        for name, objs in res.items():
            klass = klasses[name]
            fresh = [o for o in objs if read_ptr(self._h, o) == klass]
            seen: set[int] = set(fresh)
            merged = list(fresh)
            for o in old_objs.get(name) or []:
                if o in seen:
                    continue
                if read_ptr(self._h, o) != klass:
                    continue
                seen.add(o)
                merged.append(o)
            self._objs[name] = merged

    def dump_ui_state_live_tops(self, type_name: str, *, limit: int = 12) -> list[dict]:
        """成对探针：活体 NewProfile TopCategory / CraftingHub CategoryType 抽样（不上栏）。"""
        assert self._h is not None
        klass = self._klasses.get(type_name)
        if not klass or type_name not in self._objs:
            return []
        off = None
        field_key = "top"
        if type_name == "NewProfileUIState":
            off = self._off("NewProfileUIState.TopCategory")
            field_key = "top"
        elif type_name == "CraftingHubUIState":
            off = self._off("CraftingHubUIState.CategoryType")
            field_key = "cat"
        rows: list[dict] = []
        for obj in self._objs.get(type_name) or []:
            if read_ptr(self._h, obj) != klass:
                continue
            if abs(obj - klass) < _KLASS_NEAR:
                continue
            if not self._is_live_component(type_name, obj):
                continue
            raw = read_u32(self._h, obj + off) if off is not None else None
            val: int | None = None
            if raw is not None:
                si = raw if raw < 0x80000000 else raw - 0x100000000
                val = int(si)
            rows.append({"addr": hex(obj), field_key: val})
            if len(rows) >= limit:
                break
        return rows

    def craft_hub_diag(self) -> dict[str, Any]:
        """最近一次 ``_read_craft_hub_category`` 诊断（成对帧落盘用）。"""
        return dict(self._last_craft_hub_diag)

    def _klass_obj_n(self, type_name: str) -> int | None:
        if type_name not in self._klasses or type_name not in self._objs:
            return None
        return len(self._objs.get(type_name) or [])

    def _read_hud_raw_pair(self) -> tuple[int | None, int | None]:
        """首个活 Hud 的 HideAll + HudUIStep；不参与 menu 判定。"""
        assert self._h is not None
        klass = self._klasses.get("HudUIVisibleComponent")
        objs = self._objs.get("HudUIVisibleComponent") or []
        if not klass or not objs:
            return None, None
        off_hide = self._off("HudUIVisible.HideAll")
        for obj in objs:
            if read_ptr(self._h, obj) != klass:
                continue
            if abs(obj - klass) < _KLASS_NEAR:
                continue
            if not self._is_live_component("HudUIVisibleComponent", obj):
                continue
            hide = read_u8(self._h, obj + off_hide)
            step = read_u32(self._h, obj + 0x10)
            hide_v = int(hide) if hide in (0, 1) else None
            step_v: int | None = None
            if step is not None:
                si = step if step < 0x80000000 else step - 0x100000000
                if -1 <= si <= 23:
                    step_v = int(si)
            return hide_v, step_v
        return None, None

    def _live_i32_tuple(self, type_name: str) -> tuple[int, ...]:
        rows = self.dump_ui_state_live_tops(type_name)
        key = "top" if type_name == "NewProfileUIState" else "cat"
        return tuple(int(r[key]) for r in rows if r.get(key) is not None)

    def _fill_ui_dump_extras(self, info: MemSceneInfo) -> None:
        hide, step = self._read_hud_raw_pair()
        info.hud_hide_all = hide
        info.hud_ui_step = step
        info.profile_tops = self._live_i32_tuple("NewProfileUIState")
        info.hub_cats = self._live_i32_tuple("CraftingHubUIState")
        info.sn_live_n = self._count_ui_state_live("SideNavigationUIState")
        info.hub_live_n = self._count_ui_state_live("CraftingHubUIState")
        info.hud_live_n = self._count_ui_state_live("HudUIVisibleComponent")
        info.np_obj_n = self._klass_obj_n("NewProfileUIState")
        info.sn_obj_n = self._klass_obj_n("SideNavigationUIState")
        info.hub_obj_n = self._klass_obj_n("CraftingHubUIState")
        info.hud_obj_n = self._klass_obj_n("HudUIVisibleComponent")

    def _read_once(self) -> MemSceneInfo:
        assert self._h is not None
        info = MemSceneInfo(status="ok")
        hits: list[str] = []

        # T5 缓：保持现读法；闪 1→? 勿当稳态（B2 不接线 classify）
        info.repair_list_active = self._flag_or(
            "EquipmentRepairListUIComponent", "EquipmentRepairList.isActive"
        )
        info.repair_modal_open = self._flag_or(
            "EquipmentRepairListUIComponent", "EquipmentRepairList.isModalOpened"
        )
        if info.repair_list_active is not None or info.repair_modal_open is not None:
            hits.append("repair")

        info.chat_focus = self._flag_first(
            "ChattingUIStoredDataComponent", "ChattingUIStored.IsFocus"
        )
        full = self._flag_first("ChatDisplayDataComponent", "ChatDisplay.IsFullScreen")
        mid = self._flag_first("ChatDisplayDataComponent", "ChatDisplay.IsMiddleScreen")
        if full is not None or mid is not None:
            info.chat_fullscreen = bool(full) or bool(mid)
        if info.chat_focus is not None or info.chat_fullscreen is not None:
            info.chat_open = bool(info.chat_focus) or bool(info.chat_fullscreen)
            hits.append("chat")

        # 开图会新建 MapUIVisible；勿同步全堆扫（会卡死状态栏 lean 拍）。
        map_vis = self._flag_or("MapUIVisibleComponent", "MapUIVisible.IsVisible")
        if map_vis is None:
            self._schedule_map_rescan()
        if map_vis is not None:
            hits.append("map_vis")

        # B4：Hub 仅在 menu 已开且非地图时写入；幽灵 cat=1 多实例则弃
        hub_cat = self._read_craft_hub_category()
        if hub_cat is not None:
            hits.append("craft_hub")

        info.side_nav_open = self._read_ui_state_present("SideNavigationUIState")
        if info.side_nav_open is not None:
            hits.append("side_nav")
        # 人物：NewProfile 活体计数启发式（live≥2 开 / 0 关 / 1 不定不上栏）
        live_n = self._count_ui_state_live("NewProfileUIState")
        info.profile_live_n = live_n
        info.profile_open = resolve_profile_open_from_live_n(live_n)
        if live_n is not None:
            hits.append("profile")

        # 地图主信号：须 map_vis=True（mt 只填类型）
        mt = self._read_viewing_map_type()
        info.map_open, resolved_mt = reconcile_map_open_signal(
            map_vis=map_vis, viewing_mt=mt
        )
        if resolved_mt is not None and info.map_open:
            info.map_type = resolved_mt
            info.map_type_name = MAP_TYPE_CN.get(resolved_mt, str(resolved_mt))
            hits.append("viewing_map")
        elif is_valid_viewing_map_type(mt) and info.map_open:
            info.map_type = mt
            info.map_type_name = MAP_TYPE_CN.get(mt, str(mt))
            hits.append("viewing_map")

        # hub 旁证：不改写 menu（离开主界面只信 UI 栈 HudUIStepState）
        if info.map_open is True and map_vis is True:
            info.craft_hub_category = 0
            if "map_hub0" not in hits:
                hits.append("map_align")
        elif hub_cat is not None:
            info.craft_hub_category = hub_cat

        obj, cid = self._best_channel_obj()
        if obj is not None and cid is not None:
            info.channel_sheet_id = cid
            hits.append("channel")
            info.in_dungeon = bool(
                read_u8(self._h, obj + self._off("ChannelIdentifier.IsPlayerInDungeon")) or 0
            )
            info.in_normal_field = bool(
                read_u8(
                    self._h, obj + self._off("ChannelIdentifier.IsPlayerInNormalField")
                )
                or 0
            )
            info.changing_channel = bool(
                read_u8(
                    self._h, obj + self._off("ChannelIdentifier.IsChangingChannel")
                )
                or 0
            )

        name = self._read_display_name(info.channel_sheet_id)
        if name:
            info.display_name = name
            hits.append("display_name")

        try:
            self._fill_ui_dump_extras(info)
        except Exception:
            pass

        info.source_hits = hits
        if not hits:
            info.status = "instance_not_found"
        return info

    def _read_combat_mode_ui(self) -> bool | None:
        """SkillButtonsUIComponent → ui.isCombatModeLast（校验 0/1 + UI klass）。"""
        assert self._h is not None
        comp_klass = self._klasses.get("SkillButtonsUIComponent")
        ui_klass = self._klasses.get("SkillButtonsUI")
        objs = self._objs.get("SkillButtonsUIComponent") or []
        if not comp_klass or not ui_klass or not objs:
            return self._flag_first("SkillButtonsUI", "SkillButtonsUI.IsCombatModeLast")
        off_ui = self._off("SkillButtonsUIComponent.UI")
        off_hud = self._off("SkillButtonsUIComponent.IsHudUIVisible")
        off_cui = self._off("SkillButtonsUIComponent.IsCombatUIVisible")
        off_mode = self._off("SkillButtonsUI.IsCombatModeLast")
        alive: list[int] = []
        for obj in objs:
            if read_ptr(self._h, obj) != comp_klass:
                continue
            alive.append(obj)
            hud = read_u8(self._h, obj + off_hud)
            cui = read_u8(self._h, obj + off_cui)
            if hud not in (0, 1) or cui not in (0, 1):
                continue
            ui = read_ptr(self._h, obj + off_ui)
            if not ui or read_ptr(self._h, ui) != ui_klass:
                continue
            mode = read_u8(self._h, ui + off_mode)
            if mode not in (0, 1):
                continue
            # 优先 HUD 可见的那份；cbt 用 IsCombatUIVisible（IsCombatModeLast 城镇常误亮）
            if hud == 1:
                self._objs["SkillButtonsUIComponent"] = alive
                return bool(cui)
        self._objs["SkillButtonsUIComponent"] = alive
        return None

    def _read_combat_state(self) -> int | None:
        """优先 SystemContainers → ControlCharacterStates；否则堆扫首个。"""
        assert self._h is not None
        sc_klass = self._klasses.get("SystemContainers")
        sc_objs = self._objs.get("SystemContainers") or []
        off_sc = self._cfg["offsets"].get(
            "SystemContainers.ControlCharacterStatesComponent"
        )
        off_cs = self._off("ControlCharacterStates.CombatState")
        states_klass = self._klasses.get("ControlCharacterStatesComponent")
        if sc_klass and sc_objs and off_sc is not None and states_klass:
            for sc in sc_objs:
                if read_ptr(self._h, sc) != sc_klass:
                    continue
                comp = read_ptr(self._h, sc + int(off_sc))
                if not comp:
                    continue
                if read_ptr(self._h, comp) != states_klass:
                    continue
                v = read_u32(self._h, comp + off_cs)
                if v is not None and 0 <= v <= 3:
                    return int(v)
        # 堆扫回退
        for obj in self._objs.get("ControlCharacterStatesComponent") or []:
            if not states_klass or read_ptr(self._h, obj) != states_klass:
                continue
            v = read_u32(self._h, obj + off_cs)
            if v is not None and 0 <= v <= 3:
                return int(v)
        return None

    def _read_display_name(self, channel_id: int | None) -> str:
        """优先 GameSpaceAPI.CurrentGameSpaceSheetData.DisplayName。"""
        assert self._h is not None
        name = self._read_display_from_gamespace_api()
        if name:
            return name
        # ChannelSheetData.Id 与 ChannelSheetId 数值空间不一致，全堆扫无效且极慢；保留缓存兜底
        return self._display_name_cache

    def _read_display_from_gamespace_api(self, *, allow_rescan: bool = False) -> str:
        assert self._h is not None
        api_klass = self._klasses.get("GameSpaceAPI")
        gs_klass = self._klasses.get("GameSpaceSheetData")
        if not api_klass:
            return ""
        objs = list(self._objs.get("GameSpaceAPI") or [])
        if not objs and allow_rescan:
            self._prime_gamespace_api()
            objs = list(self._objs.get("GameSpaceAPI") or [])
        off_data = self._off("GameSpaceAPI.CurrentGameSpaceSheetData")
        alive: list[int] = []
        for api in objs:
            if read_ptr(self._h, api) != api_klass:
                continue
            alive.append(api)
            gs = read_ptr(self._h, api + off_data)
            if not gs:
                continue
            if gs_klass and read_ptr(self._h, gs) != gs_klass:
                continue
            name = self._read_display_from_gamespace(gs)
            if name:
                self._objs["GameSpaceAPI"] = alive
                self._display_name_cache = name
                return name
        if alive:
            self._objs["GameSpaceAPI"] = alive
        return ""

    def _read_display_from_gamespace(self, gs_obj: int) -> str:
        assert self._h is not None
        str_ptr = read_ptr(self._h, gs_obj + self._off("GameSpaceSheetData.DisplayName"))
        if not str_ptr:
            return ""
        return read_il2cpp_string(self._h, str_ptr) or ""

    def _read_str_field(self, obj: int, off_key: str, *, max_chars: int = 256) -> str:
        assert self._h is not None
        if off_key not in self._cfg["offsets"]:
            return ""
        str_ptr = read_ptr(self._h, obj + self._off(off_key))
        if not str_ptr:
            return ""
        return read_il2cpp_string(self._h, str_ptr, max_chars=max_chars) or ""

    def scan_channel_map_paths(
        self,
        *,
        hwnd: int | None = None,
        max_hits: int = 400,
        name_substr: str = "",
    ) -> list[dict[str, Any]]:
        """堆扫 ChannelSheetData，读 MapImagePath / MapBackgroundImagePath。

        不以 ChannelSheetId 唯匹配（与 sheet.Id 数值空间可能不一致）；
        可用 DisplayName / Nexon 名 / 路径子串过滤。
        """
        if not self._ensure_proc(hwnd):
            return []
        with self._lock:
            if self._ga is None or not self._klasses:
                if not self._resolve_klasses():
                    return []
            klass = self._klasses.get("ChannelSheetData")
            gs_klass = self._klasses.get("GameSpaceSheetData")
            if not klass or self._h is None:
                self._last_err = "ChannelSheetData_klass_missing"
                return []
            rows: list[dict[str, Any]] = []
            needle = (name_substr or "").strip().lower()
            for obj in scan_klass_objects(
                self._h, klass, max_hits=max_hits, should_stop=lambda: self._closed
            ):
                if read_ptr(self._h, obj) != klass:
                    continue
                sheet_id = read_u32(self._h, obj + self._off("ChannelSheetData.Id"))
                nexon = self._read_str_field(
                    obj, "ChannelSheetData.NexonServiceLogDisplayName", max_chars=128
                )
                map_path = self._read_str_field(obj, "ChannelSheetData.MapImagePath")
                bg_path = self._read_str_field(
                    obj, "ChannelSheetData.MapBackgroundImagePath"
                )
                display = ""
                gs = read_ptr(self._h, obj + self._off("ChannelSheetData.GameSpace"))
                if gs and (not gs_klass or read_ptr(self._h, gs) == gs_klass):
                    display = self._read_display_from_gamespace(gs)
                blob = " ".join(
                    p for p in (display, nexon, map_path, bg_path) if p
                ).lower()
                if needle and needle not in blob:
                    continue
                if not map_path and not bg_path and not display and not nexon:
                    continue
                rows.append(
                    {
                        "obj": hex(obj),
                        "sheet_id": int(sheet_id) if sheet_id is not None else None,
                        "nexon_name": nexon,
                        "display_name": display,
                        "map_image_path": map_path,
                        "map_background_path": bg_path,
                    }
                )
            return rows

    def schedule_resolve(self, hwnd: int | None = None) -> None:
        with self._lock:
            if self._resolving or self._closed or self._pause_resolve:
                return
            self._resolving = True

        def _work() -> None:
            try:
                if self._pause_resolve or self._closed:
                    return
                # Background startup needs the base scene types, not an
                # unbounded scan for every optional UI component.
                self.warm_bundle(hwnd, max_wall_s=4.5)
            except Exception as exc:
                log.warning(
                    "il2cpp scene resolve fail: %s: %r",
                    type(exc).__name__,
                    exc,
                )
            finally:
                self._resolving = False

        self._bg = threading.Thread(target=_work, name="il2cpp-scene", daemon=True)
        self._bg.start()

    def read(
        self,
        hwnd: int | None = None,
        *,
        allow_scan: bool = True,
        refresh_ui: bool | None = None,
        lean_ui_refresh: bool = False,
    ) -> MemSceneInfo | None:
        """读界面态。

        allow_scan：未解析时是否同步堆扫。
        refresh_ui：是否调度后台 UI 对象补扫。默认仅 allow_scan=True 时补扫；
        lean/状态栏应传 False，避免数十秒堆扫拖垮刷新——除非 lean_ui_refresh=True
        （仅地名空时补 GameSpace，不再 lean 堆扫 Hud/Inv/Dlg）。
        """
        t0 = time.perf_counter()
        do_refresh = allow_scan if refresh_ui is None else bool(refresh_ui)
        if lean_ui_refresh:
            do_refresh = True
        if self._closed:
            self._last_err = "closed"
            return None
        if not self._ensure_proc(hwnd=hwnd):
            return None
        # 短超时：若偶发被其它路径占锁，先吐缓存，别把状态栏卡成几十秒一拍
        got = self._lock.acquire(timeout=0.05)
        if not got:
            self._last_err = "busy"
            return self._cache
        try:
            if self._ga is None or not self._klasses:
                if not allow_scan:
                    self.schedule_resolve(hwnd=hwnd)
                    self._last_err = "resolving"
                    return self._cache
                if not self._resolve_klasses():
                    return None
            if self._resolved and self._objs:
                info = self._read_once()
                info = self._sticky_merge(info)
                info.age_ms = (time.perf_counter() - t0) * 1000.0
                self._cache = info
                self._last_err = "" if info.status == "ok" else info.status
                schedule_refresh = do_refresh
                self._emit_warm_progress()
            else:
                schedule_refresh = False
                info = None  # type: ignore[assignment]
        finally:
            self._lock.release()
        if self._resolved and self._objs and schedule_refresh:
            self._maybe_refresh_ui_objs(lean=bool(lean_ui_refresh))
            return info
        if not allow_scan:
            self.schedule_resolve(hwnd=hwnd)
            self._last_err = "resolving"
            return self._cache
        with self._lock:
            if not self._klasses and not self._resolve_klasses():
                return None
        self._resolve_all_objs()
        with self._lock:
            info = self._read_once()
            info = self._sticky_merge(info)
            info.age_ms = (time.perf_counter() - t0) * 1000.0
            self._cache = info
            self._last_err = "" if info.status == "ok" else info.status
            self._emit_warm_progress()
            return info

    def _sticky_merge(self, info: MemSceneInfo) -> MemSceneInfo:
        """读丢频道/战斗标志时沿用上次，避免面板来回跳。"""
        prev = self._cache
        if prev is None:
            return info
        # 脏 channel 当缺失，便于沿用上次可信值
        if not plausible_channel_sheet_id(info.channel_sheet_id):
            info.channel_sheet_id = None
        if info.channel_sheet_id is None and plausible_channel_sheet_id(
            prev.channel_sheet_id
        ):
            info.channel_sheet_id = prev.channel_sheet_id
            info.in_dungeon = prev.in_dungeon if info.in_dungeon is None else info.in_dungeon
            info.in_normal_field = (
                prev.in_normal_field
                if info.in_normal_field is None
                else info.in_normal_field
            )
            if "channel" not in info.source_hits and "channel" in (prev.source_hits or []):
                info.source_hits = list(info.source_hits) + ["channel*"]
        if not info.display_name and prev.display_name:
            info.display_name = prev.display_name
        if info.combat_mode_ui is None and prev.combat_mode_ui is not None:
            info.combat_mode_ui = prev.combat_mode_ui
            info.in_combat = prev.combat_mode_ui
        if info.in_combat is None and prev.in_combat is not None:
            info.in_combat = prev.in_combat
        if info.combat_state is None and prev.combat_state is not None:
            info.combat_state = prev.combat_state
        # 面板开关不 sticky：实例消失时为 None，粘住旧 True 会让 dlg/inv 永远亮
        return info
