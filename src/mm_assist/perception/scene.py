"""游戏框内场景状态：界面态 + 地理（像素/OCR）+ 内存坐标。

只读截屏与内存；不点击。供 state_hud 0.5s 轮询。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from mm_assist.config import AppConfig
from mm_assist.perception.game_frame import GameFrame, ensure_game_frame
from mm_assist.perception.list_ocr import box_center_norm
from mm_assist.perception.map_classify import (
    CONTINENT_NEEDLES,
    DIALOG_NEEDLES,
    GOTO_NEEDLES,
    STOP_NEEDLES,
    _blob,
    _is_inventory_blob,
    _map_level,
    _map_opened_blob,
    classify_state,
)
from mm_assist.memory.readers.scene import (
    Il2CppSceneReader,
    MemSceneInfo,
    default_path as scene_mem_path,
    is_repair_ui_open,
    resolve_menu_subdivide_detail,
)
from mm_assist.memory.readers.mem_position import MemPositionReader, default_path as mem_pos_path
from mm_assist.perception.ocr import OcrLine, ocr_image
from mm_assist.perception.list_ocr import normalize_text
from mm_assist.platform.window import WindowInfo

log = logging.getLogger(__name__)

UiKind = Literal[
    "hud",
    "combat",
    "map_continent",
    "map_region",
    "map_town",
    "npc_card",
    "traveling",
    "dialog",
    "repair",
    "inventory",
    "menu",
    "quest",
    "chat",
    "unknown",
]

INVENTORY_MARKS = (
    "自動配戴",
    "自动配戴",
    "武器符",
    "防具符",
    "飾品符",
    "饰品符",
    "寶石·符文",
    "宝石·符文",
    "背包",
    "裝備",
    "装备",
)
REPAIR_MARKS = (
    "修理所有",
    "修理費用",
    "修理费用",
    "修理已穿戴",
    "維修",
    "需要修理",
    "裝備損壞",
    "装备损坏",
)
MENU_MARKS = (
    "Stellagram",
    "信箱",
    "成就",
    "設定",
    "设置",
    "遊戲選項",
    "游戏选项",
)
GEO_TOWN_MARKS = ("杜巴頓", "杜巴顿", "杜巴頔", "尚杜巴")
GEO_OTHER_MARKS = (
    "库漢",
    "庫漢",
    "库汉",
    "莱比",
    "萊比",
    "希德",
    "堤爾",
    "堤尔",
    "歐拉大陸",
    "欧拉大陆",
    "欧拉大睦",
    "歐拉大睦",
)

UI_LABELS: dict[str, str] = {
    "hud": "非战斗",
    "combat": "战斗",
    "map_continent": "总地图",
    "map_region": "地图",
    "map_town": "地图·城镇",
    "npc_card": "NPC卡片",
    "traveling": "行进中",
    "dialog": "对话",
    "repair": "修理",
    "inventory": "物品",
    "menu": "菜单",
    "quest": "任务",
    "chat": "聊天",
    "unknown": "未知",
}

UI_DETAIL_LABELS: dict[str, str] = {
    "map_continent": "大陆",
    "map_region": "地区",
    "map_town": "城镇",
    "npc_card": "前往卡",
    "traveling": "自动移动",
    "dialog": "NPC对话",
    "repair": "修理面板",
    "hud": "主界面",
    "combat": "战斗中",
    "inventory": "背包",
    "menu": "系统菜单",
    "quest": "任务簿",
    "chat": "聊天",
    "unknown": "",
}

MAP_LEVEL_LABELS: dict[str, str] = {
    "continent": "大陆",
    "region": "地区",
    "town": "城镇",
    "unknown": "—",
}

MEM_STATUS_LABELS: dict[str, str] = {
    "ok": "正常",
    "pending": "等待",
    "resolving": "首次定位中…",
    "cache_miss": "首次定位中…",
    "addr_not_locked": "未锁定（可跑 probe）",
    "xy_zero": "读数无效",
    "pid_changed_rescan": "进程已变，重定位中",
    "process_not_found": "无游戏进程",
    "gameassembly_not_found": "未找到 GameAssembly",
    "typeinfo_uninitialized": "TypeInfo 未就绪（进场景后再试）",
    "instance_not_found": "未找到坐标组件",
    "read_failed": "读取失败",
    "il2cpp_and_fixed_failed": "IL2CPP/回退均失败",
    "il2cpp_failed": "IL2CPP 读坐标失败",
    "value_out_of_range": "数值异常",
    "open_failed": "无法打开进程",
    "client_build_unregistered": "版本待驗證",
    "configuration_changed": "讀取配置已變更，請核對版本",
    "client_session_changed": "讀取配置已變更，請重開狀態欄",
    "client_build_hashing": "正在核對遊戲版本…",
    "closed": "已关闭",
}

MEM_SOURCE_LABELS: dict[str, str] = {
    "map_component": "地图坐标",
    "transform_xz": "Transform·XZ",
    "inaccurate": "粗定位±1m",
    "il2cpp_comp": "IL2CPP",
    "il2cpp_sc": "IL2CPP",
    "fixed": "固定地址",
}

CHANNEL_REGIONS_REL = Path("maps/channel_regions.json")


def channel_regions_path(knowledge_root: Path | None = None) -> Path:
    if knowledge_root is None:
        return Path.cwd() / "knowledge" / CHANNEL_REGIONS_REL
    return Path(knowledge_root) / CHANNEL_REGIONS_REL


def load_channel_regions(path: Path | None = None) -> dict[str, str]:
    """返回 channel_sheet_id(str) -> 中文地区名。"""
    p = path or channel_regions_path()
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    regions = raw.get("regions") or {}
    out: dict[str, str] = {}
    for k, v in regions.items():
        if isinstance(v, dict):
            name = str(v.get("name") or "").strip()
        else:
            name = str(v or "").strip()
        if name:
            out[str(k)] = name
    return out


def resolve_geo_name(
    channel_sheet_id: int | None,
    *,
    in_dungeon: bool | None = None,
    regions: dict[str, str] | None = None,
    display_name: str = "",
) -> str:
    """地理主文案：权威 DisplayName 优先；地下城未知名 → 地下城；站桩表仅兜底。"""
    name = (display_name or "").strip()
    if name:
        return name
    table = regions if regions is not None else load_channel_regions()
    if channel_sheet_id is not None:
        pinned = table.get(str(int(channel_sheet_id)))
        if pinned:
            return pinned
    if in_dungeon:
        return "地下城"
    return "未知"


def _from_mem_scene(
    ms: MemSceneInfo | None,
    *,
    regions: dict[str, str] | None = None,
) -> tuple[UiKind, str, str, str, str, str]:
    """界面 / 地图层 / 地区：业务主标签 + 权威 DisplayName。

    栈就绪时按类名分流。未就绪不把空栈当主界面。
    """
    if ms is None:
        return "unknown", "内存定位中", "unknown", "定位中…", "", ""

    from mm_assist.memory.readers.ui_stack import stack_inv, stack_kind

    detail = ""
    hub = getattr(ms, "craft_hub_category", None)
    dlg = _mem_flag(ms, "dialog_active", "dialog_active")
    names = list(getattr(ms, "ui_stack_names", ()) or ())
    ready = bool(getattr(ms, "ui_stack_ready", False))
    inv = stack_inv(names) if ready else None
    menu = _mem_flag(ms, "menu_open", "menu_open")
    map_on = _mem_flag(ms, "map_open", "map_open")
    kind = stack_kind(
        names,
        ready=ready,
        empty_ok=bool(getattr(ms, "ui_stack_empty_ok", False)),
    )

    if kind == "dialog" or (kind is None and dlg is True):
        ui: UiKind = "dialog"
        mem_ui = UI_LABELS["dialog"]
    elif kind == "inventory" or (kind is None and inv is True):
        ui = "inventory"
        mem_ui = UI_LABELS["inventory"]
    elif kind == "main" or (kind is None and menu is False):
        ui = "hud"
        mem_ui = UI_LABELS["hud"]
        if getattr(ms, "changing_channel", False):
            detail = "换频中"
    elif kind == "map" or (kind is None and menu is True and map_on is True):
        mt = getattr(ms, "map_type", None)
        if mt == 1:
            ui = "map_continent"
        elif mt == 2:
            ui = "map_town"
        else:
            ui = "map_region"
        mem_ui = UI_LABELS[ui]
        detail = UI_DETAIL_LABELS.get(ui, "") or ""
    elif kind == "overlay":
        ui = "unknown"
        mem_ui = "未知"
    elif kind in ("side_nav", "life_skill", "profile", "craft_hub", "repair") or (
        kind is None and menu is True
    ):
        ui = "menu"
        mem_ui = UI_LABELS["menu"]
        detail = resolve_menu_subdivide_detail(craft_hub_category=hub)
    else:
        ui = "unknown"
        mem_ui = "未知"

    lv = "unknown"
    if ui.startswith("map_") or map_on is True:
        mt = getattr(ms, "map_type", None)
        if mt == 1:
            lv = "continent"
        elif mt == 2:
            lv = "town"
        else:
            lv = "region"

    mem_geo = ms.geo_hint or ""
    geo = resolve_geo_name(
        ms.channel_sheet_id,
        in_dungeon=ms.in_dungeon,
        regions=regions,
        display_name=ms.display_name,
    )
    return ui, detail, lv, geo, mem_ui, mem_geo


# 1920×1440：小地图右下开图钮（world_map.md；旁证 ROI，非绝对可点宣称）
_MAP_OPEN_BTN_CLIENT_XY: tuple[int, int] = (1811, 176)


def format_ui_map_hint(ms: MemSceneInfo | None) -> str:
    """地图旁证：map=1/0 + mapact=@开图钮。仅 map_open 有读数时非空。"""
    if ms is None or ms.map_open is None:
        return ""
    bit = "1" if ms.map_open else "0"
    x, y = _MAP_OPEN_BTN_CLIENT_XY
    return f"map={bit} mapact=@{x},{y}"


def format_ui_profile_hint(ms: MemSceneInfo | None) -> str:
    """人物旁证：prf=1/0 [+ prf_n]。仅 profile_open 有读数时非空（live==1 不定不上栏）。"""
    if ms is None or ms.profile_open is None:
        return ""
    bit = "1" if ms.profile_open else "0"
    parts = [f"prf={bit}"]
    if ms.profile_live_n is not None:
        parts.append(f"prf_n={int(ms.profile_live_n)}")
    return " ".join(parts)


def _mem_flag(ms: Any, *names: str) -> bool | None:
    for name in names:
        v = getattr(ms, name, None)
        if v is not None:
            return bool(v)
    return None


def _menu_display_bit(ms: MemSceneInfo) -> bool | None:
    """离开主界面：栈含 HudUIStepState（写入 ms.menu_open）。"""
    return _mem_flag(ms, "menu_open", "menu_open")


def format_ui_params(
    ms: MemSceneInfo | None,
    icons: Any | None = None,
    action: Any | None = None,
    menu_nav: str | None = None,
) -> str:
    """先核栈就绪，再显示独立停止态；菜单/设施不把行动中的1改为0。"""
    from mm_assist.perception.ui_fine import HudActionSummary

    if ms is None:
        return "—"
    _ = icons
    _ = menu_nav
    if not bool(getattr(ms, "ui_stack_ready", False)):
        return "—"
    stop = None
    if action is not None and isinstance(action, HudActionSummary):
        stop = getattr(action, "stop_visible", None)
    if stop is True:
        return "stp=1"
    return "stp=0" if stop is False else "stp=?"


STACK_REUSE_MAX_S = 1.5
STACK_REUSE_MAX_TICKS = 2
STOP_SAMPLE_MAX_AGE_S = 2.5


def apply_stack_bits(ms: MemSceneInfo, snap: Any) -> None:
    """用 UI 栈覆盖 menu/dlg。未 ready 则两位为 None。"""
    from mm_assist.memory.readers.ui_stack import left_main_hud, stack_dlg

    names = [str(n) for n in list(getattr(snap, "names", None) or [])]
    ready = bool(getattr(snap, "ready", False))
    ms.ui_stack_names = tuple(names)
    ms.ui_stack_ready = ready
    ms.ui_stack_error = str(getattr(snap, "error", "") or "")
    ms.memory_process = dict(getattr(snap, 'process_identity', {}) or {})
    ms.ui_stack_empty_ok = bool(getattr(snap, "empty_confirmed", False))
    # No facts survive an unavailable/reused stack or a closed instance.
    ms.profile_category = None
    ms.profile_identity = None
    ms.craft_current_category = None
    ms.craft_identity = None
    ms.map_identity = None
    ms.map_view = None
    try:
        ms.ui_stack_ts = float(getattr(snap, "sampled_ts", 0.0) or 0.0)
    except (TypeError, ValueError):
        ms.ui_stack_ts = 0.0
    if not ready:
        ms.menu_open = None
        ms.dialog_active = None
        return
    if not ms.ui_stack_error and ms.ui_stack_ts > 0:
        ms.profile_category = getattr(snap, "profile_category", None)
        ms.profile_identity = getattr(snap, "profile_identity", None)
        ms.craft_current_category = getattr(snap, "craft_current_category", None)
        ms.craft_identity = getattr(snap, "craft_identity", None)
        ms.map_identity = getattr(snap, "map_identity", None)
        ms.map_view = getattr(snap, "map_view", None)
    ms.menu_open = bool(left_main_hud(names))
    ms.dialog_active = stack_dlg(names)
    cat = getattr(snap, "craft_category", None)
    if cat in (0, 1, 2, 3):
        ms.craft_hub_category = int(cat)


_LIFE_SKILL_NAV_LABELS = frozenset(
    {
        "伐木",
        "伐木",
        "采矿",
        "采矿",
        "药草",
        "药草",
        "剪羊毛",
        "剪羊毛",
        "收割",
        "收割",
        "锄地",
        "锄地",
        "昆虫采集",
        "昆虫采集",
        "钓鱼",
        "钓鱼",
    }
)
_PROFILE_NAV_LABELS = frozenset({"角色界面", "角色", "人物界面", "人物"})


def _nav_labels(menu_nav: str | None) -> list[str]:
    s = str(menu_nav or "").strip()
    if not s:
        return []
    if s.startswith("nav="):
        s = s[4:]
    return [p.strip() for p in s.split(",") if p.strip()]


def _relation_combat_hud(
    ms: MemSceneInfo | None,
    ui: str,
    icons: Any | None = None,
) -> bool | None:
    """主界面战斗态：True=战斗界面 / False=非战斗界面 / None=未知。"""
    from mm_assist.perception.ui_fine import HudIconSummary, format_ui_combat_from_icons

    if ui == "combat":
        return True
    if icons is not None and isinstance(icons, HudIconSummary):
        cbt = format_ui_combat_from_icons(icons)
        if cbt == "cbt=1":
            return True
        if cbt == "cbt=0":
            return False
    if ms is not None:
        if ms.combat_mode_ui is True:
            return True
        if ms.combat_mode_ui is False:
            return False
    return None


def format_ui_relation(
    ms: MemSceneInfo | None,
    ui: str,
    detail: str = "",
    *,
    icons: Any | None = None,
    menu_nav: str | None = None,
) -> str:
    """顶部「界面」行人话主态。与 format_node_zh 同一套中文。"""
    from mm_assist.life_plan.node_inspect import format_node_zh
    from mm_assist.memory.readers.ui_stack import stack_node

    detail_s = str(detail or "")
    nav = _nav_labels(menu_nav)
    sn = ""
    if ms is not None:
        sn = stack_node(
            list(getattr(ms, "ui_stack_names", ()) or ()),
            ready=bool(getattr(ms, "ui_stack_ready", False)),
            empty_ok=bool(getattr(ms, "ui_stack_empty_ok", False)),
        )
    if sn:
        if sn == "map" or sn == "map_dunbarton" or str(sn).startswith("map"):
            return "地图"
        if sn == "overlay":
            if any(x in _LIFE_SKILL_NAV_LABELS for x in nav):
                return "角色-生活技能"
            if any(x in _PROFILE_NAV_LABELS for x in nav):
                return "角色"
            return "离开主界面"
        if sn == "main_hud":
            combat = _relation_combat_hud(ms, ui, icons)
            if combat is True or ui == "combat":
                return "主界面战斗"
            return "主界面非战斗"
        zh = format_node_zh(sn)
        if zh and zh not in {"?", "—"}:
            return zh

    if ms is not None:
        if ms.dialog_active:
            return "对话"
        names = list(getattr(ms, "ui_stack_names", ()) or ())
        from mm_assist.memory.readers.ui_stack import stack_inv

        if stack_inv(names):
            return "背包"
        mapish = _mem_flag(ms, "map_open", "map_open") is True or str(ui).startswith(
            "map_"
        )
        if mapish:
            return "地图"
        if any(x in _LIFE_SKILL_NAV_LABELS for x in nav):
            return "角色-生活技能"
        if ms.profile_open is True or any(x in _PROFILE_NAV_LABELS for x in nav):
            return "角色"
        hub = ms.craft_hub_category
        if hub in (1, 2, 3) or detail_s.startswith("craft_hub:"):
            return "制作"
        if ui == "repair" or is_repair_ui_open(
            repair_list_active=ms.repair_list_active,
            repair_modal_open=ms.repair_modal_open,
        ):
            return "修理界面"
        if ms.menu_open is True:
            return "菜单"
        if ms.menu_open is False or ui in ("hud", "combat"):
            combat = _relation_combat_hud(ms, ui, icons)
            if combat is True or ui == "combat":
                return "主界面战斗"
            return "主界面非战斗"

    if ui == "dialog":
        return "对话"
    if ui == "inventory":
        return "背包"
    if str(ui).startswith("map_"):
        return "地图"
    if any(x in _LIFE_SKILL_NAV_LABELS for x in nav):
        return "角色-生活技能"
    if any(x in _PROFILE_NAV_LABELS for x in nav):
        return "角色"
    if ui == "repair":
        return "修理界面"
    if ui == "menu":
        return "菜单"
    if detail_s.startswith("craft_hub:"):
        return "制作"
    if ui == "combat":
        return "主界面战斗"
    if ui == "hud":
        combat = _relation_combat_hud(ms, ui, icons)
        return "主界面战斗" if combat is True else "主界面非战斗"
    if ui == "quest":
        return "任务界面"
    if ui == "chat":
        return "聊天"
    if ui == "traveling":
        return "行进中"
    if ui == "npc_card":
        return "NPC卡片"
    return UI_LABELS.get(ui, ui or "未知")


def _hard_panel_open(ms: MemSceneInfo) -> bool:
    """可信硬面板：对话 / 背包 / 菜单 / 地图。"""
    from mm_assist.memory.readers.ui_stack import stack_inv

    if ms.dialog_active:
        return True
    names = list(getattr(ms, "ui_stack_names", ()) or ())
    if stack_inv(names) and ms.menu_open is None:
        return True
    if ms.menu_open:
        return True
    return False


def classify_ui_mode(
    ms: MemSceneInfo | None,
    *,
    img: np.ndarray | None = None,
    fingerprints: dict[str, str] | None = None,
    samples_dir: Path | None = None,
    regions: dict[str, str] | None = None,
) -> tuple[UiKind, str, str, str, str, str, str]:
    """内存优先（硬面板）+ 指纹兜底。

    返回 (ui, detail, map_level, geo, mem_ui, mem_geo, source)
    source: mem | fp | roi | unknown
    """
    from mm_assist.memory.readers.ui_stack import stack_inv
    from mm_assist.perception.ui_mode_fp import (
        chat_roi_hint,
        compute_fingerprints,
        match_fingerprint,
    )

    ui, detail, lv, geo, mem_ui, mem_geo = _from_mem_scene(ms, regions=regions)
    fps = fingerprints if fingerprints is not None else compute_fingerprints(img)

    def _fp_hit() -> tuple[UiKind, str, str, str, str, str, str] | None:
        hit = match_fingerprint(fps, samples_dir=samples_dir)
        if hit is None or hit.label not in (
            "hud",
            "combat",
            "inventory",
            "map_continent",
            "map_region",
            "map_town",
            "chat",
            "dialog",
            "repair",
            "menu",
            "quest",
            "traveling",
            "npc_card",
        ):
            return None
        label: UiKind = hit.label  # type: ignore[assignment]
        return (
            label,
            UI_DETAIL_LABELS.get(label, "") or f"fp={hit.distance:.1f}",
            lv,
            geo,
            UI_LABELS.get(label, label),
            mem_geo,
            "fp",
        )

    if ms is None:
        hit_t = _fp_hit()
        if hit_t is not None:
            return hit_t
        return ui, detail, lv, geo, mem_ui, mem_geo, "unknown"

    # 硬面板（地图/对话/菜单/修理）内存为真 → 直接信内存（不含 inv/chat）
    if _hard_panel_open(ms) and ui not in ("unknown", "hud"):
        return ui, detail, lv, geo, mem_ui, mem_geo, "mem"

    hit_t = _fp_hit()
    if hit_t is not None:
        return hit_t

    if (
        not _hard_panel_open(ms)
        and chat_roi_hint(img)
        and not (ms.map_open or stack_inv(list(getattr(ms, "ui_stack_names", ()) or ())))
    ):
        return "chat", "ROI 启发", lv, geo, UI_LABELS["chat"], mem_geo, "roi"

    return ui, detail, lv, geo, mem_ui, mem_geo, "mem" if ms.status == "ok" else "unknown"


@dataclass
class SceneSnapshot:
    ui: UiKind = "unknown"
    ui_detail: str = ""
    ui_source: str = ""  # mem | fp | roi | unknown
    ui_params: str = ""  # 状态栏：栈就绪后才 stp=0/1；未就绪 —
    ui_stack: str = ""  # 状态栏新行：UIAPI 面板栈（类名中文）
    ui_relation: str = ""  # 状态栏：人话关系（主界面·细节 + 叠加）
    ui_fine: str = ""  # 旁证字段；主栏不展（2026-08-02 收敛恒空）
    map_level: str = "unknown"
    geo: str = "未知"
    in_dunbarton_town: bool | None = None
    mem_xy: tuple[float, float] | None = None
    mem_addr: str = ""
    mem_status: str = "pending"  # ok | resolving | …
    memory_diagnostic: dict = field(default_factory=dict)
    stop_diagnostic: dict = field(default_factory=dict)
    mem_source: str = ""
    mem_ui: str = ""  # 内存判定的界面提示
    mem_geo: str = ""  # 内存频道/地图层（非地名）
    pixel_hints: list[str] = field(default_factory=list)
    ocr_ms: float = 0.0
    mem_ms: float = 0.0
    total_ms: float = 0.0
    ts: float = 0.0
    blob_sample: str = ""

    @property
    def ui_label(self) -> str:
        return UI_LABELS.get(self.ui, self.ui)

    @property
    def map_level_label(self) -> str:
        return MAP_LEVEL_LABELS.get(self.map_level, self.map_level)

    @property
    def mem_status_label(self) -> str:
        if self.mem_status.startswith("open_failed:"):
            reason = self.mem_status.removeprefix("open_failed:")
            return MEM_STATUS_LABELS.get(reason, MEM_STATUS_LABELS["open_failed"])
        return MEM_STATUS_LABELS.get(self.mem_status, self.mem_status)

    @property
    def mem_source_label(self) -> str:
        return MEM_SOURCE_LABELS.get(self.mem_source, self.mem_source or "—")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ui_label"] = self.ui_label
        d["mem_status_label"] = self.mem_status_label
        return d


def _pixel_hints(img: np.ndarray) -> list[str]:
    """轻量像素特征（不依赖 OCR）。"""
    hints: list[str] = []
    h, w = img.shape[:2]
    if h < 40 or w < 40:
        return hints
    # 右上：小地图区有一定纹理 → 可能 HUD
    corner = img[0 : max(1, h // 6), int(w * 0.85) : w]
    if float(corner.std()) > 12:
        hints.append("minimap_tex")
    # 右下绿块：前往/自动移动
    br = img[int(h * 0.78) : h, int(w * 0.70) : w]
    if br.size:
        b, g, r = cv2.split(br)
        green = (g.astype(np.int16) > b.astype(np.int16) + 25) & (
            g.astype(np.int16) > r.astype(np.int16) + 15
        )
        if float(np.mean(green)) > 0.01:
            hints.append("green_br")
    # 中右大面板（背包/菜单）偏暗遮罩
    mid = img[int(h * 0.15) : int(h * 0.85), int(w * 0.55) : int(w * 0.95)]
    if mid.size and float(np.mean(mid)) < 55 and float(mid.std()) < 40:
        hints.append("dark_panel")
    return hints


def classify_ui(
    lines: list[OcrLine],
    img: np.ndarray | None,
    win: WindowInfo,
    *,
    pixel_hints: list[str] | None = None,
) -> tuple[UiKind, str, str]:
    """返回 (ui, detail, map_level)。未知则 unknown。"""
    b = _blob(lines)
    hints = pixel_hints or []
    lv = "unknown"
    if _map_opened_blob(b) or any(n in b for n in CONTINENT_NEEDLES):
        lv = _map_level(b, lines, win)

    if _is_inventory_blob(b) or sum(1 for k in INVENTORY_MARKS if k in b) >= 2:
        return "inventory", "装备/背包面板", "unknown"

    repair_hits = sum(1 for k in REPAIR_MARKS if k in b)
    if repair_hits >= 1 and ("修理" in b or "維修" in b or "维修" in b):
        return "repair", "修理相关界面", lv if lv != "unknown" else "unknown"

    if sum(1 for k in MENU_MARKS if k in b) >= 2:
        return "menu", "系统/ESC 菜单", "unknown"

    st = classify_state(lines, img)
    if st == "traveling":
        return "traveling", "自动移动/停走", lv
    if st == "npc_card":
        return "npc_card", "NPC 卡片/前往", lv
    if st == "dialog":
        return "dialog", "NPC 对话", lv
    if st in ("continent", "region") or lv in ("continent", "region", "town"):
        if lv == "town":
            return "map_town", "城镇详图", "town"
        if lv == "continent" or st == "continent":
            return "map_continent", "大陆/世界地图", "continent"
        if lv == "region" or st == "region":
            return "map_region", "地区中景", "region"
        return "map_region", "地图（层未明）", lv

    if any(n in b for n in STOP_NEEDLES):
        return "traveling", "停走标识", lv
    if any(n in b for n in GOTO_NEEDLES):
        return "npc_card", "见前往文案", lv
    if any(n in b for n in DIALOG_NEEDLES) and ("選擇" in b or "选择" in b or "再見" in b or "再见" in b):
        return "dialog", "对话选项", lv

    if st == "hud" or ("minimap_tex" in hints and not _map_opened_blob(b)):
        return "hud", "主界面 HUD", "unknown"

    if not b.strip() and "minimap_tex" in hints:
        return "hud", "像素像 HUD（OCR 空）", "unknown"

    return "unknown", "未识别", lv


def classify_geo(lines: list[OcrLine], win: WindowInfo, ui: UiKind, map_level: str) -> tuple[str, bool | None]:
    """地理标签 + 是否杜巴頓城内（None=未知）。"""
    b = _blob(lines)
    titles = []
    for L in lines:
        c = box_center_norm(L, win.width, win.height)
        if not c or c[1] > 0.12:
            continue
        titles.append(normalize_text(L.text or ""))
    title_blob = "".join(titles)

    if any(k in title_blob or k in b for k in GEO_TOWN_MARKS):
        if map_level == "town" or ui == "map_town":
            return "杜巴頓·城镇详图", True
        if map_level == "region":
            return "杜巴頓·地区", None
        return "杜巴頓附近", None

    for k in ("库漢", "庫漢", "库汉"):
        if k in title_blob:
            return "库漢（非杜巴）", False
    if "莱比" in b or "萊比" in b:
        return "莱比敦方向", False
    if any(n in b for n in CONTINENT_NEEDLES) or "欧拉大睦" in b:
        if map_level == "continent" or ui == "map_continent":
            return "歐拉大陸·全图", None
        return "歐拉大陸", None

    # HUD 弱线索
    if ui == "hud":
        if "杜巴" in b:
            return "HUD提及杜巴", None
        return "未知（HUD未开图）", None

    if ui in ("inventory", "menu", "repair", "dialog", "npc_card", "traveling"):
        return "未知（非地图界面）", None

    return "未知", None


def skip_fullframe_ocr_on_main_hud(ms: MemSceneInfo | None) -> bool:
    """主界面禁止全帧OCR；dlg仅表对话态，认页规格见PRODUCT REQ_UI_PARAMS。"""
    if ms is None:
        return False
    return ms.menu_open is False


def _ocr_scaled(img: np.ndarray, scale: float = 0.55, min_conf: float = 0.35) -> list[OcrLine]:
    if scale >= 0.99:
        return ocr_image(img, min_confidence=min_conf)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    lines = ocr_image(small, min_confidence=min_conf)
    # box 坐标缩放回原图（供 map_level 顶栏判断）
    inv = 1.0 / scale
    out: list[OcrLine] = []
    for L in lines:
        box = None
        if L.box:
            box = [[float(p[0]) * inv, float(p[1]) * inv] for p in L.box]
        out.append(OcrLine(text=L.text, confidence=L.confidence, box=box))
    return out


class SceneMonitor:
    """截屏+OCR+内存 → SceneSnapshot。"""

    def __init__(
        self,
        cfg: AppConfig,
        frame: GameFrame | None = None,
        *,
        mem_path: Path | None = None,
        ocr_scale: float = 0.55,
        icon_refresh_s: float | None = None,
        action_refresh_s: float | None = None,
    ) -> None:
        import os
        import threading

        self.cfg = cfg
        self.frame = frame
        self.ocr_scale = ocr_scale
        root = Path(cfg.resolve(cfg.paths.knowledge))
        self._mem = MemPositionReader(mem_path or mem_pos_path(root))
        self._scene_mem = Il2CppSceneReader(scene_mem_path(root))
        self._regions = load_channel_regions(channel_regions_path(root))
        self.last: SceneSnapshot | None = None
        # 本拍内存态：状态栏落盘界面态（state_ui_feed）取 hub / prf 用
        self.last_ms: MemSceneInfo | None = None
        # icon 堆扫在后台；默认 0.5s，避免 cbt 卡 2s 才变
        if icon_refresh_s is None:
            try:
                icon_refresh_s = float(os.environ.get("MM_UI_FINE_ICON_S", "0.5"))
            except ValueError:
                icon_refresh_s = 0.5
        self._icon_refresh_s = max(0.2, float(icon_refresh_s))
        # 动作中旁证（像素）：跟状态栏主周期，默认 0.5s；与 icon 解耦
        if action_refresh_s is None:
            try:
                action_refresh_s = float(os.environ.get("MM_UI_FINE_ACTION_S", "0.5"))
            except ValueError:
                action_refresh_s = 0.5
        self._action_refresh_s = max(0.2, float(action_refresh_s))
        self._icon_reader = None
        self._icon_summary = None
        self._icon_last_ts = 0.0
        self._icon_lock = threading.Lock()
        self._icon_busy = False
        self._icon_client = (1920, 1440)
        self._action_reader = None
        self._action_summary = None
        self._action_last_ts = 0.0
        self._action_busy = False
        self._stop_mem: bool | None = None
        self._stop_image: bool | None = None
        self._stop_image_ts = 0.0
        self._stop_mem_last_ts = 0.0
        self._stop_mem_attempt_ts = 0.0
        self._stop_pixel_busy = False
        self._stop_pixel_target = None
        self._stop_pixel_generation = 0
        self._stop_pixel_closed = False
        self._stop_pixel_wake = threading.Event()
        self._stop_pixel_thread = None
        # 动作中像素跟主周期；下限 0.35s（原 1.0s 让 stp 钝）
        self._stop_pixel_min_s = max(0.35, float(self._action_refresh_s))
        self._last_win = None
        # 菜单图标旁证：后台刷新；默认 0.8s（原 0.35s 同步 OCR 会拖死四参上栏）
        try:
            menu_nav_s = float(os.environ.get("MM_MENU_NAV_S", "0.8"))
        except ValueError:
            menu_nav_s = 0.8
        self._menu_nav_refresh_s = max(0.35, min(2.0, float(menu_nav_s)))
        self._menu_nav_line = ""
        self._menu_nav_last_ts = 0.0
        self._menu_nav_busy = False
        self._menu_nav_img = None
        self._menu_nav_menu_open: bool | None = None
        self._params_menu_open: bool | None = None  # menu 边沿催 icon/stp/nav
        self._params_miss_since = 0.0  # params=— 起算，用于触发补预热
        self._stack_reader = None
        self._stack_reuse_started: float | None = None
        self._stack_reuse_n = 0

    def warm_load_progress(self) -> tuple[int, int]:
        return self._scene_mem.warm_load_progress()

    def set_warm_progress(self, fn) -> None:
        self._scene_mem.set_warm_progress(fn)

    def close(self) -> None:
        self._close_stop_pixel()
        self._mem.close()
        self._scene_mem.close()
        if self._icon_reader is not None:
            try:
                self._icon_reader.close()
            except Exception:
                pass
            self._icon_reader = None
        if self._action_reader is not None:
            try:
                self._action_reader.close()
            except Exception:
                pass
            self._action_reader = None
        if self._stack_reader is not None:
            try:
                self._stack_reader.close()
            except Exception:
                pass
            self._stack_reader = None

    def reload_regions(self) -> None:
        root = Path(self.cfg.resolve(self.cfg.paths.knowledge))
        self._regions = load_channel_regions(channel_regions_path(root))

    def _refresh_icons(self, *, client_w: int = 1920, client_h: int = 1440) -> None:
        """后台读 icon（重堆扫）。动作中旁证走 ``_maybe_refresh_action``，互不堵。"""
        import threading

        self._icon_client = (int(client_w), int(client_h))
        now = time.time()
        with self._icon_lock:
            if self._icon_busy:
                return
            has_icon = self._icon_summary is not None
            if has_icon and (now - self._icon_last_ts) < self._icon_refresh_s:
                return
            self._icon_busy = True

        def _work() -> None:
            try:
                from mm_assist.memory.readers.hud_icons import Il2CppHudMainIconsReader
                from mm_assist.perception.ui_fine import summarize_hud_icons

                cw, ch = self._icon_client
                if self._icon_reader is None:
                    self._icon_reader = Il2CppHudMainIconsReader()
                # 状态栏只要 Show/可点旁证；跳过 rect 链
                frame = self._icon_reader.capture(
                    client_w=cw, client_h=ch, need_rect=False, max_wall_s=1.2
                )
                if frame is not None:
                    summary = summarize_hud_icons(frame)
                    with self._icon_lock:
                        self._icon_summary = summary
                        self._icon_last_ts = time.time()
            except Exception as exc:
                log.debug("hud icons refresh: %s", exc)
            finally:
                with self._icon_lock:
                    self._icon_busy = False

        threading.Thread(target=_work, name="hud-icons-bg", daemon=True).start()

    def _maybe_kick_stop_pixel(self, win: WindowInfo | None) -> None:
        """One recurring pixel reader, independent of slow memory/UI ticks."""
        import threading

        key = self._stop_pixel_key(win)
        with self._icon_lock:
            if getattr(self, '_stop_pixel_closed', False):
                return
            self._stop_pixel_closed = False
            if not hasattr(self, '_stop_pixel_wake'):
                self._stop_pixel_wake = threading.Event()
            target = getattr(self, '_stop_pixel_target', None)
            if key != self._stop_pixel_key(target):
                self._stop_pixel_generation = getattr(self, '_stop_pixel_generation', 0) + 1
                self._stop_image = self._stop_mem = None
                self._stop_image_ts = self._stop_mem_last_ts = 0.
                self._stop_pixel_target = win
                self._stop_pixel_wake.set()
            if win is None or getattr(self, '_stop_pixel_busy', False):
                return
            self._stop_pixel_busy = True

        def _work() -> None:
            try:
                while True:
                    with self._icon_lock:
                        target = self._stop_pixel_target
                        if self._stop_pixel_closed or target is None:
                            return
                        generation = self._stop_pixel_generation
                        self._stop_pixel_wake.clear()
                        sampled_at = time.time()
                    try:
                        pix = self._sample_stop_pixel_now(target)
                        self._apply_stop_pixel(pix, sampled_at=sampled_at, generation=generation)
                    except Exception as exc:
                        log.debug('stop pixel bg: %s', exc)
                    self._stop_pixel_wake.wait(self._stop_pixel_min_s)
            finally:
                with self._icon_lock:
                    self._stop_pixel_busy = False

        # Publish the single worker before starting it, under the busy flag.
        self._stop_pixel_thread = threading.Thread(target=_work, name='stop-pixel-bg', daemon=True)
        self._stop_pixel_thread.start()

    @staticmethod
    def _stop_pixel_key(win):
        if win is None:
            return None
        return tuple(getattr(win, name, None) for name in ('hwnd', 'left', 'top', 'width', 'height'))

    def _close_stop_pixel(self) -> None:
        with self._icon_lock:
            self._stop_pixel_closed = True
            self._stop_pixel_generation = getattr(self, '_stop_pixel_generation', 0) + 1
            self._stop_pixel_target = None
            self._stop_image = self._stop_mem = None
            self._stop_image_ts = self._stop_mem_last_ts = 0.
            wake = getattr(self, '_stop_pixel_wake', None)
            if wake is not None:
                wake.set()

    def _sample_stop_pixel_now(self, win: WindowInfo | None) -> bool | None:
        if win is None:
            return None
        try:
            from mm_assist.platform.capture import grab_client
            from mm_assist.perception.hud_sense import detect_stop_icon
            from mm_assist.platform.window import refresh_window
            from mm_assist.memory.process import pid_from_hwnd

            current = refresh_window(win)
            pid = pid_from_hwnd(current.hwnd)
            if not pid:
                return None
            img = grab_client(current, method="printwindow")
            if (img is None or not img.size or img.ndim != 3 or img.shape[2] != 3
                    or min(img.shape[:2]) < 240 or np.max(img) < 5
                    or img.shape[:2] != (current.height, current.width)
                    or self._stop_pixel_key(refresh_window(current)) != self._stop_pixel_key(current)
                    or pid_from_hwnd(current.hwnd) != pid):
                return None
            return detect_stop_icon(img, ocr_space=False)
        except Exception as exc:
            log.debug("stop icon pixel: %s", exc)
            return None

    def _publish_fused_stop_unlocked(self) -> None:
        from mm_assist.perception.ui_fine import HudActionSummary, merge_stop_visible

        now = time.time()
        mem = self._stop_mem if 0 <= now-self._stop_mem_last_ts <= STOP_SAMPLE_MAX_AGE_S else None
        pixel = self._stop_image if 0 <= now-self._stop_image_ts <= STOP_SAMPLE_MAX_AGE_S else None
        fused, _src = merge_stop_visible(mem, pixel)
        if self._action_summary is None:
            self._action_summary = HudActionSummary()
        self._action_summary.stop_visible = fused
        if fused is not None:
            self._action_last_ts = time.time()

    def _apply_stop_pixel(self, pix: bool | None, *, sampled_at: float | None = None,
                          generation: int | None = None) -> None:
        """写入图像降级通道再合并；``None`` 不覆盖旧图像结论。"""
        if pix is not True and pix is not False:
            return
        stamp = time.time() if sampled_at is None else sampled_at
        with self._icon_lock:
            if (getattr(self, '_stop_pixel_closed', False)
                    or (generation is not None and generation != self._stop_pixel_generation)):
                return
            if stamp < self._stop_image_ts or not 0 <= time.time()-stamp <= STOP_SAMPLE_MAX_AGE_S:
                return
            self._stop_image = pix
            self._stop_image_ts = stamp
            self._publish_fused_stop_unlocked()

    def _maybe_refresh_action(self) -> None:
        """周期内存轻扫停钮，与图像通道并行融合（不挡主拍）。"""
        import threading

        now = time.time()
        with self._icon_lock:
            if self._action_busy:
                return
            if (now - self._stop_mem_attempt_ts) < max(2.0, self._action_refresh_s):
                return
            self._action_busy = True
            self._stop_mem_attempt_ts = now
        threading.Thread(
            target=self._refresh_action_light,
            name="hud-action-bg",
            daemon=True,
        ).start()

    def _refresh_action_light(self) -> None:
        """内存轻扫停钮，写入 ``_stop_mem`` 后合并（内存优先）。"""
        try:
            from mm_assist.memory.readers.hud_action import Il2CppHudActionReader
            from mm_assist.perception.ui_fine import summarize_hud_action

            with self._icon_lock:
                generation = getattr(self, '_stop_pixel_generation', 0)
            cw, ch = self._icon_client
            if self._action_reader is None:
                self._action_reader = Il2CppHudActionReader(max_ui=24, max_buttons=96)
            sampled_at = time.time()
            aframe = self._action_reader.capture(
                client_w=cw,
                client_h=ch,
                need_rect=False,
                max_wall_s=0.6,
                stop_on_strong=True,
            )
            if aframe is None:
                return
            mem_summary = summarize_hud_action(aframe)
            mem_stop = getattr(mem_summary, "stop_visible", None)
            if mem_stop is not True and mem_stop is not False:
                return
            with self._icon_lock:
                if (getattr(self, '_stop_pixel_closed', False)
                        or generation != getattr(self, '_stop_pixel_generation', 0)
                        or sampled_at < self._stop_mem_last_ts
                        or not 0 <= time.time()-sampled_at <= STOP_SAMPLE_MAX_AGE_S):
                    return
                self._stop_mem = mem_stop
                self._stop_mem_last_ts = sampled_at
                self._publish_fused_stop_unlocked()
        except Exception as exc:
            log.debug("hud action refresh: %s", exc)
        finally:
            with self._icon_lock:
                self._action_busy = False

    def _fine_line(self, ms: MemSceneInfo | None) -> str:
        """主栏不展细界面（2026-08-02 收敛）；旁证仅 readonly_ui_fine job。"""
        return ""

    def _kick_scene_warm(self, hwnd: int | None) -> None:
        """params 长期空时后台限时 warm（互斥 pause_resolve）。"""
        import threading

        scene = self._scene_mem
        if (scene._pause_resolve or scene.resolving
                or scene._ui_refreshing or scene._map_rescanning):
            return
        scene._pause_resolve = True

        def _work() -> None:
            try:
                warm, _hits = scene.warm_bundle(hwnd, max_wall_s=4.5, max_regions=0)
                log.info(
                    "scene re-warm status=%s menu=%s err=%s",
                    getattr(warm, "status", None),
                    getattr(warm, "menu_open", None),
                    scene.last_error,
                )
            except Exception as exc:
                log.warning(
                    "scene re-warm fail: %s: %r",
                    type(exc).__name__,
                    exc,
                )
            finally:
                scene._pause_resolve = False
                if scene._cache is None:
                    scene.schedule_resolve(hwnd=hwnd)

        threading.Thread(target=_work, name="scene-rewarm", daemon=True).start()

    def ensure_frame(self, win: WindowInfo | None = None) -> GameFrame:
        if self.frame is None:
            if win is None:
                raise ValueError("SceneMonitor: 首次 tick 需要 WindowInfo")
            self.frame = ensure_game_frame(self.cfg, win)
        else:
            self.frame.refresh()
        return self.frame

    def _params_line(self, ms: MemSceneInfo | None) -> str:
        with self._icon_lock:
            self._publish_fused_stop_unlocked()
            icons = self._icon_summary
            action = self._action_summary
            nav = self._menu_nav_line
        return format_ui_params(ms, icons=icons, action=action, menu_nav=nav)

    def _current_stop_unlocked(self, now: float):
        from mm_assist.perception.ui_fine import merge_stop_visible
        mem_age = now-self._stop_mem_last_ts
        image_age = now-self._stop_image_ts
        mem = self._stop_mem if 0 <= mem_age <= STOP_SAMPLE_MAX_AGE_S else None
        image = self._stop_image if 0 <= image_age <= STOP_SAMPLE_MAX_AGE_S else None
        value, source = merge_stop_visible(mem, image)
        stamp = (self._stop_mem_last_ts if source == 'mem' else
                 self._stop_image_ts if source == 'image' else 0.)
        return value, source, stamp

    def read_stop_sample(self) -> dict:
        """Read cached stop evidence independently of the slow full watch tick."""
        with self._icon_lock:
            value, _, stamp = self._current_stop_unlocked(time.time())
            if (getattr(self, '_stop_pixel_closed', False)
                    or getattr(self, '_stop_pixel_target', None) is None
                    or not getattr(self.last_ms, 'ui_stack_ready', False)):
                value, stamp = None, 0.
            return {'stp': int(value) if value is not None else None,
                    'timestamp': stamp,
                    'generation': getattr(self, '_stop_pixel_generation', 0)}

    def stop_diagnostic(self) -> dict:
        """Source ages only; never a separate input or page permission."""
        with self._icon_lock:
            now = time.time()
            image_age = now-self._stop_image_ts
            value, source, _ = self._current_stop_unlocked(now)
            return {'value': value, 'source': source,
                    'memory_sampled_at': self._stop_mem_last_ts,
                    'image_sampled_at': self._stop_image_ts,
                    'image_age_s': round(image_age, 3) if self._stop_image_ts else None,
                    'pixel_worker': bool(getattr(self, '_stop_pixel_busy', False)),
                    'generation': getattr(self, '_stop_pixel_generation', 0)}

    def _clear_stack_reuse(self) -> None:
        self._stack_reuse_started = None
        self._stack_reuse_n = 0

    def _reuse_stack_snap(self, *, now: float | None = None) -> Any | None:
        """栈读失败时沿用上一拍 names/ready/line；超时则未就绪。无上一拍则 None。"""
        last_ms = self.last_ms
        if last_ms is None:
            return None
        names = [str(n) for n in list(getattr(last_ms, "ui_stack_names", ()) or ())]
        ready = bool(getattr(last_ms, "ui_stack_ready", False))
        if not ready and not names:
            return None
        from mm_assist.memory.readers.ui_stack import UiStackSnap

        t = float(now if now is not None else time.time())
        if getattr(self, "_stack_reuse_started", None) is None:
            self._stack_reuse_started = t
            self._stack_reuse_n = 0
        self._stack_reuse_n = int(getattr(self, "_stack_reuse_n", 0) or 0) + 1
        elapsed = t - float(self._stack_reuse_started)
        if elapsed > STACK_REUSE_MAX_S or self._stack_reuse_n > STACK_REUSE_MAX_TICKS:
            return UiStackSnap(
                names=[],
                line="—",
                n=0,
                error="stack_reuse_expired",
                ready=False,
            )
        line = "—"
        if self.last is not None:
            line = str(getattr(self.last, "ui_stack", "") or "—")
        return UiStackSnap(
            names=names,
            line=line,
            n=len(names),
            error="stack_read_reused",
            ready=ready,
        )

    def _stack_snap(self) -> Any:
        from mm_assist.memory.readers.ui_stack import shared_ui_stack_reader, UiStackSnap

        try:
            if self._stack_reader is None:
                self._stack_reader = shared_ui_stack_reader()
            snap = self._stack_reader.read()
            self._clear_stack_reuse()
            return snap
        except Exception as exc:
            log.debug("ui stack: %s", exc)
            reused = self._reuse_stack_snap()
            if reused is not None:
                return reused
            return UiStackSnap(line="—", error="stack_read_failed", ready=False)

    def _bind_stack(self, ms: MemSceneInfo | None) -> Any:
        """读栈并写 bits；失败不抛、可复用上一拍。调用方仍须写出 mem_xy/geo。"""
        from mm_assist.memory.readers.ui_stack import UiStackSnap

        stack: Any = None
        try:
            stack = self._stack_snap()
            if ms is not None:
                apply_stack_bits(ms, stack)
            return stack
        except Exception as exc:
            log.debug("ui stack bind: %s", exc)
            stack = self._reuse_stack_snap() or UiStackSnap(
                line="—", error="stack_read_failed", ready=False
            )
            if ms is not None:
                try:
                    apply_stack_bits(ms, stack)
                except Exception as apply_exc:
                    log.debug("ui stack apply: %s", apply_exc)
            return stack

    def _maybe_refresh_menu_nav(
        self,
        img: np.ndarray | None = None,
        win: WindowInfo | None = None,
        *,
        menu_open: bool | None = None,
    ) -> None:
        """后台菜单识别（模糊图标 + OCR 辅判），供关系判定；不上栏 nav=。

        不得同步堵 tick_lean：PrintWindow+OCR 可达数百 ms～数秒，会让四参显示变钝。
        """
        import threading

        now = time.time()
        with self._icon_lock:
            if self._menu_nav_busy:
                return
            menu_changed = (
                menu_open is not None and menu_open != self._menu_nav_menu_open
            )
            due = now - float(self._menu_nav_last_ts or 0.0) >= self._menu_nav_refresh_s
            if not menu_changed and not due:
                return
            self._menu_nav_busy = True
            if menu_open is not None:
                self._menu_nav_menu_open = menu_open
            open_flag = self._menu_nav_menu_open

        frame = img
        use_win = win or self._last_win

        def _work() -> None:
            try:
                from mm_assist.perception.menu_icons import (
                    format_menu_nav,
                    probe_menu_icons_fast,
                )

                grab = frame
                if grab is None and use_win is not None:
                    from mm_assist.platform.capture import grab_client

                    grab = grab_client(use_win, method="printwindow")
                if grab is None:
                    return
                probe = probe_menu_icons_fast(grab, menu_open=open_flag)
                line = format_menu_nav(probe)
                with self._icon_lock:
                    self._menu_nav_line = line
                    self._menu_nav_last_ts = time.time()
            except Exception as exc:
                log.debug("menu nav refresh: %s", exc)
            finally:
                with self._icon_lock:
                    self._menu_nav_busy = False

        threading.Thread(target=_work, name="menu-nav-bg", daemon=True).start()

    def tick(self, win: WindowInfo | None = None) -> SceneSnapshot:
        t0 = time.perf_counter()
        fr = self.ensure_frame(win)
        img = fr.grab(self.cfg)
        winfo = fr.to_window()
        hints = _pixel_hints(img)

        t_mem = time.perf_counter()
        # 状态栏禁止同步堆扫（首次约 10s）；后台 resolve
        mp = self._mem.read(hwnd=winfo.hwnd, allow_scan=False)
        ms: MemSceneInfo | None = self._scene_mem.read(
            hwnd=winfo.hwnd, allow_scan=False, refresh_ui=False, lean_ui_refresh=True
        )
        stack = self._bind_stack(ms)
        mem_ms = (time.perf_counter() - t_mem) * 1000.0

        # OCR仅作摘要/调试旁证；主界面禁止全帧OCR（PRODUCT REQ_UI_PARAMS）。
        t_ocr = time.perf_counter()
        lines: list[OcrLine] = []
        if skip_fullframe_ocr_on_main_hud(ms):
            ocr_ms = 0.0
        else:
            try:
                lines = _ocr_scaled(img, scale=self.ocr_scale)
            except Exception as exc:
                log.debug("ocr fail: %s", exc)
                lines = []
            ocr_ms = (time.perf_counter() - t_ocr) * 1000.0

        mem_xy = mp.as_xy() if mp else None
        mem_source = mp.source if mp else ""
        mem_addr = ""
        if mp:
            mem_status = "ok"
        else:
            err = self._mem.last_error or "pending"
            if self._mem.resolving or err in ("resolving", "cache_miss"):
                mem_status = "resolving"
            else:
                mem_status = err

        ui, detail, lv, geo, mem_ui, mem_geo, ui_src = classify_ui_mode(
            ms, img=img, regions=self._regions
        )

        self._last_win = winfo
        self._maybe_kick_stop_pixel(winfo)
        self._refresh_icons(client_w=int(winfo.width), client_h=int(winfo.height))
        self._maybe_refresh_action()
        self._maybe_refresh_menu_nav(
            img, winfo, menu_open=None if ms is None else ms.menu_open
        )

        b = _blob(lines)
        with self._icon_lock:
            rel_icons = self._icon_summary
            rel_nav = self._menu_nav_line
        snap = self._apply_sticky(
            SceneSnapshot(
                ui=ui,
                ui_detail=detail,
                ui_source=ui_src,
                ui_params=self._params_line(ms),
                ui_stack=str(getattr(stack, "line", "") or "—"),
                ui_relation=format_ui_relation(
                    ms, ui, detail, icons=rel_icons, menu_nav=rel_nav
                ),
                ui_fine=self._fine_line(ms),
                map_level=lv,
                geo=geo,
                in_dunbarton_town=None,
                mem_xy=mem_xy,
                mem_addr=mem_addr,
                mem_status=mem_status,
                mem_source=mem_source,
                mem_ui=mem_ui,
                mem_geo=mem_geo,
                pixel_hints=hints,
                ocr_ms=ocr_ms,
                mem_ms=mem_ms,
                total_ms=(time.perf_counter() - t0) * 1000.0,
                ts=time.time(),
                blob_sample=re.sub(r"\s+", "", b)[:96],
            )
        )
        snap.memory_diagnostic = self.memory_diagnostic(stack_error=str(getattr(stack, 'error', '') or ''))
        snap.stop_diagnostic = self.stop_diagnostic()
        self.last = snap
        self.last_ms = ms
        return snap

    def tick_lean(self, win: WindowInfo | None = None) -> SceneSnapshot:
        """高效：只读内存（坐标 + 界面 + 地理），不截屏、不 OCR。"""
        t0 = time.perf_counter()
        self._maybe_kick_stop_pixel(win)
        hwnd = win.hwnd if win is not None else None
        t_mem = time.perf_counter()
        mp = self._mem.read(hwnd=hwnd, allow_scan=False)
        ms: MemSceneInfo | None = self._scene_mem.read(
            hwnd=hwnd, allow_scan=False, refresh_ui=False, lean_ui_refresh=True
        )
        if ms is None:
            # 预热已写 cache 但本拍抢锁失败 / 瞬时 None → 仍用缓存，勿整栏空白
            ms = getattr(self._scene_mem, "_cache", None)
        if ms is None:
            # 地区定位可暂缺，独立 UI 栈读数仍须传给状态栏；不伪造地名或就绪位。
            ms = MemSceneInfo()
        stack = self._bind_stack(ms)
        mem_ms = (time.perf_counter() - t_mem) * 1000.0
        mem_xy = mp.as_xy() if mp else None
        mem_source = mp.source if mp else ""
        if mp:
            mem_status = "ok"
        else:
            err = self._mem.last_error or "pending"
            if self._mem.resolving or err in ("resolving", "cache_miss"):
                mem_status = "resolving"
            else:
                mem_status = err
        # 预热失败或 HUD 未扫到（menu=None）时 params 会长期 ?/—：超时再 warm
        weak_params = ms is None or not str(getattr(ms, "display_name", "") or "").strip()
        if weak_params:
            now_m = time.time()
            if self._params_miss_since <= 0:
                self._params_miss_since = now_m
            elif (
                now_m - self._params_miss_since >= 12.0
                and not self._scene_mem._pause_resolve
                and not self._scene_mem.resolving
            ):
                self._params_miss_since = now_m
                self._kick_scene_warm(hwnd)
        else:
            self._params_miss_since = 0.0
        # menu 边沿：立刻催 icon / stp / nav / lean 对象补扫（界面参数少半拍）
        cur_menu = None if ms is None else ms.menu_open
        if cur_menu is not None and cur_menu != self._params_menu_open:
            self._params_menu_open = cur_menu
            with self._icon_lock:
                self._icon_last_ts = 0.0
                self._menu_nav_last_ts = 0.0
            try:
                self._scene_mem._next_ui_refresh = 0.0
            except Exception:
                pass
        elif cur_menu is not None:
            self._params_menu_open = cur_menu
        ui, detail, lv, geo, mem_ui, mem_geo, ui_src = classify_ui_mode(
            ms, img=None, regions=self._regions
        )
        cw, ch = 1920, 1440
        if win is not None:
            cw, ch = int(win.width), int(win.height)
            self._last_win = win
        # 预热中跳过截屏；停止钮/菜单识别均后台，主拍只吐内存四参
        warming = bool(getattr(self._scene_mem, "_pause_resolve", False))
        self._refresh_icons(client_w=cw, client_h=ch)
        self._maybe_refresh_action()
        if not warming:
            self._maybe_refresh_menu_nav(
                img=None,
                win=win,
                menu_open=None if ms is None else ms.menu_open,
            )
        with self._icon_lock:
            rel_icons = self._icon_summary
            rel_nav = self._menu_nav_line
        snap = self._apply_sticky(
            SceneSnapshot(
                ui=ui,
                ui_detail=detail,
                ui_source=ui_src,
                ui_params=self._params_line(ms),
                ui_stack=str(getattr(stack, "line", "") or "—"),
                ui_relation=format_ui_relation(
                    ms, ui, detail, icons=rel_icons, menu_nav=rel_nav
                ),
                ui_fine=self._fine_line(ms),
                map_level=lv,
                geo=geo,
                in_dunbarton_town=None,
                mem_xy=mem_xy,
                mem_addr="",
                mem_status=mem_status,
                mem_source=mem_source,
                mem_ui=mem_ui,
                mem_geo=mem_geo,
                pixel_hints=[],
                ocr_ms=0.0,
                mem_ms=mem_ms,
                total_ms=(time.perf_counter() - t0) * 1000.0,
                ts=time.time(),
                blob_sample="",
            )
        )
        snap.memory_diagnostic = self.memory_diagnostic(stack_error=str(getattr(stack, 'error', '') or ''))
        snap.stop_diagnostic = self.stop_diagnostic()
        self.last = snap
        self.last_ms = ms
        return snap

    def memory_diagnostic(self, *, stack_error: str = '') -> dict:
        from mm_assist.memory.diagnostics import memory_diagnostic
        from mm_assist.memory.process import pid_from_hwnd
        win = getattr(self, '_last_win', None)
        pid = pid_from_hwnd(int(getattr(win, 'hwnd', 0) or 0)) if win else None
        stack = getattr(self, '_stack_reader', None)
        return memory_diagnostic(pid, stack_error=stack_error or str(getattr(stack, 'last_error', '') or ''))

    def _apply_sticky(self, snap: SceneSnapshot) -> SceneSnapshot:
        """ms 短暂丢失时保持上次界面/地理，避免面板闪烁。

        注意：ui_params **不** sticky——参数随本拍内存走，
        否则 weak_ui 时会把 inv/menu/dlg 粘死成「卡住」。
        """
        prev = self.last
        if prev is None:
            return snap
        weak_ui = snap.ui == "unknown" or "定位" in (snap.ui_detail or "")
        weak_geo = snap.geo in ("", "—", "未知", "定位中…") or "定位" in snap.geo
        if weak_ui and prev.ui != "unknown":
            snap.ui = prev.ui
            snap.ui_detail = prev.ui_detail
            snap.mem_ui = prev.mem_ui
            snap.ui_source = prev.ui_source or snap.ui_source
            snap.ui_relation = prev.ui_relation or snap.ui_relation
        if weak_geo and prev.geo not in ("", "—", "未知", "定位中…"):
            snap.geo = prev.geo
            snap.mem_geo = prev.mem_geo
            snap.map_level = prev.map_level
        # 坐标：本拍 resolving/空时沿用上次，避免预热已成功仍显示「定位中」
        if snap.mem_xy is None and prev.mem_xy is not None:
            if snap.mem_status in ("", "resolving", "pending", "cache_miss") or "定位" in (
                snap.mem_status_label or ""
            ):
                snap.mem_xy = prev.mem_xy
                snap.mem_status = "ok"
                snap.mem_source = prev.mem_source or snap.mem_source
        return snap
