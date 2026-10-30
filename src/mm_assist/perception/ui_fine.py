"""细界面判态：MainHud icon / HudAction 旁证。

原则（REQ_UI_PARAMS）：
- 主栏「界面：」内存栈优先认页；`stp` 在内存行。inv/menu/dlg 不上栏（由栈类名判断）。
- 旧 cbt/cs/icn/hub/map/prf/act **不上栏**。
- icon **不作唯一主判**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mm_assist.perception.hud_sense import TRAVEL_ROI
from mm_assist.memory.readers.hud_action import HudActionFrame
from mm_assist.memory.readers.hud_icons import HudMainIconSnap, HudMainIconsFrame
from mm_assist.memory.readers.scene import MemSceneInfo

FineKind = Literal[
    "dialog",
    "map",
    "craft",
    "inventory",
    "menu_other",
    "combat_hud",
    "clear_hud",
    "clear_hud_weak",
    "conflict",
    "unknown",
]

Confidence = Literal["high", "mid", "low"]

_KIND_KEY = {
    "InventoryButtonUIComponent": "inventory",
    "MainMenuButtonUIComponent": "menu_esc",
    "MainHudLeftBottomButtonUIComponent": "left_bottom",
    "MainHudTopRightButtonSetUIComponent": "camera",
}

FINE_LABELS: dict[str, str] = {
    "dialog": "对话",
    "map": "地图",
    "craft": "製作",
    "inventory": "物品",
    "menu_other": "其它菜单",
    "combat_hud": "战斗HUD",
    "clear_hud": "清HUD",
    "clear_hud_weak": "清HUD?",
    "conflict": "冲突",
    "unknown": "未知",
}


# 1920×1440 标定 ROI 左上（knowledge/schema/roi_overrides.json · hud_main_icons.md）
_ICON_ROI_CLIENT_XY: dict[str, tuple[int, int]] = {
    "menu_esc": (1592, 30),
    "camera": (1492, 34),
}
_STATE_SHORT = {"Show": "S", "Hide": "H", "Combat": "C"}

# 停止钮 TRAVEL_ROI 左上（1920×1440；绝对客户区未解，用标定 ROI）
_TRAVEL_ROI_CLIENT_XY: tuple[int, int] = (
    int(round(1920 * TRAVEL_ROI[0])),
    int(round(1440 * TRAVEL_ROI[1])),
)


@dataclass
class IconBtnView:
    """单钮旁证。"""

    state: str | None = None  # Show | Hide | Combat | …
    interactable: bool | None = None
    modular_visible: bool | None = None
    rect_w: float | None = None
    rect_h: float | None = None

    @property
    def show_ok(self) -> bool:
        return self.state == "Show" and self.interactable is True

    @property
    def is_combat(self) -> bool:
        return self.state == "Combat"

    @property
    def is_hide(self) -> bool:
        return self.state == "Hide"

    def short(self) -> str:
        st = self.state or "?"
        if self.interactable is True:
            return f"{st[0] if st else '?'}/可"
        if self.interactable is False:
            return f"{st[0] if st else '?'}/禁"
        return st[0] if st else "?"

    def rect_short(self) -> str:
        if self.rect_w is None or self.rect_h is None:
            return ""
        w, h = int(round(self.rect_w)), int(round(self.rect_h))
        if w <= 0 or h <= 0:
            return ""
        if w == h:
            return f"{w}²"
        return f"{w}×{h}"


@dataclass
class HudIconSummary:
    inventory: IconBtnView | None = None
    menu_esc: IconBtnView | None = None
    left_bottom: IconBtnView | None = None
    camera: IconBtnView | None = None
    age_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def has_any(self) -> bool:
        return any(
            x is not None
            for x in (self.inventory, self.menu_esc, self.left_bottom, self.camera)
        )


@dataclass
class HudActionSummary:
    """停止钮（HudAction）旁证：可见态 + ROI/local 尺寸。

    ``stop_visible``：True=停止钮在（行进中）；False=消失（已停）；None=无读数。
    """

    stop_visible: bool | None = None
    layer_visible: bool | None = None
    rect_w: float | None = None
    rect_h: float | None = None
    age_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def has_stop(self) -> bool:
        return self.stop_visible is not None


@dataclass
class FineUiVerdict:
    kind: FineKind
    label: str
    confidence: Confidence
    evidence: str

    def format_line(self) -> str:
        mark = {"high": "✓", "mid": "~", "low": "?"}[self.confidence]
        if self.evidence:
            return f"{self.label}{mark} {self.evidence}"
        return f"{self.label}{mark}"


def _view_from_snap(snap: HudMainIconSnap) -> IconBtnView:
    state = snap.button_state or snap.mhb_state
    if state is None:
        if snap.is_visible is True:
            state = "Show"
        elif snap.is_visible is False:
            state = "Hide"
    rect_w = snap.rect.w if snap.rect is not None else None
    rect_h = snap.rect.h if snap.rect is not None else None
    return IconBtnView(
        state=state,
        interactable=snap.interactable,
        modular_visible=snap.modular_visible,
        rect_w=rect_w,
        rect_h=rect_h,
    )


def _snap_score(snap: HudMainIconSnap) -> tuple[int, int, int]:
    """Prefer Show+interactable, then Show, then has button."""
    v = _view_from_snap(snap)
    show = 2 if v.state == "Show" else (1 if v.state == "Combat" else 0)
    inter = 2 if snap.interactable is True else (1 if snap.interactable is False else 0)
    has_btn = 1 if snap.button_addr else 0
    return (show, inter, has_btn)


def summarize_hud_icons(frame: HudMainIconsFrame | None) -> HudIconSummary:
    """从一帧 icon 探针折叠成四钮摘要。"""
    out = HudIconSummary()
    if frame is None:
        return out
    out.age_ms = float(frame.age_ms or 0.0)
    out.errors = list(frame.errors or [])
    best: dict[str, HudMainIconSnap] = {}
    for snap in frame.icons or []:
        key = _KIND_KEY.get(snap.kind)
        if not key:
            continue
        # 无 Show/Hide/Combat/可见位的幽灵不作旁证
        view = _view_from_snap(snap)
        if view.state not in ("Show", "Hide", "Combat"):
            continue
        prev = best.get(key)
        if prev is None or _snap_score(snap) > _snap_score(prev):
            best[key] = snap
    if "inventory" in best:
        out.inventory = _view_from_snap(best["inventory"])
    if "menu_esc" in best:
        out.menu_esc = _view_from_snap(best["menu_esc"])
    if "left_bottom" in best:
        out.left_bottom = _view_from_snap(best["left_bottom"])
    if "camera" in best:
        out.camera = _view_from_snap(best["camera"])
    return out


def _fmt_icon_part(
    key: str,
    view: IconBtnView,
    roi_xy: tuple[int, int] | None,
) -> str:
    st = _STATE_SHORT.get(view.state or "", "?")
    if roi_xy and view.state in ("Show", "Combat"):
        return f"{key}={st}@{roi_xy[0]},{roi_xy[1]}"
    rect = view.rect_short()
    if rect:
        return f"{key}={st}/{rect}"
    return f"{key}={st}"


def format_ui_combat_from_icons(icons: HudIconSummary | None) -> str:
    """状态栏战斗位：``cbt=1/0`` ← MainHud icon ``Combat``/``Show``。

    2026-08-03 成对：非战斗 inv=Show、战斗 inv=Combat；旧内存 cbt/cs 作废。
    无 Show/Combat 读数则不上栏（避免 ? 充数）。≠够用。
    """
    if icons is None or not icons.has_any():
        return ""
    states: list[str] = []
    for v in (icons.inventory, icons.menu_esc, icons.left_bottom, icons.camera):
        if v is None or v.state not in ("Show", "Hide", "Combat"):
            continue
        states.append(v.state)
    if not states:
        return ""
    if any(s == "Combat" for s in states):
        return "cbt=1"
    if any(s == "Show" for s in states):
        return "cbt=0"
    return ""


def format_ui_icon_hint(icons: HudIconSummary | None) -> str:
    """战斗/非战斗旁证：esc/inv/lb/cam 状态 + ROI 左上或内存 local 尺寸。"""
    if icons is None or not icons.has_any():
        return ""
    parts: list[str] = []
    if icons.menu_esc is not None:
        parts.append(
            _fmt_icon_part("e", icons.menu_esc, _ICON_ROI_CLIENT_XY.get("menu_esc"))
        )
    if icons.inventory is not None:
        parts.append(_fmt_icon_part("i", icons.inventory, None))
    if icons.left_bottom is not None:
        parts.append(_fmt_icon_part("l", icons.left_bottom, None))
    if icons.camera is not None:
        parts.append(
            _fmt_icon_part("c", icons.camera, _ICON_ROI_CLIENT_XY.get("camera"))
        )
    if not parts:
        return ""
    return "icn " + " ".join(parts)


def summarize_hud_action(frame: HudActionFrame | None) -> HudActionSummary:
    """从 HudAction 一帧折叠停止钮可见态 + local 尺寸。

    判定（2026-08-03）：
    - **强钮**：``base_visible`` 且（``interactable`` 或有 rect）→ ``stop_visible=True``
    - 幽灵（可见但无 rect 且不可点）不计
    - **层 IsVisible 单独不足以判停**（易假阳）
    - 扫到关态钮且无强钮 → ``False``；钮列表空/未扫到 → ``None``（勿误报已停）
    """
    out = HudActionSummary()
    if frame is None:
        return out
    out.age_ms = float(frame.age_ms or 0.0)
    out.errors = list(frame.errors or [])

    layer_on: bool | None = None
    for c in frame.ui_components or []:
        vis = c.get("IsVisible") if isinstance(c, dict) else None
        if vis is True:
            layer_on = True
            break
        if vis is False and layer_on is None:
            layer_on = False
    out.layer_visible = layer_on

    strong = False
    saw_off = False
    saw_ghost = False
    best_w: float | None = None
    best_h: float | None = None
    for b in frame.buttons or []:
        if b.base_visible is True:
            if b.rect is None and b.interactable is not True:
                saw_ghost = True
                continue
            strong = True
            if b.rect is not None and (best_w is None or (b.rect.w or 0) > (best_w or 0)):
                best_w, best_h = b.rect.w, b.rect.h
        elif b.base_visible is False:
            saw_off = True

    out.rect_w = best_w
    out.rect_h = best_h
    if strong:
        out.stop_visible = True
    elif saw_off or (saw_ghost and layer_on is not True):
        # 有关态证据，或仅幽灵且层未亮 → 视为停止钮不在
        out.stop_visible = False
    else:
        # 无强钮：可能堆扫超时钮列表空，保持未知
        out.stop_visible = None
    return out


def pixel_stop_visible(travel_state: str | None) -> bool | None:
    """兼容旧 travel_state 字符串旁证（动作中采证优先走 ``detect_stop_icon``）。

    - ``stop``（OCR 停止类）→ True
    - ``hidden`` → False
    - ``navigate`` / 其它 → None（仅绿量不可靠）
    """
    if not travel_state:
        return None
    st = str(travel_state).strip().lower()
    if st in ("stop", "stop_icon"):
        return True
    if st in ("hidden", "no_stop_icon"):
        return False
    return None


def merge_stop_visible(
    mem: bool | None,
    pixel: bool | None,
) -> tuple[bool | None, str]:
    """主界面停态：内存 ``HudAction`` 优先；无读数再降级图像/OCR。返回 (stop_visible, src)。"""
    if mem is True or mem is False:
        return mem, "mem"
    if pixel is True or pixel is False:
        return pixel, "image"
    return None, "none"


def status_bar_stop_visible(
    mem: bool | None,
    pixel: bool | None,
) -> tuple[bool | None, str]:
    """与 ``merge_stop_visible`` 同一口径。"""
    return merge_stop_visible(mem, pixel)


def format_ui_stop_hint(action: HudActionSummary | None) -> str:
    """动作中旁证：stp=1/0 + act=@ROI[/local尺寸]。有效才返回非空。"""
    if action is None or not action.has_stop():
        return ""
    stp = "1" if action.stop_visible else "0"
    x, y = _TRAVEL_ROI_CLIENT_XY
    act = f"act=@{x},{y}"
    if action.stop_visible and action.rect_w is not None and action.rect_h is not None:
        w, h = int(round(action.rect_w)), int(round(action.rect_h))
        if w > 0 and h > 0:
            if w == h:
                act = f"{act}/{w}²"
            else:
                act = f"{act}/{w}×{h}"
    return f"stp={stp} {act}"


def _icon_evidence(icons: HudIconSummary | None) -> str:
    if icons is None or not icons.has_any():
        return "无icon"
    parts: list[str] = []
    if icons.menu_esc:
        parts.append(f"esc={icons.menu_esc.short()}")
    if icons.inventory:
        parts.append(f"inv={icons.inventory.short()}")
    if icons.left_bottom:
        parts.append(f"lb={icons.left_bottom.short()}")
    return " ".join(parts) if parts else "无icon"


def classify_fine_ui(
    ms: MemSceneInfo | None,
    icons: HudIconSummary | None = None,
) -> FineUiVerdict:
    """粗四位为主，icon 交叉验证清 HUD / 战斗。"""
    ev_icon = _icon_evidence(icons)

    if ms is None:
        return FineUiVerdict("unknown", FINE_LABELS["unknown"], "low", ev_icon)

    if ms.dialog_active is True:
        return FineUiVerdict("dialog", FINE_LABELS["dialog"], "high", f"dlg=1 {ev_icon}".strip())

    menu = ms.menu_open
    hub = ms.craft_hub_category
    from mm_assist.memory.readers.ui_stack import stack_inv

    inv = stack_inv(list(getattr(ms, "ui_stack_names", ()) or ()))

    if menu is True and hub == 0:
        return FineUiVerdict("map", FINE_LABELS["map"], "high", f"menu=1 hub=0 {ev_icon}".strip())

    if menu is True and hub is not None and hub in (1, 2, 3):
        return FineUiVerdict(
            "craft", FINE_LABELS["craft"], "high", f"menu=1 hub={hub} {ev_icon}".strip()
        )

    if inv is True and menu is not True:
        # inv=1 且非明确菜单层 → 物品（历史：inv 不单独抬主标签，细界面仍标出）
        return FineUiVerdict(
            "inventory", FINE_LABELS["inventory"], "mid", f"inv=1 {ev_icon}".strip()
        )

    if menu is True:
        hub_s = "-" if hub is None else str(hub)
        return FineUiVerdict(
            "menu_other",
            FINE_LABELS["menu_other"],
            "mid",
            f"menu=1 hub={hub_s} {ev_icon}".strip(),
        )

    # menu=0 / None：清 HUD 或战斗；靠 icon 交叉
    combat_ui = ms.combat_mode_ui is True or ms.in_combat is True
    icon_combat = bool(
        icons
        and (
            (icons.inventory and icons.inventory.is_combat)
            or (icons.menu_esc and icons.menu_esc.is_combat)
            or (icons.left_bottom and icons.left_bottom.is_combat)
        )
    )
    if combat_ui or icon_combat:
        return FineUiVerdict(
            "combat_hud",
            FINE_LABELS["combat_hud"],
            "high" if (combat_ui and icon_combat) else "mid",
            ev_icon,
        )

    if menu is False:
        if icons and icons.has_any():
            esc_ok = icons.menu_esc is not None and icons.menu_esc.show_ok
            inv_ok = icons.inventory is not None and icons.inventory.show_ok
            esc_bad = icons.menu_esc is not None and (
                icons.menu_esc.is_hide or icons.menu_esc.interactable is False
            )
            inv_bad = icons.inventory is not None and (
                icons.inventory.is_hide or icons.inventory.interactable is False
            )
            if esc_ok and inv_ok:
                return FineUiVerdict(
                    "clear_hud", FINE_LABELS["clear_hud"], "high", ev_icon
                )
            # 单侧 Hide/禁点常见于幽灵实例；双侧才算冲突
            if esc_bad and inv_bad:
                return FineUiVerdict(
                    "conflict",
                    FINE_LABELS["conflict"],
                    "mid",
                    f"menu=0但icon异常 {ev_icon}",
                )
            if esc_ok or inv_ok:
                return FineUiVerdict(
                    "clear_hud_weak",
                    FINE_LABELS["clear_hud_weak"],
                    "mid",
                    f"menu=0部分icon {ev_icon}",
                )
            if esc_bad or inv_bad:
                return FineUiVerdict(
                    "clear_hud_weak",
                    FINE_LABELS["clear_hud_weak"],
                    "mid",
                    f"menu=0 icon弱 {ev_icon}",
                )
            return FineUiVerdict(
                "clear_hud_weak", FINE_LABELS["clear_hud_weak"], "mid", ev_icon
            )
        return FineUiVerdict(
            "clear_hud_weak",
            FINE_LABELS["clear_hud_weak"],
            "low",
            "menu=0 无icon",
        )

    # menu=?
    return FineUiVerdict("unknown", FINE_LABELS["unknown"], "low", ev_icon)


def format_fine_ui(verdict: FineUiVerdict | None) -> str:
    if verdict is None:
        return "—"
    return verdict.format_line()


def verdict_to_dict(verdict: FineUiVerdict) -> dict[str, Any]:
    return {
        "kind": verdict.kind,
        "label": verdict.label,
        "confidence": verdict.confidence,
        "evidence": verdict.evidence,
        "line": verdict.format_line(),
    }
