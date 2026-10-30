"""状态栏落盘的权威界面态（`logs/state_watch_ui.json`）。

为什么单独一份：`state_watch.heartbeat.json` 只有 pid/时间，是「state-watch 活着吗」的门控；
本文件是「现在在哪个界面」的**内存权威判据**，供 gather/feed/ui_switch/ui_escape 只读消费，
避免继续用文字 OCR 主判界面族（`10-live-test`：实机路径 OCR 只作旁证）。

粗族由内存决定，**不可被 OCR 推翻**；`menu` 层内的细分（人物 / 生活技能大格 / 設施）
内存分不出，留给调用方用 OCR 细化——但只能在 `family == FAMILY_MENU` 时细化。
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mm_assist.memory.readers.scene import craft_hub_tab_label

FEED_NAME = "state_watch_ui.json"

# 消费侧默认新鲜度：state-watch lean tick 约 0.5–1s 一拍，2.5s 内视为可信
DEFAULT_MAX_AGE_S = 2.5

FAMILY_MAIN_HUD = "main_hud"
FAMILY_COMBAT = "combat"
FAMILY_MENU = "menu"
FAMILY_CRAFT_HUB = "craft_hub"
FAMILY_PRF = "prf"
FAMILY_MAP = "map"
FAMILY_DIALOG = "dialog"
FAMILY_INVENTORY = "inventory"
FAMILY_REPAIR = "repair"
FAMILY_UNKNOWN = "unknown"

# 内存能独立坐实的硬面板：这些族禁止被 OCR 推翻
HARD_FAMILIES = frozenset(
    {
        FAMILY_MAIN_HUD,
        FAMILY_COMBAT,
        FAMILY_CRAFT_HUB,
        FAMILY_MAP,
        FAMILY_DIALOG,
        FAMILY_INVENTORY,
        FAMILY_REPAIR,
    }
)


def _repo_root() -> Path:
    from mm_assist.config import ROOT

    return ROOT


def state_ui_path(root: Path | None = None) -> Path:
    base = root or _repo_root()
    return base / "logs" / FEED_NAME


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def family_from_stack_names(names: list[str] | None, *, ready: bool) -> str | None:
    """栈就绪时的业务族。未就绪 / 无法分类 → None。"""
    from mm_assist.memory.readers.ui_stack import stack_kind

    kind = stack_kind(names, ready=ready)
    if kind is None:
        return None
    if kind == "dialog":
        return FAMILY_DIALOG
    if kind == "inventory":
        return FAMILY_INVENTORY
    if kind == "repair":
        return FAMILY_REPAIR
    if kind == "craft_hub":
        return FAMILY_CRAFT_HUB
    if kind == "map":
        return FAMILY_MAP
    if kind == "profile":
        return FAMILY_PRF
    if kind in ("side_nav", "life_skill"):
        return FAMILY_MENU
    if kind == "overlay":
        return FAMILY_UNKNOWN
    if kind == "main":
        return FAMILY_MAIN_HUD
    return FAMILY_UNKNOWN


def family_from_scene(
    ui: str,
    *,
    craft_hub_category: int | None = None,
    profile_open: bool | None = None,
    ui_stack_names: list[str] | tuple[str, ...] | None = None,
    ui_stack_ready: bool | None = None,
) -> str:
    """`scene_status` 的 ui 标签 → 业务族。栈就绪时栈类名优先。

    `menu` + hub∈{1,2,3} 才是製作总览；hub=0 是大地图（`_from_mem_scene` 已分流）。
    人物页（prf）只有 `profile_open` 为真时才敢报，否则留在 `menu` 让 OCR 细分。
    """
    if ui_stack_ready:
        fam = family_from_stack_names(
            list(ui_stack_names or ()), ready=True
        )
        if fam is not None:
            if fam == FAMILY_MAIN_HUD and str(ui or "").strip() == "combat":
                return FAMILY_COMBAT
            return fam
    u = str(ui or "").strip()
    if u == "dialog":
        return FAMILY_DIALOG
    if u == "repair":
        return FAMILY_REPAIR
    if u == "inventory":
        return FAMILY_INVENTORY
    if u.startswith("map_"):
        return FAMILY_MAP
    if u == "combat":
        return FAMILY_COMBAT
    if u == "hud":
        return FAMILY_MAIN_HUD
    if u == "menu":
        try:
            hub = int(craft_hub_category) if craft_hub_category is not None else None
        except (TypeError, ValueError):
            hub = None
        if hub in (1, 2, 3):
            return FAMILY_CRAFT_HUB
        if profile_open is True:
            return FAMILY_PRF
        return FAMILY_MENU
    return FAMILY_UNKNOWN


_PARAM_BIT_RE = re.compile(r"(?:^|\s)([a-z]+)=([01?])")


def parse_ui_param_bit(params: str, key: str) -> int | None:
    """从 ``ui_params`` 取 ``cbt=0`` / ``stp=1`` 等；``?`` 或缺省 → None。"""
    k = str(key or "").strip()
    if not k:
        return None
    for m in _PARAM_BIT_RE.finditer(str(params or "")):
        if m.group(1) != k:
            continue
        bit = m.group(2)
        if bit == "?":
            return None
        return int(bit)
    return None


def build_state_ui_payload(
    snap: Any,
    ms: Any = None,
    *,
    pid: int | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """由 `SceneSnapshot`（+ 可选 `MemSceneInfo`）拼出落盘 payload。零额外堆扫。"""
    ui = str(getattr(snap, "ui", "") or "")
    hub = getattr(ms, "craft_hub_category", None)
    prf = getattr(ms, "profile_open", None)
    params = str(getattr(snap, "ui_params", "") or "")
    cbt = parse_ui_param_bit(params, "cbt")
    stp = parse_ui_param_bit(params, "stp")
    craft_tab = craft_hub_tab_label(hub)
    payload: dict[str, Any] = {
        "pid": int(pid if pid is not None else os.getpid()),
        "ts_unix": float(now if now is not None else time.time()),
        "family": family_from_scene(
            ui,
            craft_hub_category=hub,
            profile_open=prf,
            ui_stack_names=list(getattr(ms, "ui_stack_names", ()) or ()),
            ui_stack_ready=bool(getattr(ms, "ui_stack_ready", False)),
        ),
        "ui": ui,
        "ui_detail": str(getattr(snap, "ui_detail", "") or ""),
        "ui_source": str(getattr(snap, "ui_source", "") or ""),
        "ui_params": params,
        "ui_stack": str(getattr(snap, "ui_stack", "") or ""),
        "ui_stack_names": list(getattr(ms, "ui_stack_names", ()) or ()),
        "ui_stack_ready": bool(getattr(ms, "ui_stack_ready", False)),
        "ui_stack_empty_ok": bool(getattr(ms, "ui_stack_empty_ok", False)),
        "ui_relation": str(getattr(snap, "ui_relation", "") or ""),
        "geo": str(getattr(snap, "geo", "") or ""),
        "map_type": getattr(ms, "map_type", None),
        "mem_status": str(getattr(snap, "mem_status", "") or ""),
        "memory_diagnostic": dict(getattr(snap, 'memory_diagnostic', {}) or {}),
        "stop_diagnostic": dict(getattr(snap, 'stop_diagnostic', {}) or {}),
        "memory_process": dict(getattr(ms, 'memory_process', {}) or {}),
        "menu_open": getattr(ms, "menu_open", None),
        "dialog_active": getattr(ms, "dialog_active", None),
        "map_open": getattr(ms, "map_open", None),
        "profile_open": prf,
        "profile_category": getattr(ms, "profile_category", None),
        "profile_identity": getattr(ms, "profile_identity", None),
        "craft_current_category": getattr(ms, "craft_current_category", None),
        "craft_identity": getattr(ms, "craft_identity", None),
        "map_identity": getattr(ms, "map_identity", None),
        "map_view": getattr(ms, "map_view", None),
        "craft_hub_category": hub,
        "cbt": cbt,
        "stp": stp,
        "craft_tab": craft_tab,
    }
    err = str(getattr(ms, "ui_stack_error", "") or "")
    if err:
        payload["ui_stack_error"] = err
    try:
        stack_ts = float(getattr(ms, "ui_stack_ts", 0.0) or 0.0)
    except (TypeError, ValueError):
        stack_ts = 0.0
    if stack_ts > 0.0:
        payload["ui_stack_ts"] = stack_ts
    return payload


def write_state_ui(
    snap: Any,
    ms: Any = None,
    *,
    pid: int | None = None,
    root: Path | None = None,
    now: float | None = None,
) -> Path:
    """state-watch 每拍调用：原子写界面态。异常不得掀翻状态栏主循环。"""
    p = state_ui_path(root)
    payload = build_state_ui_payload(snap, ms, pid=pid, now=now)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(p)
    return p


def clear_state_ui(root: Path | None = None) -> None:
    p = state_ui_path(root)
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        p.with_name("ui_inspect.json").unlink(missing_ok=True)
    except OSError:
        pass


@dataclass(frozen=True)
class StateUiView:
    family: str
    ui: str
    detail: str
    params: str
    relation: str
    geo: str
    age_s: float
    fresh: bool
    pid: int
    pid_alive: bool
    menu_open: bool | None
    dialog_active: bool | None
    map_open: bool | None
    profile_open: bool | None
    craft_hub_category: int | None
    cbt: int | None
    stp: int | None
    craft_tab: str | None
    raw: dict[str, Any]

    @property
    def usable(self) -> bool:
        """可当权威判据：新鲜 + 写者还活着 + 族不是 unknown。"""
        return self.fresh and self.pid_alive and self.family != FAMILY_UNKNOWN

    @property
    def hard(self) -> bool:
        """内存独立坐实的硬族；OCR 不得推翻。"""
        return self.usable and self.family in HARD_FAMILIES


def read_state_ui(
    *,
    max_age_s: float = DEFAULT_MAX_AGE_S,
    root: Path | None = None,
    now: float | None = None,
) -> StateUiView | None:
    """只读界面态。缺文件 / 坏 JSON → None（调用方回退 OCR 并标 `ocr_fallback`）。"""
    p = state_ui_path(root)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        ts = float(raw.get("ts_unix") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    age = max(0.0, float(now if now is not None else time.time()) - ts)
    try:
        pid = int(raw.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    params = str(raw.get("ui_params") or "")
    cbt_raw = raw.get("cbt")
    stp_raw = raw.get("stp")
    try:
        cbt = int(cbt_raw) if cbt_raw is not None else parse_ui_param_bit(params, "cbt")
    except (TypeError, ValueError):
        cbt = parse_ui_param_bit(params, "cbt")
    try:
        stp = int(stp_raw) if stp_raw is not None else parse_ui_param_bit(params, "stp")
    except (TypeError, ValueError):
        stp = parse_ui_param_bit(params, "stp")
    craft_tab = raw.get("craft_tab")
    if craft_tab is not None:
        craft_tab = str(craft_tab) or None
    elif raw.get("craft_hub_category") is not None:
        craft_tab = craft_hub_tab_label(raw.get("craft_hub_category"))
    return StateUiView(
        family=str(raw.get("family") or FAMILY_UNKNOWN),
        ui=str(raw.get("ui") or ""),
        detail=str(raw.get("ui_detail") or ""),
        params=str(raw.get("ui_params") or ""),
        relation=str(raw.get("ui_relation") or ""),
        geo=str(raw.get("geo") or ""),
        age_s=age,
        fresh=ts > 0.0 and age <= float(max_age_s),
        pid=pid,
        pid_alive=_pid_alive(pid),
        menu_open=raw.get("menu_open"),
        dialog_active=raw.get("dialog_active"),
        map_open=raw.get("map_open"),
        profile_open=raw.get("profile_open"),
        craft_hub_category=raw.get("craft_hub_category"),
        cbt=cbt,
        stp=stp,
        craft_tab=craft_tab,
        raw=raw,
    )


def authoritative_family(
    *,
    max_age_s: float = DEFAULT_MAX_AGE_S,
    root: Path | None = None,
    now: float | None = None,
    view: StateUiView | None = None,
) -> tuple[str | None, str]:
    """返回 (族, 来源)。不可用时族为 None，来源说明原因，便于 events 归因。"""
    v = view if view is not None else read_state_ui(
        max_age_s=max_age_s, root=root, now=now
    )
    if v is None:
        return None, "statusbar_missing"
    if not v.pid_alive:
        return None, "statusbar_pid_dead"
    if not v.fresh:
        return None, f"statusbar_stale_{v.age_s:.1f}s"
    if v.family == FAMILY_UNKNOWN:
        return None, "statusbar_unknown"
    return v.family, "statusbar_mem"


def _frame_looks_main_hud(v: StateUiView) -> bool:
    """干净主界面：栈已就绪且无 HudUIStepState，且非动作中。"""
    raw = v.raw if isinstance(v.raw, dict) else {}
    if not bool(raw.get("ui_stack_ready")):
        return False
    names = [str(x) for x in (raw.get("ui_stack_names") or [])]
    if "HudUIStepState" in names:
        return False
    if v.stp not in (0, None) and v.stp != 0:
        return False
    if v.stp == 1:
        return False
    if v.map_open is True:
        return False
    try:
        hub = int(v.craft_hub_category) if v.craft_hub_category is not None else None
    except (TypeError, ValueError):
        hub = None
    if hub in (1, 2, 3):
        return False
    return True


def confirm_main_hud_frames(
    *,
    gap_s: float = 0.6,
    max_age_s: float = DEFAULT_MAX_AGE_S,
    root: Path | None = None,
    sleep_fn: Any = None,
    read_fn: Any = None,
) -> dict[str, Any]:
    """脱困成功判据：状态栏连续 2 帧干净主界面（无 HudUIStepState、stp≠1，间隔 ≥ gap_s）。

    OCR 不单独抬成功。状态栏不可用时 ``available=False``，调用方回退旧 menu_open 路径。
    """
    reader = read_fn or (lambda: read_state_ui(max_age_s=max_age_s, root=root))
    sleeper = sleep_fn or time.sleep
    out: dict[str, Any] = {
        "ok": False,
        "available": False,
        "why": "",
        "gap_s": float(gap_s),
        "frames": [],
    }
    v1 = reader()
    if v1 is None or not v1.usable:
        out["why"] = "frame1_unusable"
        return out
    out["available"] = True
    out["frames"].append(
        {
            "family": v1.family,
            "age_s": v1.age_s,
            "menu_open": v1.menu_open,
            "cbt": v1.cbt,
            "stp": v1.stp,
        }
    )
    if not _frame_looks_main_hud(v1):
        out["why"] = f"frame1_not_hud:{v1.family}"
        if v1.stp == 1:
            out["why"] = "frame1_stp_not_0"
        elif v1.menu_open is True:
            out["why"] = "frame1_menu_open"
        return out
    sleeper(max(0.0, float(gap_s)))
    v2 = reader()
    if v2 is None or not v2.usable:
        out["why"] = "frame2_unusable"
        return out
    out["frames"].append(
        {
            "family": v2.family,
            "age_s": v2.age_s,
            "menu_open": v2.menu_open,
            "cbt": v2.cbt,
            "stp": v2.stp,
        }
    )
    if not _frame_looks_main_hud(v2):
        out["why"] = f"frame2_not_hud:{v2.family}"
        if v2.stp == 1:
            out["why"] = "frame2_stp_not_0"
        elif v2.menu_open is True:
            out["why"] = "frame2_menu_open"
        return out
    out["ok"] = True
    out["why"] = "two_frame_main_hud"
    return out
