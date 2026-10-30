"""一键修理：清主界面 → map_goto(杜巴頓/奈麗絲) → 对话链。

验收后成功=对话链完成（耐久只记录）。禁止点名牌旁路。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mm_assist.config import ROOT
from mm_assist.life_plan.types import ToolDurability
from mm_assist.perception.ocr import OcrLine

log = logging.getLogger(__name__)

_OP_CANCEL = threading.Event()

REPAIR_REGION = "杜巴頓"
REPAIR_NPC = "奈麗絲"
DEFAULT_REPAIR_VIA = "one_click"
DEFAULT_REPAIR_ENTRY = "run_repair_round"
FORBID_FIRST_REPAIR: tuple[str, ...] = ("repair_dialog", "npc_nameplate")
REPAIR_AFTER_ARRIVE = "skip_then_repair_entry"
REPAIR_ENTRY_ICON_MISS = "repair_icon_miss"
REPAIR_ENTRY_DEBT = "到站后缺铁砧修理入口；禁改点名牌"
REPAIR_NAMEPLATE_MISS = "nameplate_miss:need_approach_or_map"
STATUSBAR_EQUIP_CACHE_MAX_AGE_S = 8.0

DEFAULT_REPAIR_NEEDLES: tuple[str, ...] = (
    "需要修理",
    "耐久度危險",
    "耐久度危险",
    "工具已損",
    "工具已坏",
    "請修理",
    "请修理",
    "已損壞",
    "已损坏",
    "無法採集",
    "无法采集",
    "無法探集",
    "無法使用",
    "无法使用",
    "耐久不足",
    "損壞",
    "损坏",
    "破損",
    "破损",
    "維修",
    "维修",
    "修理",
)

_STATUSBAR_EQUIP_CACHE: Any = None
_STATUSBAR_EQUIP_CACHE_TS = 0.0
REPAIR_FLOW_ACCEPTED_REL = Path("knowledge/ui/contracts/repair_flow_accepted.json")


def repair_flow_accepted(path: Path | None = None) -> bool:
    """人工印章。仅 accepted 严格为 true 才算验收后；缺文件/坏 JSON 一律未验收。"""
    p = path or (ROOT / REPAIR_FLOW_ACCEPTED_REL)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(raw, dict):
        return False
    return raw.get("accepted") is True


def resolve_require_durability_rose(explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return not repair_flow_accepted()


@dataclass
class RepairProbeResult:
    needed: bool
    source: str
    matched: str = ""
    note: str = ""


def request_op_cancel() -> None:
    _OP_CANCEL.set()
    try:
        from mm_assist.safety.input import request_stop

        request_stop()
    except Exception:
        pass


def clear_op_cancel() -> None:
    _OP_CANCEL.clear()


def op_cancelled() -> bool:
    return _OP_CANCEL.is_set()


MIN_OP_STEP_S = 1.0


def pace_op_step(t0: float | None = None, *, min_s: float | None = None) -> None:
    need = float(MIN_OP_STEP_S if min_s is None else min_s)
    if need <= 0:
        return
    if t0 is None:
        end = time.monotonic() + need
    else:
        end = float(t0) + need
    while True:
        if op_cancelled():
            return
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(0.2, left))


def set_statusbar_equip_cache(snap: Any) -> None:
    global _STATUSBAR_EQUIP_CACHE, _STATUSBAR_EQUIP_CACHE_TS
    if snap is None:
        return
    tools = list(getattr(snap, "tools", None) or [])
    if not tools:
        return
    _STATUSBAR_EQUIP_CACHE = snap
    _STATUSBAR_EQUIP_CACHE_TS = time.monotonic()


def get_statusbar_equip_cache(
    *, max_age_s: float = STATUSBAR_EQUIP_CACHE_MAX_AGE_S
) -> Any | None:
    if _STATUSBAR_EQUIP_CACHE is None:
        return None
    age = time.monotonic() - float(_STATUSBAR_EQUIP_CACHE_TS or 0.0)
    if age > float(max_age_s):
        return None
    return _STATUSBAR_EQUIP_CACHE


def clear_statusbar_equip_cache() -> None:
    global _STATUSBAR_EQUIP_CACHE, _STATUSBAR_EQUIP_CACHE_TS
    _STATUSBAR_EQUIP_CACHE = None
    _STATUSBAR_EQUIP_CACHE_TS = 0.0


def _tools_from_snap(snap: Any) -> list[Any]:
    return list(getattr(snap, "tools", None) or [])


def equip_durability_map(snap: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    if snap is None:
        return out
    for i, it in enumerate(_tools_from_snap(snap)):
        if isinstance(it, ToolDurability):
            name = it.name or "?"
            cur = it.durability
        else:
            name = (
                getattr(it, "display_name", None)
                or getattr(it, "name", None)
                or "?"
            )
            cur = getattr(it, "durability", None)
        if cur is None:
            continue
        out[f"tool:{name}:{i}"] = int(cur)
    return out


def durability_rose(before: dict[str, int], after: dict[str, int]) -> bool:
    for key, old in before.items():
        new = after.get(key)
        if new is not None and int(new) > int(old):
            return True
    return False


def _read_equip_snapshot(*, allow_scan: bool = True) -> Any | None:
    _ = allow_scan
    try:
        from mm_assist.life_plan.snapshot import take_snapshot

        snap = take_snapshot()
        if not snap.ok:
            return None
        return snap
    except Exception as exc:
        log.debug("equip snapshot failed: %s", exc)
        return None


def any_tool_durability_below(
    threshold: float = 1.0,
    *,
    name_substrings: tuple[str, ...] | None = None,
    prefer_statusbar_cache: bool = True,
    allow_mem_scan: bool = True,
) -> RepairProbeResult:
    snap = None
    source = "tool_dur"
    if prefer_statusbar_cache:
        snap = get_statusbar_equip_cache()
        if snap is not None:
            source = "statusbar_equip_cache"
    if snap is None and allow_mem_scan:
        snap = _read_equip_snapshot(allow_scan=True)
        source = "tool_dur"
    if snap is None:
        note = (
            "statusbar_equip_cache_miss"
            if prefer_statusbar_cache and not allow_mem_scan
            else "equip_unavailable"
        )
        return RepairProbeResult(False, "none", note=note)
    needles = tuple(s for s in (name_substrings or ()) if s)
    for it in _tools_from_snap(snap):
        if isinstance(it, ToolDurability):
            name, cur = it.name, it.durability
        else:
            name = str(getattr(it, "display_name", None) or getattr(it, "name", "") or "?")
            cur = getattr(it, "durability", None)
        if cur is None:
            continue
        if needles and not any(s in name for s in needles):
            continue
        if float(cur) < float(threshold):
            return RepairProbeResult(
                True,
                source,
                matched=f"{name}:{cur}",
                note=f"tool_durability<{threshold:g}",
            )
    return RepairProbeResult(
        False,
        source,
        note="tools_ok" if not needles else "tools_ok_filtered",
    )


def text_blob(lines: list[OcrLine]) -> str:
    return "".join(str(getattr(ln, "text", "") or "") for ln in lines)


def detect_repair_needed(
    lines: list[OcrLine],
    needles: tuple[str, ...] | list[str] | None = None,
) -> str | None:
    blob = text_blob(lines)
    if not blob:
        return None
    pool = [n for n in (needles or DEFAULT_REPAIR_NEEDLES) if n]
    pool.sort(key=len, reverse=True)
    for n in pool:
        if n in blob:
            return n
    return None


def plan_repair_dialog_entry(
    *,
    arrived: bool = True,
    dialog_open: bool | None = None,
    has_skip: bool = False,
    has_repair_icon: bool = False,
    has_repair_ocr: bool = False,
    has_equipped_icon: bool = False,
    skip_attempted: bool = False,
    nameplate_found: bool = False,
) -> dict[str, Any]:
    """已到站后的对话入口：先跳過對話，再铁砧。永不点名牌。"""
    _ = nameplate_found
    out: dict[str, Any] = {
        "allow_click_nameplate": False,
        "forbid_nameplate": True,
        "stage": "entry",
        "fail_tag": "",
        "debt": "",
        "error": "",
    }
    if has_equipped_icon:
        out.update(ok=True, op="click_equipped", reason="equipped_ui")
        return out
    if not skip_attempted and has_skip:
        out.update(ok=True, op="skip_then_find_entry", reason="skip_to_reveal_menu")
        return out
    if has_repair_icon or has_repair_ocr:
        out.update(
            ok=True,
            op="click_repair_entry",
            reason="repair_icon" if has_repair_icon else "repair_ocr",
        )
        return out
    if not skip_attempted and arrived and dialog_open is True:
        out.update(ok=True, op="skip_then_find_entry", reason="dialog_open_skip_first")
        return out
    out.update(
        ok=False,
        op="reject",
        reason=REPAIR_ENTRY_ICON_MISS,
        fail_tag=REPAIR_ENTRY_ICON_MISS,
        debt=REPAIR_ENTRY_DEBT,
        error="对话入口「修理」图标未找到",
    )
    return out


def plan_durability_repair(
    *,
    needed: bool,
    matched: str = "",
    note: str = "",
) -> dict[str, Any]:
    if not needed:
        return {
            "ok": True,
            "needed": False,
            "op": "skip",
            "via": None,
            "entry": None,
            "reason": "tools_ok",
            "forbid_first": (),
            "matched": matched,
            "note": note,
        }
    return {
        "ok": True,
        "needed": True,
        "op": "one_click",
        "via": DEFAULT_REPAIR_VIA,
        "entry": DEFAULT_REPAIR_ENTRY,
        "reason": "durability_below_1",
        "forbid_first": FORBID_FIRST_REPAIR,
        "after_arrive": REPAIR_AFTER_ARRIVE,
        "matched": matched,
        "note": note,
    }


NavFn = Callable[..., tuple[Any, dict]]
DialogFn = Callable[..., tuple[Any, dict]]
DurFn = Callable[[], dict[str, int]]


def _default_nav(cfg, win, **_k):
    from mm_assist.nav.map_atlas import map_goto

    region = str(_k.get("region") or REPAIR_REGION)
    npc = str(_k.get("npc") or REPAIR_NPC)
    out = map_goto(
        cfg,
        win,
        region=region,
        npc=npc,
        idle_abort_s=0.0,
        wait_arrive=bool(_k.get("wait_arrive", True)),
    )
    ok = bool(out.get("ok"))
    return win, {
        "ok": ok,
        "stage": str(out.get("stage") or ("arrived" if ok else "goto")),
        "error": str(out.get("error") or out.get("detail") or ""),
        "goto": out,
    }


def _default_dialog(cfg, win, **_k):
    from mm_assist.life_execute.repair_dialog import run_repair_dialog_chain

    return run_repair_dialog_chain(cfg, win, **_k)


def run_repair_round(
    cfg: Any,
    win: Any,
    *,
    region: str = REPAIR_REGION,
    npc: str = REPAIR_NPC,
    near_only: bool = False,
    skip_loc_probe: bool = True,
    on_progress: Any | None = None,
    nav_fn: NavFn | None = None,
    dialog_fn: DialogFn | None = None,
    dur_before: dict[str, int] | None = None,
    dur_after_fn: DurFn | None = None,
    require_durability_rose: bool | None = None,
    phase: str = "all",
) -> tuple[Any, dict]:
    """去修一轮。未验收：对话链 ok 且内存耐久回升。验收后：对话链 ok 即可。"""

    def _prog(msg: str) -> None:
        log.info("repair progress: %s", msg)
        if on_progress is not None:
            try:
                on_progress(msg)
            except Exception:
                pass

    result: dict[str, Any] = {
        "ok": False,
        "stage": "start",
        "region": region,
        "npc": npc,
        "near_only": near_only,
        "skip_loc_probe": skip_loc_probe,
        "durability_rose": False,
        "require_durability_rose": resolve_require_durability_rose(
            require_durability_rose
        ),
    }
    if op_cancelled():
        result["error"] = "已停止操作"
        result["stage"] = "cancel"
        return win, result

    before = dur_before
    if before is None:
        before = equip_durability_map(_read_equip_snapshot() or get_statusbar_equip_cache())
    result["dur_before"] = dict(before)

    _prog("① 清界面…")
    result["stage"] = "menu"

    goto_fn = nav_fn or _default_nav
    dialog = dialog_fn or _default_dialog
    last_nav = {"ok": False}
    step = str(phase or "all").strip() or "all"
    wait_arrive = step == "all"
    already_dlg = False
    try:
        from mm_assist.life_execute.repair_dialog import _peek_dialog_active

        already_dlg = _peek_dialog_active() is True
    except Exception:
        already_dlg = False
    if step != "dialog" and already_dlg:
        _prog("已在对话，跳过找城")
        last_nav = {"ok": True, "stage": "already_dialog"}
        result["goto"] = last_nav
        if step == "nav":
            result["ok"] = True
            result["stage"] = "at_dialog"
            return win, result
    elif step != "dialog":
        if op_cancelled():
            result["error"] = "已停止操作"
            result["stage"] = "cancel"
            return win, result
        _prog(f"③ 开图→{region}→{npc}")
        win, last_nav = goto_fn(
            cfg, win, region=region, npc=npc, wait_arrive=wait_arrive
        )
        result["goto"] = last_nav
        if str(last_nav.get("stage") or "") == "nav_not_ported":
            result["stage"] = "nav_not_ported"
            result["error"] = str(last_nav.get("error") or "nav_not_ported")
            return win, result
        if not last_nav.get("ok"):
            result["stage"] = str(last_nav.get("stage") or "region")
            result["error"] = str(last_nav.get("error") or "map_goto_fail")
            result["far_fail"] = True
            return win, result
        if step == "nav":
            result["ok"] = True
            result["stage"] = str(last_nav.get("stage") or "walking")
            return win, result
    elif step == "dialog":
        last_nav = {"ok": True, "stage": "dialog_only"}
        result["goto"] = last_nav

    if op_cancelled():
        result["error"] = "已停止操作"
        result["stage"] = "cancel"
        return win, result
    _prog("④ 对话修理链…")
    win, dialog_out = dialog(cfg, win)
    result["dialog"] = dialog_out
    if not dialog_out.get("ok"):
        result["stage"] = str(dialog_out.get("stage") or "dialog")
        result["error"] = str(dialog_out.get("error") or "dialog_fail")
        return win, result

    after_fn = dur_after_fn or (lambda: equip_durability_map(_read_equip_snapshot() or {}))
    after = after_fn()
    result["dur_after"] = dict(after)
    rose = durability_rose(before, after)
    result["durability_rose"] = rose
    if dialog_out.get("dlg_left") is False:
        result["stage"] = "dialog"
        result["error"] = "对话链完成但未离开对话层"
        return win, result
    need_rose = bool(result["require_durability_rose"])
    if need_rose and not rose:
        result["stage"] = "durability"
        result["error"] = "对话链完成但内存耐久未回升"
        return win, result
    result["ok"] = True
    result["stage"] = "done"
    return win, result


def one_click_repair(
    cfg: Any,
    win: Any,
    *,
    force: bool = False,
    on_progress: Any | None = None,
    near_only: bool = False,
    skip_loc_probe: bool = True,
    nav_fn: NavFn | None = None,
    dialog_fn: DialogFn | None = None,
    dur_before: dict[str, int] | None = None,
    dur_after_fn: DurFn | None = None,
    require_durability_rose: bool | None = None,
    phase: str = "all",
) -> tuple[Any, dict]:
    if force:
        needed, matched, note = True, "", "force"
        probe_d = None
    else:
        probe = any_tool_durability_below(1.0)
        needed, matched, note = bool(probe.needed), probe.matched, probe.note
        probe_d = {
            "needed": probe.needed,
            "source": probe.source,
            "matched": probe.matched,
            "note": probe.note,
        }
    plan = plan_durability_repair(needed=needed, matched=matched, note=note)
    if plan["op"] == "skip":
        return win, {"ok": True, "stage": "skip", "via": None, "plan": plan, "probe": probe_d}
    win, result = run_repair_round(
        cfg,
        win,
        near_only=near_only,
        skip_loc_probe=skip_loc_probe,
        on_progress=on_progress,
        nav_fn=nav_fn,
        dialog_fn=dialog_fn,
        dur_before=dur_before,
        dur_after_fn=dur_after_fn,
        require_durability_rose=require_durability_rose,
        phase=phase,
    )
    result["via"] = DEFAULT_REPAIR_VIA
    result["plan"] = plan
    if probe_d is not None:
        result["probe"] = probe_d
    return win, result
