"""内存快照：背包 + 工具耐久 + 设施队列。失败不降 OCR。点开始不限时；核对可限 3 秒。"""

from __future__ import annotations

import threading
import time
import logging
import sys
import traceback
from contextlib import ExitStack
from typing import Callable

from mm_assist.life_plan.types import LifeSnapshot, SkillGrowth
from mm_assist.memory.process import close_handle, find_game_pid, open_process, read_identity
from mm_assist.memory.evidence import ReadEvidence, bind_evidence
from mm_assist.memory.readers.consumable import read_consumable_counts
from mm_assist.memory.readers.equip_tools import read_equipped_tools
from mm_assist.memory.readers.facility_queue import read_facility_queues
from mm_assist.memory.readers.living_skill import read_living_skills
from mm_assist.memory.component_hints import prepare_hints

SNAP_VERIFY_S = 3.0
_SCAN_LOCK = threading.Lock()
heavy_read_lock = threading.Lock()
log = logging.getLogger(__name__)


def snapshot_in_progress() -> bool:
    return _SCAN_LOCK.locked()


class ItemsReadBusy(RuntimeError):
    """读取尚未开始，正等待共用读锁；不代表库存未知。"""


def refresh_items(*, budget_s: float = 0.25) -> dict[str, int] | None:
    """Paired current stock only; no full snapshot, heap fallback or worker."""
    from mm_assist.memory.il2cpp import ui_memory_read_slot
    from mm_assist.memory.readers.consumable import read_consumable_counts as read_items
    deadline = time.monotonic() + max(0.0, budget_s)
    remaining = lambda: max(0.0, deadline-time.monotonic())
    try:
        with ExitStack() as cleanup:
            for name, lock in (('snapshot', _SCAN_LOCK), ('heavy', heavy_read_lock)):
                if not lock.acquire(timeout=remaining()):
                    raise ItemsReadBusy(name)
                cleanup.callback(lock.release)
            with ui_memory_read_slot(timeout_s=remaining()) as acquired:
                if not acquired or not remaining():
                    raise ItemsReadBusy('memory')
                pid = find_game_pid()
                if not pid:
                    return None
                h = open_process(pid)
                if not h:
                    return None
                cleanup.callback(close_handle, h)
                identity = read_identity(h, pid)
                evidence = ReadEvidence(identity)
                with bind_evidence(evidence):
                    first = read_items(h, pid, allow_scan=False)
                if not first or not remaining():
                    return None
                with bind_evidence(evidence):
                    second = read_items(h, pid, allow_scan=False)
                if first != second or not remaining() or find_game_pid() != pid:
                    return None
                if read_identity(h, pid) != identity or not evidence.recheck(h) or not remaining():
                    return None
                return dict(second)
    except ItemsReadBusy:
        raise
    except Exception:
        return None


def failed_snapshot(error: str, *, taken_at: float | None = None, evidence: dict | None = None) -> LifeSnapshot:
    return LifeSnapshot(ok=False, error=error, taken_at=float(taken_at if taken_at is not None else time.time()),
                        evidence=evidence or {})


def _snapshot_body(ts: float, *, include_tools: bool = True, include_skills: bool = False) -> LifeSnapshot:
    pid = find_game_pid()
    if not pid:
        return failed_snapshot("no_game", taken_at=ts)
    h = None
    evidence = ReadEvidence({})
    def fail(reason):
        return failed_snapshot(reason, taken_at=ts, evidence=evidence.to_dict())
    try:
        h = open_process(pid)
        evidence.process = read_identity(h, pid)
        log.info("life_snapshot game_pid=%s handle=%s", pid, h)
        if not prepare_hints(pid):
            return fail("mem_warm_failed")
        def read(name, fn):
            started = time.monotonic()
            started_at = time.time()
            log.info("life_snapshot begin %s", name)
            result = None
            try:
                with bind_evidence(evidence):
                    result = fn(h, pid)
            finally:
                evidence.components[name] = {'started_at': started_at,
                    'elapsed_s': time.monotonic()-started, 'readable': result is not None}
            log.info("life_snapshot end %s elapsed=%.3fs readable=%s", name, time.monotonic() - started, result is not None)
            return result
        items = read("items", read_consumable_counts)
        tools = read("tools", read_equipped_tools) if include_tools else ()
        queues = read("facilities", read_facility_queues)
        if not items:
            return fail("consumable_unreadable")
        if queues is None:
            return fail("facility_unreadable")
        # 工具耐久读不到不挡必要物品 / 开 10h；十分钟扫仍可补 tools。
        err = ""
        if tools is None:
            tools = ()
            err = "tools_unreadable"
        skills: tuple[SkillGrowth, ...] = ()
        if include_skills:
            raw_skills = read("skills", read_living_skills)
            if not raw_skills:
                err = (err + ";" if err else "") + "skills_unreadable"
            else:
                skills = tuple(
                    SkillGrowth(name=str(s["name"]), level=int(s["level"]), exp=int(s["exp"]))
                    for s in raw_skills
                )
        if find_game_pid() != pid or read_identity(h, pid) != evidence.process:
            return fail('process_binding_changed')
        if not evidence.recheck(h):
            return fail(evidence.error)
        return LifeSnapshot(
            ok=True,
            items=items,
            tools=tuple(tools),
            facilities=queues,
            skills=skills,
            error=err,
            taken_at=ts,
            evidence=evidence.to_dict(),
        )
    except Exception as exc:
        from mm_assist.memory.client_build import BuildRejected
        from mm_assist.memory.access_diagnostic import ProcessReadDenied
        if isinstance(exc, ProcessReadDenied):
            evidence.components['access'] = exc.diagnostic
            return fail('process_read_denied' if exc.winerror == 5 else 'process_read_failed')
        return fail(exc.reason if isinstance(exc, BuildRejected) else f"mem_error:{exc}")
    finally:
        if h:
            close_handle(h)


def take_snapshot(
    *,
    now: float | None = None,
    budget_s: float = 0.0,
    include_tools: bool = True,
    include_skills: bool = False,
    on_complete: Callable[[LifeSnapshot], None] | None = None,
) -> LifeSnapshot:
    """Budget caps caller waiting; on_complete receives the same read's final result."""
    ts = float(now if now is not None else time.time())
    limit = float(budget_s)
    if not _SCAN_LOCK.acquire(blocking=False):
        return failed_snapshot("mem_busy", taken_at=ts)
    box: dict[str, LifeSnapshot] = {}

    def _run() -> None:
        try:
            try:
                with heavy_read_lock:
                    box["v"] = _snapshot_body(
                        ts, include_tools=include_tools, include_skills=include_skills
                    )
            except Exception as exc:
                box["v"] = failed_snapshot(f"mem_error:{exc}", taken_at=ts)
            if on_complete is not None:
                on_complete(box["v"])
        finally:
            _SCAN_LOCK.release()

    if limit <= 0:
        _run()
        return box["v"]

    t = threading.Thread(target=_run, daemon=True, name="life-snap")
    t.start()
    t.join(limit)
    got = box.get("v")
    if got is None:
        frame = sys._current_frames().get(t.ident)
        if frame is not None:
            log.warning("life_snapshot timeout after %.1fs; reader stack:\n%s", limit, "".join(traceback.format_stack(frame)))
        return failed_snapshot("mem_budget", taken_at=ts)
    return got


def refresh_tools() -> tuple | None:
    """只读装备工具；与整包及watch读取串行，不覆盖读不到的旧值。"""
    with _SCAN_LOCK, heavy_read_lock:
        pid = find_game_pid()
        if not pid:
            return None
        h = None
        try:
            h = open_process(pid)
            identity = read_identity(h, pid)
            evidence = ReadEvidence(identity)
            with bind_evidence(evidence):
                result = read_equipped_tools(h, pid)
            return result if (find_game_pid() == pid and read_identity(h, pid) == identity
                              and evidence.recheck(h)) else None
        except Exception:
            return None
        finally:
            if h:
                close_handle(h)
