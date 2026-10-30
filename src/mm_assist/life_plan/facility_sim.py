"""加工进度：10 分钟同步堆扫一次；期间用 endAt 墙钟模拟。暂停冻结计时。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace

from mm_assist.life_plan.types import (
    CLAIM_DONE_MIN,
    FACILITY_ORDER,
    FACILITY_SCAN_INTERVAL_S,
    FacilityId,
    LifeSnapshot,
    QueueSlot,
)
from mm_assist.memory.readers.facility_queue import progress_from_ticks

_LOCK = threading.Lock()


@dataclass
class FacilitySimState:
    ok: bool = False
    error: str = ""
    last_scan_at: float = 0.0
    accrued_s: float = 0.0
    tick_at: float | None = None
    did_scan: bool = False
    slots: dict[FacilityId, tuple[QueueSlot, ...]] = field(
        default_factory=lambda: {
            "metal": (),
            "wood": (),
            "cloth": (),
            "medicine": (),
        }
    )


_STATE = FacilitySimState()
_READ = None


def reset_facility_sim_for_tests() -> None:
    global _STATE, _READ
    with _LOCK:
        _STATE = FacilitySimState()
        _READ = None


def seed_facility_slots(
    slots: dict[FacilityId, tuple[QueueSlot, ...]],
    *,
    now: float,
    ok: bool = True,
) -> None:
    with _LOCK:
        st = _STATE
        st.ok = bool(ok)
        st.error = "" if ok else "seed_failed"
        st.last_scan_at = float(now)
        st.accrued_s = 0.0
        st.tick_at = float(now)
        st.slots = {
            "metal": tuple(slots.get("metal") or ()),
            "wood": tuple(slots.get("wood") or ()),
            "cloth": tuple(slots.get("cloth") or ()),
            "medicine": tuple(slots.get("medicine") or ()),
        }


def set_facility_read_for_tests(fn) -> None:
    global _READ
    with _LOCK:
        _READ = fn


def forget_claimed_slots(facility: FacilityId) -> dict[FacilityId, tuple[QueueSlot, ...]]:
    """领取流程完成后移除本地已完成槽，未完成槽保留；下次内存扫描校正。"""
    with _LOCK:
        _STATE.slots[facility] = tuple(s for s in _STATE.slots.get(facility, ()) if not s.done)
        return dict(_STATE.slots)


def facility_sim_ok() -> bool:
    with _LOCK:
        return bool(_STATE.ok)


def peek_facility_sim() -> FacilitySimState:
    """只读当前模拟态；不扫描、不推进计时。"""
    with _LOCK:
        st = _STATE
        return FacilitySimState(
            ok=st.ok,
            error=st.error,
            last_scan_at=st.last_scan_at,
            accrued_s=st.accrued_s,
            tick_at=st.tick_at,
            did_scan=False,
            slots={
                "metal": tuple(st.slots.get("metal") or ()),
                "wood": tuple(st.slots.get("wood") or ()),
                "cloth": tuple(st.slots.get("cloth") or ()),
                "medicine": tuple(st.slots.get("medicine") or ()),
            },
        )


def simulate_slot(slot: QueueSlot, *, now_ms: int) -> QueueSlot:
    if int(slot.end_ms) <= 0:
        return slot
    start = int(slot.start_ms)
    end = int(slot.end_ms)
    req = 0
    ratio, remain = progress_from_ticks(
        start, end, now=int(now_ms), required_ticks=req, ticks_per_second=1000,
    )
    done = bool(int(now_ms) >= end)
    return QueueSlot(
        recipe=slot.recipe,
        done=done,
        progress_ratio=round(ratio, 4),
        remain_s=round(remain, 1),
        sheet_id=int(slot.sheet_id),
        start_ms=start,
        end_ms=end,
    )


def simulate_facilities(
    slots: dict[FacilityId, tuple[QueueSlot, ...]], *, now_s: float
) -> dict[FacilityId, tuple[QueueSlot, ...]]:
    now_ms = int(float(now_s) * 1000)
    out: dict[FacilityId, tuple[QueueSlot, ...]] = {
        "metal": (),
        "wood": (),
        "cloth": (),
        "medicine": (),
    }
    for fac, rows in slots.items():
        out[fac] = tuple(simulate_slot(s, now_ms=now_ms) for s in rows)
    return out


def done_count(slots: tuple[QueueSlot, ...]) -> int:
    return sum(1 for s in slots if s.done)


def should_claim_medicine(slots: dict[FacilityId, tuple[QueueSlot, ...]]) -> bool:
    return done_count(slots.get("medicine", ())) >= CLAIM_DONE_MIN


def should_claim_now(slots: dict[FacilityId, tuple[QueueSlot, ...]]) -> bool:
    return any(done_count(slots.get(fac, ())) >= CLAIM_DONE_MIN for fac in FACILITY_ORDER)


def _do_scan() -> tuple[bool, str, dict[FacilityId, tuple[QueueSlot, ...]]]:
    reader = _READ
    if reader is not None:
        raw = reader()
        if not raw:
            return False, "facility_scan_empty", {}
        return True, "", dict(raw)
    from mm_assist.memory.process import close_handle, find_game_pid, open_process
    from mm_assist.memory.readers.facility_queue import read_facility_queues

    pid = find_game_pid()
    if not pid:
        return False, "no_game", {}
    h = None
    try:
        h = open_process(pid)
        q = read_facility_queues(h, pid)
        if not q:
            return False, "facility_scan_empty", {}
        return True, "", dict(q)
    except Exception as exc:
        return False, f"facility_scan:{exc}", {}
    finally:
        if h:
            close_handle(h)


def tick_facility_sim(
    *,
    running: bool,
    now: float | None = None,
    force_scan: bool = False,
    scan_fn=None,
) -> FacilitySimState:
    """running=False 冻结 10 分钟计时。扫描失败 → ok=False，禁止领料。"""
    ts = float(now if now is not None else time.time())
    with _LOCK:
        st = _STATE
        if running:
            if st.tick_at is not None:
                st.accrued_s += max(0.0, ts - st.tick_at)
            st.tick_at = ts
        else:
            st.tick_at = None
        due = force_scan or (running and (st.last_scan_at <= 0 or st.accrued_s >= FACILITY_SCAN_INTERVAL_S))
        scanned = False
        if due:
            scanned = True
            ok, err, slots = (scan_fn or _do_scan)()
            if err != "mem_pending":
                st.last_scan_at = ts
                st.accrued_s = 0.0
            else:
                scanned = False  # Polling a read is not another whole-bag scan.
            if ok:
                st.ok = True
                st.error = ""
                st.slots = {
                    "metal": tuple(slots.get("metal") or ()),
                    "wood": tuple(slots.get("wood") or ()),
                    "cloth": tuple(slots.get("cloth") or ()),
                    "medicine": tuple(slots.get("medicine") or ()),
                }
            else:
                st.ok = False
                st.error = err
        simulated = simulate_facilities(st.slots, now_s=ts) if st.ok else st.slots
        if st.ok:
            st.slots = simulated
        return FacilitySimState(
            ok=st.ok,
            error=st.error,
            last_scan_at=st.last_scan_at,
            accrued_s=st.accrued_s,
            tick_at=st.tick_at,
            did_scan=scanned,
            slots=dict(st.slots),
        )


def overlay_snapshot(snap: LifeSnapshot, slots: dict[FacilityId, tuple[QueueSlot, ...]]) -> LifeSnapshot:
    return replace(snap, items=dict(snap.items), facilities=dict(slots))
