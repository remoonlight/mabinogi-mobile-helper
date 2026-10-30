"""生活调度状态机：开始=重排，暂停=保留队列，恢复=不清队列再跑。不点游戏。"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field, replace
from mm_assist.life_plan.action_finish import PendingAction
from mm_assist.life_plan import execution_records

from mm_assist.life_plan.planner import (
    training_item,
    training_target_allowed,
    group_gather_items,
    apply_done,
    insert_repair_next,
    needs_repair,
    plan_life,
)
from mm_assist.life_plan.snapshot import take_snapshot, failed_snapshot, snapshot_in_progress
from mm_assist.life_plan.types import (
    FACILITY_SCAN_INTERVAL_S,
    FACILITY_ORDER,
    PLAN_CAP,
    LifeSnapshot,
    PlanItem,
)
from mm_assist.tool_repair import clear_op_cancel, request_op_cancel

_LOCK = threading.Lock()
_STATE: LifeController | None = None


@dataclass
class _PeriodicRead:
    owner: LifeController
    run_started_at: float
    result: LifeSnapshot | None = None
    done: threading.Event = field(default_factory=threading.Event)
    started: bool = True

    def complete(self, result: LifeSnapshot) -> None:
        self.result = result
        self.done.set()


_PERIODIC_READ: _PeriodicRead | None = None


@dataclass
class LifeController:
    running: bool = False
    paused: bool = False
    pause_on_failure: bool = False
    start_blocked: bool = False
    block_reason: str = ""
    snapshot: LifeSnapshot | None = None
    queue: list[PlanItem] = field(default_factory=list)
    current: PlanItem | None = None
    current_action: str = "未开始"
    seq: int = 1
    fail_item_id: str = ""
    fail_n: int = 0
    blocked_facilities: set[str] = field(default_factory=set)
    unsupported: set[str] = field(default_factory=set)
    sense_item_id: str = ""
    sense_started_at: float = 0.0
    sense_backoff_s: float = 0.0
    life_started_at: float = 0.0
    preparing: bool = False
    prepare_token: int = 0
    gather_pending_id: str = ""
    gather_pending_run: float = 0.0
    gather_pending_busy: bool = False
    pending_action: PendingAction | None = None
    skipped_claim_facilities: set[str] = field(default_factory=set)
    skipped_plan_keys: set[tuple[str, ...]] = field(default_factory=set)
    feed_spend: dict[str, tuple[float, dict[str, int]]] = field(default_factory=dict)
    refresh_requested: bool = False
    refresh_after: float = 0.0
    replan_requested: bool = False
    batch_refresh_pending: bool = False
    training_counts: dict[str, int] = field(default_factory=dict)
    tools_checked_at: float = 0.0

    def view(self) -> LifeController:
        return LifeController(
            running=self.running,
            paused=self.paused,
            pause_on_failure=self.pause_on_failure,
            start_blocked=self.start_blocked,
            block_reason=self.block_reason,
            snapshot=self.snapshot,
            queue=list(self.queue),
            current=self.current,
            current_action=self.current_action,
            seq=self.seq,
            fail_item_id=self.fail_item_id,
            fail_n=self.fail_n,
            blocked_facilities=set(self.blocked_facilities),
            unsupported=set(self.unsupported),
            sense_item_id=self.sense_item_id,
            sense_started_at=self.sense_started_at,
            sense_backoff_s=self.sense_backoff_s,
            life_started_at=self.life_started_at,
            preparing=self.preparing,
            prepare_token=self.prepare_token,
            gather_pending_id=self.gather_pending_id,
            gather_pending_run=self.gather_pending_run,
            gather_pending_busy=self.gather_pending_busy,
            pending_action=(replace(self.pending_action, idle=replace(self.pending_action.idle))
                            if self.pending_action is not None else None),
            skipped_claim_facilities=set(self.skipped_claim_facilities),
            skipped_plan_keys=set(self.skipped_plan_keys),
            feed_spend={key:(ts,dict(cost)) for key,(ts,cost) in self.feed_spend.items()},
            refresh_requested=self.refresh_requested,
            refresh_after=self.refresh_after,
            replan_requested=self.replan_requested,
            batch_refresh_pending=self.batch_refresh_pending,
            training_counts=dict(self.training_counts),
            tools_checked_at=self.tools_checked_at,
        )


def get_controller() -> LifeController:
    global _STATE
    with _LOCK:
        if _STATE is None:
            _STATE = LifeController()
        return _STATE.view()


def execution_items(st: LifeController) -> dict[str, int]:
    """Conservative remaining stock; never credit unfinished feed output.

    Snapshot time is scan start. A scan begun during a feed can already include
    its consumption; keep that reservation until a scan starts after completion
    rather than risk double-spending. The original read evidence stays intact.
    """
    snap = st.snapshot
    items = dict(snap.items) if snap is not None else {}
    for completed_at,cost in st.feed_spend.values():
        if snap is not None and completed_at > snap.taken_at:
            for name,qty in cost.items():
                items[name] = max(0,items.get(name,0)-qty)
    return items


def select_ready_current() -> LifeController:
    """按原顺序执行，已发动作等待期间钉住当前项。"""
    with _LOCK:
        if _STATE is None:
            return LifeController()
        _hold_gather_pending(_STATE)
        if _STATE.pending_action is not None:
            pending = _STATE.pending_action
            if pending.run_started_at == _STATE.life_started_at:
                _STATE.current = next((x for x in _STATE.queue if x.id == pending.item_id), _STATE.current)
        return _STATE.view()


def _hold_gather_pending(st: LifeController) -> None:
    if not st.gather_pending_id:
        return
    if float(st.gather_pending_run) != float(st.life_started_at):
        st.gather_pending_id = ""
        st.gather_pending_run = 0.0
        st.gather_pending_busy = False
        return
    hit = next((item for item in st.queue if item.id == st.gather_pending_id), None)
    if hit is not None:
        st.current = hit


def mark_gather_pending(
    item_id: str, run_started_at: float, *, saw_busy: bool = False
) -> LifeController:
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        same = (
            str(st.gather_pending_id) == str(item_id or "")
            and float(st.gather_pending_run) == float(run_started_at)
        )
        st.gather_pending_id = str(item_id or "")
        st.gather_pending_run = float(run_started_at)
        st.gather_pending_busy = bool(saw_busy or (same and st.gather_pending_busy))
        _hold_gather_pending(st)
        return st.view()


def clear_gather_pending() -> None:
    with _LOCK:
        if _STATE is None:
            return
        _STATE.gather_pending_id = ""
        _STATE.gather_pending_run = 0.0
        _STATE.gather_pending_busy = False


def wait_action_completion(item_id: str, run_started_at: float, result, *, actions_complete: bool = False) -> None:
    """Save dispatch evidence across pause/resume, rejecting old generations."""
    with _LOCK:
        if _STATE is None or _STATE.life_started_at != run_started_at:
            return
        if not any(x.id == item_id for x in _STATE.queue):
            return
        pending = _STATE.pending_action
        if pending is None or pending.item_id != item_id:
            _STATE.pending_action = PendingAction(item_id, run_started_at, result, actions_complete)
        else:
            pending.result = result
            pending.actions_complete = pending.actions_complete or actions_complete


def observe_action_completion(sample, *, now: float, item_id: str, run_started_at: float) -> bool:
    with _LOCK:
        if _STATE is None or not _STATE.running or _STATE.pending_action is None:
            return False
        pending = _STATE.pending_action
        if (not pending.actions_complete or pending.run_started_at != _STATE.life_started_at
                or pending.run_started_at != run_started_at or pending.item_id != item_id):
            return False
        return pending.idle.observe(sample, now)


def gather_pending_active(item_id: str, run_started_at: float) -> bool:
    with _LOCK:
        if _STATE is None:
            return False
        return (
            str(_STATE.gather_pending_id) == str(item_id)
            and float(_STATE.gather_pending_run) == float(run_started_at)
            and float(_STATE.life_started_at) == float(run_started_at)
        )


def gather_pending_saw_busy(item_id: str, run_started_at: float) -> bool:
    with _LOCK:
        if _STATE is None:
            return False
        if not (
            str(_STATE.gather_pending_id) == str(item_id)
            and float(_STATE.gather_pending_run) == float(run_started_at)
            and float(_STATE.life_started_at) == float(run_started_at)
        ):
            return False
        return bool(_STATE.gather_pending_busy)


def reset_controller_for_tests() -> None:
    global _STATE, _PERIODIC_READ
    with _LOCK:
        _STATE = LifeController()
        _PERIODIC_READ = None
    from mm_assist.life_plan.facility_sim import reset_facility_sim_for_tests

    reset_facility_sim_for_tests()
    from mm_assist.life_plan.bar_ui import reset_pump_settle, reset_recovery_for_tests

    reset_pump_settle()
    reset_recovery_for_tests()
    clear_op_cancel()


def tick_facility_progress(*, now: float | None = None, force_scan: bool = False) -> LifeController:
    """Refresh both views only at the batch boundary, through one shared read."""
    from mm_assist.life_plan.facility_sim import (
        overlay_snapshot,
        tick_facility_sim,
    )

    ts = float(now if now is not None else time.time())
    with _LOCK:
        global _STATE, _PERIODIC_READ
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        running = bool(st.running) and not st.paused
        owner, run_at = st, st.life_started_at
        # Keep the displayed processing/material snapshot fixed for this batch.
        # Periodic and explicit requests cannot interrupt remaining plans, nor
        # publish a scan while paused. Tool-only failure checks stay separate.
        if st.queue or st.paused:
            return st.view()
        if _PERIODIC_READ is not None and (
            _PERIODIC_READ.owner is not owner or _PERIODIC_READ.run_started_at != run_at
        ):
            _PERIODIC_READ = None
        if running and (st.refresh_requested or _PERIODIC_READ is not None):
            force_scan = True
    refreshed = None

    def scan_all():
        nonlocal refreshed
        global _PERIODIC_READ
        with _LOCK:
            # Consume once, including when sharing a read already in flight.
            # A failed/empty result must not re-arm a scan on every pump tick.
            if _STATE is owner and owner.life_started_at == run_at:
                owner.refresh_requested = False
            job = _PERIODIC_READ
            if job is not None and not job.started and not snapshot_in_progress():
                # Another caller held the scan lock; this periodic read can now start.
                _PERIODIC_READ = None
                job = None
            if job is not None and job.done.is_set():
                refreshed = job.result
                _PERIODIC_READ = None
                # A long pause must not publish an old completed read.
                if (refreshed is not None and refreshed.taken_at >= owner.refresh_after
                        and 0 <= ts - refreshed.taken_at < FACILITY_SCAN_INTERVAL_S):
                    return refreshed.ok, refreshed.error, refreshed.facilities
                refreshed = None
                job = None
            if job is not None:
                refreshed = failed_snapshot("mem_pending", taken_at=ts)
                return False, refreshed.error, {}
            job = _PeriodicRead(owner, run_at)
            _PERIODIC_READ = job
        refreshed = take_snapshot(now=ts, budget_s=3.0, include_tools=True,
                                  include_skills=True, on_complete=job.complete)
        with _LOCK:
            if refreshed.error not in {"mem_budget", "mem_busy"}:
                if _PERIODIC_READ is job:
                    _PERIODIC_READ = None
            else:
                # mem_busy started no worker; wait for the scan lock next tick.
                if refreshed.error == "mem_busy":
                    job.started = False
                refreshed = failed_snapshot("mem_pending", taken_at=ts)
        return refreshed.ok, refreshed.error, refreshed.facilities

    sim = tick_facility_sim(running=running, now=ts, force_scan=force_scan, scan_fn=scan_all)
    with _LOCK:
        st = _STATE
        if st is None or st is not owner or st.life_started_at != run_at:
            return st.view() if st is not None else LifeController()
        if refreshed is not None:
            if (refreshed.ok and st.snapshot is not None
                    and st.tools_checked_at > refreshed.taken_at):
                refreshed = replace(refreshed, tools=st.snapshot.tools,
                    error=';'.join(e for e in refreshed.error.split(';') if e != 'tools_unreadable'))
            st.snapshot = refreshed
            if not refreshed.ok and st.running and not st.queue:
                st.current_action = ("等待本次扫描完成" if refreshed.error == "mem_pending"
                                     else f"材料及工具耐久刷新失败：{refreshed.error}")
            if refreshed.ok:
                logging.getLogger(__name__).info(
                    "periodic snapshot accepted run=%.3f taken_at=%.3f items=%d tools=%d skills=%d error=%s",
                    run_at, refreshed.taken_at, len(refreshed.items), len(refreshed.tools),
                    len(refreshed.skills), refreshed.error)
                st.blocked_facilities.clear()
                st.skipped_claim_facilities.clear()
                st.skipped_plan_keys.clear()
                st.feed_spend = {key:spent for key,spent in st.feed_spend.items()
                                 if spent[0] > refreshed.taken_at}
                if st.running:
                    st.replan_requested = True
                    _rebuild_waiting(st, refreshed)
        if st.snapshot is not None and st.snapshot.ok and sim.ok:
            st.snapshot = overlay_snapshot(st.snapshot, sim.slots)
            # Completed slots inform the next batch. Do not insert a higher
            # numbered claim ahead of this batch's remaining visible plans.
        elif not sim.ok and st.running:
            if st.current is not None and st.current.kind == "claim":
                st.current_action = f"扫描失败不领：{sim.error}"
        if (
            sim.did_scan
            and st.running
            and st.snapshot is not None
            and st.snapshot.ok
        ):
            if needs_repair(st.snapshot) and st.current is not None:
                st.queue, st.seq = insert_repair_next(
                    st.queue, after_id=st.current.id, seq=st.seq
                )
                _order_repair_ids(st)
        return st.view()


_SENSE_BACKOFF_CAP_S = 8.0


def _item_key_part(item: PlanItem) -> str:
    if item.kind == "claim":
        return str(item.facility or "")
    return str(item.recipe or item.detail or "")


def _unsupported_key(item: PlanItem) -> str:
    if str(item.kind or "") == "claim":
        return f"claim|{item.facility or ''}"
    return f"{item.kind}|{_item_key_part(item)}"


def _filter_unsupported(state: LifeController, queue: list[PlanItem]) -> list[PlanItem]:
    from mm_assist.life_plan.recipes import LIFE_TRAINING_EXCLUDED_SKILLS, fishing_plan

    uns = state.unsupported
    pinned = (state.pending_action.item_id if state.pending_action is not None
              and state.pending_action.run_started_at == state.life_started_at else '')
    if not pinned and state.gather_pending_run == state.life_started_at:
        pinned = state.gather_pending_id
    return [item for item in queue if (_unsupported_key(item) not in uns
            and (item.id == pinned or (not fishing_plan(item)
                 and not (item.kind == 'gather' and item.gather_mode == 'skill'
                          and item.skill in LIFE_TRAINING_EXCLUDED_SKILLS))))]


def _apply_current_after_queue(state: LifeController, *, now: float | None = None) -> None:
    state.queue = _filter_unsupported(state, state.queue)
    if not state.queue:
        state.current = None
        state.current_action = "暂无可做的计划"
        return
    state.current = state.queue[0]
    _hold_gather_pending(state)
    if state.pending_action is not None and state.pending_action.run_started_at == state.life_started_at:
        state.current = next((p for p in state.queue if p.id == state.pending_action.item_id), state.current)
    state.current_action = state.current.title


def _order_repair_ids(state: LifeController) -> None:
    """Safety repair stays next; renumber only its never-issued following tail."""
    if not any(p.kind == 'repair' for p in state.queue):
        return
    previous = 0
    ordered = []
    for item in state.queue:
        number = int(item.id.removeprefix('p'))
        if number <= previous:
            number = max(state.seq, previous + 1)
            item = replace(item, id=f'p{number}')
            state.seq = number + 1
        previous = number
        ordered.append(item)
    state.queue = ordered


def _dup_key(item: PlanItem) -> tuple[str, str, str]:
    return (str(item.kind or ""), str(item.facility or ""), str(item.recipe or ""))


def _plan_key(item: PlanItem) -> tuple[str, ...]:
    return tuple(str(value or "") for value in (
        item.kind, item.facility, item.recipe, item.skill, item.detail, item.gather_mode
    ))


def _plan_from_state(state: LifeController, snap: LifeSnapshot) -> list[PlanItem]:
    from mm_assist.life_plan.recipes import RECIPES

    reserved = {}
    for _, cost in state.feed_spend.values():
        recipe = next((name for name, rec in RECIPES.items() if rec.inputs == cost), None)
        if recipe is not None:
            reserved[recipe] = reserved.get(recipe, 0) + 1
    return plan_life(replace(snap, items=execution_items(state)), start_id=state.seq,
                     blocked_facilities=frozenset(state.blocked_facilities),
                     training_counts=state.training_counts, reserved_feeds=reserved,
                     training_target=next((p for p in state.queue if p.gather_mode == 'skill'), None))


def _rebuild_waiting(state: LifeController, snap: LifeSnapshot) -> None:
    """Keep this batch fixed; a post-batch read creates up to ten new items."""
    if (not state.replan_requested or state.pending_action is not None or not snap.ok
            or (state.gather_pending_id and state.gather_pending_run == state.life_started_at)):
        return
    if state.queue:
        state.queue = _filter_unsupported(state, state.queue)
        _apply_current_after_queue(state)
        if not state.queue:
            _refill_empty(state, snap)
        state.replan_requested = False
        return
    if state.batch_refresh_pending and snap.taken_at < state.refresh_after:
        return
    state.batch_refresh_pending = False
    state.replan_requested = False
    _top_up(state, snap)


def _top_up(state: LifeController, snap: LifeSnapshot) -> None:
    """保留现有队列与编号，只补到 PLAN_CAP。不整表重排。"""
    state.snapshot = snap
    state.queue = _filter_unsupported(state, state.queue)
    if not snap.ok:
        _replan(state, snap)
        return
    state.start_blocked = False
    state.block_reason = ""
    room = PLAN_CAP - len(state.queue)
    if room <= 0:
        _apply_current_after_queue(state)
        return
    existing_n = len(state.queue)
    fresh = _plan_from_state(state, snap)
    have_claim = {
        _dup_key(x) for x in state.queue if str(x.kind or "") == "claim"
    }
    added: list[PlanItem] = []
    for item in fresh:
        if _plan_key(item) in state.skipped_plan_keys:
            continue
        key = _dup_key(item)
        if item.kind == "claim" and (key in have_claim or item.facility in state.skipped_claim_facilities):
            continue
        added.append(replace(item, id=f"p{int(state.seq)}"))
        if str(item.kind or "") == "claim":
            have_claim.add(key)
        state.seq = int(state.seq) + 1
        if len(added) >= room:
            break
    state.queue.extend(added)
    state.queue = _filter_unsupported(state, state.queue)
    if (any(p.gather_mode == 'process_mat' for p in state.queue)
            and not any(p.gather_mode == 'skill' for p in fresh)):
        supply = list({_plan_key(p): p for p in state.queue
                       if p.gather_mode == 'process_mat'
                       and _plan_key(p) not in state.skipped_plan_keys}.values())
        index = 0
        while supply and len(state.queue) < PLAN_CAP:
            state.queue.append(replace(supply[index % len(supply)], id=f'p{int(state.seq)}'))
            state.seq += 1
            index += 1
        state.queue[existing_n:] = group_gather_items(state.queue[existing_n:])
        _apply_current_after_queue(state)
        return
    counts = dict(state.training_counts)
    target = next((p for p in state.queue if p.gather_mode == 'skill'
                   and training_target_allowed(p, snap, automatic=True)), None)
    # Choose once for this batch; only a new batch can select another target.
    for _ in range(PLAN_CAP):
        if len(state.queue) >= PLAN_CAP:
            break
        g = target or training_item(snap, counts)
        if (_unsupported_key(g) in state.unsupported
                or _plan_key(g) in state.skipped_plan_keys):
            if target is not None:
                break
            counts[g.skill] = max(counts.values(), default=0) + 1
            continue
        target = g
        break
    if (target is not None and _unsupported_key(target) not in state.unsupported
            and _plan_key(target) not in state.skipped_plan_keys):
        while len(state.queue) < PLAN_CAP:
            state.queue.append(replace(target, id=f"p{int(state.seq)}"))
            state.seq = int(state.seq) + 1
    state.queue[existing_n:] = group_gather_items(state.queue[existing_n:])
    _apply_current_after_queue(state)


def _refill_empty(state: LifeController, snap: LifeSnapshot, *, now: float | None = None) -> None:
    state.current = None
    if not state.batch_refresh_pending:
        state.batch_refresh_pending = True
        state.refresh_requested = True
        state.refresh_after = max(state.refresh_after, float(now if now is not None else time.time()))
    state.current_action = "本批完成，等待刷新必要物品及加工状态"


def _replan(state: LifeController, snap: LifeSnapshot) -> None:
    state.snapshot = snap
    if not snap.ok:
        state.running = False
        state.start_blocked = True
        state.block_reason = snap.error or "snapshot_failed"
        state.queue = []
        state.current = None
        state.fail_item_id = ""
        state.fail_n = 0
        state.current_action = f"快照失败：{state.block_reason}"
        return
    state.start_blocked = False
    state.block_reason = ""
    state.queue = _plan_from_state(state, snap)
    if state.queue:
        state.seq = int(state.queue[-1].id.replace("p", "") or 0) + 1
    state.queue = _filter_unsupported(state, state.queue)
    state.fail_item_id = ""
    state.fail_n = 0
    _apply_current_after_queue(state)


def _commit_start(st: LifeController, snap: LifeSnapshot, now: float | None) -> None:
    execution_records.interrupt_current()
    st.pending_action = None
    st.skipped_claim_facilities.clear()
    st.skipped_plan_keys.clear()
    st.feed_spend.clear()
    st.refresh_requested = False
    st.refresh_after = 0.0
    st.replan_requested = False
    st.batch_refresh_pending = False
    st.training_counts.clear()
    st.tools_checked_at = 0.0
    st.queue = []
    st.paused = False
    st.preparing = False
    st.running = bool(snap.ok)
    st.blocked_facilities.clear()
    st.unsupported.clear()
    st.seq = 1
    st.life_started_at = float(now if now is not None else time.time())
    st.sense_item_id = ""
    st.sense_started_at = 0.0
    st.sense_backoff_s = 0.0
    st.gather_pending_id = ""
    st.gather_pending_run = 0.0
    st.gather_pending_busy = False
    clear_op_cancel()
    try:
        from mm_assist.safety.autofarm_gates import reset_stage_gates

        reset_stage_gates()
    except Exception:
        pass
    try:
        from mm_assist.life_plan.bar_ui import reset_pump_settle

        reset_pump_settle()
    except Exception:
        pass
    _replan(st, snap)


def mark_life_preparing(*, pause_on_failure: bool = False) -> tuple[int, LifeController]:
    """界面线程只打准备中，不扫内存。返回 token，过期快照不得提交。"""
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        st.prepare_token += 1
        st.pause_on_failure = bool(pause_on_failure)
        st.preparing = True
        st.running = False
        st.paused = False
        st.start_blocked = False
        st.block_reason = ""
        st.current_action = "准备中"
        clear_op_cancel()
        return st.prepare_token, st.view()


def apply_prepared_snapshot(
    snap: LifeSnapshot,
    token: int,
    *,
    now: float | None = None,
) -> LifeController:
    """后台快照回来后提交。暂停/换代/取消则丢弃，不得置 running。"""
    ts = float(now if now is not None else time.time())
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        cancelled = False
        try:
            from mm_assist.tool_repair import op_cancelled

            cancelled = bool(op_cancelled())
        except Exception:
            cancelled = False
        if int(st.prepare_token) != int(token):
            return st.view()
        if (not st.preparing) or st.paused or cancelled:
            st.preparing = False
            if not st.running:
                st.current_action = "准备已取消"
            return st.view()
        _commit_start(st, snap, ts)
        view = st.view()
    if view.running:
        from mm_assist.life_plan.facility_sim import seed_facility_slots

        seed_facility_slots(snap.facilities, now=ts)
        with _LOCK:
            return _STATE.view() if _STATE is not None else view
    return view


def start_life(*, snapshot: LifeSnapshot | None = None, now: float | None = None,
               pause_on_failure: bool = False) -> LifeController:
    """开始=立刻新快照并重排，丢弃暂停队列。测试可注入 snapshot。"""
    snap = snapshot if snapshot is not None else take_snapshot(now=now, budget_s=0, include_skills=True)
    ts = float(now if now is not None else time.time())
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        st.pause_on_failure = bool(pause_on_failure)
        _commit_start(st, snap, ts)
        view = st.view()
    if view.running:
        from mm_assist.life_plan.facility_sim import seed_facility_slots

        seed_facility_slots(snap.facilities, now=ts)
        with _LOCK:
            return _STATE.view() if _STATE is not None else view
    return view


def pause_life(*, run_started_at: float | None = None, item_id: str | None = None,
               reason: str = "") -> LifeController:
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        if run_started_at is not None and (st.life_started_at != run_started_at or not st.running
                or st.current is None or st.current.id != item_id):
            return st.view()
        st.running = False
        st.paused = True
        if st.current is not None:
            execution_records.suspend(st.life_started_at, st.current.id)
        if st.pending_action is not None:
            st.pending_action.idle.reset()
        st.preparing = False
        st.prepare_token += 1
        request_op_cancel()
        try:
            from mm_assist.safety.input import release_mouse_buttons

            release_mouse_buttons()
        except Exception:
            pass
        if st.current is not None:
            st.current_action = f"已暂停：{st.current.title}"
        else:
            st.current_action = "已暂停"
        if reason:
            st.current_action = f"已暂停：{reason}"
        view = st.view()
    # A pause releases input; it is not evidence that prior clicks succeeded.
    try:
        from mm_assist.life_plan.bar_ui import reset_pump_settle

        reset_pump_settle()
    except Exception:
        pass
    from mm_assist.life_plan.facility_sim import tick_facility_sim

    tick_facility_sim(running=False, now=time.time(), force_scan=False)
    return view


def resume_life(*, now: float | None = None, pause_on_failure: bool = False) -> LifeController:
    """保留队列、当前条及未验证输入预算；只清业务失败次数，不重排。"""
    ts = float(now if now is not None else time.time())
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        st.pause_on_failure = bool(pause_on_failure)
        if st.start_blocked:
            st.current_action = f"开始不可用：{st.block_reason}"
            return st.view()
        waiting_refresh = st.batch_refresh_pending or st.refresh_requested or (
            _PERIODIC_READ is not None and _PERIODIC_READ.owner is st
            and _PERIODIC_READ.run_started_at == st.life_started_at)
        _apply_current_after_queue(st, now=ts)
        if (not st.queue or st.current is None) and not waiting_refresh:
            st.paused = True
            st.running = False
            st.current_action = "没有可恢复的计划，请点开始"
            return st.view()
        st.paused = False
        st.running = True
        if st.current is not None:
            execution_records.resume(st.life_started_at, st.current.id)
        st.fail_item_id = ""
        st.fail_n = 0
        st.sense_item_id = ""
        st.sense_started_at = 0.0
        st.sense_backoff_s = 0.0
        clear_op_cancel()
        st.current_action = st.current.title if st.current is not None else "等待刷新材料及工具耐久"
        view = st.view()
    # Only a later independent page observation can verify a dispatched step.
    try:
        from mm_assist.life_plan.bar_ui import reset_pump_settle

        reset_pump_settle()
    except Exception:
        pass
    from mm_assist.life_plan.facility_sim import tick_facility_sim

    tick_facility_sim(running=True, now=ts, force_scan=False)
    return view


def acknowledge_done(
    item_id: str,
    *,
    now: float | None = None,
    snapshot: LifeSnapshot | None = None,
    run_started_at: float | None = None,
    claim_received: bool = True,
) -> LifeController:
    """执行器调用：删一条；队空则补缺，不整表重排（测试可注入 snapshot）。"""
    ts = float(now if now is not None else time.time())
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        if not st.running:
            return st.view()
        if run_started_at is not None and float(st.life_started_at) != float(run_started_at):
            return st.view()
        st.fail_item_id = ""
        st.fail_n = 0
        if st.pending_action is not None and st.pending_action.item_id == item_id:
            st.pending_action = None
        if st.gather_pending_id == item_id:
            st.gather_pending_id = ""
            st.gather_pending_run = 0.0
            st.gather_pending_busy = False
        done = next((item for item in st.queue if item.id == item_id), None)
        if done is not None:
            execution_records.finish(st.life_started_at, item_id, 'completed',
                                     '调度完成；业务到账按原独立验收口径')
        if done is not None and done.kind == 'gather' and done.skill:
            from mm_assist.life_plan.recipes import SKILL_GATHER
            if done.skill in {skill for skill, _ in SKILL_GATHER}:
                st.training_counts[done.skill] = st.training_counts.get(done.skill, 0) + 1
        if done is not None and done.kind == 'feed' and done.recipe:
            from mm_assist.life_plan.recipes import RECIPES
            recipe = RECIPES.get(done.recipe)
            if recipe is not None:
                st.feed_spend[item_id] = (ts,dict(recipe.inputs))
        if not claim_received and done is not None and done.kind == "claim":
            st.skipped_claim_facilities.add(str(done.facility or ""))
        if claim_received and done is not None and done.kind == "claim" and done.facility:
            from mm_assist.life_plan.facility_sim import forget_claimed_slots, overlay_snapshot

            slots = forget_claimed_slots(done.facility)
            if st.snapshot is not None:
                st.snapshot = overlay_snapshot(st.snapshot, slots)
            st.blocked_facilities.discard(str(done.facility))
            st.skipped_claim_facilities.discard(str(done.facility))
        st.queue = apply_done(st.queue, item_id)
        if st.replan_requested and st.snapshot is not None:
            _rebuild_waiting(st, st.snapshot)
        if st.queue:
            _apply_current_after_queue(st, now=ts)
            return st.view()
        if snapshot is not None:
            snap = snapshot
            st.snapshot = snap
            _top_up(st, snap)
        else:
            snap = st.snapshot or LifeSnapshot(ok=False, error="snapshot_missing")
            _refill_empty(st, snap, now=ts)
        st.seq = max(st.seq, 1)
        st.running = True
        return st.view()


def set_current_action(text: str, *, run_started_at: float | None = None) -> LifeController:
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        if run_started_at is None or _STATE.life_started_at == run_started_at:
            _STATE.current_action = str(text or "")
            execution_records.phase(_STATE.current_action)
        return _STATE.view()


def set_action(title: str, step: str = "") -> LifeController:
    from mm_assist.life_execute.ui_act import format_act

    return set_current_action(format_act(title, None, step))


def note_sense_pending(item_id: str, *, now: float | None = None) -> float:
    """同一 item 退避 1→2→4→8；换条重置。返回本拍应 sleep 秒数。"""
    ts = float(now if now is not None else time.time())
    iid = str(item_id or "")
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        if st.sense_item_id != iid:
            st.sense_item_id = iid
            st.sense_started_at = ts
            st.sense_backoff_s = 1.0
        else:
            prev = float(st.sense_backoff_s or 0.0)
            st.sense_backoff_s = min(_SENSE_BACKOFF_CAP_S, max(1.0, prev) * 2.0)
        return float(st.sense_backoff_s)


def sense_elapsed(item_id: str, *, now: float | None = None) -> float:
    ts = float(now if now is not None else time.time())
    iid = str(item_id or "")
    with _LOCK:
        st = _STATE
        if st is None or st.sense_item_id != iid or st.sense_started_at <= 0.0:
            return 0.0
        return max(0.0, ts - float(st.sense_started_at))


def clear_sense_pending() -> None:
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        st.sense_item_id = ""
        st.sense_started_at = 0.0
        st.sense_backoff_s = 0.0


def note_exec_fail(item_id: str) -> int:
    """同一条计划失败次数。换条则从 1 计。"""
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        iid = str(item_id or "")
        if st.fail_item_id != iid:
            st.fail_item_id = iid
            st.fail_n = 0
        st.fail_n += 1
        return int(st.fail_n)


def reset_exec_fail() -> None:
    with _LOCK:
        if _STATE is None:
            return
        _STATE.fail_item_id = ""
        _STATE.fail_n = 0


def check_failed_training_durability(*, run_started_at: float, item_id: str) -> bool | None:
    """Read tools before failure transfer; None rejects an interrupted/old run."""
    from mm_assist.life_plan.snapshot import refresh_tools
    from mm_assist.life_plan.anomalies import log_life_anomaly

    with _LOCK:
        owner = _STATE
        item = owner.current if owner is not None else None
        if (owner is None or not owner.running or owner.life_started_at != run_started_at
                or item is None or item.id != item_id or item.kind != 'gather'
                or item.gather_mode != 'skill'):
            return None
        owner.current_action = '练采集失败，检查工具耐久'
    started = time.time()
    try:
        tools = refresh_tools()
    except Exception:
        tools = None
    finished = time.time()
    with _LOCK:
        if (_STATE is not owner or not owner.running or owner.life_started_at != run_started_at
                or owner.current != item):
            return None
        repair_needed = False
        if tools and owner.snapshot is not None:
            owner.snapshot = replace(owner.snapshot, tools=tuple(tools))
            owner.tools_checked_at = finished
            repair_needed = needs_repair(owner.snapshot)
    detail = ('；'.join(f'{t.name}={t.durability:g}' for t in tools)
              if tools else '工具耐久未读到，不推断为零')
    log_life_anomaly(plan_title=item.title, plan_id=item.id, kind=item.kind,
        tag='training_failure_tools', stage='low_durability' if repair_needed else 'tools_checked',
        ok=bool(tools), extra=f'{detail}；读取开始={started:.3f}，结束={finished:.3f}',
        run_started_at=run_started_at)
    return repair_needed


def skip_current(
    *,
    reason: str,
    escaped: bool | None = None,
    extra: str = "",
    action: str = "",
    stage: str = "",
    from_node: str = "",
    run_started_at: float | None = None,
    item_id: str | None = None,
    repair_needed: bool = False,
) -> LifeController:
    """丢掉失败当前条，立刻切下一条。不暂停 10h。队空则补缺，不整表重排。"""
    from mm_assist.life_plan.anomalies import log_life_anomaly

    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        item = st.current
        if run_started_at is not None and (st.life_started_at != run_started_at or not st.running
                or item is None or item.id != item_id):
            return st.view()
        if item is None:
            return st.view()
        title = item.title
        pid = item.id
        kind = item.kind
        execution_records.finish(st.life_started_at, item.id, 'skipped', extra or reason)
        if st.pending_action is not None and st.pending_action.item_id == item.id:
            st.pending_action = None
        if st.gather_pending_id == item.id:
            st.gather_pending_id = ''
            st.gather_pending_run = 0.0
            st.gather_pending_busy = False
        act = str(action or st.current_action or title)
        fac = str(item.facility or "")
        # Filter only newly generated plans from this unchanged observation.
        # Already queued duplicates remain eligible; no timer or class cooldown.
        st.skipped_plan_keys.add(_plan_key(item))
        if kind == "claim":
            # Do not recreate a skipped item from the same unchanged facility snapshot.
            # Existing queued claims remain eligible; a fresh scan permits new planning.
            st.skipped_claim_facilities.add(fac)
        if reason == "slots_full" and kind == "feed" and fac:
            st.blocked_facilities.add(fac)
            st.queue = [
                q
                for q in st.queue
                if not (q.kind == "feed" and str(q.facility or "") == fac)
            ]
        st.sense_item_id = ""
        st.sense_started_at = 0.0
        st.sense_backoff_s = 0.0
        st.fail_item_id = ""
        st.fail_n = 0
        st.queue = apply_done(st.queue, item.id)
        st.queue = _filter_unsupported(st, st.queue)
        if repair_needed:
            repair = next((p for p in st.queue if p.kind == 'repair'), None)
            if repair is not None:
                st.queue = [repair, *(p for p in st.queue if p.id != repair.id)]
            else:
                st.queue, st.seq = insert_repair_next(st.queue, after_id='', seq=st.seq)
                if reason == 'gather_repair_required':
                    st.queue[0] = replace(st.queue[0], reason='采集菜单出现 IngredientGuideUIInstance')
            _order_repair_ids(st)
        if st.queue:
            _apply_current_after_queue(st)
        else:
            snap = st.snapshot or LifeSnapshot(ok=False, error="snapshot_missing")
            _refill_empty(st, snap)
            st.running = True
        view = st.view()
    log_life_anomaly(
        plan_title=title,
        plan_id=pid,
        kind=kind,
        action=act,
        tag=reason,
        escaped=escaped,
        extra=extra,
        stage=stage,
        from_node=from_node,
    )
    return view


def mark_unsupported_and_skip(
    *,
    reason: str,
    extra: str = "",
    action: str = "",
    stage: str = "",
) -> LifeController:
    """会话内标记 unsupported，移出队列但不进冷却。"""
    from mm_assist.life_plan.anomalies import log_life_anomaly

    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        item = st.current
        if item is None:
            return st.view()
        title = item.title
        pid = item.id
        kind = item.kind
        act = str(action or st.current_action or title)
        st.unsupported.add(_unsupported_key(item))
        st.sense_item_id = ""
        st.sense_started_at = 0.0
        st.sense_backoff_s = 0.0
        st.fail_item_id = ""
        st.fail_n = 0
        st.queue = apply_done(st.queue, item.id)
        st.queue = _filter_unsupported(st, st.queue)
        if st.queue:
            _apply_current_after_queue(st)
        else:
            snap = st.snapshot or LifeSnapshot(ok=False, error="snapshot_missing")
            _refill_empty(st, snap)
            st.running = True
        view = st.view()
    log_life_anomaly(
        plan_title=title,
        plan_id=pid,
        kind=kind,
        action=act,
        tag=reason,
        extra=extra,
        stage=stage,
    )
    return view


def requeue_current() -> LifeController:
    """当前条原 id 挪到队尾，不丢条、不重编号。"""
    with _LOCK:
        global _STATE
        if _STATE is None:
            _STATE = LifeController()
        st = _STATE
        item = st.current
        if item is None:
            return st.view()
        rest = [x for x in st.queue if x.id != item.id]
        st.sense_item_id = ""
        st.sense_started_at = 0.0
        st.sense_backoff_s = 0.0
        st.fail_item_id = ""
        st.fail_n = 0
        st.queue = rest + [item]
        _apply_current_after_queue(st)
        return st.view()
