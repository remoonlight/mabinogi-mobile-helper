"""状态栏生活块文案 + 10h 开始后泵计划。"""

from __future__ import annotations

import threading
import time

from mm_assist.life_plan.controller import LifeController
from mm_assist.life_plan import execution_records
from mm_assist.life_plan.types import FACILITY_ORDER, PLAN_CAP, SLOT_CAP, QueueSlot

_FACILITY_ZH = {
    "metal": "金属",
    "wood": "木材",
    "cloth": "布料",
    "medicine": "藥品",
}

_IDLE_HOLD_S = 2.0
_HUD_IDLE_SINCE = 0.0
IDLE_STALL_S = 300.0
SENSE_TIMEOUT_S = 30.0
_IDLE_KEY = ""
_IDLE_T0 = -1.0
_INPUT_PAGE_WAIT_KEY = None
_INPUT_PAGE_WAIT_SINCE = 0.0
INPUT_PAGE_WAIT_S = 60.0
_UNISSUED_BUSY_KEY = None
_UNISSUED_BUSY_STAMP = 0.0
_RECOVERY_KEY = None
_RECOVERY_COUNT = 0
_RECOVERY_FAILURE = None
_RECOVERY_AFTER = 0.0
_RECOVERY_WAIT_SINCE = None
RECOVERY_PAGE_WAIT_S = 60.0


def reset_recovery_for_tests() -> None:
    global _RECOVERY_KEY, _RECOVERY_COUNT, _RECOVERY_FAILURE, _RECOVERY_AFTER
    _RECOVERY_KEY, _RECOVERY_COUNT = None, 0
    _RECOVERY_FAILURE, _RECOVERY_AFTER = None, 0.0
    global _RECOVERY_WAIT_SINCE
    _RECOVERY_WAIT_SINCE = None


def reset_pump_settle() -> None:
    global _HUD_IDLE_SINCE, _IDLE_KEY, _IDLE_T0
    global _INPUT_PAGE_WAIT_KEY, _INPUT_PAGE_WAIT_SINCE
    global _UNISSUED_BUSY_KEY, _UNISSUED_BUSY_STAMP
    global _RECOVERY_WAIT_SINCE
    _RECOVERY_WAIT_SINCE = None
    _HUD_IDLE_SINCE = 0.0
    _IDLE_KEY = ""
    _IDLE_T0 = -1.0
    _INPUT_PAGE_WAIT_KEY = None
    _INPUT_PAGE_WAIT_SINCE = 0.0
    _UNISSUED_BUSY_KEY = None
    _UNISSUED_BUSY_STAMP = 0.0


def idle_stall_due(
    *,
    plan_id: str,
    param_key: str,
    stp: int | None,
    now: float | None = None,
) -> bool:
    """人没在干活（stp 不是 1）且计划+四参不变满 5 分钟。到期后重新计时，避免连跳。"""
    global _IDLE_KEY, _IDLE_T0
    ts = float(now if now is not None else time.time())
    if stp == 1:
        _IDLE_KEY = ""
        _IDLE_T0 = -1.0
        return False
    key = f"{plan_id}|{param_key}"
    if key != _IDLE_KEY:
        _IDLE_KEY = key
        _IDLE_T0 = ts
        return False
    if _IDLE_T0 < 0:
        _IDLE_T0 = ts
        return False
    if ts - _IDLE_T0 < IDLE_STALL_S:
        return False
    _IDLE_T0 = ts
    return True


def _maybe_idle_stall(win):
    import os

    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    from mm_assist.life_execute.ui_logic import ui_param_key
    from mm_assist.life_plan.controller import get_controller
    from mm_assist.perception.state_ui import parse_ui_param_bit

    st = get_controller()
    if not st.running or st.paused or st.current is None:
        return
    if st.pending_action is not None:
        reset_pump_settle()
        return
    params, node = ui_param_key()
    stp = parse_ui_param_bit(params, "stp")
    if not idle_stall_due(
        plan_id=str(st.current.id),
        param_key=f"{params}|{node}",
        stp=stp,
    ):
        return
    from mm_assist.life_plan.types import ExecuteResult
    return ExecuteResult(False, 'idle_stall', 'stp!=1 五分钟计划与四参未变')


def _main_hud_stp0(view: object | None) -> bool:
    """主界面待机只信栏上「界面：」。family=unknown 不得否决待机。"""
    if view is None:
        return False
    stp = getattr(view, "stp", None)
    try:
        stp_i = int(stp) if stp is not None else None
    except (TypeError, ValueError):
        stp_i = None
    if stp_i == 1:
        return False
    from mm_assist.life_execute.ui_act import inspect_is_hud_idle

    return inspect_is_hud_idle()


def _pump_wait_hud_idle(_win) -> bool:
    """开链前只认真主界面待机；2 秒保持在寻找附近之后（skill_nearby）。pytest 跳过。"""
    global _HUD_IDLE_SINCE
    from mm_assist.life_execute.ui_logic import skip_wait_ui_params
    from mm_assist.life_plan.controller import set_current_action

    if skip_wait_ui_params():
        return True
    from mm_assist.perception.state_ui import read_state_ui

    if _main_hud_stp0(read_state_ui()):
        _HUD_IDLE_SINCE = 0.0
        return True
    _HUD_IDLE_SINCE = 0.0
    set_current_action("准备中 等主界面待机")
    return False


def format_process_status(
    *,
    ok: bool,
    last_scan_at: float,
    slots: dict[str, tuple[QueueSlot, ...]] | None = None,
) -> str:
    """每格 1=完成 / 2=加工或排队 / —=空。不读百分比。"""
    if not ok:
        if float(last_scan_at) <= 0:
            return "加工状态：未开始（暂无进度）"
        return "加工状态：扫描失败（不使用旧数据）"
    lines = ["加工状态："]
    data = slots or {}
    for fac in FACILITY_ORDER:
        rows = list(data.get(fac) or ())[:SLOT_CAP]
        codes = ["1" if s.done else "2" for s in rows]
        while len(codes) < SLOT_CAP:
            codes.append("—")
        lines.append(f"{_FACILITY_ZH[fac]}：{' '.join(codes)}")
    return "\n".join(lines)


def max_remain_s(slots: dict[str, tuple[QueueSlot, ...]] | None) -> float | None:
    best: float | None = None
    for rows in (slots or {}).values():
        for slot in rows:
            if slot.done:
                continue
            if slot.remain_s is None:
                continue
            val = float(slot.remain_s)
            best = val if best is None else max(best, val)
    return best


def format_need_items(
    *,
    status: bool | None,
    have: dict[str, int] | None = None,
    error: str = "",
    taken_at: float = 0.0,
    remain_s: float | None = None,
) -> str:
    """必要物品。status False 不用旧背包。"""
    if status is False:
        why = error or "unknown"
        if why == "mem_pending":
            return "必要物品：读取中（等待本次扫描完成）"
        return f"必要物品：内存读取失败：{why}"
    if status is None or have is None:
        return "必要物品：未读取"
    from mm_assist.life_plan.planner import necessary_desired
    from mm_assist.life_plan.recipes import display_need_totals

    lines = ["必要物品："]
    if taken_at > 0:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(taken_at)))
        extra = f"；最长剩余：{int(remain_s)}s" if remain_s is not None else ""
        lines.append(f"读数时刻：{stamp}{extra}")
    desired = necessary_desired(have)
    from mm_assist.life_plan.acquisition import acquisition_plan
    routes = acquisition_plan(have, desired)
    labels = {row['name']: label for key, label in
              [('manual_materials', '自行准备'), ('manual_processing', '人工加工'),
               ('manual_crafting', '人工製作')]
              for row in routes[key]}
    from mm_assist.life_plan.recipes import GATHER_NODE, material_recipes
    from mm_assist.life_plan.elixir_reserve import RESERVE_GOALS
    automatic = set(GATHER_NODE) | set(material_recipes()) | set(RESERVE_GOALS)
    for name, got, want in display_need_totals(have, desired):
        if name not in automatic:
            continue
        marker = f'（{labels[name]}）' if name in labels else ''
        lines.append(f"{name} {got}/{want}{marker}")
    return "\n".join(lines)


_PUMP: threading.Thread | None = None
_PUMP_LOCK = threading.Lock()
_REPAIR: threading.Thread | None = None
_REPAIR_LOCK = threading.Lock()


def pump_thread_alive() -> bool:
    t = _PUMP
    return t is not None and t.is_alive()


def pump_is_current_thread() -> bool:
    t = _PUMP
    return t is not None and threading.current_thread() is t


def repair_busy() -> bool:
    t = _REPAIR
    return t is not None and t.is_alive()


def try_start_one_click_repair(cfg, get_win, on_hint) -> bool:
    """状态栏「一键修理」。生活调度运行中或已在修则拒绝。"""
    from mm_assist.life_plan.controller import get_controller
    from mm_assist.tool_repair import clear_op_cancel

    if get_controller().running or pump_thread_alive():
        on_hint("生活调度进行中，先暂停")
        return False
    with _REPAIR_LOCK:
        global _REPAIR
        if repair_busy():
            on_hint("修理中…先点停止操作")
            return False
        clear_op_cancel()
        try:
            from mm_assist.config import ROOT
            import json as _json

            p = ROOT / "logs" / "repair_click.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                _json.dumps({"ts": time.time(), "via": "statusbar_one_click"}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

        def _run() -> None:
            from mm_assist.life_execute.repair import execute_repair

            try:
                on_hint("修理中…")
                win = get_win() if get_win is not None else None
                r = execute_repair(cfg, win, force=True, on_progress=on_hint)
                if r.status == "cancel":
                    on_hint("已停止")
                else:
                    on_hint(r.message)
            except Exception as exc:
                on_hint(f"异常:{exc}")

        t = threading.Thread(target=_run, daemon=True, name="one-click-repair")
        _REPAIR = t
        t.start()
    return True


def request_stop_repair() -> None:
    from mm_assist.tool_repair import request_op_cancel

    request_op_cancel()
    try:
        from mm_assist.safety.input import release_mouse_buttons

        release_mouse_buttons()
    except Exception:
        pass


def format_life_plan_block(state: LifeController) -> str:
    lines = ["测试跟踪：失败时暂停"] if state.pause_on_failure else []
    if state.start_blocked:
        lines.append(f"开始不可用：{state.block_reason}")
    elif state.paused:
        lines.append("状态：暂停")
    elif state.running:
        lines.append("状态：运行")
    else:
        lines.append("状态：未开始")
    cur = state.current
    lines.append(f"当前计划：{cur.id + ' ' + cur.title if cur else '—'}")
    lines.append(f"当前动作：{state.current_action or '—'}")
    lines.append(f"计划表（{len(state.queue)}/{PLAN_CAP}）:")
    started = float(getattr(state, "life_started_at", 0.0) or 0.0)
    if started > 0:
        lines.append(
            "  本次开始："
            + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))
        )
    if not state.queue:
        lines.append("  （空）")
    else:
        for item in state.queue:
            mark = ">" if cur is not None and item.id == cur.id else " "
            lines.append(f" {mark} {item.id} {item.title}")
    return "\n".join(lines)


def format_life_process_block(state: LifeController) -> str:
    _ = state
    from mm_assist.life_plan.facility_sim import peek_facility_sim

    peek = peek_facility_sim()
    if not peek.ok and peek.error == "mem_pending":
        return "加工状态：扫描中（等待本次读取完成）"
    return format_process_status(ok=peek.ok, last_scan_at=peek.last_scan_at, slots=peek.slots)


def format_life_needs_block(state: LifeController) -> str:
    snap = state.snapshot
    if snap is not None and (snap.ok or snap.items):
        return format_need_items(
            status=True,
            have=dict(snap.items),
            taken_at=float(snap.taken_at or 0.0),
            remain_s=max_remain_s(snap.facilities),
        )
    if snap is not None and not snap.ok:
        return format_need_items(
            status=False,
            error=str(snap.error or "snapshot_failed"),
            taken_at=float(snap.taken_at or 0.0),
        )
    return format_need_items(status=None)


def format_life_needs_columns(state: LifeController) -> tuple[str, str, str]:
    """Read down the left list, then the right; keep quantities and route labels."""
    lines = format_life_needs_block(state).splitlines()
    if not lines or lines[0] != '必要物品：':
        return '\n'.join(lines), '', ''
    caption = []
    if len(lines) > 1 and lines[1].startswith('读数时刻：'):
        caption.append(lines.pop(1))
    rows = lines[1:]
    split = (len(rows) + 1) // 2
    return '\n'.join(caption), '\n'.join(rows[:split]), '\n'.join(rows[split:])


def format_life_panel(state: LifeController) -> str:
    return "\n".join(
        [
            format_life_plan_block(state),
            format_life_process_block(state),
            format_life_needs_block(state),
        ]
    )


def _note_fail_and_continue(*, win, item, reason: str, extra: str = "", trigger: str = "",
                            run_started_at: float | None = None) -> None:
    """正常运行留证并跳过；显式错误跟踪暂停保留计划。"""
    from mm_assist.life_plan.controller import get_controller, skip_current, pause_life, set_current_action
    from mm_assist.life_plan.anomalies import cache_fail_shot, log_life_anomaly
    st = get_controller()
    if (not st.running or st.current is None or st.current.id != item.id
            or (run_started_at is not None and st.life_started_at != run_started_at)):
        return
    pending = st.pending_action
    if pending is not None and pending.actions_complete:
        return
    try:
        shot = cache_fail_shot(win, tag=str(reason or "fail"))
        if shot:
            log_life_anomaly(plan_title=item.title, plan_id=item.id, kind=item.kind,
                             tag="fail_shot", extra=shot, stage=reason)
    except Exception:
        pass
    latest = get_controller()
    if not latest.running or latest.life_started_at != st.life_started_at or latest.current != item:
        return
    repair_needed = False
    if item.kind == 'gather' and item.gather_mode == 'skill':
        from mm_assist.life_plan.controller import check_failed_training_durability
        repair_needed = check_failed_training_durability(
            run_started_at=st.life_started_at, item_id=item.id)
        if repair_needed is None:
            return
        latest = get_controller()
        if not latest.running or latest.life_started_at != st.life_started_at or latest.current != item:
            return
    if latest.pause_on_failure:
        pause_life(run_started_at=st.life_started_at, item_id=item.id)
        set_current_action(f"已暂停：测试发现失败（{extra or reason}）", run_started_at=st.life_started_at)
        log_life_anomaly(plan_title=item.title, plan_id=item.id, kind=item.kind,
            tag='tracking_failure_paused', extra=extra, stage=reason, run_started_at=st.life_started_at)
        return
    skip_current(reason=reason or "fail", extra=extra, action=st.current_action, stage=reason,
                 run_started_at=st.life_started_at, item_id=item.id, repair_needed=repair_needed)



def _finish_pending_action() -> None:
    from mm_assist.life_plan.game_stop import game_stop_interrupted
    if game_stop_interrupted(force=True,allow_frame_wait=True):
        return
    from mm_assist.life_plan import action_finish
    from mm_assist.life_plan.controller import (get_controller, observe_action_completion,
        acknowledge_done, set_current_action)
    st = get_controller()
    pending = st.pending_action
    if not st.running or pending is None:
        return
    if not pending.actions_complete:
        set_current_action("当前计划动作链未完成", run_started_at=pending.run_started_at)
        return
    item = next((x for x in st.queue if x.id == pending.item_id), None)
    if item is None:
        return
    if item.kind == 'gather':
        execution_records.phase('等待采集结束', kind='stp_wait')
        now = time.monotonic()
        if not observe_action_completion(action_finish.read_completion_stp(), now=now,
                item_id=pending.item_id, run_started_at=pending.run_started_at):
            latest = get_controller().pending_action
            count = latest.idle.completed_segments(now) if latest is not None else 0
            set_current_action(f"等待采集结束：stp=0 连续2秒 {count}/2", run_started_at=pending.run_started_at)
            return
    from mm_assist.life_plan.anomalies import log_life_anomaly
    log_life_anomaly(plan_title=item.title, plan_id=item.id, kind=item.kind,
        tag="action_finished", stage="stp_idle_2x2" if item.kind == 'gather' else 'plan_actions_complete',
        extra="连续两个2秒段 stp=0；调度完成" if item.kind == 'gather' else '全部计划动作完成',
        run_started_at=pending.run_started_at)
    acknowledge_done(item.id, run_started_at=pending.run_started_at,
                     claim_received=pending.result.ok and pending.result.status == 'done')
    # acknowledge_done clears this action atomically; never clear a new generation here.


def _execution_page_ready(win) -> bool:
    """An empty ready stack can precede the actual HUD during loading.

    Gate new dispatch only, independently of the user's stp completion rule.
    Known facilities and other pages keep their executor's preparation path.
    """
    if not int(getattr(win, 'hwnd', 0) or 0):
        return True
    from mm_assist.life_execute.ui_logic import resolve_navigation, graph_node, hud_page_pending
    from mm_assist.life_execute.ui_nav import _hud_xy, _verify_late_edges
    from mm_assist.memory.readers.ui_stack import (
        input_blocking_layers, facility_travel_popup_has_only_own_shield,
    )
    from mm_assist.platform.capture import grab_client

    frame_started_at = time.time()
    frame = grab_client(win)
    where = resolve_navigation(frame)
    if (input_blocking_layers(where.stack_names)
            and not facility_travel_popup_has_only_own_shield(where.stack_names)):
        # A known loading/input shield is a readonly wait, not a failed plan
        # or permission to close the visible page underneath it.
        return False
    if hud_page_pending(where):
        return False
    _verify_late_edges(where, frame_started_at=frame_started_at)
    if graph_node(where.node) == 'unknown':
        return False
    if graph_node(where.node) != 'main_hud':
        return True
    if _hud_xy(frame, 'menu', root=None) is not None:
        return True
    # A recognized NPC exit is ready for the executor's preparation, even
    # though the HUD menu is covered. Loading/unknown frames remain blocked.
    from mm_assist.life_execute.repair_dialog import npc_end_dialog_visible, npc_goodbye_visible

    return npc_end_dialog_visible(frame) or npc_goodbye_visible(frame)


def _interrupt_gather_for_repair(win, item, run_at, *, evidence='') -> None:
    from mm_assist.life_plan.controller import get_controller, skip_current, pause_life, set_current_action
    from mm_assist.life_plan.anomalies import cache_fail_shot, log_life_anomaly
    from mm_assist.life_plan.gather_repair import REPAIR_REQUIRED

    st = get_controller()
    if not st.running or st.life_started_at != run_at or st.current != item:
        return
    issued = st.pending_action is not None or st.gather_pending_id == item.id
    evidence = f'{evidence}；主要动作已发出={issued}；中断不记采集成功'
    try:
        shot = cache_fail_shot(win, tag=REPAIR_REQUIRED)
    except Exception:
        shot = ''
    st = get_controller()
    if not st.running or st.life_started_at != run_at or st.current != item:
        return
    log_life_anomaly(plan_title=item.title, plan_id=item.id, kind=item.kind,
        tag=REPAIR_REQUIRED, extra=f'{evidence} · 原图={shot}', run_started_at=run_at)
    if st.pause_on_failure:
        pause_life(run_started_at=run_at, item_id=item.id)
        set_current_action('已暂停：采集菜单提示需要一键修理', run_started_at=run_at)
        return
    skip_current(reason=REPAIR_REQUIRED, extra=evidence,
                 run_started_at=run_at, item_id=item.id, repair_needed=True)


from mm_assist.life_plan.game_stop import guard_life_pump


@execution_records.trace_pump
@guard_life_pump
def pump_life_once(cfg, win, *, initial_result=None) -> None:
    """采集完成等两段2秒；未发失败最多两次已验脱困，再决定跳过。"""
    from mm_assist.life_plan.controller import (select_ready_current, get_controller, execution_items,
        set_action, set_current_action, wait_action_completion)
    from mm_assist.life_plan.execute import try_execute
    from mm_assist.life_plan.types import ExecuteResult
    from mm_assist.life_execute.ui_act import pump_aborted
    st = select_ready_current()
    if not st.running or st.current is None or pump_aborted():
        return
    from mm_assist.life_plan.gather_repair import GatherRepairWatch, REPAIR_REQUIRED, watch_gather_repair
    if st.current.kind == 'gather':
        watch = GatherRepairWatch(win)
        if watch.check():
            _interrupt_gather_for_repair(win, st.current, st.life_started_at,
                evidence=f'栈={watch.names}；采样={watch.sampled_ts}')
            return
        latest = get_controller()
        if not latest.running or latest.life_started_at != st.life_started_at or latest.current != st.current:
            return
    if st.pending_action is not None:
        _finish_pending_action()
        return
    if st.refresh_requested:
        set_current_action('等待刷新材料、耐久及加工状态', run_started_at=st.life_started_at)
        return
    if st.snapshot is None or not st.snapshot.ok:
        pending = st.snapshot is not None and st.snapshot.error == "mem_pending"
        set_action(st.current.title, "等待本次扫描完成" if pending else "扫描失败，等待下次读取")
        return
    item = st.current
    run_at = st.life_started_at
    recovering = _RECOVERY_KEY == (run_at, item.id) and _RECOVERY_FAILURE is not None
    if recovering:
        initial_result = _RECOVERY_FAILURE
    elif int(getattr(win, 'hwnd', 0) or 0):
        from mm_assist.life_plan.action_finish import read_completion_stp
        sample = read_completion_stp()
        if sample.fresh and sample.stp == 1:
            set_current_action('等待移动或当前游戏动作结束，不发出新动作', run_started_at=run_at)
            return
    global _UNISSUED_BUSY_KEY, _UNISSUED_BUSY_STAMP
    if _UNISSUED_BUSY_KEY == (run_at, item.id):
        from mm_assist.life_plan.action_finish import read_completion_stp
        sample = read_completion_stp()
        if not sample.fresh or sample.stp != 0 or sample.timestamp <= _UNISSUED_BUSY_STAMP:
            set_current_action('等待当前游戏动作结束，不发出新动作', run_started_at=run_at)
            return
        _UNISSUED_BUSY_KEY = None
        _UNISSUED_BUSY_STAMP = 0.0
    global _INPUT_PAGE_WAIT_KEY, _INPUT_PAGE_WAIT_SINCE
    page_failure = initial_result
    if page_failure is None and not _execution_page_ready(win):
        key = (run_at, item.id)
        now = time.monotonic()
        if _INPUT_PAGE_WAIT_KEY != key:
            _INPUT_PAGE_WAIT_KEY, _INPUT_PAGE_WAIT_SINCE = key, now
        set_current_action('等待页面就绪：当前界面尚不可操作', run_started_at=run_at)
        if now - _INPUT_PAGE_WAIT_SINCE < INPUT_PAGE_WAIT_S:
            return
        page_failure = ExecuteResult(False, 'input_page_timeout',
                                     '当前界面60秒仍未就绪；未发出动作')
    _INPUT_PAGE_WAIT_KEY, _INPUT_PAGE_WAIT_SINCE = None, 0.0
    # Each executor prepares its own page. Repair must reach prepare_to(main_hud)
    # from a facility; waiting for HUD here prevents that navigation from running.
    set_action(item.title)
    def issued():
        wait_action_completion(item.id, run_at, ExecuteResult(False,'action_pending',action_issued=True))
    def progress(msg):
        set_current_action(str(msg or ''), run_started_at=run_at)
    started = time.monotonic()
    from mm_assist.safety.input import foreground_input_scope, pointer_trace_scope
    with pointer_trace_scope(bool(getattr(getattr(cfg, 'window', None), 'input_trace', False))), \
            foreground_input_scope(int(getattr(win, 'hwnd', 0) or 0), win=win), watch_gather_repair(
            win if item.kind == 'gather' else None) as watch:
        r = _execute_with_recovery(try_execute, item, initial_result=page_failure,
                        cfg=cfg, win=win, items=execution_items(st),
                        facilities=st.snapshot.facilities, on_progress=progress,
                        defer_completion=True, on_action_issued=issued)
        if watch.check():
            r = ExecuteResult(False, REPAIR_REQUIRED,
                f'栈={watch.names}；采样={watch.sampled_ts}；主要动作已发出={r.action_issued}',
                action_issued=r.action_issued)
    from mm_assist.life_plan.anomalies import log_life_anomaly
    from mm_assist.life_plan.game_stop import game_stop_interrupted
    if game_stop_interrupted(force=True):
        return
    now_st = get_controller()
    was_issued = bool(r.action_issued or now_st.pending_action is not None or (r.ok and r.status=='done'))
    waiting = r.status in {'stp_busy', 'recovery_wait', 'stock_read_wait', 'input_layer_wait'} and not was_issued
    log_life_anomaly(plan_title=item.title, plan_id=item.id, kind=item.kind,
        action=now_st.current_action, tag='action_waiting' if waiting else 'execute_result', stage=r.status,
        extra=r.message, ok=r.ok, elapsed_s=round(time.monotonic()-started,3), run_started_at=run_at)
    if now_st.life_started_at != run_at:
        return
    if r.status == REPAIR_REQUIRED:
        _interrupt_gather_for_repair(win, item, run_at, evidence=r.message)
        return
    if r.status == 'recovery_wait':
        return
    if r.status == 'input_layer_wait' and waiting:
        if now_st.running:
            set_current_action('等待系统弹层退出：未发出本次输入，重新认页后定位', run_started_at=run_at)
        return
    if r.status == 'stock_read_wait' and waiting:
        if now_st.running:
            set_current_action('等待库存读取：共享读锁忙，未发出动作', run_started_at=run_at)
        return
    if waiting and now_st.running:
        from mm_assist.life_plan.action_finish import read_completion_stp
        sample = read_completion_stp()
        _UNISSUED_BUSY_KEY = (run_at, item.id)
        _UNISSUED_BUSY_STAMP = sample.timestamp if sample.fresh else time.time()
        set_current_action('等待当前游戏动作结束，不发出新动作', run_started_at=run_at)
        return
    if was_issued:
        wait_action_completion(item.id, run_at, r,
            actions_complete=r.status in {'done', 'action_pending', 'gather_pending'})
    if r.status == 'cancel' or not now_st.running:
        return
    if was_issued:
        if get_controller().pending_action is not None and not get_controller().pending_action.actions_complete:
            _note_fail_and_continue(win=win, item=item, reason=r.status, extra=r.message)
            return
        _finish_pending_action()
        return
    _note_fail_and_continue(win=win, item=item, reason=r.status, extra=r.message)


def _known_input_layer_refusal(message: str) -> bool:
    from mm_assist.memory.readers.ui_stack import INPUT_BLOCKING_NAMES
    text = str(message or '')
    if text.startswith('{') and len(text) <= 512:
        import ast
        try:
            wrapped = ast.literal_eval(text)
            text = str(wrapped.get('error') or '') if isinstance(wrapped, dict) else ''
        except (SyntaxError, ValueError):
            return False
    prefix = 'click_ui_blocked: '
    if text == 'click_ui_not_ready: 网络等待后页面或目标已变化，须重新认页定位':
        return True
    if not text.startswith(prefix):
        return False
    names = text[len(prefix):].split(',')
    return bool(names) and all(name in INPUT_BLOCKING_NAMES for name in names)


def _execute_with_recovery(execute, item, *, initial_result=None, **kwargs):
    """Keep the same item; two verified recoveries, never replay an issued action."""
    from mm_assist.life_plan.controller import get_controller, set_current_action
    from mm_assist.life_plan.anomalies import cache_fail_shot, log_life_anomaly
    from mm_assist.life_execute.ui_recover import MAX_RECOVERIES, recover_once
    from mm_assist.life_execute.ui_act import pump_aborted
    from mm_assist.life_plan.types import ExecuteResult

    initial = get_controller()
    run_at = initial.life_started_at
    global _RECOVERY_KEY, _RECOVERY_COUNT, _RECOVERY_FAILURE, _RECOVERY_AFTER
    global _RECOVERY_WAIT_SINCE

    def active():
        st = get_controller()
        return (st.running and st.life_started_at == run_at
                and st.current == item and not pump_aborted())

    def call():
        try:
            result = execute(item, **kwargs)
        except Exception as exc:
            result = ExecuteResult(False, 'exec_exception', str(exc))
        # This exact native refusal proves this input never reached mouse-down.
        # The next pump must reacquire a new frame/stack and locate again; no
        # coordinates or old authorization survive this readonly wait.
        if (not result.ok and not result.action_issued
                and get_controller().pending_action is None
                and _known_input_layer_refusal(result.message)):
            return ExecuteResult(False, 'input_layer_wait', result.message)
        return result

    def eligible(result):
        st = get_controller()
        safety = ('click_', 'human_busy', 'focus_lost', 'pointer_moved',
                  'may_click=false', 'lease', 'request_stop')
        text = f'{result.status} {result.message}'.lower()
        return (active() and not st.pause_on_failure and not result.ok
                and not result.action_issued and st.pending_action is None
                and result.status not in {'cancel', 'stp_busy', 'stock_read_wait', 'input_layer_wait', 'no_mats', 'not_ready', 'gather_repair_required'}
                and not any(token in text for token in safety))

    result = initial_result if initial_result is not None else call()
    if not eligible(result):
        return result
    key = (run_at, item.id)
    if _RECOVERY_KEY != key:
        _RECOVERY_KEY, _RECOVERY_COUNT = key, 0
        _RECOVERY_FAILURE, _RECOVERY_AFTER = None, 0.0
        _RECOVERY_WAIT_SINCE = None
    recovery_capture = {}
    if (item.kind == 'repair' and kwargs.get('cfg') is not None
            and int(getattr(kwargs.get('win'), 'hwnd', 0) or 0)):
        from mm_assist.nav.map_atlas import _grab_navigation_client
        # Keep repair recovery on the same rendered-frame backend as its map
        # navigation; never authorize Esc from the known stale BitBlt map.
        recovery_capture['grab'] = lambda: _grab_navigation_client(kwargs['cfg'], kwargs['win'])
    # Capture once, before any recovery input. Waiting for a fresh page must
    # neither replace the original failure frame nor flood the evidence folder.
    if _RECOVERY_FAILURE is None:
        try:
            shot = cache_fail_shot(kwargs.get('win'), tag=str(result.status or 'fail'))
        except Exception:
            shot = ''
        log_life_anomaly(plan_title=item.title, plan_id=item.id, kind=item.kind,
            tag='recovery_origin', stage=result.status, extra=f'{result.message} · 原图={shot}',
            run_started_at=run_at)
    while _RECOVERY_COUNT < MAX_RECOVERIES or _RECOVERY_AFTER:
        if not eligible(result):
            break
        attempt = _RECOVERY_COUNT + (0 if _RECOVERY_AFTER else 1)
        set_current_action('重新识别当前界面，保留当前计划', run_started_at=run_at)
        _RECOVERY_FAILURE = result
        count_before = _RECOVERY_COUNT
        def dispatched(stamp):
            global _RECOVERY_COUNT, _RECOVERY_AFTER
            _RECOVERY_COUNT += 1
            _RECOVERY_AFTER = stamp
            set_current_action(f'脱困 {_RECOVERY_COUNT}/{MAX_RECOVERIES}：Esc已发出，等待刷新后验页',
                               run_started_at=run_at)
        try:
            recovered = recover_once(result.status, win=kwargs.get('win'),
                                     cancelled=lambda: not active(),
                                     after_input_at=_RECOVERY_AFTER, on_dispatch=dispatched,
                                     **recovery_capture) or {}
        except Exception as exc:
            recovered = {'ok': False, 'error': str(exc), 'waiting': bool(_RECOVERY_AFTER)}
        # Older injected executors do not expose dispatch metadata. Production
        # always reports esc_sent and calls dispatched before waiting/observing.
        if 'esc_sent' not in recovered and _RECOVERY_COUNT == count_before and not _RECOVERY_AFTER:
            _RECOVERY_COUNT += 1
        if recovered.get('esc_sent') and _RECOVERY_COUNT == count_before:
            _RECOVERY_COUNT += 1
            _RECOVERY_AFTER = float(recovered.get('sent_at') or 0)
        if not active():
            return ExecuteResult(False, 'cancel', '脱困期间暂停或开始新一轮')
        log_life_anomaly(plan_title=item.title, plan_id=item.id, kind=item.kind,
            tag=f'recovery_result_{attempt}', stage=result.status, ok=bool(recovered.get('ok')),
            extra=str(recovered), run_started_at=run_at)
        if any(token in str(recovered).lower() for token in
               ('click_', 'human_busy', 'focus_lost', 'pointer_moved', 'lease')):
            _RECOVERY_FAILURE = None
            return ExecuteResult(False, 'recovery_yield', str(recovered))
        if recovered.get('waiting'):
            if recovered.get('status') == 'stp_busy':
                # Known movement is not a recognition failure or a retry.
                _RECOVERY_WAIT_SINCE = None
            else:
                now = time.monotonic()
                if _RECOVERY_WAIT_SINCE is None:
                    _RECOVERY_WAIT_SINCE = now
                if now - _RECOVERY_WAIT_SINCE >= RECOVERY_PAGE_WAIT_S:
                    message = f'界面{RECOVERY_PAGE_WAIT_S:g}秒仍未就绪；原始失败：{result.status} {result.message}'
                    log_life_anomaly(plan_title=item.title, plan_id=item.id, kind=item.kind,
                        tag='recovery_page_timeout', stage=result.status, extra=message,
                        run_started_at=run_at)
                    if _RECOVERY_AFTER:
                        # An issued Esc still lacks independent verification.
                        # Preserve its budget and current plan for safe resume.
                        from mm_assist.life_plan.controller import pause_life
                        pause_life(run_started_at=run_at, item_id=item.id)
                        set_current_action('已暂停：已发脱困动作未能验页；' + message,
                                           run_started_at=run_at)
                        return ExecuteResult(False, 'cancel', message)
                    _RECOVERY_FAILURE = None
                    _RECOVERY_WAIT_SINCE = None
                    return ExecuteResult(False, 'recovery_page_timeout', message)
            message = ('等待移动或当前游戏动作结束，不发出新动作'
                       if recovered.get('status') == 'stp_busy' else
                       '等待界面刷新并重新识别，保留当前计划')
            set_current_action(message, run_started_at=run_at)
            return ExecuteResult(False, 'recovery_wait', message)
        _RECOVERY_WAIT_SINCE = None
        _RECOVERY_AFTER = 0.0
        if recovered.get('ok'):
            set_current_action(f'脱困已确认：继续{item.title}', run_started_at=run_at)
            result = call()
            if recovered.get('esc_sent') is False:
                # Already independently at HUD: don't open its menu with Esc,
                # nor repeatedly replay a failing executor without an input.
                break
        elif recovered.get('esc_sent') is False and not recovered.get('sent_at'):
            break
    _RECOVERY_FAILURE = None
    return result



def _resolve_life_win(cfg, get_win):
    win = None
    if get_win is not None:
        try:
            win = get_win()
        except Exception:
            win = None
    hwnd = int(getattr(win, "hwnd", 0) or 0) if win is not None else 0
    alive = False
    if hwnd > 0:
        try:
            import win32gui

            alive = bool(win32gui.IsWindow(hwnd))
        except Exception:
            alive = True
    if win is not None and alive:
        return win
    title = str(getattr(getattr(cfg, "window", None), "title_substr", "") or "")
    try:
        from mm_assist.platform.window import find_window

        return find_window(title) or find_window("Mobile") or find_window("瑪奇")
    except Exception:
        return win


def _pump_loop(cfg, get_win) -> None:
    from mm_assist.life_plan.controller import get_controller, tick_facility_progress

    try:
        while get_controller().running:
            tick_facility_progress()
            st = get_controller()
            if st.current is None:
                time.sleep(0.2)
                continue
            win = _resolve_life_win(cfg, get_win)
            try:
                initial_result = _maybe_idle_stall(win)
                pump_life_once(cfg, win, initial_result=initial_result)
            except Exception as exc:
                failed = get_controller()
                cur = failed.current
                if (cur is not None and failed.life_started_at == st.life_started_at
                        and cur == st.current):
                    if failed.pending_action is None:
                        from mm_assist.life_plan.types import ExecuteResult
                        try:
                            pump_life_once(cfg, win, initial_result=ExecuteResult(
                                False, 'exec_exception', str(exc)))
                        except Exception:
                            _note_fail_and_continue(win=win, item=cur,
                                reason='exec_exception', extra=str(exc))
                    else:
                        _note_fail_and_continue(win=win, item=cur,
                            reason='exec_exception', extra=str(exc))
                from mm_assist.life_plan.anomalies import log_life_anomaly

                log_life_anomaly(
                    tag="exec_exception",
                    extra=str(exc),
                    action=get_controller().current_action,
                )
            time.sleep(0.05)
    except Exception as exc:
        from mm_assist.life_plan.anomalies import log_life_anomaly

        log_life_anomaly(tag="pump_died", extra=str(exc))


def start_life_pump(cfg, get_win) -> None:
    global _PUMP
    with _PUMP_LOCK:
        if _PUMP is not None and _PUMP.is_alive():
            return
        t = threading.Thread(target=_pump_loop, args=(cfg, get_win), daemon=True, name="life-repair-pump")
        _PUMP = t
        t.start()


def start_scenario_test(cfg, get_win, scenario: str, on_hint) -> bool:
    """与生产调度共用输入线程位置，不能并发测试、修理、备料。"""
    from mm_assist.life_plan.controller import get_controller
    from mm_assist.life_plan.live_test import run_scenario, scenarios
    if scenario not in scenarios():
        raise ValueError("未知测试场景")
    global _PUMP
    with _PUMP_LOCK:
        if pump_thread_alive() or repair_busy() or get_controller().running or get_controller().preparing:
            on_hint("已有任务运行，不能同时测试")
            return False
        def run():
            try:
                run_scenario(cfg, _resolve_life_win(cfg, get_win), scenario, on_hint)
            except Exception as exc:
                on_hint(f"测试启动失败：{exc}")
        _PUMP = threading.Thread(target=run, daemon=True, name="life-scenario-test")
        _PUMP.start()
        return True


def start_escape_recovery(cfg, get_win, on_hint) -> bool:
    """手动脱困占用唯一输入线程；暂停中的计划与发出事实不变。"""
    from mm_assist.life_plan.controller import get_controller
    from mm_assist.life_execute.ui_recover import MAX_RECOVERIES, recover_once
    from mm_assist.tool_repair import clear_op_cancel, op_cancelled

    global _PUMP
    with _PUMP_LOCK:
        if pump_thread_alive() or repair_busy() or get_controller().running or get_controller().preparing:
            on_hint('正在停止操作，结束后再脱困')
            return False
        run_at = get_controller().life_started_at
        clear_op_cancel()
        def stopped():
            st = get_controller()
            return op_cancelled() or st.running or st.life_started_at != run_at
        def run():
            try:
                win = _resolve_life_win(cfg, get_win)
                count, after = 0, 0.0
                wait_started = time.monotonic()
                def dispatched(stamp):
                    nonlocal count, after
                    count += 1
                    after = stamp
                    on_hint(f'脱困 {count}/{MAX_RECOVERIES}：Esc已发出，等待刷新后验页')
                while count < MAX_RECOVERIES or after:
                    if stopped():
                        return
                    on_hint('重新识别当前界面')
                    before = count
                    result = recover_once('manual_escape', win=win, cancelled=stopped,
                                          after_input_at=after, on_dispatch=dispatched)
                    if 'esc_sent' not in result and count == before:
                        count += 1
                    if stopped():
                        return
                    if any(token in str(result).lower() for token in
                           ('click_', 'human_busy', 'focus_lost', 'pointer_moved', 'lease')):
                        on_hint(f'脱困停止：{result}')
                        return
                    if result.get('waiting'):
                        on_hint('等待移动结束或界面刷新，不发出新动作')
                        if time.monotonic() - wait_started >= INPUT_PAGE_WAIT_S:
                            on_hint('界面未就绪，停止本次脱困')
                            return
                        time.sleep(0.1)
                        continue
                    after = 0.0
                    if result.get('ok'):
                        on_hint('脱困已确认：主界面')
                        return
                    if result.get('esc_sent') is False and not result.get('sent_at'):
                        on_hint(f'脱困停止：{result}')
                        return
                on_hint('两次脱困未确认主界面')
            except Exception as exc:
                on_hint(f'脱困停止：{exc}')
            finally:
                _clear_pump_if_self()
        _PUMP = threading.Thread(target=run, daemon=True, name='life-escape-recovery')
        _PUMP.start()
        return True


def start_life_prepare(cfg, get_win, *, snapshot_fn=None, on_hint=None,
                       pause_on_failure: bool = False) -> bool:
    """与测试/修理互斥。快照在泵位线程；暂停后即使扫完也不能置 running。"""
    from mm_assist.life_plan.controller import (
        apply_prepared_snapshot,
        get_controller,
        mark_life_preparing,
    )
    from mm_assist.life_plan.snapshot import take_snapshot

    global _PUMP
    with _PUMP_LOCK:
        if pump_thread_alive() or repair_busy() or get_controller().running or get_controller().preparing:
            return False
        token, _ = mark_life_preparing(pause_on_failure=pause_on_failure)

        def run():
            hint = on_hint or (lambda _m: None)
            start_pump = False
            try:
                hint("准备中")
                snap_fn = snapshot_fn or take_snapshot
                snap = snap_fn(budget_s=0, include_skills=True)
                view = apply_prepared_snapshot(snap, token)
                if not view.running:
                    hint(view.current_action or "准备已取消")
                    return
                start_pump = cfg is not None and get_win is not None
            except Exception as exc:
                from mm_assist.life_plan.snapshot import failed_snapshot

                apply_prepared_snapshot(failed_snapshot(f"prepare_error:{exc}"), token)
                hint(f"开始失败:{exc}")
            finally:
                _clear_pump_if_self()
            if start_pump:
                start_life_pump(cfg, get_win)

        _PUMP = threading.Thread(target=run, daemon=True, name="life-start-prepare")
        _PUMP.start()
        return True


def _clear_pump_if_self() -> None:
    global _PUMP
    with _PUMP_LOCK:
        if _PUMP is threading.current_thread():
            _PUMP = None


def attach_life_plan_controls(parent, *, tk_mod, cfg=None, get_win=None) -> None:
    import tkinter as tk

    from mm_assist.life_plan.controller import get_controller, pause_life, resume_life

    _ = tk_mod
    font = ("Segoe UI", 8)
    box = tk.LabelFrame(parent, text="生活技能", font=font)
    box.pack(fill="x", padx=8, pady=(0, 8))
    btns = tk.Frame(box)
    btns.pack(anchor="w", padx=6, pady=(4, 2))
    hint = tk.StringVar(value="")
    tk.Label(box, textvariable=hint, font=font, justify="left", wraplength=640).pack(
        anchor="w", padx=6, pady=(0, 4)
    )

    plan_var = tk.StringVar()
    proc_var = tk.StringVar()
    need_caption = tk.StringVar()
    need_left = tk.StringVar()
    need_right = tk.StringVar()
    plan_fr = tk.LabelFrame(box, text="计划表", font=font)
    plan_fr.pack(fill="x", padx=6, pady=(0, 4))
    tk.Label(
        plan_fr, textvariable=plan_var, font=font, justify="left", wraplength=640, anchor="nw"
    ).pack(anchor="w", padx=4, pady=2)
    proc_fr = tk.LabelFrame(box, text="加工状态", font=font)
    proc_fr.pack(fill="x", padx=6, pady=(0, 4))
    tk.Label(
        proc_fr, textvariable=proc_var, font=font, justify="left", wraplength=640, anchor="nw"
    ).pack(anchor="w", padx=4, pady=2)
    need_fr = tk.LabelFrame(box, text="必要物品", font=font)
    need_fr.pack(fill="x", padx=6, pady=(0, 4))
    need_fr.columnconfigure(0, weight=1, uniform='need_columns')
    need_fr.columnconfigure(1, weight=1, uniform='need_columns')
    tk.Label(need_fr, textvariable=need_caption, font=font, justify='left',
             wraplength=620, anchor='nw').grid(row=0, column=0, columnspan=2, sticky='nw', padx=4, pady=2)
    for column, variable in enumerate((need_left, need_right)):
        tk.Label(need_fr, textvariable=variable, font=font, justify='left',
                 wraplength=300, anchor='nw').grid(row=1, column=column, sticky='nw', padx=4, pady=2)

    def _refresh() -> None:
        st = get_controller()
        plan_var.set(format_life_plan_block(st))
        proc_var.set(format_life_process_block(st))
        caption, left, right = format_life_needs_columns(st)
        need_caption.set(caption)
        need_left.set(left)
        need_right.set(right)

    _refresh()

    def _start(*, pause_on_failure: bool = False) -> None:
        if pump_thread_alive() or repair_busy():
            _hint("已有任务运行，先停止再开始")
            return
        _hint("准备中")
        try:
            if not start_life_prepare(cfg, get_win, on_hint=_hint, pause_on_failure=pause_on_failure):
                _hint("已有任务运行，先停止再开始")
        except Exception as exc:
            _hint(f"开始失败:{exc}")
        _refresh()

    def _pause() -> None:
        pause_life()
        _refresh()

    def _resume(*, pause_on_failure: bool | None = None) -> None:
        if pump_thread_alive() or repair_busy():
            _hint("已有任务仍在结束，请稍后恢复")
            return
        try:
            if pause_on_failure is None:
                pause_on_failure = get_controller().pause_on_failure
            st = resume_life(pause_on_failure=pause_on_failure)
            if st.running and cfg is not None and get_win is not None:
                start_life_pump(cfg, get_win)
            elif not st.running:
                _hint(st.current_action or "无法恢复")
        except Exception as exc:
            _hint(f"恢复失败:{exc}")
        _refresh()

    def _hint(msg: str) -> None:
        def _apply() -> None:
            hint.set(str(msg)[:160])
            try:
                if str(msg).startswith("修理中"):
                    repair_btn.configure(text="修理中…")
                else:
                    repair_btn.configure(text="一键修理")
            except Exception:
                pass

        try:
            parent.after(0, _apply)
        except Exception:
            _apply()

    def _one_click() -> None:
        try:
            if cfg is None or get_win is None:
                _hint("无游戏窗上下文")
                return
            try_start_one_click_repair(cfg, get_win, _hint)
        except Exception as exc:
            _hint(f"异常:{exc}")

    def _stop_ops() -> None:
        try:
            from mm_assist.config import ROOT
            import json as _json

            p = ROOT / "logs" / "stop_ops_click.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                _json.dumps({"ts": time.time(), "via": "statusbar_stop_ops"}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
        _pause()
        request_stop_repair()
        _hint("已停止操作")

    tk.Button(btns, text="生活备料", command=_start, font=font).pack(side="left", padx=4)
    tk.Button(btns, text="生活備料（調試）", command=lambda: _start(pause_on_failure=True),
              font=font).pack(side="left", padx=4)
    tk.Button(btns, text="暂停", command=_pause, font=font).pack(side="left", padx=4)
    tk.Button(btns, text="恢复", command=_resume, font=font).pack(side="left", padx=4)
    tk.Button(btns, text="执行记录", command=lambda: execution_records.show_records(parent), font=font).pack(side="left", padx=4)
    repair_btn = tk.Button(btns, text="一键修理", command=_one_click, font=font)
    repair_btn.pack(side="left", padx=4)
    tk.Button(btns, text="停止操作", command=_stop_ops, font=font).pack(side="left", padx=4)

    def _test(scenario: str) -> None:
        start_scenario_test(cfg, get_win, scenario, _hint)

    def _escape_via_cmd() -> None:
        _pause()
        start_escape_recovery(cfg, get_win, _hint)

    def _poll() -> None:
        execution_records.checkpoint()
        try:
            from mm_assist.life_plan.cmd_file import consume_life_cmd

            applied = consume_life_cmd(
                start_fn=_start,
                pause_fn=_pause,
                resume_fn=_resume,
                escape_fn=_escape_via_cmd,
                test_fn=_test,
            )
            if applied == "start":
                _hint("已按文件指令开始")
            elif applied == "pause":
                _hint("已按文件指令暂停")
            elif applied == "resume":
                _hint("已按文件指令恢复")
        except Exception:
            pass
        _refresh()
        try:
            parent.after(500, _poll)
        except Exception:
            pass

    try:
        parent.after(500, _poll)
    except Exception:
        pass
