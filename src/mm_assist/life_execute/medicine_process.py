"""加工投领短链。ExecuteResult done/ok 仅表示加工过程结果，不是材料到账或队列入队。

独立材料/队列验收在 life_plan.live_test / TESTING。不得凭设施结算宣称材料净增。
不得宣称 C2 / 10h 可用。
"""

from __future__ import annotations

from mm_assist.life_plan.game_stop import operation_monotonic, operation_wall_time

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from mm_assist.config import ROOT
from mm_assist.life_plan.recipes import RECIPES, can_afford
from mm_assist.life_plan.types import ExecuteResult, facility_slot_cap
from mm_assist.life_plan.execution_records import trace_wait

FACILITY = "medicine"
ALLOWED = ("metal", "wood", "cloth", "medicine")
_FAC_ZH = {"metal": "金屬", "wood": "木材", "cloth": "布料", "medicine": "藥品"}
MAX_CLICKS = 2
WAIT_WALK_S = 180.0
SLOT_ROI = (0.02, 0.86, 0.42, 0.12)
PROCESS_NAVIGATION_CONFIG = Path('knowledge/ui/config/process_navigation.json')


def _travel_still_busy(read_stp=None) -> bool:
    if read_stp is not None:
        return read_stp() == 1
    from mm_assist.life_plan.action_finish import read_completion_stp

    sample = read_completion_stp()
    return sample.fresh and sample.stp == 1


def _recipe_list_down_points(frame: Any, *, root: Path | None = None) -> tuple[int, int, int, int]:
    """Configured vertical gesture inside the independently recognized list."""
    from mm_assist.perception.process_sense import FACILITY_LIST_ROI

    spec = json.loads(((root or ROOT) / PROCESS_NAVIGATION_CONFIG).read_text(encoding='utf-8'))['recipe_list_down']
    x, y0, y1 = spec['x'], spec['start_y'], spec['end_y']
    rx, ry, rw, rh = FACILITY_LIST_ROI
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (x, y0, y1)) or not rx < x < rx + rw or not ry < y1 < y0 < ry + rh):
        raise ValueError('invalid vertical recipe-list drag configuration')
    h, w = frame.shape[:2]
    points = int(w * x), int(h * y0), int(w * x), int(h * y1)
    if points[1] <= points[3]:
        raise ValueError('recipe-list drag frame is too small')
    return points


def _client_of(hit: Any) -> tuple[int, int] | None:
    from mm_assist.life_execute.ui_logic import recognized_click_xy

    return recognized_click_xy(hit)


def _recipe_detail_open(recipe: str, btn: dict[str, Any]) -> bool:
    """明细弹层已开且加工钮在：再点列表格会关掉弹层。"""
    if not btn.get("ok"):
        return False
    from mm_assist.perception.item_sense import fold_zh
    from mm_assist.life_execute.ui_logic import read_inspect_page, read_inspect_record

    want = fold_zh(recipe)
    if not want:
        return False
    page = fold_zh(read_inspect_page() or "")
    zh = fold_zh(str((read_inspect_record() or {}).get("zh") or ""))
    blob = fold_zh(str(btn.get("materials_ocr") or ""))
    return want in page or want in zh or want in blob


def slot_signature(frame: Any) -> tuple[float, float]:
    if frame is None:
        return (0.0, 0.0)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    x, y, ww, hh = SLOT_ROI
    x0, y0 = int(w * x), int(h * y)
    x1, y1 = int(w * (x + ww)), int(h * (y + hh))
    crop = frame[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
    if crop.size == 0:
        return (0.0, 0.0)
    return float(crop.mean()), float(crop.std())


def slot_changed(before: tuple[float, float], after: tuple[float, float]) -> bool:
    return abs(after[0] - before[0]) >= 1.5 or abs(after[1] - before[1]) >= 1.5


def execute_medicine_claim(
    cfg: Any = None,
    win: Any = None,
    *,
    items: dict[str, int] | None = None,
    do_click: Callable[[int, int], bool] | None = None,
    read_items: Callable[[], dict[str, int] | None] | None = None,
    grab: Callable[[], Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
    root: Path | None = None,
    **kwargs: Any,
) -> ExecuteResult:
    _ = (cfg, kwargs)
    return execute_facility_claim(
        cfg,
        win,
        facility=FACILITY,
        items=items,
        do_click=do_click,
        read_items=read_items,
        grab=grab,
        on_progress=on_progress,
        root=root,
        **kwargs,
    )


def execute_facility_claim(
    cfg: Any = None,
    win: Any = None,
    *,
    facility: str,
    items: dict[str, int] | None = None,
    do_click: Callable[[int, int], bool] | None = None,
    read_items: Callable[[], dict[str, int] | None] | None = None,
    grab: Callable[[], Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
    root: Path | None = None,
    **kwargs: Any,
) -> ExecuteResult:
    _ = (cfg, kwargs)
    fac = str(facility or "").strip()
    if fac not in ALLOWED:
        return ExecuteResult(ok=False, status="bad_facility", message="不是可领设施")
    return _run_process(
        win,
        facility=fac,
        recipe="",
        mode="claim",
        items=items,
        do_click=do_click,
        read_items=read_items,
        grab=grab,
        on_progress=on_progress,
        root=root,
        sleep_fn=kwargs.get("sleep_fn"),
        read_stp=kwargs.get("read_stp"),
        facilities=kwargs.get("facilities"),
        defer_completion=bool(kwargs.get("defer_completion")),
        on_action_issued=kwargs.get("on_action_issued"),
    )


def execute_medicine_feed(
    cfg: Any = None,
    win: Any = None,
    *,
    recipe: str,
    items: dict[str, int] | None = None,
    do_click: Callable[[int, int], bool] | None = None,
    read_items: Callable[[], dict[str, int] | None] | None = None,
    grab: Callable[[], Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
    root: Path | None = None,
    **kwargs: Any,
) -> ExecuteResult:
    _ = (cfg, kwargs)
    name = str(recipe or "").strip()
    rec = RECIPES.get(name)
    if rec is None or rec.facility != FACILITY:
        return ExecuteResult(ok=False, status="bad_recipe", message="不是藥品配方")
    return execute_facility_feed(
        cfg,
        win,
        facility=FACILITY,
        recipe=name,
        items=items,
        do_click=do_click,
        read_items=read_items,
        grab=grab,
        on_progress=on_progress,
        root=root,
        sleep_fn=kwargs.get("sleep_fn"),
        read_stp=kwargs.get("read_stp"),
        facilities=kwargs.get("facilities"),
        facility_slot_caps=kwargs.get("facility_slot_caps"),
        defer_completion=bool(kwargs.get("defer_completion")),
        on_action_issued=kwargs.get("on_action_issued"),
    )


def execute_facility_feed(
    cfg: Any = None,
    win: Any = None,
    *,
    facility: str,
    recipe: str,
    items: dict[str, int] | None = None,
    do_click: Callable[[int, int], bool] | None = None,
    read_items: Callable[[], dict[str, int] | None] | None = None,
    grab: Callable[[], Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
    root: Path | None = None,
    **kwargs: Any,
) -> ExecuteResult:
    _ = (cfg, kwargs)
    fac = str(facility or "").strip()
    if fac not in ALLOWED:
        return ExecuteResult(ok=False, status="bad_facility", message="不是可投设施")
    name = str(recipe or "").strip()
    rec = RECIPES.get(name)
    if rec is None or rec.facility != fac:
        return ExecuteResult(ok=False, status="bad_recipe", message="配方与设施不符")
    from mm_assist.life_plan import recipes as recipe_registry
    if not recipe_registry.feed_registered(name):
        return ExecuteResult(False, 'not_ready', f'{name}配方定位尚未登记，不投料')
    have = dict(items or {})
    if items is None:
        from mm_assist.life_plan.snapshot import take_snapshot

        snap = take_snapshot()
        have = dict(snap.items) if snap.ok else {}
    if not can_afford(name, have) and (read_items is not None or (do_click is None and win is not None)):
        # A batch snapshot predates its claims. Recheck actual stock only on a
        # shortage; never credit predicted output or rescan the whole snapshot.
        from mm_assist.life_execute.ui_act import pump_aborted
        from mm_assist.life_plan.snapshot import ItemsReadBusy, refresh_items
        try:
            fresh = (read_items or refresh_items)()
        except ItemsReadBusy as exc:
            if pump_aborted():
                return ExecuteResult(False, 'cancel', '已停止操作')
            return ExecuteResult(False, 'stock_read_wait', f'等待库存读取：{exc}读锁忙；未发出动作')
        except Exception:
            fresh = None
        if pump_aborted():
            return ExecuteResult(False, 'cancel', '已停止操作')
        if fresh is None:
            return ExecuteResult(False, 'items_unknown', '当前库存未确认，不投料')
        have = dict(fresh)
    if not can_afford(name, have):
        return ExecuteResult(ok=False, status="no_mats", message=f"{name}材料不够")
    return _run_process(
        win,
        facility=fac,
        recipe=name,
        mode="feed",
        items=have,
        do_click=do_click,
        read_items=read_items,
        grab=grab,
        on_progress=on_progress,
        root=root,
        sleep_fn=kwargs.get("sleep_fn"),
        read_stp=kwargs.get("read_stp"),
        facilities=kwargs.get("facilities"),
        facility_slot_caps=kwargs.get("facility_slot_caps"),
        defer_completion=bool(kwargs.get("defer_completion")),
        on_action_issued=kwargs.get("on_action_issued"),
    )


def _run_process(
    win: Any,
    *,
    facility: str,
    recipe: str,
    mode: str,
    items: dict[str, int] | None,
    do_click: Callable[[int, int], bool] | None,
    read_items: Callable[[], dict[str, int] | None] | None,
    grab: Callable[[], Any] | None,
    on_progress: Callable[[str], None] | None,
    root: Path | None,
    sleep_fn: Callable[[float], None] | None = None,
    read_stp: Callable[[], int | None] | None = None,
    facilities: dict | None = None,
    facility_slot_caps: dict | None = None,
    defer_completion: bool = False,
    on_action_issued: Callable[[], None] | None = None,
    memory_reader: Callable[[], dict | None] | None = None,
) -> ExecuteResult:
    fac = str(facility or "").strip()
    title = _FAC_ZH.get(fac, fac)

    def _prog(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    def _click(x: int, y: int) -> bool:
        if do_click is not None:
            return bool(do_click(x, y))
        if win is None:
            return False
        from mm_assist.safety.input import click_client

        return bool(click_client(win, x, y))

    memory_frame = None
    memory_view = None
    memory_error = ''
    memory_required = memory_reader is not None or (do_click is None and win is not None)

    def _read_memory() -> dict | None:
        nonlocal memory_error
        if memory_reader is not None:
            return memory_reader()
        if do_click is not None or win is None:
            return None
        from mm_assist.memory.readers.ui_stack import shared_ui_stack_reader
        diagnostics={}
        fact=shared_ui_stack_reader().read_facility_ui(diagnostics=diagnostics)
        memory_error=str(diagnostics)
        return fact

    def _frame(*, wait_memory: bool = False) -> Any:
        nonlocal memory_frame, memory_view
        deadline = operation_monotonic() + 2.0 if wait_memory else 0.0
        for observation in range(24):
            before = _read_memory()
            if grab is not None:
                frame = grab()
            elif win is None:
                frame = None
            else:
                from mm_assist.platform.capture import grab_client
                frame = grab_client(win)
            after = _read_memory()
            memory_frame = frame
            memory_view = after if before == after else {'error': 'memory_changed'}
            if memory_required and (memory_view is None or memory_view.get('error')):
                import logging
                logging.getLogger(__name__).info('facility memory pending: %s paired=%s',memory_error,before==after)
            if (not wait_memory or (isinstance(memory_view, dict)
                    and memory_view.get('facility') and not memory_view.get('error'))):
                break
            from mm_assist.life_execute.ui_act import pump_aborted
            if pump_aborted() or operation_monotonic() >= deadline or observation == 23:
                break
            _prog('等待设施内存稳定')
            sleeper(min(0.1, max(0.0, deadline-operation_monotonic())))
        return frame

    def _memory_for(frame):
        return memory_view if frame is memory_frame else None

    def _claim_button(frame):
        memory = _memory_for(frame)
        if memory is None and memory_required:
            return {'ok': False, 'found': False, 'src': 'memory', 'error': 'memory_unavailable'}
        kwargs = {'memory_view': memory, 'facility': fac} if memory is not None else {}
        return locate_claim_all(frame, root=root, **kwargs)

    def _go_button(frame):
        memory = _memory_for(frame)
        if memory is None and memory_required:
            return {'ok': False, 'found': False, 'src': 'memory', 'error': 'memory_unavailable'}
        kwargs = {'memory_view': memory, 'facility': fac} if memory is not None else {}
        return locate_go_equip(frame, root=root, **kwargs)

    if grab is None and win is None:
        return ExecuteResult(ok=False, status="need_context", message=f"{title}投领需要游戏窗")

    if mode == "feed":
        if facilities is None:
            from mm_assist.life_plan.snapshot import take_snapshot as take_slots

            occ = take_slots()
            if occ.ok:
                facilities = occ.facilities
                facility_slot_caps = occ.facility_slot_caps
        cap = facility_slot_cap(facility_slot_caps, fac)
        if facilities is None or cap is None:
            return ExecuteResult(ok=False, status="facility_unknown", message="槽位读数缺失，不投料")
        if len(facilities.get(fac) or ()) >= cap:
            return ExecuteResult(ok=False, status="slots_full", message=f"{title}槽已满不投")

    armed = do_click is None
    if armed:
        from mm_assist.safety.lease import prefer_visible_clicks, production_mode

        production_mode(True)
        prefer_visible_clicks(True)
    try:
        from mm_assist.perception.process_sense import (
            detect_facility_page,
            locate_claim_all,
            locate_go_equip,
            locate_hub_card,
            locate_process_button,
            locate_process_recipe,
        )
        from mm_assist.platform.window import focus_window, is_foreground

        from mm_assist.life_execute.ui_act import (
            confirm_go_facility_popup,
            dismiss_facility_result,
            hold_facility_settle,
        )
        from mm_assist.life_execute.ui_logic import skip_wait_ui_params

        sleeper = sleep_fn or time.sleep
        from mm_assist.safety.input import foreground_input_enabled
        if win is not None and not foreground_input_enabled(win.hwnd) and not is_foreground(win.hwnd):
            focus_window(win.hwnd, force=True)
            sleeper(0.25)

        def _stp() -> int | None:
            if read_stp is not None:
                return read_stp()
            if not armed:
                return None
            from mm_assist.perception.state_ui import read_state_ui

            view = read_state_ui()
            if view is None:
                return None
            try:
                return int(view.stp) if view.stp is not None else None
            except (TypeError, ValueError):
                return None

        def _one(stage: str, pt: tuple[int, int] | None) -> ExecuteResult | None:
            from mm_assist.life_execute.ui_act import click_spot

            err = click_spot(_click, pt, stage, on_progress=_prog,
                on_dispatch=on_action_issued if stage in {"加工", "全部領取"} else None)
            if err is not None:
                return err
            return None

        def _confirm_go_if_popup() -> ExecuteResult | None:
            return confirm_go_facility_popup(_click, _frame, on_progress=_prog, root=root)

        def _cancelled() -> ExecuteResult | None:
            from mm_assist.life_execute.ui_act import pump_aborted

            if pump_aborted():
                return ExecuteResult(False, "cancel", "已停止操作")
            return None

        def _wait_walk_stop() -> ExecuteResult | None:
            """等 stp=1 再到 stp=0，或「前往設備」消失（加工页盖住主界面时停钮可能不变）。"""
            if skip_wait_ui_params() and sleep_fn is None:
                return None
            deadline = operation_wall_time() + WAIT_WALK_S
            saw1 = False
            asked = False
            gone_since = 0.0
            while operation_wall_time() < deadline:
                cancelled = _cancelled()
                if cancelled is not None:
                    return cancelled
                if not asked:
                    from mm_assist.life_execute.ui_act import stack_has_common_popup

                    if stack_has_common_popup():
                        asked = True
                        go = _confirm_go_if_popup()
                        if go is not None:
                            return go
                cur = _stp()
                if cur == 1:
                    saw1 = True
                elif saw1 and cur == 0:
                    return None
                frame = _frame()
                away = _go_button(frame)
                on_fac = _on_this_facility(frame)
                # 关加工页去走路时「前往設備」也会消失，不能当成已到站。
                memory = _memory_for(frame)
                known_absence = (memory is not None
                    and memory.get('buttons', {}).get('go_equip', {}).get('present') is False)
                if on_fac and (known_absence if memory_required else not away.get("ok")):
                    if gone_since <= 0.0:
                        gone_since = operation_wall_time()
                    if operation_wall_time() - gone_since >= 1.0:
                        return None
                else:
                    gone_since = 0.0
                sleeper(0.2)
            return ExecuteResult(ok=False, status="walk_timeout", message="走到设施超时")

        @trace_wait('等待设施停钮')
        def _wait_stop_then_process(btn: dict[str, Any]) -> ExecuteResult | None:
            """走到后栏上仍可以是配方页；已知停钮灭再点加工。未知/仍在走不投。"""
            _ = btn
            if skip_wait_ui_params() and sleep_fn is None:
                return None
            deadline = operation_wall_time() + WAIT_WALK_S
            while operation_wall_time() < deadline:
                cancelled = _cancelled()
                if cancelled is not None:
                    return cancelled
                cur = _stp()
                if cur == 0:
                    return None
                if cur == 1:
                    _prog("等待停钮")
                    sleeper(0.2)
                    continue
                if skip_wait_ui_params():
                    return ExecuteResult(ok=False, status="stp_unknown", message="停钮未知，不点加工")
                sleeper(0.2)
            if _stp() == 1:
                return ExecuteResult(ok=False, status="stp_busy", message="仍在走，不点加工、不做2+2")
            return ExecuteResult(ok=False, status="stp_unknown", message="停钮未知，不点加工")

        def _process_stance(btn: dict[str, Any]) -> str | None:
            if not btn.get("ok"):
                return None
            label = str(btn.get("label") or "")
            at = btn.get("at_facility")
            if at is False or label == "前往加工":
                if at is True:
                    return None
                return "away"
            if at is True and label == "加工":
                return "here"
            return None

        def _confirm_process_btn(
            *,
            for_feed: bool = False,
        ) -> tuple[dict[str, Any] | None, ExecuteResult | None]:
            def _pair() -> tuple[dict[str, Any], dict[str, Any], str | None, str | None]:
                first_frame = _frame()
                first = locate_process_button(first_frame, root=root)
                fresh_frame = _frame()
                fresh = locate_process_button(fresh_frame, root=root)
                s0, s1 = _process_stance(first), _process_stance(fresh)
                if armed and (s0 is None or s1 is None or s0 != s1):
                    import logging
                    from mm_assist.life_plan.anomalies import cache_fail_shot

                    paths = [cache_fail_shot(frame=first_frame, tag="process_pair_first"),
                             cache_fail_shot(frame=fresh_frame, tag="process_pair_fresh")]
                    logging.getLogger(__name__).warning(
                        "process_pair facility=%s recipe=%s for_feed=%s stance=%s->%s first=%r fresh=%r frames=%r",
                        fac, recipe, for_feed, s0, s1, first, fresh, paths,
                    )
                return first, fresh, s0, s1

            def _refuse_unknown(fresh: dict[str, Any], s1: str | None) -> ExecuteResult:
                if s1 is None and not fresh.get("ok"):
                    return ExecuteResult(ok=False, status="no_process_btn", message="没有加工钮")
                if s1 is None:
                    return ExecuteResult(ok=False, status="btn_ambiguous", message="加工钮语义不清，不点")
                return ExecuteResult(ok=False, status="btn_unstable", message="加工钮未稳住，不点")

            first, fresh, s0, s1 = _pair()
            if for_feed:
                if s0 == "here" and s1 == "here":
                    return fresh, None
                if not (skip_wait_ui_params() and sleep_fn is None):
                    sleeper(0.2)
                    _, extra, e0, e1 = _pair()
                    if e0 == "here" and e1 == "here":
                        return extra, None
                    fresh, s1 = extra, e1
                return None, _refuse_unknown(fresh, s1)
            if s1 is None:
                return None, _refuse_unknown(fresh, s1)
            if s0 == "here" and s1 == "away":
                return fresh, None
            if s0 is not None and s0 != s1:
                return None, ExecuteResult(ok=False, status="btn_unstable", message="加工钮未稳住，不点")
            return fresh, None

        def _facility_identity(frame: Any) -> str:
            memory = _memory_for(frame)
            if memory is not None:
                return str(memory.get('facility') or '')
            page = detect_facility_page(frame, expected=fac, root=root)
            if page.get("ok"):
                return str(page.get("facility") or fac)
            seen = str(page.get("facility") or "").strip()
            if seen:
                return seen
            from mm_assist.life_execute.ui_logic import (
                facility_id_of,
                read_inspect_page,
                resolve_ui_node,
            )

            fid = facility_id_of(str(read_inspect_page(root=root) or ""))
            if fid:
                return fid
            where = resolve_ui_node(frame=frame, root=root)
            fid = facility_id_of(str(getattr(where, "node", "") or ""))
            if fid:
                return fid
            return str(getattr(where, "facility_id", "") or "").strip()

        def _leave_wrong(frame: Any) -> bool:
            memory = _memory_for(frame)
            if memory is not None:
                seen = str(memory.get('facility') or '')
                return bool(seen) and seen != fac
            page = detect_facility_page(frame, expected=fac, root=root)
            seen = str(page.get("facility") or "").strip()
            return bool(seen) and seen != fac

        def _on_this_facility(frame: Any) -> bool:
            memory = _memory_for(frame)
            if memory is not None:
                return memory.get('facility') == fac
            from mm_assist.life_execute.ui_act import stack_has_facility_recipe

            if mode == "claim" and stack_has_facility_recipe():
                return False
            page = detect_facility_page(frame, expected=fac, root=root)
            # The process button is shared by every facility. A positively
            # identified different facility must go through navigation first.
            seen = str(page.get("facility") or "").strip()
            if seen and seen != fac:
                return False
            detect_this = bool(page.get("ok")) and str(page.get("facility") or "") == fac
            if mode == "claim":
                if _facility_identity(frame) == fac:
                    return True
                return detect_this
            if page.get("ok"):
                return True
            here = locate_process_button(frame, root=root)
            return here.get("at_facility") is True and str(here.get("label") or "") == "加工"

        def _prepare() -> ExecuteResult:
            from mm_assist.life_execute.ui_act import prepare_to

            return prepare_to(
                "facility",
                grab=_frame,
                click=_click,
                on_progress=_prog,
                root=root,
                facility=fac,
                title=f"{'领取' if mode == 'claim' else '投料'} {title}",
                already_ok=_on_this_facility,
                leave_wrong_page=_leave_wrong,
                read_stp=(lambda: 0) if not armed else None,
                sleep_fn=sleep_fn,
            )

        def _after_nav(nav: ExecuteResult) -> ExecuteResult | None:
            if not nav.ok:
                return nav
            frame = _frame(wait_memory=memory_required and mode == 'claim')
            cancelled = _cancelled()
            if cancelled is not None:
                return cancelled
            if memory_required and mode == 'claim' and not (_memory_for(frame) or {}).get('facility'):
                return ExecuteResult(False, 'facility_memory_pending', '设施内存未稳定，保留当前页，不点')
            if not _on_this_facility(frame):
                return ExecuteResult(ok=False, status="no_facility_page", message=f"没进{title}加工页")
            return None

        def _open_page() -> ExecuteResult | None:
            from mm_assist.life_execute.ui_act import stack_has_facility_result, stack_has_facility_recipe

            if stack_has_facility_result():
                dismissed = dismiss_facility_result(_click, _frame, on_progress=_prog, root=root)
                if dismissed is not None:
                    return dismissed
            frame = _frame()
            if mode == "claim" and stack_has_facility_recipe():
                return _after_nav(_prepare())
            if _on_this_facility(frame):
                _prog(f"已在{title}设施页")
                return None
            ident = _facility_identity(frame)
            if ident and ident != fac:
                return _after_nav(_prepare())
            if not ident:
                return _after_nav(_prepare())
            _prog(f"已在{title}设施页")
            return None

        opened = _open_page()
        if opened is not None:
            return opened

        if mode == "claim":
            frame = _frame(wait_memory=memory_required)
            cancelled = _cancelled()
            if cancelled is not None:
                return cancelled
            if not _on_this_facility(frame):
                return ExecuteResult(ok=False, status="facility_unknown", message="认不出本设施，不点")
            from mm_assist.life_execute.ui_act import stack_has_common_popup as _has_go_popup

            def _arrival_frame():
                # stp can settle while ToyTransparent/MMML is still switching.
                # Reuse the bounded readonly pairing gate after travel too.
                fresh = _frame(wait_memory=memory_required)
                cancelled = _cancelled()
                if cancelled is not None:
                    return fresh,cancelled
                memory = _memory_for(fresh)
                if memory_required and (not isinstance(memory,dict)
                        or memory.get('error') or not memory.get('facility')):
                    from mm_assist.life_execute.ui_logic import hud_page_pending, resolve_navigation

                    if hud_page_pending(resolve_navigation(fresh)):
                        return fresh,ExecuteResult(False,'input_layer_wait',
                            '前往設備的页面仍未就绪；只读等待，不重发前往或领取')
                    # A brief idle while travelling can precede renewed motion.
                    # Missing facility widgets then cannot prove an arrival
                    # failure. The pump retains the item and waits for newer
                    # idle before reacquiring the page, without repeating go.
                    if _travel_still_busy(read_stp):
                        cancelled = _cancelled()
                        if cancelled is not None:
                            return fresh,cancelled
                        return fresh,ExecuteResult(False,'stp_busy',
                            '仍在前往設備；只读等待，不重发前往或领取')
                    return fresh,ExecuteResult(False,'facility_memory_pending',
                        '到站后设施内存未稳定，保留当前页，不重发前往或领取')
                return fresh,None

            def _finish_claim(claim_hit: dict[str, Any]) -> ExecuteResult:
                if claim_hit.get('src') == 'memory':
                    # At the dispatch boundary a changed PID/window must stop,
                    # not settle onto another fact and reuse the earlier action.
                    fresh_frame = _frame()
                    cancelled = _cancelled()
                    if cancelled is not None:
                        return cancelled
                    fresh_memory = _memory_for(fresh_frame)
                    if fresh_memory is None:
                        return ExecuteResult(False, 'no_claim_ui', '领取前内存状态不可用，不点')
                    fresh_hit = _claim_button(fresh_frame)
                    if (not fresh_hit.get('ok')
                            or fresh_hit.get('src') != 'memory'
                            or fresh_memory.get('buttons', {}).get('go_equip', {}).get('present', True)):
                        return ExecuteResult(False, 'no_claim_ui', '领取前状态改变或未到设备，不点')
                    claim_hit = fresh_hit
                err = _one("全部領取", _client_of(claim_hit))
                if err:
                    return err
                had_popup = _has_go_popup()
                if had_popup:
                    go = confirm_go_facility_popup(_click, _frame, on_progress=_prog, root=root, present=True)
                    if go is not None:
                        return go
                    walked = _wait_walk_stop()
                    if walked is not None:
                        return walked
                from mm_assist.life_execute.ui_act import stack_has_facility_result

                # Observe the independent result layer; a dispatched click by
                # itself does not prove that the claim took effect.
                for _ in range(40):
                    cancelled = _cancelled()
                    if cancelled is not None:
                        return cancelled
                    if stack_has_facility_result():
                        break
                    sleeper(0.2)
                else:
                    return ExecuteResult(False, "claim_result_missing", "领取后未出现设施结果，不记完成")
                dismissed = dismiss_facility_result(_click, _frame, on_progress=_prog, root=root, present=True)
                if dismissed is not None:
                    return dismissed
                if defer_completion:
                    return ExecuteResult(True, "done", f"领取 {title}", action_issued=True)
                if _stp() == 1:
                    return ExecuteResult(ok=True, status="done", message=f"领取 {title}")
                settled = hold_facility_settle(
                    sleep_fn=sleep_fn,
                    read_stp=_stp,
                    on_progress=_prog,
                )
                if not settled.ok:
                    return settled
                return ExecuteResult(ok=True, status="done", message=f"领取 {title}")

            # The active travel popup takes priority over covered facility
            # buttons. Destination text does not choose a different response.
            if _has_go_popup():
                go = confirm_go_facility_popup(_click, _frame, on_progress=_prog, root=root, present=True)
                if go is not None:
                    return go
                walked = _wait_walk_stop()
                if walked is not None:
                    return walked
                frame,arrival_error = _arrival_frame()
                if arrival_error is not None:
                    return arrival_error
                if not _on_this_facility(frame):
                    reopened = _open_page()
                    if reopened is not None:
                        return reopened
                    frame,arrival_error = _arrival_frame()
                    if arrival_error is not None:
                        return arrival_error
            claim = _claim_button(frame)
            go_eq = _go_button(frame)
            # 看得见「前往設備」就必须先走到；远程点全部領取槽不清、材料不涨。
            if go_eq.get("ok"):
                err = _one("前往設備", _client_of(go_eq))
                if err:
                    return err
                go = _confirm_go_if_popup()
                if go is not None:
                    return go
                walked = _wait_walk_stop()
                if walked is not None:
                    return walked
                frame,arrival_error = _arrival_frame()
                if arrival_error is not None:
                    return arrival_error
                claim = _claim_button(frame)
                if not claim.get("ok"):
                    if _stp() == 1:
                        return ExecuteResult(ok=False, status="walk_timeout", message="仍在走，不点领取")
                    if not _on_this_facility(frame):
                        reopened = _open_page()
                        if reopened is not None:
                            return reopened
                        frame,arrival_error = _arrival_frame()
                        if arrival_error is not None:
                            return arrival_error
                        claim = _claim_button(frame)
                    if not claim.get("ok"):
                        return ExecuteResult(ok=False, status="no_claim_ui", message="走到后看不见全部領取，不点")
                return _finish_claim(claim)
            if claim.get("ok"):
                return _finish_claim(claim)
            return ExecuteResult(ok=False, status="no_claim_ui", message="看不见全部領取，不点")

        frame = _frame()
        from mm_assist.perception.process_sense import recipe_click_variant

        btn = locate_process_button(frame, root=root)
        if _recipe_detail_open(recipe, btn):
            _prog(f"已在{recipe}明细")
        else:
            recipe_var = recipe_click_variant(recipe)
            item = locate_process_recipe(
                frame, facility=fac, name=recipe, variant=recipe_var, root=root,
                **({'memory_view': _memory_for(frame) or {'error': 'memory_unavailable'}} if memory_required else {})
            )
            n = 0
            while not item.get("ok") and n < MAX_CLICKS:
                # Only a known recipe absent from a recognized facility list
                # authorizes paging. A new wrong/unknown page must stop here.
                if item.get("error") != "not_found":
                    break
                cancelled = _cancelled()
                if cancelled is not None:
                    return cancelled
                _prog("下翻配方")
                n += 1
                if win is not None and do_click is None:
                    from mm_assist.safety.input import drag_client

                    points = _recipe_list_down_points(frame, root=root)
                    if not drag_client(win, *points, end_hold_s=0.4):
                        return ExecuteResult(False, "drag_rejected", "翻页输入未发出，已停止")
                sleeper(0.5)
                cancelled = _cancelled()
                if cancelled is not None:
                    return cancelled
                frame = _frame()
                item = locate_process_recipe(
                    frame, facility=fac, name=recipe, variant=recipe_var, root=root,
                    **({'memory_view': _memory_for(frame) or {'error': 'memory_unavailable'}} if memory_required else {})
                )
            err = _one(f"点{recipe}", _client_of(item))
            if err:
                return err
            btn = {"ok": False}
            for _ in range(3):
                frame = _frame()
                btn = locate_process_button(frame, root=root)
                if btn.get("ok"):
                    break
                sleeper(0.8)
            if not btn.get("ok"):
                return ExecuteResult(ok=False, status="no_process_btn", message="没有加工钮")
        btn, stance_err = _confirm_process_btn()
        if stance_err is not None:
            return stance_err
        assert btn is not None
        mats_ok = btn.get("materials_ok")
        if _process_stance(btn) == "away":
            err = _one("前往加工", _client_of(btn))
            if err:
                return err
            go = _confirm_go_if_popup()
            if go is not None:
                return go
            deadline = operation_wall_time() + WAIT_WALK_S
            while operation_wall_time() < deadline:
                cancelled = _cancelled()
                if cancelled is not None:
                    return cancelled
                travel_stp = _stp()
                if armed and read_stp is None:
                    from mm_assist.life_plan.action_finish import read_completion_stp
                    sample = read_completion_stp()
                    travel_stp = sample.stp if sample.fresh else None
                if travel_stp != 0 and (armed or read_stp is not None):
                    _prog('等待移动或当前游戏动作')
                    sleeper(1.2)
                    continue
                seen = locate_process_button(_frame(), root=root)
                if _process_stance(seen) == "here":
                    confirmed, pending = _confirm_process_btn()
                    if pending is None and _process_stance(confirmed or {}) == 'here':
                        btn = confirmed
                        break
                sleeper(1.2)
            else:
                return ExecuteResult(ok=False, status="walk_timeout", message="走到设施超时")
            assert btn is not None
            if _process_stance(btn) == "away":
                return ExecuteResult(ok=False, status="still_away", message="仍是前往加工，停以免误投")
            if _process_stance(btn) != "here":
                return ExecuteResult(ok=False, status="btn_ambiguous", message="走到后加工钮语义不清，不点")
            mats_ok = btn.get("materials_ok")
        if _process_stance(btn) != "here":
            return ExecuteResult(ok=False, status="btn_ambiguous", message="加工钮不是在设施前的加工，不点")
        if mats_ok is False and not can_afford(recipe, items or {}):
            return ExecuteResult(ok=False, status="no_mats", message="设施前显示缺料")
        stopped = _wait_stop_then_process(btn)
        if stopped is not None:
            return stopped
        feed_btn, stance_err = _confirm_process_btn(for_feed=True)
        if stance_err is not None:
            return stance_err
        assert feed_btn is not None
        if _process_stance(feed_btn) != "here":
            return ExecuteResult(ok=False, status="btn_unstable", message="投料前加工钮已不是加工，不点")
        err = _one("加工", _client_of(feed_btn))
        if err:
            return err
        if memory_required:
            # The native click returning only proves dispatch. A retained recipe
            # can otherwise consume a local slot despite the game ignoring it.
            # read_active_facility rejects recipe/other covering players and
            # rechecks the current window association; reuse its bounded pair.
            # This checks page response, not material deductions or queue entry.
            try:
                response = _frame(wait_memory=True)
                cancelled = _cancelled()
                if cancelled is not None:
                    return ExecuteResult(False, 'cancel', cancelled.message, action_issued=True)
                fact = _memory_for(response)
                if (not isinstance(fact, dict) or fact.get('error') or fact.get('facility') != fac
                        or fact.get('buttons', {}).get('go_equip', {}).get('present') is not False):
                    return ExecuteResult(False, 'feed_page_unverified',
                        '加工已发出，未验明返回当前设施；保留动作，不补点', action_issued=True)
                page = detect_facility_page(response, expected=fac, root=root)
                if not page.get('ok') or page.get('facility') != fac:
                    return ExecuteResult(False, 'feed_page_unverified',
                        '加工已发出，当前设施画面未验明；保留动作，不补点', action_issued=True)
            except Exception as exc:
                return ExecuteResult(False, 'feed_page_unverified',
                    f'加工已发出，后置认页异常：{exc}；不补点', action_issued=True)
        if defer_completion:
            return ExecuteResult(True, "done", f"投料 {recipe}", action_issued=True)
        settled = hold_facility_settle(
            sleep_fn=sleep_fn,
            read_stp=_stp,
            on_progress=_prog,
        )
        if not settled.ok:
            return settled
        return ExecuteResult(ok=True, status="done", message=f"投料 {recipe}")
    except Exception as exc:
        return ExecuteResult(ok=False, status="error", message=str(exc))
    finally:
        if armed:
            from mm_assist.safety.input import release_mouse_buttons
            from mm_assist.safety.lease import production_mode

            production_mode(False)
            try:
                release_mouse_buttons()
            except Exception:
                pass
