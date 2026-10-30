"""树液：物品卡 → 获得方法 → 第一行采集点 → 寻找附近。煤炭走技能寻找附近。"""

from __future__ import annotations

from mm_assist.life_plan.game_stop import operation_wall_time

import time
from typing import Any, Callable

from mm_assist.life_execute.gather_wait import HOLD_IDLE_S, WAIT_GATHER_S, wait_gather_idle
from mm_assist.life_plan.types import ExecuteResult

OBTAIN_MATS = frozenset({"樹液"})
_HOME = {"煤炭": ("metal", "合金鋼錠"), "樹液": ("wood", "高級木材")}
MAX_HOPS = 8
_sap_visits = 0


def reset_sap_visits() -> None:
    global _sap_visits
    _sap_visits = 0


def _next_spot(node: str, mat: str) -> str:
    key = str(node or "")
    if key.startswith("obtain_method:") and mat in key:
        from mm_assist.life_plan.node_inspect import spots_for_node

        spots = spots_for_node(key)
        return str(spots[0]) if spots else ""
    if key.startswith("item_info:") and mat in key:
        return "获得方法"
    if mat == "樹液" and (
        key == "fine:facility_detail"
        or key.startswith("facility:wood:高級木材")
    ):
        return "樹液"
    if mat == "樹液" and (
        key == "facility:wood"
        or (key.startswith("facility:wood:") and "高級木材" not in key)
    ):
        return "高級木材"
    if mat == "樹液" and key == "inventory":
        return "樹液"
    if mat == "樹液" and key.startswith("facility:") and not key.startswith("facility:wood"):
        return ""
    if key in {"inventory", "facility"} or key.startswith("facility:"):
        return mat
    return ""


def execute_obtain_gather(
    cfg: Any = None,
    win: Any = None,
    *,
    mat: str,
    items: dict[str, int] | None = None,
    do_click: Callable[[int, int], bool] | None = None,
    read_items: Callable[[], dict[str, int] | None] | None = None,
    grab: Callable[[], Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
    read_stp: Callable[[], int | None] | None = None,
    read_page: Callable[[], Any] | None = None,
    observe: Callable[[], Any] | None = None,
    now_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    hold_s: float | None = None,
    wait_s: float | None = None,
    observe_only: bool = False,
    already_started: bool = False,
    root: Any = None,
    **kwargs: Any,
) -> ExecuteResult:
    _ = (cfg, kwargs)
    name = str(mat or "").strip()
    if name not in OBTAIN_MATS:
        return ExecuteResult(ok=False, status="bad_mat", message="不是获得方法链材料")
    fac, _recipe = _HOME[name]
    have = dict(items or {})

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

    def _frame() -> Any:
        if grab is not None:
            return grab()
        if win is None:
            return None
        from mm_assist.platform.capture import grab_client

        return grab_client(win)

    if grab is None and win is None:
        return ExecuteResult(ok=False, status="need_context", message="获得方法采集需要游戏窗")

    armed = do_click is None
    if armed:
        from mm_assist.safety.lease import prefer_visible_clicks, production_mode

        production_mode(True)
        prefer_visible_clicks(True)
    try:
        from mm_assist.life_execute.ui_act import click_spot, format_act, prepare_to
        from mm_assist.life_execute.ui_logic import gather_detail_visible, resolve_ui_node
        from mm_assist.life_plan.node_inspect import compute_inspect, locate_spot
        from mm_assist.perception.item_sense import locate_find_nearby

        def _one(label: str, xy: tuple[int, int] | None) -> ExecuteResult | None:
            err = click_spot(_click, xy, label, on_progress=_prog,
                on_dispatch=kwargs.get("on_action_issued") if label == "寻找附近位置" else None)
            if err is not None:
                return err
            return None

        def _stp() -> int | None:
            if read_stp is not None:
                return read_stp()
            from mm_assist.perception.state_ui import read_state_ui

            view = read_state_ui()
            if view is None:
                return None
            raw = getattr(view, "raw", None)
            payload = raw if isinstance(raw, dict) else {}
            if payload.get("stp") is None:
                return None
            try:
                return int(payload.get("stp"))
            except (TypeError, ValueError):
                return None

        def _wait_end() -> ExecuteResult:
            from mm_assist.life_execute.ui_act import pump_aborted

            if kwargs.get("defer_completion"):
                return ExecuteResult(False, "action_pending", "动作已发出，等待连续两段2秒 stp=0", action_issued=True)
            return wait_gather_idle(
                observe=observe,
                read_stp=read_stp if observe is None else None,
                read_page=read_page if observe is None else None,
                now_fn=now_fn,
                sleep_fn=sleep_fn,
                cancelled=pump_aborted,
                wait_s=WAIT_GATHER_S if wait_s is None else wait_s,
                hold_s=HOLD_IDLE_S if hold_s is None else hold_s,
                read_items=read_items,
                message_ok=f"{name} 主界面待机",
                already_started=bool(already_started) if observe_only else False,
                issued=True,
                root=root,
                grab=_frame if observe is None and read_stp is None and read_page is None else None,
            )

        if observe_only:
            return _wait_end()

        def _wait_bag_gain() -> ExecuteResult:
            return _wait_end()

        def _finish_nearby(frame: Any, nearby: dict[str, Any] | None = None) -> ExecuteResult:
            from mm_assist.life_execute.ui_act import pump_aborted
            from mm_assist.life_execute.ui_logic import recognized_click_xy, skip_wait_ui_params

            if nearby is None:
                nearby = locate_find_nearby(frame, root=root)
            if not nearby.get("ok"):
                if skip_wait_ui_params():
                    return ExecuteResult(ok=False, status="no_nearby", message="看不见寻找附近，不点")
                deadline_ui = operation_wall_time() + WAIT_GATHER_S
                while operation_wall_time() < deadline_ui:
                    if pump_aborted():
                        return ExecuteResult(ok=False, status="cancel", message="已停止操作")
                    time.sleep(0.4)
                    frame = _frame()
                    nearby = locate_find_nearby(frame, root=root)
                    if nearby.get("ok"):
                        break
            err = _one("寻找附近位置", recognized_click_xy(nearby))
            if err:
                return err
            return _wait_end()

        opened = False
        sap_chain = False
        switched_bag_props = False

        def _out(res: ExecuteResult) -> ExecuteResult:
            global _sap_visits
            if name == "樹液" and (res.ok or sap_chain):
                _sap_visits += 1
            return res

        for _ in range(MAX_HOPS):
            frame = _frame()
            where = resolve_ui_node(frame=frame, root=root)
            info = compute_inspect(frame)
            node = str(info.get("node") or "")
            _prog(format_act(f"采集 {name}", where, ""))
            if name == "樹液" and node.startswith("obtain_method:") and name in node:
                sap_chain = True
            if gather_detail_visible(frame, root=root):
                return _out(_finish_nearby(frame))
            nearby = locate_find_nearby(frame, root=root)
            if nearby.get("ok"):
                return _out(_finish_nearby(frame, nearby))
            spot = _next_spot(node, name)
            if not spot:
                if opened:
                    return _out(
                        ExecuteResult(
                            ok=False,
                            status="obtain_stuck",
                            message=f"{node} 上没有{name}可点",
                        )
                    )
                if name == "樹液" and _sap_visits > 0:
                    nav = prepare_to(
                        "inventory",
                        grab=_frame,
                        click=_click,
                        on_progress=_prog,
                        root=root,
                        title=f"采集 {name}",
                        read_stp=(lambda: 0) if not armed else None,
                    )
                else:
                    nav = prepare_to(
                        "facility",
                        grab=_frame,
                        click=_click,
                        on_progress=_prog,
                        root=root,
                        facility=fac,
                        title=f"采集 {name}",
                        read_stp=(lambda: 0) if not armed else None,
                    )
                if not nav.ok:
                    return _out(nav)
                opened = True
                continue
            xy = locate_spot(node, spot, frame)
            if name == "樹液" and node == "inventory" and xy is None:
                # The second visit uses 物品 -> 道具. A historical visit count
                # does not prove the current bag tab or sap card is visible.
                if not switched_bag_props:
                    from mm_assist.perception.inventory_sense import props_tab_selected

                    if not props_tab_selected(frame, root=root):
                        err = _one("道具", locate_spot(node, "道具", frame))
                        if err:
                            return _out(err)
                    switched_bag_props = True
                from mm_assist.life_execute.ui_act import pump_aborted

                for _ in range(10):
                    if pump_aborted():
                        return _out(ExecuteResult(False, "cancel", "已停止操作"))
                    (sleep_fn or time.sleep)(0.2)
                    frame = _frame()
                    if str(compute_inspect(frame).get("node") or "") != "inventory":
                        return _out(ExecuteResult(False, "bag_page_changed", "道具切页后已离开物品，不点"))
                    xy = locate_spot("inventory", name, frame)
                    if xy is not None:
                        break
                if xy is None:
                    return _out(ExecuteResult(False, "sap_card_missing", "道具页未看见樹液，不点"))
            err = _one(spot, xy)
            if err:
                return _out(err)
            if name == "樹液" and spot == "获得方法":
                sap_chain = True
            if name == "樹液" and spot == "伐木":
                return _out(_wait_bag_gain())
        return _out(
            ExecuteResult(ok=False, status="obtain_timeout", message=f"未走完{name}获得方法")
        )
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
