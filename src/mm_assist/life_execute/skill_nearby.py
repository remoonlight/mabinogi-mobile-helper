"""生活技能格 → 详情物 → 寻找附近。流程结束=曾 stp=1 后新鲜主界面明确待机。"""

from __future__ import annotations

from mm_assist.life_plan.game_stop import operation_wall_time

import time
from pathlib import Path
from typing import Any, Callable

from mm_assist.life_execute.gather_wait import HOLD_IDLE_S, WAIT_GATHER_S, wait_gather_idle
from mm_assist.life_plan.types import ExecuteResult

_SKILL_CELL = {
    "daily": Path("knowledge") / "templates" / "life_grid" / "daily.png",
    "herb": Path("knowledge") / "templates" / "life_grid" / "herb.png",
    "woodcut": Path("knowledge") / "templates" / "life_grid" / "woodcut.png",
    "mine": Path("knowledge") / "templates" / "life_grid" / "mine.png",
    "shear": Path("knowledge") / "templates" / "life_grid" / "shear.png",
    "harvest": Path("knowledge") / "templates" / "life_grid" / "harvest.png",
    "hoe": Path("knowledge") / "templates" / "life_grid" / "hoe.png",
    "insect": Path("knowledge") / "templates" / "life_grid" / "insect.png",
    "fish": Path("knowledge") / "templates" / "life_grid" / "fish.png",
}


def _client_of(hit: Any) -> tuple[int, int] | None:
    from mm_assist.life_execute.ui_logic import recognized_click_xy

    return recognized_click_xy(hit)


def gather_resume_layer(zh: str, node: str) -> str:
    """栏上已在采集链哪一层。detail/grid/profile 禁止先清回主界面。"""
    from mm_assist.life_execute.ui_logic import graph_node

    text = str(zh or "")
    g = graph_node(str(node or ""))
    if "技能详情" in text or g == "gather_detail":
        return "detail"
    if "生活技能" in text or g in {"life_grid", "gather_skill"}:
        return "grid"
    if "角色" in text or g == "profile":
        return "profile"
    if "主界面-待机" in text or g == "main_hud":
        return "hud"
    return ""


def execute_skill_nearby(
    cfg: Any = None,
    win: Any = None,
    *,
    skill_id: str,
    item_name: str,
    output: str,
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
    root: Path | None = None,
    **kwargs: Any,
) -> ExecuteResult:
    """已在技能详情则直接寻找附近；已在角色/生活技能则接着开，不先点退出。"""
    _ = (cfg, kwargs)
    sid = str(skill_id or "").strip()
    detail = str(item_name or "").strip()
    out_name = str(output or "").strip()
    have = dict(items or {})
    cell = _SKILL_CELL.get(sid)
    if not sid or not detail or not out_name or cell is None:
        return ExecuteResult(ok=False, status="bad_skill", message="未知采集技能")

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

    def _bag() -> dict[str, int]:
        if read_items is not None:
            return dict(read_items() or {})
        return have

    if grab is None and win is None:
        return ExecuteResult(ok=False, status="need_context", message="采集需要游戏窗")

    armed = do_click is None
    if armed:
        from mm_assist.safety.lease import prefer_visible_clicks, production_mode

        production_mode(True)
        prefer_visible_clicks(True)
    try:
        from mm_assist.life_execute.ui_act import click_spot, prepare_to
        from mm_assist.life_execute.ui_logic import graph_node, read_inspect_record
        from mm_assist.perception.item_sense import (
            locate_find_nearby,
            locate_skill_item,
        )
        from mm_assist.platform.window import is_foreground
        from mm_assist.safety.input import foreground_input_enabled

        if armed and win is not None and not foreground_input_enabled(win.hwnd) and not is_foreground(win.hwnd):
            return ExecuteResult(
                ok=False,
                status="not_foreground",
                message="游戏窗口未前台：请手动切到游戏后再测",
            )

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
                message_ok=f"{out_name} 主界面待机",
                already_started=bool(already_started) if observe_only else False,
                issued=True,
                root=root,
                grab=_frame if observe is None and read_stp is None and read_page is None else None,
            )

        if observe_only:
            return _wait_end()

        def _one(stage: str, pt: tuple[int, int] | None) -> ExecuteResult | None:
            err = click_spot(_click, pt, stage, on_progress=_prog,
                on_dispatch=kwargs.get("on_action_issued") if stage == "寻找附近位置" else None)
            if err is not None:
                return err
            return None

        rec = read_inspect_record(root=root)
        node_s = str((rec or {}).get("node") or "")
        insp = graph_node(node_s)
        zh = str((rec or {}).get("zh") or "").strip()
        layer = gather_resume_layer(zh, node_s or insp)

        def _wanted_skill_list() -> bool:
            if insp != "gather_skill":
                return False
            suffix = node_s.split(":", 1)[1] if ":" in node_s else ""
            return suffix == sid
        if "动作中" in zh and layer in {"", "hud"}:
            return ExecuteResult(
                ok=False,
                status="stp_busy",
                message="主界面动作中，不点、不换条",
            )
        if not layer:
            if insp == "main_hud" and _stp() == 1:
                return ExecuteResult(
                    ok=False,
                    status="stp_busy",
                    message="主界面动作中，不点、不换条",
                )
            if insp in {"", "unknown"}:
                return ExecuteResult(
                    ok=False,
                    status="unknown_ui",
                    message="认不出界面，不点、不换条",
                )
            layer = "resume"

        # 已在对应技能列表则不再进退页；错技能列表仍走 prepare。禁止先退主界面。
        if layer != "detail" and not _wanted_skill_list():
            nav = prepare_to(
                "gather_skill",
                grab=_frame,
                click=_click,
                on_progress=_prog,
                root=root,
                skill_id=sid,
                title=f"采集 {out_name}",
                read_stp=_stp,
            )
            if not nav.ok:
                return nav

        captured_at = time.time()
        frame = _frame()
        from mm_assist.life_plan.node_inspect import gather_detail_for_target, gather_detail_images_only
        from mm_assist.perception.item_sense import fold_zh
        images_only = gather_detail_images_only(detail)
        detail_node = gather_detail_for_target(frame, detail) if layer == 'detail' else ''
        if layer == 'detail' and fold_zh(detail_node) != fold_zh(f'gather_detail:{detail}'):
            return ExecuteResult(False, 'wrong_gather_detail',
                                 f'当前详情不是{detail}，不点被遮挡的列表或尋找附近')
        # The newly captured popup title has already proved this target.
        # A coarse/stale display label must not send input to its covered list.
        need_item = layer != "detail"
        item: dict[str, Any] = {"ok": False}
        if need_item:
            _prog(f"定位 {detail}")
            t_loc = time.monotonic()
            item = locate_skill_item(frame, skill_id=sid, name=detail, root=root)
            _prog(f"定位 {detail} {time.monotonic() - t_loc:.2f}s")
            from mm_assist.life_execute import skill_list
            from mm_assist.life_execute.ui_act import pump_aborted
            from mm_assist.life_plan.controller import get_controller
            owner = get_controller()
            def list_cancelled():
                if pump_aborted():
                    return True
                current = get_controller()
                return bool(armed and owner.running and (not current.running
                    or current.life_started_at != owner.life_started_at or current.current != owner.current))
            def drag(x,y0,xx,y1):
                if kwargs.get('do_drag') is not None:
                    return kwargs['do_drag'](x,y0,xx,y1,end_hold_s=.4)
                from mm_assist.safety.input import drag_client
                return drag_client(win,x,y0,xx,y1,end_hold_s=.4)
            resolver = kwargs.get('resolve')
            if resolver is None and armed:
                from mm_assist.life_execute.ui_logic import resolve_navigation
                resolver = resolve_navigation
            def validate(image, captured_at):
                where = resolver(image)
                return (getattr(where,'stack_ready',False)
                    and str(where.node) == 'gather_skill:'+sid
                    and captured_at < float(getattr(where,'stack_sampled_ts',0) or 0) <= time.time()+.05)
            frame, item, error = skill_list.seek_item(frame,item,skill_id=sid,name=detail,
                grab=_frame,drag=drag if kwargs.get('do_drag') is not None or (armed and win is not None) else None,
                validate=validate if resolver is not None else None,cancelled=list_cancelled,
                captured_at=captured_at,sleep=time.sleep,on_scroll=lambda:_prog('下翻详情物'),root=root)
            if error is not None:
                return error
            err = _one(f"点{detail}", _client_of(item))
            if err:
                return err
            frame = _frame()
            detail_node = ''
        nearby = locate_find_nearby(frame, root=root, allow_ocr=not images_only)
        if not nearby.get("ok"):
            from mm_assist.life_execute.ui_logic import skip_wait_ui_params

            if not skip_wait_ui_params():
                until = operation_wall_time() + 2.0
                while operation_wall_time() < until:
                    time.sleep(0.25)
                    frame = _frame()
                    detail_node = ''  # A new capture needs its own title, even if the buffer is reused.
                    nearby = locate_find_nearby(frame, root=root, allow_ocr=not images_only)
                    if nearby.get("ok"):
                        break
        actual = detail_node or gather_detail_for_target(frame, detail)
        expected = f'gather_detail:{detail}'
        if fold_zh(actual) != fold_zh(expected):
            return ExecuteResult(False, 'wrong_gather_detail',
                                 f'详情目标不符或未认出：{actual or "未知"}；需要{detail}')
        err = _one("寻找附近位置", _client_of(nearby))
        if err:
            return err
        return _wait_end()
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
