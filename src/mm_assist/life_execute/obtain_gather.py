"""樹液经木材加工的柔軟木材配方打开獲得方法。"""

from __future__ import annotations

import time
from typing import Any, Callable

from mm_assist.life_execute.gather_wait import HOLD_IDLE_S, WAIT_GATHER_S, wait_gather_idle
from mm_assist.life_plan.types import ExecuteResult
from mm_assist.perception import sap_sense

OBTAIN_MATS = frozenset({'樹液'})


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
    from mm_assist.life_execute.ui_act import click_spot, prepare_to, pump_aborted
    from mm_assist.life_execute.ui_logic import resolve_navigation

    name = str(mat or '').strip()
    if name not in OBTAIN_MATS:
        return ExecuteResult(False, 'bad_mat', '不是獲得方法采集材料')
    if grab is None and win is None:
        return ExecuteResult(False, 'need_context', '獲得方法采集需要游戏窗')

    def click(x: int, y: int) -> bool:
        if do_click is not None:
            return bool(do_click(x, y))
        from mm_assist.safety.input import click_client

        return bool(click_client(win, x, y)) if win is not None else False

    def frame_now() -> Any:
        if grab is not None:
            return grab()
        from mm_assist.platform.capture import grab_client

        return grab_client(win)

    def wait_end() -> ExecuteResult:
        if kwargs.get('defer_completion'):
            return ExecuteResult(False, 'action_pending', '伐木已发出，等待主界面待机', action_issued=True)
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
            message_ok=f'{name} 主界面待机',
            already_started=bool(already_started) if observe_only else False,
            issued=True,
            root=root,
            grab=frame_now if observe is None and read_stp is None and read_page is None else None,
        )

    issued = bool(observe_only)

    def action_issued() -> None:
        nonlocal issued
        issued = True
        callback = kwargs.get('on_action_issued')
        if callback is not None:
            callback()

    armed = do_click is None
    if armed:
        from mm_assist.safety.lease import prefer_visible_clicks, production_mode

        production_mode(True)
        prefer_visible_clicks(True)
    try:
        if observe_only:
            return wait_end()
        for target, item_name in (('facility_recipe', sap_sense.RECIPE), ('obtain_method', name)):
            nav = prepare_to(target, grab=frame_now, click=click, on_progress=on_progress,
                             root=root, facility='wood', item_name=item_name,
                             read_stp=read_stp or ((lambda: 0) if not armed else None),
                             sleep_fn=sleep_fn)
            if not nav.ok:
                return nav
        frame = frame_now()
        where = resolve_navigation(frame)
        sampled = where.stack_sampled_ts
        if (not where.stack_ready or not sampled or not 0 <= time.time() - sampled <= 3
                or where.node != sap_sense.OBTAIN_PAGE):
            return ExecuteResult(False, 'need_verify', '未确认樹液的獲得方法页，暂不伐木')
        xy = sap_sense.locate(frame, 'woodcut', root=root)
        if xy is None:
            return ExecuteResult(False, 'sap_woodcut_missing', '找不到樹液獲得方法的第一行伐木')
        err = click_spot(click, xy, '伐木', on_progress=on_progress, wait_refresh=False,
                         root=root, on_dispatch=action_issued)
        return err if err is not None else wait_end()
    except Exception as exc:
        return ExecuteResult(False, 'error', str(exc), action_issued=issued)
    finally:
        if armed:
            from mm_assist.safety.input import release_mouse_buttons
            from mm_assist.safety.lease import production_mode

            production_mode(False)
            try:
                release_mouse_buttons()
            except Exception:
                pass
