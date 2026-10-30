"""独立 Esc 脱困：每次一次，等待两秒后重新认页，不复用失败导航。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

MAX_RECOVERIES = 2
ESC_INTERVAL_S = 2.0


def recover_once(
    trigger: str,
    *,
    win: Any = None,
    grab: Callable[[], Any] | None = None,
    click: Callable[[int, int], bool] | None = None,
    root: Path | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    press_esc: Callable[..., None] | None = None,
    resolve: Callable[..., Any] | None = None,
    cancelled: Callable[[], bool] | None = None,
    after_input_at: float = 0.0,
    on_dispatch: Callable[[float], None] | None = None,
) -> dict:
    """未到主界面才按 Esc；成功须新帧、输入后的就绪栈及独立 HUD。"""
    del trigger, click  # 保留旧调用签名；脱困不点击确认、加工标签或业务按钮。
    from mm_assist.life_execute.ui_logic import graph_node, resolve_navigation
    from mm_assist.life_execute.ui_nav import _hud_xy, _verify_late_edges
    from mm_assist.memory.readers.ui_stack import input_blocking_layers
    from mm_assist.safety.input import check_abort, press_escape, release_mouse_buttons
    from mm_assist.safety.lease import in_production_mode, in_test_mode, production_mode
    from mm_assist.tool_repair import op_cancelled

    if grab is None:
        if win is None:
            return {'ok': False, 'action': 'press_escape', 'status': 'need_context', 'where_node': ''}
        from mm_assist.platform.capture import grab_client
        grab = lambda: grab_client(win)
    if press_esc is None and win is None:
        return {'ok': False, 'action': 'press_escape', 'status': 'need_context', 'where_node': ''}
    resolver = resolve or resolve_navigation
    sleeper = sleep_fn or time.sleep
    own_production = win is not None and not in_test_mode() and not in_production_mode()
    sent_at = after_input_at
    esc_sent = False

    def stopped():
        check_abort()
        return op_cancelled() or (cancelled is not None and cancelled())

    def fresh(where, *, after=0.0):
        sampled = float(getattr(where, 'stack_sampled_ts', 0) or 0)
        return (where.stack_ready and after < sampled <= time.time()
                and time.time() - sampled <= 16.0
                and not input_blocking_layers(where.stack_names))

    def hud(where, frame):
        return (graph_node(where.node) == 'main_hud' and where.menu == 0 and where.dlg == 0
                and _hud_xy(frame, 'menu', root=root) is not None)

    def idle(where, *, after=0.0):
        # Direct navigation identifies the page/stack and intentionally has no
        # action bit. Read that fact separately; None never means idle.
        if where.stp is not None:
            return where.stp
        from mm_assist.life_plan.action_finish import read_completion_stp
        sample = read_completion_stp()
        return sample.stp if sample.fresh and sample.timestamp > after else None

    def result(ok, status, node='', *, waiting=False):
        return {'ok': ok, 'action': 'press_escape', 'status': status, 'where_node': node,
                'esc_sent': esc_sent, 'sent_at': sent_at, 'waiting': waiting}

    def wait_refresh():
        elapsed = 0.0
        while elapsed < ESC_INTERVAL_S:
            if stopped():
                return False
            step = min(0.1, ESC_INTERVAL_S - elapsed)
            sleeper(step)
            elapsed += step
        return not stopped()

    if own_production:
        production_mode(True)
    try:
        if stopped():
            return result(False, 'cancel')
        if sent_at and time.time() - sent_at < ESC_INTERVAL_S and not wait_refresh():
            return result(False, 'cancel')
        frame_started = time.time()
        frame = grab()
        where = resolver(frame)
        if stopped():
            return result(False, 'cancel')
        stp = idle(where, after=sent_at)
        if not fresh(where, after=sent_at) or not where.known or stp != 0:
            return result(False, 'stp_busy' if stp == 1 else 'page_not_ready',
                          where.node, waiting=True)
        _verify_late_edges(where, frame_started_at=frame_started)
        if hud(where, frame):
            return result(True, 'at_main_hud', where.node)
        if graph_node(where.node) == 'main_hud':
            return result(False, 'page_not_ready', where.node, waiting=True)
        if sent_at:
            # Resume observing an already sent Esc; never send it again just
            # because its first post-input frame was late/unready.
            return result(False, 'escape_unverified', where.node)
        if press_esc is not None:
            press_esc()
        else:
            from mm_assist.life_plan.live_test import inspect_nav_watch_ready
            _, refuse = inspect_nav_watch_ready(root=root)
            if refuse:
                return result(False, f'click_watch_not_ready: {refuse}', where.node)
            if stopped():
                return result(False, 'cancel', where.node)
            press_escape(win=win)
        sent_at = time.time()
        esc_sent = True
        if on_dispatch is not None:
            on_dispatch(sent_at)
        # 短段等待，暂停/F8/换代不等到两秒结束才响应。
        if not wait_refresh():
            return result(False, 'cancel', where.node)
        frame_started = time.time()
        frame = grab()
        after = resolver(frame)
        if stopped():
            return result(False, 'cancel', after.node)
        verified = fresh(after, after=sent_at)
        if verified:
            _verify_late_edges(after, frame_started_at=frame_started)
        stp = idle(after, after=sent_at)
        ready = bool(verified and after.known and stp == 0
                     and (graph_node(after.node) != 'main_hud' or hud(after, frame)))
        ok = bool(ready and hud(after, frame))
        return result(ok, 'at_main_hud' if ok else 'escape_unverified', after.node,
                      waiting=not ready)
    finally:
        if own_production:
            production_mode(False)
        release_mouse_buttons()
