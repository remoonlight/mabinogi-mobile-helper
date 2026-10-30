"""默认关闭的方法 A 单击基准。不改通用 open_test_job。"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from mm_assist.safety.lease import (
    ACTION_CLICK,
    ACTION_DRAG,
    GuardError,
    NotAuthorized,
    assert_protocol_ready,
    begin_authorization,
    enter_test_mode,
    reset_guard_state,
    revoke,
    wait_for_user_start,
)
from mm_assist.safety.readonly import (
    ReadOnlyProfileError,
    live_benchmark_env_enabled,
    live_benchmark_input_scope,
)
from mm_assist.jobs.bar import require_state_watch_running
from mm_assist.platform.window import WindowInfo, is_foreground

log = logging.getLogger(__name__)

MAX_LEASE_S = 60.0
MAX_STAGE_FAIL = 2
LIVE_VIEW_SETTLE_S = 1.0
ALLOWED_CLICK_METHODS = frozenset({"sendinput", "si", "unity_hook", "hook"})
ALLOWED_CLICK_METHODS = frozenset({"sendinput", "si", "unity_hook", "hook"})
ALLOWED_DRAG_METHODS = ALLOWED_CLICK_METHODS | frozenset(
    {"postmessage", "post", "background"}
)


class LiveBenchmarkError(RuntimeError):
    pass


def prepare_live_view(
    win: WindowInfo, *, settle_s: float = LIVE_VIEW_SETTLE_S
) -> None:
    """截图或键鼠前：先等 settle_s，再强制游戏前台，立刻看/点。"""
    from mm_assist.platform.window import bring_to_foreground

    hwnd = int(win.hwnd)
    time.sleep(max(0.0, float(settle_s)))
    bring_to_foreground(hwnd, force=True)
    if not is_foreground(hwnd):
        raise NotAuthorized("游戏窗口未前台：前置并等待后仍非前台")


def grab_live_client(win: WindowInfo, *, settle_s: float = LIVE_VIEW_SETTLE_S):
    """LIVE 采帧：先前置游戏、等 settle_s，再截客户区。"""
    from mm_assist.platform.capture import grab_client

    prepare_live_view(win, settle_s=settle_s)
    return grab_client(win)


def assert_live_benchmark_ready(*, hwnd: int, root=None) -> None:
    if not live_benchmark_env_enabled():
        raise ReadOnlyProfileError("FAIL-CLOSED: 未设 MM_LIVE_BENCHMARK=1")
    assert_protocol_ready()
    require_state_watch_running(root=root)
    if int(hwnd) <= 0:
        raise LiveBenchmarkError("hwnd 无效")


def _guard_method(method: str, *, for_drag: bool = False) -> str:
    m = str(method or "sendinput").strip().lower()
    if m in ("postmessage", "post", "background", "sendmessage", "sendmsg"):
        if for_drag and m in ("postmessage", "post", "background"):
            return m
        raise GuardError("测试模式禁止 postmessage/sendmessage，仅允许 sendinput/unity_hook")
    allowed = ALLOWED_DRAG_METHODS if for_drag else ALLOWED_CLICK_METHODS
    if m not in allowed:
        raise GuardError(f"未知 click method={method!r}")
    return m


def run_one_click(
    win: WindowInfo,
    cx: int,
    cy: int,
    *,
    method: str = "sendinput",
    need_ui: str,
    stage: str,
    fail_count: int = 0,
    lease_s: float = MAX_LEASE_S,
    auto_arm: bool = True,
) -> dict[str, Any]:
    """一次租约、一次点击。同 stage 失败达 2 次则拒绝第三次。"""
    if fail_count >= MAX_STAGE_FAIL:
        raise LiveBenchmarkError(
            f"stage={stage!r} 已失败 {fail_count} 次；禁止第三次盲试"
        )
    if not str(need_ui or "").strip():
        raise LiveBenchmarkError("need_ui 必填")
    lease_s = min(float(lease_s), MAX_LEASE_S)
    m = _guard_method(method)
    assert_live_benchmark_ready(hwnd=int(win.hwnd))
    prepare_live_view(win)

    tid = f"benchmark_click:{stage}:{uuid.uuid4().hex[:12]}"
    enter_test_mode()
    begin_authorization(
        hwnd=int(win.hwnd),
        intent=f"live_benchmark:{stage}:{need_ui}",
        actions=[ACTION_CLICK],
        lease_s=lease_s,
    )
    wait_for_user_start(timeout_s=0.0 if auto_arm else None, auto_start_on_timeout=auto_arm)
    t0 = time.perf_counter()
    clicked = False
    err: str | None = None
    try:
        from mm_assist.safety.input import click_client, release_mouse_buttons

        with live_benchmark_input_scope():
            clicked = bool(
                click_client(win, int(cx), int(cy), method=m, transition_id=tid)
            )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            from mm_assist.safety.input import release_mouse_buttons

            release_mouse_buttons()
        except Exception:
            pass
        revoke("live_benchmark_done")
        reset_guard_state()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "ok": bool(clicked) and err is None,
        "clicked": clicked,
        "method": m,
        "transition_id": tid,
        "elapsed_ms": elapsed_ms,
        "need_ui": need_ui,
        "stage": stage,
        "error": err,
    }


def run_one_drag(
    win: WindowInfo,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    method: str = "sendinput",
    need_ui: str,
    stage: str,
    fail_count: int = 0,
    lease_s: float = MAX_LEASE_S,
    auto_arm: bool = True,
) -> dict[str, Any]:
    """一次租约、一次列表拖动（右栏向下滚）。"""
    if fail_count >= MAX_STAGE_FAIL:
        raise LiveBenchmarkError(
            f"stage={stage!r} 已失败 {fail_count} 次；禁止第三次盲试"
        )
    if not str(need_ui or "").strip():
        raise LiveBenchmarkError("need_ui 必填")
    lease_s = min(float(lease_s), MAX_LEASE_S)
    m = _guard_method(method, for_drag=True)
    assert_live_benchmark_ready(hwnd=int(win.hwnd))
    prepare_live_view(win)

    tid = f"benchmark_drag:{stage}:{uuid.uuid4().hex[:12]}"
    enter_test_mode()
    begin_authorization(
        hwnd=int(win.hwnd),
        intent=f"live_benchmark:{stage}:{need_ui}",
        actions=[ACTION_DRAG],
        lease_s=lease_s,
    )
    wait_for_user_start(timeout_s=0.0 if auto_arm else None, auto_start_on_timeout=auto_arm)
    t0 = time.perf_counter()
    dragged = False
    err: str | None = None
    try:
        from mm_assist.safety.input import drag_client, release_mouse_buttons

        with live_benchmark_input_scope():
            dragged = bool(
                drag_client(
                    win,
                    int(x0),
                    int(y0),
                    int(x1),
                    int(y1),
                    method=m,
                    transition_id=tid,
                )
            )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            from mm_assist.safety.input import release_mouse_buttons

            release_mouse_buttons()
        except Exception:
            pass
        revoke("live_benchmark_done")
        reset_guard_state()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "ok": bool(dragged) and err is None,
        "dragged": dragged,
        "method": m,
        "transition_id": tid,
        "elapsed_ms": elapsed_ms,
        "need_ui": need_ui,
        "stage": stage,
        "error": err,
    }


def run_one_wheel(
    win: WindowInfo,
    cx: int,
    cy: int,
    *,
    method: str = "sendinput",
    need_ui: str,
    stage: str,
    delta: int = -120,
    notches: int = 1,
    fail_count: int = 0,
    lease_s: float = MAX_LEASE_S,
    auto_arm: bool = True,
) -> dict[str, Any]:
    """一次租约、滚轮滚动列表（path=item_list_scroll）。"""
    if fail_count >= MAX_STAGE_FAIL:
        raise LiveBenchmarkError(
            f"stage={stage!r} 已失败 {fail_count} 次；禁止第三次盲试"
        )
    if not str(need_ui or "").strip():
        raise LiveBenchmarkError("need_ui 必填")
    lease_s = min(float(lease_s), MAX_LEASE_S)
    m = _guard_method(method, for_drag=True)
    assert_live_benchmark_ready(hwnd=int(win.hwnd))
    prepare_live_view(win)

    tid = f"benchmark_wheel:{stage}:{uuid.uuid4().hex[:12]}"
    enter_test_mode()
    begin_authorization(
        hwnd=int(win.hwnd),
        intent=f"live_benchmark:{stage}:{need_ui}",
        actions=[ACTION_DRAG],
        lease_s=lease_s,
    )
    wait_for_user_start(timeout_s=0.0 if auto_arm else None, auto_start_on_timeout=auto_arm)
    t0 = time.perf_counter()
    wheeled = False
    err: str | None = None
    try:
        from mm_assist.safety.input import release_mouse_buttons, wheel_client

        with live_benchmark_input_scope():
            wheeled = bool(
                wheel_client(
                    win,
                    int(cx),
                    int(cy),
                    path="item_list_scroll",
                    delta=int(delta),
                    notches=int(notches),
                    method=m,
                )
            )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            from mm_assist.safety.input import release_mouse_buttons

            release_mouse_buttons()
        except Exception:
            pass
        revoke("live_benchmark_done")
        reset_guard_state()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "ok": bool(wheeled) and err is None,
        "wheeled": wheeled,
        "method": m,
        "transition_id": tid,
        "elapsed_ms": elapsed_ms,
        "need_ui": need_ui,
        "stage": stage,
        "error": err,
    }
