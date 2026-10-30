"""客户区左键点击 / 拖拽（唯一对外输入入口）。

方法矩阵与业界对照见 knowledge/ui/input_methods.md。
粘滞自救：release_mouse_buttons / ``python -m mm_assist mouse-unlock``。
"""
from __future__ import annotations

import atexit
import logging
import signal
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Sequence

from mm_assist.safety.click_policy import (
    HitBox,
)
from mm_assist.config import AppConfig
from mm_assist.platform.window import WindowInfo, focus_window, is_foreground

log = logging.getLogger(__name__)

_MOUSE_SAFETY_INSTALLED = False
_INPUT_HUD_HWND = 0
_INPUT_HUD_RESTORE = False
_FOREGROUND_INPUT_HWND = ContextVar('life_foreground_input_hwnd', default=0)
_FOREGROUND_INPUT_WINDOW = ContextVar('life_foreground_input_window', default=None)
_FACILITY_CONFIRM_POINT = ContextVar('facility_confirm_point', default=None)
_POINTER_TRACE = ContextVar('life_pointer_trace', default=False)


@contextmanager
def pointer_trace_scope(enabled: bool):
    """Optional diagnostics scoped to the resident executor, never an input permit."""
    token = _POINTER_TRACE.set(bool(enabled))
    try:
        yield
    finally:
        _POINTER_TRACE.reset(token)


class _PointerTrace:
    def __init__(self, kind, win, points, **timing):
        self.enabled = _POINTER_TRACE.get()
        if self.enabled:
            self.kind, self.win = kind, win
            self.points, self.timing = points, timing
            self.started = time.monotonic()
            self.samples = []

    def sample(self, phase, sx, sy):
        if not self.enabled:
            return
        # A failed diagnostic read cannot suppress button release or authorize input.
        try:
            import win32gui
            px, py = win32gui.GetCursorPos()
            actual, error = [px - self.win.left, py - self.win.top], ''
        except Exception as exc:
            actual, error = None, str(exc)
        self.samples.append(dict(phase=phase, requested_client=[sx-self.win.left, sy-self.win.top],
            actual_client=actual, elapsed_s=time.monotonic()-self.started, error=error))

    def finish(self, error=''):
        if not self.enabled:
            return
        try:
            from mm_assist.life_plan.execution_records import pointer_trace
            pointer_trace(kind=self.kind, hwnd=self.win.hwnd,
                coordinate_space='game_frame_client_pixels',
                client_origin=[self.win.left, self.win.top],
                client_size=[self.win.width, self.win.height],
                planned_client_points=[[x-self.win.left, y-self.win.top] for x, y in self.points],
                timing=self.timing, samples=self.samples,
                elapsed_s=time.monotonic()-self.started, error=error)
        except Exception:
            log.debug('pointer trace unavailable', exc_info=True)


@contextmanager
def facility_confirm_input_scope(xy: tuple[int, int], frame_size: tuple[int, int]):
    """Only the freshly located travel-confirm point gets a temporary shield exception."""
    token = _FACILITY_CONFIRM_POINT.set((tuple(xy), tuple(frame_size), time.time()))
    try:
        yield
    finally:
        _FACILITY_CONFIRM_POINT.reset(token)


@contextmanager
def foreground_input_scope(hwnd: int, *, win: WindowInfo | None = None):
    """Authorized resident life operation may focus this window per dispatch."""
    bound_hwnd = int(hwnd or 0)
    if win is not None and int(getattr(win, 'hwnd', 0) or 0) != bound_hwnd:
        raise RuntimeError('click_window_mismatch: resident input window differs')
    token = _FOREGROUND_INPUT_HWND.set(bound_hwnd)
    window_token = _FOREGROUND_INPUT_WINDOW.set(win if bound_hwnd > 0 else None)
    try:
        yield
    finally:
        _FOREGROUND_INPUT_WINDOW.reset(window_token)
        _FOREGROUND_INPUT_HWND.reset(token)


def foreground_input_enabled(hwnd: int) -> bool:
    return int(hwnd or 0) > 0 and _FOREGROUND_INPUT_HWND.get() == int(hwnd)


def _focus_for_dispatch(win: WindowInfo) -> float | None:
    if not foreground_input_enabled(win.hwnd):
        return None
    check_abort()
    if is_foreground(win.hwnd):
        # The one-second budget bounds an actual activation, not observation
        # work on a game that was already foreground. Rechecks still run.
        return None
    focus_window(win.hwnd)
    if not is_foreground(win.hwnd):
        raise RuntimeError('click_focus_lost: cannot foreground game')
    # The product deadline starts at confirmed activation. Focus advice and
    # preparation while another app still owns the foreground are not elapsed
    # game-foreground time; later checks still share this single deadline.
    return time.monotonic() + 1.0


def _check_focus_deadline(deadline: float | None) -> None:
    if deadline is not None:
        elapsed = time.monotonic() - (deadline - 1.0)
        log.info('life focus_to_dispatch_s=%.4f', elapsed)
        if elapsed >= 1.0:
            raise RuntimeError('focus_click_timeout')


def register_input_hud(hwnd: int) -> None:
    global _INPUT_HUD_HWND
    _INPUT_HUD_HWND = int(hwnd)


class AbortRequested(RuntimeError):
    pass


def _check_abort_after_verified_frame() -> None:
    """Recheck stop/generation without recapturing this guarded input's pixels."""
    from mm_assist.safety.lease import cancel_requested
    if cancel_requested():
        raise AbortRequested("user abort")
    from mm_assist.life_plan.game_stop import game_stop_scope_cancelled
    if game_stop_scope_cancelled():
        raise AbortRequested("game_stop_paused")


def check_abort() -> None:
    """强制结束 / F8 撤权后，禁止继续键鼠。"""
    _check_abort_after_verified_frame()
    from mm_assist.life_plan.game_stop import game_stop_interrupted
    # Do not suspend a held gesture. The fresh boundary before each new input
    # and the executor's between-action checks handle transient network waits.
    if game_stop_interrupted(check_network=False):
        raise AbortRequested("game_stop_paused")


def _segment_intersects_rect(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> bool:
    """折线边与窗口矩形是否相交。GetWindowRect 的 right/bottom 为开区间。"""
    xmin = float(left)
    xmax = float(right) - 1e-9
    ymin = float(top)
    ymax = float(bottom) - 1e-9
    if xmax < xmin or ymax < ymin:
        return False
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, float(x0) - xmin),
        (dx, xmax - float(x0)),
        (-dy, float(y0) - ymin),
        (dy, ymax - float(y0)),
    ):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        t = q / p
        if p < 0.0:
            if t > t1:
                return False
            if t > t0:
                t0 = t
        else:
            if t < t0:
                return False
            if t < t1:
                t1 = t
    return True


def _hud_window_rect() -> tuple[int, int, int, int] | None:
    """本程序 HUD 顶窗矩形。查询失败返回 None（调用方按挡住处理）。"""
    import win32gui
    try:
        l, t, r, b = win32gui.GetWindowRect(int(_INPUT_HUD_HWND))
        return int(l), int(t), int(r), int(b)
    except Exception:
        return None


def _hud_blocks_any_point(points: Sequence[tuple[int, int]]) -> bool:
    """本程序 HUD 或其子窗是否盖住任一屏幕点。查询失败视为挡住（安全让行）。"""
    import win32gui
    hud = int(_INPUT_HUD_HWND)
    for sx, sy in points:
        try:
            hit = int(win32gui.WindowFromPoint((int(sx), int(sy))) or 0)
        except Exception:
            return True
        if hit == hud or (hit and win32gui.IsChild(hud, hit)):
            return True
    return False


def _hud_blocks_polyline(points: Sequence[tuple[int, int]]) -> bool:
    """采样点 WindowFromPoint，或折线边与 HUD 矩形相交。几何查询失败视为挡住。"""
    if _hud_blocks_any_point(points):
        return True
    if len(points) < 2:
        return False
    rect = _hud_window_rect()
    if rect is None:
        return True
    left, top, right, bottom = rect
    prev = points[0]
    for cur in points[1:]:
        if _segment_intersects_rect(
            int(prev[0]), int(prev[1]), int(cur[0]), int(cur[1]),
            left, top, right, bottom,
        ):
            return True
        prev = cur
    return False


def _refresh_same_client_size(win: WindowInfo):
    """refresh 后原点可用；客户区宽高变化则拒绝，避免沿旧识别坐标输入。"""
    from mm_assist.platform.window import refresh_window

    current = refresh_window(win)
    if (int(current.width), int(current.height)) != (int(win.width), int(win.height)):
        raise RuntimeError("click_window_resized: reobserve controls")
    return current


def _require_pointer_geometry(win: WindowInfo) -> None:
    """Once the pointer moves, keep the same client origin through button release."""
    current = _refresh_same_client_size(win)
    if (int(current.left), int(current.top)) != (int(win.left), int(win.top)):
        raise RuntimeError('click_window_moved: reobserve controls')


def _wait_input(seconds: float, check) -> None:
    """Bound uninterrupted holds without changing the requested total wait."""
    remaining = max(0., float(seconds))
    while remaining > 0:
        check()
        part = min(.025, remaining)
        time.sleep(part)
        remaining -= part
    check()


def _linear_client_screen_points(
    origin_left: int,
    origin_top: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    steps: int,
) -> list[tuple[int, int]]:
    n = max(2, int(steps))
    pts: list[tuple[int, int]] = []
    for i in range(n + 1):
        t = i / n
        cx = int(x0 + (x1 - x0) * t)
        cy = int(y0 + (y1 - y0) * t)
        pts.append((int(origin_left) + cx, int(origin_top) + cy))
    return pts


def _park_hud_outside_game(game_hwnd: int) -> bool:
    """Move our visible bar into observed monitor space, without activation."""
    import win32api
    import win32con
    import win32gui
    try:
        hl, ht, hr, hb = win32gui.GetWindowRect(_INPUT_HUD_HWND)
        gl, gt, gr, gb = win32gui.GetWindowRect(game_hwnd)
        work = win32api.GetMonitorInfo(win32api.MonitorFromWindow(game_hwnd, 2))['Work']
        wl, wt, wr, wb = work
        width, height = hr-hl, hb-ht
        if width <= 0 or height <= 0 or width > wr-wl or height > wb-wt:
            return False
        x = min(max(hl, wl), wr-width)
        y = min(max(ht, wt), wb-height)
        candidates = [(gl-width-8, y), (gr+8, y), (x, gt-height-8), (x, gb+8)]
        for left, top in candidates:
            if left < wl or top < wt or left+width > wr or top+height > wb:
                continue
            win32gui.SetWindowPos(_INPUT_HUD_HWND, 0, left, top, 0, 0,
                win32con.SWP_NOSIZE | win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE)
            rect = _hud_window_rect()
            if rect is not None and (rect[2] <= gl or rect[0] >= gr or rect[3] <= gt or rect[1] >= gb):
                log.info('life HUD moved outside game rect=%s', rect)
                return True
        return False
    except Exception:
        log.debug('life HUD positioning failed', exc_info=True)
        return False


def hud_suspend_for_input(*, points: Sequence[tuple[int, int]] | None = None) -> None:
    """仅当本程序 HUD 挡住本次落点或拖拽路径时临时隐藏；无 points 不藏栏。"""
    global _INPUT_HUD_RESTORE
    if sys.platform != "win32" or not _INPUT_HUD_HWND:
        return
    if not points:
        return
    import win32gui
    if not win32gui.IsWindow(_INPUT_HUD_HWND):
        return
    if not win32gui.IsWindowVisible(_INPUT_HUD_HWND):
        return
    if not _hud_blocks_polyline(points):
        return
    game_hwnd = _FOREGROUND_INPUT_HWND.get()
    if game_hwnd:
        if not _park_hud_outside_game(game_hwnd) or _hud_blocks_polyline(points):
            raise RuntimeError('click_hud_occluded: no visible space outside game')
        return
    _INPUT_HUD_RESTORE = True
    win32gui.ShowWindow(_INPUT_HUD_HWND, 0)  # Hide our own topmost HUD only.


def hud_resume_after_input() -> None:
    global _INPUT_HUD_RESTORE
    if _INPUT_HUD_RESTORE:
        import win32gui
        _INPUT_HUD_RESTORE = False
        if win32gui.IsWindow(_INPUT_HUD_HWND):
            win32gui.ShowWindow(_INPUT_HUD_HWND, 4)  # Restore without activation.


def require_game_point(hwnd: int, sx: int, sy: int) -> None:
    import win32gui
    if not is_foreground(hwnd):
        raise RuntimeError("click_focus_lost")
    hit = int(win32gui.WindowFromPoint((int(sx), int(sy))) or 0)
    if hit != int(hwnd) and not win32gui.IsChild(int(hwnd), hit):
        raise RuntimeError(f"click_occluded: point belongs to hwnd={hit}, game={hwnd}")


def request_stop() -> None:
    return None


class InvalidClientClick(ValueError):
    """无有效客户区坐标 → 禁止点击（防屏外/左上角默认点）。"""


def require_valid_client_xy(
    win: WindowInfo | None,
    cx: int,
    cy: int,
) -> tuple[int, int]:
    """客户区坐标门闩：负值 / 出界 / (0,0) 毒默认 / 窗尺寸无效 → fail-closed。"""
    if win is None:
        raise InvalidClientClick("click_client: win is None")
    try:
        x = int(cx)
        y = int(cy)
    except (TypeError, ValueError) as exc:
        raise InvalidClientClick(f"click_client: bad xy {cx!r},{cy!r}") from exc
    try:
        w = int(getattr(win, "width", 0) or 0)
        h = int(getattr(win, "height", 0) or 0)
    except (TypeError, ValueError):
        w, h = 0, 0
    if w < 100 or h < 100:
        raise InvalidClientClick(f"click_client: win size invalid {w}x{h}")
    # (0,0) = 未解析/OCR miss 默认；合法近角用 (30,40) 等，不放行原点
    if x == 0 and y == 0:
        raise InvalidClientClick("click_client: refuse (0,0) poison default")
    if x < 0 or y < 0 or x >= w or y >= h:
        raise InvalidClientClick(
            f"click_client: out of client ({x},{y}) size={w}x{h}"
        )
    return x, y

# 与 knowledge/ui/input_methods.md 方法矩阵一致；未知 method 禁止静默回退
INPUT_METHODS = frozenset(
    {
        "sendinput",
        "si",
        "unity_hook",
        "hook",
        "postmessage",
        "post",
        "background",
        "sendmessage",
        "sendmsg",
    }
)
TEST_CLICK_METHODS = frozenset({"sendinput", "si", "unity_hook", "hook"})


def normalize_input_method(method: str | None, *, default: str = "sendinput") -> str:
    """规范化并校验输入 method；未知值直接报错（禁止静默当 sendinput）。"""
    m = str(method or default).strip().lower() or default
    if m not in INPUT_METHODS:
        raise ValueError(
            f"未知 input method={method!r}；允许 {sorted(INPUT_METHODS)}。"
            "见 knowledge/ui/input_methods.md"
        )
    return m


def _guard_before_input(
    kind: str,
    win: WindowInfo | None,
    *,
    method: str,
    transition_id: str | None = None,
    client_xy: tuple[int, int] | None = None,
) -> None:
    """输入边界：测试租约 / 生产旁路 / 默认 fail-closed。

    测试模式禁止 postmessage/sendmessage 点击；unity_hook 允许失焦。
    拖拽：测试模式允许 postmessage（地图回退）；禁止 sendmessage。
    禁止自动抢前台（sendinput 由调用方保证已前台）。
    """
    from mm_assist.safety.lease import (
        ACTION_CLICK,
        ACTION_DRAG,
        ACTION_KEY,
        GuardError,
        in_test_mode,
        require_action,
    )
    
    # This boundary forces a fresh game-stop frame below. Calling check_abort
    # first can capture another frame immediately beforehand, inside the
    # one-second foreground dispatch budget.
    from mm_assist.safety.lease import cancel_requested
    if cancel_requested():
        raise AbortRequested("user abort")
    from mm_assist.life_plan.game_stop import game_stop_interrupted, network_wait_seconds, network_input_unchanged
    if game_stop_interrupted(force=True):
        raise AbortRequested("game_stop_paused")
    if network_wait_seconds() > 0 and not network_input_unchanged(client_xy):
        raise RuntimeError('click_ui_not_ready: 网络等待后页面或目标已变化，须重新认页定位')
    from mm_assist.jobs.bar import click_owner_block_reason

    owner = click_owner_block_reason()
    if owner:
        raise GuardError(owner)
    m = normalize_input_method(method)
    # 点击：测试模式仅 sendinput / unity_hook。
    # 拖拽：优先 unity_hook；地图画布仍允许 postmessage。
    if in_test_mode() and m in (
        "postmessage",
        "post",
        "background",
        "sendmessage",
        "sendmsg",
    ):
        allow_drag_post = kind == ACTION_DRAG and m in (
            "postmessage",
            "post",
            "background",
        )
        if not allow_drag_post:
            raise GuardError(
                "测试模式禁止 postmessage/sendmessage，仅允许 sendinput/unity_hook"
            )

    hwnd = int(win.hwnd) if win is not None else 0
    k = kind
    if k not in (ACTION_CLICK, ACTION_DRAG, ACTION_KEY):
        k = ACTION_CLICK
    need_fg = bool(win is not None and hwnd > 0 and m not in ("unity_hook", "hook"))
    if m in ('sendinput','si') and foreground_input_enabled(hwnd):
        # Validate authorization before focusing. Actual dispatch still requires
        # foreground, current client geometry and unobstructed game pixels.
        need_fg = False
    require_action(
        k,
        hwnd,
        transition_id=transition_id,
        require_foreground=need_fg,
        method=m,
    )
    if win is not None and hwnd > 0:
        from mm_assist.memory.readers.ui_stack import shared_ui_stack_reader, input_blocking_layers

        stack = shared_ui_stack_reader().read(force=True)
        log.debug("input_ui ready=%s sampled_ts=%.3f names=%s", stack.ready,
                  stack.sampled_ts, ",".join(stack.names))
        if not stack.ready:
            raise RuntimeError("click_ui_not_ready: 当前界面栈未就绪")
        from mm_assist.life_plan.game_stop import network_located_result_stack_current
        if not network_located_result_stack_current(stack.names,client_xy,win.hwnd):
            raise RuntimeError('click_ui_not_ready: 网络等待后页面或目标已变化，须重新认页定位')
        blocked = input_blocking_layers(stack.names)
        permit = _FACILITY_CONFIRM_POINT.get()
        from mm_assist.memory.readers.ui_stack import facility_travel_popup_has_only_own_shield
        own_popup = bool(permit is not None and k == ACTION_CLICK
            and client_xy == permit[0]
            and (win.width, win.height) == permit[1]
            and 0 <= time.time() - permit[2] <= 2.0
            and 0 <= time.time() - stack.sampled_ts <= 2.0
            and facility_travel_popup_has_only_own_shield(stack.names))
        if blocked and not own_popup:
            raise RuntimeError("click_ui_blocked: " + ",".join(blocked))
        if permit is not None and not own_popup:
            raise RuntimeError("click_ui_blocked: 前往确认画面或落点已改变")


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    INPUT_MOUSE = 0
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_WHEEL = 0x0800
    MOUSEEVENTF_ABSOLUTE = 0x8000
    MOUSEEVENTF_VIRTUALDESK = 0x4000

    WM_MOUSEMOVE = 0x0200
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    WM_MOUSEWHEEL = 0x020A
    MK_LBUTTON = 0x0001

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]

    def _send_mouse(flags: int, x: int = 0, y: int = 0) -> None:
        abs_x, abs_y = x, y
        if flags & MOUSEEVENTF_MOVE and flags & MOUSEEVENTF_ABSOLUTE:
            user32 = ctypes.windll.user32
            left, top, width, height = (user32.GetSystemMetrics(i) for i in (76, 77, 78, 79))
            if width <= 1 or height <= 1 or not (left <= x < left+width and top <= y < top+height):
                raise RuntimeError('click_desktop_geometry: invalid screen point or virtual desktop')
            abs_x = round((x-left) * 65535 / (width-1))
            abs_y = round((y-top) * 65535 / (height-1))
            flags |= MOUSEEVENTF_VIRTUALDESK
        extra = ctypes.c_ulong(0)
        mi = MOUSEINPUT(abs_x, abs_y, 0, flags, 0, ctypes.pointer(extra))
        inp = INPUT(INPUT_MOUSE, mi)
        sent = ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        if sent != 1:
            raise RuntimeError(f"SendInput rejected flags={flags:#x}")
        if flags & MOUSEEVENTF_MOVE and flags & MOUSEEVENTF_ABSOLUTE:
            log.debug('sendinput desktop=(%s,%s,%s,%s) requested=(%s,%s) normalized=(%s,%s)',
                      left, top, width, height, x, y, abs_x, abs_y)

    def _makelparam(cx: int, cy: int) -> int:
        return (int(cy) << 16) | (int(cx) & 0xFFFF)

    def _post_click(hwnd: int, cx: int, cy: int, hold_s: float = 0.02) -> None:
        user32 = ctypes.windll.user32
        lp = _makelparam(cx, cy)
        user32.PostMessageW(hwnd, WM_MOUSEMOVE, 0, lp)
        time.sleep(0.01)
        user32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lp)
        time.sleep(max(hold_s, 0.001))
        user32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lp)

    def _sendmsg_click(hwnd: int, cx: int, cy: int, hold_s: float = 0.02) -> None:
        user32 = ctypes.windll.user32
        lp = _makelparam(cx, cy)
        user32.SendMessageW(hwnd, WM_MOUSEMOVE, 0, lp)
        time.sleep(0.01)
        user32.SendMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lp)
        time.sleep(max(hold_s, 0.001))
        user32.SendMessageW(hwnd, WM_LBUTTONUP, 0, lp)
else:  # pragma: no cover

    def _send_mouse(flags: int, x: int = 0, y: int = 0) -> None:
        raise RuntimeError("actuation only supported on Windows")

    def _post_click(hwnd: int, cx: int, cy: int, hold_s: float = 0.02) -> None:
        raise RuntimeError("postmessage only on Windows")

    def _sendmsg_click(hwnd: int, cx: int, cy: int, hold_s: float = 0.02) -> None:
        raise RuntimeError("sendmessage only on Windows")


def _sendinput_yield_unfocused(win: WindowInfo, cx: int, cy: int) -> bool:
    """未对上游戏前台且抢焦闸命中 → 让行。

    True 表示调用方须立刻 return False，且不得再 SetCursorPos / _send_mouse。
    sendinput 动的是真实光标：没对上焦就会点到人正在用的窗口上。
    """
    if is_foreground(win.hwnd):
        return False
    from mm_assist.platform.window import focus_steal_block_reason

    blocked = focus_steal_block_reason()
    if not blocked:
        return False
    log.warning(
        "sendinput yield_focus (%d,%d)：人在操作且未对上前台（%s）",
        int(cx),
        int(cy),
        blocked,
    )
    return True


def _stamp_sendinput_heartbeat() -> None:
    """sendinput 成功走完后再打点，使 idle_s ≈ hb_age。让行路径不得调用。"""
    try:
        from mm_assist.jobs.focus_watch import stamp_agent_input_heartbeat

        stamp_agent_input_heartbeat(who="sendinput")
    except Exception:
        pass


def click_client(
    win: WindowInfo,
    cx: int,
    cy: int,
    *,
    method: str = "sendinput",
    refocus: bool = False,
    hold_s: float | None = None,
    transition_id: str | None = None,
) -> bool:
    from mm_assist.safety.readonly import deny_input

    deny_input("click_client")
    """
    method（见 knowledge/ui/input_methods.md）：
      - sendinput：前台真实鼠标（默认；验收）
      - unity_hook：受控注入伪造 Unity/`NXPInput` 鼠标读口（可失焦；长挂）
      - postmessage：后台消息 ——测试模式禁止点击
      - sendmessage：同步消息 ——测试模式禁止；勿作验收回退
    hold_s：按下时长（秒）；None 时用短默认 0.02。
    返回 True=已点；False=让行未点（日志带 yield_focus）。调用方可忽略。
    任意路径结束必 release_mouse_buttons（含 force_hook_idle）。
    """
    try:
        from mm_assist.tool_repair import op_cancelled

        if op_cancelled():
            return False
    except Exception:
        pass
    from mm_assist.safety.lease import ACTION_CLICK, in_test_mode

    m = normalize_input_method(method, default="sendinput")
    cx, cy = require_valid_client_xy(win, cx, cy)
    from mm_assist.life_plan.live_test import record_input, check_input_deadline

    input_deadline = record_input(win, "click", [cx, cy])
    _guard_before_input(ACTION_CLICK, win, method=m, transition_id=transition_id, client_xy=(cx, cy))
    hold = 0.02 if hold_s is None else max(float(hold_s), 0.001)
    try:
        if m in ("unity_hook", "hook"):
            from mm_assist.unity_input_hook import click_client_unity_hook

            click_client_unity_hook(win, cx, cy, hold_s=hold)
            return True
        if m in ("postmessage", "post", "background"):
            _post_click(win.hwnd, cx, cy, hold_s=hold)
            return True
        if m in ("sendmessage", "sendmsg"):
            _sendmsg_click(win.hwnd, cx, cy, hold_s=hold)
            return True
        # sendinput / si
        install_mouse_safety()
        try:
            focus_deadline = _focus_for_dispatch(win)
            if (not foreground_input_enabled(win.hwnd) and not in_test_mode()
                    and (refocus or not is_foreground(win.hwnd))):
                focus_window(win.hwnd)
                time.sleep(0.05)
            if _sendinput_yield_unfocused(win, cx, cy):
                return False
            if foreground_input_enabled(win.hwnd) and not is_foreground(win.hwnd):
                raise RuntimeError('click_focus_lost: game lost focus after preparation')
            current = _refresh_same_client_size(win)
            sx, sy = int(current.left) + cx, int(current.top) + cy
            hud_suspend_for_input(points=[(sx, sy)])
            trace = _PointerTrace('click', current, ((sx, sy),), before_down_s=.08, hold_s=hold)
            error = ''
            try:
                _send_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, sx, sy)
                # Give the just-activated game time to process its pointer move.
                # The first-click deadline still bounds the entire dispatch.
                time.sleep(0.08)
                import win32gui
                pointer = win32gui.GetCursorPos()
                log.info("click dispatch hwnd=%s client=(%s,%s) origin=(%s,%s) screen=(%s,%s)",
                         win.hwnd, cx, cy, current.left, current.top, sx, sy)
                log.info("click pointer before_down=%s expected=(%s,%s)", pointer, sx, sy)
                if max(abs(pointer[0] - sx), abs(pointer[1] - sy)) > 2:
                    raise RuntimeError("click_pointer_moved: yield to mouse input")
                trace.sample('before_down', sx, sy)
                _require_pointer_geometry(current)
                if foreground_input_enabled(win.hwnd):
                    _guard_before_input(ACTION_CLICK, win, method=m, client_xy=(cx, cy))
                require_game_point(win.hwnd, sx, sy)
                if foreground_input_enabled(win.hwnd):
                    _check_abort_after_verified_frame()
                else:
                    check_abort()
                from mm_assist.tool_repair import op_cancelled
                if op_cancelled():
                    raise AbortRequested('user abort')
                check_input_deadline(input_deadline)
                # The stack/frame recheck can take time. Reuse neither the
                # old cursor position nor the old window geometry afterward.
                _require_pointer_geometry(current)
                pointer = win32gui.GetCursorPos()
                if max(abs(pointer[0] - sx), abs(pointer[1] - sy)) > 2:
                    raise RuntimeError('click_pointer_moved: yield to mouse input')
                require_game_point(win.hwnd, sx, sy)
                _check_focus_deadline(focus_deadline)
                _send_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE, sx, sy)
                def check_hold():
                    check_abort()
                    if op_cancelled():
                        raise AbortRequested('user abort')
                    check_input_deadline(input_deadline)
                    if not is_foreground(win.hwnd):
                        raise RuntimeError('click_focus_lost')
                    _require_pointer_geometry(current)
                _wait_input(hold, check_hold)
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                try:
                    _send_mouse(MOUSEEVENTF_LEFTUP)
                except Exception as exc:
                    if not error:
                        error = str(exc)
                        raise
                    error += f'; release_error={exc}'
                    log.warning('mouse release failed while handling %s', error)
                finally:
                    trace.sample('after_up', sx, sy)
                    trace.finish(error)
            _stamp_sendinput_heartbeat()
            return True
        finally:
            hud_resume_after_input()
    finally:
        release_mouse_buttons()


def release_mouse_buttons(*, also_right: bool = False) -> None:
    """松开左键 + 取消 ClipCursor + 清空 unity_hook SHM。

    默认只抬左键；应急解锁和显式相机协议可 ``also_right=True`` 同时抬右键。
    """
    if sys.platform != "win32":
        return
    try:
        extra = ctypes.c_ulong(0)
        for _ in range(2):
            mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
            inp = INPUT(INPUT_MOUSE, mi)
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        if also_right:
            for _ in range(2):
                mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_RIGHTUP, 0, ctypes.pointer(extra))
                inp = INPUT(INPUT_MOUSE, mi)
                ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        try:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            if also_right:
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        except Exception:
            pass
        ctypes.windll.user32.ClipCursor(None)
        try:
            ctypes.windll.user32.BlockInput(False)
        except Exception:
            pass
    except Exception:
        log.debug("release_mouse_buttons failed", exc_info=True)
    try:
        from mm_assist.unity_input_hook.backend import force_hook_idle

        force_hook_idle()
    except Exception:
        log.debug("force_hook_idle from release failed", exc_info=True)


def _clip_client_xy(win: WindowInfo, cx: int, cy: int, box: dict[str, Any]) -> tuple[int, int]:
    x0 = int(box["x0"])
    y0 = int(box["y0"])
    x1 = int(box["x1"])
    y1 = int(box["y1"])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("palette box empty")
    cx = max(x0 + 2, min(x1 - 2, int(cx)))
    cy = max(y0 + 2, min(y1 - 2, int(cy)))
    # 亦夹客户区
    cx = max(0, min(int(win.width) - 1, cx))
    cy = max(0, min(int(win.height) - 1, cy))
    return cx, cy


# 右键/滚轮默认禁；仅下列 path 放行（显式主界面相机等协议）
RIGHT_WHEEL_ALLOWED_PATHS = frozenset(
    {
        "camera_input_smoke",
        "item_list_scroll",
    }
)


def _require_right_wheel_path(path: str) -> str:
    p = str(path or "").strip()
    if p not in RIGHT_WHEEL_ALLOWED_PATHS:
        raise RuntimeError(
            f"右键/滚轮未放行 path={p!r}；允许={sorted(RIGHT_WHEEL_ALLOWED_PATHS)}"
        )
    return p


def _client_clip_box(win: WindowInfo) -> dict[str, int]:
    """主界面冒烟：夹在客户区内缩，避开边栏按钮。"""
    m = 80
    return {
        "x0": m,
        "y0": m,
        "x1": max(m + 8, int(win.width) - m),
        "y1": max(m + 8, int(win.height) - m),
    }


def wheel_client(
    win: WindowInfo,
    cx: int,
    cy: int,
    *,
    path: str,
    clip_box: dict[str, Any] | None = None,
    delta: int = -120,
    notches: int = 1,
    method: str = "sendinput",
    refocus: bool = True,
) -> bool:
    """滚轮缩放。须显式 ``path``（``camera_input_smoke`` / ``item_list_scroll``）。

    返回 True=已滚；False=让行未滚（日志带 yield_focus）。
    """
    from mm_assist.safety.readonly import deny_input

    deny_input("wheel_client")
    from mm_assist.safety.lease import ACTION_DRAG, in_test_mode
    
    _require_right_wheel_path(path)
    if sys.platform != "win32":
        raise RuntimeError("wheel only on Windows")
    m = normalize_input_method(method, default="sendinput")
    _guard_before_input(ACTION_DRAG, win, method=m)
    box = clip_box if clip_box is not None else _client_clip_box(win)
    cx, cy = _clip_client_xy(win, cx, cy, box)
    try:
        from mm_assist.unity_input_hook.backend import force_hook_idle

        force_hook_idle()
    except Exception:
        pass
    install_mouse_safety()
    try:
        if not in_test_mode() and (refocus or not is_foreground(win.hwnd)):
            focus_window(win.hwnd)
            time.sleep(0.06)
        if _sendinput_yield_unfocused(win, cx, cy):
            return False
        if m in ("postmessage", "post", "background"):
            user32 = ctypes.windll.user32
            lp = _makelparam(cx, cy)
            n = max(1, int(notches))
            step = int(delta)
            for _ in range(n):
                wp = ctypes.c_ulong((step << 16) & 0xFFFFFFFF).value
                user32.PostMessageW(int(win.hwnd), WM_MOUSEWHEEL, wp, lp)
                time.sleep(0.06)
            return True
        current = _refresh_same_client_size(win)
        sx, sy = int(current.left + cx), int(current.top + cy)
        hud_suspend_for_input(points=[(sx, sy)])
        _send_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, sx, sy)
        ctypes.windll.user32.SetCursorPos(int(sx), int(sy))
        time.sleep(0.05)
        n = max(1, int(notches))
        step = int(delta)
        for _ in range(n):
            extra = ctypes.c_ulong(0)
            mi = MOUSEINPUT(
                0, 0, step & 0xFFFFFFFF, MOUSEEVENTF_WHEEL, 0, ctypes.pointer(extra)
            )
            inp = INPUT(INPUT_MOUSE, mi)
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
            time.sleep(0.06)
        return True
    finally:
        try:
            hud_resume_after_input()
        except Exception:
            pass
        release_mouse_buttons(also_right=True)


def install_mouse_safety() -> None:
    """进程退出/Ctrl+C 时必抬鼠标键；可重复调用。"""
    global _MOUSE_SAFETY_INSTALLED
    if _MOUSE_SAFETY_INSTALLED:
        return
    _MOUSE_SAFETY_INSTALLED = True
    atexit.register(release_mouse_buttons)

    def _on_signal(signum, _frame) -> None:  # noqa: ANN001
        try:
            release_mouse_buttons()
        except Exception:
            pass
        try:
            
            request_stop(f"signal:{signum}")
        except Exception:
            pass
        try:
            release_mouse_buttons()
        except Exception:
            pass
        raise SystemExit(130)

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None), getattr(signal, "SIGBREAK", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass


def click_hitbox(
    cfg: AppConfig,
    win: WindowInfo,
    hitbox: HitBox | Sequence[float],
    *,
    method: str = "sendinput",
    refocus: bool = False,
    use_center: bool = False,
) -> tuple[float, float]:
    del cfg, win, hitbox, method, refocus, use_center
    from mm_assist.safety.readonly import deny_input

    deny_input("click_hitbox")
    return 0.0, 0.0


def press_vk(vk: int, *, hwnd: int | None = None, win: WindowInfo | None = None) -> None:
    """SendInput 虚拟键（需前台焦点场景；如 Escape=0x1B）。"""
    from mm_assist.safety.readonly import deny_input

    deny_input("press_vk")
    if sys.platform != "win32":
        raise RuntimeError("press_vk only on Windows")
    from mm_assist.safety.lease import (
        ACTION_KEY,
        NotAuthorized,
        current_lease,
        in_production_mode,
        in_test_mode,
        require_action,
    )
    
    check_abort()
    bound_window = _FOREGROUND_INPUT_WINDOW.get()
    if win is None and bound_window is not None:
        if hwnd is not None and int(hwnd) != int(bound_window.hwnd):
            raise RuntimeError('click_window_mismatch: key targets another window')
        win = bound_window
    h = int(win.hwnd) if win is not None else hwnd
    if h is None:
        lease = current_lease()
        if lease is not None:
            h = int(lease.hwnd)
    if win is not None:
        from mm_assist.life_plan.live_test import record_input, check_input_deadline
        from mm_assist.tool_repair import op_cancelled

        if op_cancelled():
            raise AbortRequested('user abort')
        deadline = record_input(win, 'key', [int(vk)])
        _guard_before_input(ACTION_KEY, win, method='sendinput')
        if not is_foreground(win.hwnd):
            raise RuntimeError('click_focus_lost: Esc requires foreground game')
        # 同客户区、新鲜就绪栈、输入所有者和租约；Esc 不抢前台。
        _refresh_same_client_size(win)
        check_input_deadline(deadline)
        check_abort()
        if op_cancelled():
            raise AbortRequested('user abort')
        if not is_foreground(win.hwnd):
            raise RuntimeError('click_focus_lost: game lost focus before Esc')
    elif in_test_mode():
        if not h:
            raise NotAuthorized("press_vk 测试模式需要 hwnd/租约")
        require_action(ACTION_KEY, int(h), require_foreground=True, method="sendinput")
    elif not in_production_mode():
        require_action(
            ACTION_KEY, int(h or 0), require_foreground=False, method="sendinput"
        )

    _send_key(int(vk))
    _stamp_sendinput_heartbeat()


def _send_key(vk: int) -> None:
    """Win32 INPUT 的联合体必须包含 MOUSEINPUT，不能传短键盘结构。"""
    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        class _U(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    extra = ctypes.c_ulong(0)
    down = INPUT(INPUT_KEYBOARD, INPUT._U(KEYBDINPUT(vk, 0, 0, 0, ctypes.pointer(extra))))
    up = INPUT(
        INPUT_KEYBOARD, INPUT._U(KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP, 0, ctypes.pointer(extra)))
    )
    user32 = ctypes.windll.user32
    if user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT)) != 1:
        raise RuntimeError('click_key_failed: key down was not dispatched')
    try:
        time.sleep(0.05)
    finally:
        if user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT)) != 1:
            raise RuntimeError('click_key_failed: key up was not dispatched')


def press_escape(*, hwnd: int | None = None, win: WindowInfo | None = None) -> None:
    press_vk(0x1B, hwnd=hwnd, win=win)


def drag_client(
    win: WindowInfo,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    method: str = "sendinput",
    refocus: bool = False,
    steps: int = 12,
    duration_s: float = 0.35,
    end_hold_s: float = 0.0,
    transition_id: str | None = None,
) -> bool:
    """客户区拖动：按下 → 插值到终点 → 原地按住 end_hold_s → 松开。

    返回 True=已拖；False=让行未拖（日志带 yield_focus）。调用方可忽略。
    任意路径结束必 release_mouse_buttons。见 knowledge/ui/input_methods.md。
    """
    from mm_assist.safety.readonly import deny_input

    deny_input("drag_client")
    if sys.platform != "win32":
        raise RuntimeError("drag only supported on Windows")
    from mm_assist.safety.lease import ACTION_DRAG, in_test_mode

    m = normalize_input_method(method, default="sendinput")
    x0, y0 = require_valid_client_xy(win, x0, y0)
    x1, y1 = require_valid_client_xy(win, x1, y1)
    from mm_assist.life_plan.live_test import record_input, check_input_deadline

    input_deadline = record_input(win, "drag", [x0, y0, x1, y1])
    _guard_before_input(ACTION_DRAG, win, method=m, transition_id=transition_id)
    n = max(2, int(steps))
    pause = max(0.01, float(duration_s) / n)
    hold = float(end_hold_s)
    if not 0 <= hold <= 2:
        raise ValueError('end_hold_s must be between 0 and 2 seconds')

    def check_drag():
        check_abort()
        from mm_assist.tool_repair import op_cancelled
        if op_cancelled():
            raise RuntimeError('已停止操作')
        check_input_deadline(input_deadline)
        if m not in ('unity_hook', 'hook') and not is_foreground(win.hwnd):
            raise RuntimeError('drag_focus_lost')
        if pointer_window is not None:
            _require_pointer_geometry(pointer_window)

    pointer_window = None

    def hold_endpoint():
        until = time.monotonic() + hold
        while True:
            check_drag()
            remaining = until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(.025, remaining))

    try:
        if m in ("unity_hook", "hook"):
            from mm_assist.unity_input_hook import drag_client_unity_hook

            drag_client_unity_hook(
                win, x0, y0, x1, y1, steps=n, duration_s=float(duration_s),
                **({'end_hold_s': hold} if hold else {})
            )
            return True

        if m in ("postmessage", "post", "background"):
            user32 = ctypes.windll.user32
            lp0 = _makelparam(x0, y0)
            user32.PostMessageW(win.hwnd, WM_MOUSEMOVE, 0, lp0)
            time.sleep(0.02)
            cx, cy = x0, y0
            try:
                user32.PostMessageW(win.hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lp0)
                for i in range(1, n + 1):
                    check_drag()
                    t = i / n
                    cx = int(x0 + (x1 - x0) * t)
                    cy = int(y0 + (y1 - y0) * t)
                    user32.PostMessageW(
                        win.hwnd, WM_MOUSEMOVE, MK_LBUTTON, _makelparam(cx, cy)
                    )
                    time.sleep(pause)
                hold_endpoint()
            finally:
                user32.PostMessageW(win.hwnd, WM_LBUTTONUP, 0, _makelparam(cx, cy))
            return True

        if m in ("sendmessage", "sendmsg"):
            user32 = ctypes.windll.user32
            lp0 = _makelparam(x0, y0)
            user32.SendMessageW(win.hwnd, WM_MOUSEMOVE, 0, lp0)
            time.sleep(0.02)
            cx, cy = x0, y0
            try:
                user32.SendMessageW(win.hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lp0)
                for i in range(1, n + 1):
                    check_drag()
                    t = i / n
                    cx = int(x0 + (x1 - x0) * t)
                    cy = int(y0 + (y1 - y0) * t)
                    user32.SendMessageW(
                        win.hwnd, WM_MOUSEMOVE, MK_LBUTTON, _makelparam(cx, cy)
                    )
                    time.sleep(pause)
                hold_endpoint()
            finally:
                user32.SendMessageW(win.hwnd, WM_LBUTTONUP, 0, _makelparam(cx, cy))
            return True

        # sendinput / si
        install_mouse_safety()
        try:
            focus_deadline = _focus_for_dispatch(win)
            if (not foreground_input_enabled(win.hwnd) and not in_test_mode()
                    and (refocus or not is_foreground(win.hwnd))):
                focus_window(win.hwnd)
                time.sleep(0.05)
            if _sendinput_yield_unfocused(win, x0, y0):
                return False
            if foreground_input_enabled(win.hwnd) and not is_foreground(win.hwnd):
                raise RuntimeError('click_focus_lost: game lost focus after preparation')
            current = _refresh_same_client_size(win)
            pointer_window = current
            # Freeze one path for HUD checks, dispatch and diagnostic replay.
            points = tuple(_linear_client_screen_points(current.left, current.top, x0, y0, x1, y1, n))
            hud_suspend_for_input(points=points)
            sx0, sy0 = points[0]
            sx, sy = sx0, sy0
            user32 = ctypes.windll.user32
            after_down = max(0.08, min(0.2, float(duration_s) * 0.12))
            trace = _PointerTrace('drag', current, points, before_down_s=.05,
                                  after_down_s=after_down, point_interval_s=pause, end_hold_s=hold)
            error = ''
            try:
                check_drag()
                user32.SetCursorPos(int(sx0), int(sy0))
                _wait_input(.05, check_drag)
                trace.sample('before_down', sx, sy)
                check_drag()
                if foreground_input_enabled(win.hwnd):
                    _guard_before_input(ACTION_DRAG, win, method=m)
                    require_game_point(win.hwnd, int(sx0), int(sy0))
                check_input_deadline(input_deadline)
                _check_focus_deadline(focus_deadline)
                _send_mouse(MOUSEEVENTF_LEFTDOWN)
                # 列表/ScrollRect：按下后稍停再移，否则游戏常吃不到拖
                _wait_input(after_down, check_drag)
                for sx, sy in points[1:]:
                    check_drag()
                    if foreground_input_enabled(win.hwnd):
                        require_game_point(win.hwnd, int(sx), int(sy))
                    user32.SetCursorPos(int(sx), int(sy))
                    _wait_input(pause, check_drag)
                    trace.sample('move', sx, sy)
                hold_endpoint()
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                try:
                    _send_mouse(MOUSEEVENTF_LEFTUP)
                except Exception as exc:
                    if not error:
                        error = str(exc)
                        raise
                    error += f'; release_error={exc}'
                    log.warning('mouse release failed while handling %s', error)
                finally:
                    trace.sample('after_up', sx, sy)
                    trace.finish(error)
            _stamp_sendinput_heartbeat()
            return True
        finally:
            hud_resume_after_input()
    finally:
        release_mouse_buttons()
