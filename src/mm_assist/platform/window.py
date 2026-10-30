from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

if sys.platform == "win32":
    import ctypes

    import win32con
    import win32gui
else:  # pragma: no cover
    ctypes = None  # type: ignore
    win32con = None  # type: ignore
    win32gui = None  # type: ignore

log = logging.getLogger(__name__)

# 标定客户区（与 config.window.client_size / game_frame 一致）
DESIGN_CLIENT_W = 1920
DESIGN_CLIENT_H = 1440
FULL_CLIENT_MIN_W = 1900
FULL_CLIENT_MIN_H = 1400

# 未 enable DPI 时，125%/150% 缩放常把 1920×1440 误读成 ≈960×720（假半窗）
_DPI_AWARE_DONE = False
_DPI_AWARE_OK = False


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


def enable_dpi_aware() -> bool:
    """进程 DPI 感知（幂等）。读客户区前须调用，否则易假半窗。

    返回是否已尝试启用（非 win32 视为无需启用且 ok）。
    """
    global _DPI_AWARE_DONE, _DPI_AWARE_OK
    if _DPI_AWARE_DONE:
        return _DPI_AWARE_OK
    if sys.platform != "win32" or ctypes is None:
        _DPI_AWARE_DONE = True
        _DPI_AWARE_OK = True
        return True
    ok = False
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        ok = True
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            ok = True
        except Exception:
            # 已由清单/先前调用设过时 API 会失败；仍标 done，靠尺寸复核
            ok = False
    _DPI_AWARE_DONE = True
    _DPI_AWARE_OK = bool(ok)
    return _DPI_AWARE_OK


def ensure_dpi_aware() -> bool:
    """读窗尺寸前默认入口（幂等）。"""
    if _DPI_AWARE_DONE:
        return _DPI_AWARE_OK
    return enable_dpi_aware()


def dpi_aware_active() -> bool:
    """本进程是否已成功调用过 DPI API（失败/未调 → False）。"""
    return bool(_DPI_AWARE_OK)


def reset_dpi_aware_for_tests() -> None:
    """单测复位 DPI 旗标（勿用于生产路径）。"""
    global _DPI_AWARE_DONE, _DPI_AWARE_OK
    _DPI_AWARE_DONE = False
    _DPI_AWARE_OK = False


def looks_like_dpi_false_half(
    width: int,
    height: int,
    *,
    design: tuple[int, int] = (DESIGN_CLIENT_W, DESIGN_CLIENT_H),
    tol: int = 16,
) -> bool:
    """客户区 ≈ 标定的一半 → 典型「未开 DPI」假半窗（如 960×720）。"""
    try:
        w = int(width or 0)
        h = int(height or 0)
    except (TypeError, ValueError):
        return False
    dw, dh = int(design[0]), int(design[1])
    if dw < 200 or dh < 200:
        return False
    return abs(w - dw // 2) <= int(tol) and abs(h - dh // 2) <= int(tol)


def assess_client_size(
    width: int,
    height: int,
    *,
    design: tuple[int, int] = (DESIGN_CLIENT_W, DESIGN_CLIENT_H),
    min_full: tuple[int, int] = (FULL_CLIENT_MIN_W, FULL_CLIENT_MIN_H),
) -> dict:
    """窗尺寸只读判定（供探针 / ui_escape fail-closed）。

    ``reason``:
    - ``ok``：接近标定全窗
    - ``dpi_false_half_suspect``：≈标定一半（假半窗嫌疑；须先 ensure_dpi 再复读）
    - ``half_window``：其它不足全窗
    """
    try:
        w = int(width or 0)
        h = int(height or 0)
    except (TypeError, ValueError):
        w = h = 0
    full_ok = w >= int(min_full[0]) and h >= int(min_full[1])
    false_half = looks_like_dpi_false_half(w, h, design=design)
    dpi_ok = dpi_aware_active()
    if full_ok:
        reason = "ok"
        msg = f"客户区{w}x{h}合格"
    elif false_half:
        reason = "dpi_false_half_suspect"
        msg = (
            f"客户区{w}x{h}≈标定{design[0]}x{design[1]}之半；"
            f"疑未开 DPI 假半窗（dpi_aware={dpi_ok}）。"
            "须 ensure_dpi_aware 后重读；勿当真半窗硬停空转。"
        )
    else:
        reason = "half_window"
        msg = f"客户区{w}x{h}不足全窗（须≥{min_full[0]}x{min_full[1]}）"
    return {
        "width": w,
        "height": h,
        "design": [int(design[0]), int(design[1])],
        "full_ok": full_ok,
        "likely_dpi_false_half": false_half,
        "dpi_aware": dpi_ok,
        "reason": reason,
        "message": msg,
    }


def _client_geometry(hwnd: int) -> tuple[int, int, int, int] | None:
    # 默认开 DPI：避免 FindWindow/GetClientRect 在缩放屏上假半窗
    ensure_dpi_aware()
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w, h = right - left, bottom - top
    if w < 100 or h < 100:
        return None
    pt = win32gui.ClientToScreen(hwnd, (0, 0))
    return pt[0], pt[1], w, h


def find_window(title_substr: str) -> WindowInfo | None:
    ensure_dpi_aware()
    if sys.platform != "win32":
        return None
    found: list[WindowInfo] = []

    def enum_cb(hwnd: int, _: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if title_substr not in title:
            return True
        geo = _client_geometry(hwnd)
        if not geo:
            return True
        left, top, w, h = geo
        found.append(WindowInfo(hwnd, title, left, top, w, h))
        return True

    win32gui.EnumWindows(enum_cb, None)
    return found[0] if found else None


def refresh_window(win: WindowInfo) -> WindowInfo:
    """刷新客户区屏幕坐标与尺寸（窗口移动/缩放后必调）。"""
    ensure_dpi_aware()
    if sys.platform != "win32":
        return win
    if not win32gui.IsWindow(win.hwnd):
        return win
    title = win32gui.GetWindowText(win.hwnd) or win.title
    geo = _client_geometry(win.hwnd)
    if not geo:
        return win
    left, top, w, h = geo
    return WindowInfo(win.hwnd, title, left, top, w, h)


def is_foreground(hwnd: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        return int(win32gui.GetForegroundWindow()) == int(hwnd)
    except Exception:
        return False


def restore_no_activate(hwnd: int) -> None:
    """最小化时恢复显示但不抢前台焦点。"""
    if sys.platform != "win32":
        return
    try:
        # SW_SHOWNOACTIVATE = 4
        win32gui.ShowWindow(hwnd, 4)
    except Exception:
        pass


# 抢焦码：锁屏 / IME（勿糊成笼统 game_not_foreground）
LOCK_SCREEN_STEALS = "lock_screen_steals_foreground"
IME_STEALS = "ime_steals_foreground"
# 已熔断：AllowSetForeground + TOPMOST + Switch + 客户区中心点整序
FORBIDDEN_RECLAIM_ALLOWSET_LEGACY = "allowset_topmost_switch_center"


def _hwnd_exe_basename(hwnd: int) -> str:
    """只读：前台进程映像名（失败→空串；无键鼠）。"""
    if sys.platform != "win32" or ctypes is None or not hwnd:
        return ""
    try:
        pid = ctypes.c_ulong(0)
        ctypes.windll.user32.GetWindowThreadProcessId(int(hwnd), ctypes.byref(pid))
        if not int(pid.value or 0):
            return ""
        # PROCESS_QUERY_LIMITED_INFORMATION
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid.value))
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(520)
            size = ctypes.c_uint(520)
            ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(
                h, 0, buf, ctypes.byref(size)
            )
            if not ok:
                return ""
            path = str(buf.value or "")
            return path.rsplit("\\", 1)[-1] if path else ""
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    except Exception:
        return ""


def classify_foreground_blocker(
    *,
    game_foreground: bool,
    fg_title: str = "",
    fg_cls: str = "",
    fg_hwnd: int = 0,
    fg_exe: str = "",
) -> str:
    """纯函数：前台诊断 → 阻塞码（空串=游戏已前台/无阻塞）。

    证据口径：
    - LockApp / 标题含「锁屏」→ ``lock_screen_steals_foreground``（勿并入 IME）
    - Default IME / MSCTFIME /「Windows 默认输入法」+ CoreWindow / TextInputHost
      → ``ime_steals_foreground``
    SendInput 不进游戏客户区（见 dual_entry_focus / DevB L2a LockApp checkpoint）。
    """
    if game_foreground:
        return ""
    title = str(fg_title or "")
    cls = str(fg_cls or "")
    exe = str(fg_exe or "")
    title_u = title.upper()
    cls_u = cls.upper()
    exe_u = exe.upper()
    # 锁屏：先于 IME（「Windows 默认锁屏界面」无「输入」字样，旧码会糊成 game_not_foreground）
    if (
        ("LOCKAPP" in exe_u)
        or ("锁屏" in title)
        or ("LOCK SCREEN" in title_u)
        or ("DEFAULT LOCK" in title_u)
        or (
            "COREWINDOW" in cls_u
            and ("锁屏" in title or "LOCK" in title_u and "IME" not in title_u)
        )
    ):
        return LOCK_SCREEN_STEALS
    if (
        ("输入法" in title)
        or ("输入" in title)
        or ("IME" in title_u)
        or ("MSCTF" in title_u)
        or ("TEXTINPUTHOST" in cls_u)
        or ("MSCTFIME" in cls_u)
    ):
        return IME_STEALS
    # CoreWindow：空标题或含 Input/TextInput → 常见 IME/触控键盘宿主
    if "COREWINDOW" in cls_u and (
        (not title.strip())
        or ("INPUT" in title_u)
        or ("TEXTINPUT" in cls_u)
        or ("WINDOWS.UI.CORE.COREWINDOW" in cls_u and "IME" in title_u)
    ):
        return IME_STEALS
    # 无标题的 Windows.UI.Core.CoreWindow：与 IME 宿主同形，保守判 IME
    if (not title.strip()) and "WINDOWS.UI.CORE.COREWINDOW" in cls_u:
        return IME_STEALS
    if int(fg_hwnd or 0) == 0 and not title and not cls:
        return "no_foreground"
    return "game_not_foreground"


def plan_focus_reclaim_path(
    *,
    blocker: str = "",
    game_foreground: bool = False,
    fg_title: str = "",
    fg_cls: str = "",
    fg_exe: str = "",
    prior_l2a_attempts: int = 0,
) -> dict:
    """新夺焦路径草案（纯函数 · 假快照可测 · 禁实机）。

    与已熔断 ``AllowSetForeground+TOPMOST+Switch+中心点`` 旧序显式区分：
    任何路径的 ``forbidden`` 均含该旧序 id。
    """
    forbidden = [FORBIDDEN_RECLAIM_ALLOWSET_LEGACY, "client_click_while_stolen"]
    if game_foreground or not str(blocker or "").strip():
        return {
            "path_id": "none_needed",
            "allowed": False,
            "actions": [],
            "forbidden": list(forbidden),
            "reason": "already_foreground_or_no_blocker",
            "next_live_ok_when": "game_foreground=true",
            "prior_l2a_attempts": int(prior_l2a_attempts),
        }
    code = str(blocker or "").strip()
    if not code:
        code = classify_foreground_blocker(
            game_foreground=False,
            fg_title=fg_title,
            fg_cls=fg_cls,
            fg_exe=fg_exe,
        )
    if code == LOCK_SCREEN_STEALS:
        return {
            "path_id": "lock_screen_await_human_unlock",
            "allowed": False,
            "actions": [
                "readonly_resnapshot",
                "await_human_unlock",
                "resume_only_when_not_lockapp",
            ],
            "forbidden": list(forbidden)
            + [
                "minimize_lockapp_loop",
                "hide_lockapp_loop",
                "focus_window_allowset_legacy",
            ],
            "reason": LOCK_SCREEN_STEALS,
            "next_live_ok_when": (
                "snapshot_not_lockapp:"
                "exe!=LockApp.exe and title无锁屏 and blocker空"
            ),
            "prior_l2a_attempts": int(prior_l2a_attempts),
            "note": "L2a×2 已熔断；关锁屏后历史 reclaim 曾 true，禁无人循环",
        }
    if code == IME_STEALS:
        attempts = int(prior_l2a_attempts)
        if attempts >= 2:
            return {
                "path_id": "ime_l2a_exhausted",
                "allowed": False,
                "actions": ["readonly_resnapshot", "escalate_l5"],
                "forbidden": list(forbidden) + ["l2a_third_attempt"],
                "reason": IME_STEALS,
                "next_live_ok_when": "new_focus_evidence_or_human",
                "prior_l2a_attempts": attempts,
            }
        return {
            "path_id": "ime_l2a_minimize_then_focus",
            "allowed": True,
            "actions": [
                "minimize_ime_corewindow_once",
                "restore_game_no_allowset_legacy",
                "resnapshot",
            ],
            "forbidden": list(forbidden),
            "reason": IME_STEALS,
            "next_live_ok_when": "game_foreground=true and blocker空",
            "prior_l2a_attempts": attempts,
            "max_attempts": 2,
        }
    return {
        "path_id": "generic_no_auto_reclaim",
        "allowed": False,
        "actions": ["readonly_resnapshot"],
        "forbidden": list(forbidden),
        "reason": code or "game_not_foreground",
        "next_live_ok_when": "game_foreground=true",
        "prior_l2a_attempts": int(prior_l2a_attempts),
    }


def foreground_snapshot(hwnd: int | None = None) -> dict:
    """只读：当前前台窗与游戏是否前台（供闸门/烟测诊断，无键鼠）。"""
    out: dict = {
        "game_hwnd": int(hwnd or 0),
        "game_foreground": False,
        "fg_hwnd": 0,
        "fg_title": "",
        "fg_cls": "",
        "fg_exe": "",
        "blocker": "no_foreground",
    }
    if sys.platform != "win32" or win32gui is None:
        out["blocker"] = "non_win32"
        return out
    try:
        fg = int(win32gui.GetForegroundWindow() or 0)
    except Exception:
        fg = 0
    out["fg_hwnd"] = fg
    if fg:
        try:
            out["fg_title"] = str(win32gui.GetWindowText(fg) or "")[:80]
        except Exception:
            pass
        try:
            out["fg_cls"] = str(win32gui.GetClassName(fg) or "")[:80]
        except Exception:
            pass
        try:
            out["fg_exe"] = str(_hwnd_exe_basename(fg) or "")[:80]
        except Exception:
            pass
    game_fg = bool(hwnd and is_foreground(int(hwnd)))
    out["game_foreground"] = game_fg
    if game_fg:
        out["blocker"] = ""
        return out
    out["blocker"] = classify_foreground_blocker(
        game_foreground=False,
        fg_title=str(out.get("fg_title") or ""),
        fg_cls=str(out.get("fg_cls") or ""),
        fg_hwnd=int(out.get("fg_hwnd") or 0),
        fg_exe=str(out.get("fg_exe") or ""),
    )
    return out


_FOCUS_ADVICE_MAX_AGE_S = 10.0


def _advice_age_s(ts: str) -> float | None:
    """FocusWatch 建议的年龄（秒）；无法解析返回 None。"""
    from datetime import datetime, timezone

    t = (ts or "").strip()
    if not t:
        return None
    try:
        if t.endswith("Z"):
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(t)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return None


def focus_advice_fail_open_reason(
    adv: dict | None, *, max_age_s: float = _FOCUS_ADVICE_MAX_AGE_S
) -> str:
    """无文件 / ok=false / 超龄 → 原因码并 warning；可采信返回空串。

    不改变 fail-open：调用方仍应放行，只让 focus-watch 失效看得见。
    """
    if not adv or not adv.get("ok"):
        why = "missing" if not adv else "ok_false"
        log.warning("focus_advice_stale（fail-open 放行）：%s", why)
        return why
    age = _advice_age_s(str(adv.get("ts") or ""))
    if age is None:
        log.warning("focus_advice_stale（fail-open 放行）：unparsable_ts")
        return "unparsable_ts"
    if age > float(max_age_s):
        log.warning("focus_advice_stale（fail-open 放行）：stale age=%.1fs", age)
        return "stale"
    return ""


def focus_steal_block_reason(*, max_age_s: float = _FOCUS_ADVICE_MAX_AGE_S) -> str:
    """人在用电脑时该让行：返回拦截原因，空串=放行。

    抢前台会 TOPMOST 闪窗并空按 Alt，人正在打字时被切走极其扰人。FocusWatch
    早就把判据写进 ``logs/focus_watch.json``，但抢焦入口从不读它——本闸门补上。

    只采信**新鲜**建议：watcher 没跑或文件过期即放行，否则离线工具全瘫。
    陈旧/缺文件仍 fail-open，但会 ``focus_advice_stale`` warning，避免挂了看不见。
    LIVE 租约活跃、``in_test_mode`` 放行。agent 心跳**不再**一刀切放行（预打点
    会把长挂让行废掉）；自家 sendinput 与人操作的区分由 FocusWatch
    ``decide_advice(..., agent_hb_age_s=...)`` 比时序写进 advice。
    """
    try:
        from mm_assist.safety.lease import current_lease, in_test_mode

        if in_test_mode():
            return ""
        lease = current_lease()
        if lease is not None and lease.alive():
            return ""
    except Exception:
        pass
    try:
        from mm_assist.jobs.focus_watch import (
            DEFAULT_ADVICE_PATH,
            _abs_log_path,
            read_focus_advice,
        )

        advice_path = _abs_log_path(DEFAULT_ADVICE_PATH)
        adv = read_focus_advice(advice_path)
        if focus_advice_fail_open_reason(adv, max_age_s=max_age_s):
            return ""
        if bool(adv.get("may_steal_focus")) or not bool(adv.get("human_busy")):
            return ""
        return str(adv.get("advice") or "human_busy")
    except Exception as exc:  # 判不了就别挡路
        log.debug("focus steal gate skip: %s", exc)
        return ""


def focus_window(hwnd: int, *, force: bool = False) -> bool:
    """尽量抢前台；失败时用 AllowSetForeground / TOPMOST / AttachThreadInput / Alt。

    ``force=False`` 时先过 :func:`focus_steal_block_reason`：人在操作电脑就让行，
    返回 False 让调用方走 soft_skip，而不是把人的画面切走。
    """
    if sys.platform != "win32":
        return False
    if not force:
        blocked = focus_steal_block_reason()
        if blocked:
            log.info("focus_window 让行（人在操作）：%s", blocked)
            return False
    user32 = ctypes.windll.user32
    try:
        # ASFW_ANY=-1：解除本进程前台锁（对 IME 霸占仍可能失败，但无害）
        user32.AllowSetForegroundWindow(-1)
    except Exception:
        pass
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    except Exception:
        pass
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    if is_foreground(hwnd):
        return True
    # TOPMOST 闪一下再卸：有时能从非 IME 前台抢走焦点
    try:
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST,
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW,
        )
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_NOTOPMOST,
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW,
        )
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    if is_foreground(hwnd):
        return True
    # Windows 常拦 SetForegroundWindow：附加到前台线程再试
    try:
        fg = int(user32.GetForegroundWindow() or 0)
        tid_fg = int(user32.GetWindowThreadProcessId(fg, None)) if fg else 0
        tid_tg = int(user32.GetWindowThreadProcessId(int(hwnd), None))
        if tid_fg and tid_tg and tid_fg != tid_tg:
            user32.AttachThreadInput(tid_fg, tid_tg, True)
            try:
                win32gui.BringWindowToTop(hwnd)
                win32gui.SetForegroundWindow(hwnd)
            finally:
                user32.AttachThreadInput(tid_fg, tid_tg, False)
        else:
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    if is_foreground(hwnd):
        return True
    # Alt 空按：解除前台锁后再 SetForeground
    try:
        VK_MENU = 0x12
        KEYUP = 0x0002
        user32.keybd_event(VK_MENU, 0, 0, 0)
        user32.keybd_event(VK_MENU, 0, KEYUP, 0)
        # 这是**真实**键盘注入，会重置 GetLastInputInfo。不打点则下一拍
        # FocusWatch 把自家 Alt 当「有人在操作」→ human_busy 假阳 → 门闩拦点，
        # 拦了又重抢前台，形成自锁环（2026-08-19 夜间 06:00 86/180 即此）。
        try:
            from mm_assist.jobs.focus_watch import stamp_agent_input_heartbeat

            stamp_agent_input_heartbeat(who="focus_alt_tap")
        except Exception:
            pass
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    return is_foreground(hwnd)


def bring_to_foreground(
    hwnd: int, *, retries: int = 6, settle_s: float = 0.12, force: bool = False
) -> bool:
    """多次强制前置游戏窗口；返回是否已是前台。

    人在操作电脑时整轮直接让行——否则 retries 次重试会连按多下 Alt、反复闪窗。
    """
    import time

    if sys.platform != "win32":
        return False
    if not force:
        blocked = focus_steal_block_reason()
        if blocked:
            log.info("bring_to_foreground 让行（人在操作）：%s", blocked)
            return False
    for i in range(max(1, int(retries))):
        focus_window(hwnd, force=True)
        time.sleep(settle_s)
        if is_foreground(hwnd):
            return True
        # 奇数次再试 ShowWindow / SwitchToThisWindow
        if i % 2 == 1:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            except Exception:
                pass
            try:
                ctypes.windll.user32.SwitchToThisWindow(int(hwnd), True)
            except Exception:
                pass
    return is_foreground(hwnd)
