"""实机测试授权租约：人工/到期自动开始、60 秒、fail-closed。

协议版本须与 `.cursor/rules/10-live-test.mdc` / `docs/TESTING.md` 一致。
Agent / scripts/verify 测试入口必须 `assert_protocol_ready()`；无租约不得 SendInput。
用户主动启动的 life_level_loop 通过 `production_mode()` 旁路租约（仍可 F8）。
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from mm_assist.platform.win_spawn import run_hidden

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "LIVE_TEST_PROTOCOL_V2"
DEFAULT_LEASE_S = 60.0
# 兼容：旧 prepare 默认等待秒数（现 prepare 默认 0）
DEFAULT_WAIT_START_TIMEOUT_S = 0.0
# V2：等待「确认测试完毕」默认秒数；到期视为已确认
DEFAULT_FINISH_ACK_TIMEOUT_S = 3.0

# 允许的动作种类（白名单）
ACTION_CLICK = "click"
ACTION_DRAG = "drag"
ACTION_KEY = "key"
ALLOWED_ACTIONS = frozenset({ACTION_CLICK, ACTION_DRAG, ACTION_KEY})


class GuardError(RuntimeError):
    """授权/协议失败：fail-closed。"""


class LeaseExpired(GuardError):
    pass


class NotAuthorized(GuardError):
    pass


class ProtocolNotReady(GuardError):
    pass


@dataclass
class TestLease:
    """一次人工授权的租约。不可自动续期。"""

    session_id: str
    hwnd: int
    intent: str
    actions_plan: tuple[str, ...]
    started_at: float
    lease_s: float = DEFAULT_LEASE_S
    used_transitions: set[str] = field(default_factory=set)

    def remaining_s(self, now: float | None = None) -> float:
        t = time.monotonic() if now is None else now
        return max(0.0, self.started_at + self.lease_s - t)

    def alive(self, now: float | None = None) -> bool:
        return self.remaining_s(now) > 0.0


_lock = threading.RLock()
_lease: TestLease | None = None
_test_mode: bool = False  # True = Agent/verify 测试，强制租约
_production: bool = False  # True = 用户长挂，旁路租约
_lease_scope_active: bool = False
_armed: bool = False  # 租约已武装（prepare auto-arm）
_pending_intent: str = ""
_pending_actions: tuple[str, ...] = ()
_pending_hwnd: int = 0
_pending_lease_s: float = DEFAULT_LEASE_S
_start_event = threading.Event()
_cancel_event = threading.Event()
_session_atexit_registered = False


def _repo_root() -> Path:
    from mm_assist.config import ROOT

    return ROOT


def live_test_registry_path() -> Path:
    """已登记实机测试会话 PID 的仓库内 registry（logs/ 已 gitignore）。"""
    return _repo_root() / "logs" / "live_test_sessions.json"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_pid(pid: int) -> None:
    """终止已登记的测试进程。不按命令行匹配，故不会误杀 life_level_loop。"""
    if pid <= 0 or pid == os.getpid():
        return
    try:
        if sys.platform == "win32":
            run_hidden(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        else:
            os.kill(pid, signal.SIGTERM)
        log.warning("terminated prior live-test pid=%s", pid)
    except Exception as exc:
        log.warning("terminate live-test pid=%s failed: %s", pid, exc)


def _read_session_registry() -> list[dict]:
    path = live_test_registry_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        sessions = data.get("sessions", data if isinstance(data, list) else [])
        return [s for s in sessions if isinstance(s, dict)]
    except Exception:
        return []


def _write_session_registry(sessions: list[dict]) -> None:
    path = live_test_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sessions": sessions, "protocol": PROTOCOL_VERSION}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def claim_test_session() -> None:
    """新会话：杀掉仍存活的其它已登记测试 PID，清理死 PID，登记本进程。

    仅处理 registry 内 PID，绝不按命令行扫描；故不杀未登记的 life_level_loop。
    """
    global _session_atexit_registered
    my_pid = os.getpid()
    kept: list[dict] = []
    for entry in _read_session_registry():
        try:
            pid = int(entry.get("pid", 0))
        except (TypeError, ValueError):
            continue
        if pid == my_pid:
            continue
        if _pid_alive(pid):
            log.warning("claim_test_session: ending prior session pid=%s", pid)
            _terminate_pid(pid)
        # 死进程 / 已杀：不保留
    kept.append({"pid": my_pid, "started_at": time.time()})
    _write_session_registry(kept)
    if not _session_atexit_registered:
        atexit.register(release_test_session)
        _session_atexit_registered = True
    log.info("claim_test_session: registered pid=%s", my_pid)


def release_test_session() -> None:
    """注销本进程 PID（finish / 退出时调用）。"""
    my_pid = os.getpid()
    sessions = []
    for entry in _read_session_registry():
        try:
            pid = int(entry.get("pid", 0))
        except (TypeError, ValueError):
            continue
        if pid != my_pid:
            sessions.append(entry)
    try:
        _write_session_registry(sessions)
    except Exception:
        pass


def reset_guard_state() -> None:
    """单测用：清空全部守卫状态。"""
    global _lease, _test_mode, _production, _armed
    global _pending_intent, _pending_actions, _pending_hwnd, _pending_lease_s
    with _lock:
        _lease = None
        _test_mode = False
        _production = False
        _armed = False
        _pending_intent = ""
        _pending_actions = ()
        _pending_hwnd = 0
        _pending_lease_s = DEFAULT_LEASE_S
        _start_event.clear()
        _cancel_event.clear()


@contextmanager
def bounded_test_lease(*, hwnd: int, intent: str, lease_s: float = 60.0):
    """状态栏单场景测试：执行器的 production_mode 不能绕过这次限时租约。"""
    global _lease_scope_active, _test_mode, _production
    if not 0 < float(lease_s) <= DEFAULT_LEASE_S:
        raise ValueError("test lease must be within 60 seconds")
    with _lock:
        if _lease_scope_active:
            raise GuardError("已有状态栏实机测试")
        _lease_scope_active = True
    timer = None
    try:
        assert_protocol_ready()
        begin_authorization(hwnd=hwnd, intent=intent, actions=(ACTION_CLICK, ACTION_DRAG), lease_s=lease_s)
        signal_user_start()
        def expire():
            # 到期只收缴点击；寻找附近后走路/待机等待不再点，不得因租约 abort。
            revoke("scenario_timeout")
        timer = threading.Timer(float(lease_s), expire)
        timer.daemon = True
        timer.start()
        from mm_assist.safety.readonly import statebar_test_input_scope
        with statebar_test_input_scope():
            yield
    finally:
        if timer is not None:
            timer.cancel()
            timer.join(timeout=1)
        revoke("scenario_finished")
        with _lock:
            _lease_scope_active = False
            _production = False
            _test_mode = False


def enter_test_mode() -> None:
    """标记当前进程为 Agent/verify 实机测试（强制租约）。"""
    global _test_mode, _production
    with _lock:
        _test_mode = True
        _production = False


def production_mode(enabled: bool = True) -> None:
    """用户主动长挂：旁路租约检查（仍受 Abort/F8 约束）。"""
    global _production, _test_mode
    with _lock:
        _production = bool(enabled) and not _lease_scope_active
        if enabled and not _lease_scope_active:
            _test_mode = False
        else:
            # 关 production 时一并关掉「可见鼠标」偏好，避免泄漏到下一段
            global _prefer_visible_clicks
            _prefer_visible_clicks = False


_prefer_visible_clicks = False


def prefer_visible_clicks(enabled: bool | None = None) -> bool:
    """状态栏一键修：要看得到系统鼠标（sendinput），但仍禁止反复抢前台。

    enabled=None → 只读；True/False → 设置并返回当前值。
    """
    global _prefer_visible_clicks
    with _lock:
        if enabled is not None:
            _prefer_visible_clicks = bool(enabled)
        return _prefer_visible_clicks


def in_test_mode() -> bool:
    with _lock:
        return _test_mode and not _production


def in_production_mode() -> bool:
    with _lock:
        return _production


def assert_protocol_ready() -> None:
    """启动自检：版本常量可用即通过；缺模块则导入失败。

    测试入口在任何输入前必须调用。失败 → fail-closed。
    """
    if not PROTOCOL_VERSION or not PROTOCOL_VERSION.startswith("LIVE_TEST_PROTOCOL_"):
        raise ProtocolNotReady(f"协议版本无效: {PROTOCOL_VERSION!r}")
    # 规则文件存在性（仓库根相对）
    try:
        from mm_assist.config import ROOT

        rule = ROOT / ".cursor" / "rules" / "10-live-test.mdc"
        proto = ROOT / "docs" / "TESTING.md"
        if not rule.is_file():
            raise ProtocolNotReady(f"缺少规则文件: {rule}")
        text = rule.read_text(encoding="utf-8")
        if PROTOCOL_VERSION not in text:
            raise ProtocolNotReady(
                f"规则文件未声明 {PROTOCOL_VERSION}；禁止「先跑再补」"
            )
        if not proto.is_file():
            raise ProtocolNotReady(f"缺少协议文档: {proto}")
        ptext = proto.read_text(encoding="utf-8")
        if PROTOCOL_VERSION not in ptext:
            raise ProtocolNotReady(f"TESTING.md 未声明 {PROTOCOL_VERSION}")
    except ProtocolNotReady:
        raise
    except Exception as exc:
        raise ProtocolNotReady(f"协议自检失败: {exc}") from exc
    log.info("protocol ready %s", PROTOCOL_VERSION)


def begin_authorization(
    *,
    hwnd: int,
    intent: str,
    actions: Sequence[str],
    lease_s: float = DEFAULT_LEASE_S,
) -> None:
    """展示计划后调用：等待用户点「开始」前的挂起态。不抢前台、不发输入。"""
    global _armed, _lease, _pending_intent, _pending_actions, _pending_hwnd, _pending_lease_s
    with _lock:
        enter_test_mode()
        _armed = False
        _lease = None
        _pending_hwnd = int(hwnd)
        _pending_intent = str(intent)
        _pending_actions = tuple(str(a) for a in actions)
        _pending_lease_s = float(lease_s)
        _start_event.clear()
        _cancel_event.clear()


def signal_user_start() -> None:
    """武装租约（原 HUD「开始测试」；现由 prepare 自动调用）。"""
    global _armed, _lease
    with _lock:
        if _pending_hwnd <= 0:
            log.warning("signal_user_start without pending auth")
            return
        _armed = True
        _lease = TestLease(
            session_id=str(uuid.uuid4()),
            hwnd=_pending_hwnd,
            intent=_pending_intent,
            actions_plan=_pending_actions,
            started_at=time.monotonic(),
            lease_s=_pending_lease_s,
        )
        log.info(
            "lease armed session=%s hwnd=%s lease_s=%.0f intent=%s",
            _lease.session_id[:8],
            _lease.hwnd,
            _lease.lease_s,
            _lease.intent[:40],
        )
    _start_event.set()


def signal_user_cancel(reason: str = "用户取消") -> None:
    """取消等待或撤权。"""
    revoke(reason)
    _cancel_event.set()
    _start_event.set()  # 解除 wait


def cancel_requested() -> bool:
    return _cancel_event.is_set()


def wait_for_user_start(
    *,
    timeout_s: float | None = None,
    auto_start_on_timeout: bool = False,
) -> TestLease:
    """阻塞直到租约武装或取消。

    auto_start_on_timeout=True（prepare 默认）：超时立刻 ``signal_user_start``；
    F8 / request_stop 仍取消。timeout_s=0 时立即 auto-arm。
    """
    ok = _start_event.wait(timeout=None if timeout_s is None else float(timeout_s))
    if not ok:
        if auto_start_on_timeout and not _cancel_event.is_set():
            log.info(
                "wait_for_user_start: %.1fs → 自动武装",
                float(timeout_s or 0),
            )
            signal_user_start()
        else:
            revoke("等待开始超时")
            raise NotAuthorized("等待测试授权超时")
    if _cancel_event.is_set() and not _armed:
        raise NotAuthorized("用户取消测试")
    with _lock:
        if _lease is None or not _armed:
            raise NotAuthorized("未获得测试授权")
        return _lease


def current_lease() -> TestLease | None:
    with _lock:
        return _lease


def revoke(reason: str = "revoked") -> None:
    """撤租约并抬键。不回调 request_stop，避免与 signal_user_cancel 递归。"""
    global _lease, _armed
    with _lock:
        if _lease is not None:
            log.warning("lease revoke session=%s reason=%s", _lease.session_id[:8], reason)
        _lease = None
        _armed = False
    try:
        from mm_assist.safety.input import release_mouse_buttons

        release_mouse_buttons()
    except Exception:
        pass


# 只有真移系统光标 / 刷 GetLastInputInfo 的路径才预打点。
# unity_hook / postmessage / sendmessage 不刷 LastInput；误 stamp 后
# idle_s + 1.5 < hb_age 约 1.6s 内把真人判成 not busy，闸门放行抢焦。
_STAMP_LAST_INPUT_METHODS = frozenset({"sendinput", "si"})


def _input_method_stamps_last_input(method: str | None) -> bool:
    """未传 method 视为 SendInput（press_vk 等旧调用）。"""
    if method is None:
        return True
    m = str(method).strip().lower()
    if not m:
        return True
    return m in _STAMP_LAST_INPUT_METHODS


def require_action(
    kind: str,
    hwnd: int,
    *,
    transition_id: str | None = None,
    require_foreground: bool = True,
    method: str | None = None,
) -> TestLease | None:
    """输入边界调用。生产模式返回 None；测试模式返回有效租约或抛错。

    ``method``：仅 sendinput/si（及未传，视为 SendInput）预打心跳；
    unity_hook/postmessage/sendmessage 不 stamp。
    """
    global _lease, _armed
    if kind not in ALLOWED_ACTIONS:
        raise GuardError(f"动作不在白名单: {kind}")

    stamp_who: str | None = None
    lease: TestLease | None = None
    with _lock:
        if _production and not _test_mode:
            stamp_who = "production"
        elif not _test_mode:
            # 未显式进测试模式：为防 Agent 误触，默认拒绝真输入
            # 可用环境变量 MM_ALLOW_UNGUARDED=1 仅供紧急调试（文档禁止日常使用）
            if os.environ.get("MM_ALLOW_UNGUARDED", "").strip() in ("1", "true", "yes"):
                log.warning("unguarded sendinput allowed by MM_ALLOW_UNGUARDED")
                stamp_who = "unguarded"
            else:
                raise NotAuthorized(
                    "未进入测试模式且非生产长挂：拒绝 SendInput（fail-closed）"
                )
        else:
            lease = _lease
            if lease is None or not _armed:
                raise NotAuthorized(
                    "无有效租约：须由协议列明的入口建立受限测试租约"
                )
            if not lease.alive():
                _lease = None
                _armed = False
                raise LeaseExpired(f"租约已过期（上限 {lease.lease_s:.0f}s）")
            if int(hwnd) != int(lease.hwnd):
                raise NotAuthorized(
                    f"HWND 不匹配 lease={lease.hwnd} got={hwnd}"
                )
            if transition_id:
                if transition_id in lease.used_transitions:
                    raise GuardError(
                        f"transition 已执行过，禁止重试: {transition_id}"
                    )
                lease.used_transitions.add(transition_id)
            stamp_who = "lease"

    if require_foreground and stamp_who == "lease":
        from mm_assist.platform.window import is_foreground

        if not is_foreground(int(hwnd)):
            revoke("游戏未前台")
            raise NotAuthorized("游戏窗口未前台：请手动切到游戏后再测")
    # sendinput 预打点：否则刷 GetLastInputInfo 后 FocusWatch 误标 human_busy，
    # 无租约闸门让行 → 开图 focus_fail 自锁。非 sendinput 不 stamp。
    if stamp_who and _input_method_stamps_last_input(method):
        try:
            from mm_assist.jobs.focus_watch import stamp_agent_input_heartbeat

            stamp_agent_input_heartbeat(who=stamp_who)
        except Exception:
            pass
    return lease


def mark_transition_used(transition_id: str) -> None:
    with _lock:
        if _lease is not None:
            _lease.used_transitions.add(transition_id)


def lease_remaining_s() -> float:
    with _lock:
        if _lease is None:
            return 0.0
        return _lease.remaining_s()
