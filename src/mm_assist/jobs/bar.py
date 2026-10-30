"""状态栏常驻测试 job：热读 logs/state_bar_job.json 派发动作。

协议：docs/TESTING.md — 实机采证/验收经 watch（内部进程名 state-watch），禁止独立验收主进程。
逃生阀：MM_ALLOW_STANDALONE_LIVE=1（仅本机调试，非验收）。
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

log = logging.getLogger(__name__)

# 空转熔断：协议层（docs/TESTING.md），本模块不实现失败计数器。
# 成对模板 → jobs.pairs。

# B：await_start →（开始采集）queued；await_capture 仅兼容旧 job
# A：本仓禁止；open_test_job 不得直接 queued
JOB_STATUSES = frozenset(
    {
        "idle",
        "await_start",
        "await_capture",
        "queued",
        "running",
        "ok",
        "fail",
    }
)
KNOWN_ACTIONS = frozenset(
    {
        "mem_warm_retry",
        "readonly_snap_ocr",
        "readonly_ui_fine",
        "benchmark_memory",
    }
)
# 本仓只读 profile：不注册任何含键鼠 LIVE action
LIVE_ACTIONS = frozenset()
# B 闸门无自动倒计时（须手点「开始采集」）；字段保留兼容
DEFAULT_AUTO_START_S = 0.0

# 开测前必须写清「需要界面」；缺省时用本表，Agent 仍应显式传入 need_ui
DEFAULT_NEED_UI: dict[str, str] = {
    "mem_warm_retry": "游戏进程在跑即可（只读内存预热，不要求特定 UI）",
    "readonly_snap_ocr": "按本次 title 所指画面摆好（只读截屏/OCR）",
    "readonly_ui_fine": (
        "按本次 title 所指画面摆好（只读：粗四位+MainHud icon→细界面；零键鼠）"
    ),
    "benchmark_memory": "游戏在主界面静止（只读内存基准，零键鼠）",
}

_STANDALONE_MSG = (
    "FAIL-CLOSED: 实机验收须经常驻 state-watch + logs/state_bar_job.json；"
    "勿另起本进程。调试可设 MM_ALLOW_STANDALONE_LIVE=1（非验收）。"
)

# 修理对话菜单旁证（不含奈麗絲名牌：名牌只说明人在附近，不等于已开对话）
REPAIR_DIALOG_HINTS = (
    "跳過",
    "跳过",
    "修理",
    "修復",
    "對話",
    "对话",
    "結束",
    "结束",
    "配戴",
    "佩戴",
)
NERYS_NAMEPLATE_NEEDLES = ("奈麗絲", "奈麗斯", "奈丽丝", "奈麗", "奈丽")
REPAIR_PRE_NAMEPLATE_MISS = "nameplate_miss:need_approach_or_map"
REPAIR_PRE_NAMEPLATE_DEBT = "需先走近/开地图"


def nerys_nameplate_in_text(text: str) -> bool:
    t = str(text or "")
    return any(n in t for n in NERYS_NAMEPLATE_NEEDLES)


def plan_repair_pre_dialog(
    *,
    blob: str = "",
    has_repair_icon: bool = False,
    has_equipped_icon: bool = False,
    nameplate_found: bool = False,
) -> dict[str, Any]:
    """纯函数：修理对话前置。缺名牌且无对话 → fail-closed，禁盲点。"""
    blob_s = str(blob or "")
    dialogish = (
        bool(has_repair_icon)
        or bool(has_equipped_icon)
        or any(k in blob_s for k in REPAIR_DIALOG_HINTS)
    )
    if dialogish:
        return {
            "ok": True,
            "op": "run_chain",
            "allow_click": False,
            "reason": "dialog_visible",
            "fail_tag": "",
            "debt": "",
            "stage": "pre_dialog",
            "error": "",
        }
    if nameplate_found or nerys_nameplate_in_text(blob_s):
        return {
            "ok": True,
            "op": "click_nameplate",
            "allow_click": True,
            "reason": "nameplate_visible",
            "fail_tag": "",
            "debt": "",
            "stage": "pre_dialog",
            "error": "",
        }
    return {
        "ok": False,
        "op": "reject",
        "allow_click": False,
        "reason": "nameplate_miss",
        "fail_tag": REPAIR_PRE_NAMEPLATE_MISS,
        "debt": REPAIR_PRE_NAMEPLATE_DEBT,
        "stage": "pre_dialog",
        "error": (
            "前置证据不足：主界面未见奈麗絲名牌；需先走近或开地图"
        ),
    }

# state-watch 心跳：主循环刷新；开测前 require_state_watch_running()
# 菜单/生活技能等重 tick 可超过 5s；过严会导致 transition-accept 误报「心跳过期」
HEARTBEAT_MAX_AGE_S = 12.0
_HEARTBEAT_NAME = "state_watch.heartbeat.json"
_heartbeat_write_lock = threading.Lock()
# 进程级单例（Windows named mutex）；与 scripts/state_watch_kill.ps1 配合：先杀旧再开新
# 测试可设 MM_STATE_WATCH_MUTEX 隔离
_STATE_WATCH_MUTEX_NAME = os.environ.get(
    "MM_STATE_WATCH_MUTEX", "Local\\mm_assist_state_watch_singleton"
)
_state_watch_mutex_handle: Any | None = None


def try_acquire_state_watch_singleton() -> bool:
    """同一时刻只允许一个 state-watch。成功持锁则进程内常驻；失败返回 False。

    Windows：CreateMutex；非 Windows：退化用心跳 pid（尽力而为）。
    """
    global _state_watch_mutex_handle
    if _state_watch_mutex_handle is not None:
        return True
    name = os.environ.get("MM_STATE_WATCH_MUTEX", _STATE_WATCH_MUTEX_NAME)
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        # bInitialOwner=True：创建即持有
        handle = kernel32.CreateMutexW(None, True, name)
        if not handle:
            return False
        ERROR_ALREADY_EXISTS = 183
        if int(kernel32.GetLastError()) == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        _state_watch_mutex_handle = handle
        return True
    # 非 Windows：心跳 pid 仍存活则视为已有实例
    try:
        raw = heartbeat_path().read_text(encoding="utf-8")
        data = json.loads(raw)
        pid = int(data.get("pid") or 0)
        if pid > 0 and pid != os.getpid() and _pid_alive(pid):
            return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    _state_watch_mutex_handle = True  # type: ignore[assignment]
    return True


def release_state_watch_singleton_for_tests() -> None:
    """仅测试用：释放单例互斥，避免污染后续用例。"""
    global _state_watch_mutex_handle
    h = _state_watch_mutex_handle
    _state_watch_mutex_handle = None
    if sys.platform == "win32" and h not in (None, True):
        try:
            ctypes.windll.kernel32.CloseHandle(h)
        except Exception:
            pass


def _repo_root() -> Path:
    from mm_assist.config import ROOT

    return ROOT


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def job_path(root: Path | None = None) -> Path:
    base = root or _repo_root()
    return base / "logs" / "state_bar_job.json"


def heartbeat_path(root: Path | None = None) -> Path:
    base = root or _repo_root()
    return base / "logs" / _HEARTBEAT_NAME


def standalone_live_allowed() -> bool:
    return os.environ.get("MM_ALLOW_STANDALONE_LIVE", "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    )


def assert_standalone_live_allowed() -> None:
    """独立验收主入口门控；未开逃生阀则 SystemExit。"""
    if standalone_live_allowed():
        return
    raise SystemExit(_STANDALONE_MSG)


def write_state_watch_heartbeat(
    *,
    pid: int | None = None,
    root: Path | None = None,
) -> Path:
    """state-watch 启动/主循环调用：写 logs/state_watch.heartbeat.json。"""
    p = heartbeat_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": int(pid if pid is not None else os.getpid()),
        "updated_at": _utc_now(),
        "updated_at_unix": time.time(),
    }
    tmp = p.with_suffix(".tmp")
    # 主循环和独立心跳线程共用文件，必须把写临时文件和替换作为一个操作。
    with _heartbeat_write_lock:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(p)
    return p


def clear_state_watch_heartbeat(root: Path | None = None) -> None:
    p = heartbeat_path(root)
    try:
        p.unlink(missing_ok=True)
    except OSError as exc:
        log.debug("clear heartbeat: %s", exc)


def _pid_alive(pid: int) -> bool:
    """心跳 pid 是否仍存活（防僵死文件误放行）。"""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # PROCESS_QUERY_LIMITED_INFORMATION
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def click_owner_block_reason() -> str:
    """状态栏心跳进程或 10h 泵持有键鼠时，其它进程/线程不得点。pytest 放行。"""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return ""
    hb_pid = 0
    try:
        data = json.loads(heartbeat_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            hb_pid = int(data.get("pid") or 0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        hb_pid = 0
    me = os.getpid()
    if hb_pid > 0 and hb_pid != me and _pid_alive(hb_pid):
        return f"状态栏 pid={hb_pid} 持有键鼠，外进程禁点"
    try:
        from mm_assist.life_plan.bar_ui import pump_is_current_thread, pump_thread_alive
    except Exception:
        return ""
    if pump_thread_alive() and not pump_is_current_thread():
        return "10h 泵持有键鼠，其它线程禁点"
    return ""


def require_state_watch_running(
    *,
    max_age_s: float = HEARTBEAT_MAX_AGE_S,
    root: Path | None = None,
) -> dict[str, Any]:
    """开测前确认 state-watch 在跑；无心跳/过期/pid 已死则 raise RuntimeError。

    返回心跳 payload（含 pid / updated_at / age_s）。
    """
    p = heartbeat_path(root)
    if not p.is_file():
        raise RuntimeError(
            "FAIL: state-watch 未运行（无 logs/state_watch.heartbeat.json）；"
            "请先：python -m mm_assist spawn-hidden -- watch --interval 0.5"
        )
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"FAIL: state-watch 心跳不可读: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("FAIL: state-watch 心跳格式无效")
    try:
        ts = float(raw.get("updated_at_unix") or 0.0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("FAIL: state-watch 心跳缺 updated_at_unix") from exc
    age = time.time() - ts
    if age > float(max_age_s):
        raise RuntimeError(
            f"FAIL: state-watch 心跳过期 age={age:.1f}s > {max_age_s}s；"
            "请确认 state-watch 仍在跑"
        )
    try:
        pid = int(raw.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if not _pid_alive(pid):
        raise RuntimeError(
            f"FAIL: state-watch 心跳 pid={pid} 已不存在（僵死文件）；"
            "请重启：python -m mm_assist spawn-hidden -- watch --interval 0.5"
        )
    out = dict(raw)
    out["age_s"] = age
    return out


def empty_job(*, action: str = "", method: str = "B") -> dict[str, Any]:
    return {
        "id": _utc_now(),
        "action": action,
        "method": method,
        "args": {},
        "status": "idle",
        "result": None,
        "error": None,
        "auto_start_s": DEFAULT_AUTO_START_S,
        "title": "",
        "need_ui": "",
        "updated_at": _utc_now(),
    }


def resolve_need_ui(action: str, need_ui: str | None = None) -> str:
    """开测前「需要界面」文案；显式优先，否则用 DEFAULT_NEED_UI。"""
    text = str(need_ui or "").strip()
    if text:
        return text
    return str(DEFAULT_NEED_UI.get(str(action or "").strip()) or "").strip()


def open_test_job(
    action: str,
    *,
    method: str = "B",
    args: dict[str, Any] | None = None,
    title: str = "",
    need_ui: str = "",
    auto_start_s: float = DEFAULT_AUTO_START_S,
    path: Path | None = None,
) -> dict[str, Any]:
    """打开只读采证 job：本仓只允许 method=B。

    B：``await_start``，状态栏【开始采集】后再 queued。
    禁止盲 Sleep 代替按钮。方法 A / LIVE 动作一律拒绝。
    """
    action = str(action or "").strip()
    if action not in KNOWN_ACTIONS:
        raise ValueError(f"未知 action={action!r}；已知 {sorted(KNOWN_ACTIONS)}")
    if action in LIVE_ACTIONS:
        raise ValueError(f"FAIL-CLOSED: 只读 profile 禁止 LIVE action={action!r}")
    need = resolve_need_ui(action, need_ui)
    if not need:
        raise ValueError(
            f"open_test_job 须标明 need_ui（需要的界面状态）；action={action!r}"
        )
    args_out = dict(args or {})
    if "pair_id" in args_out and args_out["pair_id"] is not None:
        args_out["pair_id"] = str(args_out["pair_id"])
    if "pair_role" in args_out and args_out["pair_role"] is not None:
        args_out["pair_role"] = str(args_out["pair_role"])
    method_u = str(method or "B").strip().upper() or "B"
    if method_u != "B":
        raise ValueError(
            f"FAIL-CLOSED: 只读 profile 禁止 method={method_u!r}；仅允许 method='B'"
        )
    job = empty_job(action=action, method=method_u)
    job["status"] = "await_start"
    job["args"] = args_out
    try:
        job["auto_start_s"] = float(auto_start_s)
    except (TypeError, ValueError):
        job["auto_start_s"] = DEFAULT_AUTO_START_S
    if method_u == "B":
        job["auto_start_s"] = 0.0
    job["title"] = str(title or action)
    job["need_ui"] = need
    job["result"] = None
    job["error"] = None
    save_job(job, path)
    return job


def ui_fine_sample_confidence(
    *,
    ui_params: str | None,
    icon_errors: list[str] | None = None,
    icons: list[Any] | None = None,
    stop_visible: bool | None = None,
    pair_role: str | None = None,
    need_icon_corroboration: bool = True,
    panel_probe: dict[str, Any] | None = None,
) -> Literal["high", "low"]:
    """只读采证置信度（B3）：低置信勿当有效 / 勿过称够用。

    low 任一：params 空；icon errors 含 ``wall_timeout``；icons 空且需要
    icon 旁证；有 pair_role 时与预期冲突。

    地图/人物：主栏已撤 map=/prf=（REQ_UI_PARAMS），成对看 ``panel_probe``。
    """
    params = str(ui_params or "").strip()
    if not params or params == "—":
        return "low"
    errs = list(icon_errors or [])

    role = str(pair_role or "").strip().lower()
    # 地图/人物/制作/对话成对：主判在 panel_probe，不强制 MainHud icons
    panel_roles = (
        "map_open",
        "map_on",
        "map_closed",
        "map_off",
        "profile_open",
        "craft_open",
        "craft_closed",
        "dlg_open",
        "dlg_closed",
    )
    if role in panel_roles:
        need_icon_corroboration = False
    elif any("wall_timeout" in str(e) for e in errs):
        return "low"
    if need_icon_corroboration and not icons:
        return "low"
    if not role:
        return "high"

    probe = panel_probe or {}

    # 有预期 role：与 stop / 战斗·地图·人物探针冲突 → low
    if role == "moving":
        if stop_visible is not True:
            return "low"
    elif role == "stopped":
        if stop_visible is not False:
            return "low"
    elif role == "combat":
        if "cbt=1" not in params:
            return "low"
    elif role == "noncombat":
        if "cbt=1" in params:
            return "low"
        if "cbt=0" not in params:
            return "low"
    elif role in ("map_open", "map_on"):
        if probe:
            if probe.get("map_open") is not True:
                return "low"
            if probe.get("profile_open") is True:
                return "low"
        elif "map=1" not in params:
            return "low"
        elif "prf=1" in params:
            return "low"
    elif role in ("map_closed", "map_off"):
        if probe:
            if probe.get("map_open") is not False:
                return "low"
        elif "map=0" not in params:
            return "low"
    elif role == "profile_open":
        if probe:
            if probe.get("profile_open") is not True:
                return "low"
            if probe.get("map_open") is True:
                return "low"
            live_n = probe.get("profile_live_n")
            if live_n is not None:
                try:
                    if int(live_n) < 2:
                        return "low"
                except (TypeError, ValueError):
                    return "low"
        else:
            if "prf=1" not in params:
                return "low"
            if "map=1" in params:
                return "low"
    elif role == "craft_open":
        hub = probe.get("craft_hub_category")
        mem_ok = False
        try:
            mem_ok = hub is not None and int(hub) in (1, 2, 3)
        except (TypeError, ValueError):
            mem_ok = False
        if not mem_ok:
            # 内存空：可信图像 fail-open（模板/六页签）；OCR 旁证仍 low
            sig = (
                probe.get("craft_open_signal")
                if isinstance(probe.get("craft_open_signal"), dict)
                else {}
            )
            if not (sig.get("open") is True and sig.get("trusted") is True):
                return "low"
    elif role == "craft_closed":
        hub = probe.get("craft_hub_category")
        if hub is not None:
            try:
                if int(hub) in (1, 2, 3):
                    return "low"
            except (TypeError, ValueError):
                pass
        sig = (
            probe.get("craft_open_signal")
            if isinstance(probe.get("craft_open_signal"), dict)
            else {}
        )
        if sig.get("open") is True and sig.get("trusted") is True:
            return "low"
    elif role == "dlg_open":
        names = [str(x) for x in (probe.get("ui_stack_names") or [])]
        if probe.get("dialog_active") is not True and "DialogueInteractionUIState" not in names:
            return "low"
    elif role == "dlg_closed":
        names = [str(x) for x in (probe.get("ui_stack_names") or [])]
        if probe.get("dialog_active") is True or "DialogueInteractionUIState" in names:
            return "low"
    return "high"


def advance_job_gate(
    *,
    to: str,
    path: Path | None = None,
    expected_id: str | None = None,
) -> dict[str, Any]:
    """状态栏按钮推进 B 闸门（docs/TESTING.md：先确认界面再采集）。

    - ``await_capture``：await_start → 旧两步闸门兼容
    - ``queued`` / ``manual``：await_start 或 await_capture → 开始采集
    - ``idle``：等待/执行中/完成 → 停止
    """
    to = str(to or "").strip()
    job = load_job(path)
    if expected_id is not None and str(job.get("id") or "") != str(expected_id):
        return job
    st = str(job.get("status") or "idle")
    if to == "await_capture":
        if st != "await_start":
            return job
        job["status"] = "await_capture"
    elif to in ("queued", "manual"):
        if st not in ("await_start", "await_capture"):
            return job
        job["status"] = "queued"
    elif to == "idle":
        if st not in (
            "await_start",
            "await_capture",
            "queued",
            "running",
            "ok",
            "fail",
        ):
            return job
        job["status"] = "idle"
    elif to == "await_start":
        return job
    else:
        raise ValueError(f"非法 gate to={to!r}")
    save_job(job, path)
    return job


def load_job(path: Path | None = None) -> dict[str, Any]:
    p = path or job_path()
    if not p.is_file():
        return empty_job()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("state_bar_job load fail: %s", exc)
        return empty_job()
    if not isinstance(raw, dict):
        return empty_job()
    job = empty_job()
    job.update({k: raw.get(k, job.get(k)) for k in job})
    # 兼容旧文件多出来的字段
    for extra in (
        "result",
        "error",
        "auto_start_s",
        "title",
        "need_ui",
        "id",
        "action",
        "method",
    ):
        if extra in raw and extra not in job:
            job[extra] = raw[extra]
    job["args"] = raw.get("args") if isinstance(raw.get("args"), dict) else {}
    job["need_ui"] = str(raw.get("need_ui") or job.get("need_ui") or "")
    try:
        job["auto_start_s"] = float(raw.get("auto_start_s", job.get("auto_start_s") or DEFAULT_AUTO_START_S))
    except (TypeError, ValueError):
        job["auto_start_s"] = DEFAULT_AUTO_START_S
    job["title"] = str(raw.get("title") or job.get("title") or "")
    st = str(job.get("status") or "idle")
    if st not in JOB_STATUSES:
        job["status"] = "idle"
    return job


def save_job(job: dict[str, Any], path: Path | None = None) -> Path:
    p = path or job_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(job)
    out["updated_at"] = _utc_now()
    text = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    tmp = p.with_suffix(".tmp")
    last_exc: OSError | None = None
    for attempt in range(6):
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(p)
            return p
        except OSError as exc:
            # WinError 5：另一端正读 json（JobWatcher/轮询）时 replace 可能拒写
            last_exc = exc
            time.sleep(0.02 * (attempt + 1))
    try:
        p.write_text(text, encoding="utf-8")
        return p
    except OSError:
        if last_exc is not None:
            raise last_exc
        raise


def claim_queued(job: dict[str, Any]) -> dict[str, Any] | None:
    """若 status=queued 则改为 running 并返回副本；否则 None。"""
    if str(job.get("status") or "") != "queued":
        return None
    action = str(job.get("action") or "").strip()
    if not action:
        job = dict(job)
        job["status"] = "fail"
        job["error"] = "queued 但缺 action"
        return job
    out = dict(job)
    out["status"] = "running"
    out["error"] = None
    return out


def finish_job(
    job: dict[str, Any],
    *,
    ok: bool,
    result: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    out = dict(job)
    out["status"] = "ok" if ok else "fail"
    out["result"] = result
    out["error"] = None if ok else (error or "fail")
    return out


@dataclass
class JobContext:
    """state-watch 注入的运行时句柄。"""

    cfg: Any
    win: Any
    mon: Any = None
    equip_r: Any = None
    fac_r: Any = None
    hud: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


def _action_mem_warm_retry(ctx: JobContext, args: dict[str, Any]) -> dict[str, Any]:
    del args
    if ctx.mon is None:
        return {"ok": False, "error": "无 SceneMonitor"}
    win = ctx.win
    errors: list[str] = []
    try:
        warm = ctx.mon._scene_mem.read(hwnd=win.hwnd, allow_scan=True)
        status = getattr(warm, "status", None)
        if status not in ("ok", "cached", True) and getattr(warm, "error", None):
            errors.append(str(getattr(warm, "error", status)))
    except Exception as exc:
        errors.append(f"scene:{exc}")
    if ctx.equip_r is not None:
        try:
            es = ctx.equip_r.read(allow_scan=True, enrich=False)
            if es and ctx.hud is not None:
                ctx.hud.push_equip(es)
            if es and es.error:
                errors.append(f"equip:{es.error}")
        except Exception as exc:
            errors.append(f"equip:{exc}")
    if ctx.fac_r is not None:
        try:
            fs = ctx.fac_r.read(allow_scan=True)
            if fs and ctx.hud is not None:
                ctx.hud.push_facility(fs)
            if fs and fs.error:
                errors.append(f"facility:{fs.error}")
        except Exception as exc:
            errors.append(f"facility:{exc}")
    if errors:
        return {"ok": False, "error": "; ".join(errors)[:240], "detail": errors}
    return {"ok": True, "detail": "mem_warm_ok"}


def _action_benchmark_memory(ctx: JobContext, args: dict[str, Any]) -> dict[str, Any]:
    from mm_assist.benchmarks.memory import run_live_memory_benchmark

    hwnd = int(getattr(ctx.win, "hwnd", 0) or 0)
    warmup = int(args.get("warmup") or 2)
    repeats = int(args.get("repeats") or 8)
    include_heap = bool(args.get("include_heap", True))
    usb_safe = bool(args.get("usb_safe", True))
    out = run_live_memory_benchmark(
        hwnd=hwnd,
        warmup=warmup,
        repeats=repeats,
        include_heap=include_heap,
        usb_safe=usb_safe,
    )
    return {"ok": True, "detail": out}


def _action_readonly_snap_ocr(ctx: JobContext, args: dict[str, Any]) -> dict[str, Any]:
    del args
    from mm_assist.perception.scene import skip_fullframe_ocr_on_main_hud

    ms = getattr(getattr(ctx, "mon", None), "last_ms", None)
    if skip_fullframe_ocr_on_main_hud(ms):
        return {
            "ok": False,
            "error": "主界面(menu=0)禁止全屏 OCR；用 tick_lean/内存四参或单入口 ROI（hud_entry vision）",
        }
    from mm_assist.platform.capture import configure_grabber
    from mm_assist.perception.list_ocr import normalize_text
    from mm_assist.perception.map_classify import _blob, snap
    from mm_assist.platform.grab import prepare_grabber

    prepare_grabber(ctx.cfg)
    configure_grabber(ctx.cfg.window.capture_method)
    win, _img, lines, _ = snap(ctx.cfg, ctx.win)
    blob = _blob(lines)
    sample = normalize_text(blob)[:200]
    return {
        "ok": True,
        "detail": {
            "w": win.width,
            "h": win.height,
            "n_lines": len(lines),
            "blob_sample": sample,
        },
    }


def _action_readonly_ui_fine(ctx: JobContext, args: dict[str, Any]) -> dict[str, Any]:
    """只读：粗四位 + MainHud icon → 细界面判态（零键鼠）。

    必须复用 state-watch 的 SceneMonitor：冷 Il2CppSceneReader + allow_scan=False
    会永远 ms=None（ui_params=— / 未知）；另起 icon reader 还会与 lean 后台叠扫。

    result 含 confidence（B3）与可选 pair_id/pair_role（B2）；证据包字段见 detail（B5）。
    icon wall_timeout → confidence=low，不阻塞粗四位+stp 落盘。
    """
    if str(args.get("sense") or "") == "vision" and str(args.get("entry_id") or "").strip():
        from mm_assist.jobs.hud_entries import run_hud_entry_vision

        return run_hud_entry_vision(ctx, args)

    pair_id = args.get("pair_id")
    pair_role = args.get("pair_role")
    from mm_assist.memory.readers.hud_action import Il2CppHudActionReader
    from mm_assist.memory.readers.hud_icons import Il2CppHudMainIconsReader
    from mm_assist.perception.scene import format_ui_params, format_ui_relation, classify_ui_mode
    from mm_assist.perception.ui_fine import (
        HudIconSummary,
        classify_fine_ui,
        format_fine_ui,
        merge_stop_visible,
        pixel_stop_visible,
        summarize_hud_action,
        summarize_hud_icons,
        verdict_to_dict,
    )

    if ctx.mon is None:
        return {"ok": False, "error": "无 SceneMonitor（readonly_ui_fine 须经 state-watch 派发）"}

    cw = int(getattr(ctx.win, "width", 1920) or 1920)
    ch = int(getattr(ctx.win, "height", 1440) or 1440)
    hwnd = getattr(ctx.win, "hwnd", None)

    role_early = str(args.get("pair_role") or "").strip().lower()
    node_early = str(args.get("node_id") or "").strip()
    scene = ctx.mon._scene_mem
    # 地图/人物成对：开图会新建 MapUIVisible；lean 禁扫会永远 map_open=False。
    # 仅这些 role 同步补扫少量类型（墙钟可控），再 lean 读。
    if role_early in (
        "map_open",
        "map_on",
        "profile_open",
        "map_closed",
        "map_town",
        "map_region",
        "map_continent",
        "map_dungeon",
        "map_moongate",
        "craft_open",
        "craft_closed",
        "dlg_open",
        "dlg_closed",
    ) or node_early == "map_dunbarton":
        try:
            from mm_assist.memory.readers.scene import (
                _PAIR_UI_SYNC_LIMITS,
                _PAIR_UI_SYNC_WALL_S,
            )

            # 先只扫 MapUIVisible（开图新建实例常要 20–40s；勿全堆 allow_scan，易超租约）
            if role_early.startswith("map") or node_early == "map_dunbarton" or role_early in (
                "profile_open",
                "map_on",
                "map_off",
            ):
                scene._sync_rescan_types(
                    {"MapUIVisibleComponent": 192},
                    max_wall_s=40.0,
                )
            scene._sync_rescan_types(
                {
                    k: v
                    for k, v in dict(_PAIR_UI_SYNC_LIMITS).items()
                    if k != "MapUIVisibleComponent"
                },
                max_wall_s=max(10.0, float(_PAIR_UI_SYNC_WALL_S)),
            )
        except Exception:
            pass
        ms = scene.read(
            hwnd=hwnd, allow_scan=False, refresh_ui=False, lean_ui_refresh=True
        )
        # 仍空：再单扫 MapUIVisible 一轮（仍禁全堆）
        if role_early.startswith("map") and (
            ms is None or getattr(ms, "map_open", None) is None
        ):
            try:
                scene._sync_rescan_types(
                    {"MapUIVisibleComponent": 192}, max_wall_s=25.0
                )
                ms = scene.read(
                    hwnd=hwnd, allow_scan=False, refresh_ui=False, lean_ui_refresh=True
                )
            except Exception:
                pass
    else:
        # 与 lean tick 同路：已 warm 的 _scene_mem，禁止同步堆扫
        ms = scene.read(
            hwnd=hwnd, allow_scan=False, refresh_ui=False, lean_ui_refresh=False
        )

    summary: HudIconSummary | None = None
    icon_src = "none"
    icon_cache_age_s: float | None = None
    icon_lock = getattr(ctx.mon, "_icon_lock", None)
    # 采证要求近鲜：墙钟 >3s 或单次扫描 >5s 的缓存不可信
    max_cache_age_s = 3.0
    max_scan_ms = 5000.0

    def _cache_usable(s: HudIconSummary | None, last_ts: float) -> bool:
        if s is None or not s.has_any():
            return False
        # 至少一枚 Show/Hide/Combat；Invalid 不算
        usable_btn = False
        for v in (s.menu_esc, s.inventory):
            if v is not None and v.state in ("Show", "Hide", "Combat"):
                usable_btn = True
                break
        if not usable_btn:
            return False
        age = time.time() - float(last_ts or 0.0)
        if age < 0 or age > max_cache_age_s:
            return False
        if float(s.age_ms or 0.0) > max_scan_ms:
            return False
        return True

    # 与 lean 后台串行：busy 时等；缓存过期则强制 capture
    wait_deadline = time.time() + 30.0
    while time.time() < wait_deadline:
        busy = False
        last_ts = 0.0
        if icon_lock is not None:
            with icon_lock:
                busy = bool(getattr(ctx.mon, "_icon_busy", False))
                cached = getattr(ctx.mon, "_icon_summary", None)
                last_ts = float(getattr(ctx.mon, "_icon_last_ts", 0.0) or 0.0)
                if _cache_usable(cached, last_ts):
                    summary = cached
                    icon_src = "mon_cache"
                    icon_cache_age_s = time.time() - last_ts
        else:
            cached = getattr(ctx.mon, "_icon_summary", None)
            last_ts = float(getattr(ctx.mon, "_icon_last_ts", 0.0) or 0.0)
            if _cache_usable(cached, last_ts):
                summary = cached
                icon_src = "mon_cache"
                icon_cache_age_s = time.time() - last_ts
        if summary is not None:
            break
        if busy:
            time.sleep(0.25)
            continue
        break

    if summary is None:
        icons_r = getattr(ctx.mon, "_icon_reader", None)
        if icons_r is None:
            icons_r = Il2CppHudMainIconsReader(max_per_kind=8)
            ctx.mon._icon_reader = icons_r
        if icon_lock is not None:
            with icon_lock:
                ctx.mon._icon_busy = True
        try:
            frame = icons_r.capture(
                client_w=cw, client_h=ch, need_rect=False, max_wall_s=2.5
            )
            summary = summarize_hud_icons(frame)
            icon_src = "capture"
            icon_cache_age_s = 0.0
            if icon_lock is not None:
                with icon_lock:
                    ctx.mon._icon_summary = summary
                    ctx.mon._icon_last_ts = time.time()
        finally:
            if icon_lock is not None:
                with icon_lock:
                    ctx.mon._icon_busy = False

    assert summary is not None
    verdict = classify_fine_ui(ms, summary)
    ui, detail, lv, geo, mem_ui, mem_geo, ui_src = classify_ui_mode(ms, img=None)

    # 动作中采证：扩大堆扫 + 扫满（不早停）；独立 reader，不占 _icon_busy。
    action = None
    action_src = "none"
    action_age_ms: float | None = None

    # 等 icon 后台让路（最多 ~2s）；不等 action_busy 死等
    wait_act = time.time() + 2.0
    while time.time() < wait_act:
        busy = False
        if icon_lock is not None:
            with icon_lock:
                busy = bool(getattr(ctx.mon, "_icon_busy", False))
        else:
            busy = bool(getattr(ctx.mon, "_icon_busy", False))
        if not busy:
            break
        time.sleep(0.1)

    sample_ar = Il2CppHudActionReader(max_ui=24, max_buttons=128)
    try:
        aframe = sample_ar.capture(
            client_w=cw,
            client_h=ch,
            need_rect=False,
            max_wall_s=8.0,
            stop_on_strong=False,
        )
        if aframe is not None:
            action = summarize_hud_action(aframe)
            action_src = "capture"
            action_age_ms = float(getattr(aframe, "age_ms", 0.0) or 0.0)
            if icon_lock is not None:
                with icon_lock:
                    ctx.mon._action_summary = action
                    ctx.mon._action_last_ts = time.time()
            else:
                ctx.mon._action_summary = action
                ctx.mon._action_last_ts = time.time()
    except Exception:
        action = None
    finally:
        try:
            sample_ar.close()
        except Exception:
            pass

    if action is None:
        if icon_lock is not None:
            with icon_lock:
                action = getattr(ctx.mon, "_action_summary", None)
        else:
            action = getattr(ctx.mon, "_action_summary", None)
        if action is not None:
            action_src = "mon_cache"

    mem_stop = getattr(action, "stop_visible", None) if action else None
    # 内存无读数再降级：绿圆+白方停 / stop.png 模板 / ROI OCR
    travel_px: str | None = None
    stop_src = "none"
    if mem_stop is not True and mem_stop is not False:
        img = None
        try:
            from mm_assist.perception.hud_sense import detect_stop_icon

            if ctx.mon is not None and getattr(ctx.mon, "frame", None) is not None:
                try:
                    img = ctx.mon.frame.grab(ctx.cfg) if ctx.cfg is not None else None
                except Exception:
                    img = None
            if img is None and ctx.win is not None and ctx.cfg is not None:
                try:
                    from mm_assist.perception.game_frame import ensure_game_frame

                    fr = ensure_game_frame(ctx.cfg, ctx.win, persist=False)
                    img = fr.grab(ctx.cfg)
                except Exception:
                    img = None
            icon_hit = detect_stop_icon(img, ocr_space=True) if img is not None else None
            if icon_hit is True:
                travel_px = "stop_icon"
            elif icon_hit is False:
                travel_px = "no_stop_icon"
        except Exception:
            travel_px = None
    pix_stop = pixel_stop_visible(travel_px)
    merged_stop, stop_src = merge_stop_visible(mem_stop, pix_stop)
    if action is not None:
        action.stop_visible = merged_stop
    elif merged_stop is not None:
        from mm_assist.perception.ui_fine import HudActionSummary

        action = HudActionSummary(stop_visible=merged_stop)
        action_src = action_src if action_src != "none" else "pixel_only"

    icon_rows: list[dict[str, Any]] = []
    for key, view in (
        ("esc", summary.menu_esc),
        ("inv", summary.inventory),
        ("lb", summary.left_bottom),
        ("cam", summary.camera),
    ):
        if view is None:
            continue
        icon_rows.append(
            {
                "key": key,
                "state": view.state,
                "interactable": view.interactable,
                "short": view.short(),
            }
        )

    stop_vis = merged_stop
    ui_params = format_ui_params(ms, icons=summary, action=action)
    icon_errors = list(summary.errors)
    role_s = str(pair_role) if pair_role is not None else None
    # 成对重找落盘（≠上栏、≠可用）：战斗 + 地图/人物候选 + 扩扫抽样
    panel_probe: dict[str, Any] = {
        "combat_mode_ui": getattr(ms, "combat_mode_ui", None) if ms else None,
        "combat_state": getattr(ms, "combat_state", None) if ms else None,
        "in_combat": getattr(ms, "in_combat", None) if ms else None,
        "menu_open": getattr(ms, "menu_open", None) if ms else None,
        "dialog_active": getattr(ms, "dialog_active", None) if ms else None,
        "ui_stack_names": list(getattr(ms, "ui_stack_names", ()) or []) if ms else [],
        "map_open": getattr(ms, "map_open", None) if ms else None,
        "map_type": getattr(ms, "map_type", None) if ms else None,
        "map_type_name": getattr(ms, "map_type_name", None) if ms else None,
        "minimap_visible": getattr(ms, "minimap_visible", None) if ms else None,
        "profile_open": getattr(ms, "profile_open", None) if ms else None,
        "profile_live_n": getattr(ms, "profile_live_n", None) if ms else None,
        "craft_hub_category": getattr(ms, "craft_hub_category", None) if ms else None,
        "side_nav_open": getattr(ms, "side_nav_open", None) if ms else None,
    }
    try:
        panel_probe["map_vis_raw"] = scene._flag_or(
            "MapUIVisibleComponent", "MapUIVisible.IsVisible"
        )
        panel_probe["np_live_tops"] = scene.dump_ui_state_live_tops(
            "NewProfileUIState", limit=16
        )
        panel_probe["craft_hub_live"] = scene.dump_ui_state_live_tops(
            "CraftingHubUIState", limit=16
        )
        panel_probe["craft_hub_diag"] = scene.craft_hub_diag()
        panel_probe["obj_n"] = {
            t: len(scene._objs.get(t) or [])
            for t in (
                "MapUIVisibleComponent",
                "NewProfileUIState",
                "SideNavigationUIState",
                "CraftingHubUIState",
            )
        }
    except Exception:
        pass

    # 製作成对 / hub 空：内存优先后 fail-open 图像模板（OCR 仅旁证）
    need_craft_visual = role_s in ("craft_open", "craft_closed", "craft_tabs") or (
        panel_probe.get("menu_open") is True
        and panel_probe.get("craft_hub_category") is None
    )
    if need_craft_visual:
        try:
            from mm_assist.safety.autofarm_gates import (
                probe_craft_visual_open,
                resolve_craft_open_signal,
            )
            from mm_assist.perception.ui_fine import FineUiVerdict, FINE_LABELS

            grab = img
            if grab is None and ctx.mon is not None and getattr(ctx.mon, "frame", None) is not None:
                try:
                    grab = ctx.mon.frame.grab(ctx.cfg) if ctx.cfg is not None else None
                except Exception:
                    grab = None
            if grab is None and ctx.win is not None and ctx.cfg is not None:
                try:
                    from mm_assist.perception.game_frame import ensure_game_frame

                    grab = ensure_game_frame(ctx.cfg, ctx.win, persist=False).grab(ctx.cfg)
                except Exception:
                    grab = None
            vis = probe_craft_visual_open(
                grab, menu_open=panel_probe.get("menu_open"), ocr_blob=""
            )
            panel_probe["craft_icon_hits"] = vis.get("craft_icon_hits")
            panel_probe["craft_panel"] = vis.get("craft_panel")
            panel_probe["panel_has_image_cue"] = vis.get("panel_has_image_cue")
            diag = (
                panel_probe.get("craft_hub_diag")
                if isinstance(panel_probe.get("craft_hub_diag"), dict)
                else {}
            )
            sig = resolve_craft_open_signal(
                craft_hub_category=panel_probe.get("craft_hub_category")
                if isinstance(panel_probe.get("craft_hub_category"), int)
                else None,
                craft_hub_diag_tag=str(diag.get("tag") or "") or None,
                craft_icon_hits=vis.get("craft_icon_hits") or {},
                craft_panel_open=(vis.get("craft_panel") or {}).get("open"),
                ocr_craft=bool(vis.get("ocr_craft")),
                panel_has_image_cue=bool(vis.get("panel_has_image_cue")),
            )
            panel_probe["craft_open_signal"] = sig
            # 细判：mem 落到 menu_other 但图像可信开态 → 抬到 craft（mid；≠够用）
            if (
                isinstance(verdict, FineUiVerdict)
                and verdict.kind == "menu_other"
                and sig.get("open") is True
                and sig.get("trusted") is True
            ):
                src = str(sig.get("source") or "visual")
                verdict = FineUiVerdict(
                    "craft",
                    FINE_LABELS["craft"],
                    "mid",
                    f"menu=1 hub=- visual={src}",
                )
        except Exception:
            pass

    craft_open = False
    try:
        craft_open = int(panel_probe.get("craft_hub_category")) in (1, 2, 3)
    except (TypeError, ValueError):
        craft_open = False
    if not craft_open:
        craft_signal = (
            panel_probe.get("craft_open_signal")
            if isinstance(panel_probe.get("craft_open_signal"), dict)
            else {}
        )
        craft_open = bool(
            craft_signal.get("open") is True and craft_signal.get("trusted") is True
        )
    panel_probe["craft_open"] = craft_open
    panel_probe["stop_visible"] = stop_vis

    conf = ui_fine_sample_confidence(
        ui_params=ui_params,
        icon_errors=icon_errors,
        icons=icon_rows,
        stop_visible=stop_vis,
        pair_role=role_s,
        need_icon_corroboration=True,
        panel_probe=panel_probe,
    )
    out_detail: dict[str, Any] = {
        "ui_fine": format_fine_ui(verdict),
        "verdict": verdict_to_dict(verdict),
        "ui_params": ui_params,
        "ui_relation": format_ui_relation(ms, ui, detail),
        "coarse_ui": ui,
        "coarse_src": ui_src,
        "mem_ui": mem_ui,
        "geo": geo,
        "map_level": lv,
        "mem_geo": mem_geo,
        "icons": icon_rows,
        "icon_scan_ms": summary.age_ms,
        "icon_cache_age_s": icon_cache_age_s,
        "icon_errors": icon_errors,
        "icon_src": icon_src,
        "action_src": action_src,
        "action_scan_ms": action_age_ms,
        "stop_visible": stop_vis,
        "stop_src": stop_src,
        "travel_pixel": travel_px,
        "confidence": conf,
        "client": {"w": cw, "h": ch},
        "combat_probe": panel_probe,
        "panel_probe": panel_probe,
    }
    # 菜单图标候选对拍：args.probe_menu=1 时对当前帧 matchTemplate（零键鼠、无 OCR）
    if bool(args.get("probe_menu")):
        menu_probe: dict[str, Any] = {"ok": False}
        try:
            from mm_assist.perception.menu_icons import probe_menu_icons

            grab = img
            if grab is None and ctx.mon is not None and getattr(ctx.mon, "frame", None) is not None:
                try:
                    grab = ctx.mon.frame.grab(ctx.cfg) if ctx.cfg is not None else None
                except Exception:
                    grab = None
            if grab is None and ctx.win is not None and ctx.cfg is not None:
                try:
                    from mm_assist.perception.game_frame import ensure_game_frame

                    grab = ensure_game_frame(ctx.cfg, ctx.win, persist=False).grab(ctx.cfg)
                except Exception:
                    grab = None
            if grab is None:
                menu_probe = {"ok": False, "error": "no_frame"}
            else:
                hits = probe_menu_icons(grab)
                hit_nodes = {
                    nid: (row.get("hit") or {}).get("score")
                    for nid, row in hits.items()
                    if row.get("hit")
                }
                missing = [nid for nid, row in hits.items() if row.get("missing_template")]
                frame_rel = None
                try:
                    import cv2

                    from mm_assist.config import ROOT

                    role_safe = str(pair_role or "frame").strip() or "frame"
                    pid_safe = str(pair_id or "menu_icons").strip() or "menu_icons"
                    dest_dir = ROOT / "knowledge" / "captures" / "menu_icons" / pid_safe
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / f"{role_safe}.png"
                    cv2.imwrite(str(dest), grab)
                    frame_rel = str(dest.relative_to(ROOT)).replace("\\", "/")
                except Exception as save_exc:
                    frame_rel = f"save_fail:{save_exc}"
                menu_probe = {
                    "ok": True,
                    "hits": hits,
                    "hit_scores": hit_nodes,
                    "missing_template": missing,
                    "frame_png": frame_rel,
                }
        except Exception as exc:
            menu_probe = {"ok": False, "error": str(exc)}
        out_detail["menu_probe"] = menu_probe
    if pair_id is not None:
        out_detail["pair_id"] = str(pair_id)
    if pair_role is not None:
        out_detail["pair_role"] = str(pair_role)
    entry_id = str(args.get("entry_id") or "").strip()
    if entry_id:
        if img is None:
            return {"ok": False, "error": "hud-entry 证据归档缺原始游戏帧"}
        try:
            from mm_assist.jobs.hud_entries import (
                archive_hud_entry_frame,
                entry_by_id,
                load_main_hud_catalog,
            )

            catalog = load_main_hud_catalog()
            entry = entry_by_id(catalog, entry_id)
            capture_id = str(args.get("capture_id") or pair_id or "").strip()
            role = str(pair_role or "").strip().lower()
            evidence = archive_hud_entry_frame(
                entry,
                img,
                capture_id=capture_id,
                role=role,  # type: ignore[arg-type]
                detail=out_detail,
                screen_origin=(
                    int(getattr(ctx.win, "left", 0) or 0),
                    int(getattr(ctx.win, "top", 0) or 0),
                ),
            )
            out_detail["hud_entry_evidence"] = evidence
            out_detail["locator"] = evidence["locator"]
        except Exception as exc:
            return {
                "ok": False,
                "error": f"hud-entry 证据归档失败:{type(exc).__name__}:{exc}",
            }
    return {"ok": True, "detail": out_detail}


_DISPATCH: dict[str, Callable[[JobContext, dict[str, Any]], dict[str, Any]]] = {
    "mem_warm_retry": _action_mem_warm_retry,
    "readonly_snap_ocr": _action_readonly_snap_ocr,
    "readonly_ui_fine": _action_readonly_ui_fine,
    "benchmark_memory": _action_benchmark_memory,
}


def run_action(
    action: str,
    ctx: JobContext,
    args: dict[str, Any] | None = None,
    *,
    method: str = "A",
) -> dict[str, Any]:
    method_u = str(method or "B").strip().upper() or "B"
    if method_u != "B":
        return {
            "ok": False,
            "error": f"FAIL-CLOSED: 只读 profile 禁止 method={method_u!r}",
        }
    if action in LIVE_ACTIONS:
        return {
            "ok": False,
            "error": f"FAIL-CLOSED: 只读 profile 禁止 LIVE action={action!r}",
        }
    fn = _DISPATCH.get(action)
    if fn is None:
        return {"ok": False, "error": f"未知 action={action!r}；已知 {sorted(KNOWN_ACTIONS)}"}
    try:
        return fn(ctx, args or {})
    except SystemExit as exc:
        # 动作内部不应退出进程；记 fail
        return {"ok": False, "error": f"SystemExit:{exc}"}
    except Exception as exc:
        log.exception("state_bar_job action=%s", action)
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}


class JobWatcher:
    """供 state-watch 主循环轮询：queued → 后台线程执行 → 写回。"""

    def __init__(
        self,
        ctx_factory: Callable[[], JobContext],
        *,
        path: Path | None = None,
        on_status: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._ctx_factory = ctx_factory
        self._path = path or job_path()
        self._on_status = on_status
        self._lock = threading.Lock()
        self._busy = False
        self._last_ui: dict[str, Any] = empty_job()
        self._last_push_sig: tuple[Any, ...] | None = None

    @staticmethod
    def _job_sig(job: dict[str, Any]) -> tuple[Any, ...]:
        return (
            job.get("id"),
            job.get("status"),
            job.get("action"),
            job.get("title"),
            job.get("method"),
            job.get("error"),
            job.get("need_ui"),
        )

    def _emit(self, job: dict[str, Any]) -> None:
        if not self._on_status:
            return
        sig = self._job_sig(job)
        if sig == self._last_push_sig:
            return
        self._last_push_sig = sig
        self._on_status(job)

    @property
    def last_ui(self) -> dict[str, Any]:
        return dict(self._last_ui)

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._busy)

    def poll_once(self) -> None:
        with self._lock:
            if self._busy:
                return
            job = load_job(self._path)
            self._last_ui = job
            claimed = claim_queued(job)
            if claimed is None:
                self._emit(job)
                return
            if str(claimed.get("status")) == "fail":
                save_job(claimed, self._path)
                self._last_ui = claimed
                self._emit(claimed)
                return
            save_job(claimed, self._path)
            self._last_ui = claimed
            self._busy = True
            self._emit(claimed)

        def _worker() -> None:
            method = str(claimed.get("method") or "B").strip().upper() or "B"
            try:
                ctx = self._ctx_factory()
                action = str(claimed.get("action") or "")
                args = claimed.get("args") if isinstance(claimed.get("args"), dict) else {}
                out = run_action(action, ctx, args, method=method)
                ok = bool(out.get("ok"))
                finished = finish_job(
                    claimed,
                    ok=ok,
                    result=out.get("detail", out),
                    error=None if ok else str(out.get("error") or "fail"),
                )
                save_job(finished, self._path)
                self._last_ui = finished
                self._emit(finished)
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_worker, name="state-bar-job", daemon=True).start()
