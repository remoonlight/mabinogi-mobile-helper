"""最小只读状态栏：界面名 + 可点位置红圈 + 生活调度。"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Any

from mm_assist.config import AppConfig
from mm_assist.perception.game_frame import ensure_game_frame
from mm_assist.perception.scene import SceneMonitor
from mm_assist.memory.readers.position import near_origin
from mm_assist.memory.readers.scene import WARM_BASE_LIMITS
from mm_assist.jobs.bar import (
    JobContext,
    JobWatcher,
    clear_state_watch_heartbeat,
    load_job,
    try_acquire_state_watch_singleton,
    write_state_watch_heartbeat,
)
from mm_assist.perception.state_ui import (
    clear_state_ui,
    write_state_ui,
)
from mm_assist.platform.window import WindowInfo, refresh_window
from mm_assist.platform.win_spawn import hide_own_console
from mm_assist.platform.hint_overlay import ClickMarkOverlay, overlay_marker_from_job

log = logging.getLogger(__name__)

_ENTRY_LABELS = {
    "menu": "菜单",
    "bag": "背包",
    "avatar": "头像",
    "craft": "制作",
    "map": "地图",
    "combat": "战斗",
    "stop": "停止",
}
_ROLE_LABELS = {"source": "源帧", "target": "目标帧"}
_STATUS_LABELS = {
    "idle": "已停止",
    "await_start": "等待采集",
    "await_capture": "等待采集",
    "queued": "采集中",
    "running": "采集中",
    "ok": "已完成",
    "fail": "失败",
}


def format_job_content(job: dict[str, Any] | None) -> str:
    """测试内容：入口 / 帧角色 / 标题 / 需要的界面。"""
    job = job if isinstance(job, dict) else {}
    args = job.get("args") if isinstance(job.get("args"), dict) else {}
    entry_id = str(args.get("entry_id") or "").strip()
    role = str(args.get("pair_role") or "").strip()
    title = str(job.get("title") or "").strip()
    need_ui = str(job.get("need_ui") or "").strip()
    action = str(job.get("action") or "").strip()
    parts: list[str] = []
    if entry_id:
        parts.append(_ENTRY_LABELS.get(entry_id, entry_id))
    if role:
        parts.append(_ROLE_LABELS.get(role, role))
    if title and title not in parts:
        parts.append(title)
    if need_ui and need_ui not in parts:
        parts.append(need_ui)
    if not parts and action:
        parts.append(action)
    if not parts:
        return "无"
    return " · ".join(parts)


def format_job_status(
    job: dict[str, Any] | None,
    *,
    overlay_available: bool = True,
) -> str:
    """实际状态：采集进度 + 红点确认，不暴露内部状态码。"""
    job = job if isinstance(job, dict) else {}
    st = str(job.get("status") or "idle")
    base = _STATUS_LABELS.get(st, "已停止")
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    confirm = (
        result.get("overlay_confirmation")
        if isinstance(result.get("overlay_confirmation"), dict)
        else {}
    )
    decision = str(confirm.get("decision") or "")
    if decision == "correct":
        return "位置正确"
    if decision == "wrong":
        return "位置错误"
    marker = overlay_marker_from_job(job)
    if marker.get("visible") is True:
        if overlay_available is not True:
            return "获取框不可用"
        return "请拖动获取框后点确认"
    if st == "fail":
        err = str(job.get("error") or "").strip()
        return f"失败：{err}" if err else "失败"
    return base


def _stp_token(params: str) -> str | None:
    for tok in str(params or "").split():
        if tok in ("stp=0", "stp=1"):
            return tok
    return None


def format_mem_panel(snap: Any, ms: Any = None) -> str:
    """状态栏：内存状态 + 坐标 + 地点 + stp + 栈中文。不上 inv/menu/dlg，不另起「界面参数」行。"""
    _ = ms
    if snap is None:
        return "内存：等待首拍…"
    st = str(
        getattr(snap, "mem_status_label", None)
        or getattr(snap, "mem_status", None)
        or "—"
    )
    xy = getattr(snap, "mem_xy", None)
    if isinstance(xy, (tuple, list)) and len(xy) >= 2:
        try:
            x, y = float(xy[0]), float(xy[1])
            xy_s = "—" if near_origin((x, y)) else f"{x:.1f}, {y:.1f}"
        except (TypeError, ValueError):
            xy_s = "—"
    else:
        xy_s = "—"
    geo = str(getattr(snap, "geo", "") or "").strip()
    geo_s = f"  地点：{geo}" if geo and geo not in {"未知", "—", "定位中…"} else ""
    stp = _stp_token(str(getattr(snap, "ui_params", "") or "")) or "stp=—"
    stack = str(getattr(snap, "ui_stack", "") or "").strip()
    stack_s = f"  {stack}" if stack and stack != "—" else ""
    return f"内存：{st}  坐标：{xy_s}{geo_s}  {stp}{stack_s}"


def format_warm_progress(loaded: int, total: int) -> str:
    return f"读取 {int(loaded)}/{int(total)}"


READY_FEED = "state_watch_ready.json"
READY_MAX_AGE_S = 12.0
OPTIONAL_WARM_TYPES = frozenset({"CurrentViewingMapUIComponent"})
REQUIRED_WARM_TYPES = tuple(n for n in WARM_BASE_LIMITS if n not in OPTIONAL_WARM_TYPES)
_ready_write_lock = threading.Lock()


def watch_ready_path(root: Path | None = None) -> Path:
    from mm_assist.config import ROOT as REPO

    return (root or REPO) / "logs" / READY_FEED


def _warm_type_has_objs(scene: Any, name: str) -> bool:
    """只看已缓存对象列表，不再 RPM。地图组件开页前可以是空的。"""
    if scene is None:
        return False
    objs = getattr(scene, "_objs", None)
    if not isinstance(objs, dict):
        return False
    return bool(objs.get(name))


def build_watch_ready_payload(
    mon: Any | None = None,
    *,
    tick_completed: bool = False,
    busy: bool = False,
    pid: int | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    scene = getattr(mon, "_scene_mem", None) if mon is not None else None
    present = [n for n in WARM_BASE_LIMITS if _warm_type_has_objs(scene, n)]
    missing = [n for n in WARM_BASE_LIMITS if n not in present]
    optional_absent = [n for n in missing if n in OPTIONAL_WARM_TYPES]
    required_pending = [n for n in missing if n not in OPTIONAL_WARM_TYPES]
    loaded, total = len(present), len(WARM_BASE_LIMITS)
    if mon is not None:
        try:
            loaded, total = mon.warm_load_progress()
        except Exception:
            pass
    warming = bool(getattr(scene, "_pause_resolve", False)) if scene is not None else False
    resolving = bool(getattr(scene, "resolving", False)) if scene is not None else False
    params_incomplete = False
    if scene is not None:
        try:
            params_incomplete = bool(scene._params_incomplete())
        except Exception:
            params_incomplete = False
    return {
        "pid": int(pid if pid is not None else os.getpid()),
        "updated_at_unix": float(time.time() if now is None else now),
        "tick_completed": bool(tick_completed),
        "warming": warming,
        "busy": bool(busy or warming or resolving),
        "warm_loaded": int(loaded),
        "warm_total": int(total),
        "required_pending": required_pending,
        "optional_absent": optional_absent,
        "present": present,
        "params_incomplete": params_incomplete,
        "memory_diagnostic": dict(getattr(getattr(mon, 'last', None), 'memory_diagnostic', {}) or {}),
    }


def write_watch_ready(
    payload: dict[str, Any] | None = None,
    *,
    root: Path | None = None,
    mon: Any | None = None,
    tick_completed: bool = False,
    busy: bool = False,
    pid: int | None = None,
    now: float | None = None,
) -> Path:
    p = watch_ready_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = payload if payload is not None else build_watch_ready_payload(
        mon, tick_completed=tick_completed, busy=busy, pid=pid, now=now
    )
    tmp = p.with_suffix(".tmp")
    text = json.dumps(row, ensure_ascii=False, indent=2) + "\n"
    with _ready_write_lock:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(p)
    return p


def clear_watch_ready(root: Path | None = None) -> None:
    p = watch_ready_path(root)
    try:
        p.unlink(missing_ok=True)
    except OSError as exc:
        log.debug("clear watch ready: %s", exc)


def load_watch_ready(*, root: Path | None = None, now: float | None = None) -> dict[str, Any]:
    p = watch_ready_path(root)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {"ok": False, "reason": "watch_ready_missing"}
    if not isinstance(raw, dict):
        return {"ok": False, "reason": "watch_ready_missing"}
    row = dict(raw)
    ts = 0.0
    try:
        ts = float(row.get("updated_at_unix") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    age = float(time.time() if now is None else now) - ts
    row["age_s"] = round(age, 3)
    return row


def watch_ready_refuse_reason(
    row: dict[str, Any] | None,
    *,
    now: float | None = None,
    expected_pid: int | None = None,
    max_age_s: float = READY_MAX_AGE_S,
    for_page_nav: bool = False,
) -> str:
    if not row:
        return "watch_ready_missing"
    if row.get("reason") == "watch_ready_missing" and "tick_completed" not in row:
        return "watch_ready_missing"
    try:
        pid = int(row.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    want = int(os.getpid() if expected_pid is None else expected_pid)
    if pid != want:
        return "watch_ready_wrong_pid"
    try:
        ts = float(row.get("updated_at_unix") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    age = float(time.time() if now is None else now) - ts
    if age > float(max_age_s):
        return "watch_ready_stale"
    if not row.get("tick_completed"):
        return "watch_ready_initializing"
    diagnostic = row.get('memory_diagnostic') or {}
    build = diagnostic.get('build') or {}
    if build and not build.get('verified'):
        return str(build.get('reason') or 'client_build_unverified')
    if diagnostic.get('access'):
        return 'process_read_denied'
    if for_page_nav:
        # 切页只认即时栈和当前帧；地点预热未齐、后台仍 resolving 不挡开界面。
        return ""
    if row.get("warming"):
        return "watch_ready_warming"
    if row.get("busy"):
        return "watch_ready_busy"
    pending = row.get("required_pending") or []
    if pending:
        return "watch_ready_required_pending"
    return ""


def run_state_watch(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    interval_s: float = 0.25,
    duration_s: float | None = None,
    lean: bool = True,
) -> int:
    if not try_acquire_state_watch_singleton():
        log.error("state-watch 已在运行；请先关旧窗")
        return 2
    from mm_assist.session_log import start_watch_session_log

    start_watch_session_log()
    hide_own_console()
    write_watch_ready(tick_completed=False, busy=True)

    mon = SceneMonitor(cfg, icon_refresh_s=max(0.35, float(interval_s)))
    try:
        ensure_game_frame(cfg, win, persist=True)
    except Exception as exc:
        log.warning("game_frame 失败: %s", exc)

    root = tk.Tk()
    root.title("mm-assist 状态栏")
    root.attributes("-topmost", True)
    root.update_idletasks()
    import win32gui
    from mm_assist.safety.input import register_input_hud
    register_input_hud(win32gui.GetAncestor(root.winfo_id(), 2))
    header = tk.Frame(root)
    header.pack(fill="x", padx=8, pady=(4, 0))
    from mm_assist.platform.tk_mailbox import TkMailbox

    stop = threading.Event()
    ui_mailbox = TkMailbox(root, stop=stop)
    load_var = tk.StringVar(value=format_warm_progress(0, len(WARM_BASE_LIMITS)))
    tk.Label(header, textvariable=load_var, font=("Segoe UI", 8)).pack(side="right")

    def _on_warm(loaded: int, total: int) -> None:
        ui_mailbox.post('warm', load_var.set, format_warm_progress(loaded, total))

    mon.set_warm_progress(_on_warm)
    mem_var = tk.StringVar(value="内存：等待首拍…")
    job_updates: queue.SimpleQueue[tuple[str, dict[str, Any]]] = queue.SimpleQueue()
    mark: ClickMarkOverlay | None = None
    try:
        mark = ClickMarkOverlay(root)
    except Exception as exc:
        log.warning("点位红圈初始化失败: %s", exc)

    def _life_game_win():
        current = refresh_window(win)
        try:
            import win32gui

            hwnd = int(getattr(current, "hwnd", 0) or 0)
            if hwnd > 0 and win32gui.IsWindow(hwnd):
                return current
        except Exception:
            if getattr(current, "hwnd", 0):
                return current
        from mm_assist.platform.window import find_window

        title = str(getattr(cfg.window, "title_substr", "") or "")
        return find_window(title) or find_window("Mobile") or find_window("瑪奇") or current

    def _grab_frame():
        from mm_assist.platform.capture import grab_client

        return grab_client(_life_game_win())

    apply_inspect = None
    try:
        from mm_assist.life_plan.node_inspect import attach_node_inspect

        apply_inspect = attach_node_inspect(
            root,
            tk_mod=tk,
            get_win=_life_game_win,
            grab=_grab_frame,
            mark=mark,
            stop=stop,
            post_ui=ui_mailbox.post,
        )
    except Exception as exc:
        log.warning("界面点位栏失败: %s", exc)

    tk.Label(
        root,
        textvariable=mem_var,
        font=("Segoe UI", 8),
        justify="left",
        wraplength=640,
    ).pack(anchor="w", padx=8, pady=(0, 4))

    try:
        from mm_assist.life_plan.bar_ui import attach_life_plan_controls

        attach_life_plan_controls(
            root,
            tk_mod=tk,
            cfg=cfg,
            get_win=_life_game_win,
        )
    except Exception as exc:
        log.warning("生活调度栏失败: %s", exc)

    def _on_job(job: dict[str, Any]) -> None:
        job_updates.put(("", dict(job)))

    def _drain_job_updates() -> None:
        if stop.is_set():
            return
        while True:
            try:
                job_updates.get_nowait()
            except queue.Empty:
                break
        root.after(200, _drain_job_updates)

    root.after(200, _drain_job_updates)

    def _ctx() -> JobContext:
        w = refresh_window(win) if win else win
        return JobContext(cfg=cfg, win=w, mon=mon)

    watcher = JobWatcher(_ctx, on_status=_on_job)
    try:
        watcher.poll_once()
        write_state_watch_heartbeat()
        load_job()
    except Exception as exc:
        log.warning("state-watch 启动: %s", exc)

    def _on_close() -> None:
        mon._close_stop_pixel()
        try:
            from mm_assist.safety.input import release_mouse_buttons

            release_mouse_buttons()
        except Exception:
            pass
        stop.set()
        if mark is not None:
            mark.destroy()
        try:
            clear_state_watch_heartbeat()
            clear_state_ui()
            clear_watch_ready()
        except Exception:
            pass
        root.after(200, root.destroy)

    if duration_s:
        root.after(max(200, int(float(duration_s) * 1000)), _on_close)

    def _loop() -> None:
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                # 心跳先于重 tick：避免菜单页感知卡住时验收误判「过期」
                write_state_watch_heartbeat()
                from mm_assist.life_plan.snapshot import heavy_read_lock, snapshot_in_progress

                if snapshot_in_progress():
                    write_watch_ready(mon=mon, tick_completed=bool(getattr(_loop, "tick_ok", False)), busy=True)
                    stop.wait(0.1)
                    continue
                w = refresh_window(win)
                with heavy_read_lock:
                    snap = mon.tick_lean(w) if lean else mon.tick(w)
                _loop.tick_ok = True
                write_state_watch_heartbeat()
                ms = getattr(mon, "last_ms", None) or getattr(mon, "last_ms", None)
                try:
                    def _ui(s=snap, m=ms):
                        mem_var.set(format_mem_panel(s, m))
                        if apply_inspect is not None:
                            apply_inspect(m, s)

                    ui_mailbox.post('snapshot', _ui)
                except Exception:
                    pass
                try:
                    loaded, total = mon.warm_load_progress()
                    _on_warm(loaded, total)
                except Exception as exc:
                    log.debug("warm progress: %s", exc)
                try:
                    write_state_ui(snap, ms)
                except Exception:
                    pass
                try:
                    write_watch_ready(mon=mon, tick_completed=True, busy=False)
                except Exception as exc:
                    log.debug("watch ready: %s", exc)
                watcher.poll_once()
            except Exception as exc:
                log.debug("tick: %s", exc)
                try:
                    write_watch_ready(mon=mon, tick_completed=bool(getattr(_loop, "tick_ok", False)), busy=True)
                except Exception:
                    pass
            # 按目标间隔补睡，避免「tick 已慢还硬睡满 interval」
            left = float(interval_s) - (time.perf_counter() - t0)
            time.sleep(max(0.05, left))

    def _heartbeat_pulse() -> None:
        while not stop.is_set():
            try:
                write_state_watch_heartbeat()
            except Exception:
                pass
            time.sleep(1.0)

    from mm_assist.life_plan.action_finish import completion_stp_source
    with completion_stp_source(mon.read_stop_sample):
        threading.Thread(target=_heartbeat_pulse, name="state-watch-hb", daemon=True).start()
        threading.Thread(target=_loop, name="state-watch-lean", daemon=True).start()
        root.protocol("WM_DELETE_WINDOW", _on_close)
        try:
            root.mainloop()
        finally:
            stop.set()
            mon._close_stop_pixel()
    try:
        clear_state_watch_heartbeat()
        clear_state_ui()
        clear_watch_ready()
    except Exception:
        pass
    return 0
