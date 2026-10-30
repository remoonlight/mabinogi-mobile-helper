"""重启本仓 state-watch：只关自己的状态栏进程，不杀其它 Python。

不用 taskkill /IM python.exe，也不用隐藏的 EncodedCommand。
先给本仓状态栏发 WM_CLOSE，超时才 TerminateProcess，且映像必须是 python/pythonw。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from mm_assist.config import ROOT
from mm_assist.jobs.bar import (
    _pid_alive,
    clear_state_watch_heartbeat,
    heartbeat_path,
)

log = logging.getLogger(__name__)

_WM_CLOSE = 0x0010
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED = 0x1000
_TITLE_MARKERS = ("mm-assist", "mm_assist")
_GAME_MARKERS = ("瑪奇", "玛奇 Mobile", "Mabinogi")
_PYTHON_NAMES = frozenset({"python.exe", "pythonw.exe"})

SHORTCUT_NAME = "玛奇状态栏.lnk"


def _heartbeat_pid(root: Path | None = None) -> int:
    path = heartbeat_path(root)
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0
    try:
        return int(data.get("pid") or 0)
    except (TypeError, ValueError):
        return 0


def _python_image(pid: int) -> bool:
    if sys.platform != "win32" or pid <= 0:
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED, False, int(pid))
    if not handle:
        return False
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buf))
        ok = kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        if not ok:
            return False
        name = Path(buf.value).name.lower()
        return name in _PYTHON_NAMES
    except Exception:
        return False
    finally:
        kernel32.CloseHandle(handle)


def _close_bar_windows(*, pids: set[int]) -> int:
    if sys.platform != "win32":
        return 0
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    closed = 0

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd: int, _lp: int) -> bool:
        nonlocal closed
        if not user32.IsWindowVisible(hwnd):
            return True
        n = int(user32.GetWindowTextLengthW(hwnd) or 0)
        buf = ctypes.create_unicode_buffer(n + 2)
        user32.GetWindowTextW(hwnd, buf, n + 2)
        title = buf.value or ""
        if any(g in title for g in _GAME_MARKERS):
            return True
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        owned = int(pid.value) in pids if pids else False
        marked = any(m in title for m in _TITLE_MARKERS)
        if owned or marked:
            user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0)
            closed += 1
        return True

    user32.EnumWindows(_enum, 0)
    return closed


def _terminate_python_pid(pid: int) -> bool:
    if sys.platform != "win32" or pid <= 0 or pid == os.getpid():
        return False
    if not _python_image(pid):
        log.warning("heartbeat pid=%s 映像非 python，仍结束以免双点", pid)
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(_PROCESS_TERMINATE, False, int(pid))
    if not handle:
        return False
    try:
        return bool(ctypes.windll.kernel32.TerminateProcess(handle, 1))
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def stop_existing_state_watch(
    *,
    root: Path | None = None,
    grace_s: float = 4.0,
) -> dict[str, Any]:
    """关掉旧状态栏（心跳 pid + 本仓标题窗）。不碰其它 Python 进程。"""
    hb_pid = _heartbeat_pid(root)
    self_pid = os.getpid()
    targets: list[int] = []
    if hb_pid > 0 and hb_pid != self_pid and _pid_alive(hb_pid):
        targets.append(hb_pid)
    closed = _close_bar_windows(pids=set(targets))
    deadline = time.time() + max(0.5, float(grace_s))
    while targets and time.time() < deadline:
        targets = [p for p in targets if _pid_alive(p)]
        if not targets:
            break
        time.sleep(0.2)
    killed: list[int] = []
    for pid in list(targets):
        if _pid_alive(pid) and _terminate_python_pid(pid):
            killed.append(pid)
    leftover = [p for p in targets if _pid_alive(p)]
    if hb_pid and hb_pid != self_pid and not _pid_alive(hb_pid):
        clear_state_watch_heartbeat(root)
    return {
        "heartbeat_pid": hb_pid,
        "closed_windows": closed,
        "terminated": killed,
        "still_alive": leftover,
    }


def desktop_dir() -> Path:
    if sys.platform == "win32":
        try:
            import win32com.client

            shell = win32com.client.Dispatch("WScript.Shell")
            folder = str(shell.SpecialFolders("Desktop") or "")
            if folder:
                return Path(folder)
        except Exception:
            pass
    return Path.home() / "Desktop"


def install_watch_shortcut(*, python_exe: str | None = None) -> Path:
    """桌面 .lnk 指向明文 .bat（工作目录为本仓）。"""
    if sys.platform != "win32":
        raise RuntimeError("仅 Windows 可装桌面快捷方式")
    import win32com.client

    py = str(python_exe or sys.executable)
    bat = ROOT / "scripts" / "start_state_watch.bat"
    if not bat.is_file():
        raise FileNotFoundError(bat)
    dest = desktop_dir() / SHORTCUT_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    shell = win32com.client.Dispatch("WScript.Shell")
    sc = shell.CreateShortcut(str(dest))
    sc.TargetPath = str(bat)
    sc.WorkingDirectory = str(ROOT)
    sc.WindowStyle = 1
    sc.Description = "重启并打开 mm-assist 状态栏（只关本仓旧 watch）"
    sc.IconLocation = f"{py},0"
    sc.Save()
    return dest


stop_existing_state_watch = stop_existing_state_watch
install_watch_shortcut = install_watch_shortcut
