"""Windows 后台拉起子进程：不弹 python/cmd 黑窗，日志仍可重定向到文件。

``CREATE_NO_WINDOW`` 与 ``DETACHED_PROCESS`` / ``CREATE_NEW_CONSOLE`` 互斥
（后者会让前者被忽略），故本模块从不带 DETACHED。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# winbase.h：控制台程序不分配可见窗
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_NEW_CONSOLE = 0x00000010
SW_HIDE = 0

_STEMS: tuple[tuple[str, str], ...] = (
    ("state-watch", "state_watch"),
    ("watch", "state_watch"),
)


def no_window_kwargs(*, new_group: bool = False) -> dict[str, Any]:
    """合并进 ``Popen`` / ``run``：Windows 下藏控制台。"""
    if sys.platform != "win32":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = SW_HIDE
    flags = int(getattr(subprocess, "CREATE_NO_WINDOW", CREATE_NO_WINDOW))
    if new_group:
        flags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", CREATE_NEW_PROCESS_GROUP))
    return {"startupinfo": si, "creationflags": flags}


def merge_hidden_kwargs(kwargs: dict[str, Any], *, new_group: bool = False) -> dict[str, Any]:
    extra = no_window_kwargs(new_group=new_group)
    out = dict(kwargs)
    user_flags = int(out.pop("creationflags", 0) or 0)
    extra_flags = int(extra.get("creationflags", 0) or 0)
    flags = (user_flags | extra_flags) & ~DETACHED_PROCESS & ~CREATE_NEW_CONSOLE
    if extra.get("startupinfo") is not None and "startupinfo" not in out:
        out["startupinfo"] = extra["startupinfo"]
    if flags:
        out["creationflags"] = flags
    return out


def popen_hidden(cmd: list[str], **kwargs: Any) -> subprocess.Popen:
    return subprocess.Popen(cmd, **merge_hidden_kwargs(kwargs, new_group=True))


def run_hidden(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, **merge_hidden_kwargs(kwargs, new_group=False))


def spawn_logged(
    cmd: list[str],
    *,
    cwd: str | Path | None,
    env: dict[str, str] | None,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.Popen:
    """后台拉起并把 stdout/stderr 追加进日志文件（不丢输出）。"""
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(stdout_path, "a", encoding="utf-8")
    err = open(stderr_path, "a", encoding="utf-8")
    return popen_hidden(
        cmd,
        cwd=None if cwd is None else str(cwd),
        env=env,
        stdout=out,
        stderr=err,
        close_fds=False,
    )


def build_spawn_argv(remainder: list[str], *, executable: str | None = None) -> list[str]:
    """把 ``spawn-hidden`` 余参编成 python argv。"""
    py = executable or sys.executable
    args = [str(a) for a in remainder]
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        raise ValueError("spawn-hidden 需要子命令，例如：watch --interval 0.5")
    if args[0] == "-m" or args[0].endswith(".py") or "/" in args[0] or "\\" in args[0]:
        return [py, *args]
    return [py, "-m", "mm_assist", *args]


def infer_log_stem(argv: list[str]) -> str:
    blob = " ".join(str(a) for a in argv)
    for needle, stem in _STEMS:
        if needle in blob:
            return stem
    return "mm_spawn"


def hide_own_console() -> bool:
    """把本进程控制台 SW_HIDE。pytest / ``MM_KEEP_CONSOLE=1`` 不藏。"""
    if sys.platform != "win32":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    keep = (os.environ.get("MM_KEEP_CONSOLE") or "").strip().lower()
    if keep in ("1", "true", "yes", "on"):
        return False
    try:
        import ctypes

        hwnd = int(ctypes.windll.kernel32.GetConsoleWindow() or 0)
        if not hwnd:
            return False
        ctypes.windll.user32.ShowWindow(hwnd, SW_HIDE)
        return True
    except Exception:
        return False
