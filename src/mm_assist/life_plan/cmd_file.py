"""状态栏进程读 logs/life_cmd.json 执行开始/暂停/脱困。外进程点不到 Tk 按钮。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from mm_assist.config import ROOT

CMD_REL = Path("logs") / "life_cmd.json"
STALE_S = 30.0
ALLOWED = frozenset({"start", "pause", "resume", "escape", "test"})


def cmd_path(*, root: Path | None = None) -> Path:
    return (root or ROOT) / CMD_REL


def request_life_cmd(cmd: str, *, root: Path | None = None, now: float | None = None,
                     scenario: str = "", pause_on_failure: bool = False) -> Path:
    c = str(cmd or "").strip()
    if c not in ALLOWED:
        raise ValueError(f"life_cmd 只允许 start/pause/resume/escape，收到 {c!r}")
    path = cmd_path(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cmd": c, "ts": float(now if now is not None else time.time())}
    if c in {"start", "resume"} and pause_on_failure:
        payload["pause_on_failure"] = True
    if c == "test":
        from mm_assist.life_plan.live_test import scenarios
        if scenario not in scenarios():
            raise ValueError("未知测试场景")
        payload["scenario"] = scenario
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def consume_life_cmd(
    *,
    start_fn: Callable[..., Any] | None = None,
    pause_fn: Callable[[], Any] | None = None,
    resume_fn: Callable[..., Any] | None = None,
    escape_fn: Callable[[], Any] | None = None,
    test_fn: Callable[[str], Any] | None = None,
    root: Path | None = None,
    now: float | None = None,
) -> str:
    """状态栏 _poll 调用。读到则执行并删文件。过期指令丢弃。"""
    path = cmd_path(root=root)
    if not path.is_file():
        return ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return ""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    if not isinstance(raw, dict):
        return ""
    cmd = str(raw.get("cmd") or "").strip()
    try:
        ts = float(raw.get("ts") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    age = float(now if now is not None else time.time()) - ts
    if age < -2 or age > STALE_S or cmd not in ALLOWED:
        return ""
    if cmd == "start" and start_fn is not None:
        start_fn(pause_on_failure=True) if raw.get("pause_on_failure") is True else start_fn()
    elif cmd == "pause" and pause_fn is not None:
        pause_fn()
    elif cmd == "resume" and resume_fn is not None:
        resume_fn(pause_on_failure=True) if raw.get("pause_on_failure") is True else resume_fn()
    elif cmd == "escape" and escape_fn is not None:
        escape_fn()
    elif cmd == "test" and test_fn is not None:
        from mm_assist.life_plan.live_test import scenarios
        scenario = str(raw.get("scenario") or "")
        if scenario not in scenarios():
            return ""
        test_fn(scenario)
    return cmd
