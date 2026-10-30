"""轻量耗时打点：`MM_HUD_PROF=1` 开启，关闭时只多一次布尔判断。

用于定位状态栏/游戏掉帧的真实开销（尤其堆扫）。
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

_TRUE = ("1", "true", "on", "yes")
_ENABLED = (os.environ.get("MM_HUD_PROF") or "").strip().lower() in _TRUE

_lock = threading.Lock()
# name -> [n, total_ms, max_ms, extra]
_stats: dict[str, list[float]] = {}


def enabled() -> bool:
    return _ENABLED


def set_enabled(on: bool) -> None:
    global _ENABLED
    _ENABLED = bool(on)


def record(name: str, ms: float, extra: float = 0.0) -> None:
    if not _ENABLED:
        return
    with _lock:
        s = _stats.get(name)
        if s is None:
            _stats[name] = [1.0, ms, ms, extra]
        else:
            s[0] += 1.0
            s[1] += ms
            s[2] = max(s[2], ms)
            s[3] += extra


@contextmanager
def span(name: str):
    if not _ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record(name, (time.perf_counter() - t0) * 1000.0)


def drain() -> str:
    """取走并清空统计，返回一行摘要（extra 视作字节数）。"""
    if not _ENABLED:
        return ""
    with _lock:
        items = sorted(_stats.items(), key=lambda kv: -kv[1][1])
        _stats.clear()
    if not items:
        return ""
    parts = []
    for name, (n, total, mx, extra) in items:
        seg = f"{name} n={int(n)} tot={total:.0f} avg={total / n:.1f} max={mx:.1f}"
        if extra:
            seg += f" MB={extra / 1048576:.0f}"
        parts.append(seg)
    return " | ".join(parts)


__all__ = ["enabled", "set_enabled", "record", "span", "drain"]
