"""地图导航共用的输入策略、安全检查与耗时上下文。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from mm_assist.config import AppConfig, load_config
from mm_assist.safety.input import AbortRequested, check_abort
from mm_assist.safety.lease import (
    GuardError,
    in_production_mode,
    in_test_mode,
    prefer_visible_clicks,
)

_timing_stack: ContextVar[tuple] = ContextVar("nav_timing_stack", default=())


def prepare_grabber(cfg: AppConfig) -> None:
    from mm_assist.platform.capture import configure_grabber

    method = "auto"
    try:
        method = str(getattr(cfg.window, "capture_method", "") or "auto")
    except Exception:
        pass
    configure_grabber(method)


def production_click_method(cfg: AppConfig | None = None) -> tuple[str, bool]:
    c = cfg or load_config()
    m = str(getattr(c.window, "input_method", "") or "sendinput").lower()
    if m in ("unity_hook", "hook"):
        return "unity_hook", False
    if prefer_visible_clicks():
        return "sendinput", True
    if m in ("sendinput", "si"):
        return "sendinput", True
    return "sendinput", True


@contextmanager
def timed_action(action: str, note: str = "", **_k: Any) -> Iterator[dict]:
    ctx: dict = {"ok": False, "note": note, "action": action}
    token = _timing_stack.set(_timing_stack.get() + (ctx,))
    try:
        yield ctx
    finally:
        try:
            _timing_stack.reset(token)
        except Exception:
            pass


__all__ = [
    "AbortRequested",
    "GuardError",
    "check_abort",
    "in_production_mode",
    "in_test_mode",
    "prefer_visible_clicks",
    "prepare_grabber",
    "production_click_method",
    "timed_action",
    "_timing_stack",
]
