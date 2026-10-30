"""只读 profile 硬闸：禁止一切游戏键鼠。"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_tls = threading.local()

_LIVE_ENV = "MM_LIVE_BENCHMARK"


class ReadOnlyProfileError(RuntimeError):
    """本仓运行环境不允许发送输入。"""


def live_benchmark_env_enabled() -> bool:
    return os.environ.get(_LIVE_ENV, "").strip().lower() in ("1", "true", "yes")


def live_benchmark_input_armed() -> bool:
    return bool(getattr(_tls, "live_benchmark", False)) and live_benchmark_env_enabled()


@contextmanager
def statebar_test_input_scope() -> Iterator[None]:
    from mm_assist.safety.lease import current_lease, in_test_mode
    if not in_test_mode() or current_lease() is None:
        raise ReadOnlyProfileError("状态栏测试缺少有效租约")
    prev = bool(getattr(_tls, "statebar_test", False))
    _tls.statebar_test = True
    try:
        yield
    finally:
        _tls.statebar_test = prev


@contextmanager
def live_benchmark_input_scope() -> Iterator[None]:
    """仅 live_benchmark 在环境变量已开且租约武装后进入。"""
    if not live_benchmark_env_enabled():
        raise ReadOnlyProfileError(
            f"FAIL-CLOSED: 未设 {_LIVE_ENV}=1，禁止 live_benchmark 输入"
        )
    prev = bool(getattr(_tls, "live_benchmark", False))
    _tls.live_benchmark = True
    try:
        yield
    finally:
        _tls.live_benchmark = prev


def deny_input(op: str) -> None:
    if live_benchmark_input_armed():
        return
    if getattr(_tls, "statebar_test", False):
        from mm_assist.safety.lease import current_lease, in_test_mode
        if in_test_mode() and current_lease() is not None:
            return  # 输入层随后仍验证动作、目标窗口、前台和租约到期时间。
    try:
        from mm_assist.safety.lease import in_production_mode

        if in_production_mode():
            return
    except ImportError:
        pass
    raise ReadOnlyProfileError(
        f"FAIL-CLOSED: 只读 profile 禁止 {op}；本仓仅允许方法 B 截屏/内存采证"
        f"（live_benchmark 须 {_LIVE_ENV}=1 且进入租约作用域；"
        "生活一键修理须 production_mode）"
    )
