# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mm_assist.benchmarks.live_click import (
    LiveBenchmarkError,
    assert_live_benchmark_ready,
    prepare_live_view,
    run_one_click,
)
from mm_assist.safety.readonly import (
    ReadOnlyProfileError,
    deny_input,
    live_benchmark_input_scope,
)
from mm_assist.platform.window import WindowInfo


def _win() -> WindowInfo:
    return WindowInfo(hwnd=1, title="x", left=0, top=0, width=1920, height=1440)


def test_default_still_denies_click(monkeypatch):
    from mm_assist.safety.lease import production_mode

    monkeypatch.delenv("MM_LIVE_BENCHMARK", raising=False)
    production_mode(False)
    with pytest.raises(ReadOnlyProfileError):
        deny_input("click_client")


def test_env_without_scope_still_denies(monkeypatch):
    from mm_assist.safety.lease import production_mode

    monkeypatch.setenv("MM_LIVE_BENCHMARK", "1")
    production_mode(False)
    with pytest.raises(ReadOnlyProfileError):
        deny_input("click_client")


def test_scope_without_env_fails(monkeypatch):
    monkeypatch.delenv("MM_LIVE_BENCHMARK", raising=False)
    with pytest.raises(ReadOnlyProfileError):
        with live_benchmark_input_scope():
            deny_input("click_client")


def test_scope_with_env_allows_deny_passthrough(monkeypatch):
    monkeypatch.setenv("MM_LIVE_BENCHMARK", "1")
    with live_benchmark_input_scope():
        deny_input("click_client")


def test_missing_need_ui(monkeypatch):
    monkeypatch.setenv("MM_LIVE_BENCHMARK", "1")
    with pytest.raises(LiveBenchmarkError, match="need_ui"):
        run_one_click(_win(), 10, 10, need_ui="", stage="hud")


def test_third_blind_attempt_blocked(monkeypatch):
    monkeypatch.setenv("MM_LIVE_BENCHMARK", "1")
    with pytest.raises(LiveBenchmarkError, match="第三次"):
        run_one_click(_win(), 10, 10, need_ui="主界面", stage="hud", fail_count=2)


def test_postmessage_rejected(monkeypatch):
    from mm_assist.safety.lease import GuardError

    monkeypatch.setenv("MM_LIVE_BENCHMARK", "1")
    with pytest.raises(GuardError, match="postmessage"):
        run_one_click(
            _win(), 10, 10, method="postmessage", need_ui="主界面", stage="hud"
        )


def test_ready_requires_env(monkeypatch):
    monkeypatch.delenv("MM_LIVE_BENCHMARK", raising=False)
    with pytest.raises(ReadOnlyProfileError):
        assert_live_benchmark_ready(hwnd=1)


def test_ready_requires_state_watch(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MM_LIVE_BENCHMARK", "1")
    with pytest.raises(RuntimeError, match="state-watch"):
        assert_live_benchmark_ready(hwnd=1, root=tmp_path)


def test_prepare_live_view_waits_then_checks(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(
        "mm_assist.benchmarks.live_click.time.sleep", lambda s: sleeps.append(s)
    )
    monkeypatch.setattr(
        "mm_assist.platform.window.bring_to_foreground",
        lambda hwnd, **kwargs: True,
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.live_click.is_foreground", lambda hwnd: True
    )
    prepare_live_view(_win())
    assert sleeps == [1.0]


def test_prepare_live_view_fail_closed_if_not_foreground(monkeypatch):
    from mm_assist.safety.lease import NotAuthorized

    monkeypatch.setattr("mm_assist.benchmarks.live_click.time.sleep", lambda s: None)
    monkeypatch.setattr(
        "mm_assist.platform.window.bring_to_foreground",
        lambda hwnd, **kwargs: False,
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.live_click.is_foreground", lambda hwnd: False
    )
    with pytest.raises(NotAuthorized, match="前置并等待"):
        prepare_live_view(_win())


def test_duplicate_transition_blocked(monkeypatch):
    from mm_assist.safety.lease import (
        ACTION_CLICK,
        GuardError,
        begin_authorization,
        enter_test_mode,
        require_action,
        reset_guard_state,
        wait_for_user_start,
    )

    monkeypatch.setenv("MM_LIVE_BENCHMARK", "1")
    reset_guard_state()
    enter_test_mode()
    begin_authorization(hwnd=42, intent="t", actions=[ACTION_CLICK], lease_s=60)
    wait_for_user_start(timeout_s=0.0, auto_start_on_timeout=True)
    require_action(ACTION_CLICK, 42, transition_id="tid-1", require_foreground=False)
    with pytest.raises(GuardError, match="已执行"):
        require_action(ACTION_CLICK, 42, transition_id="tid-1", require_foreground=False)
    reset_guard_state()
