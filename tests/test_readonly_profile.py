# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import time

import pytest

from mm_assist.safety.readonly import ReadOnlyProfileError
from mm_assist.jobs.pairs import open_perceive_sample
from mm_assist.jobs.bar import advance_job_gate, open_test_job


def test_open_test_job_method_b_gate(tmp_path: Path):
    p = tmp_path / "state_bar_job.json"
    j = open_test_job(
        "readonly_ui_fine",
        method="B",
        title="细界面只读",
        need_ui="清 HUD",
        path=p,
    )
    assert j["status"] == "await_start"
    assert j["method"] == "B"
    direct = advance_job_gate(to="queued", expected_id=j["id"], path=p)
    assert direct["status"] == "queued"


def test_open_test_job_legacy_two_step_gate_still_works(tmp_path: Path):
    p = tmp_path / "state_bar_job.json"
    j = open_test_job(
        "readonly_ui_fine",
        method="B",
        title="细界面只读",
        need_ui="清 HUD",
        path=p,
    )
    j2 = advance_job_gate(to="await_capture", expected_id=j["id"], path=p)
    assert j2["status"] == "await_capture"
    j3 = advance_job_gate(to="queued", expected_id=j["id"], path=p)
    assert j3["status"] == "queued"


def test_stop_gate_from_await_start(tmp_path: Path):
    p = tmp_path / "state_bar_job.json"
    j = open_test_job(
        "readonly_ui_fine",
        method="B",
        title="细界面只读",
        need_ui="清 HUD",
        path=p,
    )
    stopped = advance_job_gate(to="idle", expected_id=j["id"], path=p)
    assert stopped["status"] == "idle"


def test_open_test_job_rejects_method_a(tmp_path: Path):
    with pytest.raises(ValueError, match="禁止 method"):
        open_test_job("readonly_ui_fine", method="A", path=tmp_path / "j.json")


def test_open_test_job_rejects_live_action(tmp_path: Path):
    with pytest.raises(ValueError, match="未知 action"):
        open_test_job("repair_dialog", method="B", path=tmp_path / "j.json")


def test_method_b_watcher_never_releases_mouse(tmp_path: Path, monkeypatch):
    from mm_assist.jobs import bar
    from mm_assist.safety import input as safety_input

    p = tmp_path / "state_bar_job.json"
    job = open_test_job(
        "readonly_ui_fine",
        method="B",
        title="只读采集",
        need_ui="清 HUD",
        path=p,
    )
    advance_job_gate(to="queued", expected_id=job["id"], path=p)

    release_calls: list[bool] = []
    monkeypatch.setattr(
        safety_input,
        "release_mouse_buttons",
        lambda **_: release_calls.append(True),
    )
    monkeypatch.setattr(
        bar,
        "run_action",
        lambda *_args, **_kwargs: {"ok": True, "detail": {"sample": "ok"}},
    )

    watcher = bar.JobWatcher(
        lambda: bar.JobContext(cfg=None, win=None, mon=None),
        path=p,
    )
    watcher.poll_once()
    deadline = time.time() + 2.0
    while (
        watcher.last_ui.get("status") not in {"ok", "fail"} or watcher.busy
    ) and time.time() < deadline:
        time.sleep(0.01)

    assert watcher.last_ui["status"] == "ok"
    assert watcher.busy is False
    assert release_calls == []


def test_click_client_denied():
    from mm_assist.safety.input import click_client, drag_client, press_vk, wheel_client
    from mm_assist.platform.window import WindowInfo
    from mm_assist.safety.lease import production_mode

    production_mode(False)

    win = WindowInfo(hwnd=1, title="x", left=0, top=0, width=1920, height=1440)
    with pytest.raises(ReadOnlyProfileError):
        click_client(win, 100, 100)
    with pytest.raises(ReadOnlyProfileError):
        drag_client(win, 100, 100, 120, 120)
    with pytest.raises(ReadOnlyProfileError):
        press_vk(0x1B)
    with pytest.raises(ReadOnlyProfileError):
        wheel_client(win, 100, 100, path="camera_input_smoke")


def test_deny_input_allows_production_mode():
    from mm_assist.safety.lease import production_mode
    from mm_assist.safety.readonly import deny_input

    production_mode(True)
    try:
        deny_input("click_client")
    finally:
        production_mode(False)


def test_cli_has_no_live_commands():
    from mm_assist.cli import ALLOWED_CMDS

    forbidden = {"gather", "craft", "map-goto", "map-town-click", "probe-backend"}
    assert not (forbidden & set(ALLOWED_CMDS))
    assert "watch" in ALLOWED_CMDS
    assert "capture" in ALLOWED_CMDS
    assert "validate" in ALLOWED_CMDS


def test_open_perceive_sample(tmp_path: Path):
    j = open_perceive_sample(
        pair_id="craft_unit",
        pair_role="craft_closed",
        need_ui="关制作",
        path=tmp_path / "j.json",
    )
    assert j["status"] == "await_start"
    assert j["method"] == "B"
    assert j["action"] == "readonly_ui_fine"
