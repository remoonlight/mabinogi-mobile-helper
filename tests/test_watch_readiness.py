import os
import time
from types import SimpleNamespace

import pytest

from mm_assist.jobs.state_watch import (
    REQUIRED_WARM_TYPES,
    build_watch_ready_payload,
    load_watch_ready,
    watch_ready_path,
    watch_ready_refuse_reason,
    write_watch_ready,
)
from mm_assist.life_plan import live_test as lt
from mm_assist.life_plan.types import ExecuteResult


def _ready(**over):
    row = {
        "pid": os.getpid(),
        "updated_at_unix": time.time(),
        "tick_completed": True,
        "warming": False,
        "busy": False,
        "warm_loaded": 2,
        "warm_total": 3,
        "required_pending": [],
        "optional_absent": ["CurrentViewingMapUIComponent"],
        "present": list(REQUIRED_WARM_TYPES),
        "params_incomplete": True,
    }
    row.update(over)
    return row


def test_build_payload_map_absent_is_optional_not_pending():
    class Scene:
        _objs = {"GameSpaceAPI": [1], "ChannelIdentifierComponent": [2]}
        _pause_resolve = False
        resolving = False

        def _params_incomplete(self):
            return True

    class Mon:
        _scene_mem = Scene()

        def warm_load_progress(self):
            return 2, 3

    row = build_watch_ready_payload(Mon(), tick_completed=True)
    assert row["required_pending"] == []
    assert row["optional_absent"] == ["CurrentViewingMapUIComponent"]
    assert row["warm_loaded"] == 2
    assert row["warm_total"] == 3
    assert row["params_incomplete"] is True
    assert watch_ready_refuse_reason(row) == ""


def test_write_file_is_what_live_test_inspects(tmp_path):
    write_watch_ready(_ready(), root=tmp_path)
    assert watch_ready_path(tmp_path).is_file()
    loaded = load_watch_ready(root=tmp_path)
    row, reason = lt.inspect_nav_watch_ready(root=tmp_path)
    assert reason == ""
    assert loaded["pid"] == os.getpid()
    assert row["optional_absent"] == ["CurrentViewingMapUIComponent"]


def test_refuse_stale_wrong_pid_initializing():
    now = 1_000_000.0
    assert watch_ready_refuse_reason(_ready(updated_at_unix=now - 40), now=now) == "watch_ready_stale"
    assert watch_ready_refuse_reason(_ready(pid=1)) == "watch_ready_wrong_pid"
    assert watch_ready_refuse_reason(_ready(tick_completed=False)) == "watch_ready_initializing"
    assert watch_ready_refuse_reason(_ready(warming=True)) == "watch_ready_warming"
    assert watch_ready_refuse_reason(_ready(required_pending=["GameSpaceAPI"])) == "watch_ready_required_pending"
    assert watch_ready_refuse_reason({}) == "watch_ready_missing"


def _nav_harness(monkeypatch, tmp_path, payload):
    from contextlib import nullcontext

    from mm_assist.life_execute.ui_logic import UiWhere

    write_watch_ready(payload, root=tmp_path)
    grabs: list[int] = []
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(lt, "ROOT", tmp_path)
    monkeypatch.setattr(lt.Evidence, "capture", lambda *a, **k: None)
    monkeypatch.setattr(lt, "probe_focus", lambda h: {"kind": "game", "is_game": True})
    monkeypatch.setattr("mm_assist.jobs.bar.require_state_watch_running", lambda: {})
    monkeypatch.setattr("mm_assist.safety.lease.bounded_test_lease", lambda **k: nullcontext())
    monkeypatch.setattr("mm_assist.platform.capture.grab_client", lambda w: grabs.append(1) or object())
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_navigation",
        lambda f: UiWhere("life_grid", 1, 0, None, "direct", stack_ready=True),
    )

    def _click(win, x, y):
        clicks.append((x, y))

    monkeypatch.setattr("mm_assist.safety.input.click_client", _click)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.ensure_node",
        lambda *a, **k: ExecuteResult(True, "at_node"),
    )
    report = lt.run_scenario(None, SimpleNamespace(hwnd=1), "nav:life_grid")
    return report, grabs, clicks


def test_nav_stale_refuses_before_input(monkeypatch, tmp_path):
    report, grabs, clicks = _nav_harness(
        monkeypatch, tmp_path, _ready(updated_at_unix=time.time() - 40)
    )
    assert report["status"] == "blocked"
    assert report["reason"] == "watch_ready_stale"
    assert grabs == []
    assert clicks == []


def test_nav_wrong_pid_refuses_before_input(monkeypatch, tmp_path):
    report, grabs, clicks = _nav_harness(monkeypatch, tmp_path, _ready(pid=1))
    assert report["status"] == "blocked"
    assert report["reason"] == "watch_ready_wrong_pid"
    assert grabs == []
    assert clicks == []


def test_nav_initializing_refuses_before_input(monkeypatch, tmp_path):
    report, grabs, clicks = _nav_harness(monkeypatch, tmp_path, _ready(tick_completed=False))
    assert report["status"] == "blocked"
    assert report["reason"] == "watch_ready_initializing"
    assert grabs == []
    assert clicks == []


def test_nav_warming_gamespace_pending_still_starts(monkeypatch, tmp_path):
    report, grabs, clicks = _nav_harness(
        monkeypatch,
        tmp_path,
        _ready(warming=True, busy=True, required_pending=["GameSpaceAPI"]),
    )
    assert report["status"] == "passed"
    assert report["reason"] == "page_matched"
    assert grabs
    assert clicks == []


def test_nav_initialized_optional_map_absent_can_pass(monkeypatch, tmp_path):
    report, grabs, clicks = _nav_harness(monkeypatch, tmp_path, _ready())
    assert report["status"] == "passed"
    assert report["reason"] == "page_matched"
    assert report["watch_ready"]["optional_absent"] == ["CurrentViewingMapUIComponent"]
    assert grabs
    assert clicks == []
