from __future__ import annotations

import json
import os
from pathlib import Path

from mm_assist.jobs import watch_lifecycle as wl


def test_stop_existing_ignores_dead_heartbeat(tmp_path: Path, monkeypatch):
    hb = tmp_path / "state_watch.heartbeat.json"
    hb.write_text(json.dumps({"pid": 424242}), encoding="utf-8")
    monkeypatch.setattr(wl, "heartbeat_path", lambda root=None: hb)
    monkeypatch.setattr(wl, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(wl, "_close_bar_windows", lambda pids: 0)
    monkeypatch.setattr(wl, "clear_state_watch_heartbeat", lambda root=None: hb.unlink())
    out = wl.stop_existing_state_watch(root=tmp_path)
    assert out["heartbeat_pid"] == 424242
    assert out["terminated"] == []
    assert not hb.is_file()


def test_stop_existing_skips_own_pid(tmp_path: Path, monkeypatch):
    hb = tmp_path / "state_watch.heartbeat.json"
    hb.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    monkeypatch.setattr(wl, "heartbeat_path", lambda root=None: hb)
    monkeypatch.setattr(wl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(wl, "_close_bar_windows", lambda pids: 0)

    def _boom(_pid: int) -> bool:
        raise AssertionError("must not kill self")

    monkeypatch.setattr(wl, "_terminate_python_pid", _boom)
    out = wl.stop_existing_state_watch(root=tmp_path)
    assert out["terminated"] == []
    assert out["heartbeat_pid"] == os.getpid()


def test_start_watch_bat_exists():
    from mm_assist.config import ROOT

    bat = ROOT / "scripts" / "start_state_watch.bat"
    text = bat.read_text(encoding="utf-8")
    assert "--restart" in text
    assert "--detach" in text
    assert "taskkill" not in text.lower()


def test_click_owner_blocks_foreign_watch_pid(tmp_path: Path, monkeypatch):
    from mm_assist.jobs import bar as bar_mod

    hb = tmp_path / "state_watch.heartbeat.json"
    hb.write_text(json.dumps({"pid": 424242}), encoding="utf-8")
    monkeypatch.setattr(bar_mod, "heartbeat_path", lambda root=None: hb)
    monkeypatch.setattr(bar_mod, "_pid_alive", lambda pid: pid == 424242)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    why = bar_mod.click_owner_block_reason()
    assert "424242" in why
    assert "外进程禁点" in why


def test_terminate_non_python_heartbeat_still_opens(monkeypatch):
    opened: list[int] = []

    class _K:
        def OpenProcess(self, *_a, **_k):
            opened.append(1)
            return 0

        def TerminateProcess(self, *_a, **_k):
            raise AssertionError("no handle")

        def CloseHandle(self, *_a, **_k):
            return True

    monkeypatch.setattr(wl, "_python_image", lambda pid: False)
    import ctypes

    monkeypatch.setattr(ctypes.windll, "kernel32", _K())
    assert wl._terminate_python_pid(99) is False
    assert opened == [1]
