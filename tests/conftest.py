# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_life_evidence(tmp_path, monkeypatch):
    """模拟故障不能覆盖正式日记，或清理实机失败图。"""
    from mm_assist.life_plan import anomalies

    monkeypatch.setattr(anomalies, "ROOT", tmp_path)
    monkeypatch.setattr(anomalies, "_LAST", {})


@pytest.fixture(autouse=True)
def _offline_life_snapshot(monkeypatch):
    """离线快照不得受正在运行的游戏影响；内存单测显式注入读数。"""
    monkeypatch.setattr("mm_assist.life_plan.snapshot.find_game_pid", lambda: None)
    monkeypatch.setattr("mm_assist.life_plan.snapshot.prepare_hints", lambda pid: True)
    # Synthetic reader handles must never reach OS identity queries.
    monkeypatch.setattr("mm_assist.life_plan.snapshot.read_identity", lambda h, pid:
                        {'pid': pid, 'birth': 'offline-fixture', 'build_id': 'offline-fixture'})
    monkeypatch.setattr("mm_assist.memory.component_hints._birth", lambda pid: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr("mm_assist.memory.readers.ui_stack.find_game_pid", lambda: None)
    monkeypatch.setattr("mm_assist.memory.readers.ui_stack.read_identity", lambda h, pid:
                        {'pid': pid, 'birth': 'offline-fixture', 'build_id': 'offline-fixture'})
    for reader in ('scene', 'position'):
        monkeypatch.setattr(f"mm_assist.memory.readers.{reader}.read_identity", lambda h, pid:
                            {'pid': pid, 'birth': 'offline-fixture', 'build_id': 'offline-fixture'})
    # The Life dialog guard must not capture a real window in offline tests.
    # Dedicated dialog tests replace this with retained actual client pixels.
    import numpy as np
    monkeypatch.setattr("mm_assist.life_plan.game_stop._capture",
                        lambda *a: np.full((720,960,3),30,np.uint8))
    monkeypatch.setattr("mm_assist.life_plan.game_stop._FRAME_FAILURE", None)


@pytest.fixture(autouse=True)
def _offline_watch_feed(tmp_path, monkeypatch):
    """Default UI feed is test-local; explicit fixture roots remain available."""
    monkeypatch.setattr("mm_assist.perception.state_ui._repo_root", lambda: tmp_path)


@pytest.fixture(autouse=True)
def _guard_default_production():
    from mm_assist.safety.lease import production_mode, reset_guard_state
    from mm_assist.tool_repair import clear_op_cancel
    from mm_assist.life_execute.skill_list import _pending

    _pending.clear()
    reset_guard_state()
    clear_op_cancel()
    production_mode(True)
    yield
    _pending.clear()
    reset_guard_state()
    clear_op_cancel()
