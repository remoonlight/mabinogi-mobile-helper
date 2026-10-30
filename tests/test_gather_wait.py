from __future__ import annotations

import json
import os
from pathlib import Path

from mm_assist.life_execute.gather_wait import (
    HOLD_IDLE_S,
    WAIT_GATHER_S,
    GatherObservation,
    is_gather_idle,
    read_gather_observation,
    wait_gather_idle,
)
from mm_assist.life_plan.types import ExecuteResult


class _Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += float(s)


def test_wait_constants_match_contract():
    assert WAIT_GATHER_S == 90.0
    assert HOLD_IDLE_S == 2.0


def test_busy_with_bag_gain_is_not_done():
    clock = _Clock()
    bags = [{"蘑菇": 5}, {"蘑菇": 8}]
    scans = {"n": 0}

    def observe() -> GatherObservation:
        return GatherObservation(stp=1, page="main_hud", fresh=True)

    def read_items():
        scans["n"] += 1
        return bags[min(scans["n"] - 1, len(bags) - 1)]

    r = wait_gather_idle(
        observe=observe,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        read_items=read_items,
        cancelled=lambda: False,
    )
    assert r.ok is False
    assert r.status == "gather_pending"
    assert clock.t >= WAIT_GATHER_S
    assert scans["n"] >= 1


def test_busy_then_none_is_not_idle():
    clock = _Clock()
    seq = [GatherObservation(stp=1, page="main_hud", fresh=True)]

    def observe() -> GatherObservation:
        if seq:
            return seq[0]
        return GatherObservation(stp=None, page="unknown", fresh=False)

    r = wait_gather_idle(
        observe=observe,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        hold_s=0.0,
        cancelled=lambda: False,
    )
    assert r.ok is False
    assert r.status == "gather_pending"
    assert "待机" not in (r.message or "")


def test_stale_or_unknown_page_is_not_idle():
    clock = _Clock()
    n = {"i": 0}

    def observe() -> GatherObservation:
        n["i"] += 1
        if n["i"] == 1:
            return GatherObservation(stp=1, page="main_hud", fresh=True)
        return GatherObservation(stp=0, page="main_hud", fresh=False)

    r = wait_gather_idle(
        observe=observe,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        hold_s=0.0,
        cancelled=lambda: False,
    )
    assert r.ok is False
    assert r.status == "gather_pending"

    clock2 = _Clock()
    n2 = {"i": 0}

    def obs2() -> GatherObservation:
        n2["i"] += 1
        if n2["i"] == 1:
            return GatherObservation(stp=1, page="main_hud", fresh=True)
        return GatherObservation(stp=0, page="unknown", fresh=True)

    r2 = wait_gather_idle(
        observe=obs2,
        now_fn=clock2.now,
        sleep_fn=clock2.sleep,
        hold_s=0.0,
        cancelled=lambda: False,
    )
    assert r2.ok is False
    assert r2.status == "gather_pending"


def test_known_idle_stable_completes_once():
    clock = _Clock()
    n = {"i": 0, "done": 0}

    def observe() -> GatherObservation:
        n["i"] += 1
        if n["i"] == 1:
            return GatherObservation(stp=1, page="main_hud", fresh=True)
        return GatherObservation(stp=0, page="main_hud", fresh=True)

    r = wait_gather_idle(
        observe=observe,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        hold_s=HOLD_IDLE_S,
        cancelled=lambda: False,
    )
    assert r.ok is True
    assert r.status == "done"
    assert r.action_started is True
    assert clock.t >= HOLD_IDLE_S
    assert n["i"] >= 2


def test_cancel_stops_without_done():
    clock = _Clock()

    def observe() -> GatherObservation:
        return GatherObservation(stp=1, page="main_hud", fresh=True)

    r = wait_gather_idle(
        observe=observe,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        cancelled=lambda: True,
    )
    assert r.ok is False
    assert r.status == "cancel"


def test_wait_does_not_call_take_snapshot(monkeypatch):
    from mm_assist.life_plan import snapshot as snap_mod

    clock = _Clock()
    hits = {"n": 0}

    def boom(**_k):
        hits["n"] += 1
        raise AssertionError("take_snapshot must not run in gather wait")

    monkeypatch.setattr(snap_mod, "take_snapshot", boom)

    seq = {"i": 0}

    def observe() -> GatherObservation:
        seq["i"] += 1
        if seq["i"] == 1:
            return GatherObservation(stp=1, page="main_hud", fresh=True)
        return GatherObservation(stp=0, page="main_hud", fresh=True)

    r = wait_gather_idle(
        observe=observe,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        hold_s=0.0,
        cancelled=lambda: False,
    )
    assert r.status == "done"
    assert hits["n"] == 0


def test_injected_observe_must_not_fall_back_to_game(monkeypatch):
    from mm_assist.perception import state_ui

    clock = _Clock()
    monkeypatch.setattr(
        state_ui,
        "read_state_ui",
        lambda **_k: (_ for _ in ()).throw(AssertionError("no live state_ui")),
    )
    seq = {"i": 0}

    def observe() -> GatherObservation:
        seq["i"] += 1
        if seq["i"] == 1:
            return GatherObservation(stp=1, page="main_hud", fresh=True)
        return GatherObservation(stp=0, page="main_hud", fresh=True)

    r = wait_gather_idle(
        observe=observe,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        hold_s=0.0,
        cancelled=lambda: False,
    )
    assert isinstance(r, ExecuteResult)
    assert r.status == "done"


def _write_state_ui(root: Path, *, now: float, **fields: object) -> None:
    payload = {
        "pid": int(os.getpid()),
        "ts_unix": float(now),
        "family": "unknown",
        "ui": "hud",
        "ui_detail": "",
        "ui_source": "mem",
        "ui_params": "stp=0",
        "ui_stack": "主界面",
        "ui_stack_names": [],
        "ui_stack_ready": True,
        "ui_stack_empty_ok": True,
        "ui_relation": "主界面非战斗",
        "menu_open": False,
        "dialog_active": False,
        "map_open": None,
        "craft_hub_category": None,
        "stp": 0,
    }
    payload.update(fields)
    path = root / "logs" / "state_watch_ui.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_inspect(root: Path, *, now: float, **fields: object) -> None:
    payload = {
        "ts_unix": float(now),
        "pid": int(os.getpid()),
        "node": "main_hud",
        "zh": "主界面-待机",
        "stack_names": [],
        "stack_ready": True,
        "source": "mem",
    }
    payload.update(fields)
    path = root / "logs" / "ui_inspect.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _hud_fix(name: str):
    import cv2

    path = Path(__file__).resolve().parent / "fixtures" / "gather_hud" / name
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert frame is not None and frame.size > 0
    return frame


def _seed_hud_assets(dst: Path) -> None:
    import shutil

    from mm_assist.config import ROOT

    files = [
        Path("knowledge") / "ui" / "contracts" / "main_hud_entry_capture.json",
    ]
    src_dir = ROOT / "knowledge" / "templates" / "hud_entries"
    for src in files:
        target = dst / src
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, target)
    if src_dir.is_dir():
        for png in src_dir.glob("*.png"):
            target = dst / "knowledge" / "templates" / "hud_entries" / png.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, target)


def test_read_gather_observation_empty_stack_without_hud_is_not_idle(tmp_path):
    now = 1_000.0
    _write_state_ui(tmp_path, now=now, stp=0)
    obs = read_gather_observation(root=tmp_path, now=now)
    assert obs.fresh is True
    assert obs.stp == 0
    assert obs.page == "unknown"
    assert is_gather_idle(obs) is False


def test_read_gather_observation_rejects_unready_or_unconfirmed_or_none(tmp_path):
    now = 1_000.0
    _write_state_ui(tmp_path, now=now, ui_stack_ready=False, stp=0)
    obs = read_gather_observation(root=tmp_path, now=now)
    assert obs.page == "unknown"
    assert is_gather_idle(obs) is False

    _write_state_ui(
        tmp_path,
        now=now,
        ui_stack_ready=True,
        ui_stack_empty_ok=False,
        ui_stack="未知",
        stp=0,
    )
    obs2 = read_gather_observation(root=tmp_path, now=now)
    assert obs2.page == "unknown"
    assert is_gather_idle(obs2) is False

    missing = tmp_path / "empty_root"
    missing.mkdir()
    obs3 = read_gather_observation(root=missing, now=now)
    assert obs3.fresh is False
    assert obs3.page == "unknown"
    assert obs3.stp is None


def test_read_gather_observation_rejects_stale_foreign_pid_frozen_inspect(tmp_path):
    now = 1_000.0
    _write_state_ui(tmp_path, now=now - 10.0, stp=0)
    stale = read_gather_observation(root=tmp_path, now=now)
    assert stale.fresh is False
    assert is_gather_idle(stale) is False

    _write_state_ui(
        tmp_path,
        now=now,
        ui_stack_empty_ok=False,
        ui_stack="",
        stp=0,
    )
    _write_inspect(tmp_path, now=now, pid=int(os.getpid()) + 99)
    foreign = read_gather_observation(root=tmp_path, now=now)
    assert foreign.page == "unknown"

    _write_inspect(tmp_path, now=now - 10.0, pid=int(os.getpid()))
    frozen = read_gather_observation(root=tmp_path, now=now)
    assert frozen.page == "unknown"


def test_read_gather_observation_fresh_inspect_pair_confirms_main(tmp_path):
    now = 1_000.0
    _write_state_ui(
        tmp_path,
        now=now,
        ui_stack_empty_ok=False,
        ui_stack="",
        stp=0,
        pid=int(os.getpid()),
    )
    _write_inspect(
        tmp_path,
        now=now,
        pid=int(os.getpid()),
        stack_names=[],
        node="main_hud",
        zh="主界面-待机",
        stp=0,
    )
    obs = read_gather_observation(root=tmp_path, now=now)
    assert obs.page == "unknown"
    assert is_gather_idle(obs) is False


def test_read_gather_observation_cutscene_frame_is_not_idle(tmp_path):
    now = 1_000.0
    _write_state_ui(tmp_path, now=now, stp=0)
    _write_inspect(tmp_path, now=now, pid=int(os.getpid()), stack_names=[], node="main_hud", zh="主界面-待机", stp=0)
    cut = _hud_fix("cutscene_goddess_garden.png")
    grabs = {"n": 0}

    def grab():
        grabs["n"] += 1
        return cut

    obs = read_gather_observation(root=tmp_path, now=now, grab=grab)
    assert obs.stp == 0
    assert obs.page == "unknown"
    assert is_gather_idle(obs) is False
    assert grabs["n"] >= 1


def test_read_gather_observation_idle_hud_frame_is_idle(tmp_path):
    now = 1_000.0
    _seed_hud_assets(tmp_path)
    _write_state_ui(tmp_path, now=now, stp=0)
    idle = _hud_fix("idle_main_hud.png")
    obs = read_gather_observation(root=tmp_path, now=now, grab=lambda: idle)
    assert obs.stp == 0
    assert obs.page == "main_hud"
    assert is_gather_idle(obs) is True


def test_read_gather_observation_missing_error_stale_frame_rejected(tmp_path):
    now = 1_000.0
    _write_state_ui(tmp_path, now=now, stp=0)
    idle = _hud_fix("idle_main_hud.png")

    missing = read_gather_observation(root=tmp_path, now=now, grab=lambda: None)
    assert missing.page == "unknown"
    assert is_gather_idle(missing) is False

    def boom():
        raise RuntimeError("grab failed")

    broken = read_gather_observation(root=tmp_path, now=now, grab=boom)
    assert broken.page == "unknown"
    assert is_gather_idle(broken) is False

    stale = read_gather_observation(
        root=tmp_path,
        now=now,
        grab=lambda: {"frame": idle, "captured_at": now - 10.0},
    )
    assert stale.page == "unknown"
    assert is_gather_idle(stale) is False


def test_wait_cutscene_after_busy_is_pending_not_done(tmp_path):
    clock = _Clock(1_000.0)
    _write_state_ui(tmp_path, now=clock.t, stp=1)
    cut = _hud_fix("cutscene_goddess_garden.png")
    bags = [{"鐵礦石": 1}, {"鐵礦石": 9}]
    scans = {"n": 0}

    def grab():
        return cut

    def sleep(s: float) -> None:
        clock.t += float(s)
        _write_state_ui(tmp_path, now=clock.t, stp=0)

    def read_items():
        scans["n"] += 1
        return bags[min(scans["n"] - 1, len(bags) - 1)]

    r = wait_gather_idle(
        root=tmp_path,
        grab=grab,
        now_fn=clock.now,
        sleep_fn=sleep,
        hold_s=HOLD_IDLE_S,
        cancelled=lambda: False,
        already_started=True,
        issued=True,
        read_items=read_items,
    )
    assert r.ok is False
    assert r.status == "gather_pending"
    assert r.action_issued is True
    assert r.action_started is True
    assert clock.t >= WAIT_GATHER_S
    assert scans["n"] >= 1


def test_wait_real_idle_hud_stable_hold_completes_once(tmp_path):
    from mm_assist.life_execute.gather_wait import HUD_GRAB_PERIOD_S, POLL_S

    clock = _Clock(1_000.0)
    _seed_hud_assets(tmp_path)
    _write_state_ui(tmp_path, now=clock.t, stp=1)
    idle = _hud_fix("idle_main_hud.png")
    grabs = {"n": 0}

    def grab():
        grabs["n"] += 1
        return {"frame": idle, "captured_at": clock.t}

    def sleep(s: float) -> None:
        clock.t += float(s)
        _write_state_ui(tmp_path, now=clock.t, stp=0)

    r = wait_gather_idle(
        root=tmp_path,
        grab=grab,
        now_fn=clock.now,
        sleep_fn=sleep,
        hold_s=HOLD_IDLE_S,
        cancelled=lambda: False,
        already_started=True,
        issued=True,
    )
    assert r.ok is True
    assert r.status == "done"
    assert r.action_issued is True
    assert r.action_started is True
    assert clock.t >= HOLD_IDLE_S
    polls = int(HOLD_IDLE_S / POLL_S)
    assert grabs["n"] >= 1
    assert grabs["n"] <= int(HOLD_IDLE_S / HUD_GRAB_PERIOD_S) + 3
    assert grabs["n"] < polls


def test_wait_busy_phase_does_not_grab(tmp_path):
    clock = _Clock(1_000.0)
    _write_state_ui(tmp_path, now=clock.t, stp=1)
    grabs = {"n": 0}

    def grab():
        grabs["n"] += 1
        raise AssertionError("busy wait must not grab")

    def sleep(s: float) -> None:
        clock.t += float(s)
        _write_state_ui(tmp_path, now=clock.t, stp=1)

    r = wait_gather_idle(
        root=tmp_path,
        grab=grab,
        now_fn=clock.now,
        sleep_fn=sleep,
        wait_s=0.2,
        hold_s=HOLD_IDLE_S,
        cancelled=lambda: False,
        already_started=True,
        issued=True,
    )
    assert r.status == "gather_pending"
    assert r.action_started is True
    assert grabs["n"] == 0


def test_wait_issued_unknown_hangs_unissued_is_no_stp():
    clock = _Clock()

    def observe() -> GatherObservation:
        return GatherObservation(stp=None, page="unknown", fresh=False)

    hung = wait_gather_idle(
        observe=observe,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        cancelled=lambda: False,
        issued=True,
    )
    assert hung.ok is False
    assert hung.status == "gather_watch"
    assert hung.action_issued is True
    assert hung.action_started is False
    assert clock.t >= WAIT_GATHER_S

    clock2 = _Clock()
    missed = wait_gather_idle(
        observe=observe,
        now_fn=clock2.now,
        sleep_fn=clock2.sleep,
        cancelled=lambda: False,
        issued=False,
    )
    assert missed.ok is False
    assert missed.status == "no_stp"


def test_default_wait_uses_file_observation_not_stp_parts(tmp_path, monkeypatch):
    from mm_assist.life_execute import gather_wait as gw

    clock = _Clock(1_000.0)
    state = {"stp": 1}

    def boom(**_k):
        raise AssertionError("default wait must not use observation_from_parts")

    monkeypatch.setattr(gw, "observation_from_parts", boom)

    _seed_hud_assets(tmp_path)
    idle = _hud_fix("idle_main_hud.png")

    def write(now: float) -> None:
        _write_state_ui(tmp_path, now=now, stp=state["stp"])

    write(clock.t)

    def sleep(s: float) -> None:
        clock.t += float(s)
        state["stp"] = 0
        write(clock.t)

    r = wait_gather_idle(
        root=tmp_path,
        grab=lambda: idle,
        now_fn=clock.now,
        sleep_fn=sleep,
        hold_s=0.0,
        cancelled=lambda: False,
    )
    assert r.ok is True
    assert r.status == "done"
    assert r.action_started is True


def test_skill_nearby_default_chain_reads_state_ui_file(tmp_path, monkeypatch):
    import numpy as np
    from mm_assist.life_execute import gather_wait as gw
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import ExecuteResult

    clock = _Clock(2_000.0)
    clicks: list[tuple[int, int]] = []
    state = {"stp": 1}

    def boom(**_k):
        raise AssertionError("executor default wait must not use observation_from_parts")

    # This test supplies a HUD fixture for the subsequent idle wait. Inject the
    # independent target proof, now required for the registered copper title.
    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:銅礦脈')
    monkeypatch.setattr(gw, "observation_from_parts", boom)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "gather_detail:銅礦脈", "zh": "技能详情-銅礦脈"},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("gather_detail", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.prepare_to",
        lambda *_a, **_k: ExecuteResult(ok=True, status="at_node", message="ok"),
    )

    def write(now: float) -> None:
        _write_state_ui(tmp_path, now=now, stp=state["stp"])

    idle = _hud_fix("idle_main_hud.png")
    _seed_hud_assets(tmp_path)
    write(clock.t)

    def sleep(s: float) -> None:
        clock.t += float(s)
        state["stp"] = 0
        write(clock.t)

    def click(x: int, y: int) -> bool:
        clicks.append((x, y))
        return True

    r = execute_skill_nearby(
        skill_id="mine",
        item_name="銅礦脈",
        output="銅礦石",
        do_click=click,
        grab=lambda: idle,
        now_fn=clock.now,
        sleep_fn=sleep,
        hold_s=0.0,
        wait_s=5.0,
        root=tmp_path,
    )
    assert clicks == [(30, 40)]
    assert r.ok is True
    assert r.status == "done"
    assert r.action_started is True


def test_hud_templates_missing_from_caller_root_do_not_use_default_root(tmp_path):
    now = 1_000.0
    _write_state_ui(tmp_path, now=now, stp=0)
    idle = _hud_fix("idle_main_hud.png")
    obs = read_gather_observation(root=tmp_path, now=now, grab=lambda: idle)
    assert (tmp_path / "knowledge" / "templates" / "hud_entries").exists() is False
    assert obs.page == "unknown"
    assert is_gather_idle(obs) is False


def test_explicit_bad_frame_timestamps_are_rejected(tmp_path):
    now = 1_000.0
    _seed_hud_assets(tmp_path)
    _write_state_ui(tmp_path, now=now, stp=0)
    idle = _hud_fix("idle_main_hud.png")

    def reject(payload):
        obs = read_gather_observation(root=tmp_path, now=now, grab=lambda: payload)
        assert obs.page == "unknown"
        assert is_gather_idle(obs) is False

    reject({"frame": idle, "captured_at": float("nan")})
    reject({"frame": idle, "captured_at": float("inf")})
    reject({"frame": idle, "captured_at": "nope"})
    reject({"frame": idle, "captured_at": now + 5.0})
    reject((idle, float("nan")))
    raw = read_gather_observation(root=tmp_path, now=now, grab=lambda: idle)
    assert raw.page == "main_hud"
    assert is_gather_idle(raw) is True
