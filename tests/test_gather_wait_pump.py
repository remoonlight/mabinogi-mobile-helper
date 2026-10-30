from __future__ import annotations

import numpy as np
import pytest

from mm_assist.life_execute.gather_wait import GatherObservation
from mm_assist.life_execute.ui_logic import UiWhere
from mm_assist.life_plan.bar_ui import pump_life_once
from mm_assist.life_plan.controller import (
    get_controller,
    pause_life,
    reset_controller_for_tests,
    resume_life,
    start_life,
)
from mm_assist.life_plan.types import ExecuteResult, PlanItem
from test_life_plan import _snap

RUN0 = 10.0
RUN1 = 20.0


class _Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += float(s)


def _two_gathers(_snap, start_id=1, **_k):
    i = int(start_id)
    return [
        PlanItem(
            id=f"p{i}",
            kind="gather",
            title="采集 銅礦石",
            recipe="銅礦石",
            skill="採礦",
            detail="銅礦脈",
            gather_mode="process_mat",
        ),
        PlanItem(
            id=f"p{i + 1}",
            kind="gather",
            title="采集 原木",
            recipe="原木",
            skill="伐木",
            detail="樹木",
            gather_mode="process_mat",
        ),
    ]


def _click_fn(clicks: list[tuple[int, int]]):
    def _click(x: int, y: int) -> bool:
        clicks.append((x, y))
        return True

    return _click


def _patch_skill_io(monkeypatch, *, stp_fn, page_fn, clock, clicks, bag=None, aborted=None):
    from mm_assist.life_execute import skill_nearby as sn
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby as real_nearby

    from mm_assist.life_plan import action_finish
    # Empty fake frames exercise scheduling, not the registered title pixels.
    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:銅礦脈')
    monkeypatch.setattr('mm_assist.life_plan.bar_ui.time.monotonic',clock.now)
    monkeypatch.setattr(action_finish,'read_completion_stp',lambda:action_finish.StpSample(stp_fn(),clock.t,1,True))
    bag = bag if bag is not None else {"銅礦石": 3}
    scans = {"n": 0}

    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_skill_item",
        lambda *a, **k: {"ok": True, "cx": 10, "cy": 20},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "gather_detail:銅礦脈", "zh": "技能详情-銅礦脈"},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("gather_detail", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.prepare_to",
        lambda *_a, **_k: ExecuteResult(ok=True, status="at_node", message="gather_skill"),
    )
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    if aborted is not None:
        monkeypatch.setattr("mm_assist.life_execute.ui_act.pump_aborted", aborted)

    def take_boom(**_k):
        scans["n"] += 1
        raise AssertionError("formal gather wait must not take_snapshot")

    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", take_boom)

    def fake_obs(*, root=None, now=None, **_k):
        page = page_fn()
        node = str(getattr(page, "node", "") or "unknown")
        fresh = bool(getattr(page, "fresh", False))
        return GatherObservation(stp=stp_fn(), page=node, fresh=fresh)

    monkeypatch.setattr("mm_assist.life_execute.gather_wait.read_gather_observation", fake_obs)

    def wrapped(*a, **k):
        monkeypatch.setattr("mm_assist.life_plan.node_inspect.gather_detail_for_target",
                            lambda frame, target: f"gather_detail:{k['item_name']}")
        if k['item_name'] == '樹木':
            # The second target begins on its own list, not a covered copper
            # popup falsely reporting a freshly verified tree title.
            monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                                lambda **kw: {'node': 'gather_skill:woodcut', 'zh': '伐木'})
        k.setdefault("do_click", _click_fn(clicks))
        k.setdefault("grab", lambda: np.zeros((16, 16, 3), dtype=np.uint8))
        k.setdefault("read_stp", stp_fn)
        k.setdefault("read_page", page_fn)
        k.setdefault("now_fn", clock.now)
        k.setdefault("sleep_fn", clock.sleep)
        k.setdefault("read_items", lambda: dict(bag))
        return real_nearby(*a, **k)

    monkeypatch.setattr(sn, "execute_skill_nearby", wrapped)
    return scans


def _start_two(monkeypatch, *, now: float = RUN0):
    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 100, "高級木材": 100, "高級布料": 100, "淨淨蘑菇汁液": 180, "銅礦石": 3})
    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", _two_gathers)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    return start_life(snapshot=snap, now=now)


def test_pump_busy_gain_does_not_switch_item(monkeypatch):
    clock = _Clock(1.0)
    clicks: list[tuple[int, int]] = []
    bag = {"銅礦石": 3}

    def stp():
        bag["銅礦石"] = 9
        return 1

    _patch_skill_io(
        monkeypatch,
        stp_fn=stp,
        page_fn=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
        clock=clock,
        clicks=clicks,
        bag=bag,
    )
    st0 = _start_two(monkeypatch)
    first = st0.current.id
    queue0 = [item.id for item in st0.queue]
    pump_life_once(None, None)
    st = get_controller()
    assert st.current is not None
    assert st.current.id == first
    assert st.current.recipe == "銅礦石"
    assert [item.id for item in st.queue] == queue0
    assert clicks == [(30, 40)]
    assert bag["銅礦石"] == 9


def test_pump_pending_does_not_click_again(monkeypatch):
    clock = _Clock(1.0)
    clicks: list[tuple[int, int]] = []

    _patch_skill_io(
        monkeypatch,
        stp_fn=lambda: 1,
        page_fn=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
        clock=clock,
        clicks=clicks,
    )
    first = _start_two(monkeypatch).current.id
    pump_life_once(None, None)
    assert clicks == [(30, 40)]
    assert get_controller().current.id == first
    pump_life_once(None, None)
    assert clicks == [(30, 40)]
    assert get_controller().current.id == first


def test_pump_pending_then_idle_acks_once_and_next_clicks_once(monkeypatch):
    clock = _Clock(1.0)
    clicks: list[tuple[int, int]] = []
    mode = {"busy": True}

    def stp():
        return 1 if mode["busy"] else 0

    def page():
        return type("P", (), {"node": "main_hud", "fresh": True})()

    _patch_skill_io(monkeypatch, stp_fn=stp, page_fn=page, clock=clock, clicks=clicks)
    first = _start_two(monkeypatch).current.id
    pump_life_once(None, None)
    assert get_controller().current.id == first
    assert clicks == [(30, 40)]
    mode["busy"] = False
    pump_life_once(None, None)
    for _ in range(3):
        clock.sleep(1)
        pump_life_once(None,None)
        assert get_controller().current.id == first
        assert clicks == [(30,40)]
    clock.sleep(1)
    pump_life_once(None, None)
    st = get_controller()
    assert st.current is not None
    assert st.current.id != first
    assert st.current.recipe == "原木"
    assert clicks == [(30, 40)]
    mode["busy"] = True
    assert not st.refresh_requested
    assert clicks == [(30, 40)]
    # 本批尚有下一项：不刷新库存、不重复派发已完成动作。
    from dataclasses import replace
    from mm_assist.life_plan import controller
    remaining = st.current
    monkeypatch.setattr(controller, "plan_life", lambda *a, **k: [remaining])
    monkeypatch.setattr(controller, "take_snapshot", lambda **k:
                        pytest.fail('remaining batch must not scan inventory'))
    controller.tick_facility_progress(now=RUN0 + 600)
    assert not get_controller().refresh_requested
    assert get_controller().current.id == remaining.id
    assert get_controller().snapshot == st.snapshot
    pump_life_once(None, None)
    assert clicks == [(30, 40), (10, 20), (30, 40)]
    pump_life_once(None, None)
    assert clicks == [(30, 40), (10, 20), (30, 40)]


def test_pump_unissued_busy_waits_without_redispatch_or_skip(monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult as ER
    from mm_assist.life_plan import action_finish
    from mm_assist.life_plan import bar_ui

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 100, "高級木材": 100, "高級布料": 100, "淨淨蘑菇汁液": 180})
    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", _two_gathers)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    calls, sample = [], [action_finish.StpSample(1, 100, 1, True)]
    def execute(item, **kw):
        calls.append(item.id)
        return ER(False, 'stp_busy', '动作中') if len(calls) == 1 else ER(
            False, 'action_pending', action_issued=True)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    monkeypatch.setattr(action_finish, 'read_completion_stp', lambda: sample[0])
    monkeypatch.setattr(bar_ui, '_note_fail_and_continue', lambda **k: pytest.fail('busy is readiness, not failure'))
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly', lambda **k: None)
    first = start_life(snapshot=snap, now=RUN0).current.id
    pump_life_once(None, None)
    assert get_controller().current.id == first
    assert get_controller().pending_action is None
    for next_sample in [action_finish.StpSample(1, 101, 1, True),
                        action_finish.StpSample(0, 102, 1, False),
                        action_finish.StpSample(None, 103, 1, True),
                        action_finish.StpSample(0, 100, 1, True)]:
        sample[0] = next_sample
        pump_life_once(None, None)
        assert calls == [first] and get_controller().current.id == first
    sample[0] = action_finish.StpSample(0, 104, 1, True)
    pump_life_once(None, None)
    assert calls == [first, first]
    assert get_controller().current.id == first
    assert get_controller().pending_action.actions_complete


def test_pump_legacy_done_busy_does_not_switch(monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult as ER

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 100, "高級木材": 100, "高級布料": 100, "淨淨蘑菇汁液": 180})
    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", _two_gathers)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ER(ok=True, status="done", message="旧执行器"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.gather_wait.read_gather_observation",
        lambda **k: GatherObservation(stp=1, page="main_hud", fresh=True),
    )
    first = start_life(snapshot=snap, now=RUN0).current.id
    queue0 = [item.id for item in get_controller().queue]
    pump_life_once(None, None)
    st = get_controller()
    assert st.current is not None
    assert st.current.id == first
    assert [item.id for item in st.queue] == queue0


def test_pump_legacy_done_unknown_does_not_switch(monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult as ER

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 100, "高級木材": 100, "高級布料": 100, "淨淨蘑菇汁液": 180})
    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", _two_gathers)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ER(ok=True, status="done", message="旧执行器"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.gather_wait.read_gather_observation",
        lambda **k: GatherObservation(stp=None, page="unknown", fresh=False),
    )
    first = start_life(snapshot=snap, now=RUN0).current.id
    pump_life_once(None, None)
    assert get_controller().current.id == first
    assert get_controller().current.recipe == "銅礦石"


def test_pause_resume_keeps_gather_pending(monkeypatch):
    clock = _Clock(1.0)
    clicks: list[tuple[int, int]] = []
    _patch_skill_io(
        monkeypatch,
        stp_fn=lambda: 1,
        page_fn=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
        clock=clock,
        clicks=clicks,
    )
    first = _start_two(monkeypatch).current.id
    pump_life_once(None, None)
    pause_life()
    resume_life()
    assert get_controller().current.id == first
    pump_life_once(None, None)
    assert clicks == [(30, 40)]
    assert get_controller().current.id == first


def test_new_start_clears_pending_late_ack_ignored(monkeypatch):
    clock = _Clock(1.0)
    clicks: list[tuple[int, int]] = []
    _patch_skill_io(
        monkeypatch,
        stp_fn=lambda: 1,
        page_fn=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
        clock=clock,
        clicks=clicks,
    )
    st0 = _start_two(monkeypatch, now=RUN0)
    run0 = st0.life_started_at
    first = st0.current.id
    queue0 = [item.id for item in st0.queue]
    pump_life_once(None, None)
    from mm_assist.life_plan.controller import acknowledge_done

    snap2 = _snap({"合金鋼錠": 100, "高級木材": 100, "高級布料": 100, "淨淨蘑菇汁液": 180, "銅礦石": 3})
    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", _two_gathers)
    st1 = start_life(snapshot=snap2, now=RUN1)
    assert st1.life_started_at == RUN1
    assert run0 == RUN0
    assert st1.life_started_at != run0
    before_ack = [item.id for item in st1.queue]
    current_id = st1.current.id
    acknowledge_done(first, run_started_at=run0)
    st = get_controller()
    assert [item.id for item in st.queue] == before_ack
    assert st.current is not None
    assert st.current.id == current_id
    assert st.current.id == st1.current.id
    assert [item.recipe for item in st.queue] == [item.recipe for item in st1.queue]
    assert queue0 == ["p1", "p2"]
    assert before_ack == ["p1", "p2"]


def test_cancel_does_not_keep_clicking(monkeypatch):
    clock = _Clock(1.0)
    clicks: list[tuple[int, int]] = []
    stop = {"on": False}

    def aborted():
        return bool(stop["on"])

    _patch_skill_io(
        monkeypatch,
        stp_fn=lambda: 1,
        page_fn=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
        clock=clock,
        clicks=clicks,
        aborted=aborted,
    )
    _start_two(monkeypatch)
    stop["on"] = True
    pump_life_once(None, None)
    assert clicks == []


def test_pump_issued_unknown_does_not_reclick(monkeypatch):
    clock = _Clock(1.0)
    clicks: list[tuple[int, int]] = []

    _patch_skill_io(
        monkeypatch,
        stp_fn=lambda: None,
        page_fn=lambda: type("P", (), {"node": "unknown", "fresh": False})(),
        clock=clock,
        clicks=clicks,
    )
    first = _start_two(monkeypatch).current.id
    queue0 = [item.id for item in get_controller().queue]
    pump_life_once(None, None)
    assert clicks == [(30, 40)]
    assert get_controller().current.id == first
    assert clock.t < 4.0
    clock.sleep(90.0)
    pump_life_once(None, None)
    st = get_controller()
    assert clicks == [(30, 40)]
    assert st.current is not None
    assert st.current.id == first
    assert [item.id for item in st.queue] == queue0


def test_pause_resume_issued_unknown_does_not_reclick(monkeypatch):
    clock = _Clock(1.0)
    clicks: list[tuple[int, int]] = []
    _patch_skill_io(
        monkeypatch,
        stp_fn=lambda: None,
        page_fn=lambda: type("P", (), {"node": "unknown", "fresh": False})(),
        clock=clock,
        clicks=clicks,
    )
    first = _start_two(monkeypatch).current.id
    pump_life_once(None, None)
    pause_life()
    resume_life()
    pump_life_once(None, None)
    assert clicks == [(30, 40)]
    assert get_controller().current.id == first


@pytest.mark.parametrize("seen_busy", [False, True])
def test_pause_during_first_wait_keeps_issued_action(monkeypatch,seen_busy):
    clock=_Clock(1.0)
    clicks=[]
    _patch_skill_io(monkeypatch,stp_fn=lambda:1 if seen_busy else None,
        page_fn=lambda:type('P',(),{'node':'unknown','fresh':False})(),clock=clock,clicks=clicks)
    started=_start_two(monkeypatch)
    def click(x,y):
        clicks.append((x,y))
        if (x,y)==(30,40):pause_life()
        return True
    # _patch_skill_io obtains this factory at dispatch, so retain the real executor.
    monkeypatch.setattr(__import__(__name__), '_click_fn', lambda _:click)
    pump_life_once(None,None)
    st=get_controller()
    assert st.paused and st.pending_action is not None
    assert st.pending_action.item_id==started.current.id
    assert st.pending_action.idle.zero_since is None
    assert clicks==[(30,40)]
    resume_life()
    pump_life_once(None,None)
    assert clicks==[(30,40)] and get_controller().current.id==started.current.id


def _hud_fix(name: str):
    import cv2
    from pathlib import Path

    path = Path(__file__).resolve().parent / "fixtures" / "gather_hud" / name
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert frame is not None and frame.size > 0
    return frame


def _mem_view(stp: int, *, now: float = 1.0):
    import os
    from types import SimpleNamespace

    payload = {
        "pid": int(os.getpid()),
        "ts_unix": float(now),
        "stp": int(stp),
        "ui_stack_names": [],
        "ui_stack_ready": True,
        "ui_stack_empty_ok": True,
        "ui_stack": "主界面",
        "family": "unknown",
    }
    return SimpleNamespace(
        fresh=True,
        pid_alive=True,
        pid=int(os.getpid()),
        family="unknown",
        stp=int(stp),
        menu_open=False,
        dialog_active=False,
        map_open=None,
        craft_hub_category=None,
        raw=payload,
    )


def _patch_default_nearby(monkeypatch, *, view_fn, grab_fn, clock, clicks, bag=None, ack_grab_fn=None):
    from mm_assist.life_execute import skill_nearby as sn
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby as real_nearby

    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:銅礦脈')
    monkeypatch.setattr('mm_assist.life_plan.bar_ui.time.monotonic',clock.now)
    monkeypatch.setattr('mm_assist.life_plan.action_finish.time.time',clock.now)
    bag = bag if bag is not None else {"銅礦石": 3}
    ack = ack_grab_fn or grab_fn
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_skill_item",
        lambda *a, **k: {"ok": True, "cx": 10, "cy": 20},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "gather_detail:銅礦脈", "zh": "技能详情-銅礦脈"},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("gather_detail", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.prepare_to",
        lambda *_a, **_k: ExecuteResult(ok=True, status="at_node", message="gather_skill"),
    )
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.snapshot.take_snapshot",
        lambda **_k: (_ for _ in ()).throw(AssertionError("no snapshot")),
    )
    monkeypatch.setattr("mm_assist.perception.state_ui.read_state_ui", lambda **k: view_fn())
    monkeypatch.setattr("mm_assist.platform.capture.grab_client", lambda *_a, **_k: ack())

    def wrapped(*a, **k):
        monkeypatch.setattr("mm_assist.life_plan.node_inspect.gather_detail_for_target",
                            lambda frame, target: f"gather_detail:{k['item_name']}")
        k.setdefault("do_click", _click_fn(clicks))
        k.setdefault("grab", grab_fn)
        k.setdefault("now_fn", clock.now)
        k.setdefault("sleep_fn", clock.sleep)
        k.setdefault("read_items", lambda: dict(bag))
        k.pop("read_stp", None)
        k.pop("read_page", None)
        k.pop("observe", None)
        return real_nearby(*a, **k)

    monkeypatch.setattr(sn, "execute_skill_nearby", wrapped)


def test_pump_busy_cutscene_idle2s_busy_keeps_item_no_reclick(monkeypatch):
    clock = _Clock(1.0)
    clicks: list[tuple[int, int]] = []
    cut = _hud_fix("cutscene_goddess_garden.png")
    seq = {"n": 0}

    def view():
        seq["n"] += 1
        if seq["n"] <= 2:
            stp = 1
        elif seq["n"] <= 50:
            stp = 0
        else:
            stp = 1
        return _mem_view(stp, now=clock.t)

    _patch_default_nearby(monkeypatch, view_fn=view, grab_fn=lambda: cut, clock=clock, clicks=clicks)
    st0 = _start_two(monkeypatch)
    first = st0.current.id
    queue0 = [item.id for item in st0.queue]
    pump_life_once(None, object())
    st = get_controller()
    assert st.current is not None
    assert st.current.id == first
    assert st.current.recipe == "銅礦石"
    assert [item.id for item in st.queue] == queue0
    assert clicks == [(30, 40)]
    pump_life_once(None, object())
    assert clicks == [(30, 40)]
    assert get_controller().current.id == first


def test_pump_default_chain_idle_hud_acks_once_then_next(monkeypatch):
    clock = _Clock(1.0)
    clicks: list[tuple[int, int]] = []
    idle = _hud_fix("idle_main_hud.png")
    mode = {"busy": True}

    def view():
        return _mem_view(1 if mode["busy"] else 0, now=clock.t)

    _patch_default_nearby(monkeypatch, view_fn=view, grab_fn=lambda: idle, clock=clock, clicks=clicks)
    first = _start_two(monkeypatch).current.id
    pump_life_once(None, object())
    assert get_controller().current.id == first
    assert clicks == [(30, 40)]
    mode["busy"] = False
    pump_life_once(None, object())
    for _ in range(4):
        clock.sleep(1)
        pump_life_once(None,object())
    st = get_controller()
    assert st.current is not None
    assert st.current.id != first
    assert st.current.recipe == "原木"
    assert clicks == [(30, 40)]


def test_pump_ack_cutscene_does_not_switch(monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult as ER

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 100, "高級木材": 100, "高級布料": 100, "淨淨蘑菇汁液": 180, "銅礦石": 3})
    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", _two_gathers)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ER(ok=True, status="done", message="执行器done"),
    )
    cut = _hud_fix("cutscene_goddess_garden.png")
    monkeypatch.setattr("mm_assist.perception.state_ui.read_state_ui", lambda **k: _mem_view(0, now=20.0))
    monkeypatch.setattr("mm_assist.platform.capture.grab_client", lambda *_a, **_k: cut)
    first = start_life(snapshot=snap, now=RUN0).current.id
    queue0 = [item.id for item in get_controller().queue]
    pump_life_once(None, object())
    st = get_controller()
    assert st.current is not None
    assert st.current.id == first
    assert [item.id for item in st.queue] == queue0


def test_pump_pending_observe_only_pause_resume_and_old_gen_late_ack(monkeypatch):
    clock = _Clock(1.0)
    clicks: list[tuple[int, int]] = []
    cut = _hud_fix("cutscene_goddess_garden.png")
    _patch_default_nearby(
        monkeypatch,
        view_fn=lambda: _mem_view(1, now=clock.t),
        grab_fn=lambda: cut,
        clock=clock,
        clicks=clicks,
    )
    st0 = _start_two(monkeypatch, now=RUN0)
    run0 = st0.life_started_at
    first = st0.current.id
    pump_life_once(None, object())
    assert clicks == [(30, 40)]
    pause_life()
    resume_life()
    pump_life_once(None, object())
    assert clicks == [(30, 40)]
    assert get_controller().current.id == first
    from mm_assist.life_plan.controller import acknowledge_done

    snap2 = _snap({"合金鋼錠": 100, "高級木材": 100, "高級布料": 100, "淨淨蘑菇汁液": 180, "銅礦石": 3})
    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", _two_gathers)
    st1 = start_life(snapshot=snap2, now=RUN1)
    acknowledge_done(first, run_started_at=run0)
    st = get_controller()
    assert st.life_started_at == RUN1
    assert st.current is not None
    assert st.current.id == st1.current.id


def test_pump_complete_chain_waits_four_seconds_after_resume(monkeypatch):
    clock=_Clock(1.0)
    clicks=[]
    idle=_hud_fix('idle_main_hud.png')
    _patch_default_nearby(monkeypatch,view_fn=lambda:_mem_view(0,now=clock.t),
        grab_fn=lambda:idle,clock=clock,clicks=clicks)
    started=_start_two(monkeypatch)
    first=started.current.id
    pump_life_once(None,object())
    st=get_controller()
    assert st.current.id==first and st.pending_action.actions_complete
    assert clicks==[(30,40)]
    pause_life();clock.sleep(10);resume_life()
    pump_life_once(None,object())
    for _ in range(3):
        clock.sleep(1);pump_life_once(None,object())
        assert get_controller().current.id==first
    clock.sleep(1);pump_life_once(None,object())
    assert get_controller().current.recipe=='原木'
    assert clicks==[(30,40)]
