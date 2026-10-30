import json

import pytest

from mm_assist.life_plan.live_test import (
    GATHER_VERIFY_S,
    SNAP_VERIFY_BUDGET_S,
    precondition,
    scenarios,
    snapshot_after_execute,
    verdict,
)
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem, QueueSlot, SkillGrowth, ToolDurability


@pytest.fixture(autouse=True)
def _isolated_cancel_state():
    from mm_assist.tool_repair import clear_op_cancel
    clear_op_cancel()
    yield
    clear_op_cancel()


def test_feed_verdict_matches_iron_ingot_variant_slot():
    item = scenarios()["feed:鐵錠"]
    before = LifeSnapshot(ok=True, items={"鐵礦石": 102}, facilities={"metal": ()})
    after = LifeSnapshot(
        ok=True,
        items={"鐵礦石": 92},
        facilities={
            "metal": (
                QueueSlot(recipe="鐵錠(鐵礦石)", done=False, progress_ratio=0.3),
            )
        },
    )
    status, reason = verdict(item, before, after, ExecuteResult(True, "done"))
    assert status == "passed"
    assert reason == "materials_consumed_and_queued"


def test_click_rejects_occluding_window_even_when_game_has_focus(monkeypatch):
    import win32gui
    from mm_assist.safety.input import require_game_point
    monkeypatch.setattr("mm_assist.safety.input.is_foreground", lambda h: True)
    monkeypatch.setattr(win32gui, "WindowFromPoint", lambda xy: 22)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    with pytest.raises(RuntimeError, match="click_occluded"):
        require_game_point(11, 100, 200)
    monkeypatch.setattr(win32gui, "WindowFromPoint", lambda xy: 11)
    require_game_point(11, 100, 200)
    monkeypatch.setattr("mm_assist.safety.input.is_foreground", lambda h: False)
    with pytest.raises(RuntimeError, match="click_focus_lost"):
        require_game_point(11, 100, 200)


def test_input_hides_own_hud_and_restores_without_stealing_focus(monkeypatch):
    import win32gui
    from mm_assist.safety import input as inputs
    monkeypatch.setattr(inputs, "_INPUT_HUD_HWND", 42)
    monkeypatch.setattr(inputs, "_INPUT_HUD_RESTORE", False)
    monkeypatch.setattr(win32gui, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(win32gui, "IsWindow", lambda h: True)
    seen = []
    monkeypatch.setattr(win32gui, "WindowFromPoint", lambda xy: 42)
    monkeypatch.setattr(win32gui, "IsChild", lambda parent, child: False)
    monkeypatch.setattr(win32gui, "ShowWindow", lambda h, mode: seen.append((h, mode)))
    inputs.hud_suspend_for_input(points=[(10, 10)])
    inputs.hud_resume_after_input()
    assert seen == [(42, 0), (42, 4)]


@pytest.mark.parametrize("pointer_moved", [False, True])
def test_click_yields_before_press_when_user_moves_pointer(monkeypatch, pointer_moved):
    import win32gui
    from types import SimpleNamespace
    from mm_assist.safety import input as inputs
    win = SimpleNamespace(hwnd=1, left=100, top=200, width=500, height=400)
    events = []
    pointer = [120, 220]
    monkeypatch.setattr("mm_assist.safety.readonly.deny_input", lambda *a: None)
    monkeypatch.setattr("mm_assist.life_plan.live_test.record_input", lambda *a: None)
    monkeypatch.setattr("mm_assist.platform.window.refresh_window", lambda w: w)
    monkeypatch.setattr("mm_assist.safety.lease.in_test_mode", lambda: True)
    monkeypatch.setattr(inputs, "_guard_before_input", lambda *a, **k: None)
    monkeypatch.setattr(inputs, "install_mouse_safety", lambda: None)
    monkeypatch.setattr(inputs, "hud_suspend_for_input", lambda *a, **k: events.append("hide"))
    monkeypatch.setattr(inputs, "hud_resume_after_input", lambda: events.append("restore"))
    monkeypatch.setattr(inputs, "release_mouse_buttons", lambda: events.append("release"))
    monkeypatch.setattr(inputs, "_sendinput_yield_unfocused", lambda *a: False)
    monkeypatch.setattr(inputs, "is_foreground", lambda _: True)
    monkeypatch.setattr(inputs, "require_game_point", lambda *a: None)
    monkeypatch.setattr(inputs, "_stamp_sendinput_heartbeat", lambda: None)
    monkeypatch.setattr(inputs, "_send_mouse", lambda flags, *xy: events.append(flags))
    monkeypatch.setattr(win32gui, "GetCursorPos", lambda: tuple(pointer))
    def sleep(_):
        if pointer_moved:
            pointer[:] = [300, 400]
    monkeypatch.setattr(inputs.time, "sleep", sleep)
    if pointer_moved:
        with pytest.raises(RuntimeError, match="click_pointer_moved"):
            inputs.click_client(win, 20, 20)
    else:
        assert inputs.click_client(win, 20, 20)
    downs = [f for f in events if isinstance(f, int) and f & inputs.MOUSEEVENTF_LEFTDOWN]
    assert bool(downs) is not pointer_moved
    assert events[-2:] == ["restore", "release"]


def test_rejected_windows_mouse_input_is_not_reported_as_sent(monkeypatch):
    from mm_assist.safety import input as inputs
    monkeypatch.setattr(inputs.ctypes.windll.user32, "SendInput", lambda *a: 0)
    with pytest.raises(RuntimeError, match="SendInput rejected"):
        inputs._send_mouse(inputs.MOUSEEVENTF_MOVE | inputs.MOUSEEVENTF_ABSOLUTE, 10, 20)


def test_input_evidence_finishes_before_focus_and_deadline(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from mm_assist.life_plan import live_test as lt
    order = []
    evidence = lt.Evidence(tmp_path, SimpleNamespace(hwnd=1), "nav:profile")
    monkeypatch.setattr(evidence, "capture", lambda *a: order.append("capture"))
    evidence.before_input = lambda: order.append("focus") or 101.0
    monkeypatch.setattr(lt, "_ACTIVE", evidence)
    assert lt.record_input(evidence.win, "click", [20, 30]) == 101.0
    assert order == ["capture", "focus"]
    monkeypatch.setattr(lt.time, "monotonic", lambda: 100.9)
    lt.check_input_deadline(101.0)
    monkeypatch.setattr(lt.time, "monotonic", lambda: 101.0)
    with pytest.raises(RuntimeError, match="focus_click_timeout"):
        lt.check_input_deadline(101.0)


def test_parse_nav_detail_and_page_match():
    from mm_assist.life_plan.live_test import nav_page_matched, parse_nav_detail
    assert parse_nav_detail("facility:metal") == ("facility", "", "metal", "")
    assert parse_nav_detail("gather_skill:woodcut") == ("gather_skill", "woodcut", "", "")
    assert parse_nav_detail("craft_quick") == ("craft_quick", "", "", "")
    assert parse_nav_detail("gather_detail:herb:淨淨蘑菇") == ("gather_detail", "herb", "", "淨淨蘑菇")
    assert parse_nav_detail("item_info:mine:鐵礦脈") == ("item_info", "mine", "", "鐵礦脈")
    assert parse_nav_detail("obtain_method:wood:樹液") == ("obtain_method", "", "wood", "樹液")
    assert parse_nav_detail("item_info:wood:樹液") == ("item_info", "", "wood", "樹液")
    assert parse_nav_detail("facility_recipe:metal:鐵錠") == ("facility_recipe", "", "metal", "鐵錠")
    assert nav_page_matched("facility:metal", "facility:metal")
    assert nav_page_matched("gather_skill:herb", "gather_skill:herb")
    assert nav_page_matched("profile:info", "profile")
    assert nav_page_matched("item_info:鐵礦脈", "item_info:mine:鐵礦脈")
    assert not nav_page_matched("item_info", "item_info:mine:鐵礦脈")
    assert nav_page_matched("obtain_method:樹液", "obtain_method:wood:樹液")
    assert not nav_page_matched("obtain_method", "obtain_method:wood:樹液")
    assert nav_page_matched("item_info:樹液", "item_info:wood:樹液")
    assert nav_page_matched("gather_detail:淨淨蘑菇", "gather_detail:herb:淨淨蘑菇")
    assert nav_page_matched("gather_detail:淨淨蘑菇", "item_info:herb:淨淨蘑菇")
    assert not nav_page_matched("gather_detail", "item_info:herb:淨淨蘑菇")
    assert not nav_page_matched("gather_detail:淨淨蘑菇", "item_info:shear:羊")
    assert nav_page_matched("gather_detail:淨淨蘑菇", "item_info:herb:淨淨蘑菇")
    assert not nav_page_matched("gather_detail", "item_info:herb:淨淨蘑菇")
    assert not nav_page_matched("gather_detail:淨淨蘑菇", "item_info:shear:羊")
    assert nav_page_matched("facility:metal:鐵錠", "facility_recipe:metal:鐵錠")
    assert not nav_page_matched("gather_detail:羊毛", "item_info:shear:羊")
    assert not nav_page_matched("facility:wood", "facility:metal")
    assert not nav_page_matched("gather_detail:鐵礦脈", "gather_detail:herb:淨淨蘑菇")
    assert not nav_page_matched("facility:wood:木材", "facility_recipe:metal:鐵錠")


def test_every_material_recipe_and_facility_has_scenario():
    from mm_assist.life_plan.recipes import GATHER_NODE, RECIPES
    cases = scenarios()
    assert all(f"gather:{mat}" in cases for mat in GATHER_NODE)
    assert all(f"feed:{recipe}" in cases for recipe in RECIPES)
    assert all(f"claim:{fac}" in cases for fac in ("metal", "wood", "cloth", "medicine"))
    assert all(f"nav:facility:{fac}" in cases for fac in ("metal", "wood", "cloth", "medicine"))
    assert "nav:gather_skill:woodcut" in cases
    assert "nav:craft_quick" in cases
    assert "nav:item_info:mine:鐵礦脈" in cases
    assert "nav:item_info:woodcut:樹木" in cases
    assert "nav:item_info:herb:淨淨蘑菇" in cases
    assert "nav:item_info:daily:蜘蛛網" in cases
    assert "nav:item_info:daily:水井" in cases
    assert "nav:item_info:wood:樹液" in cases
    assert "nav:obtain_method:wood:樹液" in cases
    assert "nav:obtain_method:mine:鐵礦脈" not in cases
    assert "nav:obtain_method:woodcut:樹木" not in cases
    assert "nav:facility_recipe:metal:鋼錠" in cases


@pytest.mark.parametrize("pending_after", [False, True])
def test_navigation_only_never_reads_materials_or_executes_business(monkeypatch, tmp_path, pending_after):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from mm_assist.jobs.state_watch import REQUIRED_WARM_TYPES, write_watch_ready
    from mm_assist.life_plan import live_test as lt
    from mm_assist.life_execute.ui_logic import UiWhere
    import os
    import time

    write_watch_ready(
        {
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
        },
        root=tmp_path,
    )
    monkeypatch.setattr(lt, "ROOT", tmp_path)
    monkeypatch.setattr(lt.Evidence, "capture", lambda *a, **k: None)
    monkeypatch.setattr(lt, "probe_focus", lambda h: {"kind": "game", "is_game": True})
    monkeypatch.setattr("mm_assist.jobs.bar.require_state_watch_running", lambda: {})
    monkeypatch.setattr("mm_assist.safety.lease.bounded_test_lease", lambda **k: nullcontext())
    monkeypatch.setattr("mm_assist.platform.capture.grab_client", lambda w: object())
    ready = UiWhere("life_grid", 1, 0, None, "direct", stack_ready=True)
    seq = [ready, ready]
    if pending_after:
        seq.extend([UiWhere("unknown", None, None, None, "direct_pending"), ready, ready])
    seq.extend([ready, ready, ready])
    rows = iter(seq)
    monkeypatch.setattr("mm_assist.life_execute.ui_logic.resolve_navigation", lambda f: next(rows, ready))
    seen = {}
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.ensure_node",
        lambda *a, **k: seen.update(k) or ExecuteResult(True, "at_node"),
    )
    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: pytest.fail("navigation must not read materials"))
    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", lambda *a, **k: pytest.fail("navigation must not run business"))
    assert lt.run_scenario(None, SimpleNamespace(hwnd=1), "nav:life_grid")["status"] == "passed"
    assert callable(seen.get("drag_fn"))


def test_no_material_and_empty_claim_are_blocked():
    snap = LifeSnapshot(ok=True)
    assert precondition(scenarios()["feed:鋼錠"], snap) == "no_mats"
    assert precondition(scenarios()["claim:metal"], snap) == "nothing_to_claim"
    assert precondition(scenarios()["gather:裝水的瓶子"], snap) == "no_empty_bottle"


def test_ui_success_without_actual_gain_is_failed():
    before = LifeSnapshot(ok=True, items={"淨淨蘑菇": 40})
    item = scenarios()["gather:淨淨蘑菇"]
    assert verdict(item, before, before, ExecuteResult(True, "done"))[0] == "failed"
    after = LifeSnapshot(ok=True, items={"淨淨蘑菇": 41})
    assert verdict(item, before, after, ExecuteResult(True, "done"))[0] == "passed"


def test_train_skill_exp_gain_is_passed_ui_ok_without_exp_is_failed():
    item = scenarios()["train:甲胄樹"]
    assert item.gather_mode == "skill"
    assert item.skill == "伐木"
    before = LifeSnapshot(ok=True, skills=(SkillGrowth("伐木", 12, 100),))
    same = LifeSnapshot(ok=True, skills=(SkillGrowth("伐木", 12, 100),))
    gained = LifeSnapshot(ok=True, skills=(SkillGrowth("伐木", 12, 140),))
    assert verdict(item, before, same, ExecuteResult(True, "done")) == ("failed", "no_skill_gain")
    assert verdict(item, before, gained, ExecuteResult(True, "done"))[0] == "passed"
    unread = LifeSnapshot(ok=True, error="skills_unreadable")
    assert verdict(item, before, unread, ExecuteResult(True, "done")) == ("unverified", "skill_unreadable")


def test_repair_verdict_uses_tool_durability_not_executor_ok():
    item = scenarios()["repair"]
    before = LifeSnapshot(ok=True, tools=(ToolDurability("斧", 12.0, 100.0),))
    same = LifeSnapshot(ok=True, tools=(ToolDurability("斧", 12.0, 100.0),))
    rose = LifeSnapshot(ok=True, tools=(ToolDurability("斧", 80.0, 100.0),))
    empty = LifeSnapshot(ok=True, tools=())
    assert verdict(item, before, same, ExecuteResult(True, "done")) == ("failed", "no_durability_gain")
    assert verdict(item, before, rose, ExecuteResult(True, "done")) == ("passed", "durability_rose")
    assert verdict(item, before, empty, ExecuteResult(True, "done")) == ("unverified", "tools_unreadable")
    assert verdict(item, before, rose, ExecuteResult(True, "skip")) == ("unverified", "repair_skipped")


@pytest.mark.parametrize("nav_status, elapsed, can_dialog", [
    ("walking", 10.0, True), ("walking", 61.0, False),
    ("at_dialog", 10.0, True), ("at_dialog", 61.0, False),
])
def test_repair_scenario_has_one_deadline_across_navigation_and_dialog(
    monkeypatch, tmp_path, nav_status, elapsed, can_dialog
):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from mm_assist.life_plan import live_test as lt

    now = [0.0]
    scopes = []
    phases = []
    active = [False]

    @contextmanager
    def lease_scope(**kwargs):
        scopes.append(kwargs)
        active[0] = True
        try:
            yield
        finally:
            active[0] = False

    def execute(*args, phase, **kwargs):
        assert active[0]
        phases.append(phase)
        if phase == "nav":
            if nav_status == "at_dialog":
                now[0] += elapsed
            return ExecuteResult(True, nav_status)
        return ExecuteResult(True, "done")

    def arrival(*args, **kwargs):
        now[0] += elapsed
        return True, 0, ""

    monkeypatch.setattr(lt, "ROOT", tmp_path)
    monkeypatch.setattr(lt.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(lt.Evidence, "capture", lambda *a, **k: None)
    monkeypatch.setattr(lt, "probe_focus", lambda h: {"kind": "game", "is_game": True})
    monkeypatch.setattr(lt, "wait_hud_idle", lambda *a, **k: {"node": "main_hud", "stp": 0})
    monkeypatch.setattr("mm_assist.jobs.bar.require_state_watch_running", lambda: {})
    monkeypatch.setattr("mm_assist.safety.lease.bounded_test_lease", lease_scope)
    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: LifeSnapshot(ok=True))
    monkeypatch.setattr("mm_assist.life_execute.ui_logic.read_inspect_record",
                        lambda: {"node": "main_hud", "stack_ready": True, "stp": 0})
    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", execute)
    monkeypatch.setattr("mm_assist.nav.map_atlas.wait_stop_then_dialog", arrival)
    monkeypatch.setattr(lt, "snapshot_after_execute", lambda *a, **k: LifeSnapshot(ok=True))
    report = lt.run_scenario(None, SimpleNamespace(hwnd=1), "repair")
    assert len(scopes) == 1
    assert scopes[0]["lease_s"] == 60
    assert phases == (["nav", "dialog"] if can_dialog else ["nav"])
    if not can_dialog:
        assert report["status"] == "failed"
        assert report["reason"] == "scenario_timeout"


def test_snapshot_after_execute_skill_waits_idle_then_exp():
    item = scenarios()["train:甲胄樹"]
    before = LifeSnapshot(ok=True, skills=(SkillGrowth("伐木", 8, 10),))
    result = ExecuteResult(True, "done")
    now = [0.0]
    skills_kw: list[bool] = []

    def take(**k):
        skills_kw.append(bool(k.get("include_skills")))
        if now[0] < 3.5:
            return LifeSnapshot(ok=True, skills=(SkillGrowth("伐木", 8, 10),))
        return LifeSnapshot(ok=True, skills=(SkillGrowth("伐木", 8, 22),))

    def read_ui():
        if now[0] < 3.2:
            return {"zh": "主界面-动作中", "stp": 1, "node": "main_hud"}
        return {"zh": "主界面-待机", "stp": 0, "node": "main_hud"}

    after = snapshot_after_execute(
        item,
        before,
        result,
        include_tools=False,
        include_skills=True,
        take_snapshot=take,
        read_ui=read_ui,
        clock=lambda: now[0],
        sleep=lambda s: now.__setitem__(0, now[0] + float(s)),
        timeout_s=12.0,
    )
    status, reason = verdict(item, before, after, result)
    assert status == "passed"
    assert "skill_delta=" in reason
    assert after.skills[0].exp == 22
    assert now[0] > 3.0
    assert skills_kw
    assert all(skills_kw)


def test_gather_after_3s_stale_then_later_gain_is_passed(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from mm_assist.life_plan import live_test as lt

    snaps = iter(
        [
            LifeSnapshot(ok=True, items={"淨淨蘑菇": 7}),
            LifeSnapshot(ok=True, items={"淨淨蘑菇": 7}),
            LifeSnapshot(ok=True, items={"淨淨蘑菇": 37}),
        ]
    )
    budgets: list[float | None] = []

    def take(**k):
        budgets.append(k.get("budget_s"))
        return next(snaps)

    monkeypatch.setattr(lt, "ROOT", tmp_path)
    monkeypatch.setattr(lt.Evidence, "capture", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.jobs.bar.require_state_watch_running", lambda: {})
    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", take)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda: {"node": "main_hud", "stack_ready": True, "zh": "主界面-待机", "stp": 0},
    )
    monkeypatch.setattr(lt, "probe_focus", lambda _h: {"kind": "game", "is_game": True})
    monkeypatch.setattr("mm_assist.safety.lease.bounded_test_lease", lambda **k: nullcontext())
    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", lambda *a, **k: ExecuteResult(True, "done"))
    monkeypatch.setattr(lt.time, "sleep", lambda _s: None)
    report = lt.run_scenario(None, SimpleNamespace(hwnd=1), "gather:淨淨蘑菇")
    assert report["status"] == "passed"
    assert "item_delta=" in str(report.get("reason") or "")
    assert budgets[0] == 45
    assert budgets[1:] == [3, 3]


def test_gather_after_retries_timeout_without_gain_is_failed(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from mm_assist.life_plan import live_test as lt

    def take(**k):
        return LifeSnapshot(ok=True, items={"淨淨蘑菇": 7})

    now = [0.0]

    monkeypatch.setattr(lt, "ROOT", tmp_path)
    monkeypatch.setattr(lt.Evidence, "capture", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.jobs.bar.require_state_watch_running", lambda: {})
    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", take)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda: {"node": "main_hud", "stack_ready": True, "zh": "主界面-待机", "stp": 0},
    )
    monkeypatch.setattr(lt, "probe_focus", lambda _h: {"kind": "game", "is_game": True})
    monkeypatch.setattr("mm_assist.safety.lease.bounded_test_lease", lambda **k: nullcontext())
    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", lambda *a, **k: ExecuteResult(True, "done"))
    monkeypatch.setattr(lt, "GATHER_VERIFY_S", 1.0)
    monkeypatch.setattr(lt.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(lt.time, "sleep", lambda s: now.__setitem__(0, now[0] + float(s)))
    report = lt.run_scenario(None, SimpleNamespace(hwnd=1), "gather:淨淨蘑菇")
    assert report["status"] == "failed"
    assert report["reason"] == "no_item_gain"


def test_snapshot_after_execute_waits_idle_then_gain():
    item = scenarios()["gather:裝水的瓶子"]
    assert item.gather_mode != "skill"
    before = LifeSnapshot(ok=True, items={"裝水的瓶子": 4})
    result = ExecuteResult(True, "done")
    now = [0.0]
    budgets: list[float | None] = []

    def take(**k):
        budgets.append(k.get("budget_s"))
        if now[0] < 3.5:
            return LifeSnapshot(ok=True, items={"裝水的瓶子": 4})
        return LifeSnapshot(ok=True, items={"裝水的瓶子": 6})

    def read_ui():
        if now[0] < 3.2:
            return {"zh": "主界面-动作中", "stp": 1, "node": "main_hud"}
        return {"zh": "主界面-待机", "stp": 0, "node": "main_hud"}

    after = snapshot_after_execute(
        item,
        before,
        result,
        include_tools=False,
        take_snapshot=take,
        read_ui=read_ui,
        clock=lambda: now[0],
        sleep=lambda s: now.__setitem__(0, now[0] + float(s)),
        timeout_s=12.0,
    )
    status, reason = verdict(item, before, after, result)
    assert status == "passed"
    assert "item_delta=" in reason
    assert after.items["裝水的瓶子"] == 6
    assert now[0] > 3.0
    assert budgets
    assert all(b == SNAP_VERIFY_BUDGET_S for b in budgets)
    assert len(budgets) >= 2
    assert GATHER_VERIFY_S >= 90.0


def test_snapshot_after_execute_retries_mem_budget():
    item = scenarios()["gather:高級羊毛"]
    before = LifeSnapshot(ok=True, items={"高級羊毛": 31})
    result = ExecuteResult(True, "done")
    now = [4.0]
    n = {"i": 0}

    def take(**k):
        n["i"] += 1
        if n["i"] == 1:
            return LifeSnapshot(ok=False, error="mem_budget")
        return LifeSnapshot(ok=True, items={"高級羊毛": 39})

    after = snapshot_after_execute(
        item,
        before,
        result,
        include_tools=False,
        take_snapshot=take,
        read_ui=lambda: {"zh": "主界面-待机", "stp": 0, "node": "main_hud"},
        clock=lambda: now[0],
        sleep=lambda s: now.__setitem__(0, now[0] + float(s)),
        timeout_s=12.0,
    )
    status, reason = verdict(item, before, after, result)
    assert status == "passed"
    assert reason == "item_delta=8"
    assert n["i"] == 2


def test_snapshot_after_execute_used_for_non_nav_gather_not_single_budget(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from mm_assist.life_plan import live_test as lt

    seen: dict = {}
    budgets: list[float | None] = []

    def fake_snap(item, before, result, **k):
        seen["kind"] = item.kind
        seen["gather_mode"] = item.gather_mode
        seen["timeout_s"] = k.get("timeout_s")
        return LifeSnapshot(ok=True, items={item.recipe: before.items.get(item.recipe, 0) + 2})

    def take(**k):
        budgets.append(k.get("budget_s"))
        return LifeSnapshot(ok=True, items={"裝水的瓶子": 1, "空瓶": 8})

    monkeypatch.setattr(lt, "ROOT", tmp_path)
    monkeypatch.setattr(lt.Evidence, "capture", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.jobs.bar.require_state_watch_running", lambda: {})
    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", take)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda: {"node": "main_hud", "stack_ready": True, "zh": "主界面-待机", "stp": 0},
    )
    monkeypatch.setattr(lt, "probe_focus", lambda _h: {"kind": "game", "is_game": True})
    monkeypatch.setattr("mm_assist.safety.lease.bounded_test_lease", lambda **k: nullcontext())
    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", lambda *a, **k: ExecuteResult(True, "done"))
    monkeypatch.setattr(lt, "snapshot_after_execute", fake_snap)
    monkeypatch.setattr(lt.time, "sleep", lambda _s: None)
    report = lt.run_scenario(None, SimpleNamespace(hwnd=1), "gather:裝水的瓶子")
    assert report["status"] == "passed"
    assert seen.get("kind") == "gather"
    assert seen.get("gather_mode") != "skill"
    assert seen.get("timeout_s") != SNAP_VERIFY_BUDGET_S
    assert budgets == [45]
    assert GATHER_VERIFY_S != SNAP_VERIFY_BUDGET_S


def test_feed_requires_consumption_and_real_queue():
    before = LifeSnapshot(ok=True, items={"鐵錠": 3, "煤炭": 4})
    item = scenarios()["feed:鋼錠"]
    consumed = LifeSnapshot(ok=True, items={})
    assert verdict(item, before, consumed, ExecuteResult(True, "done"))[0] == "failed"
    queued = LifeSnapshot(ok=True, facilities={"metal": (QueueSlot("鋼錠", False),)})
    assert verdict(item, before, queued, ExecuteResult(True, "done"))[0] == "passed"


def test_test_command_roundtrip_and_invalid_scenario(tmp_path):
    from mm_assist.life_plan.cmd_file import consume_life_cmd, request_life_cmd
    calls = []
    request_life_cmd("test", scenario="gather:淨淨蘑菇", root=tmp_path, now=1000)
    assert consume_life_cmd(root=tmp_path, now=1001, test_fn=calls.append) == "test"
    assert calls == ["gather:淨淨蘑菇"]
    with pytest.raises(ValueError):
        request_life_cmd("test", scenario="anything", root=tmp_path)


def test_production_executor_cannot_disable_bounded_lease(monkeypatch):
    from mm_assist.safety import lease
    monkeypatch.setattr(lease, "assert_protocol_ready", lambda: None)
    monkeypatch.setattr("mm_assist.safety.input.release_mouse_buttons", lambda: None)
    with lease.bounded_test_lease(hwnd=1, intent="offline"):
        lease.production_mode(True)
        assert lease.in_test_mode()
        assert lease.current_lease() is not None
        lease.revoke("test")
        lease.production_mode(True)
        with pytest.raises(lease.NotAuthorized):
            lease.require_action("click", 1, require_foreground=False)
    assert not lease.in_production_mode()


def test_lease_timeout_revokes_clicks_without_op_cancel(monkeypatch):
    import time
    from mm_assist.safety import lease
    from mm_assist.tool_repair import clear_op_cancel, op_cancelled

    monkeypatch.setattr(lease, "assert_protocol_ready", lambda: None)
    monkeypatch.setattr("mm_assist.safety.input.release_mouse_buttons", lambda: None)
    clear_op_cancel()
    try:
        with lease.bounded_test_lease(hwnd=1, intent="offline", lease_s=0.2):
            time.sleep(0.45)
            assert op_cancelled() is False
            with pytest.raises((lease.NotAuthorized, lease.LeaseExpired)):
                lease.require_action("click", 1, require_foreground=False)
    finally:
        clear_op_cancel()


def test_conditions_block_real_executor_and_write_report(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from mm_assist.life_plan import live_test
    monkeypatch.setattr(live_test, "ROOT", tmp_path)
    monkeypatch.setattr("mm_assist.jobs.bar.require_state_watch_running", lambda: {})
    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: LifeSnapshot(ok=True))
    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", lambda *a, **k: pytest.fail("must not execute"))
    report = live_test.run_scenario(None, SimpleNamespace(hwnd=1), "claim:metal")
    assert report["status"] == "blocked"
    assert json.loads((tmp_path / "logs/life_test_latest.json").read_text())["reason"] == "nothing_to_claim"


def test_start_waits_for_recognized_page_not_merely_nonempty_record():
    from mm_assist.life_plan.live_test import wait_start_ui
    rows = iter([{"node": "", "stack_ready": False}, {"node": "inventory", "stack_ready": True}])
    now = [0.0]
    def sleep(dt):
        now[0] += dt
    assert wait_start_ui(lambda: next(rows), clock=lambda: now[0], sleep=sleep)["node"] == "inventory"
    assert now[0] > 0
    assert wait_start_ui(lambda: {"node": "", "stack_ready": False}, timeout_s=0.2,
                         clock=lambda: now[0], sleep=sleep) == {}


def test_wait_hud_idle_requires_stable_idle():
    from mm_assist.life_plan.live_test import wait_hud_idle

    now = [0.0]
    rows = iter(
        [
            {"node": "main_hud", "zh": "主界面-动作中", "stp": 1},
            {"node": "main_hud", "zh": "主界面-待机", "stp": 0},
            {"node": "main_hud", "zh": "主界面-待机", "stp": 0},
            {"node": "main_hud", "zh": "主界面-待机", "stp": 0},
            {"node": "main_hud", "zh": "主界面-待机", "stp": 0},
        ]
    )

    def sleep(dt):
        now[0] += dt

    rec = wait_hud_idle(
        lambda: next(rows),
        timeout_s=5.0,
        stable_s=0.8,
        sleep=sleep,
        clock=lambda: now[0],
    )
    assert rec.get("stp") == 0
    assert now[0] >= 0.8


def test_busy_start_does_not_call_executor(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from mm_assist.life_plan import live_test as lt
    monkeypatch.setattr(lt, "ROOT", tmp_path)
    monkeypatch.setattr(lt.Evidence, "capture", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.jobs.bar.require_state_watch_running", lambda: {})
    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: LifeSnapshot(ok=True))
    monkeypatch.setattr(lt, "START_STP_WAIT_S", 0.0)
    monkeypatch.setattr("mm_assist.life_execute.ui_logic.read_inspect_record",
                        lambda: {"node": "main_hud", "stack_ready": True, "zh": "主界面-动作中"})
    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", lambda *a, **k: pytest.fail("busy must not execute"))
    report = lt.run_scenario(None, SimpleNamespace(hwnd=1), "gather:淨淨蘑菇")
    assert report["status"] == "blocked"
    assert report["reason"] == "stp_busy"


def test_nav_busy_hud_still_opens_pages(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from mm_assist.jobs.state_watch import write_watch_ready
    from mm_assist.life_plan import live_test as lt
    import os
    import time

    write_watch_ready(
        {
            "pid": os.getpid(),
            "updated_at_unix": time.time(),
            "tick_completed": True,
            "warming": True,
            "busy": True,
            "required_pending": ["GameSpaceAPI"],
            "optional_absent": ["CurrentViewingMapUIComponent"],
            "present": ["ChannelIdentifierComponent"],
            "params_incomplete": True,
        },
        root=tmp_path,
    )
    called = []
    monkeypatch.setattr(lt, "ROOT", tmp_path)
    monkeypatch.setattr(lt.Evidence, "capture", lambda *a, **k: None)
    monkeypatch.setattr(lt, "probe_focus", lambda h: {"kind": "game", "is_game": True})
    monkeypatch.setattr("mm_assist.jobs.bar.require_state_watch_running", lambda: {})
    monkeypatch.setattr("mm_assist.safety.lease.bounded_test_lease", lambda **k: nullcontext())
    rows = iter([
        {"node": "main_hud", "stack_ready": True, "stp": 1, "zh": "主界面-动作中"},
        {"node": "system_menu", "stack_ready": True},
        {"node": "system_menu", "stack_ready": True},
    ])
    monkeypatch.setattr(lt, "wait_start_ui", lambda *a, **k: next(rows))
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.ensure_node",
        lambda *a, **k: called.append("ensure") or ExecuteResult(True, "at_node", "system_menu"),
    )
    monkeypatch.setattr("mm_assist.platform.capture.grab_client", lambda w: object())
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_navigation",
        lambda f: type("W", (), {"node": "system_menu", "stack_ready": True, "cost_ms": {}})(),
    )
    report = lt.run_scenario(None, SimpleNamespace(hwnd=1), "nav:system_menu")
    assert called == ["ensure"]
    assert report["status"] == "passed", report.get("reason")
    assert report["reason"] == "page_matched"


def test_statusbar_lease_opens_readonly_gate_only_in_own_thread(monkeypatch):
    import threading
    from mm_assist.safety import lease, readonly
    monkeypatch.setattr(lease, "assert_protocol_ready", lambda: None)
    monkeypatch.setattr("mm_assist.safety.input.release_mouse_buttons", lambda: None)
    with pytest.raises(readonly.ReadOnlyProfileError):
        with readonly.statebar_test_input_scope():
            pass
    with lease.bounded_test_lease(hwnd=1, intent="offline"):
        readonly.deny_input("click_client")
        denied = []
        def other_thread():
            try:
                readonly.deny_input("click_client")
            except readonly.ReadOnlyProfileError:
                denied.append(True)
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(timeout=1)
        assert denied == [True]
        lease.revoke("offline")
        with pytest.raises(readonly.ReadOnlyProfileError):
            readonly.deny_input("click_client")
    with pytest.raises(readonly.ReadOnlyProfileError):
        readonly.deny_input("click_client")


def test_user_switched_aborts_before_executor(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from mm_assist.life_plan import live_test as lt

    monkeypatch.setattr(lt, "ROOT", tmp_path)
    monkeypatch.setattr(lt.Evidence, "capture", lambda *a, **k: None)
    monkeypatch.setattr("mm_assist.jobs.bar.require_state_watch_running", lambda: {})
    monkeypatch.setattr(
        "mm_assist.life_plan.snapshot.take_snapshot",
        lambda **k: LifeSnapshot(ok=True, items={"淨淨蘑菇": 1}),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda: {"node": "gather_skill:herb", "stack_ready": True, "zh": "角色-生活技能-採集藥草"},
    )
    monkeypatch.setattr(lt, "probe_focus", lambda _h: {"kind": "user", "is_game": False, "fg_exe": "chrome.exe"})
    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", lambda *a, **k: pytest.fail("user left"))
    report = lt.run_scenario(None, SimpleNamespace(hwnd=1), "gather:淨淨蘑菇")
    assert report["status"] == "failed"
    assert report["reason"] == "user_switched_away"
    assert report["stages"]


def test_probe_focus_kinds_offline():
    from mm_assist.life_plan.live_test import classify_focus, probe_focus

    assert classify_focus(is_game=True, fg_exe="", fg_title="", same_pid=False) == "game"
    assert classify_focus(is_game=False, fg_exe="python.exe", fg_title="mm-assist", same_pid=True) == "statusbar"
    assert classify_focus(is_game=False, fg_exe="explorer.exe", fg_title="Program Manager", same_pid=False) == "operator"
    assert classify_focus(is_game=False, fg_exe="chrome.exe", fg_title="Google", same_pid=False) == "user"
    row = probe_focus(0)
    assert "kind" in row
    assert "game_hwnd" in row
