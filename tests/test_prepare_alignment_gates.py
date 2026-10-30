"""Insect navigation must account for the same menu input as ordinary plans."""
import time

import pytest

from mm_assist.life_execute import insect_obtain, ui_act, ui_nav
from mm_assist.life_execute.ui_logic import UiWhere
from mm_assist.safety import autofarm_gates as gates


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    insect_obtain.reset_insect_visits_for_tests()
    monkeypatch.setattr(ui_act, "pump_aborted", lambda: False)
    monkeypatch.setattr(ui_act, "stack_has_common_popup", lambda **_: False)
    monkeypatch.setattr(ui_act, "stack_has_facility_result", lambda **_: False)
    monkeypatch.setattr(ui_act, "inspect_is_hud_idle", lambda **_: False)
    monkeypatch.setattr(ui_act, "inspect_is_true_hud", lambda **_: False)
    yield
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    insect_obtain.reset_insect_visits_for_tests()


def test_insect_menu_arrival_does_not_exhaust_other_plans_menu_budget(monkeypatch):
    from mm_assist.perception import insect_obtain_sense, ocr

    page = ["main_hud"]
    clicks = []
    real_time, elapsed = time.time, [0.0]
    monkeypatch.setattr(time, "time", lambda: real_time() + elapsed[0])
    monkeypatch.setattr(insect_obtain_sense, "phase", lambda *a, **k: "")
    monkeypatch.setattr(ocr, "ocr_roi", lambda *a, **k: [])
    monkeypatch.setattr(ui_act, "inspect_is_menu", lambda: page[0] == "system_menu")
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (10, 20))

    def resolve(_):
        return UiWhere(page[0], int(page[0] != "main_hud"), 0, 0, "fixture",
                       stack_ready=True, stack_sampled_ts=time.time())

    def click(*xy):
        clicks.append(xy)
        page[0] = "system_menu" if page[0] == "main_hud" else "craft_make"
        return True

    for _ in range(3):
        elapsed[0] += 10.0  # Separate plans, beyond the short duplicate-input memo.
        page[0] = "main_hud"
        result = insect_obtain.execute_insect_obtain(mat="滴答螢火蟲",
            grab=lambda: object(), resolve=resolve, do_click=click,
            read_stp=lambda: 0, sleep_fn=lambda _: None)
        # Stop at a separate, deliberately unrecognized category. The earlier
        # independently observed menu arrival must still be verified.
        assert result.status == "道具"
        assert gates._stage_counts["main_hud_menu_system_menu"] == 0
    assert len(clicks) == 6


@pytest.mark.parametrize("case", ["old_stack", "not_ready", "wrong_page", "blocked", "old_frame"])
def test_alignment_display_alone_cannot_verify_menu_input(monkeypatch, case):
    stage = "main_hud_menu_system_menu"
    page = ["main_hud"]
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (10, 20))
    # The display can claim arrival before a fresh independent stack exists.
    monkeypatch.setattr(ui_act, "inspect_is_menu", lambda: page[0] == "system_menu")

    def click(*_):
        page[0] = "system_menu"
        return True

    def resolve(_):
        return UiWhere(page[0], int(page[0] != "main_hud"), 0, 0, "fixture",
                       stack_ready=True, stack_sampled_ts=1.0)

    result = ui_act.prepare_to("system_menu", grab=lambda: object(), click=click,
        resolve=resolve, sleep_fn=lambda _: None, read_stp=lambda: 0)
    assert result.ok
    assert gates._stage_counts[stage] == 1
    assert stage in ui_nav._PENDING_EDGES
    sent_at = ui_nav._PENDING_EDGES[stage][2]
    where = UiWhere("system_menu", 1, 0, 0, "fixture", stack_ready=case != "not_ready",
        stack_sampled_ts=sent_at - 1 if case == "old_stack" else time.time(),
        stack_names=("ToyDimLayerUIInstance",) if case == "blocked" else ())
    if case == "wrong_page":
        where = UiWhere("main_hud", 0, 0, 0, "fixture", stack_ready=True,
                        stack_sampled_ts=time.time())
    ui_nav._verify_late_edges(where,
        frame_started_at=sent_at - 1 if case == "old_frame" else time.time())
    assert gates._stage_counts[stage] == 1
    assert stage in ui_nav._PENDING_EDGES
    fresh = UiWhere("system_menu", 1, 0, 0, "fixture", stack_ready=True,
                    stack_sampled_ts=time.time())
    ui_nav._verify_late_edges(fresh, frame_started_at=time.time())
    assert gates._stage_counts[stage] == 0
    assert stage not in ui_nav._PENDING_EDGES


def test_unverified_alignment_blocks_third_input_across_plan_calls(monkeypatch):
    page, clicks = ["main_hud"], []
    monkeypatch.setattr(ui_act, "inspect_is_menu", lambda: page[0] == "system_menu")
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (10, 20))

    def click(*xy):
        clicks.append(xy)
        page[0] = "system_menu"
        return True

    def resolve(_):
        return UiWhere(page[0], int(page[0] != "main_hud"), 0, 0, "fixture",
                       stack_ready=True, stack_sampled_ts=1.0)

    for i in range(3):
        page[0] = "main_hud"
        result = ui_act.prepare_to("system_menu", grab=lambda: object(), click=click,
            resolve=resolve, sleep_fn=lambda _: None, read_stp=lambda: 0)
        assert result.ok if i < 2 else result.status == "need_verify"
    assert len(clicks) == 2
    assert gates._stage_counts["main_hud_menu_system_menu"] == 2


def test_refused_alignment_creates_no_pending_input(monkeypatch):
    monkeypatch.setattr(ui_act, "inspect_is_menu", lambda: False)
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (10, 20))
    result = ui_act.prepare_to("system_menu", grab=lambda: object(), click=lambda *_: False,
        resolve=lambda _: UiWhere("main_hud", 0, 0, 0, "fixture"),
        sleep_fn=lambda _: None, read_stp=lambda: 0)
    assert result.status == "click_yield"
    assert gates._stage_counts["main_hud_menu_system_menu"] == 0
    assert not ui_nav._PENDING_EDGES
