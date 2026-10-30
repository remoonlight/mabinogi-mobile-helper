"""A repair plan must reach its own page preparation from an open facility."""
from types import SimpleNamespace
import time

import numpy as np
import pytest

from mm_assist.life_execute import repair, ui_act, ui_logic, ui_nav, ui_recover
from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.types import LifeSnapshot, PlanItem
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests


@pytest.mark.parametrize("arrived", [True, False])
def test_repair_plan_prepares_open_facility_before_repairing(monkeypatch, arrived):
    controller.reset_controller_for_tests()
    reset_stage_gates_for_tests()
    current = PlanItem("p1", "repair", "一鍵修理全部工具")
    following = PlanItem("p2", "feed", "投料 高級木材", facility="wood", recipe="高級木材")
    monkeypatch.setattr(controller, "plan_life", lambda *a, **k: [current, following])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    # Reproduce the production gate: stp=0, but the current page is a facility.
    monkeypatch.setattr(ui_logic, "skip_wait_ui_params", lambda: False)
    monkeypatch.setattr(ui_act, "inspect_is_hud_idle", lambda: False)
    monkeypatch.setattr("mm_assist.perception.state_ui.read_state_ui",
                        lambda **k: SimpleNamespace(stp=0))
    frame = np.zeros((1442, 1912, 3), dtype=np.uint8)
    monkeypatch.setattr("mm_assist.platform.capture.grab_client", lambda win: frame)
    clicks, repairs, progress, keys = [], [], [], []
    original_prepare = ui_act.prepare_to

    def resolve(_frame):
        page = "main_hud" if arrived and clicks else "facility:wood"
        return ui_logic.UiWhere(page, 0, 0, None, "fixture")

    def prepare(target, **kwargs):
        progress.append(target)
        return original_prepare(target, **kwargs, resolve=resolve,
                                sleep_fn=lambda _: None, read_stp=lambda: 0)

    monkeypatch.setattr(ui_act, "prepare_to", prepare)
    original_recover = ui_recover.recover_once
    def recover(trigger, **kwargs):
        # User changed recovery from a repeated prepare_to chain to Esc.
        # Preserve the real pump, recovery budget and unchanged-page failure.
        return original_recover(trigger, grab=lambda: frame,
            press_esc=lambda: keys.append(1), sleep_fn=lambda _: None,
            resolve=lambda _: ui_logic.UiWhere('facility:wood', 0, 0, 0, 'fixture',
                stack_ready=True, stack_sampled_ts=time.time()),
            cancelled=kwargs.get('cancelled'))
    monkeypatch.setattr(ui_recover, 'recover_once', recover)
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (1848, 57))
    monkeypatch.setattr(ui_act, "stack_has_common_popup", lambda **k: False)
    monkeypatch.setattr(ui_act, "stack_has_facility_result", lambda *a, **k: False)
    monkeypatch.setattr("mm_assist.safety.input.click_client",
                        lambda win, x, y: clicks.append((x, y)) or True)
    monkeypatch.setattr(repair, "one_click_repair",
                        lambda cfg, win, **k: (repairs.append("repair") or win,
                                              {"ok": True, "stage": "done"}))
    monkeypatch.setattr(repair, "release_mouse_buttons", lambda: None)
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr("mm_assist.life_plan.action_finish.read_completion_stp",
                        lambda: pytest.fail("repair must not wait for completion stp"))
    ui_nav.reset_edge_memo_for_tests()
    try:
        bar_ui.pump_life_once(SimpleNamespace(incomplete_plan_policy="pause"), object())
        assert progress == ["main_hud"]
        assert keys == ([] if arrived else [1, 1])
        # Two recovery attempts still cannot permit a third unverified close.
        assert clicks == [(1848, 57)]
        assert repairs == (["repair"] if arrived else [])
        assert controller.get_controller().current == following
        assert controller.get_controller().pending_action is None
    finally:
        reset_stage_gates_for_tests()
        ui_nav.reset_edge_memo_for_tests()
        controller.reset_controller_for_tests()
