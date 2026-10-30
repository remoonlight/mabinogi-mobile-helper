"""A later independent page observation verifies the earlier dispatched edge."""
from types import SimpleNamespace

import pytest

from mm_assist.life_execute import ui_act, ui_nav
from mm_assist.life_execute.ui_logic import UiWhere
from mm_assist.safety import autofarm_gates as gates


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    monkeypatch.setattr(ui_act, "stack_has_common_popup", lambda **_: False)
    monkeypatch.setattr(ui_act, "stack_has_facility_result", lambda **_: False)
    monkeypatch.setattr(ui_act, "inspect_is_hud_idle", lambda **_: False)
    monkeypatch.setattr(ui_act, "inspect_is_true_hud", lambda **_: False)
    monkeypatch.setattr(ui_act, "inspect_is_menu", lambda **_: False)
    yield
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()


def dispatched_role_timeout(monkeypatch):
    clicks = []
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (1327, 463))
    def resolve(_):
        return UiWhere("unknown" if clicks else "system_menu", 1, 0, 0, "fixture")
    result = ui_nav.ensure_node("life_grid", grab=lambda: object(), resolve=resolve,
        click=lambda *xy: clicks.append(xy) or True, sleep_fn=lambda _: None)
    assert result.status == "nav_verify_timeout"
    assert clicks == [(1327, 463)]
    assert gates._stage_counts["system_menu_role_profile"] == 1


def test_later_profile_observation_clears_role_gate_before_next_navigation(monkeypatch):
    dispatched_role_timeout(monkeypatch)
    # A new navigation call recognizes the destination of the earlier input.
    # Closing the profile must not leave its *entry* permanently charged.
    clicks = []
    def resolve(_):
        where = UiWhere("main_hud" if clicks else "profile:info", 0, 0, 0, "direct",
                       stack_names=() if clicks else ("NewProfileUIInstance",),
                       stack_ready=True)
        return SimpleNamespace(**(vars(where) | {"stack_sampled_ts": ui_nav.time.time()}))
    result = ui_act.prepare_to("main_hud", grab=lambda: object(), resolve=resolve,
        click=lambda *xy: clicks.append(xy) or True, sleep_fn=lambda _: None)
    assert result.ok
    assert len(clicks) == 1
    assert gates._stage_counts["system_menu_role_profile"] == 0


@pytest.mark.parametrize("case", ["stale", "missing_time", "not_ready", "wrong_page", "blocked", "future"])
def test_later_unproven_page_keeps_role_attempt(monkeypatch, case):
    dispatched_role_timeout(monkeypatch)
    where = SimpleNamespace(**(vars(UiWhere("profile:info", 0, 0, 0, "direct",
                    stack_names=("NewProfileUIInstance",), stack_ready=True)) |
                    {"stack_sampled_ts": ui_nav.time.time()}))
    if case == "stale":
        where.stack_sampled_ts = 1.0
    elif case == "missing_time":
        where.stack_sampled_ts = 0.0
    elif case == "not_ready":
        where.stack_ready = False
    elif case == "wrong_page":
        where.node = "inventory"
        where.stack_names = ("EquipmentInventoryUIInstance",)
    elif case == "blocked":
        where.stack_names += ("ToyDimLayerUIInstance",)
        # The real entry point waits for blocking layers; test the evidence
        # check directly to avoid modelling an entire popup lifecycle here.
        ui_nav._verify_late_edges(where, frame_started_at=ui_nav.time.time())
        assert gates._stage_counts["system_menu_role_profile"] == 1
        return
    else:
        where.stack_sampled_ts += 100
    # Already at this call's destination; that alone cannot verify role entry.
    ui_nav.ensure_node(where.node, grab=lambda: object(), resolve=lambda _: where,
        click=lambda *_: pytest.fail("observation must be read-only"), sleep_fn=lambda _: None)
    assert gates._stage_counts["system_menu_role_profile"] == 1


def test_pre_dispatch_frame_cannot_clear_role_gate(monkeypatch):
    dispatched_role_timeout(monkeypatch)
    sent_at = ui_nav._PENDING_EDGES["system_menu_role_profile"][2]
    where = UiWhere("profile:info", 0, 0, 0, "direct", stack_ready=True,
                    stack_names=("NewProfileUIInstance",), stack_sampled_ts=ui_nav.time.time())
    ui_nav._verify_late_edges(where, frame_started_at=sent_at - 1)
    assert gates._stage_counts["system_menu_role_profile"] == 1


def test_immediate_navigation_carries_older_of_two_independent_stack_timestamps(monkeypatch):
    from mm_assist.life_execute import ui_logic
    samples = iter([SimpleNamespace(ready=True, names=["NewProfileUIInstance"],
        empty_confirmed=False, craft_category=None, sampled_ts=ts) for ts in (10.0, 11.0)])
    reader = SimpleNamespace(read=lambda **_: next(samples))
    monkeypatch.setattr("mm_assist.life_plan.node_inspect.compute_inspect",
                        lambda *a, **k: {"node": "profile:info", "source": "stack+tpl"})
    where = ui_logic.resolve_navigation(object(), reader=reader)
    assert where.stack_ready and where.node == "profile:info"
    assert where.stack_sampled_ts == 10.0


@pytest.mark.parametrize('fresh', [True, False])
def test_pump_page_observation_verifies_menu_without_another_menu_click(monkeypatch, fresh):
    from mm_assist.life_plan import bar_ui
    from mm_assist.platform import capture
    stage = 'main_hud_menu_system_menu'
    assert gates.allow_stage_action(stage) and gates.allow_stage_action(stage)
    ui_nav.note_edge_clicked(stage, 'system_menu', stage=stage)
    sent = ui_nav._PENDING_EDGES[stage][2]
    where = UiWhere('system_menu', 1, 0, 0, 'direct+mem', stack_ready=True,
        stack_names=('SideNavigationUIState', 'SideNavigationUIInstance'),
        stack_sampled_ts=ui_nav.time.time() if fresh else sent-1)
    monkeypatch.setattr(capture, 'grab_client', lambda _: object())
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation', lambda _: where)
    assert bar_ui._execution_page_ready(SimpleNamespace(hwnd=11))
    assert gates._stage_counts[stage] == (0 if fresh else 2)


@pytest.mark.parametrize('clears', [True, False])
def test_verification_waits_for_transient_shield_without_clicking_it(monkeypatch, clears):
    clicks, clock = [], [0.0]
    monkeypatch.setattr(ui_nav, 'locate_edge', lambda *a, **k: (1620, 55))
    def resolve(_):
        blocked = bool(clicks) and (not clears or clock[0] < 1.4)
        node = 'overlay' if blocked else ('system_menu' if clicks else 'main_hud')
        return UiWhere(node, 1 if clicks else 0, 0, 0, 'direct', stack_ready=True,
            stack_names=('ClickShieldUIState',) if blocked else ('SideNavigationUIInstance',),
            stack_sampled_ts=ui_nav.time.time())
    result = ui_nav.ensure_node('system_menu', grab=lambda: object(), resolve=resolve,
        click=lambda *xy: clicks.append(xy) or True,
        sleep_fn=lambda s: clock.__setitem__(0, clock[0]+s))
    assert result.ok is clears
    assert clicks == [(1620, 55)]
    assert gates._stage_counts['main_hud_menu_system_menu'] == (0 if clears else 1)


def paused_navigation_controller(monkeypatch, tmp_path):
    from mm_assist.life_plan import controller, execution_records
    from mm_assist.life_plan.action_finish import PendingAction
    from mm_assist.life_plan.types import ExecuteResult, PlanItem

    controller.reset_controller_for_tests()
    item = PlanItem('p1', 'gather', '采集 樹液')
    issued = PendingAction(item.id, 1000., ExecuteResult(False, 'action_pending', action_issued=True))
    st = controller.LifeController(running=True, queue=[item], current=item,
        life_started_at=1000., pause_on_failure=True, pending_action=issued)
    monkeypatch.setattr(controller, '_STATE', st)
    monkeypatch.setattr(execution_records, '_RECORDER',
                        execution_records.ExecutionRecorder(tmp_path / 'records.json'))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr('mm_assist.life_plan.facility_sim.tick_facility_sim', lambda **_: None)
    return controller, st


@pytest.mark.parametrize('operation', ['pause', 'resume', 'cycle'])
@pytest.mark.parametrize('attempts', [1, 2])
def test_pause_resume_keeps_unverified_input_budget_and_issued_action(monkeypatch, tmp_path, operation, attempts):
    controller, st = paused_navigation_controller(monkeypatch, tmp_path)
    stage = 'main_hud_menu_system_menu'
    clicks = []
    click = lambda *xy: clicks.append(xy) or True
    for _ in range(attempts):
        assert gates.stage_click(stage, click, 1620, 55)
    ui_nav.note_edge_clicked(stage, 'system_menu', stage=stage)
    pending = dict(ui_nav._PENDING_EDGES)
    if operation in {'pause', 'cycle'}:
        controller.pause_life()
    if operation in {'resume', 'cycle'}:
        controller.resume_life(pause_on_failure=True)
    assert controller.get_controller().life_started_at == 1000.
    assert controller.get_controller().pending_action.result.action_issued
    assert ui_nav._PENDING_EDGES == pending
    if attempts == 1:
        assert gates.stage_click(stage, click, 1620, 55)
    assert gates.stage_click(stage, click, 1620, 55) is None
    assert len(clicks) == 2


@pytest.mark.parametrize('fresh', [True, False])
def test_after_resume_only_new_destination_evidence_reopens_its_step(monkeypatch, tmp_path, fresh):
    controller, _ = paused_navigation_controller(monkeypatch, tmp_path)
    stage, unrelated = 'main_hud_menu_system_menu', 'facility_back_craft_hub'
    for key in [stage, unrelated]:
        assert gates.allow_stage_action(key) and gates.allow_stage_action(key)
    ui_nav.note_edge_clicked(stage, 'system_menu', stage=stage)
    sent = ui_nav._PENDING_EDGES[stage][2]
    controller.pause_life()
    controller.resume_life(pause_on_failure=True)
    where = UiWhere('system_menu', 1, 0, 0, 'direct', stack_ready=True,
        stack_names=('SideNavigationUIInstance',),
        stack_sampled_ts=ui_nav.time.time() if fresh else sent-1)
    ui_nav._verify_late_edges(where, frame_started_at=ui_nav.time.time())
    assert gates.allow_stage_action(stage) is fresh
    assert not gates.allow_stage_action(unrelated)
