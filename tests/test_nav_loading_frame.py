"""Internal navigation must observe loading without attempting HUD input."""
import cv2
import pytest
from mm_assist.config import ROOT
from mm_assist.life_execute import ui_nav, ui_act
from mm_assist.life_execute.ui_logic import UiWhere
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests


@pytest.fixture(autouse=True)
def isolated_navigation():
    reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    yield
    reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()


def setup_navigation(monkeypatch, recover_at):
    loading = cv2.imread(str(ROOT/'tests/fixtures/process_stance/claim_loading.png'))
    hud = cv2.imread(str(ROOT/'tests/fixtures/process_stance/map_open_stale_point.png'))
    assert loading is not None and hud is not None
    loading.setflags(write=False)
    hud.setflags(write=False)
    # These tests replay immutable source frames hundreds of times to exercise
    # the real navigation timeout/cancel loop. Match each source once using the
    # real locator; fresh/changed production frames never use this test cache.
    locate_hud = ui_nav._hud_xy
    locations = {}
    def static_hud_xy(image, entry_id, *, root=None):
        if image is None or image.flags.writeable:
            return locate_hud(image, entry_id, root=root)
        key = (id(image), entry_id, root)
        record = locations.get(key)
        if record is None or record[0] is not image:
            record = (image, locate_hud(image, entry_id, root=root))
            locations[key] = record
        return record[1]
    monkeypatch.setattr(ui_nav, '_hud_xy', static_hud_xy)
    clock, clicks = [0.0], []
    monkeypatch.setattr(ui_nav.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(ui_nav, 'resolve_navigation', lambda _: UiWhere(
        'system_menu' if clicks else 'main_hud', 0, 0, 0, 'fixture'))
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **k: False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda: False)
    monkeypatch.setattr('mm_assist.tool_repair.op_cancelled', lambda: False)
    reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    def sleep(seconds):
        clock[0] += seconds
    def click(x,y):
        assert clock[0] >= recover_at
        clicks.append((x,y))
        return True
    return clock, clicks, lambda: hud if clock[0] >= recover_at else loading, sleep, click


def test_internal_menu_navigation_waits_through_loading_then_clicks_once(monkeypatch):
    clock, clicks, grab, sleep, click = setup_navigation(monkeypatch, recover_at=4.0)
    result = ui_nav.ensure_node('system_menu', grab=grab, click=click, sleep_fn=sleep)
    assert result.ok and len(clicks) == 1 and clock[0] >= 4.0


def test_loading_navigation_times_out_without_input(monkeypatch):
    clock, clicks, grab, sleep, click = setup_navigation(monkeypatch, recover_at=100.0)
    result = ui_nav.ensure_node('system_menu', grab=grab, click=click, sleep_fn=sleep)
    assert not result.ok and not clicks
    assert 60 <= clock[0] < 61


def test_loading_navigation_cancel_stops_read_only_wait(monkeypatch):
    clock, clicks, grab, sleep, click = setup_navigation(monkeypatch, recover_at=100.0)
    monkeypatch.setattr('mm_assist.tool_repair.op_cancelled', lambda: clock[0] >= .3)
    result = ui_nav.ensure_node('system_menu', grab=grab, click=click, sleep_fn=sleep)
    assert result.status == 'cancel' and not clicks and clock[0] < 1


@pytest.mark.parametrize('recover_at', [4.0, 100.0])
def test_prepare_to_live_resolver_waits_for_hud_controls(monkeypatch, recover_at):
    # prepare_to supplies its own live resolver to ensure_node. That production
    # path must retain the same read-only loading wait as direct navigation.
    clock, clicks, grab, sleep, click = setup_navigation(monkeypatch, recover_at)
    cinematic = cv2.imread(str(ROOT/'tests/fixtures/process_stance/mushroom_region_cinematic.png'))
    assert cinematic is not None
    cinematic.setflags(write=False)
    monkeypatch.setattr(ui_act, 'resolve_navigation', ui_nav.resolve_navigation, raising=False)
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation', ui_nav.resolve_navigation)
    monkeypatch.setattr(ui_act, 'promote_true_hud', lambda where: where)
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr('mm_assist.life_execute.repair_dialog.dismiss_world_npc_menu', lambda *a, **k: False)
    monkeypatch.setattr('mm_assist.life_execute.repair_dialog.npc_goodbye_visible', lambda *a: False)
    result = ui_act.prepare_to('system_menu',
        grab=lambda: grab() if clock[0] >= recover_at else cinematic,
        click=click, sleep_fn=sleep, read_stp=lambda: 0)
    assert result.ok == (recover_at < 60)
    assert len(clicks) == (1 if recover_at < 60 else 0)
    assert clock[0] >= min(recover_at, 60)
