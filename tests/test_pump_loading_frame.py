"""An stp=0 loading screen completes the old rule, but accepts no new input."""
from types import SimpleNamespace
import cv2
import pytest
from mm_assist.config import ROOT
from mm_assist.life_execute import ui_nav, ui_logic
from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.types import LifeSnapshot, PlanItem, ExecuteResult


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    controller.reset_controller_for_tests()
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly', lambda **k: None)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: '')
    yield
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()


def test_loading_does_not_dispatch_or_skip_next_plan_then_ready_hud_dispatches(monkeypatch):
    item = PlanItem('p1', 'feed', '投料', facility='cloth', recipe='布料')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    loading = cv2.imread(str(ROOT/'tests/fixtures/process_stance/gather_loading.png'))
    hud = cv2.imread(str(ROOT/'tests/fixtures/process_stance/map_open_stale_point.png'))
    assert loading is not None and hud is not None
    images = [loading, hud]
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda _: images[0])
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda _: ui_logic.UiWhere('main_hud',0,0,0,'fixture'))
    calls = []
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
                        lambda item, **k: calls.append(item) or ExecuteResult(True,'done'))
    win = SimpleNamespace(hwnd=7)
    bar_ui.pump_life_once(None,win)
    assert calls == [] and controller.get_controller().current == item
    assert controller.get_controller().running
    images.pop(0)
    bar_ui.pump_life_once(None,win)
    assert calls == [item]


def test_known_facility_is_not_forced_to_hud_before_repair(monkeypatch):
    item = PlanItem('p1','repair','修理')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda _: object())
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda _: ui_logic.UiWhere('facility:cloth',0,0,0,'fixture'))
    monkeypatch.setattr(ui_nav, '_hud_xy', lambda *a, **k: pytest.fail('known facility is prepared by executor'))
    calls = []
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
                        lambda item, **k: calls.append(item) or ExecuteResult(True,'done'))
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=7))
    assert calls == [item]


def test_page_wait_is_bounded_and_skips_unissued_item_without_input(monkeypatch):
    first = PlanItem('p1','gather','采集',recipe='淨淨蘑菇')
    next_item = PlanItem('p2','repair','修理')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [first,next_item])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    monkeypatch.setattr(bar_ui,'_execution_page_ready',lambda _:False)
    recoveries = []
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',
        lambda *a, **k: recoveries.append(1) or {'ok': False})
    clock = [0.0]
    monkeypatch.setattr(bar_ui.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda *a,**k:pytest.fail('loading accepts no input'))
    for ts in (0.0,59.0):
        clock[0] = ts
        bar_ui.pump_life_once(None,SimpleNamespace(hwnd=7))
        assert controller.get_controller().current == first
    clock[0] = 60.0
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=7))
    assert controller.get_controller().current == next_item
    assert recoveries == [1, 1]


@pytest.mark.parametrize(('entry_menu', 'allow_dialog'), [(False, False), (True, False), (False, True)])
def test_recognized_npc_goodbye_reaches_real_prepare_and_closes_before_dispatch(monkeypatch, entry_menu, allow_dialog):
    from mm_assist.life_execute import ui_act
    from mm_assist.safety import autofarm_gates as gates
    gates.reset_stage_gates_for_tests()
    item = PlanItem('p1', 'gather', '采集', recipe='淨淨蘑菇')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    goodbye = cv2.imread(str(ROOT/'tests/fixtures/repair/nerys_after_end_dialog.png'))
    hud = cv2.imread(str(ROOT/'tests/fixtures/process_stance/map_open_stale_point.png'))
    frames = [goodbye, hud]
    if entry_menu:
        frames.insert(0, cv2.imread(str(ROOT/'tests/fixtures/repair/nerys_repair_done_end.png')))
    monkeypatch.setattr('mm_assist.perception.ocr.ocr_roi',
                        lambda *a, **k: pytest.fail('exit needs independent templates, not OCR'))
    def grab():
        return frames[0]
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda _: grab())
    monkeypatch.setattr(ui_logic, 'resolve_navigation',
        lambda _: ui_logic.UiWhere('main_hud', 0, 0, 0, 'fixture', stack_ready=True,
                                   stack_sampled_ts=__import__('time').time()))
    clicks = []
    dispatched = []
    def click(x, y):
        clicks.append((x, y))
        frames.pop(0)
        return True
    def execute(item, **_):
        # Keep the actual preparation and goodbye detector/click/wait chain.
        result = ui_act.prepare_to('main_hud', grab=grab, click=click,
                                  sleep_fn=lambda _: None, read_stp=lambda: 0,
                                  allow_dialog=allow_dialog)
        if result.ok:
            assert frames[0] is hud
            dispatched.append(item)
        return result
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **_: False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda **_: False)
    monkeypatch.setattr(ui_act, 'inspect_is_true_hud', lambda **_: False)
    monkeypatch.setattr(ui_act, 'inspect_is_hud_idle', lambda **_: False)
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=7))
    assert dispatched == [item]
    assert len(clicks) == (2 if entry_menu else 1)
    assert clicks[-1] == (round(goodbye.shape[1] * .5), round(goodbye.shape[0] * .5))
    if entry_menu:
        assert 1100 < clicks[0][0] < 1180 and 1310 < clicks[0][1] < 1380
    gates.reset_stage_gates_for_tests()
