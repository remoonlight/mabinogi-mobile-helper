"""Retained p2 loading title has a ready HUD shell, but no closable page."""
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import ui_act, ui_logic, ui_nav
from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.action_finish import StpSample
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem
from mm_assist.memory.readers.ui_stack import UiStackSnap
from mm_assist.safety import autofarm_gates as gates


@pytest.fixture
def loading(monkeypatch):
    image = cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/life_faults_20261024/loading_to_coal.png',
                                    dtype=np.uint8), 1)
    assert image is not None
    reader = SimpleNamespace(read=lambda **_: UiStackSnap(ready=True,
        names=['HudUIStepState', 'TutorialTooltipUIState'], sampled_ts=time.time()))
    native_resolve = ui_logic.resolve_navigation
    resolve = lambda frame: native_resolve(frame, reader=reader)
    assert resolve(image).node == 'overlay'
    monkeypatch.setattr(ui_logic, 'resolve_navigation', resolve)
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda _: image)
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr(ui_act, 'inspect_is_hud_idle', lambda **_: True)
    monkeypatch.setattr(ui_act, 'inspect_is_true_hud', lambda **_: True)
    monkeypatch.setattr(ui_act, 'inspect_is_menu', lambda **_: True)  # stale display
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **_: False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda **_: False)
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    yield image, resolve
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()


def test_loading_title_is_not_ready_for_new_dispatch(loading):
    assert not bar_ui._execution_page_ready(SimpleNamespace(hwnd=11))


@pytest.mark.parametrize('entry', ['prepare_native', 'prepare_injected', 'ensure_native', 'ensure_injected'])
def test_loading_after_pump_gate_never_searches_for_close_or_clicks(loading, monkeypatch, entry):
    image, resolve = loading
    monkeypatch.setattr(ui_nav, 'resolve_navigation', resolve)
    monkeypatch.setattr(ui_nav, 'locate_edge', lambda *a, **k: pytest.fail('loading has no usable exit'))
    # Keep any prior input facts charged while the new frame is loading.
    gates._stage_counts['main_hud_menu_system_menu'] = 1
    ui_nav.note_edge_clicked('main_hud_menu_system_menu', 'system_menu')
    before = dict(ui_nav._PENDING_EDGES)
    execute = ui_act.prepare_to if entry.startswith('prepare') else ui_nav.ensure_node
    result = execute('gather_skill', skill_id='mine', grab=lambda: image,
        click=lambda *xy: pytest.fail('loading must be read-only'),
        resolve=resolve if entry.endswith('injected') else None,
        sleep_fn=lambda _: None, **({'read_stp': lambda: 0} if entry.startswith('prepare') else {}))
    assert not result.ok and not result.action_issued and result.status == 'input_layer_wait'
    assert gates._stage_counts['main_hud_menu_system_menu'] == 1
    assert ui_nav._PENDING_EDGES == before


@pytest.mark.parametrize('interruption', ['none', 'pause', 'new_start', 'exception'])
def test_real_pump_loading_wait_preserves_queue_and_run(loading, monkeypatch, interruption):
    first = PlanItem('p2', 'gather', '采集 煤炭')
    tail = PlanItem('p3', 'claim', '领取 木材', facility='wood')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [first, tail])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000, pause_on_failure=True)
    monkeypatch.setattr('mm_assist.life_plan.action_finish.read_completion_stp',
                        lambda: StpSample(0, time.time(), 1, True))
    gate = bar_ui._execution_page_ready
    def page_ready(_):
        value = gate(SimpleNamespace(hwnd=11))
        if interruption == 'pause': controller.pause_life()
        elif interruption == 'new_start':
            controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000, pause_on_failure=True)
        elif interruption == 'exception': raise RuntimeError('offline capture failed')
        return value
    monkeypatch.setattr(bar_ui, '_execution_page_ready', page_ready)
    calls = []
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
        lambda *a, **k: calls.append('execute') or ExecuteResult(True, 'done'))
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',
                        lambda *a, **k: pytest.fail('loading is not recovery'))
    if interruption == 'exception':
        with pytest.raises(RuntimeError, match='offline capture failed'):
            bar_ui.pump_life_once(None, SimpleNamespace(hwnd=0))
    else:
        bar_ui.pump_life_once(None, SimpleNamespace(hwnd=0))
    state = controller.get_controller()
    assert not calls and state.current == first and state.queue == [first, tail]
    assert state.pending_action is None
    assert state.paused if interruption == 'pause' else state.running
    assert state.life_started_at == (2000 if interruption == 'new_start' else 1000)
    if interruption == 'none':
        monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda _: True)
        bar_ui.pump_life_once(None, SimpleNamespace(hwnd=0))
        assert calls == ['execute']


@pytest.mark.parametrize('node,names', [
    ('facility:metal', ['AlteringFacilityPopupUIInstance']),
    ('overlay', ['AlteringFacilityPopupUIInstance', 'ClickShieldUIState',
                 'ButtonCommonPopupUIState', 'ButtonCommonPopupInstance']),
])
def test_identified_facility_and_its_confirmation_keep_preparation_path(loading, monkeypatch, node, names):
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda _: ui_logic.UiWhere(
        node, 1, 0, 0, 'fixture', stack_ready=True, stack_names=names))
    assert bar_ui._execution_page_ready(SimpleNamespace(hwnd=11))
