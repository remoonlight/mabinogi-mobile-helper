"""Bangor loading title with only the tutorial shell is unknown, not a facility."""
import hashlib
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import ui_logic
from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.action_finish import StpSample
from mm_assist.life_plan.types import LifeSnapshot, PlanItem
from mm_assist.memory.readers.ui_stack import UiStackSnap
from test_claim_travel_busy import travel


RAW = ROOT/'tests/fixtures/life_faults_20261025/loading_to_bangor_claim.png'


def loading_frame():
    frame = cv2.imdecode(np.fromfile(RAW, np.uint8), 1)
    assert frame is not None
    return frame


def native_reader():
    return SimpleNamespace(read=lambda **_: UiStackSnap(ready=True,
        names=['TutorialTooltipUIState'], sampled_ts=time.time()))


def test_loading_fixture_preserves_the_original_frame():
    meta = json.loads(RAW.with_suffix('.json').read_text('utf-8'))
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == meta['source_sha256']
    assert meta['major_action_issued'] is False and meta['input_requests'] == 7


def test_real_unknown_loading_page_is_only_a_readonly_wait():
    where = ui_logic.resolve_navigation(loading_frame(), reader=native_reader())
    assert where.node == 'unknown' and where.stack_ready
    assert ui_logic.hud_page_pending(where)
    assert where.node != 'main_hud' and not where.node.startswith('facility')


@pytest.mark.parametrize('value', [0, None])
def test_unknown_loading_after_travel_does_not_fail_or_repeat_go(travel, monkeypatch, value):
    state, clicks, run = travel
    original = ui_logic.resolve_navigation
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda frame: original(frame, reader=native_reader()))
    state.update(arrival_image=loading_frame(), moving=value)
    result = run()
    assert result.status == 'input_layer_wait' and not result.action_issued
    assert clicks == [(223, 305)] and state['clock'] <= 1002.5


def test_real_pump_keeps_unknown_loading_plan_without_reexecuting_travel(travel, monkeypatch):
    state, clicks, run = travel
    frame = loading_frame()
    state.update(arrival_image=frame, moving=0)
    original = ui_logic.resolve_navigation
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda frame: original(frame, reader=native_reader()))
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda _: frame.copy())
    item = PlanItem('p1', 'claim', '领取 木材', facility='wood')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000, pause_on_failure=True)
    monkeypatch.setattr('mm_assist.life_plan.action_finish.read_completion_stp',
        lambda: StpSample(0, state['clock'], 1, True))
    real_gate = bar_ui._execution_page_ready
    monkeypatch.setattr(bar_ui, '_execution_page_ready',
        lambda _: real_gate(SimpleNamespace(hwnd=7)) if state['travel'] else True)
    calls = []
    def execute(*a, **k):
        calls.append(1)
        return run()
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',
        lambda *a, **k: pytest.fail('loading cannot authorize recovery'))
    for _ in range(3):
        bar_ui.pump_life_once(None, SimpleNamespace(hwnd=0))
        state['clock'] += .2
        current = controller.get_controller()
        assert current.running and not current.paused and current.current == item
        assert current.pending_action is None
    assert calls == [1] and clicks == [(223, 305)]


@pytest.mark.parametrize('node,ready,names', [
    ('unknown', False, ('TutorialTooltipUIState',)),
    ('unknown', True, ()),
    ('unknown', True, ('TutorialTooltipUIState', 'UnknownPopup')),
    ('facility:wood', True, ('TutorialTooltipUIState', 'AlteringFacilityPopupUIInstance')),
    ('gather_skill:herb', True, ('LivingSkillUIInstance',)),
])
def test_pending_does_not_turn_other_or_unready_pages_into_hud(node, ready, names):
    where = ui_logic.UiWhere(node, None, None, None, 'offline', stack_ready=ready, stack_names=names)
    assert not ui_logic.hud_page_pending(where)
