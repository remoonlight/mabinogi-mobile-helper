"""Retained click-shield stack must block input on the visible underlying hub."""
from types import SimpleNamespace
import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import ui_act, ui_logic, ui_nav
from mm_assist.memory.readers.ui_stack import UiStackSnap


@pytest.mark.parametrize('stopped', [False, True])
def test_input_guard_checks_game_stop_once_and_still_checks_cancellation(monkeypatch, stopped):
    from mm_assist.safety import input as inputs, lease
    from mm_assist.life_plan import game_stop
    from mm_assist.memory.readers import ui_stack
    events = []
    monkeypatch.setattr(lease, 'cancel_requested', lambda: False)
    monkeypatch.setattr(game_stop, 'game_stop_interrupted',
                        lambda **kw: events.append(('frame', kw.get('force', False))) or stopped)
    monkeypatch.setattr('mm_assist.jobs.bar.click_owner_block_reason', lambda: '')
    monkeypatch.setattr(lease, 'in_test_mode', lambda: False)
    monkeypatch.setattr(lease, 'require_action', lambda *a, **kw: events.append('authorized'))
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: SimpleNamespace(read=lambda **kw:
        events.append('stack') or UiStackSnap(ready=True, names=['TutorialTooltipUIState'])))
    if stopped:
        with pytest.raises(inputs.AbortRequested, match='game_stop_paused'):
            inputs._guard_before_input('click', SimpleNamespace(hwnd=11), method='sendinput')
    else:
        inputs._guard_before_input('click', SimpleNamespace(hwnd=11), method='sendinput')
    assert events == [('frame', True)] + ([] if stopped else ['authorized', 'stack'])
    events.clear()
    monkeypatch.setattr(lease, 'cancel_requested', lambda: True)
    with pytest.raises(inputs.AbortRequested, match='user abort'):
        inputs._guard_before_input('click', SimpleNamespace(hwnd=11), method='sendinput')
    assert not events


@pytest.mark.parametrize('layer', ['ClickShieldUIState', 'ToyTransparentLayerUIInstance'])
def test_pump_known_shield_waits_before_dispatching_a_plan(monkeypatch, layer):
    from mm_assist.life_plan import bar_ui
    from mm_assist.platform import capture
    monkeypatch.setattr(capture, 'grab_client', lambda _: object())
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda _: ui_logic.UiWhere(
        'overlay', 1, 0, 0, 'direct+mem', stack_ready=True,
        stack_names=('NewProfileUIInstance', layer), stack_sampled_ts=1.0))
    assert not bar_ui._execution_page_ready(SimpleNamespace(hwnd=11))


def test_pump_preserves_identified_facility_travel_confirmation(monkeypatch):
    from mm_assist.life_plan import bar_ui
    from mm_assist.platform import capture
    monkeypatch.setattr(capture, 'grab_client', lambda _: object())
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda _: ui_logic.UiWhere(
        'overlay', 1, 0, 0, 'direct+mem', stack_ready=True,
        stack_names=('AlteringFacilityPopupUIInstance', 'ClickShieldUIState',
                     'ButtonCommonPopupUIState', 'ButtonCommonPopupInstance')))
    # Only the existing exact popup/point guard may authorize its foreground
    # button; a pump-level wait must not make that existing branch unreachable.
    assert bar_ui._execution_page_ready(SimpleNamespace(hwnd=11))


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests
    reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **k: False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda *a, **k: False)
    monkeypatch.setattr('mm_assist.life_plan.node_inspect._ocr_blob', lambda *a, **k: '')
    yield
    reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()


@pytest.mark.parametrize('mode', ['clears', 'timeout', 'cancel'])
def test_live_hub_stack_waits_for_shield_before_real_card_location(monkeypatch, mode):
    frame = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/hub_with_toy_layer.png'))
    arrived = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/metal_standalone.png'))
    clock, clicks = [0.0], []
    def blocked():
        return mode != 'clears' or clock[0] < .45
    class Reader:
        def read(self, **k):
            names = ['CraftingHubPopupUIInstance']
            if clicks:
                names.append('AlteringFacilityPopupUIInstance')
            elif blocked():
                names.append('ToyTransparentLayerUIInstance')
            return UiStackSnap(names=names, ready=True)
    reader = Reader()
    def click(x, y):
        assert not blocked(), 'visible underlying card is shielded'
        clicks.append((x,y))
        return True
    monkeypatch.setattr('mm_assist.tool_repair.op_cancelled', lambda: mode == 'cancel' and clock[0] >= .3)
    result = ui_nav.ensure_node('facility', facility='metal',
        grab=lambda: arrived if clicks else frame, click=click,
        resolve=lambda f: ui_logic.resolve_navigation(f, reader=reader),
        sleep_fn=lambda s: clock.__setitem__(0, clock[0]+s))
    if mode == 'clears':
        assert result.ok and clicks == [(416,402)]
    else:
        assert result.status == ('cancel' if mode == 'cancel' else 'click_ui_blocked')
        assert not clicks


@pytest.mark.parametrize('ready,names,error', [
    (True,['CraftingHubPopupUIInstance','ToyTransparentLayerUIInstance'],'click_ui_blocked'),
    (False,[],'click_ui_not_ready'),
    (True,['CraftingHubPopupUIInstance'],None),
])
def test_native_dispatch_rechecks_shield_after_authorization(monkeypatch, ready, names, error):
    from mm_assist.safety import input as inputs, lease
    from mm_assist.memory.readers import ui_stack
    events = []
    monkeypatch.setattr(inputs, 'check_abort', lambda: None)
    monkeypatch.setattr('mm_assist.jobs.bar.click_owner_block_reason', lambda: '')
    monkeypatch.setattr(lease, 'in_test_mode', lambda: False)
    monkeypatch.setattr(lease, 'require_action', lambda *a, **k: events.append('authorized'))
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: SimpleNamespace(
        read=lambda **k: events.append('stack') or UiStackSnap(ready=ready, names=names)))
    if error:
        with pytest.raises(RuntimeError, match=error):
            inputs._guard_before_input('click', SimpleNamespace(hwnd=11), method='sendinput')
    else:
        inputs._guard_before_input('click', SimpleNamespace(hwnd=11), method='sendinput')
    assert events == ['authorized','stack']


@pytest.mark.parametrize('case', ['allowed', 'underlying_recipe', 'underlying', 'drag', 'key', 'no_popup',
    'no_facility', 'toy_layer', 'recipe', 'resized', 'expired', 'stale', 'not_ready'])
def test_confirm_scope_checks_actual_target_and_native_stack(monkeypatch, case):
    import time
    from mm_assist.safety import input as inputs, lease
    from mm_assist.memory.readers import ui_stack
    names = ['ClickShieldUIState', 'AlteringFacilityPopupUIInstance', 'ButtonCommonPopupInstance']
    if case == 'underlying_recipe':
        names.insert(2, 'AlteringPanelPopupUIInstance')
    if case == 'no_popup':
        names.remove('ButtonCommonPopupInstance')
    if case == 'no_facility':
        names.remove('AlteringFacilityPopupUIInstance')
    if case == 'toy_layer':
        names.append('ToyTransparentLayerUIInstance')
    if case == 'recipe':
        names.append('AlteringPanelPopupUIInstance')
    now = [time.time()]
    monkeypatch.setattr(inputs.time, 'time', lambda: now[0])
    monkeypatch.setattr(inputs, 'check_abort', lambda: None)
    monkeypatch.setattr('mm_assist.jobs.bar.click_owner_block_reason', lambda: '')
    monkeypatch.setattr(lease, 'in_test_mode', lambda: False)
    monkeypatch.setattr(lease, 'require_action', lambda *a, **k: None)
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: SimpleNamespace(read=lambda **k:
        UiStackSnap(ready=case != 'not_ready', names=names, sampled_ts=now[0]-(3 if case == 'stale' else 0))))
    win = SimpleNamespace(hwnd=11, width=1000, height=700 if case == 'resized' else 800)
    kind = case if case in {'drag', 'key'} else 'click'
    point = (100, 200) if case == 'underlying' else (600, 700)
    with inputs.facility_confirm_input_scope((600, 700), (1000, 800)):
        if case == 'expired':
            now[0] += 3
        if case in {'allowed', 'underlying_recipe'}:
            inputs._guard_before_input(kind, win, method='sendinput', client_xy=point)
        else:
            with pytest.raises(RuntimeError, match='click_ui_'):
                inputs._guard_before_input(kind, win, method='sendinput', client_xy=point)
    # Leaving the scope, including a failed dispatch, restores the default shield ban.
    with pytest.raises(RuntimeError, match='click_ui_'):
        inputs._guard_before_input('click', win, method='sendinput', client_xy=(600, 700))
