"""Original Bankor popup across the real claim and shared confirmation path."""
import hashlib
import json
import time
from types import SimpleNamespace

import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import ui_act
from mm_assist.life_execute.medicine_process import execute_facility_claim
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests


@pytest.fixture(autouse=True)
def isolated_gates():
    reset_stage_gates_for_tests()
    yield
    reset_stage_gates_for_tests()


@pytest.mark.parametrize('after_claim', [False, True])
@pytest.mark.parametrize('confirmation', ['closed', 'refused', 'unchanged'])
@pytest.mark.parametrize('destination', ['bankor', 'kuhan'])
def test_claim_uses_confirm_on_actual_destination_popup(monkeypatch, after_claim, confirmation, destination):
    from mm_assist.safety import input as inputs, lease
    from mm_assist.memory.readers import ui_stack
    path = ROOT/f'tests/fixtures/process_stance/go_facility_{destination}_20261006.png'
    meta = json.loads(path.with_suffix('.json').read_text('utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['source_sha256']
    frame = cv2.imread(str(path))
    window = cv2.imread(str(ROOT/meta['source_window']))
    assert hashlib.sha256((ROOT/meta['source_window']).read_bytes()).hexdigest() == meta['source_window_sha256']
    import numpy as np
    x, y, w, h = meta['client_from_window_px']
    assert np.array_equal(frame, window[y:y+h, x:x+w])
    bx, by, bw, bh = meta['expected_confirm_rect_px']
    state = {'popup': not after_claim, 'result': False, 'claimed': False}
    clicks = []
    # Reproduce the native guard that previously rejected the popup's own shield.
    monkeypatch.setattr(inputs, 'check_abort', lambda: None)
    monkeypatch.setattr('mm_assist.jobs.bar.click_owner_block_reason', lambda: '')
    monkeypatch.setattr(lease, 'in_test_mode', lambda: False)
    monkeypatch.setattr(lease, 'require_action', lambda *a, **k: None)
    def native_stack(**_):
        names = ['AlteringFacilityPopupUIInstance']
        if state['popup']:
            names += ['ClickShieldUIState', 'ButtonCommonPopupUIState', 'ButtonCommonPopupInstance']
        return ui_stack.UiStackSnap(ready=True, names=names, sampled_ts=time.time())
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: SimpleNamespace(read=native_stack))
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_recipe', lambda: False)
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **_: state['popup'])
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda **_: state['result'])
    monkeypatch.setattr('mm_assist.perception.process_sense.detect_facility_page',
                        lambda *a, **k: {'ok': True, 'facility': 'metal'})
    # A visible underlying travel button must not win over the active popup.
    monkeypatch.setattr('mm_assist.perception.process_sense.locate_go_equip',
                        lambda *a, **k: {'ok': state['popup'], 'click_client': [200, 300]})
    monkeypatch.setattr('mm_assist.perception.process_sense.locate_claim_all',
                        lambda *a, **k: {'ok': True, 'click_client': [80, 1200]})
    monkeypatch.setattr('mm_assist.perception.process_sense.locate_result_confirm',
                        lambda *a, **k: {'ok': True, 'click_client': [960, 1280]})
    def view(**_):
        names = ['AlteringFacilityPopupUIInstance']
        if state['popup']:
            names.append('ButtonCommonPopupInstance')
        if state['result']:
            names.append('FacilityResultUIInstance')
        return SimpleNamespace(fresh=True, pid_alive=True, raw={
            'ui_stack_names': names, 'ui_stack_ready': True, 'ui_stack_ts': time.time()})
    monkeypatch.setattr(ui_act, 'read_state_ui', view)
    monkeypatch.setattr('mm_assist.perception.state_ui.read_state_ui', view)
    def click(cx, cy):
        inputs._guard_before_input('click', SimpleNamespace(hwnd=11, width=frame.shape[1],
            height=frame.shape[0]), method='sendinput', client_xy=(cx, cy))
        clicks.append((cx, cy))
        if bx <= cx <= bx+bw and by <= cy <= by+bh:
            if confirmation == 'refused':
                return False
            if confirmation == 'closed':
                state['popup'] = False
                state['result'] = state['claimed']
            return True
        if (cx, cy) == (80, 1200):
            assert not state['popup'], 'clicked covered claim button'
            state['claimed'] = True
            state['popup'] = after_claim
            state['result'] = not after_claim
            return True
        if (cx, cy) == (960, 1280):
            state['result'] = False
            return True
        pytest.fail(f'Unexpected input {cx, cy}; travel popup must use recognized confirm')
    result = execute_facility_claim(facility='metal', items={},
        do_click=click, grab=lambda: frame, read_stp=lambda: 0, defer_completion=True)
    confirm_clicks = [(cx, cy) for cx, cy in clicks if bx <= cx <= bx+bw and by <= cy <= by+bh]
    assert len(confirm_clicks) == 1
    if confirmation == 'closed':
        assert result.ok and result.status == 'done'
        assert clicks.count((80, 1200)) == 1
    else:
        assert not result.ok
        assert clicks.count((80, 1200)) == int(after_claim)
        assert (960, 1280) not in clicks
