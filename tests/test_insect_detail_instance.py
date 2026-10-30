"""The potion's material card may publish only DetailCardUIInstance."""
import time
from types import SimpleNamespace

import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import insect_obtain as flow, ui_act, ui_logic
from mm_assist.life_plan import node_inspect as inspect
from mm_assist.memory.readers.ui_stack import UiStackSnap
from mm_assist.perception import insect_obtain_sense as sense
from mm_assist.safety import autofarm_gates as gates

NAMES = ['TutorialTooltipUIState', 'CraftingHubPopupUIInstance',
         'CraftingFacilityPopupUIInstance', 'CraftingPanelPopupUIInstance', 'DetailCardUIInstance']


def pixels():
    return cv2.imread(str(ROOT / 'tests/fixtures/insect_obtain/firefly_card_instance_20261021.png'))


def view(**extra):
    values = dict(ui_stack_names=list(NAMES), ui_stack_ready=True,
                  ui_stack_ts=time.time(), ui_stack_error='')
    values.update(extra)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    flow.reset_insect_visits_for_tests()
    gates.reset_stage_gates_for_tests()
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr(inspect, '_ocr_lines_roi', lambda *a, **kw: [])
    yield
    flow.reset_insect_visits_for_tests()
    gates.reset_stage_gates_for_tests()


def test_native_instance_card_is_named_item_info_before_underlying_facility():
    frame = pixels()
    assert sense.phase(frame, '滴答螢火蟲') == 'card'
    result = inspect.compute_inspect(frame, view=view())
    assert result['node'] == 'item_info:滴答螢火蟲'
    assert result['cost']['ocr_n'] == 0


def test_late_native_card_verifies_prior_material_click_without_resending_it():
    stage = '虫料入口:滴答螢火蟲:滴答螢火蟲'
    flow._entry_after[stage] = ('card', 999.)
    flow._entry_attempts[stage] = 1
    assert gates.allow_stage_action(stage)
    clicks = []
    reader = SimpleNamespace(read=lambda **kw: UiStackSnap(
        ready=True, names=list(NAMES), sampled_ts=time.time()))
    result = flow.execute_insect_obtain(mat='滴答螢火蟲', grab=pixels,
        resolve=lambda frame: ui_logic.resolve_navigation(frame, reader=reader),
        do_click=lambda *xy: clicks.append(xy) or False,
        now_fn=lambda: 1000., sleep_fn=lambda _: None)
    assert clicks == [sense.locate(pixels(), 'obtain_button')]
    assert not result.ok and not result.action_issued and not flow._visits
    assert stage not in flow._entry_after and gates._stage_counts[stage] == 0


def test_card_stack_change_during_pixel_recognition_is_rejected():
    samples = iter([UiStackSnap(ready=True, names=list(NAMES), sampled_ts=time.time()),
                    UiStackSnap(ready=True, names=NAMES[:-1], sampled_ts=time.time())])
    where = ui_logic.resolve_navigation(pixels(), reader=SimpleNamespace(read=lambda **kw: next(samples)))
    assert where.node == 'unknown' and where.source == 'direct_transition'


@pytest.mark.parametrize('case', ['unready', 'stale', 'error', 'covered', 'no_card',
                                  'wrong_context', 'hidden_title', 'hidden_method'])
def test_instance_without_independent_current_card_proof_is_not_named(case):
    frame, extra = pixels(), {}
    if case == 'unready':
        extra['ui_stack_ready'] = False
    elif case == 'stale':
        extra['ui_stack_ts'] = time.time() - 10
    elif case == 'error':
        extra['ui_stack_error'] = 'stack_read_reused'
    elif case == 'covered':
        extra['ui_stack_names'] = NAMES + ['UnknownPopup']
    elif case == 'no_card':
        extra['ui_stack_names'] = NAMES[:-1]
    elif case == 'wrong_context':
        extra['ui_stack_names'] = ['DetailCardUIInstance']
    elif case == 'hidden_title':
        frame[445:500, 675:900] = 0
    elif case == 'hidden_method':
        frame[945:995, 675:820] = 0
    assert inspect.compute_inspect(frame, view=view(**extra))['node'] != 'item_info:滴答螢火蟲'
