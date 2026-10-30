"""A later insect visit opens the bag after independently leaving crafting."""
import time
from types import SimpleNamespace

import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import insect_obtain as flow, ui_act, ui_logic, ui_nav
from mm_assist.life_plan import node_inspect
from mm_assist.memory.readers.ui_stack import UiStackSnap
from mm_assist.safety import autofarm_gates as gates


def hud():
    return cv2.imread(str(ROOT/'tests/fixtures/insect_obtain/revisit_hud_20261021.png'))


def hud_where(frame):
    reader = SimpleNamespace(read=lambda **_: UiStackSnap(ready=True,
        names=['TutorialTooltipUIState'], sampled_ts=time.time()))
    return ui_logic.resolve_navigation(frame, reader=reader)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    flow.reset_insect_visits_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    gates.reset_stage_gates_for_tests()
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **_: False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda: False)
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **kw: [])
    yield
    flow.reset_insect_visits_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    gates.reset_stage_gates_for_tests()


def test_prepare_bag_from_native_hud_dispatches_registered_bag_edge():
    image, clicks = hud(), []
    assert hud_where(image).node == 'main_hud'
    result = ui_act.prepare_to('inventory', grab=lambda:image, resolve=hud_where,
        click=lambda *xy:clicks.append(xy) or False, sleep_fn=lambda _:None)
    edge = ui_nav.path_edges('main_hud', 'inventory')[0]
    assert clicks == [ui_nav.locate_edge(edge, image)]
    assert not result.ok and result.status == 'click_yield'
    assert gates._stage_counts[ui_nav.edge_stage(edge)] == 0


def test_second_insect_visit_uses_actual_prepare_bag_and_never_recipe_or_source():
    flow._visits['滴答螢火蟲'] = 1
    clicks = []
    image = hud()
    result = flow.execute_insect_obtain(mat='滴答螢火蟲', grab=lambda:image,
        resolve=hud_where, do_click=lambda *xy:clicks.append(xy) or False,
        sleep_fn=lambda _:None)
    edge = ui_nav.path_edges('main_hud', 'inventory')[0]
    assert clicks == [ui_nav.locate_edge(edge, image)]
    assert result.status == 'click_yield' and not result.action_issued
    assert flow._visits == {'滴答螢火蟲': 1}
    assert not flow._source_attempts and not flow._entry_after


def test_close_facility_then_follow_verified_menu_bag_path(monkeypatch):
    roles = ['facility:medicine', 'main_hud', 'system_menu', 'inventory']
    index, clicks = [0], []
    def observe(frame):
        node = roles[index[0]]
        if node == 'main_hud':
            return hud_where(frame)
        names = {'facility:medicine': ('AlteringFacilityPopupUIInstance',),
                 'system_menu': ('SideNavigationUIInstance',),
                 'inventory': ('EquipmentInventoryUIInstance',)}[node]
        return ui_logic.UiWhere(node, 1, 0, 0, 'fixture', stack_ready=True,
            stack_names=names, stack_sampled_ts=time.time())
    def click(*xy):
        clicks.append(roles[index[0]])
        index[0] += 1
        return True
    # Locator substitution is only for pages outside this retained HUD image.
    # The real path, layer preparation, dispatch accounting and verification run.
    original = ui_nav.locate_edge
    monkeypatch.setattr(ui_nav, 'locate_edge', lambda edge, frame, **kw:
        original(edge, frame, **kw) if roles[index[0]] == 'main_hud' else (10, 20))
    monkeypatch.setattr(ui_act, 'wait_inspect_menu', lambda **_: True)
    result = ui_act.prepare_to('inventory', grab=hud, resolve=observe, click=click,
        sleep_fn=lambda _:None)
    assert result.ok and roles[index[0]] == 'inventory'
    assert clicks == ['facility:medicine', 'main_hud', 'system_menu']
    assert gates._stage_counts['main_hud_menu_system_menu'] == 0
    assert gates._stage_counts['system_menu_bag_inventory'] == 0


@pytest.mark.parametrize('case', ['unknown', 'unready', 'cancel'])
def test_bag_preparation_still_refuses_unproved_or_cancelled_page(monkeypatch, case):
    if case == 'cancel':
        monkeypatch.setattr(ui_act, 'pump_aborted', lambda: True)
    where = ui_logic.UiWhere('unknown' if case == 'unknown' else 'main_hud',
        0, 0, 0, 'fixture', stack_ready=case != 'unready')
    result = ui_act.prepare_to('inventory', grab=hud, resolve=lambda _:where,
        click=lambda *_:pytest.fail('unproved page cannot enter the bag'),
        sleep_fn=lambda _:None)
    assert not result.ok
