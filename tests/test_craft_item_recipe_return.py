"""A potion material card returns to its crafting recipe, then the items list."""
import time
from types import SimpleNamespace

import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import ui_act, ui_logic, ui_nav, insect_obtain as flow
from mm_assist.life_plan import node_inspect as inspect
from mm_assist.memory.readers.ui_stack import UiStackSnap
from mm_assist.perception import insect_obtain_sense as sense
from mm_assist.safety import autofarm_gates as gates

NAMES = ['TutorialTooltipUIState', 'CraftingHubPopupUIInstance',
         'CraftingFacilityPopupUIInstance', 'CraftingPanelPopupUIInstance']


def pixels(name='firefly_recipe_return_20261021'):
    return cv2.imread(str(ROOT / 'tests/fixtures/insect_obtain' / (name + '.png')))


def resolve(frame, names):
    reader = SimpleNamespace(read=lambda **kw: UiStackSnap(ready=True,
        names=names, sampled_ts=time.time()))
    return ui_logic.resolve_navigation(frame, reader=reader)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    ui_nav.reset_edge_memo_for_tests()
    gates.reset_stage_gates_for_tests()
    flow.reset_insect_visits_for_tests()
    monkeypatch.setattr(inspect, '_ocr_lines_roi', lambda *a, **kw: [])
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **kw: False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda **kw: False)
    yield
    ui_nav.reset_edge_memo_for_tests()
    gates.reset_stage_gates_for_tests()
    flow.reset_insect_visits_for_tests()


def test_native_recipe_is_a_child_of_items_not_a_shell_with_exposed_process_tab():
    where = resolve(pixels(), NAMES)
    assert where.node == 'craft_item_recipe'
    assert inspect.format_node_zh(where.node) == '制作-道具-配方'
    hops = ui_nav.path_edges(where.node, 'craft_hub')
    assert [e['id'] for e in hops] == ['craft_item_recipe_back_craft_items',
        'craft_items_back_craft_make', 'craft_make_tab_craft_hub']


@pytest.mark.parametrize('whole_child', [False, True])
def test_real_navigation_closes_card_then_recipe_and_items_before_process(whole_child):
    images = dict(card=pixels('firefly_card_instance_20261021'), recipe=pixels(),
        items=pixels('items_bottom_20261020'),
        make=cv2.imread(str(ROOT/'tests/fixtures/process_stance/craft_make_20261014.png')),
        hub=cv2.imread(str(ROOT/'tests/fixtures/process_stance/craft_process_gate_20261005.png')))
    roles = ['card', 'recipe'] + ([] if whole_child else ['items']) + ['make', 'hub']
    state, clicks = [0], []
    def where(frame):
        role = roles[state[0]]
        if role in {'card', 'recipe', 'items'}:
            names = NAMES + ['DetailCardUIInstance'] if role == 'card' else (
                NAMES if role == 'recipe' else NAMES[:-1])
            return resolve(frame, names)
        return ui_logic.UiWhere('craft_make' if role == 'make' else 'craft_hub', 1, 0, 0,
            'fixture', stack_names=('CraftingHubPopupUIInstance',), stack_ready=True,
            stack_sampled_ts=time.time())
    def click(*xy):
        clicks.append(xy)
        state[0] += 1
        return True
    result = ui_nav.ensure_node('craft_hub', grab=lambda:images[roles[state[0]]], resolve=where,
        click=click, sleep_fn=lambda _:None)
    assert result.ok, result
    assert len(clicks) == (3 if whole_child else 4)
    assert clicks[0][0] > 1800
    assert all(xy == (60, 57) for xy in clicks[1:-1])
    assert 580 < clicks[-1][1] < 740
    assert gates._stage_counts['清界面'] == 0


@pytest.mark.parametrize('case', ['covered', 'stale', 'unready', 'no_panel', 'hidden_title'])
def test_recipe_identity_requires_current_stack_and_independent_pixels(case):
    image, names = pixels(), list(NAMES)
    if case == 'covered': names.append('UnknownPopup')
    if case == 'no_panel': names.pop()
    if case == 'hidden_title': image[630:675, 1230:1450] = 0
    view = SimpleNamespace(ui_stack_names=names, ui_stack_ready=case != 'unready',
        ui_stack_ts=time.time()-(10 if case == 'stale' else 0), ui_stack_error='')
    assert inspect.compute_inspect(image, view=view)['node'] != 'craft_item_recipe'


@pytest.mark.parametrize('case', ['unchanged', 'stale', 'unready', 'covered'])
def test_card_exit_without_current_independent_recipe_proof_keeps_attempt(case):
    clicks = []
    def where(frame):
        names = NAMES + ['DetailCardUIInstance'] if not clicks or case == 'unchanged' else list(NAMES)
        if clicks and case == 'covered': names.append('UnknownPopup')
        reader = SimpleNamespace(read=lambda **kw: UiStackSnap(names=names,
            ready=not (clicks and case == 'unready'),
            sampled_ts=time.time()-(10 if clicks and case == 'stale' else 0)))
        return ui_logic.resolve_navigation(frame, reader=reader)
    def frame():
        return pixels('firefly_card_instance_20261021') if not clicks or case == 'unchanged' else pixels()
    result = ui_nav.ensure_node('craft_hub', grab=frame, resolve=where,
        click=lambda *xy:clicks.append(xy) or True, sleep_fn=lambda _:None)
    assert not result.ok and len(clicks) == 1
    assert gates._stage_counts['清界面'] == 1
