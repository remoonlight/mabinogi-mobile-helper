from __future__ import annotations

import time

import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.perception import sap_sense

from mm_assist.life_execute.obtain_gather import execute_obtain_gather
from mm_assist.life_plan.execute import try_execute
from mm_assist.life_plan.types import PlanItem


def test_sap_repeated_plan_leaves_inventory_for_softwood_recipe(monkeypatch):
    from mm_assist.life_execute import ui_act, ui_logic
    from mm_assist.life_plan import node_inspect
    from mm_assist.life_plan.types import ExecuteResult

    monkeypatch.setattr(node_inspect, 'compute_inspect', lambda *a, **k: {'node': 'inventory'})
    monkeypatch.setattr(node_inspect, 'locate_spot', lambda *a, **k: (100, 100))
    monkeypatch.setattr(ui_logic, 'gather_detail_visible', lambda *a, **k: False)
    monkeypatch.setattr('mm_assist.perception.item_sense.locate_find_nearby', lambda *a, **k: {'ok': False})
    prepare, clicks = [], []

    def prepare_to(target, **kwargs):
        prepare.append((target, kwargs.get('facility'), kwargs.get('item_name')))
        return ExecuteResult(False, 'fixture_stop')

    monkeypatch.setattr(ui_act, 'prepare_to', prepare_to)
    for _ in range(2):
        result = execute_obtain_gather(mat='樹液', grab=object,
            do_click=lambda x, y: clicks.append((x, y)) or True)
        assert result.status == 'fixture_stop'
    assert prepare == [('facility_recipe', 'wood', '柔軟木材')] * 2
    assert not clicks


def test_coal_not_in_obtain_mats():
    from mm_assist.life_execute.obtain_gather import OBTAIN_MATS

    assert "煤炭" not in OBTAIN_MATS
    assert "樹液" in OBTAIN_MATS
    r = execute_obtain_gather(mat="煤炭", items={"煤炭": 0})
    assert r.status == "bad_mat"


def test_obtain_gather_needs_window():
    r = execute_obtain_gather(mat="樹液", items={"樹液": 0})
    assert r.status == "need_context"


def test_try_execute_sap_obtain():
    item = PlanItem(
        id="p1",
        kind="gather",
        title="采集 樹木",
        skill="伐木",
        detail="樹木",
        recipe="樹液",
        gather_mode="process_mat",
    )
    assert try_execute(item, items={}).status == "need_context"



@pytest.fixture
def sap_frames():
    folder = ROOT / 'tests/fixtures/process_stance/sap_softwood_20261126'
    return {name: cv2.imread(str(folder / f'{name}.png'))
            for name in ('list', 'recipe', 'card', 'obtain')}


def test_sap_navigation_replays_recipe_card_and_source_twice(monkeypatch, sap_frames):
    from mm_assist.life_execute import ui_act, ui_logic, ui_nav
    from mm_assist.life_plan.node_inspect import refine_inspect_node

    original_locate_edge = ui_nav.locate_edge
    ui_nav.reset_edge_memo_for_tests()
    screen = 'inventory'
    clicks, issued = [], []
    pages = {
        'inventory': ('inventory', 'InventoryUIInstance'),
        'main_hud': ('main_hud', 'HudUIInstance'),
        'system_menu': ('system_menu', 'SystemMenuUIInstance'),
        'craft_hub': ('craft_hub', 'CraftingHubPopupUIInstance'),
        'list': ('fine:facility', 'AlteringFacilityPopupUIInstance'),
        'recipe': ('fine:facility_recipe', 'AlteringPanelPopupUIInstance'),
        'card': ('fine:facility_detail', 'DetailCardUIInstance'),
        'obtain': ('obtain_method', 'HowToGetSelectUIInstance'),
    }
    next_screen = {
        (10, 10): 'main_hud', (20, 20): 'system_menu', (30, 30): 'craft_hub',
        (40, 40): 'list', (1689, 436): 'recipe', (1605, 1065): 'card',
        (746, 896): 'obtain', (1168, 344): 'main_hud',
    }
    generic_points = {
        'inventory_close_main_hud': (10, 10), 'main_hud_menu_system_menu': (20, 20),
        'system_menu_craft_craft_hub': (30, 30), 'craft_hub_card_facility': (40, 40),
    }

    def grab():
        return sap_frames.get(screen)

    def resolve(frame):
        coarse, stack = pages[screen]
        node = refine_inspect_node(frame, sn=coarse) if frame is not None else coarse
        names = ('AlteringPanelPopupUIInstance', stack) if screen in {'card', 'obtain'} else (stack,)
        return ui_logic.UiWhere(node, int(screen != 'main_hud'), 0, 0, 'fixture',
                                stack_names=names, stack_ready=True, stack_sampled_ts=time.time())

    def click(x, y):
        nonlocal screen
        clicks.append((x, y))
        screen = next_screen[(x, y)]
        return True

    def locate(edge, frame, **kwargs):
        if edge.get('sap_control'):
            return original_locate_edge(edge, frame, **kwargs)
        return generic_points[edge['id']]

    def prepare(target, **kwargs):
        kwargs.pop('read_stp')
        return ui_nav.ensure_node(target, resolve=resolve, **kwargs)

    monkeypatch.setattr(ui_act, 'prepare_to', prepare)
    monkeypatch.setattr(ui_logic, 'resolve_navigation', resolve)
    monkeypatch.setattr(ui_nav, 'locate_edge', locate)
    monkeypatch.setattr(ui_nav, 'craft_hub_visible', lambda *a, **k: True)
    for _ in range(2):
        result = execute_obtain_gather(mat='樹液', grab=grab, do_click=click,
            sleep_fn=lambda _: None, defer_completion=True,
            on_action_issued=lambda: issued.append(len(clicks)))
        assert result.status == 'action_pending' and result.action_issued, result
    expected = [(20, 20), (30, 30), (40, 40), (1689, 436), (1605, 1065), (746, 896), (1168, 344)]
    assert clicks == [(10, 10)] + expected * 2
    assert issued == [8, 15]


@pytest.mark.parametrize('node, frame_name, expected', [
    ('obtain_method:煤炭', 'obtain', 'need_verify'),
    (sap_sense.OBTAIN_PAGE, 'card', 'sap_woodcut_missing'),
])
def test_sap_never_dispatches_from_wrong_source(monkeypatch, sap_frames, node, frame_name, expected):
    from mm_assist.life_execute import ui_act, ui_logic
    from mm_assist.life_plan.types import ExecuteResult

    monkeypatch.setattr(ui_act, 'prepare_to', lambda *a, **k: ExecuteResult(True, 'at_node'))
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda frame:
        ui_logic.UiWhere(node, 1, 0, 0, 'fixture', stack_ready=True, stack_sampled_ts=time.time()))
    result = execute_obtain_gather(mat='樹液', grab=lambda: sap_frames[frame_name],
        do_click=lambda *a: pytest.fail('unexpected click'))
    assert result.status == expected


def test_sap_keeps_dispatch_when_interrupted_and_does_not_reopen_for_observation(monkeypatch, sap_frames):
    from mm_assist.life_execute import ui_act, ui_logic
    from mm_assist.life_plan.types import ExecuteResult
    from mm_assist.life_execute import obtain_gather

    nav = []
    monkeypatch.setattr(ui_act, 'prepare_to', lambda target, **kw:
                        nav.append(target) or ExecuteResult(True, 'at_node'))
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda frame:
        ui_logic.UiWhere(sap_sense.OBTAIN_PAGE, 1, 0, 0, 'fixture',
                         stack_ready=True, stack_sampled_ts=time.time()))
    monkeypatch.setattr(obtain_gather, 'wait_gather_idle', lambda **kw:
                        ExecuteResult(False, 'gather_pending', action_issued=kw['issued']))
    sent = []
    result = execute_obtain_gather(mat='樹液', grab=lambda: sap_frames['obtain'],
        do_click=lambda x, y: sent.append((x, y)) or True,
        on_action_issued=lambda: (_ for _ in ()).throw(RuntimeError('interrupted after input')))
    assert result.status == 'error' and result.action_issued
    assert sent == [(1168, 344)]
    result = execute_obtain_gather(mat='樹液', grab=lambda: sap_frames['obtain'],
        do_click=lambda *a: pytest.fail('reissued'), observe_only=True)
    assert result.status == 'gather_pending' and result.action_issued
    assert nav == ['facility_recipe', 'obtain_method']


def test_sap_inventory_gain_while_busy_does_not_finish(monkeypatch, sap_frames):
    from mm_assist.life_execute import ui_act, ui_logic
    from mm_assist.life_plan.types import ExecuteResult

    monkeypatch.setattr(ui_act, 'prepare_to', lambda *a, **k: ExecuteResult(True, 'at_node'))
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda frame:
        ui_logic.UiWhere(sap_sense.OBTAIN_PAGE, 1, 0, 0, 'fixture',
                         stack_ready=True, stack_sampled_ts=time.time()))
    elapsed = [0.0]
    sent = []
    result = execute_obtain_gather(mat='樹液', grab=lambda: sap_frames['obtain'],
        do_click=lambda x, y: sent.append((x, y)) or True,
        read_items=lambda: {'樹液': 33}, read_stp=lambda: 1, wait_s=1,
        read_page=lambda: type('Page', (), {'node': 'main_hud', 'fresh': True})(),
        now_fn=lambda: elapsed[0], sleep_fn=lambda s: elapsed.__setitem__(0, elapsed[0] + s))
    assert result.status == 'gather_pending' and result.action_issued
    assert sent == [(1168, 344)]


@pytest.mark.parametrize('after_node, fresh', [('facility:wood:木材+', True), (sap_sense.RECIPE_PAGE, False)])
def test_sap_unverified_recipe_does_not_open_material(monkeypatch, sap_frames, after_node, fresh):
    from mm_assist.life_execute import ui_logic, ui_nav

    ui_nav.reset_edge_memo_for_tests()
    clicks = []
    previous_stamp = time.time() - 1

    def resolve(frame):
        node = after_node if clicks else 'facility:wood'
        stamp = time.time() if fresh else previous_stamp
        return ui_logic.UiWhere(node, 1, 0, 0, 'fixture', stack_ready=True, stack_sampled_ts=stamp)

    result = ui_nav.ensure_node('facility_recipe', facility='wood', item_name=sap_sense.RECIPE,
        grab=lambda: sap_frames['list'], click=lambda x, y: clicks.append((x, y)) or True,
        resolve=resolve, sleep_fn=lambda _: None)
    assert result.status == 'nav_verify_timeout'
    assert clicks == [(1689, 436)]
    ui_nav.reset_edge_memo_for_tests()
