"""Observed successful intermediate exits must clear their own input gate."""
from types import SimpleNamespace

import cv2
import pytest
import time

from mm_assist.config import ROOT
from mm_assist.life_execute import ui_nav, ui_act
from mm_assist.life_execute.ui_logic import UiWhere
from mm_assist.safety import autofarm_gates as gates

_real_result_presence = ui_act.stack_has_facility_result
_real_common_presence = ui_act.stack_has_common_popup


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **k: False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda *a, **k: False)
    yield
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()


def test_original_cloth_failure_locates_real_close_control():
    frame = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/cloth_standalone.png'))
    assert frame is not None
    assert ui_nav.locate_edge({'id': 'facility_back_craft_hub'}, frame) == (1857, 60)


@pytest.mark.parametrize('source,arrived', [('facility:cloth', 'main_hud'),
                                          ('facility_recipe:cloth:x', 'facility:cloth')])
def test_verified_intermediate_exit_does_not_exhaust_gate(monkeypatch, source, arrived):
    # The requested outer destination is still not reached. Only the observed
    # intermediate transition may reset the exit gate, not the later failure.
    for _ in range(3):
        ui_nav.reset_edge_memo_for_tests()
        clicks = []
        monkeypatch.setattr(ui_nav, 'locate_edge', lambda edge, *a, **k:
                            (1857, 60) if not clicks and edge['id'] == 'facility_back_craft_hub' else None)
        ui_nav.ensure_node('craft_hub', grab=lambda: object(),
            resolve=lambda _: UiWhere(arrived if clicks else source, 0, 0, None, 'fixture'),
            click=lambda x, y: clicks.append((x, y)) or True, sleep_fn=lambda _: None)
        assert clicks == [(1857, 60)]
        assert gates._stage_counts['facility_back_craft_hub'] == 0


def test_unchanged_facility_keeps_gate_closed_after_two_actions(monkeypatch):
    monkeypatch.setattr(ui_nav, 'locate_edge', lambda *a, **k: (1857, 60))
    for attempt in range(3):
        ui_nav.reset_edge_memo_for_tests()
        clicks = []
        result = ui_nav.ensure_node('main_hud', grab=lambda: object(),
            resolve=lambda _: UiWhere('facility:cloth', 0, 0, None, 'fixture'),
            click=lambda x, y: clicks.append((x, y)) or True, sleep_fn=lambda _: None)
        assert len(clicks) == (1 if attempt < 2 else 0)
    assert result.status == 'need_verify'


def test_coarse_medicine_name_with_recipe_stack_verifies_two_distinct_exits(monkeypatch):
    frame = cv2.imread(str(ROOT/'tests/fixtures/process_stance/medicine_standalone.png'))
    clicks = []
    def where(_):
        if len(clicks) >= 2:
            return UiWhere('main_hud',0,0,0,'fixture',stack_ready=True)
        names = ['AlteringFacilityPopupUIInstance']
        if not clicks:
            names.append('AlteringPanelPopupUIInstance')
        return UiWhere('facility:medicine',0,0,0,'fixture',stack_names=tuple(names),stack_ready=True)
    result = ui_nav.ensure_node('main_hud', grab=lambda:frame, resolve=where,
        click=lambda x,y:clicks.append((x,y)) or True, sleep_fn=lambda _:None)
    assert result.ok and clicks == [(1857,60),(1857,60)]
    assert gates._stage_counts['facility_back_craft_hub'] == 0


@pytest.mark.parametrize('via',['nav','spot'])
@pytest.mark.parametrize('failure',['click_pointer_moved','click_focus_lost',
    'focus_click_timeout','click_ui_blocked','click_ui_not_ready','yield'])
def test_rejected_input_does_not_exhaust_close_gate(monkeypatch,via,failure):
    frame = cv2.imread(str(ROOT/'tests/fixtures/process_stance/hub_with_toy_layer.png'))
    delivered = []
    def rejected(x,y):
        if failure == 'yield':return False
        raise RuntimeError(f'{failure}: refused before mouse down')
    def invoke(click):
        if via == 'spot':
            return ui_act.click_spot(click,(1854,56),'点X',stage='清界面',wait_refresh=False)
        return ui_nav.ensure_node('main_hud',grab=lambda:frame,click=click,
            resolve=lambda _:UiWhere('main_hud' if delivered else 'craft_hub',0,0,None,'fixture'),
            sleep_fn=lambda _:None)
    for _ in range(3):
        if failure == 'yield':
            assert invoke(rejected).status == 'click_yield'
        else:
            with pytest.raises(RuntimeError,match=failure):invoke(rejected)
    result = invoke(lambda x,y:delivered.append((x,y)) or True)
    assert len(delivered) == 1
    if via == 'spot':
        assert result is None
    else:
        assert result.ok


def test_unknown_dispatch_error_keeps_attempt_budget():
    def uncertain(x,y):raise RuntimeError('unknown error after possible dispatch')
    for _ in range(2):
        with pytest.raises(RuntimeError,match='unknown error'):
            ui_act.click_spot(uncertain,(10,10),'点X',stage='清界面',wait_refresh=False)
    clicks=[]
    result=ui_act.click_spot(lambda x,y:clicks.append((x,y)) or True,(10,10),
        '点X',stage='清界面',wait_refresh=False)
    assert result.status == 'need_verify' and clicks == []


@pytest.mark.parametrize('layer', ['IngredientGuideUIInstance', 'DetailCardUIInstance'])
@pytest.mark.parametrize('removed', [True, False])
def test_detail_exit_checks_actual_underlying_facility(monkeypatch, layer, removed):
    # The same detail popup can be opened from a skill or a processing recipe.
    # Closing it must verify its disappearance, not invent a skill page.
    frame = cv2.imread(str(ROOT/'tests/fixtures/process_stance/wood_away_anim_first.png'))
    monkeypatch.setattr(ui_nav, 'locate_edge', lambda edge,*a,**k:
                        (1857,60) if edge['from'] in {'item_info','gather_detail','obtain_method'} else None)
    for _ in range(3):
        ui_nav.reset_edge_memo_for_tests()
        clicks = []
        def where(_):
            names = ['AlteringFacilityPopupUIInstance','AlteringPanelPopupUIInstance']
            if not clicks or not removed:
                names.append(layer)
                node = 'item_info:樹液'
            else:
                node = 'facility:wood:高級木材'
            return UiWhere(node,1,0,0,'fixture',stack_names=tuple(names),stack_ready=True)
        result = ui_nav.ensure_node('life_grid', grab=lambda:frame, resolve=where,
            click=lambda x,y:clicks.append((x,y)) or True, sleep_fn=lambda _:None)
        if removed:
            assert clicks == [(1857,60)]
            assert gates._stage_counts['清界面'] == 0
        elif _ == 2:
            assert clicks == [] and result.status == 'need_verify'
        else:
            assert clicks == [(1857,60)]


def test_nested_ingredient_and_item_layers_close_to_recipe(monkeypatch):
    frame = cv2.imread(str(ROOT/'tests/fixtures/process_stance/wood_away_anim_first.png'))
    monkeypatch.setattr(ui_nav, 'locate_edge', lambda edge,*a,**k:
                        (1857,60) if edge['from'] == 'item_info' else None)
    clicks = []
    def where(_):
        names = ['AlteringFacilityPopupUIInstance','AlteringPanelPopupUIInstance']
        if len(clicks) < 2:
            names.append('DetailCardUIInstance')
        if not clicks:
            names.append('IngredientGuideUIInstance')
        return UiWhere('item_info:樹液' if len(clicks)<2 else 'facility:wood:高級木材',
            1,0,0,'fixture',stack_names=tuple(names),stack_ready=True)
    ui_nav.ensure_node('life_grid', grab=lambda:frame, resolve=where,
        click=lambda x,y:clicks.append((x,y)) or True, sleep_fn=lambda _:None)
    assert clicks == [(1857,60),(1857,60)]
    assert gates._stage_counts['清界面'] == 0


def test_detail_disappearance_without_ready_stack_does_not_clear_gate(monkeypatch):
    frame = cv2.imread(str(ROOT/'tests/fixtures/process_stance/wood_away_anim_first.png'))
    clicks = []
    monkeypatch.setattr(ui_nav, 'locate_edge', lambda *a,**k:(1857,60))
    def where(_):
        return UiWhere('facility:wood:高級木材' if clicks else 'item_info:樹液',1,0,0,'fixture',
            stack_names=('AlteringPanelPopupUIInstance',) if clicks else ('DetailCardUIInstance',),
            stack_ready=not bool(clicks))
    result = ui_nav.ensure_node('life_grid',grab=lambda:frame,resolve=where,
        click=lambda x,y:clicks.append((x,y)) or True,sleep_fn=lambda _:None)
    assert not result.ok and clicks == [(1857,60)]
    assert gates._stage_counts['清界面'] == 1


def _verify_three_popup_cycles(monkeypatch, *, evidence, stage, layer, run):
    """共用验栈边界的三轮场景，保留已发次数及其它步骤不被清零的断言。"""
    calls = []
    current = {'clicked': False}

    def view(**_):
        removed = current['clicked'] and evidence != 'unchanged'
        names = ['AlteringFacilityPopupUIInstance'] + ([] if removed else [layer])
        if current['clicked'] and evidence == 'missing':
            return None
        raw = {'ui_stack_names': names, 'ui_stack_ready': evidence != 'not_ready',
            'ui_stack_ts': time.time() - 10 if evidence == 'before_click' else time.time(),
            'ui_stack': '加工设施' if removed else ('设施结果 > 加工设施' if layer == 'FacilityResultUIInstance' else '确认弹窗 > 加工设施')}
        if evidence == 'no_stack_stamp':
            raw.pop('ui_stack_ts')
        if current['clicked'] and evidence == 'no_names':
            raw.pop('ui_stack_names')
        if current['clicked'] and evidence == 'empty_unconfirmed':
            raw['ui_stack_names'] = []
        return SimpleNamespace(raw=raw, ui_stack=raw['ui_stack'],
            fresh=evidence != 'stale', pid_alive=evidence != 'dead')

    monkeypatch.setattr(ui_act, 'read_state_ui', view)
    monkeypatch.setattr('mm_assist.perception.state_ui.read_state_ui', view)
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.wait_ui_params_refresh', lambda **_: True)
    assert gates.allow_stage_action('unrelated')

    def click(x, y):
        calls.append((x, y))
        current['clicked'] = True
        return True

    for cycle in range(3):
        current['clicked'] = False
        result = run(click)
        if evidence == 'valid':
            assert result is None
            assert len(calls) == cycle + 1
            assert gates._stage_counts[stage] == 0
        else:
            assert result is not None and not result.ok
            assert len(calls) == min(cycle + 1, 2)
            assert gates._stage_counts[stage] == min(cycle + 1, 2)
            if cycle == 2:
                assert result.status == 'need_verify'
        assert gates._stage_counts['unrelated'] == 1


@pytest.mark.parametrize('evidence', ['valid', 'unchanged', 'not_ready', 'stale',
    'dead', 'before_click', 'no_stack_stamp', 'missing', 'no_names', 'empty_unconfirmed'])
def test_shared_popup_verification_requires_fresh_post_dispatch_stack(monkeypatch, evidence):
    """十种证据边界只在真实共用检查器完整展开一次。"""
    def run(click):
        return ui_act._click_popup_and_verify(click, (956, 1356), '確認',
            stage='設施結果確認', popup_names=frozenset({'FacilityResultUIInstance'}),
            miss_status='confirm_miss', miss_message='设施结果未关闭')
    _verify_three_popup_cycles(monkeypatch, evidence=evidence,
        stage='設施結果確認', layer='FacilityResultUIInstance', run=run)


@pytest.mark.parametrize('operation,stage,layer', [
    ('dismiss_facility_result', '設施結果確認', 'FacilityResultUIInstance'),
    ('confirm_go_facility_popup', '前往设施确认', 'ButtonCommonPopupInstance'),
])
@pytest.mark.parametrize('evidence', ['valid', 'unchanged'])
def test_popup_entry_wires_locator_layer_and_stage(monkeypatch, operation, stage, layer, evidence):
    """三个业务入口各验成功及未退层，真实原图与独立门闩保留。"""
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', _real_result_presence)
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', _real_common_presence)
    frame = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/medicine_result_confirm.png'))
    assert frame is not None
    for locator, xy in [('locate_common_popup_confirm', (111, 222))]:
        monkeypatch.setattr('mm_assist.perception.process_sense.' + locator,
            lambda *a, point=xy, **k: {'ok': True, 'click_client': list(point)})
    delegated = []
    verify = ui_act._click_popup_and_verify
    def checked_verify(click, xy, label, **kwargs):
        delegated.append(kwargs)
        assert kwargs['stage'] == stage
        assert layer in kwargs['popup_names']
        assert 'AlteringFacilityPopupUIInstance' not in kwargs['popup_names']
        if operation != 'dismiss_facility_result':
            assert xy == (111, 222)
        return verify(click, xy, label, **kwargs)
    monkeypatch.setattr(ui_act, '_click_popup_and_verify', checked_verify)
    def run(click):
        return getattr(ui_act, operation)(click, lambda: frame)
    _verify_three_popup_cycles(monkeypatch, evidence=evidence, stage=stage, layer=layer, run=run)
    assert len(delegated) == 3
