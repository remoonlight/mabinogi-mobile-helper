"""10/22 top skipped jobs: retained pixels and cross-module recovery, zero input."""
import time
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import insect_obtain as insect, skill_list, ui_act, ui_logic, ui_nav
from mm_assist.life_plan import node_inspect
from mm_assist.memory.readers.ui_stack import UiStackSnap
from mm_assist.perception import item_sense
from mm_assist.safety import autofarm_gates as gates


BAG_NAMES = ('HudUIStepState', 'EquipmentInventoryUIState', 'DetailCardUIState',
    'TutorialTooltipUIState', 'UILayerHideState', 'EquipmentInventoryUIInstance',
    'SizeAdjustmentSender', 'DetailCardUIInstance')


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    insect.reset_insect_visits_for_tests()
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    skill_list._pending.clear()
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda:False)
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **k:[])
    yield
    insect.reset_insect_visits_for_tests()
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    skill_list._pending.clear()


@pytest.mark.parametrize('extra', ['', 'UnknownPopup', 'ClickShieldUIState'])
def test_actual_bag_shell_verifies_icon_and_only_clicks_method(monkeypatch, extra):
    frame = cv2.imread(str(ROOT/'tests/fixtures/life_faults_20261022/firefly_bag_card.png'))
    names = BAG_NAMES + ((extra,) if extra else ())
    reader = SimpleNamespace(read=lambda **_:UiStackSnap(ready=True,
        names=list(names), sampled_ts=time.time()))
    resolve = lambda f:ui_logic.resolve_navigation(f, reader=reader)
    stage = '虫料入口:滴答螢火蟲:滴答螢火蟲'
    insect._visits['滴答螢火蟲'] = 1
    insect._entry_after[stage] = ('card', time.time()-1)
    insect._entry_attempts[stage] = 2
    gates.allow_stage_action(stage); gates.allow_stage_action(stage)
    clicks = []
    monkeypatch.setattr(ui_act, 'prepare_to', lambda *a, **k:pytest.fail('card cannot navigate away'))
    ticks = [time.time()]
    result = insect.execute_insect_obtain(mat='滴答螢火蟲', grab=lambda:frame,
        resolve=resolve, do_click=lambda *xy:clicks.append(xy) or False,
        now_fn=lambda:ticks[0], sleep_fn=lambda s:ticks.__setitem__(0,ticks[0]+s))
    assert not result.action_issued
    if extra:
        assert not clicks and insect._entry_attempts[stage] == 2
    else:
        assert resolve(frame).node == 'item_info:滴答螢火蟲'
        assert clicks == [(744,867)]
        assert stage not in insect._entry_after and gates._stage_counts[stage] == 0


@pytest.mark.parametrize('recovers', [True, False])
def test_herb_scroll_waits_for_fresh_list_without_repeating_drag(monkeypatch, recovers):
    # Same failure boundary as p181: drag sent, first post-drag stack unready.
    before = np.zeros((1440,1920,3),np.uint8)
    after = np.ones((1440,1920,3),np.uint8)
    miss = {'ok':False,'scroll_plan':{'direction':'down'}}
    target = {'ok':True,'click_client':[1275,1335]}
    drags, validations, clock = [], [], [1000.0]
    monkeypatch.setattr(skill_list.time,'time',lambda:clock[0])
    monkeypatch.setattr(item_sense,'locate_skill_item',lambda f, **k:dict(target if drags else miss))
    monkeypatch.setattr(item_sense,'item_list_row_positions',lambda *a, **k:{'藥草':700})
    def validate(f, captured):
        validations.append((len(drags),captured))
        return not drags or (recovers and sum(bool(n) for n,_ in validations)>1)
    _,hit,error = skill_list.seek_item(before,miss,skill_id='herb',name='增幅蘑菇',
        grab=lambda:after if drags else before,
        drag=lambda *a:drags.append(a) or True, validate=validate,
        sleep=lambda s:clock.__setitem__(0,clock[0]+s))
    assert len(drags) == 1
    if recovers:
        assert error is None and hit['ok'] and not skill_list._pending
    else:
        assert error.status == 'item_list_unknown'
        assert ('herb','增幅蘑菇') in skill_list._pending


def test_ambiguous_menu_dispatch_can_be_verified_later_without_third_click(monkeypatch):
    monkeypatch.setattr(ui_nav,'locate_edge',lambda *a, **k:(1620,55))
    def where(node):
        return ui_logic.UiWhere(node, int(node=='system_menu'),0,0,'direct',
            stack_ready=True, stack_sampled_ts=time.time(),
            stack_names=('SideNavigationUIInstance',) if node=='system_menu' else ('TutorialTooltipUIState',))
    def ambiguous(*a):
        raise RuntimeError('possible failure after mouse-up')
    for _ in range(2):
        with pytest.raises(RuntimeError, match='after mouse-up'):
            ui_nav.ensure_node('system_menu', grab=lambda:object(), resolve=lambda _:where('main_hud'),
                click=ambiguous, sleep_fn=lambda _:None)
        ui_nav._EDGE_MEMO.clear()  # short memo expiration does not reset the stage gate
    stage = 'main_hud_menu_system_menu'
    assert gates._stage_counts[stage] == 2
    result = ui_nav.ensure_node('system_menu', grab=lambda:object(), resolve=lambda _:where('system_menu'),
        click=lambda *a:pytest.fail('already arrived'), sleep_fn=lambda _:None)
    assert result.ok and gates._stage_counts[stage] == 0


def test_native_unpainted_rows_wait_for_registered_row_progress(monkeypatch):
    before = cv2.imread(str(ROOT/'tests/fixtures/life_faults_20261020/herb_list_after_two_drags.png'))
    loading = cv2.imread(str(ROOT/'tests/fixtures/life_faults_20261022/herb_post_scroll.png'))
    settled = cv2.imread(str(ROOT/'tests/fixtures/process_stance/herb_middle_failed_20261010.png'))
    assert item_sense.item_list_row_positions(loading,'herb') == {}
    old = item_sense.item_list_row_positions(before,'herb')
    new = item_sense.item_list_row_positions(settled,'herb')
    assert new['新芽蘑菇'] < old['新芽蘑菇'] - 28
    miss = {'ok':False,'scroll_plan':{'direction':'down'}}
    monkeypatch.setattr(item_sense,'locate_skill_item',lambda *a, **k:dict(miss))
    drags, samples = [], []
    def grab():
        if not drags:
            return before
        samples.append(1)
        return loading if len(samples)<=2 else settled
    def drag(*a):
        drags.append(a)
        return len(drags)==1  # only first gesture sent; refuse the next verified segment
    _,_,error = skill_list.seek_item(before,miss,skill_id='herb',name='增幅蘑菇',
        grab=grab,drag=drag,sleep=lambda _:None,validate=lambda *a:True)
    assert error.status == 'drag_rejected' and len(drags)==2
    assert not skill_list._pending and gates._stage_counts['详情物下翻:herb:增幅蘑菇']==0


@pytest.mark.parametrize('currency', [False, True])
def test_recorded_currency_and_underlying_repair_never_authorize_hud_menu(monkeypatch, currency):
    from mm_assist.memory.readers import ui_stack
    image = cv2.imread(str(ROOT/'tests/fixtures/life_faults_20261022/currency_over_repair.png'))
    names = ['TutorialTooltipUIState','EquipmentRepairUIInstance','SizeAdjustmentSender']
    if currency:
        names.append('CurrencyUIInstance')
    expected = 'currency_wallet' if currency else 'overlay'
    assert ui_stack.stack_left_overlay(names)
    assert ui_stack.stack_node(names,ready=True) == expected
    reader = SimpleNamespace(read=lambda **_:UiStackSnap(ready=True,
        names=names, sampled_ts=time.time()))
    # Even a menu-like pixel must never override these known active instances.
    monkeypatch.setattr(ui_logic,'_main_hud_confirmed',lambda *a, **k:True)
    monkeypatch.setattr(node_inspect,'refine_inspect_node',lambda *a, **k:'main_hud')
    assert ui_logic.resolve_navigation(image,reader=reader).node == expected


@pytest.mark.parametrize('target', ['craft_hub','facility:metal'])
def test_currency_page_never_uses_underlying_craft_controls(monkeypatch, target):
    # The old currency->craft_hub path alias must not send input to a covered card.
    where = ui_logic.UiWhere('currency_wallet',1,0,0,'direct',stack_ready=True,
        stack_names=('CurrencyUIInstance',),stack_sampled_ts=time.time())
    monkeypatch.setattr(ui_nav,'locate_edge',lambda *a, **k:(416,404))
    result = ui_nav.ensure_node(target,grab=lambda:object(),resolve=lambda _:where,
        click=lambda *a:pytest.fail('currency covers crafting controls'),sleep_fn=lambda _:None)
    assert not result.ok and result.status=='wrong_page'


@pytest.mark.parametrize('interruption', ['pause','new_start','exception'])
def test_slow_list_read_never_accepts_late_result_or_clears_drag(monkeypatch, interruption):
    frame = np.zeros((64,64,3),np.uint8)
    miss = {'ok':False,'scroll_plan':{'direction':'down'}}
    key = 'herb','增幅蘑菇'
    skill_list._pending[key] = (time.time()-1,frame.shape[:2],{'藥草':50})
    stopped = [False]
    def validate(*a):
        if interruption == 'exception':
            raise RuntimeError('stack read failed')
        stopped[0] = True  # pause/new generation arrived while resolver was reading
        return True
    kwargs = dict(skill_id=key[0],name=key[1],grab=lambda:frame,validate=validate,
        cancelled=lambda:stopped[0],sleep=lambda _:None,
        drag=lambda *a:pytest.fail('pending drag cannot be repeated'))
    if interruption == 'exception':
        with pytest.raises(RuntimeError,match='stack read failed'):
            skill_list.seek_item(frame,miss,**kwargs)
    else:
        _,_,error = skill_list.seek_item(frame,miss,**kwargs)
        assert error.status == 'cancel'
    assert key in skill_list._pending


@pytest.mark.parametrize('case', ['closed','unready','old_frame','old_stack','future','pre_capture',
    'blocked','same_list','wrong_node','list_instance_remains'])
def test_recovery_only_retires_scroll_after_independent_list_close(monkeypatch, case):
    from mm_assist.life_execute import ui_recover
    frame = np.zeros((64,64,3),np.uint8)
    key = 'herb','增幅蘑菇'
    sent = time.time()-1
    skill_list._pending[key] = (sent,frame.shape[:2],{'藥草':50})
    stage = '详情物下翻:herb:增幅蘑菇'
    gates.allow_stage_action(stage)
    names = ('NewProfileUIInstance','TutorialTooltipUIState')
    if case in {'same_list','list_instance_remains'}:
        names += ('LivingSkillUIInstance',)
    if case == 'blocked':
        names += ('ClickShieldUIState',)
    where = ui_logic.UiWhere('gather_skill:herb' if case=='same_list' else
        ('unknown' if case=='wrong_node' else 'life_grid'),0,0,0,'direct',
        stack_names=names,stack_ready=case!='unready',
        stack_sampled_ts=sent-1 if case=='old_stack' else time.time()+(100 if case=='future' else 0))
    if case == 'old_frame':
        ui_nav._verify_late_edges(where,frame_started_at=sent-1)
    else:
        # Read-only half of recovery: prior Esc already sent, no new Esc allowed.
        ui_recover.recover_once('item_scroll_unverified',grab=lambda:frame,
            resolve=lambda _:where if case in {'old_stack','future','pre_capture'} else
                replace(where,stack_sampled_ts=time.time()),after_input_at=time.time()-3,
            press_esc=lambda:pytest.fail('recovery must not send again'),sleep_fn=lambda _:None)
    assert (key not in skill_list._pending) == (case=='closed')
    assert gates._stage_counts[stage] == (0 if case=='closed' else 1)
