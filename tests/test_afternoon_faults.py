"""10点后高频故障：原图／真实调用链先复现，输入全部替换。"""
from types import SimpleNamespace

import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import insect_obtain as insect, ui_act, ui_logic
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem
from mm_assist.safety import autofarm_gates as gates


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    insect.reset_insect_visits_for_tests()
    gates.reset_stage_gates_for_tests()
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    yield
    insect.reset_insect_visits_for_tests()
    gates.reset_stage_gates_for_tests()


ITEM_NAMES = ('HudUIStepState', 'CraftingHubUIState', 'TutorialTooltipUIState',
              'CraftingHubPopupUIInstance', 'CraftingFacilityPopupUIInstance')


def test_real_items_pixels_with_observed_stack_verify_two_prior_inputs(monkeypatch):
    from mm_assist.life_plan import node_inspect
    frame = cv2.imread(str(ROOT/'tests/fixtures/insect_obtain/items_failed_20261014.png'))
    assert frame is not None
    stage = '虫料入口:滴答螢火蟲:道具'
    insect._entry_attempts[stage] = 2
    insect._entry_after[stage] = ('items', 999)
    gates._stage_counts[stage] = 2
    gates._stage_need_verify[stage] = True
    monkeypatch.setattr(ui_act, 'prepare_to', lambda *a, **k: ExecuteResult(False, 'unexpected_navigation'))
    result = insect.execute_insect_obtain(mat='滴答螢火蟲', entry_only=True,
        grab=lambda: frame, do_click=lambda *a: pytest.fail('已到道具页不能第三次点'),
        resolve=lambda _: SimpleNamespace(node='craft_shell', stack_ready=True,
            stack_names=ITEM_NAMES, stack_sampled_ts=1000.01), now_fn=lambda:1000)
    assert result.ok and result.status == 'page_matched'
    assert stage not in insect._entry_after and stage not in insect._entry_attempts
    assert gates._stage_counts[stage] == 0
    monkeypatch.setattr(node_inspect.time, 'time', lambda:1000)
    view = SimpleNamespace(ui_stack_names=ITEM_NAMES, ui_stack_ready=True, ui_stack_ts=1000,
                           ui_stack_error='', fresh=True, pid_alive=True)
    assert node_inspect._craft_items_from_view(frame, view)


def test_original_list_needs_more_than_two_verified_viewports(monkeypatch):
    from mm_assist.perception import item_sense
    from mm_assist.life_execute import skill_nearby as sn
    paths = ['tests/fixtures/process_stance/herb_list_failed_20261008.png',
             'tests/fixtures/life_faults_20261020/herb_list_after_two_drags.png',
             'tests/fixtures/process_stance/herb_middle_failed_20261010.png',
             'tests/fixtures/process_stance/herb_list_updated_20261006.png']
    frames = [cv2.imread(str(ROOT/p)) for p in paths]
    assert all(f is not None for f in frames)
    # The final archived viewport is 1912x1442. Replay a fixed 1920x1440
    # client by dropping its last two rows and padding eight right-edge columns;
    # preserve all target/anchor pixels and coordinates, without changing files.
    frames[-1] = cv2.copyMakeBorder(frames[-1][:1440],0,0,0,8,cv2.BORDER_CONSTANT)
    original = item_sense.locate_skill_item
    hits = {id(f): original(f, skill_id='herb', name='淨淨蘑菇') for f in frames}
    assert all(not hits[id(f)]['ok'] and hits[id(f)]['scroll_plan']['direction']=='down' for f in frames[:3])
    assert hits[id(frames[-1])]['ok']
    monkeypatch.setattr(item_sense, 'locate_skill_item', lambda f, **k: dict(hits[id(f)]))
    monkeypatch.setattr(ui_logic, 'read_inspect_record', lambda **k:
                        {'node':'gather_skill:herb','zh':'生活技能'})
    monkeypatch.setattr(sn.time, 'sleep', lambda _: None)
    index, drags, clicks = [0], [], []
    def drag(*points, **kw):
        drags.append(points)
        index[0] += 1
        assert index[0] <= 3
        return True
    result = sn.execute_skill_nearby(skill_id='herb', item_name='淨淨蘑菇', output='淨淨蘑菇',
        grab=lambda:frames[index[0]], do_click=lambda *xy: clicks.append(xy) or False,
        do_drag=drag, defer_completion=True)
    assert len(drags) == 3
    assert clicks == [tuple(hits[id(frames[-1])]['click_client'])]
    assert result.status == 'click_yield' and not result.action_issued


@pytest.mark.parametrize('tracking', [False, True])
def test_real_native_shield_refusal_waits_without_skipping_or_esc(monkeypatch, tracking):
    from mm_assist.life_plan import controller, bar_ui
    from mm_assist.safety import input as inputs, lease
    from mm_assist.memory.readers import ui_stack
    controller.reset_controller_for_tests()
    plan = PlanItem('p155','repair','测试返回路径')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k:[plan])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000, pause_on_failure=tracking)
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda _:True)
    monkeypatch.setattr(bar_ui, '_note_fail_and_continue', lambda **k:pytest.fail('遮罩出现前未按下，只读等待'))
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', lambda *a, **k:pytest.fail('不能Esc'))
    monkeypatch.setattr(inputs, 'check_abort', lambda:None)
    monkeypatch.setattr('mm_assist.jobs.bar.click_owner_block_reason', lambda:'')
    monkeypatch.setattr(lease, 'require_action', lambda *a, **k:None)
    blocked = [True]
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda:SimpleNamespace(read=lambda **k:
        ui_stack.UiStackSnap(ready=True,names=['AlteringFacilityPopupUIInstance']+
                            (['ClickShieldUIState'] if blocked[0] else []))))
    def execute(*a, **k):
        def click(x,y):
            inputs._guard_before_input('click', SimpleNamespace(hwnd=11), method='sendinput')
            return True
        try:
            assert gates.stage_click('返回', click, 100,100)
        except RuntimeError as exc:
            return ExecuteResult(False,'error',str(exc))
        return ExecuteResult(True,'done')
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    bar_ui.pump_life_once(None,None)
    st = controller.get_controller()
    assert st.running and not st.paused and st.current == plan
    assert st.pending_action is None and gates._stage_counts['返回'] == 0
    assert '弹层' in st.current_action
    blocked[0] = False
    bar_ui.pump_life_once(None,None)
    assert not controller.get_controller().queue
    assert gates._stage_counts['返回'] == 1


@pytest.mark.parametrize('extra', ['ClickShieldUIState','UnknownPopup','CraftingPanelPopupUIInstance'])
def test_items_pixels_cannot_authorize_covered_list_or_clear_pending(monkeypatch, extra):
    frame = cv2.imread(str(ROOT/'tests/fixtures/insect_obtain/items_failed_20261014.png'))
    stage = '虫料入口:滴答螢火蟲:道具'
    insect._entry_attempts[stage] = 2
    insect._entry_after[stage] = 'items', 999
    monkeypatch.setattr(ui_act,'prepare_to',lambda *a,**k:ExecuteResult(False,'blocked'))
    result = insect.execute_insect_obtain(mat='滴答螢火蟲',entry_only=True,
        grab=lambda:frame,do_click=lambda *a:pytest.fail('covered'),
        resolve=lambda _:SimpleNamespace(node='craft_shell',stack_ready=True,
            stack_names=ITEM_NAMES+(extra,),stack_sampled_ts=1000.01),now_fn=lambda:1000)
    assert not result.ok and insect._entry_attempts[stage] == 2


@pytest.mark.parametrize('message,wait', [
    ('click_ui_blocked: ClickShieldUIState',True),
    ("{'ok': False, 'error': 'click_ui_blocked: ClickShieldUIState', 'waiting': False}",True),
    ('click_ui_blocked: ClickShieldUIState,ToyDimLayerUIState',True),
    ('click_ui_blocked: 前往确认画面或落点已改变',False),
    ('click_ui_blocked: UnknownPopup',False), ('click_focus_lost: game lost focus',False),
])
def test_only_exact_known_preinput_layer_refusal_is_readiness(message,wait):
    from mm_assist.life_plan.bar_ui import _known_input_layer_refusal
    assert _known_input_layer_refusal(message) == wait


@pytest.mark.parametrize('interruption', ['unchanged','pause','new_start','exception','wrong_page','old_clock','resized'])
def test_scroll_never_reissues_an_unverified_gesture(monkeypatch,interruption):
    import numpy as np
    from mm_assist.life_execute import skill_list
    from mm_assist.perception import item_sense
    frame = np.zeros((64,64,3),np.uint8)
    resized = np.zeros((65,64,3),np.uint8)
    miss = {'ok':False,'scroll_plan':{'direction':'down','needs_scroll':8}}
    clock, stopped, drags = [1000.0],[False],[]
    monkeypatch.setattr(skill_list.time,'time',lambda:clock[0])
    monkeypatch.setattr(item_sense,'locate_skill_item',lambda *a,**k:dict(miss))
    monkeypatch.setattr(item_sense,'item_list_row_positions',lambda *a,**k:{'藥草':50})
    def drag(*xy):
        drags.append(xy)
        if interruption=='exception': raise RuntimeError('possible after mouse-down')
        if interruption in {'pause','new_start'}: stopped[0] = True
        return True
    def sleep(sec): clock[0] += -1 if interruption=='old_clock' else sec
    validations = []
    def validate(*a):
        validations.append(1)
        # Initial and pre-drag checks pass; the *post-drag* page changes.
        return interruption!='wrong_page' or len(validations)<=2
    args = dict(skill_id='herb',name='淨淨蘑菇',
        grab=lambda:resized if interruption=='resized' and drags else frame,
        drag=drag,validate=validate,cancelled=lambda:stopped[0],sleep=sleep)
    if interruption=='exception':
        with pytest.raises(RuntimeError):skill_list.seek_item(frame,miss,**args)
    else:
        _,_,error=skill_list.seek_item(frame,miss,**args)
        assert error is not None
    assert len(drags)==1 and ('herb','淨淨蘑菇') in skill_list._pending
    # A new call/shared gate reset cannot erase the issued-input fact.
    stopped[0]=False
    gates.reset_stage_gates_for_tests()
    _,_,error=skill_list.seek_item(frame,miss,**args)
    assert error is not None and len(drags)==1


def test_target_motion_must_settle_in_two_distinct_frames(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import skill_list
    from mm_assist.perception import item_sense
    frames=[np.zeros((64,64,3),np.uint8) for _ in range(3)]
    coords=iter([{'ok':True,'click_client':[30,40]},
                 {'ok':True,'click_client':[30,50]},
                 {'ok':True,'click_client':[30,50]}])
    monkeypatch.setattr(item_sense,'locate_skill_item',lambda *a,**k:next(coords))
    captures=iter(frames)
    f,hit,error=skill_list.seek_item(frames[0],{'ok':True,'click_client':[30,20]},
        skill_id='herb',name='淨淨蘑菇',grab=lambda:next(captures),sleep=lambda _:None)
    assert error is None and hit['click_client']==[30,50] and f is frames[-1]


def test_initial_wrong_list_cannot_authorize_first_drag(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import skill_list
    f=np.zeros((64,64,3),np.uint8)
    miss={'ok':False,'scroll_plan':{'direction':'down'}}
    captures=[]
    # A wrong initial sample may be reobserved without input. Persistent wrong
    # pages still cannot authorize the first drag or clear a pending gesture.
    _,_,error=skill_list.seek_item(f,miss,skill_id='herb',name='淨淨蘑菇',
        captured_at=1000,grab=lambda:captures.append(1) or f,
        drag=lambda *a:pytest.fail('wrong initial page'),validate=lambda *a:False,sleep=lambda _:None)
    assert error.status=='item_list_unknown' and not skill_list._pending
    assert len(captures)==30


def test_no_drag_until_current_viewport_rows_stop_moving(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import skill_list
    from mm_assist.perception import item_sense
    frames=[np.zeros((64,64,3),np.uint8) for _ in range(4)]
    ys={id(f):y for f,y in zip(frames,[50,40,30,30])}
    miss={'ok':False,'scroll_plan':{'direction':'down'}}
    captures=iter(frames[1:]);seen=[]
    def grab():
        f=next(captures);seen.append(ys[id(f)]);return f
    monkeypatch.setattr(item_sense,'locate_skill_item',lambda *a,**k:dict(miss))
    monkeypatch.setattr(item_sense,'item_list_row_positions',lambda f,*a,**k:{'藥草':ys[id(f)]})
    drags=[]
    def drag(*xy):
        assert seen==[40,30,30]
        drags.append(xy);return False
    _,_,error=skill_list.seek_item(frames[0],miss,skill_id='herb',name='淨淨蘑菇',
        grab=grab,drag=drag,sleep=lambda _:None)
    assert error.status=='drag_rejected' and len(drags)==1 and not skill_list._pending


@pytest.mark.parametrize('interrupt', ['pause','new_start'])
def test_late_target_after_pause_or_new_run_keeps_issued_scroll_fact(monkeypatch, interrupt):
    import numpy as np
    from mm_assist.life_execute import skill_list
    from mm_assist.life_plan import controller
    from mm_assist.perception import item_sense
    controller.reset_controller_for_tests()
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:[PlanItem('p1','repair','test')])
    owner=controller.start_life(snapshot=LifeSnapshot(ok=True),now=1000)
    f=np.zeros((64,64,3),np.uint8)
    miss={'ok':False,'scroll_plan':{'direction':'down'}}
    drags=[]
    def locate(*a,**k):
        if drags:
            if interrupt=='pause': controller.pause_life()
            else: controller.start_life(snapshot=LifeSnapshot(ok=True),now=2000)
            return {'ok':True,'click_client':[30,40]}
        return dict(miss)
    monkeypatch.setattr(item_sense,'locate_skill_item',locate)
    monkeypatch.setattr(item_sense,'item_list_row_positions',lambda *a,**k:{'藥草':50})
    def cancelled():
        current=controller.get_controller()
        return not current.running or current.life_started_at!=owner.life_started_at
    _,_,error=skill_list.seek_item(f,miss,skill_id='herb',name='淨淨蘑菇',
        grab=lambda:f,drag=lambda *a:drags.append(a) or True,
        cancelled=cancelled,sleep=lambda _:None)
    assert error.status=='cancel' and len(drags)==1
    assert ('herb','淨淨蘑菇') in skill_list._pending
