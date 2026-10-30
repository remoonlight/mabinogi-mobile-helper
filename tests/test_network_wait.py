"""Real network toast must suspend the current call, without replay or new input."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_plan import controller, execution_records
from mm_assist.life_plan.types import LifeSnapshot, PlanItem


@pytest.fixture
def waiting(monkeypatch, tmp_path):
    m = importlib.import_module('mm_assist.life_plan.game_stop')
    controller.reset_controller_for_tests()
    items = [PlanItem('p1','feed','投料 鋼錠'), PlanItem('p2','gather','采集 樹液')]
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:list(items))
    controller.start_life(snapshot=LifeSnapshot(ok=True),now=1000,pause_on_failure=True)
    monkeypatch.setattr(execution_records,'_RECORDER', execution_records.ExecutionRecorder(tmp_path/'records.json'))
    monkeypatch.setattr(m.GameStopWatch,'_evidence',lambda *a:None)
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons',lambda:None)
    raw = ROOT/'tests/fixtures/game_stop/network_unstable_20261024.png'
    toast = cv2.imdecode(np.fromfile(raw,np.uint8),1)
    clear = toast.copy()
    clear[112:215,790:1140] = 30
    clock, scans = [0.], []
    monkeypatch.setattr(m.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(m.time,'sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    def capture(*_):
        scans.append(clock[0])
        return toast if clock[0] < 4 else clear
    monkeypatch.setattr(m,'_capture',capture)
    yield SimpleNamespace(m=m,toast=toast,clear=clear,clock=clock,scans=scans,items=items)
    controller.reset_controller_for_tests()


@pytest.mark.parametrize('tracking',[False,True])
def test_real_toast_waits_two_seconds_between_checks_then_continues_same_call(waiting, tracking):
    w = waiting
    controller._STATE.pause_on_failure = tracking
    calls = []
    @w.m.guard_life_pump
    def work(*_):
        calls.append((w.clock[0],controller.get_controller().current.id))
    work(None,SimpleNamespace(hwnd=123))
    assert calls == [(4.,'p1')]
    assert w.scans == pytest.approx([0.,2.,4.])
    state = controller.get_controller()
    assert state.running and not state.paused and state.queue == w.items


@pytest.mark.parametrize('event',['pause','new_start','f8'])
def test_wait_cannot_undo_user_stop_or_write_into_new_generation(waiting,monkeypatch,event):
    w = waiting
    def sleep(dt):
        w.clock[0] += dt
        if w.clock[0] >= 1:
            if event == 'pause':
                controller.pause_life()
            elif event == 'new_start':
                controller.start_life(snapshot=LifeSnapshot(ok=True),now=2000)
            else:
                from mm_assist.safety import lease
                lease._cancel_event.set()
    monkeypatch.setattr(w.m.time,'sleep',sleep)
    calls = []
    @w.m.guard_life_pump
    def work(*_): calls.append('input')
    work(None,SimpleNamespace(hwnd=123))
    assert calls == [] and w.scans == [0.]
    state = controller.get_controller()
    assert (state.paused if event == 'pause' else state.life_started_at == (2000 if event == 'new_start' else 1000))


def test_warning_after_major_action_resumes_verification_without_replaying_it(waiting,monkeypatch):
    w = waiting
    captures = []
    def capture(*_):
        captures.append(w.clock[0])
        return w.clear if len(captures) == 1 or w.clock[0] >= 4 else w.toast
    monkeypatch.setattr(w.m,'_capture',capture)
    calls = []
    @w.m.guard_life_pump
    def work(*_):
        calls.append('major')
        assert not w.m.game_stop_interrupted(force=True)
        calls.append(('verify',w.clock[0]))
    work(None,SimpleNamespace(hwnd=123))
    assert calls == ['major',('verify',4.)]
    assert captures == pytest.approx([0.,0.,2.,4.])


def test_black_poll_does_not_count_as_toast_disappearing(waiting,monkeypatch):
    w = waiting
    def capture(*_):
        w.scans.append(w.clock[0])
        if w.clock[0] < 2: return w.toast
        return np.zeros_like(w.toast) if w.clock[0] < 4 else w.clear
    monkeypatch.setattr(w.m,'_capture',capture)
    calls = []
    @w.m.guard_life_pump
    def work(*_): calls.append(w.clock[0])
    work(None,SimpleNamespace(hwnd=123))
    assert calls == [4.] and w.scans == pytest.approx([0.,2.,4.])


def test_actual_disconnect_during_wait_still_pauses(waiting,monkeypatch):
    w = waiting
    fatal = cv2.imdecode(np.fromfile(ROOT/'tests/fixtures/game_stop/both_1920.png',np.uint8),1)
    monkeypatch.setattr(w.m,'_capture',lambda *_:w.toast if w.clock[0] < 2 else fatal)
    @w.m.guard_life_pump
    def work(*_): pytest.fail('no input beneath a disconnection dialog')
    work(None,SimpleNamespace(hwnd=123))
    assert controller.get_controller().paused
    assert '連線已中斷' in controller.get_controller().current_action


def test_expired_test_lease_does_not_keep_waiting_or_resume_input(waiting,monkeypatch):
    w = waiting
    from mm_assist.safety import lease
    monkeypatch.setattr(lease,'in_test_mode',lambda:True)
    monkeypatch.setattr(lease,'current_lease',lambda:SimpleNamespace(alive=lambda:w.clock[0] < 1))
    @w.m.guard_life_pump
    def work(*_): pytest.fail('lease must not be renewed by network waiting')
    work(None,SimpleNamespace(hwnd=123))
    assert w.clock[0] < 2 and w.scans == [0.]


@pytest.mark.parametrize('scale',[1.,.5,.385416667])
def test_registered_full_caption_matches_real_pixels_at_client_scales(waiting,scale):
    from mm_assist.perception.network_toast import network_unstable
    frame = cv2.resize(waiting.toast,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    assert network_unstable(frame)


@pytest.mark.parametrize('case',['caption_removed','caption_elsewhere','other_toast','empty'])
def test_wrong_or_unobserved_region_cannot_establish_warning(waiting,case):
    from mm_assist.perception.network_toast import network_unstable
    if case == 'other_toast':
        frame = cv2.imdecode(np.fromfile(ROOT/'tests/fixtures/life_faults_20261024/medicine_juice_list.png',np.uint8),1)
    elif case == 'empty':
        with pytest.raises(ValueError): network_unstable(None)
        with pytest.raises(ValueError): network_unstable(np.zeros_like(waiting.toast))
        return
    else:
        frame = waiting.toast.copy()
        caption = frame[140:180,888:1083].copy()
        frame[112:215,790:1140] = 30
        if case == 'caption_elsewhere': frame[500:540,888:1083] = caption
    assert not network_unstable(frame)


def test_pending_major_fact_survives_and_old_idle_time_is_discarded(waiting,monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult
    w = waiting
    controller.wait_action_completion('p1',1000,ExecuteResult(True,'done',action_issued=True),actions_complete=True)
    controller._STATE.pending_action.idle.zero_since = -10
    def capture(*_):
        w.scans.append(w.clock[0])
        if w.clock[0] >= 2:
            p = controller.get_controller().pending_action
            assert p is not None and p.result.action_issued and p.actions_complete
            assert p.idle.zero_since is None
        return w.toast if w.clock[0] < 4 else w.clear
    monkeypatch.setattr(w.m,'_capture',capture)
    @w.m.guard_life_pump
    def work(*_):
        p = controller.get_controller().pending_action
        assert p is not None and p.result.action_issued and p.actions_complete
    work(None,SimpleNamespace(hwnd=123))
    assert w.scans == pytest.approx([0.,2.,4.])


def test_warning_at_native_boundary_waits_before_authorization_and_new_stack(waiting,monkeypatch):
    from mm_assist.safety import input as inputs, lease
    from mm_assist.memory.readers import ui_stack
    w = waiting
    captures, events = [], []
    def capture(*_):
        captures.append(w.clock[0])
        return w.clear if len(captures) == 1 or w.clock[0] >= 4 else w.toast
    monkeypatch.setattr(w.m,'_capture',capture)
    monkeypatch.setattr('mm_assist.jobs.bar.click_owner_block_reason',lambda:'')
    monkeypatch.setattr(lease,'require_action',lambda *a,**k:events.append(('authorization',w.clock[0])))
    monkeypatch.setattr(ui_stack,'shared_ui_stack_reader',lambda:SimpleNamespace(read=lambda **k:
        events.append(('fresh_stack',w.clock[0])) or ui_stack.UiStackSnap(ready=True,names=['TutorialTooltipUIState'])))
    @w.m.guard_life_pump
    def work(*_):
        inputs._guard_before_input('click',SimpleNamespace(hwnd=123),method='sendinput',client_xy=(10,10))
        events.append(('dispatch_allowed',w.clock[0]))
    work(None,SimpleNamespace(hwnd=123))
    assert events == [('authorization',4.),('fresh_stack',4.),('dispatch_allowed',4.)]


def test_held_gesture_abort_poll_defers_network_wait_to_next_boundary(monkeypatch):
    from mm_assist.safety import input as inputs
    m = importlib.import_module('mm_assist.life_plan.game_stop')
    calls = []
    monkeypatch.setattr(m,'game_stop_interrupted',lambda **kw:calls.append(kw) or False)
    inputs.check_abort()
    assert calls == [{'check_network':False}]


@pytest.mark.parametrize('earlier_check',[False,True])
def test_native_boundary_does_not_reuse_target_changed_during_network_wait(waiting,monkeypatch,earlier_check):
    from mm_assist.safety import input as inputs, autofarm_gates as gates
    from mm_assist.safety import lease
    from mm_assist.memory.readers import ui_stack
    w = waiting
    changed = w.clear.copy()
    changed[0:110,0:700] = 20
    captures = []
    def capture(*_):
        captures.append(w.clock[0])
        if len(captures) == 1: return w.clear
        return w.toast if w.clock[0] < 4 else changed
    monkeypatch.setattr(w.m,'_capture',capture)
    monkeypatch.setattr('mm_assist.jobs.bar.click_owner_block_reason',lambda:'')
    monkeypatch.setattr(lease,'require_action',lambda *a,**k:None)
    monkeypatch.setattr(ui_stack,'shared_ui_stack_reader',lambda:SimpleNamespace(read=lambda **k:
        ui_stack.UiStackSnap(ready=True,names=['TutorialTooltipUIState'])))
    gates.reset_stage_gates_for_tests()
    @w.m.guard_life_pump
    def work(*_):
        if earlier_check:
            assert not w.m.game_stop_interrupted(force=True)
        with pytest.raises(RuntimeError,match='click_ui_not_ready: 网络等待后'):
            gates.stage_click('network_native',lambda x,y:inputs._guard_before_input(
                'click',SimpleNamespace(hwnd=123),method='sendinput',client_xy=(x,y)),10,10)
        assert gates._stage_counts['network_native'] == 0
    work(None,SimpleNamespace(hwnd=123))
    gates.reset_stage_gates_for_tests()


@pytest.mark.parametrize('earlier_check',[False,True])
def test_click_spot_relocates_instead_of_using_point_from_before_wait(waiting,monkeypatch,earlier_check):
    from mm_assist.life_execute.ui_act import click_spot
    w = waiting
    changed = w.clear.copy()
    changed[:110,:700] = 20
    captures,clicks = [],[]
    def capture(*_):
        captures.append(w.clock[0])
        if len(captures) == 1: return w.clear
        return w.toast if w.clock[0] < 4 else changed
    monkeypatch.setattr(w.m,'_capture',capture)
    @w.m.guard_life_pump
    def work(*_):
        w.clock[0] = .3
        if earlier_check:
            assert not w.m.game_stop_interrupted(force=True)
        result = click_spot(lambda x,y:clicks.append((x,y)) or True,(10,10),'加工',wait_refresh=False)
        assert result.status == 'input_layer_wait'
        assert clicks == []
        assert controller.get_controller().running
    work(None,SimpleNamespace(hwnd=123))


@pytest.mark.parametrize('pending_major',[False,True])
def test_network_target_refusal_reacquires_only_before_a_major_action(waiting,pending_major):
    from mm_assist.life_plan import bar_ui
    from mm_assist.life_plan.types import ExecuteResult
    item = controller.get_controller().current
    if pending_major:
        controller.wait_action_completion(item.id,1000,
            ExecuteResult(False,'error','major already sent',action_issued=True))
    @waiting.m.guard_life_pump
    def work(*_):
        result = bar_ui._execute_with_recovery(lambda *_a,**_k:ExecuteResult(False,'error',
            'click_ui_not_ready: 网络等待后页面或目标已变化，须重新认页定位'),item)
        assert result.status == ('error' if pending_major else 'input_layer_wait')
        state = controller.get_controller()
        assert state.running and not state.paused and state.current == item
        assert (state.pending_action is not None) == pending_major
    work(None,SimpleNamespace(hwnd=123))


@pytest.mark.parametrize('event',['pause','new_start','exception'])
def test_late_poll_cannot_resume_or_overwrite_newer_state(waiting,monkeypatch,event):
    w = waiting
    def capture(*_):
        if w.clock[0] < 2: return w.toast
        if event == 'pause': controller.pause_life()
        elif event == 'new_start': controller.start_life(snapshot=LifeSnapshot(ok=True),now=2000)
        else: raise OSError('capture failed')
        return w.clear
    monkeypatch.setattr(w.m,'_capture',capture)
    @w.m.guard_life_pump
    def work(*_): pytest.fail('late observation must not resume this call')
    work(None,SimpleNamespace(hwnd=123))
    state = controller.get_controller()
    assert state.life_started_at == (2000 if event == 'new_start' else 1000)
    assert (not state.paused if event == 'new_start' else state.paused)


def test_network_wait_does_not_exhaust_page_refresh_budget(waiting,monkeypatch):
    from mm_assist.life_execute import ui_logic
    w = waiting
    captures = []
    def capture(*_):
        captures.append(w.clock[0])
        return w.clear if len(captures) == 1 or w.clock[0] >= 4 else w.toast
    monkeypatch.setattr(w.m,'_capture',capture)
    monkeypatch.setattr(w.m.time,'time',lambda:1000+w.clock[0])
    monkeypatch.setattr(ui_logic,'skip_wait_ui_params',lambda:False)
    stamps = iter([(1.,1.),(2.,2.),(3.,3.)])
    monkeypatch.setattr(ui_logic,'ui_param_stamps',lambda **_:next(stamps))
    monkeypatch.setattr(ui_logic,'ui_param_key',lambda **_:('facility','medicine'))
    stack_stamps = iter([2.,3.])
    monkeypatch.setattr(ui_logic,'_ui_stack_sample_ts',lambda **_:next(stack_stamps))
    @w.m.guard_life_pump
    def work(*_):
        assert ui_logic.wait_ui_params_refresh(timeout_s=2,pre_wait_s=0,
            cancelled=lambda:w.m.game_stop_interrupted(force=True))
    work(None,SimpleNamespace(hwnd=123))
    assert w.clock[0] >= 4


def test_navigation_keeps_remaining_page_budget_after_network_hold(waiting,monkeypatch):
    from mm_assist.life_execute import ui_nav
    from mm_assist.life_execute.ui_logic import UiWhere
    w = waiting
    captures = []
    def capture(*_):
        captures.append(w.clock[0])
        return w.clear if len(captures) == 1 or w.clock[0] >= 4 else w.toast
    monkeypatch.setattr(w.m,'_capture',capture)
    monkeypatch.setattr(w.m.time,'time',lambda:1000+w.clock[0])
    monkeypatch.setattr(ui_nav,'resolve_navigation',
        lambda _:UiWhere('main_hud' if w.clock[0] >= 7 else 'unknown',0,0,0,'fixture'))
    @w.m.guard_life_pump
    def work(*_):
        result = ui_nav.ensure_node('main_hud',grab=lambda:object(),
            click=lambda *_:pytest.fail('page waiting must not click'),already_ok=lambda _:True)
        assert result.ok, result
    work(None,SimpleNamespace(hwnd=123))
    assert w.clock[0] >= 7


def test_skill_list_still_waits_for_stable_target_after_long_network_hold(waiting,monkeypatch):
    from mm_assist.life_execute import skill_list,ui_act
    from mm_assist.perception import item_sense
    w = waiting
    captures,hits = [],[]
    def capture(*_):
        captures.append(w.clock[0])
        return w.clear if len(captures) == 1 or w.clock[0] >= 8 else w.toast
    monkeypatch.setattr(w.m,'_capture',capture)
    def locate(*a,**k):
        hits.append(w.clock[0])
        return dict(ok=True,click_client=(40,40))
    monkeypatch.setattr(item_sense,'locate_skill_item',locate)
    @w.m.guard_life_pump
    def work(*_):
        _,item,error = skill_list.seek_item(w.clear,dict(ok=True,click_client=(10,10)),
            skill_id='herb',name='淨淨蘑菇',grab=lambda:w.clear,
            cancelled=ui_act.pump_aborted,sleep=w.m.time.sleep)
        assert error is None, error
        assert item['click_client'] == (40,40) and len(hits) == 2
    work(None,SimpleNamespace(hwnd=123))
    assert w.clock[0] >= 8


def test_insect_fresh_stack_budget_survives_long_network_hold(waiting,monkeypatch):
    from mm_assist.life_execute import insect_obtain
    from mm_assist.perception import insect_obtain_sense as sense
    w = waiting
    captures,observations,clicks = [],[],[]
    def capture(*_):
        captures.append(w.clock[0])
        return w.clear if len(captures) == 1 or w.clock[0] >= 8 else w.toast
    monkeypatch.setattr(w.m,'_capture',capture)
    def observe(_):
        observations.append(w.clock[0])
        if len(observations) == 1:
            w.clock[0] = .3
            return SimpleNamespace(node='unknown',stack_ready=False,stack_sampled_ts=0)
        return SimpleNamespace(node='obtain_method',stack_ready=True,stack_sampled_ts=1000+w.clock[0]+.001)
    monkeypatch.setattr(sense,'phase',lambda *a,**k:'obtain')
    monkeypatch.setattr(sense,'locate',lambda *a,**k:(10,20))
    insect_obtain.reset_insect_visits_for_tests()
    @w.m.guard_life_pump
    def work(*_):
        result = insect_obtain.execute_insect_obtain(mat='獨角仙',grab=lambda:w.clear,
            do_click=lambda *xy:clicks.append(xy) or True,resolve=observe,
            sleep_fn=w.m.time.sleep,now_fn=lambda:1000+w.clock[0],defer_completion=True)
        assert result.status == 'action_pending' and result.action_issued, result
    try:
        work(None,SimpleNamespace(hwnd=123))
        assert clicks == [(10,20)] and w.clock[0] >= 8
    finally:
        insect_obtain.reset_insect_visits_for_tests()


@pytest.mark.parametrize('capture_cost,interruption',[
    (.35,None),(1.2,None),(.35,'pause'),(.35,'new_start'),(.35,'f8')])
def test_foreground_dispatch_uses_one_fresh_frame_boundary_within_original_budget(waiting,monkeypatch,capture_cost,interruption):
    import win32gui
    from mm_assist.safety import input as inputs,lease
    from mm_assist.memory.readers import ui_stack
    w=waiting
    win=SimpleNamespace(hwnd=123,left=0,top=0,width=1919,height=1440)
    frame=cv2.imdecode(np.fromfile(ROOT/'tests/fixtures/life_faults_20261025/focus_timeout_hud.png',np.uint8),1)
    foreground,events=[False],[]
    def capture(*_):
        w.clock[0]+=capture_cost
        return frame
    def stack(**_):
        w.clock[0]+=.18
        events.append('fresh_stack')
        if foreground[0] and interruption:
            if interruption=='pause':controller.pause_life()
            elif interruption=='new_start':controller.start_life(snapshot=LifeSnapshot(ok=True),now=2000)
            else:lease._cancel_event.set()
        return ui_stack.UiStackSnap(ready=True,names=['TutorialTooltipUIState'],sampled_ts=1000+w.clock[0])
    monkeypatch.setattr(w.m,'_capture',capture)
    monkeypatch.setattr('mm_assist.safety.readonly.deny_input',lambda *_:None)
    monkeypatch.setattr('mm_assist.life_plan.live_test.record_input',lambda *_:None)
    monkeypatch.setattr(lease,'require_action',lambda *a,**k:None)
    monkeypatch.setattr('mm_assist.jobs.bar.click_owner_block_reason',lambda:'')
    monkeypatch.setattr(ui_stack,'shared_ui_stack_reader',lambda:SimpleNamespace(read=stack))
    monkeypatch.setattr(inputs,'install_mouse_safety',lambda:None)
    monkeypatch.setattr(inputs,'is_foreground',lambda _:foreground[0])
    monkeypatch.setattr(inputs,'focus_window',lambda _:foreground.__setitem__(0,True))
    monkeypatch.setattr(inputs,'_sendinput_yield_unfocused',lambda *a:False)
    monkeypatch.setattr(inputs,'_refresh_same_client_size',lambda _:win)
    monkeypatch.setattr(inputs,'_require_pointer_geometry',lambda _:None)
    monkeypatch.setattr(inputs,'require_game_point',lambda *a:None)
    monkeypatch.setattr(inputs,'hud_suspend_for_input',lambda **_:w.clock.__setitem__(0,w.clock[0]+.16))
    monkeypatch.setattr(inputs,'hud_resume_after_input',lambda:None)
    monkeypatch.setattr(inputs,'_stamp_sendinput_heartbeat',lambda:None)
    monkeypatch.setattr(inputs,'_send_mouse',lambda flags,*_:events.append(flags))
    monkeypatch.setattr(win32gui,'GetCursorPos',lambda:(20,20))
    @w.m.guard_life_pump
    def work(*_):
        with inputs.foreground_input_scope(win.hwnd):
            if interruption:
                with pytest.raises(inputs.AbortRequested):inputs.click_client(win,20,20)
            elif capture_cost>1:
                with pytest.raises(RuntimeError,match='focus_click_timeout'):inputs.click_client(win,20,20)
            else:
                assert inputs.click_client(win,20,20)
    work(None,win)
    downs=[x for x in events if isinstance(x,int) and x&inputs.MOUSEEVENTF_LEFTDOWN]
    assert len(downs)==(1 if capture_cost<1 and not interruption else 0)
    assert events.count('fresh_stack')==2


@pytest.mark.parametrize('change',['none','wrong_popup','new_wait','expired','pause','new_start','exception'])
def test_real_new_result_target_after_network_does_not_replay_claim(waiting,monkeypatch,change):
    from mm_assist.life_execute import ui_act,ui_logic
    from mm_assist.life_plan.types import ExecuteResult
    from mm_assist.safety import input as inputs,lease,autofarm_gates as gates
    from mm_assist.memory.readers import ui_stack
    w=waiting
    result_frame=cv2.imdecode(np.fromfile(ROOT/'tests/fixtures/life_faults_20261025/network_claim_result.png',np.uint8),1)
    captures,clicked=[],[]
    target_ready=[False]
    popup=['FacilityResultUIState','FacilityResultUIInstance','AlteringFacilityPopupUIInstance','TutorialTooltipUIState']
    def capture(*_):
        captures.append(w.clock[0])
        if len(captures)==1:return w.clear
        return w.toast if w.clock[0]<4 or (change=='new_wait' and target_ready[0] and w.clock[0]<6) else result_frame
    monkeypatch.setattr(w.m,'_capture',capture)
    monkeypatch.setattr(w.m.time,'time',lambda:1000+w.clock[0])
    monkeypatch.setattr('mm_assist.jobs.bar.click_owner_block_reason',lambda:'')
    monkeypatch.setattr(lease,'require_action',lambda *a,**k:None)
    monkeypatch.setattr(ui_stack,'shared_ui_stack_reader',lambda:SimpleNamespace(read=lambda **_:
        ui_stack.UiStackSnap(ready=True,names=['TutorialTooltipUIState'] if change=='wrong_popup' else popup,
            sampled_ts=1000+w.clock[0]+.01)))
    monkeypatch.setattr(ui_logic,'wait_ui_params_refresh',lambda **_:True)
    monkeypatch.setattr(ui_act,'read_state_ui',lambda **_:SimpleNamespace(fresh=True,pid_alive=True,
        raw=dict(ui_stack_ready=True,ui_stack_names=['AlteringFacilityPopupUIInstance'] if clicked else popup,
            ui_stack_ts=1000+w.clock[0]+.01)))
    def grab():
        target_ready[0]=True
        if change=='expired':w.clock[0]+=3
        elif change=='pause':controller.pause_life()
        elif change=='new_start':controller.start_life(snapshot=LifeSnapshot(ok=True),now=2000)
        return result_frame
    def click(x,y):
        inputs._guard_before_input('click',SimpleNamespace(hwnd=123),method='sendinput',client_xy=(x,y))
        if change=='exception':raise OSError('uncertain dispatch exception')
        clicked.append((x,y))
        return True
    @w.m.guard_life_pump
    def work(*_):
        assert not w.m.game_stop_interrupted(force=True)
        controller.wait_action_completion('p1',1000,ExecuteResult(False,'error','claim already sent',action_issued=True))
        error=None
        try:
            error=ui_act.dismiss_facility_result(click,grab,present=True)
        except RuntimeError as exc:
            assert str(exc).startswith('click_ui_not_ready'),str(exc)
            error=exc
        except OSError as exc:
            assert change=='exception'
            error=exc
        if change=='none':
            assert error is None,error
            assert len(clicked)==1
            assert gates._stage_counts.get('設施結果確認',0)==0
        else:
            assert error is not None and not clicked
        state=controller.get_controller()
        if change=='new_start':
            assert state.life_started_at==2000 and state.pending_action is None
        else:
            assert state.pending_action is not None
        if change=='exception':assert gates._stage_counts['設施結果確認']==1
        assert w.m._LOCATED_RESULT.get() is None
    work(None,SimpleNamespace(hwnd=123))
