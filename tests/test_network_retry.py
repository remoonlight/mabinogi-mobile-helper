"""User retry dialog: five second hold, guarded click, fresh postcondition."""
import importlib
import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_plan import controller, execution_records
from mm_assist.life_plan.types import LifeSnapshot, PlanItem, ExecuteResult


def pixels():
    raw=(ROOT/'tests/fixtures/game_stop/network_retry_user_20261027.png').read_bytes()
    return cv2.imdecode(np.frombuffer(raw,np.uint8),1)[25:577,4:740]


def test_original_and_registered_crops_retain_independent_fingerprints():
    assets=ROOT/'knowledge/templates/network_retry'
    row=json.loads((assets/'catalog.json').read_text('utf-8'))
    assert hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest()==row['source_sha256']
    for anchor in row['anchors']:
        raw=(assets/anchor['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==anchor['sha256']
        x1,y1,x2,y2=anchor['crop_xyxy']
        assert np.array_equal(cv2.imdecode(np.frombuffer(raw,np.uint8),1),pixels()[y1:y2,x1:x2])


@pytest.mark.parametrize('scale',[1.,2.,1440/552])
def test_real_dialog_locates_button_at_client_scales(scale):
    from mm_assist.perception.network_retry import locate_network_retry
    frame=cv2.resize(pixels(),None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    assert locate_network_retry(frame)==pytest.approx((432*scale,497*scale),abs=2)


@pytest.mark.parametrize('remove',['title','button','black','other_page','fatal'])
def test_partial_wrong_and_fatal_pages_never_authorize_retry(remove):
    from mm_assist.perception.network_retry import locate_network_retry
    frame=pixels().copy()
    if remove=='title': frame[332:354,299:437]=30
    elif remove=='button': frame[478:515,370:495]=30
    elif remove=='black': frame[:]=0
    elif remove=='other_page':
        frame=cv2.imread(str(ROOT/'tests/fixtures/game_stop/network_unstable_20261024.png'))
    else: frame=cv2.imread(str(ROOT/'tests/fixtures/game_stop/both_1920.png'))
    assert locate_network_retry(frame) is None


@pytest.fixture
def scene(monkeypatch,tmp_path):
    m=importlib.import_module('mm_assist.life_plan.game_stop')
    controller.reset_controller_for_tests()
    item=PlanItem('p1','feed','投料 布料')
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:[item])
    controller.start_life(snapshot=LifeSnapshot(ok=True),now=1000)
    monkeypatch.setattr(execution_records,'_RECORDER',execution_records.ExecutionRecorder(tmp_path/'records.json'))
    from mm_assist.safety import autofarm_gates
    autofarm_gates.reset_stage_gates_for_tests()
    clock=[0.]
    monkeypatch.setattr(m.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(m.time,'sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons',lambda:None)
    frame=pixels()
    clear=np.full_like(frame,30)
    clicks=[]
    monkeypatch.setattr(m,'_capture',lambda *_:clear if clicks else frame)
    def click(win,x,y,**kw):
        # Real native guard must be able to reenter without recursively retrying.
        assert not m.game_stop_interrupted(force=True)
        clicks.append((clock[0],x,y,kw['method']))
        return True
    monkeypatch.setattr('mm_assist.safety.input.click_client',click)
    yield SimpleNamespace(m=m,clock=clock,clicks=clicks,frame=frame,clear=clear,item=item,
        win=SimpleNamespace(hwnd=123,width=736,height=552))
    controller.reset_controller_for_tests()
    autofarm_gates.reset_stage_gates_for_tests()


def run(scene):
    calls=[]
    @scene.m.guard_life_pump
    def work(*_): calls.append('business')
    work(None,scene.win)
    return calls


@pytest.mark.parametrize('tracking',[False,True])
def test_wait_five_click_once_then_new_pump_rechecks_without_replaying_pending(scene,tracking):
    controller._STATE.pause_on_failure=tracking
    controller.wait_action_completion('p1',1000,ExecuteResult(False,'action_pending',action_issued=True))
    pending=controller.get_controller().pending_action
    assert run(scene)==[]
    assert len(scene.clicks)==1 and scene.clicks[0][:3]==pytest.approx((5.,432,497))
    assert controller.get_controller().pending_action==pending
    assert not controller.get_controller().paused
    assert run(scene)==['business']


@pytest.mark.parametrize('event',['pause','new_start','f8','lease'])
def test_cancellation_during_hold_never_clicks_or_overwrites_new_state(scene,monkeypatch,event):
    from mm_assist.safety import lease
    if event=='lease':
        monkeypatch.setattr(lease,'in_test_mode',lambda:True)
        monkeypatch.setattr(lease,'current_lease',lambda:SimpleNamespace(alive=lambda:scene.clock[0]<1))
    def sleep(dt):
        scene.clock[0]+=dt
        if scene.clock[0]>=1:
            if event=='pause': controller.pause_life()
            elif event=='new_start': controller.start_life(snapshot=LifeSnapshot(ok=True),now=2000)
            elif event=='f8': lease._cancel_event.set()
    monkeypatch.setattr(scene.m.time,'sleep',sleep)
    assert run(scene)==[] and scene.clicks==[]
    if event=='new_start': assert controller.get_controller().life_started_at==2000


@pytest.mark.parametrize('change',['clear','black','fatal','exception'])
def test_changed_or_unavailable_frame_before_dispatch_never_clicks(scene,monkeypatch,change):
    def capture(*_):
        if scene.clock[0]<5: return scene.frame
        if scene.clock[0]>=10 and change in {'exception','black'}: return scene.clear
        if change=='exception': raise OSError('render failed')
        if change=='clear': return scene.clear
        if change=='black': return np.zeros_like(scene.frame)
        return cv2.imread(str(ROOT/'tests/fixtures/game_stop/both_1920.png'))
    monkeypatch.setattr(scene.m,'_capture',capture)
    assert run(scene)==[] and scene.clicks==[]


def test_unresponsive_button_keeps_retrying_every_five_seconds_without_pause(scene,monkeypatch):
    monkeypatch.setattr(scene.m,'_capture',lambda *_:scene.clear if len(scene.clicks)>=3 else scene.frame)
    assert run(scene)==[]
    assert [c[0] for c in scene.clicks]==pytest.approx([5.,10.,15.])
    assert not controller.get_controller().paused
    assert controller.get_controller().queue==[scene.item]


def test_input_refusal_waits_for_new_evidence_without_pausing(scene,monkeypatch):
    from mm_assist.safety import input as inputs
    calls=[]
    def refuse(*a,**kw):
        calls.append(scene.clock[0])
        if len(calls)==1: raise RuntimeError('click_ui_not_ready: target changed')
        scene.clicks.append((scene.clock[0],a[1],a[2],kw['method']))
        return True
    monkeypatch.setattr(inputs,'click_client',refuse)
    assert run(scene)==[]
    assert calls==pytest.approx([5.,10.])
    assert not controller.get_controller().paused


def test_real_native_input_guard_rechecks_retry_pixels_and_ready_stack(scene,monkeypatch):
    from mm_assist.safety import input as inputs,lease
    from mm_assist.memory.readers import ui_stack
    checks=[]
    monkeypatch.setattr(ui_stack,'shared_ui_stack_reader',lambda:SimpleNamespace(read=lambda **kw:
        checks.append(kw) or SimpleNamespace(ready=True,sampled_ts=scene.m.time.time(),names=['HudUIStepState'])))
    monkeypatch.setattr('mm_assist.jobs.bar.click_owner_block_reason',lambda:'')
    monkeypatch.setattr(lease,'require_action',lambda *a,**kw:checks.append(kw))
    def click(win,x,y,**kw):
        inputs._guard_before_input('click',win,method=kw['method'],client_xy=(x,y))
        scene.clicks.append((scene.clock[0],x,y,kw['method']))
        return True
    monkeypatch.setattr(inputs,'click_client',click)
    assert run(scene)==[] and len(scene.clicks)==1
    assert {'force':True} in checks
    assert not controller.get_controller().paused


def test_fatal_during_retry_keeps_exact_fatal_frame_and_original_pause_rule(scene,monkeypatch):
    fatal=cv2.imread(str(ROOT/'tests/fixtures/game_stop/both_1920.png'))
    saved=[]
    monkeypatch.setattr(scene.m,'_capture',lambda *_:scene.frame if scene.clock[0]<5 else fatal)
    monkeypatch.setattr(scene.m.GameStopWatch,'_evidence',lambda self,frame,*args:saved.append(frame))
    assert run(scene)==[] and scene.clicks==[]
    assert controller.get_controller().paused
    assert '連線已中斷' in controller.get_controller().current_action
    assert len(saved)==1 and np.array_equal(saved[0],fatal)
