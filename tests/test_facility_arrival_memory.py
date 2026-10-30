"""Arrival HUD/stp may settle before the current claimable MMML pair is ready."""
import copy
from types import SimpleNamespace
import cv2
import pytest
from mm_assist.config import ROOT
from mm_assist.life_execute import medicine_process as process,ui_act,ui_logic
from mm_assist.memory.readers.facility_ui import select_facility_view
from mm_assist.life_plan.types import ExecuteResult
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests
from test_facility_memory import sample


@pytest.mark.parametrize('case',['ready','timeout','pause','wrong_facility'])
def test_travel_arrival_waits_for_current_memory_without_repeating_go(monkeypatch,case):
    reset_stage_gates_for_tests()
    away=select_facility_view(sample()['render'],0)
    arrived=select_facility_view(sample('metal_arrived')['render'],0)
    image=cv2.imread(str(ROOT/'tests/fixtures/facility_memory/arrival_pending_20261014.png'))
    state={'travel':False,'landed':False,'stp_reads':0,'memory_reads':0,'result':False,'clock':0.0}
    clicks=[]
    def stp():
        if not state['travel']:return 0
        state['stp_reads']+=1
        if state['stp_reads']==1:return 1
        state['landed']=True
        return 0
    def memory():
        if not state['landed']:return copy.deepcopy(away)
        state['memory_reads']+=1
        if case=='timeout' or state['memory_reads']<=4:return None
        fact=copy.deepcopy(arrived)
        if case=='wrong_facility':fact['facility']='wood'
        return fact
    def click(x,y):
        clicks.append((x,y))
        if (x,y)==(223,305):state['travel']=True
        elif (x,y)==(187,1087):
            assert state['landed'] and state['memory_reads']>=6
            state['result']=True
        else:pytest.fail('unexpected input')
        return True
    def sleep(n):state['clock']+=n
    monkeypatch.setattr(process.time,'monotonic',lambda:state['clock'])
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:case=='pause' and state['landed'] and state['memory_reads']>0)
    monkeypatch.setattr(ui_act,'stack_has_facility_recipe',lambda **k:False)
    monkeypatch.setattr(ui_act,'stack_has_common_popup',lambda **k:False)
    monkeypatch.setattr(ui_act,'confirm_go_facility_popup',lambda *a,**k:None)
    monkeypatch.setattr(ui_act,'stack_has_facility_result',lambda **k:state['result'])
    monkeypatch.setattr(ui_act,'dismiss_facility_result',lambda *a,**k:None)
    monkeypatch.setattr(ui_act,'prepare_to',lambda *a,**k:ExecuteResult(False,'wrong_facility'))
    monkeypatch.setattr('mm_assist.perception.process_sense.detect_facility_page',lambda *a,**k:{'ok':True,'facility':'metal'})
    monkeypatch.setattr(ui_logic,'read_inspect_page',lambda **k:'facility:metal')
    monkeypatch.setattr(ui_logic,'resolve_ui_node',lambda **k:SimpleNamespace(node='facility:metal',facility_id='metal'))
    result=process._run_process(None,facility='metal',recipe='',mode='claim',items={},
        do_click=click,grab=lambda:image.copy(),read_items=None,on_progress=None,root=None,
        sleep_fn=sleep,read_stp=stp,defer_completion=True,memory_reader=memory)
    assert result.ok is (case=='ready')
    assert clicks==([(223,305),(187,1087)] if case=='ready' else [(223,305)])
    assert state['clock']<=2.5
    if case=='timeout':assert result.status=='facility_memory_pending'
    if case=='pause':assert result.status=='cancel'
    reset_stage_gates_for_tests()
