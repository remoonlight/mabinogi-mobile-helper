"""A transient post-navigation memory pair is pending, never another click."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from mm_assist.life_execute import medicine_process as process, ui_act


@pytest.mark.parametrize('case', ['ready','timeout','pause','wrong_facility'])
def test_claim_waits_for_a_new_matching_memory_pair_after_navigation(monkeypatch, case):
    from mm_assist.memory.readers.facility_ui import select_facility_view
    from test_facility_memory import sample
    from mm_assist.life_plan.types import ExecuteResult
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests
    reset_stage_gates_for_tests()
    fact=select_facility_view(sample('metal_arrived')['render'],0)
    state={'nav':False,'reads':0,'clock':0.,'result':False};clicks=[]
    def memory():
        if not state['nav']:return None
        state['reads']+=1
        # First post-navigation frame straddles unavailable -> current page.
        if state['reads']==1:return None
        if state['reads']==2 or case in ('ready','pause','wrong_facility'):
            current=copy.deepcopy(fact)
            if case=='wrong_facility' and state['reads']>2:current['facility']='wood'
            return current
        return None
    def prepare(*a,**k):
        state['nav']=True
        return ExecuteResult(True,'at_node','facility')
    def click(*xy):
        assert state['reads']>=4
        clicks.append(xy);state['result']=True;return True
    monkeypatch.setattr(process.time,'monotonic',lambda:state['clock'])
    def sleep(seconds):state['clock']+=seconds
    monkeypatch.setattr(ui_act,'prepare_to',prepare)
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:case=='pause' and state['clock']>0)
    monkeypatch.setattr(ui_act,'stack_has_facility_recipe',lambda **k:False)
    monkeypatch.setattr(ui_act,'stack_has_facility_result',lambda **k:state['result'])
    monkeypatch.setattr(ui_act,'stack_has_common_popup',lambda **k:False)
    monkeypatch.setattr(ui_act,'dismiss_facility_result',lambda *a,**k:None)
    # The page is unproven before navigation; no retained display can substitute.
    monkeypatch.setattr('mm_assist.perception.process_sense.detect_facility_page',lambda *a,**k:{'ok':False})
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_page',lambda **k:'')
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_ui_node',lambda **k:SimpleNamespace(node='unknown',facility_id=''))
    out=process._run_process(None,facility='metal',recipe='',mode='claim',items={},do_click=click,
        read_items=None,grab=lambda:np.zeros((1440,1920,3),np.uint8),on_progress=None,
        root=None,sleep_fn=sleep,read_stp=lambda:0,defer_completion=True,memory_reader=memory)
    assert out.ok is (case=='ready')
    assert clicks==([(187,1087)] if case=='ready' else [])
    if case=='timeout':
        assert out.status=='facility_memory_pending'
        assert state['clock']<=2.1
    elif case=='pause':
        assert out.status=='cancel' and state['clock']<=0.1
    elif case=='wrong_facility':
        assert out.status=='no_facility_page'
