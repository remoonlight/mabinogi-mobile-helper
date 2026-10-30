"""An animated 加工 match while moving must not end the travel observation."""
import pytest
from test_medicine_process import _patch_metal_alloy_ready
from mm_assist.life_execute.medicine_process import execute_facility_feed
from mm_assist.life_plan.types import ExecuteResult


@pytest.mark.parametrize('moving',[1,None])
def test_transient_here_during_movement_waits_for_memory_idle(monkeypatch,moving):
    _patch_metal_alloy_ready(monkeypatch)
    monkeypatch.setattr('mm_assist.life_execute.ui_act.stack_has_common_popup',lambda **k:False)
    monkeypatch.setattr('mm_assist.life_execute.ui_act.stack_has_facility_recipe',lambda:False)
    monkeypatch.setattr('mm_assist.life_execute.ui_act.pump_aborted',lambda:False)
    monkeypatch.setattr('mm_assist.life_execute.ui_act.prepare_to',lambda *a,**k:ExecuteResult(True,'page_matched'))
    clicks=[];waits=[];located=[];go_read_index=[0]
    def locate(*a,**k):
        # Before departure the page says away. The first moving frame wrongly
        # matches the shorter suffix, then the button disappears in HUD.
        n=len(located);located.append(n)
        if len(clicks)==0:
            return {'ok':True,'at_facility':False,'label':'前往加工','click_client':[10,20]}
        if not waits and n==go_read_index[0]:
            return {'ok':True,'at_facility':True,'label':'加工','click_client':[10,20]}
        if len(waits)<2:return {'ok':False}
        return {'ok':True,'at_facility':True,'label':'加工','click_client':[10,20]}
    monkeypatch.setattr('mm_assist.perception.process_sense.locate_process_button',locate)
    monkeypatch.setattr('mm_assist.perception.process_sense.locate_process_recipe',lambda *a,**k:pytest.fail('already in recipe'))
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_ui_node',lambda **k:
        __import__('types').SimpleNamespace(node='facility:metal:合金鋼錠',facility_id='metal'))
    def sleep(seconds):waits.append(seconds)
    def click(x,y):
        if clicks:assert len(waits)>=2
        else:go_read_index[0]=len(located)
        clicks.append((x,y));return True
    result=execute_facility_feed(facility='metal',recipe='合金鋼錠',
        items={'鋼錠':3,'銅礦石':15,'煤炭':8},do_click=click,grab=lambda:object(),
        sleep_fn=sleep,read_stp=lambda:moving if len(waits)<2 else 0,defer_completion=True)
    assert result.ok and result.action_issued and len(clicks)==2 and len(waits)>=2
