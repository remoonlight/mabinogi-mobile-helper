"""Actual 2026-10-23 iron popup must verify its full title before nearby input."""
import cv2
import numpy as np
import pytest
from mm_assist.config import ROOT
from mm_assist.life_plan import node_inspect
from mm_assist.life_execute.skill_nearby import execute_skill_nearby
from mm_assist.life_execute import ui_logic
from mm_assist.perception import item_sense

def pixels(path):
    frame=cv2.imdecode(np.fromfile(ROOT/path,dtype=np.uint8),1)
    assert frame is not None
    return frame

@pytest.fixture(autouse=True)
def no_ocr(monkeypatch):
    def forbidden(*a,**k):pytest.fail('registered iron input must be image verified')
    monkeypatch.setattr(node_inspect,'_ocr_lines_roi',forbidden)
    monkeypatch.setattr(item_sense,'ocr_image',forbidden)
    monkeypatch.setattr(item_sense,'ocr_roi',forbidden)

@pytest.mark.parametrize('path',[
    'tests/fixtures/life_faults_20261023/iron_detail.png',
    'tests/fixtures/process_stance/iron_detail_failed_20261010.png',
])
def test_current_and_old_popup_title_permit_exactly_one_nearby(monkeypatch,path):
    frame=pixels(path)
    assert node_inspect.gather_detail_for_target(frame,'鐵礦脈')=='gather_detail:鐵礦脈'
    monkeypatch.setattr(ui_logic,'read_inspect_record',lambda **k:{'node':'gather_detail','zh':'技能详情'})
    clicks=[];issued=[]
    result=execute_skill_nearby(skill_id='mine',item_name='鐵礦脈',output='鐵礦石',
        grab=lambda:frame,do_click=lambda *p:clicks.append(p) or True,
        on_action_issued=lambda:issued.append(True),defer_completion=True)
    assert result.action_issued and result.status=='action_pending'
    assert len(clicks)==1 and issued==[True]

@pytest.mark.parametrize('kind',['wrong_title','missing_title','missing_nearby'])
def test_other_or_incomplete_popup_never_uses_plausible_display_name(monkeypatch,kind):
    frame=pixels('tests/fixtures/life_faults_20261023/iron_detail.png')
    if kind=='wrong_title':frame=pixels('tests/fixtures/process_stance/mushroom_detail_20261008.png')
    elif kind=='missing_title':frame[390:450,675:810]=0
    else:frame[780:830,675:860]=0
    monkeypatch.setattr(ui_logic,'read_inspect_record',lambda **k:{'node':'gather_detail:鐵礦脈','zh':'技能详情-鐵礦脈'})
    result=execute_skill_nearby(skill_id='mine',item_name='鐵礦脈',output='鐵礦石',
        grab=lambda:frame,do_click=lambda *p:pytest.fail('no input on unproved popup'),defer_completion=True)
    assert not result.ok and not result.action_issued

def test_actual_list_to_popup_chain_requires_both_registered_names(monkeypatch):
    listing=pixels('tests/fixtures/life_faults_20261023/mine_list.png')
    popup=pixels('tests/fixtures/life_faults_20261023/iron_detail.png')
    frames=iter([listing,listing.copy(),popup])
    monkeypatch.setattr(ui_logic,'read_inspect_record',lambda **k:{'node':'gather_skill:mine','zh':'生活技能'})
    clicks=[]
    result=execute_skill_nearby(skill_id='mine',item_name='鐵礦脈',output='鐵礦石',
        grab=lambda:next(frames),do_click=lambda *p:clicks.append(p) or True,defer_completion=True)
    assert result.status=='action_pending' and result.action_issued and len(clicks)==2
