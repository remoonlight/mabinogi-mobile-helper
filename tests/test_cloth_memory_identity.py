"""Real cloth overview was classified as wood before locating visible 布料+."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from mm_assist.perception import process_sense as sense

FIX=Path(__file__).parent/'fixtures'

def fixture():
    from test_facility_memory import HISTORICAL_NAMES
    row=json.loads((FIX/'facility_memory/cloth_identity_20261013.json').read_text('utf-8'))
    for attribute in row['render']['root_attributes']:
        attribute['name']=HISTORICAL_NAMES[attribute['name_id']]
    frame=cv2.imdecode(np.fromfile(FIX/'process_stance/cloth_facility_misread_20261013.png',np.uint8),cv2.IMREAD_COLOR)
    return frame,row

def test_original_cloth_pixels_and_native_render_independently_prove_cloth(monkeypatch):
    from mm_assist.memory.readers.facility_ui import select_facility_view
    frame,row=fixture()
    monkeypatch.setattr(sense,'_ocr_lines',lambda *a,**k:[])
    # The original failure image stays unchanged. The title fix must now
    # recognize it correctly; pinning the former "wood" error blocks that fix.
    image_hit=sense.locate_process_recipe(frame,facility='cloth',name='布料+')
    assert sense._template_title_facility(frame)=='cloth' and image_hit['ok']
    native=select_facility_view(row['render'],2)
    assert native=={k:v for k,v in row['fact'].items() if k not in ('game_pid','identity')}

def test_memory_identity_locates_real_recipe_without_ocr(monkeypatch):
    frame,row=fixture()
    monkeypatch.setattr(sense,'_ocr_lines',lambda *a,**k:[])
    monkeypatch.setattr(sense,'_template_title_facility',lambda *a,**k:pytest.fail('old title identity'))
    hit=sense.locate_process_recipe(frame,facility='cloth',name='布料+',memory_view=row['fact'])
    assert hit['ok'] and hit['src']=='template'
    assert abs(hit['click_client'][0]-1276)<5 and abs(hit['click_client'][1]-391)<5

@pytest.mark.parametrize('change',['wrong','error','identity','window','screen','empty'])
def test_invalid_memory_identity_cannot_authorize_recipe_or_fall_back(monkeypatch,change):
    frame,row=fixture();fact=copy.deepcopy(row['fact'])
    if change=='wrong':fact['facility']='metal'
    if change=='error':fact={'error':'memory_changed'}
    if change=='identity':fact['identity'][0]=0
    if change=='window':fact['window']+=1
    if change=='screen':fact['screen_size']=[100,100]
    if change=='empty':frame=np.zeros((0,0,3),np.uint8)
    monkeypatch.setattr(sense,'_template_title_facility',lambda *a,**k:pytest.fail('fallback'))
    monkeypatch.setattr(sense,'_ocr_lines',lambda *a,**k:pytest.fail('OCR fallback'))
    assert not sense.locate_process_recipe(frame,facility='cloth',name='布料+',memory_view=fact)['ok']

@pytest.mark.parametrize('changed',[False,True])
def test_executor_binds_paired_native_identity_to_recipe_locator(monkeypatch,changed):
    from mm_assist.life_execute import medicine_process as process,ui_act,ui_logic
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests
    reset_stage_gates_for_tests()
    frame,row=fixture();clicks=[];reads=[0]
    def memory():
        reads[0]+=1
        if changed and reads[0]>=4 and reads[0]%2==0:return None
        return copy.deepcopy(row['fact'])
    monkeypatch.setattr(sense,'_ocr_lines',lambda *a,**k:[])
    monkeypatch.setattr(sense,'locate_process_button',lambda *a,**k:
        {'ok':True,'at_facility':True,'label':'加工','click_client':[1493,1354]} if clicks else {'ok':False})
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:False)
    monkeypatch.setattr(ui_act,'stack_has_facility_result',lambda **k:False)
    monkeypatch.setattr(ui_act,'stack_has_facility_recipe',lambda **k:False)
    monkeypatch.setattr(ui_logic,'resolve_ui_node',lambda **k:SimpleNamespace(node='facility:cloth',facility_id='cloth'))
    out=process._run_process(None,facility='cloth',recipe='布料+',mode='feed',items={'布料':5,'羊毛':99},
        do_click=lambda x,y:clicks.append((x,y)) or True,read_items=None,grab=lambda:frame,
        on_progress=None,root=None,sleep_fn=lambda s:None,read_stp=lambda:0,facilities={'cloth':()},
        defer_completion=True,memory_reader=memory)
    if changed:
        assert not out.ok and not clicks
    else:
        assert out.ok and out.action_issued and len(clicks)==2
        assert abs(clicks[0][0]-1276)<5 and clicks[1]==(1493,1354)
