"""2026-10-24 debug p9: full coal-vein heading, never its OCR material alias."""
import hashlib,json,time
from types import SimpleNamespace
import cv2,numpy as np
from PIL import Image
import pytest
from mm_assist.config import ROOT
from mm_assist.life_plan import node_inspect
from mm_assist.life_execute import ui_logic,ui_act
from mm_assist.life_execute.skill_nearby import execute_skill_nearby
from mm_assist.memory.readers.ui_stack import UiStackSnap

FIXTURES=ROOT/'tests/fixtures/life_faults_20261024'
def pixels(name='coal_detail.png'):
    result=cv2.imdecode(np.fromfile(FIXTURES/name,dtype=np.uint8),1)
    assert result is not None
    return result
def names():return json.loads((FIXTURES/'coal_detail.json').read_text('utf-8'))['stack_names']
def forbid_ocr(monkeypatch):
    monkeypatch.setattr(node_inspect,'_ocr_lines_roi',lambda *a,**k:pytest.fail('coal must use full registered heading'))

def test_native_failed_frame_identifies_complete_coal_vein(monkeypatch):
    forbid_ocr(monkeypatch)
    assert node_inspect.gather_detail_for_target(pixels(),'煤炭礦脈')=='gather_detail:煤炭礦脈'

def test_default_shared_resolver_uses_current_known_stack_and_full_heading(monkeypatch):
    forbid_ocr(monkeypatch)
    reader=SimpleNamespace(read=lambda **k:UiStackSnap(ready=True,names=names(),sampled_ts=time.time()))
    actual=ui_logic.resolve_navigation(pixels(),reader=reader)
    assert actual.stack_ready and actual.node=='gather_detail:煤炭礦脈'

def test_executor_from_current_detail_dispatches_one_nearby_without_reopening(monkeypatch):
    forbid_ocr(monkeypatch)
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:False)
    monkeypatch.setattr(ui_act,'prepare_to',lambda *a,**k:pytest.fail('already on detail'))
    monkeypatch.setattr(ui_logic,'read_inspect_record',lambda **k:{'node':'gather_detail','zh':'技能详情'})
    clicks=[];issued=[]
    result=execute_skill_nearby(skill_id='mine',item_name='煤炭礦脈',output='煤炭',
        grab=pixels,do_click=lambda *xy:clicks.append(xy) or True,
        on_action_issued=lambda:issued.append(True),defer_completion=True)
    assert result.action_issued and result.status=='action_pending'
    assert len(clicks)==1 and issued==[True]
    assert 675<clicks[0][0]<860 and 780<clicks[0][1]<835

@pytest.mark.parametrize('case',['iron','copper','missing_title','missing_nearby'])
def test_wrong_or_incomplete_popup_never_accepts_stale_coal_display(monkeypatch,case):
    image=pixels()
    if case in ('iron','copper'):
        path=('tests/fixtures/life_faults_20261023/iron_detail.png' if case=='iron'
              else 'tests/fixtures/process_stance/copper_detail_20261011.png')
        image=cv2.imdecode(np.fromfile(ROOT/path,dtype=np.uint8),1)
    elif case=='missing_title':image[393:447,677:843]=0
    else:image[780:835,675:870]=0
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:False)
    monkeypatch.setattr(ui_logic,'read_inspect_record',lambda **k:{'node':'gather_detail:煤炭礦脈','zh':'技能详情-煤炭礦脈'})
    result=execute_skill_nearby(skill_id='mine',item_name='煤炭礦脈',output='煤炭',
        grab=lambda:image,do_click=lambda *xy:pytest.fail('unproved popup cannot click'),defer_completion=True)
    assert not result.ok and not result.action_issued

@pytest.mark.parametrize('case',['unknown_cover','missing_skill_context'])
def test_same_pixels_do_not_override_unknown_or_wrong_stack(case):
    stack=names()
    if case=='unknown_cover':stack.append('UnknownPopup')
    else:stack=[n for n in stack if n not in ('LivingSkillUIInstance','LivingSkillUIState')]
    view=SimpleNamespace(ui_stack_names=stack,ui_stack_ready=True,ui_stack_empty_ok=False,
                         ui_stack_ts=time.time(),ui_stack_error='')
    assert not node_inspect.compute_inspect(pixels(),view=view)['node'].startswith('gather_detail')

def test_coal_registration_is_exact_native_title_crop_at_original_threshold():
    catalog=node_inspect._gather_detail_title_catalog()
    assert catalog['threshold']==.97
    row=next(r for r in catalog['titles'] if r['name']=='煤炭礦脈')
    source=ROOT/row['source_frame'];template=ROOT/row['template']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
    assert hashlib.sha256(template.read_bytes()).hexdigest()==row['template_sha256']
    x,y,w,h=row['crop_client_px']
    assert Image.open(source).crop((x,y,x+w,y+h)).tobytes()==Image.open(template).tobytes()
