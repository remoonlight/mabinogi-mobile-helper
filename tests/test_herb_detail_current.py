"""Debug p7 clicked the correct mushroom but failed independent title recognition."""
import cv2
import pytest
import time
import hashlib
from types import SimpleNamespace
from mm_assist.config import ROOT
from mm_assist.life_execute import skill_nearby, ui_act, ui_logic
from mm_assist.life_plan import node_inspect as inspect


def pixels():
    return cv2.imread(str(ROOT/'tests/fixtures/process_stance/herb_detail_20261020.png'))


def native_view(extra=(),skill=True):
    names=['HudUIStepState','NewProfileUIState','DetailCardUIState','TutorialTooltipUIState',
        'UILayerHideState','NewProfileUIInstance','SizeAdjustmentSender','DetailCardUIInstance']
    if skill:
        names+=['LivingSkillUIState','LivingSkillUIInstance']
    return SimpleNamespace(ui_stack_names=names+list(extra),ui_stack_ready=True,
        ui_stack_empty_ok=False,ui_stack_ts=time.time(),ui_stack_error='')


def test_actual_failed_mushroom_detail_identifies_full_target():
    frame = pixels()
    assert inspect.gather_detail_for_target(frame,'淨淨蘑菇')=='gather_detail:淨淨蘑菇'
    assert inspect.gather_detail_for_target(frame,'咻咻蘑菇')!='gather_detail:咻咻蘑菇'


def test_executor_resumes_actual_target_detail_with_one_nearby_click(monkeypatch):
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:False)
    monkeypatch.setattr(ui_act,'prepare_to',lambda *a,**k:pytest.fail('already at target detail'))
    record=inspect.compute_inspect(pixels(),view=native_view())
    monkeypatch.setattr(ui_logic,'read_inspect_record',lambda **kw:record)
    clicks=[]
    result=skill_nearby.execute_skill_nearby(skill_id='herb',item_name='淨淨蘑菇',output='淨淨蘑菇',
        grab=pixels,do_click=lambda *xy:clicks.append(xy) or False,read_stp=lambda:0,
        sleep_fn=lambda n:None,defer_completion=True)
    assert len(clicks)==1 and 670<clicks[0][0]<860 and 740<clicks[0][1]<820
    assert not result.ok and not result.action_issued


def test_list_label_cannot_substitute_for_hidden_popup_title():
    frame = pixels()
    frame[410:465,675:850]=0
    assert inspect.gather_detail_for_target(frame,'淨淨蘑菇')!='gather_detail:淨淨蘑菇'


def test_native_skill_card_is_skill_detail_with_independent_image_proof(monkeypatch):
    monkeypatch.setattr(inspect,'_ocr_lines_roi',lambda *a,**k:[])
    result=inspect.compute_inspect(pixels(),view=native_view())
    assert result['node']=='gather_detail:淨淨蘑菇'
    assert result['cost']['ocr_n']==0


@pytest.mark.parametrize('case',['covered','other_context','no_pixels'])
def test_same_card_stack_without_safe_skill_context_is_not_gather_detail(case):
    image=pixels() if case!='no_pixels' else None
    view=native_view(extra=('UnknownPopup',) if case=='covered' else (),skill=case!='other_context')
    result=inspect.compute_inspect(image,view=view)
    assert not result['node'].startswith('gather_detail')


def test_current_title_registration_preserves_original_pixels_and_threshold():
    catalog=inspect._gather_detail_title_catalog()
    assert catalog['threshold']==.97
    row=next(r for r in catalog['titles'] if r['name']=='淨淨蘑菇')
    source=ROOT/row['source_frame'];target=ROOT/row['template']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
    assert hashlib.sha256(target.read_bytes()).hexdigest()==row['template_sha256']
    from PIL import Image
    x,y,w,h=row['crop_client_px']
    assert Image.open(source).crop((x,y,x+w,y+h)).tobytes()==Image.open(target).tobytes()
