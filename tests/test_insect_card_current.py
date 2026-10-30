"""2026-10-20 material card failed after the verified recipe/material clicks."""
from types import SimpleNamespace
import hashlib

import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import insect_obtain as flow, ui_act
from mm_assist.perception import insect_obtain_sense as sense
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests


def pixels(name='firefly_card_20261020'):
    return cv2.imread(str(ROOT / 'tests/fixtures/insect_obtain' / (name+'.png')))


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    flow.reset_insect_visits_for_tests()
    reset_stage_gates_for_tests()
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    yield
    flow.reset_insect_visits_for_tests()
    reset_stage_gates_for_tests()


def test_failed_original_is_target_card_with_obtain_button():
    frame = pixels()
    assert sense.phase(frame, '滴答螢火蟲') == 'card'
    xy = sense.locate(frame, 'obtain_button')
    assert xy is not None and 680 < xy[0] < 810 and 950 < xy[1] < 990
    assert sense.phase(frame, '獨角仙') == ''


def test_actual_recipe_to_failed_card_verifies_material_then_only_clicks_method():
    ticks, state, clicks = [1000.0], [0], []
    frames = [pixels('firefly_recipe_20261014'), pixels()]
    names = ('HudUIStepState', 'CraftingHubUIState', 'TutorialTooltipUIState',
             'CraftingHubPopupUIInstance', 'CraftingFacilityPopupUIInstance',
             'CraftingPanelPopupUIInstance')

    def resolve(frame):
        return SimpleNamespace(node='craft_shell' if state[0] == 0 else 'item_info',
            stack_ready=True, stack_sampled_ts=ticks[0]+.001,
            stack_names=names if state[0] == 0 else names+('DetailCardUIState','DetailCardUIInstance'))

    def click(*xy):
        clicks.append(xy)
        if state[0] == 0:
            state[0] = 1
            return True
        return False  # Stop before method dispatch; no source/gameplay action.

    result = flow.execute_insect_obtain(mat='滴答螢火蟲', grab=lambda:frames[state[0]],
        do_click=click, resolve=resolve, now_fn=lambda:ticks[0],
        sleep_fn=lambda n:ticks.__setitem__(0,ticks[0]+n))
    assert clicks == [(1530,950), sense.locate(frames[1], 'obtain_button')]
    assert '虫料入口:滴答螢火蟲:滴答螢火蟲' not in flow._entry_after
    assert not result.ok and not result.action_issued and not flow._visits


@pytest.mark.parametrize('bad', ['hidden_method','hidden_title','wrong_material','stale','unready'])
def test_missing_independent_card_proof_never_clicks(bad):
    frame = pixels()
    if bad == 'hidden_method':
        frame[945:995,675:820] = 0
    if bad == 'hidden_title':
        frame[445:500,670:900] = 0
    clicks = []
    result = flow.execute_insect_obtain(mat='獨角仙' if bad=='wrong_material' else '滴答螢火蟲',
        grab=lambda:frame, do_click=lambda *xy:clicks.append(xy) or False,
        resolve=lambda _:SimpleNamespace(node='item_info',stack_ready=bad!='unready',
            stack_names=('DetailCardUIState','DetailCardUIInstance'),
            stack_sampled_ts=999 if bad=='stale' else 1000.001),
        now_fn=lambda:1000, sleep_fn=lambda n:None)
    assert not result.ok and not result.action_issued and not clicks


def test_current_button_variant_is_lossless_registered_crop_with_old_thresholds():
    from PIL import Image
    row = sense._catalog(ROOT)['rows']['obtain_button']
    assert row['threshold'] == .94 and row['min_margin'] == .04
    variant = next(v for v in row['variants'] if v['file']=='obtain_button_20261020.png')
    assert variant['threshold'] == row['threshold'] and variant['min_margin'] == row['min_margin']
    source = ROOT / variant['source_frame']
    template = ROOT / 'knowledge/templates/insect_obtain' / variant['file']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == variant['source_sha256']
    assert hashlib.sha256(template.read_bytes()).hexdigest() == variant['template_sha256']
    x,y,w,h = variant['crop_client_px']
    assert Image.open(source).crop((x,y,x+w,y+h)).tobytes() == Image.open(template).tobytes()


@pytest.mark.parametrize('mat,name,node', [
    ('滴答螢火蟲','make_items_20261020','craft_make'),
    ('滴答螢火蟲','../process_stance/craft_process_tab_failed_20261014','craft_items'),
    ('獨角仙','../process_stance/craft_process_tab_failed_20261014','craft_items'),
])
def test_registered_entry_labels_use_images_before_ocr(monkeypatch,mat,name,node):
    from mm_assist.perception import ocr
    monkeypatch.setattr(ocr,'ocr_roi',lambda *a,**k:pytest.fail('image hit must skip OCR'))
    clicks = []
    ticks = [1000.0]
    result = flow.execute_insect_obtain(mat=mat,grab=lambda:pixels(name),
        do_click=lambda *xy:clicks.append(xy) or False,
        resolve=lambda _:SimpleNamespace(node=node,stack_ready=True,
            stack_names=('CraftingHubPopupUIInstance','CraftingFacilityPopupUIInstance'),
            stack_sampled_ts=ticks[0]+.001),
        now_fn=lambda:ticks[0],sleep_fn=lambda n:ticks.__setitem__(0,ticks[0]+n))
    assert len(clicks) == 1 and not result.ok and not result.action_issued


def test_label_image_miss_falls_back_to_ocr_after_independent_items_proof(monkeypatch):
    from mm_assist.perception import ocr
    frame = pixels('../process_stance/craft_process_tab_failed_20261014')
    frame[1267:1307,1587:1745] = 0
    calls, clicks, ticks = [], [], [1000.0]
    original = sense.locate

    def locate(image,key,**kw):
        if key=='firefly_potion_label':
            calls.append('image')
        return original(image,key,**kw)

    def labels(*a,**kw):
        calls.append('ocr')
        return [ocr.OcrLine('連擊強化秘藥',.99,[[1590,1270],[1740,1270],[1740,1300],[1590,1300]])]

    monkeypatch.setattr(sense,'locate',locate)
    monkeypatch.setattr(ocr,'ocr_roi',labels)
    result = flow.execute_insect_obtain(mat='滴答螢火蟲',grab=lambda:frame.copy(),
        do_click=lambda *xy:clicks.append(xy) or False,
        resolve=lambda _:SimpleNamespace(node='craft_items',stack_ready=True,
            stack_names=('CraftingHubPopupUIInstance','CraftingFacilityPopupUIInstance'),
            stack_sampled_ts=ticks[0]+.001),
        now_fn=lambda:ticks[0],sleep_fn=lambda n:ticks.__setitem__(0,ticks[0]+n))
    assert calls == ['image','ocr','image','ocr']
    assert clicks == [(1665,1285)] and not result.ok and not result.action_issued


@pytest.mark.parametrize('mat,key', [('滴答螢火蟲','firefly_potion_label'),
                                    ('獨角仙','beetle_potion_label')])
def test_current_bottom_row_is_located_before_any_scroll_or_ocr(monkeypatch,mat,key):
    from mm_assist.perception import ocr
    monkeypatch.setattr(ocr,'ocr_roi',lambda *a,**kw:pytest.fail('visible full label uses image'))
    ticks,clicks,drags = [1000.0],[],[]
    frame = pixels('items_bottom_20261020')
    assert sense.phase(frame,mat)=='items'
    result = flow.execute_insect_obtain(mat=mat,grab=lambda:frame.copy(),
        do_click=lambda *xy:clicks.append(xy) or False,
        do_drag=lambda *a,**kw:drags.append(a) or False,
        resolve=lambda _:SimpleNamespace(node='craft_items',stack_ready=True,
            stack_names=('HudUIStepState','CraftingHubUIState','TutorialTooltipUIState',
                         'CraftingHubPopupUIInstance','CraftingFacilityPopupUIInstance'),
            stack_sampled_ts=ticks[0]+.001),
        now_fn=lambda:ticks[0],sleep_fn=lambda n:ticks.__setitem__(0,ticks[0]+n))
    assert len(clicks)==1 and 1360<clicks[0][1]<1405 and not drags
    assert not result.ok and not result.action_issued


def test_partially_hidden_bottom_label_is_not_image_evidence():
    frame = pixels('items_bottom_20261020')
    frame[1378:,1580:1750] = 0
    assert sense.locate(frame,'firefly_potion_label',
        roi=sense.region('recipe_labels')) is None


@pytest.mark.parametrize('prefix',['firefly','beetle'])
def test_bottom_variant_has_exact_registered_pixels_and_unchanged_threshold(prefix):
    from PIL import Image
    row = sense._catalog(ROOT)['rows'][prefix+'_potion_label']
    variant = row['variants'][-1]
    source = ROOT/variant['source_frame']
    crop = ROOT/'knowledge/templates/insect_obtain'/variant['file']
    assert row['threshold']==variant['threshold']==.94
    assert row['min_margin']==variant['min_margin']==.04
    assert hashlib.sha256(source.read_bytes()).hexdigest()==variant['source_sha256']
    assert hashlib.sha256(crop.read_bytes()).hexdigest()==variant['template_sha256']
    x,y,w,h = variant['crop_client_px']
    assert Image.open(source).crop((x,y,x+w,y+h)).tobytes()==Image.open(crop).tobytes()


def test_transient_stack_read_waits_for_new_frame_before_source_dispatch():
    ticks,reads,clicks = [1000.0],[0],[]
    frame = pixels('firefly_obtain_20261020')
    assert sense.phase(frame,'滴答螢火蟲')=='obtain'

    def resolve(_):
        reads[0] += 1
        ready = reads[0]>1
        return SimpleNamespace(node='obtain_method' if ready else 'unknown',
            stack_ready=ready,stack_sampled_ts=ticks[0]+.001 if ready else 0,
            stack_names=('DetailCardUIInstance','HowToGetSelectUIInstance'))

    result = flow.execute_insect_obtain(mat='滴答螢火蟲',grab=lambda:frame.copy(),
        do_click=lambda *xy:clicks.append(xy) or True,resolve=resolve,
        now_fn=lambda:ticks[0],sleep_fn=lambda n:ticks.__setitem__(0,ticks[0]+n),
        defer_completion=True)
    assert reads[0]>=2 and len(clicks)==1
    assert result.status=='action_pending' and result.action_issued


@pytest.mark.parametrize('mode',['stale','unready','pause'])
def test_read_wait_never_dispatches_from_unproven_or_cancelled_stack(monkeypatch,mode):
    ticks,reads,clicks = [1000.0],[0],[]
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:mode=='pause' and reads[0]>1)
    stage = '虫料入口:滴答螢火蟲:獲得方法'
    flow._entry_after[stage] = ('obtain',999)
    flow._entry_attempts[stage] = 2

    def resolve(_):
        reads[0]+=1
        ready = mode=='stale' or (mode=='pause' and reads[0]>1)
        return SimpleNamespace(node='obtain_method',stack_ready=ready,
            stack_sampled_ts=(ticks[0]+.001 if mode=='pause' else 999),
            stack_names=('HowToGetSelectUIInstance',))

    result = flow.execute_insect_obtain(mat='滴答螢火蟲',grab=lambda:pixels('firefly_obtain_20261020'),
        do_click=lambda *xy:clicks.append(xy) or True,resolve=resolve,
        now_fn=lambda:ticks[0],sleep_fn=lambda n:ticks.__setitem__(0,ticks[0]+n))
    assert not clicks and not result.action_issued
    assert flow._entry_after[stage]==('obtain',999) and flow._entry_attempts[stage]==2
    assert result.status==('cancel' if mode=='pause' else 'stp_busy')
    assert reads[0]>1 and ticks[0]-1000<=6.5


@pytest.mark.parametrize('change',['pause','new_run','exception'])
def test_late_stack_observation_cannot_clear_old_gate_or_dispatch(monkeypatch,change):
    from mm_assist.life_plan import controller
    owner = SimpleNamespace(running=True,preparing=False,life_started_at=90,current='old')
    state = [owner]
    monkeypatch.setattr(controller,'get_controller',lambda:state[0])
    stage = '虫料入口:滴答螢火蟲:獲得方法'
    flow._entry_after[stage] = ('obtain',999)
    flow._entry_attempts[stage] = 2
    clicks = []

    def resolve(_):
        if change=='exception':
            raise RuntimeError('read failed')
        state[0] = SimpleNamespace(running=change!='pause',preparing=False,
            life_started_at=91 if change=='new_run' else 90,current='old')
        return SimpleNamespace(node='obtain_method',stack_ready=True,stack_sampled_ts=1000.001)

    result = flow.execute_insect_obtain(mat='滴答螢火蟲',grab=lambda:pixels('firefly_obtain_20261020'),
        do_click=lambda *xy:clicks.append(xy) or True,resolve=resolve,
        now_fn=lambda:1000,sleep_fn=lambda n:None,defer_completion=True)
    assert not clicks and not result.action_issued
    assert flow._entry_after[stage]==('obtain',999) and flow._entry_attempts[stage]==2
    assert result.status==('error' if change=='exception' else 'cancel')
    if change=='exception':
        second = flow.execute_insect_obtain(mat='滴答螢火蟲',
            grab=lambda:pixels('firefly_obtain_20261020'),
            do_click=lambda *xy:clicks.append(xy) or True,
            resolve=lambda _:SimpleNamespace(node='obtain_method',stack_ready=True,
                stack_sampled_ts=1000.001),now_fn=lambda:1000,sleep_fn=lambda n:None,
            defer_completion=True)
        assert second.status=='action_pending' and second.action_issued and len(clicks)==1
