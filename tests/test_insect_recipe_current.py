"""The current recipe popup has its own native layer and a registered title."""
from types import SimpleNamespace
import cv2
import pytest
from mm_assist.config import ROOT
from mm_assist.life_execute import insect_obtain as flow,ui_act
from mm_assist.life_plan.types import ExecuteResult
from mm_assist.perception import insect_obtain_sense as sense
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

NAMES=('CraftingHubPopupUIInstance','CraftingFacilityPopupUIInstance','CraftingPanelPopupUIInstance')

def pixels():return cv2.imread(str(ROOT/'tests/fixtures/insect_obtain/firefly_recipe_20261014.png'))


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    flow.reset_insect_visits_for_tests();reset_stage_gates_for_tests()
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:False)
    yield
    flow.reset_insect_visits_for_tests();reset_stage_gates_for_tests()


def test_current_title_registration_does_not_lower_old_threshold():
    import hashlib
    from PIL import Image
    assert sense.phase(pixels(),'滴答螢火蟲')=='recipe'
    assert sense.phase(pixels(),'獨角仙')==''
    row=sense._catalog(ROOT)['rows']['firefly_recipe_title']
    assert row['threshold']==.98 and row['variants'][0]['threshold']==.98
    variant=row['variants'][0]
    source=ROOT/variant['source_frame']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==variant['source_sha256']
    crop=ROOT/'knowledge/templates/insect_obtain'/variant['file']
    assert hashlib.sha256(crop.read_bytes()).hexdigest()==variant['template_sha256']
    x,y,w,h=variant['crop_client_px']
    assert Image.open(source).crop((x,y,x+w,y+h)).tobytes()==Image.open(crop).tobytes()


def test_open_current_recipe_clicks_only_material_without_reopening_potion(monkeypatch):
    ticks=[1000.0];clicks=[]
    # Stop at the injected material click; no subsequent input is needed to
    # prove that the already-open current popup bypasses the potion list.
    monkeypatch.setattr(ui_act,'prepare_to',lambda *a,**k:pytest.fail('already at the target recipe'))
    result=flow.execute_insect_obtain(mat='滴答螢火蟲',grab=pixels,
        do_click=lambda *xy:clicks.append(xy) or False,
        resolve=lambda _:SimpleNamespace(node='craft_shell',stack_ready=True,
            stack_names=NAMES,stack_sampled_ts=ticks[0]+.001),
        now_fn=lambda:ticks[0],sleep_fn=lambda n:ticks.__setitem__(0,ticks[0]+n))
    assert clicks==[(1530,950)] and not result.ok and not result.action_issued


@pytest.mark.parametrize('bad',['covered','missing_panel','title_hidden'])
def test_unknown_or_wrong_popup_cannot_click_its_material(monkeypatch,bad):
    frame=pixels();names=NAMES
    if bad=='covered':names=NAMES+('UnknownPopup',)
    if bad=='missing_panel':names=NAMES[:-1]
    if bad=='title_hidden':frame[620:680,1220:1470]=0
    monkeypatch.setattr(ui_act,'prepare_to',lambda *a,**k:ExecuteResult(False,'not_current_recipe'))
    clicks=[]
    result=flow.execute_insect_obtain(mat='滴答螢火蟲',grab=lambda:frame,
        do_click=lambda *xy:clicks.append(xy) or False,
        resolve=lambda _:SimpleNamespace(node='craft_shell',stack_ready=True,
            stack_names=names,stack_sampled_ts=1000.001),now_fn=lambda:1000,sleep_fn=lambda n:None)
    assert not result.ok and not clicks and not result.action_issued
