"""Current cloth title must not be identified from its shared 加工 suffix."""
from types import SimpleNamespace
import hashlib
import json
import time

import cv2
import pytest
from PIL import Image

from mm_assist.config import ROOT
from mm_assist.perception import process_sense as sense


def frame(name):
    return cv2.imread(str(ROOT/'tests/fixtures/process_stance'/(name+'.png')))


def test_current_cloth_title_is_not_wood_in_shared_facility_stack(monkeypatch):
    from mm_assist.life_plan.node_inspect import compute_inspect
    monkeypatch.setattr(sense,'_ocr_lines',lambda *a,**k:pytest.fail('registered title uses image'))
    image = frame('cloth_title_20261020')
    assert sense._template_title_facility(image)=='cloth'
    view = SimpleNamespace(ui_stack_names=['CraftingHubUIState','TutorialTooltipUIState',
        'AlteringFacilityPopupUIInstance'],ui_stack_ready=True,ui_stack_empty_ok=False,
        ui_stack_ts=time.time(),ui_stack_error='')
    result = compute_inspect(image,view=view)
    assert result['node'].startswith('facility:cloth')


def test_shared_suffix_without_complete_facility_name_is_not_identity():
    image = frame('cloth_title_20261020')
    image[25:90,28:135] = 0
    assert sense._template_title_facility(image) is None


@pytest.mark.parametrize('facility',['metal','wood','medicine','cloth'])
def test_existing_standalone_facility_title_stays_independent(facility):
    assert sense._template_title_facility(frame(facility+'_standalone'))==facility


def test_current_cloth_title_crop_is_registered_and_lossless():
    folder = ROOT/'knowledge/templates/facility'
    row = json.loads((folder/'title_cloth_20261020.json').read_text(encoding='utf-8'))
    source = ROOT/row['source_frame']
    crop = folder/row['file']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
    assert hashlib.sha256(crop.read_bytes()).hexdigest()==row['template_sha256']
    x,y,w,h = row['crop_client_px']
    assert Image.open(source).crop((x,y,x+w,y+h)).tobytes()==Image.open(crop).tobytes()
