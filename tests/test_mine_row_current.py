"""2026-10-23 visible iron row: icon is good, OCR cannot authorize its name."""
import cv2
import numpy as np
import pytest
from mm_assist.config import ROOT
from mm_assist.perception import item_sense
from mm_assist.life_execute import skill_list

def pixels(path):
    frame=cv2.imdecode(np.fromfile(ROOT/path,dtype=np.uint8),1)
    assert frame is not None
    return frame

@pytest.fixture(autouse=True)
def no_ocr(monkeypatch):
    monkeypatch.setattr(item_sense,'ocr_image',lambda *a,**k:[])
    monkeypatch.setattr(item_sense,'ocr_roi',lambda *a,**k:[])
    skill_list._pending.clear()
    yield
    skill_list._pending.clear()

def test_current_iron_row_needs_independent_full_name():
    frame=pixels('tests/fixtures/life_faults_20261023/mine_list.png')
    hit=item_sense.locate_skill_item(frame,skill_id='mine',name='鐵礦脈')
    assert hit['ok'] and hit['src']=='template+name_template',hit
    result=skill_list.seek_item(frame,hit,skill_id='mine',name='鐵礦脈',
        grab=lambda:frame.copy(),drag=lambda *a:pytest.fail('visible target cannot scroll'),
        validate=lambda *a:True,sleep=lambda _:None)
    assert result[2] is None and result[1]['ok']

@pytest.mark.parametrize('path',[
    'tests/fixtures/process_stance/mine_list_updated_20261006.png',
    'tests/fixtures/process_stance/mine_list_updated_20261005.png',
])
def test_archived_iron_row_clipped_under_top_edge_remains_rejected(path):
    # These older viewports show dim iron text under the rounded popup top;
    # they are negative visibility cases, not fully exposed iron rows.
    assert not item_sense.locate_skill_item(pixels(path),skill_id='mine',name='鐵礦脈')['ok']

def test_wood_facility_does_not_provide_mine_row():
    frame=pixels('tests/fixtures/life_faults_20261023/wood_rendered.png')
    assert not item_sense.locate_skill_item(frame,skill_id='mine',name='鐵礦脈')['ok']

def test_icon_without_name_cannot_authorize_iron_row():
    frame=pixels('tests/fixtures/life_faults_20261023/mine_list.png')
    # Deliberate in-memory negative: preserve the icon, remove the name.
    frame[1040:1140,1190:1390]=0
    assert not item_sense.locate_skill_item(frame,skill_id='mine',name='鐵礦脈')['ok']
