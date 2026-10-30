"""Coal prep must accept the actual popup and retain independent input gates."""
import hashlib
import json

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_plan import node_inspect


def load(path):
    return cv2.imdecode(np.fromfile(ROOT / path, dtype=np.uint8), cv2.IMREAD_COLOR)


CURRENT = 'tests/fixtures/process_stance/coal_detail_20261028.png'
OLD = 'tests/fixtures/life_faults_20261024/coal_detail.png'


@pytest.mark.parametrize('path', [CURRENT, OLD])
@pytest.mark.parametrize('display_node', ['gather_detail', 'gather_detail:煤炭礦脈', 'gather_skill:mine'])
def test_coal_actual_popup_reaches_nearby_without_ocr(monkeypatch, path, display_node):
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.perception import item_sense

    def forbidden(*a, **k):
        pytest.fail('registered coal detail and nearby must use independent images')

    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', forbidden)
    monkeypatch.setattr(item_sense, 'ocr_image', forbidden)
    monkeypatch.setattr(item_sense, 'ocr_roi', forbidden)
    image = load(path)
    assert node_inspect.gather_detail_for_target(image, '煤炭礦脈') == 'gather_detail:煤炭礦脈'
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **k: {'node': display_node, 'zh': '技能详情'})
    clicks, issued = [], []
    result = execute_skill_nearby(skill_id='mine', item_name='煤炭礦脈', output='煤炭',
        grab=lambda: image, do_click=lambda x, y: clicks.append((x, y)) or True,
        on_action_issued=lambda: issued.append(True), defer_completion=True)
    assert result.status == 'action_pending' and result.action_issued
    assert issued == [True] and len(clicks) == 1
    assert 679 <= clicks[0][0] <= 830 and 783 <= clicks[0][1] <= 820


def test_current_coal_crop_registration():
    catalog = json.loads((ROOT / 'knowledge/templates/gather_detail/title_catalog.json').read_text(encoding='utf-8'))
    row = next(r for r in catalog['titles'] if r.get('source_frame') == CURRENT)
    source = ROOT / CURRENT
    template = ROOT / row['template']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_sha256']
    assert hashlib.sha256(template.read_bytes()).hexdigest() == row['template_sha256']
    x, y, w, h = row['crop_client_px']
    assert np.array_equal(load(row['template']), load(CURRENT)[y:y+h, x:x+w])
    assert catalog['threshold'] == .97


@pytest.mark.parametrize('wrong', ['title_missing', 'nearby_missing', 'black', 'iron', 'old_wrong_flower'])
def test_wrong_or_incomplete_coal_popup_never_clicks(monkeypatch, wrong):
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    image = load(CURRENT)
    if wrong == 'title_missing':
        image[370:460, 640:860] = 0
    elif wrong == 'nearby_missing':
        image[775:830, 640:880] = 0
    elif wrong == 'black':
        image[:] = 0
    elif wrong == 'iron':
        image = load('tests/fixtures/process_stance/iron_detail_failed_20261010.png')
    else:
        image = load('tests/fixtures/ui_benchmark/herb_detail_scar_flower.png')
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **k: [])
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **k: {'node': 'gather_detail:煤炭礦脈', 'zh': '技能详情'})
    clicks = []
    result = execute_skill_nearby(skill_id='mine', item_name='煤炭礦脈', output='煤炭',
        grab=lambda: image, do_click=lambda x, y: clicks.append((x, y)) or True,
        defer_completion=True)
    assert not clicks and not result.action_issued and result.status == 'wrong_gather_detail'
