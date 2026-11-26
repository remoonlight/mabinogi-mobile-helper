import hashlib
import json

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute.skill_nearby import execute_skill_nearby
from mm_assist.life_plan import node_inspect
from mm_assist.perception import item_sense


FIXTURES = ROOT / 'tests/fixtures/season1_gather_20261125'
CASES = [
    ('daily', '蜘蛛絲團'), ('woodcut', '雷擊樹'), ('mine', '隕鐵礦脈'),
    ('herb', '柔柔蘑菇'), ('shear', '雲絮羊'), ('harvest', '稻米'),
    ('hoe', '南瓜'), ('insect', '荒涼的昆蟲群'),
]


def frame(kind):
    return cv2.imread(str(FIXTURES / f'{kind}.png'))


@pytest.mark.parametrize(('sid', 'name'), CASES)
def test_s1_target_uses_native_rows_title_and_nearby(monkeypatch, sid, name):
    def forbidden(*args, **kwargs):
        pytest.fail('this S1 target must not rely on OCR for a location')

    monkeypatch.setattr(item_sense, 'ocr_image', forbidden)
    monkeypatch.setattr(item_sense, 'ocr_roi', forbidden)
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', forbidden)
    listing, popup = frame(f'{sid}_list'), frame(f'{sid}_detail')
    assert node_inspect._detect_gather_skill_sid(listing) == sid
    hit = item_sense.locate_skill_item(listing, skill_id=sid, name=name)
    assert hit['ok'] and 1307 <= hit['click_client'][1] < 1345
    assert node_inspect.gather_detail_for_target(popup, name) == f'gather_detail:{name}'
    nearby = item_sense.locate_find_nearby(popup, allow_ocr=False)
    assert nearby['ok']

    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **kw: {'node': f'gather_skill:{sid}', 'zh': '生活技能'})
    captures = iter([listing, listing, popup])
    clicks = []
    result = execute_skill_nearby(skill_id=sid, item_name=name, output=name,
        grab=lambda: next(captures), do_click=lambda x, y: clicks.append((x, y)) or True,
        defer_completion=True)
    assert result.status == 'action_pending' and result.action_issued
    assert clicks == [tuple(hit['click_client']), tuple(nearby['click_client'])]

    # Keep neighbouring rows/icons intact. 荒蕪 has the same icon as 荒涼.
    hidden_name = listing.copy()
    hidden_name[1307:1345, 1205:1412] = listing[1325, 1190]
    assert not item_sense.locate_skill_item(hidden_name, skill_id=sid, name=name)['ok']
    assert not item_sense.locate_skill_item(popup, skill_id=sid, name=name)['ok']
    other_sid = 'insect' if sid != 'insect' else 'woodcut'
    assert not item_sense.locate_skill_item(frame(f'{other_sid}_list'), skill_id=sid, name=name)['ok']
    assert node_inspect.gather_detail_for_target(frame(f'{other_sid}_detail'), name) != f'gather_detail:{name}'


@pytest.mark.parametrize(('sid', 'name'), CASES)
def test_s1_assets_are_exact_crops_of_registered_source(sid, name):
    icon = item_sense.item_template_path(sid, name)
    meta = json.loads(icon.with_suffix('.json').read_text(encoding='utf-8'))
    source = ROOT / meta['source_frame']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == meta['source_sha256']
    listing = frame(f'{sid}_list')
    for path, coordinates, digest in (
        (icon, meta['client_px'], meta['template_sha256']),
        (icon.with_name(meta['row_name_template']), meta['row_name_client_px'], meta['row_name_sha256']),
    ):
        template_bytes = path.read_bytes()
        assert hashlib.sha256(template_bytes).hexdigest() == digest
        x, y, w, h = (coordinates[key] for key in ('x', 'y', 'w', 'h'))
        saved = cv2.imdecode(np.frombuffer(template_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(saved, listing[y:y+h, x:x+w])
