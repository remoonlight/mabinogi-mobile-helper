"""Current-run list failures must use independent image evidence, without OCR."""
import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_plan import node_inspect
from mm_assist.perception import item_sense
from mm_assist.life_execute.skill_nearby import execute_skill_nearby


def frame(name):
    return cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/process_stance' / name,
                                   dtype=np.uint8), cv2.IMREAD_COLOR)


@pytest.fixture(autouse=True)
def no_ocr(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('registered image-only target must not use OCR')
    monkeypatch.setattr(item_sense, '_ocr_lines_in_item_list', forbidden)
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', forbidden)
    monkeypatch.setattr('mm_assist.life_execute.skill_nearby.time.sleep', lambda _: None)


def test_current_visible_sheep_row_reaches_verified_popup_and_nearby(monkeypatch):
    listing = frame('shear_list_failed_20261008.png')
    hit = item_sense.locate_skill_item(listing, skill_id='shear', name='捲毛羊')
    assert hit['ok']
    assert 1080 <= hit['click_client'][0] <= 1330
    assert 1030 <= hit['click_client'][1] <= 1140
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **kw: {'node': 'gather_skill:shear', 'zh': '生活技能'})
    captures = iter([listing, listing, frame('sheep_detail_20261008.png')])
    clicks = []
    result = execute_skill_nearby(skill_id='shear', item_name='捲毛羊', output='高級羊毛',
        grab=lambda: next(captures), do_click=lambda x, y: clicks.append((x, y)) or True,
        defer_completion=True)
    assert result.status == 'action_pending' and result.action_issued
    assert len(clicks) == 2 and clicks[0] == tuple(hit['click_client'])
    hidden = listing.copy()
    hidden[1030:1145, 1190:1360] = 0
    assert not item_sense.locate_skill_item(hidden, skill_id='shear', name='捲毛羊')['ok']


def test_current_mushroom_below_viewport_has_image_scroll_evidence(monkeypatch):
    listing = frame('herb_list_failed_20261008.png')
    miss = item_sense.locate_skill_item(listing, skill_id='herb', name='咻咻蘑菇')
    assert not miss['ok'] and 'click_client' not in miss
    assert miss['scroll_plan']['direction'] == 'down'
    assert '藥草' in miss['scroll_plan']['visible_names']
    visible = frame('herb_list_updated_20261006.png')
    captures = iter([listing, visible, visible, frame('mushroom_detail_20261008.png')])
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **kw: {'node': 'gather_skill:herb', 'zh': '生活技能'})
    clicks = []
    result = execute_skill_nearby(skill_id='herb', item_name='咻咻蘑菇', output='咻咻蘑菇',
        grab=lambda: next(captures), do_click=lambda x, y: clicks.append((x, y)) or True,
        defer_completion=True)
    assert result.status == 'action_pending' and result.action_issued
    assert len(clicks) == 2
    assert 735 <= clicks[0][1] <= 790


def test_image_only_miss_without_full_row_anchor_does_not_scroll_or_click(monkeypatch):
    listing = frame('herb_list_failed_20261008.png')
    listing[:, 1000:] = 0
    miss = item_sense.locate_skill_item(listing, skill_id='herb', name='咻咻蘑菇')
    assert not miss['ok'] and 'click_client' not in miss
    assert (miss.get('scroll_plan') or {}).get('direction') != 'down'
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **kw: {'node': 'gather_skill:herb', 'zh': '生活技能'})
    clicks = []
    result = execute_skill_nearby(skill_id='herb', item_name='咻咻蘑菇', output='咻咻蘑菇',
        grab=lambda: listing, do_click=lambda x, y: clicks.append((x, y)) or True,
        defer_completion=True)
    assert result.status == 'item_not_visible' and not result.action_issued
    assert clicks == []


def test_next_viewport_without_image_evidence_stops_before_second_drag(monkeypatch):
    listing = frame('herb_list_failed_20261008.png')
    empty = np.zeros_like(listing)
    captures = iter([listing, empty])
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **kw: {'node': 'gather_skill:herb', 'zh': '生活技能'})
    clicks, progress = [], []
    result = execute_skill_nearby(skill_id='herb', item_name='咻咻蘑菇', output='咻咻蘑菇',
        grab=lambda: next(captures), do_click=lambda x, y: clicks.append((x, y)) or True,
        on_progress=progress.append, defer_completion=True)
    assert result.status == 'item_not_visible' and not result.action_issued
    assert progress.count('下翻详情物') == 1
    assert clicks == []
