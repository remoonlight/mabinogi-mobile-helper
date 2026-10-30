"""Retained 31232/run1791621986.6935394 failures, with OCR disabled."""
import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_plan import node_inspect
from mm_assist.perception import item_sense


def frame(name):
    return cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/process_stance' / name,
                                   dtype=np.uint8), cv2.IMREAD_COLOR)


def test_iron_detail_is_independently_named_without_ocr(monkeypatch):
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **k: [])
    assert node_inspect.gather_detail_for_target(frame('iron_detail_failed_20261010.png'), '鐵礦脈') == 'gather_detail:鐵礦脈'


def test_iron_executor_accepts_real_title_and_rejects_masked_or_wrong_popup(monkeypatch):
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **k: pytest.fail('OCR'))
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
        lambda **kw: {'node': 'gather_detail', 'zh': '技能详情'})
    original = frame('iron_detail_failed_20261010.png')
    masked = original.copy()
    masked[393:445, 677:805] = 0
    for image, expected in [(original, 'action_pending'), (masked, 'wrong_gather_detail'),
                            (frame('sheep_detail_20261008.png'), 'wrong_gather_detail')]:
        clicks = []
        result = execute_skill_nearby(skill_id='mine', item_name='鐵礦脈', output='鐵礦石',
            grab=lambda: image, do_click=lambda x, y: clicks.append((x, y)) or True,
            defer_completion=True)
        assert result.status == expected
        assert len(clicks) == (1 if expected == 'action_pending' else 0)


def test_latest_list_to_popup_uses_two_stable_image_observations(monkeypatch):
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **k: pytest.fail('OCR'))
    monkeypatch.setattr(item_sense, '_ocr_lines_in_item_list', lambda *a, **k: pytest.fail('OCR'))
    monkeypatch.setattr('mm_assist.life_execute.skill_nearby.time.sleep', lambda _: None)
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
        lambda **kw: {'node': 'gather_skill:herb', 'zh': '生活技能'})
    listing = frame('herb_middle_failed_20261010.png')
    captures = iter([listing, listing.copy(), frame('mushroom_detail_20261008.png')])
    clicks = []
    result = execute_skill_nearby(skill_id='herb', item_name='咻咻蘑菇', output='咻咻蘑菇',
        grab=lambda: next(captures), do_click=lambda x, y: clicks.append((x, y)) or True,
        defer_completion=True)
    assert result.status == 'action_pending' and len(clicks) == 2
    assert 1116 <= clicks[0][1] <= 1216


def test_new_registered_crops_keep_originals_and_exact_pixels():
    import hashlib
    import json
    rows = [r for r in node_inspect._gather_detail_title_catalog()['titles'] if r['name'] == '鐵礦脈']
    for name in ['新芽蘑菇', '壯壯蘑菇', '毅力草', '咻咻蘑菇']:
        path = ROOT / f'knowledge/templates/gather_skill/herb/{name}.png'
        rows.append(dict(template=str(path), **json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))))
    for row in rows:
        source, path = ROOT / row['source_frame'], ROOT / row['template']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_sha256']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['template_sha256']
        original = cv2.imdecode(np.fromfile(source, dtype=np.uint8), 1)
        x, y, w, h = row['crop_client_px']
        assert np.array_equal(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1), original[y:y+h, x:x+w])


@pytest.mark.parametrize('scale', [1, .5])
def test_configured_list_drag_keeps_existing_endpoints_and_short_second_step(scale):
    image = cv2.resize(frame('herb_middle_failed_20261010.png'), None, fx=scale, fy=scale)
    assert item_sense.item_list_scroll_drag_points(image) == (
        (int(1497.6 * scale), int(1120.32 * scale)), (int(1497.6 * scale), int(672.768 * scale)))
    assert item_sense.item_list_scroll_drag_points(image, step=1) == (
        (int(1497.6 * scale), int(1120.32 * scale)), (int(1497.6 * scale), int(874.1664 * scale)))


def test_middle_herb_viewport_locates_visible_mushroom_without_ocr(monkeypatch):
    monkeypatch.setattr(item_sense, '_ocr_lines_in_item_list', lambda *a, **k: pytest.fail('OCR'))
    hit = item_sense.locate_skill_item(frame('herb_middle_failed_20261010.png'), skill_id='herb', name='咻咻蘑菇')
    assert hit['ok'] and hit['src'] == 'template+row_anchor'
    assert 1116 <= hit['click_client'][1] <= 1216


def test_herb_middle_anchors_cannot_authorize_scroll_when_masked(monkeypatch):
    monkeypatch.setattr(item_sense, '_ocr_lines_in_item_list', lambda *a, **k: pytest.fail('OCR'))
    image = frame('herb_middle_failed_20261010.png')
    image[:, 1190:1450] = 0
    miss = item_sense.locate_skill_item(image, skill_id='herb', name='咻咻蘑菇')
    assert not miss['ok'] and miss['scroll_plan']['direction'] is None
