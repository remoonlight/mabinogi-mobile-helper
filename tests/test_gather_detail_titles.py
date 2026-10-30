"""Current-run popup titles must work without OCR and reject other pages."""
import hashlib
import json

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_plan import node_inspect

CASES = [('sheep_detail_20261008.png', '捲毛羊', 'shear', '高級羊毛'),
         ('mushroom_detail_20261008.png', '咻咻蘑菇', 'herb', '咻咻蘑菇')]


def frame(filename):
    return cv2.imdecode(np.fromfile(ROOT/'tests/fixtures/process_stance'/filename,
                                   dtype=np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize('coarse_display', [False, True])
@pytest.mark.parametrize('filename,name,sid,output', CASES)
def test_real_popup_title_and_executor_without_ocr(monkeypatch, filename, name, sid, output, coarse_display):
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **kw: [])
    original = frame(filename)
    assert node_inspect._gather_detail_named(original) == f'gather_detail:{name}'
    title_reads = []
    original_title = node_inspect.gather_detail_for_target
    def read_title(image, target):
        title_reads.append(image)
        return original_title(image, target)
    monkeypatch.setattr(node_inspect, 'gather_detail_for_target', read_title)
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **kw: {'node': 'gather_detail' if coarse_display else f'gather_detail:{name}',
                                      'zh': '技能详情' if coarse_display else f'技能详情-{name}'})
    clicks, issued = [], []
    result = execute_skill_nearby(skill_id=sid, item_name=name, output=output,
        grab=lambda: original, do_click=lambda x, y: clicks.append((x, y)) or True,
        on_action_issued=lambda: issued.append(True), defer_completion=True)
    assert result.status == 'action_pending' and result.action_issued
    assert len(clicks) == 1 and issued == [True]
    # The independent original input target is inside the popup's nearby label.
    assert 680 <= clicks[0][0] <= 860 and 765 <= clicks[0][1] <= 815
    assert len(title_reads) == 1, 'one captured detail must not repeat the same title recognition'


@pytest.mark.parametrize('next_page', ['correct', 'wrong', 'black', 'exception'])
def test_nearby_wait_rechecks_title_after_every_capture_even_with_reused_buffer(monkeypatch, next_page):
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.perception import item_sense
    from mm_assist.life_execute import ui_logic
    original = frame(CASES[0][0])
    changed = frame(CASES[1][0])
    captures, clicks, nearby_calls = [], [], []
    def grab():
        captures.append(True)
        if len(captures) > 1:
            if next_page == 'exception':
                raise RuntimeError('offline recapture failed')
            if next_page == 'wrong':
                original[:] = changed
            elif next_page == 'black':
                original[:] = 0
        return original
    def nearby(*a, **k):
        nearby_calls.append(True)
        return {'ok': len(nearby_calls) > 1, 'click_client': [760, 790]}
    monkeypatch.setattr(ui_logic, 'read_inspect_record', lambda **k: {'node': 'gather_detail:捲毛羊', 'zh': '技能详情'})
    monkeypatch.setattr(ui_logic, 'skip_wait_ui_params', lambda: False)
    monkeypatch.setattr(item_sense, 'locate_find_nearby', nearby)
    monkeypatch.setattr('time.sleep', lambda *a: None)
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **k: [])
    result = execute_skill_nearby(skill_id='shear', item_name='捲毛羊', output='高級羊毛',
        grab=grab, do_click=lambda x,y: clicks.append((x,y)) or True, defer_completion=True)
    assert len(captures) == 2
    assert len(clicks) == (1 if next_page == 'correct' else 0)
    assert result.status == ('action_pending' if next_page == 'correct' else
                             'error' if next_page == 'exception' else 'wrong_gather_detail')


@pytest.mark.parametrize('filename,name,sid,output', CASES)
def test_registered_original_crop_and_masked_title(monkeypatch, filename, name, sid, output):
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **kw: [])
    catalog = json.loads((ROOT/'knowledge/templates/gather_detail/title_catalog.json').read_text(encoding='utf-8'))
    row = next(r for r in catalog['titles'] if r['name'] == name)
    original = frame(filename)
    assert hashlib.sha256((ROOT/row['source_frame']).read_bytes()).hexdigest() == row['source_sha256']
    path = ROOT/row['template']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['template_sha256']
    crop = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    x, y, w, h = row['crop_client_px']
    assert np.array_equal(crop, original[y:y+h, x:x+w])
    absent = original.copy()
    absent[y:y+h, x:x+w] = 0
    assert node_inspect._gather_detail_named(absent) == 'gather_detail'
    # The same material in the dark underlying list cannot name this popup.
    assert node_inspect._gather_detail_named(np.zeros_like(original)) == ''


@pytest.mark.parametrize('filename,name,sid,output', CASES)
def test_wrong_actual_popup_never_dispatches(monkeypatch, filename, name, sid, output):
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **kw: [])
    other = frame(CASES[1 if sid == 'shear' else 0][0])
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **kw: {'node': f'gather_detail:{name}', 'zh': f'技能详情-{name}'})
    clicks = []
    result = execute_skill_nearby(skill_id=sid, item_name=name, output=output,
        grab=lambda: other, do_click=lambda x, y: clicks.append((x, y)) or True,
        defer_completion=True)
    assert result.status == 'wrong_gather_detail' and not result.action_issued and not clicks


def test_old_wrong_flower_still_rejected_without_ocr(monkeypatch):
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **kw: [])
    original = cv2.imdecode(np.fromfile(ROOT/'tests/fixtures/ui_benchmark/herb_detail_scar_flower.png',
                                        dtype=np.uint8), cv2.IMREAD_COLOR)
    assert node_inspect._gather_detail_named(original) not in {'gather_detail:捲毛羊', 'gather_detail:咻咻蘑菇'}


@pytest.mark.parametrize('threshold', [0.8, float('nan'), 1.01, None, 'invalid'])
def test_bad_threshold_cannot_relax_title_match(monkeypatch, threshold):
    catalog = json.loads((ROOT/'knowledge/templates/gather_detail/title_catalog.json').read_text(encoding='utf-8'))
    monkeypatch.setattr(node_inspect, '_gather_detail_title_catalog', lambda: {**catalog, 'threshold': threshold})
    assert node_inspect._gather_detail_title_name(frame(CASES[0][0])) == ''


@pytest.mark.parametrize('filename,name,sid,output', CASES)
def test_list_to_popup_chain_uses_only_images(monkeypatch, filename, name, sid, output):
    from mm_assist.perception import item_sense
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    def forbidden(*a, **kw):
        pytest.fail('registered list/title/nearby paths must never use OCR')
    monkeypatch.setattr(item_sense, 'ocr_image', forbidden)
    monkeypatch.setattr(item_sense, 'ocr_roi', forbidden)
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', forbidden)
    list_name = 'shear_list_updated_20261005.png' if sid == 'shear' else 'herb_list_updated_20261006.png'
    listing, popup = frame(list_name), frame(filename)
    hit = item_sense.locate_skill_item(listing, skill_id=sid, name=name)
    assert hit['ok'] and hit['src'] == 'template+name_template'
    captures = iter([listing, listing, popup])
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **kw: {'node': f'gather_skill:{sid}', 'zh': '生活技能'})
    clicks = []
    result = execute_skill_nearby(skill_id=sid, item_name=name, output=output,
        grab=lambda: next(captures), do_click=lambda x, y: clicks.append((x, y)) or True,
        defer_completion=True)
    assert result.status == 'action_pending' and result.action_issued
    assert len(clicks) == 2 and clicks[0] == tuple(hit['click_client'])
    absent = listing.copy()
    absent[:, 1190:1420] = 0
    assert not item_sense.locate_skill_item(absent, skill_id=sid, name=name)['ok']
    # Missing pixels cannot fall back to a plausible OCR title or button.
    assert node_inspect.gather_detail_for_target(np.zeros_like(popup), name) == ''
    assert not item_sense.locate_find_nearby(np.zeros_like(popup), allow_ocr=False)['ok']


def test_sheep_list_name_is_exact_registered_crop():
    path = ROOT/'knowledge/templates/items/shear/捲毛羊_name.png'
    row = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/row['source_frame']).read_bytes()).hexdigest() == row['source_sha256']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['template_sha256']
    original = frame('shear_list_updated_20261005.png')
    x, y, w, h = row['crop_client_px']
    assert np.array_equal(cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR),
                          original[y:y+h, x:x+w])
