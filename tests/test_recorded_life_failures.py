"""Failures retained from watch 41344, life run 1791712040.511027."""
import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_plan import node_inspect
from mm_assist.perception import insect_obtain_sense as insects, process_sense


def frame(name):
    return cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/process_stance' / name,
                                    dtype=np.uint8), cv2.IMREAD_COLOR)


def no_ocr(*a, **kw):
    return []


def test_copper_real_detail_without_ocr_dispatches_only_nearby(monkeypatch):
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', no_ocr)
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **kw: {'node': 'gather_detail', 'zh': '技能详情'})
    clicks = []
    result = execute_skill_nearby(skill_id='mine', item_name='銅礦脈', output='銅礦石',
        grab=lambda: frame('copper_detail_20261011.png'),
        do_click=lambda x, y: clicks.append((x, y)) or True, defer_completion=True)
    assert result.status == 'action_pending' and result.action_issued
    assert len(clicks) == 1 and 680 <= clicks[0][0] <= 860 and 765 <= clicks[0][1] <= 815


@pytest.mark.parametrize('name', ['iron_detail_failed_20261010.png', 'mushroom_detail_20261008.png'])
def test_copper_rejects_other_registered_popups(monkeypatch, name):
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', no_ocr)
    assert node_inspect.gather_detail_for_target(frame(name), '銅礦脈') != 'gather_detail:銅礦脈'
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **kw: {'node': 'gather_detail:銅礦脈', 'zh': '技能详情-銅礦脈'})
    clicks = []
    result = execute_skill_nearby(skill_id='mine', item_name='銅礦脈', output='銅礦石',
        grab=lambda: frame(name), do_click=lambda x, y: clicks.append((x, y)) or True,
        defer_completion=True)
    assert result.status == 'wrong_gather_detail' and not result.action_issued and not clicks


def test_copper_title_crop_provenance_and_hidden_heading(monkeypatch):
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', no_ocr)
    config = json.loads((ROOT / 'knowledge/templates/gather_detail/title_catalog.json').read_text(encoding='utf-8'))
    row = next(r for r in config['titles'] if r['name'] == '銅礦脈')
    original = frame('copper_detail_20261011.png')
    path = ROOT / row['template']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['template_sha256']
    assert hashlib.sha256((ROOT / row['source_frame']).read_bytes()).hexdigest() == row['source_sha256']
    x, y, w, h = row['crop_client_px']
    assert np.array_equal(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1), original[y:y+h, x:x+w])
    original[y:y+h, x:x+w] = 0
    assert node_inspect.gather_detail_for_target(original, '銅礦脈') != 'gather_detail:銅礦脈'


def test_current_items_page_and_blank_wrong_size_rejected():
    original = frame('insect_items_20261011.png')
    assert insects.phase(original, '滴答螢火蟲') == 'items'
    assert insects.phase(original, '獨角仙') == 'items'
    assert insects.phase(np.zeros_like(original), '滴答螢火蟲') == ''
    assert insects.phase(original[:, :-1], '滴答螢火蟲') == ''
    masked = original.copy()
    masked[:130, :450] = 0
    assert insects.phase(masked, '滴答螢火蟲') == ''


@pytest.mark.parametrize('fresh', [False, True])
def test_real_items_page_only_clears_pending_entry_on_new_stack(monkeypatch, fresh):
    from mm_assist.life_execute import insect_obtain as flow, ui_act
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests
    flow.reset_insect_visits_for_tests()
    reset_stage_gates_for_tests()
    stage = '虫料入口:滴答螢火蟲:道具'
    flow._entry_attempts[stage] = 2
    flow._entry_after[stage] = ('items', 1000.0)
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr('mm_assist.perception.ocr.ocr_roi', no_ocr)
    clicks = []
    try:
        flow.execute_insect_obtain(mat='滴答螢火蟲',
            grab=lambda: frame('insect_items_20261011.png'),
            resolve=lambda f: SimpleNamespace(node='craft_shell', stack_ready=True,
                stack_sampled_ts=1001 if fresh else 999), now_fn=lambda: 1000.5,
            do_click=lambda x, y: clicks.append((x, y)) or True)
        assert (stage not in flow._entry_attempts) == fresh
        assert not clicks  # No target potion is visible in this captured viewport.
    finally:
        flow.reset_insect_visits_for_tests()
        reset_stage_gates_for_tests()


@pytest.mark.parametrize('name', ['medicine_list_20261011.png', 'medicine_standalone.png'])
def test_visible_mushroom_juice_old_and_new_viewports_without_ocr(monkeypatch, name):
    monkeypatch.setattr(process_sense, '_ocr_lines', no_ocr)
    monkeypatch.setattr(process_sense, '_gray_from_ocr', lambda *a, **kw: False)
    result = process_sense.locate_process_recipe(frame(name), facility='medicine', name='咻咻蘑菇汁液')
    assert result['ok'] and result['src'] == 'template'
    assert 1190 <= result['click_client'][0] <= 1360 and 430 <= result['click_client'][1] <= 490


def test_mushroom_juice_masked_name_wrong_page_and_blank_never_match(monkeypatch):
    monkeypatch.setattr(process_sense, '_ocr_lines', no_ocr)
    original = frame('medicine_list_20261011.png')
    original[440:490, 1185:1355] = 0
    for image in [original, frame('insect_items_20261011.png'), np.zeros_like(original)]:
        assert not process_sense.locate_process_recipe(image, facility='medicine', name='咻咻蘑菇汁液')['ok']


def test_retained_failure_frames_have_original_hashes():
    rows = json.loads((ROOT / 'tests/fixtures/process_stance/life_records_20261011.json').read_text(encoding='utf-8'))
    for row in rows['frames']:
        assert hashlib.sha256((ROOT / row['source_frame']).read_bytes()).hexdigest() == row['source_sha256']
        assert list(frame(row['source_frame'].split('/')[-1]).shape[1::-1]) == row['source_size']
    path = ROOT / 'knowledge/templates/process/medicine/咻咻蘑菇汁液_1920.png'
    row = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['template_sha256']
    original = frame('medicine_list_20261011.png')
    assert hashlib.sha256((ROOT / row['source_frame']).read_bytes()).hexdigest() == row['source_sha256']
    x, y, w, h = row['crop_client_px']
    assert np.array_equal(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1), original[y:y+h, x:x+w])
