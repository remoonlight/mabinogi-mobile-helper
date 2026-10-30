"""Oct 5/6 catalog updates: level-gated targets and no fishing plans."""
import hashlib
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_plan import planner, controller
from mm_assist.life_plan.recipes import SKILL_GATHER
from mm_assist.life_plan.types import LifeSnapshot, PlanItem, SkillGrowth, ExecuteResult
from mm_assist.perception.item_sense import items_for_skill, locate_skill_item

TARGETS = {'合金鋼錠': 100, '高級木材': 100, '高級布料': 100, '淨淨蘑菇汁液': 150,
           '高級混合強化秘藥(山崩)': 100,
           '增幅蘑菇汁液': 150, '箭花': 250, '血紅藥草': 250, '滴答螢火蟲': 250, '獨角仙': 250}
FISH = PlanItem('old1', 'gather', '练采集 東側城牆釣魚場', skill='釣魚',
    recipe='東側城牆釣魚場', detail='東側城牆釣魚場', gather_mode='skill')
WOODCUT = PlanItem('old1', 'gather', '练采集 甲胄樹', skill='伐木',
    recipe='甲胄樹', detail='甲胄樹', gather_mode='skill')


@pytest.fixture(autouse=True)
def isolated():
    controller.reset_controller_for_tests()
    yield
    controller.reset_controller_for_tests()


def snap():
    return LifeSnapshot(ok=True, items=TARGETS, taken_at=1000)


@pytest.mark.parametrize('skill,detail', [
    ('伐木', '甲胄樹'), ('伐木', '幽暮樹'),
    ('剪羊毛', '捲毛羊'), ('剪羊毛', '烏雲羊'),
    ('採礦', '白銅礦脈'), ('採礦', '銀礦脈'),
    ('採集藥草', '淨淨蘑菇'), ('採集藥草', '傷痕花'), ('採集藥草', '增幅蘑菇'),
])
def test_excluded_automatic_training_even_as_held_lowest_skill(skill, detail):
    s = replace(snap(), skills=(SkillGrowth(skill, 26, 0),))
    held = replace(WOODCUT, skill=skill, detail=detail, recipe=detail)
    counts = {name: 10 for name, _ in SKILL_GATHER if name != skill}
    assert planner.training_item(s, counts).skill != skill
    plan = planner.plan_life(s, training_counts=counts, training_target=held)
    assert len(plan) == 10 and all(p.skill != skill for p in plan)


@pytest.mark.parametrize('operation', ['refresh', 'top_up', 'resume'])
@pytest.mark.parametrize('skill,detail,material_name,material_detail', [
    ('伐木', '甲胄樹', '原木', '樹木'),
    ('伐木', '幽暮樹', '高級原木', '成材樹'),
    ('剪羊毛', '捲毛羊', '羊毛', '羊'),
    ('剪羊毛', '烏雲羊', '高級羊毛', '捲毛羊'),
    ('採礦', '白銅礦脈', '鐵礦石', '鐵礦脈'),
    ('採礦', '銀礦脈', '煤炭', '煤炭礦脈'),
    ('採集藥草', '淨淨蘑菇', '淨淨蘑菇', '淨淨蘑菇'),
    ('採集藥草', '傷痕花', '箭花', '箭花'),
    ('採集藥草', '增幅蘑菇', '增幅蘑菇', '增幅蘑菇'),
])
def test_unissued_training_filtered_but_material_gather_kept(
        operation, skill, detail, material_name, material_detail):
    controller.start_life(snapshot=snap(), now=1000)
    state = controller._STATE
    training = replace(WOODCUT, skill=skill, detail=detail, recipe=detail)
    material = replace(training, id='old2', detail=material_detail, recipe=material_name,
                       gather_mode='process_mat')
    keep = state.queue[0]
    state.queue = [training, material, keep]
    state.current = training
    if operation == 'refresh':
        state.replan_requested = True
        controller._rebuild_waiting(state, snap())
    elif operation == 'top_up':
        controller._top_up(state, snap())
    else:
        controller.pause_life()
        controller.resume_life(now=1001)
    assert training.id not in {p.id for p in state.queue}
    assert material in state.queue and keep in state.queue
    assert all(p.skill != skill or p.gather_mode != 'skill' for p in state.queue)


@pytest.mark.parametrize('pending_kind', ['action', 'legacy_gather'])
@pytest.mark.parametrize('skill,detail', [
    ('伐木', '甲胄樹'), ('剪羊毛', '捲毛羊'),
    ('採礦', '銀礦脈'), ('採集藥草', '增幅蘑菇'),
])
def test_issued_training_remains_pinned_without_extending_batch(monkeypatch, pending_kind, skill, detail):
    controller.start_life(snapshot=snap(), now=1000)
    state = controller._STATE
    item = replace(WOODCUT, skill=skill, detail=detail, recipe=detail)
    state.queue = [item, replace(item, id='old2')]
    state.current = item
    if pending_kind == 'action':
        controller.wait_action_completion(item.id, 1000, ExecuteResult(True, 'done'), actions_complete=True)
    else:
        controller.mark_gather_pending(item.id, 1000, saw_busy=True)
    controller.pause_life()
    st = controller.resume_life(now=1001)
    assert st.current.id == item.id
    assert [p.id for p in st.queue] == [item.id]
    # A refill must not duplicate the protected, already-issued training target.
    controller._top_up(state, snap())
    assert [p.id for p in state.queue if p.skill == skill] == [item.id]
    st = controller.acknowledge_done(item.id, now=1002)
    assert all(p.skill != skill for p in st.queue)


def test_fishing_excluded_even_if_lowest_level_or_held_target():
    s = replace(snap(), skills=(SkillGrowth('釣魚', 1, 0),))
    assert len(SKILL_GATHER) == 8 and all(skill != '釣魚' for skill, _ in SKILL_GATHER)
    assert all(p.skill != '釣魚' for p in planner.plan_life(s, training_target=FISH))
    from mm_assist.life_plan.live_test import scenarios
    assert 'train:東側城牆釣魚場' not in scenarios()


@pytest.mark.parametrize('operation', ['refresh', 'top_up', 'resume'])
def test_unissued_old_fishing_removed_from_remaining_queue(monkeypatch, operation):
    controller.start_life(snapshot=snap(), now=1000)
    state = controller._STATE
    keep = state.queue[0]
    state.queue = [FISH, keep]
    state.current = FISH
    if operation == 'refresh':
        state.replan_requested = True
        controller._rebuild_waiting(state, snap())
    elif operation == 'top_up':
        controller._top_up(state, snap())
    else:
        controller.pause_life()
        controller.resume_life(now=1001)
    assert state.queue and all(p.skill != '釣魚' for p in state.queue)
    assert keep.id in {p.id for p in state.queue}
    assert state.current.skill != '釣魚'


def test_issued_fishing_fact_remains_until_completion_not_replayed(monkeypatch):
    controller.start_life(snapshot=snap(), now=1000)
    state = controller._STATE
    state.queue = [FISH, replace(FISH, id='old2')]
    state.current = FISH
    controller.wait_action_completion(FISH.id, 1000, ExecuteResult(True, 'done'), actions_complete=True)
    controller.pause_life()
    st = controller.resume_life(now=1001)
    assert st.current.id == FISH.id and st.pending_action.item_id == FISH.id
    assert [p.id for p in st.queue] == [FISH.id]
    st = controller.acknowledge_done(FISH.id, now=1002)
    assert not st.queue and st.refresh_requested
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(snap(), taken_at=k['now']))
    st = controller.tick_facility_progress(now=1003)
    assert st.queue and all(p.skill != '釣魚' for p in st.queue)


def test_legacy_issued_fishing_remains_pinned_across_resume():
    controller.start_life(snapshot=snap(), now=1000)
    state = controller._STATE
    keep = state.queue[0]
    state.queue = [keep, FISH, replace(FISH, id='old2')]
    controller.mark_gather_pending(FISH.id, 1000, saw_busy=True)
    controller.pause_life()
    st = controller.resume_life(now=1001)
    assert st.gather_pending_id == FISH.id and st.current.id == FISH.id
    assert [p.id for p in st.queue] == [keep.id, FISH.id]
    state.replan_requested = True
    controller._rebuild_waiting(state, snap())
    assert state.current.id == FISH.id and state.gather_pending_busy


READY = [
    ('伐木', 'woodcut', '幽暮樹', 25),
    ('採礦', 'mine', '銀礦脈', 25),
    ('採集藥草', 'herb', '傷痕花', 20),
    ('採集藥草', 'herb', '增幅蘑菇', 25),
    ('剪羊毛', 'shear', '烏雲羊', 25),
    ('昆蟲採集', 'insect', '荒蕪的昆蟲群', 20),
]


@pytest.mark.parametrize('skill,sid,name,level', READY)
def test_upgrade_levels_unknown_baseline_and_live_precondition(skill, sid, name, level):
    from mm_assist.life_plan.live_test import scenarios, precondition
    counts = {s: 1 for s, _ in SKILL_GATHER if s != skill}
    base = dict(SKILL_GATHER)[skill]
    candidate = PlanItem('new', 'gather', name, skill=skill, detail=name,
                         recipe=name, gather_mode='skill')
    for known in (None, level - 1, level):
        s = replace(snap(), skills=() if known is None else (SkillGrowth(skill, known, 0),))
        expected = name if known == level else base
        if name == '增幅蘑菇' and known == 24:
            expected = '傷痕花'
        if skill in {'伐木', '採礦', '採集藥草', '剪羊毛'}:
            assert planner.training_item(s, counts).skill != skill
        else:
            assert planner.training_item(s, counts).detail == expected
        assert planner.training_target_allowed(candidate, s) == (known == level)
        planned = planner.plan_life(s, training_counts=counts, training_target=candidate)
        if skill in {'伐木', '採礦', '採集藥草', '剪羊毛'}:
            assert all(p.skill != skill for p in planned)
        else:
            assert all(p.detail == expected for p in planned)
        assert f'train:{name}' in scenarios()
        assert precondition(scenarios()[f'train:{name}'], s) == (
            '' if known == level else 'training_level_unverified')


@pytest.mark.parametrize('skill,sid,name,level', READY)
def test_ready_original_crop_hash_and_wrong_page(skill, sid, name, level):
    path = ROOT/f'knowledge/templates/items/{sid}/{name}.png'
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    source = ROOT/meta['source_frame']
    assert meta['min_level'] == level and meta['status'] == 'ready'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == meta['source_sha256']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['template_sha256']
    frame = cv2.imread(str(source))
    out = locate_skill_item(frame, skill_id=sid, name=name)
    assert out['ok'] and out['src'] in {'template+ocr', 'template_fuzzy+ocr'}
    x, y = out['click_client']
    assert 1180 <= x <= 1360 and meta['client_px']['y'] <= y <= meta['client_px']['y'] + 110
    absent = frame.copy()
    row_y = meta['client_px']['y']
    absent[row_y-20:row_y+120, 1050:1810] = 0
    assert not locate_skill_item(absent, skill_id=sid, name=name)['ok']
    wrong_sid = 'herb' if sid == 'mine' else 'mine'
    wrong = cv2.imread(str(ROOT/f'tests/fixtures/process_stance/{wrong_sid}_list_updated_20261005.png'))
    assert not locate_skill_item(wrong, skill_id=sid, name=name)['ok']


@pytest.mark.parametrize('sid,name', [('herb', '增幅蘑菇')])
def test_gray_unlocked_later_asset_never_provides_click(sid, name, tmp_path, monkeypatch):
    path = ROOT/f'knowledge/templates/items/{sid}/{name}.json'
    meta = json.loads(path.read_text(encoding='utf-8'))
    # The new unlocked attachment supersedes the production metadata. Keep the
    # gray-source refusal guard independently, with its original frame.
    meta['status'] = 'deferred_level_gray'
    assert meta['min_level'] == 25
    metadata = tmp_path/f'knowledge/templates/items/{sid}/{name}.json'
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps(meta), encoding='utf-8')
    frame = cv2.imread(str(ROOT/'tests/fixtures/process_stance/herb_list_updated_20261005.png'))
    monkeypatch.setattr('mm_assist.perception.item_sense.match_template',
                        lambda *a, **k: pytest.fail('gray asset reached matching'))
    out = locate_skill_item(frame, skill_id=sid, name=name, root=tmp_path)
    assert not out['ok'] and out['error'] == 'deferred_level_gray'
    assert 'click_client' not in out


@pytest.mark.parametrize('sid,name', [
    ('woodcut', '甲胄樹'), ('mine', '白銅礦脈'), ('herb', '淨淨蘑菇'),
    ('shear', '捲毛羊'), ('insect', '冰冷的光群'),
])
def test_existing_items_still_locate_on_new_list_frames(sid, name):
    frame = cv2.imread(str(ROOT/f'tests/fixtures/process_stance/{sid}_list_updated_20261005.png'))
    out = locate_skill_item(frame, skill_id=sid, name=name)
    assert out['ok'] and out['name'] == name


def test_mining_standalone_keeps_silver_level_gate_without_automatic_training():
    from mm_assist.life_plan.live_test import scenarios, precondition

    counts = {s: 1 for s, _ in SKILL_GATHER if s != '採礦'}
    for level in (24, 25, 26):
        s = replace(snap(), skills=(SkillGrowth('採礦', level, 0),))
        assert precondition(scenarios()['train:銀礦脈'], s) == (
            '' if level >= 25 else 'training_level_unverified')
        assert precondition(scenarios()['train:白銅礦脈'], s) == ''
        assert planner.training_item(s, counts).skill != '採礦'
        assert all(p.skill != '採礦' for p in planner.plan_life(s, training_counts=counts))


@pytest.mark.parametrize('initial_level', [24, 25])
def test_mining_level_change_preserves_eligible_batch_without_silver_training(initial_level):
    def mining_snap(level):
        return replace(snap(), skills=tuple(
            SkillGrowth(skill, level if skill == '採礦' else 30, 0)
            for skill, _ in SKILL_GATHER))
    controller.start_life(snapshot=mining_snap(initial_level), now=1000)
    for item in list(controller._STATE.queue)[:7]:
        controller.acknowledge_done(item.id, now=1001)
    state = controller._STATE
    assert all(p.skill != '採礦' for p in state.queue)
    remaining_target = state.queue[0].detail
    remaining_ids = [p.id for p in state.queue]
    fresh = mining_snap(26)
    state.replan_requested = True
    controller._rebuild_waiting(state, fresh)
    assert [p.id for p in state.queue] == remaining_ids and len(state.queue) == 3
    assert all(p.detail == remaining_target for p in state.queue)
    # A new start selects a fresh batch rather than replacing remaining actions.
    new = controller.start_life(snapshot=fresh, now=1002)
    assert len(new.queue) == 10 and all(p.skill != '採礦' for p in new.queue)


@pytest.mark.parametrize('level,expected', [
    (None, '淨淨蘑菇'), (19, '淨淨蘑菇'), (20, '傷痕花'),
    (24, '傷痕花'), (25, '增幅蘑菇'), (26, '增幅蘑菇'),
])
def test_herb_standalone_keeps_tier_gates_without_automatic_training(level, expected):
    from mm_assist.life_plan.live_test import scenarios, precondition

    counts = {s: 1 for s, _ in SKILL_GATHER if s != '採集藥草'}
    s = replace(snap(), skills=() if level is None else (SkillGrowth('採集藥草', level, 0),))
    assert precondition(scenarios()[f'train:{expected}'], s) == ''
    for detail, threshold in [('傷痕花', 20), ('增幅蘑菇', 25)]:
        assert precondition(scenarios()[f'train:{detail}'], s) == (
            '' if level is not None and level >= threshold else 'training_level_unverified')
    assert planner.training_item(s, counts).skill != '採集藥草'
    assert all(p.skill != '採集藥草' for p in planner.plan_life(s, training_counts=counts))


@pytest.mark.parametrize('initial_level', [19, 24, 25])
def test_herb_level_change_preserves_eligible_batch_without_herb_training(initial_level):
    def herb_snap(level):
        return replace(snap(), skills=tuple(
            SkillGrowth(skill, level if skill == '採集藥草' else 30, 0)
            for skill, _ in SKILL_GATHER))
    controller.start_life(snapshot=herb_snap(initial_level), now=1000)
    for item in list(controller._STATE.queue)[:7]:
        controller.acknowledge_done(item.id, now=1001)
    state = controller._STATE
    assert all(p.skill != '採集藥草' for p in state.queue)
    remaining_target = state.queue[0].detail
    remaining_ids = [p.id for p in state.queue]
    fresh = herb_snap(26)
    state.replan_requested = True
    controller._rebuild_waiting(state, fresh)
    assert [p.id for p in state.queue] == remaining_ids and len(state.queue) == 3
    assert all(p.detail == remaining_target for p in state.queue)
    new = controller.start_life(snapshot=fresh, now=1002)
    assert len(new.queue) == 10 and all(p.skill != '採集藥草' for p in new.queue)


@pytest.mark.parametrize('sid,name', [('mine', '銀礦脈'), ('herb', '增幅蘑菇')])
def test_unlocked_attachment_registration_and_exact_icon_crop(sid, name):
    path = ROOT/f'knowledge/templates/items/{sid}/{name}.png'
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    window_path = ROOT/meta['source_window']
    assert hashlib.sha256(window_path.read_bytes()).hexdigest() == meta['source_window_sha256']
    window = cv2.imread(str(window_path))
    frame = cv2.imread(str(ROOT/meta['source_frame']))
    assert hashlib.sha256((ROOT/meta['source_frame']).read_bytes()).hexdigest() == meta['source_sha256']
    x, y, w, h = meta['client_from_window_px']
    assert (w, h) == (1912, 1442)
    assert np.array_equal(frame, window[y:y+h, x:x+w])
    box = meta['client_px']
    x, y, w, h = (box[k] for k in ('x', 'y', 'w', 'h'))
    # Unicode-safe decoding reads the actual registered path on Windows.
    icon = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['template_sha256']
    assert np.array_equal(icon, frame[y:y+h, x:x+w])
    from mm_assist.perception.item_sense import ITEM_LIST_ROI, TEMPLATE_THRESHOLD
    from mm_assist.platform.match import match_template
    hit = match_template(frame, path, roi=ITEM_LIST_ROI,
                         threshold=TEMPLATE_THRESHOLD, min_margin=0.02)
    assert hit is not None and (hit.x, hit.y, hit.w, hit.h) == (x, y, w, h)
    # Registration alone does not prove the page/name: the original OCR-backed
    # wrong-page and absent-row assertions remain in the catalog regression.


def test_silver_update_preserves_white_copper_on_both_list_frames():
    frame = cv2.imread(str(ROOT/'tests/fixtures/process_stance/mine_list_updated_20261006.png'))
    old = cv2.imread(str(ROOT/'tests/fixtures/process_stance/mine_list_updated_20261005.png'))
    assert locate_skill_item(old, skill_id='mine', name='白銅礦脈')['ok']
    assert locate_skill_item(frame, skill_id='mine', name='白銅礦脈')['ok']


@pytest.mark.parametrize('change', ['new_view', 'new_pixels'])
def test_list_ocr_cache_only_reuses_same_unchanged_frame(monkeypatch, change):
    from mm_assist.perception import item_sense as mod
    from mm_assist.perception.ocr import OcrLine
    mod._ocr_list_cache.clear()
    calls = []
    def read(frame, *args, **kwargs):
        calls.append(frame)
        return [OcrLine(str(len(calls)), 1, None)]
    monkeypatch.setattr(mod, 'ocr_roi', read)
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    first = mod._ocr_lines_in_item_list(frame)
    assert mod._ocr_lines_in_item_list(frame) is first and len(calls) == 1
    if change == 'new_view':
        new = frame.view()  # 相同缓冲地址仍是另一帧，不能复用旧识别。
    else:
        frame[200, 600] = 255
        new = frame
    fresh = mod._ocr_lines_in_item_list(new)
    assert len(calls) == 2 and fresh[0].text == '2'


@pytest.mark.parametrize('skill,sid,details,materials', [
    ('伐木', 'woodcut', ('甲胄樹', '幽暮樹'), (('原木', '樹木'), ('高級原木', '成材樹'))),
    ('剪羊毛', 'shear', ('捲毛羊', '烏雲羊'), (('羊毛', '羊'), ('高級羊毛', '捲毛羊'))),
    ('採礦', 'mine', ('白銅礦脈', '銀礦脈'), (('鐵礦石', '鐵礦脈'), ('煤炭', '煤炭礦脈'))),
    ('採集藥草', 'herb', ('淨淨蘑菇', '傷痕花', '增幅蘑菇'),
     (('淨淨蘑菇', '淨淨蘑菇'), ('增幅蘑菇', '增幅蘑菇'), ('箭花', '箭花'))),
])
def test_catalog_and_standalone_training_retained_without_automatic_training(
        monkeypatch, skill, sid, details, materials):
    from mm_assist.life_plan.execute import try_execute
    from mm_assist.life_plan.live_test import scenarios, precondition

    assert all(detail in items_for_skill(sid) for detail in details)
    assert (ROOT/f'knowledge/templates/items/{sid}/{details[-1]}.png').is_file()
    skills = tuple(SkillGrowth(name, 26 if name == skill else 30, 0) for name, _ in SKILL_GATHER)
    plan = planner.plan_life(replace(snap(), skills=skills))
    assert plan and all(p.skill != skill for p in plan)
    calls = []
    monkeypatch.setattr('mm_assist.life_execute.skill_nearby.execute_skill_nearby',
                        lambda *a, **k: calls.append(k) or ExecuteResult(True, 'done'))
    monkeypatch.setattr('mm_assist.life_execute.herb_mushroom.execute_skill_nearby',
                        lambda *a, **k: calls.append(k) or ExecuteResult(True, 'done'))
    for detail in details:
        item = scenarios()[f'train:{detail}']
        assert item.skill == skill
        assert precondition(item, replace(snap(), skills=(SkillGrowth(skill, 26, 0),))) == ''
        assert try_execute(item).ok
        assert (calls[-1]['skill_id'], calls[-1]['item_name']) == (sid, detail)
    for material, detail in materials:
        assert try_execute(scenarios()[f'gather:{material}']).ok
        assert (calls[-1]['skill_id'], calls[-1]['item_name']) == (sid, detail)


def test_original_woodcut_frame_correct_row_and_absent_negative():
    frame = cv2.imread(str(ROOT/'tests/fixtures/process_stance/woodcut_list_twilight_tree.png'))
    out = locate_skill_item(frame, skill_id='woodcut', name='幽暮樹')
    assert out['ok'] and out['src'] == 'template+ocr'
    x, y = out['click_client']
    assert 1180 <= x <= 1320 and 1300 <= y <= 1380
    absent = frame.copy()
    absent[1260:1410, 1050:1810] = 0
    missing = locate_skill_item(absent, skill_id='woodcut', name='幽暮樹')
    assert not missing['ok'] and 'click_client' not in missing
