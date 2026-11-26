"""Current-slot supply and balanced training through the real planner/controller."""
from dataclasses import replace

import pytest

from mm_assist.life_plan import controller, planner, bar_ui, snapshot
from mm_assist.life_plan.recipes import LIFE_TRAINING_GATHER
from mm_assist.life_plan.types import LifeSnapshot, PlanItem, QueueSlot, SkillGrowth, ExecuteResult

TARGETS = {'合金鋼錠': 100, '高級木材': 100, '高級布料': 100, '淨淨蘑菇汁液': 150,
           '高級混合強化秘藥(山崩)': 10,
           '增幅蘑菇汁液': 150, '箭花': 250, '血紅藥草': 250, '滴答螢火蟲': 250, '獨角仙': 250}


def snap(items=None, facilities=None, skills=()):
    # Preserve training/slot tests with the same 30-unit remaining supply.
    return LifeSnapshot(ok=True, items={'增幅蘑菇汁液': 150, '箭花': 250, '血紅藥草': 250,
                                      '高級混合強化秘藥(山崩)': 10,
                                      '滴答螢火蟲': 250, '獨角仙': 250,
                                       **dict(items or {})}, facilities=facilities or {},
                        skills=skills, taken_at=1000)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    controller.reset_controller_for_tests()
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly', lambda **k: None)
    yield
    controller.reset_controller_for_tests()


def test_ready_wood_feed_precedes_missing_metal_materials():
    plan = planner.plan_life(snap({'原木': 50}))
    assert plan[0].kind == 'feed' and plan[0].facility == 'wood'
    feeds = [i for i, p in enumerate(plan) if p.kind == 'feed']
    mats = [i for i, p in enumerate(plan) if p.gather_mode == 'process_mat']
    assert len(feeds) == 5 and max(feeds) < min(mats)


def test_ten_unknown_yield_gathers_do_not_credit_expected_feed_stock():
    items = dict(TARGETS, 合金鋼錠=0)
    plan = planner.plan_life(snap(items))
    gathers = [p for p in plan if p.gather_mode == 'process_mat']
    assert len(gathers) == 10 and all(p.recipe == '鐵礦石' for p in gathers)
    assert not any(p.kind == 'feed' for p in plan)
    assert '×50' in gathers[0].reason


@pytest.mark.parametrize('stock,expected', [(0, 40), (98, 8)])
def test_sap_shortfall_only_covers_free_slots_and_final_target(stock, expected):
    items = dict(TARGETS, 高級木材=stock)
    items.update({'木材+': 15, '高級原木': 100, '樹液': 0})
    plan = planner.plan_life(snap(items))
    gathers = [p for p in plan if p.gather_mode == 'process_mat']
    assert len(gathers) == 10 and all(p.recipe == '樹液' for p in gathers)
    assert f'×{expected}' in gathers[0].reason


def test_unfinished_intermediate_output_is_not_available_for_feed():
    items = dict(TARGETS, 高級木材=0)
    items.update({'樹液': 100, '高級原木': 100})
    queued = {'wood': (QueueSlot('木材+', False, remain_s=400),)}
    plan = planner.plan_life(snap(items, queued))
    assert not any(p.kind == 'feed' and p.recipe == '高級木材' for p in plan)


def test_missing_paper_and_empty_bottles_do_not_trigger_useless_gathering():
    items = dict(TARGETS, 淨淨蘑菇汁液=0)
    plan = planner.plan_life(snap(items))
    assert all(p.gather_mode == 'skill' for p in plan)


def test_full_facilities_train_one_target_for_the_whole_batch():
    facilities = {fac: tuple(QueueSlot(recipe, False, remain_s=600) for _ in range(5))
        for fac, recipe in [('metal', '合金鋼錠'), ('wood', '高級木材'),
                            ('cloth', '高級布料'), ('medicine', '淨淨蘑菇汁液')]}
    plan = planner.plan_life(snap({}, facilities))
    assert all(p.gather_mode == 'skill' for p in plan)
    assert len(plan) == 10
    assert len({(p.skill, p.detail, p.recipe) for p in plan}) == 1


def test_balanced_training_prefers_low_level_within_round_without_starvation():
    skills = tuple(SkillGrowth(skill, 1 if skill == '收割' else 20, 0) for skill, _ in LIFE_TRAINING_GATHER)
    plan = planner.plan_life(snap(TARGETS, skills=skills))
    assert plan[0].skill == '收割'
    assert all(p.skill == '收割' and p.detail == '黃豆' for p in plan)
    later = planner.plan_life(snap(TARGETS, skills=skills), training_counts={'收割': 1})
    assert later[0].skill != '收割'
    assert len({(p.skill, p.detail) for p in later}) == 1


@pytest.mark.parametrize('skill,detail', [('收割', '黃豆'), ('鋤地', '防風草')])
def test_crop_training_preserves_batch_on_refresh(monkeypatch, skill, detail):
    from mm_assist.perception.item_sense import items_for_skill

    skills = tuple(SkillGrowth(name, 1 if name == skill else 20, 0)
                   for name, _ in LIFE_TRAINING_GATHER)
    st = controller.start_life(snapshot=snap(TARGETS, skills=skills), now=1000)
    assert len(st.queue) == 10
    assert all((p.skill, p.detail, p.recipe, p.title) ==
               (skill, detail, detail, f'练采集 {detail}') for p in st.queue)
    assert detail in items_for_skill('harvest' if skill == '收割' else 'hoe')
    for item in list(st.queue)[:7]:
        st = controller.acknowledge_done(item.id, now=1001)
    remaining = list(st.queue)
    fresh = snap(TARGETS, skills=(SkillGrowth('採礦', 1, 0),))
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(fresh, taken_at=k['now']))
    st = controller.tick_facility_progress(now=1600)
    assert [p.id for p in st.queue] == [p.id for p in remaining]
    assert len(st.queue) == 3
    assert all(p.detail == detail for p in st.queue)


def test_two_training_batches_use_new_crop_targets(monkeypatch):
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(snap(TARGETS), taken_at=k['now']))
    st = controller.start_life(snapshot=snap(TARGETS), now=1000)
    harvest_details = []
    hoe_details = []
    for batch in range(len(LIFE_TRAINING_GATHER)):
        original = list(st.queue)
        assert all(p.detail not in {'玉米', '貝類'} for p in original)
        harvest_details.extend(p.detail for p in original if p.skill == '收割')
        hoe_details.extend(p.detail for p in original if p.skill == '鋤地')
        for n, item in enumerate(original):
            st = controller.acknowledge_done(item.id, now=1001 + batch * 10 + n)
        assert st.current is None and st.refresh_requested
        st = controller.tick_facility_progress(now=1011 + batch * 10)
    assert harvest_details == ['黃豆'] * 10
    assert hoe_details == ['防風草'] * 10


def test_controller_finishes_same_target_batch_then_rotates_eight_skills(monkeypatch):
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(snap(TARGETS), taken_at=k['now']))
    st = controller.start_life(snapshot=snap(TARGETS), now=1000)
    seen = []
    for batch in range(len(LIFE_TRAINING_GATHER)):
        original = list(st.queue)
        target = (original[0].skill, original[0].detail)
        assert len(original) == 10
        assert all((p.skill, p.detail) == target for p in original)
        seen.append(target[0])
        for n, item in enumerate(original):
            st = controller.acknowledge_done(item.id, now=1001 + batch * 10 + n)
            if n < 9:
                assert [p.id for p in st.queue] == [p.id for p in original[n + 1:]]
        assert st.current is None and st.refresh_requested
        st = controller.tick_facility_progress(now=1011 + batch * 10)
    assert set(seen) == {'日常採集', '伐木', '採礦', '採集藥草', '剪羊毛', '收割', '鋤地', '昆蟲採集'}
    assert len(seen) == 8 and '釣魚' not in seen


def test_mid_batch_refresh_preserves_target_remaining_count_and_feed_priority(monkeypatch):
    st = controller.start_life(snapshot=snap(TARGETS), now=1000)
    for item in list(st.queue)[:7]:
        st = controller.acknowledge_done(item.id, now=1001)
    remaining = list(st.queue)
    fresh = snap(dict(TARGETS, 高級木材=0, 原木=20),
                 skills=(SkillGrowth('日常採集', 30, 0), SkillGrowth('採礦', 1, 0)))
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(fresh, taken_at=k['now']))
    st = controller.tick_facility_progress(now=1600)
    training = [p for p in st.queue if p.gather_mode == 'skill']
    assert [p.id for p in training] == [p.id for p in remaining]
    assert all((p.skill, p.detail) == (remaining[0].skill, remaining[0].detail) for p in training)
    assert st.current.id == remaining[0].id
    assert not any(p.kind == 'feed' for p in st.queue)


def test_failed_training_continues_same_batch_then_uses_one_other_target(monkeypatch):
    monkeypatch.setattr(controller.time, 'time', lambda: 1001)
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(snap(TARGETS), taken_at=k['now']))
    st = controller.start_life(snapshot=snap(TARGETS), now=1000)
    original = list(st.queue)
    for n, item in enumerate(original):
        assert st.current.id == item.id
        st = controller.skip_current(reason='failure')
        if n < 9:
            assert [p.id for p in st.queue] == [p.id for p in original[n + 1:]]
    assert st.current is None and st.refresh_requested
    st = controller.tick_facility_progress(now=1002)
    assert len(st.queue) == 10
    assert len({(p.skill, p.detail) for p in st.queue}) == 1


def test_new_start_discards_previous_batch_target():
    original = controller.start_life(snapshot=snap(TARGETS), now=1000)
    assert all(p.skill == '收割' for p in original.queue)
    fresh = snap(TARGETS, skills=(SkillGrowth('收割', 30, 0), SkillGrowth('鋤地', 1, 0)))
    st = controller.start_life(snapshot=fresh, now=2000)
    assert all(p.skill == '鋤地' for p in st.queue)


def test_material_completion_defers_one_refresh_until_training_batch_finishes(monkeypatch):
    st = controller.start_life(snapshot=snap({'原木': 50}), now=1000)
    # Finish this boundary with the real supply/training controller, without input.
    mat = PlanItem('p50', 'gather', '采集 樹液', recipe='樹液', gather_mode='process_mat')
    training = PlanItem('p51', 'gather', '练采集 黃豆', skill='收割',
                        detail='黃豆', recipe='黃豆', gather_mode='skill')
    controller._STATE.queue = [mat, training]
    controller._STATE.current = mat
    controller.acknowledge_done(mat.id, now=1001)
    assert not controller.get_controller().refresh_requested
    assert controller.get_controller().current == training
    reads = []
    def read(**kwargs):
        reads.append(kwargs)
        return replace(snap({'原木': 50}), taken_at=kwargs['now'])
    monkeypatch.setattr(controller, 'take_snapshot', read)
    fresh = controller.tick_facility_progress(now=1002)
    assert reads == [] and fresh.current == training
    controller.acknowledge_done(training.id, now=1003)
    fresh = controller.tick_facility_progress(now=1004)
    assert len(reads) == 1 and reads[0]['include_tools'] and reads[0]['include_skills']
    assert fresh.current.kind == 'feed' and fresh.current.facility == 'wood'
    controller.tick_facility_progress(now=1005)
    assert len(reads) == 1


def test_refresh_preserves_issued_training_until_completion_then_promotes_feed(monkeypatch):
    st = controller.start_life(snapshot=snap(TARGETS), now=1000)
    current = st.current
    controller.wait_action_completion(current.id, 1000, ExecuteResult(True, 'done', action_issued=True),
                                      actions_complete=True)
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(snap({'原木': 50}),
                                                                       taken_at=k['now']))
    during = controller.tick_facility_progress(now=1600)
    assert during.current.id == current.id and during.pending_action is not None
    done = controller.acknowledge_done(current.id, now=1601)
    assert done.current.id == st.queue[1].id
    assert done.current.gather_mode == 'skill'
    assert done.training_counts[current.skill] == 1


def test_late_snapshot_feed_reservation_limits_new_slots_and_stock(monkeypatch):
    old = snap({'原木': 50})
    st = controller.start_life(snapshot=old, now=1000)
    controller.acknowledge_done(st.current.id, now=1010)
    # The read started before feed completion: conservatively retain reservation.
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(old, taken_at=1009))
    fresh = controller.tick_facility_progress(now=1600)
    assert sum(p.kind == 'feed' and p.facility == 'wood' for p in fresh.queue) == 4
    assert controller.execution_items(fresh)['原木'] == 40


def test_scan_started_before_material_finish_cannot_replan_from_old_stock(monkeypatch):
    st = controller.start_life(snapshot=snap(TARGETS), now=1000)
    mat = PlanItem('p50', 'gather', '采集 原木', recipe='原木', skill='伐木', gather_mode='process_mat')
    controller._STATE.queue = [mat]
    controller._STATE.current = mat
    controller.wait_action_completion(mat.id, 1000, ExecuteResult(True, 'done', action_issued=True),
                                      actions_complete=True)
    reads = []
    def body(ts, **kwargs):
        reads.append(ts)
        return replace(snap({'原木': 10 if ts < 1001.5 else 50}), taken_at=ts)
    monkeypatch.setattr(snapshot, '_snapshot_body', body)
    real_take = snapshot.take_snapshot
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: real_take(**dict(k, budget_s=.01)))
    try:
        with snapshot.heavy_read_lock:
            controller.tick_facility_progress(now=1001, force_scan=True)
            controller.acknowledge_done(mat.id, now=1001.5)
        with snapshot._SCAN_LOCK:
            pass
        done = controller.tick_facility_progress(now=1002)
        assert reads == [1002]  # no scan starts while the last action is still pending
        assert done.snapshot.items['原木'] == 50
        assert sum(p.kind == 'feed' and p.facility == 'wood' for p in done.queue) == 5
    finally:
        with snapshot._SCAN_LOCK:
            pass
