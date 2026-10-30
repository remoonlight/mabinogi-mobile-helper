"""Current material shortages, executable supply, and manual preparation."""
import pytest

from mm_assist.life_plan import controller, planner
from mm_assist.life_plan.types import LifeSnapshot, QueueSlot


def satisfied():
    # Retain the 30-remaining scenarios under the fixed 40 finished goal.
    have = {'高級混合強化秘藥(山崩)': 10}
    return {**planner.necessary_desired(have), **have}


def test_manual_list_deducts_stock_and_insect_medicine_routes_are_now_wired():
    from mm_assist.life_plan.acquisition import acquisition_plan
    have = dict(satisfied(), 增幅蘑菇汁液=3, 強化秘藥乳化劑=24,
                生命的魔力石=136, 憤怒的碎片=92, 滴答螢火蟲=11, 獨角仙=178,
                增幅蘑菇孢子=27, 咻咻蘑菇孢子=46, 咻咻蘑菇=89,
                增幅蘑菇=13, 紙=1000, 裝水的瓶子=11, 空瓶=1000)
    result = acquisition_plan(have)
    assert {r['name']: r['missing'] for r in result['manual_materials']} == {
        '強化秘藥乳化劑': 126, '生命的魔力石': 239, '憤怒的碎片': 808}
    assert not result['manual_processing']
    assert {r['name'] for r in result['processing']} == {
        '增幅蘑菇汁液', '增幅蘑菇孢子', '咻咻蘑菇汁液', '咻咻蘑菇孢子'}
    assert {r['name']: r['missing'] for r in result['gather']} == {
        '增幅蘑菇': 557, '咻咻蘑菇': 1021, '裝水的瓶子': 49,
        '滴答螢火蟲': 239, '獨角仙': 72}


@pytest.mark.parametrize('fac,recipe,ingredients', [
    ('metal', '合金鋼錠', {'鋼錠': 30, '銅礦石': 150, '煤炭': 80}),
    ('wood', '高級木材', {'木材+': 30, '高級原木': 150, '樹液': 80}),
    ('cloth', '高級布料', {'布料+': 30, '高級羊毛': 150, '羊毛': 80}),
    ('medicine', '淨淨蘑菇汁液', {'淨淨蘑菇孢子': 100, '裝水的瓶子': 20}),
])
@pytest.mark.parametrize('occupied', [0, 2, 5])
def test_each_affordable_facility_fills_all_free_slots_before_training(fac, recipe, ingredients, occupied):
    have = {**satisfied(), **ingredients, recipe: 0}
    slots = {fac: tuple(QueueSlot(recipe, False) for _ in range(occupied))}
    plan = planner.plan_life(LifeSnapshot(ok=True, items=have, facilities=slots))
    feeds = [p for p in plan if p.kind == 'feed']
    assert len(feeds) == 5 - occupied
    assert all(p.facility == fac and p.recipe == recipe for p in feeds)
    assert not any(p.gather_mode == 'skill' for p in plan[:len(feeds)])


def test_targeted_necessary_gathering_runs_before_training_even_when_processing_is_full():
    have = dict(satisfied(), 箭花=23, 血紅藥草=7)
    slots = {'medicine': tuple(QueueSlot('淨淨蘑菇汁液', False) for _ in range(5))}
    plan = planner.plan_life(LifeSnapshot(ok=True, items=have, facilities=slots))
    assert plan[0].recipe == '血紅藥草' and plan[0].gather_mode == 'process_mat'
    assert '×243' in plan[0].reason


def test_full_goal_does_not_feed_and_manual_only_shortages_do_not_stall_training():
    plan = planner.plan_life(LifeSnapshot(ok=True, items={**satisfied(), '強化秘藥乳化劑': 0}))
    assert all(p.gather_mode == 'skill' for p in plan)


def test_necessary_gather_success_defers_refresh_until_batch_end(monkeypatch):
    controller.reset_controller_for_tests()
    st = controller.start_life(snapshot=LifeSnapshot(ok=True, items=dict(satisfied(), 箭花=0)), now=1000)
    assert st.current.recipe == '箭花'
    done = controller.acknowledge_done(st.current.id, now=1001)
    assert not done.refresh_requested and not done.replan_requested
    assert done.queue == st.queue[1:] and done.snapshot == st.snapshot
    controller.reset_controller_for_tests()
