"""Current four goals through real reserve math, display and planner; no game IO."""
import pytest

from mm_assist.life_plan import planner, bar_ui
from mm_assist.life_plan.acquisition import acquisition_plan
from mm_assist.life_plan.elixir_reserve import MOUNTAIN_NAME, mountain_targets
from mm_assist.life_plan.recipes import RECIPES, GATHER_NODE, display_need_rows
from mm_assist.life_plan.types import LifeSnapshot, QueueSlot


GOALS = {'合金鋼錠': 100, '高級木材': 100, '高級布料': 100, MOUNTAIN_NAME: 40}


def test_four_requested_goals_are_visible_even_with_finished_stock():
    text = bar_ui.format_need_items(status=True, have=GOALS)
    for name, amount in GOALS.items():
        assert f'{name} {amount}/{amount}' in text
        assert text.count(f'{name} ') == 1
    assert not any('蘑菇' in line for line in text.splitlines())
    assert all(p.gather_mode == 'skill' for p in planner.plan_life(LifeSnapshot(ok=True, items=GOALS)))


def test_hundred_mountain_shared_demand_is_rounded_after_combining():
    # Explicit larger batches remain supported after reducing the default goal.
    targets = mountain_targets(100)
    assert targets == {'強化秘藥乳化劑': 500, '生命的魔力石': 1250,
        '憤怒的碎片': 3000, '礦山的魔力石': 600, '鍊金術碎屑': 3680,
        '增幅蘑菇汁液': 500, '淨淨蘑菇汁液': 505,
        '血紅藥草': 840, '獨角仙': 840, '箭花': 840, '滴答螢火蟲': 840}
    desired = planner.necessary_desired({})
    assert {name: desired[name] for name in GOALS} == GOALS
    assert '咻咻蘑菇汁液' not in desired


def test_forty_mountain_default_expands_materials_and_reaches_display_and_planner():
    assert mountain_targets() == {
        '強化秘藥乳化劑': 200, '生命的魔力石': 500,
        '憤怒的碎片': 1200, '礦山的魔力石': 240, '鍊金術碎屑': 1480,
        '增幅蘑菇汁液': 200, '淨淨蘑菇汁液': 205,
        '血紅藥草': 340, '獨角仙': 340, '箭花': 340, '滴答螢火蟲': 340,
    }
    desired = planner.necessary_desired({})
    assert {name: desired[name] for name in GOALS} == GOALS
    assert f'{MOUNTAIN_NAME} 0/40' in bar_ui.format_need_items(status=True, have={})
    assert all(p.gather_mode == 'skill' for p in planner.plan_life(
        LifeSnapshot(ok=True, items=GOALS)))


def test_finished_and_partially_crafted_elixirs_reduce_material_reserve():
    assert mountain_targets(have={MOUNTAIN_NAME: 40}) == {}
    partial = mountain_targets(have={MOUNTAIN_NAME: 30})
    assert partial['淨淨蘑菇汁液'] == 55
    assert partial['增幅蘑菇汁液'] == 50
    assert partial['箭花'] == partial['血紅藥草'] == 90
    intermediates = mountain_targets(have={'強擊強化秘藥': 100, '連擊強化秘藥': 100})
    assert intermediates.get('淨淨蘑菇汁液', 0) == 0
    assert intermediates['增幅蘑菇汁液'] == 200
    assert '箭花' not in intermediates and '血紅藥草' not in intermediates


@pytest.mark.parametrize('finished_mountain', [0, 40])
@pytest.mark.parametrize('stock', [0, 990, 1000])
def test_cancelled_detonator_and_materials_leave_default_supply(finished_mountain, stock):
    DETONATOR = '魔力起爆劑'
    MATERIAL_ROUTES = ('蘊含魔力的石頭', '魔力藥草', '閃閃發亮的苔蘚')
    have = {**GOALS, MOUNTAIN_NAME: finished_mountain, DETONATOR: stock}
    excluded = {DETONATOR, *MATERIAL_ROUTES}
    desired = planner.necessary_desired(have)
    assert excluded.isdisjoint(desired)
    assert excluded.isdisjoint(name for name, _, _ in display_need_rows(have, desired)[1])
    routes = acquisition_plan(have)
    assert all(excluded.isdisjoint(row['name'] for row in rows) for rows in routes.values())
    text = bar_ui.format_need_items(status=True, have=have)
    assert all(name not in text for name in excluded)
    for stocked in (have, {**have, **dict.fromkeys(MATERIAL_ROUTES, 10000)}):
        plan = planner.plan_life(LifeSnapshot(ok=True, items=stocked))
        assert all(p.recipe not in excluded for p in plan)
    # 2026-10-27 user retired the implementation as well as its default goal.
    assert DETONATOR not in RECIPES
    assert set(MATERIAL_ROUTES).isdisjoint(GATHER_NODE)


def test_legacy_queue_counts_observed_output_without_spending_unfinished_output():
    snap = LifeSnapshot(ok=True, items={**GOALS, '魔力起爆劑': 990},
                        facilities={'medicine': (QueueSlot('魔力起爆劑', False),)})
    assert planner._virtual_have(snap)['魔力起爆劑'] == 1000
    assert snap.items['魔力起爆劑'] == 990


def test_complete_four_goals_remove_all_raw_gaps_without_hidden_juice_floor():
    desired = planner.necessary_desired(GOALS)
    assert desired.get('淨淨蘑菇汁液', 0) == 0
    _, gaps = display_need_rows(GOALS, desired)
    assert gaps == []
    assert bar_ui.format_need_items(status=False, have=GOALS, error='unreadable') == '必要物品：内存读取失败：unreadable'


def test_exact_consumption_stock_does_not_force_a_spare_juice_batch():
    have = {**planner.necessary_desired({}), MOUNTAIN_NAME: 0, '淨淨蘑菇汁液': 204}
    desired = planner.necessary_desired(have)
    assert desired['淨淨蘑菇汁液'] == 204
    plan = planner.plan_life(LifeSnapshot(ok=True, items={**have,
        '淨淨蘑菇孢子': 100, '裝水的瓶子': 10}))
    assert not any(p.kind == 'feed' and p.recipe == '淨淨蘑菇汁液' for p in plan)


def test_stock_consumption_after_goal_reopens_only_actual_remaining_reserve():
    before = planner.necessary_desired(GOALS)
    after = planner.necessary_desired({**GOALS, MOUNTAIN_NAME: 30})
    assert before.get('淨淨蘑菇汁液', 0) == 0
    assert after['淨淨蘑菇汁液'] == 55 and after['增幅蘑菇汁液'] == 50
    assert after['強化秘藥乳化劑'] == 50


def test_reserve_count_boundaries_and_stock_never_mutate_caller():
    have = {MOUNTAIN_NAME: 41, '高級強擊強化秘藥': 50}
    original = dict(have)
    assert mountain_targets(have=have) == {}
    assert have == original and mountain_targets(0) == {}
    with pytest.raises(ValueError, match='negative'):
        mountain_targets(-1)
