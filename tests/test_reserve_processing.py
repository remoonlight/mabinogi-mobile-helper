"""Reserve processing follows both branches with real spendable inputs."""
import pytest

from mm_assist.life_plan import planner
from mm_assist.life_plan.recipes import RECIPES
from mm_assist.life_plan.types import LifeSnapshot, QueueSlot


def stock(**changes):
    # 10 finished, 30 remaining: original branch quantities stay meaningful.
    have = {'高級混合強化秘藥(山崩)': 10}
    return {**planner.necessary_desired(have), **have, **changes}


@pytest.mark.parametrize('recipe,ingredients', [
    ('增幅蘑菇汁液', {'增幅蘑菇孢子': 300, '咻咻蘑菇汁液': 300}),
    ('增幅蘑菇孢子', {'增幅蘑菇': 600, '紙': 1000}),
    ('咻咻蘑菇汁液', {'咻咻蘑菇孢子': 600, '裝水的瓶子': 60}),
    ('咻咻蘑菇孢子', {'咻咻蘑菇': 1200, '紙': 1000}),
])
def test_reserve_recipe_fills_five_slots_before_training(recipe, ingredients):
    plan = planner.plan_life(LifeSnapshot(ok=True, items=stock(增幅蘑菇汁液=0, **ingredients)))
    feeds = [p for p in plan if p.kind == 'feed']
    assert [p.recipe for p in feeds] == [recipe] * 5
    assert all(p.facility == 'medicine' for p in feeds)
    assert all(p.kind == 'feed' for p in plan[:5])


def test_both_enhancement_branches_have_independent_demand():
    counts = planner._needed_batches('medicine', stock(增幅蘑菇汁液=3,
        增幅蘑菇孢子=27, 咻咻蘑菇孢子=46), planner.necessary_desired(stock()))
    assert {k: counts[k] for k in ('增幅蘑菇汁液', '增幅蘑菇孢子', '咻咻蘑菇汁液', '咻咻蘑菇孢子')} == {
        '增幅蘑菇汁液': 30, '增幅蘑菇孢子': 19, '咻咻蘑菇汁液': 60, '咻咻蘑菇孢子': 37}


def test_queued_intermediates_reduce_demand_but_cannot_be_spent():
    items = stock(增幅蘑菇汁液=145, 增幅蘑菇孢子=10)
    snap = LifeSnapshot(ok=True, items=items, facilities={'medicine': (
        QueueSlot('咻咻蘑菇汁液', False), QueueSlot('咻咻蘑菇汁液', False))})
    assert not any(p.kind == 'feed' for p in planner.plan_life(snap))
    items['咻咻蘑菇汁液'] = 10
    plan = planner.plan_life(LifeSnapshot(ok=True, items=items))
    assert [p.recipe for p in plan if p.kind == 'feed'] == ['增幅蘑菇汁液']


def test_finished_enhancement_stops_all_its_intermediate_feeding():
    plan = planner.plan_life(LifeSnapshot(ok=True, items=stock(
        增幅蘑菇孢子=500, 咻咻蘑菇汁液=500, 增幅蘑菇=1000, 咻咻蘑菇=1000, 紙=1000)))
    assert not any(p.kind == 'feed' for p in plan)


def test_new_recipes_equal_observed_current_game_bom():
    from mm_assist.life_plan.elixir_reserve import observed_recipes
    for name in ('增幅蘑菇孢子', '增幅蘑菇汁液', '咻咻蘑菇孢子', '咻咻蘑菇汁液'):
        row = observed_recipes()[name]
        assert RECIPES[name].output_n == row['output_n']
        assert RECIPES[name].inputs == {i['name']: i['amount'] for i in row['ingredients']}
