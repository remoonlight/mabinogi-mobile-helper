"""Thirty remaining of the fixed 40 goal; preserve branch/safety regressions."""
from mm_assist.life_plan import bar_ui, planner
from mm_assist.life_plan.recipes import display_need_rows
from mm_assist.life_plan.types import LifeSnapshot


OTHER_TARGETS = {'合金鋼錠': 100, '高級木材': 100, '高級布料': 100,
                 '高級混合強化秘藥(山崩)': 10}


def test_thirty_elixirs_round_batches_before_reserving_materials():
    from mm_assist.life_plan.elixir_reserve import mountain_targets

    assert mountain_targets(30) == {
        '淨淨蘑菇汁液': 150, '增幅蘑菇汁液': 150,
        '強化秘藥乳化劑': 150, '生命的魔力石': 375, '憤怒的碎片': 900,
        '礦山的魔力石': 180, '鍊金術碎屑': 1100,
        '箭花': 250, '滴答螢火蟲': 250, '血紅藥草': 250, '獨角仙': 250,
    }


def test_necessary_view_expands_both_juice_chains_without_extra_squeak_reserve():
    desired = planner.necessary_desired(OTHER_TARGETS)
    assert desired['淨淨蘑菇汁液'] == desired['增幅蘑菇汁液'] == 150
    assert '咻咻蘑菇汁液' not in desired
    _, rows = display_need_rows(OTHER_TARGETS, desired)
    gaps = {name: short for name, _, short in rows}
    assert {name: gaps[name] for name in (
        '淨淨蘑菇', '增幅蘑菇', '咻咻蘑菇', '紙', '裝水的瓶子', '咻咻蘑菇汁液',
    )} == {'淨淨蘑菇': 600, '增幅蘑菇': 600, '咻咻蘑菇': 1200,
           '紙': 240, '裝水的瓶子': 90, '咻咻蘑菇汁液': 300}


def test_finished_juices_remove_their_raw_gaps_and_intermediate_stock_is_spent_once():
    have = dict(OTHER_TARGETS, 淨淨蘑菇汁液=150, 增幅蘑菇汁液=150)
    _, rows = display_need_rows(have, planner.necessary_desired(have))
    assert not any('蘑菇' in name or name in {'紙', '裝水的瓶子'} for name, _, _ in rows)
    have.pop('增幅蘑菇汁液')
    have.update({'咻咻蘑菇汁液': 295, '增幅蘑菇孢子': 300})
    _, rows = display_need_rows(have, planner.necessary_desired(have))
    gaps = {name: short for name, _, short in rows}
    assert gaps['咻咻蘑菇汁液'] == 5
    assert gaps['咻咻蘑菇'] == 30 and gaps['紙'] == 3 and gaps['裝水的瓶子'] == 1
    assert '增幅蘑菇' not in gaps and '淨淨蘑菇' not in gaps


def test_status_bar_shows_thirty_reserve_and_deducts_existing_stock():
    have = dict(OTHER_TARGETS, 淨淨蘑菇汁液=145, 增幅蘑菇汁液=150, 箭花=249)
    text = bar_ui.format_need_items(status=True, have=have)
    assert '淨淨蘑菇汁液 145/150' in text
    assert '增幅蘑菇汁液 150/150' in text
    assert '箭花 249/250' in text and text.count('箭花 ') == 1


def test_existing_juice_feed_stops_at_150_and_needed_herb_precedes_training():
    have = dict(OTHER_TARGETS, 淨淨蘑菇汁液=145, 淨淨蘑菇孢子=100, 裝水的瓶子=20)
    plan = planner.plan_life(LifeSnapshot(ok=True, items=have))
    feeds = [p for p in plan if p.kind == 'feed']
    assert len(feeds) == 1 and feeds[0].recipe == '淨淨蘑菇汁液'
    reserve = [p for p in plan if p.recipe == '咻咻蘑菇' and p.gather_mode == 'process_mat']
    assert len(reserve) == 1 and '×1200' in reserve[0].reason
    assert plan.index(reserve[0]) < next(i for i, p in enumerate(plan) if p.gather_mode == 'skill')
    assert not any(p.kind == 'feed' and p.recipe.startswith(('增幅', '咻咻')) for p in plan)


def test_no_reserve_herb_gather_when_juices_and_other_herbs_are_sufficient():
    have = dict(OTHER_TARGETS, 淨淨蘑菇汁液=150, 增幅蘑菇汁液=150, 箭花=250, 血紅藥草=250,
                滴答螢火蟲=250, 獨角仙=250)
    plan = planner.plan_life(LifeSnapshot(ok=True, items=have))
    assert all(p.gather_mode == 'skill' for p in plan)


def test_two_vertical_lists_preserve_all_quantities_and_manual_markers():
    from mm_assist.life_plan.controller import LifeController
    state = LifeController(snapshot=LifeSnapshot(ok=True, items=OTHER_TARGETS, taken_at=1000))
    caption, left, right = bar_ui.format_life_needs_columns(state)
    assert '读数时刻' in caption
    columns = [left.splitlines(), right.splitlines()]
    assert abs(len(columns[0]) - len(columns[1])) <= 1
    flat = '\n'.join([left, right])
    for line in bar_ui.format_life_needs_block(state).splitlines()[2:]:
        if not line.startswith(('需自行准备：', '需人工加工')):
            assert line in flat
    assert '強化秘藥乳化劑' not in flat and '自行准备' not in flat
    assert '增幅蘑菇汁液 0/150' in flat and '（人工加工）' not in flat
    assert '獨角仙 0/250（自行准备）' not in flat
    caption, left, right = bar_ui.format_life_needs_columns(LifeController())
    assert '未读取' in caption and not left and not right
