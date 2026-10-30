"""Reported wrong herb, duplicate needs, and drifting ten-item batches."""
from dataclasses import replace

import numpy as np
import pytest

from mm_assist.life_plan import bar_ui, controller, planner
from mm_assist.life_plan.types import LifeSnapshot, QueueSlot


def test_needs_have_one_row_per_name_with_total_quantity():
    have = dict(planner.necessary_desired({}), 箭花=249, 增幅蘑菇汁液=3,
                **{'高級混合強化秘藥(山崩)': 10},
                增幅蘑菇=13, 咻咻蘑菇=89, 強化秘藥乳化劑=24)
    st = controller.LifeController(snapshot=LifeSnapshot(ok=True, items=have, taken_at=1000))
    caption, left, right = bar_ui.format_life_needs_columns(st)
    rows = (left + '\n' + right).splitlines()
    names = [line.split(' ', 1)[0] for line in rows]
    assert len(names) == len(set(names))
    assert '箭花 249/250' in rows
    assert '增幅蘑菇 13/600' in rows
    assert '咻咻蘑菇 89/1200' in rows
    assert '強化秘藥乳化劑' not in names
    assert '自行准备' not in left + right
    assert '紙' not in names and '空瓶' not in names
    assert '需自行准备：' not in bar_ui.format_life_needs_block(st)
    assert '读数时刻' in caption


def test_total_demand_sums_shared_ingredients_and_intermediate_reserve():
    from mm_assist.life_plan.recipes import display_need_totals
    rows = display_need_totals({'煤炭': 5}, {'合金鋼錠': 3, '鋼錠': 3})
    totals = {name: (have, total) for name, have, total in rows}
    assert len(rows) == len(totals)
    assert totals['鋼錠'] == (0, 6)
    assert totals['煤炭'] == (5, 16)


def test_current_batch_keeps_p6_to_p10_on_material_and_facility_refresh(monkeypatch):
    controller.reset_controller_for_tests()
    old = LifeSnapshot(ok=True, items=planner.necessary_desired({}), taken_at=1000)
    st = controller.start_life(snapshot=old, now=1000)
    for p in st.queue[:5]:
        controller.acknowledge_done(p.id, now=1001)
    remaining = controller.get_controller().queue
    fresh = LifeSnapshot(ok=True, items={'原木': 100}, taken_at=1600,
                         facilities={'metal': tuple(QueueSlot('鋼錠', True) for _ in range(4))})
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: fresh)
    st = controller.tick_facility_progress(now=1600)
    assert [p.id for p in st.queue] == ['p6', 'p7', 'p8', 'p9', 'p10']
    assert st.queue == remaining and st.current.id == 'p6'
    assert '当前计划：p6 ' in bar_ui.format_life_plan_block(st)
    assert st.snapshot == old  # user now requires both views to stay fixed mid-batch


@pytest.mark.parametrize('fixture', ['herb_list_updated_20261006.png', 'herb_list_updated_20261005.png'])
def test_registered_mushroom_full_name_never_selects_neighbor(fixture):
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.perception.item_sense import locate_skill_item
    f = cv2.imdecode(np.fromfile(ROOT/'tests/fixtures/process_stance'/fixture, dtype=np.uint8), cv2.IMREAD_COLOR)
    hit = locate_skill_item(f, skill_id='herb', name='咻咻蘑菇')
    assert hit['ok'] and hit['src'] == 'template+name_template'
    assert 735 <= hit['click_client'][1] <= 790
    absent = f.copy()
    absent[695:825, 1000:1820] = 0
    miss = locate_skill_item(absent, skill_id='herb', name='咻咻蘑菇')
    assert not miss['ok'] and 'click_client' not in miss


def test_actual_wrong_flower_detail_frame_cannot_start_mushroom_gather(monkeypatch):
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    frame = cv2.imdecode(np.fromfile(ROOT/'tests/fixtures/ui_benchmark/herb_detail_scar_flower.png',
                                    dtype=np.uint8), cv2.IMREAD_COLOR)
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **k: {'node': 'gather_detail:傷痕花', 'zh': '技能详情-傷痕花'})
    clicks = []
    result = execute_skill_nearby(skill_id='herb', item_name='咻咻蘑菇', output='咻咻蘑菇',
        grab=lambda: frame, do_click=lambda x, y: clicks.append((x, y)) or True,
        defer_completion=True)
    assert result.status == 'wrong_gather_detail' and not result.action_issued
    assert clicks == []


def test_batch_end_waits_for_fresh_read_before_p11_to_p20(monkeypatch):
    controller.reset_controller_for_tests()
    snap = LifeSnapshot(ok=True, items=planner.necessary_desired({}), taken_at=1000)
    st = controller.start_life(snapshot=snap, now=1000)
    for n, p in enumerate(st.queue):
        st = controller.acknowledge_done(p.id, now=1001+n)
    assert st.running and st.current is None and st.queue == [] and st.refresh_requested
    calls = []
    def read(**kw):
        calls.append(kw)
        return replace(snap, taken_at=kw['now'])
    monkeypatch.setattr(controller, 'take_snapshot', read)
    st = controller.tick_facility_progress(now=1011)
    assert len(calls) == 1 and calls[0]['include_tools'] and calls[0]['include_skills']
    assert [p.id for p in st.queue] == [f'p{i}' for i in range(11, 21)]
    assert st.current.id == 'p11'
    controller.tick_facility_progress(now=1012)
    assert len(calls) == 1


def test_template_ocr_must_not_move_click_to_another_row(monkeypatch):
    from mm_assist.perception import item_sense as sense
    from mm_assist.perception.ocr import OcrLine
    from mm_assist.platform.match import MatchHit
    frame = np.zeros((1442, 1912, 3), dtype=np.uint8)
    hit = MatchHit(.99, 1125, 760, 1077, 715, 96, 89)
    monkeypatch.setattr(sense, 'ocr_image', lambda *a, **k: [OcrLine('咻咻蘑菇', .9)])
    # A bad list OCR line must not override an independently located icon row.
    monkeypatch.setattr(sense, '_ocr_lines_in_item_list', lambda *a, **k:
        [OcrLine('咻咻蘑菇', .9, [[1200, 880], [1340, 880], [1340, 920], [1200, 920]])])
    out = sense._click_from_template_hit(hit, item='咻咻蘑菇', sid='herb',
                                        frame=frame, catalog=('咻咻蘑菇', '躲躲花'))
    assert out is None or not out['ok'] or 715 <= out['click_client'][1] <= 804


@pytest.mark.parametrize('after_name', ['躲躲花', ''])
def test_wrong_or_unknown_detail_does_not_dispatch_nearby(monkeypatch, after_name):
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_plan import node_inspect
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **k: {'node': 'gather_skill:herb', 'zh': '採集藥草'})
    monkeypatch.setattr('mm_assist.perception.item_sense.locate_skill_item',
                        lambda *a, **k: {'ok': True, 'click_client': [1270, 760]})
    monkeypatch.setattr('mm_assist.perception.item_sense.locate_find_nearby',
                        lambda *a, **k: {'ok': True, 'click_client': [750, 784]})
    monkeypatch.setattr(node_inspect, '_gather_detail_named',
                        lambda f: 'gather_detail:' + after_name if after_name else 'gather_detail')
    clicks = []
    result = execute_skill_nearby(skill_id='herb', item_name='咻咻蘑菇', output='咻咻蘑菇',
        grab=lambda: np.zeros((1442, 1912, 3), dtype=np.uint8),
        do_click=lambda x, y: clicks.append((x, y)) or True, defer_completion=True)
    assert clicks == [(1270, 760)]
    assert not result.action_issued and result.status == 'wrong_gather_detail'


def test_list_moving_between_captures_never_clicks_old_coordinate(monkeypatch):
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',
                        lambda **k: {'node': 'gather_skill:herb', 'zh': '採集藥草'})
    # The repaired flow observes movement without input until two fresh frames
    # settle. Keep this target moving for the whole observation budget rather
    # than exhausting a two-frame iterator; retain both no-input assertions.
    from itertools import cycle
    hits = cycle([{'ok': True, 'click_client': [1270, 760]},
                  {'ok': True, 'click_client': [1270, 900]}])
    monkeypatch.setattr('mm_assist.perception.item_sense.locate_skill_item',
                        lambda *a, **k: next(hits))
    clicks = []
    result = execute_skill_nearby(skill_id='herb', item_name='咻咻蘑菇', output='咻咻蘑菇',
        grab=lambda: np.zeros((1442, 1912, 3), dtype=np.uint8),
        do_click=lambda x, y: clicks.append((x, y)) or True, defer_completion=True)
    assert result.status == 'item_list_moving' and not result.action_issued
    assert clicks == []


def test_failed_batch_read_does_not_rearm_or_use_old_snapshot(monkeypatch):
    controller.reset_controller_for_tests()
    snap = LifeSnapshot(ok=True, items=planner.necessary_desired({}), taken_at=1000)
    st = controller.start_life(snapshot=snap, now=1000)
    for n, p in enumerate(st.queue):
        controller.acknowledge_done(p.id, now=1001+n)
    calls = []
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k:
        calls.append(k) or LifeSnapshot(ok=False, error='test_unreadable', taken_at=k['now']))
    failed = controller.tick_facility_progress(now=1011)
    assert failed.batch_refresh_pending and failed.current is None and failed.running
    controller.tick_facility_progress(now=1012)
    assert len(calls) == 1 and controller.get_controller().queue == []
