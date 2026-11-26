"""Current season training caps through the real planner and controller."""
from dataclasses import replace

import pytest

from mm_assist.life_plan import controller, planner
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem, SkillGrowth


def full_snapshot(harvest=35, hoe=35):
    from mm_assist.life_plan.recipes import SKILL_GATHER

    levels = {'收割': harvest, '鋤地': hoe, '日常採集': 30}
    return LifeSnapshot(
        ok=True, items=planner.necessary_desired({}), taken_at=1000,
        skills=tuple(SkillGrowth(skill, levels.get(skill, 35), 0) for skill, _ in SKILL_GATHER),
    )


@pytest.fixture(autouse=True)
def isolated():
    controller.reset_controller_for_tests()
    yield
    controller.reset_controller_for_tests()


@pytest.mark.parametrize('level', [30, 34, 35, 36])
def test_only_unfinished_skill_can_be_selected_despite_round_counts(level):
    snap = full_snapshot(harvest=level)
    plan = planner.plan_life(snap, training_counts={'收割': 20})
    if level < 35:
        assert len(plan) == 10
        assert all(p.skill == '收割' and p.detail == '稻米' for p in plan)
    else:
        assert plan == []


@pytest.mark.parametrize(('skill', 'threshold', 'current', 'previous'), [
    ('日常採集', 20, '蜘蛛絲團', '蜘蛛網'),
    ('收割', 20, '稻米', '黃豆'),
    ('鋤地', 20, '南瓜', '防風草'),
    ('採礦', 30, '隕鐵礦脈', '銀礦脈'),
    ('伐木', 30, '雷擊樹', '幽暮樹'),
    ('剪羊毛', 30, '雲絮羊', '烏雲羊'),
    ('採集藥草', 30, '柔柔蘑菇', '增幅蘑菇'),
    ('昆蟲採集', 25, '荒涼的昆蟲群', '荒蕪的昆蟲群'),
])
def test_highest_registered_tier_and_lower_level_fallback(skill, threshold, current, previous):
    snap = full_snapshot()
    for level, expected in ((threshold - 1, previous), (threshold, current)):
        skills = tuple(replace(row, level=level) if row.name == skill else row for row in snap.skills)
        plan = planner.plan_life(replace(snap, skills=skills))
        assert [(p.skill, p.detail) for p in plan] == [(skill, expected)] * 10


def test_daily_training_caps_at_30_in_planner_and_remaining_queue():
    capped = full_snapshot()
    before = replace(capped, skills=tuple(replace(row, level=29) if row.name == '日常採集' else row
                                          for row in capped.skills))
    state = controller.start_life(snapshot=before, now=1000)
    assert len(state.queue) == 10 and all(p.skill == '日常採集' for p in state.queue)
    controller._STATE.snapshot = capped
    controller.pause_life()
    assert controller.resume_life(now=1001).queue == []


def test_capped_held_target_switches_to_other_skill():
    held = PlanItem('old', 'gather', '练采集 黃豆', skill='收割',
                    recipe='黃豆', detail='黃豆', gather_mode='skill')
    snap = full_snapshot(hoe=34)
    assert not planner.training_target_allowed(held, snap, automatic=True)
    plan = planner.plan_life(snap, training_target=held)
    assert len(plan) == 10 and all(p.skill == '鋤地' for p in plan)


def test_capped_skills_still_allow_material_supply_and_feeding():
    snap = full_snapshot()
    items = dict(snap.items, 高級木材=0)
    material_plan = planner.plan_life(replace(snap, items=items))
    assert material_plan and all(p.gather_mode == 'process_mat' for p in material_plan)
    items.update({'木材+': 100, '高級原木': 100, '樹液': 100})
    feed_plan = planner.plan_life(replace(snap, items=items))
    assert any(p.kind == 'feed' and p.facility == 'wood' for p in feed_plan)
    assert not any(p.gather_mode == 'skill' for p in feed_plan)


@pytest.mark.parametrize('pending', ['none', 'action', 'legacy_gather'])
def test_pause_resume_filters_capped_queue_and_preserves_issued_action(pending):
    controller.start_life(snapshot=full_snapshot(34, 34), now=1000)
    state = controller._STATE
    issued = state.current
    if pending == 'action':
        controller.wait_action_completion(issued.id, 1000, ExecuteResult(True, 'done'),
                                          actions_complete=True)
    elif pending == 'legacy_gather':
        controller.mark_gather_pending(issued.id, 1000, saw_busy=True)
    state.snapshot = full_snapshot()
    controller.pause_life()
    resumed = controller.resume_life(now=1001)
    assert [p.id for p in resumed.queue] == ([] if pending == 'none' else [issued.id])
    controller._top_up(state, state.snapshot)
    assert [p.id for p in state.queue] == ([] if pending == 'none' else [issued.id])
    if pending != 'none':
        completed = controller.acknowledge_done(issued.id, now=1002)
        assert completed.queue == []
    restarted = controller.start_life(snapshot=full_snapshot(), now=2000)
    assert restarted.current is None and restarted.queue == []


def test_controller_stops_training_after_post_batch_cap_refresh(monkeypatch):
    controller.start_life(snapshot=full_snapshot(34, 34), now=1000)
    monkeypatch.setattr(controller, 'take_snapshot',
                        lambda **kwargs: replace(full_snapshot(), taken_at=kwargs['now']))
    for item in list(controller.get_controller().queue):
        controller.acknowledge_done(item.id, now=1001)
    refreshed = controller.tick_facility_progress(now=1002)
    assert refreshed.snapshot.skills == full_snapshot().skills
    assert refreshed.queue == [] and refreshed.current is None
