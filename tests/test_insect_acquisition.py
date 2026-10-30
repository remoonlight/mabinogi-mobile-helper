"""Both insect reserves use observed obtain-method entries, not guessed groups."""
from mm_assist.life_plan import planner, execute
from mm_assist.life_plan.acquisition import acquisition_plan
from mm_assist.life_plan.types import LifeSnapshot


def thirty_remaining():
    have = {'高級混合強化秘藥(山崩)': 10}
    return {**planner.necessary_desired(have), **have}


def test_insects_are_targeted_before_training_and_stop_at_reserve():
    for name, got in (('滴答螢火蟲', 11), ('獨角仙', 178)):
        have = {**thirty_remaining(), name: got}
        result = acquisition_plan(have)
        row = next(r for r in result['gather'] if r['name'] == name)
        assert row['skill'] == '昆蟲採集' and row['detail'] == '獲得方法'
        assert row['missing'] == 250 - got
        assert not any(r['name'] == name for r in result['manual_materials'])
        plan = planner.plan_life(LifeSnapshot(ok=True, items=have))
        assert plan[0].recipe == name and plan[0].gather_mode == 'process_mat'
        assert plan[0].detail == '獲得方法'
        assert all(p.gather_mode == 'skill' for p in planner.plan_life(
            LifeSnapshot(ok=True, items=planner.necessary_desired({}))))


def test_insect_dispatch_uses_material_obtain_flow_not_skill_group(monkeypatch):
    from mm_assist.life_execute import insect_obtain
    from mm_assist.life_plan.types import ExecuteResult
    seen = []
    monkeypatch.setattr(insect_obtain, 'execute_insect_obtain', lambda *a, **kw:
        (seen.append(kw['mat']) or ExecuteResult(False, 'need_context')))
    for name in ('滴答螢火蟲', '獨角仙'):
        p = planner.plan_life(LifeSnapshot(ok=True, items={**thirty_remaining(), name: 0}))[0]
        execute.try_execute(p, items={name: 0})
    assert seen == ['滴答螢火蟲', '獨角仙']


def test_each_insect_supply_completion_defers_inventory_read_until_batch_end():
    from mm_assist.life_plan import controller
    for name in ('滴答螢火蟲', '獨角仙'):
        controller.reset_controller_for_tests()
        state = controller.start_life(snapshot=LifeSnapshot(ok=True,
            items={**thirty_remaining(), name: 0}, taken_at=1000), now=1000)
        assert state.current.recipe == name
        done = controller.acknowledge_done(state.current.id, now=1001)
        assert not done.refresh_requested and not done.replan_requested
        assert done.queue == state.queue[1:] and done.snapshot == state.snapshot
    controller.reset_controller_for_tests()
