"""Free processing slots with gatherable shortages must precede skill batches."""
from mm_assist.life_plan import controller, planner
from mm_assist.life_plan.types import LifeSnapshot, QueueSlot


def missing_stock():
    return dict(planner.necessary_desired({}), 高級木材=55, 高級布料=22,
                高級原木=8, 木材=27, **{'木材+': 38, '布料+': 11},
                布料=6, 羊毛=2, 高級羊毛=66, 樹液=2)


def test_user_empty_wood_and_cloth_slots_form_supply_batch_without_training():
    snap = LifeSnapshot(ok=True, items=missing_stock(), facilities={
        'wood': (QueueSlot('高級木材', False),),
        'cloth': (QueueSlot('高級布料', False),),
    })
    plan = planner.plan_life(snap)
    assert len(plan) == 10
    assert not any(p.gather_mode == 'skill' for p in plan)
    assert {'高級原木', '羊毛'} <= {p.recipe for p in plan if p.gather_mode == 'process_mat'}
    assert [p.id for p in plan] == [f'p{i}' for i in range(1, len(plan) + 1)]


def test_supply_batch_end_requires_new_snapshot_then_feeds_empty_wood_slots(monkeypatch):
    controller.reset_controller_for_tests()
    try:
        have = dict(planner.necessary_desired({}), 高級木材=0, 高級原木=0,
                    木材=100, 樹液=100, **{'木材+': 100})
        st = controller.start_life(snapshot=LifeSnapshot(ok=True, items=have, taken_at=1000), now=1000)
        assert len(st.queue) == 10 and not any(p.gather_mode == 'skill' for p in st.queue)
        old_ids = [int(p.id[1:]) for p in st.queue]
        for item in list(st.queue):
            controller.acknowledge_done(item.id, now=1001)
        st = controller.get_controller()
        assert not st.queue and st.refresh_requested
        assert st.running and st.current is None
        # Still short after a batch: fill another ten supply actions,
        # without trusting expected output or adding skill placeholders.
        still_short = LifeSnapshot(ok=True, items={**have, '高級原木': 1}, taken_at=1002)
        monkeypatch.setattr(controller, 'take_snapshot', lambda **kwargs: still_short)
        st = controller.tick_facility_progress(now=1002)
        assert len(st.queue) == 10 and not any(p.gather_mode == 'skill' for p in st.queue)
        for item in list(st.queue):
            controller.acknowledge_done(item.id, now=1003)
        fresh = LifeSnapshot(ok=True, items={**have, '高級原木': 200}, taken_at=1004)
        monkeypatch.setattr(controller, 'take_snapshot', lambda **kwargs: fresh)
        st = controller.tick_facility_progress(now=1004)
        assert st.queue[0].kind == 'feed' and st.queue[0].facility == 'wood'
        assert int(st.queue[0].id[1:]) > max(old_ids)
        assert sum(p.kind == 'feed' and p.facility == 'wood' for p in st.queue) == 5
    finally:
        controller.reset_controller_for_tests()


def test_processing_full_or_manual_only_shortage_does_not_stall_training():
    have = missing_stock()
    full = {fac: tuple(QueueSlot(recipe, False) for _ in range(5)) for fac, recipe in
            [('wood', '高級木材'), ('cloth', '高級布料')]}
    plan = planner.plan_life(LifeSnapshot(ok=True, items=have, facilities=full))
    assert any(p.gather_mode == 'skill' for p in plan)


def test_supply_padding_groups_materials_preserving_feed_priority():
    snap = LifeSnapshot(ok=True, items=dict(missing_stock(), 高級原木=38, 樹液=18), facilities={
        'wood': (QueueSlot('高級木材', False),),
        'cloth': (QueueSlot('高級布料', False),),
    })
    plan = planner.plan_life(snap, start_id=21)
    assert len(plan) == 10
    assert [p.id for p in plan] == [f'p{i}' for i in range(21, 31)]
    supply = [p for p in plan if p.gather_mode == 'process_mat']
    targets = list(dict.fromkeys(p.recipe for p in supply))
    assert len(targets) >= 2
    assert [p.recipe for p in supply] == ['高級原木'] * 4 + ['羊毛'] * 4
    first_supply = next(i for i, p in enumerate(plan) if p.gather_mode == 'process_mat')
    assert first_supply == 2
    assert all(p.kind == 'feed' for p in plan[:first_supply])
    assert not any(p.kind == 'feed' for p in plan[first_supply:])


def test_filtered_supply_groups_padding_with_each_remaining_material():
    have = dict(missing_stock(), 合金鋼錠=0)
    snap = LifeSnapshot(ok=True, items=have, taken_at=1000)
    state = controller.LifeController(snapshot=snap, running=True, seq=21,
                                     unsupported={'gather|高級原木'})
    controller._top_up(state, snap)
    assert len(state.queue) == 10
    assert all(p.kind == 'gather' for p in state.queue)
    assert [p.recipe for p in state.queue] == ['鐵礦石'] * 6 + ['羊毛'] * 4
    assert [int(p.id[1:]) for p in state.queue] == sorted(int(p.id[1:]) for p in state.queue)
    assert len({p.id for p in state.queue}) == 10
    assert state.current == state.queue[0]


def test_filtered_supply_replenishes_only_remaining_supported_material():
    snap = LifeSnapshot(ok=True, items=missing_stock(), taken_at=1000)
    state = controller.LifeController(snapshot=snap, running=True, seq=21,
                                     unsupported={'gather|高級原木'})
    controller._top_up(state, snap)
    assert len(state.queue) == 10
    assert [p.id for p in state.queue] == sorted([p.id for p in state.queue], key=lambda x: int(x[1:]))
    assert len(set(p.id for p in state.queue)) == 10
    assert all(p.recipe != '高級原木' for p in state.queue)
    assert not any(p.gather_mode == 'skill' for p in state.queue)


def test_supply_refresh_keeps_remaining_batch_without_new_padding(monkeypatch):
    controller.reset_controller_for_tests()
    try:
        snap = LifeSnapshot(ok=True, items=missing_stock(), taken_at=1000)
        st = controller.start_life(snapshot=snap, now=1000)
        assert len(st.queue) == 10
        st = controller.acknowledge_done(st.current.id, now=1001)
        remaining = list(st.queue)
        fresh = LifeSnapshot(ok=True, items=planner.necessary_desired({}), taken_at=1002)
        monkeypatch.setattr(controller, 'take_snapshot', lambda **kw: fresh)
        st = controller.tick_facility_progress(now=1002)
        assert st.queue == remaining and len(st.queue) == 9
    finally:
        controller.reset_controller_for_tests()


def test_unsupported_supply_is_filtered_before_selecting_available_training():
    have = dict(planner.necessary_desired({}), 高級木材=0, 高級原木=0,
                木材=100, 樹液=100, **{'木材+': 100})
    snap = LifeSnapshot(ok=True, items=have, taken_at=1000)
    state = controller.LifeController(snapshot=snap, running=True,
                                      unsupported={'gather|高級原木'})
    controller._top_up(state, snap)
    assert len(state.queue) == 10
    assert state.queue[0].kind == 'feed' and state.queue[0].facility == 'wood'
    assert all(p.gather_mode == 'skill' for p in state.queue[1:])
    assert not any(p.gather_mode == 'process_mat' for p in state.queue)
    manual = dict(planner.necessary_desired({}), 增幅蘑菇汁液=0, 紙=0, 空瓶=0)
    plan = planner.plan_life(LifeSnapshot(ok=True, items=manual))
    assert any(p.gather_mode == 'skill' for p in plan)
