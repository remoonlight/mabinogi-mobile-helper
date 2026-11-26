"""A visible batch must finish before either whole-bag view is refreshed."""
from dataclasses import replace

import pytest

from mm_assist.life_plan import controller, planner, facility_sim
from mm_assist.life_plan.types import LifeSnapshot, PlanItem, QueueSlot


@pytest.fixture(autouse=True)
def reset():
    controller.reset_controller_for_tests()
    yield
    controller.reset_controller_for_tests()


@pytest.mark.parametrize('kind', ['material', 'claim', 'training'])
def test_ten_items_defer_requested_periodic_and_forced_scans(monkeypatch, kind):
    items = planner.necessary_desired({})
    if kind == 'material':
        items['箭花'] = 0
    slots = {'wood': tuple(QueueSlot('木材', True) for _ in range(5))} if kind == 'claim' else {}
    old = LifeSnapshot(ok=True, items=items, facilities=slots, taken_at=1000)
    state = controller.start_life(snapshot=old, now=1000)
    original = list(state.queue)
    assert len(original) == 10
    reads = []
    monkeypatch.setattr(controller, 'take_snapshot', lambda **kw:
                        reads.append(kw) or replace(old, taken_at=kw['now']))
    for n, item in enumerate(original):
        state = controller.tick_facility_progress(now=1600 + n, force_scan=True)
        assert not reads and state.snapshot.items == old.items
        assert facility_sim.peek_facility_sim().last_scan_at == 1000
        assert state.current.id == item.id
        state = controller.acknowledge_done(item.id, now=1600.5 + n)
        assert [p.id for p in state.queue] == [p.id for p in original[n + 1:]]
        if n < 9:
            assert not state.refresh_requested
    assert state.current is None and state.refresh_requested
    assert not reads
    state = controller.tick_facility_progress(now=1611)
    assert len(reads) == 1 and reads[0]['include_tools'] and reads[0]['include_skills']
    assert len(state.queue) == 10 and not state.refresh_requested
    controller.tick_facility_progress(now=2211, force_scan=True)
    assert len(reads) == 1


def test_pause_and_new_start_keep_batch_snapshot(monkeypatch):
    old = LifeSnapshot(ok=True, items=planner.necessary_desired({}), taken_at=1000)
    state = controller.start_life(snapshot=old, now=1000)
    monkeypatch.setattr(controller, 'take_snapshot', lambda **kw: pytest.fail('batch read'))
    controller.pause_life()
    assert controller.tick_facility_progress(now=1700, force_scan=True).snapshot == state.snapshot
    fresh = replace(old, taken_at=2000)
    started = controller.start_life(snapshot=fresh, now=2000)
    assert controller.tick_facility_progress(now=2700).snapshot == started.snapshot


def test_capacity_upgrade_replans_at_batch_boundary(monkeypatch):
    stock = {**planner.necessary_desired({}), '合金鋼錠': 0,
             '鋼錠': 30, '銅礦石': 200, '煤炭': 80}
    old = LifeSnapshot(ok=True, items=stock, facilities={'metal': (QueueSlot('合金鋼錠', False),) * 5},
                       taken_at=1000, facility_slot_caps={'metal': 5})
    state = controller.start_life(snapshot=old, now=1000)
    first_batch = list(state.queue)
    assert len(first_batch) == 10 and all(p.kind != 'feed' for p in first_batch)
    calls = []

    def refresh(**kw):
        calls.append(kw['now'])
        return replace(old, taken_at=kw['now'], facility_slot_caps={'metal': 6})

    monkeypatch.setattr(controller, 'take_snapshot', refresh)
    assert controller.tick_facility_progress(now=1001, force_scan=True).queue == first_batch
    assert not calls
    for idx, item in enumerate(first_batch):
        controller.acknowledge_done(item.id, now=1010 + idx)
    new = controller.tick_facility_progress(now=1020)
    assert calls == [1020]
    assert [(p.facility, p.recipe) for p in new.queue if p.kind == 'feed'] == [('metal', '合金鋼錠')]
