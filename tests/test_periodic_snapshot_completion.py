"""A delayed read remains one read and must finish through the real scheduler."""
import threading

import pytest

from mm_assist.life_plan import bar_ui, controller, facility_sim, snapshot
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem, QueueSlot, ToolDurability


def test_tool_only_read_waits_for_shared_snapshot_lock(monkeypatch):
    attempted, read = threading.Event(), threading.Event()
    monkeypatch.setattr(snapshot, 'find_game_pid', lambda: 1)
    monkeypatch.setattr(snapshot, 'open_process', lambda pid: 7)
    monkeypatch.setattr(snapshot, 'close_handle', lambda h: None)
    monkeypatch.setattr(snapshot, 'read_equipped_tools', lambda *a: read.set() or ())
    result = []
    def work():
        attempted.set()
        result.append(snapshot.refresh_tools())
    worker = threading.Thread(target=work)
    try:
        with snapshot._SCAN_LOCK:
            worker.start()
            assert attempted.wait(1)
            assert not read.wait(.03)
    finally:
        worker.join(2)
    assert not worker.is_alive() and read.is_set() and result == [()]


def exhaust(monkeypatch):
    monkeypatch.setattr(controller.time, 'time', lambda: 1000.5)
    controller.reset_controller_for_tests()
    item = PlanItem('p1', 'gather', '采集 羊毛', recipe='羊毛')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item])
    monkeypatch.setattr(controller, 'training_item', lambda *a, **k: item)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly', lambda **k: None)
    controller.start_life(snapshot=LifeSnapshot(ok=True, items={'羊毛': 10},
        tools=(ToolDurability('鐮刀', 10),), taken_at=1000), now=1000)
    return controller.skip_current(reason='failure')


@pytest.mark.parametrize('failure', [False, True])
def test_exhaustion_delayed_refresh_reads_materials_and_tools_once(monkeypatch, failure):
    empty = exhaust(monkeypatch)
    assert empty.running and not empty.queue
    calls = []
    new = LifeSnapshot(ok=not failure, items={'羊毛': 40},
        tools=(ToolDurability('鐮刀', .5),), taken_at=1001,
        facilities={'cloth': (QueueSlot('布料', False, remain_s=40),)},
        error='facility_unreadable' if failure else '')
    def body(*args, **kwargs):
        calls.append(kwargs)
        return new
    monkeypatch.setattr(snapshot, '_snapshot_body', body)
    real_take = snapshot.take_snapshot
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: real_take(**dict(k, budget_s=.01)))
    try:
        with snapshot.heavy_read_lock:
            waiting = controller.tick_facility_progress(now=1001)
            assert waiting.snapshot.error == 'mem_pending'
            controller.tick_facility_progress(now=1001.1)
            assert not calls
        with snapshot._SCAN_LOCK:
            pass
        done = controller.tick_facility_progress(now=1002)
        assert len(calls) == 1 and calls[0]['include_tools'] is True
        assert done.snapshot.ok is (not failure)
        if not failure:
            assert done.snapshot.items == new.items and done.snapshot.tools == new.tools
            assert facility_sim.peek_facility_sim().slots['cloth'] == new.facilities['cloth']
            assert any(item.kind == 'repair' for item in done.queue)
        else:
            assert not done.queue and not done.paused
        controller.tick_facility_progress(now=1003)
        assert len(calls) == 1
    finally:
        with snapshot._SCAN_LOCK:
            pass
        controller.reset_controller_for_tests()


def test_exhaustion_success_without_eligible_plans_does_not_loop(monkeypatch):
    exhaust(monkeypatch)
    controller._STATE.unsupported.add('gather|羊毛')
    reads = []
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: reads.append(k) or
        LifeSnapshot(ok=True, taken_at=k['now']))
    try:
        assert not controller.tick_facility_progress(now=1001).queue
        controller.tick_facility_progress(now=1002)
        assert len(reads) == 1
        controller.tick_facility_progress(now=1601)
        assert len(reads) == 2
    finally:
        controller.reset_controller_for_tests()


def test_exhaustion_pause_preserves_refresh_request_until_resume(monkeypatch):
    exhaust(monkeypatch)
    reads = []
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: reads.append(k) or
        LifeSnapshot(ok=True, taken_at=k['now']))
    try:
        controller.pause_life()
        controller.tick_facility_progress(now=1001)
        assert not reads
        resumed = controller.resume_life(now=1002)
        assert resumed.running and not resumed.paused and resumed.life_started_at == 1000
        assert controller.tick_facility_progress(now=1002).queue
        assert len(reads) == 1
    finally:
        controller.reset_controller_for_tests()


def test_exhaustion_refresh_uses_real_snapshot_readers_in_one_handle(monkeypatch):
    exhaust(monkeypatch)
    calls = []
    handle = object()
    monkeypatch.setattr(snapshot, 'find_game_pid', lambda: 7)
    monkeypatch.setattr(snapshot, 'open_process', lambda pid: handle)
    monkeypatch.setattr(snapshot, 'prepare_hints', lambda pid: True)
    monkeypatch.setattr(snapshot, 'close_handle', lambda h: calls.append(('close', h)))
    def reader(name, result):
        def read(h, pid):
            calls.append((name, h))
            assert pid == 7
            return result
        return read
    monkeypatch.setattr(snapshot, 'read_consumable_counts', reader('items', {'羊毛': 40}))
    tools = (ToolDurability('鐮刀', .5),)
    monkeypatch.setattr(snapshot, 'read_equipped_tools', reader('tools', tools))
    queues = {'metal': (), 'wood': (), 'medicine': (),
              'cloth': (QueueSlot('布料', False, remain_s=40),)}
    monkeypatch.setattr(snapshot, 'read_facility_queues', reader('facilities', queues))
    caps = {'metal': 5, 'wood': 5, 'cloth': 6, 'medicine': 4}
    monkeypatch.setattr(snapshot, 'read_facility_slot_caps', reader('facility_slot_caps', caps))
    monkeypatch.setattr(snapshot, 'read_living_skills', reader('skills', [
        {'name': '採礦', 'level': 20, 'exp': 1}]))
    monkeypatch.setattr(controller, 'take_snapshot', snapshot.take_snapshot)
    try:
        done = controller.tick_facility_progress(now=1001)
        assert done.snapshot.items == {'羊毛': 40} and done.snapshot.tools == tools
        assert done.snapshot.facilities == queues
        assert done.snapshot.facility_slot_caps == caps
        assert facility_sim.peek_facility_sim().slots['cloth'] == queues['cloth']
        assert done.snapshot.skills[0].name == '採礦'
        assert calls == [(name, handle) for name in
                         ['items', 'tools', 'facilities', 'facility_slot_caps', 'skills', 'close']]
        assert any(item.kind == 'repair' for item in done.queue)
    finally:
        with snapshot._SCAN_LOCK:
            pass
        controller.reset_controller_for_tests()


def setup(monkeypatch):
    controller.reset_controller_for_tests()
    plans = [PlanItem('p1', 'gather', '采集 羊毛', recipe='羊毛'),
             PlanItem('p2', 'feed', '投料 布料', recipe='布料', facility='cloth')]
    old = LifeSnapshot(ok=True, items={'羊毛': 10}, facilities={'cloth': ()}, taken_at=1000)
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: list(plans))
    controller.start_life(snapshot=old, now=1000)
    real_take = snapshot.take_snapshot
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: real_take(**dict(k, budget_s=.01)))
    calls = []
    new = LifeSnapshot(ok=True, items={'羊毛': 40}, facilities={
        'cloth': (QueueSlot('布料', False, remain_s=40),)}, taken_at=1600)
    monkeypatch.setattr(snapshot, '_snapshot_body', lambda *a, **k: calls.append(1) or new)
    return plans, old, new, calls


def request_batch_refresh():
    # 2026-10-10: asynchronous full reads are only legal at a batch boundary.
    controller._STATE.queue = []
    controller._STATE.current = None
    controller._STATE.batch_refresh_pending = True
    controller._STATE.refresh_requested = True
    controller._STATE.refresh_after = 1000.5


@pytest.mark.parametrize('failure', [False, True])
def test_lock_delayed_periodic_read_is_consumed_once(monkeypatch, failure):
    plans, old, new, calls = setup(monkeypatch)
    request_batch_refresh()
    if failure:
        new = LifeSnapshot(ok=False, error='facility_unreadable', taken_at=1600)
        monkeypatch.setattr(snapshot, '_snapshot_body', lambda *a, **k: calls.append(1) or new)
    try:
        with snapshot.heavy_read_lock:
            first = controller.tick_facility_progress(now=1600)
            assert not first.snapshot.ok
            # Waiting is shown separately from a completed failure.
            assert '内存读取失败' not in bar_ui.format_life_needs_block(first)
            assert '扫描中' in bar_ui.format_life_process_block(first)
            monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda *a, **k: pytest.fail('pending scan cannot dispatch'))
            bar_ui.pump_life_once(None, None)
            controller.tick_facility_progress(now=1600.1)
            assert not calls
        with snapshot._SCAN_LOCK:
            pass  # Wait for the real worker to exit before assertions/teardown.
        done = controller.tick_facility_progress(now=1601)
        assert done.snapshot.ok is (not failure)
        assert done.snapshot.error == new.error
        if not failure:
            assert done.snapshot.items == new.items
            assert facility_sim.peek_facility_sim().slots['cloth'] == new.facilities['cloth']
        else:
            assert not facility_sim.facility_sim_ok()
            monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda *a, **k: pytest.fail('failed read cannot dispatch'))
            bar_ui.pump_life_once(None, None)
        controller.tick_facility_progress(now=1602)
        assert calls == [1]
        if not failure:
            assert [p.kind for p in controller.get_controller().queue[:2]] == [p.kind for p in plans]
        else:
            assert controller.get_controller().queue == []
    finally:
        # This also handles a red assertion before the worker releases its lock.
        with snapshot._SCAN_LOCK:
            pass
        controller.reset_controller_for_tests()


def test_old_periodic_completion_cannot_overwrite_new_start(monkeypatch):
    plans, old, new, calls = setup(monkeypatch)
    request_batch_refresh()
    try:
        with snapshot.heavy_read_lock:
            controller.tick_facility_progress(now=1600)
            replacement = LifeSnapshot(ok=True, items={'羊毛': 99}, facilities={}, taken_at=1600.5)
            controller.start_life(snapshot=replacement, now=1600.5)
        with snapshot._SCAN_LOCK:
            pass
        state = controller.tick_facility_progress(now=1601)
        assert state.snapshot.items == replacement.items
        assert state.life_started_at == 1600.5
        assert calls == [1]
    finally:
        with snapshot._SCAN_LOCK:
            pass
        controller.reset_controller_for_tests()


def test_issued_gather_completion_ignores_periodic_read_failure(monkeypatch):
    plans, old, new, calls = setup(monkeypatch)
    controller.wait_action_completion('p1', 1000, ExecuteResult(True, 'done', action_issued=True), actions_complete=True)
    controller._STATE.snapshot = LifeSnapshot(ok=False, error='mem_budget')
    from mm_assist.life_plan import action_finish
    # Use the real 4-second completion logic, with only external samples replaced.
    now = [10.0]
    monkeypatch.setattr(action_finish, 'read_completion_stp', lambda: action_finish.StpSample(0, now[0], 1, True))
    monkeypatch.setattr(bar_ui.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr('mm_assist.life_execute.ui_act.pump_aborted', lambda: False)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda *a, **k: pytest.fail('must not redispatch'))
    for value in [10, 11, 12, 13]:
        now[0] = value
        bar_ui.pump_life_once(None, None)
        assert controller.get_controller().current == plans[0]
    now[0] = 14
    bar_ui.pump_life_once(None, None)
    assert controller.get_controller().current == plans[1]
    assert controller.get_controller().pending_action is None
    assert not calls
    controller.reset_controller_for_tests()


def test_delayed_reader_exception_is_published_and_releases_lock(monkeypatch):
    plans, old, new, calls = setup(monkeypatch)
    request_batch_refresh()
    def broken(*a, **k):
        raise RuntimeError('reader_broke')
    monkeypatch.setattr(snapshot, '_snapshot_body', broken)
    try:
        with snapshot.heavy_read_lock:
            controller.tick_facility_progress(now=1600)
        with snapshot._SCAN_LOCK:
            pass
        failed = controller.tick_facility_progress(now=1601)
        assert not failed.snapshot.ok
        assert failed.snapshot.error == 'mem_error:reader_broke'
        assert not snapshot.snapshot_in_progress()
        monkeypatch.setattr(snapshot, '_snapshot_body', lambda *a, **k: new)
        assert controller.tick_facility_progress(now=2201).snapshot.ok
    finally:
        with snapshot._SCAN_LOCK:
            pass
        controller.reset_controller_for_tests()


def test_pause_holds_completion_until_same_run_resumes(monkeypatch):
    plans, old, new, calls = setup(monkeypatch)
    request_batch_refresh()
    try:
        with snapshot.heavy_read_lock:
            waiting = controller.tick_facility_progress(now=1600)
            controller.pause_life()
        with snapshot._SCAN_LOCK:
            pass
        paused = controller.tick_facility_progress(now=1601)
        assert paused.paused and not paused.snapshot.ok
        assert paused.queue == waiting.queue
        resumed = controller.resume_life(now=1602)
        assert resumed.running and resumed.life_started_at == 1000
        resumed = controller.tick_facility_progress(now=1602)
        assert resumed.snapshot.ok
        assert resumed.snapshot.items == new.items
        assert calls == [1]
    finally:
        with snapshot._SCAN_LOCK:
            pass
        controller.reset_controller_for_tests()


def test_forced_refresh_waits_for_other_reader_without_losing_request(monkeypatch):
    plans, old, new, calls = setup(monkeypatch)
    request_batch_refresh()
    try:
        with snapshot.heavy_read_lock:
            assert snapshot.take_snapshot(budget_s=.01).error == 'mem_budget'
            pending = controller.tick_facility_progress(now=1001, force_scan=True)
            assert pending.snapshot.error == 'mem_pending'
            controller.tick_facility_progress(now=1001.1)
            assert not calls
        with snapshot._SCAN_LOCK:
            pass
        # Complete before the normal 600-second deadline, without another force.
        done = controller.tick_facility_progress(now=1002)
        assert done.snapshot.ok and done.snapshot.items == new.items
        assert calls == [1, 1]  # One other caller's read and one periodic read.
        controller.tick_facility_progress(now=1003)
        assert len(calls) == 2
    finally:
        with snapshot._SCAN_LOCK:
            pass
        controller.reset_controller_for_tests()


def test_last_gather_finishes_while_read_pending_then_replans_from_new_data(monkeypatch):
    plans, old, new, calls = setup(monkeypatch)
    controller._STATE.queue = plans[:1]
    monkeypatch.setattr(controller.time, 'time', lambda: 1599.5)
    controller.wait_action_completion('p1', 1000, ExecuteResult(True, 'done', action_issued=True), actions_complete=True)
    from mm_assist.life_plan import action_finish
    now = [10.0]
    monkeypatch.setattr(bar_ui.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(action_finish, 'read_completion_stp', lambda: action_finish.StpSample(0, now[0], 1, True))
    monkeypatch.setattr('mm_assist.life_execute.ui_act.pump_aborted', lambda: False)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', lambda *a, **k: pytest.fail('must not redispatch'))
    try:
        with snapshot.heavy_read_lock:
            controller.tick_facility_progress(now=1600)
            for value in [10, 11, 12, 13, 14]:
                now[0] = value
                bar_ui.pump_life_once(None, None)
            state = controller.get_controller()
            assert state.pending_action is None and not state.queue
            assert state.running and not state.start_blocked
            assert not calls
            controller.tick_facility_progress(now=1600.5)
        with snapshot._SCAN_LOCK:
            pass
        completed = controller.tick_facility_progress(now=1601)
        assert completed.running and completed.queue
        assert completed.snapshot.items == new.items
        assert calls == [1]
    finally:
        with snapshot._SCAN_LOCK:
            pass
        controller.reset_controller_for_tests()
