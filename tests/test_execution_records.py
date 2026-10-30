"""Timing is observational: one plan spans polls, pauses and late callbacks."""
from contextlib import contextmanager
from types import SimpleNamespace
import json
import threading

import pytest

from mm_assist.life_plan import execution_records as records


@pytest.fixture
def clock():
    return SimpleNamespace(wall=100000., mono=10.)


@pytest.fixture
def recorder(tmp_path, clock, monkeypatch):
    rec = records.ExecutionRecorder(tmp_path / 'records.json',
        wall=lambda: clock.wall, mono=lambda: clock.mono, pid=42)
    monkeypatch.setattr(records, '_RECORDER', rec)
    return rec


def advance(clock, seconds):
    clock.wall += seconds
    clock.mono += seconds


@contextmanager
def plan(recorder, run=90000., item='p1'):
    key = recorder.begin(run, SimpleNamespace(id=item, title='采集 銅礦石', kind='gather'))
    with records.bind(key):
        yield key


def test_one_plan_accumulates_page_stage_and_stp_across_polls(recorder, clock):
    with plan(recorder) as key:
        records.phase('进入菜单')
        records.navigation(SimpleNamespace(node='system_menu', stack_ready=True,
            stack_names=('Menu',), stack_sampled_ts=clock.wall, cost_ms={}), .25)
        records.input_requested('click', [123, 456], [1920, 1440])
        advance(clock, 3)
        records.phase('等待采集结束', kind='stp_wait')
        records.stp_sample(SimpleNamespace(stp=1, fresh=True, timestamp=clock.wall, pid=42))
    advance(clock, 7)
    with plan(recorder) as same:
        assert same == key
        records.phase('等待采集结束', kind='stp_wait')
        advance(clock, 4)
        records.finish(90000., 'p1', 'completed')
    row = recorder.read()[0]
    assert row['elapsed_s'] == 14 and row['stp_wait_s'] == 11
    assert row['active_s'] == 14 and row['paused_s'] == 0
    assert row['status'] == 'completed' and len(recorder.read()) == 1
    assert any(s['node'] == 'system_menu' and s['duration_s'] == 3 for s in row['segments'])
    assert any(e.get('coords') == [123,456] for e in row['events'])
    assert row['navigation_s'] == .25
    assert json.loads(recorder.path.read_text(encoding='utf-8'))['plans'][0]['stp_wait_s'] == 11


def test_pause_is_separate_and_late_old_context_cannot_change_new_plan(recorder, clock):
    with plan(recorder) as old:
        records.phase('等待采集结束', kind='stp_wait')
    advance(clock, 5)
    records.suspend(90000., 'p1')
    advance(clock, 30)
    with records.bind(old):
        records.phase('迟到回调')
    records.resume(90000., 'p1')
    advance(clock, 2)
    with plan(recorder, run=91000., item='p1'):
        with records.bind(old):
            records.phase('旧代同编号回调')
    old_row = next(r for r in recorder.read() if r['run_started_at'] == 90000.)
    assert old_row['status'] == 'interrupted'
    assert old_row['stp_wait_s'] == 5 and old_row['paused_s'] == 30
    assert old_row['active_s'] == 7 and old_row['elapsed_s'] == 37
    assert not any('迟到' in s['phase'] or '旧代' in s['phase'] for r in recorder.read() for s in r['segments'])


def test_pointer_trace_cannot_write_into_paused_or_new_generation(recorder, clock):
    with plan(recorder) as old:
        records.pointer_trace(kind='click', samples=[{'actual_client': [20, 30]}])
    records.suspend(90000., 'p1')
    with records.bind(old):
        records.pointer_trace(kind='paused_late', samples=[])
    records.resume(90000., 'p1')
    with plan(recorder, run=91000., item='p1'):
        with records.bind(old):
            records.pointer_trace(kind='old_generation', samples=[])
        records.pointer_trace(kind='new', samples=[])
    events = [e for row in recorder.read() for e in row['events'] if e['event'] == 'pointer_trace']
    assert [e['kind'] for e in events] == ['click', 'new']


def test_retention_active_long_plan_restart_and_unknown_are_explicit(recorder, clock):
    with plan(recorder):
        records.phase('等待页面就绪', kind='page_wait')
    advance(clock, 12*3600+1)
    rows = recorder.read()
    assert len(rows) == 1 and rows[0]['status'] == 'running'
    assert rows[0]['segments'][0]['node'] == 'unknown'
    records.finish(90000., 'p1', 'skipped', 'unknown_page')
    advance(clock, 12*3600+1)
    assert recorder.read() == []
    with plan(recorder, run=92000.):
        advance(clock, 2)
        recorder.checkpoint(force=True)
    reopened = records.ExecutionRecorder(recorder.path, wall=lambda:clock.wall,
        mono=lambda:clock.mono, pid=43)
    row = reopened.read()[0]
    assert row['status'] == 'interrupted' and row['pid'] == 42
    assert row['elapsed_s'] == 2  # no invented execution after the last checkpoint


def test_monotonic_elapsed_and_io_failure_do_not_break_executor(recorder, clock, monkeypatch):
    with plan(recorder):
        clock.wall -= 1000
        clock.mono += 5
        records.phase('下一步')
        assert recorder.read()[0]['elapsed_s'] == 5
        monkeypatch.setattr(recorder, '_save', lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))
        records.phase('仍可执行')
        records.finish(90000., 'p1', 'completed')


def test_pending_completion_uses_real_pump_and_no_second_action(recorder, clock, monkeypatch):
    from mm_assist.life_plan import controller, bar_ui, action_finish
    from mm_assist.life_plan.types import LifeSnapshot, PlanItem, ExecuteResult
    controller.reset_controller_for_tests()
    item = PlanItem('p1','gather','采集 銅礦石', recipe='銅礦石')
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:[item])
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly',lambda **k:None)
    monkeypatch.setattr('mm_assist.life_plan.gather_repair.GatherRepairWatch.check',lambda self:False)
    monkeypatch.setattr('mm_assist.life_execute.ui_act.pump_aborted',lambda:False)
    controller.start_life(snapshot=LifeSnapshot(ok=True),now=90000.)
    calls=[]
    def execute(*a, **kw):
        calls.append(True)
        kw['on_progress']('进入菜单')
        advance(clock,3)
        kw['on_action_issued']()
        return ExecuteResult(False,'action_pending',action_issued=True)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',execute)
    monkeypatch.setattr(action_finish,'read_completion_stp',lambda: action_finish.StpSample(0, clock.wall, 42, True))
    monkeypatch.setattr(bar_ui.time,'monotonic',lambda:clock.mono)
    try:
        bar_ui.pump_life_once(None,None)
        advance(clock,2)
        bar_ui.pump_life_once(None,None)
        advance(clock,2)
        bar_ui.pump_life_once(None,None)
        row=recorder.read()[0]
        assert calls == [True] and row['status'] == 'completed'
        assert row['elapsed_s'] == 7 and row['stp_wait_s'] == 4
    finally:
        controller.reset_controller_for_tests()


def test_old_thread_after_new_start_cannot_append_to_new_record(recorder, clock):
    ready=threading.Event();go=threading.Event()
    with plan(recorder) as key:
        def worker():
            with records.bind(key):
                ready.set();assert go.wait(3)
                records.phase('旧线程结束')
        thread=threading.Thread(target=worker);thread.start()
    assert ready.wait(3)
    with plan(recorder,run=91000.):
        go.set();thread.join(3)
        assert not thread.is_alive()
    assert not any(s['phase']=='旧线程结束' for r in recorder.read() for s in r['segments'])


def test_ui_checkpoint_flushes_blocked_worker_and_keeps_repeated_input_requests(recorder, clock):
    with plan(recorder):
        records.phase('等待页面就绪')
        records.input_requested('click', [123,456], [1920,1440])
        records.input_requested('click', [123,456], [1920,1440])
    advance(clock, 17)
    # No executor callback: the existing UI timer is the only observer here.
    records.checkpoint()
    row=json.loads(recorder.path.read_text(encoding='utf-8'))['plans'][0]
    assert row['elapsed_s']==17 and row['stp_wait_s']==0
    assert len([e for e in row['events'] if e['event']=='input_requested'])==2


def test_corrupt_row_does_not_hide_valid_record_or_break_viewer(recorder, clock):
    with plan(recorder):
        records.finish(90000.,'p1','completed')
    saved=json.loads(recorder.path.read_text(encoding='utf-8'))
    saved['plans'] += [{'updated_at':clock.wall,'segments':[None],'events':[]},None]
    recorder.path.write_text(json.dumps(saved),encoding='utf-8')
    reopened=records.ExecutionRecorder(recorder.path,wall=lambda:clock.wall)
    assert len(reopened.read())==1
    assert '最后观测界面' in records.format_detail(reopened.read()[0])


def test_recorder_initialization_failure_and_offline_default_cannot_block_pump(tmp_path, monkeypatch):
    from mm_assist.life_plan import controller
    monkeypatch.setattr(records,'_RECORDER',None)
    monkeypatch.setattr(records,'PATH',tmp_path/'formal.json')
    assert records._get() is None  # pytest does not create production records
    records.checkpoint()
    assert not records.PATH.exists()
    monkeypatch.delenv('PYTEST_CURRENT_TEST')
    monkeypatch.setattr(records,'ExecutionRecorder',lambda *a: (_ for _ in ()).throw(OSError('denied')))
    monkeypatch.setattr(controller,'get_controller',lambda:SimpleNamespace(running=True,current=object()))
    @records.trace_pump
    def pump():
        return 'unchanged'
    assert pump()=='unchanged'
    records.interrupt_current()
    records.suspend(90000.,'p1')
    assert not records.PATH.exists()


def test_blocking_gather_wait_times_only_wait_and_restores_stage(recorder, clock):
    from mm_assist.life_execute.gather_wait import wait_gather_idle, GatherObservation
    with plan(recorder):
        records.phase('准备采集')
        result=wait_gather_idle(observe=lambda:GatherObservation(0,'main_hud',True),
            already_started=True,issued=True,now_fn=lambda:clock.mono,
            sleep_fn=lambda _:advance(clock,1),hold_s=2)
        assert result.ok
        advance(clock,3)
        records.finish(90000.,'p1','completed')
    row=recorder.read()[0]
    assert row['stp_wait_s']==2 and row['elapsed_s']==5
    assert row['segments'][-1]['phase']=='准备采集'


def test_wait_exception_preserves_executor_exception_and_restores_stage(recorder, clock):
    @records.trace_wait('等待停钮')
    def waiting():
        advance(clock,4)
        raise ValueError('original')
    with plan(recorder):
        records.phase('进入加工')
        with pytest.raises(ValueError,match='original'):
            waiting()
        advance(clock,1)
        records.finish(90000.,'p1','skipped','original')
    row=recorder.read()[0]
    assert row['stp_wait_s']==4 and row['elapsed_s']==5
