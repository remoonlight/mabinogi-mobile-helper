"""Formal scheduling: two continuous 2-second idle segments after dispatch."""
import pytest
from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem


def setup(monkeypatch, result, *, kind='gather'):
    controller.reset_controller_for_tests()
    first=PlanItem('p1',kind,'当前计划',facility='metal',recipe='合金鋼錠')
    duplicate=PlanItem('p2',kind,'下一计划',facility='metal',recipe='合金鋼錠')
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:[first,duplicate])
    controller.start_life(snapshot=LifeSnapshot(ok=True),now=1000)
    calls=[]
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',lambda item,**k:calls.append(item.id) or result)
    clock=[0.0];value=[0];fresh=[True];stamp=[1.0]
    from mm_assist.life_plan import action_finish
    monkeypatch.setattr(bar_ui.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(action_finish,'read_completion_stp',lambda:action_finish.StpSample(value[0],stamp[0],1,fresh[0]))
    def tick(t,stp=0,valid=True):
        clock[0]=t;value[0]=stp;fresh[0]=valid;stamp[0]=t+1
        bar_ui.pump_life_once(None,None)
    return first,duplicate,calls,tick


def test_dispatched_action_needs_four_seconds_and_never_repeats(monkeypatch):
    first,nxt,calls,tick=setup(monkeypatch,ExecuteResult(True,'done',action_issued=True))
    tick(0)
    for t in (1,2,3,3.99):
        tick(t)
        assert controller.get_controller().current==first
    tick(4)
    assert controller.get_controller().current==nxt
    assert calls==['p1']


@pytest.mark.parametrize('stp,valid',[(1,True),(None,True),(0,False)])
def test_busy_unknown_or_stale_resets_both_segments(monkeypatch,stp,valid):
    first,nxt,calls,tick=setup(monkeypatch,ExecuteResult(False,'gather_pending',action_issued=True))
    tick(0);tick(2);tick(3,stp,valid)
    for t in (4,5,6,7):
        tick(t)
        assert controller.get_controller().current==first
    tick(8)
    assert controller.get_controller().current==nxt
    assert calls==['p1']


def test_unissued_failure_skips_without_cooling_duplicate(monkeypatch):
    first,nxt,calls,tick=setup(monkeypatch,ExecuteResult(False,'no_mats','材料不够'))
    tick(0)
    assert first not in controller.get_controller().queue
    assert controller.get_controller().current==nxt
    tick(.1)
    assert calls==['p1','p2']


def test_pause_keeps_dispatched_action_and_restarts_idle_segments(monkeypatch):
    first,nxt,calls,tick=setup(monkeypatch,ExecuteResult(True,'done',action_issued=True))
    tick(0);tick(2);controller.pause_life();tick(6)
    assert controller.get_controller().current==first
    controller.resume_life(now=1001)
    for t in (7,8,9,10):
        tick(t)
        assert controller.get_controller().current==first
    tick(11)
    assert controller.get_controller().current==nxt
    assert calls==['p1']


def test_late_dispatch_result_does_not_modify_new_generation(monkeypatch):
    first,nxt,calls,tick=setup(monkeypatch,ExecuteResult(True,'done',action_issued=True))
    def late(item,**k):
        controller.start_life(snapshot=LifeSnapshot(ok=True),now=2000)
        return ExecuteResult(True,'done',action_issued=True)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',late)
    tick(0)
    st=controller.get_controller()
    assert st.life_started_at==2000 and st.current==first
    assert st.pending_action is None


def test_incomplete_claim_pauses_without_erasing_slots(monkeypatch):
    from mm_assist.life_plan.types import QueueSlot
    first,nxt,calls,tick=setup(monkeypatch,ExecuteResult(False,'claim_fail',action_issued=True))
    claim=PlanItem('p1','claim','领取 金屬',facility='metal')
    snap=LifeSnapshot(ok=True,facilities={'metal':(QueueSlot('鋼錠',True),)})
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:[claim,nxt])
    controller.start_life(snapshot=snap,now=1001,pause_on_failure=True)
    for t in (0,1,2,3,4): tick(t)
    st=controller.get_controller()
    assert st.current==claim and st.paused and not st.running
    assert st.pending_action is not None and not st.pending_action.actions_complete
    assert st.snapshot.facilities['metal']==snap.facilities['metal']


@pytest.mark.parametrize('kind',['feed','claim','repair'])
def test_non_gather_chain_completes_without_stp_wait(monkeypatch,kind):
    first,nxt,calls,tick=setup(monkeypatch,ExecuteResult(True,'done',action_issued=True),kind=kind)
    monkeypatch.setattr(bar_ui,'_pump_wait_hud_idle',lambda _:True)
    from mm_assist.life_plan import action_finish
    monkeypatch.setattr(action_finish,'read_completion_stp',lambda:(_ for _ in ()).throw(AssertionError('no completion stp read')))
    tick(0,1)
    st=controller.get_controller()
    assert st.current==nxt and st.pending_action is None
    assert calls==['p1']


def test_normal_incomplete_chain_skips_even_with_old_pause_config(monkeypatch):
    from types import SimpleNamespace
    first,nxt,calls,tick=setup(monkeypatch,ExecuteResult(False,'confirm_miss',action_issued=True))
    bar_ui.pump_life_once(SimpleNamespace(incomplete_plan_policy='pause'),None)
    st=controller.get_controller()
    assert st.current==nxt and first not in st.queue
    assert st.pending_action is None and st.running and not st.paused
    assert calls==['p1']


@pytest.mark.parametrize('during_capture', ['pause', 'new_start'])
def test_failure_capture_cannot_mutate_paused_or_new_plan(monkeypatch,during_capture):
    first,nxt,calls,tick=setup(monkeypatch,ExecuteResult(False,'no_mats'))
    def capture(*a,**k):
        if during_capture=='pause':controller.pause_life()
        else:controller.start_life(snapshot=LifeSnapshot(ok=True),now=2000)
        return 'original_failure.png'
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot',capture)
    tick(0)
    st=controller.get_controller()
    assert st.current==first and st.queue==[first,nxt]
    if during_capture=='pause':assert st.paused and not st.running
    else:assert st.life_started_at==2000


def test_count_starts_after_all_plan_steps_return(monkeypatch):
    from mm_assist.life_plan import action_finish
    first,nxt,calls,tick=setup(monkeypatch,ExecuteResult(True,'done',action_issued=True))
    clock=[0.0]
    monkeypatch.setattr(bar_ui.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(action_finish,'read_completion_stp',lambda:action_finish.StpSample(0,clock[0]+1,1,True))
    def chain(item,**kwargs):
        kwargs['on_action_issued']()
        assert controller.get_controller().pending_action.idle.zero_since is None
        clock[0]=10.0  # Further planned steps, including result confirmation.
        assert not controller.observe_action_completion(action_finish.StpSample(0,11,1,True),
            now=10,item_id=item.id,run_started_at=1000)
        assert controller.get_controller().current==first
        return ExecuteResult(True,'done',action_issued=True)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',chain)
    bar_ui.pump_life_once(None,None)
    assert controller.get_controller().pending_action.idle.zero_since==10
    for t in (11,12,13,13.99):
        clock[0]=t;bar_ui.pump_life_once(None,None)
        assert controller.get_controller().current==first
    clock[0]=14;bar_ui.pump_life_once(None,None)
    assert controller.get_controller().current==nxt


def test_repeated_frame_cannot_supply_two_idle_segments():
    from mm_assist.life_plan.action_finish import IdleSegments,StpSample
    idle=IdleSegments()
    for t in (0,1,2,3,4):
        assert not idle.observe(StpSample(0,1,1,True),t)
    assert idle.zero_since==3  # Lost observation continuity resets the streak.


@pytest.mark.parametrize('change', ['gap','pid','clock','stamp','invalid_clock'])
def test_observation_discontinuity_resets_four_seconds(change):
    from mm_assist.life_plan.action_finish import IdleSegments,StpSample
    idle=IdleSegments()
    assert not idle.observe(StpSample(0,10,1,True),10)
    assert not idle.observe(StpSample(0,12,1,True),12)
    at,stamp,pid=13,13,1
    if change=='gap':at=15;stamp=15
    if change=='pid':pid=2
    if change=='clock':at=11
    if change=='stamp':stamp=11
    if change=='invalid_clock':at=float('nan')
    assert not idle.observe(StpSample(0,stamp,pid,True),at)
    assert idle.zero_since==(None if change=='invalid_clock' else at)


@pytest.mark.parametrize('override', [dict(fresh=False),dict(pid_alive=False),dict(pid=999999),
    dict(raw={'ts_unix':96}),dict(raw={'ts_unix':101}),dict(raw={'ts_unix':'bad'})])
def test_completion_reader_rejects_stale_dead_foreign_and_invalid_samples(monkeypatch,override):
    import os
    from types import SimpleNamespace
    from mm_assist.life_plan import action_finish
    values=dict(fresh=True,pid_alive=True,pid=os.getpid(),stp=0,raw={'ts_unix':100})
    monkeypatch.setattr(action_finish.time,'time',lambda:100)
    monkeypatch.setattr('mm_assist.perception.state_ui.read_state_ui',lambda:SimpleNamespace(**values))
    assert action_finish.read_completion_stp().fresh
    values.update(override)
    assert not action_finish.read_completion_stp().fresh
