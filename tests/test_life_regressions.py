"""跨计划器、调度泵和加工进度验证产品行为；只替换外部输入输出。"""

import pytest

from mm_assist.life_plan import bar_ui, controller, facility_sim, planner
from mm_assist.life_plan.execute import is_nav_result, is_sense_result
from mm_assist.life_plan.types import CYCLE_SIZES, ExecuteResult, LifeSnapshot, PlanItem, QueueSlot


def snapshot():
    return LifeSnapshot(
        ok=True,
        items={},
        facilities={"metal": tuple(QueueSlot("鋼錠", True) for _ in range(3))},
    )


def setup_queue(monkeypatch, queue, snap=None, *, pause_on_failure=False):
    controller.reset_controller_for_tests()
    facility_sim.set_facility_read_for_tests(lambda: (snap or snapshot()).facilities)
    monkeypatch.setattr("mm_assist.life_plan.snapshot.refresh_tools", lambda: ())
    monkeypatch.setattr(controller, "plan_life", lambda *a, **k: list(queue))
    monkeypatch.setattr(controller, "take_snapshot", lambda **k: snap or snapshot())
    monkeypatch.setattr("mm_assist.life_execute.ui_recover.recover_once", lambda *a, **k: {"ok": True})
    return controller.start_life(snapshot=snap or snapshot(), now=1000,pause_on_failure=pause_on_failure)


def test_finished_targets_only_train_and_claim():
    snap = LifeSnapshot(ok=True, items={**CYCLE_SIZES, "鋼錠": 999, "煤炭": 999,
                                        "高級混合強化秘藥(山崩)": 100,
                                        "增幅蘑菇汁液": 150, "箭花": 250, "血紅藥草": 250,
                                        "滴答螢火蟲": 250, "獨角仙": 250})
    plan = planner.plan_life(snap)
    assert plan
    assert all(p.kind == "gather" and p.gather_mode == "skill" for p in plan)
    assert "合金鋼錠 100/100" in bar_ui.format_need_items(status=True, have=snap.items)


def test_failed_claim_is_skipped_until_new_snapshot(monkeypatch):
    claim = PlanItem('p1', 'claim', '领取 金屬', facility='metal')
    gather = PlanItem('p2', 'gather', '采集 蘑菇', recipe='淨淨蘑菇')
    st = setup_queue(monkeypatch, [claim, gather])
    bar_ui._note_fail_and_continue(win=None, item=st.current, reason='no_claim_ui')
    st = controller.tick_facility_progress(now=1001)
    assert st.current == gather and claim not in st.queue
    assert facility_sim.peek_facility_sim().slots['metal']  # No fake receipt.
    st = controller.tick_facility_progress(now=1601)
    assert st.current == gather  # Fresh data does not replace the visible batch.
    from dataclasses import replace
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(snapshot(), taken_at=k['now']))
    controller.acknowledge_done(gather.id, now=1602)
    st = controller.tick_facility_progress(now=1603)
    assert st.current.kind == 'claim' and st.current.id != claim.id


def test_successful_claim_not_recreated_from_old_slots(monkeypatch):
    from dataclasses import replace
    claim = PlanItem("p1", "claim", "领取 金屬", facility="metal")
    gather = replace(planner.training_item(snapshot()), id='p2')
    setup_queue(monkeypatch, [claim, gather])
    controller.acknowledge_done(claim.id, now=1001)
    # A successful claim requests a new authoritative read. Keep the real
    # planner and model the actual cleared slots rather than replaying old ones.
    monkeypatch.setattr(controller, 'plan_life', planner.plan_life)
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(
        snapshot(), facilities={'metal': ()}, items={**CYCLE_SIZES, '增幅蘑菇汁液': 150,
            '箭花': 250, '血紅藥草': 250, '滴答螢火蟲': 250, '獨角仙': 250}, taken_at=k['now']))
    st = controller.tick_facility_progress(now=1002)
    assert st.current == gather
    assert not any(p.kind == "claim" for p in st.queue)
    assert not facility_sim.peek_facility_sim().slots["metal"]


def test_failure_does_not_block_next_queued_duplicate(monkeypatch):
    first = PlanItem("p1", "gather", "采集 蘑菇", recipe="淨淨蘑菇")
    duplicate = PlanItem("p2", "gather", "采集 蘑菇", recipe="淨淨蘑菇")
    other = PlanItem("p3", "gather", "采集 銅礦石", recipe="銅礦石")
    st = setup_queue(monkeypatch, [first, duplicate, other], LifeSnapshot(ok=True))
    calls = []
    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", lambda item, **k: calls.append(item) or ExecuteResult(True, "done"))
    bar_ui._note_fail_and_continue(win=None, item=st.current, reason="nav_verify_timeout")
    bar_ui.pump_life_once(None, None)
    assert calls == [duplicate]


def test_failure_frame_is_saved_before_skip_without_recovery(monkeypatch):
    events = []
    first = PlanItem("p1", "gather", "采集 蘑菇", recipe="淨淨蘑菇")
    other = PlanItem("p2", "gather", "采集 銅礦石", recipe="銅礦石")
    st = setup_queue(monkeypatch, [first, other], LifeSnapshot(ok=True))
    monkeypatch.setattr("mm_assist.life_plan.anomalies.cache_fail_shot", lambda *a, **k: events.append("capture") or "failure.png")
    monkeypatch.setattr("mm_assist.life_execute.ui_recover.recover_once", lambda *a, **k: events.append("recover") or {"ok": True})
    bar_ui._note_fail_and_continue(win=None, item=st.current, reason="nav_verify_timeout")
    assert events == ["capture"]
    assert controller.get_controller().current == other
    assert first not in controller.get_controller().queue


def test_pending_action_waits_without_executing_again(monkeypatch):
    item = PlanItem('p1','gather','采集 蘑菇',recipe='淨淨蘑菇')
    setup_queue(monkeypatch, [item], LifeSnapshot(ok=True))
    controller.wait_action_completion(item.id, 1000, ExecuteResult(False,'action_pending',action_issued=True), actions_complete=True)
    calls=[]
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',lambda *a,**k: calls.append(1))
    bar_ui.pump_life_once(None,None)
    assert calls==[] and controller.get_controller().queue==[item]
    assert '等待采集结束' in controller.get_controller().current_action


def test_unissued_unknown_skips_but_busy_preserves_plan(monkeypatch):
    first=PlanItem('p1','gather','采集 蘑菇',recipe='淨淨蘑菇')
    other=PlanItem('p2','gather','采集 銅礦石',recipe='銅礦石')
    for status in ('stp_busy','unknown_ui'):
        setup_queue(monkeypatch,[first,other],LifeSnapshot(ok=True))
        monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',lambda *a,**k: ExecuteResult(False,status))
        bar_ui.pump_life_once(None,None)
        if status == 'stp_busy':
            assert controller.get_controller().current == first
            assert first in controller.get_controller().queue
        else:
            assert controller.get_controller().current==other
            assert first not in controller.get_controller().queue


def test_claim_removes_only_done_slots():
    pending = QueueSlot("鋼錠", False, end_ms=9999999999999)
    facility_sim.reset_facility_sim_for_tests()
    facility_sim.seed_facility_slots({"metal": (QueueSlot("鋼錠", True), pending)}, now=1000)
    assert facility_sim.forget_claimed_slots("metal")["metal"] == (pending,)


def test_execute_result_records_run_and_duration(monkeypatch):
    import json
    from mm_assist.life_plan import anomalies

    first = PlanItem("p1", "gather", "采集 蘑菇", recipe="淨淨蘑菇")
    other = PlanItem("p2", "gather", "采集 銅礦石", recipe="銅礦石")
    setup_queue(monkeypatch, [first, other], LifeSnapshot(ok=True))
    times = iter([10.0, 12.5, 12.5])
    monkeypatch.setattr(bar_ui.time, "monotonic", lambda: next(times))
    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", lambda *a, **k: ExecuteResult(True, "done"))
    bar_ui.pump_life_once(None, None)
    row = json.loads((anomalies.ROOT / anomalies.LOG_REL).read_text(encoding="utf-8").splitlines()[-1])
    assert row["tag"] == "execute_result"
    assert row["elapsed_s"] == 2.5
    assert row["run_started_at"] == 1000
    assert row["stage"] == "done" and row["ok"] is True


def test_select_current_preserves_fifo_without_failure_cooldown(monkeypatch):
    first=PlanItem('p1','gather','采集 蘑菇',recipe='淨淨蘑菇')
    other=PlanItem('p2','gather','采集 銅礦石',recipe='銅礦石')
    setup_queue(monkeypatch,[first,other],LifeSnapshot(ok=True))
    assert controller.select_ready_current().current==first
    controller.skip_current(reason='fail')
    st=controller.select_ready_current()
    assert st.current==other and st.queue==[other]
    assert not hasattr(st,'cooldown')


def _no_mats_snap():
    return LifeSnapshot(
        ok=True,
        items={"鋼錠": 2, "合金鋼錠": 0, "銅礦石": 0},
        facilities={
            "metal": (QueueSlot("鋼錠", True), QueueSlot("鋼錠", True)),
            "wood": (),
            "cloth": (),
            "medicine": (),
        },
    )


def test_no_mats_first_result_does_not_retry_or_recover(monkeypatch):
    import json
    from mm_assist.life_plan import anomalies

    recovers: list[str] = []
    scans: list[int] = []
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_recover.recover_once",
        lambda trigger, **k: recovers.append(str(trigger)) or {"ok": True},
    )
    feed = PlanItem("p1", "feed", "投料 合金鋼錠", facility="metal", recipe="合金鋼錠")
    gather = PlanItem("p2", "gather", "采集 銅礦石", recipe="銅礦石")
    snap = _no_mats_snap()
    st = setup_queue(monkeypatch, [feed, gather], snap)
    monkeypatch.setattr(
        controller,
        "take_snapshot",
        lambda **k: scans.append(1) or snap,
    )
    calls: list[str] = []

    def fake_exec(item, **k):
        calls.append(item.id)
        return ExecuteResult(False, "no_mats", "合金鋼錠材料不够")

    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", fake_exec)
    items_before = dict(st.snapshot.items)
    fac_before = {k: tuple(v) for k, v in st.snapshot.facilities.items()}
    bar_ui.pump_life_once(None, None)
    st1 = controller.get_controller()
    assert calls == ["p1"]
    assert recovers == []
    assert scans == []
    assert st1.fail_n == 0
    assert st1.paused is False
    assert feed not in st1.queue
    assert st1.current == gather
    assert st1.snapshot.items == items_before
    assert {k: tuple(v) for k, v in st1.snapshot.facilities.items()} == fac_before
    log = anomalies.ROOT / anomalies.LOG_REL
    tags = [json.loads(line)["tag"] for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert "no_mats" in tags
    assert "fail_shot" not in tags


def test_no_mats_next_pump_selects_other_recipe_or_gather(monkeypatch):
    feed = PlanItem("p1", "feed", "投料 合金鋼錠", facility="metal", recipe="合金鋼錠")
    gather = PlanItem("p2", "gather", "采集 銅礦石", recipe="銅礦石")
    setup_queue(monkeypatch, [feed, gather], _no_mats_snap())
    calls: list[str] = []

    def fake_exec(item, **k):
        calls.append(item.id)
        if item.kind == "feed":
            return ExecuteResult(False, "no_mats", "合金鋼錠材料不够")
        return ExecuteResult(False, "cancel", "stop")

    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", fake_exec)
    bar_ui.pump_life_once(None, None)
    bar_ui.pump_life_once(None, None)
    assert calls == ["p1", "p2"]
    st = controller.get_controller()
    assert st.current == gather
    assert feed not in st.queue
    assert gather in st.queue


def test_no_mats_next_duplicate_executes_once(monkeypatch):
    first = PlanItem("p1", "feed", "投料 合金鋼錠", facility="metal", recipe="合金鋼錠")
    duplicate = PlanItem("p2", "feed", "投料 合金鋼錠", facility="metal", recipe="合金鋼錠")
    other = PlanItem("p3", "gather", "采集 銅礦石", recipe="銅礦石")
    setup_queue(monkeypatch, [first, duplicate, other], _no_mats_snap())
    calls: list[str] = []

    def fake_exec(item, **k):
        calls.append(item.id)
        if item.kind == "feed":
            return ExecuteResult(False, "no_mats", "合金鋼錠材料不够")
        return ExecuteResult(False, "cancel", "stop")

    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", fake_exec)
    bar_ui.pump_life_once(None, None)
    bar_ui.pump_life_once(None, None)
    assert calls == ["p1", "p2"]
    st = controller.get_controller()
    assert first not in st.queue
    assert duplicate not in st.queue
    assert st.current == other


def test_no_mats_does_not_mark_item_done(monkeypatch):
    feed = PlanItem("p1", "feed", "投料 合金鋼錠", facility="metal", recipe="合金鋼錠")
    gather = PlanItem("p2", "gather", "采集 銅礦石", recipe="銅礦石")
    setup_queue(monkeypatch, [feed, gather], _no_mats_snap())
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda item, **k: ExecuteResult(False, "no_mats", "缺料"),
    )
    bar_ui.pump_life_once(None, None)
    st = controller.get_controller()
    assert "p1" not in [x.id for x in st.queue]
    assert st.snapshot.items == _no_mats_snap().items
    assert st.pending_action is None


def test_no_mats_paused_or_new_generation_does_not_mutate_queue(monkeypatch):
    feed = PlanItem("p1", "feed", "投料 合金鋼錠", facility="metal", recipe="合金鋼錠")
    gather = PlanItem("p2", "gather", "采集 銅礦石", recipe="銅礦石")
    snap = _no_mats_snap()
    st0 = setup_queue(monkeypatch, [feed, gather], snap)
    queue0 = list(st0.queue)

    def fake_paused(*_a, **_k):
        controller.pause_life()
        return ExecuteResult(False, "no_mats", "缺料")

    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", fake_paused)
    bar_ui.pump_life_once(None, None)
    paused = controller.get_controller()
    assert paused.paused is True
    assert list(paused.queue) == queue0
    assert paused.pending_action is None

    st1 = setup_queue(monkeypatch, [feed, gather], snap)
    run1 = st1.life_started_at
    before = list(st1.queue)
    items_before = dict(st1.snapshot.items)

    def fake_new_gen(*_a, **_k):
        controller.start_life(snapshot=snap, now=run1 + 50)
        return ExecuteResult(False, "no_mats", "缺料")

    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", fake_new_gen)
    bar_ui.pump_life_once(None, None)
    st2 = controller.get_controller()
    assert st2.life_started_at != run1
    assert st2.pending_action is None
    assert [x.id for x in st2.queue] == [x.id for x in before]
    assert st2.snapshot.items == items_before


_PROCESS_WAIT_STATUSES = (
    ("btn_unstable", "加工钮未稳住，不点"),
    ("btn_ambiguous", "加工钮语义不清，不点"),
    ("stp_unknown", "停钮未知，不点加工"),
)


@pytest.mark.parametrize("status,message", _PROCESS_WAIT_STATUSES)
def test_unissued_process_wait_status_skips_and_preserves_evidence(monkeypatch,status,message):
    wait=ExecuteResult(False,status,message)
    assert not is_nav_result(wait) and is_sense_result(wait)
    feed=PlanItem('p1','feed','投料 合金鋼錠',facility='metal',recipe='合金鋼錠')
    gather=PlanItem('p2','gather','采集 銅礦石',recipe='銅礦石')
    st0=setup_queue(monkeypatch,[feed,gather],LifeSnapshot(ok=True))
    calls=[];recovers=[];shots=[]
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',lambda *a,**k:recovers.append(1))
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot',lambda *a,**k:shots.append('shot') or 'failure.png')
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',lambda item,**k:calls.append(item.id) or wait)
    bar_ui.pump_life_once(None,None)
    st=controller.get_controller()
    # 2026-10-04: user requests two recovery attempts before final skip.
    assert calls==['p1']
    assert len(recovers) == (0 if status == 'click_yield' else 2)
    assert shots == (['shot'] if status == 'click_yield' else ['shot', 'shot'])
    assert st.current==gather and feed not in st.queue
    assert st.life_started_at==st0.life_started_at and st.fail_n==0
    import json
    from mm_assist.life_plan import anomalies
    rows=[json.loads(x) for x in (anomalies.ROOT/anomalies.LOG_REL).read_text(encoding='utf-8').splitlines()]
    assert any(r['tag']=='execute_result' and r['extra']==message and not r['ok'] for r in rows)


def test_tracking_issued_settle_failure_pauses_without_retry_or_recovery(monkeypatch):
    from mm_assist.life_plan import action_finish
    fail=ExecuteResult(False,'settle_fail','未稳住仅设施名',action_issued=True)
    assert not is_sense_result(fail)
    feed=PlanItem('p1','feed','投料 合金鋼錠',facility='metal',recipe='合金鋼錠')
    gather=PlanItem('p2','gather','采集 銅礦石',recipe='銅礦石')
    setup_queue(monkeypatch,[feed,gather],LifeSnapshot(ok=True),pause_on_failure=True)
    calls=[];recovers=[]
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',lambda *a,**k:recovers.append(1))
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',lambda item,**k:calls.append(item.id) or fail)
    monkeypatch.setattr(action_finish,'read_completion_stp',lambda:action_finish.StpSample(1,1,1,True))
    for _ in range(3):bar_ui.pump_life_once(None,None)
    assert calls==['p1'] and recovers==[]
    assert controller.get_controller().current==feed
    assert controller.get_controller().paused
    assert controller.get_controller().pending_action.result==fail
