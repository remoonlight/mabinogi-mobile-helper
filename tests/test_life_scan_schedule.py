from mm_assist.life_plan import controller, facility_sim
from mm_assist.life_plan.types import LifeSnapshot, PlanItem


def test_start_scans_once_and_periodic_scan_waits_for_batch_end(monkeypatch):
    controller.reset_controller_for_tests()
    scans = []
    def read(**kwargs):
        scans.append(kwargs)
        return LifeSnapshot(ok=True, items={"淨淨蘑菇": len(scans)}, taken_at=kwargs.get("now", 1000))
    monkeypatch.setattr(controller, "take_snapshot", read)
    monkeypatch.setattr(facility_sim, "_do_scan", lambda: (_ for _ in ()).throw(AssertionError("duplicate scan")))
    started = controller.start_life(now=1000)
    controller.tick_facility_progress(now=1599)
    assert len(scans) == 1
    st = controller.tick_facility_progress(now=1600)
    assert len(scans) == 1 and st.snapshot == started.snapshot
    for item in started.queue:
        controller.acknowledge_done(item.id, now=1601)
    st = controller.tick_facility_progress(now=1602)
    assert len(scans) == 2
    assert st.snapshot.items["淨淨蘑菇"] == 2


def test_empty_queue_does_not_rescan_bag(monkeypatch):
    controller.reset_controller_for_tests()
    monkeypatch.setattr(controller, "take_snapshot", lambda **k: (_ for _ in ()).throw(AssertionError("extra scan")))
    monkeypatch.setattr(controller, "plan_life", lambda *a, **k: [PlanItem("p1", "gather", "采集", recipe="淨淨蘑菇")])
    controller.start_life(snapshot=LifeSnapshot(ok=True, items={"淨淨蘑菇": 1}), now=1000)
    controller.acknowledge_done("p1", now=1001)
