"""Facility updates must retain the skill evidence acquired in the same read."""

from mm_assist.life_plan import controller, facility_sim
from mm_assist.life_plan.types import LifeSnapshot, QueueSlot, SkillGrowth


def test_claim_and_progress_overlay_keep_skill_evidence(monkeypatch):
    controller.reset_controller_for_tests()
    facility_sim.reset_facility_sim_for_tests()
    skills = (SkillGrowth("採礦", 12, 345),)
    slots = {"metal": (QueueSlot("鐵錠", True),) * 3}
    snap = LifeSnapshot(
        ok=True, items={"鐵錠": 7}, facilities=slots,
        skills=skills, taken_at=1000,
    )
    monkeypatch.setattr("mm_assist.life_plan.snapshot.refresh_tools", lambda: ())
    try:
        started = controller.start_life(snapshot=snap, now=1000)
        assert started.current.kind == "claim"
        progressed = controller.tick_facility_progress(now=1001)
        assert progressed.snapshot.skills == skills
        claimed = controller.acknowledge_done(started.current.id, now=1002)
        assert claimed.snapshot.skills == skills
        assert claimed.snapshot.items == snap.items
        assert claimed.snapshot.taken_at == snap.taken_at
        assert claimed.snapshot.facilities["metal"] == ()
        assert snap.facilities["metal"] == slots["metal"]
    finally:
        controller.reset_controller_for_tests()
        facility_sim.reset_facility_sim_for_tests()
