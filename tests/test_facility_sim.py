from __future__ import annotations

from mm_assist.life_plan.facility_sim import (
    facility_sim_ok,
    peek_facility_sim,
    reset_facility_sim_for_tests,
    seed_facility_slots,
    set_facility_read_for_tests,
    should_claim_medicine,
    simulate_slot,
    tick_facility_sim,
)
from mm_assist.life_plan.types import FACILITY_SCAN_INTERVAL_S, QueueSlot


def test_simulate_marks_done_at_end():
    start = 1_700_000_000_000
    slot = QueueSlot(
        recipe="淨淨蘑菇汁液",
        done=False,
        start_ms=start,
        end_ms=start + 3_600_000,
        sheet_id=393824971,
    )
    mid = simulate_slot(slot, now_ms=start + 1_800_000)
    assert mid.done is False
    assert 0.4 < float(mid.progress_ratio or 0) < 0.6
    done = simulate_slot(slot, now_ms=start + 3_600_000)
    assert done.done is True
    assert done.progress_ratio == 1.0


def test_should_claim_requires_three():
    two = {
        "medicine": (
            QueueSlot("淨淨蘑菇汁液", True),
            QueueSlot("淨淨蘑菇汁液", True),
            QueueSlot("淨淨蘑菇汁液", False),
        )
    }
    three = {
        "medicine": tuple(QueueSlot("淨淨蘑菇汁液", True) for _ in range(3))
    }
    assert should_claim_medicine(two) is False
    assert should_claim_medicine(three) is True


def test_scan_interval_freezes_when_paused():
    reset_facility_sim_for_tests()
    scans = {"n": 0}

    def _read():
        scans["n"] += 1
        return {
            "medicine": (QueueSlot("淨淨蘑菇汁液", False, start_ms=1, end_ms=9),)
        }

    set_facility_read_for_tests(_read)
    t0 = 1000.0
    a = tick_facility_sim(running=True, now=t0, force_scan=True)
    assert a.ok and scans["n"] == 1
    tick_facility_sim(running=False, now=t0 + FACILITY_SCAN_INTERVAL_S + 10)
    b = tick_facility_sim(running=True, now=t0 + FACILITY_SCAN_INTERVAL_S + 20)
    assert scans["n"] == 1
    assert b.accrued_s < FACILITY_SCAN_INTERVAL_S
    tick_facility_sim(running=True, now=t0 + FACILITY_SCAN_INTERVAL_S + 20 + FACILITY_SCAN_INTERVAL_S)
    assert scans["n"] == 2


def test_peek_does_not_scan():
    reset_facility_sim_for_tests()
    scans = {"n": 0}

    def _read():
        scans["n"] += 1
        return {"medicine": (QueueSlot("淨淨蘑菇汁液", True),)}

    set_facility_read_for_tests(_read)
    seed_facility_slots(
        {"medicine": (QueueSlot("淨淨蘑菇汁液", True), QueueSlot("淨淨蘑菇汁液", False))},
        now=5.0,
        ok=True,
    )
    peek = peek_facility_sim()
    assert scans["n"] == 0
    assert peek.ok is True
    assert peek.last_scan_at == 5.0
    assert peek.slots["medicine"][0].done is True
    reset_facility_sim_for_tests()
    set_facility_read_for_tests(_read)
    failed = peek_facility_sim()
    assert scans["n"] == 0
    assert failed.ok is False
    assert failed.last_scan_at == 0.0


def test_failed_scan_not_ok():
    reset_facility_sim_for_tests()
    set_facility_read_for_tests(lambda: None)
    st = tick_facility_sim(running=True, now=1.0, force_scan=True)
    assert st.ok is False
    assert facility_sim_ok() is False
