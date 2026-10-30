"""Read-only verification of material decisions; never dispatch game input.

Offline cases measure avoided work in deterministic fixtures, not elapsed game
time. --live adds one current memory snapshot after watch/protocol checks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mm_assist.life_plan.planner import plan_life
from mm_assist.life_plan.recipes import can_afford, explode_needs
from mm_assist.life_plan.types import CYCLE_SIZES, LifeSnapshot, QueueSlot


def offline_cases() -> list[dict]:
    """Compare the same target with and without each relevant observation."""
    bag = {**CYCLE_SIZES, "合金鋼錠": 97}
    desired = dict(CYCLE_SIZES)
    without = explode_needs(bag, desired)
    with_steel = explode_needs({**bag, "鋼錠": 3}, desired)
    avoided = {name: without.get(name, 0) - with_steel.get(name, 0)
               for name in without if without.get(name, 0) > with_steel.get(name, 0)}
    assert avoided == {"鐵礦石": 10, "煤炭": 4}, avoided
    rows = [{"element": "intermediate_stock", "passed": True,
             "without_stock": without, "with_stock": with_steel,
             "avoided_raw_units": avoided}]

    available = {**bag, "鋼錠": 3, "銅礦石": 15, "煤炭": 8}
    snap = LifeSnapshot(ok=True, items=available, taken_at=1000)
    without_queue = plan_life(snap)
    in_work = replace(snap, facilities={"metal": (QueueSlot("合金鋼錠", False),)})
    with_queue = plan_life(in_work)
    feeds_before = sum(p.kind == "feed" and p.recipe == "合金鋼錠" for p in without_queue)
    feeds_after = sum(p.kind == "feed" and p.recipe == "合金鋼錠" for p in with_queue)
    assert (feeds_before, feeds_after) == (1, 0)
    rows.append({"element": "queued_output", "passed": True,
                 "alloy_feed_plans_without_queue": feeds_before,
                 "alloy_feed_plans_with_queue": feeds_after})

    from mm_assist.life_plan.controller import LifeController, execution_items
    depleted = LifeController(snapshot=LifeSnapshot(
        ok=True, items={"淨淨蘑菇孢子": 15, "裝水的瓶子": 7}, taken_at=1000),
        feed_spend={"p1": (1010, {"淨淨蘑菇孢子": 10, "裝水的瓶子": 1})})
    original = dict(depleted.snapshot.items)
    remaining = execution_items(depleted)
    assert can_afford("淨淨蘑菇汁液", original)
    assert not can_afford("淨淨蘑菇汁液", remaining)
    assert depleted.snapshot.items == original
    rows.append({"element": "spent_stock", "passed": True,
                 "before": original, "available_after_feed": remaining,
                 "second_feed_affordable": False})

    future = LifeController(snapshot=LifeSnapshot(ok=True,
        items={**bag, "銅礦石": 15, "煤炭": 8},
        facilities={"metal": (QueueSlot("鋼錠", False),)}))
    assert not can_afford("合金鋼錠", execution_items(future))
    rows.append({"element": "unfinished_output_gate", "passed": True,
                 "queued_steel_is_available_stock": False})

    from mm_assist.life_plan.facility_sim import (
        overlay_snapshot, reset_facility_sim_for_tests, seed_facility_slots,
        tick_facility_sim,
    )
    slots = {"metal": (QueueSlot("鐵錠", True), QueueSlot("鐵錠", True),
                       QueueSlot("鐵錠", False, start_ms=1000000, end_ms=1010000))}
    scans = []
    def unexpected_scan():
        scans.append(True)
        raise AssertionError("completion should use the already-read end time")
    reset_facility_sim_for_tests()
    try:
        seed_facility_slots(slots, now=1000)
        before = tick_facility_sim(running=True, now=1009, scan_fn=unexpected_scan)
        after = tick_facility_sim(running=True, now=1010, scan_fn=unexpected_scan)
        snap = LifeSnapshot(ok=True, items=dict(CYCLE_SIZES), facilities=slots)
        n0 = sum(p.kind == "claim" for p in plan_life(overlay_snapshot(snap, before.slots)))
        n1 = sum(p.kind == "claim" for p in plan_life(overlay_snapshot(snap, after.slots)))
        assert (n0, n1, len(scans)) == (0, 1, 0)
        rows.append({"element": "completion_time", "passed": True,
                     "claim_plans_before": n0, "claim_plans_after": n1,
                     "extra_memory_scans": len(scans)})
    finally:
        reset_facility_sim_for_tests()
    return rows


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src/mm_assist").rglob("*.py"))
    paths += sorted((ROOT / "knowledge/ui/config").glob("*.json"))
    paths += [Path(__file__).resolve()]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def live_snapshot() -> dict:
    from mm_assist.jobs.bar import require_state_watch_running
    from mm_assist.safety.lease import assert_protocol_ready
    from mm_assist.memory.process import find_game_pid
    from mm_assist.life_plan.snapshot import take_snapshot

    assert_protocol_ready()
    hb = require_state_watch_running()
    ready = json.loads((ROOT / "logs/state_watch_ready.json").read_text(encoding="utf-8"))
    age = time.time() - float(ready.get("updated_at_unix", 0))
    if (ready.get("pid") != hb["pid"] or not 0 <= age <= 5
            or not ready.get("tick_completed") or ready.get("warming")
            or ready.get("busy") or ready.get("required_pending")
            or ready.get("params_incomplete")):
        raise RuntimeError("watch required components are not freshly ready")
    pid = find_game_pid()
    if not pid:
        raise RuntimeError("game not running")
    started = time.perf_counter()
    snap = take_snapshot(include_skills=True)
    elapsed = time.perf_counter() - started
    finished_at = time.time()
    if find_game_pid() != pid:
        raise RuntimeError("game process changed during read")
    if not snap.ok:
        raise RuntimeError(snap.error)
    if set(snap.facilities) != {"metal", "wood", "cloth", "medicine"}:
        raise RuntimeError("incomplete facility read")
    timing = []
    for facility, slots in snap.facilities.items():
        for slot in slots:
            expected = max(0.0, slot.end_ms / 1000 - finished_at)
            # Facilities are read before optional skills, so account for the
            # measured total read span rather than inventing an atomic sample.
            valid = slot.remain_s is not None and abs(slot.remain_s - expected) <= elapsed + 1
            timing.append({"facility": facility, "recipe": slot.recipe,
                           "reported_s": slot.remain_s, "at_read_end_s": round(expected, 1),
                           "within_read_span": valid})
    if not all(row["within_read_span"] for row in timing):
        raise RuntimeError("remaining time conflicts with end timestamp")
    return {"game_pid": pid, "watch_pid": hb["pid"], "read_elapsed_s": elapsed,
            "snapshot": asdict(snap), "item_count": len(snap.items),
            "tool_count": len(snap.tools), "skill_count": len(snap.skills),
            "timing_checks": timing,
            "tools_readable": bool(snap.tools), "skills_readable": bool(snap.skills),
            "plan": [asdict(p) for p in plan_life(snap)],
            "input_events": 0, "limitation": "one read; no action or throughput comparison"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="one read-only current game snapshot")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "source_fingerprint": source_fingerprint(), "offline": offline_cases(),
              "live": {"status": "not_requested"}}
    code = 0
    if args.live:
        try:
            report["live"] = {"status": "readable", **live_snapshot()}
        except Exception as exc:
            report["live"] = {"status": "unverified", "reason": str(exc)}
            code = 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    live = report["live"]
    print(json.dumps({"offline_passed": len(report["offline"]),
                      "live_status": live["status"], "reason": live.get("reason"),
                      "items": live.get("item_count"), "tools": live.get("tools_readable"),
                      "skills": live.get("skills_readable"), "path": str(args.out)}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
