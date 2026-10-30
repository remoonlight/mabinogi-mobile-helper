from __future__ import annotations

import struct
from unittest.mock import patch

import pytest

from mm_assist.life_plan.types import SLOT_CAP, QueueSlot
from mm_assist.memory.readers.facility_queue import (
    clamp_slot_cap,
    pick_latest_slots,
    progress_from_ticks,
    resolve_sheet_name,
    sheet_facility_type,
    ticks_to_seconds,
)
from mm_assist.memory.readers.mm_dict import iter_key_int_pairs
from mm_assist.perception.process_sense import parse_queue_percents


def test_sheet_id_maps_juice():
    assert resolve_sheet_name(393824971) == "淨淨蘑菇汁液"


def test_metal_facility_type_is_zero():
    assert sheet_facility_type(332636885) == 0
    assert sheet_facility_type(1793283870) == 0
    assert sheet_facility_type(999999999) == -1


def test_progress_half_hour_on_one_hour_job():
    hour = 36_000_000_000
    start = 1_000_000_000_000_000
    end = start + hour
    ratio, remain = progress_from_ticks(start, end, now=start + hour // 2)
    assert 0.49 <= ratio <= 0.51
    assert 1700 <= remain <= 1900


def test_progress_done_when_past_end():
    start, end = 100, 200
    ratio, remain = progress_from_ticks(start, end, now=250)
    assert ratio == 1.0
    assert remain == 0.0


def test_ticks_to_seconds_dotnet():
    assert abs(ticks_to_seconds(10_000_000) - 1.0) < 1e-6


@pytest.mark.parametrize("remaining_ms", [500, 10_000_000, 12_800_000])
def test_remaining_seconds_use_timestamp_unit_not_delta_size(remaining_ms):
    # Current read fixture: Unix milliseconds. Long queues crossed the old
    # 10,000,000 threshold and were incorrectly displayed as about one second.
    now = 1_790_866_419_819
    start = now - 1000
    ratio, remain = progress_from_ticks(start, now + remaining_ms, now=now)
    assert 0 <= ratio <= 1
    assert remain == pytest.approx(remaining_ms / 1000)


def test_tick_timestamp_keeps_subsecond_precision_near_completion():
    start = 1_790_866_419_819_0000
    ratio, remain = progress_from_ticks(start, start + 10_000_000, now=start + 9_500_000)
    assert ratio == pytest.approx(0.95)
    assert remain == pytest.approx(0.05)


def test_ocr_percent_glue():
    assert parse_queue_percents("0% 淨淨蘑菇汁液") == [0]
    assert parse_queue_percents("12％ 槽2 88%") == [12, 88]
    assert parse_queue_percents("无进度") == []


def test_pick_latest_slots_dedupes_and_keeps_five():
    recs = []
    for i, start in enumerate((10, 20, 30, 40, 20, 50)):
        recs.append(
            (
                start,
                QueueSlot(
                    recipe="淨淨蘑菇汁液",
                    done=False,
                    progress_ratio=0.1 * i,
                    sheet_id=393824971,
                ),
            )
        )
    slots = pick_latest_slots(recs, cap=5)
    assert len(slots) == 5
    assert [round(float(s.progress_ratio or 0), 1) for s in slots] == [0.0, 0.1, 0.2, 0.3, 0.5]


def test_pick_latest_slots_six_recs_keeps_five():
    recs = []
    for i, start in enumerate((10, 20, 30, 40, 50, 60)):
        recs.append(
            (
                start,
                QueueSlot(
                    recipe="淨淨蘑菇汁液",
                    done=False,
                    progress_ratio=0.1 * i,
                    sheet_id=393824971,
                ),
            )
        )
    slots = pick_latest_slots(recs)
    assert SLOT_CAP == 5
    assert len(slots) == 5
    assert [round(float(s.progress_ratio or 0), 1) for s in slots] == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_clamp_slot_cap_fail_closed_five():
    assert clamp_slot_cap(5) == 5
    assert clamp_slot_cap(1) == 1
    assert clamp_slot_cap(8) == 8
    assert clamp_slot_cap(0) == SLOT_CAP
    assert clamp_slot_cap(9) == SLOT_CAP
    assert clamp_slot_cap(None) == SLOT_CAP
    assert clamp_slot_cap(-1) == SLOT_CAP
    assert SLOT_CAP == 5


def test_iter_key_int_pairs_reads_i32_stride4():
    offs = {
        "count": 0x34,
        "keys": 0x10,
        "values": 0x18,
        "arr_max": 0x18,
        "arr_vec": 0x20,
    }
    blobs: dict[int, bytes] = {
        0x1010: (0x2000).to_bytes(8, "little"),
        0x1018: (0x3000).to_bytes(8, "little"),
        0x1034: (2).to_bytes(4, "little"),
        0x2018: (2).to_bytes(4, "little"),
        0x2020: (0).to_bytes(4, "little"),
        0x2024: (1).to_bytes(4, "little"),
        0x3018: (2).to_bytes(4, "little"),
        0x3020: (5).to_bytes(4, "little", signed=True),
        0x3024: (5).to_bytes(4, "little", signed=True),
    }

    def fake_read_mem(_h, addr: int, size: int) -> bytes | None:
        raw = blobs.get(int(addr))
        if raw is None:
            return None
        return raw[:size]

    def fake_read_u32(_h, addr: int) -> int | None:
        raw = fake_read_mem(_h, addr, 4)
        if not raw or len(raw) < 4:
            return None
        return struct.unpack("<I", raw)[0]

    def fake_read_ptr(_h, addr: int) -> int | None:
        raw = fake_read_mem(_h, addr, 8)
        if not raw or len(raw) < 8:
            return None
        return struct.unpack("<Q", raw)[0]

    with (
        patch("mm_assist.memory.readers.mm_dict.read_mem", fake_read_mem),
        patch("mm_assist.memory.readers.mm_dict.read_u32", fake_read_u32),
        patch("mm_assist.memory.readers.mm_dict.read_ptr", fake_read_ptr),
    ):
        pairs = list(iter_key_int_pairs(object(), 0x1000, offs))
    assert pairs == [(0, 5), (1, 5)]
