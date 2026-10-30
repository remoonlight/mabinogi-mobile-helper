# -*- coding: utf-8 -*-
from __future__ import annotations

from mm_assist.benchmarks.memory import (
    CURRENT_PRODUCTION,
    ReadMeter,
    build_scorecard,
    decide_default,
    rank_static,
    run_sparse_poll,
    run_static_ladder,
    sha256_hex,
)


def test_heap_scan_does_not_shadow_chunk_size():
    import inspect

    from mm_assist.memory import il2cpp

    src = inspect.getsource(il2cpp._scan_multi)
    assert "n = min(chunk," not in src
    assert "payload = read_chunk" in src
    assert "on_first_hit" in src


def test_usb_safe_sizes_skip_megabyte():
    from mm_assist.benchmarks.memory import USB_SAFE_SIZES

    assert max(USB_SAFE_SIZES) <= 65536


def test_production_is_rpm_into():
    assert CURRENT_PRODUCTION == "rpm_into"


def test_scorecard_keeps_rpm_into():
    rows = [
        {
            "workload": "static_ladder",
            "method": "rpm_into",
            "hash_match": True,
            "ok_ratio": 1.0,
            "p95_ms": 2.0,
        }
    ]
    card = build_scorecard(rows)
    dec = decide_default(rank_static(rows), card)
    assert dec["keep"] == "rpm_into"
    assert card[0]["accuracy_ok"] is True


def test_inaccurate_run_still_keeps_production_name():
    rows = [
        {
            "workload": "static_ladder",
            "method": "rpm_into",
            "hash_match": False,
            "ok_ratio": 0.0,
            "p95_ms": 2.0,
        }
    ]
    card = build_scorecard(rows)
    assert card[0]["accuracy_ok"] is False
    dec = decide_default(rank_static(rows), card)
    assert dec["keep"] == "rpm_into"


def test_static_ladder_mock(monkeypatch):
    from mm_assist.benchmarks import memory as mb

    payload = b"\x01" * 65536

    def fake_read(hproc, addr, size, *, method, meter, buf=None):
        meter.record(size, size)
        return payload[:size]

    monkeypatch.setattr(mb, "_read", fake_read)
    rows = run_static_ladder(object(), 0x1000, 65536, warmup=0, repeats=2)
    assert rows
    assert all(r["hash_match"] for r in rows)
    assert all(r["method"] == "rpm_into" for r in rows)
    assert sha256_hex(payload[:8]) == sha256_hex(payload[:8])


def test_sparse_poll_records_ok(monkeypatch):
    from mm_assist.benchmarks import memory as mb

    def fake_read(hproc, addr, size, *, method, meter, buf=None):
        meter.record(size, size)
        return b"\x00" * size

    monkeypatch.setattr(mb, "_read", fake_read)
    fields = [(0x1000, 8), (0x1008, 8)]
    rows = run_sparse_poll(object(), fields, warmup=0, repeats=3)
    assert rows[0]["method"] == "rpm_into"
    assert rows[0]["ok_ratio"] == 1.0
    assert ReadMeter().fails == 0
