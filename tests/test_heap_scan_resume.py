import struct

import pytest

from mm_assist.memory import il2cpp as il


@pytest.fixture
def heap(monkeypatch):
    now = [0.0]
    reads = []
    monkeypatch.setattr(il.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(il, "_hot_key", lambda h: h)
    monkeypatch.setattr(il, "_HOT_REGIONS", {1: {0x10000: 50}})
    monkeypatch.setattr(il, "_SCAN_CURSORS", {}, raising=False)
    monkeypatch.setattr(il, "_save_hot_regions", lambda *a: None)
    monkeypatch.setattr(il, "_load_hot_regions", lambda *a: None)
    monkeypatch.setattr(il, "_heap_scan_workers", lambda: 1)
    # 热区之后还有较大冷区，旧40%/60%分段仍会反复停在冷区前半段。
    regions = [(0x10000, 256, 4, 0x20000), (0x18000, 256, 4, 0x20000),
               (0x20000, 32, 4, 0x20000)]
    monkeypatch.setattr(il, "_list_heap_regions", lambda *a, **k: list(regions))
    def read(h, addr, n):
        reads.append((h, addr))
        now[0] += 0.1
        raw = bytearray(n)
        if addr <= 0x20008 < addr + n - 7:
            struct.pack_into("<Q", raw, 0x20008 - addr, 0x90000)
        return raw, n
    monkeypatch.setattr(il, "read_chunk", read)
    return now, reads, regions


def test_repeated_bounded_location_scan_reaches_cold_region(heap):
    _, reads, _ = heap
    for _ in range(16):
        assert il.scan_klass_objects(1, 0x90000, chunk=32, max_hits=1,
                                    max_wall_s=0.25) == []
    assert not any(addr == 0x20000 for _, addr in reads)
    reads.clear()
    result = []
    for _ in range(16):
        result = il.scan_klass_objects(1, 0x90000, chunk=32, max_hits=1,
                                       max_wall_s=0.25, resume=True)
        if result:
            break
    assert result == [0x20008]
    assert len(reads) < 32


def test_resume_inside_false_hit_chunk_does_not_starve_valid_candidate(heap, monkeypatch):
    now, _, regions = heap
    regions[:] = [(0x10000, 80, 4, 0x20000)]
    raw = struct.pack("<Q", 0x90000) * 10
    monkeypatch.setattr(il, "read_chunk", lambda *a: (raw, len(raw)))
    visited = []
    def valid(name, addr):
        visited.append(addr)
        now[0] += 0.1
        return addr == 0x10048
    result = []
    for _ in range(5):
        result = il.scan_klass_objects(1, 0x90000, max_hits=1,
            max_wall_s=0.25, hit_filter=valid, resume=True)
        if result:
            break
    assert result == [0x10048]
    assert visited == list(range(0x10000, 0x10050, 8))


def test_resume_restarts_after_success_and_full_miss(heap):
    _, reads, regions = heap
    regions[:] = [(0x20000, 32, 4, 0x20000)]
    for _ in range(2):
        assert il.scan_klass_objects(1, 0x90000, max_hits=1,
            max_wall_s=0.25, resume=True) == [0x20008]
    assert reads == [(1, 0x20000)] * 2
    for _ in range(2):
        assert il.scan_klass_objects(1, 0xA0000, max_wall_s=0.25, resume=True) == []
    assert len(reads) == 4


def test_resume_is_scoped_to_process_type_and_reset(heap):
    _, reads, _ = heap
    il.scan_klass_objects(1, 0x90000, chunk=32, max_wall_s=0.25, resume=True)
    for pid, klass in [(2, 0x90000), (1, 0xA0000)]:
        start = len(reads)
        il.scan_klass_objects(pid, klass, chunk=32, max_wall_s=0.25, resume=True)
        assert reads[start] == (pid, 0x10000)
    il.reset_scan_cache(1)
    start = len(reads)
    il.scan_klass_objects(1, 0x90000, chunk=32, max_wall_s=0.25, resume=True)
    assert reads[start] == (1, 0x10000)


def test_stopped_resume_scan_does_not_read(heap):
    _, reads, _ = heap
    assert il.scan_klass_objects(1, 0x90000, resume=True,
                                should_stop=lambda: True) == []
    assert reads == []


def test_resume_tolerates_disappeared_region(heap):
    _, reads, regions = heap
    il.scan_klass_objects(1, 0x90000, chunk=32, max_wall_s=0.25, resume=True)
    regions[:] = regions[-1:]
    assert il.scan_klass_objects(1, 0x90000, chunk=32, max_hits=1,
                                max_wall_s=0.25, resume=True) == [0x20008]


def test_multi_type_resume_keeps_progress_inside_each_needle(heap, monkeypatch):
    now, _, regions = heap
    regions[:] = [(0x10000, 80, 4, 0x20000)]
    raw = struct.pack("<Q", 0x90000) * 9 + struct.pack("<Q", 0xA0000)
    monkeypatch.setattr(il, "read_chunk", lambda *a: (raw, len(raw)))
    visited = []
    def valid(name, addr):
        visited.append((name, addr))
        now[0] += 0.1
        return name == "position"
    result = {}
    for _ in range(5):
        result = il.scan_klass_objects_multi(1, {"ghost": 0x90000, "position": 0xA0000},
            max_wall_s=0.25, hit_filter=valid, resume=True)
        if result["position"]:
            break
    assert result["position"] == [0x10048]
    assert len(visited) == 10


def test_reader_timeout_waits_on_one_scan_without_starting_another(monkeypatch):
    import threading
    entered, release = threading.Event(), threading.Event()
    calls = []
    def scan(*a, **k):
        calls.append(1)
        entered.set()
        assert release.wait(1)
        return {"x": [0x10000]}
    monkeypatch.setattr(il, "_scan_multi", scan)
    worker = threading.Thread(target=lambda: il.scan_klass_objects_multi(1, {"x": 0x90000}))
    worker.start()
    try:
        assert entered.wait(1)
        assert il.scan_klass_objects_multi(1, {"x": 0x90000}, max_wall_s=0.01) == {"x": []}
        assert calls == [1]
    finally:
        release.set()
        worker.join(1)
    assert not worker.is_alive()


def test_reset_during_scan_does_not_accept_late_old_cursor(heap, monkeypatch):
    now, _, _ = heap
    def read(*a):
        il.reset_scan_cache(1)
        now[0] += 0.3
        return b"\0" * 32, 32
    monkeypatch.setattr(il, "read_chunk", read)
    il.scan_klass_objects(1, 0x90000, chunk=32, max_wall_s=0.25, resume=True)
    assert il._SCAN_CURSORS == {}


def test_pause_mid_chunk_resumes_after_processed_candidate(heap, monkeypatch):
    _, _, regions = heap
    regions[:] = [(0x10000, 16, 4, 0x20000)]
    raw = struct.pack("<Q", 0x90000) * 2
    monkeypatch.setattr(il, "read_chunk", lambda *a: (raw, len(raw)))
    stopped, visited = [False], []
    def valid(name, addr):
        visited.append(addr)
        stopped[0] = True
        return addr == 0x10008
    assert il.scan_klass_objects(1, 0x90000, max_hits=1, resume=True,
        should_stop=lambda: stopped[0], hit_filter=valid) == []
    stopped[0] = False
    assert il.scan_klass_objects(1, 0x90000, max_hits=1, resume=True,
        should_stop=lambda: stopped[0], hit_filter=valid) == [0x10008]
    assert visited == [0x10000, 0x10008]


def test_scan_exception_releases_shared_reader(monkeypatch):
    calls = []
    def scan(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("read failed")
        return {"x": [0x10000]}
    monkeypatch.setattr(il, "_scan_multi", scan)
    with pytest.raises(RuntimeError, match="read failed"):
        il.scan_klass_objects_multi(1, {"x": 0x90000}, max_wall_s=0.1)
    assert il.scan_klass_objects_multi(1, {"x": 0x90000}, max_wall_s=0.1) == {"x": [0x10000]}
