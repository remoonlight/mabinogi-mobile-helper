import json
from concurrent.futures import ThreadPoolExecutor

from mm_assist.life_plan.types import QueueSlot, ToolDurability
from mm_assist.memory.readers import facility_queue as fq


def test_mmlist_count_is_not_version(monkeypatch):
    # 实机 get_Count @ RVA 0x3C86950: 8B 41 1C C3 = mov eax,[rcx+1C]; ret.
    off = {k: int(v, 0) for k, v in fq._cfg()["offsets"].items()}
    words = {0x10018: 91, 0x1001C: 2, 0x20018: 4}
    ptrs = {0x10010: 0x20000, 0x20020: 0x30000, 0x20028: 0x40000}
    monkeypatch.setattr(fq, "read_u32", lambda h, a: words.get(a))
    monkeypatch.setattr(fq, "read_ptr", lambda h, a: ptrs.get(a))
    assert fq._read_ptr_list(None, 0x10000, off) == [0x30000, 0x40000]
    assert fq._read_ptr_list(None, 0x50000, off) is None


def test_status_queue_follows_work_list_pointer_and_allows_verified_empty(monkeypatch):
    off = {k: int(v, 0) for k, v in fq._cfg()["offsets"].items()}
    monkeypatch.setattr(fq, "read_u32", lambda *a: 4)
    monkeypatch.setattr(fq, "iter_key_ptr_pairs", lambda *a: [(0, 10), (1, 20), (2, 30), (4, 40)])
    lists = {10: [100], 20: [], 30: [], 40: [], 200: [300]}
    monkeypatch.setattr(fq, "_read_ptr_list", lambda h, p, o, **k: lists.get(p))
    monkeypatch.setattr(fq, "read_ptr", lambda h, a: 200 if a == 100 + 0x38 else None)
    monkeypatch.setattr(fq, "_work_to_slot", lambda *a: QueueSlot("鋼錠", True))
    assert fq._read_status_queues(None, 1, off)["metal"] == (QueueSlot("鋼錠", True),)
    lists[200] = []  # 领取后挂接列表为空，不能回退到堆里的旧 WorkStatus。
    assert fq._read_status_queues(None, 1, off) == fq._empty()
    lists[200] = None
    assert fq._read_status_queues(None, 1, off) == {}  # 未读懂不等于空队列。


def test_status_queue_preserves_sixth_slot_even_without_capacity(monkeypatch):
    off = {k: int(v, 0) for k, v in fq._cfg()['offsets'].items()}
    monkeypatch.setattr(fq, 'iter_key_ptr_pairs', lambda *a: [(0, 10), (1, 20), (2, 30), (4, 40)])
    work_array = 0x20000
    words = {1 + off['MMDictionary.count']: 4, 100 + off['MMList._size']: 6,
             work_array + off['Il2CppArray.max_length']: 8}
    works = [0x30000 + i * 0x100 for i in range(6)]
    ptrs = {100 + off['MMList._items']: work_array,
            **{work_array + off['Il2CppArray.vector'] + i * 8: work for i, work in enumerate(works)}}
    read_list = fq._read_ptr_list
    monkeypatch.setattr(fq, 'read_u32', lambda h, a: words.get(a))
    monkeypatch.setattr(fq, 'read_ptr', lambda h, a: 100 if a == 200 + off['AlteringWorkStatusQueue.workQueue'] else ptrs.get(a))
    monkeypatch.setattr(fq, '_read_ptr_list', lambda h, p, o, **kw:
                        {10: [200], 20: [], 30: [], 40: []}[p] if p != 100 else read_list(h, p, o, **kw))
    monkeypatch.setattr(fq, '_work_to_slot', lambda h, p, o: QueueSlot('鋼錠', p == works[-1]))
    metal = fq._read_status_queues(None, 1, off)['metal']
    assert len(metal) == 6 and metal[-1].done


def test_facility_cache_rereads_membership_without_rescanning(monkeypatch):
    monkeypatch.setattr(fq, "_COMPONENTS", {})
    monkeypatch.setattr(fq, "module_base", lambda *a: 1)
    monkeypatch.setattr(fq, "resolve_klass", lambda *a: 99)
    off = {k: int(v, 0) for k, v in fq._cfg()["offsets"].items()}
    ptrs = {1000: 99, 1000 + off["AlteringComponent.EntireAlteringStatusData"]: 2000, 2016: 3000}
    monkeypatch.setattr(fq, "read_ptr", lambda h, a: ptrs.get(a))
    live = fq._empty()
    live["metal"] = (QueueSlot("鋼錠", True),)
    monkeypatch.setattr(fq, "_read_status_queues", lambda *a: dict(live))
    scans = []
    def scan(h, k, **kw):
        scans.append(1)
        assert kw["hit_filter"]("_", 1000)
        kw["hit_filter"]("_", 9000)  # 并行扫描晚到的坏命中不能覆盖好结果。
        return [1000]
    monkeypatch.setattr(fq, "scan_klass_objects", scan)
    assert len(fq.read_facility_queues(None, 7)["metal"]) == 1
    live["metal"] = ()
    assert fq.read_facility_queues(None, 7)["metal"] == ()
    assert len(scans) == 1


def test_tool_cache_reads_new_durability_and_retries_missing(monkeypatch):
    from mm_assist.memory.readers import equip_tools as et
    et.reset_tools_cache_for_tests()
    monkeypatch.setattr(et, "module_base", lambda *a: 1)
    monkeypatch.setattr(et, "resolve_klass", lambda *a: 99)
    monkeypatch.setattr(et, "read_ptr", lambda h, a: 99)
    monkeypatch.setattr(et, "read_sc_component", lambda *a: None)
    current = [ToolDurability("鎬", 10, 100)]
    monkeypatch.setattr(et, "_decode_tools", lambda *a: list(current))
    def scan(h, k, **kw):
        return [1000] if kw["hit_filter"]("_", 1000) else []
    monkeypatch.setattr(et, "scan_klass_objects", scan)
    assert et.read_equipped_tools(None, 7)[0].durability == 10
    current[:] = [ToolDurability("鎬", 1, 100)]
    assert et.read_equipped_tools(None, 7)[0].durability == 1
    current.clear()
    assert et.read_equipped_tools(None, 7) is None
    current[:] = [ToolDurability("鎬", 100, 100)]
    monkeypatch.setattr(et, "_TOOLS_RETRY_AT", 0)
    assert et.read_equipped_tools(None, 7)[0].durability == 100
    et.reset_tools_cache_for_tests()


def test_two_heartbeat_writers_do_not_collide(tmp_path):
    from mm_assist.jobs.bar import write_state_watch_heartbeat
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda i: write_state_watch_heartbeat(pid=i + 1, root=tmp_path), range(50)))
    assert json.loads(paths[-1].read_text())["pid"] > 0


def test_heap_scan_checks_deadline_inside_false_hit_chunk(monkeypatch):
    import struct
    from mm_assist.memory import il2cpp as il
    monkeypatch.setattr(il, "_hot_key", lambda h: 1)
    monkeypatch.setattr(il, "_HOT_REGIONS", {1: {}})
    monkeypatch.setattr(il, "_save_hot_regions", lambda *a: None)
    monkeypatch.setattr(il, "_load_hot_regions", lambda *a: None)
    monkeypatch.setattr(il, "_list_heap_regions", lambda *a, **k: [(0x10000, 8000, 0, 0)])
    monkeypatch.setattr(il, "_heap_scan_workers", lambda: 1)
    payload = bytearray(struct.pack("<Q", 0x20000) * 1000)
    monkeypatch.setattr(il, "read_chunk", lambda *a: (payload, len(payload)))
    elapsed = [0.0]
    monkeypatch.setattr(il.time, "perf_counter", lambda: elapsed[0])
    def slow_false_hit(*a):
        elapsed[0] += 0.1
        return False
    assert il._scan_multi(None, {"x": 0x20000}, max_wall_s=0.25, hit_filter=slow_false_hit) == {"x": []}
    assert elapsed[0] < 0.4


def test_region_enumeration_honors_stop_even_before_readable_region(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from mm_assist.memory import process as pm
    calls = []
    def query(h, addr, out, size):
        calls.append(1)
        mbi = ctypes.cast(out, ctypes.POINTER(pm.MEMORY_BASIC_INFORMATION)).contents
        mbi.BaseAddress = addr.value or 0
        mbi.RegionSize = 4096
        mbi.State = 0  # 未提交页不 yield，取消仍须生效。
        return size
    monkeypatch.setattr(pm, "kernel32", SimpleNamespace(VirtualQueryEx=query))
    assert list(pm.iter_regions(None, should_stop=lambda: len(calls) >= 3)) == []
    assert len(calls) == 3


def test_region_cache_shared_by_same_game_process(monkeypatch):
    from mm_assist.memory import il2cpp as il
    from mm_assist.memory import client_build
    # Both fake handles belong to the same verified process lifetime.
    monkeypatch.setattr(client_build, 'discovery_token', lambda h: (7, 123, 0x10000, 'fixture'))
    monkeypatch.setattr(il, "_REGION_CACHE", {})
    monkeypatch.setattr(il, "_hot_key", lambda h: 7)
    now = [100.0]
    monkeypatch.setattr(il.time, "perf_counter", lambda: now[0])
    calls = []
    def enum(*a, **k):
        calls.append(1)
        return iter([(0x10000, 4096, 4, 0x20000)])
    monkeypatch.setattr(il, "iter_regions", enum)
    il._list_heap_regions(10)
    now[0] += 2
    il._list_heap_regions(11)
    assert len(calls) == 1
    now[0] += 31
    il._list_heap_regions(11)
    assert len(calls) == 2


def test_heap_callback_cannot_overwrite_remaining_chunk(monkeypatch):
    import struct
    from mm_assist.memory import il2cpp as il
    monkeypatch.setattr(il, "_hot_key", lambda h: 1)
    monkeypatch.setattr(il, "_HOT_REGIONS", {1: {}})
    monkeypatch.setattr(il, "_save_hot_regions", lambda *a: None)
    monkeypatch.setattr(il, "_load_hot_regions", lambda *a: None)
    monkeypatch.setattr(il, "_heap_scan_workers", lambda: 1)
    monkeypatch.setattr(il, "_list_heap_regions", lambda *a, **k: [(0x10000, 16, 0, 0)])
    tls_buffer = bytearray(struct.pack("<Q", 0x20000) * 2)
    monkeypatch.setattr(il, "read_chunk", lambda *a: (tls_buffer, 16))
    def reread_memory(*a):
        tls_buffer[:] = b"\0" * 16
        return True
    hits = il._scan_multi(None, {"x": 0x20000}, max_hits=2, hit_filter=reread_memory)
    assert hits["x"] == [0x10000, 0x10008]


def test_snapshot_waits_for_inflight_recognition_without_second_scan(monkeypatch):
    import threading
    from mm_assist.life_plan import snapshot as sm
    from mm_assist.life_plan.types import LifeSnapshot
    called = threading.Event()
    monkeypatch.setattr(sm, "_snapshot_body", lambda ts, **_k: called.set() or LifeSnapshot(ok=True))
    with sm.heavy_read_lock:
        assert sm.take_snapshot(budget_s=0.01).error == "mem_budget"
        assert sm.snapshot_in_progress()
        assert not called.is_set()
        assert sm.take_snapshot(budget_s=0.01).error == "mem_busy"
    assert called.wait(1)
    with sm._SCAN_LOCK:
        pass


def test_component_hint_rejects_reused_pid_and_wrong_type(monkeypatch, tmp_path):
    from mm_assist.memory import component_hints as ch
    monkeypatch.setattr(ch, "ROOT", tmp_path)
    monkeypatch.setattr(ch, "_birth", lambda pid: 100.0)
    monkeypatch.setattr(ch, "read_ptr", lambda h, ptr: 99)
    ch.save_hint(7, "bag", 0x10000)
    assert ch.load_hint(None, 7, "bag", 99) == 0x10000
    assert ch.load_hint(None, 7, "bag", 88) == 0
    assert ch.load_hint(None, 8, "bag", 99) == 0
    monkeypatch.setattr(ch, "_birth", lambda pid: 200.0)
    assert ch.load_hint(None, 7, "bag", 99) == 0


def test_cold_discovery_is_read_only_bounded_and_not_repeated(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from mm_assist.memory import component_hints as ch
    monkeypatch.setattr(ch, "ROOT", tmp_path)
    monkeypatch.setattr(ch, "_birth", lambda pid: "same-game")
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        ch.save_hint(7, "bag", 100)
        ch.save_hint(7, "facilities", 200)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ch.subprocess, "run", run)
    assert ch.prepare_hints(7)
    assert ch.prepare_hints(7)
    assert len(calls) == 1
    assert calls[0][0][1:] == ["-m", "mm_assist.memory.component_hints", "--warm", "7"]
    assert calls[0][1]["timeout"] == 35


def test_cold_discovery_timeout_does_not_allow_execution(monkeypatch, tmp_path):
    from mm_assist.memory import component_hints as ch
    monkeypatch.setattr(ch, "ROOT", tmp_path)
    monkeypatch.setattr(ch, "_birth", lambda pid: "same-game")
    def timeout(*a, **k):
        raise ch.subprocess.TimeoutExpired("readonly", 35)
    monkeypatch.setattr(ch.subprocess, "run", timeout)
    assert not ch.prepare_hints(7)
