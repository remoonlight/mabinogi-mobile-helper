"""IL2CPP 只读公共原语：模块基址、TypeInfo→klass、堆扫对象。"""

from __future__ import annotations

import json
import logging
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from mm_assist import prof
from mm_assist.memory import client_build
from mm_assist.memory.process import (
    coalesce_regions,
    decode_utf16_le,
    find_module_base,
    iter_regions,
    kernel32,
    read_floats,
    read_mem,
    read_chunk,
    read_ptr,
    read_u8,
    read_u32,
)

log = logging.getLogger(__name__)


def _heap_scan_workers() -> int:
    """并行扫区数。默认 1，避免多线程 RPM 抬高 DPC 导致 USB 掉线。上限 2。"""
    try:
        n = int(os.environ.get("MM_HEAP_SCAN_WORKERS", "1"))
    except ValueError:
        n = 1
    return max(1, min(2, n))


def parse_int(v) -> int:
    if isinstance(v, int):
        return v
    s = str(v).strip().lower()
    if s.startswith("0x"):
        return int(s, 16)
    return int(s)


def resolve_klass(hproc, ga_base: int, typeinfo_rva: int) -> int | None:
    """读 TypeInfo→klass；已是堆指针则按 PID+基址+RVA 缓存。

    不缓存 0 / 低位 token（TypeInfo 未初始化时常见 0x200xxxxx），以便稍后重试。
    """
    if not ga_base or not typeinfo_rva:
        return None
    pid = _hot_key(hproc)
    key = (int(pid), client_build.discovery_token(hproc), int(ga_base), int(typeinfo_rva))
    with _KLASS_LOCK:
        hit = _KLASS_CACHE.get(key)
    if hit:
        return int(hit)
    p = read_ptr(hproc, ga_base + typeinfo_rva)
    if p and p >= _KLASS_HEAP_MIN:
        with _KLASS_LOCK:
            _KLASS_CACHE[key] = int(p)
        return int(p)
    return p


# TypeInfo 已初始化后的 Il2CppClass 指针（x64 用户堆）；未初始化的 token 不缓存。
_KLASS_HEAP_MIN = 0x1_0000_0000
_KLASS_CACHE: dict[tuple, int] = {}
_KLASS_LOCK = threading.Lock()
# VirtualQuery 结果短缓存：同一 tick 里多个 reader 不必各走一遍地址空间。
_REGION_TTL_S = 30.0
_REGION_CACHE: dict[tuple, tuple[float, list[tuple[int, int, int, int]]]] = {}
_REGION_LOCK = threading.Lock()
_DEFAULT_CHUNK = 2 * 1024 * 1024

# 命中过对象的堆区 base → 命中次数；按 PID 共享（多 OpenProcess 句柄同热区）
_HOT_REGIONS: dict[int, dict[int, int]] = {}
from mm_assist.config import ROOT as _REPO_ROOT

_HOT_FILE = _REPO_ROOT / "logs" / "il2cpp_heap_hot.json"

# 限时定位续扫，只保留位置，不复用旧对象或旧读取结果。
_SCAN_CURSORS: dict[tuple, dict] = {}
_SCAN_CURSOR_LOCK = threading.Lock()
_SCAN_EPOCH = 0
_HEAP_READ_LOCK = threading.Lock()
_UI_MEMORY_READ_PENDING = threading.Event()
_UI_MEMORY_REQUEST_LOCK = threading.Lock()
_UI_MEMORY_REQUESTS = 0


@contextmanager
def ui_memory_read_slot(*, timeout_s: float = 0.25):
    """Bounded direct reads preempt cooperative scans without spawning readers."""
    global _UI_MEMORY_REQUESTS
    with _UI_MEMORY_REQUEST_LOCK:
        _UI_MEMORY_REQUESTS += 1
        _UI_MEMORY_READ_PENDING.set()
    acquired = False
    try:
        acquired = _HEAP_READ_LOCK.acquire(timeout=max(0.0, timeout_s))
        yield acquired
    finally:
        if acquired:
            _HEAP_READ_LOCK.release()
        with _UI_MEMORY_REQUEST_LOCK:
            _UI_MEMORY_REQUESTS -= 1
            if not _UI_MEMORY_REQUESTS:
                _UI_MEMORY_READ_PENDING.clear()


def _hot_key(hproc) -> int:
    try:
        pid = int(kernel32.GetProcessId(hproc) or 0)
        if pid:
            return pid
    except Exception:
        pass
    return int(hproc)


def _load_hot_regions(pid: int) -> None:
    """跨 state-watch 重启复用热区排序（同游戏 PID）。"""
    # 单测假句柄多为小整数；只持久化真实 PID
    if pid < 256 or pid in _HOT_REGIONS:
        return
    try:
        if not _HOT_FILE.is_file():
            return
        raw = json.loads(_HOT_FILE.read_text(encoding="utf-8"))
        entry = raw.get(str(pid)) if isinstance(raw, dict) else None
        if not isinstance(entry, dict):
            return
        hot = {int(k, 0): int(v) for k, v in entry.items() if int(v) > 0}
        if hot:
            _HOT_REGIONS[pid] = hot
            log.debug("heap hot loaded pid=%s regions=%d", pid, len(hot))
    except Exception as exc:
        log.debug("heap hot load: %s", exc)


def _save_hot_regions(pid: int) -> None:
    if pid < 256:
        return
    hot = _HOT_REGIONS.get(pid) or {}
    if not hot:
        return
    try:
        _HOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        raw: dict = {}
        if _HOT_FILE.is_file():
            try:
                raw = json.loads(_HOT_FILE.read_text(encoding="utf-8"))
            except Exception:
                raw = {}
        if not isinstance(raw, dict):
            raw = {}
        # 只留 top 热区，避免文件膨胀
        top = sorted(hot.items(), key=lambda kv: -kv[1])[:64]
        raw[str(pid)] = {hex(b): int(c) for b, c in top}
        tmp = _HOT_FILE.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(_HOT_FILE)
    except Exception as exc:
        log.debug("heap hot save: %s", exc)


def reset_scan_cache(hproc=None) -> None:
    """进程句柄失效时清热区 / TypeInfo / VirtualQuery 短缓存。"""
    global _SCAN_EPOCH
    pid = _hot_key(hproc) if hproc is not None else None
    from mm_assist.memory.diagnostics import invalidate_scans
    invalidate_scans(pid)
    with _SCAN_CURSOR_LOCK:
        _SCAN_EPOCH += 1
        for key in list(_SCAN_CURSORS):
            if pid is None or key[0] == pid:
                _SCAN_CURSORS.pop(key, None)
    if hproc is None:
        _HOT_REGIONS.clear()
        with _KLASS_LOCK:
            _KLASS_CACHE.clear()
        with _REGION_LOCK:
            _REGION_CACHE.clear()
        return
    _HOT_REGIONS.pop(pid, None)
    with _KLASS_LOCK:
        for k in [k for k in _KLASS_CACHE if k[0] == pid]:
            _KLASS_CACHE.pop(k, None)
    with _REGION_LOCK:
        for k in [k for k in _REGION_CACHE if k[0] == pid]:
            _REGION_CACHE.pop(k, None)


def _list_heap_regions(
    hproc,
    *,
    private_only: bool = True,
    writable_only: bool = True,
    should_stop: Callable[[], bool] | None = None,
) -> list[tuple[int, int, int, int]]:
    """同 PID 30 秒复用地址区枚举；超时的部分枚举不缓存。"""
    pid = _hot_key(hproc)
    key = (int(pid), client_build.discovery_token(hproc), bool(private_only), bool(writable_only))
    now = time.perf_counter()
    with _REGION_LOCK:
        ent = _REGION_CACHE.get(key)
        if ent and (now - ent[0]) < _REGION_TTL_S:
            return list(ent[1])
    regions = [
        r
        for r in iter_regions(hproc, private_only=private_only, writable_only=writable_only, should_stop=should_stop)
        if r[3] == 0x20000  # MEM_PRIVATE
    ]
    regions = coalesce_regions(regions)
    if should_stop is None or not should_stop():
        with _REGION_LOCK:
            _REGION_CACHE[key] = (time.perf_counter(), regions)
    return list(regions)


def scan_klass_objects(
    hproc,
    klass: int,
    *,
    max_hits: int = 32,
    chunk: int = _DEFAULT_CHUNK,
    should_stop: Callable[[], bool] | None = None,
    max_wall_s: float | None = None,
    hit_filter: Callable[[str, int], bool] | None = None,
    resume: bool = False,
) -> list[int]:
    res = scan_klass_objects_multi(
        hproc,
        {"_": klass},
        max_hits=max_hits,
        chunk=chunk,
        should_stop=should_stop,
        max_wall_s=max_wall_s,
        hit_filter=hit_filter,
        resume=resume,
        _caller_depth=2,
    )
    return res["_"]


def scan_klass_objects_multi(
    hproc,
    klasses: dict[str, int],
    *,
    max_hits: int | dict[str, int] = 32,
    chunk: int = _DEFAULT_CHUNK,
    should_stop: Callable[[], bool] | None = None,
    max_wall_s: float | None = None,
    max_regions: int | None = None,
    hit_filter: Callable[[str, int], bool] | None = None,
    on_first_hit: Callable[[str, list[int]], None] | None = None,
    _caller_depth: int = 1,
    resume: bool = False,
) -> dict[str, list[int]]:
    """一次遍历堆匹配多个 klass：页只读一次，多 needle 共享。

    max_wall_s：墙钟上限（秒），超时早停（保证预热可控）。
    max_regions：最多扫多少个 MEM_PRIVATE 区（热区优先）。
    hit_filter：可选 (type_name, obj_addr)→bool；仅 True 计入 max_hits。
      用于 Hud 等假阳极多的类型——否则幽灵先占满限额，活例永远扫不到。
    on_first_hit：某类型第一次命中时回调 (name, objs_so_far)，供状态栏进度。
    resume：限时结束后保留区／块／候选位置，下次接着扫；找到或扫完后重置。
    """
    caller = sys._getframe(_caller_depth).f_code.co_name if prof.enabled() else ""
    read = [0]
    t0 = time.perf_counter()
    from mm_assist.memory.diagnostics import begin_scan, publish_scan
    diagnostic_pid = _hot_key(hproc)
    diagnostic_seq = begin_scan(diagnostic_pid)
    diagnostic = {'coverage_complete': False, 'bytes_read': 0, 'regions_visited': 0,
                  'candidate_count': 0, 'reason': 'heap_read_busy'}
    def stop_scan():
        if _UI_MEMORY_READ_PENDING.is_set():
            diagnostic['interrupted_by'] = 'ui_read_request'
            return True
        if should_stop and should_stop():
            diagnostic['interrupted_by'] = 'caller'
            return True
        return False
    # 所有 reader 共用一份堆读取；等待占用同一预算，超时不另开扫描。
    while True:
        remain = None if max_wall_s is None else max_wall_s - (time.perf_counter() - t0)
        stopped = stop_scan()
        if stopped or (remain is not None and remain <= 0):
            diagnostic.update(reason=('cancelled' if diagnostic.get('interrupted_by') == 'caller' else 'heap_read_busy'),
                              elapsed_s=time.perf_counter()-t0)
            publish_scan(diagnostic_pid, diagnostic_seq, diagnostic)
            return {name: [] for name in klasses}
        if _HEAP_READ_LOCK.acquire(timeout=min(0.05, remain) if remain is not None else 0.05):
            break
    try:
        return _scan_multi(
            hproc,
            klasses,
            max_hits=max_hits,
            chunk=chunk,
            should_stop=stop_scan,
            max_wall_s=(None if max_wall_s is None else max(0.0, max_wall_s - (time.perf_counter() - t0))),
            max_regions=max_regions,
            hit_filter=hit_filter,
            on_first_hit=on_first_hit,
            _read_acc=read if caller else None,
            _resume=resume,
            _diagnostic=diagnostic,
        )
    except Exception:
        diagnostic.update(reason='scan_error', coverage_complete=False)
        raise
    finally:
        _HEAP_READ_LOCK.release()
        diagnostic['elapsed_s'] = time.perf_counter()-t0
        if diagnostic.get('interrupted_by') and diagnostic.get('reason') != 'scan_error':
            diagnostic['reason'] = 'heap_read_busy' if diagnostic['interrupted_by'] == 'ui_read_request' else 'cancelled'
        publish_scan(diagnostic_pid, diagnostic_seq, diagnostic)
        if caller:
            prof.record(
                f"heap_scan:{caller}", (time.perf_counter() - t0) * 1000.0, float(read[0])
            )


def _scan_multi(
    hproc,
    klasses: dict[str, int],
    *,
    max_hits: int | dict[str, int] = 32,
    chunk: int = _DEFAULT_CHUNK,
    should_stop: Callable[[], bool] | None = None,
    max_wall_s: float | None = None,
    max_regions: int | None = None,
    hit_filter: Callable[[str, int], bool] | None = None,
    on_first_hit: Callable[[str, list[int]], None] | None = None,
    _read_acc: list[int] | None = None,
    _resume: bool = False,
    _diagnostic: dict | None = None,
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {name: [] for name in klasses}
    needles: dict[str, bytes] = {}
    limits: dict[str, int] = {}
    for name, klass in klasses.items():
        if not klass or klass < 0x10000:
            continue
        lim = max_hits if isinstance(max_hits, int) else int(max_hits.get(name, 32))
        if lim <= 0:
            continue
        needles[name] = struct.pack("<Q", klass)
        limits[name] = lim
    pending = set(needles)
    if not pending:
        if _diagnostic is not None:
            _diagnostic.update(reason='typeinfo_uninitialized', coverage_complete=False)
        return out

    t0 = time.perf_counter()
    wall = float(max_wall_s) if max_wall_s is not None else None
    def enumeration_stop():
        return bool((should_stop and should_stop()) or (wall is not None and time.perf_counter() - t0 >= wall))

    pid = _hot_key(hproc)
    _load_hot_regions(pid)
    hot = _HOT_REGIONS.setdefault(pid, {})
    regions = _list_heap_regions(hproc, should_stop=enumeration_stop)
    all_regions = list(regions)
    enumeration_incomplete = enumeration_stop()
    visited, finished = set(), set()
    stats = {'bytes_read': 0, 'candidate_count': 0, 'rejected_candidates': 0, 'read_failures': 0}
    if threading.current_thread().name == "life-snap":
        log.info("life_scan pid=%s klasses=%s regions=%s enum_s=%.3f", pid, {k: hex(v) for k, v in klasses.items()}, len(regions), time.perf_counter() - t0)
    # 热区优先；有热区时先只扫热区，剩余时间再扫冷区
    hot_regs = [r for r in regions if hot.get(r[0], 0) > 0]
    cold_regs = [r for r in regions if hot.get(r[0], 0) <= 0]
    hot_regs.sort(key=lambda r: -hot.get(r[0], 0))
    # 冷区：大块优先（IL2CPP 托管堆通常更大）
    cold_regs.sort(key=lambda r: -r[1])
    cursor_key = None
    cursor = None
    cursor_epoch = None
    if _resume:
        cursor_key = (pid, tuple(sorted((name, klasses[name]) for name in needles)), chunk)
        with _SCAN_CURSOR_LOCK:
            cursor_epoch = _SCAN_EPOCH
            saved = _SCAN_CURSORS.get(cursor_key, {})
            cursor = {"base": saved.get("base"), "offset": saved.get("offset", 0),
                      "starts": dict(saved.get("starts", {}))}
        ordered = hot_regs + cold_regs
        for index, (base, size, *_rest) in enumerate(ordered):
            if base == cursor["base"]:
                if cursor["offset"] >= size:
                    index += 1
                    cursor = {"base": None, "offset": 0, "starts": {}}
                ordered = ordered[index:] + ordered[:index]
                break
        phases = [ordered]
    elif max_regions is not None and max_regions > 0:
        # 截断时仍保留「热+冷」比例
        regions = (hot_regs + cold_regs)[: int(max_regions)]
        phases = [regions]
    elif hot_regs and max_wall_s is not None:
        # 两阶段：热区最多用 40% 预算，剩余给冷区（避免陈旧热区吃光时间）
        phases = [hot_regs, cold_regs]
    else:
        phases = [hot_regs + cold_regs]

    lock = threading.Lock()
    workers = 1 if _resume else _heap_scan_workers()
    # 当前阶段截止（热区 40% 预算）；并行 worker 共用
    phase_deadline_box: list[float | None] = [None]

    def _timed_stop(phase_deadline: float | None = None) -> bool:
        if should_stop and should_stop():
            return True
        now = time.perf_counter()
        if wall is not None and (now - t0) >= wall:
            return True
        pd = phase_deadline if phase_deadline is not None else phase_deadline_box[0]
        if pd is not None and now >= pd:
            return True
        return False

    def _scan_one_region(base: int, size: int, _prot: int, _typ: int) -> None:
        """单区扫描；命中写入 out，满额从 pending 剔除（带锁）。"""
        nonlocal pending
        block_size = chunk
        with lock:
            if not pending or _timed_stop():
                return
            local_pending = set(pending)
        off = cursor["offset"] if cursor is not None and cursor["base"] == base else 0
        starts = dict(cursor["starts"]) if cursor is not None and cursor["base"] == base else {}
        with lock:
            visited.add(base)
        while off < size:
            if cursor is not None:
                cursor.update(base=base, offset=off, starts=starts)
            if _timed_stop():
                return
            with lock:
                if not pending:
                    return
                local_pending = set(pending)
            n = min(block_size, size - off)
            if off > 0:
                read_off = off - 7
                read_n = n + 7
            else:
                read_off = off
                read_n = n
            payload = read_chunk(hproc, base + read_off, read_n)
            with lock:
                stats['bytes_read'] += int(payload[1]) if payload else 0
                if not payload or payload[1] != read_n:
                    stats['read_failures'] += 1
            if _read_acc is not None:
                with lock:
                    _read_acc[0] += read_n
            if not payload:
                off += n
                starts = {}
                continue
            raw, got = payload
            if hit_filter is not None or on_first_hit is not None:
                # read_chunk 返回线程复用缓冲；回调的 read_ptr/read_mem 会覆盖它。
                raw = bytes(memoryview(raw)[:got])
            for name in sorted(local_pending):
                needle = needles[name]
                lim = limits[name]
                start = starts.get(name, 0)
                while True:
                    # 一块内可能有海量假命中；过滤器会继续读内存，块内也须检查预算。
                    if _timed_stop():
                        return
                    with lock:
                        if name not in pending:
                            break
                    i = raw.find(needle, start, got)
                    if i < 0:
                        starts[name] = got
                        break
                    abs_addr = base + read_off + i
                    starts[name] = i + 8
                    if abs_addr % 8 == 0:
                        with lock:
                            stats['candidate_count'] += 1
                        # 过滤在锁外：活体判定可能读内存，勿堵其它区
                        snapshot: list[int] | None = None
                        if hit_filter is not None:
                            try:
                                if not hit_filter(name, abs_addr):
                                    with lock:
                                        stats['rejected_candidates'] += 1
                                    start = i + 8
                                    continue
                            except Exception:
                                with lock:
                                    stats['rejected_candidates'] += 1
                                start = i + 8
                                continue
                        with lock:
                            hits = out[name]
                            if name not in pending:
                                break
                            if abs_addr not in hits:
                                first = len(hits) == 0
                                hits.append(abs_addr)
                                hot[base] = hot.get(base, 0) + 1
                                if first:
                                    snapshot = list(hits)
                            if len(hits) >= lim:
                                pending.discard(name)
                        if snapshot is not None and on_first_hit is not None:
                            try:
                                on_first_hit(name, snapshot)
                            except Exception:
                                pass
                    start = i + 8
            off += n
            starts = {}
            if cursor is not None:
                cursor.update(base=base, offset=off, starts=starts)
        with lock:
            finished.add(base)

    def _scan_region_list(
        regs: list, *, phase_deadline: float | None = None
    ) -> None:
        nonlocal pending
        if not regs:
            return
        phase_deadline_box[0] = phase_deadline
        try:
            if workers <= 1 or len(regs) < 2:
                for base, size, prot, typ in regs:
                    if not pending or _timed_stop():
                        break
                    _scan_one_region(base, size, prot, typ)
                return
            # 按区并行：同句柄只读；多 klass 仍一次匹配（比 warm 两轮串行扫堆快）
            log.debug(
                "heap scan parallel workers=%d regions=%d types=%d",
                workers,
                len(regs),
                len(needles),
            )
            with ThreadPoolExecutor(max_workers=min(workers, len(regs))) as pool:
                futs = [
                    pool.submit(_scan_one_region, base, size, prot, typ)
                    for base, size, prot, typ in regs
                ]
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except Exception as exc:
                        with lock:
                            stats['read_failures'] += 1
                        log.debug("heap parallel region: %s", exc)
                    if _timed_stop():
                        # 不 cancel：ReadProcessMemory 中途打断不稳；靠 _timed_stop 早退
                        break
        finally:
            phase_deadline_box[0] = None

    if len(phases) == 2 and wall is not None:
        hot_deadline = t0 + wall * 0.4
        _scan_region_list(phases[0], phase_deadline=hot_deadline)
        if pending and not _timed_stop():
            _scan_region_list(phases[1], phase_deadline=None)
    else:
        for regs in phases:
            if not pending or _timed_stop():
                break
            _scan_region_list(regs)

    _save_hot_regions(pid)
    if _diagnostic is not None:
        # Slice statistics only. A resumed tail, hit limit or failed read cannot prove full coverage.
        full = (bool(all_regions) and bool(pending) and not enumeration_incomplete and not stats['read_failures']
                and len(finished) == len(all_regions) and not _timed_stop()
                and (not _resume or not cursor or not saved.get('base')))
        reason = ('read_failed' if stats['read_failures'] else 'coverage_complete' if full
                  else 'target_limit_reached' if not pending else 'coverage_incomplete')
        _diagnostic.update(stats, regions_visited=len(visited), regions_total=len(all_regions),
                           coverage_complete=bool(full), reason=reason,
                           accepted_candidates=sum(len(v) for v in out.values()),
                           resumed=bool(_resume), coverage_scope='current_region_snapshot_slice')
    if cursor_key is not None:
        stopped = _timed_stop()
        with _SCAN_CURSOR_LOCK:
            # reset 后迟到的旧扫描不复活已清除的位置。
            if cursor_epoch == _SCAN_EPOCH:
                if pending and stopped and cursor is not None:
                    _SCAN_CURSORS[cursor_key] = cursor
                else:
                    _SCAN_CURSORS.pop(cursor_key, None)
    return out


def module_base(pid: int, name: str = "GameAssembly.dll") -> int | None:
    return find_module_base(pid, name)


def read_xy(hproc, addr: int) -> tuple[float, float] | None:
    vals = read_floats(hproc, addr, 2)
    if not vals:
        return None
    return float(vals[0]), float(vals[1])


def read_xyz(hproc, addr: int) -> tuple[float, float, float] | None:
    vals = read_floats(hproc, addr, 3)
    if not vals:
        return None
    return float(vals[0]), float(vals[1]), float(vals[2])


def plausible_xy(x: float, y: float) -> bool:
    if abs(x) < 1e-4 and abs(y) < 1e-4:
        return False
    if abs(x) > 1e6 or abs(y) > 1e6:
        return False
    if x != x or y != y:
        return False
    return True


def read_il2cpp_string(hproc, addr: int, *, max_chars: int = 256) -> str | None:
    """读 System.String：length@0x10，chars@0x14（UTF-16LE）。

    默认 max_chars=256，覆盖资源路径；短标签可传更小值。
    """
    if not addr or addr < 0x10000:
        return None
    raw_len = read_mem(hproc, addr + 0x10, 4)
    if not raw_len or len(raw_len) < 4:
        return None
    length = struct.unpack("<i", raw_len)[0]
    if length <= 0 or length > max_chars:
        return None
    raw = read_mem(hproc, addr + 0x14, length * 2)
    if not raw or len(raw) < length * 2:
        return None
    return decode_utf16_le(raw[: length * 2])


# re-export
__all__ = [
    "parse_int",
    "resolve_klass",
    "scan_klass_objects",
    "scan_klass_objects_multi",
    "module_base",
    "read_xy",
    "read_xyz",
    "plausible_xy",
    "read_il2cpp_string",
    "read_ptr",
    "read_u8",
    "read_u32",
    "read_floats",
    "reset_scan_cache",
    "coalesce_regions",
]
