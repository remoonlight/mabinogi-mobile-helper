"""游戏进程只读内存获取基准（零键鼠）。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mm_assist.config import ROOT
from mm_assist.memory.il2cpp import parse_int, resolve_klass, scan_klass_objects_multi
from mm_assist.memory.process import (
    find_game_pid,
    find_module_base,
    iter_regions,
    open_process,
    close_handle,
    read_chunk,
    read_mem,
)

log = logging.getLogger(__name__)

LEDGER_DIR = ROOT / "logs" / "benchmarks" / "memory"
SCHEMA = "memory_benchmark_run.v1"
METHODS = ("rpm_into",)
CURRENT_PRODUCTION = "rpm_into"
# 1–2 MiB 连读会抬高 DPC，USB 主机控制器容易掉设备；默认不上这些档。
SIZES = (8, 64, 4096, 65536, 1 * 1024 * 1024, 2 * 1024 * 1024)
USB_SAFE_SIZES = (8, 64, 4096, 65536)
BELOW_NORMAL = 0x00004000
NORMAL_PRIORITY = 0x00000020
IMPROVE_PCT = 15.0
ACCURACY_OK_MIN = 1.0
from mm_assist.knowledge_paths import POSITION_JSON as POS_JSON


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    k = (len(ys) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ys) - 1)
    frac = k - lo
    return float(ys[lo] * (1.0 - frac) + ys[hi] * frac)


def _stats(ms: list[float]) -> dict[str, float]:
    if not ms:
        return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    return {
        "n": float(len(ms)),
        "p50_ms": round(_percentile(ms, 50), 4),
        "p95_ms": round(_percentile(ms, 95), 4),
        "p99_ms": round(_percentile(ms, 99), 4),
        "mean_ms": round(float(statistics.fmean(ms)), 4),
    }


def usb_yield(*, nbytes: int = 0, extra_s: float = 0.0) -> None:
    """大块读取后稍停，降低 USB 掉线风险。"""
    pause = extra_s
    if nbytes >= 65536:
        pause += 0.008
    elif nbytes >= 4096:
        pause += 0.001
    if pause > 0:
        time.sleep(pause)


def set_benchmark_priority(*, low: bool) -> None:
    from mm_assist.memory.process import kernel32

    cls = BELOW_NORMAL if low else NORMAL_PRIORITY
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), cls)


class ReadMeter:
    def __init__(self) -> None:
        self.calls = 0
        self.bytes_req = 0
        self.bytes_got = 0
        self.fails = 0

    def record(self, req: int, got: int) -> None:
        self.calls += 1
        self.bytes_req += int(req)
        self.bytes_got += int(got)
        if got <= 0:
            self.fails += 1

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "bytes_req": self.bytes_req,
            "bytes_got": self.bytes_got,
            "fails": self.fails,
        }


def _read(
    hproc,
    addr: int,
    size: int,
    *,
    method: str,
    meter: ReadMeter,
    buf=None,
) -> bytes | None:
    del buf, method
    chunk = read_chunk(hproc, addr, size)
    if not chunk:
        meter.record(size, 0)
        return None
    ba, got = chunk
    meter.record(size, got)
    return bytes(ba[:got])


def pick_module_read_base(hproc, pid: int, *, module: str = "GameAssembly.dll") -> tuple[int, int] | None:
    base = find_module_base(pid, module)
    if not base:
        return None
    for rbase, rsize, _prot, _typ in iter_regions(hproc):
        if rbase <= base < rbase + rsize:
            return int(rbase), int(rsize)
        if rbase == base:
            return int(rbase), int(rsize)
    raw = read_mem(hproc, base, 4096)
    if raw:
        return int(base), 4096
    return None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_static_ladder(
    hproc,
    base: int,
    region_size: int,
    *,
    methods: tuple[str, ...] = METHODS,
    warmup: int = 3,
    repeats: int = 20,
    sizes: tuple[int, ...] | None = None,
    usb_safe: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ref_hash: dict[int, str] = {}
    order = list(methods)
    ladder = sizes if sizes is not None else (USB_SAFE_SIZES if usb_safe else SIZES)
    for size in ladder:
        if size > region_size:
            continue
        for method in order:
            meter = ReadMeter()
            times: list[float] = []
            hashes: list[str] = []
            ok_n = 0
            for i in range(max(0, warmup) + max(1, repeats)):
                t0 = time.perf_counter()
                raw = _read(hproc, base, size, method=method, meter=meter)
                dt = (time.perf_counter() - t0) * 1000.0
                if usb_safe:
                    usb_yield(nbytes=size)
                if i < warmup:
                    continue
                times.append(dt)
                if raw and len(raw) == size:
                    ok_n += 1
                    hashes.append(sha256_hex(raw))
            hx = hashes[0] if hashes else ""
            if size not in ref_hash and hx:
                ref_hash[size] = hx
            match = bool(hx) and all(h == ref_hash[size] for h in hashes)
            mb_s = 0.0
            if times and ok_n:
                mean_s = (sum(times) / len(times)) / 1000.0
                if mean_s > 0:
                    mb_s = (size / (1024 * 1024)) / mean_s
            rows.append(
                {
                    "workload": "static_ladder",
                    "method": method,
                    "size": size,
                    "ok_ratio": ok_n / max(1, repeats),
                    "hash_match": match,
                    "mb_s": round(mb_s, 3),
                    **_stats(times),
                    **meter.as_dict(),
                }
            )
        random.shuffle(order)
        if usb_safe:
            usb_yield(extra_s=0.02)
    return rows


def _load_pos_cfg() -> dict[str, Any]:
    if not POS_JSON.is_file():
        return {}
    return json.loads(POS_JSON.read_text(encoding="utf-8"))


def resolve_sparse_fields(hproc, pid: int) -> list[tuple[int, int]]:
    """解析坐标主源对象上的若干相邻字段。失败返回空。"""
    cfg = _load_pos_cfg()
    ga = find_module_base(pid, str(cfg.get("module") or "GameAssembly.dll"))
    if not ga:
        return []
    src = (cfg.get("sources") or {}).get("map_component") or {}
    rva = parse_int(src.get("typeinfo_rva") or 0)
    off = parse_int(src.get("xy_offset") or 0x10)
    klass = resolve_klass(hproc, ga, rva)
    if not klass:
        return []
    prev = os.environ.get("MM_HEAP_SCAN_WORKERS")
    os.environ["MM_HEAP_SCAN_WORKERS"] = "1"
    try:
        hits = scan_klass_objects_multi(
            hproc,
            {"map_component": klass},
            max_hits=4,
            max_wall_s=4.0,
            max_regions=12,
            chunk=1024 * 1024,
        )
    finally:
        if prev is None:
            os.environ.pop("MM_HEAP_SCAN_WORKERS", None)
        else:
            os.environ["MM_HEAP_SCAN_WORKERS"] = prev
    objs = list(hits.get("map_component") or [])
    if not objs:
        return []
    obj = int(objs[0])
    return [(obj + off, 8), (obj + off + 8, 8)]


def run_sparse_poll(
    hproc,
    fields: list[tuple[int, int]],
    *,
    methods: tuple[str, ...] = METHODS,
    warmup: int = 3,
    repeats: int = 20,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order = list(methods)
    random.shuffle(order)
    for method in order:
        meter = ReadMeter()
        times: list[float] = []
        ok_n = 0
        plausible = 0
        for i in range(max(0, warmup) + max(1, repeats)):
            t0 = time.perf_counter()
            ok = True
            xy_ok = False
            for addr, size in fields:
                raw = _read(hproc, addr, size, method=method, meter=meter)
                if not raw:
                    ok = False
                    break
                if size >= 8:
                    xy_ok = True
            dt = (time.perf_counter() - t0) * 1000.0
            if i < warmup:
                continue
            times.append(dt)
            if ok:
                ok_n += 1
            if xy_ok:
                plausible += 1
            usb_yield(extra_s=0.001)
        rows.append(
            {
                "workload": "sparse_fields",
                "method": method,
                "fields": len(fields),
                "ok_ratio": ok_n / max(1, repeats),
                "plausible_ratio": plausible / max(1, repeats),
                **_stats(times),
                **meter.as_dict(),
            }
        )
    return rows


def run_heap_scan(
    hproc,
    pid: int,
    *,
    warmup: int = 0,
    repeats: int = 1,
    workers: tuple[int, ...] = (1,),
    chunks: tuple[int, ...] = (1024 * 1024,),
    max_wall_s: float = 4.0,
    usb_safe: bool = True,
) -> list[dict[str, Any]]:
    cfg = _load_pos_cfg()
    ga = find_module_base(pid, str(cfg.get("module") or "GameAssembly.dll"))
    if not ga:
        return [{"workload": "heap_scan", "error": "no_gameassembly"}]
    src = (cfg.get("sources") or {}).get("map_component") or {}
    rva = parse_int(src.get("typeinfo_rva") or 0)
    klass = resolve_klass(hproc, ga, rva)
    if not klass:
        return [{"workload": "heap_scan", "error": "no_klass"}]
    rows: list[dict[str, Any]] = []
    prev = os.environ.get("MM_HEAP_SCAN_WORKERS")
    try:
        for w in workers:
            os.environ["MM_HEAP_SCAN_WORKERS"] = str(w)
            for chunk in chunks:
                times: list[float] = []
                hits_n: list[int] = []
                for i in range(max(0, warmup) + max(1, repeats)):
                    t0 = time.perf_counter()
                    res = scan_klass_objects_multi(
                        hproc,
                        {"map_component": klass},
                        max_hits=8,
                        chunk=chunk,
                        max_wall_s=max_wall_s,
                        max_regions=16 if usb_safe else None,
                    )
                    dt = (time.perf_counter() - t0) * 1000.0
                    if usb_safe:
                        usb_yield(extra_s=0.2)
                    if i < warmup:
                        continue
                    times.append(dt)
                    hits_n.append(len(res.get("map_component") or []))
                rows.append(
                    {
                        "workload": "heap_scan",
                        "method": "rpm_into",
                        "workers": w,
                        "chunk": chunk,
                        "hits_mean": round(float(statistics.fmean(hits_n or [0])), 2),
                        **_stats(times),
                    }
                )
    finally:
        if prev is None:
            os.environ.pop("MM_HEAP_SCAN_WORKERS", None)
        else:
            os.environ["MM_HEAP_SCAN_WORKERS"] = prev
    return rows


def rank_static(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ladder = [r for r in rows if r.get("workload") == "static_ladder" and r.get("hash_match")]
    by_m: dict[str, list[float]] = {}
    for r in ladder:
        by_m.setdefault(str(r["method"]), []).append(float(r.get("p95_ms") or 0.0))
    ranked = []
    for method, xs in by_m.items():
        ranked.append({"method": method, "p95_ms_mean": round(float(statistics.fmean(xs)), 4)})
    ranked.sort(key=lambda x: x["p95_ms_mean"])
    return ranked


def build_scorecard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """准确性优先。生产只保留 rpm_into。"""
    methods = list(METHODS)
    card: list[dict[str, Any]] = []
    by_m: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        m = r.get("method")
        if not m:
            continue
        by_m.setdefault(str(m), []).append(r)
    for method in methods:
        items = by_m.get(method) or []
        ladder = [r for r in items if r.get("workload") == "static_ladder"]
        sparse = [r for r in items if r.get("workload") == "sparse_fields"]
        hash_ok = [float(bool(r.get("hash_match"))) for r in ladder]
        ok_ratio = [float(r.get("ok_ratio") or 0.0) for r in ladder]
        p95 = [float(r.get("p95_ms") or 0.0) for r in ladder]
        plaus = [float(r.get("plausible_ratio") or 0.0) for r in sparse]
        acc = True
        if ladder:
            acc = all(h >= ACCURACY_OK_MIN for h in hash_ok) and all(
                o >= ACCURACY_OK_MIN for o in ok_ratio
            )
        elif not items:
            acc = False
        card.append(
            {
                "method": method,
                "role": "current" if method == CURRENT_PRODUCTION else "candidate",
                "accuracy_ok": acc,
                "hash_match": round(float(statistics.fmean(hash_ok)), 3) if hash_ok else 0.0,
                "read_ok": round(float(statistics.fmean(ok_ratio)), 3) if ok_ratio else 0.0,
                "plausible": round(float(statistics.fmean(plaus)), 3) if plaus else None,
                "p95_ms": round(float(statistics.fmean(p95)), 4) if p95 else None,
            }
        )
    return card


def decide_default(
    ranked: list[dict[str, Any]],
    scorecard: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    keep = CURRENT_PRODUCTION
    drop: list[str] = []
    card = {r["method"]: r for r in (scorecard or [])}
    if card:
        for method, row in card.items():
            if method == CURRENT_PRODUCTION:
                continue
            if not row.get("accuracy_ok"):
                drop.append(method)
        accurate = [
            r
            for r in ranked
            if (card.get(r["method"]) or {}).get("accuracy_ok")
        ]
        ranked = accurate or []
    if not ranked:
        return {
            "keep": keep,
            "drop": drop,
            "reason": "no_ranked_rows",
            "current": CURRENT_PRODUCTION,
        }
    base = next((r for r in ranked if r["method"] == CURRENT_PRODUCTION), None)
    if base is None:
        return {
            "keep": keep,
            "drop": drop + [r["method"] for r in ranked],
            "reason": "baseline_missing_or_inaccurate",
            "current": CURRENT_PRODUCTION,
        }
    best = ranked[0]
    base_p = float(base["p95_ms_mean"]) or 1.0
    gain = (base_p - float(best["p95_ms_mean"])) / base_p * 100.0
    if best["method"] != CURRENT_PRODUCTION and gain >= IMPROVE_PCT:
        keep = str(best["method"])
        drop = [m for m in METHODS if m != keep]
        return {
            "keep": keep,
            "drop": drop,
            "reason": f"accuracy_ok and p95 improved {gain:.1f}% vs {CURRENT_PRODUCTION}",
            "gain_pct": round(gain, 1),
            "current": CURRENT_PRODUCTION,
        }
    drop = [m for m in METHODS if m != CURRENT_PRODUCTION]
    return {
        "keep": CURRENT_PRODUCTION,
        "drop": drop,
        "reason": "no_material_gain_keep_current",
        "best": best["method"],
        "gain_pct": round(gain, 1),
        "current": CURRENT_PRODUCTION,
    }


def format_rank_table(out: dict[str, Any]) -> str:
    return format_scorecard_table(out)


def format_scorecard_table(out: dict[str, Any]) -> str:
    lines = [
        "method\trole\taccuracy_ok\thash\tread_ok\tplausible\tp95_ms",
        "------\t----\t-----------\t----\t-------\t---------\t------",
    ]
    for r in out.get("scorecard") or []:
        lines.append(
            "\t".join(
                [
                    str(r.get("method")),
                    str(r.get("role")),
                    str(r.get("accuracy_ok")),
                    str(r.get("hash_match")),
                    str(r.get("read_ok")),
                    str(r.get("plausible")),
                    str(r.get("p95_ms")),
                ]
            )
        )
    dec = out.get("decision") or {}
    lines.append(f"current\t{dec.get('current') or CURRENT_PRODUCTION}")
    lines.append(f"keep\t{dec.get('keep')} ({dec.get('reason')})")
    if dec.get("drop"):
        lines.append("drop\t" + ",".join(str(x) for x in dec["drop"]))
    return "\n".join(lines) + "\n"


def run_live_memory_benchmark(
    *,
    hwnd: int | None = None,
    warmup: int = 2,
    repeats: int = 8,
    include_heap: bool = True,
    usb_safe: bool = True,
    ledger_dir: Path | None = None,
) -> dict[str, Any]:
    pid = find_game_pid(hwnd=hwnd)
    if not pid:
        raise RuntimeError("game_process_missing")
    if usb_safe:
        set_benchmark_priority(low=True)
    hproc = open_process(pid)
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    try:
        picked = pick_module_read_base(hproc, pid)
        if not picked:
            raise RuntimeError("gameassembly_unreadable")
        base, rsize = picked
        rows.extend(
            run_static_ladder(
                hproc,
                base,
                rsize,
                warmup=warmup,
                repeats=repeats,
                usb_safe=usb_safe,
            )
        )
        fields = resolve_sparse_fields(hproc, pid)
        if fields:
            rows.extend(run_sparse_poll(hproc, fields, warmup=warmup, repeats=repeats))
        else:
            rows.append({"workload": "sparse_fields", "error": "no_fields"})
        if include_heap:
            rows.extend(
                run_heap_scan(
                    hproc,
                    pid,
                    warmup=0,
                    repeats=1,
                    usb_safe=usb_safe,
                )
            )
    finally:
        close_handle(hproc)
        if usb_safe:
            set_benchmark_priority(low=False)
    ranked = rank_static(rows)
    scorecard = build_scorecard(rows)
    out = {
        "schema": SCHEMA,
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pid": pid,
        "elapsed_s": round(time.time() - t0, 3),
        "usb_safe": usb_safe,
        "current": CURRENT_PRODUCTION,
        "results": rows,
        "ranked": ranked,
        "scorecard": scorecard,
        "decision": decide_default(ranked, scorecard),
    }
    dest = Path(ledger_dir or LEDGER_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"run_{out['ts']}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = dest / "latest.md"
    md.write_text(format_rank_table(out), encoding="utf-8")
    out["path"] = str(path)
    return out


def wait_job_done(*, timeout_s: float = 600.0, path: Path | None = None) -> dict[str, Any]:
    from mm_assist.jobs.bar import load_job

    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = load_job(path)
        st = str(last.get("status") or "")
        if st in ("ok", "fail"):
            return last
        time.sleep(0.25)
    raise TimeoutError(f"job 超时 status={last.get('status')!r}")
