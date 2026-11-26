"""加工设施队列 + 单槽进度只读。配置 knowledge/ui/config/il2cpp_facility.json。

进度不在 ValueTuple 里，而在 AlteringComponent.EntireAlteringStatusData
（AlteringWorkStatus.startAt/endAt，与 UI ProgressRatio 同口径）。关菜单可读。
OCR 槽位%仅开设施页时旁证。不得宣称 C2。
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

from mm_assist.knowledge_paths import ROOT, ui_config
from mm_assist.life_plan.types import FACILITY_ORDER, FacilityId, QueueSlot, SLOT_CAP
from mm_assist.memory.il2cpp import (
    module_base,
    parse_int,
    read_il2cpp_string,
    resolve_klass,
    scan_klass_objects,
)
from mm_assist.memory.process import read_mem, read_ptr, read_u32, read_u8
from mm_assist.memory.readers.mm_dict import iter_key_int_pairs, iter_key_ptr_pairs
from mm_assist.memory.component_hints import load_hint, save_hint
from mm_assist.memory.evidence import root_observation

# dump FacilityType：0金属 1木材 2布料 3皮革 4药品 5食材
_KEY_TO_FACILITY: dict[int, FacilityId] = {
    0: "metal",
    1: "wood",
    2: "cloth",
    4: "medicine",
}

_DOTNET_UNIX_TICKS = 621355968000000000
_CATALOG: dict[int, str] | None = None
_CATALOG_FT: dict[int, int] | None = None
_COMPONENTS: dict[tuple[int, str], int] = {}
_MAX_QUEUE_LIST = 32


def _cfg() -> dict[str, Any]:
    return json.loads(ui_config("il2cpp_facility.json").read_text(encoding="utf-8"))


def _i64(hproc, addr: int) -> int | None:
    raw = read_mem(hproc, addr, 8)
    if not raw or len(raw) < 8:
        return None
    return struct.unpack("<q", raw)[0]


def _i32(hproc, addr: int) -> int | None:
    raw = read_mem(hproc, addr, 4)
    if not raw or len(raw) < 4:
        return None
    return struct.unpack("<i", raw)[0]


def _load_catalog(*, root: Path | None = None) -> tuple[dict[int, str], dict[int, int]]:
    global _CATALOG, _CATALOG_FT
    if _CATALOG is not None and _CATALOG_FT is not None:
        return _CATALOG, _CATALOG_FT
    path = (root or ROOT) / "knowledge" / "items" / "altering_process_catalog.json"
    names: dict[int, str] = {}
    fts: dict[int, int] = {}
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        for rows in (raw.get("by_facility") or {}).values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sid = int(row.get("altering_sheet_id") or 0)
                name = str(row.get("display_name") or row.get("result_name") or "")
                if sid > 0 and name:
                    names[sid] = name
                    raw_ft = row.get("facility_type")
                    fts[sid] = int(raw_ft) if raw_ft is not None else -1
    _CATALOG, _CATALOG_FT = names, fts
    return names, fts


def sheet_name_map(*, root: Path | None = None) -> dict[int, str]:
    return _load_catalog(root=root)[0]


def sheet_facility_type(sheet_id: int) -> int:
    return int(_load_catalog()[1].get(int(sheet_id), -1))


def resolve_sheet_name(sheet_id: int, *, fallback: str = "") -> str:
    if int(sheet_id) <= 0:
        return fallback
    return sheet_name_map().get(int(sheet_id), fallback or str(int(sheet_id)))


def now_raw_like(sample: int) -> int:
    """把本机时钟对齐到样本 RealTime.rawValue 的量级。"""
    unix_ticks = int(time.time() * 10_000_000)
    n = int(sample)
    if n >= 5 * 10**17:
        return unix_ticks + _DOTNET_UNIX_TICKS
    if n >= 10**14:
        return unix_ticks
    if n >= 10**11:
        return int(time.time() * 1000)
    return int(time.time())


def ticks_to_seconds(delta: int, *, ticks_per_second: int = 10_000_000) -> float:
    """Convert with the timestamp's unit; a duration's size cannot identify it."""
    return int(delta) / float(ticks_per_second)


def progress_from_ticks(
    start: int,
    end: int,
    *,
    now: int | None = None,
    required_ticks: int = 0,
    ticks_per_second: int | None = None,
) -> tuple[float, float]:
    """返回 (progress_ratio 0–1, remain_s)。"""
    dur = int(end) - int(start)
    if dur <= 0:
        dur = int(required_ticks)
    if dur <= 0:
        return 0.0, 0.0
    reference = start if start > 0 else end
    clock = int(now if now is not None else now_raw_like(reference))
    elapsed = clock - int(start)
    ratio = max(0.0, min(1.0, elapsed / float(dur)))
    scale = ticks_per_second
    if scale is None:
        # Same units as now_raw_like: .NET/Unix ticks, Unix ms, or seconds.
        scale = 10_000_000 if reference >= 10**14 else (1000 if reference >= 10**11 else 1)
    remain = max(0.0, ticks_to_seconds(int(start) + dur - clock, ticks_per_second=scale))
    return ratio, remain


def pick_latest_slots(
    recs: list[tuple[int, QueueSlot]], *, cap: int = SLOT_CAP
) -> tuple[QueueSlot, ...]:
    """同一设施多份堆对象时，按 start 去重后取最近 cap 格（当前队列）。"""
    seen: set[tuple[int, int, str]] = set()
    uniq: list[tuple[int, QueueSlot]] = []
    for start, slot in recs:
        key = (int(slot.sheet_id), int(start), slot.recipe)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((int(start), slot))
    uniq.sort(key=lambda x: x[0], reverse=True)
    chosen = list(reversed(uniq[:cap]))
    return tuple(s for _, s in chosen)


def _empty() -> dict[FacilityId, tuple[QueueSlot, ...]]:
    return {fac: () for fac in FACILITY_ORDER}


def _dict_offs(off: dict[str, int]) -> dict[str, int]:
    return {
        "count": off["MMDictionary.count"],
        "keys": off["MMDictionary.keys"],
        "values": off["MMDictionary.values"],
        "arr_max": off["Il2CppArray.max_length"],
        "arr_vec": off["Il2CppArray.vector"],
    }


def _read_ptr_list(hproc, list_ptr: int, offs: dict[str, int]) -> list[int] | None:
    if not list_ptr:
        return None
    size = read_u32(hproc, list_ptr + offs["MMList._size"])
    if size is None or size > _MAX_QUEUE_LIST:
        return None
    if size == 0:
        return []
    items = read_ptr(hproc, list_ptr + offs["MMList._items"])
    if not items:
        return None
    n_arr = int(read_u32(hproc, items + offs["Il2CppArray.max_length"]) or 0)
    if n_arr < size:
        return None
    vec = items + offs["Il2CppArray.vector"]
    out: list[int] = []
    for i in range(size):
        p = read_ptr(hproc, vec + i * 8)
        if p and p > 0x10000:
            out.append(int(p))
        else:
            return None
    return out


def _read_unique(hproc, obj: int, field_off: int) -> str:
    us = read_ptr(hproc, obj + field_off)
    if not us:
        return ""
    direct = read_il2cpp_string(hproc, us, max_chars=64)
    if direct:
        return direct
    inner = read_ptr(hproc, us)
    return read_il2cpp_string(hproc, inner, max_chars=64) or "" if inner else ""


def _work_to_slot(hproc, work: int, off: dict[str, int]) -> QueueSlot | None:
    if not work:
        return None
    base = work + off["AlteringWorkStatus.replicatedData"]
    sheet_id = int(_i32(hproc, base + off["AlteringWorkReplicatedData.SheetId"]) or 0)
    required = int(_i64(hproc, base + off["AlteringWorkReplicatedData.RequiredTimeTicks"]) or 0)
    start = int(_i64(hproc, work + off["AlteringWorkStatus.startAt"]) or 0)
    end = int(_i64(hproc, work + off["AlteringWorkStatus.endAt"]) or 0)
    sheet = read_ptr(hproc, work + off["AlteringWorkStatus.alteringSheetData"])
    name = ""
    if sheet:
        name = _read_unique(hproc, sheet, off["AlteringSheetData.DisplayName"])
        if sheet_id <= 0:
            sheet_id = int(_i32(hproc, sheet + off["AlteringSheetData.Id"]) or 0)
    if not name:
        name = resolve_sheet_name(sheet_id)
    if not name:
        return None
    ratio, remain = progress_from_ticks(start, end, required_ticks=required)
    done = bool(ratio >= 0.999 or (end > 0 and now_raw_like(end) >= end))
    return QueueSlot(
        recipe=name,
        done=done,
        progress_ratio=round(ratio, 4),
        remain_s=round(remain, 1),
        sheet_id=int(sheet_id),
        start_ms=int(start) if int(start) > 0 else 0,
        end_ms=int(end) if int(end) > 0 else 0,
    )


def _read_status_queues(
    hproc,
    infos: int,
    off: dict[str, int],
) -> dict[FacilityId, tuple[QueueSlot, ...]]:
    cur = _empty()
    count = read_u32(hproc, infos + off["MMDictionary.count"])
    if count is None or not 1 <= count <= 6:
        return {}
    pairs = list(iter_key_ptr_pairs(hproc, infos, _dict_offs(off)))
    if len(pairs) != count or len({key for key, _ in pairs}) != count or any(key > 5 for key, _ in pairs):
        return {}
    if not {0, 1, 2, 4}.issubset({key for key, _ in pairs}):
        return {}
    for key, lst in pairs:
        fac = _KEY_TO_FACILITY.get(int(key))
        if fac is None:
            continue
        slots: list[QueueSlot] = []
        queues = _read_ptr_list(hproc, lst, off)
        if queues is None:
            return {}
        for q in queues:
            works = _read_ptr_list(
                hproc, read_ptr(hproc, q + off["AlteringWorkStatusQueue.workQueue"]), off
            )
            if works is None or len(slots) + len(works) > _MAX_QUEUE_LIST:
                return {}
            for work in works:
                slot = _work_to_slot(hproc, work, off)
                if slot is None:
                    return {}
                slots.append(slot)
        cur[fac] = tuple(slots)
    return cur


def _scan_slot_cap_dict(
    hproc,
    pid: int,
    cfg: dict[str, Any],
    ga: int,
    off: dict[str, int],
    *,
    typeinfo_key: str,
    field: str,
) -> dict[FacilityId, int]:
    rva = parse_int((cfg.get("typeinfo_rva") or {}).get(typeinfo_key) or 0)
    if not rva:
        return {}
    klass = resolve_klass(hproc, ga, rva)
    if not klass:
        return {}
    dict_offs = _dict_offs(off)
    field_off = off.get(field)
    if field_off is None:
        return {}

    def decode(comp: int) -> dict[FacilityId, int]:
        if read_ptr(hproc, comp) != klass:
            return {}
        addr = comp + field_off
        dptr = read_ptr(hproc, addr)
        if not dptr:
            return {}
        count_addr = dptr + dict_offs['count']
        count = read_u32(hproc, count_addr)
        if count is None or not 1 <= count <= 6:
            return {}
        pairs = dict(iter_key_int_pairs(hproc, dptr, dict_offs))
        if (len(pairs) != count or dict(iter_key_int_pairs(hproc, dptr, dict_offs)) != pairs
                or read_u32(hproc, count_addr) != count or read_ptr(hproc, addr) != dptr):
            return {}
        return {fac: pairs[key] for key, fac in _KEY_TO_FACILITY.items()
                if key in pairs and 1 <= pairs[key] <= 8}

    def verify(comp: int, expected: dict[FacilityId, int] | None = None) -> dict[FacilityId, int]:
        with root_observation('facility_slot_caps', hproc, comp, (0, field_off), accepted=False) as root:
            caps = decode(comp)
            root.accepted = bool(caps) and (expected is None or caps == expected)
            return caps if root.accepted else {}

    hinted = load_hint(hproc, pid, typeinfo_key, klass)
    if hinted:
        caps = verify(hinted)
        if caps:
            return caps

    candidates: dict[int, dict[FacilityId, int]] = {}

    def valid(_name: str, comp: int) -> bool:
        caps = decode(comp)
        if caps:
            candidates[comp] = caps
        return bool(caps)

    hits = scan_klass_objects(hproc, klass, max_hits=4, max_wall_s=4.0, hit_filter=valid)
    if not hits:
        return {}
    comp = hits[0]
    expected = candidates[comp]
    if any(candidates[hit] != expected for hit in hits[1:]):
        return {}
    caps = verify(comp, expected)
    if len(caps) == len(FACILITY_ORDER):
        save_hint(pid, typeinfo_key, comp)
    return caps


def read_facility_slot_caps(hproc, pid: int) -> dict[FacilityId, int] | None:
    """优先 queueCapacities，其次 FacilitySlotCapacities；只覆盖四类加工。"""
    cfg = _cfg()
    ga = module_base(pid, str(cfg.get("module") or "GameAssembly.dll"))
    if not ga:
        return None
    off = {k: parse_int(v) for k, v in cfg["offsets"].items()}
    primary = _scan_slot_cap_dict(
        hproc, pid,
        cfg,
        int(ga),
        off,
        typeinfo_key="AlteringFacilityQueueComponent",
        field="AlteringFacilityQueueComponent.queueCapacities",
    )
    if len(primary) == len(FACILITY_ORDER):
        return primary
    secondary = _scan_slot_cap_dict(
        hproc, pid,
        cfg,
        int(ga),
        off,
        typeinfo_key="FacilityComponent",
        field="FacilityComponent.FacilitySlotCapacities",
    )
    return {**secondary, **primary} or None


def read_facility_queues(
    hproc, pid: int
) -> dict[FacilityId, tuple[QueueSlot, ...]] | None:
    """只读当前组件挂接的队列；堆中孤立 WorkStatus 不能证明仍在队列。"""
    cfg = _cfg()
    ga = module_base(pid, str(cfg.get("module") or "GameAssembly.dll"))
    if not ga:
        return None
    off = {k: parse_int(v) for k, v in cfg["offsets"].items()}
    rva = parse_int(cfg["typeinfo_rva"].get("AlteringComponent") or 0)
    klass = resolve_klass(hproc, ga, rva) if rva else None
    if not klass:
        return None
    key = (int(pid), "AlteringComponent")

    def decode(comp):
        if not comp or read_ptr(hproc, comp) != klass:
            return None
        with root_observation('facilities', hproc, comp, (0, off['AlteringComponent.EntireAlteringStatusData']), accepted=False) as root:
            status = read_ptr(hproc, comp + off["AlteringComponent.EntireAlteringStatusData"])
            if not status:
                return None
            with root_observation('facilities', hproc, status, (0, off['EntireAlteringStatus.alteringWorkStatusQueueListMap']), accepted=False) as status_root:
                infos = read_ptr(hproc, status + off["EntireAlteringStatus.alteringWorkStatusQueueListMap"])
                result = (_read_status_queues(hproc, infos, off) or None) if infos else None
                root.accepted = status_root.accepted = result is not None
                return result

    if key not in _COMPONENTS:
        _COMPONENTS[key] = load_hint(hproc, pid, "facilities", klass)
    cached = decode(_COMPONENTS.get(key))
    if cached is not None:
        return cached
    _COMPONENTS.pop(key, None)
    found = None

    def valid(_name, comp):
        nonlocal found
        rows = decode(comp)
        if rows is None:
            return False
        found = rows
        _COMPONENTS[key] = int(comp)
        return True

    scan_klass_objects(hproc, klass, max_hits=1, max_wall_s=20.0, hit_filter=valid)
    if found is not None:
        save_hint(pid, "facilities", _COMPONENTS[key])
    return found
