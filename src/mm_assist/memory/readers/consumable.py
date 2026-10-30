"""消耗品背包只读。配置 knowledge/ui/config/il2cpp_consumable.json。"""

from __future__ import annotations

import json
import struct
from typing import Any

from mm_assist.knowledge_paths import ui_config
from mm_assist.memory.il2cpp import (
    module_base,
    parse_int,
    read_il2cpp_string,
    resolve_klass,
    scan_klass_objects,
)
from mm_assist.memory.process import read_mem, read_ptr
from mm_assist.memory.readers.mm_dict import iter_ptr_values
from mm_assist.memory.readers.system_containers import read_sc_component
from mm_assist.memory.component_hints import load_hint, save_hint
from mm_assist.memory.evidence import root_observation


def _cfg() -> dict[str, Any]:
    return json.loads(ui_config("il2cpp_consumable.json").read_text(encoding="utf-8"))


def _i64(hproc, addr: int) -> int | None:
    raw = read_mem(hproc, addr, 8)
    if not raw or len(raw) < 8:
        return None
    return int(struct.unpack("<q", raw)[0])


def _plausible_name(name: str) -> bool:
    if not name or len(name) > 64:
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in name)


def _read_unique(hproc, obj: int, field_off: int) -> str:
    us = read_ptr(hproc, obj + field_off)
    if not us:
        return ""
    direct = read_il2cpp_string(hproc, us, max_chars=64)
    if direct:
        return direct
    inner = read_ptr(hproc, us)
    return read_il2cpp_string(hproc, inner, max_chars=64) or "" if inner else ""


def _decode_bag(hproc, comp: int, off: dict[str, int], stack_klass: int | None) -> dict[str, int]:
    stacks = read_ptr(hproc, comp + off["ConsumableInventoryComponent.ConsumableStacks"])
    if not stacks:
        return {}
    dict_offs = {
        "count": off["MMDictionary.count"],
        "values": off["MMDictionary.values"],
        "arr_max": 0x18,
        "arr_vec": 0x20,
    }
    bag: dict[str, int] = {}
    for stk in iter_ptr_values(hproc, stacks, dict_offs):
        if stack_klass and read_ptr(hproc, stk) != stack_klass:
            continue
        sheet = read_ptr(hproc, stk + off["ConsumableStack.SheetData"])
        cnt = _i64(hproc, stk + off["ConsumableStack.Count"])
        if not sheet or cnt is None or cnt <= 0 or cnt > 10**12:
            continue
        name = _read_unique(hproc, sheet, off["ConsumableSheetData.DisplayName"])
        if not _plausible_name(name):
            continue
        bag[name] = bag.get(name, 0) + int(cnt)
    return bag


_BAG_PID = 0
_BAG_COMP = 0


def reset_bag_cache_for_tests() -> None:
    global _BAG_PID, _BAG_COMP
    _BAG_PID = 0
    _BAG_COMP = 0


def _try_bag(
    hproc, comp: int, klass: int, off: dict[str, int], stack_klass: int | None
) -> dict[str, int]:
    if not comp or read_ptr(hproc, comp) != klass:
        return {}
    with root_observation('items', hproc, comp, (0, off['ConsumableInventoryComponent.ConsumableStacks']), accepted=False) as root:
        result = _decode_bag(hproc, comp, off, stack_klass)
        root.accepted = bool(result)
        return result


def read_consumable_counts(hproc, pid: int, *, allow_scan: bool = True) -> dict[str, int] | None:
    cfg = _cfg()
    ga = module_base(pid, str(cfg.get("module") or "GameAssembly.dll"))
    if not ga:
        return None
    rva = parse_int(cfg["typeinfo_rva"]["ConsumableInventoryComponent"])
    klass = resolve_klass(hproc, ga, rva)
    if not klass:
        return None
    off = {k: parse_int(v) for k, v in cfg["offsets"].items()}
    stack_rva = parse_int(cfg["typeinfo_rva"].get("ConsumableStack") or 0)
    stack_klass = resolve_klass(hproc, ga, stack_rva) if stack_rva else None
    sc_rva = parse_int(cfg["typeinfo_rva"].get("SystemContainers") or 0)
    global _BAG_PID, _BAG_COMP
    if _BAG_PID != int(pid) or not _BAG_COMP:
        _BAG_PID = int(pid)
        _BAG_COMP = load_hint(hproc, pid, "bag", klass)
    if _BAG_PID == int(pid) and _BAG_COMP:
        cached = _try_bag(hproc, _BAG_COMP, klass, off, stack_klass)
        if cached:
            return cached
        _BAG_COMP = 0
    sc_comp = (
        read_sc_component(
            hproc,
            ga,
            sc_rva,
            off["SystemContainers.ConsumableInventoryComponent"],
            klass,
        )
        if sc_rva
        else None
    )
    if sc_comp:
        bag = _try_bag(hproc, sc_comp, klass, off, stack_klass)
        if bag:
            _BAG_PID = int(pid)
            _BAG_COMP = int(sc_comp)
            save_hint(pid, "bag", _BAG_COMP)
            return bag
    if not allow_scan:
        return None
    best: dict[str, int] = {}
    best_comp = 0

    def _real_bag(_name: str, addr: int) -> bool:
        nonlocal best, best_comp
        bag = _try_bag(hproc, addr, klass, off, stack_klass)
        if not bag:
            return False
        if len(bag) > len(best):
            best = bag
            best_comp = int(addr)
        return True

    scan_klass_objects(
        hproc,
        klass,
        max_hits=4,
        max_wall_s=20.0,
        hit_filter=_real_bag,
    )
    if not best:
        return None
    _BAG_PID = int(pid)
    _BAG_COMP = best_comp
    save_hint(pid, "bag", _BAG_COMP)
    return best
