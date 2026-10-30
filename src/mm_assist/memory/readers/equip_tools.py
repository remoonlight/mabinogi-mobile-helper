"""已装备生活工具耐久。配置 knowledge/ui/config/il2cpp_equip.json。"""

from __future__ import annotations

import json
import struct
import time
from typing import Any

from mm_assist.knowledge_paths import ui_config
from mm_assist.life_plan.types import ToolDurability
from mm_assist.memory.il2cpp import (
    module_base,
    parse_int,
    read_il2cpp_string,
    resolve_klass,
    scan_klass_objects,
)
from mm_assist.memory.process import read_mem, read_ptr, read_u32
from mm_assist.memory.readers.mm_dict import iter_ptr_values
from mm_assist.memory.readers.system_containers import read_sc_component
from mm_assist.memory.component_hints import load_hint, save_hint
from mm_assist.memory.evidence import root_observation


def _cfg() -> dict[str, Any]:
    return json.loads(ui_config("il2cpp_equip.json").read_text(encoding="utf-8"))


def _i32(hproc, addr: int) -> int | None:
    raw = read_mem(hproc, addr, 4)
    if not raw or len(raw) < 4:
        return None
    return int(struct.unpack("<i", raw)[0])


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


def _decode_tools(
    hproc, comp: int, off: dict[str, int], tool_klass: int | None
) -> list[ToolDurability]:
    slots = read_ptr(hproc, comp + off["EquipmentToolInventoryComponent.equipmentToolSlots"])
    if not slots:
        return []
    dict_offs = {
        "count": off["MMDictionary.count"],
        "values": off["MMDictionary.values"],
        "arr_max": off["Il2CppArray.max_length"],
        "arr_vec": off["Il2CppArray.vector"],
    }
    tools: list[ToolDurability] = []
    for obj in iter_ptr_values(hproc, slots, dict_offs):
        if tool_klass and read_ptr(hproc, obj) != tool_klass:
            continue
        dur = _i32(hproc, obj + off["EquipmentTool.Durability"])
        sheet = read_ptr(hproc, obj + off["EquipmentTool.sheetData"])
        name = ""
        mx = 0
        if sheet:
            name = _read_unique(hproc, sheet, off["EquipmentToolSheetData.DisplayName"])
            mx_v = read_u32(hproc, sheet + off["EquipmentToolSheetData.MaxDurability"])
            mx = int(mx_v or 0)
        if dur is None:
            continue
        if name and not _plausible_name(name):
            continue
        tools.append(
            ToolDurability(
                name=name or "工具", durability=float(dur), max_durability=float(mx)
            )
        )
    return tools


_TOOLS_PID = 0
_TOOLS_COMP = 0
_TOOLS_RETRY_AT = 0.0


def reset_tools_cache_for_tests() -> None:
    global _TOOLS_PID, _TOOLS_COMP, _TOOLS_RETRY_AT
    _TOOLS_PID, _TOOLS_COMP, _TOOLS_RETRY_AT = 0, 0, 0.0


def read_equipped_tools(hproc, pid: int) -> tuple[ToolDurability, ...] | None:
    cfg = _cfg()
    ga = module_base(pid, str(cfg.get("module") or "GameAssembly.dll"))
    if not ga:
        return None
    rva = parse_int(cfg["typeinfo_rva"]["EquipmentToolInventoryComponent"])
    klass = resolve_klass(hproc, ga, rva)
    if not klass:
        return None
    off = {k: parse_int(v) for k, v in cfg["offsets"].items()}
    tool_rva = parse_int(cfg["typeinfo_rva"].get("EquipmentTool") or 0)
    tool_klass = resolve_klass(hproc, ga, tool_rva) if tool_rva else None
    sc_rva = parse_int(cfg["typeinfo_rva"].get("SystemContainers") or 0)
    global _TOOLS_PID, _TOOLS_COMP, _TOOLS_RETRY_AT
    if _TOOLS_PID != int(pid):
        _TOOLS_PID, _TOOLS_COMP, _TOOLS_RETRY_AT = int(pid), load_hint(hproc, pid, "tools", klass), 0.0

    def decode(comp):
        if not comp or read_ptr(hproc, comp) != klass:
            return None
        with root_observation('tools', hproc, comp, (0, off['EquipmentToolInventoryComponent.equipmentToolSlots']), accepted=False) as root:
            rows = _decode_tools(hproc, comp, off, tool_klass)
            root.accepted = bool(rows)
        return tuple(rows) if rows else None

    cached = decode(_TOOLS_COMP)
    if cached is not None:
        return cached  # 缓存地址，每次重读耐久；不缓存耐久数值。
    _TOOLS_COMP = 0
    if time.monotonic() < _TOOLS_RETRY_AT:
        return None
    sc_comp = read_sc_component(hproc, ga, sc_rva, off["SystemContainers.EquipmentToolInventoryComponent"], klass) if sc_rva else None
    found = decode(sc_comp)
    if found is not None:
        _TOOLS_COMP = int(sc_comp)
        save_hint(pid, "tools", _TOOLS_COMP)
        return found

    def valid(_name, comp):
        nonlocal found
        rows = decode(comp)
        if rows is None:
            return False
        found = rows
        return True

    hits = scan_klass_objects(hproc, klass, max_hits=1, max_wall_s=8.0, hit_filter=valid)
    if hits:
        _TOOLS_COMP = int(hits[0])
        save_hint(pid, "tools", _TOOLS_COMP)
        return decode(_TOOLS_COMP)
    _TOOLS_RETRY_AT = time.monotonic() + 30.0
    return None
