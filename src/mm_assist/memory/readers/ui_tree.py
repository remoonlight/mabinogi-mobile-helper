"""IL2CPP 只读：优先 UI 组件一帧快照（地址 + 已知字段）。

配置：
- `knowledge/ui/config/il2cpp_ui_priority.json`（TypeInfo）
- `knowledge/ui/config/il2cpp_scene.json`（已验证偏移，复用）

调用方：`python -m mm_assist.tools.il2cpp_ui_tree_probe`
产出 schema：`{pid, gameassembly, types: {TypeName: [{address, fields, plausible}]}}`
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm_assist.memory.readers.scene import load_config as load_scene_config
from mm_assist.memory.il2cpp import module_base, parse_int, resolve_klass, scan_klass_objects
from mm_assist.memory.process import close_handle, find_game_pid, open_process, read_u8, read_u32

log = logging.getLogger(__name__)

from mm_assist.knowledge_paths import UI_PRIORITY_JSON

PRIORITY_REL = UI_PRIORITY_JSON

KNOWN_FIELDS: dict[str, list[tuple[str, int, str]]] = {
    "DialoguePlayerComponent": [
        ("DialogueIndex", 0x10, "u32"),
        ("DialoguePlayType", 0x28, "u32"),
        ("DialoguePlayState", 0x44, "u32"),
        ("IsActive", 0x48, "bool"),
        ("IsRequested", 0x49, "bool"),
        ("IsReloading", 0x4A, "bool"),
        ("CurrentKeyframeIndex", 0x50, "u32"),
    ],
    "MapUIVisibleComponent": [("IsVisible", 0x10, "bool")],
    "MinimapUIVisibleComponent": [("IsVisible", 0x28, "bool")],
    "InventoryVisibleDataComponent": [("IsInventoryVisible", 0x10, "bool")],
    "CombatUIComponent": [("IsUIVisible", 0x38, "bool")],
    "HudUIVisibleComponent": [("HideAll", 0x28, "bool")],
    "CurrentViewingMapUIComponent": [
        ("MapType", 0x20, "u32"),
        ("GeoDataKey", 0x24, "u32"),
        ("ViewingChannelSheetId", 0x2C, "u32"),
        ("CurrentChannelSheetId", 0x3C, "u32"),
    ],
    "SkillButtonsUIComponent": [
        ("IsHudUIVisible", 0x30, "bool"),
        ("IsCombatUIVisible", 0x31, "bool"),
    ],
    "HudActionUIComponent": [
        ("IsVisible", 0x42, "bool"),
        ("IsTweening", 0x43, "bool"),
        ("IsNeedUpdate", 0x44, "bool"),
    ],
    "BasicHudActionButtonControl": [
        ("ControlKeyId", 0x38, "u32"),
    ],
    "BaseHudActionButtonControl": [
        ("IsVisible", 0x28, "bool"),
    ],
    "EquipmentRepairListUIComponent": [
        ("isShowReserved", 0x48, "bool"),
        ("isActive", 0x49, "bool"),
        ("isModalOpened", 0x4A, "bool"),
    ],
    "RadioDialoguePlayerComponent": [
        ("isPlayingRadioDialogue", 0x24, "bool"),
    ],
    "ChattingUIStoredDataComponent": [
        ("IsFocus", 0x40, "bool"),
    ],
    "ChatDisplayDataComponent": [
        ("IsFullScreen", 0x18, "bool"),
        ("IsMiddleScreen", 0x24, "bool"),
    ],
}


def default_priority_path(knowledge_root: Path | None = None) -> Path:
    if knowledge_root is None:
        return Path.cwd() / PRIORITY_REL
    return Path(knowledge_root) / "ui" / "config" / "il2cpp_ui_priority.json"


def load_priority(path: Path | None = None) -> dict[str, int]:
    p = path or default_priority_path()
    out: dict[str, int] = {}
    if p.is_file():
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
        for k, v in (raw.get("typeinfo_rva") or {}).items():
            out[k] = parse_int(v)
    scene = load_scene_config()
    for k, v in scene["typeinfo_rva"].items():
        out.setdefault(k, int(v))
    return out


@dataclass
class UiObjectSnap:
    type_name: str
    address: int
    fields: dict[str, Any] = field(default_factory=dict)
    plausible: bool = True


@dataclass
class UiFrame:
    pid: int
    gameassembly: int
    age_ms: float
    types: dict[str, list[UiObjectSnap]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "gameassembly": hex(self.gameassembly),
            "age_ms": round(self.age_ms, 1),
            "type_count": len(self.types),
            "object_count": sum(len(v) for v in self.types.values()),
            "errors": self.errors,
            "types": {
                name: [
                    {
                        "address": hex(o.address),
                        "plausible": o.plausible,
                        "fields": o.fields,
                    }
                    for o in objs
                ]
                for name, objs in self.types.items()
            },
        }


def _read_field(hproc, base: int, off: int, kind: str) -> Any | None:
    addr = base + off
    if kind == "bool":
        v = read_u8(hproc, addr)
        if v is None:
            return None
        if v not in (0, 1):
            return {"raw": v, "ok": False}
        return bool(v)
    if kind == "u32":
        return read_u32(hproc, addr)
    return None


def _fields_plausible(fields: dict[str, Any]) -> bool:
    for v in fields.values():
        if isinstance(v, dict) and v.get("ok") is False:
            return False
    return True


class Il2CppUiTreeReader:
    def __init__(
        self,
        *,
        priority_path: Path | None = None,
        max_hits: int = 8,
    ) -> None:
        self._priority = load_priority(priority_path)
        self._max_hits = max_hits
        self._pid: int | None = None
        self._h = None
        self._ga: int | None = None
        self.last_error: str | None = None

    def close(self) -> None:
        if self._h:
            close_handle(self._h)
            self._h = None

    def _ensure(self) -> bool:
        if self._h and self._ga:
            return True
        pid = find_game_pid()
        if not pid:
            self.last_error = "process_not_found"
            return False
        h = open_process(pid)
        if not h:
            self.last_error = "open_process_failed"
            return False
        ga = module_base(pid, "GameAssembly.dll")
        if not ga:
            close_handle(h)
            self.last_error = "gameassembly_not_found"
            return False
        self._pid = pid
        self._h = h
        self._ga = ga
        return True

    def capture(
        self,
        *,
        type_names: list[str] | None = None,
        allow_scan: bool = True,
    ) -> UiFrame | None:
        t0 = time.perf_counter()
        if not self._ensure():
            return None
        assert self._pid and self._h and self._ga
        names = type_names or sorted(KNOWN_FIELDS.keys())
        frame = UiFrame(pid=self._pid, gameassembly=self._ga, age_ms=0.0)
        for name in names:
            rva = self._priority.get(name)
            if not rva:
                frame.errors.append(f"{name}:no_rva")
                continue
            klass = resolve_klass(self._h, self._ga, rva)
            if not klass:
                frame.errors.append(f"{name}:no_klass")
                continue
            if not allow_scan:
                continue
            objs = scan_klass_objects(self._h, klass, max_hits=self._max_hits)
            snaps: list[UiObjectSnap] = []
            field_defs = KNOWN_FIELDS.get(name, [])
            for addr in objs:
                fields: dict[str, Any] = {}
                for fname, off, kind in field_defs:
                    fields[fname] = _read_field(self._h, addr, off, kind)
                snap = UiObjectSnap(
                    type_name=name,
                    address=addr,
                    fields=fields,
                    plausible=_fields_plausible(fields) if field_defs else True,
                )
                snaps.append(snap)
            if field_defs:
                good = [s for s in snaps if s.plausible]
                snaps = good or snaps
            frame.types[name] = snaps
        frame.age_ms = (time.perf_counter() - t0) * 1000
        return frame
