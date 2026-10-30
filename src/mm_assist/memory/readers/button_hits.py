"""IL2CPP 只读：枚举当前堆内按钮控制体的触发矩形。

路径：
- TweeningButtonControl.raycastTargetRect / TweeningToggleButtonControl.buttonRect（托管 Rect）
- ButtonControl → UISlot.Raw / RaycastTarget → Graphic.m_RectTransform → m_CachedPtr
  再对 native 指针做启发式 float 扫描，筛客户区内合理框

配置：`knowledge/ui/config/il2cpp_button_hits.json`
探针：`python -m mm_assist.tools.il2cpp_button_hits_probe`
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm_assist.memory.il2cpp import (
    module_base,
    parse_int,
    resolve_klass,
    scan_klass_objects_multi,
)
from mm_assist.memory.process import (
    close_handle,
    find_game_pid,
    open_process,
    read_floats,
    read_ptr,
    read_u8,
)

from mm_assist.knowledge_paths import BUTTON_HITS_JSON

CONFIG_REL = BUTTON_HITS_JSON


def default_config_path(knowledge_root: Path | None = None) -> Path:
    if knowledge_root is None:
        return BUTTON_HITS_JSON
    return Path(knowledge_root) / "ui" / "config" / "il2cpp_button_hits.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or default_config_path()
    raw = json.loads(p.read_text(encoding="utf-8-sig"))
    offs = {k: parse_int(v) for k, v in (raw.get("offsets") or {}).items()}
    tis = {k: parse_int(v) for k, v in (raw.get("typeinfo_rva") or {}).items()}
    return {
        "path": p,
        "typeinfo_rva": tis,
        "offsets": offs,
        "native_rect_probe_offsets": [
            int(x) for x in (raw.get("native_rect_probe_offsets") or [])
        ],
        "max_hits": int(raw.get("max_hits") or 512),
    }


@dataclass
class HitRect:
    x: float
    y: float
    w: float
    h: float
    space: str  # client_px | local | unknown | clip

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]

    def plausible_client(self, cw: float, ch: float) -> bool:
        if self.w < 4 or self.h < 4 or self.w > cw or self.h > ch:
            return False
        if self.x < -40 or self.y < -40:
            return False
        if self.x + self.w > cw + 40 or self.y + self.h > ch + 40:
            return False
        return True


@dataclass
class ButtonHit:
    kind: str
    address: int
    interactable: bool | None = None
    enabled: bool | None = None
    raycast_target: bool | None = None
    rect: HitRect | None = None
    rect_source: str | None = None
    graphic: int | None = None
    rect_transform: int | None = None
    native_ptr: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "address": hex(self.address),
            "interactable": self.interactable,
            "enabled": self.enabled,
            "raycast_target": self.raycast_target,
            "rect_source": self.rect_source,
            "graphic": hex(self.graphic) if self.graphic else None,
            "rect_transform": hex(self.rect_transform) if self.rect_transform else None,
            "native_ptr": hex(self.native_ptr) if self.native_ptr else None,
            "notes": self.notes,
        }
        if self.rect:
            d["rect"] = {
                "x": round(self.rect.x, 2),
                "y": round(self.rect.y, 2),
                "w": round(self.rect.w, 2),
                "h": round(self.rect.h, 2),
                "space": self.rect.space,
                "xywh": [round(v, 2) for v in self.rect.as_list()],
            }
        else:
            d["rect"] = None
        return d


def _read_rect(hproc, addr: int) -> HitRect | None:
    fs = read_floats(hproc, addr, 4)
    if not fs or len(fs) < 4:
        return None
    x, y, w, h = (float(fs[0]), float(fs[1]), float(fs[2]), float(fs[3]))
    if any(v != v or abs(v) > 1e7 for v in (x, y, w, h)):
        return None
    return HitRect(x, y, w, h, "unknown")


def _score_client_rect(r: HitRect, cw: float, ch: float) -> float:
    if not r.plausible_client(cw, ch):
        return -1.0
    # Prefer interior, non-tiny, not full-screen
    area = r.w * r.h
    full = cw * ch
    if area > 0.85 * full:
        return 0.1
    score = 1.0
    if 16 <= r.w <= 400 and 16 <= r.h <= 200:
        score += 1.0
    if 0 <= r.x and 0 <= r.y and r.x + r.w <= cw and r.y + r.h <= ch:
        score += 1.5
    return score


def _read_local_rect_native(hproc, native: int) -> HitRect | None:
    """Unity RectTransform native 缓存 rect：本机构造为 +0xB8 (x,y,w,h)。"""
    if not native or native < 0x10000:
        return None
    base = native & ~7
    r = _read_rect(hproc, base + 0xB8)
    if r is None:
        return None
    if r.w < 2 or r.h < 2 or r.w > 4000 or r.h > 4000:
        return None
    if any(v != v for v in (r.x, r.y, r.w, r.h)):
        return None
    r.space = "local"
    return r


def _probe_native_rect(
    hproc,
    native: int,
    *,
    cw: float,
    ch: float,
    offsets: list[int],
) -> tuple[HitRect | None, str | None]:
    """优先读已验证的 local rect@+0xB8；其余 offset 仅作屏幕框候选。"""
    local = _read_local_rect_native(hproc, native)
    if local is not None:
        # 若已是客户区坐标（少见），升级标记
        if local.plausible_client(cw, ch) and local.x >= 0 and local.y >= 0:
            local.space = "client_px"
            return local, "native+0xb8"
        return local, "native+0xb8"
    if not native or native < 0x10000:
        return None, None
    base = native & ~7
    best: HitRect | None = None
    best_score = -1.0
    best_off: int | None = None
    for off in offsets:
        if off == 0xB8:
            continue
        r = _read_rect(hproc, base + off)
        if r is None:
            continue
        r.space = "client_px"
        sc = _score_client_rect(r, cw, ch)
        if sc > best_score:
            best_score = sc
            best = r
            best_off = off
    if best is None or best_score < 2.0:
        return None, None
    return best, f"native+0x{best_off:x}"


def _resolve_graphic_from_button(
    hproc, btn: int, offs: dict[str, int]
) -> tuple[int | None, list[str]]:
    notes: list[str] = []
    # Prefer ButtonControl.RaycastTarget UISlot → Raw
    slot = read_ptr(hproc, btn + offs["ButtonControl.RaycastTarget"])
    if slot:
        raw = read_ptr(hproc, slot + offs["UISlot.Raw"])
        if raw:
            notes.append("via_RaycastTarget_slot")
            return raw, notes
    # Fallback: uiSlot → ButtonSlot.RaycastTarget
    ui_slot = read_ptr(hproc, btn + offs["ButtonControl.uiSlot"])
    if not ui_slot:
        notes.append("no_uiSlot")
        return None, notes
    button_slot = read_ptr(hproc, ui_slot + offs["UISlot.Raw"])
    if not button_slot:
        notes.append("no_ButtonSlot")
        return None, notes
    graphic = read_ptr(hproc, button_slot + offs["ButtonSlot.RaycastTarget"])
    if graphic:
        notes.append("via_ButtonSlot.RaycastTarget")
    else:
        notes.append("no_graphic")
    return graphic, notes


def _rect_from_graphic(
    hproc,
    graphic: int,
    offs: dict[str, int],
    *,
    cw: float,
    ch: float,
    native_offsets: list[int],
) -> tuple[HitRect | None, str | None, int | None, int | None]:
    rt = read_ptr(hproc, graphic + offs["Graphic.m_RectTransform"])
    if not rt:
        rt = read_ptr(hproc, graphic + offs["UIBehaviour.rectTransformCached"])
    native = None
    if rt:
        native = read_ptr(hproc, rt + offs["UnityObject.m_CachedPtr"])
    # MaskableGraphic previousClipRect (canvas/clip space — may match client for overlay)
    clip = _read_rect(hproc, graphic + offs["MaskableGraphic.previousClipRect"])
    valid = read_u8(hproc, graphic + offs["MaskableGraphic.previousValidRect"])
    if clip and valid == 1:
        clip.space = "clip"
        if clip.plausible_client(cw, ch):
            clip.space = "client_px"
            return clip, "MaskableGraphic.previousClipRect", rt, native
    if native:
        rect, src = _probe_native_rect(
            hproc, native, cw=cw, ch=ch, offsets=native_offsets
        )
        if rect:
            return rect, src, rt, native
    return None, None, rt, native


@dataclass
class ButtonHitsFrame:
    pid: int
    gameassembly: int
    client_w: int
    client_h: int
    age_ms: float
    buttons: list[ButtonHit] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    scan_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        with_rect = sum(1 for b in self.buttons if b.rect)
        return {
            "pid": self.pid,
            "gameassembly": hex(self.gameassembly),
            "client": [self.client_w, self.client_h],
            "age_ms": round(self.age_ms, 1),
            "button_count": len(self.buttons),
            "with_rect_count": with_rect,
            "scan_counts": self.scan_counts,
            "errors": self.errors,
            "buttons": [b.to_dict() for b in self.buttons],
        }


class Il2CppButtonHitsReader:
    def __init__(self, *, config_path: Path | None = None) -> None:
        self._cfg = load_config(config_path)
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
        client_w: int = 1920,
        client_h: int = 1440,
        max_hits: int | None = None,
    ) -> ButtonHitsFrame | None:
        if not self._ensure():
            return None
        assert self._h and self._ga and self._pid
        offs = self._cfg["offsets"]
        tis = self._cfg["typeinfo_rva"]
        native_offs = self._cfg["native_rect_probe_offsets"]
        limit = int(max_hits or self._cfg["max_hits"])
        t0 = time.perf_counter()
        errors: list[str] = []
        klasses: dict[str, int] = {}
        for name, rva in tis.items():
            klass = resolve_klass(self._h, self._ga, rva)
            if not klass:
                errors.append(f"klass_missing:{name}")
                continue
            klasses[name] = klass
        if not klasses:
            self.last_error = "no_klasses"
            return None
        objs = scan_klass_objects_multi(
            self._h, klasses, max_hits=limit, chunk=1024 * 1024
        )
        buttons: list[ButtonHit] = []
        for kind, addrs in objs.items():
            for addr in addrs:
                hit = ButtonHit(kind=kind, address=addr)
                if kind == "ButtonControl":
                    ib = read_u8(self._h, addr + offs["ButtonControl.interactable"])
                    en = read_u8(self._h, addr + offs["ButtonControl.enabled"])
                    hit.interactable = bool(ib) if ib in (0, 1) else None
                    hit.enabled = bool(en) if en in (0, 1) else None
                    graphic, notes = _resolve_graphic_from_button(self._h, addr, offs)
                    hit.notes.extend(notes)
                    hit.graphic = graphic
                    if graphic:
                        rt_flag = read_u8(
                            self._h, graphic + offs["Graphic.m_RaycastTarget"]
                        )
                        hit.raycast_target = (
                            bool(rt_flag) if rt_flag in (0, 1) else None
                        )
                        rect, src, rt, native = _rect_from_graphic(
                            self._h,
                            graphic,
                            offs,
                            cw=float(client_w),
                            ch=float(client_h),
                            native_offsets=native_offs,
                        )
                        hit.rect = rect
                        hit.rect_source = src
                        hit.rect_transform = rt
                        hit.native_ptr = native
                elif kind == "TweeningButtonControl":
                    ib = read_u8(
                        self._h, addr + offs["TweeningButtonControl.Interactable"]
                    )
                    hit.interactable = bool(ib) if ib in (0, 1) else None
                    managed = _read_rect(
                        self._h,
                        addr + offs["TweeningButtonControl.raycastTargetRect"],
                    )
                    if managed:
                        managed.space = "local"
                        hit.rect = managed
                        hit.rect_source = "TweeningButtonControl.raycastTargetRect"
                        # If already looks like client px, upgrade
                        if managed.plausible_client(float(client_w), float(client_h)):
                            managed.space = "client_px"
                    # Also try Graphic via TweeningButtonSlot
                    slot = read_ptr(self._h, addr + 0x10)  # uiSlot
                    if slot:
                        raw = read_ptr(self._h, slot + offs["UISlot.Raw"])
                        if raw:
                            # TweeningButtonSlot.RaycastTarget @ 0x18
                            graphic = read_ptr(self._h, raw + 0x18)
                            hit.graphic = graphic
                            if graphic:
                                rect, src, rt, native = _rect_from_graphic(
                                    self._h,
                                    graphic,
                                    offs,
                                    cw=float(client_w),
                                    ch=float(client_h),
                                    native_offsets=native_offs,
                                )
                                hit.rect_transform = rt
                                hit.native_ptr = native
                                if rect and (
                                    hit.rect is None or hit.rect.space != "client_px"
                                ):
                                    hit.rect = rect
                                    hit.rect_source = src
                elif kind == "TweeningToggleButtonControl":
                    ib = read_u8(
                        self._h, addr + offs["TweeningToggleButtonControl.interactable"]
                    )
                    hit.interactable = bool(ib) if ib in (0, 1) else None
                    managed = _read_rect(
                        self._h,
                        addr + offs["TweeningToggleButtonControl.buttonRect"],
                    )
                    if managed:
                        managed.space = "local"
                        hit.rect = managed
                        hit.rect_source = "TweeningToggleButtonControl.buttonRect"
                        if managed.plausible_client(float(client_w), float(client_h)):
                            managed.space = "client_px"
                buttons.append(hit)
        age = (time.perf_counter() - t0) * 1000.0
        return ButtonHitsFrame(
            pid=self._pid,
            gameassembly=self._ga,
            client_w=client_w,
            client_h=client_h,
            age_ms=age,
            buttons=buttons,
            errors=errors,
            scan_counts={k: len(v) for k, v in objs.items()},
        )


__all__ = [
    "ButtonHit",
    "ButtonHitsFrame",
    "HitRect",
    "Il2CppButtonHitsReader",
    "default_config_path",
    "load_config",
]
