"""IL2CPP 只读：MainHud 快捷钮（背包 / ESC / 左下 / 右上摄影机）可见态 + Button 本地矩形。

链（dump.cs）：
  InventoryButtonUIComponent.ui@0x10 → InventoryButtonUI.buttonControl@0x20
    → MainHudButtonControl.buttonControl@0x28 → ButtonControl → Graphic → RectTransform local@+0xB8
  可见：buttonState@0x3C（MainHudButtonState: Invalid=0 Show=1 Hide=2 Combat=3）
       · isMainHudModularUIVisible@0x38 · MainHudButtonControl.currentState@0x38

  MainMenuButtonUIComponent.ui@0x10 → MainMenuButtonUI.buttonControl@0x18 → …
    可见：isVisible@0x31 · isMainHudModularUIVisible@0x30

  MainHudLeftBottomButtonUIComponent：同 Menu 布局；buttonState@0x34

  MainHudTopRightButtonSetUIComponent.ui@0x10 → … .cameraButtonControl@0x30 → ButtonControl
    可见：isVisible@0x31

出现机制：各 Component.Update(...) 吃 ``HudUIVisibleComponent`` + 战斗/教程/解锁等，
写入 Show/Hide/Combat；非独立「Icon_* 贴图指针」。

探针：``python -m mm_assist.tools.il2cpp_hud_main_icons_probe``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm_assist.memory.readers.button_hits import (
    HitRect,
    _rect_from_graphic,
    _resolve_graphic_from_button,
    default_config_path as button_hits_config_path,
    load_config as load_button_hits_config,
)
from mm_assist.memory.readers.ui_tree import default_priority_path, load_priority
from mm_assist.memory.il2cpp import module_base, parse_int, resolve_klass, scan_klass_objects
from mm_assist.memory.process import close_handle, find_game_pid, open_process, read_ptr, read_u8, read_u32

BUTTON_STATE = {0: "Invalid", 1: "Show", 2: "Hide", 3: "Combat"}

# Component → (ui_off, path_to_MainHudButtonControl_or_Button, fields)
# path steps are offsets from previous object; last targets ButtonControl unless noted.
SPECS: dict[str, dict[str, Any]] = {
    "InventoryButtonUIComponent": {
        "ui": 0x10,
        # InventoryButtonUI.buttonControl@0x20 → MainHudButtonControl
        "to_mhb": (0x20,),
        "mod_vis": 0x38,
        "button_state": 0x3C,
        "icon": "Icon_Inventory",
    },
    "MainMenuButtonUIComponent": {
        "ui": 0x10,
        # MainMenuButtonUI.buttonControl@0x18
        "to_mhb": (0x18,),
        "mod_vis": 0x30,
        "is_visible": 0x31,
        "icon": "Icon_Menu",
    },
    "MainHudLeftBottomButtonUIComponent": {
        "ui": 0x10,
        "to_mhb": (0x18,),
        "mod_vis": 0x30,
        "button_state": 0x34,
        "icon": "Icon_Class_gstar_x2 / Icon_Crafting_x2",
    },
    "MainHudTopRightButtonSetUIComponent": {
        "ui": 0x10,
        # MainHudTopRightButtonSetUI.cameraButtonControl@0x30 → ButtonControl 直达
        "to_button": (0x30,),
        "mod_vis": 0x30,
        "is_visible": 0x31,
        "icon": "hud.end_cam (cameraButtonControl)",
    },
}

OFF_MHB_BUTTON = 0x28  # MainHudButtonControl.buttonControl
OFF_MHB_STATE = 0x38  # currentState


def _boolish(v: int | None) -> bool | None:
    if v is None:
        return None
    if v in (0, 1):
        return bool(v)
    return None


def _ptr_ok(p: int | None) -> bool:
    """Windows x64 用户态指针；拒 -1 / 非规范高位。"""
    if not p:
        return False
    if p >= 0x0000_8000_0000_0000:
        return False
    return p >= 0x1_0000_0000


def _snap_early_stop(snap: "HudMainIconSnap") -> bool:
    """lean 早停：必须 Show（或 isVisible）+ 可点，避免吃到半截 Hide 就停。"""
    if not _ptr_ok(snap.button_addr):
        return False
    if snap.interactable is not True:
        return False
    st = snap.button_state or snap.mhb_state
    if st == "Show":
        return True
    if snap.is_visible is True:
        return True
    return False


def _follow(h, base: int, offs: tuple[int, ...]) -> int | None:
    cur = base
    for off in offs:
        cur = read_ptr(h, cur + off)
        if not _ptr_ok(cur):
            return None
    return cur


@dataclass
class HudMainIconSnap:
    kind: str
    address: int
    icon_hint: str
    modular_visible: bool | None = None
    is_visible: bool | None = None
    button_state: str | None = None
    button_state_raw: int | None = None
    mhb_state: str | None = None
    button_addr: int | None = None
    interactable: bool | None = None
    enabled: bool | None = None
    rect: HitRect | None = None
    rect_source: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "address": hex(self.address),
            "icon_hint": self.icon_hint,
            "modular_visible": self.modular_visible,
            "is_visible": self.is_visible,
            "button_state": self.button_state,
            "button_state_raw": self.button_state_raw,
            "mhb_state": self.mhb_state,
            "button": hex(self.button_addr) if self.button_addr else None,
            "interactable": self.interactable,
            "enabled": self.enabled,
            "rect_source": self.rect_source,
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


@dataclass
class HudMainIconsFrame:
    pid: int
    gameassembly: int
    client_w: int
    client_h: int
    age_ms: float
    icons: list[HudMainIconSnap] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "gameassembly": hex(self.gameassembly),
            "client_w": self.client_w,
            "client_h": self.client_h,
            "age_ms": round(self.age_ms, 1),
            "icons": [i.to_dict() for i in self.icons],
            "icon_count": len(self.icons),
            "with_rect": sum(1 for i in self.icons if i.rect is not None),
            "errors": self.errors,
        }


class Il2CppHudMainIconsReader:
    def __init__(
        self,
        *,
        priority_path: Path | None = None,
        button_hits_path: Path | None = None,
        max_per_kind: int = 8,
    ) -> None:
        self._priority = load_priority(priority_path or default_priority_path())
        self._btn_cfg = load_button_hits_config(
            button_hits_path or button_hits_config_path()
        )
        self._max_per_kind = max_per_kind
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

    def _rect_from_button(self, button: int, cw: int, ch: int) -> tuple[HitRect | None, str | None, list[str]]:
        offs = self._btn_cfg["offsets"]
        native_offs = [
            parse_int(x) if not isinstance(x, int) else x
            for x in (self._btn_cfg.get("native_rect_probe_offsets") or [0xB8])
        ]
        if 0xB8 not in native_offs:
            native_offs = [0xB8, *native_offs]
        graphic, gnotes = _resolve_graphic_from_button(self._h, button, offs)
        if not graphic or not _ptr_ok(graphic):
            return None, None, gnotes
        rect, src, _rt, _nat = _rect_from_graphic(
            self._h,
            graphic,
            offs,
            cw=float(cw),
            ch=float(ch),
            native_offsets=native_offs,
        )
        return rect, src, gnotes

    def capture(
        self,
        *,
        client_w: int = 1920,
        client_h: int = 1440,
        need_rect: bool = True,
        max_wall_s: float | None = None,
    ) -> HudMainIconsFrame | None:
        t0 = time.perf_counter()
        if not self._ensure():
            return None
        assert self._pid and self._h and self._ga
        # lean/状态栏：硬墙钟，禁止再扫到几十秒
        if max_wall_s is None:
            max_wall_s = 2.5 if not need_rect else 15.0
        wall = max(0.4, float(max_wall_s))
        frame = HudMainIconsFrame(
            pid=self._pid,
            gameassembly=self._ga,
            client_w=client_w,
            client_h=client_h,
            age_ms=0.0,
        )
        offs = self._btn_cfg["offsets"]

        for kind, spec in SPECS.items():
            if (time.perf_counter() - t0) >= wall:
                frame.errors.append("wall_timeout")
                break
            rva = self._priority.get(kind)
            if not rva:
                frame.errors.append(f"{kind}:no_rva")
                continue
            klass = resolve_klass(self._h, self._ga, int(rva))
            if not klass:
                frame.errors.append(f"{kind}:no_klass")
                continue
            remain = wall - (time.perf_counter() - t0)
            if remain <= 0.05:
                frame.errors.append("wall_timeout")
                break
            addrs = scan_klass_objects(
                self._h,
                klass,
                max_hits=self._max_per_kind,
                max_wall_s=remain,
                should_stop=lambda: (time.perf_counter() - t0) >= wall,
            )
            for addr in addrs:
                snap = HudMainIconSnap(
                    kind=kind,
                    address=addr,
                    icon_hint=str(spec.get("icon") or ""),
                )
                if "mod_vis" in spec:
                    snap.modular_visible = _boolish(read_u8(self._h, addr + int(spec["mod_vis"])))
                if "is_visible" in spec:
                    snap.is_visible = _boolish(read_u8(self._h, addr + int(spec["is_visible"])))
                if "button_state" in spec:
                    raw = read_u32(self._h, addr + int(spec["button_state"]))
                    if raw is not None and raw <= 3:
                        snap.button_state_raw = raw
                        snap.button_state = BUTTON_STATE.get(raw, str(raw))
                    elif raw is not None:
                        snap.notes.append(f"button_state_odd={raw}")

                ui = read_ptr(self._h, addr + int(spec["ui"]))
                if not _ptr_ok(ui):
                    snap.notes.append("ui_ptr_bad")
                    frame.icons.append(snap)
                    continue

                button: int | None = None
                if "to_button" in spec:
                    button = _follow(self._h, ui, tuple(spec["to_button"]))
                elif "to_mhb" in spec:
                    mhb = _follow(self._h, ui, tuple(spec["to_mhb"]))
                    if mhb:
                        st = read_u32(self._h, mhb + OFF_MHB_STATE)
                        if st is not None and st <= 3:
                            snap.mhb_state = BUTTON_STATE.get(st, str(st))
                        button = read_ptr(self._h, mhb + OFF_MHB_BUTTON)
                    else:
                        snap.notes.append("mhb_ptr_bad")

                if not _ptr_ok(button):
                    snap.notes.append("button_ptr_bad")
                    frame.icons.append(snap)
                    continue
                snap.button_addr = button
                snap.interactable = _boolish(
                    read_u8(self._h, button + offs.get("ButtonControl.interactable", 0x84))
                )
                snap.enabled = _boolish(
                    read_u8(self._h, button + offs.get("ButtonControl.enabled", 0x85))
                )
                if need_rect:
                    rect, src, gnotes = self._rect_from_button(button, client_w, client_h)
                    snap.notes.extend(gnotes)
                    snap.rect = rect
                    snap.rect_source = src
                frame.icons.append(snap)
                # lean（无 rect）：找到 Show+可点 才早停
                if not need_rect and _snap_early_stop(snap):
                    break

        # 丢弃明显幽灵：无可见标志、无状态、无 button
        good = [
            i
            for i in frame.icons
            if i.modular_visible
            or i.is_visible
            or (i.button_state in ("Show", "Hide", "Combat") and i.button_addr)
            or (i.mhb_state in ("Show", "Hide", "Combat") and i.button_addr)
            or i.rect is not None
        ]
        # 有活钮用活钮；全幽灵时保留原列表供 summarize 再筛，避免 icons=[] 误报「无icon」
        if good:
            frame.icons = good

        frame.age_ms = (time.perf_counter() - t0) * 1000.0
        return frame
