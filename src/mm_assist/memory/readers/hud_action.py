"""IL2CPP 只读：HudAction 行动钮可见态 + ButtonControl 本地矩形。

链：
  HudActionUIComponent.IsVisible@0x42
  BasicHudActionButtonControl.baseControl@0x18 → BaseHudActionButtonControl
    .IsVisible@0x28 · .Button@0x38 → ButtonControl → Graphic → RectTransform local@+0xB8
  BasicHudActionButtonControl.ControlKeyId@0x40（ControlKeysSheetId.Id）

配置复用：
  knowledge/ui/config/il2cpp_ui_priority.json（TypeInfo）
  knowledge/ui/config/il2cpp_button_hits.json（ButtonControl 偏移）

探针：`python -m mm_assist.tools.il2cpp_hud_action_probe`
"""

from __future__ import annotations

import json
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

# dump.cs 偏移（HudAction 族）
OFF_HA_UI_IS_VISIBLE = 0x42
OFF_HA_UI_IS_TWEENING = 0x43
OFF_HA_UI_NEED_UPDATE = 0x44
OFF_HA_UI_COOLDOWN = 0x70  # float; 2026-11-04 field schema

OFF_BASIC_BASE_CONTROL = 0x18
OFF_BASIC_ICON_CONTROL = 0x20
OFF_BASIC_CONTROL_KEY_ID = 0x40  # ControlKeysSheetId.Id; follows RootObject@0x38

OFF_BASE_IS_VISIBLE = 0x28
OFF_BASE_BUTTON = 0x38

# 与 Basic 同布局的其它行动钮（2026-11-04：base@0x18 · key@0x40）
BUTTON_CONTROL_KINDS = (
    "BasicHudActionButtonControl",
    "CombatHudActionButtonControl",
)


def _boolish(v: int | None) -> bool | None:
    if v is None:
        return None
    if v in (0, 1):
        return bool(v)
    return None


def _ptr_ok(p: int | None) -> bool:
    # 本机 64-bit 托管对象多在 >4GB；过滤 0x20120000 一类假指针
    return bool(p and p >= 0x1_0000_0000)


@dataclass
class HudActionButtonSnap:
    address: int
    kind: str = "BasicHudActionButtonControl"
    control_key_id: int | None = None
    base_visible: bool | None = None
    button_addr: int | None = None
    interactable: bool | None = None
    enabled: bool | None = None
    rect: HitRect | None = None
    rect_source: str | None = None
    notes: list[str] = field(default_factory=list)
    plausible: bool = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "address": hex(self.address),
            "control_key_id": self.control_key_id,
            "base_visible": self.base_visible,
            "button": hex(self.button_addr) if self.button_addr else None,
            "interactable": self.interactable,
            "enabled": self.enabled,
            "rect_source": self.rect_source,
            "plausible": self.plausible,
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
class HudActionFrame:
    pid: int
    gameassembly: int
    client_w: int
    client_h: int
    age_ms: float
    ui_components: list[dict[str, Any]] = field(default_factory=list)
    buttons: list[HudActionButtonSnap] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "gameassembly": hex(self.gameassembly),
            "client_w": self.client_w,
            "client_h": self.client_h,
            "age_ms": round(self.age_ms, 1),
            "ui_components": self.ui_components,
            "buttons": [b.to_dict() for b in self.buttons],
            "button_count": len(self.buttons),
            "with_rect": sum(1 for b in self.buttons if b.rect is not None),
            "errors": self.errors,
        }


class Il2CppHudActionReader:
    def __init__(
        self,
        *,
        priority_path: Path | None = None,
        button_hits_path: Path | None = None,
        max_ui: int = 24,
        max_buttons: int = 128,
    ) -> None:
        self._priority = load_priority(priority_path or default_priority_path())
        self._btn_cfg = load_button_hits_config(
            button_hits_path or button_hits_config_path()
        )
        self._max_ui = max_ui
        self._max_buttons = max_buttons
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

    def _klass(self, name: str):
        assert self._h and self._ga
        rva = self._priority.get(name)
        if not rva:
            # 允许从 button 配置以外的硬编码回退（priority 已含）
            return None, f"{name}:no_rva"
        klass = resolve_klass(self._h, self._ga, int(rva))
        if not klass:
            return None, f"{name}:no_klass"
        return klass, None

    def capture(
        self,
        *,
        client_w: int = 1920,
        client_h: int = 1440,
        need_rect: bool = True,
        max_wall_s: float | None = None,
        stop_on_strong: bool = False,
    ) -> HudActionFrame | None:
        """读一帧 HudAction。

        need_rect=False：只读可见/可点，跳过 Graphic→rect（状态栏 lean 用，显著加速）。
        stop_on_strong：已找到强停止候选（可见层或 vis+可点/rect）即停，供采证早停。
        max_wall_s：墙钟上限，超时带回已扫到的部分帧。
        """
        t0 = time.perf_counter()
        if not self._ensure():
            return None
        assert self._pid and self._h and self._ga
        frame = HudActionFrame(
            pid=self._pid,
            gameassembly=self._ga,
            client_w=client_w,
            client_h=client_h,
            age_ms=0.0,
        )
        offs = self._btn_cfg["offsets"]
        native_offs = [
            parse_int(x) if not isinstance(x, int) else x
            for x in (self._btn_cfg.get("native_rect_probe_offsets") or [0xB8])
        ]
        if 0xB8 not in native_offs:
            native_offs = [0xB8, *native_offs]

        def _remain_wall() -> float | None:
            if max_wall_s is None:
                return None
            left = float(max_wall_s) - (time.perf_counter() - t0)
            return max(0.05, left)

        def _over_wall() -> bool:
            if max_wall_s is None:
                return False
            return (time.perf_counter() - t0) >= float(max_wall_s)

        strong = False

        # --- HudActionUIComponent ---
        klass, err = self._klass("HudActionUIComponent")
        if err:
            frame.errors.append(err)
        elif klass:
            for addr in scan_klass_objects(
                self._h,
                klass,
                max_hits=self._max_ui,
                max_wall_s=_remain_wall(),
                should_stop=_over_wall,
            ):
                if _over_wall():
                    frame.errors.append("wall_timeout_ui")
                    break
                vis = _boolish(read_u8(self._h, addr + OFF_HA_UI_IS_VISIBLE))
                tw = _boolish(read_u8(self._h, addr + OFF_HA_UI_IS_TWEENING))
                need = _boolish(read_u8(self._h, addr + OFF_HA_UI_NEED_UPDATE))
                if vis is None and tw is None and need is None:
                    continue  # 幽灵
                cool = None
                try:
                    from mm_assist.memory.process import read_floats

                    fs = read_floats(self._h, addr + OFF_HA_UI_COOLDOWN, 1)
                    if fs and abs(float(fs[0])) < 1e6:
                        cool = float(fs[0])
                except Exception:
                    cool = None
                frame.ui_components.append(
                    {
                        "address": hex(addr),
                        "IsVisible": vis,
                        "IsTweening": tw,
                        "IsNeedUpdate": need,
                        "cooldownRemaining": cool,
                    }
                )
                # 层可见不作 early-stop 条件（易假阳）；仅记组件

        # --- Basic / Combat HudActionButtonControl → Button → rect ---
        for kind in BUTTON_CONTROL_KINDS:
            if _over_wall():
                frame.errors.append("wall_timeout_buttons")
                break
            if stop_on_strong and strong:
                break
            klass, err = self._klass(kind)
            if err:
                frame.errors.append(err)
                continue
            if not klass:
                continue
            for addr in scan_klass_objects(
                self._h,
                klass,
                max_hits=self._max_buttons,
                max_wall_s=_remain_wall(),
                should_stop=_over_wall,
            ):
                if _over_wall():
                    frame.errors.append("wall_timeout_buttons")
                    break
                snap = HudActionButtonSnap(address=addr, kind=kind)
                key = read_u32(self._h, addr + OFF_BASIC_CONTROL_KEY_ID)
                # ControlKeysSheetId.Id 通常为小正整数；过大当幽灵
                if key is not None and 0 <= int(key) < 1_000_000:
                    snap.control_key_id = int(key)
                else:
                    snap.control_key_id = None
                    snap.notes.append(f"key_raw={key}")
                base = read_ptr(self._h, addr + OFF_BASIC_BASE_CONTROL)
                if not _ptr_ok(base):
                    snap.notes.append("no_baseControl")
                    snap.plausible = False
                    frame.buttons.append(snap)
                    continue
                snap.base_visible = _boolish(read_u8(self._h, base + OFF_BASE_IS_VISIBLE))
                btn = read_ptr(self._h, base + OFF_BASE_BUTTON)
                snap.button_addr = btn if _ptr_ok(btn) else None
                if not snap.button_addr:
                    snap.notes.append("no_Button")
                    snap.plausible = bool(snap.base_visible is not None)
                    frame.buttons.append(snap)
                    continue
                snap.interactable = _boolish(
                    read_u8(self._h, btn + offs["ButtonControl.interactable"])
                )
                snap.enabled = _boolish(read_u8(self._h, btn + offs["ButtonControl.enabled"]))
                if need_rect:
                    graphic, gnotes = _resolve_graphic_from_button(self._h, btn, offs)
                    snap.notes.extend(gnotes)
                    if graphic and _ptr_ok(graphic):
                        rect, src, _rt, _nat = _rect_from_graphic(
                            self._h,
                            graphic,
                            offs,
                            cw=float(client_w),
                            ch=float(client_h),
                            native_offsets=native_offs,
                        )
                        snap.rect = rect
                        snap.rect_source = src
                snap.plausible = (
                    snap.base_visible is not None
                    or snap.rect is not None
                    or snap.enabled is not None
                )
                frame.buttons.append(snap)
                # 强候选：可见且（可点或有尺寸）——与 summarize 幽灵过滤对齐
                if snap.base_visible is True and (
                    snap.interactable is True or snap.rect is not None
                ):
                    strong = True
                    if stop_on_strong:
                        break

        # 优先保留 plausible
        good = [b for b in frame.buttons if b.plausible]
        if good:
            frame.buttons = good

        frame.age_ms = (time.perf_counter() - t0) * 1000
        return frame


__all__ = [
    "HudActionButtonSnap",
    "HudActionFrame",
    "Il2CppHudActionReader",
]
