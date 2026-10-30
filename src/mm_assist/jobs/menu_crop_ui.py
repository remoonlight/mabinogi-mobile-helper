"""系统菜单四键：游戏客户区 HintOverlay 获取框。方法 B，不向游戏发键鼠。"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence
from typing import Any

from mm_assist.config import AppConfig
from mm_assist.jobs.menu_buttons import MENU_BUTTONS, MenuButtonSpec, save_menu_button_crop
from mm_assist.platform.grab import prepare_grabber, refresh_and_grab
from mm_assist.platform.hint_overlay import HintOverlay, screen_rect_to_client
from mm_assist.platform.window import enable_dpi_aware, find_window, refresh_window


def run_overlay_session(
    cfg: AppConfig,
    buttons: Sequence[MenuButtonSpec] = MENU_BUTTONS,
) -> list[dict[str, Any]]:
    enable_dpi_aware()
    win = find_window(cfg.window.title_substr)
    if not win:
        raise RuntimeError("game_window_missing")
    prepare_grabber(cfg)
    items = list(buttons)
    if not items:
        return []
    saved: list[dict[str, Any]] = []
    idx = 0

    root = tk.Tk()
    root.title("采界面键")
    root.attributes("-topmost", True)
    hint = tk.StringVar()
    box_var = tk.StringVar(value="获取框：—")
    overlay = HintOverlay(root, size=72)

    def current_win():
        return refresh_window(win)

    def place_box() -> None:
        w = current_win()
        overlay.show_rect(
            w.left + max(40, w.width // 3),
            w.top + max(40, w.height // 4),
            80,
            80,
        )

    def refresh_hint() -> None:
        spec = items[idx]
        hint.set(
            f"{idx + 1}/{len(items)}  {spec.need_ui}\n把红框套住「{spec.label}」，再点确认"
        )
        root.title(f"采界面键 · {spec.label}")

    def on_move(rect: tuple[int, int, int, int]) -> None:
        w = current_win()
        cr = screen_rect_to_client(
            rect,
            origin=(w.left, w.top),
            frame_w=w.width,
            frame_h=w.height,
        )
        box_var.set(f"获取框：客户区 x={cr[0]} y={cr[1]} w={cr[2]} h={cr[3]}")

    overlay.on_move = on_move

    def confirm(_ev: object | None = None) -> None:
        nonlocal idx
        spec = items[idx]
        screen = overlay.screen_rect()
        overlay.hide()
        root.update_idletasks()
        w, img = refresh_and_grab(cfg, current_win())
        cr = screen_rect_to_client(
            screen,
            origin=(w.left, w.top),
            frame_w=int(img.shape[1]),
            frame_h=int(img.shape[0]),
        )
        try:
            row = save_menu_button_crop(img, spec, cr)
        except Exception as exc:
            hint.set(f"保存失败：{exc}\n把红框套住「{spec.label}」后重试")
            place_box()
            return
        saved.append(row)
        idx += 1
        if idx >= len(items):
            root.destroy()
            return
        refresh_hint()
        place_box()

    def skip(_ev: object | None = None) -> None:
        nonlocal idx
        idx += 1
        if idx >= len(items):
            root.destroy()
            return
        refresh_hint()
        place_box()

    tk.Label(root, textvariable=hint, font=("Segoe UI", 12), justify="left", wraplength=420).pack(
        anchor="w", padx=8, pady=6
    )
    tk.Label(root, textvariable=box_var, font=("Segoe UI", 11)).pack(anchor="w", padx=8)
    row = tk.Frame(root)
    row.pack(fill="x", padx=8, pady=8)
    tk.Button(row, text="确认保存", command=confirm, width=12).pack(side="left", padx=4)
    tk.Button(row, text="跳过", command=skip, width=8).pack(side="left", padx=4)
    tk.Button(row, text="结束", command=root.destroy, width=8).pack(side="left", padx=4)
    root.bind("s", confirm)
    root.bind("S", confirm)
    root.bind("n", skip)
    root.bind("<Escape>", lambda _e: root.destroy())
    refresh_hint()
    place_box()
    root.mainloop()
    overlay.destroy()
    return saved
