"""Windows 游戏客户区可拖动获取框。

覆盖层只显示、可拖动，不向游戏发送输入。
"""

from __future__ import annotations

import hashlib
import sys
import tkinter as tk
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import ImageGrab

if sys.platform == "win32":
    import win32con
    import win32gui
else:  # pragma: no cover
    win32con = None  # type: ignore[assignment]
    win32gui = None  # type: ignore[assignment]

_CHROMA = "#010203"
_CHROMA_COLORREF = 0x030201
_BORDER = 4
_GRIP = 14
_MIN_BOX = 24


def screen_rect_to_client(
    screen_rect: tuple[int, int, int, int] | list[int],
    *,
    origin: tuple[int, int],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """屏幕获取框 → 游戏客户区像素矩形，并夹在画面内。"""
    sx, sy, sw, sh = (int(v) for v in screen_rect)
    ox, oy = int(origin[0]), int(origin[1])
    x = sx - ox
    y = sy - oy
    w = max(_MIN_BOX, sw)
    h = max(_MIN_BOX, sh)
    fw = max(1, int(frame_w))
    fh = max(1, int(frame_h))
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(_MIN_BOX, min(w, fw - x))
    h = max(_MIN_BOX, min(h, fh - y))
    return x, y, w, h


def client_rect_to_screen(
    client_rect: tuple[int, int, int, int] | list[int],
    *,
    origin: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in client_rect)
    ox, oy = int(origin[0]), int(origin[1])
    return ox + x, oy + y, max(_MIN_BOX, w), max(_MIN_BOX, h)


def client_rect_to_norm(
    client_rect: tuple[int, int, int, int] | list[int],
    *,
    frame_w: int,
    frame_h: int,
) -> list[float]:
    x, y, w, h = (int(v) for v in client_rect)
    fw = max(1, int(frame_w))
    fh = max(1, int(frame_h))
    return [x / fw, y / fh, w / fw, h / fh]


def crop_client_pattern(
    frame: np.ndarray,
    client_rect: tuple[int, int, int, int] | list[int],
) -> np.ndarray | None:
    if frame is None or getattr(frame, "size", 0) == 0 or frame.ndim != 3:
        return None
    fh, fw = frame.shape[:2]
    x, y, w, h = (int(v) for v in client_rect)
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))
    patch = frame[y : y + h, x : x + w]
    if patch.size == 0:
        return None
    return patch


def red_marker_center_ok(
    image_bgr: np.ndarray,
    *,
    client_xy: tuple[int, int],
    radius: int = 10,
) -> bool:
    """桌面合成截图中，预期客户区点附近是否确实出现红点。"""
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return False
    h, w = image_bgr.shape[:2]
    cx, cy = int(client_xy[0]), int(client_xy[1])
    r0 = max(2, int(radius))
    x0, x1 = max(0, cx - r0), min(w, cx + r0 + 1)
    y0, y1 = max(0, cy - r0), min(h, cy + r0 + 1)
    patch = image_bgr[y0:y1, x0:x1]
    if patch.size == 0 or patch.ndim != 3 or patch.shape[2] < 3:
        return False
    b, g, r = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    red = (r > 180) & (r.astype(np.int16) > g.astype(np.int16) + 70) & (
        r.astype(np.int16) > b.astype(np.int16) + 70
    )
    return float(np.mean(red)) >= 0.03


def red_marker_delta_ok(
    raw_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    *,
    client_xy: tuple[int, int],
    radius: int = 12,
) -> bool:
    """红点证据须相对无覆盖层原图新增红色，不能复用游戏自身红色。"""
    if (
        raw_bgr is None
        or overlay_bgr is None
        or raw_bgr.shape != overlay_bgr.shape
        or raw_bgr.ndim != 3
    ):
        return False
    h, w = raw_bgr.shape[:2]
    cx, cy = int(client_xy[0]), int(client_xy[1])
    rr = max(3, int(radius))
    x0, x1 = max(0, cx - rr), min(w, cx + rr + 1)
    y0, y1 = max(0, cy - rr), min(h, cy + rr + 1)
    before = raw_bgr[y0:y1, x0:x1].astype(np.int16)
    after = overlay_bgr[y0:y1, x0:x1].astype(np.int16)
    if before.size == 0:
        return False
    new_red = (
        (after[:, :, 2] > 180)
        & (after[:, :, 2] > after[:, :, 1] + 70)
        & (after[:, :, 2] > after[:, :, 0] + 70)
        & (after[:, :, 2] > before[:, :, 2] + 25)
    )
    return float(np.mean(new_red)) >= 0.03


def capture_overlay_evidence(
    *,
    client_box: tuple[int, int, int, int],
    click_client: tuple[int, int],
    destination: Path,
    baseline_bgr: np.ndarray | None = None,
) -> dict[str, Any]:
    """抓桌面合成客户区，证明 layered 红点真实显示在预期位置。"""
    left, top, width, height = (int(v) for v in client_box)
    if width <= 0 or height <= 0:
        return {"ok": False, "error": "invalid_client_box"}
    bbox = (left, top, left + width, top + height)
    try:
        image = ImageGrab.grab(
            bbox=bbox,
            include_layered_windows=True,
            all_screens=True,
        )
    except TypeError:  # 旧 Pillow 不支持 include_layered_windows
        image = ImageGrab.grab(bbox=bbox, all_screens=True)
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    marker_ok = (
        red_marker_delta_ok(baseline_bgr, bgr, client_xy=click_client)
        if baseline_bgr is not None
        else red_marker_center_ok(bgr, client_xy=click_client)
    )
    encoded_ok, encoded = cv2.imencode(".png", bgr)
    if not encoded_ok:
        return {"ok": False, "error": "png_encode_failed"}
    data = encoded.tobytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(destination)
    return {
        "ok": marker_ok,
        "path": str(destination),
        "sha256": hashlib.sha256(data).hexdigest(),
        "click_client": [int(click_client[0]), int(click_client[1])],
        "client_box": [left, top, width, height],
        "reason": "red_marker_present" if marker_ok else "red_marker_missing",
    }


def overlay_marker_from_job(job: dict[str, Any]) -> dict[str, Any]:
    """由完成的 hud-entry 源帧决定是否显示红点。"""
    args = job.get("args") if isinstance(job.get("args"), dict) else {}
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    locator = result.get("locator") if isinstance(result.get("locator"), dict) else {}
    if (
        str(job.get("status") or "") != "ok"
        or str(job.get("action") or "") != "readonly_ui_fine"
        or str(job.get("method") or "").upper() != "B"
        or str(args.get("pair_role") or "") != "source"
        or not str(args.get("entry_id") or "")
        or locator.get("position_ok") is not True
    ):
        return {"visible": False}
    click = locator.get("click") if isinstance(locator.get("click"), dict) else {}
    screen = click.get("screen")
    if (
        not isinstance(screen, list)
        or len(screen) != 2
        or not all(isinstance(v, (int, float)) for v in screen)
    ):
        return {"visible": False}
    provisional = bool(locator.get("provisional") is True)
    feature_ok = bool(locator.get("feature_ok") is True)
    if not feature_ok and not provisional:
        return {"visible": False}
    client_rect = locator.get("feature_rect_client")
    if (
        not isinstance(client_rect, list)
        or len(client_rect) != 4
        or not all(isinstance(v, (int, float)) for v in client_rect)
    ):
        client_xy = click.get("client")
        if isinstance(client_xy, list) and len(client_xy) == 2:
            cx, cy = int(client_xy[0]), int(client_xy[1])
            client_rect = [cx - 24, cy - 24, 48, 48]
        else:
            client_rect = [0, 0, 48, 48]
    client_xy = click.get("client")
    origin = (
        (int(screen[0]) - int(client_xy[0]), int(screen[1]) - int(client_xy[1]))
        if isinstance(client_xy, list) and len(client_xy) == 2
        else (0, 0)
    )
    sx, sy, sw, sh = client_rect_to_screen(client_rect, origin=origin)
    return {
        "visible": True,
        "screen": [int(screen[0]), int(screen[1])],
        "screen_rect": [int(sx), int(sy), int(sw), int(sh)],
        "client_rect": [int(v) for v in client_rect],
        "entry_id": str(args.get("entry_id")),
        "capture_id": str(args.get("capture_id") or args.get("pair_id") or ""),
        "accepted": bool(locator.get("accepted") is True),
        "provisional": provisional,
    }


class HintOverlay:
    """可拖动、可缩放的获取框；不向游戏发送输入。"""

    def __init__(self, parent: tk.Misc, *, size: int = 48) -> None:
        self._w = max(_MIN_BOX, int(size))
        self._h = max(_MIN_BOX, int(size))
        self._shown = False
        self._drag: tuple[str, int, int, int, int] | None = None
        self.on_move = None
        self._window = tk.Toplevel(parent)
        self._window.withdraw()
        self._window.overrideredirect(True)
        try:
            self._window.attributes("-topmost", True)
        except tk.TclError:
            pass
        try:
            self._window.attributes("-transparentcolor", _CHROMA)
        except tk.TclError:
            pass
        self._canvas = tk.Canvas(
            self._window,
            width=self._w,
            height=self._h,
            background=_CHROMA,
            highlightthickness=0,
            borderwidth=0,
            cursor="fleur",
        )
        self._canvas.pack(fill="both", expand=True)
        self._redraw()
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._window.update_idletasks()
        self._apply_window_style()

    def _redraw(self) -> None:
        c = self._canvas
        c.delete("all")
        c.config(width=self._w, height=self._h)
        c.create_rectangle(
            1,
            1,
            self._w - 2,
            self._h - 2,
            outline="#ff2020",
            width=_BORDER,
            fill=_CHROMA,
        )
        c.create_rectangle(
            self._w - _GRIP,
            self._h - _GRIP,
            self._w - 2,
            self._h - 2,
            outline="#ffffff",
            fill="#ff2020",
            width=1,
        )

    @property
    def hwnd(self) -> int:
        raw = int(self._window.winfo_id())
        if sys.platform == "win32" and win32gui is not None and win32con is not None:
            root = int(win32gui.GetAncestor(raw, win32con.GA_ROOT) or 0)
            if root:
                return root
        return raw

    def _apply_window_style(self) -> None:
        if sys.platform != "win32" or win32gui is None or win32con is None:
            return
        hwnd = self.hwnd
        exstyle = int(win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE))
        exstyle |= (
            win32con.WS_EX_LAYERED
            | win32con.WS_EX_TOOLWINDOW
            | getattr(win32con, "WS_EX_NOACTIVATE", 0x08000000)
        )
        exstyle &= ~int(getattr(win32con, "WS_EX_TRANSPARENT", 0x00000020))
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, exstyle)
        win32gui.SetLayeredWindowAttributes(
            hwnd,
            _CHROMA_COLORREF,
            0,
            win32con.LWA_COLORKEY,
        )

    def screen_rect(self) -> tuple[int, int, int, int]:
        self._window.update_idletasks()
        return (
            int(self._window.winfo_x()),
            int(self._window.winfo_y()),
            int(self._window.winfo_width() or self._w),
            int(self._window.winfo_height() or self._h),
        )

    def show(self, screen_x: int, screen_y: int) -> None:
        self.show_rect(
            int(screen_x) - self._w // 2,
            int(screen_y) - self._h // 2,
            self._w,
            self._h,
        )

    def show_rect(self, screen_x: int, screen_y: int, width: int, height: int) -> None:
        self._w = max(_MIN_BOX, int(width))
        self._h = max(_MIN_BOX, int(height))
        x, y = int(screen_x), int(screen_y)
        self._shown = True
        self._redraw()
        self._window.geometry(f"{self._w}x{self._h}+{x}+{y}")
        self._window.deiconify()
        try:
            self._window.attributes("-topmost", True)
        except tk.TclError:
            pass
        self._window.lift()
        self._window.update_idletasks()
        self._window.update()
        self._apply_window_style()
        if sys.platform == "win32" and win32gui is not None and win32con is not None:
            flags = win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW
            win32gui.SetWindowPos(
                self.hwnd,
                win32con.HWND_TOPMOST,
                x,
                y,
                self._w,
                self._h,
                flags,
            )
            win32gui.ShowWindow(
                self.hwnd,
                getattr(win32con, "SW_SHOWNOACTIVATE", 4),
            )
        self._notify_move()
        self._window.after(400, self._pulse_topmost)

    def _notify_move(self) -> None:
        cb = self.on_move
        if cb is None:
            return
        try:
            cb(self.screen_rect())
        except Exception:
            return

    def _on_press(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.x >= self._w - _GRIP and event.y >= self._h - _GRIP:
            self._drag = ("resize", event.x_root, event.y_root, self._w, self._h)
        else:
            geo = self.screen_rect()
            self._drag = ("move", event.x_root, event.y_root, geo[0], geo[1])

    def _on_drag(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._drag is None:
            return
        kind, x0, y0, a, b = self._drag
        dx = int(event.x_root) - x0
        dy = int(event.y_root) - y0
        if kind == "move":
            self._window.geometry(f"{self._w}x{self._h}+{a + dx}+{b + dy}")
        else:
            self._w = max(_MIN_BOX, a + dx)
            self._h = max(_MIN_BOX, b + dy)
            geo = self.screen_rect()
            self._redraw()
            self._window.geometry(f"{self._w}x{self._h}+{geo[0]}+{geo[1]}")
        self._notify_move()

    def _on_release(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._drag = None
        self._notify_move()

    def _pulse_topmost(self) -> None:
        if not self._shown:
            return
        try:
            self._window.attributes("-topmost", True)
            self._window.lift()
        except tk.TclError:
            return
        if sys.platform == "win32" and win32gui is not None and win32con is not None:
            try:
                win32gui.SetWindowPos(
                    self.hwnd,
                    win32con.HWND_TOPMOST,
                    0,
                    0,
                    0,
                    0,
                    win32con.SWP_NOMOVE
                    | win32con.SWP_NOSIZE
                    | win32con.SWP_NOACTIVATE
                    | win32con.SWP_SHOWWINDOW,
                )
            except Exception:
                pass
        try:
            self._window.after(400, self._pulse_topmost)
        except tk.TclError:
            return

    def hide(self) -> None:
        self._shown = False
        self._window.withdraw()

    def destroy(self) -> None:
        self._shown = False
        try:
            self._window.destroy()
        except tk.TclError:
            pass


class ClickMarkOverlay:
    """客户区红圈：只显示、点穿透，不向游戏发输入。"""

    def __init__(self, parent: tk.Misc, *, radius: int = 28) -> None:
        self._r = max(12, int(radius))
        self._shown = False
        self._window = tk.Toplevel(parent)
        self._window.withdraw()
        self._window.overrideredirect(True)
        try:
            self._window.attributes("-topmost", True)
        except tk.TclError:
            pass
        try:
            self._window.attributes("-transparentcolor", _CHROMA)
        except tk.TclError:
            pass
        side = self._r * 2 + 4
        self._canvas = tk.Canvas(
            self._window,
            width=side,
            height=side,
            background=_CHROMA,
            highlightthickness=0,
            borderwidth=0,
        )
        self._canvas.pack()
        pad = 2
        self._canvas.create_oval(
            pad,
            pad,
            side - pad,
            side - pad,
            outline="#ff2020",
            width=3,
            fill=_CHROMA,
        )
        self._window.update_idletasks()
        self._apply_style()
        self._last_pos: tuple[int, int] | None = None
        self._multi = False

    @property
    def hwnd(self) -> int:
        raw = int(self._window.winfo_id())
        if sys.platform == "win32" and win32gui is not None and win32con is not None:
            root = int(win32gui.GetAncestor(raw, win32con.GA_ROOT) or 0)
            if root:
                return root
        return raw

    def _apply_style(self) -> None:
        if sys.platform != "win32" or win32gui is None or win32con is None:
            return
        hwnd = self.hwnd
        exstyle = int(win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE))
        exstyle |= (
            win32con.WS_EX_LAYERED
            | win32con.WS_EX_TOOLWINDOW
            | getattr(win32con, "WS_EX_NOACTIVATE", 0x08000000)
            | getattr(win32con, "WS_EX_TRANSPARENT", 0x00000020)
        )
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, exstyle)
        win32gui.SetLayeredWindowAttributes(
            hwnd,
            _CHROMA_COLORREF,
            0,
            win32con.LWA_COLORKEY,
        )

    def show_client(self, origin: tuple[int, int], cx: int, cy: int) -> None:
        if self._multi:
            self._reset_single_canvas()
        side = self._r * 2 + 4
        sx = int(origin[0]) + int(cx) - self._r - 2
        sy = int(origin[1]) + int(cy) - self._r - 2
        if self._shown and self._last_pos == (sx, sy) and not self._multi:
            return
        self._last_pos = (sx, sy)
        self._shown = True
        self._window.geometry(f"{side}x{side}+{sx}+{sy}")
        self._window.deiconify()
        if sys.platform == "win32" and win32gui is not None and win32con is not None:
            win32gui.SetWindowPos(
                self.hwnd,
                win32con.HWND_TOPMOST,
                sx,
                sy,
                side,
                side,
                win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW,
            )

    def show_points(
        self,
        origin: tuple[int, int],
        size: tuple[int, int],
        points: list[tuple[int, int]],
    ) -> None:
        if not points:
            self.hide()
            return
        w = max(64, int(size[0]))
        h = max(64, int(size[1]))
        sx, sy = int(origin[0]), int(origin[1])
        self._multi = True
        self._last_pos = None
        self._shown = True
        self._canvas.delete("all")
        self._canvas.config(width=w, height=h, background=_CHROMA)
        r = self._r
        for cx, cy in points:
            x, y = int(cx), int(cy)
            self._canvas.create_oval(
                x - r,
                y - r,
                x + r,
                y + r,
                outline="#ff2020",
                width=3,
                fill=_CHROMA,
            )
        self._window.geometry(f"{w}x{h}+{sx}+{sy}")
        self._window.deiconify()
        self._apply_style()
        if sys.platform == "win32" and win32gui is not None and win32con is not None:
            win32gui.SetWindowPos(
                self.hwnd,
                win32con.HWND_TOPMOST,
                sx,
                sy,
                w,
                h,
                win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW,
            )

    def _reset_single_canvas(self) -> None:
        self._multi = False
        side = self._r * 2 + 4
        self._canvas.delete("all")
        self._canvas.config(width=side, height=side, background=_CHROMA)
        pad = 2
        self._canvas.create_oval(
            pad,
            pad,
            side - pad,
            side - pad,
            outline="#ff2020",
            width=3,
            fill=_CHROMA,
        )

    def hide(self) -> None:
        self._shown = False
        self._last_pos = None
        if self._multi:
            self._reset_single_canvas()
        self._window.withdraw()

    def destroy(self) -> None:
        self._shown = False
        try:
            self._window.destroy()
        except tk.TclError:
            pass

