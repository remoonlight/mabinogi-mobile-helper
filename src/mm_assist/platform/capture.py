from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mss import mss

from mm_assist.platform.window import WindowInfo, restore_no_activate

log = logging.getLogger(__name__)

_PW_RENDERFULLCONTENT = 0x00000002
_BI_RGB = 0
_DIB_RGB_COLORS = 0
_SRCCOPY = 0x00CC0020


def _client_is_foreground(win: WindowInfo) -> bool:
    from mm_assist.platform.window import is_foreground
    return is_foreground(int(win.hwnd))


class Grabber:
    """
    截屏：
    - auto：前后台均只用 PrintWindow 渲染图，失败／黑屏保持未知
    - bitblt / printwindow / mss：强制单一后端
    """

    def __init__(self, method: str = "auto") -> None:
        self.method = (method or "auto").lower()
        self._sct = mss()
        self._last_source: str | None = None

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass

    def grab_rect(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        mon = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
        shot = self._sct.grab(mon)
        bgra = np.asarray(shot, dtype=np.uint8)
        return np.ascontiguousarray(bgra[:, :, :3])

    def grab_client(self, win: WindowInfo) -> np.ndarray:
        method = self.method
        if method == "mss":
            img = self.grab_rect(win.left, win.top, win.width, win.height)
            self._note("mss")
            return img
        if method == "printwindow":
            img = self._printwindow_client(win)
            if img is None:
                raise RuntimeError("PrintWindow 失败（窗口可能最小化或被保护）")
            self._note("printwindow")
            return img
        if method == "bitblt":
            img = self._bitblt_client(win)
            if img is None or _looks_black(img):
                raise RuntimeError("BitBlt 失败或黑屏")
            self._note("bitblt")
            return img
        # A nonblack client DC can retain a previous scene even with this HWND
        # foreground. Current evidence pairs old dungeon-result BitBlt pixels
        # with the independently rendered wood facility from the same window.
        # Do not use that surface to establish page identity or absence of UI.
        return self._rendered_client(win)

    def _rendered_client(self, win: WindowInfo) -> np.ndarray:
        image = self._printwindow_client(win)
        if image is None or _looks_black(image):
            raise RuntimeError('PrintWindow 失败或黑屏，不能使用可能过期的客户区表面')
        self._note('printwindow')
        return image

    def grab_client_roi(
        self,
        win: WindowInfo,
        roi: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, int, int]:
        x, y, rw, rh = roi
        ox = max(0, int(x * win.width))
        oy = max(0, int(y * win.height))
        w = max(1, min(int(rw * win.width), win.width - ox))
        h = max(1, min(int(rh * win.height), win.height - oy))
        patch = self.grab_client_norm_box(
            win,
            (
                ox / float(win.width),
                oy / float(win.height),
                (ox + w) / float(win.width),
                (oy + h) / float(win.height),
            ),
            prefer_mss=False,
            fallback_full=True,
        )
        if patch is None:
            raise RuntimeError("客户区 ROI 截取失败")
        return patch, ox, oy

    def grab_client_norm_box(
        self,
        win: WindowInfo,
        box: tuple[float, float, float, float],
        *,
        prefer_mss: bool = False,
        fallback_full: bool = True,
    ) -> np.ndarray | None:
        """截客户区归一化框 ``(nx0,ny0,nx1,ny1)``。

        优先客户区 ROI BitBlt（源 hdc 偏移、目标即 ROI 大小），不先截整窗再裁。
        ``prefer_mss=True`` 且窗口前台时才直 mss ROI。
        ``fallback_full=True`` 时 BitBlt ROI 失败才 PrintWindow/整客户区再裁。
        auto前后台均只裁渲染截图；禁止整窗后备时返回None，不读取客户区DC表面。
        """
        x0, y0, x1, y1 = (float(v) for v in box)
        ox = max(0, int(win.width * x0))
        oy = max(0, int(win.height * y0))
        w = max(1, min(int(win.width * (x1 - x0)), win.width - ox))
        h = max(1, min(int(win.height * (y1 - y0)), win.height - oy))
        if self.method == 'auto':
            if not fallback_full:
                return None
            full = self.grab_client(win)
            return full[oy:oy+h, ox:ox+w].copy()
        if prefer_mss:
            from mm_assist.platform.window import is_foreground

            if not is_foreground(int(win.hwnd)):
                prefer_mss = False
        if prefer_mss:
            try:
                patch = self.grab_rect(win.left + ox, win.top + oy, w, h)
                if (patch is not None and not _looks_black(patch)
                        and _client_is_foreground(win)):
                    self._note("mss_roi")
                    return patch
            except Exception:
                pass
        # Unity 客户区常用 PrintWindow；局部 BitBlt 易黑屏/错位，条带会永远对不上图标。
        if self.method not in ("mss", "printwindow"):
            try:
                patch = self._bitblt_client(win, ox=ox, oy=oy, width=w, height=h)
            except Exception:
                patch = None
            if (patch is not None and not _looks_black(patch)
                    and (self.method != 'auto' or _client_is_foreground(win))):
                self._note("bitblt_roi")
                return patch
        if not fallback_full:
            return None
        full = self.grab_client(win)
        return full[oy : oy + h, ox : ox + w].copy()

    def _note(self, source: str) -> None:
        if self._last_source != source:
            self._last_source = source
            log.info("capture source=%s method=%s", source, self.method)

    def _printwindow_client(self, win: WindowInfo) -> np.ndarray | None:
        if sys.platform != "win32":
            return None
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        hwnd = int(win.hwnd)

        # 最小化时尝试不激活恢复，便于 PrintWindow
        try:
            if user32.IsIconic(hwnd):
                restore_no_activate(hwnd)
        except Exception:
            pass

        wr = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(wr)):
            return None
        win_left, win_top = int(wr.left), int(wr.top)
        win_w = int(wr.right - wr.left)
        win_h = int(wr.bottom - wr.top)
        if win_w <= 0 or win_h <= 0:
            return None

        hdc = user32.GetWindowDC(hwnd)
        if not hdc:
            return None
        memdc = gdi32.CreateCompatibleDC(hdc)
        bmp = gdi32.CreateCompatibleBitmap(hdc, win_w, win_h) if memdc else None
        if not memdc or not bmp:
            if memdc:
                gdi32.DeleteDC(memdc)
            user32.ReleaseDC(hwnd, hdc)
            return None
        gdi32.SelectObject(memdc, bmp)

        ok = user32.PrintWindow(hwnd, memdc, _PW_RENDERFULLCONTENT)
        if ok != 1:
            ok = user32.PrintWindow(hwnd, memdc, 0)
        if ok != 1:
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(memdc)
            user32.ReleaseDC(hwnd, hdc)
            return None

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = win_w
        bmi.bmiHeader.biHeight = -win_h
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = _BI_RGB
        buf = (ctypes.c_ubyte * (win_w * win_h * 4))()
        bits = gdi32.GetDIBits(memdc, bmp, 0, win_h, ctypes.byref(buf), ctypes.byref(bmi), _DIB_RGB_COLORS)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(memdc)
        user32.ReleaseDC(hwnd, hdc)
        if bits == 0:
            return None

        rgba = np.frombuffer(buf, dtype=np.uint8).reshape((win_h, win_w, 4))
        # 客户区相对窗口左上
        ox = int(win.left - win_left)
        oy = int(win.top - win_top)
        x1 = min(win_w, ox + win.width)
        y1 = min(win_h, oy + win.height)
        x0 = max(0, ox)
        y0 = max(0, oy)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = rgba[y0:y1, x0:x1, :3]
        return np.ascontiguousarray(crop)

    def _bitblt_client(
        self,
        win: WindowInfo,
        *,
        ox: int = 0,
        oy: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray | None:
        if sys.platform != "win32":
            return None
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        hwnd = int(win.hwnd)
        src_x = max(0, int(ox))
        src_y = max(0, int(oy))
        max_w = max(0, int(win.width) - src_x)
        max_h = max(0, int(win.height) - src_y)
        w = int(win.width if width is None else width)
        h = int(win.height if height is None else height)
        w = max(1, min(w, max_w if max_w > 0 else w))
        h = max(1, min(h, max_h if max_h > 0 else h))
        if int(win.width) <= 0 or int(win.height) <= 0 or w <= 0 or h <= 0:
            return None
        hdc = user32.GetDC(hwnd)
        if not hdc:
            return None
        memdc = gdi32.CreateCompatibleDC(hdc)
        bmp = gdi32.CreateCompatibleBitmap(hdc, w, h) if memdc else None
        if not memdc or not bmp:
            if memdc:
                gdi32.DeleteDC(memdc)
            user32.ReleaseDC(hwnd, hdc)
            return None
        gdi32.SelectObject(memdc, bmp)
        ok = gdi32.BitBlt(memdc, 0, 0, w, h, hdc, src_x, src_y, _SRCCOPY)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = _BI_RGB
        buf = (ctypes.c_ubyte * (w * h * 4))()
        bits = 0
        if ok:
            bits = gdi32.GetDIBits(memdc, bmp, 0, h, ctypes.byref(buf), ctypes.byref(bmi), _DIB_RGB_COLORS)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(memdc)
        user32.ReleaseDC(hwnd, hdc)
        if not ok or bits == 0:
            return None
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        return np.ascontiguousarray(rgba[:, :, :3])


def _looks_black(img: np.ndarray) -> bool:
    if img is None or img.size == 0:
        return True
    mean = float(img.mean())
    std = float(img.std())
    return mean < 5.0 and std < 2.0


_default: Grabber | None = None
_default_method: str = "auto"


def configure_grabber(method: str = "auto") -> Grabber:
    global _default, _default_method
    if _default is None or _default_method != method:
        if _default is not None:
            _default.close()
        _default = Grabber(method=method)
        _default_method = method
    return _default


def get_grabber() -> Grabber:
    global _default
    if _default is None:
        _default = Grabber()
    return _default


def grab_client(win: WindowInfo, method: str | None = None) -> np.ndarray:
    g = configure_grabber(method) if method else get_grabber()
    if method and g.method != method.lower():
        g = configure_grabber(method)
    return g.grab_client(win)


def grab_client_norm_box(
    win: WindowInfo,
    box: tuple[float, float, float, float],
    *,
    prefer_mss: bool = False,
    fallback_full: bool = True,
    method: str | None = None,
) -> np.ndarray | None:
    g = configure_grabber(method) if method else get_grabber()
    return g.grab_client_norm_box(
        win, box, prefer_mss=prefer_mss, fallback_full=fallback_full
    )


def save_capture(img_bgr: np.ndarray, out_dir: Path, prefix: str = "cap") -> Path:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{prefix}_{ts}.png"
    cv2.imwrite(str(path), img_bgr)
    return path
