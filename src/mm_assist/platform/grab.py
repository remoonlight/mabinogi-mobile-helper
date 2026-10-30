"""按配置初始化采图后端。"""

from __future__ import annotations

import numpy as np

from mm_assist.config import AppConfig
from mm_assist.platform.capture import configure_grabber, grab_client
from mm_assist.platform.window import WindowInfo, refresh_window, restore_no_activate


def prepare_grabber(cfg: AppConfig) -> None:
    configure_grabber(cfg.window.capture_method)


def refresh_and_grab(cfg: AppConfig, win: WindowInfo) -> tuple[WindowInfo, np.ndarray]:
    win = refresh_window(win)
    restore_no_activate(win.hwnd)
    img = grab_client(win, method=cfg.window.capture_method)
    return win, img
