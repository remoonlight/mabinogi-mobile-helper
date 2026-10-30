"""长期游戏检测/操作框：识别与点击的唯一客户区坐标系。

框 = 游戏窗口客户区；标定设计分辨率默认 1920×1440。
归一化 (nx,ny) 与标定像素 (px,py) 均相对此框。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mm_assist.platform.capture import grab_client
from mm_assist.config import AppConfig
from mm_assist.platform.window import WindowInfo, refresh_window

log = logging.getLogger(__name__)

DEFAULT_DESIGN = (1920, 1440)

@dataclass
class GameFrame:
    """检测与操作框（屏幕绝对坐标 + 设计分辨率）。"""

    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int
    design_w: int = DEFAULT_DESIGN[0]
    design_h: int = DEFAULT_DESIGN[1]

    @property
    def scale_x(self) -> float:
        return self.width / max(self.design_w, 1)

    @property
    def scale_y(self) -> float:
        return self.height / max(self.design_h, 1)

    def to_window(self) -> WindowInfo:
        return WindowInfo(
            self.hwnd, self.title, self.left, self.top, self.width, self.height
        )

    def refresh(self) -> GameFrame:
        w = refresh_window(self.to_window())
        self.hwnd = w.hwnd
        self.title = w.title
        self.left = w.left
        self.top = w.top
        self.width = w.width
        self.height = w.height
        return self

    def design_to_client(self, px: int, py: int) -> tuple[int, int]:
        """标定像素 → 当前客户区像素。"""
        return int(px * self.scale_x), int(py * self.scale_y)

    def norm_to_client(self, nx: float, ny: float) -> tuple[int, int]:
        return int(nx * self.width), int(ny * self.height)

    def client_to_norm(self, cx: int, cy: int) -> tuple[float, float]:
        return cx / max(self.width, 1), cy / max(self.height, 1)

    def screen_to_client(self, sx: int, sy: int) -> tuple[int, int]:
        """屏幕绝对坐标 → 相对客户区左上角。"""
        return int(sx) - self.left, int(sy) - self.top

    def client_inside(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.width and 0 <= cy < self.height

    def cursor_annotation(self, sx: int, sy: int) -> str:
        """角标「鼠 …」：相对客户区；窗外标「外」。"""
        cx, cy = self.screen_to_client(sx, sy)
        if not self.client_inside(cx, cy):
            return f"鼠 外 {cx},{cy}"
        nx, ny = self.client_to_norm(cx, cy)
        return f"鼠 {cx},{cy} ({nx:.3f},{ny:.3f})"

    def grab(self, cfg: AppConfig) -> np.ndarray:
        """截取框内画面（与 WindowInfo 客户区一致）。"""
        self.refresh()
        return grab_client(self.to_window(), method=cfg.window.capture_method)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "role": "long_term_detect_operate_frame",
            "design_resolution": [self.design_w, self.design_h],
            "hwnd": self.hwnd,
            "title": self.title,
            "screen": {
                "left": self.left,
                "top": self.top,
                "width": self.width,
                "height": self.height,
            },
            "scale": [round(self.scale_x, 4), round(self.scale_y, 4)],
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "识别截屏与点击均相对此客户区；标定坐标按 design_resolution 缩放",
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("game_frame saved %s %dx%d", path, self.width, self.height)
        return path


def frame_path(cfg: AppConfig) -> Path:
    root = Path(cfg.resolve(cfg.paths.knowledge))
    # knowledge/ui/config/game_frame.json
    return root / "ui" / "config" / "game_frame.json"


def from_window(
    win: WindowInfo, *, design: tuple[int, int] | None = None
) -> GameFrame:
    w = refresh_window(win)
    dw, dh = design or tuple(DEFAULT_DESIGN)
    return GameFrame(
        hwnd=w.hwnd,
        title=w.title,
        left=w.left,
        top=w.top,
        width=w.width,
        height=w.height,
        design_w=int(dw),
        design_h=int(dh),
    )


def load_frame(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# —— 全局活动框（自动脚本入口 ensure） ——
_ACTIVE: GameFrame | None = None


def get_frame() -> GameFrame | None:
    return _ACTIVE


def ensure_game_frame(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    persist: bool = True,
) -> GameFrame:
    """锁定/刷新长期游戏框（只坐标，不画描边）。以后识别与操作以此为准。"""
    global _ACTIVE
    design = tuple(cfg.window.client_size) if cfg.window.client_size else DEFAULT_DESIGN
    fr = from_window(win, design=(int(design[0]), int(design[1])))
    _ACTIVE = fr
    if persist:
        fr.save(frame_path(cfg))
    log.info(
        "game_frame active %dx%d design=%dx%d scale=%.3f",
        fr.width,
        fr.height,
        fr.design_w,
        fr.design_h,
        fr.scale_x,
    )
    return fr


def run_frame_cli(cfg: AppConfig, win: WindowInfo, *, watch_s: float = 3600.0) -> int:
    """锁定游戏框并写入 game_frame.json；可选挂起直到 Ctrl+C（不画描边）。"""
    fr = ensure_game_frame(cfg, win, persist=True)
    print(
        json.dumps(fr.to_dict(), ensure_ascii=False, indent=2),
        flush=True,
    )
    print(
        f"游戏检测框已锁定（无描边）。设计={fr.design_w}x{fr.design_h} "
        f"当前={fr.width}x{fr.height} 文件={frame_path(cfg)}",
        flush=True,
    )
    print("Ctrl+C 结束。", flush=True)
    try:
        t0 = time.time()
        while time.time() - t0 < watch_s:
            fr.refresh()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    return 0
