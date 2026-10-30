"""可控随机准确点击：内缩框内采样（像人手，非提命中）。"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HitBox:
    """客户区归一化框 [x,y,w,h]，坐标均 ∈ [0,1] 语义。"""

    x: float
    y: float
    w: float
    h: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def clamp_unit(self) -> HitBox:
        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        w = min(max(self.w, 0.0), 1.0 - x)
        h = min(max(self.h, 0.0), 1.0 - y)
        return HitBox(x, y, max(w, 1e-6), max(h, 1e-6))


def hitbox_from_point(nx: float, ny: float, half_extent: float) -> HitBox:
    """示教单点 → 邻域扩框（±half_extent 客户区宽高），再由调用方 inset。"""
    he = max(float(half_extent), 1e-4)
    return HitBox(nx - he, ny - he, 2.0 * he, 2.0 * he).clamp_unit()


def hitbox_from_roi(roi: Sequence[float]) -> HitBox:
    x, y, w, h = (float(v) for v in roi[:4])
    return HitBox(x, y, w, h).clamp_unit()


def inset_hitbox(hitbox: HitBox, inset_ratio: float) -> HitBox:
    r = min(max(float(inset_ratio), 0.0), 0.49)
    ix = hitbox.x + hitbox.w * r
    iy = hitbox.y + hitbox.h * r
    iw = hitbox.w * (1.0 - 2.0 * r)
    ih = hitbox.h * (1.0 - 2.0 * r)
    return HitBox(ix, iy, max(iw, 1e-6), max(ih, 1e-6)).clamp_unit()


def sample_point_in_hitbox(
    hitbox: HitBox,
    inset_ratio: float = 0.18,
    sigma_ratio: float = 0.12,
    *,
    mode: str = "gaussian",
) -> tuple[float, float]:
    """
    只在内缩框内采样，禁止出框。
    mode=gaussian：相对 inset 框中心，σ = sigma_ratio * 边长；超出则拒样/回退均匀。
    mode=uniform：inset 框内均匀。
    """
    box = inset_hitbox(hitbox, inset_ratio)
    x0, y0, bw, bh = box.x, box.y, box.w, box.h
    x1, y1 = x0 + bw, y0 + bh
    cx, cy = x0 + bw * 0.5, y0 + bh * 0.5

    if (mode or "gaussian").lower() == "uniform":
        return (random.uniform(x0, x1), random.uniform(y0, y1))

    sx = max(bw * float(sigma_ratio), 1e-6)
    sy = max(bh * float(sigma_ratio), 1e-6)
    for _ in range(24):
        nx = random.gauss(cx, sx)
        ny = random.gauss(cy, sy)
        if x0 <= nx <= x1 and y0 <= ny <= y1:
            return (nx, ny)
    # 拒样耗尽：均匀落在 inset 内（仍禁止出框）
    return (random.uniform(x0, x1), random.uniform(y0, y1))


def hitbox_center(hitbox: HitBox) -> tuple[float, float]:
    return (hitbox.x + hitbox.w * 0.5, hitbox.y + hitbox.h * 0.5)


def _rand_ms(span: Sequence[int] | tuple[int, int]) -> float:
    lo, hi = int(span[0]), int(span[1])
    if hi < lo:
        lo, hi = hi, lo
    return random.uniform(lo, hi) / 1000.0


def sleep_pre_delay(pre_delay_ms: Sequence[int] | tuple[int, int] = (80, 220)) -> float:
    sec = _rand_ms(pre_delay_ms)
    time.sleep(sec)
    return sec


def sleep_hold(hold_ms: Sequence[int] | tuple[int, int] = (40, 90)) -> float:
    sec = _rand_ms(hold_ms)
    time.sleep(sec)
    return sec


def sample_hold_seconds(hold_ms: Sequence[int] | tuple[int, int] = (40, 90)) -> float:
    return _rand_ms(hold_ms)
