"""点击前后像素签名（从 assist click_verify 抽出 map_goto 用到的部分）。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from mm_assist.platform.match import roi_slice


@dataclass
class FrameSig:
    gray: np.ndarray
    mean: float


def make_sig(
    img: np.ndarray, *, scale: float = 0.2, roi: tuple[float, float, float, float] | None = None
) -> FrameSig:
    crop = img
    if roi is not None:
        crop, _, _ = roi_slice(img, roi)
    if crop.size == 0:
        return FrameSig(np.zeros((8, 8), dtype=np.uint8), 0.0)
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    h, w = gray.shape[:2]
    nh = max(8, int(h * scale))
    nw = max(8, int(w * scale))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    return FrameSig(small, float(small.mean()))


def sig_delta(a: FrameSig, b: FrameSig) -> float:
    if a.gray.shape != b.gray.shape:
        return 999.0
    return float(np.mean(cv2.absdiff(a.gray, b.gray)))


def wait_pixel_change(
    grab: Callable[[], np.ndarray],
    before: FrameSig,
    *,
    threshold: float = 3.5,
    timeout_s: float = 2.2,
    poll_s: float = 0.18,
    roi: tuple[float, float, float, float] | None = None,
) -> tuple[bool, float, FrameSig]:
    t0 = time.time()
    last = before
    best = 0.0
    while time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        img = grab()
        last = make_sig(img, roi=roi)
        d = sig_delta(before, last)
        if d > best:
            best = d
        if d >= threshold:
            return True, d, last
    return False, best, last
