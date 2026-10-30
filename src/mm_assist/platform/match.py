from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_CV_METHODS = {
    "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
    "ccorr_normed": cv2.TM_CCORR_NORMED,
    "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
}


@dataclass
class MatchHit:
    score: float
    cx: int
    cy: int
    x: int
    y: int
    w: int
    h: int
    margin: float = 0.0


def roi_slice(img: np.ndarray, roi: tuple[float, float, float, float]) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    x, y, rw, rh = roi
    x0 = max(0, int(x * w))
    y0 = max(0, int(y * h))
    x1 = min(w, int((x + rw) * w))
    y1 = min(h, int((y + rh) * h))
    return img[y0:y1, x0:x1], x0, y0


@lru_cache(maxsize=256)
def _load_template_gray(path_str: str) -> tuple[np.ndarray, int, int]:
    # Windows：cv2.imread 无法读含非 ASCII 路径（会刷 WARN）；非 ASCII 直接 fromfile+imdecode
    tmpl = None
    if path_str.isascii():
        tmpl = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        try:
            raw = np.fromfile(path_str, dtype=np.uint8)
        except OSError as exc:
            raise FileNotFoundError(f"template missing: {path_str}") from exc
        if raw.size == 0:
            raise FileNotFoundError(f"template missing: {path_str}")
        tmpl = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise FileNotFoundError(f"template missing: {path_str}")
    th, tw = tmpl.shape[:2]
    return tmpl, tw, th


def clear_template_cache() -> None:
    _load_template_gray.cache_clear()


def _peak_and_margin(res: np.ndarray, tw: int, th: int, higher_is_better: bool) -> tuple[float, tuple[int, int], float]:
    if higher_is_better:
        _, best, _, best_loc = cv2.minMaxLoc(res)
        best_f = float(best)
    else:
        best_raw, _, best_loc, _ = cv2.minMaxLoc(res)
        best_f = 1.0 - float(best_raw)

    rad = max(5, min(tw, th, 24) // 4)
    x, y = int(best_loc[0]), int(best_loc[1])
    masked = res.copy()
    x0, y0 = max(0, x - rad), max(0, y - rad)
    x1, y1 = min(res.shape[1], x + rad + 1), min(res.shape[0], y + rad + 1)

    if higher_is_better:
        masked[y0:y1, x0:x1] = -1.0
        _, second, _, _ = cv2.minMaxLoc(masked)
        second_f = float(second)
        if second_f < 0:
            return best_f, best_loc, best_f
        margin = best_f - second_f
    else:
        masked[y0:y1, x0:x1] = 2.0
        second_raw, _, _, _ = cv2.minMaxLoc(masked)
        second_f = 1.0 - float(second_raw)
        margin = best_f - second_f

    return best_f, best_loc, margin


def match_template(
    haystack_bgr: np.ndarray,
    template_path: Path,
    roi: tuple[float, float, float, float] | None = None,
    threshold: float = 0.88,
    method: str = "ccoeff_normed",
    min_margin: float = 0.05,
    scale: float = 1.0,
    check_margin: bool = True,
) -> MatchHit | None:
    """固定分辨率单尺度灰度匹配；低纹理模板直接拒绝。

    scale<1 时对搜索区与模板同比例缩小（坐标映射回原图）；状态栏快路径用。
    check_margin=False 时跳过次峰 margin（更快）。
    """
    tmpl, tw, th = _load_template_gray(str(template_path.resolve()))
    if float(np.std(tmpl)) < 5.0:
        return None

    region, ox, oy = (haystack_bgr, 0, 0) if roi is None else roi_slice(haystack_bgr, roi)
    region_g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    sc = float(scale) if scale and scale > 0 else 1.0
    if sc != 1.0:
        region_g = cv2.resize(
            region_g,
            (max(1, int(region_g.shape[1] * sc)), max(1, int(region_g.shape[0] * sc))),
            interpolation=cv2.INTER_AREA,
        )
        tmpl = cv2.resize(
            tmpl,
            (max(1, int(tw * sc)), max(1, int(th * sc))),
            interpolation=cv2.INTER_AREA,
        )
        tw, th = tmpl.shape[1], tmpl.shape[0]
    if region_g.shape[0] < th or region_g.shape[1] < tw:
        return None

    cv_method = _CV_METHODS.get(method, cv2.TM_CCOEFF_NORMED)
    higher = cv_method != cv2.TM_SQDIFF_NORMED
    res = cv2.matchTemplate(region_g, tmpl, cv_method)
    if check_margin and min_margin > 0:
        score, max_loc, margin = _peak_and_margin(res, tw, th, higher)
        if score < threshold or margin < min_margin:
            return None
    else:
        if higher:
            _, best, _, best_loc = cv2.minMaxLoc(res)
            score = float(best)
        else:
            best_raw, _, best_loc, _ = cv2.minMaxLoc(res)
            score = 1.0 - float(best_raw)
        max_loc = best_loc
        margin = score
        if score < threshold:
            return None
    # 缩放坐标映回原图 ROI 原点
    inv = 1.0 / sc
    x = int(max_loc[0] * inv) + ox
    y = int(max_loc[1] * inv) + oy
    ow = int(round(tw * inv))
    oh = int(round(th * inv))
    return MatchHit(score, x + ow // 2, y + oh // 2, x, y, ow, oh, margin)


def suppress_red_badge(bgr: np.ndarray) -> np.ndarray:
    """把高饱和红点（角标）涂成暗色，避免破坏图标灰度相关。"""
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 70, 70), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 70, 70), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    if int(cv2.countNonZero(mask)) == 0:
        return bgr
    out = bgr.copy()
    out[mask > 0] = (36, 36, 36)
    return out


def match_icon_fuzzy(
    haystack_bgr: np.ndarray,
    template_path: Path,
    roi: tuple[float, float, float, float] | None = None,
    threshold: float = 0.58,
    scales: tuple[float, ...] = (0.78, 0.88, 0.96, 1.0, 1.08, 1.18, 1.32),
) -> MatchHit | None:
    """多尺度灰度相关；去红点角标；不要求次峰 margin（图标允许部分被挡）。"""
    work = suppress_red_badge(haystack_bgr)
    best: MatchHit | None = None
    for sc in scales:
        hit = match_template(
            work,
            template_path,
            roi=roi,
            threshold=threshold,
            min_margin=0.0,
            scale=float(sc),
            check_margin=False,
        )
        if hit is None:
            continue
        if best is None or hit.score > best.score:
            best = hit
    return best


def find_hamburger_bars(
    haystack_bgr: np.ndarray,
    roi: tuple[float, float, float, float] | None = None,
) -> MatchHit | None:
    """找圆底上的三根横条（等长或左对齐长短条均可），忽略红点。"""
    work = suppress_red_badge(haystack_bgr)
    region, ox, oy = (work, 0, 0) if roi is None else roi_slice(work, roi)
    if region.size == 0:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    h, w = gray.shape[:2]
    if h < 12 or w < 12:
        return None
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -8
    )
    kw = max(5, w // 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    row_energy = (horiz > 0).sum(axis=1).astype(np.float32)
    if float(row_energy.max(initial=0.0)) < kw:
        return None
    gate = max(float(kw) * 0.85, float(row_energy.max()) * 0.35)
    active = row_energy >= gate
    bands: list[tuple[int, int]] = []
    i = 0
    while i < h:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < h and active[j]:
            j += 1
        if 1 <= (j - i) <= max(6, h // 8):
            bands.append((i, j))
        i = j
    if len(bands) < 3:
        return None
    # 取能量最高的三带，再按 y 排序
    ranked = sorted(
        bands,
        key=lambda b: float(row_energy[b[0] : b[1]].sum()),
        reverse=True,
    )[:3]
    ranked.sort(key=lambda b: b[0])
    if len(ranked) < 3:
        return None
    gaps = [ranked[k + 1][0] - ranked[k][1] for k in range(2)]
    if max(gaps) > max(10, h // 2) or min(gaps) < 1:
        return None
    if max(gaps) > min(gaps) * 3.2:
        return None
    y0, y1 = ranked[0][0], ranked[2][1]
    mid_rows = horiz[y0:y1]
    col_energy = (mid_rows > 0).sum(axis=0)
    xs = np.where(col_energy >= max(1, int(0.4 * (y1 - y0))))[0]
    if xs.size == 0:
        return None
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    bw_box = max(8, x1 - x0)
    bh_box = max(8, y1 - y0)
    score = min(0.95, 0.55 + 0.08 * 3)
    return MatchHit(
        score=float(score),
        cx=ox + (x0 + x1) // 2,
        cy=oy + (y0 + y1) // 2,
        x=ox + x0,
        y=oy + y0,
        w=bw_box,
        h=bh_box,
        margin=0.12,
    )
