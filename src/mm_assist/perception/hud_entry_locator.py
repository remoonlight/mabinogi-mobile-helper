"""主界面入口图像定位与客户区点击点计算。

只读：本模块只分析已经抓取的游戏客户区图像，不发送任何输入。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mm_assist.config import ROOT
from mm_assist.platform.match import (
    find_hamburger_bars,
    match_icon_fuzzy,
    match_template,
    roi_slice,
)

_SAFE_CAPTURE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_capture_id(value: str) -> str:
    """校验可用作单层证据目录名的采集 ID。"""
    raw = str(value or "")
    text = raw.strip()
    if raw != text or text.endswith("."):
        raise ValueError("capture_id 不得包含首尾空白或尾随点")
    if not _SAFE_CAPTURE_ID.fullmatch(text):
        raise ValueError("capture_id 仅允许 1-64 位 ASCII 字母、数字、点、横线和下划线")
    if text.upper().split(".", 1)[0] in _WINDOWS_RESERVED:
        raise ValueError(f"capture_id 不得使用 Windows 保留名：{text!r}")
    return text


def _normalized_roi(locator: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = locator.get("roi")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1.0001 or y + h > 1.0001:
        return None
    return x, y, w, h


def _inflate_roi(
    roi: tuple[float, float, float, float] | None,
    *,
    pad: float = 0.05,
) -> tuple[float, float, float, float] | None:
    if roi is None:
        return None
    x, y, w, h = roi
    x2 = max(0.0, x - pad)
    y2 = max(0.0, y - pad)
    right = min(1.0, x + w + pad)
    bottom = min(1.0, y + h + pad)
    return x2, y2, max(0.01, right - x2), max(0.01, bottom - y2)


# 主界面背包在右下簇（I），左为製作 K、右为停止 Space。扩大搜索后用归一化门闩防误点。
_BAG_NX = (0.80, 0.89)
_BAG_NY = (0.86, 0.98)
_BAG_NW = (0.035, 0.10)
_BAG_NH = (0.04, 0.12)


def _bag_geometry_ok(
    hit: tuple[int, int, int, int, float, float] | Any,
    frame_w: int,
    frame_h: int,
) -> bool:
    """背包命中须在右下图标带，且尺寸接近 HUD 圆钮。"""
    if frame_w <= 0 or frame_h <= 0:
        return False
    if hasattr(hit, "cx"):
        cx, cy, w, h = float(hit.cx), float(hit.cy), float(hit.w), float(hit.h)
    else:
        x, y, w, h = hit[0], hit[1], hit[2], hit[3]
        cx, cy = x + w / 2.0, y + h / 2.0
    nx, ny = cx / float(frame_w), cy / float(frame_h)
    nw, nh = float(w) / float(frame_w), float(h) / float(frame_h)
    return (
        _BAG_NX[0] <= nx <= _BAG_NX[1]
        and _BAG_NY[0] <= ny <= _BAG_NY[1]
        and _BAG_NW[0] <= nw <= _BAG_NW[1]
        and _BAG_NH[0] <= nh <= _BAG_NH[1]
    )


def _want_fuzzy(locator: dict[str, Any], entry_id: str) -> bool:
    if bool(locator.get("fuzzy")):
        return True
    mode = str(locator.get("match") or locator.get("mode") or "").strip().lower()
    if mode in {"fuzzy", "feature", "icon"}:
        return True
    return entry_id in {"menu", "bag", "avatar"}


def _point_payload(
    cx: int,
    cy: int,
    *,
    frame_w: int,
    frame_h: int,
    screen_origin: tuple[int, int],
) -> dict[str, list[float] | list[int]]:
    sx0, sy0 = int(screen_origin[0]), int(screen_origin[1])
    return {
        "client": [int(cx), int(cy)],
        "norm": [float(cx) / max(frame_w, 1), float(cy) / max(frame_h, 1)],
        "screen": [sx0 + int(cx), sy0 + int(cy)],
    }


def _provisional_result(
    entry_id: str,
    locator: dict[str, Any],
    *,
    frame_w: int,
    frame_h: int,
    screen_origin: tuple[int, int],
    reason: str,
) -> dict[str, Any]:
    roi = _normalized_roi(locator)
    click = None
    rect = None
    if roi is not None:
        x, y, w, h = roi
        x0, y0 = int(x * frame_w), int(y * frame_h)
        rw, rh = int(w * frame_w), int(h * frame_h)
        rect = [x0, y0, rw, rh]
        click = _point_payload(
            x0 + rw // 2,
            y0 + rh // 2,
            frame_w=frame_w,
            frame_h=frame_h,
            screen_origin=screen_origin,
        )
    return {
        "entry_id": entry_id,
        "status": "calibration_required",
        "reason": reason,
        "detector": str(locator.get("detector") or ""),
        "maturity": str(locator.get("maturity") or "calibration"),
        "feature_ok": False,
        "position_ok": click is not None,
        "provisional": click is not None,
        "accepted": False,
        "feature_rect_client": rect,
        "click": click,
    }


def _find_stop_square(
    frame: np.ndarray,
    roi: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int, float] | None:
    region, ox, oy = (frame, 0, 0) if roi is None else roi_slice(frame, roi)
    if region.size == 0 or region.shape[0] < 20 or region.shape[1] < 20:
        return None
    b, g, r = cv2.split(region)
    green = (
        (g.astype(np.int16) > b.astype(np.int16) + 18)
        & (g.astype(np.int16) > r.astype(np.int16) + 12)
        & (g > 70)
    )
    white = (
        (r > 170)
        & (g > 170)
        & (b > 170)
        & (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 45)
        & (np.abs(g.astype(np.int16) - b.astype(np.int16)) < 45)
    )
    green_d = cv2.dilate(green.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2)
    mask = ((white.astype(np.uint8) > 0) & (green_d > 0)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, int, int, int, int]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        if area < 36 or w < 6 or h < 6:
            continue
        aspect = w / max(float(h), 1.0)
        extent = area / max(float(w * h), 1.0)
        fill = float(np.mean(white[y : y + h, x : x + w]))
        if not 0.75 <= aspect <= 1.35 or extent < 0.72 or fill < 0.70:
            continue
        green_near = float(np.mean(green_d[max(0, y - 5) : y + h + 5, max(0, x - 5) : x + w + 5]))
        candidates.append((extent + green_near, x + ox, y + oy, w, h))
    if not candidates:
        return None
    score, x, y, w, h = max(candidates, key=lambda item: item[0])
    return x, y, w, h, min(1.0, score / 2.0)


def locate_hud_entry(
    entry: dict[str, Any],
    frame: np.ndarray,
    *,
    root: Path | None = None,
    screen_origin: tuple[int, int] = (0, 0),
    prefer_roi_center: bool = False,
) -> dict[str, Any]:
    """定位一个入口并输出可审计的客户区、归一化和屏幕坐标。"""
    entry_id = str(entry.get("id") or "").strip()
    locator = entry.get("locator") if isinstance(entry.get("locator"), dict) else {}
    if not entry_id or frame is None or getattr(frame, "size", 0) == 0:
        return {
            "entry_id": entry_id,
            "status": "invalid_frame",
            "feature_ok": False,
            "position_ok": False,
            "provisional": False,
            "accepted": False,
            "click": None,
        }
    frame_h, frame_w = frame.shape[:2]
    detector = str(locator.get("detector") or "").strip()
    roi = _normalized_roi(locator)
    maturity = str(locator.get("maturity") or "calibration")
    if prefer_roi_center:
        return _provisional_result(
            entry_id,
            locator,
            frame_w=frame_w,
            frame_h=frame_h,
            screen_origin=screen_origin,
            reason="roi_center",
        )

    hit: tuple[int, int, int, int, float, float] | None = None
    if detector == "template":
        ref = str(locator.get("template") or "").strip()
        template_path = Path(ref)
        if not template_path.is_absolute():
            template_path = (root or ROOT) / template_path
        if not ref or not template_path.is_file():
            return _provisional_result(
                entry_id,
                locator,
                frame_w=frame_w,
                frame_h=frame_h,
                screen_origin=screen_origin,
                reason="missing_template",
            )
        # 菜单不外扩 ROI：外扩易吃到摄像机/生活入口（实机曾点偏未开系统菜单）。
        search_roi = roi if entry_id == "menu" else (
            _inflate_roi(roi) if _want_fuzzy(locator, entry_id) else roi
        )

        # 每个候选先验位置；错误纹理不能挡住后续图标检测。
        def _position_checked(candidate):
            if entry_id != "menu" or candidate is None or frame_w <= 0:
                return candidate
            left = float((roi or (0.0, 0.0, 1.0, 1.0))[0])
            nx = float(candidate.cx) / float(frame_w)
            if left >= 0.7 and nx < 0.83:
                return None
            if left >= 0.83 and nx >= 0.87:
                return None
            return candidate

        matched = None
        if entry_id == "menu":
            matched = _position_checked(
                find_hamburger_bars(
                    frame, roi=search_roi or roi or (0.83, 0.0, 0.14, 0.12)
                )
            )
        if matched is None and _want_fuzzy(locator, entry_id):
            matched = _position_checked(
                match_icon_fuzzy(
                    frame,
                    template_path,
                    roi=search_roi,
                    threshold=float(locator.get("threshold") or 0.58),
                )
            )
        if matched is None:
            matched = _position_checked(
                match_template(
                    frame,
                    template_path,
                    roi=roi,
                    threshold=float(locator.get("threshold") or 0.8),
                    min_margin=float(
                        locator.get("min_margin") or locator.get("min_margin") or 0.02
                    ),
                ),
            )
        if (
            entry_id == "bag"
            and matched is not None
            and not _bag_geometry_ok(matched, frame_w, frame_h)
        ):
            matched = None
        if matched is not None:
            hit = matched.x, matched.y, matched.w, matched.h, matched.score, matched.margin
    elif detector == "stop_shape":
        stopped = _find_stop_square(frame, roi)
        if stopped is not None:
            x, y, w, h, score = stopped
            hit = x, y, w, h, score, score
    else:
        return _provisional_result(
            entry_id,
            locator,
            frame_w=frame_w,
            frame_h=frame_h,
            screen_origin=screen_origin,
            reason="unknown_detector",
        )

    if hit is None:
        return {
            "entry_id": entry_id,
            "status": "no_match",
            "detector": detector,
            "maturity": maturity,
            "feature_ok": False,
            "position_ok": False,
            "provisional": False,
            "accepted": False,
            "feature_rect_client": None,
            "click": None,
        }

    x, y, w, h, score, margin = hit
    cx, cy = x + w // 2, y + h // 2
    inside = 0 <= cx < frame_w and 0 <= cy < frame_h
    accepted = bool(inside and maturity == "accepted")
    return {
        "entry_id": entry_id,
        "status": "matched",
        "detector": detector,
        "maturity": maturity,
        "feature_ok": True,
        "position_ok": inside,
        "provisional": False,
        "accepted": accepted,
        "score": float(score),
        "margin": float(margin),
        "feature_rect_client": [int(x), int(y), int(w), int(h)],
        "click": _point_payload(
            cx,
            cy,
            frame_w=frame_w,
            frame_h=frame_h,
            screen_origin=screen_origin,
        )
        if inside
        else None,
    }
