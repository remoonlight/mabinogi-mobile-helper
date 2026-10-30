"""状态栏感知：右下移动态 + 修理相关 OCR 标识命中。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mm_assist.config import ROOT
from mm_assist.perception.list_ocr import normalize_text
from mm_assist.perception.map_classify import GOTO_NEEDLES, STOP_NEEDLES
from mm_assist.platform.match import match_template, roi_slice
from mm_assist.perception.ocr import OcrLine, ocr_image

# 修理路径关键 OCR（繁中 + 常见误识）
REPAIR_OCR_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("前往此处", ("前往此處", "前往此处", "前往此", "前往")),
    ("歐拉大陸", ("歐拉大陸", "欧拉大陆", "歐拉大睦", "欧拉大睦", "歐拉", "欧拉")),
    ("杜巴頓", ("杜巴頓", "杜巴顿", "杜巴頔", "尚杜巴")),
    ("奈麗絲", ("奈麗絲", "奈麗", "奈丽", "武器商人")),
    ("跳過對話", ("跳過對話", "跳過对話", "跳过对话", "跳過")),
    ("修理", ("修理", "修復", "修复", "維修")),
    ("修理已佩戴", ("修理已佩戴的裝備", "修理已佩戴的装备", "修理已佩戴")),
    ("結束對話", ("結束對話", "结束对话", "再見", "再见")),
)

# 归一化 ROI：(x0, y0, x1, y1)
TRAVEL_ROI = (0.70, 0.78, 1.0, 1.0)
# 模板搜索：TRAVEL 同区的 xywh（match_template 用 xywh，不是 x0y0x1y1）
STOP_TEMPLATE_SEARCH_ROI = (0.70, 0.78, 0.30, 0.22)
STOP_TEMPLATE_PATH = ROOT / "knowledge" / "templates" / "hud_entries" / "stop.png"
# 对话选项带（中下），比大半屏小很多
DIALOG_ROI = (0.22, 0.52, 0.78, 0.92)


def detect_stop_icon_patch(
    br: np.ndarray | None,
    *,
    ocr_space: bool = False,
    min_confidence: float = 0.35,
) -> bool | None:
    """在已裁好的右下 TRAVEL 块上认「绿圆 + 实心白方停」。"""
    if br is None or getattr(br, "size", 0) == 0:
        return None
    bh, bw = br.shape[:2]
    if bh < 20 or bw < 20:
        return None
    b, g, r = cv2.split(br)
    # 浅绿圆底（seafoam）：G 明显高于 R/B
    green = (
        (g.astype(np.int16) > b.astype(np.int16) + 18)
        & (g.astype(np.int16) > r.astype(np.int16) + 12)
        & (g > 70)
    )
    # 白方角停：高亮且通道接近
    white = (
        (r.astype(np.int16) > 170)
        & (g.astype(np.int16) > 170)
        & (b.astype(np.int16) > 170)
        & (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 45)
        & (np.abs(g.astype(np.int16) - b.astype(np.int16)) < 45)
    )
    kernel = np.ones((5, 5), np.uint8)
    green_d = cv2.dilate(green.astype(np.uint8), kernel, iterations=2)
    white_on_green = (white.astype(np.uint8) > 0) & (green_d > 0)
    green_mean = float(np.mean(green))

    min_area = max(36.0, 0.0015 * bh * bw)
    max_area = 0.12 * bh * bw
    found_square = False
    cnts, _ = cv2.findContours(
        white_on_green.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        _x, _y, cw, ch = cv2.boundingRect(c)
        if cw < 6 or ch < 6:
            continue
        ar = cw / float(ch)
        if not (0.75 <= ar <= 1.35):
            continue
        extent = area / float(cw * ch)
        # 实心方 ~0.95；举手白掌 ~0.52 — 必须够实
        if extent < 0.72:
            continue
        fill = float(np.mean(white[_y : _y + ch, _x : _x + cw]))
        if fill < 0.70:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # 方块约 4 顶点；允许轻微圆角到 6
        if len(approx) > 6:
            continue
        found_square = True
        break

    if found_square and green_mean >= 0.004:
        return True

    if ocr_space:
        try:
            lines = ocr_image(br, min_confidence=min_confidence)
        except Exception:
            lines = []
        blob = normalize_text("".join(L.text for L in lines))
        # 禁止裸 Space：举手/鸟/钓鱼都有 Space 键位条
        if any(normalize_text(n) in blob for n in STOP_NEEDLES):
            return True

    return False


def match_stop_template(
    img: np.ndarray | None,
    *,
    threshold: float = 0.78,
    template_path: Path | None = None,
) -> bool | None:
    """右下停钮模板。缺文件 → None；未命中 → False；命中 → True。"""
    if img is None or getattr(img, "size", 0) == 0:
        return None
    path = Path(template_path) if template_path is not None else STOP_TEMPLATE_PATH
    if not path.is_file():
        return None
    hit = match_template(
        img,
        path,
        roi=STOP_TEMPLATE_SEARCH_ROI,
        threshold=threshold,
        check_margin=False,
    )
    return hit is not None


def detect_stop_icon(
    img: np.ndarray | None,
    *,
    ocr_space: bool = False,
    min_confidence: float = 0.35,
) -> bool | None:
    """识别右下停钮：绿圆白方形状 **或** ``stop.png`` 模板。

    反例（同为绿底 / Space，但不是停）：举手、鸟绿环、钓鱼竿等。
    形状须白块近似正方形且 bbox 填充高；禁止仅凭 OCR ``Space`` 判停。
    ``ocr_space=True`` 时仅补 OCR「停止/取消移動」类针，不含 Space。
    """
    if img is None or getattr(img, "size", 0) == 0:
        return None
    h, w = img.shape[:2]
    if h < 40 or w < 40:
        return None
    br = img[
        int(h * TRAVEL_ROI[1]) : int(h * TRAVEL_ROI[3]),
        int(w * TRAVEL_ROI[0]) : int(w * TRAVEL_ROI[2]),
    ]
    shape = detect_stop_icon_patch(
        br, ocr_space=ocr_space, min_confidence=min_confidence
    )
    tmpl = match_stop_template(img)
    if tmpl is True or shape is True:
        return True
    if tmpl is False and shape is False:
        return False
    if tmpl is None:
        return shape
    return False


def detect_travel_state(img: np.ndarray | None, lines: list[OcrLine] | None = None) -> str:
    """右下角移动态：navigate | stop | hidden。优先 OCR 针，再像素粗判。"""
    blob = ""
    if lines:
        blob = normalize_text("".join(L.text for L in lines))
    if any(normalize_text(n) in blob for n in STOP_NEEDLES):
        return "stop"
    if any(normalize_text(n) in blob for n in GOTO_NEEDLES):
        return "navigate"
    if img is None or img.size == 0:
        return "hidden"
    h, w = img.shape[:2]
    if h < 40 or w < 40:
        return "hidden"
    br = img[int(h * 0.78) : h, int(w * 0.70) : w]
    if br.size == 0:
        return "hidden"
    b, g, r = cv2.split(br)
    green = (g.astype(np.int16) > b.astype(np.int16) + 25) & (
        g.astype(np.int16) > r.astype(np.int16) + 15
    )
    if float(np.mean(green)) > 0.012:
        return "navigate"
    redish = (r.astype(np.int16) > g.astype(np.int16) + 20) & (
        r.astype(np.int16) > b.astype(np.int16) + 20
    )
    if float(np.mean(redish)) > 0.015:
        return "stop"
    return "hidden"


def travel_state_label(state: str) -> str:
    return {"navigate": "导航", "stop": "停止", "hidden": "不显示"}.get(state, "不显示")


def match_repair_ocr_markers(lines: list[OcrLine] | None) -> dict[str, bool]:
    """修理相关标识是否出现在当前 OCR 行。"""
    blob = normalize_text("".join(L.text for L in (lines or [])))
    out: dict[str, bool] = {}
    for label, needles in REPAIR_OCR_MARKERS:
        out[label] = any(normalize_text(n) in blob for n in needles)
    return out


def format_marker_line(hits: dict[str, bool]) -> str:
    parts = [f"{'✓' if v else '·'}{k}" for k, v in hits.items()]
    return " ".join(parts)


def empty_marker_hits() -> dict[str, bool]:
    return {k: False for k, _ in REPAIR_OCR_MARKERS}


def _crop_scaled(
    img: np.ndarray,
    roi: tuple[float, float, float, float],
    *,
    max_w: int,
) -> np.ndarray | None:
    crop, _, _ = roi_slice(img, roi)
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    h, w = crop.shape[:2]
    if w > max_w and w > 0:
        scale = max_w / w
        crop = cv2.resize(
            crop, (max_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    return crop


def sense_from_image(
    img: np.ndarray,
    *,
    min_confidence: float = 0.35,
    need_markers: bool = True,
    travel_ocr: bool = False,
) -> tuple[str, str]:
    """窄 ROI 感知。

    - 移动态：默认只像素（RapidOCR 兜底约 250ms，状态栏默认关）
    - 修理标识：仅 need_markers 时 OCR 对话选项带
    """
    travel = detect_travel_state(img, None)
    if travel == "hidden" and travel_ocr:
        br = _crop_scaled(img, TRAVEL_ROI, max_w=360)
        if br is not None:
            travel = detect_travel_state(
                img, ocr_image(br, min_confidence=min_confidence)
            )
    if need_markers:
        dlg = _crop_scaled(img, DIALOG_ROI, max_w=420)
        lines = ocr_image(dlg, min_confidence=min_confidence) if dlg is not None else []
        hits = match_repair_ocr_markers(lines)
    else:
        hits = empty_marker_hits()
    return travel, format_marker_line(hits)


def should_ocr_markers(*, ui: str = "", ui_params: str = "") -> bool:
    """对话/修理/地图相关时才扫标识 OCR。"""
    if ui in ("dialog", "repair", "map_town", "map_continent", "map_region"):
        return True
    p = ui_params or ""
    return "dlg=1" in p or "map=1" in p


__all__ = [
    "REPAIR_OCR_MARKERS",
    "TRAVEL_ROI",
    "DIALOG_ROI",
    "detect_stop_icon",
    "match_stop_template",
    "STOP_TEMPLATE_PATH",
    "STOP_TEMPLATE_SEARCH_ROI",
    "detect_travel_state",
    "travel_state_label",
    "match_repair_ocr_markers",
    "format_marker_line",
    "empty_marker_hits",
    "sense_from_image",
    "should_ocr_markers",
]
