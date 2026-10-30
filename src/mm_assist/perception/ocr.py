from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from mm_assist.platform.match import roi_slice

_ocr_lock = threading.Lock()

# legacy=旧 rapidocr_onnxruntime（简体偏）；v5=PP-OCRv5 中英日混合（繁体矿字更准）；cht=繁体专用 v3
_DEFAULT_BACKEND = "v5"
# 离线榜：v5+up2 主用，v5+clahe 备用
_DEFAULT_PREPROCESS = "up2,clahe"
# crop 像素超过则本趟把预处理名 up2 换成 none（仍可走列表里的 clahe）
UP2_MAX_PIXELS = 600 * 300


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: list[list[float]] | None = None


def _backend_name() -> str:
    raw = (os.environ.get("MM_OCR_BACKEND") or _DEFAULT_BACKEND).strip().lower()
    if raw in ("legacy", "old", "onnxruntime", "rapidocr_onnxruntime"):
        return "legacy"
    if raw in ("cht", "chinese_cht", "trad", "traditional"):
        return "cht"
    if raw in ("v5", "ppocrv5", "rapid", "rapidocr", "default"):
        return "v5"
    return "v5"


def _preprocess_names() -> list[str]:
    raw = (os.environ.get("MM_OCR_PREPROCESS") or _DEFAULT_PREPROCESS).strip().lower()
    names = [p.strip() for p in raw.split(",") if p.strip()]
    return names or ["up2", "clahe"]


def preprocess_ocr(img_bgr: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    """返回 (图像, 坐标缩放)。up2 的 box 需除以 2 映回原图。"""
    if name in ("none", "raw", ""):
        return img_bgr, 1.0
    if name == "up2":
        return cv2.resize(img_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC), 2.0
    if name == "clahe":
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(clahe.apply(g), cv2.COLOR_GRAY2BGR), 1.0
    return img_bgr, 1.0


def _scale_lines(lines: list[OcrLine], scale: float) -> list[OcrLine]:
    if scale == 1.0:
        return lines
    inv = 1.0 / scale
    out: list[OcrLine] = []
    for line in lines:
        box = None
        if line.box:
            box = [[float(p[0]) * inv, float(p[1]) * inv] for p in line.box]
        out.append(OcrLine(line.text, line.confidence, box))
    return out


@lru_cache(maxsize=4)
def _engine(backend: str):
    if backend == "legacy":
        from rapidocr_onnxruntime import RapidOCR

        return ("legacy", RapidOCR())

    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    if backend == "cht":
        params = {
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.CHINESE_CHT,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV4,
        }
    else:
        # PP-OCRv5 ch：文档标明含繁体；实测礦脈类远优于旧引擎 / chinese_cht-only
        params = {
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.CH,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        }
    return ("rapid", RapidOCR(params=params))


def _parse_legacy(result) -> list[OcrLine]:
    lines: list[OcrLine] = []
    if not result:
        return lines
    for item in result:
        box, text, score = item[0], item[1], float(item[2])
        lines.append(OcrLine(text=str(text), confidence=score, box=box))
    return lines


def _parse_rapid(out) -> list[OcrLine]:
    lines: list[OcrLine] = []
    if out is None:
        return lines
    boxes = getattr(out, "boxes", None)
    txts = getattr(out, "txts", None) or ()
    scores = getattr(out, "scores", None) or ()
    if boxes is None:
        return lines
    for i, text in enumerate(txts):
        score = float(scores[i]) if i < len(scores) else 0.0
        box = boxes[i]
        box_list = [[float(p[0]), float(p[1])] for p in box]
        lines.append(OcrLine(text=str(text), confidence=score, box=box_list))
    return lines


def ocr_image(
    img_bgr: np.ndarray,
    min_confidence: float = 0.5,
    *,
    backend: str | None = None,
) -> list[OcrLine]:
    """OCR 非线程安全：全局锁串行化。

    默认 ``MM_OCR_BACKEND=v5``，预处理 ``up2`` 主用、``clahe`` 备用（空结果才回退）。
    ``backend`` 可强制 ``v5`` / ``cht`` / ``legacy``。
    """
    use = (backend or _backend_name()).strip().lower()
    if use in ("cht", "chinese_cht", "trad", "traditional"):
        use = "cht"
    elif use in ("legacy", "old", "onnxruntime", "rapidocr_onnxruntime"):
        use = "legacy"
    elif use in ("v5", "ppocrv5", "rapid", "rapidocr", "default", ""):
        use = "v5"
    names = _preprocess_names()
    try:
        h, w = int(img_bgr.shape[0]), int(img_bgr.shape[1])
        pixels = h * w
    except Exception:
        pixels = 0
    if pixels > UP2_MAX_PIXELS:
        names = ["none" if n == "up2" else n for n in names]
    last: list[OcrLine] = []
    with _ocr_lock:
        kind, engine = _engine(use)
        for i, name in enumerate(names):
            work, scale = preprocess_ocr(img_bgr, name)
            if kind == "legacy":
                result, _ = engine(work)
                lines = _parse_legacy(result)
            else:
                lines = _parse_rapid(engine(work))
            lines = _scale_lines(lines, scale)
            kept = [L for L in lines if L.confidence >= min_confidence]
            last = kept
            if kept:
                return kept
            if i + 1 < len(names):
                continue
    return last


def ocr_roi(
    img_bgr: np.ndarray,
    roi: tuple[float, float, float, float],
    min_confidence: float = 0.5,
    *,
    backend: str | None = None,
    offset_boxes: bool = True,
) -> list[OcrLine]:
    crop, ox, oy = roi_slice(img_bgr, roi)
    if crop is None or getattr(crop, "size", 0) == 0:
        return []
    lines = ocr_image(crop, min_confidence=min_confidence, backend=backend)
    if not offset_boxes:
        return lines
    out: list[OcrLine] = []
    for line in lines:
        box = None
        if line.box:
            box = [[float(p[0]) + ox, float(p[1]) + oy] for p in line.box]
        out.append(OcrLine(line.text, line.confidence, box))
    return out


def text_contains_any(lines: list[OcrLine], needles: list[str]) -> bool:
    blob = "".join(l.text for l in lines)
    return any(n and n in blob for n in needles)
