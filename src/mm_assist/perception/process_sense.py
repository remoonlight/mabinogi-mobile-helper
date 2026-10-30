"""加工枢纽六卡 + 金属/木材/布料设施内配方行：图模板优先，失败繁体 OCR 降级。

只读，不发送键鼠。OCR 只作定位旁证，不得当制作成功主判。
可见即 locate ≠ 可投料 / 制作成功。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from functools import lru_cache
from typing import Any

import numpy as np

from mm_assist.config import ROOT
from mm_assist.perception.item_sense import compact_zh, item_name_variants, line_matches_name
from mm_assist.perception.ocr import OcrLine, ocr_roi
from mm_assist.platform.match import match_template, MatchHit, roi_slice, _peak_and_margin

HUB_CARD_IDS = ("metal", "wood", "leather", "cloth", "medicine", "food")
RECIPE_FACILITIES = ("metal", "wood", "cloth", "medicine")

TEMPLATE_THRESHOLD = 0.76
CARD_AMBIGUOUS_DELTA = 0.04
CARD_MARGIN_MIN = 0.03

# 相对客户区 (x, y, w, h)
HUB_CARDS_ROI = (0.10, 0.10, 0.82, 0.80)
# 六卡中心（1920×1440 实机）；模板/OCR 都丢时，其它卡已命中才用。
HUB_CARD_CLICK: dict[str, tuple[float, float]] = {
    "metal": (0.218, 0.280),
    "wood": (0.432, 0.280),
    "leather": (0.648, 0.280),
    "cloth": (0.862, 0.280),
    "medicine": (0.218, 0.580),
    "food": (0.432, 0.580),
}
# 配方是右上横排卡片。
# 配方是右栏卡片；下沿须盖住第三排（藥品汁液在底部）。
FACILITY_LIST_ROI = (0.38, 0.06, 0.60, 0.78)
PROCESS_BUTTON_ROI = (0.48, 0.48, 0.50, 0.50)
HOLD_NEED_ROI = (0.38, 0.42, 0.60, 0.50)
# 左下「全部領取」：单排槽约 y=1204/1440；五槽两行时字上移到约 0.72。
CLAIM_ALL_ROI = (0.02, 0.68, 0.30, 0.24)
# 总览「前往設備」在标题下方左侧，不在右下加工钮 ROI。
GO_EQUIP_ROI = (0.02, 0.14, 0.28, 0.18)
# 加工完成底部绿色「確認」（客户区比例）；未命中不得猜点正中。
RESULT_CONFIRM_ROI = (0.28, 0.72, 0.44, 0.24)
# 通用确认弹窗（附近没有加工设备 / 前往堤爾克那）的「確認」；比加工完成更高。
COMMON_POPUP_CONFIRM_ROI = (0.28, 0.48, 0.50, 0.40)
# 设施页左下五槽待标定（与 medicine_process.SLOT_ROI 同口径）；OCR 旁证，不成主判。
QUEUE_SLOT_ROI = (0.02, 0.86, 0.42, 0.12)
_PCT_RE = re.compile(r"(\d{1,3})\s*[%％]")
TITLE_ROI = (0.00, 0.00, 0.36, 0.09)

# 六卡文字：优先「××加工」全称，再短名；禁单字「藥」以免误撞。
HUB_OCR_NEEDLES: dict[str, tuple[str, ...]] = {
    "metal": ("金屬加工", "金属加工", "金屬", "金属"),
    "wood": ("木材加工", "木材"),
    "leather": ("皮革加工", "皮革"),
    "cloth": ("布料加工", "布料"),
    "medicine": ("藥品加工", "药品加工", "藥品", "药品"),
    "food": ("食材加工", "食材"),
}

FACILITY_TITLES: dict[str, tuple[str, ...]] = {
    "metal": ("金屬加工", "金属加工", "屬加工", "属加工"),
    "wood": ("木材加工",),
    "leather": ("皮革加工",),
    "cloth": ("布料加工",),
    "medicine": ("藥品加工", "药品加工", "品加工"),
    "food": ("食材加工",),
}

# 点选配方后详情里的行动钮是「前往加工」；禁点它投料。
PROCESS_BTN_NEEDLES = ("前往加工", "加工")
GO_EQUIP_NEEDLES = ("前往設備", "前往设备")
GRAY_NEEDLES = ("未解鎖", "未解锁", "鎖定", "锁定", "灰色", "灰")
K_ASSIST_NEEDLES = ("協助", "协助", "K協助", "K协助")

_CATALOG_CACHE: dict[str, dict[str, Any]] = {}


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else ROOT


def catalog_path(*, root: Path | None = None) -> Path:
    return _root(root) / "knowledge" / "templates" / "process" / "catalog.json"


def load_catalog(*, root: Path | None = None) -> dict[str, Any]:
    path = catalog_path(root=root)
    key = str(path.resolve()) if path.exists() else str(path)
    cached = _CATALOG_CACHE.get(key)
    if cached is not None:
        return cached
    if not path.is_file():
        data: dict[str, Any] = {"schema": "process_sense_catalog_v1", "facilities": {}, "material_reuse": {}}
        _CATALOG_CACHE[key] = data
        return data
    data = json.loads(path.read_text(encoding="utf-8"))
    _CATALOG_CACHE[key] = data
    return data


def clear_catalog_cache() -> None:
    _CATALOG_CACHE.clear()


def facility_title_roi(*, root: Path | None = None) -> tuple[float, float, float, float] | None:
    raw = (load_catalog(root=root).get("facility_titles") or {}).get("roi", TITLE_ROI)
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        if any(isinstance(v, bool) for v in raw):
            return None
        x, y, w, h = (float(v) for v in raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
        return None
    return x, y, w, h


def facility_title_paths(fid: str, *, root: Path | None = None) -> tuple[Path, ...]:
    variants = (load_catalog(root=root).get("facility_titles") or {}).get("variants") or {}
    names = [f"title_{fid}.png"] + list(variants.get(fid) or [])
    return tuple(_root(root) / "knowledge/templates/facility" / str(name) for name in names)


def _template_title_facility(frame: np.ndarray, *, root: Path | None = None, on_template=None) -> str | None:
    roi = facility_title_roi(root=root)
    if roi is None:
        return None
    scores = {}
    for fid in HUB_CARD_IDS:
        for path in facility_title_paths(fid, root=root):
            if not path.is_file():
                continue
            if on_template is not None:
                on_template()
            # A shared 加工 suffix can score >.8 on a different facility.
            # Require the full registered title; OCR remains the locator's fallback.
            hit = match_template(frame, path, roi=roi, threshold=0.94)
            if hit is not None:
                scores[fid] = max(scores.get(fid, 0.0), float(hit.score))
    ranked = sorted(scores.items(), key=lambda row: row[1], reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[0][1] - ranked[1][1] < CARD_AMBIGUOUS_DELTA):
        return None
    return ranked[0][0]


def hub_card_path(card_id: str, *, root: Path | None = None) -> Path:
    return _root(root) / "knowledge" / "templates" / "craft_hub" / f"card_{card_id}.png"


def process_btn_path(*, root: Path | None = None, variant_id: str | None = None) -> Path:
    vid = str(variant_id or "").strip() or "process_btn_short_away"
    for row in action_rows("process_btn", root=root):
        if str(row.get("id") or "") == vid:
            return _root(root) / "knowledge" / "templates" / "process" / str(row.get("file") or "")
    named = _root(root) / "knowledge" / "templates" / "process" / "process_btn_short_away.png"
    legacy = _root(root) / "knowledge" / "templates" / "process" / "process_btn.png"
    return named if named.is_file() else legacy


def claim_all_path(*, root: Path | None = None) -> Path:
    for row in action_rows("claim_all", root=root):
        return _root(root) / "knowledge" / "templates" / "process" / str(row.get("file") or "claim_all.png")
    return _root(root) / "knowledge" / "templates" / "process" / "claim_all.png"


def action_rows(kind: str, *, root: Path | None = None) -> tuple[dict[str, Any], ...]:
    cat = load_catalog(root=root)
    rows = (cat.get("actions") or {}).get(kind) or []
    return tuple(r for r in rows if isinstance(r, dict))


def recipe_template_path(
    facility: str,
    name: str,
    *,
    variant: str | None = None,
    root: Path | None = None,
) -> Path:
    rec = _find_recipe(facility, name, variant=variant, root=root)
    if rec is not None:
        rel = rec.get("file") or f"{name}.png"
        rel_dir = rec.get("rel_dir") or f"knowledge/templates/process/{facility}"
        return _root(root) / str(rel_dir) / str(rel)
    stem = _recipe_stem(name, variant)
    return _root(root) / "knowledge" / "templates" / "process" / facility / f"{stem}.png"


def _recipe_stem(name: str, variant: str | None) -> str:
    if variant:
        return f"{name}_{variant}"
    return str(name)


def recipes_for_facility(facility: str, *, root: Path | None = None) -> tuple[dict[str, Any], ...]:
    fid = str(facility or "").strip()
    cat = load_catalog(root=root)
    rows = (cat.get("facilities") or {}).get(fid) or []
    if isinstance(rows, dict):
        rows = rows.get("recipes") or []
    return tuple(dict(r) for r in rows if isinstance(r, dict))


def _find_recipe(
    facility: str,
    name: str,
    *,
    variant: str | None = None,
    root: Path | None = None,
) -> dict[str, Any] | None:
    want = compact_zh(name)
    var = compact_zh(variant or "")
    hits = [
        row
        for row in recipes_for_facility(facility, root=root)
        if compact_zh(str(row.get("name") or row.get("label") or "")) == want
    ]
    if not hits:
        return None
    if var:
        for row in hits:
            if compact_zh(str(row.get("variant") or "")) == var:
                return row
        return None
    if len(hits) == 1:
        return hits[0]
    unmarked = [row for row in hits if not compact_zh(str(row.get("variant") or ""))]
    return unmarked[0] if len(unmarked) == 1 else None


def gather_material_ref(name: str, *, root: Path | None = None) -> dict[str, Any]:
    n = compact_zh(name)
    if not n:
        return {"ok": False, "error": "missing_name", "kind": "gap"}
    cat = load_catalog(root=root)
    reuse = cat.get("material_reuse") or {}
    row = reuse.get(name) or reuse.get(n)
    if not isinstance(row, dict):
        return {"ok": False, "error": "gap", "kind": "gap", "name": name}
    kind = str(row.get("kind") or "").strip()
    if kind == "gap":
        return {
            "ok": False,
            "error": "gap",
            "kind": "gap",
            "name": name,
            "reason": row.get("reason"),
        }
    rel = str(row.get("path") or "")
    path = _root(root) / rel if rel and not Path(rel).is_absolute() else Path(rel)
    exists = path.is_file()
    return {
        "ok": bool(exists),
        "kind": "reuse",
        "name": name,
        "path": str(path),
        "exists": exists,
        "error": None if exists else "missing_template",
    }


def _bbox(hit: Any) -> dict[str, int]:
    return {"x": int(hit.x), "y": int(hit.y), "w": int(hit.w), "h": int(hit.h)}


def _ocr_lines(frame: np.ndarray, roi: tuple[float, float, float, float]) -> list[OcrLine]:
    lines = ocr_roi(frame, roi, min_confidence=0.35, backend="cht")
    if not lines:
        lines = ocr_roi(frame, roi, min_confidence=0.35, backend="v5")
    return lines


def _blob(lines: list[OcrLine]) -> str:
    return compact_zh("".join(ln.text for ln in lines))


def _needles_in(text: str, needles: tuple[str, ...]) -> bool:
    blob = compact_zh(text)
    for n in needles:
        c = compact_zh(n)
        if c and c in blob:
            return True
    return False


def _line_center(line: OcrLine) -> tuple[int, int] | None:
    if not line.box:
        return None
    xs = [float(p[0]) for p in line.box]
    ys = [float(p[1]) for p in line.box]
    return int((min(xs) + max(xs)) / 2), int((min(ys) + max(ys)) / 2)


def _empty_frame(frame: np.ndarray | None) -> bool:
    return frame is None or getattr(frame, "size", 0) == 0


def _is_half_window(frame: np.ndarray) -> bool:
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if w < 200 or h < 200:
        return True
    return (w / max(h, 1)) < 0.35


def _match_card(
    frame: np.ndarray,
    card_id: str,
    *,
    root: Path | None = None,
) -> Any | None:
    path = hub_card_path(card_id, root=root)
    if not path.is_file():
        return None
    return match_template(
        frame,
        path,
        roi=HUB_CARDS_ROI,
        threshold=TEMPLATE_THRESHOLD,
        min_margin=0.02,
    )


def locate_hub_card(
    frame: np.ndarray,
    card_id: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    cid = str(card_id or "").strip()
    if _empty_frame(frame):
        return {"ok": False, "found": False, "card_id": cid, "error": "empty_frame"}
    if cid not in HUB_CARD_IDS:
        return {"ok": False, "found": False, "card_id": cid, "error": "unknown_card"}

    scores: dict[str, Any] = {}
    for other in HUB_CARD_IDS:
        hit = _match_card(frame, other, root=root)
        if hit is not None:
            scores[other] = hit

    path = hub_card_path(cid, root=root)
    hit = scores.get(cid)
    ambiguous = False
    if hit is not None:
        others = [(oid, h) for oid, h in scores.items() if oid != cid]
        reject = False
        if others:
            best_other = max(others, key=lambda kv: float(kv[1].score))
            margin = float(hit.score) - float(best_other[1].score)
            close = abs(float(hit.score) - float(best_other[1].score)) <= CARD_AMBIGUOUS_DELTA
            other_hit = best_other[1]
            dist = (
                (float(hit.cx) - float(other_hit.cx)) ** 2
                + (float(hit.cy) - float(other_hit.cy)) ** 2
            ) ** 0.5
            # 六卡会同时高分；只有峰值叠在同一点才算撞车。
            same_spot = dist < 96.0
            if same_spot and close and margin < CARD_MARGIN_MIN:
                reject = True
            if same_spot and float(best_other[1].score) >= float(hit.score) and margin < CARD_MARGIN_MIN:
                reject = True
        if not reject:
            return {
                "ok": True,
                "src": "template",
                "found": True,
                "peak": float(hit.score),
                "bbox": _bbox(hit),
                "click_client": [int(hit.cx), int(hit.cy)],
                "card_id": cid,
            }
        ambiguous = True

    lines = _ocr_lines(frame, HUB_CARDS_ROI)
    for line in lines:
        if _hub_ocr_card_id(line.text) != cid:
            continue
        center = _line_center(line)
        if center is None:
            continue
        x0 = min(int(p[0]) for p in line.box)
        y0 = min(int(p[1]) for p in line.box)
        x1 = max(int(p[0]) for p in line.box)
        y1 = max(int(p[1]) for p in line.box)
        return {
            "ok": True,
            "src": "ocr_cht",
            "found": True,
            "peak": float(line.confidence),
            "bbox": {"x": x0, "y": y0, "w": max(1, x1 - x0), "h": max(1, y1 - y0)},
            # Use the observed caption itself. A height-based offset can land
            # on the facility model instead of the card's text button.
            "click_client": [center[0], center[1]],
            "card_id": cid,
        }

    if (not ambiguous) and cid not in scores:
        frac = HUB_CARD_CLICK.get(cid)
        hubish = len(scores) >= 2
        if not hubish:
            title = compact_zh(_blob(_ocr_lines(frame, TITLE_ROI)))
            hubish = bool(title) and _title_facility(title) is None and (
                "加工" in title or "製作" in title or "制作" in title
            )
        if frac is not None and hubish:
            h, w = int(frame.shape[0]), int(frame.shape[1])
            return {
                "ok": True,
                "src": "hub_grid",
                "found": True,
                "peak": 0.0,
                "click_client": [int(w * frac[0]), int(h * frac[1])],
                "card_id": cid,
            }

    err = "ambiguous_card" if ambiguous else ("missing_template" if not path.is_file() else "not_found")
    out = {"ok": False, "found": False, "card_id": cid, "error": err, "path": str(path)}
    if hit is not None:
        out["peak"] = float(hit.score)
        out["src"] = "template"
    return out


def _hub_ocr_card_id(text: str) -> str | None:
    """六卡 OCR 最长针命中；全称优先于短名，互撞则不定。"""
    blob = compact_zh(text)
    if not blob:
        return None
    best_id: str | None = None
    best_len = 0
    tied = False
    for hid, needles in HUB_OCR_NEEDLES.items():
        for needle in needles:
            c = compact_zh(needle)
            if not c or c not in blob:
                continue
            nlen = len(c)
            if nlen > best_len:
                best_len = nlen
                best_id = hid
                tied = False
            elif nlen == best_len and hid != best_id:
                tied = True
    if tied or best_len <= 0:
        return None
    return best_id


def _title_facility(blob: str) -> str | None:
    compact = compact_zh(blob)
    if not compact:
        return None
    best_id: str | None = None
    best_len = 0
    tied = False
    for fid, titles in FACILITY_TITLES.items():
        for title in titles:
            c = compact_zh(title)
            if not c or c not in compact:
                continue
            nlen = len(c)
            if nlen > best_len:
                best_len = nlen
                best_id = fid
                tied = False
            elif nlen == best_len and fid != best_id:
                tied = True
    if tied or best_len <= 0:
        return None
    return best_id


def _any_hub_card(frame: np.ndarray, *, root: Path | None = None) -> bool:
    n_tmpl = sum(1 for cid in HUB_CARD_IDS if _match_card(frame, cid, root=root) is not None)
    if n_tmpl >= 2:
        return True
    found: set[str] = set()
    for line in _ocr_lines(frame, HUB_CARDS_ROI):
        cid = _hub_ocr_card_id(line.text)
        if cid in HUB_CARD_IDS:
            found.add(cid)
        if len(found) >= 2:
            return True
    return False


_FACILITY_TITLE_NEEDLES = (
    "金屬加工",
    "金属加工",
    "木材加工",
    "皮革加工",
    "布料加工",
    "藥品加工",
    "药品加工",
    "食材加工",
)


def _process_action_kind(blob: str) -> str | None:
    """从一行或一小段 OCR 判断行动钮：away=前往加工，here=加工。"""
    bb = compact_zh(blob)
    if not bb:
        return None
    if compact_zh("全部領取") in bb or compact_zh("全部领取") in bb:
        return None
    if compact_zh("前往加工") in bb:
        return "away"
    rest = bb
    for needle in ("加工時間", "加工时间") + _FACILITY_TITLE_NEEDLES:
        rest = rest.replace(compact_zh(needle), "")
    if compact_zh("加工") in rest:
        return "here"
    return None


def infer_materials_ok(blob: str) -> bool | None:
    """从明细 OCR 猜材料是否凑齐。解析失败返回 None。"""
    import re

    text = str(blob or "")
    if compact_zh("所需材料") not in compact_zh(text):
        return None
    if re.search(r"(?<!\d)0\s*/", text) or "０/" in text:
        return False
    # 16/316/4 → 16/3 16/4（三位数中段=一位需求+两位持有）
    text = re.sub(r"(\d+)/(\d)(\d{2})/(\d+)", r"\1/\2 \3/\4", text)
    # 粘连 24/1020/1 → 24/10 20/1
    text = re.sub(r"(\d+)/(\d{2})(\d{2,})/(\d+)", r"\1/\2 \3/\4", text)
    # 37/30 11/3 粘成 37/30113
    text = re.sub(r"(\d+)/(\d{2})(\d{2})(\d{1,2})(?!\d)", r"\1/\2 \3/\4", text)
    pairs = re.findall(r"(\d+)\s*/\s*(\d+)", text)
    if not pairs:
        return None
    for a, b in pairs:
        ia, ib = int(a), int(b)
        if ib > 0 and ia < ib:
            return False
    return True


def classify_process_btn_state(btn_blob: str, materials_blob: str) -> dict[str, Any]:
    """四态：材料齐/缺 × 是否在设施前。前往加工=不在设施前；加工=在设施前。

    明细里的「加工時間 / ××加工設備」不得把「加工」钮判成未知。
    """
    kind = _process_action_kind(btn_blob)
    mats_ok = infer_materials_ok(materials_blob)
    variant_id = None
    at: bool | None = None
    if kind == "away":
        at = False
        variant_id = "process_btn_full_away" if mats_ok is True else "process_btn_short_away"
    elif kind == "here":
        at = True
        variant_id = "process_btn_full_here" if mats_ok is True else "process_btn_short_here"
    return {
        "at_facility": at,
        "materials_ok": mats_ok,
        "variant_id": variant_id,
        "label": "前往加工" if kind == "away" else ("加工" if kind == "here" else None),
    }


def _pick_process_action_line(lines: list[Any]) -> Any | None:
    go = None
    short = None
    for line in lines:
        kind = _process_action_kind(getattr(line, "text", "") or "")
        if kind == "away":
            go = line
            break
        if kind == "here":
            short = short or line
    return go or short


def _btn_text_from_template_path(path: str | None) -> str:
    name = Path(str(path or "")).stem
    if name.endswith("_here"):
        return "加工"
    if name.endswith("_away") or name == "process_btn":
        return "前往加工"
    return ""


def _process_btn_stance_from_path(path: str | Path | None) -> str | None:
    name = Path(str(path or "")).stem
    if name.endswith("_here"):
        return "here"
    if name.endswith("_away") or name == "process_btn":
        return "away"
    return None


def _pick_template_process_hit(
    hits: list[tuple[Any, Path]],
) -> tuple[Any, Path] | None:
    """可见「前往加工」时不用「加工」子串模板。away 过线则取最高分 away。"""
    if not hits:
        return None
    away = [(hit, path) for hit, path in hits if _process_btn_stance_from_path(path) == "away"]
    pool = away or hits
    return max(pool, key=lambda item: float(item[0].score))


@lru_cache(maxsize=32)
def _process_label_channel(path: str) -> np.ndarray | None:
    import cv2

    image = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    # White/grey letters have signal in every channel; the animated cyan/green
    # background does not. Preserve antialiasing rather than threshold pixels.
    return np.min(image,axis=2) if image is not None else None


def _match_complete_away_label(frame: np.ndarray, paths: list[Path]) -> tuple[Any,Path] | None:
    """Check the full phrase before accepting its shorter 加工 suffix.

    Recorded opening frames change the label size by about 5%. Match that small
    range with the existing score/margin gates; never infer away from OCR.
    """
    import cv2

    if frame.ndim != 3:
        return None
    crop,ox,oy = roi_slice(frame,PROCESS_BUTTON_ROI)
    crop = np.min(crop,axis=2)
    matches = []
    for path in paths:
        if not path.stem.endswith('_away'):
            continue
        label = _process_label_channel(str(path.resolve()))
        if label is None or float(np.std(label)) < 5:
            continue
        for scale in (.95,1.,1.05):
            template = cv2.resize(label,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
            h,w = template.shape
            if h > crop.shape[0] or w > crop.shape[1]:
                continue
            scores = cv2.matchTemplate(crop,template,cv2.TM_CCOEFF_NORMED)
            score,point,margin = _peak_and_margin(scores,w,h,True)
            if score >= TEMPLATE_THRESHOLD and margin >= .02:
                x,y = point[0]+ox,point[1]+oy
                matches.append((MatchHit(score,x+w//2,y+h//2,x,y,w,h,margin),path))
    return _pick_template_process_hit(matches)


def _decorate_process_btn(
    frame: np.ndarray,
    out: dict[str, Any],
    *,
    btn_text: str | None = None,
) -> dict[str, Any]:
    mats = _blob(_ocr_lines(frame, HOLD_NEED_ROI))
    text = str(btn_text or "").strip()
    if not text:
        picked = _pick_process_action_line(_ocr_lines(frame, PROCESS_BUTTON_ROI))
        text = str(getattr(picked, "text", "") or "")
    if not text:
        text = _btn_text_from_template_path(str(out.get("path") or ""))
    out.update(classify_process_btn_state(text, mats))
    stem = Path(str(out.get("path") or "")).stem
    if _process_btn_stance_from_path(stem) is not None:
        out["variant_id"] = stem
    out["materials_ocr"] = mats
    return out


def locate_process_button(
    frame: np.ndarray,
    *,
    root: Path | None = None,
    prefer_ocr: bool = False,
) -> dict[str, Any]:
    base = {"ok": False, "found": False, "name": "加工"}
    if _empty_frame(frame):
        return {**base, "error": "empty_frame"}
    paths: list[Path] = []
    for row in action_rows("process_btn", root=root):
        p = process_btn_path(root=root, variant_id=str(row.get("id") or ""))
        if p.is_file() and p not in paths:
            paths.append(p)
    legacy = _root(root) / "knowledge" / "templates" / "process" / "process_btn.png"
    if legacy.is_file() and legacy not in paths:
        paths.append(legacy)
    if paths and not prefer_ocr:
        scored: list[tuple[Any, Path]] = []
        for path in paths:
            hit = match_template(
                frame,
                path,
                roi=PROCESS_BUTTON_ROI,
                threshold=TEMPLATE_THRESHOLD,
                min_margin=0.02,
            )
            if hit is None:
                continue
            scored.append((hit, path))
        picked = _pick_template_process_hit(scored)
        if picked is not None and _process_btn_stance_from_path(picked[1]) == 'here':
            # Gradient/scale animation can defeat the whole-label grey match
            # while the shorter suffix still matches. Require full-label evidence.
            picked = _match_complete_away_label(frame,paths) or picked
        if picked is not None:
            best_hit, best_path = picked
            return _decorate_process_btn(
                frame,
                {
                    **base,
                    "ok": True,
                    "found": True,
                    "src": "template",
                    "peak": float(best_hit.score),
                    "bbox": _bbox(best_hit),
                    "click_client": [int(best_hit.cx), int(best_hit.cy)],
                    "path": str(best_path),
                },
                btn_text=_btn_text_from_template_path(best_path),
            )
    path = process_btn_path(root=root)
    lines = _ocr_lines(frame, PROCESS_BUTTON_ROI)

    def _line_hit(line: Any) -> dict[str, Any] | None:
        center = _line_center(line)
        if center is None or not line.box:
            return None
        x0 = min(int(p[0]) for p in line.box)
        y0 = min(int(p[1]) for p in line.box)
        x1 = max(int(p[0]) for p in line.box)
        y1 = max(int(p[1]) for p in line.box)
        return {
            **base,
            "ok": True,
            "found": True,
            "src": "ocr_cht",
            "peak": float(line.confidence),
            "bbox": {"x": x0, "y": y0, "w": max(1, x1 - x0), "h": max(1, y1 - y0)},
            "click_client": [center[0], center[1]],
        }

    hit_line = _pick_process_action_line(lines)
    if hit_line is not None:
        out = _line_hit(hit_line)
        if out is not None:
            return _decorate_process_btn(frame, out, btn_text=str(hit_line.text or ""))
    err = "missing_template" if not path.is_file() else "not_found"
    return {**base, "error": err, "path": str(path)}


def _memory_facility_button(frame, memory_view, kind, facility):
    if memory_view is None:
        return None
    base = {'ok': False, 'found': False, 'name': '全部領取' if kind == 'claim_all' else '前往設備', 'src': 'memory'}
    try:
        if _empty_frame(frame):
            return {**base, 'error': 'empty_frame'}
        if facility and memory_view['facility'] != facility:
            return {**base, 'error': 'wrong_facility'}
        button = memory_view['buttons'][kind]
        if not button['present'] or not button['usable']:
            return {**base, 'error': 'not_available'}
        sh, sw = frame.shape[:2]
        vw, vh = memory_view['screen_size']
        if min(vw, vh) <= 0 or abs((sw/sh)/(vw/vh)-1) > 0.01:
            return {**base, 'error': 'client_mismatch'}
        x, y, w, h = button['rect']
        x, y, w, h = x*sw/vw, y*sh/vh, w*sw/vw, h*sh/vh
        if min(w,h) < 4 or min(x,y) < 0 or x+w > sw+1 or y+h > sh+1:
            return {**base, 'error': 'invalid_rect'}
        return {**base, 'ok': True, 'found': True, 'peak': 1.0,
                'bbox': {'x': round(x), 'y': round(y), 'w': round(w), 'h': round(h)},
                'click_client': [round(x+w/2), round(y+h/2)], 'facility': memory_view['facility']}
    except (KeyError,TypeError,ValueError,ZeroDivisionError):
        return {**base, 'error': 'invalid_memory_view'}


def locate_claim_all(
    frame: np.ndarray,
    *,
    root: Path | None = None,
    prefer_ocr: bool = False,
    memory_view: dict[str, Any] | None = None,
    facility: str = '',
) -> dict[str, Any]:
    """全部領取：在/不在设施前 UI 无差。可见 ≠ 已领到。"""
    memory = _memory_facility_button(frame, memory_view, 'claim_all', facility)
    if memory is not None:
        return memory
    base = {
        "ok": False,
        "found": False,
        "name": "全部領取",
        "at_facility_ui_same": True,
    }
    if _empty_frame(frame):
        return {**base, "error": "empty_frame"}
    path = claim_all_path(root=root)
    if path.is_file() and not prefer_ocr:
        hit = match_template(
            frame,
            path,
            roi=CLAIM_ALL_ROI,
            threshold=TEMPLATE_THRESHOLD,
            min_margin=0.02,
        )
        if hit is not None:
            return {
                **base,
                "ok": True,
                "found": True,
                "src": "template",
                "peak": float(hit.score),
                "bbox": _bbox(hit),
                "click_client": [int(hit.cx), int(hit.cy)],
            }
    for line in _ocr_lines(frame, CLAIM_ALL_ROI):
        blob = compact_zh(line.text)
        if compact_zh("全部領取") not in blob and compact_zh("全部领取") not in blob:
            continue
        center = _line_center(line)
        if center is None or not line.box:
            continue
        x0 = min(int(p[0]) for p in line.box)
        y0 = min(int(p[1]) for p in line.box)
        x1 = max(int(p[0]) for p in line.box)
        y1 = max(int(p[1]) for p in line.box)
        return {
            **base,
            "ok": True,
            "found": True,
            "src": "ocr_cht",
            "peak": float(line.confidence),
            "bbox": {"x": x0, "y": y0, "w": max(1, x1 - x0), "h": max(1, y1 - y0)},
            "click_client": [center[0], center[1]],
        }
    err = "missing_template" if not path.is_file() else "not_found"
    return {**base, "error": err, "path": str(path)}


def locate_go_equip(
    frame: np.ndarray,
    *,
    root: Path | None = None,
    memory_view: dict[str, Any] | None = None,
    facility: str = '',
) -> dict[str, Any]:
    """人不在设施前时标题下「前往設備」。可见才点，不盲点。"""
    memory = _memory_facility_button(frame, memory_view, 'go_equip', facility)
    if memory is not None:
        return memory
    _ = root
    base = {
        "ok": False,
        "found": False,
        "name": "前往設備",
    }
    if _empty_frame(frame):
        return {**base, "error": "empty_frame"}
    for roi in (GO_EQUIP_ROI, PROCESS_BUTTON_ROI):
        for line in _ocr_lines(frame, roi):
            blob = compact_zh(line.text)
            if not _needles_in(blob, GO_EQUIP_NEEDLES):
                continue
            center = _line_center(line)
            if center is None or not line.box:
                continue
            x0 = min(int(p[0]) for p in line.box)
            y0 = min(int(p[1]) for p in line.box)
            x1 = max(int(p[0]) for p in line.box)
            y1 = max(int(p[1]) for p in line.box)
            return {
                **base,
                "ok": True,
                "found": True,
                "src": "ocr_cht",
                "peak": float(line.confidence),
                "bbox": {"x": x0, "y": y0, "w": max(1, x1 - x0), "h": max(1, y1 - y0)},
                "click_client": [center[0], center[1]],
            }
    return {**base, "error": "not_found"}


def _locate_confirm_in_roi(
    frame: np.ndarray,
    roi: tuple[float, float, float, float],
) -> dict[str, Any]:
    base = {
        "ok": False,
        "found": False,
        "name": "確認",
    }
    if _empty_frame(frame):
        return {**base, "error": "empty_frame"}
    for line in _ocr_lines(frame, roi):
        blob = compact_zh(line.text)
        if compact_zh("確認") not in blob and compact_zh("确认") not in blob:
            continue
        center = _line_center(line)
        if center is None or not line.box:
            continue
        x0 = min(int(p[0]) for p in line.box)
        y0 = min(int(p[1]) for p in line.box)
        x1 = max(int(p[0]) for p in line.box)
        y1 = max(int(p[1]) for p in line.box)
        return {
            **base,
            "ok": True,
            "found": True,
            "src": "ocr_cht",
            "peak": float(line.confidence),
            "bbox": {"x": x0, "y": y0, "w": max(1, x1 - x0), "h": max(1, y1 - y0)},
            "click_client": [center[0], center[1]],
        }
    return {**base, "error": "not_found"}


def locate_result_confirm(
    frame: np.ndarray,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """加工完成底部「確認」。OCR 定位，未命中 fail-closed。"""
    _ = root
    return _locate_confirm_in_roi(frame, RESULT_CONFIRM_ROI)


def locate_common_popup_confirm(
    frame: np.ndarray,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """通用确认弹窗的「確認」。先弹窗 ROI，再退回加工完成底部 ROI。"""
    _ = root
    hit = _locate_confirm_in_roi(frame, COMMON_POPUP_CONFIRM_ROI)
    if hit.get("ok"):
        return hit
    return _locate_confirm_in_roi(frame, RESULT_CONFIRM_ROI)


def _process_button_ok(frame: np.ndarray, *, root: Path | None = None) -> bool:
    return bool(locate_process_button(frame, root=root).get("ok"))


def _list_evidence(frame: np.ndarray) -> bool:
    lines = _ocr_lines(frame, FACILITY_LIST_ROI)
    blob = _blob(lines)
    return len(blob) >= 2


def _overview_evidence(frame: np.ndarray, *, root: Path | None = None) -> bool:
    """人不在设施前的总览：标题下「前往設備」，或左下「全部領取」。不是配方列表。"""
    blob = _blob(_ocr_lines(frame, PROCESS_BUTTON_ROI))
    if _needles_in(blob, GO_EQUIP_NEEDLES):
        return True
    blob_go = _blob(_ocr_lines(frame, GO_EQUIP_ROI))
    if _needles_in(blob_go, GO_EQUIP_NEEDLES):
        return True
    return bool(locate_claim_all(frame, root=root).get("ok"))


def detect_facility_page(
    frame: np.ndarray,
    *,
    expected: str | None = None,
    root: Path | None = None,
    memory_view: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _empty_frame(frame):
        return {"ok": False, "error": "empty_frame"}
    if _is_half_window(frame):
        return {"ok": False, "error": "half_window"}

    if memory_view is not None:
        # The executor binds this rechecked native fact to this exact frame.
        # An explicit invalid fact must not fall back to a conflicting title.
        try:
            fid = memory_view['facility']
            identity = memory_view['identity']
            vw, vh = memory_view['screen_size']
            sh, sw = frame.shape[:2]
            if (memory_view.get('error') or fid not in FACILITY_TITLES
                    or len(identity) != 5 or any(type(v) is not int or v <= 0 for v in identity)
                    or memory_view['window'] != identity[4]
                    or list(memory_view['view_root']) != list(identity[2:4])
                    or any(not math.isfinite(v) or not 100 <= v <= 16384 for v in (vw,vh))
                    or abs((sw/sh)/(vw/vh)-1) > 0.01):
                return {'ok': False, 'error': 'invalid_memory_view', 'src': 'memory'}
            if expected and fid != expected:
                return {'ok': False, 'error': 'wrong_facility', 'facility': fid, 'expected': expected, 'src': 'memory'}
            return {'ok': True, 'facility': fid, 'src': 'memory'}
        except (KeyError,TypeError,ValueError,ZeroDivisionError):
            return {'ok': False, 'error': 'invalid_memory_view', 'src': 'memory'}

    roi = facility_title_roi(root=root)
    if roi is None:
        return {"ok": False, "error": "invalid_title_roi"}
    fid = _template_title_facility(frame, root=root)
    src = "template_title" if fid else "ocr_title"
    title_blob = "" if fid else _blob(_ocr_lines(frame, roi))
    if fid is None:
        fid = _title_facility(title_blob)
    want = str(expected or "").strip() or None

    if fid is None:
        if _needles_in(title_blob, K_ASSIST_NEEDLES):
            return {"ok": False, "error": "k_assist_or_hud", "title": title_blob}
        if _any_hub_card(frame, root=root):
            return {"ok": False, "error": "unknown_page", "title": title_blob}
        if title_blob:
            return {"ok": False, "error": "unknown_page", "title": title_blob}
        return {"ok": False, "error": "k_assist_or_hud"}

    if want and fid != want:
        return {"ok": False, "error": "wrong_facility", "facility": fid, "expected": want}

    if not (
        _process_button_ok(frame, root=root)
        or _overview_evidence(frame, root=root)
        or _list_evidence(frame)
    ):
        return {"ok": False, "error": "missing_list_or_process", "facility": fid}

    return {"ok": True, "facility": fid, "src": src}


def _iron_need_variant(name: str, variant: str | None) -> bool:
    return compact_zh(name) == "鐵錠" and not compact_zh(variant or "")


def recipe_click_variant(name: str) -> str | None:
    """投料点配方：鐵錠必须消歧到鐵礦石档。"""
    if compact_zh(name) == "鐵錠":
        return "iron_ore"
    return None


def _ocr_needles_for_recipe(row: dict[str, Any] | None, name: str, variant: str | None) -> tuple[str, ...]:
    if row and row.get("ocr_needles"):
        return tuple(str(x) for x in row["ocr_needles"])
    base = tuple(item_name_variants(name)) + (name,)
    var = compact_zh(variant or "")
    if compact_zh(name) == "鐵錠":
        if var == "iron_ore":
            return ("鐵錠(鐵礦石)", "鐵錠(铁矿石)", "鐵礦石", "铁矿石")
        return ()
    if var:
        return base + (var,)
    return base


def _recipe_name_catalog(facility: str, *, root: Path | None = None) -> tuple[str, ...]:
    names: list[str] = []
    for row in recipes_for_facility(facility, root=root):
        n = str(row.get("name") or "").strip()
        if n:
            names.append(n)
    return tuple(dict.fromkeys(names))


def _ocr_line_matches_recipe(
    text: str,
    name: str,
    variant: str | None,
    needles: tuple[str, ...],
    *,
    facility: str,
    root: Path | None = None,
) -> bool:
    blob = compact_zh(text)
    if compact_zh(name) == "鐵錠":
        var = compact_zh(variant or "")
        if var == "iron_ore":
            if _needles_in(blob, ("鐵錠(礦石)", "鐵錠(矿石)")) and not _needles_in(
                blob, ("鐵礦石", "铁矿石", "鐵錠(鐵礦石)", "鐵錠(铁矿石)")
            ):
                return False
            return _needles_in(blob, ("鐵礦石", "铁矿石", "鐵錠(鐵礦石)", "鐵錠(铁矿石)"))
        return False
    long_needles = tuple(n for n in needles if len(compact_zh(n)) >= 5)
    if long_needles and _needles_in(blob, long_needles):
        return True
    catalog = _recipe_name_catalog(facility, root=root)
    if catalog:
        return line_matches_name(text, name, catalog)
    return _needles_in(blob, needles) or any(v in blob for v in item_name_variants(name))


def _gray_from_ocr(frame: np.ndarray, bbox: dict[str, int] | None) -> bool:
    lines = _ocr_lines(frame, FACILITY_LIST_ROI)
    extra = _ocr_lines(frame, HOLD_NEED_ROI)
    blob = _blob(lines) + _blob(extra)
    return _needles_in(blob, GRAY_NEEDLES)


def locate_process_recipe(
    frame: np.ndarray,
    *,
    facility: str,
    name: str,
    variant: str | None = None,
    root: Path | None = None,
    prefer_ocr: bool = False,
    memory_view: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fid = str(facility or "").strip()
    item = str(name or "").strip()
    var = str(variant).strip() if variant else None
    base = {
        "facility": fid,
        "name": item,
        "variant": var,
        "found": False,
        "ok": False,
    }
    if _empty_frame(frame):
        return {**base, "error": "empty_frame"}
    if fid not in RECIPE_FACILITIES:
        return {**base, "error": "unsupported_facility"}
    if not item:
        return {**base, "error": "missing_name"}
    if _iron_need_variant(item, var):
        return {**base, "error": "need_variant"}

    page = detect_facility_page(frame, expected=fid, root=root,
        **({'memory_view': memory_view} if memory_view is not None else {}))
    if not page.get("ok"):
        return {**base, "error": page.get("error") or "unknown_page", "page": page}

    row = _find_recipe(fid, item, variant=var, root=root)
    if row is None:
        return {**base, "error": "unknown_recipe"}
    path = recipe_template_path(fid, item, variant=var, root=root)
    needles = _ocr_needles_for_recipe(row, item, var)

    if path.is_file() and not prefer_ocr:
        try:
            recipe_threshold = float(row.get('threshold', TEMPLATE_THRESHOLD))
        except (TypeError, ValueError):
            return {**base, 'error': 'invalid_template_threshold'}
        if not math.isfinite(recipe_threshold) or recipe_threshold > 1.0:
            return {**base, 'error': 'invalid_template_threshold'}
        threshold = max(TEMPLATE_THRESHOLD, recipe_threshold)
        # Retain the original viewport; registered full-label variants use the
        # same page proof, threshold and ambiguity margin without rescaling.
        paths = [path] + [path.parent / filename for filename in row.get('template_variants', [])]
        hit = None
        for candidate in paths:
            if not candidate.is_file():
                continue
            hit = match_template(frame, candidate, roi=FACILITY_LIST_ROI,
                                 threshold=threshold, min_margin=0.02)
            if hit is not None:
                break
        if hit is not None:
            bbox = _bbox(hit)
            out = {
                **base,
                "ok": True,
                "found": True,
                "src": "template",
                "peak": float(hit.score),
                "bbox": bbox,
                "click_client": [int(hit.cx), int(hit.cy)],
            }
            if _gray_from_ocr(frame, bbox):
                out["gray"] = True
            return out

    lines = _ocr_lines(frame, FACILITY_LIST_ROI)
    for line in lines:
        if not _ocr_line_matches_recipe(
            line.text, item, var, needles, facility=fid, root=root
        ):
            continue
        center = _line_center(line)
        if center is None:
            continue
        x0 = min(int(p[0]) for p in line.box)
        y0 = min(int(p[1]) for p in line.box)
        x1 = max(int(p[0]) for p in line.box)
        y1 = max(int(p[1]) for p in line.box)
        bbox = {"x": x0, "y": y0, "w": max(1, x1 - x0), "h": max(1, y1 - y0)}
        out = {
            **base,
            "ok": True,
            "found": True,
            "src": "ocr_cht",
            "peak": float(line.confidence),
            "bbox": bbox,
            "click_client": [center[0], center[1]],
        }
        if _needles_in(line.text, GRAY_NEEDLES) or _gray_from_ocr(frame, bbox):
            out["gray"] = True
        return out

    err = "missing_template" if not path.is_file() else "not_found"
    return {**base, "error": err, "path": str(path)}


def parse_queue_percents(text: str) -> list[int]:
    """从 OCR 粘连文案抽出 0–100 的槽位百分比。开菜单旁证，不成败主判。"""
    blob = str(text or "").replace("％", "%")
    out: list[int] = []
    for m in _PCT_RE.finditer(blob):
        v = int(m.group(1))
        if 0 <= v <= 100:
            out.append(v)
        if len(out) >= 5:
            break
    return out


def ocr_queue_percents(
    frame: Any, *, roi: tuple[float, float, float, float] = QUEUE_SLOT_ROI
) -> list[int]:
    """设施页左下槽 OCR 百分比。关菜单无字 → 空列表；进度主源走内存。"""
    if frame is None:
        return []
    lines = ocr_roi(frame, roi, min_confidence=0.35, backend="cht")
    if not lines:
        lines = ocr_roi(frame, roi, min_confidence=0.35, backend="v5")
    blob = "".join(ln.text for ln in lines)
    return parse_queue_percents(blob)
