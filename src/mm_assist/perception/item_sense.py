"""技能详情右栏：详情物图模板优先，失败则繁体 OCR 认图标右侧文字。

只读，不发送键鼠。纯图命中不算可点；须图+字或 OCR 文字行。
"""

from __future__ import annotations

import json
import hashlib
import weakref
from pathlib import Path
from typing import Any

import numpy as np

from mm_assist.config import ROOT
from mm_assist.perception.ocr import OcrLine, ocr_image, ocr_roi
from mm_assist.platform.match import MatchHit, match_template

# 技能详情弹窗内 tiers 列表（仅右栏物名区，勿含左侧九门格；含末行）
ITEM_LIST_ROI = (0.56, 0.26, 0.40, 0.74)
# 列表顶/底留边：半截下一行仍可能落在 ROI 内，禁止点
ITEM_LIST_EDGE_INSET = 0.045
# 详情页底部动作区（尋找附近、问号）；y0 须 ≤0.40，否则弹窗按钮在 0.53 附近会被裁掉
GATHER_ACTION_ROI = (0.0, 0.38, 1.0, 0.60)
TEMPLATE_THRESHOLD = 0.76
ACTION_TEMPLATE_THRESHOLD = 0.78
MAX_SCROLL_STEPS = 8
ITEM_LIST_ROWS_VISIBLE = 5

GATHER_SKILL_IDS = (
    "woodcut",
    "mine",
    "herb",
    "shear",
    "harvest",
    "hoe",
    "insect",
    "fish",
    "daily",
)

FIND_NEARBY_REL = "knowledge/templates/gather_detail/find_nearby.png"
HELP_REL = "knowledge/templates/gather_detail/help.png"
FIND_NEARBY_OCR_NEEDLES = ("尋找附近", "寻找附近", "附近位置", "尋找附近位置")
HELP_OCR_NEEDLES = ("?", "？", "說明", "说明", "幫助", "帮助")


def compact_zh(text: str) -> str:
    return str(text or "").replace(" ", "").replace("　", "").strip()


_ZH_FOLD = str.maketrans(
    {
        "净": "淨",
        "凈": "淨",
        "铁": "鐵",
        "锭": "錠",
        "壮": "壯",
        "网": "網",
        "矿": "礦",
        "药": "藥",
        "采": "採",
        "胄": "冑",
        "铜": "銅",
        "脉": "脈",
        "树": "樹",
    }
)


def fold_zh(text: str) -> str:
    """繁简/OCR 近形折成同一串，便于全名比对。"""
    return compact_zh(text).translate(_ZH_FOLD)


def item_name_variants(name: str) -> tuple[str, ...]:
    """台服繁简别名（如 粗壮/粗壯）。"""
    base = compact_zh(name)
    if not base:
        return ()
    variants = [base]
    pairs = (
        ("壮", "壯"),
        ("网", "網"),
        ("矿", "礦"),
        ("药", "藥"),
        ("采", "採"),
        ("胄", "冑"),
        ("净", "淨"),
        ("凈", "淨"),  # OCR 常把「淨」认成「凈」
        ("铜", "銅"),
        ("脉", "脈"),
    )
    for a, b in pairs:
        if a in base:
            variants.append(base.replace(a, b))
        if b in base:
            variants.append(base.replace(b, a))
    return tuple(dict.fromkeys(variants))


def names_matching_line(text: str, catalog: tuple[str, ...]) -> list[str]:
    """OCR 行内命中的 catalog 名（子串；含繁简别名）。"""
    blob = fold_zh(text)
    if not blob:
        return []
    hits: list[str] = []
    for name in catalog:
        n = fold_zh(name)
        if n and n in blob:
            hits.append(name)
    return hits


def line_matches_name(
    text: str,
    name: str,
    catalog: tuple[str, ...] | None = None,
) -> bool:
    """严格消歧：短名不得撞长名（如 冰 vs 冰冷的光群）。"""
    needle = fold_zh(name)
    blob = fold_zh(text)
    if not needle or needle not in blob:
        return False
    if catalog:
        hits = names_matching_line(text, catalog)
        if not any(fold_zh(h) == needle for h in hits):
            return False
        longest = max(hits, key=lambda x: len(fold_zh(x)))
        if needle != fold_zh(longest):
            return False
        if blob == needle:
            return True
        if len(needle) < 2:
            return False
        if blob.startswith(needle):
            return True
        if blob.endswith(needle):
            prefix = blob[: -len(needle)]
            return prefix.isdigit()
        return False
    if blob == needle or any(blob == v for v in item_name_variants(name)):
        return True
    if len(needle) >= 2 and blob.startswith(needle):
        return True
    return False


def ocr_name_matches(text: str, name: str, *, catalog: tuple[str, ...] | None = None) -> bool:
    return line_matches_name(text, name, catalog)


def line_center(line: OcrLine) -> tuple[int, int] | None:
    if not getattr(line, "box", None):
        return None
    xs = [float(p[0]) for p in line.box]
    ys = [float(p[1]) for p in line.box]
    return int((min(xs) + max(xs)) / 2), int((min(ys) + max(ys)) / 2)


def _list_inner_bounds(frame: np.ndarray) -> tuple[float, float, float, float]:
    h, w = int(frame.shape[0]), int(frame.shape[1])
    rx, ry, rw, rh = ITEM_LIST_ROI
    inset = float(ITEM_LIST_EDGE_INSET) * h
    x0 = w * rx
    y0 = h * ry + inset
    x1 = w * (rx + rw)
    y1 = h * (ry + rh) - inset
    if y1 <= y0:
        y0 = h * ry
        y1 = h * (ry + rh)
    return x0, y0, x1, y1


def box_fully_in_item_list(
    frame: np.ndarray,
    y0: float,
    y1: float,
    *,
    x0: float | None = None,
    x1: float | None = None,
) -> bool:
    """完整行：上下（及可选左右）都在列表 ROI 内带内，半截下一行不算可见。"""
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    ix0, iy0, ix1, iy1 = _list_inner_bounds(frame)
    top, bot = float(min(y0, y1)), float(max(y0, y1))
    if top < iy0 or bot > iy1 or bot <= top:
        return False
    if x0 is None or x1 is None:
        return True
    left, right = float(min(x0, x1)), float(max(x0, x1))
    return left >= ix0 and right <= ix1


def icon_fully_in_item_list(frame: np.ndarray, hit: MatchHit) -> bool:
    return box_fully_in_item_list(
        frame,
        float(hit.y),
        float(hit.y + hit.h),
        x0=float(hit.x),
        x1=float(hit.x + hit.w),
    )


def ocr_line_fully_in_item_list(frame: np.ndarray, line: OcrLine) -> bool:
    if not line.box:
        return False
    xs = [float(p[0]) for p in line.box]
    ys = [float(p[1]) for p in line.box]
    return box_fully_in_item_list(frame, min(ys), max(ys), x0=min(xs), x1=max(xs))


def _catalog_best_name(text: str, catalog: tuple[str, ...]) -> str:
    hits = names_matching_line(text, catalog)
    if not hits:
        return ""
    return max(hits, key=lambda x: len(fold_zh(x)))


def _clipped_miss(*, item: str, sid: str, visible: list[str] | None = None, root: Path | None = None) -> dict[str, Any]:
    plan = scroll_plan_for_item(sid, item, visible or [], root=root)
    if plan.get("direction") != "down":
        plan = {
            "needs_scroll": 1,
            "direction": "down",
            "hint": "clipped_row",
        }
    else:
        plan = dict(plan)
        plan["hint"] = "clipped_row"
        plan["needs_scroll"] = max(1, int(plan.get("needs_scroll") or 1))
    return {
        "ok": False,
        "error": "not_found",
        "reason": "clipped_row",
        "name": item,
        "skill_id": sid,
        "scroll_plan": plan,
    }


def crop_right_of_icon(frame: np.ndarray, hit: MatchHit, *, pad: int = 6) -> np.ndarray:
    h, w = int(frame.shape[0]), int(frame.shape[1])
    x0 = min(w - 1, max(0, int(hit.x + hit.w + pad)))
    y0 = min(h - 1, max(0, int(hit.y) - 4))
    x1 = min(w, max(x0 + 8, int(hit.x + hit.w + max(80, w * 0.22))))
    y1 = min(h, max(y0 + 8, int(hit.y + hit.h + 4)))
    return np.ascontiguousarray(frame[y0:y1, x0:x1])


def item_template_path(skill_id: str, name: str, *, root: Path | None = None) -> Path:
    return (root or ROOT) / "knowledge" / "templates" / "items" / skill_id / f"{name}.png"


def items_for_skill(skill_id: str, *, root: Path | None = None) -> tuple[str, ...]:
    sid = str(skill_id or "").strip()
    return tuple(
        row["name"]
        for row in iter_tier_items(root=root)
        if row["skill_id"] == sid
    )


_ocr_list_cache: dict[int, tuple[weakref.ReferenceType, bytes, list[OcrLine]]] = {}


def _ocr_lines_in_item_list(frame: np.ndarray) -> list[OcrLine]:
    # A buffer address can be reused by another screenshot. Only this unchanged
    # frame may reuse OCR, including when a caller mutates its array in place.
    cache_key = id(frame)
    h, w = frame.shape[:2]
    x, y, rw, rh = ITEM_LIST_ROI
    pixels = frame[int(y*h):min(h, int((y+rh)*h)), int(x*w):min(w, int((x+rw)*w))]
    fingerprint = hashlib.blake2b(pixels.tobytes(), digest_size=16).digest()
    cached = _ocr_list_cache.get(cache_key)
    if cached is not None and cached[0]() is frame and cached[1] == fingerprint:
        return cached[2]
    lines = ocr_roi(frame, ITEM_LIST_ROI, min_confidence=0.35, backend="cht")
    if not lines:
        lines = ocr_roi(frame, ITEM_LIST_ROI, min_confidence=0.35, backend="v5")
    if len(_ocr_list_cache) > 8:
        _ocr_list_cache.clear()
    _ocr_list_cache[cache_key] = (weakref.ref(frame), fingerprint, lines)
    return lines


def _click_from_template_hit(
    hit: MatchHit,
    *,
    item: str,
    sid: str,
    frame: np.ndarray,
    catalog: tuple[str, ...],
) -> dict[str, Any] | None:
    if not icon_fully_in_item_list(frame, hit):
        return {"ok": False, "error": "clipped_row", "name": item, "skill_id": sid}
    strip = crop_right_of_icon(frame, hit)
    lines = ocr_image(strip, min_confidence=0.35, backend="cht")
    if not lines:
        lines = ocr_image(strip, min_confidence=0.35, backend="v5")
    blob = "".join(str(line.text or "") for line in lines)
    best = _catalog_best_name(blob, catalog)
    if best and fold_zh(best) != fold_zh(item):
        return None
    if not any(line_matches_name(line.text, item, catalog) for line in lines):
        return None
    cx = int(hit.x + hit.w + max(80, int(hit.w)))
    cy = int(hit.cy)
    list_ocr = _click_from_ocr_lines(
        _ocr_lines_in_item_list(frame), item=item, sid=sid, catalog=catalog, frame=frame
    )
    if list_ocr is not None and list_ocr.get("ok"):
        snapped = list_ocr.get("click_client") or []
        if isinstance(snapped, (list, tuple)) and len(snapped) >= 2:
            if not hit.y <= int(snapped[1]) <= hit.y + hit.h:
                return None
            cx, cy = int(snapped[0]), int(snapped[1])
    if not box_fully_in_item_list(frame, cy - 6, cy + 6):
        return {"ok": False, "error": "clipped_row", "name": item, "skill_id": sid}
    return {
        "ok": True,
        "src": "template+ocr",
        "name": item,
        "skill_id": sid,
        "click_client": [cx, cy],
        "icon": {"x": hit.x, "y": hit.y, "w": hit.w, "h": hit.h, "score": hit.score},
    }


def _click_from_ocr_lines(
    lines: list[OcrLine],
    *,
    item: str,
    sid: str,
    catalog: tuple[str, ...],
    frame: np.ndarray | None = None,
) -> dict[str, Any] | None:
    matches: list[tuple[OcrLine, tuple[int, int]]] = []
    clipped = False
    for line in lines:
        best = _catalog_best_name(line.text, catalog)
        if best and fold_zh(best) != fold_zh(item):
            continue
        if not line_matches_name(line.text, item, catalog):
            continue
        center = line_center(line)
        if center is None:
            continue
        if frame is not None and not ocr_line_fully_in_item_list(frame, line):
            clipped = True
            continue
        matches.append((line, center))
    if not matches:
        if clipped:
            return {"ok": False, "error": "clipped_row", "name": item, "skill_id": sid}
        return None
    if len(matches) > 1:
        ys = [c[1] for _, c in matches]
        if max(ys) - min(ys) > 8:
            return {
                "ok": False,
                "error": "ambiguous",
                "name": item,
                "skill_id": sid,
                "matches": len(matches),
            }
    line, center = matches[0]
    return {
        "ok": True,
        "src": "ocr_cht",
        "name": item,
        "skill_id": sid,
        "click_client": [center[0], center[1]],
        "ocr_text": line.text,
    }


def _locate_by_row_geometry(
    frame: np.ndarray,
    *,
    item: str,
    sid: str,
    catalog: tuple[str, ...],
) -> dict[str, Any] | None:
    """固定行高列表：按 catalog 序号推算行带 OCR（scroll 失效降级）。"""
    if item not in catalog:
        return None
    idx = catalog.index(item)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    rx, ry, rw, rh = ITEM_LIST_ROI
    rows = max(1, int(ITEM_LIST_ROWS_VISIBLE))
    x0 = int(w * rx)
    x1 = int(w * (rx + rw))
    cx = int(w * (rx + rw * 0.55))
    for page in range(min(MAX_SCROLL_STEPS + 1, len(catalog))):
        local = idx - page
        if local < 0 or local >= rows:
            continue
        y0 = int(h * (ry + (local / rows) * rh))
        y1 = int(h * (ry + ((local + 1) / rows) * rh))
        if y1 <= y0:
            continue
        strip = np.ascontiguousarray(frame[y0:y1, x0:x1])
        lines = ocr_image(strip, min_confidence=0.32, backend="cht")
        if not lines:
            lines = ocr_image(strip, min_confidence=0.32, backend="v5")
        matching = [ln for ln in lines if line_matches_name(ln.text, item, catalog)]
        if matching:
            if not box_fully_in_item_list(frame, y0, y1):
                return {
                    "ok": False,
                    "error": "clipped_row",
                    "name": item,
                    "skill_id": sid,
                }
            center = line_center(matching[0])
            if center is not None:
                cx = x0 + int(center[0])
                cy = y0 + int(center[1])
            else:
                cy = int((y0 + y1) / 2)
            return {
                "ok": True,
                "src": "geometry+ocr",
                "name": item,
                "skill_id": sid,
                "click_client": [cx, cy],
                "scroll_page": page,
            }
    return None


def _image_row_anchors(frame: np.ndarray, sid: str, *, root: Path | None = None) -> list[tuple[str, MatchHit]]:
    """Registered icon + full-name rows independently prove visible catalog items."""
    folder = (root or ROOT) / 'knowledge/templates/gather_skill' / sid
    catalog = items_for_skill(sid, root=root)
    rows = []
    for metadata in sorted(folder.glob('*.json')):
        try:
            meta = json.loads(metadata.read_text(encoding='utf-8'))
            name = str(meta.get('label') or '')
            if meta.get('role') != 'page_anchor' or name not in catalog:
                continue
            path = metadata.with_suffix('.png')
            hit = match_template(frame, path, roi=ITEM_LIST_ROI, threshold=0.97, min_margin=0.02)
        except (OSError, ValueError):
            continue
        if hit is not None and icon_fully_in_item_list(frame, hit):
            rows.append((name, hit))
    return rows


def item_list_row_positions(frame, skill_id: str, *, root: Path | None = None) -> dict[str, int]:
    """只采用已登记的完整行图；供逐段下翻独立核对，不用OCR授权手势。"""
    return {name: int(hit.cy) for name, hit in _image_row_anchors(frame, skill_id, root=root)}


def locate_skill_item(
    frame: np.ndarray,
    *,
    skill_id: str,
    name: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """返回可点客户区坐标。图模板失败时用繁体 OCR 认右侧文字。"""
    sid = str(skill_id or "").strip()
    item = str(name or "").strip()
    if frame is None or getattr(frame, "size", 0) == 0:
        return {"ok": False, "error": "empty_frame"}
    if not sid or not item:
        return {"ok": False, "error": "missing_name"}
    path = item_template_path(sid, item, root=root)
    metadata = path.with_suffix('.json')
    meta = {}
    if metadata.is_file():
        try:
            meta = json.loads(metadata.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {'ok': False, 'error': 'invalid_template_metadata', 'name': item, 'skill_id': sid}
        if meta.get('status') == 'deferred_level_gray':
            return {'ok': False, 'error': 'deferred_level_gray', 'name': item, 'skill_id': sid}
    if meta.get('row_name_template'):
        # Registered full-name pixels select this row independently of OCR.
        label_path = path.parent / meta['row_name_template']
        label = match_template(frame, label_path, roi=ITEM_LIST_ROI,
                               threshold=0.97, min_margin=0.02)
        # Correlation also matches the dimmed list behind a detail popup.
        # These registered targets require visible white name pixels.
        visible_label = label is not None
        if visible_label and meta.get('require_white_name'):
            pixels = frame[label.y:label.y + label.h, label.x:label.x + label.w]
            visible_label = bool(np.count_nonzero(np.all(pixels >= 200, axis=2)) >= 8)
        if visible_label and icon_fully_in_item_list(frame, label):
            h, w = frame.shape[:2]
            row_roi = (ITEM_LIST_ROI[0], max(0, label.y - 35) / h,
                       0.10, min(h - max(0, label.y - 35), label.h + 70) / h)
            icon = match_template(frame, path, roi=row_roi,
                                  threshold=TEMPLATE_THRESHOLD, min_margin=0.02)
            if icon is not None and icon_fully_in_item_list(frame, icon):
                return {'ok': True, 'src': 'template+name_template', 'name': item,
                        'skill_id': sid, 'click_client': [label.cx, label.cy]}
        if meta.get('image_only'):
            anchors = _image_row_anchors(frame, sid, root=root)
            target = next((hit for name, hit in anchors if name == item), None)
            if target is not None and (visible_label or not meta.get('require_white_name')):
                return {'ok': True, 'src': 'template+row_anchor', 'name': item,
                        'skill_id': sid, 'click_client': [target.cx, target.cy]}
            visible = [name for name, _ in anchors]
            plan = (scroll_plan_for_item(sid, item, visible, root=root) if visible
                    else {'direction': None, 'needs_scroll': 0, 'error': 'no_visible_image_anchor'})
            plan = {**plan, 'src': 'row_anchor_template', 'visible_names': visible}
            return {'ok': False, 'error': 'name_template_not_found', 'name': item, 'skill_id': sid,
                    'scroll_plan': plan}
        visible = [r['name'] for r in enumerate_visible_skill_items(frame, skill_id=sid, root=root)
                   if r.get('ok') and r['name'] != item]
        return {'ok': False, 'error': 'name_template_not_found', 'name': item, 'skill_id': sid,
                'scroll_plan': scroll_plan_for_item(sid, item, visible, root=root)}
    catalog = items_for_skill(sid, root=root) or (item,)
    clipped = False

    if path.is_file():
        hit = match_template(
            frame,
            path,
            roi=ITEM_LIST_ROI,
            threshold=TEMPLATE_THRESHOLD,
            min_margin=0.02,
        )
        if hit is not None:
            out = _click_from_template_hit(hit, item=item, sid=sid, frame=frame, catalog=catalog)
            if out is not None and out.get("ok"):
                return out
            if out is not None and out.get("error") == "clipped_row":
                clipped = True

        from mm_assist.platform.match import match_icon_fuzzy

        try:
            fuzzy = match_icon_fuzzy(
                frame, path, roi=ITEM_LIST_ROI, threshold=min(0.68, TEMPLATE_THRESHOLD)
            )
        except (FileNotFoundError, OSError, ValueError):
            fuzzy = None
        if fuzzy is not None:
            hit_f = MatchHit(
                score=fuzzy.score,
                cx=fuzzy.cx,
                cy=fuzzy.cy,
                x=fuzzy.x,
                y=fuzzy.y,
                w=fuzzy.w,
                h=fuzzy.h,
            )
            out = _click_from_template_hit(
                hit_f, item=item, sid=sid, frame=frame, catalog=catalog
            )
            if out is not None and out.get("ok"):
                out["src"] = "template_fuzzy+ocr"
                return out
            if out is not None and out.get("error") == "clipped_row":
                clipped = True

    lines = _ocr_lines_in_item_list(frame)
    out = _click_from_ocr_lines(lines, item=item, sid=sid, catalog=catalog, frame=frame)
    if out is not None and out.get("ok"):
        return out
    if out is not None and out.get("error") == "clipped_row":
        clipped = True
    if out is not None and out.get("error") == "ambiguous":
        return out

    geo = _locate_by_row_geometry(frame, item=item, sid=sid, catalog=catalog)
    if geo is not None and geo.get("ok"):
        return geo
    if geo is not None and geo.get("error") == "clipped_row":
        clipped = True

    visible = [
        row["name"]
        for row in enumerate_visible_skill_items(frame, skill_id=sid, root=root)
        if row.get("ok")
    ]
    if clipped:
        miss = _clipped_miss(item=item, sid=sid, visible=visible, root=root)
        miss["visible_count"] = len(visible)
        return miss
    plan = scroll_plan_for_item(sid, item, visible, root=root)
    return {
        "ok": False,
        "error": "missing_template" if not path.is_file() else "not_found",
        "name": item,
        "skill_id": sid,
        "visible_count": len(visible),
        "scroll_plan": plan,
    }


def enumerate_visible_skill_items(
    frame: np.ndarray,
    *,
    skill_id: str,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """当前帧右栏可见详情物。整栏 OCR 一次，禁止按 catalog 逐条再识。"""
    sid = str(skill_id or "").strip()
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    catalog = items_for_skill(sid, root=root)
    lines = _ocr_lines_in_item_list(frame)
    rows: list[dict[str, Any]] = []
    for name in catalog:
        out = _click_from_ocr_lines(lines, item=name, sid=sid, catalog=catalog, frame=frame)
        if out is not None and out.get("ok"):
            rows.append(out)
        else:
            rows.append(
                {
                    "name": name,
                    "ok": False,
                    "error": (out or {}).get("error") if out else "not_found",
                }
            )
    return rows


def scroll_plan_for_item(
    skill_id: str,
    name: str,
    visible_names: list[str] | tuple[str, ...],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """目标不在可见集时给出向下滚动计划（禁止向上）。"""
    sid = str(skill_id or "").strip()
    target = str(name or "").strip()
    catalog = items_for_skill(sid, root=root)
    if target not in catalog:
        return {"needs_scroll": 0, "direction": None, "error": "unknown_item"}
    if target in visible_names:
        return {"needs_scroll": 0, "direction": None}
    if not visible_names:
        idx = catalog.index(target)
        viewport = 4
        steps = max(0, idx - viewport + 1)
        return {
            "needs_scroll": min(steps, MAX_SCROLL_STEPS),
            "direction": "down" if steps > 0 else None,
            "hint": "no_visible_anchor",
        }
    visible_idx = [catalog.index(v) for v in visible_names if v in catalog]
    if not visible_idx:
        return {"needs_scroll": 1, "direction": "down", "hint": "no_catalog_anchor"}
    target_idx = catalog.index(target)
    max_vis = max(visible_idx)
    if target_idx > max_vis:
        steps = target_idx - max_vis
        return {
            "needs_scroll": min(steps, MAX_SCROLL_STEPS),
            "direction": "down",
        }
    return {
        "needs_scroll": 0,
        "direction": None,
        "error": "scroll_up_forbidden",
        "hint": "target_above_viewport",
    }


def item_list_scroll_drag_points(
    frame: np.ndarray, *, step: int = 0
) -> tuple[tuple[int, int], tuple[int, int]]:
    """配置固定起止点；第二翻保留短拖，全部相对当前 game_frame。"""
    h, w = int(frame.shape[0]), int(frame.shape[1])
    spec = json.loads((ROOT / 'knowledge/ui/config/process_navigation.json').read_text(encoding='utf-8'))['gather_list_down']
    x, y0f = float(spec['x']), float(spec['start_y'])
    y1f = float(spec['second_end_y'] if int(step) >= 1 else spec['end_y'])
    rx, ry, rw, rh = ITEM_LIST_ROI
    if not (rx < x < rx + rw and ry < y1f < y0f < ry + rh):
        raise ValueError('invalid gather list drag coordinates')
    return (int(w * x), int(h * y0f)), (int(w * x), int(h * y1f))


def ocr_needle_matches(text: str, needles: tuple[str, ...]) -> bool:
    blob = compact_zh(text)
    for needle in needles:
        n = compact_zh(needle)
        if n and n in blob:
            return True
    return False


def _template_path(rel: str, *, root: Path | None = None) -> Path:
    path = Path(rel)
    if not path.is_absolute():
        path = (root or ROOT) / rel
    return path


def _locate_by_template_then_ocr(
    frame: np.ndarray,
    *,
    template_rel: str,
    roi: tuple[float, float, float, float],
    ocr_needles: tuple[str, ...],
    label: str,
    root: Path | None = None,
    threshold: float = ACTION_TEMPLATE_THRESHOLD,
    allow_ocr: bool = True,
) -> dict[str, Any]:
    """图模板优先；miss 时在 ROI 内繁体 OCR 降级。只读，不点击。"""
    if frame is None or getattr(frame, "size", 0) == 0:
        return {"ok": False, "error": "empty_frame", "label": label}
    path = _template_path(template_rel, root=root)
    if path.is_file():
        hit = match_template(
            frame,
            path,
            roi=roi,
            threshold=threshold,
            min_margin=0.02,
        )
        if hit is not None:
            return {
                "ok": True,
                "src": "template",
                "label": label,
                "click_client": [int(hit.cx), int(hit.cy)],
                "icon": {
                    "x": hit.x,
                    "y": hit.y,
                    "w": hit.w,
                    "h": hit.h,
                    "score": hit.score,
                },
            }
        from mm_assist.platform.match import match_icon_fuzzy

        fuzzy_thr = 0.65 if label == "find_nearby" else min(0.62, float(threshold))
        fuzzy = match_icon_fuzzy(frame, path, roi=roi, threshold=fuzzy_thr)
        if fuzzy is not None:
            return {
                "ok": True,
                "src": "template_fuzzy",
                "label": label,
                "click_client": [int(fuzzy.cx), int(fuzzy.cy)],
                "icon": {
                    "x": fuzzy.x,
                    "y": fuzzy.y,
                    "w": fuzzy.w,
                    "h": fuzzy.h,
                    "score": fuzzy.score,
                },
            }
    if not allow_ocr:
        return {'ok': False, 'error': 'not_found', 'label': label, 'path': str(path)}
    lines = ocr_roi(frame, roi, min_confidence=0.35, backend="cht")
    if not lines:
        lines = ocr_roi(frame, roi, min_confidence=0.35, backend="v5")
    for line in lines:
        if not ocr_needle_matches(line.text, ocr_needles):
            continue
        center = line_center(line)
        if center is None:
            continue
        return {
            "ok": True,
            "src": "ocr_cht",
            "label": label,
            "click_client": [center[0], center[1]],
            "ocr_text": line.text,
        }
    err = "missing_template" if not path.is_file() else "not_found"
    return {"ok": False, "error": err, "label": label, "path": str(path)}


def locate_find_nearby(
    frame: np.ndarray,
    *,
    root: Path | None = None,
    allow_ocr: bool = True,
) -> dict[str, Any]:
    """寻找附近位置：图优先，失败则 OCR。只读定位。"""
    return _locate_by_template_then_ocr(
        frame,
        template_rel=FIND_NEARBY_REL,
        roi=GATHER_ACTION_ROI,
        ocr_needles=FIND_NEARBY_OCR_NEEDLES,
        label="find_nearby",
        root=root,
        allow_ocr=allow_ocr,
    )


def locate_help_button(
    frame: np.ndarray,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """问号帮助：图优先，失败则 OCR。只读定位。"""
    return _locate_by_template_then_ocr(
        frame,
        template_rel=HELP_REL,
        roi=GATHER_ACTION_ROI,
        ocr_needles=HELP_OCR_NEEDLES,
        label="help",
        root=root,
    )


def iter_tier_items(*, root: Path | None = None) -> list[dict[str, str]]:
    """living_skill_gather_tiers 中全部详情物清单。"""
    import json

    tiers_path = (root or ROOT) / "knowledge" / "ui" / "config" / "living_skill_gather_tiers.json"
    if not tiers_path.is_file():
        return []
    data = json.loads(tiers_path.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for skill in data.get("skills") or []:
        if not isinstance(skill, dict):
            continue
        sid = str(skill.get("id") or "").strip()
        for tier in skill.get("tiers") or []:
            if not isinstance(tier, dict):
                continue
            for item in tier.get("items") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                if sid and name:
                    out.append({"skill_id": sid, "name": name})
    return out


def default_gather_item_name(skill_id: str, *, root: Path | None = None) -> str:
    """该 skill 在 tiers 配置中的第一个入门详情物。"""
    sid = str(skill_id or "").strip()
    for row in iter_tier_items(root=root):
        if row["skill_id"] == sid:
            return row["name"]
    raise ValueError(f"no default item for skill={sid!r}")
