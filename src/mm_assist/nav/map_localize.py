"""歐拉中景图鉴：拼接、视野定位、拖向目标。

Callers: map_atlas.town_pixel_click_npc / map_goto（城图层）；
像素双图鉴导航门闩见 docs/REQ_FAR_CITY_NAV.md。
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mm_assist.perception.list_ocr import normalize_text

log = logging.getLogger(__name__)

# 匹配时去掉顶栏/任务栏（localize 用）
_ROI_TOP = 0.10
_ROI_BOTTOM = 0.88
_ROI_LEFT = 0.08
# 右缘：采东侧画布（庫漢码头等）；顶右任务卡靠 _STITCH_HUD_MASKS 抠掉
_ROI_RIGHT = 0.88
# 拼接 / 位移估计：再内缩常驻 HUD（面包屑 / 左传送列 / 底栏）
_STITCH_ROI_TOP = 0.11
_STITCH_ROI_LEFT = 0.20
_STITCH_ROI_RIGHT = 0.92
_STITCH_ROI_BOTTOM = 0.85
# 全客户区归一化遮罩（红框）：进 stitch 后涂黑，勿当地图内容
# (nx0, ny0, nx1, ny1)
_STITCH_HUD_MASKS: tuple[tuple[float, float, float, float], ...] = (
    (0.00, 0.00, 0.34, 0.10),  # 顶左面包屑 歐拉大陸/城名
    (0.50, 0.00, 1.00, 0.36),  # 顶右任务/活动卡（造船所等）
    (0.00, 0.48, 0.23, 0.86),  # 左下地区快捷（北方廢墟等）
    (0.00, 0.86, 1.00, 1.00),  # 底栏 地圖/任務/兼職 + 定位钮
)
_FEATHER_PX = 48
_MIN_TEMPLATE_SCORE = 0.55
_MAX_SHIFT_FRAC = 0.82

# 降采样：加速 matchTemplate
_MATCH_SCALE = 0.35
_MIN_MATCH_SCORE = 0.30

DUBARTON_CANONS = ("杜巴頓", "杜巴顿", "杜巴頔")

_DEFAULT_NAV_THRESHOLDS = {
    "N_ulara": 100,
    "N_town": 70,
    "min_match_score_ulara": 0.35,
    # 实机城图曾 0.329 FAIL；略降至 0.30（与 _MIN_MATCH_SCORE 齐）
    "min_match_score_town": 0.30,
    "min_overview_canvas_score": 0.60,
    "min_overview_patch_score": 0.65,
}


@dataclass
class AtlasPlace:
    canon: str
    text: str
    ax: float
    ay: float
    score: float = 0.0


@dataclass
class LocalizeResult:
    ok: bool
    score: float
    # 视野中心在图鉴上的像素
    cx: float
    cy: float
    # ROI 左上角在图鉴上的位置（降采样还原后）
    match_x: float
    match_y: float
    note: str = ""
    # live ROI → 图鉴 的内容缩放（zoom 搜索用；默认 1）
    zoom: float = 1.0


@dataclass
class DragNearResult:
    """drag_until_near 结果：ok=中心距 < N 且定位成功。"""

    ok: bool
    loc: LocalizeResult | None
    steps: int
    note: str = ""


def _maps_parent(maps_root: Path) -> Path:
    """knowledge/maps/ulara → knowledge/maps；已是 maps 则原样。"""
    if maps_root.name == "ulara":
        return maps_root.parent
    return maps_root


def _ulara_dir(maps_root: Path) -> Path:
    """兼容传入 knowledge/maps 或 knowledge/maps/ulara。"""
    if maps_root.name == "ulara":
        return maps_root
    cand = maps_root / "ulara"
    return cand if cand.is_dir() else maps_root


def atlas_paths(
    maps_root: Path,
    *,
    kind: str = "continent",
    region_id: str | None = None,
) -> tuple[Path, Path]:
    """图鉴路径。

    - continent → ``{ulara}/ulara_atlas.png|.json``
    - town → ``knowledge/maps/{region_id}/{region_id}_town_atlas.png|.json``
    """
    if kind == "continent":
        base = _ulara_dir(maps_root)
        return base / "ulara_atlas.png", base / "ulara_atlas.json"
    if kind == "town":
        if not region_id:
            raise ValueError("region_id required for town atlas")
        base = _maps_parent(maps_root) / region_id
        return (
            base / f"{region_id}_town_atlas.png",
            base / f"{region_id}_town_atlas.json",
        )
    raise ValueError(f"unknown atlas kind: {kind!r}")


def load_nav_thresholds(maps_root: Path) -> dict[str, Any]:
    """读 ``nav_thresholds.json``；缺文件用初值。"""
    p = _ulara_dir(maps_root) / "nav_thresholds.json"
    out = dict(_DEFAULT_NAV_THRESHOLDS)
    if not p.is_file():
        return out
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(raw, dict):
        return out
    for k in _DEFAULT_NAV_THRESHOLDS:
        if k in raw:
            try:
                out[k] = type(_DEFAULT_NAV_THRESHOLDS[k])(raw[k])
            except (TypeError, ValueError):
                pass
    return out


def crop_map_roi(
    img: np.ndarray,
    *,
    mode: str = "localize",
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """裁地图画布 ROI，返回 (crop, (x0,y0,x1,y1))。

    ``mode=\"localize\"``：定位匹配用；``mode=\"stitch\"``：拼接/位移估计几何框
   （左传送列/底栏等内缩）。**不**在此涂黑 HUD——相同黑块会让位移估死锁在 0；
    遮罩只在 ``prepare_roi_tile`` 写入图鉴前做。
    """
    h, w = img.shape[:2]
    left, right, top, bottom = _ROI_LEFT, _ROI_RIGHT, _ROI_TOP, _ROI_BOTTOM
    if mode == "stitch":
        left = _STITCH_ROI_LEFT
        right = _STITCH_ROI_RIGHT
        top = _STITCH_ROI_TOP
        bottom = _STITCH_ROI_BOTTOM
    x0 = int(w * left)
    x1 = int(w * right)
    y0 = int(h * top)
    y1 = int(h * bottom)
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def apply_stitch_hud_masks(
    crop: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    full_wh: tuple[int, int],
    masks: tuple[tuple[float, float, float, float], ...] | None = None,
) -> np.ndarray:
    """把全帧 HUD 红框涂黑（相对 ``box`` 裁切坐标）。"""
    fw, fh = int(full_wh[0]), int(full_wh[1])
    if fw <= 0 or fh <= 0 or crop.size == 0:
        return crop
    x0, y0, x1, y1 = (int(v) for v in box)
    out = crop
    for nx0, ny0, nx1, ny1 in masks if masks is not None else _STITCH_HUD_MASKS:
        ax0 = max(0, int(fw * float(nx0)) - x0)
        ay0 = max(0, int(fh * float(ny0)) - y0)
        ax1 = min(out.shape[1], int(fw * float(nx1)) - x0)
        ay1 = min(out.shape[0], int(fh * float(ny1)) - y0)
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        if out is crop:
            out = crop.copy()
        out[ay0:ay1, ax0:ax1] = 0
    return out


def label_in_stitch_hud(
    nx: float,
    ny: float,
    *,
    masks: tuple[tuple[float, float, float, float], ...] | None = None,
) -> bool:
    """全帧归一化点是否落在 stitch HUD 遮罩内。"""
    for x0, y0, x1, y1 in masks if masks is not None else _STITCH_HUD_MASKS:
        if float(x0) <= float(nx) <= float(x1) and float(y0) <= float(ny) <= float(y1):
            return True
    return False


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def selection_green_weight(img: np.ndarray) -> np.ndarray:
    """选中头像绿锥/落点圈的逐像素融合权重（0~1，越绿越低）。

    针对高饱和、偏亮的霓虹绿 UI；普通草地/树也会略降权，依赖重叠区干净帧补回。
    单帧覆盖处保留下限，避免挖空。
    """
    if img.ndim == 2:
        return np.ones(img.shape[:2], dtype=np.float32)
    bgr = img[:, :, :3] if img.shape[2] >= 3 else img
    b, g, r = cv2.split(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    g16 = g.astype(np.int16)
    r16 = r.astype(np.int16)
    b16 = b.astype(np.int16)
    # OpenCV H: 绿约 35–90；霓虹选中：高 S、偏亮、G 明显大于 R/B
    green_hue = (h >= 35) & (h <= 95)
    g_dom = (g16 - np.maximum(r16, b16)) >= 28
    # 强度：饱和度 + 绿色优势
    sat_t = np.clip((s.astype(np.float32) - 90.0) / 90.0, 0.0, 1.0)
    dom_t = np.clip((g16 - np.maximum(r16, b16)).astype(np.float32) / 80.0, 0.0, 1.0)
    bright = np.clip((v.astype(np.float32) - 50.0) / 120.0, 0.0, 1.0)
    strength = sat_t * np.maximum(dom_t, 0.35) * np.maximum(bright, 0.4)
    mask = green_hue & g_dom & (s >= 85) & (g >= 70)
    w = np.ones(g.shape, dtype=np.float32)
    # 最强绿锥压到 ~0.06；轻污染压到 ~0.4
    w[mask] = 1.0 - 0.94 * strength[mask]
    return np.clip(w, 0.06, 1.0)


def stitch_tiles(
    tiles: list[dict[str, Any]],
    *,
    blend: bool = True,
    feather_px: int = _FEATHER_PX,
    suppress_selection_green: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """按 content_offset 拼接 tiles。

    每项: {img: ndarray, ox: float, oy: float, labels?: [{text,nx,ny,score}]}
    约定: atlas = (ox + sx, oy + sy)；拖拽手指 +(dx,dy) 后 ox -= dx。

    ``blend=True`` 时按距边缘羽化加权融合，减轻矩形硬边与叠影。
    ``suppress_selection_green=True`` 时压低选中 NPC 绿锥/落点圈权重，
    重叠区优先用未高亮的帧。
    """
    if not tiles:
        raise ValueError("no tiles")
    placed: list[tuple[np.ndarray, int, int]] = []
    for t in tiles:
        img = t["img"]
        if img is None:
            continue
        ox = int(round(float(t.get("ox", 0))))
        oy = int(round(float(t.get("oy", 0))))
        placed.append((img, ox, oy))
    if not placed:
        raise ValueError("no tile images")

    min_x = min(ox for _, ox, _ in placed)
    min_y = min(oy for _, _, oy in placed)
    max_x = max(ox + im.shape[1] for im, ox, _ in placed)
    max_y = max(oy + im.shape[0] for im, _, oy in placed)
    aw, ah = max_x - min_x, max_y - min_y
    acc = np.zeros((ah, aw, 3), dtype=np.float32)
    weight = np.zeros((ah, aw), dtype=np.float32)

    for im, ox, oy in placed:
        x0, y0 = ox - min_x, oy - min_y
        h, w = im.shape[:2]
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.shape[2] == 4:
            im = im[:, :, :3]
        content = (im.sum(axis=2) > 8).astype(np.float32)
        if blend and feather_px > 0:
            yy = np.minimum(np.arange(h), np.arange(h - 1, -1, -1)).astype(np.float32)
            xx = np.minimum(np.arange(w), np.arange(w - 1, -1, -1)).astype(np.float32)
            edge = np.minimum(yy[:, None], xx[None, :])
            w2d = np.clip(edge / float(feather_px), 0.0, 1.0) * content
        else:
            w2d = content
        if suppress_selection_green:
            w2d = w2d * selection_green_weight(im)
        for c in range(3):
            acc[y0 : y0 + h, x0 : x0 + w, c] += im[:, :, c].astype(np.float32) * w2d
        weight[y0 : y0 + h, x0 : x0 + w] += w2d

    canvas = np.zeros((ah, aw, 3), dtype=np.uint8)
    nz = weight > 1e-3
    for c in range(3):
        ch = acc[:, :, c]
        canvas[:, :, c][nz] = np.clip(ch[nz] / weight[nz], 0, 255).astype(np.uint8)

    places: list[AtlasPlace] = []
    for t in tiles:
        ox = float(t.get("ox", 0)) - min_x
        oy = float(t.get("oy", 0)) - min_y
        th = int(t.get("h") or (t["img"].shape[0] if t.get("img") is not None else 0))
        tw = int(t.get("w") or (t["img"].shape[1] if t.get("img") is not None else 0))
        for lab in t.get("labels") or []:
            text = str(lab.get("text") or "")
            nx = float(lab.get("nx") or 0)
            ny = float(lab.get("ny") or 0)
            if not text:
                continue
            ax = ox + nx * tw
            ay = oy + ny * th
            places.append(
                AtlasPlace(
                    canon=_canon_place(text),
                    text=text,
                    ax=ax,
                    ay=ay,
                    score=float(lab.get("score") or 0),
                )
            )

    places = _merge_places(places)
    meta = {
        "width": aw,
        "height": ah,
        "origin_shift": [min_x, min_y],
        "places": [
            {
                "canon": p.canon,
                "text": p.text,
                "ax": round(p.ax, 2),
                "ay": round(p.ay, 2),
                "score": round(p.score, 3),
            }
            for p in places
        ],
        "feather_px": int(feather_px) if blend else 0,
        "suppress_selection_green": bool(suppress_selection_green),
    }
    return canvas, meta


_QUEST_NOISE = (
    "净化",
    "淨化",
    "通關",
    "通关",
    "目標",
    "目标",
    "委托",
    "委記",
    "定期",
    "冒险家",
    "冒險家",
    "狩場",
    "狩独",
    "狩獨",
    "整體",
    "整体",
    "達成",
)


def is_quest_noise_text(text: str) -> bool:
    t = normalize_text(text)
    return any(k in t for k in _QUEST_NOISE)


def _canon_place(text: str) -> str:
    t = normalize_text(text)
    # 任务栏噪声：保留原文 canon，enrich/merge 时跳过
    if is_quest_noise_text(t):
        return t
    if any(k in t for k in DUBARTON_CANONS) or "杜巴" in t:
        if "北" in t:
            return "杜巴頓北門"
        return "杜巴頓"
    if "堤" in t and "南" in t:
        return "堤爾克那南部"
    if "堤爾" in t or "堤尔" in t or "堤克那" in t:
        return "堤爾克那"
    if "莱比" in t or "萊比" in t:
        return "莱比地下城"
    if "希德" in t or "雪原" in t:
        return "希德斯特雪原"
    if "瑪斯" in t or "马斯" in t:
        return "瑪斯地下城"
    return t


def _merge_places(places: list[AtlasPlace]) -> list[AtlasPlace]:
    best: dict[str, AtlasPlace] = {}
    for p in places:
        if is_quest_noise_text(p.text) or is_quest_noise_text(p.canon):
            continue
        key = p.canon or p.text
        prev = best.get(key)
        if prev is None or p.score > prev.score:
            best[key] = p
    return sorted(best.values(), key=lambda p: -p.score)


def find_place_xy(meta: dict[str, Any], canon: str) -> tuple[float, float] | None:
    """从图鉴 meta 取目标 atlas 坐标。"""
    want = normalize_text(canon)
    places = meta.get("places") or []
    # 精确 canon
    for p in places:
        if normalize_text(str(p.get("canon") or "")) == want:
            return float(p["ax"]), float(p["ay"])
    # 杜巴頓族
    if want in DUBARTON_CANONS or want == "杜巴頓":
        cands = [
            p
            for p in places
            if normalize_text(str(p.get("canon") or "")) == "杜巴頓"
            or (
                "杜巴" in normalize_text(str(p.get("text") or ""))
                and "北" not in normalize_text(str(p.get("text") or ""))
            )
        ]
        if not cands:
            cands = [
                p
                for p in places
                if "杜巴" in normalize_text(str(p.get("text") or p.get("canon") or ""))
            ]
        if cands:
            cands.sort(key=lambda p: -float(p.get("score") or 0))
            return float(cands[0]["ax"]), float(cands[0]["ay"])
    for p in places:
        if want in normalize_text(str(p.get("text") or "")):
            return float(p["ax"]), float(p["ay"])
    return None


def localize_viewport(
    live: np.ndarray,
    atlas: np.ndarray,
    *,
    scale: float = _MATCH_SCALE,
    min_score: float = _MIN_MATCH_SCORE,
) -> LocalizeResult:
    """当前地图截屏在图鉴上的位置（模板匹配）。"""
    if live is None or atlas is None or live.size == 0 or atlas.size == 0:
        return LocalizeResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, "empty")
    crop, (x0, y0, x1, y1) = crop_map_roi(live)
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        return LocalizeResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, "roi_too_small")

    g_atlas = _to_gray(atlas)
    g_tpl = _to_gray(crop)
    if scale != 1.0:
        g_atlas = cv2.resize(
            g_atlas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        g_tpl = cv2.resize(
            g_tpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    if (
        g_tpl.shape[0] >= g_atlas.shape[0]
        or g_tpl.shape[1] >= g_atlas.shape[1]
    ):
        return LocalizeResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, "template_larger")

    res = cv2.matchTemplate(g_atlas, g_tpl, cv2.TM_CCOEFF_NORMED)
    _min_v, max_v, _min_loc, max_loc = cv2.minMaxLoc(res)
    score = float(max_v)
    mx = max_loc[0] / scale
    my = max_loc[1] / scale
    # ROI 中心
    rw, rh = (x1 - x0), (y1 - y0)
    cx = mx + rw / 2.0
    cy = my + rh / 2.0
    ok = score >= min_score
    return LocalizeResult(
        ok=ok,
        score=score,
        cx=cx,
        cy=cy,
        match_x=mx,
        match_y=my,
        note=f"score={score:.3f}",
        zoom=1.0,
    )


def localize_viewport_zoom_search(
    live: np.ndarray,
    atlas: np.ndarray,
    *,
    zooms: tuple[float, ...] = (
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.0,
        1.1,
        1.2,
        1.35,
        1.5,
        1.7,
    ),
    match_scale: float = _MATCH_SCALE,
    min_score: float = _MIN_MATCH_SCORE,
) -> LocalizeResult:
    """缩放搜索：缓解 live 地图与图鉴缩放不一致。

    ``zoom`` = 将 live ROI 缩到图鉴尺度的倍率（<1 表示当前比图鉴更放大）。
    """
    if live is None or atlas is None or live.size == 0 or atlas.size == 0:
        return LocalizeResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, "empty")
    crop, (_x0, _y0, _x1, _y1) = crop_map_roi(live)
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        return LocalizeResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, "roi_too_small")

    g_atlas = _to_gray(atlas)
    g_atlas_s = (
        cv2.resize(
            g_atlas, None, fx=match_scale, fy=match_scale, interpolation=cv2.INTER_AREA
        )
        if match_scale != 1.0
        else g_atlas
    )
    g_crop = _to_gray(crop)
    ch, cw = g_crop.shape[:2]
    best: LocalizeResult | None = None
    # 够用即停：一键修拖图不必扫完全部 zoom
    early = max(float(min_score) + 0.12, 0.52)
    for i, z in enumerate(zooms):
        if i % 4 == 0:
            try:
                from mm_assist.nav.compat import check_abort

                check_abort()
            except Exception as exc:
                from mm_assist.nav.compat import AbortRequested

                if isinstance(exc, AbortRequested):
                    return LocalizeResult(
                        False, 0.0, 0.0, 0.0, 0.0, 0.0, "cancelled"
                    )
                raise
        tw = max(16, int(round(cw * float(z) * match_scale)))
        th = max(16, int(round(ch * float(z) * match_scale)))
        if th >= g_atlas_s.shape[0] or tw >= g_atlas_s.shape[1]:
            continue
        tpl = cv2.resize(g_crop, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(g_atlas_s, tpl, cv2.TM_CCOEFF_NORMED)
        _min_v, max_v, _min_loc, max_loc = cv2.minMaxLoc(res)
        score = float(max_v)
        mx = max_loc[0] / match_scale
        my = max_loc[1] / match_scale
        rw = cw * float(z)
        rh = ch * float(z)
        cand = LocalizeResult(
            ok=score >= min_score,
            score=score,
            cx=mx + rw / 2.0,
            cy=my + rh / 2.0,
            match_x=mx,
            match_y=my,
            note=f"score={score:.3f} zoom={z:.2f}",
            zoom=float(z),
        )
        if best is None or cand.score > best.score:
            best = cand
        if best is not None and best.score >= early:
            break
    if best is None:
        best = LocalizeResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, "no_zoom_fit")
    best.ok = best.score >= min_score
    if not best.ok:
        feature_loc = _localize_partial_features(crop, atlas, min_score=min_score)
        if feature_loc is not None:
            return feature_loc
    return best


def _localize_partial_features(
    crop: np.ndarray, atlas: np.ndarray, *, min_score: float,
) -> LocalizeResult | None:
    """Locate partial atlas overlap; reject sparse or inconsistent landmarks.

    Template containment fails when a viewport extends beyond stitched edges.
    A similarity transform needs >=40 RANSAC inliers, >=70% agreement, and
    inliers spread over both axes. Rotation must be negligible for projection.
    """
    # ORB can concentrate on the narrow town strip and miss the low-contrast
    # coast/desert. SIFT must satisfy the same geometry gate, with a stricter
    # descriptor ratio; it never relaxes the required inlier distribution.
    for detector, norm, ratio_limit, name in (
        (cv2.ORB_create(nfeatures=4000), cv2.NORM_HAMMING, .75, 'ORB'),
        (cv2.SIFT_create(nfeatures=10000), cv2.NORM_L2, .70, 'SIFT'),
    ):
        loc = _localize_feature_transform(crop, atlas, detector=detector,
            norm=norm, ratio_limit=ratio_limit, name=name, min_score=min_score)
        if loc is not None:
            return loc
    return None


def _localize_feature_transform(crop, atlas, *, detector, norm, ratio_limit,
                                name, min_score) -> LocalizeResult | None:
    source_keys, source_desc = detector.detectAndCompute(_to_gray(crop), None)
    atlas_keys, atlas_desc = detector.detectAndCompute(_to_gray(atlas), None)
    if source_desc is None or atlas_desc is None or len(atlas_desc) < 2:
        return None
    pairs = cv2.BFMatcher(norm).knnMatch(source_desc, atlas_desc, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio_limit * pair[1].distance]
    if len(good) < 40:
        return None
    source = np.float32([source_keys[m.queryIdx].pt for m in good])
    target = np.float32([atlas_keys[m.trainIdx].pt for m in good])
    transform, mask = cv2.estimateAffinePartial2D(source, target,
        method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if transform is None or mask is None:
        return None
    inliers = mask.ravel().astype(bool)
    count = int(inliers.sum())
    ratio = count / len(good)
    zoom = float(transform[0, 0])
    if (count < 40 or ratio < max(.70, min_score) or not .40 <= zoom <= 1.70
            or abs(float(transform[1, 0])) > zoom * .02):
        return None
    h, w = crop.shape[:2]
    spread = np.ptp(source[inliers], axis=0)
    if spread[0] < w * .40 or spread[1] < h * .40:
        return None
    # Remove the tiny fitted rotation consistently; use the inlier median
    # translation so projection and center use the same scale-only model.
    offset = np.median(target[inliers] - zoom * source[inliers], axis=0)
    x, y = float(offset[0]), float(offset[1])
    return LocalizeResult(True, ratio, x + w*zoom/2, y + h*zoom/2,
        x, y, f"partial_features {name} inliers={count}/{len(good)} zoom={zoom:.3f}", zoom)


def continent_canvas_matches(
    live: np.ndarray | None,
    atlas: np.ndarray | None,
    *,
    min_score: float = 0.60,
    min_patch_score: float = 0.65,
) -> bool:
    """Independent pixel evidence; a breadcrumb alone cannot prove the level.

    Require one common transform and three spatially separate matching patches.
    One quadrant can contain the quest card or the player's moving marker.
    This stricter classification check does not lower navigation thresholds.
    """
    if (live is None or atlas is None or live.size == 0 or atlas.size == 0
            or min(live.shape[:2]) < 64 or float(live.std()) < 1.0):
        return False
    loc = localize_viewport_zoom_search(live, atlas, min_score=min_score)
    if not loc.ok:
        return False
    crop, _ = crop_map_roi(live)
    crop = cv2.resize(crop, None, fx=loc.zoom, fy=loc.zoom)
    x, y = round(loc.match_x), round(loc.match_y)
    h, w = crop.shape[:2]
    # Clip to actual overlap rather than requiring an unstitched border to
    # exist. Keep each original quadrant's geometry and require half its area.
    if x + w <= 0 or y + h <= 0 or x >= atlas.shape[1] or y >= atlas.shape[0]:
        return False
    matches = 0
    for y0, y1 in ((0, h//2), (h//2, h)):
        for x0, x1 in ((0, w//2), (w//2, w)):
            cx0, cx1 = max(x0, -x), min(x1, atlas.shape[1]-x)
            cy0, cy1 = max(y0, -y), min(y1, atlas.shape[0]-y)
            if ((cx1-cx0)*(cy1-cy0) < (x1-x0)*(y1-y0)*.50
                    or cx1 <= cx0 or cy1 <= cy0):
                continue
            patch = crop[cy0:cy1, cx0:cx1]
            target = atlas[y+cy0:y+cy1, x+cx0:x+cx1]
            # Stitched atlases also have black holes inside their rectangle.
            # Compare only real reference pixels, with >=half a quadrant left.
            valid = np.any(target != 0, axis=2) if target.ndim == 3 else target != 0
            if int(valid.sum()) < (x1-x0)*(y1-y0)*.50:
                continue
            live_values = patch[valid].astype(np.float32).ravel()
            atlas_values = target[valid].astype(np.float32).ravel()
            if float(live_values.std()) < 1.0 or float(atlas_values.std()) < 1.0:
                continue
            live_values -= live_values.mean()
            atlas_values -= atlas_values.mean()
            score = float(np.dot(live_values, atlas_values)
                          / (np.linalg.norm(live_values)*np.linalg.norm(atlas_values)))
            matches += int(score >= min_patch_score)
    log.info("continent canvas evidence score=%.3f patches=%d/4 zoom=%.2f", loc.score, matches, loc.zoom)
    return matches >= 3


def known_town_canvas_matches(live: np.ndarray | None) -> bool:
    """Two registered town terrain patches under one translation, no OCR."""
    if not isinstance(live, np.ndarray) or live.size == 0:
        return False
    from mm_assist.config import ROOT
    from mm_assist.platform.match import match_template
    try:
        spec = json.loads((ROOT/'knowledge/templates/map/bankor_town_view.json').read_text(encoding='utf-8'))
        offsets = []
        for patch in spec['patches']:
            hit = match_template(live, ROOT/patch['template'], roi=tuple(spec['roi']),
                                 threshold=spec['threshold'], min_margin=spec['min_margin'])
            if hit is None:
                return False
            x, y = patch['source_origin']
            offsets.append((hit.x-x, hit.y-y))
        return len(offsets) >= 2 and bool(np.all(np.ptp(offsets, axis=0) <= spec['max_translation_delta']))
    except (OSError, ValueError, KeyError):
        return False


def finger_drag_to_target(
    localize: LocalizeResult,
    target_xy: tuple[float, float],
    *,
    max_step: float = 420.0,
) -> tuple[int, int]:
    """把目标拖到视野中心所需的手指位移 (dx, dy)，单步限幅。"""
    tx, ty = target_xy
    # dx = C.x - T.x （见模块约定）
    raw_dx = localize.cx - tx
    raw_dy = localize.cy - ty
    # 限幅
    scale = 1.0
    mag = (raw_dx * raw_dx + raw_dy * raw_dy) ** 0.5
    if mag > max_step and mag > 1e-3:
        scale = max_step / mag
    return int(round(raw_dx * scale)), int(round(raw_dy * scale))


def _load_atlas_pair(
    png: Path, js: Path
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """png 与 json 皆须存在；任一缺失 → (None, None) 硬 fail。

    png 像素尺寸须与 meta.width/height 一致；不一致则缩放到 meta（places 坐标空间）。
    """
    if not png.is_file() or not js.is_file():
        return None, None
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    try:
        meta = json.loads(js.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if not isinstance(meta, dict):
        return None, None
    try:
        mw = int(meta.get("width") or 0)
        mh = int(meta.get("height") or 0)
    except (TypeError, ValueError):
        mw, mh = 0, 0
    if mw > 0 and mh > 0 and (img.shape[1], img.shape[0]) != (mw, mh):
        log.warning(
            "atlas size mismatch png=%dx%d meta=%dx%d → resize to meta (%s)",
            img.shape[1],
            img.shape[0],
            mw,
            mh,
            png.name,
        )
        img = cv2.resize(img, (mw, mh), interpolation=cv2.INTER_AREA)
    return img, meta


def load_continent_atlas(
    maps_root: Path,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """加载欧拉大陆图鉴；缺文件 → (None, None)。"""
    png, js = atlas_paths(maps_root, kind="continent")
    return _load_atlas_pair(png, js)


def load_town_atlas(
    maps_root: Path,
    region_id: str,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """加载地区城图鉴；png/json 缺一 → (None, None) 硬 fail。"""
    png, js = atlas_paths(maps_root, kind="town", region_id=region_id)
    return _load_atlas_pair(png, js)


def load_atlas(maps_root: Path) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """兼容旧调用 = load_continent_atlas。"""
    return load_continent_atlas(maps_root)


def center_dist(loc: LocalizeResult, target_xy: tuple[float, float]) -> float:
    """视野中心到目标锚点的欧氏距离（atlas 像素）。"""
    tx, ty = target_xy
    return float(math.hypot(loc.cx - tx, loc.cy - ty))


def near_enough(
    loc: LocalizeResult,
    target_xy: tuple[float, float],
    N: float,
) -> bool:
    """成功门闩：中心距 < N（等于 N 不算够近）。"""
    if not loc.ok:
        return False
    return center_dist(loc, target_xy) < float(N)


def is_placeholder_anchor(
    meta: dict[str, Any],
    canon: str,
    *,
    kinds: tuple[str, ...] = ("npc_avatar",),
) -> bool:
    """锚未钉：source=placeholder，或 ax/ay 近似 0。"""
    want = normalize_text(canon)
    kind_set = set(kinds)
    for a in meta.get("anchors") or []:
        if not isinstance(a, dict):
            continue
        if str(a.get("kind") or "") not in kind_set:
            continue
        if normalize_text(str(a.get("canon") or "")) != want:
            continue
        if str(a.get("source") or "") == "placeholder":
            return True
        try:
            ax, ay = float(a["ax"]), float(a["ay"])
        except (KeyError, TypeError, ValueError):
            return True
        if abs(ax) + abs(ay) < 1e-3:
            return True
        return False
    return True  # 找不到也当未钉


def client_norm_to_atlas_xy(
    loc: LocalizeResult,
    nx: float,
    ny: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float]:
    """客户区归一化坐标 → 图鉴像素（需已成功 localize）。"""
    _, (x0, y0, _x1, _y1) = crop_map_roi(
        np.zeros((img_h, img_w, 3), dtype=np.uint8)
    )
    px = float(nx) * float(img_w)
    py = float(ny) * float(img_h)
    z = float(getattr(loc, "zoom", 1.0) or 1.0)
    return loc.match_x + (px - x0) * z, loc.match_y + (py - y0) * z


def atlas_xy_to_client_norm(
    loc: LocalizeResult,
    ax: float,
    ay: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float]:
    """图鉴锚点 → 客户区归一化坐标（需已成功 localize）。"""
    _, (x0, y0, _x1, _y1) = crop_map_roi(
        np.zeros((img_h, img_w, 3), dtype=np.uint8)
    )
    z = float(getattr(loc, "zoom", 1.0) or 1.0)
    if abs(z) < 1e-6:
        z = 1.0
    px = x0 + (float(ax) - loc.match_x) / z
    py = y0 + (float(ay) - loc.match_y) / z
    return px / float(img_w), py / float(img_h)


def upsert_town_npc_anchor(
    maps_root: Path,
    region_id: str,
    *,
    canon: str,
    ax: float,
    ay: float,
    text: str | None = None,
    score: float = 20.0,
    source: str = "pin",
) -> Path:
    """写入 / 替换 town atlas 的 npc_avatar 锚；去掉 placeholder。只改 json。"""
    _png, js = atlas_paths(maps_root, kind="town", region_id=region_id)
    if not js.is_file():
        raise FileNotFoundError(f"town atlas json missing: {js}")
    meta = json.loads(js.read_text(encoding="utf-8"))
    anchors = list(meta.get("anchors") or [])
    want = normalize_text(canon)
    row = {
        "canon": canon,
        "kind": "npc_avatar",
        "text": text or canon,
        "ax": round(float(ax), 2),
        "ay": round(float(ay), 2),
        "score": float(score),
        "source": source,
    }
    replaced = False
    for i, a in enumerate(anchors):
        if not isinstance(a, dict):
            continue
        if normalize_text(str(a.get("canon") or "")) != want:
            continue
        if str(a.get("kind") or "npc_avatar") != "npc_avatar":
            continue
        anchors[i] = {**a, **row}
        replaced = True
        break
    if not replaced:
        anchors.append(row)
    meta["anchors"] = anchors
    meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    js.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return js


def anchor_xy(
    meta: dict[str, Any],
    canon: str,
    *,
    kinds: tuple[str, ...] | None = None,
) -> tuple[float, float] | None:
    """从图鉴 meta 取锚点。

    - town：``anchors`` 列表；``kinds=("npc_avatar",)`` 只认头像
    - ulara：``places``；``kinds=("place",)`` 过滤设施噪音
    - kinds=None：尽量兼容 find_place_xy / 全 anchors
    """
    want = normalize_text(canon)
    kind_set = set(kinds) if kinds else None

    # town schema：anchors
    anchors = meta.get("anchors")
    if isinstance(anchors, list) and anchors:
        for a in anchors:
            if not isinstance(a, dict):
                continue
            k = str(a.get("kind") or "")
            if kind_set is not None and k not in kind_set:
                continue
            if normalize_text(str(a.get("canon") or "")) != want:
                # 软：text 含 canon
                if want not in normalize_text(str(a.get("text") or "")):
                    continue
            try:
                return float(a["ax"]), float(a["ay"])
            except (KeyError, TypeError, ValueError):
                continue
        # 若只要 npc_avatar 且未命中，不再落到 places
        if kind_set is not None and kind_set <= {"npc_avatar"}:
            return None

    # ulara places / 兼容
    if kind_set is not None and "place" in kind_set:
        # 过滤设施噪音后再找
        filtered = {
            "places": [
                p
                for p in (meta.get("places") or [])
                if isinstance(p, dict)
                and not any(
                    fac in normalize_text(str(p.get("canon") or p.get("text") or ""))
                    for fac in _TOWN_FACILITY_CANONS
                )
            ]
        }
        return find_place_xy(filtered, canon)

    if kind_set is None or not anchors:
        return find_place_xy(meta, canon)
    return None


def drag_until_near(
    *,
    atlas: np.ndarray,
    target_xy: tuple[float, float],
    N: float,
    min_score: float = 0.35,
    max_steps: int = 12,
    max_step_px: float = 420.0,
    snap_fn: Callable[[], np.ndarray],
    localize_fn: Callable[[np.ndarray, np.ndarray], LocalizeResult] | None = None,
    drag_fn: Callable[[int, int], None],
) -> DragNearResult:
    """有据拖至中心距 < N。

    纯逻辑：用 snap / localize / drag 回调；**禁止** OCR localize 兜底。
    score 失败、步数尽、或仍 ≥ N → hard fail。
    """
    locate = localize_fn or (
        lambda live, atl: localize_viewport_zoom_search(
            live, atl, min_score=min_score
        )
    )
    last: LocalizeResult | None = None
    for step in range(max_steps):
        try:
            from mm_assist.nav.compat import check_abort

            check_abort()
        except Exception as exc:
            from mm_assist.nav.compat import AbortRequested

            if isinstance(exc, AbortRequested):
                return DragNearResult(
                    ok=False,
                    loc=last or LocalizeResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, "abort"),
                    steps=step,
                    note="cancelled",
                )
            raise
        live = snap_fn()
        loc = locate(live, atlas)
        last = loc
        if not loc.ok or loc.score < float(min_score):
            return DragNearResult(
                ok=False,
                loc=loc,
                steps=step,
                note=f"match_score_fail:{loc.note or f'{loc.score:.3f}'}",
            )
        if near_enough(loc, target_xy, N):
            return DragNearResult(ok=True, loc=loc, steps=step, note="near")
        dx, dy = finger_drag_to_target(loc, target_xy, max_step=max_step_px)
        if dx == 0 and dy == 0:
            return DragNearResult(
                ok=False,
                loc=loc,
                steps=step,
                note=f"zero_drag_still_far:{center_dist(loc, target_xy):.1f}",
            )
        drag_fn(dx, dy)

    # 末步后再定位一次
    live = snap_fn()
    loc = locate(live, atlas)
    last = loc
    if loc.ok and loc.score >= float(min_score) and near_enough(loc, target_xy, N):
        return DragNearResult(ok=True, loc=loc, steps=max_steps, note="near_after_last")
    dist = center_dist(loc, target_xy) if loc.ok else float("inf")
    return DragNearResult(
        ok=False,
        loc=last,
        steps=max_steps,
        note=f"exhausted_dist={dist:.1f}",
    )


def save_atlas(
    maps_root: Path,
    canvas: np.ndarray,
    meta: dict[str, Any],
) -> tuple[Path, Path]:
    maps_root.mkdir(parents=True, exist_ok=True)
    png, js = atlas_paths(maps_root, kind="continent")
    # 控制体积：长边过大则缩小写入
    h, w = canvas.shape[:2]
    max_side = 4096
    out = canvas
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        out = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        meta = dict(meta)
        meta["write_scale"] = scale
        meta["width"] = out.shape[1]
        meta["height"] = out.shape[0]
        places = []
        for p in meta.get("places") or []:
            q = dict(p)
            q["ax"] = round(float(p["ax"]) * scale, 2)
            q["ay"] = round(float(p["ay"]) * scale, 2)
            places.append(q)
        meta["places"] = places
    cv2.imwrite(str(png), out, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    # 预览 jpg 控制体积
    preview = maps_root / "ulara_atlas_preview.jpg"
    prev = out
    if max(prev.shape[:2]) > 1600:
        s = 1600 / float(max(prev.shape[:2]))
        prev = cv2.resize(prev, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(preview), prev, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    meta = dict(meta)
    meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["atlas_png"] = png.name
    meta["preview"] = preview.name
    js.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return png, js


# phaseCorrelate / ORB 弱匹配门槛（auto 路径）
_MIN_PHASE_RESPONSE = 0.15
_MIN_ORB_INLIER_RATIO = 0.20
_MIN_ORB_INLIERS = 8
_MAX_AFFINE_ROT_DEG = 8.0
_MAX_AFFINE_SCALE_DEV = 0.15


def remap_labels_full_to_roi(
    labels: list[dict[str, Any]],
    full_wh: tuple[int, int],
    roi_box: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    """把全帧归一化 nx/ny 映射到 ROI 坐标；落在 ROI 外的标签丢弃（HUD 侧栏等）。"""
    fw, fh = full_wh
    x0, y0, x1, y1 = roi_box
    rw = max(1, x1 - x0)
    rh = max(1, y1 - y0)
    out: list[dict[str, Any]] = []
    for lab in labels or []:
        text = str(lab.get("text") or "")
        if not text:
            continue
        px = float(lab.get("nx") or 0) * fw
        py = float(lab.get("ny") or 0) * fh
        rnx = (px - x0) / rw
        rny = (py - y0) / rh
        if rnx < -0.02 or rnx > 1.02 or rny < -0.02 or rny > 1.02:
            continue
        q = dict(lab)
        q["nx"] = rnx
        q["ny"] = rny
        out.append(q)
    return out


def prepare_roi_tile(
    img: np.ndarray,
    labels: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int], list[dict[str, Any]]]:
    """裁地图 ROI（stitch 模式）+ HUD 遮罩 + 重映射 OCR，供写入图鉴。

    位移估计请用 ``crop_map_roi(..., mode=\"stitch\")``（无遮罩），避免黑块假锁。
    """
    h, w = img.shape[:2]
    crop, box = crop_map_roi(img, mode="stitch")
    crop = apply_stitch_hud_masks(crop, box, full_wh=(w, h))
    kept = [
        lab
        for lab in (labels or [])
        if not label_in_stitch_hud(float(lab.get("nx") or 0), float(lab.get("ny") or 0))
    ]
    remapped = remap_labels_full_to_roi(kept, (w, h), box)
    return crop, box, remapped


def estimate_content_shift(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    use_map_roi: bool = True,
) -> tuple[float, float, float]:
    """估计两帧之间画布内容位移（atlas offset 增量）。

    用 ``phaseCorrelate(prev, curr)``：返回 ``(sx, sy, response)``，
    含义为 curr 相对 prev 的像素位移。约定与 ``stitch_tiles`` 一致：
    ``ox_new = ox_old + sx``，``oy_new = oy_old + sy``
    （等价于手指拖 ``(-sx, -sy)`` 后 ``ox -= finger_dx``）。

    方法 B 人手拖图后，用本函数衔接 tile offset，无需脚本键鼠。
    弱 response 时调用方应判失败，不得推进 ox / 叠帧。
    """
    a = prev
    b = curr
    if use_map_roi:
        a, _ = crop_map_roi(prev, mode="stitch")
        b, _ = crop_map_roi(curr, mode="stitch")
    ga = _to_gray(a).astype(np.float64)
    gb = _to_gray(b).astype(np.float64)
    if ga.shape != gb.shape:
        h = min(ga.shape[0], gb.shape[0])
        w = min(ga.shape[1], gb.shape[1])
        ga = ga[:h, :w]
        gb = gb[:h, :w]
    if ga.size < 64 or gb.size < 64:
        return 0.0, 0.0, 0.0
    # Hann 窗抑制边缘
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_64F)
    shift, response = cv2.phaseCorrelate(ga, gb, win)
    return float(shift[0]), float(shift[1]), float(response)


def estimate_content_shift_orb(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    use_map_roi: bool = True,
) -> tuple[float, float, float, dict[str, Any]]:
    """ORB + BFMatcher(knn) + Lowe0.75 + estimateAffinePartial2D(RANSAC) 估平移。

    返回 ``(sx, sy, score, info)``；score≈inlier_ratio。
    只取平移 dx/dy（忽略旋转/缩放）；旋转过大或匹配不足 → 失败（sx=sy=score=0）。
    """
    a = prev
    b = curr
    if use_map_roi:
        a, _ = crop_map_roi(prev, mode="stitch")
        b, _ = crop_map_roi(curr, mode="stitch")
    ga = _to_gray(a)
    gb = _to_gray(b)
    if ga.shape != gb.shape:
        h = min(ga.shape[0], gb.shape[0])
        w = min(ga.shape[1], gb.shape[1])
        ga = ga[:h, :w]
        gb = gb[:h, :w]
    info: dict[str, Any] = {"ok": False, "method": "orb"}
    if ga.size < 64 or gb.size < 64:
        info["reason"] = "roi_too_small"
        return 0.0, 0.0, 0.0, info

    orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(ga, None)
    kp2, des2 = orb.detectAndCompute(gb, None)
    n1 = 0 if kp1 is None else len(kp1)
    n2 = 0 if kp2 is None else len(kp2)
    info["n_kp"] = (n1, n2)
    if des1 is None or des2 is None or n1 < 12 or n2 < 12:
        info["reason"] = "insufficient_features"
        return 0.0, 0.0, 0.0, info

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        pairs = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        info["reason"] = "matcher_error"
        return 0.0, 0.0, 0.0, info

    good: list[Any] = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < 0.75 * n.distance:
            good.append(m)
    info["n_good"] = len(good)
    if len(good) < _MIN_ORB_INLIERS:
        info["reason"] = "insufficient_matches"
        return 0.0, 0.0, 0.0, info

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    aff, mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if aff is None or mask is None:
        info["reason"] = "affine_failed"
        return 0.0, 0.0, 0.0, info

    inliers = int(mask.ravel().sum())
    inlier_ratio = float(inliers) / float(len(good))
    info["n_inliers"] = inliers
    info["inlier_ratio"] = round(inlier_ratio, 4)
    # aff: [[a, b, tx], [c, d, ty]]；partial ≈ scale*R + t
    a00, a01, tx = float(aff[0, 0]), float(aff[0, 1]), float(aff[0, 2])
    a10, a11, ty = float(aff[1, 0]), float(aff[1, 1]), float(aff[1, 2])
    scale = math.hypot(a00, a10)
    rot_deg = math.degrees(math.atan2(a10, a00))
    info["scale"] = round(scale, 4)
    info["rot_deg"] = round(rot_deg, 3)
    if inliers < _MIN_ORB_INLIERS or inlier_ratio < _MIN_ORB_INLIER_RATIO:
        info["reason"] = "weak_inliers"
        return 0.0, 0.0, 0.0, info
    if abs(rot_deg) > _MAX_AFFINE_ROT_DEG:
        info["reason"] = "rotation_too_large"
        return 0.0, 0.0, 0.0, info
    if abs(scale - 1.0) > _MAX_AFFINE_SCALE_DEV:
        info["reason"] = "scale_too_far"
        return 0.0, 0.0, 0.0, info

    info["ok"] = True
    info["reason"] = "ok"
    # 只取平移；约定与 phaseCorrelate / stitch_tiles 一致
    return tx, ty, inlier_ratio, info


def _prep_pair_gray(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    use_map_roi: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    a = prev
    b = curr
    if use_map_roi:
        a, _ = crop_map_roi(prev, mode="stitch")
        b, _ = crop_map_roi(curr, mode="stitch")
    ga = _to_gray(a)
    gb = _to_gray(b)
    if ga.shape != gb.shape:
        h = min(ga.shape[0], gb.shape[0])
        w = min(ga.shape[1], gb.shape[1])
        ga = ga[:h, :w]
        gb = gb[:h, :w]
    if ga.size < 64 or gb.size < 64:
        return None
    return ga, gb


def _shift_plausible(sx: float, sy: float, w: int, h: int) -> bool:
    return abs(sx) <= _MAX_SHIFT_FRAC * w and abs(sy) <= _MAX_SHIFT_FRAC * h


def warp_to_atlas_delta(sx: float, sy: float) -> tuple[float, float]:
    """prev→curr 点映射平移 → 图鉴放置增量。

    ORB/ECC/模板给出的是「prev 局部点落到 curr 的平移」；同名地物重合要求
    ``ox_curr = ox_prev - sx``。本函数返回可直接 ``ox +=`` 的增量。
    """
    return -float(sx), -float(sy)


def overlap_zncc(
    prev: np.ndarray,
    curr: np.ndarray,
    atlas_sx: float,
    atlas_sy: float,
    *,
    use_map_roi: bool = True,
    min_side: int = 64,
) -> tuple[float, int]:
    """将 curr 放到 ``(atlas_sx, atlas_sy)`` 后与 prev 重叠区的 ZNCC。

    返回 ``(zncc, overlap_pixels)``；重叠过小则 ``(-1.0, 0)``。
    """
    pair = _prep_pair_gray(prev, curr, use_map_roi=use_map_roi)
    if pair is None:
        return -1.0, 0
    ga, gb = pair
    sx, sy = int(round(atlas_sx)), int(round(atlas_sy))
    h, w = ga.shape[:2]
    hb, wb = gb.shape[:2]
    x0, y0 = max(0, sx), max(0, sy)
    x1, y1 = min(w, sx + wb), min(h, sy + hb)
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return -1.0, 0
    pa = ga[y0:y1, x0:x1].astype(np.float32)
    pb = gb[y0 - sy : y1 - sy, x0 - sx : x1 - sx].astype(np.float32)
    if pa.shape != pb.shape or pa.size == 0:
        return -1.0, 0
    pa = pa - float(pa.mean())
    pb = pb - float(pb.mean())
    denom = float(np.linalg.norm(pa) * np.linalg.norm(pb)) + 1e-6
    return float((pa * pb).sum() / denom), int(pa.size)


_MIN_OVERLAP_ZNCC = 0.55


def refine_poses_against_neighbors(
    rois: list[np.ndarray],
    poses: list[tuple[float, float]],
    *,
    rounds: int = 2,
    min_overlap_px: int = 60000,
) -> list[tuple[float, float]]:
    """用邻接高重叠对做 ECC/匹配，迭代拉回位姿（抑制 LS 折中残差）。"""
    out = list(poses)
    n = len(rois)
    if n < 2:
        return out
    for _ in range(rounds):
        for i in range(n):
            best: tuple[float, int, float, float] | None = None
            for j in range(n):
                if j == i:
                    continue
                dx = out[i][0] - out[j][0]
                dy = out[i][1] - out[j][1]
                z, area = overlap_zncc(
                    rois[j], rois[i], dx, dy, use_map_roi=False
                )
                if area < min_overlap_px:
                    continue
                if best is None or z > best[0]:
                    best = (z, j, dx, dy)
            if best is None:
                continue
            z0, j, _dx0, _dy0 = best
            if z0 >= 0.88:
                continue
            sx, sy, sc, info = estimate_tile_shift(
                rois[j], rois[i], method="auto", use_map_roi=False
            )
            if not info.get("ok"):
                continue
            # 新观测：pose_i = pose_j + atlas_delta
            nx = out[j][0] + sx
            ny = out[j][1] + sy
            # 高 zncc 观测权重大
            alpha = 0.65 if z0 < 0.5 else 0.35
            out[i] = (
                out[i][0] * (1.0 - alpha) + nx * alpha,
                out[i][1] * (1.0 - alpha) + ny * alpha,
            )
    return out


def estimate_content_shift_template(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    use_map_roi: bool = True,
    min_score: float = _MIN_TEMPLATE_SCORE,
) -> tuple[float, float, float, dict[str, Any]]:
    """多块 matchTemplate 估纯平移（中心 + 四边中点），取最高分且位移合理者。"""
    info: dict[str, Any] = {"ok": False, "method": "template"}
    pair = _prep_pair_gray(prev, curr, use_map_roi=use_map_roi)
    if pair is None:
        info["reason"] = "roi_too_small"
        return 0.0, 0.0, 0.0, info
    ga, gb = pair
    h, w = ga.shape[:2]
    tw, th = max(48, int(w * 0.32)), max(48, int(h * 0.32))
    # (patch_x0, patch_y0)
    cands = [
        ((w - tw) // 2, (h - th) // 2),  # center
        (int(w * 0.08), (h - th) // 2),  # left
        (int(w * 0.60), (h - th) // 2),  # right
        ((w - tw) // 2, int(h * 0.08)),  # top
        ((w - tw) // 2, int(h * 0.60)),  # bottom
    ]
    best: tuple[float, float, float, int, int] | None = None
    for x0, y0 in cands:
        x0 = int(np.clip(x0, 0, max(0, w - tw)))
        y0 = int(np.clip(y0, 0, max(0, h - th)))
        tpl = ga[y0 : y0 + th, x0 : x0 + tw]
        if tpl.shape[0] >= gb.shape[0] or tpl.shape[1] >= gb.shape[1]:
            continue
        res = cv2.matchTemplate(gb, tpl, cv2.TM_CCOEFF_NORMED)
        _mn, max_v, _mnloc, max_loc = cv2.minMaxLoc(res)
        score = float(max_v)
        sx = float(max_loc[0] - x0)
        sy = float(max_loc[1] - y0)
        if score < min_score or not _shift_plausible(sx, sy, w, h):
            continue
        if best is None or score > best[2]:
            best = (sx, sy, score, x0, y0)
    if best is None:
        info["reason"] = "weak_template"
        return 0.0, 0.0, 0.0, info
    sx, sy, score, x0, y0 = best
    info["ok"] = True
    info["reason"] = "ok"
    info["template_score"] = round(score, 4)
    info["patch_xy"] = (x0, y0)
    return sx, sy, score, info


def refine_shift_ecc(
    prev: np.ndarray,
    curr: np.ndarray,
    sx: float,
    sy: float,
    *,
    use_map_roi: bool = True,
) -> tuple[float, float, float, dict[str, Any]]:
    """ECC MOTION_TRANSLATION 精修初值 (sx,sy)。失败则退回初值。"""
    info: dict[str, Any] = {"ok": False, "method": "ecc", "init": (sx, sy)}
    pair = _prep_pair_gray(prev, curr, use_map_roi=use_map_roi)
    if pair is None:
        info["reason"] = "roi_too_small"
        return sx, sy, 0.0, info
    ga, gb = pair
    h, w = ga.shape[:2]
    ga64 = ga.astype(np.float32)
    gb64 = gb.astype(np.float32)
    warp = np.array([[1.0, 0.0, float(sx)], [0.0, 1.0, float(sy)]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        cc, warp = cv2.findTransformECC(
            ga64,
            gb64,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            None,
            1,
        )
    except cv2.error as e:
        info["reason"] = f"ecc_fail:{e}"
        return sx, sy, 0.0, info
    nsx, nsy = float(warp[0, 2]), float(warp[1, 2])
    if not _shift_plausible(nsx, nsy, w, h):
        info["reason"] = "ecc_shift_too_large"
        return sx, sy, float(cc), info
    # ECC 发散：位移比初值大太多则弃用
    init_mag = abs(sx) + abs(sy) + 1.0
    new_mag = abs(nsx) + abs(nsy)
    if new_mag > 1.85 * init_mag and new_mag > 80:
        info["reason"] = "ecc_diverged"
        return sx, sy, float(cc), info
    info["ok"] = True
    info["reason"] = "ok"
    info["cc"] = float(cc)
    return nsx, nsy, float(cc), info


def estimate_tile_shift(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    method: str = "auto",
    use_map_roi: bool = True,
    min_phase_response: float = _MIN_PHASE_RESPONSE,
    min_orb_score: float = _MIN_ORB_INLIER_RATIO,
    min_template_score: float = _MIN_TEMPLATE_SCORE,
    min_overlap_zncc: float = _MIN_OVERLAP_ZNCC,
) -> tuple[float, float, float, dict[str, Any]]:
    """统一 tile 位移：``template`` | ``orb`` | ``phase`` | ``auto``。

    返回的 ``(sx, sy)`` 是**图鉴 ox/oy 增量**（``ox_curr = ox_prev + sx``），
    已把 ORB/ECC/模板的 prev→curr 仿射平移取反，并用重叠区 ZNCC 门控。

    auto：模板匹配 → ECC 精修 →（失败）ORB →（再失败）phase。
    失败时 sx=sy=score=0 且 ``info["ok"]=False``；调用方不得推进 ox、不得叠该帧。
    """
    m = (method or "auto").lower().strip()
    if m not in ("template", "orb", "phase", "auto"):
        m = "auto"

    def _fail(reason: str, **extra: Any) -> tuple[float, float, float, dict[str, Any]]:
        out: dict[str, Any] = {"ok": False, "method": m, "reason": reason}
        out.update(extra)
        return 0.0, 0.0, 0.0, out

    def _ok_out(
        warp_sx: float,
        warp_sy: float,
        score: float,
        info: dict[str, Any],
        label: str,
    ) -> tuple[float, float, float, dict[str, Any]]:
        pair = _prep_pair_gray(prev, curr, use_map_roi=use_map_roi)
        if pair is not None:
            h, w = pair[0].shape[:2]
            if not _shift_plausible(warp_sx, warp_sy, w, h):
                return _fail("shift_too_large", method=label, warp=(warp_sx, warp_sy))
        atlas_sx, atlas_sy = warp_to_atlas_delta(warp_sx, warp_sy)
        zncc, area = overlap_zncc(
            prev, curr, atlas_sx, atlas_sy, use_map_roi=use_map_roi
        )
        if zncc < min_overlap_zncc:
            return _fail(
                "weak_overlap_zncc",
                method=label,
                warp=(warp_sx, warp_sy),
                atlas=(atlas_sx, atlas_sy),
                zncc=round(zncc, 4),
                overlap_px=area,
            )
        out = dict(info)
        out["ok"] = True
        out["method"] = label
        out["warp"] = (round(warp_sx, 3), round(warp_sy, 3))
        out["zncc"] = round(zncc, 4)
        out["overlap_px"] = area
        return float(atlas_sx), float(atlas_sy), float(score), out

    if m in ("template", "auto"):
        sx, sy, score, info = estimate_content_shift_template(
            prev, curr, use_map_roi=use_map_roi, min_score=min_template_score
        )
        if info.get("ok"):
            rsx, rsy, cc, einf = refine_shift_ecc(
                prev, curr, sx, sy, use_map_roi=use_map_roi
            )
            if einf.get("ok"):
                merged = dict(info)
                merged["ecc"] = einf
                merged["pre_ecc"] = (sx, sy)
                return _ok_out(rsx, rsy, max(score, cc), merged, "template+ecc")
            return _ok_out(sx, sy, score, info, "template")
        if m == "template":
            out = dict(info)
            out["ok"] = False
            return 0.0, 0.0, 0.0, out

    if m in ("orb", "auto"):
        sx, sy, score, info = estimate_content_shift_orb(
            prev, curr, use_map_roi=use_map_roi
        )
        if info.get("ok") and score >= min_orb_score:
            rsx, rsy, cc, einf = refine_shift_ecc(
                prev, curr, sx, sy, use_map_roi=use_map_roi
            )
            if einf.get("ok"):
                merged = dict(info)
                merged["ecc"] = einf
                return _ok_out(rsx, rsy, max(score, cc), merged, "orb+ecc")
            return _ok_out(sx, sy, score, info, "orb")
        if m == "orb":
            out = dict(info)
            out["ok"] = False
            out["method"] = "orb"
            return 0.0, 0.0, 0.0, out

    sx, sy, resp = estimate_content_shift(prev, curr, use_map_roi=use_map_roi)
    if resp >= min_phase_response:
        pair = _prep_pair_gray(prev, curr, use_map_roi=use_map_roi)
        if pair is not None and _shift_plausible(sx, sy, pair[0].shape[1], pair[0].shape[0]):
            rsx, rsy, cc, einf = refine_shift_ecc(
                prev, curr, sx, sy, use_map_roi=use_map_roi
            )
            if einf.get("ok"):
                return _ok_out(
                    rsx,
                    rsy,
                    max(resp, cc),
                    {"phase_response": resp, "ecc": einf},
                    "phase+ecc",
                )
            return _ok_out(sx, sy, resp, {"phase_response": resp}, "phase")
    return _fail("all_failed", phase_response=resp)


def solve_translation_pose_graph(
    n: int,
    edges: list[tuple[int, int, float, float, float]],
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """加权最小二乘解纯平移位姿图。

    每条边 ``(i, j, dx, dy, w)`` 表示 ``pose_j ≈ pose_i + (dx, dy)``。
    各连通分量内以最小下标为原点；分量之间横向分岛（gap）。
    """
    info: dict[str, Any] = {"n": n, "n_edges": len(edges), "components": []}
    if n <= 0:
        return [], info
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j, _dx, _dy, _w in edges:
        if 0 <= i < n and 0 <= j < n:
            union(i, j)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    for nodes in comps.values():
        nodes.sort()

    poses = [(0.0, 0.0)] * n
    island_x = 0.0
    gap = 80.0
    for root, nodes in sorted(comps.items(), key=lambda kv: kv[1][0]):
        local_root = nodes[0]
        idx_map = {node: k for k, node in enumerate(nodes)}
        mvar = len(nodes) - 1
        if mvar == 0:
            poses[local_root] = (island_x, 0.0)
            info["components"].append({"nodes": nodes, "edges": 0})
            island_x += gap
            continue

        AtA = np.zeros((2 * mvar, 2 * mvar), dtype=np.float64)
        Atb = np.zeros(2 * mvar, dtype=np.float64)

        def var(node: int, axis: int) -> int | None:
            li = idx_map[node]
            if li == 0:
                return None
            return 2 * (li - 1) + axis

        def add_eq(coeffs: dict[int, float], rhs: float, w: float) -> None:
            if not coeffs:
                return
            keys = list(coeffs.keys())
            for a in keys:
                Atb[a] += w * coeffs[a] * rhs
                for b in keys:
                    AtA[a, b] += w * coeffs[a] * coeffs[b]

        n_local_edges = 0
        for i, j, dx, dy, w in edges:
            if i not in idx_map or j not in idx_map:
                continue
            n_local_edges += 1
            ww = max(float(w), 1e-3)
            for axis, d in ((0, float(dx)), (1, float(dy))):
                coeffs: dict[int, float] = {}
                vj = var(j, axis)
                vi = var(i, axis)
                # pose_j - pose_i = d；local_root 固定 0
                rhs = d
                if vj is not None:
                    coeffs[vj] = 1.0
                else:
                    rhs = d  # pose_j=0 → -pose_i = d 已由 vi 处理
                if vi is not None:
                    coeffs[vi] = coeffs.get(vi, 0.0) - 1.0
                else:
                    # pose_i=0 → pose_j = d
                    pass
                if not coeffs:
                    continue
                # 若 j 是 root：0 - pose_i = d → pose_i = -d
                if vj is None and vi is not None:
                    add_eq({vi: 1.0}, -d, ww)
                else:
                    add_eq(coeffs, rhs, ww)

        try:
            x = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(AtA, Atb, rcond=None)[0]

        local_poses: list[tuple[float, float]] = [(0.0, 0.0)] * len(nodes)
        local_poses[0] = (0.0, 0.0)
        for li in range(1, len(nodes)):
            local_poses[li] = (float(x[2 * (li - 1)]), float(x[2 * (li - 1) + 1]))

        # 分岛：平移到 island_x
        xs = [p[0] for p in local_poses]
        ys = [p[1] for p in local_poses]
        # 无 tile 尺寸时按位姿跨度估岛宽，避免分量重叠
        span_x = (max(xs) - min(xs)) if xs else 0.0
        span_y = (max(ys) - min(ys)) if ys else 0.0
        for li, node in enumerate(nodes):
            lx, ly = local_poses[li]
            poses[node] = (lx - min(xs) + island_x, ly - min(ys))
        island_x += span_x + 1400.0 + gap
        info["components"].append(
            {
                "nodes": nodes,
                "edges": n_local_edges,
                "root": local_root,
                "span": [round(span_x, 1), round(span_y, 1)],
            }
        )

    return poses, info


def build_pairwise_shift_edges(
    rois: list[np.ndarray],
    *,
    method: str = "auto",
    min_score: float = 0.45,
    max_pair_span: int | None = None,
    consistency_px: float = 48.0,
) -> tuple[list[tuple[int, int, float, float, float]], dict[str, Any]]:
    """ROI 列表两两（或邻域）估平移边；双向一致则加权更高。

    输入须已是 stitch-ROI（``use_map_roi=False``）。
    """
    n = len(rois)
    edges: list[tuple[int, int, float, float, float]] = []
    meta: dict[str, Any] = {
        "tried": 0,
        "accepted": 0,
        "rejected_weak": 0,
        "rejected_inconsistent": 0,
    }
    span = max_pair_span if max_pair_span is not None else n
    for i in range(n):
        for j in range(i + 1, min(i + 1 + span, n)):
            meta["tried"] += 1
            sx_ij, sy_ij, sc_ij, info_ij = estimate_tile_shift(
                rois[i], rois[j], method=method, use_map_roi=False
            )
            if not info_ij.get("ok") or sc_ij < min_score:
                meta["rejected_weak"] += 1
                continue
            sx_ji, sy_ji, sc_ji, info_ji = estimate_tile_shift(
                rois[j], rois[i], method=method, use_map_roi=False
            )
            if info_ji.get("ok") and sc_ji >= min_score * 0.85:
                err = math.hypot(sx_ij + sx_ji, sy_ij + sy_ji)
                if err > consistency_px:
                    meta["rejected_inconsistent"] += 1
                    continue
                # 平均双向
                sx = 0.5 * (sx_ij - sx_ji)
                sy = 0.5 * (sy_ij - sy_ji)
                w = float(min(sc_ij, sc_ji)) * (1.0 + 0.5 * (1.0 - err / consistency_px))
            else:
                # 仅单向：邻接边仍可用，权重打折
                if j > i + 2:
                    meta["rejected_weak"] += 1
                    continue
                sx, sy = sx_ij, sy_ij
                w = float(sc_ij) * 0.45
            edges.append((i, j, float(sx), float(sy), float(w)))
            meta["accepted"] += 1
    # 保证时间邻接至少尝试留下弱边，避免全断
    if n >= 2 and not any(e[0] + 1 == e[1] for e in edges):
        for i in range(n - 1):
            sx, sy, sc, info = estimate_tile_shift(
                rois[i], rois[i + 1], method=method, use_map_roi=False
            )
            if info.get("ok"):
                edges.append((i, i + 1, float(sx), float(sy), max(float(sc), 0.2) * 0.3))
                meta["accepted"] += 1
                meta["forced_adjacent"] = True
    return edges, meta


def stitch_rois_pose_graph(
    rois: list[np.ndarray],
    *,
    labels_per_tile: list[list[dict[str, Any]]] | None = None,
    tags: list[str] | None = None,
    method: str = "auto",
    min_score: float = 0.45,
    feather_px: int = 72,
    max_pair_span: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """位姿图拼接：全对/邻域匹配 → LS 解 ox/oy → 羽化融合。

    比邻接 ORB 链更能抑制累计漂移；比 Stitcher.SCANS 更贴合「纯平移」城图。
    """
    if not rois:
        raise ValueError("no rois")
    edges, edge_meta = build_pairwise_shift_edges(
        rois,
        method=method,
        min_score=min_score,
        max_pair_span=max_pair_span,
    )
    poses, pose_info = solve_translation_pose_graph(len(rois), edges)
    poses = refine_poses_against_neighbors(rois, poses, rounds=2)
    tiles: list[dict[str, Any]] = []
    for i, img in enumerate(rois):
        ox, oy = poses[i]
        labs = (labels_per_tile or [None] * len(rois))[i] or []
        tiles.append(
            {
                "img": img,
                "ox": ox,
                "oy": oy,
                "w": int(img.shape[1]),
                "h": int(img.shape[0]),
                "labels": labs,
                "tag": (tags or [None] * len(rois))[i],
            }
        )
    canvas, sm = stitch_tiles(tiles, blend=True, feather_px=feather_px)
    # 拼接后抽样重叠 ZNCC，写入 meta 便于验收
    zncc_samples: list[dict[str, Any]] = []
    for i in range(len(rois)):
        for j in range(i + 1, len(rois)):
            dx = poses[j][0] - poses[i][0]
            dy = poses[j][1] - poses[i][1]
            z, area = overlap_zncc(rois[i], rois[j], dx, dy, use_map_roi=False)
            if area >= 80000:
                zncc_samples.append(
                    {
                        "i": i,
                        "j": j,
                        "zncc": round(z, 4),
                        "area": area,
                    }
                )
    zncc_samples.sort(key=lambda r: r["zncc"])
    sm["stitch_method"] = "pose_graph"
    sm["pose_graph"] = {
        "edges": edge_meta,
        "components": pose_info.get("components"),
        "poses": [{"ox": round(p[0], 2), "oy": round(p[1], 2)} for p in poses],
        "n_edges_used": len(edges),
        "zncc_mean": round(
            sum(r["zncc"] for r in zncc_samples) / len(zncc_samples), 4
        )
        if zncc_samples
        else None,
        "zncc_worst": zncc_samples[:3] if zncc_samples else [],
    }
    canvas, sm, rinfo = recenter_atlas_on_town(canvas, sm)
    sm["recenter_info"] = rinfo
    return canvas, sm


def stitch_rois_opencv_scans(
    rois: list[np.ndarray],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """OpenCV ``Stitcher_SCANS``；失败返回 (None, info)。"""
    info: dict[str, Any] = {"ok": False, "method": "opencv_stitcher_scans"}
    if not rois:
        info["reason"] = "empty"
        return None, info
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        status, pano = stitcher.stitch(rois)
    except Exception as e:
        info["reason"] = f"exc:{e}"
        return None, info
    info["status"] = int(status)
    if status != cv2.Stitcher_OK or pano is None or pano.size == 0:
        info["reason"] = "stitch_failed"
        return None, info
    info["ok"] = True
    info["width"] = int(pano.shape[1])
    info["height"] = int(pano.shape[0])
    return pano, info


def estimate_town_center_xy(
    canvas: np.ndarray,
    meta: dict[str, Any] | None = None,
) -> tuple[float, float, str]:
    """估计城镇中心（图鉴像素）。

    优先：城内地标（鐵匠/广场/学校等）均值 → 红屋顶质心 → 单点鐵匠 → 内容质心。
    返回 ``(cx, cy, source)``。
    """
    h, w = canvas.shape[:2]
    meta = meta or {}
    town_keys = (
        "鐵匠",
        "铁匠",
        "截匠",
        "广场",
        "廣場",
        "學校",
        "学校",
        "雜貨",
        "杂货",
        "食品",
        "書店",
        "书店",
        "銀行",
        "银行",
        "教堂",
        "治癒",
        "治愈",
        "武器",
        "服裝",
        "服装",
    )
    pts: list[tuple[float, float]] = []
    for p in meta.get("places") or []:
        if not isinstance(p, dict):
            continue
        text = str(p.get("canon") or "") + str(p.get("text") or "")
        nt = normalize_text(text)
        if any(normalize_text(k) in nt or k in text for k in town_keys):
            pts.append((float(p.get("ax") or 0), float(p.get("ay") or 0)))
    if len(pts) >= 2:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        return cx, cy, f"places_mean_n={len(pts)}"

    b, g, r = cv2.split(canvas)
    mask = canvas.sum(axis=2) > 30
    roof = (r > 90) & (r > g + 15) & (r > b + 15) & mask
    yy, xx = np.where(roof)
    if xx.size >= 500:
        return float(xx.mean()), float(yy.mean()), "roof"

    if len(pts) == 1:
        return pts[0][0], pts[0][1], "place_single"

    for a in meta.get("anchors") or []:
        if not isinstance(a, dict):
            continue
        if a.get("kind") == "npc_avatar" and abs(float(a.get("ax") or 0)) + abs(
            float(a.get("ay") or 0)
        ) > 2:
            return float(a["ax"]), float(a["ay"]), "anchor"

    yy, xx = np.where(mask)
    if xx.size:
        return float(xx.mean()), float(yy.mean()), "content"
    return w / 2.0, h / 2.0, "fallback"


def recenter_atlas_on_town(
    canvas: np.ndarray,
    meta: dict[str, Any] | None = None,
    *,
    margin: int = 40,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """平移画布使城镇中心落到内容包围盒的几何中心，四周留 margin。

    更新 meta 内 places/anchors/pose_graph.poses 的 ax/ay/ox/oy。
    返回 ``(new_canvas, new_meta, info)``。
    """
    meta = dict(meta or {})
    h, w = canvas.shape[:2]
    cx, cy, src = estimate_town_center_xy(canvas, meta)
    mask = canvas.sum(axis=2) > 30
    ys, xs = np.where(mask)
    info: dict[str, Any] = {
        "town_center_before": [round(cx, 1), round(cy, 1)],
        "center_source": src,
    }
    if xs.size == 0:
        info["ok"] = False
        info["reason"] = "empty_canvas"
        return canvas, meta, info

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # 裁到内容，再算城镇相对位置，pad 使城镇居中
    crop = canvas[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    tcx, tcy = cx - x0, cy - y0
    # 最终画布：城镇在中心 → 左边要 tcx+margin，右边要 (cw-tcx)+margin
    left = int(round(tcx + margin))
    right = int(round((cw - tcx) + margin))
    top = int(round(tcy + margin))
    bottom = int(round((ch - tcy) + margin))
    # 对称化：取左右/上下更大的那边，保证城镇真正居中
    half_w = max(left, right)
    half_h = max(top, bottom)
    nw, nh = half_w * 2, half_h * 2
    out = np.zeros((nh, nw, 3), dtype=np.uint8)
    # 城镇中心放到 (half_w, half_h)
    dx = int(round(half_w - tcx))
    dy = int(round(half_h - tcy))
    out[dy : dy + ch, dx : dx + cw] = crop

    # 旧坐标 → 新：先减 (x0,y0) 再加 (dx,dy)
    ox_add = float(dx - x0)
    oy_add = float(dy - y0)

    def _shift_items(items: list[Any] | None, xk: str, yk: str) -> list[Any]:
        out_items: list[Any] = []
        for it in items or []:
            if not isinstance(it, dict):
                out_items.append(it)
                continue
            q = dict(it)
            if xk in q and yk in q:
                q[xk] = round(float(q[xk]) + ox_add, 2)
                q[yk] = round(float(q[yk]) + oy_add, 2)
            out_items.append(q)
        return out_items

    meta["places"] = _shift_items(meta.get("places"), "ax", "ay")
    meta["anchors"] = _shift_items(meta.get("anchors"), "ax", "ay")
    meta["facilities"] = _shift_items(meta.get("facilities"), "ax", "ay")
    pg = meta.get("pose_graph")
    if isinstance(pg, dict):
        pg = dict(pg)
        pg["poses"] = _shift_items(pg.get("poses"), "ox", "oy")
        meta["pose_graph"] = pg
    meta["width"] = nw
    meta["height"] = nh
    meta["recenter"] = {
        "ox_add": round(ox_add, 2),
        "oy_add": round(oy_add, 2),
        "town_center": [round(half_w, 1), round(half_h, 1)],
        "center_source": src,
        "content_bbox_before": [x0, y0, x1, y1],
    }
    info["ok"] = True
    info["size_before"] = [w, h]
    info["size_after"] = [nw, nh]
    info["shift"] = [round(ox_add, 1), round(oy_add, 1)]
    info["town_center_after"] = [half_w, half_h]
    return out, meta, info


def save_town_atlas(
    maps_root: Path,
    region_id: str,
    canvas: np.ndarray,
    meta: dict[str, Any],
) -> tuple[Path, Path]:
    """写入 ``{region_id}/{region_id}_town_atlas.png|.json``。"""
    png, js = atlas_paths(maps_root, kind="town", region_id=region_id)
    png.parent.mkdir(parents=True, exist_ok=True)
    h, w = canvas.shape[:2]
    max_side = 4096
    out = canvas
    scale = 1.0
    meta = dict(meta)
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        out = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        meta["write_scale"] = scale
        meta["width"] = out.shape[1]
        meta["height"] = out.shape[0]
        for key in ("places", "anchors"):
            items = []
            for p in meta.get(key) or []:
                q = dict(p)
                if "ax" in q and "ay" in q:
                    q["ax"] = round(float(q["ax"]) * scale, 2)
                    q["ay"] = round(float(q["ay"]) * scale, 2)
                items.append(q)
            if key in meta:
                meta[key] = items
    else:
        meta["width"] = w
        meta["height"] = h
    cv2.imwrite(str(png), out, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    preview = png.parent / f"{region_id}_town_atlas_preview.jpg"
    prev = out
    if max(prev.shape[:2]) > 1600:
        s = 1600 / float(max(prev.shape[:2]))
        prev = cv2.resize(prev, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(preview), prev, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["region_id"] = region_id
    meta["atlas_png"] = png.name
    meta["preview"] = preview.name
    meta["source"] = meta.get("source") or "method_B"
    js.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return png, js


def town_meta_from_stitch(
    stitch_meta: dict[str, Any],
    *,
    region: str = "杜巴頓",
    region_id: str = "dunbarton",
    npc_canons: tuple[str, ...] = ("奈麗絲",),
) -> dict[str, Any]:
    """把 stitch places 收成 town atlas schema（anchors=npc_avatar）。"""
    anchors: list[dict[str, Any]] = []
    facilities: list[dict[str, Any]] = []
    for p in stitch_meta.get("places") or []:
        if not isinstance(p, dict):
            continue
        canon = str(p.get("canon") or "")
        nt = normalize_text(canon + str(p.get("text") or ""))
        is_npc = any(normalize_text(c) in nt for c in npc_canons)
        item = {
            "canon": canon or str(p.get("text") or ""),
            "text": str(p.get("text") or ""),
            "ax": float(p.get("ax") or 0),
            "ay": float(p.get("ay") or 0),
            "score": float(p.get("score") or 0),
            "source": "stitch_ocr",
        }
        if is_npc:
            item["kind"] = "npc_avatar"
            item["canon"] = next(
                (c for c in npc_canons if normalize_text(c) in nt),
                item["canon"],
            )
            anchors.append(item)
        else:
            item["kind"] = "facility"
            facilities.append(item)
    # 杜巴修理链：无 OCR 时占位奈麗絲（ax/ay=0）；其它城勿塞假锚
    if "奈麗絲" in npc_canons and not any(a.get("canon") == "奈麗絲" for a in anchors):
        anchors.append(
            {
                "canon": "奈麗絲",
                "kind": "npc_avatar",
                "text": "奈麗絲",
                "ax": 0.0,
                "ay": 0.0,
                "score": 0.0,
                "source": "placeholder",
            }
        )
    return {
        "region": region,
        "region_id": region_id,
        "width": int(stitch_meta.get("width") or 0),
        "height": int(stitch_meta.get("height") or 0),
        "origin_shift": stitch_meta.get("origin_shift"),
        "places": stitch_meta.get("places") or [],
        "anchors": anchors,
        "facilities": facilities,
        "stitch_method": stitch_meta.get("stitch_method"),
        "pose_graph": stitch_meta.get("pose_graph"),
        "feather_px": stitch_meta.get("feather_px"),
        "note": "方法 B 人手拖 + 位姿图/ROI 拼接",
    }


def has_dubarton(meta: dict[str, Any] | None) -> bool:
    if not meta:
        return False
    return find_place_xy(meta, "杜巴頓") is not None


# 定位时优先用地图像素锚（非任务噪声）
_OCR_LOCALIZE_PREFER = (
    "堤爾克那南部",
    "莱比地下城",
    "希德斯特雪原",
    "瑪斯地下城",
    "堤爾克那",
    "艾菲地下城",
    "灰色谷地",
    "阿德莉亞平原",
)

# 城镇设施不作大陆定位锚（误在详图里 OCR 到会拖飞）
_TOWN_FACILITY_CANONS = (
    "治療所",
    "治疗所",
    "旅館",
    "旅馆",
    "雜貨",
    "杂货",
    "銀行",
    "银行",
    "鐵匠",
    "铁匠",
    "學校",
    "学校",
    "工坊",
    "食品店",
    "教會",
    "教会",
    "造船",
    "碼頭",
    "码头",
    "交易所",
    "村長家",
    "兵事務",
)


def localize_by_ocr_places(
    lines: list[Any],
    win_w: int,
    win_h: int,
    meta: dict[str, Any],
    *,
    min_landmarks: int = 1,
) -> LocalizeResult:
    """用当前 OCR 地名 + 图鉴 places 绝对坐标反推视野中心。

    deprecated: 仅建图辅助；像素双图鉴主路径禁止调用（REQ_FAR_CITY_NAV）。
    """
    from mm_assist.perception.list_ocr import box_center_norm

    by_canon: dict[str, tuple[float, float, float]] = {}
    for p in meta.get("places") or []:
        c = str(p.get("canon") or "")
        text = str(p.get("text") or "")
        if not c or is_quest_noise_text(c) or is_quest_noise_text(text):
            continue
        # 「通往X」是传送口，不是地区标，勿作锚
        if "通往" in normalize_text(text) or "通向" in normalize_text(text):
            continue
        if _place_is_duba_family(c):
            continue
        if any(k in c for k in _TOWN_FACILITY_CANONS):
            continue
        try:
            by_canon[c] = (float(p["ax"]), float(p["ay"]), float(p.get("score") or 0))
        except Exception:
            continue
    if not by_canon:
        return LocalizeResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, "no_atlas_places")

    votes: list[tuple[float, float, float, str]] = []
    for L in lines:
        text = str(getattr(L, "text", "") or "")
        if not text or is_quest_noise_text(text):
            continue
        if "通往" in normalize_text(text) or "通向" in normalize_text(text):
            continue
        if any(k in normalize_text(text) for k in _TOWN_FACILITY_CANONS):
            continue
        cxy = box_center_norm(L, win_w, win_h)
        if cxy is None:
            continue
        nx, ny = cxy
        # 与 map_atlas._is_map_canvas_poi 对齐：排除顶栏 + 左下 POI 按钮条
        # （ny>0.82 的「艾菲地下城」等是侧栏按钮，不当图鉴锚）
        if not (0.06 <= nx <= 0.78 and 0.10 <= ny <= 0.82):
            continue
        if is_quest_noise_text(text) or text.rstrip().endswith("狩"):
            continue
        canon = _canon_place(text)
        if canon not in by_canon:
            # 软匹配：仅长 canon，避免「加德」误吃「杜加德走廊」
            nt = normalize_text(text)
            hit_c = None
            for c in by_canon:
                nc = normalize_text(c)
                if len(nc) < 4:
                    continue
                if nc == nt or nc in nt or (len(nt) >= 4 and nt in nc):
                    hit_c = c
                    break
            if hit_c is None:
                continue
            canon = hit_c
        elif is_quest_noise_text(canon):
            continue
        # 过短锚一律跳过
        if len(normalize_text(canon)) < 3:
            continue
        ax, ay, ps = by_canon[canon]
        sx, sy = nx * win_w, ny * win_h
        cx = ax - sx + win_w * 0.5
        cy = ay - sy + win_h * 0.5
        prefer = 8.0 if canon in _OCR_LOCALIZE_PREFER else 0.0
        if canon == "堤爾克那南部":
            prefer += 4.0
        # places_rel 估点易漂，压权重；真 OCR 钉点优先
        src = ""
        for p in meta.get("places") or []:
            if str(p.get("canon") or "") == canon:
                src = str(p.get("source") or "")
                break
        if src.startswith("places_rel"):
            prefer -= 6.0
        elif "live_ocr" in src or src.startswith("co_visible"):
            prefer += 5.0
        conf = float(
            getattr(L, "confidence", None) or getattr(L, "score", 0) or 0
        )
        score = conf + prefer + 0.1 * ps
        votes.append((cx, cy, score, canon))

    if len(votes) < min_landmarks:
        return LocalizeResult(
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            f"landmarks={len(votes)}",
        )
    # 剔除离群：相对 median 过远的票
    xs = sorted(v[0] for v in votes)
    ys = sorted(v[1] for v in votes)
    mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
    tight = [v for v in votes if abs(v[0] - mx) + abs(v[1] - my) < 550]
    if tight:
        votes = tight
    # 加权：按 score 取 top-k 均值
    votes.sort(key=lambda v: -v[2])
    top = votes[: min(5, len(votes))]
    wsum = sum(v[2] for v in top) or 1.0
    cx = sum(v[0] * v[2] for v in top) / wsum
    cy = sum(v[1] * v[2] for v in top) / wsum
    note = ",".join(v[3] for v in top[:3])
    return LocalizeResult(
        ok=True,
        score=float(top[0][2]),
        cx=cx,
        cy=cy,
        match_x=cx - win_w * 0.5,
        match_y=cy - win_h * 0.5,
        note=f"ocr:{note}",
    )


def _place_is_duba_family(canon: str) -> bool:
    c = normalize_text(canon)
    return "杜巴" in c
