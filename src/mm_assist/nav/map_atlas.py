"""歐拉大陸地图：开图 → 全图 → 扫地区 → 白名单 NPC 前往。

调用：一键修理；边界与验收见 docs/PRODUCT.md、docs/TESTING.md。
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from mm_assist.safety.input import click_client, drag_client
from mm_assist.platform.capture import grab_client, save_capture
from mm_assist.config import AppConfig, ROOT
from mm_assist.perception.list_ocr import box_center_norm, normalize_text
from mm_assist.perception.ocr import OcrLine, ocr_image
from mm_assist.nav.compat import AbortRequested, GuardError, check_abort, in_production_mode, in_test_mode, prefer_visible_clicks, prepare_grabber, production_click_method, timed_action
from mm_assist.platform.window import WindowInfo, focus_window, is_foreground, refresh_window

if TYPE_CHECKING:
    from mm_assist.life_execute.ui_logic import UiWhere

log = logging.getLogger(__name__)

MapState = Literal[
    "hud",
    "map",  # Current map parameters; no map tier or atlas authorization.
    "continent",
    "region",
    "npc_card",
    "traveling",
    "dialog",
    "unknown",
    "map_button_missing",
]

# 1920×1440 标定 · 纯鼠标（不用按键）
# 开图：小地图圆右下金色钮（CV/扫点锁定 2026-07-25）
# 下列客户区坐标均相对标定分辨率；点击前经 _scale_xy 适配当前窗口
_CAL_W, _CAL_H = 1920, 1440
MINIMAP_OPEN_XY = (1811, 176)  # 偶发无效 / 易误开「主要」目录
MINIMAP_XY = (1870, 185)  # 小地图「M」（2026-07-29 12:11 探针；此前 1848,175 / 1865,95 失效）
CLOSE_XY = (1905, 28)
# 右上关闭 X（overlay menu.close · 1920×1440 归一）
CLOSE_X_ROI: tuple[float, float, float, float] = (0.9568, 0.0069, 0.0365, 0.0486)
# 开图钮模板：只允许新帧识别按钮；缺失不回退旧标定点。
MAP_OPEN_BTN_REL = Path("knowledge/ui/map_open_btn.png")
MAP_OPEN_BTN_ROI = (0.88, 0.05, 0.12, 0.18)  # 右上小地图一带
MAP_OPEN_BTN_THRESHOLD = 0.72
# 歐拉中景「杜巴頓」白色城标（图鉴/实机模板点；不依赖 OCR）
DUBARTON_CASTLE_REL = Path("knowledge/maps/ulara/ref_dubarton_castle_icon.png")
DUBARTON_CASTLE_THRESHOLD = 0.58
DUBARTON_CASTLE_ROI = (0.10, 0.12, 0.80, 0.72)  # 地图画布，避开顶栏/底栏
# 选点后右下角绿色「自动移动」近似中心（优 CV 绿块）
GOTO_FALLBACK_XY = (1580, 1330)
# 面包屑「歐拉大陸」solo OCR 中心（分行时）
ULARA_BREADCRUMB_XY = (146, 58)
MAP_NAVIGATION_CONFIG_REL = Path('knowledge/ui/config/map_navigation.json')

CONTINENT_NEEDLES = (
    "歐拉大陸",
    "欧拉大陆",
    "歐拉大陆",
    "欧拉大陸",
    "欧拉大睦",  # OCR 误识
    "歐拉大睦",
)
GOTO_NEEDLES = ("前往此處", "前往此处", "前往此", "前往奈", "前往")
STOP_NEEDLES = (
    "停止",
    "取消移動",
    "取消移动",
    "停止移動",
    "停止移动",
    "取消自動",
    "取消自动",
    "取消移",
    "停止移",
    # OCR 常见残片
    "停正",
    "停走",
    "取消動",
)
DIALOG_NEEDLES = ("對話", "对话", "交談", "修理", "修復", "修复", "购买", "購買", "武器商人", "带著清爽", "帶著清爽", "再見")

# 地图 chrome：不当作地区名
_CHROME = (
    "地圖",
    "地图",
    "歐拉大陸",
    "欧拉大陆",
    "歐拉",
    "欧拉",
    "地下城",
    "世界",
    "關閉",
    "关闭",
    "返回",
    "搜尋",
    "搜索",
    "推薦",
    "推荐",
    "Home",
    "ESC",
    "End",
    "前往",
    "距離",
    "距离",
    "治療所",
    "治疗所",
    "教会",
    "教會",
    "武器店",
    "鐵匠",
    "铁匠",
    "旅馆",
    "旅館",
    "銀行",
    "银行",
    "商店",
)

_REGION_HINTS = (
    "杜巴頓",
    "杜巴顿",
    "杜巴頔",  # OCR
    "堤爾克那",
    "堤尔克那",
    "佛格斯",
    "艾明馬夏",
    "艾明马夏",
    "歐絲拉",
    "欧丝拉",
    "班克爾",
    "班克尔",
    "萊比",
    "莱比",
    "提爾",
    "提尔",
    "艾明",
    "史卡哈",
    "肯尼斯",
    "都柏林",
    "埃门",
    "北門",
    "北门",
)


@dataclass
class MapHit:
    kind: str  # text | icon
    text: str
    nx: float
    ny: float
    score: float = 0.0


@dataclass
class MapSnapshot:
    state: MapState
    blob: str
    lines: list[OcrLine] = field(default_factory=list)
    regions: list[MapHit] = field(default_factory=list)
    path: Path | None = None


def maps_root(cfg: AppConfig) -> Path:
    return cfg.resolve("knowledge/maps/ulara")


def load_whitelist(cfg: AppConfig) -> dict:
    p = maps_root(cfg) / "whitelist.json"
    if not p.is_file():
        return {"continent": "歐拉大陸", "npcs": []}
    return json.loads(p.read_text(encoding="utf-8"))


# 大地图拖找：库漢多在东 → 优先往西/西南多拖，再补其它方向
# 堤爾克那 → 杜巴頓：图鉴 ay 杜巴(1634) < 堤爾(2462)，需视野北移。
# 约定：手指 +(dx,dy) → 视野中心 -= (dx,dy)；故北移用手指下移 dy>0。
_SEEK_FROM_TIR_DELTAS: tuple[tuple[int, int], ...] = (
    (0, 480),
    (0, 480),
    (-360, 420),
    (360, 420),
    (0, 520),
    (-450, 280),
    (420, 300),
    (-500, 0),
    (0, 400),
    (480, 0),
    (0, 560),
    (-400, 360),
    (400, 360),
)

# 杜巴頓 town 内拖找鐵匠头像（步长略小）
_TOWN_SEEK_DRAG_DELTAS: tuple[tuple[int, int], ...] = (
    (0, 280),
    (0, -280),
    (-320, 0),
    (320, 0),
    (-280, 220),
    (280, 220),
    (-280, -220),
    (280, -220),
    (0, 360),
    (-400, 0),
)


def _seek_region_on_map(
    cfg: AppConfig,
    win: WindowInfo,
    needles: tuple[str, ...],
    *,
    max_drags: int = 8,
    lines: list[OcrLine] | None = None,
    min_ny: float = 0.09,
    deltas: tuple[tuple[int, int], ...] | None = None,
) -> tuple[WindowInfo, list[OcrLine], list[MapHit]]:
    """已退役：OCR 拖图找城。请用像素图鉴路径 / 方法 B 建图。"""
    del cfg, win, needles, max_drags, lines, min_ny, deltas
    raise RuntimeError(
        "_seek_region_on_map 已退役：禁止 OCR 拖图；用 ulara_atlas 像素路或已在詳图"
    )


def _seek_facility_on_town(
    cfg: AppConfig,
    win: WindowInfo,
    needles: tuple[str, ...],
    *,
    max_drags: int = 8,
    lines: list[OcrLine] | None = None,
    min_ny: float = 0.12,
) -> tuple[WindowInfo, list[OcrLine], list[MapHit]]:
    """已退役：OCR 拖图找设施。请用 town_pixel_click_npc。"""
    del cfg, win, needles, max_drags, lines, min_ny
    raise RuntimeError(
        "_seek_facility_on_town 已退役：禁止 OCR 拖图；用 town_pixel_click_npc"
    )


def load_places_list(cfg: AppConfig) -> list[dict]:
    """读 ulara/places.json 的 places 列表（失败返回空）。"""
    p = maps_root(cfg) / "places.json"
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    places = data.get("places") if isinstance(data, dict) else data
    return [it for it in places if isinstance(it, dict)] if isinstance(places, list) else []


def load_place_fallback(cfg: AppConfig, canon: str) -> MapHit | None:
    """OCR 找不到地名时，用 places.json 标定坐标点一次（非全图 pan）。

    注意：nx/ny 是某次扫图时的屏幕坐标，远距视野下不能当点击锚点；
    仅适合近距已露出杜巴周边时的拖中心，或顶栏已确认杜巴后的兜底点。
    """
    best = None
    for it in load_places_list(cfg):
        c = str(it.get("canon") or it.get("text") or "")
        if c != canon and canon not in c:
            continue
        # 精确城名胜过「杜巴頓北門」
        score = float(it.get("score") or 0) + (20.0 if c == canon else 0.0)
        hit = MapHit(
            kind="text",
            text=str(it.get("text") or c),
            nx=float(it["nx"]),
            ny=float(it["ny"]),
            score=score,
        )
        if best is None or hit.score > best.score:
            best = hit
    return best


# 远距拖找：画面上已见这些地标时，用 places 相对位移估杜巴頓方向（勿用绝对屏坐标盲点）
_LANDMARK_SEEK_KEYS: tuple[str, ...] = (
    "堤爾克那",
    "堤尔克那",
    "希德斯特",
    "雪原",
    "艾菲",
    "艾明",
    "莱比",
    "萊比",
    "杜加德",
    "库漢",
    "庫漢",
    "佛格",
    "伐木",
)

# 任务栏/周常文案：含地名但不是地图城标
_LANDMARK_QUEST_NOISE: tuple[str, ...] = (
    "淨化",
    "净化",
    "達成",
    "达成",
    "目標",
    "目标",
    "地區的",
    "地区的",
    "委記",
    "委托",
    "委託",
    "每週",
    "每周",
    "公會",
    "公会",
    "冒険",
    "冒險",
)


def _is_quest_noise_text(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return True
    if len(t) > 8:
        return True
    return any(k in t for k in _LANDMARK_QUEST_NOISE)


def _is_map_canvas_poi(nx: float, ny: float) -> bool:
    """地图画布内（排除顶栏面包屑 + 右侧任务栏）。"""
    return 0.06 <= nx <= 0.78 and 0.10 <= ny <= 0.82


def _place_canon_xy(places: list[dict], canon: str) -> tuple[float, float] | None:
    """精确 canon 优先，其次 canon 含子串。"""
    exact: tuple[float, float] | None = None
    soft: tuple[float, float] | None = None
    for it in places:
        c = str(it.get("canon") or it.get("text") or "")
        try:
            xy = (float(it["nx"]), float(it["ny"]))
        except (KeyError, TypeError, ValueError):
            continue
        if c == canon:
            exact = xy
        elif canon in c or c in canon:
            soft = soft or xy
    return exact or soft


def _place_is_target_family(canon: str, target: str) -> bool:
    """杜巴頓 / 杜巴頓北門 等同族，不能当地标。"""
    c, t = normalize_text(canon), normalize_text(target)
    if not c or not t:
        return False
    if c == t:
        return True
    if t in c or c in t:
        return True
    return "杜巴" in c and "杜巴" in t


def estimate_target_from_landmarks(
    cfg: AppConfig,
    lines: list[OcrLine],
    win: WindowInfo,
    *,
    target_canon: str = "杜巴頓",
) -> MapHit | None:
    """【deprecated】用地标相对位移估城标屏位；生产主路径已改像素图鉴。

    仅单测/对照。返回 kind=estimate 的 MapHit；估点可在屏外。
    """
    places = load_places_list(cfg)
    tgt = _place_canon_xy(places, target_canon)
    if not tgt:
        return None
    tnx, tny = tgt
    best: MapHit | None = None
    for it in places:
        canon = str(it.get("canon") or it.get("text") or "")
        if not canon or _place_is_target_family(canon, target_canon):
            continue
        if not any(k in canon for k in _LANDMARK_SEEK_KEYS):
            continue
        try:
            lnx, lny = float(it["nx"]), float(it["ny"])
        except (KeyError, TypeError, ValueError):
            continue
        needles = tuple(
            dict.fromkeys(
                [
                    canon,
                    str(it.get("text") or ""),
                    *[a for a in (canon[:2], canon[:3]) if len(a) >= 2],
                ]
            )
        )
        hits = [
            h
            for h in find_text_hits(lines, win, needles, min_ny=0.09)
            if _is_map_canvas_poi(h.nx, h.ny) and not _is_quest_noise_text(h.text)
        ]
        if not hits:
            # 宽松：扫到的地区名含地标关键字
            for h in scan_region_text_hits(lines, win):
                if not _is_map_canvas_poi(h.nx, h.ny) or _is_quest_noise_text(h.text):
                    continue
                ht = normalize_text(h.text)
                if any(
                    normalize_text(n) in ht
                    for n in needles
                    if len(normalize_text(n)) >= 2
                ):
                    hits.append(h)
        if not hits:
            continue
        h = hits[0]
        enx = h.nx + (tnx - lnx)
        eny = h.ny + (tny - lny)
        # 相对位移过大多半是不同扫图帧拼出来的噪声
        od = abs(tnx - lnx) + abs(tny - lny)
        if od > 0.75:
            continue
        score = float(h.score) + 5.0 / (1.0 + od) - 0.01 * (
            abs(enx - 0.5) + abs(eny - 0.48)
        )
        cand = MapHit(
            kind="estimate",
            text=f"{target_canon}←{h.text}",
            nx=enx,
            ny=eny,
            score=score,
        )
        if best is None or cand.score > best.score:
            best = cand
    return best


def far_landmarks_visible(lines: list[OcrLine], win: WindowInfo) -> bool:
    """画布上是否已见堤爾克那/雪原/艾菲等远距地标（此时禁止 places 绝对坐标盲点）。"""
    for h in scan_region_text_hits(lines, win):
        if not _is_map_canvas_poi(h.nx, h.ny) or _is_quest_noise_text(h.text):
            continue
        t = normalize_text(h.text)
        if any(k in t for k in _LANDMARK_SEEK_KEYS):
            return True
    return False


def title_other_city(lines: list[OcrLine], win: WindowInfo) -> str | None:
    """顶栏面包屑他城名（非杜巴）；用于强制点歐拉离开他城详图。"""
    keys = (
        ("堤爾", "堤爾克那"),
        ("堤尔", "堤爾克那"),
        ("库漢", "库漢"),
        ("庫漢", "库漢"),
        ("库汉", "库漢"),
        ("佛格", "佛格斯"),
        ("艾明", "艾明馬夏"),
        ("班克", "班克爾"),
        ("歐絲", "歐絲拉"),
        ("欧丝", "歐絲拉"),
    )
    for L in lines:
        c = box_center_norm(L, win.width, win.height)
        if not c or c[1] > 0.10 or c[0] > 0.55:
            continue
        t = normalize_text(L.text or "")
        if "杜巴" in t:
            return None
        for needle, name in keys:
            if needle in t:
                return name
    return None


_TOWN_FACILITY_KEYS: tuple[str, ...] = (
    "鐵匠",
    "铁匠",
    "旅館",
    "旅馆",
    "雜貨",
    "杂货",
    "銀行",
    "银行",
    "治療",
    "治疗",
    "學校",
    "学校",
    "工坊",
    "食品",
    "教會",
    "教会",
    "造船",
    "碼頭",
    "码头",
    "羊群",
    "兵事務",
    "交易所",
)


def looks_town_facility_blob(blob: str) -> bool:
    return any(k in blob for k in _TOWN_FACILITY_KEYS)


def looks_other_town_detail(blob: str, lines: list[OcrLine], win: WindowInfo) -> bool:
    """他城城镇详图：设施芯片为主；顶栏他城可加强，但缺顶栏时仍可判。"""
    if _looks_hud_chrome(blob):
        return False
    other = title_other_city(lines, win)
    # 顶栏明确他城 + 设施 = 城镇详图（面包屑常含歐拉／堤爾）
    if other is not None and looks_town_facility_blob(blob):
        return True
    # 歐拉中景/大陆：城内设施名会出现在地图 POI，勿当城镇详图
    if any(k in blob for k in ("歐拉大", "欧拉大", "歐拉大陸", "欧拉大陆", "欧拉大睦", "歐拉大睦")):
        return False
    if not looks_town_facility_blob(blob):
        return False
    # 大陆中景常见：多城/野外/地下城标签并存
    worldish = sum(
        1
        for k in (
            "莱比",
            "萊比",
            "希德",
            "瑪斯",
            "冰霜",
            "魔墟",
            "狩場",
            "狩场",
            "雪原",
            "伐木",
            "地下城",
        )
        if k in blob
    )
    if worldish >= 2:
        return False
    # 顶栏 OCR 漏城名，但有设施且不像中景 → 仍当城镇详图（如堤爾）
    if "杜巴" in blob[:60] and any(
        k in blob for k in ("鐵匠", "铁匠", "旅館", "银行", "銀行", "交易所")
    ):
        # 可能已是杜巴詳图；交给 _in_dubarton_town 近距路径
        return False
    return True


def dunbarton_town_path(cfg: AppConfig) -> Path:
    return maps_root(cfg) / "dunbarton_town.json"


# 店名 OCR → 圆头像：点店名上方（2026-07-28 实机 P匠铺@0.04 出卡；0.055 易空点）
_AVATAR_NY_OFFSET = 0.04

# 城镇详图：找鐵匠店名（OCR 常成「截匠铺」「P匠铺」）
_SMITH_SHOP_NEEDLES: tuple[str, ...] = (
    "截匠铺",
    "截匠舖",
    "截匠",
    "鐵匠铺",
    "鐵匠鋪",
    "铁匠铺",
    "鐵匠舖",
    "鐵匠",
    "铁匠",
    "P匠铺",
    "P匠舖",
    "匠铺",
    "匠舖",
)


def avatar_above_label(hit: MapHit, *, offset: float = _AVATAR_NY_OFFSET) -> MapHit:
    """店名/芯片上方一点 → 圆头像点击位。"""
    return MapHit(
        kind="avatar",
        text=hit.text,
        nx=hit.nx,
        ny=max(0.05, hit.ny - offset),
        score=hit.score,
    )


def find_smith_shop_hit(
    lines: list[OcrLine], win: WindowInfo, *, needles: tuple[str, ...] | None = None
) -> MapHit | None:
    """当前帧 OCR 铁匠店名（优先长匹配 / 截匠）。排除顶栏/底栏误匹配。"""
    use = needles or _SMITH_SHOP_NEEDLES
    hits = [h for h in find_text_hits(lines, win, use) if _is_smith_only(h.text)]
    if not hits:
        for L in lines:
            t = L.text or ""
            if not _is_smith_only(t):
                continue
            c = box_center_norm(L, win.width, win.height)
            if not c:
                continue
            hits.append(
                MapHit(
                    kind="text",
                    text=t,
                    nx=c[0],
                    ny=c[1],
                    score=8.0 + float(getattr(L, "confidence", 0.5) or 0.5),
                )
            )
    # 顶栏面包屑/底栏噪声：ny 过小或过大一律丢掉
    hits = [h for h in hits if 0.12 <= h.ny <= 0.88 and 0.05 <= h.nx <= 0.95]
    if not hits:
        return None
    # 偏中心 + 长匹配优先
    hits.sort(
        key=lambda h: (
            -len(normalize_text(h.text)),
            -h.score,
            abs(h.ny - 0.50) + abs(h.nx - 0.45),
        )
    )
    return hits[0]


def _read_dunbarton_town(cfg: AppConfig) -> dict | None:
    p = dunbarton_town_path(cfg)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def load_dunbarton_facility(
    cfg: AppConfig, needles: tuple[str, ...]
) -> MapHit | None:
    """读扫图标定的杜巴頓城镇设施大概位置（归一化 + 像素）。"""
    data = _read_dunbarton_town(cfg)
    if not data:
        return None
    facs = data.get("facilities")
    if not isinstance(facs, list):
        return None
    best: MapHit | None = None
    for it in facs:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text") or "")
        canon = str(it.get("canon") or text)
        blob = normalize_text(text + canon)
        if not any(normalize_text(n) in blob for n in needles if n):
            continue
        try:
            nx, ny = float(it["nx"]), float(it["ny"])
        except (KeyError, TypeError, ValueError):
            continue
        score = float(it.get("score") or 1.0) + (
            30.0 if any(normalize_text(n) in normalize_text(canon) for n in needles) else 0.0
        )
        hit = MapHit(kind="anchor", text=text or canon, nx=nx, ny=ny, score=score)
        if best is None or hit.score > best.score:
            best = hit
    return best


def load_dunbarton_avatar(
    cfg: AppConfig, needles: tuple[str, ...] = ("奈麗絲", "奈麗", "奈丽")
) -> MapHit | None:
    """读标定的杜巴頓 town 圆头像位置；优先 verified_card / source=pin。"""
    data = _read_dunbarton_town(cfg)
    if not data:
        return None
    avs = data.get("avatars")
    if not isinstance(avs, list):
        return None
    best: MapHit | None = None
    best_rank = -1
    for it in avs:
        if not isinstance(it, dict):
            continue
        canon = str(it.get("canon") or it.get("text") or "")
        blob = normalize_text(canon + str(it.get("text") or ""))
        if not any(normalize_text(n) in blob for n in needles if n):
            continue
        try:
            nx, ny = float(it["nx"]), float(it["ny"])
        except (KeyError, TypeError, ValueError):
            continue
        src = str(it.get("source") or "")
        verified = bool(it.get("verified_card"))
        rank = (3 if verified else 0) + (2 if src == "pin" else 1 if src == "scan_offset" else 0)
        score = float(it.get("score") or 1.0) + 40.0 + rank * 10.0
        hit = MapHit(kind="avatar", text=canon or "avatar", nx=nx, ny=ny, score=score)
        if rank > best_rank or (rank == best_rank and (best is None or hit.score > best.score)):
            best = hit
            best_rank = rank
    return best


def avatar_row_from_norm(
    *,
    canon: str,
    nx: float,
    ny: float,
    facility: str = "鐵匠铺",
    source: str = "pin",
    verified_card: bool = False,
    score: float = 10.0,
    text: str = "",
) -> dict:
    """构造 avatars[] 一条（标定分辨率像素）。"""
    return {
        "canon": canon,
        "facility": facility,
        "text": text or canon,
        "nx": round(float(nx), 4),
        "ny": round(float(ny), 4),
        "px": int(round(float(nx) * _CAL_W)),
        "py": int(round(float(ny) * _CAL_H)),
        "source": source,
        "verified_card": bool(verified_card),
        "score": round(float(score), 3),
    }


def upsert_dunbarton_avatar(cfg: AppConfig, row: dict, *, replace: bool = True) -> Path:
    """写入/更新 dunbarton_town.json 的 avatars 项（同 canon 合并）。"""
    data = _read_dunbarton_town(cfg) or {
        "region": "杜巴頓",
        "design_resolution": [_CAL_W, _CAL_H],
        "facilities": [],
        "avatars": [],
    }
    avs: list = list(data.get("avatars") or [])
    canon = str(row.get("canon") or "")
    idx = next(
        (i for i, a in enumerate(avs) if isinstance(a, dict) and a.get("canon") == canon),
        None,
    )
    if idx is not None:
        if not replace:
            raise ValueError(f"avatar {canon!r} already exists")
        # 扫图候选不覆盖已验证 pin
        old = avs[idx]
        if (
            isinstance(old, dict)
            and old.get("verified_card")
            and row.get("source") == "scan_offset"
            and not row.get("verified_card")
        ):
            pass
        else:
            avs[idx] = row
    else:
        avs.append(row)
    return save_dunbarton_town_json(
        cfg,
        list(data.get("facilities") or []),
        avatars=avs,
        note=str(data.get("note") or "avatar upsert"),
    )


def drag_norm_to_center(win: WindowInfo, nx: float, ny: float) -> None:
    """把屏幕归一化点拖到客户区中心（露出该处附近内容）。只拖不缩放。"""
    from mm_assist.nav.compat import in_test_mode
    from mm_assist.nav.compat import production_click_method

    cx, cy = win.width // 2, int(win.height * 0.48)
    x0 = int(max(80, min(win.width - 80, nx * win.width)))
    y0 = int(max(120, min(win.height - 120, ny * win.height)))
    # 贴顶标签：clamp 后位移仍可能偏短，强制加大纵向拖
    if ny < 0.14:
        y0 = int(win.height * 0.18)
        cy = int(win.height * 0.62)
    elif ny < 0.28:
        # 未进安全带：拉长纵距，避免城堡仍在顶栏外
        y0 = int(win.height * 0.26)
        cy = int(win.height * 0.58)
    if ny > 0.82:
        y0 = int(win.height * 0.78)
        cy = int(win.height * 0.45)
    if in_test_mode():
        # 与 _drag_map 一致：town 实机 sendinput 拖有效
        method, refocus = "sendinput", True
    else:
        method, refocus = production_click_method()
    drag_client(
        win,
        x0,
        y0,
        cx,
        cy,
        method=method,
        refocus=refocus,
        steps=16,
        duration_s=0.55,
    )


# 点城：文字须落在此带内，城堡才在可点区（贴顶时城堡进顶栏点不中）
_DRILL_LABEL_NY_LO = 0.28
_DRILL_LABEL_NY_HI = 0.58
_CASTLE_CLICK_NY_MIN = 0.18


def label_in_drill_zone(ny: float) -> bool:
    return _DRILL_LABEL_NY_LO <= ny <= _DRILL_LABEL_NY_HI


def is_duba_city_label(text: str) -> bool:
    """可点进详图的城名（杜巴頓/頔），不含北門 POI。"""
    t = normalize_text(text or "")
    if "北" in t:
        return False
    # 排除任务/NPC「杜巴顿·韓信」等
    if "·" in t or "・" in t or "韓" in t or "韩" in t:
        return False
    # 排除面包屑「歐拉大陸／杜巴頓」合并行
    if any(k in t for k in ("歐拉", "欧拉", "大睦", "大陸", "大陆", "／", "/")):
        return False
    if len(t) > 8:
        return False
    return any(k in t for k in ("杜巴頓", "杜巴顿", "杜巴頔")) or t in (
        "杜巴",
        "尚杜巴",
    )


def is_duba_north_gate(text: str) -> bool:
    t = normalize_text(text or "")
    return "杜巴" in t and "北" in t


def estimate_city_from_north_gate(nx: float, ny: float) -> tuple[float, float]:
    """北門东南侧为城标（places 共现 Δ≈+0.08,+0.11）。"""
    return (
        min(0.92, max(0.08, float(nx) + 0.08)),
        min(0.90, max(0.12, float(ny) + 0.11)),
    )


def castle_click_norm(
    nx: float,
    label_ny: float,
    *,
    off: float = -0.06,
    dx: float = 0.0,
) -> tuple[float, float]:
    """城标文字上方的城堡点击坐标（归一化）；禁止点进顶栏。"""
    cy = label_ny + off
    if cy < _CASTLE_CLICK_NY_MIN:
        cy = _CASTLE_CLICK_NY_MIN
    # 文字仍偏上：点击至少比文字高一点，但不得低于安全带
    if label_ny < 0.28:
        cy = max(_CASTLE_CLICK_NY_MIN, min(cy, label_ny - 0.04))
    nx2 = min(0.75, max(0.08, nx + dx))
    ny2 = min(0.80, max(_CASTLE_CLICK_NY_MIN, cy))
    return nx2, ny2


def drag_label_into_drill_zone(
    win: WindowInfo,
    nx: float,
    ny: float,
    *,
    rounds: int = 4,
) -> None:
    """把城名标签拖进可点城堡的纵向带；贴顶/贴底时加大步。"""
    for i in range(max(1, rounds)):
        if label_in_drill_zone(ny):
            break
        if ny < _DRILL_LABEL_NY_LO:
            # 越贴顶步越大（城堡在文字上方，易进顶栏）
            frac = 0.42 if ny < 0.14 else (0.36 if i == 0 else 0.26)
            step = int(win.height * frac)
            _drag_map(win, dx=0, dy=step, duration_s=0.90)
            ny = min(0.55, ny + step / max(win.height, 1))
        elif ny > _DRILL_LABEL_NY_HI:
            step = int(win.height * (0.32 if i == 0 else 0.22))
            _drag_map(win, dx=0, dy=-step, duration_s=0.85)
            ny = max(0.30, ny - step / max(win.height, 1))
        else:
            break
    if abs(nx - 0.5) > 0.15 or not label_in_drill_zone(ny):
        drag_norm_to_center(win, nx, min(0.52, max(0.32, ny)))


def save_dunbarton_town_json(
    cfg: AppConfig,
    facilities: list[dict],
    *,
    avatars: list[dict] | None = None,
    note: str = "",
) -> Path:
    out = dunbarton_town_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = _read_dunbarton_town(cfg) or {}
    if avatars is None:
        avatars = list(prev.get("avatars") or [])
    payload = {
        "region": "杜巴頓",
        "design_resolution": [_CAL_W, _CAL_H],
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note
        or "town pan-scan; nx/ny 为扫到时屏幕归一化，px/py 为标定像素；avatars 为圆头像",
        "count": len(facilities),
        "facilities": facilities,
        "avatars": avatars,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


_TOWN_FACILITY_CANON = (
    ("鐵匠铺", ("鐵匠", "铁匠", "截匠")),
    ("旅館", ("旅館", "旅馆")),
    ("雜貨店", ("雜貨", "杂货")),
    ("銀行", ("銀行", "银行")),
    ("治療所", ("治療", "治疗")),
    ("交易所", ("交易所",)),
    ("奈麗絲", ("奈麗", "奈丽")),
)


def _canon_town_facility(text: str) -> str | None:
    t = normalize_text(text)
    for canon, keys in _TOWN_FACILITY_CANON:
        if any(k in t for k in keys):
            return canon
    return None


def _merge_avatar_candidate(store: dict[str, dict], row: dict) -> None:
    """同 canon 保留更高分；verified/pin 优先于 scan_offset。"""
    canon = str(row.get("canon") or "")
    if not canon:
        return
    prev = store.get(canon)
    if prev is None:
        store[canon] = row
        return
    if prev.get("verified_card") and not row.get("verified_card"):
        return
    prev_rank = (3 if prev.get("verified_card") else 0) + (
        2 if prev.get("source") == "pin" else 1 if prev.get("source") == "scan_offset" else 0
    )
    new_rank = (3 if row.get("verified_card") else 0) + (
        2 if row.get("source") == "pin" else 1 if row.get("source") == "scan_offset" else 0
    )
    if new_rank > prev_rank or (
        new_rank == prev_rank
        and float(row.get("score") or 0) >= float(prev.get("score") or 0)
    ):
        store[canon] = row


def scan_dunbarton_town(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    max_drags: int = 10,
    settle: float = 0.55,
) -> tuple[Path, list[dict], list[dict]]:
    """在杜巴頓 town 详图内拖扫，记录设施 + 奈麗絲头像候选。

    要求当前已是杜巴頓 town（调用方保证）；只点击拖动，不缩放。
    返回 (path, facilities, avatars)。
    """
    merged: dict[str, dict] = {}
    avatars_merged: dict[str, dict] = {}
    # 设施：与上次合并，避免单轮漏扫抹掉已有点
    prev = _read_dunbarton_town(cfg) or {}
    for f in prev.get("facilities") or []:
        if isinstance(f, dict) and f.get("canon"):
            merged[str(f["canon"])] = f
    # 仅保留已验证 pin；扫图候选每轮重算，避免陈旧种子挡住新坐标
    for a in prev.get("avatars") or []:
        if not isinstance(a, dict) or not a.get("canon"):
            continue
        if a.get("verified_card") or a.get("source") == "pin":
            _merge_avatar_candidate(avatars_merged, a)

    def _ingest(tag: str) -> None:
        nonlocal win
        win, _img, lines, _ = snap(cfg, win)
        for h in scan_region_text_hits(lines, win):
            canon = _canon_town_facility(h.text)
            if not canon:
                continue
            px = int(round(h.nx * _CAL_W))
            py = int(round(h.ny * _CAL_H))
            row = {
                "canon": canon,
                "text": h.text,
                "nx": round(h.nx, 4),
                "ny": round(h.ny, 4),
                "px": px,
                "py": py,
                "score": round(h.score, 3),
                "seen_at": tag,
            }
            prev_f = merged.get(canon)
            if prev_f is None or row["score"] > float(prev_f.get("score") or 0):
                merged[canon] = row
            if canon == "奈麗絲":
                _merge_avatar_candidate(
                    avatars_merged,
                    avatar_row_from_norm(
                        canon="奈麗絲",
                        nx=h.nx,
                        ny=max(0.05, h.ny - _AVATAR_NY_OFFSET),
                        facility="鐵匠铺",
                        source="scan_offset",
                        score=h.score + 5.0,
                        text=h.text,
                    ),
                )
            elif canon == "鐵匠铺":
                _merge_avatar_candidate(
                    avatars_merged,
                    avatar_row_from_norm(
                        canon="奈麗絲",
                        nx=h.nx,
                        ny=max(0.05, h.ny - _AVATAR_NY_OFFSET),
                        facility="鐵匠铺",
                        source="scan_offset",
                        score=h.score,
                        text=h.text,
                    ),
                )
        log.info(
            "town_scan %s unique=%d avatars=%d",
            tag,
            len(merged),
            len(avatars_merged),
        )

    _ingest("origin")
    sx = win.width / _CAL_W
    sy = win.height / _CAL_H
    for i, (dx, dy) in enumerate(_TOWN_SEEK_DRAG_DELTAS[:max_drags]):
        _drag_map(win, dx=int(dx * sx), dy=int(dy * sy))
        time.sleep(settle)
        _ingest(f"drag{i+1}")

    facilities = sorted(merged.values(), key=lambda r: -float(r.get("score") or 0))
    avatars = sorted(
        avatars_merged.values(), key=lambda r: -float(r.get("score") or 0)
    )
    path = save_dunbarton_town_json(
        cfg,
        facilities,
        avatars=avatars,
        note=f"scan_dunbarton_town drags={max_drags}",
    )
    return path, facilities, avatars


def _blob(lines: list[OcrLine]) -> str:
    return normalize_text("".join(L.text for L in lines))


def _scale_xy(win: WindowInfo, cx: int, cy: int) -> tuple[int, int]:
    """标定分辨率坐标 → 当前客户区像素。"""
    return (
        int(cx * win.width / _CAL_W),
        int(cy * win.height / _CAL_H),
    )


def _click(
    win: WindowInfo,
    cx: int,
    cy: int,
    *,
    hold_s: float = 0.08,
    method: str = "sendinput",
) -> None:
    """默认走生产点击方式；测试模式默认 sendinput，显式 unity_hook 可失焦。

    状态栏一键修开了 prefer_visible_clicks 时：强制 sendinput（系统鼠标会动），
    仅在失焦时补一次 focus（不算反复抢前台）。
    """
    from mm_assist.nav.compat import check_abort
    from mm_assist.nav.compat import in_test_mode, prefer_visible_clicks
    from mm_assist.nav.compat import production_click_method
    from mm_assist.platform.window import is_foreground

    check_abort()
    t0 = time.monotonic()
    m = (method or "sendinput").lower()
    refocus = False
    from mm_assist.safety.input import foreground_input_enabled
    if foreground_input_enabled(win.hwnd):
        m = 'sendinput'
    elif in_test_mode():
        if m not in ("unity_hook", "hook"):
            m = "sendinput"
        refocus = _steal_focus_allowed() and m in ("sendinput", "si")
    elif prefer_visible_clicks() and m not in ("unity_hook", "hook", "postmessage", "post"):
        m = "sendinput"
        if not is_foreground(win.hwnd):
            focus_window(win.hwnd)
            time.sleep(0.05)
        refocus = False
    elif m in ("sendinput", "si"):
        m, _ = production_click_method()
        if m in ("sendinput", "si") and _steal_focus_allowed():
            focus_window(win.hwnd)
        refocus = _steal_focus_allowed() and m in ("sendinput", "si")
    log.info("map_click (%d,%d) method=%s refocus=%s", int(cx), int(cy), m, refocus)
    click_client(
        win,
        cx,
        cy,
        method=m,
        refocus=refocus,
        hold_s=hold_s,
    )
    if prefer_visible_clicks():
        from mm_assist.tool_repair import pace_op_step

        pace_op_step(t0)


def _click_cal(
    win: WindowInfo,
    cx: int,
    cy: int,
    *,
    hold_s: float = 0.08,
    method: str = "sendinput",
) -> None:
    """点击标定坐标（1920×1440），自动按窗口缩放。"""
    from mm_assist.nav.compat import check_abort

    check_abort()
    sx, sy = _scale_xy(win, cx, cy)
    _click(win, sx, sy, hold_s=hold_s, method=method)


def _wait_foreground_before_click() -> None:
    """切到游戏前台后先等 1s 再点（Unity 未就绪会吞点击）。"""
    try:
        from mm_assist.nav.compat import in_test_mode

        if in_test_mode():
            return
    except Exception:
        pass
    time.sleep(1.0)


def _focus_for_map_open(win: WindowInfo) -> bool:
    """开图前对焦：sendinput 须前台；禁失焦 unity_hook 点空后记 map_open_fail。

    返回是否已在前台——未对上时调用方必须放弃点击（否则点到人的窗口上）。

    状态栏「一键修理」会 ``prefer_visible_clicks``：点按钮瞬间 human_busy，须
    ``force`` 抢焦，否则 focus_fail 假「开图失败」。
    """
    from mm_assist.platform.window import bring_to_foreground, focus_window, is_foreground

    force = False
    try:
        from mm_assist.nav.compat import in_production_mode, prefer_visible_clicks

        force = bool(prefer_visible_clicks() or in_production_mode())
    except Exception:
        force = False
    try:
        if is_foreground(win.hwnd):
            _wait_foreground_before_click()
            return True
        bring_to_foreground(win.hwnd, retries=3, settle_s=0.08, force=force)
        if not is_foreground(win.hwnd):
            focus_window(win.hwnd, force=force)
            time.sleep(0.05)
        ok = bool(is_foreground(win.hwnd))
        if ok:
            _wait_foreground_before_click()
        return ok
    except Exception:
        return False


def _click_map_open(
    win: WindowInfo,
    cx: int,
    cy: int,
    *,
    hold_s: float = 0.10,
    cal: bool = False,
) -> str:
    """开图专用点击：强制 sendinput + 对焦；禁被 production 改写为 unity_hook。

    返回 ``ok`` / ``focus_fail`` / ``yield_focus``。``focused=False`` 一律不点。
    """
    from mm_assist.safety.input import release_mouse_buttons
    from mm_assist.nav.compat import check_abort

    check_abort()
    focused = _focus_for_map_open(win)
    release_mouse_buttons()
    sx, sy = _scale_xy(win, cx, cy) if cal else (int(cx), int(cy))
    if not focused:
        blocked = ""
        try:
            from mm_assist.platform.window import focus_steal_block_reason

            blocked = str(focus_steal_block_reason() or "")
        except Exception:
            blocked = ""
        extra = f" yield_focus={blocked}" if blocked else ""
        log.warning(
            "map_open_click focus_fail (%d,%d)：未对焦不点%s",
            sx,
            sy,
            extra,
        )
        return "focus_fail"
    log.info("map_open_click (%d,%d) method=sendinput focus=1", sx, sy)
    clicked = click_client(win, sx, sy, method="sendinput", refocus=False, hold_s=hold_s)
    if not clicked:
        log.warning("map_open_click yield_focus (%d,%d)：让行未点", sx, sy)
        return "yield_focus"
    return "ok"


def _click_norm(
    win: WindowInfo,
    nx: float,
    ny: float,
    *,
    method: str = "sendinput",
) -> None:
    from mm_assist.nav.compat import check_abort

    check_abort()
    _click(win, int(nx * win.width), int(ny * win.height), method=method)


def _steal_focus_allowed() -> bool:
    """一键修/长挂 production：禁止抢前台（Q1=A）。Agent 测试模式仍可。"""
    from mm_assist.nav.compat import in_production_mode, in_test_mode

    if in_production_mode():
        return False
    if in_test_mode():
        return True
    # 非 production 的长挂配置：也不主动抢（unity_hook 可失焦）
    return False


def _map_ui_settle(kind: str = "step") -> float:
    """地图链路最长等待（点/拖后等画面变化的 timeout 上限）。

    kind: open | drag | click | breadcrumb | after_goto
    """
    from mm_assist.nav.compat import in_production_mode, prefer_visible_clicks

    # 可见/长挂：仍给足动画时间，但靠像素变化早停，不再盲睡满额
    fast = {
        "open": 2.8,
        "drag": 1.4,
        "click": 2.0,
        "breadcrumb": 2.0,
        "after_goto": 0.55,
        "step": 1.6,
    }
    slow = {
        "open": 3.5,
        "drag": 2.0,
        "click": 2.5,
        "breadcrumb": 2.5,
        "after_goto": 1.0,
        "step": 2.0,
    }
    table = fast if (prefer_visible_clicks() or in_production_mode()) else slow
    return float(table.get(kind, table["step"]))


def _grab_map_frame(
    cfg: AppConfig, win: WindowInfo, *, settle_s: float = 0.05
) -> tuple[WindowInfo, np.ndarray]:
    """图鉴定位用：只截屏，不 OCR。"""
    return _foreground_grab(cfg, win, settle_s=settle_s)


def _wait_action_effect(
    cfg: AppConfig,
    win: WindowInfo,
    before_img: np.ndarray | None,
    *,
    timeout_s: float | None = None,
    threshold: float = 2.8,
    poll_s: float = 0.12,
    roi: tuple[float, float, float, float] | None = None,
) -> tuple[WindowInfo, np.ndarray | None, bool, float]:
    """游戏动作后等到画面相对 before 变化，再让调用方继续。

    返回 (win, last_img, changed, delta)。before 为空则先抓一帧当基准。
    点击类用这个；拖图请用 ``_wait_frame_stable``（等停稳再识别）。
    """
    from mm_assist.nav.click_verify import make_sig, wait_pixel_change

    if before_img is None or getattr(before_img, "size", 0) == 0:
        win, before_img = _grab_map_frame(cfg, win, settle_s=0.0)
    before = make_sig(before_img, roi=roi)
    last_img: np.ndarray | None = before_img
    to = float(_map_ui_settle("step") if timeout_s is None else timeout_s)

    def grab() -> np.ndarray:
        nonlocal win, last_img
        win, last_img = _grab_map_frame(cfg, win, settle_s=0.0)
        assert last_img is not None
        return last_img

    changed, delta, _ = wait_pixel_change(
        grab,
        before,
        threshold=threshold,
        timeout_s=to,
        poll_s=poll_s,
        roi=roi,
    )
    log.info(
        "map action effect changed=%s delta=%.2f thr=%.2f timeout=%.2f",
        changed,
        delta,
        threshold,
        to,
    )
    return win, last_img, changed, float(delta)


def _wait_frame_stable(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    timeout_s: float | None = None,
    stable_thr: float = 1.8,
    need_stable: int = 2,
    poll_s: float = 0.10,
    min_wait_s: float = 0.08,
    roi: tuple[float, float, float, float] | None = None,
) -> tuple[WindowInfo, np.ndarray | None, bool]:
    """拖图后等到连续帧几乎不变（动画停稳），再做图鉴/OCR 识别。

    返回 (win, last_img, stable_ok)。
    """
    from mm_assist.nav.click_verify import make_sig, sig_delta

    to = float(_map_ui_settle("drag") if timeout_s is None else timeout_s)
    if min_wait_s > 0:
        time.sleep(min_wait_s)
    win, prev_img = _grab_map_frame(cfg, win, settle_s=0.0)
    prev = make_sig(prev_img, roi=roi)
    last_img: np.ndarray | None = prev_img
    t0 = time.time()
    stable_n = 0
    while time.time() - t0 < to:
        time.sleep(poll_s)
        win, last_img = _grab_map_frame(cfg, win, settle_s=0.0)
        cur = make_sig(last_img, roi=roi)
        d = sig_delta(prev, cur)
        if d < float(stable_thr):
            stable_n += 1
            if stable_n >= int(need_stable):
                log.info(
                    "map frame stable delta=%.2f after=%.2fs",
                    d,
                    time.time() - t0,
                )
                return win, last_img, True
        else:
            stable_n = 0
        prev = cur
    log.warning("map frame not stable within %.2fs (识别可能偏)", to)
    return win, last_img, False


def _snap_map_light(
    cfg: AppConfig, win: WindowInfo
) -> tuple[WindowInfo, np.ndarray, list[OcrLine]]:
    """开图后轻量确认：全帧截屏 + 顶栏/中下 ROI OCR（避免全屏 OCR 卡住）。"""
    from mm_assist.perception.ocr import ocr_roi

    win, img = _grab_map_frame(cfg, win, settle_s=0.04)
    lines: list[OcrLine] = []
    if img is None or getattr(img, "size", 0) == 0:
        return win, img, lines
    # 顶栏面包屑
    lines.extend(ocr_roi(img, (0.0, 0.0, 0.60, 0.14), min_confidence=0.22))
    # 城镇设施芯片带（中下）
    lines.extend(ocr_roi(img, (0.05, 0.50, 0.90, 0.40), min_confidence=0.22))
    return win, img, lines


def _grab_navigation_client(cfg: AppConfig, win: WindowInfo) -> np.ndarray:
    """Map auto capture uses a fresh rendered frame, with no shared-backend change.

    BitBlt can keep returning a nonblack old HUD while MapUIState is open.
    PrintWindow was independently checked against the visible client frame.
    Explicit capture settings remain authoritative; failure stops navigation.
    """
    from mm_assist.platform.capture import Grabber

    method = str(cfg.window.capture_method or 'auto').lower()
    grabber = Grabber('printwindow' if method == 'auto' else method)
    try:
        return grabber.grab_client(win)
    finally:
        grabber.close()


def _foreground_grab(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    settle_s: float = 0.25,
) -> tuple[WindowInfo, np.ndarray]:
    """地图链路截「游戏框」。production 不置前（PrintWindow/mss 可读）。"""
    steal = _steal_focus_allowed()
    if steal:
        focus_window(win.hwnd)
        time.sleep(max(0.0, settle_s))
    elif settle_s > 0:
        time.sleep(min(0.05, settle_s))
    win = refresh_window(win)
    from mm_assist.perception.game_frame import get_frame

    fr = get_frame()
    if fr is not None:
        fr.refresh()
        win = fr.to_window()
    img = _grab_navigation_client(cfg, win)
    from mm_assist.safety.input import foreground_input_enabled
    if steal and not foreground_input_enabled(win.hwnd) and not is_foreground(win.hwnd):
        focus_window(win.hwnd)
        time.sleep(max(0.05, settle_s * 0.5))
        win = refresh_window(win)
        img = _grab_navigation_client(cfg, win)
    return win, img


def snap(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    min_conf: float = 0.30,
    save_prefix: str | None = None,
    grab_settle_s: float = 0.25,
) -> tuple[WindowInfo, np.ndarray, list[OcrLine], Path | None]:
    from mm_assist.nav.compat import check_abort

    check_abort()
    win, img = _foreground_grab(cfg, win, settle_s=grab_settle_s)
    lines = ocr_image(img, min_confidence=min_conf)
    path = None
    if save_prefix:
        path = save_capture(img, cfg.resolve(cfg.paths.captures), prefix=save_prefix)
    return win, img, lines, path


def _is_inventory_blob(b: str) -> bool:
    """装备/背包面板（勿当地图 town）。"""
    marks = ("自動配戴", "自动配戴", "武器符", "防具符", "飾品符", "饰品符", "寶石·符文", "宝石·符文")
    return sum(1 for k in marks if k in b) >= 2


def _is_craft_ui_blob(b: str) -> bool:
    """生活工具/製作面板（遮住小地图，开图前须关掉）。"""
    marks = (
        "所需材料",
        "前往製作",
        "前往制作",
        "建立任務",
        "建立任务",
        "生活工具",
        "製作成功率",
        "制作成功率",
        "未持有",
        "多用途製作台",
        "多用途制作台",
    )
    if sum(1 for k in marks if k in b) >= 2:
        return True
    # 顶栏「工具」+ 分类页
    if ("工具" in b[:40] or "工具" in b[:80]) and any(
        k in b for k in ("全部", "生活工具", "改良型", "藥草鋤", "药草锄")
    ):
        return True
    return False


def _is_npc_dialog_blob(b: str) -> bool:
    """NPC 对话层（遮住 HUD/小地图）。"""
    if any(
        k in b
        for k in (
            "跳過對話",
            "跳过对话",
            "跳過对話",
            "跳對话",
            "跳对话",
            "跳對話",
            "跳对話",
        )
    ):
        return True
    if any(k in b for k in ("結束對話", "结束对话", "結束对話", "糙束")):
        return True
    # 对话选项菜单（无跳過时也要关）
    if any(k in b for k in ("随便聊聊", "隨便聊聊")) and any(
        k in b for k in ("商店", "結束", "结束", "修理")
    ):
        return True
    # 仅有对白气泡、无地图/任务栏强信号
    if "對話" in b or "对话" in b or "對话" in b:
        if not any(k in b for k in ("每週", "每周", "目標", "地图", "地圖", "歐拉")):
            return True
    return False


def _is_quest_offer_blob(b: str) -> bool:
    """兼职/任务接取层：接受 + 之後再做（遮住小地图）。OCR 常糊成 兼藏/之筱再做。"""
    has_later = ("再做" in b) or any(
        k in b for k in ("之後再", "之后再", "以後再", "以后再", "之筱再", "之筱")
    )
    has_accept = "接受" in b
    has_part = any(k in b for k in ("兼職", "兼职", "兼联", "兼藏", "兼職", "兼联"))
    return (has_later and has_accept) or (has_part and has_accept)


_QUEST_LATER_NEEDLES = (
    "之後再做",
    "之后再做",
    "以後再做",
    "以后再做",
    "之筱再做",
    "之筱再",
    "之後再",
    "之后再",
    "再做",
)
# 1920×1440 底栏「之後再做」近似中心（接受在左）
_QUEST_LATER_CAL_XY = (1105, 1285)


_SKIP_DIALOG_CLICK_NEEDLES = (
    "跳過對話",
    "跳过对话",
    "跳過对話",
    "跳對话",
    "跳对话",
    "跳對話",
    "跳对話",
    "跳過",
    "跳过",
    "跳對",
    "跳对",
    "結束對話",
    "结束对话",
    "結束对話",
    "糙束對话",
    "糙束對話",
    "糙束",
    "結束",
    "结束",
)
# 1920×1440 右上「跳過對話」近似中心（OCR 常糊成跳對话）
_SKIP_DIALOG_CAL_XY = (1768, 78)
# 对话底栏「結束對話」常见区
_END_DIALOG_CAL_XY = (1766, 1243)


def _is_town_directory_blob(b: str) -> bool:
    """城镇「主要」设施目录/图例（非世界地图；勿当 town 详图）。"""
    if "主要" not in b:
        return False
    marks = (
        "帽子開關",
        "帽子开关",
        "成長指南",
        "成长指南",
        "愛心商店",
        "爱心商店",
        "任務佈告",
        "任務怖告",
        "证物交换",
    )
    return sum(1 for k in marks if k in b) >= 2


def _map_opened_blob(b: str) -> bool:
    """强信号：勿用任务栏「地下城」误判。"""
    if _is_inventory_blob(b):
        return False
    if _is_craft_ui_blob(b):
        return False
    if _is_town_directory_blob(b):
        return False
    if _looks_hud_chrome(b):
        return False
    if any(k in b for k in ("歐拉大陸", "欧拉大陆", "世界地圖", "世界地图")):
        return True
    if "地圖" in b or "地图" in b:
        # 排除任务/聊天里偶发「地下城」而无地图标题
        if "Home" in b and "ESC" in b and "歐拉" not in b and "欧拉" not in b:
            # HUD 仍可见系统钮时，通常地图未盖住全屏
            if "理科" in b or "王牌" in b:
                return False
        return True
    return False


def _looks_hud_chrome(b: str) -> bool:
    """主界面 HUD（地图未盖住）：Home/ESC + 侧栏/任务噪声，勿当地图大陆层。"""
    if "Home" not in b or "ESC" not in b:
        return False
    if any(k in b for k in ("歐拉大陸", "欧拉大陆", "欧拉大睦", "歐拉大睦")):
        return False
    # 侧栏/技能栏常见字
    return any(
        k in b
        for k in (
            "王牌",
            "理科",
            "疾風",
            "疾风",
            "LEV",
            "每週目標",
            "每周目标",
            "每凋目標",
            "每调目標",
            "每遇自",
        )
    )


def classify_state(lines: list[OcrLine], img: np.ndarray | None = None) -> MapState:
    """五态 + dialog；实机文案可在 world_map.md 增补后同步 needles。"""
    b = _blob(lines)
    texts = [normalize_text(L.text) for L in lines]

    if any(n in b for n in DIALOG_NEEDLES) and (
        "選擇" in b or "选择" in b or "確認" in b or "确认" in b or "關閉" in b or "关闭" in b
    ):
        # 弱启发：对话常与选项并存；停走优先于对话误判
        if not any(n in b for n in STOP_NEEDLES):
            if "修理" in b or "對話" in b or "对话" in b or "交談" in b:
                return "dialog"

    if any(n in b for n in STOP_NEEDLES):
        return "traveling"

    if any(n in b for n in GOTO_NEEDLES) and (
        any(a in b for a in ("奈麗", "奈丽", "武器店", "鐵匠", "截匠", "杜巴"))
        or _has_nearby_name_card(texts)
    ):
        return "npc_card"

    if any(n in b for n in GOTO_NEEDLES):
        return "npc_card"

    has_continent_title = any(n in b for n in CONTINENT_NEEDLES) or "欧拉大睦" in b or "歐拉大睦" in b
    has_map = _map_opened_blob(b)
    regionish = sum(1 for h in _REGION_HINTS if h in b)
    facility = any(
        k in b
        for k in (
            "鐵匠",
            "铁匠",
            "截匠",
            "旅館",
            "旅馆",
            "學校",
            "学校",
            "乳牛",
            "北門",
            "北门",
            "防具",
            "皮革加工",
        )
    )
    # 面包屑「歐拉大陸 / 杜巴頓」+ 设施 = 地区图；点歐拉大陸后才是全图
    if has_map or has_continent_title:
        if facility or ("兼職" in b or "兼职" in b):
            if regionish <= 2 and (facility or "杜巴" in b):
                return "region"
        if regionish >= 3:
            return "continent"
        if has_continent_title and not facility:
            return "continent"
        if facility:
            return "region"
        return "continent"

    # HUD：有小地图区但无大地图标题（弱）
    if img is not None:
        h, w = img.shape[:2]
        corner = img[0 : max(1, h // 6), int(w * 0.85) : w]
        if float(corner.std()) > 8 and not has_map:
            return "hud"

    if not has_map and regionish == 0:
        return "hud"
    return "unknown"


def _has_nearby_name_card(texts: list[str]) -> bool:
    return any(len(t) >= 2 and len(t) <= 8 and re.search(r"[\u4e00-\u9fff]", t) for t in texts)


def find_text_hits(
    lines: list[OcrLine],
    win: WindowInfo,
    needles: tuple[str, ...],
    *,
    min_ny: float = 0.0,
    max_ny: float = 1.0,
) -> list[MapHit]:
    hits: list[MapHit] = []
    for L in lines:
        t = normalize_text(L.text)
        if not t:
            continue
        score = 0.0
        for n in needles:
            nn = normalize_text(n)
            if not nn:
                continue
            if t == nn:
                score = max(score, 10.0 + L.confidence)
            elif nn in t or t in nn:
                score = max(score, 6.0 + L.confidence)
        if score <= 0:
            continue
        c = box_center_norm(L, win.width, win.height)
        if not c:
            continue
        nx, ny = c
        if ny < min_ny or ny > max_ny:
            continue
        hits.append(MapHit(kind="text", text=L.text, nx=nx, ny=ny, score=score))
    hits.sort(key=lambda h: -h.score)
    return hits


def _is_chrome(text: str) -> bool:
    t = normalize_text(text)
    if not t or len(t) < 2:
        return True
    if t.startswith("前往") or "前往" in t:
        return True
    if any(normalize_text(c) == t for c in _CHROME):
        return True
    if t in ("地圖", "地图", "地下城", "世界"):
        return True
    if re.fullmatch(r"\d+m?", t):
        return True
    if re.fullmatch(r"[\d./]+", t):
        return True
    # 英文 UI / IDE 噪声
    if re.fullmatch(r"[A-Za-z0-9_./:+-]+", t):
        return True
    return False


_HUD_NOISE = (
    "王牌",
    "活動",
    "活动",
    "歡迎",
    "欢迎",
    "目標",
    "目标",
    "冒險",
    "冒险",
    "通關",
    "通关",
    "加工",
    "設備",
    "设备",
    "武器",
    "理科",
    "女巫",
    "分鐘",
    "分钟",
    "委記",
    "委托",
    "星穹",
)


def _looks_region_name(text: str) -> bool:
    t = normalize_text(text)
    if _is_chrome(t):
        return False
    if any(n in t for n in _HUD_NOISE):
        return False
    if not re.search(r"[\u4e00-\u9fff]", t):
        return False
    if len(t) < 2 or len(t) > 10:
        return False
    # 优先已知地名；未知名仅接受纯中文短词（全图再人工筛）
    if any(h in t for h in _REGION_HINTS):
        return True
    if re.search(r"[的了是在与和Lv0-9]", t):
        return False
    if re.search(r"[A-Za-z]", t):
        return False
    return 2 <= len(t) <= 6


def scan_region_text_hits(lines: list[OcrLine], win: WindowInfo) -> list[MapHit]:
    hits: list[MapHit] = []
    for L in lines:
        if not _looks_region_name(L.text):
            continue
        c = box_center_norm(L, win.width, win.height)
        if not c:
            continue
        nx, ny = c
        # 地图画布大致居中，避开顶栏底栏
        if ny < 0.08 or ny > 0.88 or nx < 0.05 or nx > 0.95:
            continue
        score = 5.0 + L.confidence
        if any(h in normalize_text(L.text) for h in _REGION_HINTS):
            score += 3.0
        hits.append(MapHit(kind="text", text=L.text.strip(), nx=nx, ny=ny, score=score))
    return _dedupe_hits(hits)


def scan_region_icon_hits(img: np.ndarray, win: WindowInfo) -> list[MapHit]:
    """全图上找偏小的高饱和色块作图标候选（像素通路雏形）。"""
    h, w = img.shape[:2]
    y0, y1 = int(h * 0.10), int(h * 0.85)
    x0, x1 = int(w * 0.08), int(w * 0.92)
    crop = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 饱和色标（排除灰底）
    mask = cv2.inRange(hsv, (0, 80, 80), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hits: list[MapHit] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40 or area > 2500:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if aspect < 0.4 or aspect > 2.5:
            continue
        cx = x0 + x + bw / 2
        cy = y0 + y + bh / 2
        hits.append(
            MapHit(
                kind="icon",
                text=f"icon_{len(hits)}",
                nx=cx / w,
                ny=cy / h,
                score=min(3.0, area / 200.0),
            )
        )
    return _dedupe_hits(hits, dist=0.025)


def _dedupe_hits(hits: list[MapHit], dist: float = 0.03) -> list[MapHit]:
    hits = sorted(hits, key=lambda h: -h.score)
    out: list[MapHit] = []
    for h in hits:
        if any(abs(h.nx - o.nx) < dist and abs(h.ny - o.ny) < dist for o in out):
            continue
        out.append(h)
    return out


def merge_region_hits(text_hits: list[MapHit], icon_hits: list[MapHit]) -> list[MapHit]:
    """文字优先；图标若远离文字则保留。"""
    out = list(text_hits)
    for ic in icon_hits:
        if any(abs(ic.nx - t.nx) < 0.04 and abs(ic.ny - t.ny) < 0.04 for t in text_hits):
            continue
        out.append(ic)
    return sorted(out, key=lambda h: (h.kind != "text", -h.score))


def _map_level(b: str, lines: list[OcrLine], win: WindowInfo) -> str:
    """town | region | continent | unknown — 实机分层。"""
    if _is_inventory_blob(b) or _is_town_directory_blob(b):
        return "unknown"
    # HUD 任务文案常含「地下城/莱比」等，禁止据此判 continent
    if _looks_hud_chrome(b) and not _map_opened_blob(b):
        return "unknown"
    # Facility chips and an explicit other-city title describe its town map;
    # nearby mines/quest text must not promote that detail view to continent.
    if looks_other_town_detail(b, lines, win):
        return "town"
    fac = any(
        k in b
        for k in ("鐵匠", "铁匠", "截匠", "旅館", "旅馆", "雜貨", "杂货", "學校", "乳牛", "食材")
    )
    titles = [
        L.text
        for L in lines
        if box_center_norm(L, win.width, win.height)
        and box_center_norm(L, win.width, win.height)[1] < 0.12
    ]
    title_blob = "".join(titles)
    # 顶栏杜巴 + 设施芯片 = town（优先于任务栏「莱比/地下城」噪声）
    if fac and "杜巴" in title_blob:
        return "town"
    # 多城/地下城标签 = 大陆中景（即使 OCR 到「杜巴顿」地名也不算 town）
    world_marks = sum(
        1
        for k in (
            "莱比",
            "萊比",
            "希德",
            "冰霜",
            "堤爾",
            "堤尔",
            "库漢",
            "庫漢",
            "库汉",
            "地下城",
            "狩場",
            "狩场",
            "雪原",
            "伐木",
        )
        if k in b
    )
    other = sum(1 for k in ("堤爾", "堤尔", "佛格", "艾明", "班克", "歐絲", "提爾克那") if k in b)
    only_u = bool(titles) and all("杜巴" not in t for t in titles) and any(
        "欧" in t or "歐" in t for t in titles
    )
    # Any secondary place, including a city, is a detail page. City/dungeon
    # labels on that page must not override its breadcrumb and promote it.
    title_secondary = _title_has_secondary_place(lines, win)
    if title_secondary:
        return "region"
    if world_marks >= 2:
        return "continent"
    if other >= 1 or (only_u and not title_secondary):
        return "continent"
    # 标题停在他城（库漢等）且无杜巴 → 非杜巴頓 town
    if any(k in title_blob for k in ("库漢", "庫漢", "库汉", "庫汉")) and "杜巴" not in title_blob:
        return "continent" if world_marks >= 1 or only_u else "region"
    # 设施芯片 + 顶栏含杜巴 + 非多城 = town（禁止仅凭画面中城名「杜巴顿」判 town）
    if fac and "杜巴" not in title_blob and "杜巴" not in b and world_marks < 1:
        return "town"
    if "北門" in b or "北门" in b or "杜巴顿" in b or "杜巴頓" in b or "杜巴頔" in b:
        # 标题明确杜巴且无多城 → region/town 边缘，偏 region
        if "杜巴" in title_blob and world_marks == 0:
            return "region"
        # 顶栏只有歐拉（无斜杠二次地名）= 总图上的城标，不是地区详图
        if only_u and not title_secondary:
            return "continent"
        return "region"
    if _map_opened_blob(b):
        return "region"
    return "unknown"


def _breadcrumb_has_secondary_place(text: str) -> bool:
    from mm_assist.perception.map_header import breadcrumb_has_secondary_place
    return breadcrumb_has_secondary_place(text)


def _title_has_secondary_place(lines: list[OcrLine], win: WindowInfo) -> bool:
    from mm_assist.perception.map_header import title_has_secondary_place
    return title_has_secondary_place(lines, win)


def _current_map_observation(frame: np.ndarray | None, win: WindowInfo) -> UiWhere | None:
    """Read fresh page parameters, without promoting the map to any tier."""
    from mm_assist.life_execute.ui_logic import resolve_navigation

    if frame is None or getattr(frame, 'size', 0) == 0:
        return None
    if frame.shape[:2] != (win.height, win.width):
        return None
    try:
        where = resolve_navigation(frame)
        age = time.time() - float(where.stack_sampled_ts)
        if (not where.stack_ready or where.node not in {'map', 'map_continent', 'map_dunbarton'}
                or not math.isfinite(age) or not -.5 <= age <= 3):
            return None
        return where
    except Exception as exc:
        log.debug('current map parameters unavailable: %s', exc)
        return None


def _current_map_parameters(frame: np.ndarray | None, win: WindowInfo) -> str | None:
    where = _current_map_observation(frame, win)
    return where.node if where is not None else None


def _click_ulara_breadcrumb(lines: list[OcrLine], win: WindowInfo, *, frame=None) -> bool:
    """Fresh map parameters authorize one configured client-frame click."""
    node = _current_map_parameters(frame, win)
    if node is None:
        return False
    try:
        spec = json.loads((ROOT / MAP_NAVIGATION_CONFIG_REL).read_text(encoding='utf-8'))['ulara_breadcrumb']
        dw, dh = spec['design_resolution']
        x, y = spec['client_xy']
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (dw, dh, x, y)) or not 0 <= x < dw or not 0 <= y < dh):
            raise ValueError('invalid fixed map coordinate')
        cx, cy = int(x * win.width / dw), int(y * win.height / dh)
    except Exception as exc:
        log.warning('歐拉 fixed click unavailable: %s', exc)
        return False
    log.info('click 歐拉 fixed client point (%d,%d), current page=%s', cx, cy, node)
    _click(win, cx, cy, hold_s=.30, method='sendinput')
    return True


_CONTINENT_CITY_KEYS: tuple[str, ...] = (
    "杜巴頓",
    "杜巴顿",
    "杜巴頔",
    "佛格",
    "艾明",
    "班克",
    "歐絲",
    "欧丝",
    "库漢",
    "庫漢",
    "库汉",
)


def is_continent_overview(lines: list[OcrLine], win: WindowInfo) -> bool:
    """是否已在歐拉大陸总图（多城可见，而非堤爾克那/狼之森林等地区中景）。"""
    if looks_other_town_detail(_blob(lines), lines, win):
        return False
    titles = []
    for L in lines:
        c = box_center_norm(L, win.width, win.height)
        if c and c[1] <= 0.14 and c[0] < 0.70:
            titles.append(L.text or "")
    # 「歐拉大陸／狼之森林」顶栏 = 地区详图，绝非大陆总图
    if _title_has_secondary_place(lines, win):
        return False
    cities = 0
    has_duba = False
    for h in scan_region_text_hits(lines, win):
        if not _is_map_canvas_poi(h.nx, h.ny) or _is_quest_noise_text(h.text):
            continue
        t = normalize_text(h.text)
        if "北" in t and "杜巴" in t:
            continue
        if any(k in t for k in _CONTINENT_CITY_KEYS):
            cities += 1
            if "杜巴" in t:
                has_duba = True
    if has_duba or cities >= 2:
        return True
    # 顶栏无他城名芯片，且地图已开 → 多半已是总图
    if title_other_city(lines, win) is None and _map_opened_blob(_blob(lines)):
        # 地区中景常仍挂着「堤爾克那」芯片；无芯片且见歐拉标题
        tb = "".join(normalize_text(t) for t in titles)
        if ("欧拉" in tb or "歐拉" in tb or "大睦" in tb) and not any(
            k in tb
            for k in (
                "堤爾",
                "堤尔",
                "库漢",
                "庫漢",
                "佛格",
                "艾明",
                "狼",
                "希德",
                "雪原",
                "森林",
            )
        ):
            return True
    # 顶栏漏框时：全文已见歐拉总图标题、无斜杠二次、无城镇设施芯片
    tb = "".join(normalize_text(t) for t in titles)
    b = _blob(lines)
    if any(n in b for n in CONTINENT_NEEDLES) and not looks_town_facility_blob(b):
        if not any(
            k in tb
            for k in (
                "堤爾",
                "堤尔",
                "库漢",
                "庫漢",
                "狼",
                "希德",
                "雪原",
                "森林",
            )
        ):
            return True
    return False


def click_ulara_text_and_ocr(
    cfg: AppConfig, win: WindowInfo, lines: list[OcrLine]
) -> tuple[WindowInfo, list[OcrLine], str]:
    """确认地图参数后点固定歐拉位置，再独立检查升层。"""
    win, before = _grab_map_frame(cfg, win, settle_s=0.04)
    if not _click_ulara_breadcrumb(lines, win, frame=before):
        click_ulara_text_and_ocr.last_changed = False
        return win, lines, _map_level(_blob(lines), lines, win)
    win, _img, changed, delta = _wait_action_effect(
        cfg, win, before, timeout_s=_map_ui_settle("breadcrumb"), threshold=2.5
    )
    log.info("click_ulara effect changed=%s delta=%.2f", changed, delta)
    click_ulara_text_and_ocr.last_changed = bool(changed)
    win, _img, lines, _ = snap(cfg, win, min_conf=0.28, grab_settle_s=0.05)
    lv = _map_level(_blob(lines), lines, win)
    return win, lines, lv


click_ulara_text_and_ocr.last_changed = True


def leave_other_town_detail(
    cfg: AppConfig,
    win: WindowInfo,
    lines: list[OcrLine],
    *,
    max_rounds: int = 5,
) -> tuple[WindowInfo, list[OcrLine], str, bool]:
    """离开他城城镇详图到中景/大陆。返回 (win, lines, lv, left_ok)。"""
    b = _blob(lines)
    lv = _map_level(b, lines, win)
    # town 层必须尝试离开：面包屑常含「歐拉大」，looks_other_town_detail
    # 会因此假阴性，若此处直接 return 会把「仍 town」当成已离开（找城失败债）。
    if lv != "town" and not looks_other_town_detail(b, lines, win):
        return win, lines, lv, True

    # 测试模式：只允许一次 sendinput 点歐拉，失败即停（禁止多偏移/双击/postmessage）
    from mm_assist.nav.compat import in_test_mode

    if in_test_mode():
        if lv == "town" or looks_other_town_detail(b, lines, win):
            win, lines, lv = click_ulara_text_and_ocr(cfg, win, lines)
            b = _blob(lines)
            left = lv in ("continent", "region")
            log.info("leave_town test-mode left=%s lv=%s", left, lv)
            return win, lines, lv, left
        return win, lines, lv, True

    # 生产：多策略点歐拉：OCR 文字 → 标定坐标 → 左偏/双击 → 关图重开
    cal_tries = (
        ULARA_BREADCRUMB_XY,
        (120, 58),
        (100, 55),
        (160, 58),
        (146, 70),
    )
    for i in range(max(1, max_rounds)):
        from mm_assist.nav.compat import check_abort

        check_abort()
        # 成功离开：已非 town。勿仅靠 looks_other_town（歐拉面包屑假阴）。
        if lv in ("continent", "region"):
            log.info("leave_town ok after #%d lv=%s", i, lv)
            return win, lines, lv, True
        if lv != "town" and not looks_other_town_detail(b, lines, win):
            log.info("leave_town ok after #%d lv=%s", i, lv)
            return win, lines, lv, True
        log.info(
            "leave_town #%d still_town title_other=%s → click 歐拉",
            i + 1,
            title_other_city(lines, win),
        )
        if i == 0:
            win, lines, lv = click_ulara_text_and_ocr(cfg, win, lines)
        elif i == 1:
            # 标定坐标加长按
            _click_cal(
                win, cal_tries[0][0], cal_tries[0][1], hold_s=0.28, method="sendinput"
            )
            time.sleep(1.0)
            win, _img, lines, _ = snap(cfg, win)
            lv = _map_level(_blob(lines), lines, win)
        elif i < 1 + len(cal_tries):
            xy = cal_tries[i - 1]
            _click_cal(win, xy[0], xy[1], hold_s=0.22, method="sendinput")
            time.sleep(1.0)
            win, _img, lines, _ = snap(cfg, win)
            lv = _map_level(_blob(lines), lines, win)
        else:
            # 关图重开，常落在中景
            log.info("leave_town: close+reopen map")
            win = ensure_map_closed(cfg, win, rounds=2)
            time.sleep(0.4)
            opened = open_map(cfg, win, settle=0.9, skip_precheck=True)
            lines = opened.lines
            win, _img = _foreground_grab(cfg, win)
            b = _blob(lines)
            lv = _map_level(b, lines, win)
            if lv == "town" or looks_other_town_detail(b, lines, win):
                win, lines, lv = click_ulara_text_and_ocr(cfg, win, lines)
        b = _blob(lines)
        lv = _map_level(b, lines, win)

    left = lv in ("continent", "region")
    log.info("leave_town done left=%s lv=%s", left, lv)
    return win, lines, lv, left


def _map_already_open(lines: list[OcrLine], img: np.ndarray, win: WindowInfo) -> MapState | None:
    """已开图则返回有效态，避免重复点小地图。npc 卡片≠世界地图。"""
    b = _blob(lines)
    if _is_town_directory_blob(b) or _is_inventory_blob(b) or _is_craft_ui_blob(b):
        return None
    if _is_npc_dialog_blob(b):
        return None
    lv = _map_level(b, lines, win)
    # 层已判出：即使 OCR 仍扫到 Home/ESC 残影也算开图成功
    if lv in ("town", "region", "continent"):
        return "continent" if lv == "continent" else ("town" if lv == "town" else "region")
    if _map_opened_blob(b):
        return "region"
    if _looks_hud_chrome(b):
        return None
    st = classify_state(lines, img)
    if st in ("continent", "region"):
        return st
    return None


def _find_dubarton_castle_norm(img: np.ndarray, win: WindowInfo) -> tuple[float, float] | None:
    """实机帧模板找杜巴頓白色城标 → 客户区归一化中心。"""
    from mm_assist.config import ROOT
    from mm_assist.platform.match import match_template

    path = ROOT / DUBARTON_CASTLE_REL
    if not path.is_file() or img is None or getattr(img, "size", 0) == 0:
        return None
    hit = match_template(
        img,
        path,
        roi=DUBARTON_CASTLE_ROI,
        threshold=DUBARTON_CASTLE_THRESHOLD,
        check_margin=False,
        min_margin=0.0,
    )
    if hit is None:
        # 多尺度兜底（图鉴/实机缩放差）
        for sc in (0.5, 0.75, 1.25, 1.5, 2.0):
            hit = match_template(
                img,
                path,
                roi=DUBARTON_CASTLE_ROI,
                threshold=DUBARTON_CASTLE_THRESHOLD,
                scale=sc,
                check_margin=False,
                min_margin=0.0,
            )
            if hit is not None:
                break
    if hit is None:
        return None
    return float(hit.cx) / max(win.width, 1), float(hit.cy) / max(win.height, 1)


def _find_map_open_btn_xy(img: np.ndarray, win: WindowInfo) -> tuple[int, int] | None:
    """模板找开图钮客户区坐标；缺模板/未命中 → None。"""
    from mm_assist.config import ROOT
    from mm_assist.platform.match import match_template

    locator = ROOT / "knowledge/templates/map/open_hotkey.json"
    if locator.is_file() and img is not None and img.size:
        spec = json.loads(locator.read_text(encoding="utf-8"))
        hit = match_template(img, ROOT / spec['template'], roi=tuple(spec['roi']),
            threshold=float(spec['threshold']), min_margin=float(spec['min_margin']))
        if hit is not None:
            return int(hit.cx), int(hit.cy)
    path = ROOT / MAP_OPEN_BTN_REL
    if not path.is_file() or img is None or getattr(img, "size", 0) == 0:
        return None
    hit = match_template(
        img,
        path,
        roi=MAP_OPEN_BTN_ROI,
        threshold=MAP_OPEN_BTN_THRESHOLD,
        check_margin=False,
    )
    if hit is None:
        return None
    return int(hit.cx), int(hit.cy)


def _note_open_map_polls(n: int) -> None:
    """累加父层 timed_action 的 polls。无栈则忽略。不碰 walk_ms/overhead_ms。"""
    if n <= 0:
        return
    try:
        from mm_assist.nav.compat import _timing_stack

        stack = _timing_stack.get()
        if not stack:
            return
        ctx = stack[-1]
        prev = ctx.get("polls")
        try:
            base = int(prev) if prev is not None else 0
        except (TypeError, ValueError):
            base = 0
        ctx["polls"] = base + int(n)
    except Exception:
        return


def open_map(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    settle: float = 1.0,
    skip_precheck: bool = False,
) -> MapSnapshot:
    """先清遮罩，再点击新帧识别的地图按钮；失败不换点重试。

    skip_precheck=True：省掉开图前 OCR，仍检查新鲜地图参数避免重复开图。
    M按钮由独立模板识别，不匹配变化的地图地形；无按钮不发输入。
    """
    from mm_assist.safety.input import release_mouse_buttons
    from mm_assist.nav.compat import in_production_mode, in_test_mode
    from mm_assist.platform.window import bring_to_foreground

    # 一键修：不再盲睡；点后等画面变化（timeout 见 _map_ui_settle）
    from mm_assist.nav.compat import prefer_visible_clicks

    if prefer_visible_clicks():
        settle = max(float(settle), 0.35)
    elif in_production_mode() and settle > 0.45:
        settle = 0.35
    open_timeout = max(float(settle), _map_ui_settle("open"))

    if not skip_precheck:
        win = dismiss_pre_map_blockers(cfg, win)
        win, img, lines, path = snap(cfg, win)  # 不开 save_prefix，少落盘
        opened = _map_already_open(lines, img, win)
        if opened:
            return MapSnapshot(state=opened, blob=_blob(lines), lines=lines, path=path)
    else:
        lines = []
        img = None
        path = None
        # 开图前只要像素基准，避免 OCR 拖慢
        win, img = _grab_map_frame(cfg, win, settle_s=0.05)
        lines = []
        path = None

    if _current_map_parameters(img, win) is not None:
        return MapSnapshot(state="map", blob=_blob(lines), lines=lines, path=path)

    # 开图一律对焦：goal-loop 默认 unity_hook 失焦时点小地图常空 → map_open_fail
    _focus_for_map_open(win)
    release_mouse_buttons()

    def _after_open_try(name: str, before_img: np.ndarray | None) -> MapSnapshot | None:
        nonlocal win, img, lines, path
        # 1) 先等画面真变（游戏已响应）
        win, img, changed, delta = _wait_action_effect(
            cfg,
            win,
            before_img,
            timeout_s=open_timeout,
            threshold=2.8,
            poll_s=0.12,
        )
        if not changed:
            log.warning(
                "open_map no visual change after %s delta=%.2f (可能点空)",
                name,
                delta,
            )
        # 2) 开图层复核：轻量 OCR 只作「杜巴 town」快路径；
        #    其余一律全屏 OCR 一次（轻量曾把城内误判 continent → 乱拖）
        win, img, lines = _snap_map_light(cfg, win)
        path = None
        b = _blob(lines)
        if _current_map_parameters(img, win) is not None:
            return MapSnapshot(state="map", blob=b, lines=lines, path=path)
        if _is_npc_dialog_blob(b) or (
            len(b) < 100
            and not _looks_hud_chrome(b)
            and not _map_opened_blob(b)
            and any(k in b for k in ("跳", "對話", "对话", "我做的", "工藝"))
        ):
            log.info("open_map hit dialog after %s, dismiss", name)
            win = dismiss_pre_map_blockers(cfg, win)
            return None
        if (
            _is_town_directory_blob(b)
            or _is_inventory_blob(b)
            or _is_craft_ui_blob(b)
        ):
            log.info("open_map hit non-world UI after %s, close and continue", name)
            from mm_assist.safety.input import press_escape

            press_escape()
            time.sleep(0.2)
            return None

        opened = _map_already_open(lines, img, win)
        title_duba = "杜巴" in b[:120]
        fac_hint = any(
            k in b
            for k in (
                "鐵匠",
                "铁匠",
                "截匠",
                "旅館",
                "旅馆",
                "銀行",
                "银行",
                "雜貨",
                "杂货",
                "交易所",
            )
        )
        need_full = opened != "town" or not (title_duba and fac_hint)
        if need_full:
            win, img, lines, path = snap(
                cfg, win, min_conf=0.28, grab_settle_s=0.05
            )
            b = _blob(lines)
            opened = _map_already_open(lines, img, win)
            log.info(
                "open_map full OCR recheck after %s light=%s → %s",
                name,
                "town" if title_duba else "weak",
                opened,
            )
        if opened:
            log.info(
                "open_map ok via %s changed=%s delta=%.2f state=%s",
                name,
                changed,
                delta,
                opened,
            )
            return MapSnapshot(state=opened, blob=b, lines=lines, path=path)
        return None

    # Freshly recognized control only. A missing template must not become a
    # stale calibrated point, and a failed click must not try another offset.
    xy = _find_map_open_btn_xy(img, win)
    if xy is None:
        log.warning("open_map: no recognized map control; no input")
        return MapSnapshot(state="map_button_missing", blob="", lines=lines, path=path)
    before = img.copy()
    tag = _click_map_open(win, xy[0], xy[1], hold_s=0.10, cal=False)
    _note_open_map_polls(1)
    log.info("open_map try=template %s method=sendinput tag=%s", xy, tag)
    if tag == "ok":
        hit = _after_open_try("map_open_btn", before)
        if hit is not None:
            return hit

    return MapSnapshot(
        state=classify_state(lines, img),
        blob=_blob(lines) if lines else "",
        lines=lines,
        path=path,
    )


def ensure_continent(cfg: AppConfig, win: WindowInfo, *, settle: float = 1.0) -> MapSnapshot:
    """开图；town 时点歐拉升到 region 中景。大陆多城全图若未通则返回 region。"""
    open_map(cfg, win, settle=settle)
    win, img, lines, _ = snap(cfg, win)
    b = _blob(lines)
    if not _map_opened_blob(b) and classify_state(lines, img) == "hud":
        log.warning("开图失败，仍在 HUD")
        return MapSnapshot(state="hud", blob=b, lines=lines, regions=[], path=None)

    lv = _map_level(b, lines, win)
    if lv == "town" or looks_other_town_detail(b, lines, win):
        win, lines, lv, _left = leave_other_town_detail(cfg, win, lines, max_rounds=4)
        win, img, lines, _ = snap(cfg, win)
        b = _blob(lines)
        lv = _map_level(b, lines, win)

    if lv == "region":
        # 再点一次歐拉/地球，尝试升大陆全图（实机可能仍停在 region）
        _click_ulara_breadcrumb(lines, win, frame=img)
        time.sleep(settle)
        # 地球图标：文字框左侧
        for L in lines:
            c = box_center_norm(L, win.width, win.height)
            if c and c[1] < 0.12 and ("欧" in L.text or "歐" in L.text) and L.box:
                xs = [p[0] for p in L.box]
                ys = [p[1] for p in L.box]
                _click(win, max(25, int(min(xs) - 28)), int(sum(ys) / len(ys)), hold_s=0.16)
                time.sleep(settle)
                break

    win, img, lines, path = snap(cfg, win, save_prefix="map_continent")
    b = _blob(lines)
    lv = _map_level(b, lines, win)
    st: MapState = "continent" if lv == "continent" else "region" if lv in ("region", "town") else classify_state(lines, img)
    if lv != "continent":
        log.warning("大陆多城全图未达成，当前层=%s（town→region 已可用）", lv)
    regions: list[MapHit] = []
    if _map_opened_blob(b) or st in ("continent", "region"):
        regions = merge_region_hits(
            scan_region_text_hits(lines, win), scan_region_icon_hits(img, win)
        )
    return MapSnapshot(state=st, blob=b, lines=lines, regions=regions, path=path)


def probe(cfg: AppConfig, win: WindowInfo) -> MapSnapshot:
    """干跑：截屏+OCR+态+地区候选，不点击。"""
    win, img, lines, path = snap(cfg, win, save_prefix="map_probe")
    st = classify_state(lines, img)
    regions: list[MapHit] = []
    if st in ("continent", "region"):
        regions = merge_region_hits(
            scan_region_text_hits(lines, win), scan_region_icon_hits(img, win)
        )
    return MapSnapshot(state=st, blob=_blob(lines)[:500], lines=lines, regions=regions, path=path)


def save_regions_json(cfg: AppConfig, hits: list[MapHit], *, state: str) -> Path:
    root = maps_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    out = root / "regions.json"
    payload = {
        "continent": "歐拉大陸",
        "design_resolution": [1920, 1440],
        "state": state,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "regions": [asdict(h) for h in hits],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _drag_map(
    win: WindowInfo,
    *,
    dx: int,
    dy: int,
    cx: int | None = None,
    cy: int | None = None,
    duration_s: float = 0.45,
    force_method: str | None = None,
) -> None:
    """在地图画布上拖拽：内容随手指移动（dx>0 画面右移=看左边）。

    force_method：大陆段建议 ``postmessage``（REQ_C3_ONE_CLICK 不崩鼠标）；
    测试模式默认仍 sendinput（town 实机曾需）。结束必抬键。
    """
    from mm_assist.safety.input import release_mouse_buttons
    from mm_assist.nav.compat import in_test_mode
    from mm_assist.nav.compat import check_abort
    from mm_assist.nav.compat import production_click_method

    check_abort()
    if force_method:
        method = force_method
        do_focus = method in ("sendinput", "si")
    elif in_test_mode():
        # 2026-07-31 town 实机：unity_hook/postmessage 拖 delta≈0；sendinput delta≈40
        # （大陆图 postmessage 曾有效；town 详图以 sendinput 为准）
        method, do_focus = "sendinput", True
    else:
        method, do_focus = production_click_method()
    # sendinput 需前台；unity_hook/postmessage 拖不抢焦
    from mm_assist.safety.input import foreground_input_enabled
    if (do_focus or method in ("sendinput", "si")) and not foreground_input_enabled(win.hwnd):
        if not in_test_mode():
            focus_window(win.hwnd)
            time.sleep(0.08)
        elif not is_foreground(win.hwnd):
            focus_window(win.hwnd)
            time.sleep(0.08)
    x0 = cx if cx is not None else win.width // 2
    y0 = cy if cy is not None else int(win.height * 0.55)
    # 避开顶栏/底栏
    x0 = min(max(x0, int(win.width * 0.25)), int(win.width * 0.75))
    y0 = min(max(y0, int(win.height * 0.25)), int(win.height * 0.75))
    x1 = min(max(x0 + dx, 80), win.width - 80)
    y1 = min(max(y0 + dy, 120), win.height - 120)
    try:
        drag_client(
            win,
            x0,
            y0,
            x1,
            y1,
            method=method,
            refocus=False,
            steps=18,
            duration_s=duration_s,
        )
    finally:
        release_mouse_buttons()


_QUEST_NOISE = (
    "通關",
    "通关",
    "通隔",
    "通嗣",
    "目標",
    "目标",
    "委記",
    "委托",
    "冒家",
    "冒險家",
    "冒险家",
    "每凋",
    "每调",
    "每週",
    "每周",
    "重置",
    "回禮",
    "援吧",
    "進行",
    "全部",
    "喜歡",
    "喜欢",
    "呼唤",
    "呼喚",
    "魔族",
)


def _collect_map_labels(lines: list[OcrLine], img: np.ndarray, win: WindowInfo) -> list[MapHit]:
    """拖拽扫图用：地名 + 设施/地下城标签 + 图标（排除任务栏噪声）。"""
    hits = list(scan_region_text_hits(lines, win))
    extra_needles = (
        "地下城",
        "庭園",
        "庭园",
        "狩獵場",
        "狩猎场",
        "牧場",
        "牧场",
        "北門",
        "北门",
        "鐵匠",
        "铁匠",
        "旅館",
        "旅馆",
        "學校",
        "学校",
        "銀行",
        "银行",
        "治療",
        "治疗",
        "杜巴",
        "堤爾",
        "堤尔",
        "佛格",
        "艾明",
        "班克",
        "歐絲",
        "莱比",
        "萊比",
        "瑪斯",
        "女神",
        "提爾",
        "提尔",
        "肯恩",
        "都柏",
    )
    for L in lines:
        t = normalize_text(L.text)
        if len(t) < 2 or len(t) > 12:
            continue
        if any(n in t for n in _QUEST_NOISE):
            continue
        if not any(n in t for n in extra_needles) and not _looks_region_name(L.text):
            continue
        c = box_center_norm(L, win.width, win.height)
        if not c or c[1] < 0.10 or c[1] > 0.88:
            continue
        # 右侧任务条 ROI 丢掉（非地图画布）
        if c[0] > 0.72 and c[1] < 0.55:
            continue
        hits.append(MapHit(kind="text", text=L.text.strip(), nx=c[0], ny=c[1], score=4.0 + L.confidence))
    # 图标只取地图中部，避免底栏
    icons = [h for h in scan_region_icon_hits(img, win) if 0.15 < h.ny < 0.82 and h.nx < 0.85]
    hits = merge_region_hits(hits, icons[:25])
    return _dedupe_hits(hits, dist=0.028)


def pan_scan_atlas(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    passes: int = 1,
    step_px: int = 560,
    settle: float = 0.65,
) -> tuple[Path, list[MapHit], Path]:
    """开图 → 升到 region → 网格拖拽扫屏，合并 OCR/图标写入 regions.json。

    默认 1 pass、同向不双拖（约 6 步 OCR）。缺地名时再显式加大 passes。
    """
    open_map(cfg, win)
    win, img, lines, _ = snap(cfg, win)
    b = _blob(lines)
    lv = _map_level(b, lines, win)
    if lv == "town":
        _click_ulara_breadcrumb(lines, win, frame=img)
        time.sleep(0.9)
        win, img, lines, _ = snap(cfg, win)

    tile_dir = maps_root(cfg) / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    merged: dict[str, MapHit] = {}
    tile_meta: list[dict] = []

    # 四向 + 两斜；拖拽方向=手指移动，地图内容反向露出（无同向双拖）
    pattern: list[tuple[str, int, int]] = []
    for p in range(max(1, passes)):
        s = step_px + p * 40
        pattern.extend(
            [
                (f"W{p}", s, 0),
                (f"N{p}", 0, s),
                (f"E{p}", -s, 0),
                (f"S{p}", 0, -s),
                (f"NW{p}", s // 2, s // 2),
                (f"SE{p}", -s // 2, -s // 2),
            ]
        )

    def _ingest(tag: str) -> None:
        nonlocal win
        # 只写 tiles/，不刷 captures/
        win, img, lines, _path = snap(cfg, win)
        hits = _collect_map_labels(lines, img, win)
        tile_path = tile_dir / f"{tag}.png"
        cv2.imwrite(str(tile_path), img)
        tile_meta.append(
            {
                "tag": tag,
                "file": str(tile_path.relative_to(maps_root(cfg))),
                "labels": [asdict(h) for h in hits if h.kind == "text"][:40],
            }
        )
        for h in hits:
            key = f"{h.kind}:{normalize_text(h.text)}"
            prev = merged.get(key)
            if prev is None or h.score > prev.score:
                merged[key] = h
        log.info("pan %s labels=%d unique=%d", tag, len(hits), len(merged))

    _ingest("origin")
    for name, dx, dy in pattern:
        _drag_map(win, dx=dx, dy=dy)
        time.sleep(settle)
        _ingest(name)

    hits_all = sorted(merged.values(), key=lambda h: (h.kind != "text", -h.score))
    win, img, lines, _ = snap(cfg, win)
    st = classify_state(lines, img)
    path = save_regions_json(cfg, hits_all, state=st)
    index = maps_root(cfg) / "tiles_index.json"
    index.write_text(
        json.dumps(
            {
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tiles": tile_meta,
                "unique": len(hits_all),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path, hits_all, index


def build_ulara_atlas(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    passes: int = 2,
    step_px: int = 480,
    settle: float = 0.55,
) -> tuple[Path, Path, dict]:
    """开图拖扫拼接歐拉中景图鉴（ulara_atlas.png + .json）。

    手指拖 +(dx,dy) → content_offset -= (dx,dy)；atlas = offset + screen。
    """
    from mm_assist.nav.map_localize import has_dubarton, save_atlas, stitch_tiles

    win = dismiss_pre_map_blockers(cfg, win)
    # 开世界地图（拒「主要」目录）
    for attempt in range(3):
        opened = open_map(cfg, win, settle=0.65, skip_precheck=(attempt == 0))
        win, img, lines, _ = snap(cfg, win)
        b = _blob(lines)
        if _is_town_directory_blob(b) or _is_inventory_blob(b):
            log.info("atlas_build dismiss non-world UI attempt=%d", attempt)
            win = dismiss_pre_map_blockers(cfg, win)
            continue
        if _map_opened_blob(b) or _map_level(b, lines, win) in (
            "town",
            "region",
            "continent",
        ):
            break
    else:
        win, img, lines, _ = snap(cfg, win)
        b = _blob(lines)

    # 离开他城/城镇详图 → 歐拉中景
    for _ in range(4):
        b = _blob(lines)
        lv = _map_level(b, lines, win)
        if looks_other_town_detail(b, lines, win) or lv == "town":
            win, lines, _lv = click_ulara_text_and_ocr(cfg, win, lines)
            win, img = _foreground_grab(cfg, win)
            continue
        break

    b = _blob(lines)
    if _is_town_directory_blob(b):
        raise RuntimeError("atlas_build: still on town directory UI, abort")
    if not (
        _map_opened_blob(b)
        or _map_level(b, lines, win) in ("region", "continent", "town")
    ):
        raise RuntimeError("atlas_build: world map not open")

    tile_dir = maps_root(cfg) / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[dict] = []
    ox = 0.0
    oy = 0.0

    # 覆盖式路径：不回到原点；堤爾一带优先北向找杜巴（ay 更小）
    pattern: list[tuple[str, int, int]] = [("origin", 0, 0)]
    # 北向推进（堤爾→杜巴）
    for i, (dx, dy) in enumerate(_SEEK_FROM_TIR_DELTAS[:8]):
        pattern.append((f"TIR{i}", dx, dy))
    for p in range(max(1, passes)):
        s = step_px + p * 60
        pattern.extend(
            [
                (f"N{p}", 0, s),
                (f"E{p}", -s, 0),
                (f"NE{p}", -s // 2, s // 2),
                (f"W{p}", s, 0),
                (f"NW{p}", s // 2, s // 2),
                (f"S{p}", 0, -s),
            ]
        )

    def _ingest(tag: str) -> None:
        nonlocal win
        win, img_now, lines_now, _ = snap(cfg, win)
        bb = _blob(lines_now)
        if _is_town_directory_blob(bb) or _is_inventory_blob(bb):
            log.warning("atlas_tile skip %s (non-world UI)", tag)
            return
        hits = _collect_map_labels(lines_now, img_now, win)
        h, w = img_now.shape[:2]
        tile_path = tile_dir / f"atlas_{tag}.png"
        cv2.imwrite(str(tile_path), img_now)
        labels = [
            {
                "text": hi.text,
                "nx": hi.nx,
                "ny": hi.ny,
                "score": hi.score,
            }
            for hi in hits
            if hi.kind == "text"
        ]
        tiles.append(
            {
                "tag": tag,
                "img": img_now,
                "ox": ox,
                "oy": oy,
                "w": w,
                "h": h,
                "file": str(tile_path.relative_to(maps_root(cfg))),
                "labels": labels,
            }
        )
        log.info(
            "atlas_tile %s offset=(%.0f,%.0f) labels=%d",
            tag,
            ox,
            oy,
            len(labels),
        )

    # origin 不拖
    _ingest("origin")
    for name, dx, dy in pattern[1:]:
        if dx == 0 and dy == 0:
            continue
        _drag_map(win, dx=dx, dy=dy)
        ox -= float(dx)
        oy -= float(dy)
        time.sleep(settle)
        _ingest(name)

    if not tiles:
        raise RuntimeError("atlas_build: no tiles captured")

    canvas, meta = stitch_tiles(tiles)
    # 用 places.json 补杜巴頓（若本趟 OCR 漏了但扫到同帧地标）
    meta = _enrich_atlas_places_from_json(cfg, meta, tiles)
    meta["continent"] = "歐拉大陸"
    meta["passes"] = passes
    meta["step_px"] = step_px
    meta["tile_count"] = len(tiles)
    meta["tiles"] = [
        {
            "tag": t["tag"],
            "ox": t["ox"],
            "oy": t["oy"],
            "file": t.get("file"),
            "labels": len(t.get("labels") or []),
        }
        for t in tiles
    ]
    png, js = save_atlas(maps_root(cfg), canvas, meta)
    meta["has_dubarton"] = has_dubarton(meta)
    saved = json.loads(js.read_text(encoding="utf-8"))
    saved["has_dubarton"] = meta["has_dubarton"]
    saved["places"] = meta.get("places") or saved.get("places")
    js.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "atlas built %s places=%d has_dubarton=%s",
        png,
        len(meta.get("places") or []),
        meta["has_dubarton"],
    )
    return png, js, meta


def _enrich_atlas_places_from_json(
    cfg: AppConfig, meta: dict, tiles: list[dict]
) -> dict:
    """若图鉴 OCR 已有地标、places.json 有杜巴相对关系，则补写杜巴頓 UV。"""
    from mm_assist.nav.map_localize import is_quest_noise_text

    meta = dict(meta)
    plist = list(meta.get("places") or [])
    # OCR 真见杜巴则保留；去掉旧 places_rel 估点以便重算
    has_ocr_dub = any(
        str(p.get("canon") or "") == "杜巴頓"
        and not str(p.get("source") or "").startswith("places_rel")
        for p in plist
    )
    if has_ocr_dub:
        return meta
    plist = [
        p
        for p in plist
        if not (
            str(p.get("canon") or "") == "杜巴頓"
            and str(p.get("source") or "").startswith("places_rel")
        )
    ]
    meta["places"] = plist

    places_file = load_places_list(cfg)
    if not places_file:
        return meta
    have: dict[str, tuple[float, float]] = {}
    for p in plist:
        c = str(p.get("canon") or "")
        if not c or is_quest_noise_text(str(p.get("text") or "")) or is_quest_noise_text(
            c
        ):
            continue
        if _place_is_target_family(c, "杜巴頓"):
            continue
        have[c] = (float(p["ax"]), float(p["ay"]))
    file_xy: dict[str, tuple[float, float]] = {}
    for it in places_file:
        c = str(it.get("canon") or it.get("text") or "")
        if not c:
            continue
        try:
            file_xy[c] = (float(it["nx"]), float(it["ny"]))
        except Exception:
            continue
    tgt = file_xy.get("杜巴頓") or file_xy.get("杜巴顿")
    if not tgt:
        return meta
    # 优先用同纬度/同帧标定更稳的锚
    prefer = (
        "堤爾克那南部",
        "希德斯特雪原",
        "莱比地下城",
        "莱比地下",
        "堤爾克那",
        "艾菲地下城",
    )
    ordered = [c for c in prefer if c in have and c in file_xy]
    ordered += [c for c in have if c not in ordered and c in file_xy]
    tw = float(np.mean([t.get("w") or 1920 for t in tiles])) if tiles else 1920.0
    th = float(np.mean([t.get("h") or 1440 for t in tiles])) if tiles else 1440.0
    for canon in ordered:
        ax, ay = have[canon]
        fx, fy = file_xy[canon]
        dax = (tgt[0] - fx) * tw
        day = (tgt[1] - fy) * th
        est = {
            "canon": "杜巴頓",
            "text": "杜巴頓",
            "ax": round(ax + dax, 2),
            "ay": round(ay + day, 2),
            "score": 6.0,
            "source": f"places_rel←{canon}",
        }
        plist.append(est)
        meta["places"] = plist
        log.info("atlas enrich 杜巴頓 via %s -> (%.1f,%.1f)", canon, est["ax"], est["ay"])
        return meta
    return meta


def atlas_seek_city(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    target_canon: str = "杜巴頓",
    max_drags: int = 8,
    lines: list[OcrLine] | None = None,
) -> tuple[WindowInfo, list[OcrLine], list[MapHit], str]:
    """DEPRECATED：旧 OCR/盲拖寻城。主路径禁止调用。"""
    raise RuntimeError(
        "atlas_seek_city 已退役：请用 town_pixel_click_npc / 大陆段后置像素路"
    )


def scan_and_capture_regions(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    click_each: bool = False,
    pan: bool = True,
    max_regions: int = 80,
    passes: int = 1,
) -> tuple[Path, list[MapHit]]:
    """拖拽扫图写 regions.json；可选再逐点文字截 overview。"""
    if pan:
        path, hits, _ = pan_scan_atlas(cfg, win, passes=passes)
        hits = hits[:max_regions]
    else:
        snap1 = ensure_continent(cfg, win)
        hits = snap1.regions[:max_regions]
        path = save_regions_json(cfg, hits, state=snap1.state)

    if not click_each:
        return path, hits

    overview_dir = maps_root(cfg) / "regions"
    overview_dir.mkdir(parents=True, exist_ok=True)
    for i, h in enumerate(hits):
        if h.kind != "text":
            continue
        _click_norm(win, h.nx, h.ny)
        time.sleep(0.9)
        win, img, lines, _ = snap(cfg, win)
        safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", h.text)[:32] or f"r{i}"
        dest = overview_dir / safe
        dest.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest / "overview.png"), img)
        ch = find_text_hits(lines, win, CONTINENT_NEEDLES + ("歐拉", "欧拉", "返回", "地圖"))
        if ch:
            _click_norm(win, ch[0].nx, ch[0].ny)
        else:
            _click_ulara_breadcrumb(lines, win, frame=img)
        time.sleep(0.5)
    return path, hits


def find_goto_button(lines: list[OcrLine], win: WindowInfo) -> MapHit | None:
    hits = find_text_hits(lines, win, GOTO_NEEDLES, min_ny=0.70)
    if not hits:
        hits = find_text_hits(lines, win, GOTO_NEEDLES, min_ny=0.55)
    return hits[0] if hits else None


def find_green_travel_button(img: np.ndarray, win: WindowInfo) -> MapHit | None:
    """地图点角色/地点后，右下角绿色自动移动钮（OCR 常漏「前往」）。

    避开底栏「地圖」绿标（约在中右）；前往钮更靠右、更宽。
    """
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        return None
    # 再收紧：排除底栏地圖 tab（常见 nx≈0.70 附近）
    y0, y1 = int(h * 0.86), int(h * 0.96)
    x0, x1 = int(w * 0.78), int(w * 0.96)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 台服绿钮偏亮绿；两档 HSV 合并
    mask1 = cv2.inRange(hsv, (35, 60, 60), (95, 255, 255))
    mask2 = cv2.inRange(hsv, (40, 40, 100), (90, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        # 前往大钮；底栏小图标面积偏小
        if area < 800 or area > 120000:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if aspect < 1.6 or aspect > 10.0:
            continue
        cx_c = x + bw / 2.0
        cy_c = y + bh / 2.0
        corner_bonus = (cx_c / max(crop.shape[1], 1)) + (cy_c / max(crop.shape[0], 1))
        score_area = area * (1.0 + 0.35 * corner_bonus)
        if score_area > best_area:
            best_area = score_area
            best = (cx_c, cy_c, area)
    if not best:
        return None
    cx = (x0 + best[0]) / w
    cy = (y0 + best[1]) / h
    return MapHit(
        kind="green",
        text="green_travel",
        nx=float(cx),
        ny=float(cy),
        score=min(25.0, best[2] / 400.0),
    )


def resolve_goto_click(
    lines: list[OcrLine],
    img: np.ndarray,
    win: WindowInfo,
    *,
    allow_fallback: bool = False,
) -> MapHit | None:
    """OCR「前往*」→ 右下角绿钮 CV。GOTO_FALLBACK 已退役（ignore allow_fallback）。"""
    del allow_fallback
    hit = find_goto_button(lines, win)
    if hit:
        return hit
    hit = find_green_travel_button(img, win)
    if hit:
        return hit
    return None


# 修理设施：只认铁匠/奈麗/完整「武器店」；禁作台/武器零等弱匹配
_REPAIR_FACILITY_OK = ("鐵匠", "铁匠", "截匠", "P匠", "匠铺", "匠舖", "奈麗", "奈丽", "武器店")
_REPAIR_FACILITY_BAN = ("作台", "製作", "制作", "零工", "加工設", "多用途", "武器零", "武器作")


def is_repair_facility_text(text: str) -> bool:
    t = normalize_text(text)
    if not t or any(b in t for b in _REPAIR_FACILITY_BAN):
        return False
    return any(k in t for k in _REPAIR_FACILITY_OK)


def _repair_hit_preference(text: str) -> float:
    """修理点排序：奈麗絲/武器店 > 鐵匠，避免大陆图误点弱设施。"""
    t = normalize_text(text)
    if any(k in t for k in ("奈麗", "奈丽")):
        return 100.0
    if "武器店" in t:
        return 50.0
    if any(k in t for k in ("鐵匠", "铁匠", "截匠")):
        return 10.0
    return 0.0


def prefer_town_avatar_click(
    *,
    ocr_hit: MapHit | None,
    atlas_hit: MapHit | None,
) -> MapHit | None:
    """城内点头像优先级：OCR 奈麗名 > 图鉴投影 > OCR 铁匠上方。

    实机（2026-08-20）：atlas 近距后 OCR「鐵匠鋪上方」易点到设施图标/色块，
    出卡失败；同帧 atlas_xy 点头像可出奈麗絲卡。故图鉴已近时 atlas 压过铁匠 OCR。
    """
    if ocr_hit is not None and _repair_hit_preference(ocr_hit.text) >= 100.0:
        return ocr_hit
    if atlas_hit is not None:
        return atlas_hit
    return ocr_hit


def find_npc_avatar(
    lines: list[OcrLine],
    img: np.ndarray,
    win: WindowInfo,
    names: tuple[str, ...],
    *,
    allow_icon_fallback: bool = False,
    repair_facility: bool = False,
    min_ny: float = 0.0,
) -> MapHit | None:
    """先 OCR 名，再点名附近。默认不点随机图标（避免误点製作台）。

    min_ny：修理设施默认应 ≥0.12，忽略顶栏面包屑误命中。
    """
    name_hits = find_text_hits(lines, win, names, min_ny=min_ny)
    if repair_facility:
        # 奈麗丝名优先于铁匠店名（店名上方常误点设施圆标）
        npc_named = [
            h
            for h in name_hits
            if _repair_hit_preference(h.text) >= 100.0
            and 0.12 <= h.ny <= 0.88
            and 0.05 <= h.nx <= 0.95
        ]
        if npc_named:
            npc_named.sort(key=lambda h: -(h.score + _repair_hit_preference(h.text)))
            return avatar_above_label(npc_named[0])
        # 铁匠店名专用（已滤顶/底栏）优先于泛 needles
        smith = find_smith_shop_hit(lines, win)
        if smith is not None:
            return avatar_above_label(smith)
        name_hits = [h for h in name_hits if is_repair_facility_text(h.text)]
        # 扫全行再补铁匠 OCR 变体
        if not name_hits:
            for L in lines:
                if not is_repair_facility_text(L.text):
                    continue
                c = box_center_norm(L, win.width, win.height)
                if not c or c[1] < max(min_ny, 0.12):
                    continue
                # 顶栏城名/面包屑绝不当设施
                tt = normalize_text(L.text or "")
                if any(k in tt for k in ("杜巴", "歐拉", "欧拉", "大陸", "大陆", "北門", "北门")):
                    continue
                name_hits.append(
                    MapHit(kind="text", text=L.text, nx=c[0], ny=c[1], score=8.0 + L.confidence)
                )
        # 滤顶栏
        lo = max(float(min_ny), 0.12)
        name_hits = [
            h
            for h in name_hits
            if lo <= h.ny <= 0.88
            and 0.05 <= h.nx <= 0.95
            and not any(
                k in normalize_text(h.text)
                for k in ("杜巴", "歐拉", "欧拉", "大陸", "大陆", "北門", "北门")
            )
        ]
        name_hits.sort(key=lambda h: -(h.score + _repair_hit_preference(h.text)))
    if name_hits:
        h = name_hits[0]
        # 用户锁定：点「鐵匠鋪」上方任务头像，不是店名文字
        if repair_facility:
            return avatar_above_label(h)
        ny = max(0.05, h.ny - 0.04)
        return MapHit(kind="text", text=h.text, nx=h.nx, ny=ny, score=h.score)
    if allow_icon_fallback:
        icons = [
            h
            for h in scan_region_icon_hits(img, win)
            if 0.15 < h.ny < 0.82 and 0.08 < h.nx < 0.85
        ]
        return icons[0] if icons else None
    return None


def detect_map_close_x_pixel(img: np.ndarray | None) -> bool | None:
    """右上关闭钮像素：深底上的白色 X（双对角线亮笔画）。

    True=像有 X；False=不像；None=裁剪失败。
    """
    if img is None or getattr(img, "size", 0) == 0:
        return None
    from mm_assist.platform.match import roi_slice

    crop, _, _ = roi_slice(img, CLOSE_X_ROI)
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return None
    b, g, r = cv2.split(crop)
    white = (
        (r.astype(np.int16) > 175)
        & (g.astype(np.int16) > 175)
        & (b.astype(np.int16) > 175)
        & (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 40)
        & (np.abs(g.astype(np.int16) - b.astype(np.int16)) < 40)
    )
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = gray < 100
    white_frac = float(np.mean(white))
    dark_frac = float(np.mean(dark))
    # 实心白块/空白：不像描边 X；无深底也不像关闭钮
    if white_frac < 0.025 or white_frac > 0.42 or dark_frac < 0.12:
        return False

    # 主对角 / 反对角上的亮点密度（X 两臂）
    n = min(h, w)
    main = []
    anti = []
    for i in range(n):
        y = int(i * (h - 1) / max(1, n - 1))
        x1 = int(i * (w - 1) / max(1, n - 1))
        x2 = int((n - 1 - i) * (w - 1) / max(1, n - 1))
        main.append(1.0 if white[y, x1] else 0.0)
        anti.append(1.0 if white[y, x2] else 0.0)
    main_m = float(np.mean(main)) if main else 0.0
    anti_m = float(np.mean(anti)) if anti else 0.0
    # 两臂都要有一定亮点（单斜线 / 噪点不够）
    if main_m < 0.12 or anti_m < 0.12:
        return False
    return True


def detect_map_close_x(
    img: np.ndarray | None,
    lines: list[OcrLine] | None = None,
    *,
    blob: str | None = None,
) -> dict:
    """关图钮是否仍在：像素白 X 为主；OCR 地图层仅在像素未知时旁证。

    ``visible``：
    - True：像素像 X；或像素未知且 OCR 仍见地图层
    - False：像素明确不像 X（认 X 已灭）；或像素未知且 OCR 无地图
    - None：像素与 OCR 皆无可用信号
    """
    b = blob if blob is not None else (_blob(lines) if lines else "")
    ocr_map = bool(_map_opened_blob(b))
    pix = detect_map_close_x_pixel(img)
    if pix is True:
        visible: bool | None = True
    elif pix is False:
        # 点 X 后认像素灭优先；OCR 地图粘滞不挡「X 没有了」
        visible = False
    elif ocr_map:
        visible = True
    elif b == "" and img is None:
        visible = None
    else:
        visible = False if not ocr_map else None
    return {
        "visible": visible,
        "pixel": pix,
        "ocr_map": ocr_map,
        "note": f"pix={pix} ocr_map={ocr_map}",
    }


# 人物界面 OCR（繁中 + 常见误识）；与地圖层互斥
PROFILE_OCR_NEEDLES: tuple[str, ...] = (
    "生活技能",
    "生活技",
    "裝備",
    "装备",
    "道具",
    "寵物",
    "宠物",
    "稱號",
    "称号",
    "能力值",
    "角色資訊",
    "角色信息",
    "外觀",
    "外观",
)


def _profile_ocr_hit(b: str) -> bool:
    if not b or _map_opened_blob(b):
        return False
    return any(n in b for n in PROFILE_OCR_NEEDLES)


def detect_profile_panel(
    img: np.ndarray | None,
    lines: list[OcrLine] | None = None,
    *,
    blob: str | None = None,
) -> dict:
    """人物界面是否已开：OCR 文案 + 像素右上关闭 X（与关圖同套特征，方向相反）。

    ``open``：
    - True：OCR 命中人物针，或（像素像 X 且 OCR 非地图）
    - False：明确不像
    - None：无可用信号
    """
    b = blob if blob is not None else (_blob(lines) if lines else "")
    ocr_map = bool(_map_opened_blob(b))
    ocr_prf = _profile_ocr_hit(b)
    pix = detect_map_close_x_pixel(img)
    if ocr_map:
        open_: bool | None = False
    elif ocr_prf:
        open_ = True
    elif pix is True:
        # 有关钮且非地图 → 当作面板层（人物/其它 menu）；OCR 未中时仍过像素闸
        open_ = True
    elif pix is False and not ocr_prf:
        open_ = False
    else:
        open_ = None
    return {
        "open": open_,
        "pixel_x": pix,
        "ocr_profile": ocr_prf,
        "ocr_map": ocr_map,
        "note": f"pix={pix} ocr_prf={ocr_prf} ocr_map={ocr_map}",
    }


def detect_craft_panel(
    img: np.ndarray | None,
    lines: list[OcrLine] | None = None,
    *,
    blob: str | None = None,
) -> dict:
    """製作/加工面板是否已开：OCR 文案为主；右上关钮像素旁证。

    ``open`` True=像製作层；False=不像；None=无信号。
    """
    b = blob if blob is not None else (_blob(lines) if lines else "")
    if _map_opened_blob(b):
        return {
            "open": False,
            "ocr_craft": False,
            "pixel_x": detect_map_close_x_pixel(img),
            "ocr_map": True,
            "note": "ocr_map_blocks_craft",
        }
    ocr_strong = _is_craft_ui_blob(b)
    ocr_weak = any(
        k in b
        for k in (
            "所需材料",
            "前往製作",
            "前往制作",
            "生活工具",
            "製作成功率",
            "制作成功率",
            "金屬加工",
            "金属加工",
            "加工",
            "製作",
            "制作",
        )
    )
    pix = detect_map_close_x_pixel(img)
    if ocr_strong:
        open_: bool | None = True
    elif ocr_weak:
        open_ = True
    elif pix is True and ("工具" in b or "材料" in b):
        open_ = True
    elif not b and img is None:
        open_ = None
    else:
        open_ = False
    return {
        "open": open_,
        "ocr_craft": bool(ocr_strong or ocr_weak),
        "pixel_x": pix,
        "ocr_map": False,
        "note": f"pix={pix} ocr_strong={ocr_strong} ocr_weak={ocr_weak}",
    }


def ensure_map_closed(cfg: AppConfig, win: WindowInfo, *, rounds: int = 1) -> WindowInfo:
    """前往后强制关图；默认 1 轮；已关则 0 点。

    点 CLOSE 后认 X 灭；OCR 仍见地图层时继续点关。
    """
    win, img, lines, _ = snap(cfg, win, min_conf=0.28)
    st = classify_state(lines, img)
    b = _blob(lines)
    cx = detect_map_close_x(img, lines, blob=b)
    need = (
        cx.get("visible") is True
        or cx.get("ocr_map") is True
        or st in ("region", "continent", "npc_card")
    )
    if not need:
        return win
    for i in range(max(1, rounds)):
        _click_cal(win, CLOSE_XY[0], CLOSE_XY[1])
        time.sleep(0.35)
        win, img, lines, _ = snap(cfg, win, min_conf=0.28)
        st = classify_state(lines, img)
        b = _blob(lines)
        cx = detect_map_close_x(img, lines, blob=b)
        if cx.get("visible") is False and not cx.get("ocr_map"):
            return win
        if not (_map_opened_blob(b) or st in ("region", "continent", "npc_card")):
            return win
        if i + 1 >= rounds:
            break
    return win


def wait_stop_then_dialog(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    timeout_s: float = 60.0,
    poll: float = 0.50,
    idle_abort_s: float = 5.0,
    no_motion_abort_s: float = 5.0,
    settle_s: float = 0.0,
) -> tuple[bool, MapState, str]:
    """点「前往」后等人物对话。

    约定（用户拍板）：
    - 点完移动后 ``settle_s``（默认由调用方先睡 1s）内**禁止任何键鼠**；
    - ``menu=0`` 且 ``stp=1``：行进中，只等；
    - ``menu=1`` 且 ``stp=0``：已到站 UI，等到 ``dlg=1`` / 对话画面；
    - 成功：内存 dlg 或 OCR 对话选项。

    idle_abort_s：从未出现过 stp=1 才早弃；``<=0`` 关闭。
    timeout_s：``<=0`` 无限等（仍可停止操作取消）。
    """
    from mm_assist.perception.hud_sense import detect_stop_icon
    from mm_assist.memory.readers.scene import Il2CppSceneReader
    from mm_assist.tool_repair import op_cancelled

    if settle_s > 0:
        time.sleep(float(settle_s))

    t0 = time.time()
    saw_stp1 = False
    last: MapState = "unknown"
    detail = ""
    allow_idle_abort = idle_abort_s > 0
    infinite = timeout_s <= 0
    closed_map_once = False

    while infinite or time.time() - t0 < timeout_s:
        if op_cancelled():
            return False, last, "cancelled"

        menu: bool | None = None
        dlg = False
        stp: bool | None = None
        try:
            rdr = Il2CppSceneReader()
            try:
                ms = rdr.read(allow_scan=False)
            finally:
                rdr.close()
            if ms is not None:
                menu = ms.menu_open
                if ms.dialog_active is True:
                    dlg = True
        except Exception:
            pass

        # 轻量截屏只认停止钮；动作中绝不点关地圖
        win, img, lines, _ = snap(cfg, win, min_conf=0.28, grab_settle_s=0.05)
        b = _blob(lines)
        st = classify_state(lines, img)
        last = st
        if img is not None:
            try:
                icon = detect_stop_icon(img, ocr_space=True)
                if icon is not None:
                    stp = True
                elif stp is None:
                    stp = False
            except Exception:
                pass
        if st == "traveling" or any(n in b for n in STOP_NEEDLES):
            stp = True
        if stp is True:
            saw_stp1 = True
            detail = "stp1"

        # 已进对话：立刻成功（勿再点）。任务栏「交談中」可与六钮同时出现。
        if dlg or st == "dialog" or _looks_dialog(b, lines, win):
            if any(k in b for k in ("修理", "修復", "武器商人", "奈麗", "再見", "跳過")) or dlg:
                return True, "dialog", f"dlg menu={menu} stp={stp}"
            if menu is True and stp is False:
                return True, "dialog", f"menu1_stp0_dialog_ui"

        # 任务栏「前往與奈麗絲交談中」= 仍在走路，继续等
        if _is_walking_to_npc_blob(b) and stp is True:
            detail = "walk quest_tracker"
            time.sleep(poll)
            continue
        if _is_walking_to_npc_blob(b) and saw_stp1 and stp is not True:
            detail = "stuck_idle_tracker"
            time.sleep(poll)
            if time.time() - t0 >= 8.0:
                return False, last, f"stuck_idle_tracker saw_stp1={saw_stp1}"
            continue

        # menu=0 & stp=1：走路中，只等
        if menu is False and stp is True:
            detail = "walk menu0_stp1"
            time.sleep(poll)
            continue

        # menu=1 & stp=0：到站，等对话弹出
        if menu is True and stp is False:
            detail = "arrived menu1_stp0"
            time.sleep(poll)
            continue

        # 仅当地图仍开着、且已不在走路时，关一次图（禁止打断自动走）
        if (
            not closed_map_once
            and stp is not True
            and (_map_opened_blob(b) or st in ("region", "continent"))
        ):
            _click_cal(win, CLOSE_XY[0], CLOSE_XY[1])
            closed_map_once = True
            detail = detail or "closed_map_after_walk"
            time.sleep(0.3)
            continue

        if allow_idle_abort and not saw_stp1:
            if time.time() - t0 >= idle_abort_s:
                return False, last, f"idle_abort never_stp1 after {idle_abort_s:.0f}s"
        time.sleep(poll)
    return False, last, f"timeout detail={detail} saw_stp1={saw_stp1}"


def _is_smith_only(text: str) -> bool:
    t = normalize_text(text)
    if any(k in t for k in ("奈麗", "奈丽", "武器店")):
        return False
    if any(k in t for k in ("鐵匠", "铁匠", "截匠", "P匠")):
        return True
    # OCR 掉首字：…匠铺 / …匠舖（排除其它含「铺」的店需同时有「匠」）
    return ("匠铺" in t or "匠舖" in t) and "食" not in t and "药" not in t and "藥" not in t


def _card_has_repair_npc(bb: str, npc_needles: tuple[str, ...]) -> bool:
    """卡片是否出现白名单 NPC（奈麗絲卡）。无则禁止点前往。"""
    nb = normalize_text(bb)
    if any(normalize_text(n) in nb for n in npc_needles if n):
        return True
    # 台服卡片：奈麗絲 + 帶著清爽微笑的武器商人 + 前往此處
    if any(
        k in nb
        for k in (
            "奈麗",
            "奈丽",
            "武器商人",
            "带著清爽",
            "帶著清爽",
            "清爽微笑",
            "清爽微笑的",
            "清爽",
            "武器商",
        )
    ):
        return True
    # OCR 常糊：鐵匠铺卡带「修理/改造/販售」描述 + 前往
    if ("前往此" in nb or "前往这" in nb) and any(
        k in nb for k in ("修理", "改造", "販售", "贩售", "武器商", "清爽")
    ):
        return True
    return False


TELEPORT_NEEDLES = (
    "月之石",
    "傳送",
    "传送",
    "瞬間移動",
    "瞬间移动",
    "使用月之石",
    "移動到",
    "移动到",
)


_WALKING_NPC_MARKS = (
    "前往與奈麗",
    "前往与奈麗",
    "前往奈麗絲處",
    "前往奈丽丝处",
    "交談中",
    "交谈中",
)


def _is_walking_to_npc_blob(b: str) -> bool:
    """任务栏「前往與奈麗絲交談中」不是已进对话。"""
    if any(k in b for k in ("跳過對話", "跳过对话", "武器商人", "帶著清爽", "带着清爽")):
        return False
    return any(k in b for k in _WALKING_NPC_MARKS)


def _looks_dialog(b: str, lines: list[OcrLine], win: WindowInfo) -> bool:
    if any(n in b for n in ("修理", "修復", "修复", "跳過對話", "跳过对话", "武器商人", "特化", "結束對話")):
        return True
    if _is_walking_to_npc_blob(b):
        return False
    if "對話" in b or "对话" in b:
        if "前往" in b and "奈麗" in b:
            return False
        return True
    low = 0
    for L in lines:
        c = box_center_norm(L, win.width, win.height)
        if c and c[1] > 0.70:
            low += 1
    return low >= 2 and ("關閉" in b or "关闭" in b or "確認" in b or "确认" in b)


def _region_id_for_name(region: str) -> str:
    """地区显示名 → atlas 目录 id（v1 仅杜巴頓）。"""
    t = normalize_text(region)
    if "杜巴" in t:
        return "dunbarton"
    # 未知：用拼音占位会缺 atlas → 硬 fail；暂原样小写
    return re.sub(r"[^\w]+", "_", region).strip("_").lower() or "unknown"


def registered_nerys_avatar(img: np.ndarray, win: WindowInfo) -> MapHit | None:
    """Independent current portrait; atlas/OCR coordinates cannot authorize it."""
    from mm_assist.config import ROOT
    from mm_assist.platform.match import match_template
    if not isinstance(img,np.ndarray) or img.size == 0:
        return None
    try:
        spec = json.loads((ROOT/'knowledge/templates/map/nerys.json').read_text(encoding='utf-8'))
        hit = match_template(img, ROOT/'knowledge/templates/map/nerys.png',roi=tuple(spec['roi']),
                             threshold=spec['threshold'],min_margin=spec['min_margin'])
        if hit is None:
            return None
        return MapHit(kind='avatar',text='template:奈麗絲',nx=hit.cx/img.shape[1],
                      ny=hit.cy/img.shape[0],score=hit.score)
    except (OSError,ValueError,KeyError):
        return None


def town_pixel_click_npc(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    region: str = "杜巴頓",
    npc: str = "奈麗絲",
    region_id: str | None = None,
) -> dict:
    """城内找 NPC：图鉴定位当前视野 → 拖到锚点附近 → OCR/图标点头像 → 验卡。

    与歐拉大陆同思路：先认「在图哪」，再拖到合适位置，最后识图点。
    不做前往/对话。
    """
    from mm_assist.nav.compat import timed_action
    from mm_assist.nav.compat import AbortRequested, check_abort
    from mm_assist.nav.compat import in_test_mode, prefer_visible_clicks
    from mm_assist.nav.map_localize import (
        anchor_xy,
        drag_until_near,
        find_place_xy,
        is_placeholder_anchor,
        load_nav_thresholds,
        load_town_atlas,
        localize_viewport_zoom_search,
    )

    rid = region_id or _region_id_for_name(region)
    npc_needles = tuple(
        dict.fromkeys(
            [npc, "奈麗絲", "奈麗", "奈丽"] if "奈麗" in normalize_text(npc) or "奈丽" in normalize_text(npc) else [npc]
        )
    )


    from mm_assist.perception.game_frame import ensure_game_frame

    ensure_game_frame(cfg, win, persist=True)

    try:
        check_abort()
        facility_needles = tuple(
            dict.fromkeys(
                [
                    *npc_needles,
                    "鐵匠",
                    "铁匠",
                    "截匠",
                    "武器店",
                    "匠铺",
                    "匠舖",
                ]
            )
        )

        force = (
            "sendinput"
            if (prefer_visible_clicks() or in_test_mode())
            else None
        )

        def _seek_avatar(
            lines_now: list[OcrLine],
            img_now: np.ndarray,
            *,
            allow_icon: bool = False,
        ) -> MapHit | None:
            return find_npc_avatar(
                lines_now,
                img_now,
                win,
                facility_needles,
                allow_icon_fallback=allow_icon,
                repair_facility=True,
                min_ny=0.12,
            )

        # —— 1) 图鉴：当前视野在城图哪里 → 拖到奈麗/鐵匠附近 ——
        from mm_assist.nav.map_localize import atlas_xy_to_client_norm

        repair_nerys = _repair_hit_preference(npc) >= 100.0
        visible_nerys = None
        if repair_nerys:
            win, current_frame = _grab_map_frame(cfg, win, settle_s=.04)
            visible_nerys = registered_nerys_avatar(current_frame,win)
        # A visible independently recognized target needs no map movement.
        atlas_img, meta = ((None,None) if visible_nerys is not None else
                           load_town_atlas(maps_root(cfg),rid))
        tgt = None
        atlas_near_ok = False
        near_loc = None
        if atlas_img is not None and meta is not None:
            if not is_placeholder_anchor(meta, npc, kinds=("npc_avatar",)):
                tgt = anchor_xy(meta, npc, kinds=("npc_avatar",))
            if tgt is None:
                tgt = find_place_xy(meta, "鐵匠铺") or find_place_xy(meta, "鐵匠鋪")
            if tgt is not None:
                th = load_nav_thresholds(maps_root(cfg))
                N = float(th.get("N_town") or 120)
                min_score = min(float(th.get("min_match_score_town") or 0.30), 0.25)
                with timed_action("town_atlas_drag_near", note=npc) as dctx:

                    def _snap() -> np.ndarray:
                        check_abort()
                        nonlocal win
                        win, im = _grab_map_frame(cfg, win, settle_s=0.04)
                        assert im is not None
                        return im

                    def _drag(dx: int, dy: int) -> None:
                        check_abort()
                        nonlocal win
                        _drag_map(
                            win,
                            dx=int(dx),
                            dy=int(dy),
                            duration_s=0.35,
                            force_method=force,
                        )
                        # 拖完停稳再识别（勿在动画中途 localize）
                        win, _im, ok = _wait_frame_stable(
                            cfg, win, timeout_s=_map_ui_settle("drag")
                        )
                        if not ok:
                            log.warning("town atlas drag frame not stable")

                    near = drag_until_near(
                        atlas=atlas_img,
                        target_xy=tgt,
                        N=N,
                        min_score=min_score,
                        max_steps=8,
                        snap_fn=_snap,
                        localize_fn=lambda live, atl: localize_viewport_zoom_search(
                            live, atl, min_score=min_score
                        ),
                        drag_fn=_drag,
                    )
                    atlas_near_ok = bool(near.ok)
                    if near.ok and near.loc is not None:
                        near_loc = near.loc
                    dctx["ok"] = atlas_near_ok
                    dctx["note"] = str(near.note or "")[:48]
                    log.info(
                        "town atlas near ok=%s note=%s score=%s",
                        atlas_near_ok,
                        near.note,
                        getattr(near.loc, "score", None),
                    )
            else:
                log.warning("town atlas: no npc/facility anchor for %s", npc)
        elif visible_nerys is None:
            log.warning("town atlas missing rid=%s → OCR-only seek", rid)

        # —— 2) OCR / 图鉴投影 → 再拖扫；图标色块最后（禁空 OCR 抢点随机标）——
        # 图鉴已近：atlas 投影优先于「铁匠上方」OCR（后者易点设施图标，见 prefer_town_avatar_click）
        with timed_action("town_ocr_after_atlas", note=npc) as dctx:
            lines: list[OcrLine] = []
            av: MapHit | None = None
            av_ocr: MapHit | None = None
            win, img = _grab_map_frame(cfg, win, settle_s=0.04)
            if img is not None:
                win, img, lines = _snap_map_light(cfg, win)
                av_ocr = _seek_avatar(lines, img, allow_icon=False)
            if av_ocr is None:
                win, img, lines, _ = snap(cfg, win, min_conf=0.28, grab_settle_s=0.05)
                av_ocr = (
                    _seek_avatar(lines, img, allow_icon=False) if img is not None else None
                )
            av_atlas: MapHit | None = None
            if (
                atlas_near_ok
                and tgt is not None
                and near_loc is not None
                and img is not None
            ):
                nx_a, ny_a = atlas_xy_to_client_norm(
                    near_loc, tgt[0], tgt[1], img.shape[1], img.shape[0]
                )
                if 0.08 <= nx_a <= 0.92 and 0.12 <= ny_a <= 0.88:
                    av_atlas = MapHit(
                        kind="avatar",
                        text=f"atlas:{npc}",
                        nx=nx_a,
                        ny=ny_a,
                        score=float(near_loc.score),
                    )
            av = prefer_town_avatar_click(ocr_hit=av_ocr, atlas_hit=av_atlas)
            if repair_nerys:
                av = registered_nerys_avatar(img,win)
                if av is None:
                    return {'ok':False,'stage':'npc_avatar_unverified',
                    'error':'新图未独立识别奈麗絲头像，禁止投影点／随机图标回退',
                    'verified':False}
            if av is not None and av_atlas is not None and av is av_atlas:
                log.info(
                    "town click via atlas_xy (%.3f,%.3f) score=%.3f (ocr=%s)",
                    av.nx,
                    av.ny,
                    av.score,
                    (av_ocr.text if av_ocr else "none")[:24],
                )
            elif av is not None and av_ocr is not None and av is av_ocr:
                log.info(
                    "town click via ocr %r @ (%.3f,%.3f) atlas=%s",
                    av.text[:24],
                    av.nx,
                    av.ny,
                    "skip" if av_atlas is None else "deferred",
                )
            max_ocr_drags = 3 if atlas_near_ok else 8
            if av is None:
                for i, (dx, dy) in enumerate(_TOWN_SEEK_DRAG_DELTAS[:max_ocr_drags]):
                    check_abort()
                    _drag_map(
                        win,
                        dx=int(dx),
                        dy=int(dy),
                        duration_s=0.35,
                        force_method=force,
                    )
                    win, img, _ok = _wait_frame_stable(
                        cfg, win, timeout_s=_map_ui_settle("drag")
                    )
                    if img is None:
                        continue
                    win, img, lines = _snap_map_light(cfg, win)
                    av = _seek_avatar(lines, img, allow_icon=False)
                    if av is None:
                        win, img, lines, _ = snap(
                            cfg, win, min_conf=0.28, grab_settle_s=0.05
                        )
                        av = (
                            _seek_avatar(lines, img, allow_icon=False)
                            if img is not None
                            else None
                        )
                    if av is not None:
                        log.info(
                            "town found after drag#%d %r @ (%.3f,%.3f)",
                            i + 1,
                            av.text,
                            av.nx,
                            av.ny,
                        )
                        break
            if av is not None and av.ny < 0.14:
                log.info("town hit too high ny=%.3f → drag down", av.ny)
                for _ in range(3):
                    check_abort()
                    _drag_map(
                        win,
                        dx=0,
                        dy=int(win.height * 0.28),
                        duration_s=0.35,
                        force_method=force,
                    )
                    win, img, _ok = _wait_frame_stable(
                        cfg, win, timeout_s=_map_ui_settle("drag")
                    )
                    if img is None:
                        continue
                    win, img, lines = _snap_map_light(cfg, win)
                    av = _seek_avatar(lines, img, allow_icon=False)
                    if av is not None and av.ny >= 0.14:
                        break
                else:
                    av = None
            # 图标色块：仅 OCR/投影都失败时最后一搏
            if av is None and img is not None:
                av = _seek_avatar(lines, img, allow_icon=True)
                if av is not None:
                    log.info(
                        "town seek icon fallback @ (%.3f,%.3f) %r",
                        av.nx,
                        av.ny,
                        av.text,
                    )
            dctx["ok"] = av is not None
            dctx["note"] = (av.text if av else "miss")[:40]

        if av is None:
            pinned = load_dunbarton_avatar(cfg)
            if pinned is not None and 0.12 <= pinned.ny <= 0.88:
                av = pinned
                log.info(
                    "town seek fallback pinned @ (%.3f,%.3f)", av.nx, av.ny
                )
        if av is None:
            return {
                "ok": False,
                "stage": "drag",
                "error": "图鉴/OCR 均未见鐵匠·奈麗",
                "verified": False,
            }

        nx, ny = float(av.nx), float(av.ny)
        if not (0.08 <= nx <= 0.92 and 0.12 <= ny <= 0.88):
            return {
                "ok": False,
                "stage": "click",
                "error": f"命中越界 nx={nx:.3f} ny={ny:.3f}",
                "verified": False,
            }

        win, before = _grab_map_frame(cfg, win, settle_s=0.0)
        if repair_nerys:
            fresh_avatar = registered_nerys_avatar(before,win)
            if fresh_avatar is None:
                return {'ok':False,'stage':'npc_avatar_unverified',
                'error':'点击前奈麗絲头像消失，未发输入','verified':False}
            nx,ny = fresh_avatar.nx,fresh_avatar.ny
        _click_norm(win, nx, ny)
        win, img, changed, delta = _wait_action_effect(
            cfg,
            win,
            before,
            timeout_s=_map_ui_settle("click"),
            threshold=2.5,
        )
        if not changed:
            return {
                "ok": False,
                "stage": "npc_card",
                "error": f"点头像无画面变化（点空）delta={delta:.2f}",
                "clicked_nx": nx,
                "clicked_ny": ny,
                "hit": (av.text or "")[:40],
                "verified": False,
            }
        win, img, lines, _ = snap(cfg, win, min_conf=0.28, grab_settle_s=0.05)
        bb = _blob(lines)
        card_ok = _card_has_repair_npc(bb, npc_needles)
        if not card_ok and ("前往此" in bb or "前往这" in bb):
            time.sleep(0.25)
            win, img, lines, _ = snap(cfg, win, min_conf=0.28, grab_settle_s=0.05)
            bb = _blob(lines)
            card_ok = _card_has_repair_npc(bb, npc_needles)
        with timed_action("step_verify_npc_card", note=npc) as ctx:
            ctx["ok"] = card_ok
            ctx["note"] = bb[:60]
        if not card_ok:
            return {
                "ok": False,
                "stage": "npc_card",
                "error": "单次点击后未见奈麗絲卡片",
                "blob": bb[:120],
                "clicked_nx": nx,
                "clicked_ny": ny,
                "verified": False,
            }
        return {
            "ok": True,
            "stage": "npc_card",
            "detail": "npc_card_ok",
            "region": region,
            "npc": npc,
            "region_id": rid,
            "clicked_nx": nx,
            "clicked_ny": ny,
            "blob": bb[:120],
            "verified": True,
            "in_town": True,
            "seek": "ocr_drag",
        }
    except AbortRequested:
        return {"ok": False, "stage": "aborted", "error": "用户停止", "verified": False}
    except Exception as exc:
        from mm_assist.nav.compat import GuardError

        if isinstance(exc, GuardError):
            return {
                "ok": False,
                "stage": "guard",
                "error": str(exc)[:160],
                "verified": False,
            }
        raise
    finally:
        from mm_assist.safety.input import release_mouse_buttons
        release_mouse_buttons()


def map_goto(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    region: str,
    npc: str,
    timeout_s: float = 60.0,
    idle_abort_s: float = 5.0,
    prefer_npc_name: bool = True,
    max_attempts: int = 1,
    wait_arrive: bool = True,
) -> dict:
    """白名单导航：近距杜巴頓 town 直达；远距拖地图找城→绿钮（默认不重试）。

    输入授权由调用者的有效租约和输入门控负责。
    """
    from mm_assist.nav.compat import AbortRequested, check_abort

    from mm_assist.perception.game_frame import ensure_game_frame


    ensure_game_frame(cfg, win, persist=True)
    last: dict = {"ok": False, "stage": "start"}
    try:
        for attempt in range(max(1, max_attempts)):
            check_abort()
            npc_only = attempt > 0
            abort_s = idle_abort_s
            if attempt > 0:
                win = ensure_map_closed(cfg, win, rounds=1)
                time.sleep(0.25)
            last = _map_goto_once(
                cfg,
                win,
                region=region,
                npc=npc,
                timeout_s=timeout_s,
                idle_abort_s=abort_s,
                npc_only=npc_only,
                ban_smith_outside_town=True,
                wait_arrive=wait_arrive,
            )
            last["attempt"] = attempt
            if last.get("ok"):
                return last
            detail = str(last.get("detail") or "")
            err = str(last.get("error") or "")
            # 远距短失败 / 封死：禁止满重试空等
            if any(
                k in err
                for k in (
                    "无前往/传送",
                    "放弃空等",
                    "卡片无",
                    "未找到杜巴頓地名",
                    "拖地图仍未找到杜巴頓",
                    "未进城镇详图",
                    "未见杜巴頓标签",
                    "match_score_fail",
                )
            ):
                break
            retryable = (
                "idle_abort" in detail
                or "製作台" in err
                or "顶栏无杜巴" in err
                or "顶栏非杜巴" in err
            )
            if attempt + 1 < max_attempts and retryable:
                log.warning("map_goto retryable=%s attempt=%d", detail or err, attempt)
                continue
            break
        return last
    except AbortRequested:
        return {
            "ok": False,
            "stage": "aborted",
            "error": "用户停止",
            "verified": False,
        }
    except Exception as exc:
        from mm_assist.nav.compat import GuardError

        if isinstance(exc, GuardError):
            from mm_assist.safety.input import release_mouse_buttons

            release_mouse_buttons()
            return {
                "ok": False,
                "stage": "guard",
                "error": str(exc)[:160],
                "verified": False,
            }
        raise
    finally:
        # 拖拽/点击若被打断，防止左键粘滞导致系统鼠标像被锁死
        from mm_assist.safety.input import release_mouse_buttons

        release_mouse_buttons()


def dismiss_pre_map_blockers(cfg: AppConfig, win: WindowInfo) -> WindowInfo:
    """开图前清对话/制作/背包/城镇目录。勿乱点屏幕中央（易点到城内 NPC）。"""
    from mm_assist.safety.input import press_escape, release_mouse_buttons
    from mm_assist.nav.compat import in_test_mode
    from mm_assist.platform.window import bring_to_foreground, focus_window

    if _steal_focus_allowed():
        focus_window(win.hwnd)
        time.sleep(0.12)
    for _round in range(8):
        if _steal_focus_allowed():
            bring_to_foreground(win.hwnd, retries=2, settle_s=0.06)
        win, _img, lines, _ = snap(
            cfg, win, min_conf=0.25, grab_settle_s=0.05 if not _steal_focus_allowed() else 0.25
        )
        b = _blob(lines)

        # 兼职/任务接取：点「之後再做」，勿点接受（Unity 常需 sendinput）
        if _is_quest_offer_blob(b):
            log.info("dismiss quest offer before map (round %d)", _round + 1)
            clicked = False
            for L in lines:
                t = normalize_text(L.text or "")
                if not any(k in t for k in _QUEST_LATER_NEEDLES):
                    continue
                c = box_center_norm(L, win.width, win.height)
                if not c or c[1] < 0.70:
                    continue
                # 按钮偏右：OCR 框中心偏左时略右移
                nx = min(0.72, max(0.48, c[0] + 0.02))
                ny = min(0.92, max(0.78, c[1]))
                _click(
                    win,
                    int(nx * win.width),
                    int(ny * win.height),
                    hold_s=0.14,
                    method="sendinput",
                )
                clicked = True
                log.info(
                    "quest later OCR %r @ (%.3f,%.3f)->(%.3f,%.3f)",
                    L.text,
                    c[0],
                    c[1],
                    nx,
                    ny,
                )
                time.sleep(0.7)
                break
            if not clicked:
                sx, sy = _scale_xy(win, *_QUEST_LATER_CAL_XY)
                log.info("quest later cal click -> (%d,%d)", sx, sy)
                _click(win, sx, sy, hold_s=0.14, method="sendinput")
                time.sleep(0.6)
            # 仍卡：ESC 兜底（兼職层常可关）
            if _round >= 2:
                press_escape()
                time.sleep(0.4)
            release_mouse_buttons()
            continue

        # NPC 对话：优先 OCR 点「跳過*」，失败则点标定右上角
        if _is_npc_dialog_blob(b) or (
            len(b) < 80
            and any(k in b for k in ("我做的", "工藝", "工艺", "麥爾坎", "梅尔坎"))
            and not _looks_hud_chrome(b)
            and not _map_opened_blob(b)
        ):
            log.info("dismiss npc dialog before map (round %d)", _round + 1)
            menu_only = any(k in b for k in ("随便聊聊", "隨便聊聊", "商店")) and not any(
                k in b for k in ("跳過", "跳过", "跳對", "跳对")
            )
            if menu_only:
                # 已是选项菜单：优先点結束，否则 ESC（勿点商店）
                end_clicked = False
                for L in lines:
                    t = normalize_text(L.text or "")
                    if any(k in t for k in ("結束", "结束", "糙束")):
                        c = box_center_norm(L, win.width, win.height)
                        if c:
                            _click(
                                win,
                                int(c[0] * win.width),
                                int(c[1] * win.height),
                                hold_s=0.12,
                                method="sendinput",
                            )
                            end_clicked = True
                            log.info("end dialog OCR (menu) %r", L.text)
                            time.sleep(0.5)
                            break
                if not end_clicked:
                    log.info("npc menu_only — ESC")
                    press_escape()
                    time.sleep(0.45)
                release_mouse_buttons()
                continue
            clicked = False
            for L in lines:
                t = normalize_text(L.text or "")
                if not any(k in t for k in _SKIP_DIALOG_CLICK_NEEDLES):
                    continue
                c = box_center_norm(L, win.width, win.height)
                if not c or c[1] > 0.22:
                    continue
                _click(
                    win,
                    int(c[0] * win.width),
                    int(c[1] * win.height),
                    hold_s=0.12,
                    method="sendinput",
                )
                clicked = True
                log.info("skip dialog OCR click %r @ (%.3f,%.3f)", L.text, c[0], c[1])
                time.sleep(0.55)
                break
            if not clicked:
                sx, sy = _scale_xy(win, *_SKIP_DIALOG_CAL_XY)
                log.info("skip dialog cal click -> (%d,%d)", sx, sy)
                _click(win, sx, sy, hold_s=0.12, method="sendinput")
                time.sleep(0.45)
            # 选项菜单：「結束對話 / 随便聊聊」
            win, _img, lines2, _ = snap(cfg, win, min_conf=0.25)
            b2 = _blob(lines2)
            if any(k in b2 for k in ("糙束", "結束", "结束", "随便聊聊", "隨便聊聊", "商店")):
                end_clicked = False
                for L in lines2:
                    t = normalize_text(L.text or "")
                    if any(k in t for k in ("結束", "结束", "糙束")):
                        c = box_center_norm(L, win.width, win.height)
                        if c:
                            _click(
                                win,
                                int(c[0] * win.width),
                                int(c[1] * win.height),
                                hold_s=0.12,
                                method="sendinput",
                            )
                            end_clicked = True
                            log.info("end dialog OCR %r", L.text)
                            time.sleep(0.5)
                            break
                if not end_clicked:
                    # 无「結束」仅商店/聊聊：ESC 关菜单（勿盲点商店）
                    log.info("npc menu without end — ESC")
                    press_escape()
                    time.sleep(0.45)
                    if _round >= 1:
                        press_escape()
                        time.sleep(0.35)
            else:
                # 推进气泡 / ESC
                _click(
                    win,
                    int(win.width * 0.50),
                    int(win.height * 0.42),
                    hold_s=0.08,
                    method="sendinput",
                )
                time.sleep(0.25)
                press_escape()
                time.sleep(0.35)
            release_mouse_buttons()
            continue

        kind = None
        if _is_craft_ui_blob(b):
            kind = "craft_ui"
        elif _is_inventory_blob(b):
            kind = "inventory"
        elif _is_town_directory_blob(b):
            kind = "town_directory"
        if kind:
            log.info("dismiss %s before map (round %d)", kind, _round + 1)
            press_escape()
            time.sleep(0.4)
            win, _img, lines, _ = snap(cfg, win, min_conf=0.25)
            b2 = _blob(lines)
            still = (
                (kind == "craft_ui" and _is_craft_ui_blob(b2))
                or (kind == "inventory" and _is_inventory_blob(b2))
                or (kind == "town_directory" and _is_town_directory_blob(b2))
            )
            if still:
                _click_cal(win, CLOSE_XY[0], CLOSE_XY[1])
                time.sleep(0.4)
            release_mouse_buttons()
            continue
        if any(k in b for k in ("網路", "网络", "連線", "连线", "不稳定", "不穩定")):
            for L in lines:
                t = L.text or ""
                if not L.box:
                    continue
                if any(k in t for k in ("重新", "重試", "重试", "当試")) and "开始" not in t and "開始" not in t:
                    c = box_center_norm(L, win.width, win.height)
                    if not c:
                        continue
                    _click(win, int(c[0] * win.width), int(c[1] * win.height))
                    time.sleep(1.2)
                    break
            continue
        break
    release_mouse_buttons()
    return win


def _map_goto_once(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    region: str,
    npc: str,
    timeout_s: float,
    idle_abort_s: float,
    npc_only: bool,
    ban_smith_outside_town: bool,
    wait_arrive: bool = True,
) -> dict:
    """单次：近距杜巴頓 town 直达 NPC；远距再找城。卡片无白名单名禁点前往。"""
    win = dismiss_pre_map_blockers(cfg, win)
    wl = load_whitelist(cfg)
    entry = None
    for n in wl.get("npcs", []):
        if n.get("region") == region and n.get("name") == npc:
            entry = n
            break
    if entry is None:
        entry = {
            "region": region,
            "name": npc,
            "aliases": [],
            "region_aliases": [],
            "facility_aliases": [],
        }

    region_needles = tuple(
        dict.fromkeys([region, *entry.get("region_aliases", [])])
    )
    npc_needles = tuple(dict.fromkeys([npc, *entry.get("aliases", [])]))
    if npc_only:
        facility_needles = tuple(
            dict.fromkeys([*npc_needles, "武器店", *entry.get("facility_aliases", [])])
        )
        facility_needles = tuple(n for n in facility_needles if not _is_smith_only(n))
        if not facility_needles:
            facility_needles = npc_needles
    else:
        facility_needles = tuple(
            dict.fromkeys(
                [
                    *npc_needles,
                    "武器店",
                    *entry.get("facility_aliases", []),
                    "鐵匠鋪",
                    "鐵匠铺",
                    "铁匠铺",
                    "鐵匠",
                    "截匠铺",
                    "截匠鋪",
                ]
            )
        )

    # 开图：一键修/长挂走快路径（skip_precheck，只开一次），避免③前空等 5～13s
    from mm_assist.nav.compat import in_production_mode, prefer_visible_clicks

    lean_open = prefer_visible_clicks() or in_production_mode()
    if lean_open:
        opened = open_map(cfg, win, settle=_map_ui_settle("open"), skip_precheck=True)
        lines = opened.lines
        b = opened.blob if opened.blob else _blob(lines)
        win, img = _foreground_grab(cfg, win, settle_s=0.05)
        lv = _map_level(b, lines, win)
        parameter_map = _current_map_parameters(img, win) is not None
        if parameter_map and (lv not in ("town", "region", "continent") or _looks_hud_chrome(b)):
            # Page parameters only authorize the fixed breadcrumb, never a tier
            # or atlas. Drop text from a failed/stale HUD capture before ascent.
            lines, b, lv = [], "", "unknown"
        if not parameter_map and lv not in ("town", "region", "continent") and not _map_opened_blob(b):
            return {
                "ok": False,
                "stage": "open",
                "error": ("未识别地图按钮，未点击" if opened.state == "map_button_missing"
                          else "点击后未确认地图打开，已停止"),
                "blob": b[:100],
            }
    else:
        # 开图：先清遮罩；settle 略长保证 tip 关干净
        win = dismiss_pre_map_blockers(cfg, win)
        opened = open_map(cfg, win, settle=0.85, skip_precheck=False)
        lines = opened.lines
        b = opened.blob if opened.blob else _blob(lines)
        if _is_town_directory_blob(b) or _is_inventory_blob(b):
            log.info("open_map got non-world UI, dismiss and retry")
            win = dismiss_pre_map_blockers(cfg, win)
            opened = open_map(cfg, win, settle=0.85, skip_precheck=False)
            lines = opened.lines
            b = opened.blob if opened.blob else _blob(lines)
        win, img = _foreground_grab(cfg, win)
        lv = _map_level(b, lines, win)
        if lv not in ("town", "region", "continent") and not _map_opened_blob(b):
            win = dismiss_pre_map_blockers(cfg, win)
            opened = open_map(cfg, win, settle=0.9, skip_precheck=False)
            lines = opened.lines
            b = opened.blob if opened.blob else _blob(lines)
            if _is_town_directory_blob(b) or _is_inventory_blob(b):
                win = dismiss_pre_map_blockers(cfg, win)
                opened = open_map(cfg, win, settle=0.9, skip_precheck=False)
                lines = opened.lines
                b = opened.blob if opened.blob else _blob(lines)
            win, img = _foreground_grab(cfg, win)
            lv = _map_level(b, lines, win)
        if lv not in ("town", "region", "continent") and not _map_opened_blob(b):
            return {
                "ok": False,
                "stage": "open",
                "error": "开图失败",
                "blob": b[:100],
            }
    # HUD 残留：勿把主界面当成他城详图去点歐拉
    if _looks_hud_chrome(b) and not _map_opened_blob(b):
        return {
            "ok": False,
            "stage": "open",
            "error": "开图失败(仍为HUD)",
            "blob": b[:100],
        }

    dubarton_aliases = tuple(
        dict.fromkeys([*region_needles, "杜巴", "尚杜巴", "杜巴頔", "杜巴顿"])
    )

    def _has_dubarton(blob: str) -> bool:
        return any(normalize_text(n) in blob for n in dubarton_aliases if n)

    def _title_has_dubarton(lines_now: list[OcrLine]) -> bool:
        """仅看地图顶栏面包屑；库漢等他城标题一律否。"""
        title_bits: list[str] = []
        for L in lines_now:
            c = box_center_norm(L, win.width, win.height)
            # 面包屑约 y≈0.04；杜巴頓北門可偏右
            if not c or c[1] > 0.10 or c[0] > 0.55:
                continue
            t = normalize_text(L.text or "")
            title_bits.append(t)
            if any(k in t for k in ("库漢", "庫漢", "库汉", "尚库", "尚庫")):
                return False
            if any(normalize_text(n) in t for n in dubarton_aliases if n):
                # 中景「杜巴頓北門」城北门 POI，不是城镇面包屑
                if "北門" in t or "北门" in t:
                    continue
                log.info("title_dubarton hit %r @ (%.3f,%.3f)", L.text, c[0], c[1])
                return True
        tb = "".join(title_bits)
        if any(k in tb for k in ("库漢", "庫漢", "库汉")):
            return False
        return False

    def _in_dubarton_town(blob: str, level: str, lines_now: list[OcrLine]) -> bool:
        # The lean path can have no usable OCR even when the selected resource
        # and title already prove Dunbarton. Re-read the paired current identity
        # for this frame; a generic Town enum is never sufficient.
        if _current_map_parameters(img, win) == 'map_dunbarton':
            return True
        if title_other_city(lines_now, win) is not None:
            return False
        if any(k in blob for k in ("库漢", "庫漢", "库汉")) and "杜巴" not in blob[:50]:
            return False
        title_ok = _title_has_dubarton(lines_now)
        head = blob[:100]
        head_ok = "杜巴" in head and not any(
            k in head for k in ("库漢", "庫漢", "堤爾克那", "堤尔克那")
        )
        if not title_ok and not head_ok:
            return False
        town_fac = any(
            k in blob
            for k in (
                "鐵匠",
                "铁匠",
                "截匠",
                "P匠",
                "匠铺",
                "匠舖",
                "旅館",
                "旅馆",
                "雜貨",
                "杂货",
                "銀行",
                "银行",
                "治療",
                "治疗",
                "交易所",
                "乳牛",
                "學校",
                "学校",
            )
        )
        # 顶栏/前段杜巴 + 设施芯片 = 已在详图
        if town_fac:
            return True
        if sum(1 for k in ("莱比", "萊比", "希德", "冰霜") if k in blob) >= 2:
            return False
        if level == "town":
            return True
        if level == "region" and town_fac:
            return True
        return False

    def _find_region_hits(lines_now: list[OcrLine]) -> list[MapHit]:
        rhits = find_text_hits(lines_now, win, dubarton_aliases)
        if rhits:
            return rhits
        out: list[MapHit] = []
        for h in scan_region_text_hits(lines_now, win):
            if any(normalize_text(n) in normalize_text(h.text) for n in dubarton_aliases):
                out.append(h)
        return out

    # —— 近距优先：已在杜巴頓 town → 绝不先去大陆拖图 ——
    if _in_dubarton_town(b, lv, lines):

        return _map_goto_npc_arrive(
            cfg,
            win,
            img=img,
            lines=lines,
            region=region,
            npc=npc,
            npc_needles=npc_needles,
            facility_needles=facility_needles,
            level=lv,
            timeout_s=timeout_s,
            idle_abort_s=idle_abort_s,
            allow_smith=not npc_only,
            wait_arrive=wait_arrive,
        )

    # —— 大陆/中景段：像素图鉴拖杜巴頓进城（REQ_C3_ONE_CLICK）——

    entered = continent_pixel_enter_dubarton(
        cfg, win, lines=lines, img=img, already_open=True
    )
    if not entered.get("ok"):
        return {
            "ok": False,
            "stage": str(entered.get("stage") or "continent"),
            "error": str(entered.get("error") or "大陆进杜巴失败"),
            "level": entered.get("level") or lv,
            "blob": b[:100],
            "continent": entered,
            "verified": False,
        }
    win = entered.get("win") or win
    win, img, lines, _ = snap(cfg, win)
    b = _blob(lines)
    lv = _map_level(b, lines, win)
    if not _in_dubarton_town(b, lv, lines):
        return {
            "ok": False,
            "stage": "continent_enter",
            "error": "拖点后仍非杜巴頓 town",
            "level": lv,
            "blob": b[:100],
            "continent": entered,
            "verified": False,
        }
    return _map_goto_npc_arrive(
        cfg,
        win,
        img=img,
        lines=lines,
        region=region,
        npc=npc,
        npc_needles=npc_needles,
        facility_needles=facility_needles,
        level=lv,
        timeout_s=timeout_s,
        idle_abort_s=idle_abort_s,
        allow_smith=not npc_only,
        wait_arrive=wait_arrive,
    )


def continent_pixel_enter_dubarton(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    lines: list[OcrLine] | None = None,
    img: np.ndarray | None = None,
    target_canon: str = "杜巴頓",
    already_open: bool = True,
) -> dict:
    """开图后：先点「歐拉大陸」→ 再用 ulara 图鉴拖「杜巴頓」进城。

    already_open=True（默认）：调用方已开图，禁止再 open_map（避免③前空等）。
    """
    from mm_assist.safety.input import release_mouse_buttons
    from mm_assist.nav.compat import in_test_mode
    from mm_assist.nav.map_localize import (
        atlas_xy_to_client_norm,
        drag_until_near,
        find_place_xy,
        load_continent_atlas,
        load_nav_thresholds,
        localize_viewport_zoom_search,
        known_town_canvas_matches,
    )

    try:
        if not already_open:
            open_map(cfg, win, settle=1.0, skip_precheck=True)
            win, img2, lines2, _ = snap(cfg, win, min_conf=0.28, grab_settle_s=0.05)
            img = img2 if img2 is not None else img
            lines = lines2 if lines2 else (lines or [])
        else:
            if lines is None or img is None:
                win, img2, lines2, _ = snap(cfg, win, min_conf=0.28, grab_settle_s=0.05)
                img = img2 if img2 is not None else img
                lines = lines2 if lines2 else (lines or [])
        lines = lines or []
        b = _blob(lines)
        lv = _map_level(b, lines, win)

        if _current_map_parameters(img, win) == 'map_dunbarton':
            return {
                'ok': True, 'stage': 'already_town', 'level': 'town',
                'blob': b[:100], 'win': win, 'error': None,
                'near': 'current_map_dunbarton',
            }

        # City chips can remain on the actual continent canvas (original 庫漢
        # failure frame). Override the text-only detail hypothesis only with
        # independently matched terrain in three separate atlas quadrants.
        def proven_canvas(frame: np.ndarray | None) -> bool:
            from mm_assist.nav.map_localize import continent_canvas_matches

            if frame is None or frame.size == 0 or float(frame.std()) < 1.0:
                return False
            atlas_probe, _ = load_continent_atlas(maps_root(cfg))
            thresholds = load_nav_thresholds(maps_root(cfg))
            return continent_canvas_matches(frame, atlas_probe,
                min_score=float(thresholds['min_overview_canvas_score']),
                min_patch_score=float(thresholds['min_overview_patch_score']))

        def current_world_resource(frame: np.ndarray | None) -> str:
            where = _current_map_observation(frame, win)
            return where.map_resource if where and where.node == 'map_continent' else ''

        if known_town_canvas_matches(img):
            lv = "town"
        # The already-open fast path intentionally has no OCR lines. Terrain
        # evidence must not depend on text having identified a tier first.
        canvas_proven = proven_canvas(img)
        world_resource = current_world_resource(img)
        # 已在歐拉总图则禁止再点标题（实机会连点歐拉大陆、从不拖杜巴頓）
        from mm_assist.perception.map_header import cloud_region_visible
        region_header = cloud_region_visible(img)
        if world_resource or canvas_proven or (not region_header and (is_continent_overview(lines, win) or lv == "continent")):
            log.info("continent: already overview (lv=%s) skip 歐拉 click, seek %s", lv, target_canon)
            lv = "continent"
        else:
            log.info("continent: click 歐拉大陸 first (lv=%s) then seek %s", lv, target_canon)
            before_level = lv
            win, lines, lv = click_ulara_text_and_ocr(cfg, win, lines)
            b = _blob(lines)
            # Check the new frame before deciding another ascent is needed:
            # a city chip can remain after town -> continent has already worked.
            win, img = _grab_map_frame(cfg, win, settle_s=0.04)
            canvas_proven = proven_canvas(img)
            world_resource = current_world_resource(img)
            if world_resource or canvas_proven:
                lv = "continent"
            # A verified town -> region transition may need one more ascent.
            # Remaining on the same detail page is failure, not a retry budget.
            if (
                before_level == "town"
                and lv == "region"
                and getattr(click_ulara_text_and_ocr, "last_changed", True)
                and not is_continent_overview(lines, win)
            ):
                log.info("continent: town -> region verified; ascend next layer")
                win, lines, lv = click_ulara_text_and_ocr(cfg, win, lines)
                b = _blob(lines)
                win, img = _grab_map_frame(cfg, win, settle_s=0.04)
                canvas_proven = proven_canvas(img)
                world_resource = current_world_resource(img)
                if world_resource or canvas_proven:
                    lv = "continent"

        def _top_title_texts() -> list[str]:
            out: list[str] = []
            for L in lines or []:
                c = box_center_norm(L, win.width, win.height)
                if c and c[1] < 0.12:
                    out.append(L.text or "")
            return out

        def _still_region_detail() -> bool:
            """斜杠面包屑或 lv=region 且非总图 → 仍是地区详图（禁 ulara 图鉴）。"""
            if world_resource or canvas_proven:
                return False
            if cloud_region_visible(img):
                return True
            if _title_has_secondary_place(lines, win):
                return True
            return lv == "region" and not is_continent_overview(lines, win)

        if lv == "town" and (
            "杜巴" in b[:100]
            and any(
                k in b
                for k in ("鐵匠", "铁匠", "截匠", "旅館", "旅馆", "銀行", "银行", "雜貨")
            )
        ):
            log.warning(
                "continent: still town after 歐拉 but OCR 杜巴詳图 → already_town"
            )
            return {
                "ok": True,
                "stage": "already_town",
                "level": "town",
                "blob": b[:100],
                "near": "ulara_still_town_recheck",
                "win": win,
                "error": None,
            }

        # Never close/reopen or promote an unchanged detail page to overview.
        if _still_region_detail():
            log.warning("continent: ascent did not reach overview; stop (no retry/atlas)")
            return {
                "ok": False,
                "stage": "continent_region_stuck",
                "error": "找城失败:仍地区中景未升大陆总图·禁图鉴硬匹配",
                "level": lv,
                "blob": b[:100],
                "win": win,
            }

        if is_continent_overview(lines, win) or lv == "continent":
            # overview 偏严时：_map_level=continent 且无斜杠面包屑 → 仍可走图鉴（修一键开图）
            lv = "continent"
        elif lv not in ("continent", "region"):
            return {
                "ok": False,
                "stage": "continent_open",
                "error": f"点歐拉后仍非大陆/中景（lv={lv}）",
                "level": lv,
                "win": win,
            }
        else:
            # lv=region 但无斜杠、且被判定非 detail：保守再 fail（不应常见）
            log.warning(
                "continent: ambiguous region without overview → fail-closed (no atlas)"
            )
            return {
                "ok": False,
                "stage": "continent_region_stuck",
                "error": "找城失败:仍地区中景未升大陆总图·禁图鉴硬匹配",
                "level": lv,
                "blob": b[:100],
                "win": win,
            }

        mroot = maps_root(cfg)
        atlas, meta = load_continent_atlas(mroot)
        if atlas is None or meta is None:
            return {
                "ok": False,
                "stage": "continent_atlas",
                "error": "缺 ulara_atlas",
                "win": win,
            }
        target = find_place_xy(meta, target_canon)
        if target is None:
            return {
                "ok": False,
                "stage": "continent_place",
                "error": f"图鉴无 {target_canon}",
                "win": win,
            }
        thr = load_nav_thresholds(mroot)
        N = float(thr.get("N_ulara") or thr.get("near_px") or thr.get("N") or 180.0)
        # png/json 已对齐；门槛读 nav_thresholds（默认 0.35）
        min_score = float(
            thr.get("min_match_score_ulara") or thr.get("min_score") or 0.35
        )

        def _snap_fn() -> np.ndarray:
            nonlocal win, img
            # 图鉴匹配只要像素，禁止每步全屏 OCR
            win, img = _grab_map_frame(cfg, win, settle_s=0.04)
            assert img is not None
            if world_resource and current_world_resource(img) != world_resource:
                raise GuardError('当前大陆地图资源已变化，停止图鉴定位')
            return img

        from mm_assist.nav.compat import prefer_visible_clicks

        drag_method = (
            "sendinput"
            if (in_test_mode() or prefer_visible_clicks())
            else "postmessage"
        )

        def _drag_fn(dx: int, dy: int) -> None:
            nonlocal win, img
            if world_resource and current_world_resource(img) != world_resource:
                raise GuardError('当前大陆地图资源已变化，停止拖图')
            _drag_map(win, dx=int(dx), dy=int(dy), force_method=drag_method)
            # 拖完停稳再 localize
            win, img, ok = _wait_frame_stable(
                cfg, win, timeout_s=_map_ui_settle("drag")
            )
            if not ok:
                log.warning("continent drag frame not stable")

        log.info(
            "continent: atlas drag seek %s min_score=%.2f method=%s",
            target_canon,
            min_score,
            drag_method,
        )
        near = drag_until_near(
            atlas=atlas,
            target_xy=target,
            N=N,
            min_score=min_score,
            max_steps=10,
            snap_fn=_snap_fn,
            localize_fn=lambda live, atl: localize_viewport_zoom_search(
                live, atl, min_score=min_score
            ),
            drag_fn=_drag_fn,
        )
        def _click_castle_or_atlas() -> tuple[str, float, float]:
            """优先实机城标模板；否则用图鉴锚点投影。"""
            assert img is not None
            nx, ny = atlas_xy_to_client_norm(
                near.loc, target[0], target[1], img.shape[1], img.shape[0]
            )
            cn = _find_dubarton_castle_norm(img, win)
            if cn is not None:
                return "castle_tmpl", cn[0], cn[1]
            return "atlas_xy", nx, ny

        if near.ok and near.loc is not None and near.loc.ok:
            assert img is not None
            how, nx, ny = _click_castle_or_atlas()
            log.info("continent: click 杜巴頓 via %s (%.3f,%.3f)", how, nx, ny)
            win, before = _grab_map_frame(cfg, win, settle_s=0.0)
            _click_norm(win, nx, ny)
            win, img, changed, delta = _wait_action_effect(
                cfg, win, before, timeout_s=_map_ui_settle("click"), threshold=2.5
            )
            log.info("continent click effect changed=%s delta=%.2f", changed, delta)
            win, img, lines, _ = snap(cfg, win, min_conf=0.28, grab_settle_s=0.05)
            b = _blob(lines)
            lv = ('town' if _current_map_parameters(img, win) == 'map_dunbarton'
                  else _map_level(b, lines, win))
            # 未进详图：再试一次城标模板（勿依赖 OCR）
            if lv not in ("town",) and img is not None:
                cn = _find_dubarton_castle_norm(img, win)
                if cn is not None:
                    log.info("continent: re-click castle tmpl (%.3f,%.3f)", cn[0], cn[1])
                    win, before = _grab_map_frame(cfg, win, settle_s=0.0)
                    _click_norm(win, cn[0], cn[1])
                    win, img, changed, delta = _wait_action_effect(
                        cfg, win, before, timeout_s=_map_ui_settle("click"), threshold=2.5
                    )
                    win, img, lines, _ = snap(
                        cfg, win, min_conf=0.28, grab_settle_s=0.05
                    )
                    b = _blob(lines)
                    lv = ('town' if _current_map_parameters(img, win) == 'map_dunbarton'
                          else _map_level(b, lines, win))
            ok = lv == "town" or "杜巴" in b[:100]
            return {
                "ok": ok,
                "stage": "continent_enter",
                "level": lv,
                "blob": b[:100],
                "near": f"{near.note}|{how}",
                "score": near.loc.score,
                "win": win,
                "error": None if ok else "点击后未进杜巴 town",
            }

        # 图鉴分不够：先全屏 OCR 复核是否其实已在杜巴 town（误判 continent 会乱拖）
        win, img, lines, _ = snap(cfg, win, min_conf=0.28, grab_settle_s=0.05)
        b = _blob(lines)
        lv = ('town' if _current_map_parameters(img, win) == 'map_dunbarton'
              else _map_level(b, lines, win))
        if lv == "town" or (
            "杜巴" in b[:100]
            and any(
                k in b
                for k in ("鐵匠", "铁匠", "截匠", "旅館", "旅馆", "銀行", "银行", "雜貨")
            )
        ):
            log.warning(
                "continent: atlas miss (%s) but OCR says town → skip blind drag",
                getattr(near, "note", None),
            )
            return {
                "ok": True,
                "stage": "already_town",
                "level": "town",
                "blob": b[:100],
                "near": f"recheck_town|{getattr(near, 'note', '')}",
                "score": getattr(near.loc, "score", None),
                "win": win,
                "error": None,
            }

        # 图鉴未对准：诚实失败，禁 Tir 方向盲拖（现场 Δ0 / need_repair 债口）
        note = getattr(near, "note", None)
        log.warning("continent: atlas miss (%s) → fail-closed (no blind drag)", note)
        return {
            "ok": False,
            "stage": "continent_atlas_miss",
            "error": f"找城失败:图鉴未对准（{note}）·禁盲拖",
            "score": getattr(near.loc, "score", None),
            "level": lv,
            "blob": b[:100],
            "win": win,
        }
    finally:
        release_mouse_buttons()


def _map_goto_npc_arrive(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    img: np.ndarray,
    lines: list[OcrLine],
    region: str,
    npc: str,
    npc_needles: tuple[str, ...],
    facility_needles: tuple[str, ...],
    level: str,
    timeout_s: float,
    idle_abort_s: float,
    allow_smith: bool,
    wait_arrive: bool = True,
) -> dict:
    """杜巴頓 town 内：OCR/图标拖扫点头像出卡 → 前往 → 停走/对话。

    不用 town atlas 像素路；``allow_smith`` 等保留签名兼容。
    """
    del img, facility_needles, allow_smith, npc_needles
    b0 = _blob(lines)
    title_ok = False
    for L in lines:
        c = box_center_norm(L, win.width, win.height)
        if not c or c[1] > 0.10 or c[0] > 0.55:
            continue
        t = normalize_text(L.text or "")
        if "杜巴" in t:
            title_ok = True
            break
        if any(k in t for k in ("库漢", "庫漢", "库汉", "尚库", "尚庫")):
            return {
                "ok": False,
                "stage": "region",
                "error": "顶栏非杜巴頓（他城），拒点頭像",
                "level": level,
                "blob": b0[:80],
            }
    if not title_ok and "杜巴" in b0[:100]:
        title_ok = True
    if not title_ok:
        return {
            "ok": False,
            "stage": "region",
            "error": "顶栏无杜巴，拒走近距奈麗絲",
            "level": level,
            "blob": b0[:80],
        }
    if (
        sum(1 for k in ("莱比", "萊比", "希德", "冰霜", "库漢", "庫漢", "库汉") if k in b0)
        >= 2
    ):
        return {
            "ok": False,
            "stage": "region",
            "error": "当前为大陆/多城视野（近距仅杜巴頓 town）",
            "level": level,
            "blob": b0[:100],
        }

    from mm_assist.nav.compat import timed_action

    click = town_pixel_click_npc(
        cfg,
        win,
        region=region,
        npc=npc,
    )
    if not click.get("ok"):
        return {
            "ok": False,
            "stage": str(click.get("stage") or "npc"),
            "error": str(click.get("error") or "town_pixel_click_npc failed"),
            "level": level,
            "in_town": True,
            "verified": False,
            "pixel": click,
        }

    win, img, lines, _ = snap(cfg, win)
    bb = _blob(lines)
    goto = resolve_goto_click(lines, img, win, allow_fallback=False)
    with timed_action("step_verify_goto_btn", note="") as ctx:
        ctx["ok"] = bool(
            goto and (goto.kind == "green" or "前往" in (goto.text or ""))
        )
        if goto:
            ctx["note"] = f"{goto.kind}:{goto.text}"
    if not goto:
        return {
            "ok": False,
            "stage": "goto",
            "error": "无前往/绿钮（已出卡；禁止 GOTO_FALLBACK）",
            "level": level,
            "blob": bb[:120],
            "verified": False,
            "pixel": click,
            "card_ok": True,
        }

    with timed_action("green_travel", note=f"npc:{npc}") as gctx:
        gctx["ok"] = goto.kind == "green" or "前往" in (goto.text or "")
        gctx["note"] = f"{goto.kind}:{goto.text}@({goto.nx:.2f},{goto.ny:.2f})"
    _click_norm(win, goto.nx, goto.ny)
    # 点完前往后短静默，避免立刻 OCR/关图打断自动走（旧 1s）
    time.sleep(_map_ui_settle("after_goto"))
    if not wait_arrive:
        return {
            "ok": True,
            "stage": "walking",
            "detail": "已点前往，到站等待不占用点击租约",
            "region": region,
            "npc": npc,
            "level": level,
            "verified": False,
            "pixel": click,
        }

    with timed_action("map_goto", note=f"npc_arrive:{npc}") as mctx:
        # idle_abort_s<=0 → 一直等对话；timeout 放宽
        wait_timeout = timeout_s if idle_abort_s > 0 else 0.0
        ok, st, detail = wait_stop_then_dialog(
            cfg,
            win,
            timeout_s=wait_timeout if wait_timeout > 0 else 0.0,
            idle_abort_s=idle_abort_s,
            settle_s=0.0,  # 已在上方短静默
        )
        mctx["ok"] = bool(ok)
        mctx["note"] = detail or str(st)
    return {
        "ok": ok,
        "stage": "arrive",
        "state": st,
        "detail": detail,
        "region": region,
        "npc": npc,
        "level": level,
        "clicked": npc,
        "goto_kind": goto.kind,
        "in_town": True,
        "pixel": click,
    }


def run_probe_cli(cfg: AppConfig, win: WindowInfo) -> int:
    s = probe(cfg, win)
    print(f"state={s.state}")
    print(f"capture={s.path}")
    print(f"ocr_lines={len(s.lines)}")
    print(f"region_candidates={len(s.regions)}")
    for h in s.regions[:40]:
        print(f"  [{h.kind}] {h.text!r} @ ({h.nx:.3f},{h.ny:.3f}) score={h.score:.2f}")
    sample = [L.text for L in s.lines[:25]]
    print("ocr_sample:", sample)
    return 0


def run_open_cli(cfg: AppConfig, win: WindowInfo) -> int:
    s = ensure_continent(cfg, win)
    path = save_regions_json(cfg, s.regions, state=s.state)
    print(f"state={s.state} regions={len(s.regions)} -> {path}")
    if s.path:
        print(f"capture={s.path}")
    for h in s.regions[:40]:
        print(f"  [{h.kind}] {h.text!r} @ ({h.nx:.3f},{h.ny:.3f})")
    return 0 if s.state in ("continent", "region") else 1


def run_scan_cli(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    click_each: bool,
    pan: bool = True,
    passes: int = 1,
) -> int:
    path, hits = scan_and_capture_regions(
        cfg, win, click_each=click_each, pan=pan, passes=passes
    )
    print(f"wrote {path} count={len(hits)} pan={pan} passes={passes} click_each={click_each}")
    for h in hits[:50]:
        print(f"  [{h.kind}] {h.text!r} @ ({h.nx:.3f},{h.ny:.3f})")
    tiles = maps_root(cfg) / "tiles_index.json"
    if tiles.is_file():
        print(f"tiles_index={tiles}")
    return 0


def run_goto_cli(
    cfg: AppConfig,
    win: WindowInfo,
    region: str,
    npc: str,
    *,
    timeout_s: float = 60.0,
) -> int:
    r = map_goto(cfg, win, region=region, npc=npc, timeout_s=timeout_s)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0 if r.get("ok") else 1


def run_atlas_build_cli(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    passes: int = 2,
) -> int:
    """实机拖扫拼歐拉图鉴；成功条件：json 含杜巴頓。"""
    from mm_assist.nav.map_localize import has_dubarton

    try:
        png, js, meta = build_ulara_atlas(cfg, win, passes=passes)
    except RuntimeError as exc:
        print(f"FAIL {exc}")
        return 2
    places = meta.get("places") or []
    print(f"atlas={png}")
    print(f"meta={js}")
    print(f"size={meta.get('width')}x{meta.get('height')} places={len(places)}")
    print(f"has_dubarton={has_dubarton(meta)}")
    for p in places[:30]:
        print(
            f"  {p.get('canon')!r} {p.get('text')!r} @ ({p.get('ax')},{p.get('ay')})"
            f" {p.get('source') or ''}"
        )
    return 0 if has_dubarton(meta) else 2
