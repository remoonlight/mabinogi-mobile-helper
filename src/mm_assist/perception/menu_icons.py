"""菜单入口与子页识别：模糊图标为主，PP-OCRv5 文字为辅。"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from mm_assist.safety.input import click_client, release_mouse_buttons
from mm_assist.platform.capture import grab_client
from mm_assist.config import ROOT, AppConfig, load_config
from mm_assist.platform.match import MatchHit, find_hamburger_bars, match_icon_fuzzy, match_template
from mm_assist.platform.window import WindowInfo, find_window

log = logging.getLogger(__name__)

MENU_ENTRY_TEMPLATE = ROOT / "knowledge/templates/menu/icon_menu_52x44.png"

MENU_ICON_THRESHOLD = 0.72
MENU_ICON_MIN_MARGIN = 0.02

# 树节点 id
MENU_ENTRY = "menu_entry"
CHARACTER = "character"
LIFE_SKILLS = "life_skills"
WOODCUT = "woodcut"
MINE = "mine"
HERB = "herb"
SHEAR = "shear"
HARVEST = "harvest"
HOE = "hoe"
INSECT = "insect"
FISH = "fish"
DAILY = "daily"
INVENTORY = "inventory"
CRAFT = "craft"
CRAFT_METAL = "craft_metal"
CRAFT_WOOD = "craft_wood"
CRAFT_LEATHER = "craft_leather"
CRAFT_CLOTH = "craft_cloth"
CRAFT_MEDICINE = "craft_medicine"
CRAFT_FOOD = "craft_food"
EXCHANGE = "exchange"

MENU_NODE_IDS: tuple[str, ...] = (
    MENU_ENTRY,
    CHARACTER,
    LIFE_SKILLS,
    WOODCUT,
    MINE,
    HERB,
    SHEAR,
    HARVEST,
    HOE,
    INSECT,
    FISH,
    DAILY,
    INVENTORY,
    CRAFT,
    CRAFT_METAL,
    CRAFT_WOOD,
    CRAFT_LEATHER,
    CRAFT_CLOTH,
    CRAFT_MEDICINE,
    CRAFT_FOOD,
    EXCHANGE,
)

_TEMPLATE_REL_BY_NODE: dict[str, str | None] = {
    # 2026-08-04：实机裁切模板；文字仅作辅判（PP-OCRv5）
    MENU_ENTRY: "knowledge/templates/menu/icon_menu_52x44.png",  # ← hud_menu0 ESC
    CHARACTER: "knowledge/templates/menu/icon_character.png",  # ← esc_root 角色
    LIFE_SKILLS: "knowledge/templates/menu/icon_life_skills.png",  # ← life_leaves 侧栏
    WOODCUT: "knowledge/templates/menu/icon_woodcut.png",
    MINE: "knowledge/templates/menu/icon_mine.png",
    HERB: "knowledge/templates/menu/icon_herb.png",
    SHEAR: "knowledge/templates/menu/icon_shear.png",
    HARVEST: "knowledge/templates/menu/icon_harvest.png",
    HOE: "knowledge/templates/menu/icon_hoe.png",
    INSECT: "knowledge/templates/menu/icon_insect.png",
    FISH: "knowledge/templates/menu/icon_fish.png",
    DAILY: "knowledge/templates/menu/icon_daily.png",
    INVENTORY: "knowledge/templates/menu/icon_inventory.png",
    CRAFT: "knowledge/templates/menu/icon_crafting.png",
    CRAFT_METAL: "knowledge/templates/craft_hub/card_metal.png",
    CRAFT_WOOD: "knowledge/templates/craft_hub/card_wood.png",
    CRAFT_LEATHER: "knowledge/templates/craft_hub/card_leather.png",
    CRAFT_CLOTH: "knowledge/templates/craft_hub/card_cloth.png",
    CRAFT_MEDICINE: "knowledge/templates/craft_hub/card_medicine.png",
    CRAFT_FOOD: "knowledge/templates/craft_hub/card_food.png",
    EXCHANGE: "knowledge/templates/menu/icon_exchange.png",
}


def _resolve_template_path(name_or_path: str | Path) -> Path | None:
    key = str(name_or_path)
    rel = _TEMPLATE_REL_BY_NODE.get(key)
    if rel is not None:
        return ROOT / rel
    p = Path(name_or_path)
    if p.is_file():
        return p
    rooted = ROOT / key
    return rooted if rooted.is_file() else None


def _hit_to_dict(hit: MatchHit) -> dict[str, Any]:
    return {
        "score": hit.score,
        "cx": hit.cx,
        "cy": hit.cy,
        "x": hit.x,
        "y": hit.y,
        "w": hit.w,
        "h": hit.h,
        "margin": hit.margin,
    }


def find_template(
    name_or_path: str | Path,
    frame: np.ndarray,
    *,
    roi: tuple[float, float, float, float] | None = None,
    threshold: float = MENU_ICON_THRESHOLD,
    min_margin: float = MENU_ICON_MIN_MARGIN,
) -> MatchHit | None:
    path = _resolve_template_path(name_or_path)
    if path is None or not path.is_file():
        return None
    return match_template(
        frame,
        path,
        roi=roi,
        threshold=threshold,
        min_margin=min_margin,
    )


def find_menu_entry(
    bgr_or_gray_frame: np.ndarray,
    *,
    roi: tuple[float, float, float, float] | None = None,
    threshold: float = MENU_ICON_THRESHOLD,
    min_margin: float = MENU_ICON_MIN_MARGIN,
) -> MatchHit | None:
    if not MENU_ENTRY_TEMPLATE.is_file():
        raise FileNotFoundError(f"menu entry template missing: {MENU_ENTRY_TEMPLATE}")
    return match_template(
        bgr_or_gray_frame,
        MENU_ENTRY_TEMPLATE,
        roi=roi,
        threshold=threshold,
        min_margin=min_margin,
    )


def probe_menu_entry_peak(
    bgr_or_gray_frame: np.ndarray,
    *,
    roi: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """只读：菜单入口模板在 ROI 内的峰分（不点、不抬阈值门槛）。

    用于区分「遮挡低峰」与「真 miss」。roi 默认右上角菜单区。
    """
    use_roi = _ROI_MENU_ENTRY if roi is None else roi
    if not MENU_ENTRY_TEMPLATE.is_file():
        return {"peak": None, "reason": "missing_template", "roi": list(use_roi)}
    hit = match_template(
        bgr_or_gray_frame,
        MENU_ENTRY_TEMPLATE,
        roi=use_roi,
        threshold=0.0,
        min_margin=0.0,
        check_margin=False,
    )
    if hit is None:
        return {"peak": None, "reason": "no_peak", "roi": list(use_roi)}
    return {
        "peak": float(hit.score),
        "margin": float(hit.margin),
        "cx": int(hit.cx),
        "cy": int(hit.cy),
        "xy": [int(hit.x), int(hit.y)],
        "roi": list(use_roi),
        "reason": "ok",
    }


def probe_menu_icons(
    frame: np.ndarray,
    *,
    roi: tuple[float, float, float, float] | None = None,
    threshold: float = MENU_ICON_THRESHOLD,
    min_margin: float = MENU_ICON_MIN_MARGIN,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for node_id in MENU_NODE_IDS:
        rel = _TEMPLATE_REL_BY_NODE.get(node_id)
        if not rel:
            out[node_id] = {"missing_template": True}
            continue
        path = ROOT / rel
        if not path.is_file():
            out[node_id] = {"missing_template": True}
            continue
        hit = match_template(
            frame,
            path,
            roi=roi,
            threshold=threshold,
            min_margin=min_margin,
        )
        out[node_id] = {"hit": _hit_to_dict(hit) if hit is not None else None}
    return out


# 快路径 ROI（归一化）；状态栏用
_ROI_MENU_ENTRY = (0.78, 0.0, 0.22, 0.14)
# 系统菜单右上四宫格（与 path_auto._MENU_COLS / overlay menu.char 对齐）
# 列：角色 / 背包 / 製作 / 鍊金·交易所；ny≈0.20
# 列：角色 / 背包 / 製作 / 鍊金·交易所（1920×1440 菜单实机 2026-08-28）
ESC_GRID_COLS = (0.688, 0.795, 0.902, 0.968)
ESC_GRID_NY = 0.21  # 製作格图标中心（icon_crafting 标定 y+h/2）
ESC_CHAR_NX = ESC_GRID_COLS[0]
ESC_CRAFT_NX = ESC_GRID_COLS[2]
ESC_EXCHANGE_NX = ESC_GRID_COLS[3]
# 标签「製作」中心向上到图标中心；0.055 会点到上一行缝
ESC_CRAFT_LABEL_UP_NY = 0.038
ROI_ESC_CRAFT = (0.82, 0.12, 0.14, 0.22)


def _esc_row_present(frame: Any, *, skip: str | None = None) -> bool:
    """系统菜单四宫格至少认出一格（不含 skip）。"""
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    cells = (
        (CHARACTER, ROI_ESC_CHAR, 0.45),
        (INVENTORY, (0.76, 0.10, 0.12, 0.26), 0.55),
        (CRAFT, ROI_ESC_CRAFT, 0.55),
    )
    for name, roi, th in cells:
        if skip is not None and name == skip:
            continue
        hit = find_template(name, frame, roi=roi, threshold=th)
        if hit is not None and int(hit.cy) < int(0.42 * int(frame.shape[0])):
            return True
    return False


def _is_menu_craft_label(text: str) -> bool:
    """只要菜单上的「製作 / 製作(K)」，不要「特製」「重甲製作」。"""
    from mm_assist.perception.item_sense import compact_zh

    b = compact_zh(text).lower()
    if not b:
        return False
    if any(x in b for x in ("特製", "特制", "重甲", "布衣", "輕甲", "轻甲", "前往", "成功", "貨幣", "货币")):
        return False
    core = b.replace("(", "").replace(")", "").replace("k", "")
    return core in {"製作", "制作"}


def _is_menu_char_label(text: str) -> bool:
    from mm_assist.perception.item_sense import compact_zh

    b = compact_zh(text)
    if not b:
        return False
    if "生活" in b or "技能" in b or "背包" in b or "製作" in b or "制作" in b:
        return False
    return b in {"角色", "人物"} or b.startswith("角色") or b.startswith("人物")


def locate_esc_char_by_label(frame: Any) -> tuple[int, int] | None:
    """系统菜单「角色」格：ROI 内图标，否则 OCR 标签上方。"""
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    from mm_assist.perception.ocr import ocr_roi

    h, w = int(frame.shape[0]), int(frame.shape[1])
    hit = find_template(CHARACTER, frame, roi=ROI_ESC_CHAR, threshold=0.45)
    if hit is not None:
        return int(hit.cx), int(hit.cy)
    if _esc_row_present(frame, skip=CHARACTER):
        return int(w * ESC_CHAR_NX), int(h * ESC_GRID_NY)
    try:
        lines = ocr_roi(frame, ROI_ESC_CHAR, min_confidence=0.35)
    except Exception:
        lines = []
    for ln in lines:
        if not _is_menu_char_label(getattr(ln, "text", "") or ""):
            continue
        box = getattr(ln, "box", None) or []
        xs = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
        ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
        if xs and ys:
            lx, ly = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
            return lx, max(8, ly - int(0.055 * h))
    return None


def locate_esc_cell_by_label(
    frame: Any,
    needles: tuple[str, ...],
    *,
    reject: tuple[str, ...] = (),
) -> tuple[int, int] | None:
    """系统菜单「製作」：先图标模板，再按「製作」标签上移到格心。"""
    _ = needles
    _ = reject
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    from mm_assist.perception.ocr import ocr_roi

    h, w = int(frame.shape[0]), int(frame.shape[1])
    hit = find_template(CRAFT, frame, roi=ROI_ESC_CRAFT, threshold=0.50)
    if hit is not None and int(hit.cy) < int(0.42 * h):
        return int(hit.cx), int(hit.cy)
    try:
        lines = ocr_roi(frame, (0.68, 0.14, 0.30, 0.28), min_confidence=0.35)
    except Exception:
        lines = []
    label_xy: tuple[int, int] | None = None
    for ln in lines:
        if not _is_menu_craft_label(getattr(ln, "text", "") or ""):
            continue
        box = getattr(ln, "box", None) or []
        xs = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
        ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
        if xs and ys:
            label_xy = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
            break
    if label_xy is not None:
        lx, ly = label_xy
        x0 = max(0.0, (lx / w) - 0.05)
        y0 = max(0.0, (ly / h) - 0.08)
        roi = (x0, y0, 0.10, 0.08)
        hit = find_template(CRAFT, frame, roi=roi, threshold=0.55)
        if hit is not None and int(hit.cy) < int(0.42 * h):
            return int(hit.cx), int(hit.cy)
        return lx, max(8, ly - int(ESC_CRAFT_LABEL_UP_NY * h))
    for roi, th in (
        ((0.80, 0.12, 0.16, 0.22), 0.55),
        ((0.70, 0.08, 0.28, 0.36), 0.58),
    ):
        hit = find_template(CRAFT, frame, roi=roi, threshold=th)
        if hit is not None and int(hit.cy) < int(0.42 * h):
            return int(hit.cx), int(hit.cy)
    # 角色/背包格在则四宫格第三格就是製作
    if _esc_row_present(frame):
        return int(w * ESC_CRAFT_NX), int(h * ESC_GRID_NY)
    return None


# 仅角色格（左格 nx 0.66–0.80），不含製作格 0.872
ROI_ESC_CHAR = (0.66, 0.08, 0.14, 0.28)
# 四宫格整带（背包/製作/交易所）；角色请用 ROI_ESC_CHAR
_ROI_ESC = (0.68, 0.08, 0.28, 0.36)
_ROI_LIFE = (0.02, 0.08, 0.98, 0.88)
_ROI_CRAFT = (0.10, 0.08, 0.88, 0.88)

_ESC_NODES: tuple[str, ...] = (CHARACTER, INVENTORY, CRAFT, EXCHANGE)
_LIFE_NODES: tuple[str, ...] = (
    LIFE_SKILLS,
    WOODCUT,
    MINE,
    HERB,
    SHEAR,
    HARVEST,
    HOE,
    INSECT,
    FISH,
    DAILY,
)
_CRAFT_NODES: tuple[str, ...] = (
    CRAFT_METAL,
    CRAFT_WOOD,
    CRAFT_LEATHER,
    CRAFT_CLOTH,
    CRAFT_MEDICINE,
    CRAFT_FOOD,
)

MENU_NAV_SCALE = 0.5
MENU_NAV_FAST_THRESHOLD = 0.58
_FAST_FUZZY_SCALES = (0.5, 0.75, 1.0, 1.15)


def _match_node_fast(
    frame: np.ndarray,
    node_id: str,
    *,
    roi: tuple[float, float, float, float] | None,
    threshold: float,
    scale: float,
) -> dict[str, Any]:
    rel = _TEMPLATE_REL_BY_NODE.get(node_id)
    if not rel:
        return {"missing_template": True}
    path = ROOT / rel
    if not path.is_file():
        return {"missing_template": True}
    scales = tuple(sorted({float(scale), *_FAST_FUZZY_SCALES}))
    hit = match_icon_fuzzy(
        frame,
        path,
        roi=roi,
        threshold=min(float(threshold), 0.62),
        scales=scales,
    )
    source = "fuzzy"
    if hit is None and node_id == MENU_ENTRY:
        hit = find_hamburger_bars(frame, roi=roi or _ROI_MENU_ENTRY)
        source = "bars"
    payload = _hit_to_dict(hit) if hit is not None else None
    if payload is not None:
        payload["source"] = source
    return {"hit": payload}


def _ocr_blob(frame: np.ndarray, roi: tuple[float, float, float, float]) -> tuple[str, float]:
    """当前 ROI 只跑一次 OCR；失败返回空串。"""
    from mm_assist.perception.ocr import ocr_roi

    try:
        lines = ocr_roi(frame, roi, min_confidence=0.45)
    except Exception as exc:
        log.debug("menu ocr: %s", exc)
        return "", 0.0
    blob = "".join((ln.text or "").replace(" ", "") for ln in lines)
    conf = max((float(ln.confidence) for ln in lines), default=0.0)
    return blob, conf


# 繁简别名；长词优先匹配
_NODE_TEXT_ALIASES: dict[str, tuple[str, ...]] = {
    CHARACTER: ("角色界面", "人物界面", "角色", "人物"),
    LIFE_SKILLS: ("生活技能", "生活技", "生活"),
    WOODCUT: ("伐木", "伐木"),
    MINE: ("采矿", "采矿", "挖矿"),
    HERB: ("药草", "药草", "采药"),
    SHEAR: ("剪羊毛", "剪羊毛"),
    HARVEST: ("收割", "收割"),
    HOE: ("锄地", "锄地"),
    INSECT: ("昆虫采集", "昆虫采集", "昆虫"),
    FISH: ("钓鱼", "钓鱼"),
    DAILY: ("日常采集", "日常采集"),
    INVENTORY: ("背包", "物品栏", "物品"),
    CRAFT: ("制作", "制作"),
    CRAFT_METAL: ("金属加工", "金属加工"),
    CRAFT_WOOD: ("木材加工", "木材加工"),
    CRAFT_LEATHER: ("皮革加工", "皮革加工"),
    CRAFT_CLOTH: ("布料加工", "布料加工"),
    CRAFT_MEDICINE: ("药品加工", "药品加工"),
    CRAFT_FOOD: ("食材加工", "食材加工"),
    EXCHANGE: ("交易所", "交易所"),
    MENU_ENTRY: ("菜单", "系统"),
}


def _apply_ocr_to_nodes(
    out: dict[str, dict[str, Any]],
    blob: str,
    conf: float,
    node_ids: tuple[str, ...],
) -> None:
    if not blob:
        return
    needles: list[tuple[str, str]] = []
    allowed = set(node_ids)
    for nid, aliases in _NODE_TEXT_ALIASES.items():
        if nid not in allowed:
            continue
        for alias in aliases:
            needles.append((alias, nid))
    needles.sort(key=lambda t: len(t[0]), reverse=True)
    claimed: set[str] = set()
    for alias, nid in needles:
        if nid in claimed:
            continue
        if alias not in blob:
            continue
        claimed.add(nid)
        row = out.setdefault(nid, {"hit": None})
        hit = row.get("hit")
        if isinstance(hit, dict):
            hit["ocr"] = True
            continue
        row["hit"] = {
            "score": max(0.51, min(0.9, float(conf) or 0.55)),
            "cx": 0,
            "cy": 0,
            "x": 0,
            "y": 0,
            "w": 0,
            "h": 0,
            "margin": 0.0,
            "source": "ocr",
        }


def probe_menu_icons_fast(
    frame: np.ndarray,
    *,
    menu_open: bool | None = None,
    threshold: float = MENU_NAV_FAST_THRESHOLD,
    scale: float = MENU_NAV_SCALE,
) -> dict[str, dict[str, Any]]:
    """菜单快路径：模糊图标为主、ROI OCR 为辅；menu_open=False 时只认入口。"""
    out: dict[str, dict[str, Any]] = {nid: {"hit": None} for nid in MENU_NODE_IDS}
    out[MENU_ENTRY] = _match_node_fast(
        frame, MENU_ENTRY, roi=_ROI_MENU_ENTRY, threshold=threshold, scale=scale
    )
    if menu_open is not False:
        blob_e, conf_e = _ocr_blob(frame, _ROI_MENU_ENTRY)
        _apply_ocr_to_nodes(out, blob_e, conf_e, (MENU_ENTRY,))

    # 主界面：不必扫子树
    if menu_open is False:
        return out

    for nid in _ESC_NODES:
        roi = ROI_ESC_CHAR if nid == CHARACTER else _ROI_ESC
        out[nid] = _match_node_fast(
            frame, nid, roi=roi, threshold=threshold, scale=scale
        )
    blob_esc, conf_esc = _ocr_blob(frame, _ROI_ESC)
    _apply_ocr_to_nodes(out, blob_esc, conf_esc, _ESC_NODES)

    # 哨兵：生活侧栏 / 任一加工图
    out[LIFE_SKILLS] = _match_node_fast(
        frame, LIFE_SKILLS, roi=_ROI_LIFE, threshold=threshold, scale=scale
    )
    out[CRAFT_METAL] = _match_node_fast(
        frame, CRAFT_METAL, roi=_ROI_CRAFT, threshold=threshold, scale=scale
    )

    life_hit = (out.get(LIFE_SKILLS) or {}).get("hit")
    # 仅侧栏命中时扫生活叶子（避免 ESC 根菜单白扫 8 张）
    if life_hit is not None:
        for nid in _LIFE_NODES:
            if nid == LIFE_SKILLS:
                continue
            out[nid] = _match_node_fast(
                frame, nid, roi=_ROI_LIFE, threshold=threshold, scale=scale
            )
        blob_life, conf_life = _ocr_blob(frame, _ROI_LIFE)
        _apply_ocr_to_nodes(out, blob_life, conf_life, _LIFE_NODES)

    craft_hit = (out.get(CRAFT_METAL) or {}).get("hit")
    # 仅加工页哨兵命中时扫六加工（勿因 ESC「制作」入口误扩）
    if craft_hit is not None:
        for nid in _CRAFT_NODES:
            if nid == CRAFT_METAL:
                continue
            out[nid] = _match_node_fast(
                frame, nid, roi=_ROI_CRAFT, threshold=threshold, scale=scale
            )
        blob_c, conf_c = _ocr_blob(frame, _ROI_CRAFT)
        _apply_ocr_to_nodes(out, blob_c, conf_c, _CRAFT_NODES)

    return out


# 状态栏 nav= 展示优先级：叶子 > 中层 > 入口（同分取 score 高）
_NAV_PRIORITY: dict[str, int] = {
    WOODCUT: 30,
    MINE: 30,
    HERB: 30,
    SHEAR: 30,
    HARVEST: 30,
    HOE: 30,
    INSECT: 30,
    FISH: 30,
    DAILY: 30,
    CRAFT_METAL: 30,
    CRAFT_WOOD: 30,
    CRAFT_LEATHER: 30,
    CRAFT_CLOTH: 30,
    CRAFT_MEDICINE: 30,
    CRAFT_FOOD: 30,
    LIFE_SKILLS: 20,
    CHARACTER: 20,
    INVENTORY: 20,
    CRAFT: 20,
    EXCHANGE: 20,
    MENU_ENTRY: 10,
}

# 状态栏 nav= 中文；含义见 docs/REQ_UI_PARAMS.md「nav 含义」
_NAV_LABEL_ZH: dict[str, str] = {
    MENU_ENTRY: "主界面",  # 可按主菜单按钮
    CHARACTER: "角色",  # 主菜单 · 可按角色
    LIFE_SKILLS: "角色界面",  # 可按生活技能按钮
    WOODCUT: "伐木",  # 生活技能界面
    MINE: "采矿",
    HERB: "药草",
    SHEAR: "剪羊毛",
    HARVEST: "收割",
    HOE: "锄地",
    INSECT: "昆虫采集",
    FISH: "钓鱼",
    DAILY: "日常採集",
    INVENTORY: "背包",  # 主菜单 · 可按背包
    CRAFT: "制作",  # 主菜单 · 可按制作
    CRAFT_METAL: "金属加工",  # 加工界面
    CRAFT_WOOD: "木材加工",
    CRAFT_LEATHER: "皮革加工",
    CRAFT_CLOTH: "布料加工",
    CRAFT_MEDICINE: "药品加工",
    CRAFT_FOOD: "食材加工",
    EXCHANGE: "交易所",  # 主菜单 · 可按交易所
}


def format_menu_nav(
    probe: dict[str, dict[str, Any]] | None,
    *,
    max_nodes: int = 3,
) -> str:
    """把 probe 收成内部 ``nav=伐木,角色``（不上状态栏；中文见 ``_NAV_LABEL_ZH``）。无命中返回空串。"""
    if not probe:
        return ""
    scored: list[tuple[str, float]] = []
    for nid, row in probe.items():
        if not isinstance(row, dict):
            continue
        hit = row.get("hit")
        if not isinstance(hit, dict):
            continue
        try:
            sc = float(hit.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        if sc <= 0:
            continue
        scored.append((nid, sc))
    if not scored:
        return ""
    scored.sort(
        key=lambda t: (_NAV_PRIORITY.get(t[0], 0), t[1]),
        reverse=True,
    )
    labels = [
        _NAV_LABEL_ZH.get(n, n) for n, _ in scored[: max(1, int(max_nodes))]
    ]
    return "nav=" + ",".join(labels)


def click_menu_entry(
    *,
    input_method: str = "sendinput",
    roi: tuple[float, float, float, float] | None = None,
    threshold: float = MENU_ICON_THRESHOLD,
    min_margin: float = MENU_ICON_MIN_MARGIN,
    cfg: AppConfig | None = None,
    win: WindowInfo | None = None,
) -> dict[str, Any]:
    """对图命中菜单入口后点一次；未命中或缺窗则 fail-closed（不盲点）。"""
    cfg = cfg or load_config()
    win = win or find_window(cfg.window.title_substr) or find_window("Mobile")
    if win is None:
        return {"ok": False, "reason": "no_window"}

    clicked = False
    try:
        img = grab_client(win, method=cfg.window.capture_method)
        hit = find_menu_entry(
            img,
            roi=roi,
            threshold=threshold,
            min_margin=min_margin,
        )
        if hit is None:
            return {"ok": False, "reason": "not_found"}
        cx, cy = int(hit.cx), int(hit.cy)
        click_client(
            win,
            cx,
            cy,
            method=input_method,
            refocus=True,
            hold_s=0.08,
        )
        clicked = True
        log.info(
            "menu entry icon click @%d,%d score=%.3f margin=%.3f",
            cx,
            cy,
            hit.score,
            hit.margin,
        )
        return {"ok": True, "cx": cx, "cy": cy, "hit": _hit_to_dict(hit)}
    finally:
        if clicked:
            release_mouse_buttons()


# 生活技能叶子（C2 open_life_skill）
LIFE_SKILL_LEAF_IDS: tuple[str, ...] = (
    WOODCUT,
    MINE,
    HERB,
    SHEAR,
    HARVEST,
    HOE,
    INSECT,
    FISH,
    DAILY,
)

_UI_LABEL_TO_LEAF: dict[str, str] = {
    "伐木": WOODCUT,
    "采矿": MINE,
    "采集药草": HERB,
    "採集藥草": HERB,
    "剪羊毛": SHEAR,
    "收割": HARVEST,
    "锄地": HOE,
    "鋤地": HOE,
    "昆虫采集": INSECT,
    "昆蟲採集": INSECT,
    "钓鱼": FISH,
    "釣魚": FISH,
    "日常採集": DAILY,
    "日常采集": DAILY,
    "采矿": MINE,
    "採礦": MINE,
}


def leaf_id_for_ui_label(ui_label: str) -> str | None:
    """状态下拉 ui_label → 菜单叶子 node id。"""
    key = (ui_label or "").strip()
    if key in _UI_LABEL_TO_LEAF:
        return _UI_LABEL_TO_LEAF[key]
    if key in LIFE_SKILL_LEAF_IDS:
        return key
    return None


def _roi_for_node(node_id: str) -> tuple[float, float, float, float] | None:
    if node_id == MENU_ENTRY:
        return _ROI_MENU_ENTRY
    if node_id == CHARACTER:
        return ROI_ESC_CHAR
    if node_id in _ESC_NODES:
        return _ROI_ESC
    if node_id in _LIFE_NODES:
        return _ROI_LIFE
    if node_id in _CRAFT_NODES or node_id == CRAFT:
        return _ROI_CRAFT
    return None


def esc_char_grid_xy(win: Any) -> tuple[int, int]:
    """四宫格「角色」网格中心。与製作格 ESC_CRAFT_NX 不同列。"""
    w = max(int(getattr(win, "width", 0) or 0), 1)
    h = max(int(getattr(win, "height", 0) or 0), 1)
    return int(ESC_CHAR_NX * w), int(ESC_GRID_NY * h)


def xy_in_esc_craft_cell(cx: int, cy: int, win: Any) -> bool:
    """点位落在四宫格「製作」格（prf 禁点）。"""
    w = max(int(getattr(win, "width", 0) or 0), 1)
    nx = float(cx) / float(w)
    return abs(nx - ESC_CRAFT_NX) <= 0.04


def click_menu_node(
    node_id: str,
    *,
    cfg: AppConfig | None = None,
    win: WindowInfo | None = None,
    input_method: str = "sendinput",
    refocus: bool = True,
    threshold: float = MENU_ICON_THRESHOLD,
    min_margin: float = MENU_ICON_MIN_MARGIN,
    roi: tuple[float, float, float, float] | None = None,
    settle_s: float = 0.55,
    allow_full_scan: bool = True,
) -> dict[str, Any]:
    """对图命中指定菜单节点后点一次；未命中 fail-closed（不盲点）。

    ``character`` 默认禁全屏扫，避免点中左头像或製作格。
    """
    cfg = cfg or load_config()
    win = win or find_window(cfg.window.title_substr) or find_window("Mobile")
    if win is None:
        return {"ok": False, "reason": "no_window", "node": node_id}

    use_roi = roi if roi is not None else _roi_for_node(node_id)
    if node_id == CHARACTER:
        allow_full_scan = False
    clicked = False
    try:
        img = grab_client(win, method=cfg.window.capture_method)
        hit = find_template(
            node_id,
            img,
            roi=use_roi,
            threshold=threshold,
            min_margin=min_margin,
        )
        if hit is None and use_roi is not None and allow_full_scan:
            # ROI 未中再全屏扫一次（布局偏移）；角色格禁此回退
            hit = find_template(
                node_id,
                img,
                roi=None,
                threshold=threshold,
                min_margin=min_margin,
            )
        if hit is None:
            return {"ok": False, "reason": "not_found", "node": node_id}
        cx, cy = int(hit.cx), int(hit.cy)
        if node_id == CHARACTER and xy_in_esc_craft_cell(cx, cy, win):
            log.warning("menu node character hit craft cell @%d,%d; skip", cx, cy)
            return {
                "ok": False,
                "reason": "make_forbidden",
                "node": node_id,
                "cx": cx,
                "cy": cy,
            }
        click_client(
            win,
            cx,
            cy,
            method=input_method,
            refocus=refocus,
            hold_s=0.08,
        )
        clicked = True
        if settle_s > 0:
            time.sleep(settle_s)
        log.info(
            "menu node %s click @%d,%d score=%.3f",
            node_id,
            cx,
            cy,
            hit.score,
        )
        return {
            "ok": True,
            "node": node_id,
            "cx": cx,
            "cy": cy,
            "hit": _hit_to_dict(hit),
        }
    finally:
        if clicked:
            release_mouse_buttons()


def open_life_skill(
    skill_node_id: str,
    *,
    cfg: AppConfig | None = None,
    win: WindowInfo | None = None,
    input_method: str = "unity_hook",
    refocus: bool = False,
    settle_s: float = 0.55,
    threshold: float = MENU_ICON_THRESHOLD,
) -> dict[str, Any]:
    """【旁证/deprecated 主闸】模板路径开生活技能叶子。

    C2 生产主路径走 ``skill_train._open_skill_ocr``，勿再以本函数为主闸。
    模板资产仍供 nav 探测 / 单测旁证。每步对图失败即停；禁文字 OCR 认菜单名。
    """
    leaf = (skill_node_id or "").strip()
    if leaf not in LIFE_SKILL_LEAF_IDS:
        return {"ok": False, "reason": "bad_skill", "node": leaf}

    cfg = cfg or load_config()
    win = win or find_window(cfg.window.title_substr) or find_window("Mobile")
    if win is None:
        return {"ok": False, "reason": "no_window"}

    steps: list[str] = []

    def _frame() -> np.ndarray:
        return grab_client(win, method=cfg.window.capture_method)

    img = _frame()
    if find_template(leaf, img, roi=_ROI_LIFE, threshold=threshold) is not None:
        return {"ok": True, "already": True, "steps": steps, "node": leaf}

    # 已在生活技能页（侧栏或任一叶子可见）→ 只点目标叶子
    life_open = find_template(LIFE_SKILLS, img, roi=_ROI_LIFE, threshold=threshold)
    any_leaf = any(
        find_template(n, img, roi=_ROI_LIFE, threshold=threshold) is not None
        for n in LIFE_SKILL_LEAF_IDS
    )
    if life_open is not None or any_leaf:
        r = click_menu_node(
            leaf,
            cfg=cfg,
            win=win,
            input_method=input_method,
            refocus=refocus,
            settle_s=settle_s,
            threshold=threshold,
        )
        if not r.get("ok"):
            return {
                "ok": False,
                "reason": r.get("reason") or "not_found",
                "failed_at": leaf,
                "steps": steps,
            }
        steps.append(leaf)
        return {"ok": True, "steps": steps, "node": leaf, "from": "life_page"}

    # ESC 根菜单（角色/背包等）→ 生活技能 → 叶子
    esc_open = any(
        find_template(n, img, roi=_ROI_ESC, threshold=threshold) is not None
        for n in _ESC_NODES
    )
    path: list[str]
    if esc_open:
        path = [LIFE_SKILLS, leaf]
        from_tag = "esc_root"
    else:
        path = [MENU_ENTRY, CHARACTER, LIFE_SKILLS, leaf]
        from_tag = "hud"

    for node in path:
        try:
            from mm_assist.tool_repair import op_cancelled

            if op_cancelled():
                return {
                    "ok": False,
                    "reason": "cancelled",
                    "failed_at": node,
                    "steps": steps,
                    "from": from_tag,
                }
        except Exception:
            pass
        r = click_menu_node(
            node,
            cfg=cfg,
            win=win,
            input_method=input_method,
            refocus=refocus,
            settle_s=settle_s,
            threshold=threshold,
        )
        if not r.get("ok"):
            return {
                "ok": False,
                "reason": r.get("reason") or "not_found",
                "failed_at": node,
                "steps": steps,
                "from": from_tag,
            }
        steps.append(node)

    return {"ok": True, "steps": steps, "node": leaf, "from": from_tag}
