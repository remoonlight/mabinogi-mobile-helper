"""列表 OCR 选行：物品/材料/地点等可变目标点击。

Callers: tasks.gather（回放时实时选）、path_auto、cli list-ocr。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mm_assist.perception.ocr import OcrLine
from mm_assist.platform.window import WindowInfo

# 可变步骤默认候选（台服繁中 + 常见 OCR 误识）
DEFAULT_NEEDLES: dict[str, tuple[str, ...]] = {
    "04_item": (
        "鐵錠",
        "铁锭",
        "鐵锭",
        "鐵锭（",
        "鐵錠(",
        "鋼鐵錠",
        "钢锭",
        "鋼錠",
        "鋼锭",
        "合金鋼錠",
        "特殊鋼錠",
        "銀合金錠",
        "隕鐵錠",
        "焦油",
    ),
    "05_material": (
        "磺石",
        "礦石",
        "矿石",
        "鐵礦石",
        "铁矿石",
        "鐵礦",
        "煤炭",
        "煤",
    ),
    "05_ore": (
        "煤炭礦脈",
        "煤炭矿脉",
        "煤炭礦",
        "煤炭矿",
        "鐵礦脈",
        "铁矿脉",
        "鐵礦",
        "銅礦脈",
        "铜矿脉",
        "白銅礦脈",
        "白铜矿脉",
        "礦脈",
        "矿脉",
        "冰",
    ),
    "07_location": (
        "推薦",
        "推荐",
        "礦脈",
        "矿脉",
        "煤炭礦",
        "煤炭磺",
        "採集",
        "采集",
    ),
}

_MIN_NY: dict[str, float] = {
    "04_item": 0.12,
    "05_material": 0.55,
    "05_ore": 0.25,
    "07_location": 0.18,
}

OCR_SELECT_STEP_IDS = frozenset(DEFAULT_NEEDLES.keys())


@dataclass
class OcrPick:
    nx: float
    ny: float
    text: str
    score: float
    step_id: str


def normalize_text(s: str) -> str:
    return s.replace(" ", "").replace("　", "")


def box_center_norm(line: OcrLine, w: int, h: int) -> tuple[float, float] | None:
    if not line.box:
        return None
    xs = [float(p[0]) for p in line.box]
    ys = [float(p[1]) for p in line.box]
    return (sum(xs) / len(xs) / w, sum(ys) / len(ys) / h)


def score_line(text: str, needles: tuple[str, ...]) -> float:
    t = normalize_text(text)
    best = 0.0
    for n in needles:
        nn = normalize_text(n)
        if not nn or len(nn) < 2:
            continue
        if t == nn:
            best = max(best, 10.0 + len(nn))
        elif nn in t:
            best = max(best, 5.0 + len(nn) + len(nn) / max(len(t), 1))
        elif t in nn and len(t) >= max(3, int(len(nn) * 0.75)):
            best = max(best, 2.0 + len(t) / max(len(nn), 1))
    return best


def find_click_target(
    lines: list[OcrLine],
    needles: tuple[str, ...],
    win: WindowInfo,
    *,
    prefer_top_right: bool = False,
    avoid_left_strip: bool = False,
    min_ny: float = 0.0,
    prefer_right_panel: bool = False,
    prefer_iron_bar: bool = False,
) -> tuple[float, float, str, float] | None:
    """返回 (nx, ny, matched_text, score)。"""
    cands: list[tuple[float, float, float, str]] = []
    for L in lines:
        sc = score_line(L.text, needles)
        if sc <= 0:
            continue
        c = box_center_norm(L, win.width, win.height)
        if c is None:
            continue
        nx, ny = c
        if ny < min_ny:
            continue
        if prefer_top_right and (nx < 0.7 or ny > 0.2):
            sc *= 0.5
        if avoid_left_strip and nx < 0.18:
            sc *= 0.2
        if prefer_right_panel:
            if nx < 0.5:
                sc *= 0.25
            else:
                sc += 1.0
                if "薦" in L.text or "荐" in L.text:
                    sc += 3.0
                if "礦" in L.text or "矿" in L.text or "磺" in L.text:
                    sc += 2.0
        if prefer_iron_bar and any(k in L.text for k in ("鐵錠", "铁锭", "鐵锭")):
            sc += 2.5
            sc += max(0.0, 1.0 - nx)
        if "所需" in L.text:
            sc *= 0.1
        if len(normalize_text(L.text)) > 10:
            sc *= 0.15
        cands.append((sc + L.confidence * 0.1, nx, ny, L.text))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    sc, nx, ny, txt = cands[0]
    return (nx, ny, txt, sc)


def needles_for_step(
    step_id: str,
    *,
    item: str = "",
    material: str = "",
    location: str = "",
    ore: str = "",
) -> tuple[str, ...]:
    """配置主键 + 默认候选。主键放最前以提高命中。"""
    base = DEFAULT_NEEDLES.get(step_id, ())
    primary = ""
    if step_id.startswith("04"):
        primary = item.strip()
    elif step_id == "05_ore":
        primary = (ore or material).strip()
    elif step_id.startswith("05"):
        primary = material.strip()
    elif step_id.startswith("07"):
        primary = location.strip()
    if primary:
        extra = [primary]
        return tuple(dict.fromkeys([*extra, *base]))
    return base


def fallback_material(lines: list[OcrLine], win: WindowInfo) -> tuple[float, float, str, float] | None:
    mats = DEFAULT_NEEDLES["05_material"]
    hit = find_click_target(lines, mats, win, min_ny=0.55)
    if hit and len(normalize_text(hit[2])) <= 8:
        return hit
    for L in lines:
        if "所需材料" in L.text or "所需" in L.text:
            c = box_center_norm(L, win.width, win.height)
            if c:
                return (c[0], min(c[1] + 0.07, 0.85), "below_所需材料", 0.4)
    return (0.70, 0.78, "material_slot_fallback", 0.3)


def fallback_location(lines: list[OcrLine], win: WindowInfo) -> tuple[float, float, str, float] | None:
    hit = find_click_target(lines, ("地點", "地点", "區域", "区域", "採集", "采集"), win)
    if hit:
        return hit
    scored: list[tuple[float, float, str]] = []
    for L in lines:
        c = box_center_norm(L, win.width, win.height)
        if not c:
            continue
        nx, ny = c
        if 0.2 < nx < 0.9 and 0.3 < ny < 0.8 and len(normalize_text(L.text)) >= 2:
            if re.search(r"[A-Za-z0-9]", L.text) and len(L.text) < 3:
                continue
            scored.append((ny, nx, L.text))
    if not scored:
        return None
    scored.sort()
    ny, nx, txt = scored[0]
    return (nx, ny, txt, 0.4)


def pick_list_row(
    step_id: str,
    lines: list[OcrLine],
    win: WindowInfo,
    *,
    item: str = "",
    material: str = "",
    location: str = "",
    ore: str = "",
) -> OcrPick | None:
    """对 04/05/07/05_ore 做 OCR 选行；地点若点到「推薦」徽章则下移一行。"""
    needles = needles_for_step(
        step_id, item=item, material=material, location=location, ore=ore
    )
    if not needles:
        return None
    prefer_right = step_id.startswith("07") or step_id == "05_ore"
    hit = find_click_target(
        lines,
        needles,
        win,
        avoid_left_strip=step_id.startswith("04"),
        min_ny=_MIN_NY.get(step_id, 0.0),
        prefer_right_panel=prefer_right,
        prefer_iron_bar=step_id.startswith("04"),
    )
    if hit is None and step_id.startswith("05") and step_id != "05_ore":
        hit = fallback_material(lines, win)
    if hit is None and step_id.startswith("07"):
        hit = fallback_location(lines, win)
    if hit is None:
        return None
    nx, ny, txt, sc = hit
    if step_id.startswith("07") and ("薦" in txt or "荐" in txt):
        ny = min(ny + 0.04, 0.9)
        txt = f"{txt}+row"
    return OcrPick(nx=nx, ny=ny, text=txt, score=sc, step_id=step_id)
