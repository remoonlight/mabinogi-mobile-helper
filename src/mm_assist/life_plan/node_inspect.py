"""状态栏认页中文名 + 当前页可点位置（只定位，不点）。"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from mm_assist.config import ROOT

log = logging.getLogger(__name__)

_inspect_tpl_n: contextvars.ContextVar[int] = contextvars.ContextVar("inspect_tpl_n", default=0)
_inspect_ocr_n: contextvars.ContextVar[int] = contextvars.ContextVar("inspect_ocr_n", default=0)
_inspect_shot_memo: contextvars.ContextVar[dict[Any, Any] | None] = contextvars.ContextVar(
    "inspect_shot_memo", default=None
)
_inspect_stack_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "inspect_stack_key", default=""
)
_inspect_sn: contextvars.ContextVar[str] = contextvars.ContextVar("inspect_sn", default="")


def _reset_inspect_cost() -> None:
    _inspect_tpl_n.set(0)
    _inspect_ocr_n.set(0)
    _inspect_shot_memo.set({})


def _shot_memo() -> dict[Any, Any]:
    d = _inspect_shot_memo.get()
    if d is None:
        d = {}
        _inspect_shot_memo.set(d)
    return d


def _memo_get(key: Any, frame: Any) -> tuple[bool, Any]:
    rec = _shot_memo().get(key)
    if rec is None:
        return False, None
    stored, value = rec
    if stored is not frame:
        return False, None
    return True, value


def _memo_put(key: Any, frame: Any, value: Any) -> None:
    _shot_memo()[key] = (frame, value)


def _bump_tpl() -> None:
    _inspect_tpl_n.set(_inspect_tpl_n.get() + 1)


def _bump_ocr() -> None:
    _inspect_ocr_n.set(_inspect_ocr_n.get() + 1)


def _inspect_cost(t0: float) -> dict[str, Any]:
    return {
        "tpl_n": int(_inspect_tpl_n.get()),
        "ocr_n": int(_inspect_ocr_n.get()),
        "ms": (time.perf_counter() - t0) * 1000.0,
    }


@lru_cache(maxsize=512)
def _template_exists(path_str: str) -> bool:
    return Path(path_str).is_file()


NODE_ZH: dict[str, str] = {
    "main_hud": "主界面",
    "system_menu": "菜单",
    "profile": "角色",
    "profile:info": "角色-我的资讯",
    "profile:combat": "角色-战斗技能",
    "profile:career": "角色-职业",
    "life_grid": "角色-生活技能",
    "gather_skill": "角色-生活技能-技能列表",
    "gather_detail": "角色-生活技能-技能详情",
    "craft_shell": "制作",
    "craft_items": "制作-道具",
    "craft_item_recipe": "制作-道具-配方",
    "craft_quick": "制作-快速制作",
    "craft_hub": "制作-加工",
    "craft_make": "制作-制作",
    "craft_alchemy": "制作-炼金术",
    "facility": "制作-加工-设施",
    "facility_recipe": "制作-加工-加工详情",
    "facility_detail": "制作-加工-加工详情",
    "inventory": "背包",
        "map_dunbarton": "地图-杜巴顿",
    "map_continent": "地图-欧拉大陆",
    "map_kuhan": "地图-库汉",
    "map": "地图",
    "currency_wallet": "持有的货币",
    "dialog": "对话",
    "dialog_skip": "对话-跳过对话",
    "item_info": "物品详情",
    "obtain_method": "获得方法",
    "overlay": "离开主界面",
}

_SKIP_ZH = frozenset({"map_dunbarton_nerys", "unknown"})
_UNKNOWN_ZH = "未知界面"

# 生活技能大格：九门采集（与 knowledge/templates/life_grid 对齐）
LIFE_GRID_GATHER: tuple[tuple[str, str], ...] = (
    ("日常採集", "daily"),
    ("伐木", "woodcut"),
    ("採礦", "mine"),
    ("採集藥草", "herb"),
    ("剪羊毛", "shear"),
    ("收割", "harvest"),
    ("鋤地", "hoe"),
    ("昆蟲採集", "insect"),
    ("釣魚", "fish"),
)

_CRAFT_SHELL_TABS: tuple[str, ...] = ("快速制作", "加工", "制作", "炼金术")

# 当前页 10h 会点的位置（不含禁点：头像、主界面製作/背包）
SPOTS: dict[str, tuple[str, ...]] = {
    "main_hud": ("菜单",),
    "system_menu": ("角色", "製作", "背包", "右上 X"),
    "profile": ("生活技能 Tab", "关闭/X"),
    "profile:info": ("关闭/X",),
    "profile:combat": ("关闭/X",),
    "profile:career": ("关闭/X",),
    "life_grid": tuple(n for n, _sid in LIFE_GRID_GATHER) + ("关闭/X",),
    "gather_skill": ("巢穴", "蜘蛛網", "水", "水井"),
    "gather_detail": ("淨淨蘑菇", "水井", "寻找附近", "关闭详情"),
    "craft_quick": _CRAFT_SHELL_TABS + ("右上 X",),
    "craft_hub": _CRAFT_SHELL_TABS
    + (
        "金屬加工",
        "木材加工",
        "皮革加工",
        "布料加工",
        "藥品加工",
        "食材加工",
        "右上 X",
    ),
    "craft_make": _CRAFT_SHELL_TABS + ("右上 X",),
    "craft_alchemy": _CRAFT_SHELL_TABS + ("右上 X",),
    "facility": ("左上 ←", "全部領取"),
    "facility_recipe": ("前往加工", "加工"),
    "facility_detail": ("获得方法",),
    "inventory": ("樹液", "煤炭", "关闭背包"),
    "map_dunbarton": ("右上 X",),
    "map_continent": ("右上 X",),
    "map_kuhan": ("右上 X",),
    "map": ("右上 X",),
    "currency_wallet": ("右上 X",),
    "craft_shell": ("右上 X",),
    "craft_items": ("返回",),
    "craft_item_recipe": ("返回",),
    "dialog": (),
    "dialog_skip": (),
    "item_info": ("关闭背包",),
    "obtain_method": ("关闭背包",),
    "overlay": (),
}


def _gather_skill_sid(node: str) -> str:
    key = str(node or "").strip()
    if key.startswith("gather_skill:"):
        return key.split(":", 1)[1].strip()
    return ""


def _facility_fid(node: str) -> str:
    fid, _rec = _facility_parts(node)
    return fid


FACILITY_ZH: dict[str, str] = {
    "metal": "金属加工",
    "wood": "木材加工",
    "leather": "皮革加工",
    "cloth": "布料加工",
    "medicine": "药品加工",
    "food": "食材加工",
}

_INSPECT_FACILITY_EXTRA: dict[str, tuple[str, ...]] = {
    "metal": ("鐵錠", "鐵錠(礦石)", "特殊鋼錠"),
}

_RECIPE_DISP = str.maketrans("鐵錠礦藥級淨樹", "铁锭矿药级净树")


def _facility_parts(node: str) -> tuple[str, str]:
    key = str(node or "").strip()
    if not key.startswith("facility:"):
        return ("", "")
    rest = key.split(":", 1)[1].strip()
    if not rest:
        return ("", "")
    if ":" in rest:
        fid, rec = rest.split(":", 1)
        return (fid.strip(), rec.strip())
    return (rest, "")


def _recipe_zh(label: str) -> str:
    from mm_assist.perception.item_sense import fold_zh

    return fold_zh(label).translate(_RECIPE_DISP)


def _facility_item_names(fid: str) -> tuple[str, ...]:
    names: list[str] = []
    try:
        from mm_assist.perception.process_sense import recipes_for_facility

        for row in recipes_for_facility(fid):
            lab = str(row.get("label") or row.get("name") or "").strip()
            if lab and lab not in names:
                names.append(lab)
    except Exception:
        pass
    for extra in _INSPECT_FACILITY_EXTRA.get(fid, ()):
        if extra not in names:
            names.append(extra)
    return tuple(names)


def _recipe_input_names(label: str) -> tuple[str, ...]:
    from mm_assist.life_plan.recipes import RECIPES
    from mm_assist.perception.item_sense import fold_zh

    want = fold_zh(label)
    if not want:
        return ()
    for name, rec in RECIPES.items():
        if fold_zh(name) == want:
            return tuple(rec.inputs.keys())
    return ()


_ITEM_INFO_ROI = (0.26, 0.22, 0.48, 0.58)
_ITEM_PANEL_ROI = (0.28, 0.24, 0.28, 0.50)
_ITEM_INFO_HDR_ROI = (0.18, 0.20, 0.40, 0.20)
_ITEM_INFO_FOOT_ROI = (0.16, 0.50, 0.42, 0.22)
# 技能详情弹窗标题带（尋找附近钮上方；避开右栏物品列表）
_GATHER_DETAIL_HDR_ROI = (0.20, 0.10, 0.56, 0.28)
_BAG_CARD_ROI = (0.16, 0.18, 0.42, 0.60)
_OBTAIN_LIST_ROI = (0.50, 0.18, 0.46, 0.62)
_OBTAIN_HDR_ROI = (0.50, 0.08, 0.44, 0.16)
_OBTAIN_VEIN_NX, _OBTAIN_VEIN_NY = 0.64, 0.26
_INV_GRID_ROI = (0.55, 0.10, 0.44, 0.78)
_BAG_MATS = frozenset({"樹液", "树液", "煤炭"})


def _ocr_lines_roi(
    frame: Any,
    roi: tuple[float, float, float, float],
    *,
    fallback: bool = False,
) -> list[Any]:
    from mm_assist.perception.ocr import ocr_roi

    _ = fallback  # 兼容调用点；认页门闩只 cht 一趟，空结果不上第二引擎
    _bump_ocr()
    try:
        lines = ocr_roi(frame, roi, min_confidence=0.28, backend="cht")
    except Exception:
        return []
    return list(lines or [])


def _ocr_blob(
    frame: Any,
    roi: tuple[float, float, float, float],
    *,
    fallback: bool = False,
) -> str:
    from mm_assist.perception.item_sense import compact_zh, fold_zh

    blob = compact_zh(
        "".join(
            str(getattr(ln, "text", "") or "")
            for ln in _ocr_lines_roi(frame, roi, fallback=fallback)
        )
    )
    return fold_zh(blob)


def _process_btn_hit(frame: Any) -> bool:
    """只模板，不走 locate_process_button（其装饰会 OCR 整块所需材料）。"""
    from mm_assist.platform.match import match_template

    try:
        from mm_assist.perception.process_sense import (
            PROCESS_BUTTON_ROI,
            TEMPLATE_THRESHOLD,
            action_rows,
            process_btn_path,
        )
    except Exception:
        return False
    if frame is None:
        return False
    paths: list[Path] = []
    try:
        for row in action_rows("process_btn"):
            p = process_btn_path(variant_id=str(row.get("id") or ""))
            if p.is_file() and p not in paths:
                paths.append(p)
        legacy = ROOT / "knowledge" / "templates" / "process" / "process_btn.png"
        if legacy.is_file() and legacy not in paths:
            paths.append(legacy)
    except Exception:
        return False
    for path in paths:
        try:
            hit = match_template(
                frame,
                path,
                roi=PROCESS_BUTTON_ROI,
                threshold=TEMPLATE_THRESHOLD,
                min_margin=0.0,
                check_margin=False,
            )
        except Exception:
            hit = None
        if hit is not None:
            return True
    return False


def _named_line_xy(
    frame: Any,
    roi: tuple[float, float, float, float],
    name: str,
    *,
    contains: bool = False,
    fallback: bool = True,
) -> tuple[int, int] | None:
    from mm_assist.perception.item_sense import fold_zh, line_center, line_matches_name

    want = str(name or "").strip()
    if frame is None or not want:
        return None
    needles = {fold_zh(want)}
    if want in ("獲得方法", "获得方法"):
        needles.update({fold_zh("獲得方法"), fold_zh("获得方法")})
    if want in ("附近位置", "尋找附近位置", "尋找附近", "寻找附近"):
        needles.update(
            {
                fold_zh("附近位置"),
                fold_zh("尋找附近位置"),
                fold_zh("尋找附近"),
                fold_zh("寻找附近"),
            }
        )
    if want in ("煤炭礦脈", "煤炭矿脉", "礦脈"):
        needles.update({fold_zh("煤炭礦脈"), fold_zh("煤炭矿脉"), fold_zh("礦脈")})
    if want in ("推薦", "推荐"):
        needles.update({fold_zh("推薦"), fold_zh("推荐")})
    hits: list[tuple[int, int]] = []
    for ln in _ocr_lines_roi(frame, roi, fallback=fallback):
        text = str(getattr(ln, "text", "") or "")
        blob = fold_zh(text)
        if contains:
            ok = any(n and n in blob for n in needles)
        else:
            ok = line_matches_name(text, want, catalog=(want,))
        if not ok:
            continue
        xy = line_center(ln)
        if xy is None:
            continue
        hits.append((int(xy[0]), int(xy[1])))
    if not hits:
        return None
    hits.sort(key=lambda t: t[1])
    return hits[0]


def _is_coal_vein_spot(spot: str) -> bool:
    from mm_assist.perception.item_sense import fold_zh

    blob = fold_zh(spot)
    return bool(blob) and fold_zh("礦脈") in blob


def _is_bag_mat_spot(spot: str) -> bool:
    from mm_assist.perception.item_sense import fold_zh

    blob = fold_zh(spot)
    return blob in {fold_zh("樹液"), fold_zh("煤炭")}


def _bag_item_xy(frame: Any, name: str) -> tuple[int, int] | None:
    from mm_assist.perception.item_sense import fold_zh

    want = "樹液" if fold_zh(name) == fold_zh("樹液") else "煤炭"
    return _named_line_xy(frame, _INV_GRID_ROI, want, contains=True, fallback=True)


def _inventory_overlay_node(frame: Any) -> str:
    """背包上物品卡 / 获得方法列表。"""
    if frame is None:
        return ""
    om = _detect_obtain_method(frame)
    if om:
        return om
    info = _detect_item_info(frame)
    if info:
        return info
    return _detect_bag_item_card(frame)


def _frac_xy(frame: Any, nx: float, ny: float) -> tuple[int, int] | None:
    try:
        h, w = int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        return None
    return (int(w * nx), int(h * ny))


def _obtain_vein_xy(frame: Any) -> tuple[int, int] | None:
    """获得方法列表第一行煤炭矿脉：OCR 礦脈取最上，否则推荐徽下，再否则列表首行。"""
    xy = _named_line_xy(
        frame, _OBTAIN_LIST_ROI, "煤炭礦脈", contains=True, fallback=False
    )
    if xy is not None:
        return xy
    rec = _named_line_xy(frame, _OBTAIN_LIST_ROI, "推薦", contains=True, fallback=False)
    if rec is not None:
        try:
            h, w = int(frame.shape[0]), int(frame.shape[1])
            return (int(w * _OBTAIN_VEIN_NX), rec[1] + max(36, int(h * 0.055)))
        except Exception:
            return (rec[0], rec[1] + 40)
    return _frac_xy(frame, _OBTAIN_VEIN_NX, _OBTAIN_VEIN_NY)


def _material_catalog() -> tuple[str, ...]:
    from mm_assist.life_plan.recipes import RECIPES
    from mm_assist.perception.item_sense import GATHER_SKILL_IDS, items_for_skill

    names: list[str] = ["煤炭", "樹液"]
    for rec in RECIPES.values():
        names.append(rec.output)
        names.extend(rec.inputs.keys())
    for sid in GATHER_SKILL_IDS:
        try:
            names.extend(items_for_skill(sid))
        except Exception:
            continue
    return tuple(dict.fromkeys(names))


def _pick_material_from_blob(blob: str) -> str:
    from mm_assist.perception.item_sense import fold_zh, line_matches_name

    blob_f = fold_zh(blob)
    if not blob_f:
        return ""
    catalog = _material_catalog()
    hits = [n for n in catalog if line_matches_name(blob, n, catalog)]
    if not hits:
        return ""
    return max(hits, key=lambda n: len(fold_zh(n)))


def _obtain_spots_for_mat(mat: str) -> tuple[str, ...]:
    from mm_assist.perception.item_sense import fold_zh

    blob = fold_zh(mat)
    if fold_zh("煤炭") in blob:
        return ("煤炭礦脈",)
    if fold_zh("樹液") in blob:
        return ("伐木",)
    return ("伐木", "煤炭礦脈")


def _is_woodcut_spot(spot: str) -> bool:
    from mm_assist.perception.item_sense import fold_zh

    blob = fold_zh(spot)
    return blob == fold_zh("伐木")


def _obtain_first_named_xy(
    frame: Any,
    name: str,
    *,
    exclude: tuple[str, ...] = (),
) -> tuple[int, int] | None:
    from mm_assist.perception.item_sense import fold_zh, line_center

    needle = fold_zh(name)
    banned = tuple(fold_zh(x) for x in exclude if x)
    hits: list[tuple[int, int]] = []
    for ln in _ocr_lines_roi(frame, _OBTAIN_LIST_ROI, fallback=False):
        text = str(getattr(ln, "text", "") or "")
        blob = fold_zh(text)
        if needle and needle not in blob:
            continue
        if any(b and b in blob for b in banned):
            continue
        xy = line_center(ln)
        if xy is None:
            continue
        hits.append((int(xy[0]), int(xy[1])))
    if hits:
        hits.sort(key=lambda t: t[1])
        return hits[0]
    rec = _named_line_xy(frame, _OBTAIN_LIST_ROI, "推薦", contains=True, fallback=False)
    if rec is not None:
        try:
            h, w = int(frame.shape[0]), int(frame.shape[1])
            return (int(w * _OBTAIN_VEIN_NX), rec[1] + max(36, int(h * 0.055)))
        except Exception:
            return (rec[0], rec[1] + 40)
    return _frac_xy(frame, _OBTAIN_VEIN_NX, _OBTAIN_VEIN_NY)


def _detect_obtain_method(frame: Any) -> str:
    if _process_btn_hit(frame):
        return ""
    blob = _ocr_blob(frame, _OBTAIN_HDR_ROI)
    header = "請選擇獲得方法" in blob or "请选择获得方法" in blob or "選擇獲得方法" in blob
    if not (header or (("獲得方法" in blob or "获得方法" in blob) and "請選擇" in blob)):
        return ""
    mat = _pick_material_from_blob(_ocr_blob(frame, _ITEM_INFO_HDR_ROI) + blob)
    return f"obtain_method:{mat}" if mat else "obtain_method"


def _detect_item_info(frame: Any, *, require_obtain: bool = True) -> str:
    """物品卡：标题物名 + 获得方法。无闭集命中则粗 item_info，不用「材料」。"""
    hdr = _ocr_blob(frame, _ITEM_INFO_HDR_ROI)
    foot = _ocr_blob(frame, _ITEM_INFO_FOOT_ROI)
    blob = hdr + foot
    if "請選擇獲得方法" in blob or "请选择获得方法" in blob or "選擇獲得方法" in blob:
        return ""
    has_obtain = "獲得方法" in blob or "获得方法" in blob
    if require_obtain and not has_obtain:
        return ""
    mat = _pick_material_from_blob(hdr)
    if not require_obtain:
        if mat not in ("煤炭", "樹液"):
            return ""
        if "材料" not in blob and not has_obtain:
            return ""
        return f"item_info:{mat}"
    return f"item_info:{mat}" if mat else "item_info"


def _detect_bag_item_card(frame: Any) -> str:
    blob = _ocr_blob(frame, _BAG_CARD_ROI)
    if "請選擇獲得方法" in blob or "请选择获得方法" in blob or "選擇獲得方法" in blob:
        return ""
    mat = _pick_material_from_blob(blob)
    if mat not in ("煤炭", "樹液"):
        return ""
    if "材料" not in blob and "獲得方法" not in blob and "获得方法" not in blob:
        return ""
    return f"item_info:{mat}"


def _stack_labels() -> dict[str, str]:
    from mm_assist.memory.readers.ui_stack import _cfg

    raw = _cfg().get("labels") or {}
    return {str(k): str(v) for k, v in raw.items() if str(v).strip()}


def format_node_zh(node: str, *, stp: int | None = None) -> str:
    key = str(node or "").strip()
    if not key:
        return "—"
    if key.startswith("stack:"):
        cls = key.split(":", 1)[1].strip()
        return _stack_labels().get(cls) or _UNKNOWN_ZH
    if key.startswith("fine:"):
        inner = key.split(":", 1)[1].strip()
        if inner == "craft_hub":
            return "—"
        if inner == "facility":
            return NODE_ZH["facility"]
        if inner == "facility_recipe":
            return NODE_ZH["facility_recipe"]
        if inner == "facility_detail":
            return NODE_ZH["facility_detail"]
        if inner == "life_grid":
            return NODE_ZH["life_grid"]
        return format_node_zh(inner)
    sid = _gather_skill_sid(key)
    if sid:
        label = {kid: lab for lab, kid in LIFE_GRID_GATHER}.get(sid, "")
        return f"角色-生活技能-{label}" if label else "角色-生活技能-技能列表"
    fid = _facility_fid(key)
    if fid:
        name = FACILITY_ZH.get(fid, "")
        base = f"制作-加工-{name}" if name else "制作-加工-设施"
        rec = _facility_parts(key)[1]
        if rec:
            return f"{base}-{_recipe_zh(rec)}"
        return base
    if key.startswith("item_info:"):
        tail = key.split(":", 1)[1].strip()
        return f"物品详情-{_recipe_zh(tail)}" if tail else "物品详情"
    if key.startswith("obtain_method:"):
        tail = key.split(":", 1)[1].strip()
        return f"获得方法-{_recipe_zh(tail)}" if tail else "获得方法"
    if key.startswith("gather_detail:"):
        tail = key.split(":", 1)[1].strip()
        base = NODE_ZH["gather_detail"]
        return f"{base}-{_recipe_zh(tail)}" if tail else base
    if key.startswith("map:"):
        tail = key.split(":", 1)[1].strip()
        return f"地图-{tail}" if tail else "地图"
    if key.startswith("profile:"):
        return NODE_ZH.get(key, NODE_ZH["profile"])
    if key in _SKIP_ZH or key not in NODE_ZH:
        return _UNKNOWN_ZH
    if key == "main_hud":
        if stp == 1:
            return "主界面-动作中"
        if stp == 0:
            return "主界面-待机"
        return NODE_ZH[key]
    return NODE_ZH[key]


def inspect_page_zh(node: str, *, stack_ready: bool, stp: int | None = None) -> str:
    """先栈、再 stp、再中文。栈未就绪不给页名。製作壳粗名不上栏。"""
    if not stack_ready:
        return "—"
    key = str(node or "").strip()
    if not key or key in ("craft_shell", "fine:craft_hub"):
        return "—"
    return format_node_zh(key, stp=stp if key == "main_hud" else None)


HUD_HOLD_SAMPLES = 2
CRAFT_WEAK_SAMPLES = 3


def is_main_hud_node(node: str) -> bool:
    return str(node or "").strip() == "main_hud"


def _profile_shell_fine(node: str) -> bool:
    n = str(node or "").strip()
    if n == "life_grid" or n.startswith("profile:"):
        return True
    return (
        n == "gather_detail"
        or n.startswith("gather_detail:")
        or n == "gather_skill"
        or n.startswith("gather_skill:")
    )


def _life_grid_fine(node: str) -> bool:
    n = str(node or "").strip()
    return (
        n == "gather_detail"
        or n.startswith("gather_detail:")
        or n.startswith("gather_skill:")
    )


def _craft_shell_fine(node: str) -> bool:
    return str(node or "").strip() in _CRAFT_SHELL_NODES


def _item_info_fine(node: str) -> bool:
    n = str(node or "").strip()
    return n.startswith("item_info:") and bool(n.split(":", 1)[1].strip())


def _item_info_sap_or_coal(node: str) -> bool:
    from mm_assist.perception.item_sense import fold_zh

    n = str(node or "").strip()
    if not n.startswith("item_info:"):
        return False
    tail = fold_zh(n.split(":", 1)[1].strip())
    return bool(tail) and tail in {fold_zh("樹液"), fold_zh("煤炭")}


def _obtain_method_fine(node: str) -> bool:
    n = str(node or "").strip()
    return n.startswith("obtain_method:") and bool(n.split(":", 1)[1].strip())


def _gather_detail_fine(node: str) -> bool:
    n = str(node or "").strip()
    return n.startswith("gather_detail:") and bool(n.split(":", 1)[1].strip())


def _detail_mat(node: str) -> str:
    n = str(node or "").strip()
    for prefix in ("item_info:", "obtain_method:"):
        if n.startswith(prefix):
            return n.split(":", 1)[1].strip()
    return ""


def _craft_shell_coarse(node: str) -> bool:
    return str(node or "").strip() in ("craft_shell", "fine:craft_hub")


def _facility_page(node: str) -> bool:
    n = str(node or "").strip()
    return n in ("facility", "facility_recipe") or n.startswith("facility:")


def _keep_inspect_stack(prev: dict[str, Any], cur_d: dict[str, Any]) -> dict[str, Any]:
    kept = dict(prev or {})
    kept["stack_names"] = list(cur_d.get("stack_names") or kept.get("stack_names") or [])
    kept["stack_ready"] = bool(cur_d.get("stack_ready"))
    return kept


def hold_inspect_payload(
    prev: dict[str, Any] | None,
    cur: dict[str, Any],
    *,
    hud_streak: int = 0,
) -> tuple[dict[str, Any], int]:
    """换页未确认前冻住上一拍「界面：」。离开层→主界面须连续两拍。"""
    from mm_assist.memory.readers.ui_stack import craft_node_from_stack, stack_node

    cur_d = dict(cur or {})
    if any((prev or {}).get(key) for key in ("profile_mem", "map_mem", "craft_mem")):
        # A slower image job started before a newer memory tick must not overwrite it.
        if float(cur_d.get("stack_sampled_ts") or 0) < float(prev.get("stack_sampled_ts") or 0):
            return dict(prev), int(hud_streak or 0)
    prev_n = str((prev or {}).get("node") or "").strip()
    cur_n = str(cur_d.get("node") or "").strip()
    ready = bool(cur_d.get("stack_ready"))
    if cur_d.get("skew"):
        return dict(prev or {}), int(hud_streak or 0)
    if ready and not cur_n:
        filled = stack_node(
            [str(x) for x in (cur_d.get("stack_names") or [])],
            ready=True,
        )
        if filled:
            cur_d["node"] = filled
            cur_n = filled
    if not prev_n:
        return cur_d, (1 if is_main_hud_node(cur_n) and ready else 0)
    if not ready:
        return dict(prev or {}), int(hud_streak or 0)
    if not cur_n:
        return dict(prev or {}), 0
    if cur_n == prev_n:
        return cur_d, (hud_streak + 1 if is_main_hud_node(cur_n) else 0)
    # 无新画面时不要把生活技能刷成粗「角色」；已经看过画面则允许离开该 Tab。
    if (
        cur_n == "profile"
        and _profile_shell_fine(prev_n)
        and not cur_d.get("profile_looked")
    ):
        kept = dict(prev or {})
        kept["stack_names"] = list(cur_d.get("stack_names") or kept.get("stack_names") or [])
        kept["stack_ready"] = ready
        return kept, 0
    # 技能列表/详情栈仍是人物：裸「角色」不算已离开，即使看过这一拍。
    if cur_n == "profile" and _life_grid_fine(prev_n):
        kept = dict(prev or {})
        kept["stack_names"] = list(cur_d.get("stack_names") or kept.get("stack_names") or [])
        kept["stack_ready"] = ready
        return kept, 0
    # 无新画面时不要把技能详情/技能列表刷回粗大格；看过画面后允许离开详情。
    if (
        cur_n == "life_grid"
        and _life_grid_fine(prev_n)
        and not cur_d.get("life_looked")
    ):
        kept = _keep_inspect_stack(dict(prev or {}), cur_d)
        return kept, 0
    if cur_n == "item_info" and _item_info_fine(prev_n):
        return _keep_inspect_stack(dict(prev or {}), cur_d), 0
    if cur_n == "obtain_method" and _obtain_method_fine(prev_n):
        return _keep_inspect_stack(dict(prev or {}), cur_d), 0
    if cur_n == "gather_detail" and _gather_detail_fine(prev_n):
        return _keep_inspect_stack(dict(prev or {}), cur_d), 0
    # 获得方法页标题常无物名：从上一拍物品详情带过来，退回详情同样带回。
    mat = _detail_mat(prev_n)
    if cur_n == "obtain_method" and _item_info_fine(prev_n) and mat:
        out = dict(cur_d)
        out["node"] = f"obtain_method:{mat}"
        return out, 0
    if cur_n == "item_info" and _obtain_method_fine(prev_n) and mat:
        out = dict(cur_d)
        out["node"] = f"item_info:{mat}"
        return out, 0
    if _craft_shell_coarse(cur_n) and _craft_shell_fine(prev_n):
        kept = _keep_inspect_stack(dict(prev or {}), cur_d)
        return kept, 0
    if _craft_shell_fine(cur_n) and _craft_shell_fine(prev_n) and cur_n != prev_n:
        how = str(cur_d.get("craft_how") or "")
        if how == "tpl" or cur_d.get("craft_mem"):
            cur_d.pop("craft_weak", None)
            return cur_d, 0
        prev_weak = (prev or {}).get("craft_weak")
        same = (
            isinstance(prev_weak, (list, tuple))
            and len(prev_weak) >= 2
            and str(prev_weak[0]) == cur_n
        )
        n = int(prev_weak[1]) + 1 if same else 1
        if n >= CRAFT_WEAK_SAMPLES:
            cur_d.pop("craft_weak", None)
            return cur_d, 0
        kept = _keep_inspect_stack(dict(prev or {}), cur_d)
        kept["craft_weak"] = (cur_n, n)
        return kept, 0
    # 设施细页不要被制作粗页顶掉；栈已离开设施层则放行，否则六卡可见才回 craft_hub。
    if _facility_page(prev_n) and cur_n in ("craft_hub", "craft_shell", "fine:craft_hub"):
        if ready:
            layer = craft_node_from_stack(
                [str(x) for x in (cur_d.get("stack_names") or [])]
            )
            if layer not in ("fine:facility", "fine:facility_recipe"):
                return cur_d, 0
        if not cur_d.get("hub_cards"):
            return _keep_inspect_stack(dict(prev or {}), cur_d), 0
    if is_main_hud_node(cur_n) and not list(cur_d.get("stack_names") or []):
        if not cur_d.get("hud_menu_icon"):
            try:
                stp_i = int(cur_d["stp"]) if cur_d.get("stp") is not None else None
            except (TypeError, ValueError):
                stp_i = None
            if stp_i not in (0, 1) and not cur_d.get("empty_ok"):
                frozen = dict(prev or {})
                frozen["held"] = True
                return frozen, int(hud_streak or 0)
    if is_main_hud_node(cur_n) and not is_main_hud_node(prev_n):
        streak = int(hud_streak) + 1
        if streak < HUD_HOLD_SAMPLES:
            frozen = dict(prev or {})
            frozen["held"] = True
            return frozen, streak
        return cur_d, 0
    return cur_d, 0


def spots_for_node(node: str) -> tuple[str, ...]:
    key = str(node or "").strip()
    sid = _gather_skill_sid(key)
    if sid:
        try:
            from mm_assist.perception.item_sense import items_for_skill

            names = list(items_for_skill(sid))
        except Exception:
            names = []
        if sid == "daily" and "水" not in names:
            if "水井" in names:
                names.insert(names.index("水井"), "水")
            else:
                names.append("水")
        else:
            for extra in _INSPECT_EXTRA_ITEMS.get(sid, ()):
                if extra not in names:
                    names.append(extra)
        return tuple(names) if names else SPOTS.get("gather_skill", ())
    fid = _facility_fid(key)
    if fid or key == "facility":
        items = list(_facility_item_names(fid)) if fid else []
        nav = list(SPOTS.get("facility", ()))
        if _facility_parts(key)[1]:
            nav = nav + list(SPOTS.get("facility_recipe", ()))
            for mat in _recipe_input_names(_facility_parts(key)[1]):
                if mat not in items:
                    items.append(mat)
        return tuple(items) + tuple(nav)
    if key == "gather_detail" or key.startswith("gather_detail:"):
        return SPOTS.get("gather_detail", ())
    if key == "item_info" or key.startswith("item_info:"):
        return ("获得方法", "附近位置")
    if key.startswith("obtain_method:"):
        return _obtain_spots_for_mat(key.split(":", 1)[1])
    if key == "fine:craft_hub":
        return _CRAFT_SHELL_TABS + ("右上 X",)
    if key == "fine:facility_detail":
        return SPOTS.get("facility_detail", ("获得方法",))
    if key.startswith("map"):
        return SPOTS.get("map_dunbarton", ("右上 X",))
    if key.startswith("profile:"):
        return SPOTS.get(key, ("关闭/X",))
    if key in _SKIP_ZH or key not in NODE_ZH:
        return ()
    return SPOTS.get(key, ())


# 游戏右栏有、tiers 尚未配模板的项（只进状态栏下拉）
_INSPECT_EXTRA_ITEMS: dict[str, tuple[str, ...]] = {
    "daily": ("水",),
}


_FAM_NODE = {
    "main_hud": "main_hud",
    "combat": "main_hud",
    "inventory": "inventory",
    "map": "map_dunbarton",
    "craft_hub": "craft_hub",
    "prf": "profile",
}


def mem_inspect_node(view: Any = None) -> str:
    """仅内存界面参数。menu=0 主界面；menu 族就是菜单。不扫画面。"""
    from mm_assist.life_execute.ui_logic import node_from_mem
    from mm_assist.perception.state_ui import read_state_ui

    v = view if view is not None else read_state_ui()
    if v is None:
        return ""
    node, _tab = node_from_mem(v)
    if node in NODE_ZH:
        return node
    return ""


def _tmpl(
    frame: Any,
    rel: str,
    *,
    threshold: float = 0.70,
    roi: tuple[float, float, float, float] | None = None,
) -> bool:
    from mm_assist.platform.match import match_template

    key = ("tmpl", id(frame), rel, roi, threshold)
    hit_ok, cached = _memo_get(key, frame)
    if hit_ok:
        return bool(cached)
    path = ROOT / rel
    if frame is None or not _template_exists(str(path)):
        _memo_put(key, frame, False)
        return False
    try:
        _bump_tpl()
        hit = match_template(frame, path, threshold=threshold, roi=roi)
        ok = hit is not None
        if ok and rel.replace("\\", "/").startswith("knowledge/templates/life_grid/"):
            # Normalized correlation also matches the darkened grid behind a
            # skill popup. Its white icons are ~71 gray, versus 255 when active.
            # Only the lit foreground grid may prove arrival at life_grid.
            import cv2
            import numpy as np
            patch = frame[hit.y:hit.y + hit.h, hit.x:hit.x + hit.w]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            ok = bool(gray.size and np.percentile(gray, 95) >= 150)
    except Exception:
        ok = False
    _memo_put(key, frame, ok)
    return ok


# locate_spot(system_menu) 已有框，禁止另造。ROI 按 2026-09-10 采框互不重叠。
_MENU_ICON_DESIGN = (1920, 1440)
_MENU_ICON_ROI: dict[str, tuple[float, float, float, float]] = {
    "knowledge/templates/menu/icon_character.png": (0.66, 0.12, 0.08, 0.10),
    "knowledge/templates/menu/icon_crafting.png": (0.79, 0.12, 0.08, 0.10),
    "knowledge/templates/menu/icon_inventory.png": (0.73, 0.12, 0.07, 0.10),
}


def _menu_sidecar_box(rel: str) -> tuple[int, int, int, int] | None:
    path = (ROOT / rel).with_suffix(".json")
    if not path.is_file():
        return None
    try:
        box = json.loads(path.read_text(encoding="utf-8")).get("client_px") or {}
        x = int(box.get("x") or 0)
        y = int(box.get("y") or 0)
        bw = int(box.get("w") or 0)
        bh = int(box.get("h") or 0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if bw <= 0 or bh <= 0:
        return None
    return (x, y, bw, bh)


def _scale_menu_box(
    box: tuple[int, int, int, int],
    frame: Any,
) -> tuple[int, int, int, int] | None:
    try:
        h, w = int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    dw, dh = float(_MENU_ICON_DESIGN[0]), float(_MENU_ICON_DESIGN[1])
    x, y, bw, bh = box
    sx, sy = w / dw, h / dh
    return (
        int(x * sx),
        int(y * sy),
        max(1, int(bw * sx)),
        max(1, int(bh * sy)),
    )


def _menu_icon_xy(
    frame: Any,
    rel: str,
    roi: tuple[float, float, float, float] | None,
) -> tuple[int, int] | None:
    """优先采框中心，避免四键模板互配导致红圈乱跳。对图命中须落在框内才用。"""
    from mm_assist.platform.match import match_template

    path = ROOT / rel
    if frame is None or not path.is_file():
        return None
    scaled = None
    raw = _menu_sidecar_box(rel)
    if raw is not None:
        scaled = _scale_menu_box(raw, frame)
    search_roi = roi
    if scaled is not None:
        x, y, bw, bh = scaled
        try:
            fh, fw = int(frame.shape[0]), int(frame.shape[1])
        except Exception:
            fh, fw = 0, 0
        if fw > 0 and fh > 0:
            pad_x, pad_y = max(8, bw // 4), max(8, bh // 4)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(fw, x + bw + pad_x)
            y1 = min(fh, y + bh + pad_y)
            if x1 > x0 and y1 > y0:
                search_roi = (x0 / fw, y0 / fh, (x1 - x0) / fw, (y1 - y0) / fh)
    try:
        hit = match_template(
            frame,
            path,
            roi=search_roi,
            threshold=0.55,
            min_margin=0.0,
            check_margin=False,
        )
    except Exception:
        hit = None
    if hit is not None and scaled is not None:
        x, y, bw, bh = scaled
        pad_x, pad_y = max(12, bw // 2), max(12, bh // 2)
        if (x - pad_x) <= int(hit.cx) <= (x + bw + pad_x) and (
            y - pad_y
        ) <= int(hit.cy) <= (y + bh + pad_y):
            return (int(hit.cx), int(hit.cy))
        return None
    if hit is not None:
        return (int(hit.cx), int(hit.cy))
    return None


def _system_menu_visible(frame: Any) -> bool:
    hits = sum(
        1
        for rel, roi in _MENU_ICON_ROI.items()
        if _tmpl(frame, rel, threshold=0.70, roi=roi)
    )
    return hits >= 2


# 顶栏标题图标仅旁证（避开左侧 Tab 列与右侧 ? / 金币）
_CRAFT_TITLE_ICON_ROI = (0.08, 0.00, 0.12, 0.10)
_CRAFT_CLOSE_ROI = (0.88, 0.00, 0.12, 0.14)
_CRAFT_TITLE_ICONS: tuple[tuple[str, str], ...] = (
    ("craft_quick", "knowledge/templates/craft_hub/title_quick.png"),
)
_CRAFT_TAB_DESIGN = (1920, 1440)
_CRAFT_TAB_STEMS: tuple[tuple[str, str, str], ...] = (
    ("craft_quick", "快速制作", "tab_quick"),
    ("craft_hub", "加工", "tab_process"),
    ("craft_make", "制作", "tab_make"),
    ("craft_alchemy", "炼金术", "tab_alchemy"),
)
_CRAFT_SHELL_NODES = frozenset({"craft_quick", "craft_hub", "craft_make", "craft_alchemy"})
_CRAFT_MATCH_THR = 0.55
_CRAFT_CHROME_THR = 0.62
_CRAFT_ON_THR = 0.70
_CRAFT_ON_LEAD = 0.05
_CRAFT_SLOT_PAD = 0.30
_CRAFT_ROI_PAD = 0.04


@lru_cache(maxsize=1)
def _craft_tab_slots() -> tuple[tuple[str, str, str, float, float, float, float, float, float], ...]:
    """node, spot, stem, nx, ny, x0, y0, x1, y1（分数，来自 tab_*_{on,off}.json）。"""
    dw, dh = float(_CRAFT_TAB_DESIGN[0]), float(_CRAFT_TAB_DESIGN[1])
    out: list[tuple[str, str, str, float, float, float, float, float, float]] = []
    for node, spot, stem in _CRAFT_TAB_STEMS:
        xs: list[float] = []
        ys: list[float] = []
        x1s: list[float] = []
        y1s: list[float] = []
        for state in ("on", "off"):
            path = ROOT / "knowledge" / "templates" / "craft_hub" / f"{stem}_{state}.json"
            if not path.is_file():
                continue
            try:
                box = json.loads(path.read_text(encoding="utf-8")).get("client_px") or {}
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            x = float(box.get("x") or 0)
            y = float(box.get("y") or 0)
            w = float(box.get("w") or 0)
            h = float(box.get("h") or 0)
            xs.append(x)
            ys.append(y)
            x1s.append(x + w)
            y1s.append(y + h)
        if not xs:
            continue
        x0, y0, x1, y1 = min(xs), min(ys), max(x1s), max(y1s)
        out.append(
            (
                node,
                spot,
                stem,
                ((x0 + x1) / 2.0) / dw,
                ((y0 + y1) / 2.0) / dh,
                x0 / dw,
                y0 / dh,
                x1 / dw,
                y1 / dh,
            )
        )
    return tuple(out)


def _craft_tab_roi() -> tuple[float, float, float, float]:
    slots = _craft_tab_slots()
    if not slots:
        return (0.00, 0.22, 0.18, 0.58)
    x0 = max(0.0, min(s[5] for s in slots) - _CRAFT_ROI_PAD)
    y0 = max(0.0, min(s[6] for s in slots) - _CRAFT_ROI_PAD)
    x1 = min(1.0, max(s[7] for s in slots) + _CRAFT_ROI_PAD)
    y1 = min(1.0, max(s[8] for s in slots) + _CRAFT_ROI_PAD)
    return (x0, y0, x1 - x0, y1 - y0)


def _match_rel(
    frame: Any,
    rel: str,
    *,
    roi: tuple[float, float, float, float] | None,
    threshold: float,
) -> Any:
    from mm_assist.platform.match import match_template

    key = ("mr", id(frame), rel, roi, threshold)
    hit_ok, cached = _memo_get(key, frame)
    if hit_ok:
        return cached
    path = ROOT / rel
    if frame is None or not _template_exists(str(path)):
        _memo_put(key, frame, None)
        return None
    try:
        _bump_tpl()
        hit = match_template(
            frame,
            path,
            roi=roi,
            threshold=threshold,
            min_margin=0.0,
            check_margin=False,
        )
    except Exception:
        hit = None
    _memo_put(key, frame, hit)
    return hit


def _hit_in_slot(
    cx: float,
    cy: float,
    slot: tuple[str, str, str, float, float, float, float, float, float],
    fw: float,
    fh: float,
) -> bool:
    x0, y0, x1, y1 = slot[5] * fw, slot[6] * fh, slot[7] * fw, slot[8] * fh
    wx, hy = max(1.0, x1 - x0), max(1.0, y1 - y0)
    pad = _CRAFT_SLOT_PAD
    return (x0 - wx * pad) <= cx <= (x1 + wx * pad) and (y0 - hy * pad) <= cy <= (y1 + hy * pad)


def _empty_craft_tab_scores() -> dict[str, dict[str, Any]]:
    return {
        node: {"on": 0.0, "off": 0.0, "on_xy": None, "off_xy": None}
        for node, _spot, _stem in _CRAFT_TAB_STEMS
    }


def _craft_tab_scores(frame: Any) -> dict[str, dict[str, Any]]:
    key = ("cts", id(frame))
    hit_ok, cached = _memo_get(key, frame)
    if hit_ok:
        return cached
    empty = _empty_craft_tab_scores()
    try:
        fh, fw = int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        _memo_put(key, frame, empty)
        return empty
    roi = _craft_tab_roi()
    out = _empty_craft_tab_scores()
    for slot in _craft_tab_slots():
        node, _spot, stem = slot[0], slot[1], slot[2]
        for state in ("on", "off"):
            rel = f"knowledge/templates/craft_hub/{stem}_{state}.png"
            hit = _match_rel(frame, rel, roi=roi, threshold=_CRAFT_MATCH_THR)
            if hit is None:
                continue
            cx = float(getattr(hit, "cx", 0) or 0)
            cy = float(getattr(hit, "cy", 0) or 0)
            if not _hit_in_slot(cx, cy, slot, float(fw), float(fh)):
                continue
            out[node][state] = float(getattr(hit, "score", 0.0) or 0.0)
            out[node][f"{state}_xy"] = (int(cx), int(cy))
    _memo_put(key, frame, out)
    return out


def _craft_side_tab_xy(frame: Any, spot: str) -> tuple[int, int] | None:
    slot = next((s for s in _craft_tab_slots() if s[1] == spot), None)
    if slot is None:
        return None
    node = slot[0]
    scores = _craft_tab_scores(frame)
    row = scores.get(node) or {}
    on_xy, off_xy = row.get("on_xy"), row.get("off_xy")
    on_s, off_s = float(row.get("on") or 0.0), float(row.get("off") or 0.0)
    if on_xy is not None and off_xy is not None:
        return on_xy if on_s >= off_s else off_xy
    if off_xy is not None:
        return off_xy
    if on_xy is not None:
        return on_xy
    return None


def craft_tab_xy(frame: Any, spot: str) -> tuple[int, int] | None:
    """製作壳左侧 Tab：加工 / 快速制作 / 制作 / 炼金术。"""
    return _craft_side_tab_xy(frame, str(spot or "").strip())


def _cyan_score(patch: Any) -> float:
    try:
        b = patch[:, :, 0].astype(float)
        g = patch[:, :, 1].astype(float)
        r = patch[:, :, 2].astype(float)
    except Exception:
        return -999.0
    if getattr(b, "size", 0) == 0:
        return -999.0
    return float((g.mean() + b.mean()) / 2.0 - r.mean())


def _title_icon_score(frame: Any, rel: str) -> float:
    hit = _match_rel(frame, rel, roi=_CRAFT_TITLE_ICON_ROI, threshold=0.62)
    if hit is None:
        return 0.0
    return float(getattr(hit, "score", 0.0) or 0.0)


def _best_title_icon(frame: Any) -> str:
    """顶栏旁证：只留快速制作标题图，不产出其它栏。"""
    best_node = ""
    best = 0.0
    for node, rel in _CRAFT_TITLE_ICONS:
        score = _title_icon_score(frame, rel)
        if score > best:
            best_node, best = node, score
    return best_node if best >= 0.70 else ""


def _craft_chrome(frame: Any) -> bool:
    """须看到左侧製作 Tab 至少两个。右上 X 与人物/背包共用，单独不算制作壳。"""
    scores = _craft_tab_scores(frame)
    hits = sum(
        1
        for row in scores.values()
        if float(row.get("on") or 0.0) >= _CRAFT_CHROME_THR
        or float(row.get("off") or 0.0) >= _CRAFT_CHROME_THR
    )
    return hits >= 2


def _selected_craft_tab(frame: Any) -> str:
    """左侧当前高亮 Tab：选中(青)模板，没有再比青色。"""
    if frame is None or not _craft_chrome(frame):
        return ""
    ranked = sorted(
        (
            (float(row.get("on") or 0.0), node, float(row.get("off") or 0.0))
            for node, row in _craft_tab_scores(frame).items()
        ),
        reverse=True,
    )
    if ranked:
        best, node, off = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        if (best >= _CRAFT_ON_THR and (best - second) >= _CRAFT_ON_LEAD
                and (best - off) >= _CRAFT_ON_LEAD):
            return node
    try:
        h, w = int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        return ""
    scored: list[tuple[float, str]] = []
    for slot in _craft_tab_slots():
        node, nx, ny = slot[0], slot[3], slot[4]
        x, y = int(nx * w), int(ny * h)
        rad = max(8, int((slot[7] - slot[5]) * w * 0.25))
        patch = frame[max(0, y - rad) : y + rad, max(0, x - rad) : x + rad]
        scored.append((_cyan_score(patch), node))
    scored.sort(key=lambda it: it[0], reverse=True)
    if len(scored) < 2:
        return ""
    if scored[0][0] < 22.0 or (scored[0][0] - scored[1][0]) < 12.0:
        return ""
    return scored[0][1]


def _craft_title_blob(frame: Any) -> str:
    """顶栏标题 OCR（兜底）。"""
    from mm_assist.perception.item_sense import compact_zh
    from mm_assist.perception.ocr import ocr_roi

    roi = (0.18, 0.00, 0.40, 0.12)
    try:
        _bump_ocr()
        lines = ocr_roi(frame, roi, min_confidence=0.32, backend="cht")
    except Exception:
        return ""
    return compact_zh("".join(str(getattr(ln, "text", "") or "") for ln in lines or []))


_CRAFT_CAT_NODES = {
    0: "craft_quick",
    1: "craft_hub",
    2: "craft_make",
    3: "craft_alchemy",
}


def _craft_shell_from_blob(blob: str) -> str:
    if "快速" in blob and ("製作" in blob or "制作" in blob):
        return "craft_quick"
    if "鍊金" in blob or "炼金" in blob:
        return "craft_alchemy"
    if "加工" in blob:
        return "craft_hub"
    if "製作" in blob or "制作" in blob:
        return "craft_make"
    return ""


def _craft_node_from_category(category: Any) -> str:
    try:
        n = int(category) if category is not None else None
    except (TypeError, ValueError):
        return ""
    if n is None:
        return ""
    return str(_CRAFT_CAT_NODES.get(n, "") or "")


def _craft_shell_how(frame: Any, category: Any = None) -> tuple[str, str]:
    try:
        key = ("csh", id(frame), category)
        hash(key)
    except TypeError:
        key = ("csh", id(frame), repr(category))
    hit_ok, cached = _memo_get(key, frame)
    if hit_ok:
        return cached
    out = _craft_shell_how_uncached(frame, category)
    _memo_put(key, frame, out)
    return out


def _craft_shell_how_uncached(frame: Any, category: Any = None) -> tuple[str, str]:
    if not _craft_chrome(frame):
        return "", ""
    tab = _selected_craft_tab(frame)
    if tab:
        return tab, "tpl"
    mapped = _craft_node_from_category(category)
    if mapped:
        return mapped, "mem"
    blob = _craft_shell_from_blob(_craft_title_blob(frame))
    if blob:
        return blob, "ocr"
    return "", ""


def _craft_shell_node(frame: Any, category: Any = None) -> str:
    node, _how = _craft_shell_how(frame, category)
    return node


def _process_hub_visible(frame: Any) -> bool:
    """加工六卡：标题或金属/木材卡。无左栏壳则不认。"""
    if _craft_shell_node(frame) == "craft_hub":
        return True
    if not _craft_chrome(frame):
        return False
    try:
        from mm_assist.perception.process_sense import _match_card
    except Exception:
        return False
    try:
        return any(_match_card(frame, cid) is not None for cid in ("metal", "wood"))
    except Exception:
        return False


def _quick_craft_visible(frame: Any) -> bool:
    return _craft_shell_node(frame) == "craft_quick"


def _hub_card_xy(frame: Any, card_id: str) -> tuple[int, int] | None:
    """只匹配目标卡，不扫六卡、不 OCR。"""
    from mm_assist.perception.process_sense import HUB_CARD_CLICK, _match_card

    try:
        hit = _match_card(frame, card_id)
    except Exception:
        hit = None
    if hit is not None:
        return (int(hit.cx), int(hit.cy))
    frac = HUB_CARD_CLICK.get(card_id)
    if frac is None or frame is None:
        return None
    try:
        h, w = int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        return None
    return (int(w * frac[0]), int(h * frac[1]))


def _gather_detail_visible(frame: Any) -> bool:
    return _tmpl(
        frame,
        "knowledge/templates/gather_detail/find_nearby.png",
        threshold=0.70,
        roi=(0.0, 0.38, 1.0, 0.60),
    )


def _skill_detail_card_visible(frame: Any, names) -> bool:
    """Current skill cards share DetailCard classes with inventory item cards."""
    observed = set(names or ())
    required = {'NewProfileUIInstance','LivingSkillUIState',
                'LivingSkillUIInstance','DetailCardUIInstance'}
    allowed = required | {'HudUIStepState','NewProfileUIState','DetailCardUIState',
        'TutorialTooltipUIState','UILayerHideState','SizeAdjustmentSender'}
    return bool(required <= observed <= allowed and _gather_detail_visible(frame))


def _gather_detail_named(frame: Any) -> str:
    """已见尋找附近：标题行闭集命中才带物名，否则粗技能详情。"""
    if not _gather_detail_visible(frame):
        return ""
    if not _usable_inspect_frame(frame):
        return "gather_detail"
    name = _gather_detail_title_name(frame)
    if name:
        return f'gather_detail:{name}'
    from mm_assist.perception.item_sense import fold_zh, line_matches_name

    catalog = _material_catalog()
    named: list[str] = []
    for ln in _ocr_lines_roi(frame, _GATHER_DETAIL_HDR_ROI):
        text = str(getattr(ln, "text", "") or "")
        hits = [n for n in catalog if line_matches_name(text, n, catalog)]
        if hits:
            named.append(max(hits, key=lambda n: len(fold_zh(n))))
    if named:
        mat = max(named, key=lambda n: len(fold_zh(n)))
        return f"gather_detail:{mat}"
    mat = _pick_material_from_blob(_ocr_blob(frame, _GATHER_DETAIL_HDR_ROI))
    return f"gather_detail:{mat}" if mat else "gather_detail"


@lru_cache(maxsize=1)
def _gather_detail_title_catalog() -> dict:
    try:
        return json.loads((ROOT / 'knowledge/templates/gather_detail/title_catalog.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _gather_detail_title_name(frame: Any) -> str:
    """Independent full popup headings; no list icon or OCR alias authorizes input."""
    catalog = _gather_detail_title_catalog()
    try:
        threshold = float(catalog.get('threshold'))
        roi = tuple(float(v) for v in catalog['search_roi'])
        if not 0.97 <= threshold <= 1.0 or len(roi) != 4:
            return ''
        if not (0 <= roi[0] < 1 and 0 <= roi[1] < 1 and roi[2] > 0 and roi[3] > 0
                and roi[0] + roi[2] <= 1 and roi[1] + roi[3] <= 1):
            return ''
        rows = catalog['titles']
    except (KeyError, TypeError, ValueError):
        return ''
    names = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get('name') or not row.get('template'):
            continue
        if _match_rel(frame, row['template'], roi=roi, threshold=threshold) is not None:
            names.add(row['name'])
    return next(iter(names)) if len(names) == 1 else ''


def gather_detail_images_only(target: str) -> bool:
    return any(row.get('name') == target for row in _gather_detail_title_catalog().get('titles', []))


def gather_detail_for_target(frame: Any, target: str) -> str:
    """Registered targets never fall back to OCR to permit a nearby action."""
    # Each standalone observation owns its memo. A capture backend may reuse
    # the ndarray for new pixels; object identity cannot establish freshness.
    token = _inspect_shot_memo.set({})
    try:
        if not gather_detail_images_only(target):
            return _gather_detail_named(frame)
        if not _gather_detail_visible(frame):
            return ''
        name = _gather_detail_title_name(frame)
        return f'gather_detail:{name}' if name else 'gather_detail'
    finally:
        _inspect_shot_memo.reset(token)


# 主界面菜单钮 ROI = menu_icons._ROI_MENU_ENTRY（复制数字，避免 import 热模块）
_ROI_MENU_ENTRY = (0.78, 0.0, 0.22, 0.14)
# knowledge/templates/life_grid 无 json roi；menu_icons._ROI_LIFE 几乎整屏。格区估计，待 B 采。
_LIFE_GRID_CELL_ROI = (0.18, 0.18, 0.64, 0.64)
_LIFE_GRID_PNGS: tuple[str, ...] = (
    "daily.png",
    "herb.png",
    "woodcut.png",
    "mine.png",
    "shear.png",
    "harvest.png",
    "hoe.png",
    "insect.png",
    "fish.png",
)
_REFINE_STEPS_FULL: tuple[str, ...] = (
    "gather_detail",
    "obtain",
    "facility",
    "claim_all",
    "system_menu",
    "icon_menu",
    "currency",
    "gather_skill",
    "life_grid",
    "life_tab",
    "craft_shell",
    "process_hub",
)
_REFINE_ROUTE: dict[str, frozenset[str]] = {
    "": frozenset(_REFINE_STEPS_FULL),
    "overlay": frozenset({"icon_menu"}),
    "fine:craft_hub": frozenset({"craft_shell", "process_hub"}),
    "craft_shell": frozenset({"craft_shell", "process_hub"}),
    "fine:facility": frozenset({"facility", "claim_all"}),
    "fine:facility_recipe": frozenset({"facility_recipe", "facility"}),
    "fine:facility_detail": frozenset({"obtain", "facility_recipe", "facility"}),
    "fine:life_grid": frozenset({"gather_detail", "gather_skill", "life_grid", "life_tab"}),
    "item_info": frozenset({"obtain"}),
    "obtain_method": frozenset({"obtain"}),
}


def _refine_steps(sn: str) -> frozenset[str]:
    key = str(sn or "")
    return _REFINE_ROUTE.get(key, frozenset(_REFINE_STEPS_FULL))


def _hub_stack_sn(sn: str) -> bool:
    """只在制作壳上扫六卡。设施页 / 配方页不扫。"""
    s = str(sn or "")
    return s in {"fine:craft_hub", "craft_shell"}


def refine_inspect_node(frame: Any, category: Any = None, *, sn: str = "") -> str:
    """画面认页：模板先于 OCR。专页先于菜单。sn 按栈裁不可能的细页；空串全瀑布；overlay 只认菜单图标。"""
    refine_inspect_node.last_hub_cards = False
    _inspect_sn.set(str(sn or ""))
    if frame is None:
        return ""
    steps = _refine_steps(sn)
    hub = False
    if _hub_stack_sn(sn):
        hub = _hub_overview_visible(frame)
    refine_inspect_node.last_hub_cards = bool(hub)
    if _hub_stack_sn(sn):
        _reset_facility_fid_cache()
    if str(sn or "") == "fine:facility_detail":
        info = _detect_item_info(frame, require_obtain=True)
        if _item_info_sap_or_coal(info):
            return info
    if "facility_recipe" in steps and not hub:
        kind = _facility_kind_node(frame)
        fid = _facility_fid(kind)
        rec = _detect_facility_recipe(frame, fid) if fid else ""
        if fid and rec:
            return f"facility:{fid}:{rec}"
        if kind:
            return kind
        return "facility_recipe"
    if "gather_detail" in steps and _gather_detail_visible(frame):
        return _gather_detail_named(frame) or "gather_detail"
    if "obtain" in steps:
        om = _detect_obtain_method(frame)
        if om:
            return om
    if (
        "facility" in steps
        and not hub
        and _tmpl(
            frame,
            "knowledge/templates/facility/back.png",
            threshold=0.50,
            roi=(0.00, 0.00, 0.10, 0.12),
        )
    ):
        if str(sn or "") in ("", "overlay"):
            info = _detect_item_info(frame)
            if info:
                return info
        return _facility_kind_node(frame)
    # 布料等设施页有时没有左上←，只有「全部領取」。life_tab 会误认成角色。
    if "claim_all" in steps and not hub:
        claimed = _facility_if_claim_all(frame)
        if claimed:
            return claimed
    if "system_menu" in steps and _system_menu_visible(frame):
        return "system_menu"
    if "icon_menu" in steps and _tmpl(
        frame,
        "knowledge/templates/menu/icon_menu_52x44.png",
        threshold=0.70,
        roi=_ROI_MENU_ENTRY,
    ):
        return "main_hud"
    if "currency" in steps:
        try:
            from mm_assist.life_execute.ui_logic import currency_wallet_visible

            if currency_wallet_visible(frame):
                return "currency_wallet"
        except Exception:
            pass
    if "gather_skill" in steps:
        sid = _detect_gather_skill_sid(frame)
        if sid:
            return f"gather_skill:{sid}"
    if "life_grid" in steps:
        for name in _LIFE_GRID_PNGS:
            if _tmpl(
                frame,
                f"knowledge/templates/life_grid/{name}",
                threshold=0.70,
                roi=_LIFE_GRID_CELL_ROI,
            ):
                return "life_grid"
    if "life_tab" in steps and _tmpl(
        frame,
        "knowledge/templates/profile/life_tab.png",
        threshold=0.55,
        roi=_PROFILE_TAB_ROI,
    ):
        if _life_grid_confirmed(frame) or _life_skills_page_visible(frame):
            return "life_grid"
    if "craft_shell" in steps:
        shell = _craft_shell_node(frame, category=category)
        if shell:
            if shell == "craft_hub" and not hub and str(sn or "") == "":
                kind = _facility_kind_node(frame, require_fid=True)
                if kind:
                    return kind
            return shell
    if "process_hub" in steps and _process_hub_visible(frame):
        return "craft_hub"
    return ""


def _edge_xy(
    frame: Any,
    eid: str,
    frm: str,
    to: str,
    *,
    skill_id: str = "",
    facility: str = "",
) -> tuple[int, int] | None:
    from mm_assist.life_execute.ui_nav import locate_edge

    return locate_edge(
        {"id": eid, "from": frm, "to": to},
        frame,
        skill_id=skill_id,
        facility=facility,
    )


def locate_spot(node: str, spot: str, frame: Any, *, root: Path | None = None) -> tuple[int, int] | None:
    """当前帧上该可点项的客户区坐标。找不到返回 None。不点击。"""
    _ = root or ROOT
    n = str(node or "")
    s = str(spot or "")
    if frame is None or not s:
        return None
    try:
        if n in {'craft_items', 'craft_item_recipe'} and s == '返回':
            from mm_assist.perception.insect_obtain_sense import locate
            if locate(frame,'items_title',root=root) is None:
                return None
            hit = _match_rel(frame,'knowledge/templates/facility/back.png',
                             roi=(0,0,.16,.16),threshold=.75)
            return (int(hit.cx),int(hit.cy)) if hit is not None else None
        if n == "main_hud" and s == "菜单":
            return _edge_xy(frame, "main_hud_menu_system_menu", "main_hud", "system_menu")
        if n == "system_menu":
            if s == "角色":
                return _menu_icon_xy(
                    frame,
                    "knowledge/templates/menu/icon_character.png",
                    (0.66, 0.08, 0.14, 0.28),
                )
            if s == "製作":
                return _menu_icon_xy(
                    frame,
                    "knowledge/templates/menu/icon_crafting.png",
                    (0.80, 0.12, 0.16, 0.22),
                )
            if s == "背包":
                return _menu_icon_xy(
                    frame,
                    "knowledge/templates/menu/icon_inventory.png",
                    (0.72, 0.10, 0.18, 0.16),
                )
            if s == "右上 X":
                return _menu_icon_xy(
                    frame,
                    "knowledge/templates/menu/icon_close_x.png",
                    (0.88, 0.00, 0.12, 0.12),
                )
        if n == "profile" or n.startswith("profile:"):
            if s == "生活技能 Tab":
                return _edge_xy(frame, "profile_life_tab_life_grid", "profile", "life_grid")
            if s == "关闭/X":
                return _edge_xy(frame, "profile_close_main_hud", "profile", "main_hud")
        if n == "life_grid":
            if s == "关闭/X":
                return _edge_xy(frame, "life_grid_close_main_hud", "life_grid", "main_hud")
            sid = {label: kid for label, kid in LIFE_GRID_GATHER}.get(s)
            if sid:
                return _edge_xy(
                    frame,
                    "life_grid_skill_gather_detail",
                    "life_grid",
                    "gather_detail",
                    skill_id=sid,
                )
        if n == "gather_skill" or n.startswith("gather_skill:"):
            sid = _gather_skill_sid(n) or "daily"
            if not s:
                return None
            return _item_list_xy(frame, sid, s)
        if n == "gather_detail" or n.startswith("gather_detail:"):
            from mm_assist.life_execute.ui_logic import recognized_click_xy
            from mm_assist.perception.item_sense import locate_find_nearby, locate_skill_item

            if s == "获得方法":
                xy = _named_line_xy(frame, _ITEM_INFO_FOOT_ROI, "獲得方法", contains=True)
                if xy is not None:
                    return xy
                return _named_line_xy(frame, _ITEM_INFO_ROI, "獲得方法", contains=True)

            if s == "淨淨蘑菇":
                return recognized_click_xy(
                    locate_skill_item(frame, skill_id="herb", name="淨淨蘑菇")
                )
            if s == "水井":
                return recognized_click_xy(
                    locate_skill_item(frame, skill_id="daily", name="水井")
                )
            if s == "寻找附近":
                return recognized_click_xy(locate_find_nearby(frame))
            if s == "关闭详情":
                return _edge_xy(
                    frame, "gather_detail_close_life_grid", "gather_detail", "life_grid"
                )
        if n in _CRAFT_SHELL_NODES:
            if s == "右上 X":
                return _edge_xy(frame, "craft_hub_x_main_hud", "craft_hub", "main_hud")
            tab_xy = _craft_side_tab_xy(frame, s)
            if tab_xy is not None:
                return tab_xy
            if n == "craft_hub":
                cid = {
                    "金屬加工": "metal",
                    "木材加工": "wood",
                    "皮革加工": "leather",
                    "布料加工": "cloth",
                    "藥品加工": "medicine",
                    "食材加工": "food",
                }.get(s)
                if cid:
                    return _hub_card_xy(frame, cid)
        if n == "facility" or n.startswith("facility:"):
            fid = _facility_fid(n)
            if s == "左上 ←":
                return _edge_xy(frame, "facility_back_craft_hub", "facility", "craft_hub")
            if s == "全部領取":
                from mm_assist.life_execute.ui_logic import recognized_click_xy
                from mm_assist.perception.process_sense import locate_claim_all

                return recognized_click_xy(locate_claim_all(frame))
            if s in ("前往加工", "加工"):
                from mm_assist.life_execute.ui_logic import recognized_click_xy
                from mm_assist.perception.process_sense import locate_process_button

                btn = locate_process_button(frame)
                if not btn.get("ok"):
                    return None
                label = str(btn.get("label") or "")
                if s == "前往加工" and label != "前往加工":
                    return None
                if s == "加工" and label == "前往加工":
                    return None
                return recognized_click_xy(btn)
            rec = _facility_parts(n)[1]
            if rec and s in _recipe_input_names(rec):
                from mm_assist.perception.process_sense import HOLD_NEED_ROI

                xy = _named_line_xy(frame, HOLD_NEED_ROI, s)
                if xy is not None:
                    return xy
            if fid and s:
                return _facility_item_xy(frame, fid, s)
        if (n == "item_info" or n.startswith("item_info:")) and s == "获得方法":
            xy = _named_line_xy(frame, _ITEM_INFO_FOOT_ROI, "獲得方法", contains=True)
            if xy is not None:
                return xy
            return _named_line_xy(frame, _ITEM_INFO_ROI, "獲得方法", contains=True)
        if (n == "item_info" or n.startswith("item_info:")) and s == "附近位置":
            xy = _named_line_xy(frame, _ITEM_INFO_FOOT_ROI, "附近位置", contains=True)
            if xy is not None:
                return xy
            return _named_line_xy(frame, _ITEM_INFO_ROI, "附近位置", contains=True)
        if n.startswith("obtain_method:") and _is_coal_vein_spot(s):
            return _obtain_vein_xy(frame)
        if n.startswith("obtain_method:") and _is_woodcut_spot(s):
            return _obtain_first_named_xy(frame, "伐木", exclude=("道具", "道具效果"))
        if n == "facility_recipe":
            from mm_assist.life_execute.ui_logic import recognized_click_xy
            from mm_assist.perception.process_sense import locate_process_button

            btn = locate_process_button(frame)
            if not btn.get("ok"):
                return None
            label = str(btn.get("label") or "")
            if s == "前往加工" and label != "前往加工":
                return None
            if s == "加工" and label == "前往加工":
                return None
            return recognized_click_xy(btn)
        if n == "currency_wallet" and s == "右上 X":
            from mm_assist.platform.match import match_template

            for rel in (
                "knowledge/templates/craft_hub/close_x.png",
                "knowledge/templates/menu/icon_close_x.png",
                "knowledge/templates/map/close_x.png",
            ):
                path = ROOT / rel
                if not path.is_file():
                    continue
                try:
                    hit = match_template(
                        frame,
                        path,
                        roi=(0.86, 0.00, 0.14, 0.16),
                        threshold=0.55,
                        min_margin=0.0,
                        check_margin=False,
                    )
                except Exception:
                    hit = None
                if hit is not None:
                    return (int(hit.cx), int(hit.cy))
            return None
        if n == "inventory" and s == "关闭背包":
            return _edge_xy(frame, "inventory_close_main_hud", "inventory", "main_hud")
        if n in ("item_info", "obtain_method") and s == "关闭背包":
            return _edge_xy(frame, "inventory_close_main_hud", "inventory", "main_hud")
        if n == "craft_shell" and s == "右上 X":
            return _edge_xy(frame, "craft_hub_x_main_hud", "craft_hub", "main_hud")
        if n == "inventory" and _is_bag_mat_spot(s):
            return _bag_item_xy(frame, s)
        if n == "inventory" and s == "道具":
            from mm_assist.perception.inventory_sense import locate_props_tab

            return locate_props_tab(frame, root=root)
        if n == "map_dunbarton" and s == "右上 X":
            return _edge_xy(frame, "map_dunbarton_x_main_hud", "map_dunbarton", "main_hud")
    except Exception:
        return None
    return None


# 採礦可见 6 行；末行要接到客户区底，否则甲胄樹/白銅礦脈/冰冷的光群模板被裁。
_ITEM_ROW_BAND = (0.54, 0.58, 0.44, 0.42)
_ITEM_ROWS = 6


def _ocr_item_xy(frame: Any, skill_id: str, name: str) -> tuple[int, int] | None:
    """右栏 OCR 认物名（繁简）；短名不抢长名。画面没有该行则无圈。"""
    from mm_assist.perception.item_sense import (
        ITEM_LIST_ROI,
        items_for_skill,
        line_center,
        line_matches_name,
    )
    from mm_assist.perception.ocr import ocr_roi

    want = str(name or "").strip()
    if frame is None or not want:
        return None
    catalog = items_for_skill(skill_id) or (want,)
    try:
        _bump_ocr()
        lines = ocr_roi(frame, ITEM_LIST_ROI, min_confidence=0.32, backend="cht")
    except Exception:
        return None
    hits: list[tuple[int, int]] = []
    for line in lines or []:
        if not line_matches_name(getattr(line, "text", "") or "", want, catalog):
            continue
        c = line_center(line)
        if c is not None:
            hits.append(c)
    if len(hits) != 1:
        return None
    return hits[0]


def _unique_item_xy(frame: Any, skill_id: str, name: str) -> tuple[int, int] | None:
    """逐行比模板：末行不被同类树/矿脉/光群抢走；不在可见行就不画圈。"""
    from mm_assist.perception.item_sense import item_template_path, items_for_skill
    from mm_assist.platform.match import match_template

    want = str(name or "").strip()
    if frame is None or not want:
        return None
    names = list(items_for_skill(skill_id))
    rx, ry, rw, rh = _ITEM_ROW_BAND
    row_h = rh / float(_ITEM_ROWS)
    won: dict[str, tuple[float, int, int]] = {}
    for i in range(_ITEM_ROWS):
        y0 = ry + i * row_h
        if i == _ITEM_ROWS - 1:
            row_roi = (rx, min(y0, 0.82), rw, max(0.18, 1.0 - min(y0, 0.82)))
        else:
            row_roi = (rx, y0, rw, min(row_h + 0.02, max(0.01, 1.0 - y0)))
        scored: list[tuple[float, str, Any]] = []
        for n in names:
            path = item_template_path(skill_id, n)
            if not path.is_file():
                continue
            try:
                hit = match_template(
                    frame,
                    path,
                    roi=row_roi,
                    threshold=0.70,
                    min_margin=0.0,
                    check_margin=False,
                )
            except Exception:
                hit = None
            if hit is None:
                continue
            scored.append((float(hit.score), n, hit))
        if not scored:
            continue
        scored.sort(key=lambda r: r[0], reverse=True)
        if scored[0][0] < 0.72:
            continue
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.05:
            continue
        sc, n, hit = scored[0]
        prev = won.get(n)
        if prev is None or sc > prev[0]:
            won[n] = (sc, int(hit.cx), int(hit.cy))
    got = won.get(want)
    return (got[1], got[2]) if got else None


def _item_list_xy(frame: Any, skill_id: str, name: str) -> tuple[int, int] | None:
    """光群/树/矿脉图会互撞：必须 OCR 读到该物名才画圈。未显示则无圈。"""
    ocr_xy = _ocr_item_xy(frame, skill_id, name)
    if ocr_xy is None:
        return None
    tmpl_xy = _unique_item_xy(frame, skill_id, name)
    if tmpl_xy is not None and abs(int(tmpl_xy[1]) - int(ocr_xy[1])) <= 70:
        return tmpl_xy
    return ocr_xy


def _hub_overview_visible(frame: Any) -> bool:
    """加工六卡总览：卡图命中 ≥2 张即真。不走 OCR。"""
    try:
        from mm_assist.perception.process_sense import HUB_CARD_IDS, _match_card

        n = 0
        for cid in HUB_CARD_IDS:
            if _match_card(frame, cid) is not None:
                n += 1
                if n >= 2:
                    return True
        return False
    except Exception:
        return False


def _facility_kind_node(frame: Any, *, require_fid: bool = False) -> str:
    fid = _detect_facility_fid(frame)
    if not fid:
        return "" if require_fid else "facility"
    return f"facility:{fid}"


def _facility_if_claim_all(frame: Any) -> str:
    """「全部領取」只出现在设施页，用来挡住 life_tab 误认。"""
    try:
        from mm_assist.perception.process_sense import locate_claim_all

        hit = locate_claim_all(frame)
    except Exception:
        return ""
    if not (isinstance(hit, dict) and hit.get("ok") is True):
        return ""
    return _facility_kind_node(frame)


_FACILITY_TITLE_FIDS = ("metal", "wood", "leather", "cloth", "medicine", "food")


def _reset_facility_fid_cache() -> None:
    _shot_memo().pop("facility_fid", None)


def _fid_cache_id(names: list[str] | None = None) -> str:
    from mm_assist.memory.readers.ui_stack import stack_compare_key

    if names is not None:
        return stack_compare_key(list(names or []))
    return str(_inspect_stack_key.get() or "")


def _cached_facility_node(sn: str, names: list[str] | None = None) -> str:
    # All facilities share AlteringFacilityPopupUIInstance. Stack identity alone
    # cannot establish which facility is on screen.
    _ = sn, names
    return ""


def _detect_facility_fid(frame: Any) -> str:
    """左上标题：先模板 title_{fid}.png，缺图或未命中再 OCR。"""
    if frame is None:
        return ""
    hit_ok, cached = _memo_get("facility_fid", frame)
    if hit_ok:
        return str(cached or "")
    try:
        from mm_assist.perception.process_sense import facility_title_roi, _template_title_facility, _title_facility
        TITLE_ROI = facility_title_roi()
    except Exception:
        return ""
    if TITLE_ROI is None:
        return ""
    try:
        fid = _template_title_facility(frame, on_template=_bump_tpl)
    except Exception:
        fid = None
    if fid:
        _memo_put("facility_fid", frame, fid)
        return fid
    if _title_facility is None:
        return ""
    try:
        blob = _ocr_blob(frame, TITLE_ROI)
    except Exception:
        return ""
    fid = str(_title_facility(blob) or "")
    _memo_put("facility_fid", frame, fid)
    return fid


_RECIPE_DETAIL_ROI = (0.54, 0.46, 0.32, 0.12)


def _facility_recipe_open(frame: Any) -> bool:
    return _process_btn_hit(frame)


def _pick_recipe_from_blob(blob: str, names: tuple[str, ...]) -> str:
    from mm_assist.perception.item_sense import fold_zh

    blob_f = fold_zh(blob)
    if not blob_f:
        return ""
    exact = [n for n in names if fold_zh(n) == blob_f]
    if exact:
        return exact[0]
    contained = [n for n in names if fold_zh(n) and fold_zh(n) in blob_f]
    if contained:
        return max(contained, key=lambda n: len(fold_zh(n)))
    supers = [n for n in names if blob_f in fold_zh(n)]
    if len(supers) == 1:
        return supers[0]
    return ""


def _detect_facility_recipe(frame: Any, fid: str) -> str:
    """右下详情已展开（有前往加工/加工）时，读当前配方名。"""
    if frame is None or not fid or not _facility_recipe_open(frame):
        return ""
    blob = _ocr_blob(frame, _RECIPE_DETAIL_ROI)
    return _pick_recipe_from_blob(blob, _facility_item_names(fid))


def _facility_item_xy(frame: Any, fid: str, label: str) -> tuple[int, int] | None:
    """设施配方卡：模板优先，OCR 读全名才圈。"""
    from mm_assist.perception.item_sense import fold_zh, line_center, line_matches_name
    from mm_assist.perception.process_sense import (
        FACILITY_LIST_ROI,
        recipe_template_path,
        recipes_for_facility,
    )
    from mm_assist.platform.match import match_template

    want = str(label or "").strip()
    if frame is None or not fid or not want:
        return None
    catalog = _facility_item_names(fid) or (want,)
    row = None
    want_f = fold_zh(want)
    for rec in recipes_for_facility(fid):
        lab = str(rec.get("label") or rec.get("name") or "").strip()
        if fold_zh(lab) == want_f:
            row = rec
            break
    if row is not None:
        name = str(row.get("name") or "")
        var = str(row.get("variant") or "") or None
        if var == "鋼錠":
            var = None
        path = recipe_template_path(fid, name, variant=var)
        if path.is_file():
            try:
                hit = match_template(
                    frame,
                    path,
                    roi=FACILITY_LIST_ROI,
                    threshold=0.70,
                    min_margin=0.0,
                    check_margin=False,
                )
            except Exception:
                hit = None
            if hit is not None:
                return (int(hit.cx), int(hit.cy))
    try:
        from mm_assist.perception.ocr import ocr_roi

        _bump_ocr()
        lines = ocr_roi(frame, FACILITY_LIST_ROI, min_confidence=0.28, backend="cht")
    except Exception:
        return None
    needles = tuple(str(x) for x in (row or {}).get("ocr_needles") or ())
    pts: list[tuple[int, int]] = []
    for line in lines or []:
        text = str(getattr(line, "text", "") or "")
        ok = line_matches_name(text, want, catalog)
        if not ok and needles:
            blob = fold_zh(text)
            ok = any(bool(fold_zh(n)) and fold_zh(n) in blob for n in needles)
        if not ok:
            continue
        c = line_center(line)
        if c is not None:
            pts.append(c)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if max(xs) - min(xs) <= 100 and max(ys) - min(ys) <= 100:
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
    return pts[0]


_GATHER_SKILL_PAGE_ROI = (0.50, 0.28, 0.48, 0.68)


@lru_cache(maxsize=64)
def _gather_skill_page_anchor_pngs(sid: str) -> tuple[Path, ...]:
    folder = ROOT / "knowledge" / "templates" / "gather_skill" / str(sid or "").strip()
    if not folder.is_dir():
        return ()
    return tuple(sorted(folder.glob("*.png")))


def _best_template_score(
    frame: Any,
    paths: tuple[Path, ...] | list[Path],
    *,
    roi: tuple[float, float, float, float],
    threshold: float,
) -> float:
    from mm_assist.platform.match import match_template

    best = 0.0
    for path in paths:
        if not _template_exists(str(path)):
            continue
        try:
            _bump_tpl()
            hit = match_template(
                frame,
                path,
                roi=roi,
                threshold=threshold,
                min_margin=0.0,
                check_margin=False,
            )
        except Exception:
            hit = None
        if hit is not None:
            best = max(best, float(hit.score))
    return best


def _pick_skill_peak(peaks: list[tuple[float, str]]) -> str:
    peaks.sort(reverse=True)
    if not peaks or peaks[0][0] < 0.80:
        return ""
    if len(peaks) > 1 and peaks[0][0] - peaks[1][0] < 0.04:
        return ""
    return peaks[0][1]


def _detect_gather_skill_sid(frame: Any) -> str:
    """右栏认哪一门：先图标+物名页锚，再详情物图标。分差不够不定。"""
    from mm_assist.perception.item_sense import GATHER_SKILL_IDS, items_for_skill

    roi = _GATHER_SKILL_PAGE_ROI
    page_peaks: list[tuple[float, str]] = []
    for sid in GATHER_SKILL_IDS:
        pngs = _gather_skill_page_anchor_pngs(sid)
        if not pngs:
            continue
        page_peaks.append((_best_template_score(frame, pngs, roi=roi, threshold=0.55), sid))
    picked = _pick_skill_peak(page_peaks)
    if picked:
        return picked

    peaks: list[tuple[float, str]] = []
    for sid in GATHER_SKILL_IDS:
        paths = tuple(
            ROOT / f"knowledge/templates/items/{sid}/{name}.png"
            for name in items_for_skill(sid)
        )
        peaks.append((_best_template_score(frame, paths, roi=roi, threshold=0.55), sid))
    return _pick_skill_peak(peaks)


_DIRECT_STACK = frozenset(
    {
        "dialog",
        "dialog_skip",
        "system_menu",
        "life_grid",
        "item_info",
        "map",
        "map_dunbarton",
        "map_continent",
        "currency_wallet",
        "craft_quick",
        "craft_hub",
        "craft_make",
        "craft_alchemy",
    }
)
_GRAB_STACK = frozenset({"inventory"})
# 左侧四 Tab 竖列（标定 1920×1440 client_px）。xywh。
_PROFILE_TAB_ROI: tuple[float, float, float, float] = (0.00, 0.22, 0.18, 0.58)
_PROFILE_TAB_BOX: tuple[float, float, float, float] = (0.00, 0.22, 0.18, 0.80)
_PROFILE_TAB_MARGIN = 0.03
_PROFILE_TAB_DESIGN = (1920, 1440)
_PROFILE_TAB_SLOTS: tuple[tuple[str, int, int, int, int], ...] = (
    ("info", 56, 591, 80, 80),
    ("combat", 56, 671, 80, 80),
    ("life", 56, 751, 80, 80),
    ("career", 56, 831, 80, 80),
)
_PROFILE_ON_TABS: tuple[tuple[str, str], ...] = (
    ("info", "knowledge/templates/profile/tab_info_on.png"),
    ("combat", "knowledge/templates/profile/tab_combat_on.png"),
    ("life", "knowledge/templates/profile/tab_life_on.png"),
    ("life", "knowledge/templates/profile/life_tab.png"),
    ("career", "knowledge/templates/profile/tab_career_on.png"),
)
_PROFILE_INFO_CHROME = "knowledge/templates/profile/info_combat_power.png"
_PROFILE_INFO_CHROME_ROI = (0.10, 0.18, 0.30, 0.22)
_PROFILE_TAB_ICON_THR = 0.80
_PROFILE_TAB_SAT_MIN = 55.0
_PROFILE_TAB_SAT_LEAD = 18.0
_PROFILE_TAB_VISUAL: tuple[tuple[str, str], ...] = (
    ("info", "knowledge/templates/profile/tab_info_on.png"),
    ("combat", "knowledge/templates/profile/tab_combat_on.png"),
    ("life", "knowledge/templates/profile/tab_life_on.png"),
    ("career", "knowledge/templates/profile/tab_career_on.png"),
)
_MAP_TITLE_ROI = (0.00, 0.00, 0.42, 0.12)
_MAP_CONTINENT_MARKS = (
    "歐拉大陸",
    "欧拉大陆",
    "歐拉大陆",
    "欧拉大陸",
    "欧拉大睦",
    "歐拉大睦",
)
_MAP_DUNBARTON_MARKS = ("杜巴頓", "杜巴顿", "杜巴頔", "尚杜巴")


def _profile_match_roi(frame: Any) -> tuple[float, float, float, float] | None:
    """整客户区用左侧条带 ROI；已是条带则 roi=None，避免错位。"""
    try:
        h, w = int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        return _PROFILE_TAB_ROI
    if w <= 0 or h <= 0:
        return _PROFILE_TAB_ROI
    if w <= max(8, int(h * 0.45)):
        return None
    return _PROFILE_TAB_ROI


def _profile_slot_sat(patch: Any) -> float:
    """选中 Tab 高饱和；未选中灰图标饱和度低。"""
    try:
        import cv2
    except Exception:
        return 0.0
    if patch is None or getattr(patch, "size", 0) == 0:
        return 0.0
    try:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    except Exception:
        return 0.0
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = val > 40
    if int(mask.sum()) < 20:
        return 0.0
    return float(sat[mask].mean())


def detect_profile_top_tab_slots(frame: Any) -> str:
    """四格标定位置里饱和度最高且明显亮过其它格 → 当前 Tab。"""
    try:
        h, w = int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        return ""
    if w < 80 or h < 80:
        return ""
    sx = w / float(_PROFILE_TAB_DESIGN[0])
    sy = h / float(_PROFILE_TAB_DESIGN[1])
    scores: list[tuple[float, str]] = []
    for key, x, y, bw, bh in _PROFILE_TAB_SLOTS:
        x0 = max(0, int(x * sx))
        y0 = max(0, int(y * sy))
        x1 = min(w, x0 + max(8, int(bw * sx)))
        y1 = min(h, y0 + max(8, int(bh * sy)))
        if x1 <= x0 or y1 <= y0:
            continue
        scores.append((_profile_slot_sat(frame[y0:y1, x0:x1]), key))
    scores.sort(reverse=True)
    if not scores or scores[0][0] < 55.0:
        return ""
    if len(scores) > 1 and scores[0][0] - scores[1][0] < 18.0:
        return ""
    return scores[0][1]


def detect_profile_top_tab(frame: Any) -> str:
    """先看四格谁更亮（选中彩标）；模板只作后备。不用 off 定 Tab。"""
    if frame is None or not hasattr(frame, "shape"):
        return ""
    slot = detect_profile_top_tab_slots(frame)
    if slot:
        return slot
    from mm_assist.platform.match import match_template

    roi = _profile_match_roi(frame)
    peaks: list[tuple[float, str]] = []
    by_key: dict[str, float] = {}
    for key, rel in _PROFILE_ON_TABS:
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            hit = match_template(
                frame,
                path,
                roi=roi,
                threshold=0.52,
                min_margin=0.0,
                check_margin=False,
            )
        except Exception:
            hit = None
        score = float(hit.score) if hit is not None else 0.0
        if score > by_key.get(key, 0.0):
            by_key[key] = score
    peaks = [(sc, k) for k, sc in by_key.items()]
    peaks.sort(reverse=True)
    if not peaks or peaks[0][0] < 0.58:
        return ""
    if len(peaks) > 1 and peaks[0][0] - peaks[1][0] < _PROFILE_TAB_MARGIN:
        return ""
    return peaks[0][1]


def _profile_non_life_tab(frame: Any) -> str:
    """先认资讯页独有标识，再检查图标落点的选中色；不用设计格位。"""
    if frame is None or not hasattr(frame, "shape"):
        return ""
    if _tmpl(
        frame,
        _PROFILE_INFO_CHROME,
        threshold=0.70,
        roi=_PROFILE_INFO_CHROME_ROI,
    ):
        return "info"
    try:
        h, w = int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        return ""
    if w < 80 or h < 80:
        return ""
    roi = _profile_match_roi(frame)
    scores: list[tuple[float, str]] = []
    for key, rel in _PROFILE_TAB_VISUAL:
        hit = _match_rel(frame, rel, roi=roi, threshold=_PROFILE_TAB_ICON_THR)
        if hit is None:
            continue
        cx = int(getattr(hit, "cx", 0) or 0)
        cy = int(getattr(hit, "cy", 0) or 0)
        rad = 36
        patch = frame[max(0, cy - rad) : min(h, cy + rad), max(0, cx - rad) : min(w, cx + rad)]
        scores.append((_profile_slot_sat(patch), key))
    scores.sort(reverse=True)
    if not scores or scores[0][0] < _PROFILE_TAB_SAT_MIN:
        return ""
    if len(scores) > 1 and scores[0][0] - scores[1][0] < _PROFILE_TAB_SAT_LEAD:
        return ""
    key = scores[0][1]
    if key in ("info", "combat", "career"):
        return key
    return ""


def _life_skills_page_visible(frame: Any) -> bool:
    """20 门生活技能格：日常採集等大格仍在角色壳上，栈类名仍是人物。"""
    hits = 0
    for name in ("daily.png", "woodcut.png", "mine.png"):
        if _tmpl(frame, f"knowledge/templates/life_grid/{name}", threshold=0.62):
            hits += 1
    return hits >= 1


def _life_grid_confirmed(frame: Any) -> bool:
    """两个大格共同命中才走快速路径；详情覆盖层由调用者先排除。"""
    hits = 0
    for name in ("daily.png", "woodcut.png", "mine.png"):
        if _tmpl(frame, f"knowledge/templates/life_grid/{name}", threshold=0.70):
            hits += 1
            if hits >= 2:
                return True
    return False


def _usable_inspect_frame(frame: Any) -> bool:
    if frame is None:
        return False
    try:
        h, w = int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        return False
    return h > 0 and w > 0


def inspect_needs_grab(sn: str, view: Any = None) -> bool:
    """栈已能定页则不截屏；背包叠层、开图细分、制作壳/设施、粗角色页、生活技能大格才抓屏。仅 Hud（stack_hud_only）的 overlay 才抓。"""
    n = str(sn or "")
    if _craft_node_from_view(view):
        return False
    if n in ("profile", "life_grid") and _profile_node_from_view(view):
        return False
    if n == "profile" or n == "life_grid":
        return True
    if n.startswith("profile:"):
        return False
    if n == "map":
        if _map_node_from_view(view):
            return False
        mt = map_node_from_type(_view_field(view, "map_type"))
        return mt == "map"
    if n == "overlay":
        from mm_assist.memory.readers.ui_stack import stack_hud_only

        names = _view_field(view, "ui_stack_names")
        seq = list(names or []) if isinstance(names, (list, tuple)) else []
        return stack_hud_only(seq)
    if (
        n in ("item_info", "obtain_method")
        or n.startswith("item_info:")
        or n.startswith("obtain_method:")
    ):
        return True
    if n == "fine:facility_detail":
        return True
    if n == "fine:facility":
        return True
    return n in _GRAB_STACK or n.startswith("fine:") or n == "craft_shell"


def mem_should_apply_stack(sn: str, held_n: str, *, inflight: bool) -> bool:
    """内存拍是否写出「界面：」。抓屏中途也要能先写出栈页，禁止一直停在 —。"""
    held = str(held_n or "")
    n = str(sn or "")
    if n == "profile" and _profile_shell_fine(held):
        return False
    if n == "life_grid" and _life_grid_fine(held):
        return False
    if n == "fine:craft_hub" and _craft_shell_fine(held):
        return False
    if n == "fine:facility":
        if held == "facility" or (held.startswith("facility:") and not _facility_parts(held)[1]):
            return False
    if n == "fine:facility_recipe" and (
        _facility_page(held) or held == "facility_recipe"
    ):
        return False
    if n == "fine:facility_detail" and _item_info_fine(held):
        return False
    if n == "item_info" and _item_info_fine(held):
        return False
    if n == "obtain_method" and _obtain_method_fine(held):
        return False
    if n == "gather_detail" and _gather_detail_fine(held):
        return False
    if inflight and inspect_needs_grab(n) and held:
        return False
    return True


def map_node_from_type(mt: Any) -> str:
    """MapType is World/Town/Dungeon/MoonGate, never a particular region."""
    return "map"


def inspect_map_from_frame(frame: Any) -> str:
    """开图后用顶栏图/OCR 分欧拉大陆与杜巴顿；分不出只报「地图」。不用地点内存。"""
    node, _how = _inspect_map_how(frame)
    return node


def _inspect_map_how(frame: Any) -> tuple[str, str]:
    if frame is None:
        return "map", ""
    from mm_assist.perception.map_header import cloud_region_visible, breadcrumb_has_secondary_place
    if cloud_region_visible(frame):
        return "map", "tpl"
    if _tmpl(frame, "knowledge/templates/map/nerys.png", threshold=0.70):
        return "map_dunbarton", "tpl"
    blob = _ocr_blob(frame, _MAP_TITLE_ROI, fallback=True)
    has_d = any(m in blob for m in _MAP_DUNBARTON_MARKS)
    has_c = any(m in blob for m in _MAP_CONTINENT_MARKS)
    if has_d:
        return "map_dunbarton", "ocr"
    if breadcrumb_has_secondary_place(blob):
        return "map", "ocr"
    if has_c:
        return "map_continent", "ocr"
    return "map", ""


def _view_or_state(view: Any) -> Any:
    if view is not None:
        return view
    try:
        from mm_assist.perception.state_ui import read_state_ui

        return read_state_ui()
    except Exception:
        return None


def _fine_fallback(sn: str) -> str:
    if sn == "overlay":
        return ""
    if sn == "fine:craft_hub":
        return "craft_shell"
    if sn == "fine:facility":
        return "facility"
    if sn == "fine:facility_recipe":
        return "facility_recipe"
    if sn == "fine:facility_detail":
        return "facility_detail"
    if sn == "fine:life_grid":
        return "life_grid"
    if sn.startswith("fine:"):
        return sn.split(":", 1)[-1]
    return sn


def _inspect_source(node: str, *, sn: str, tpl: bool, ocr: bool = False) -> str:
    if not node:
        return "pending"
    if tpl:
        return "stack+tpl"
    if ocr:
        return "stack+ocr"
    return "stack"


def _view_field(view: Any, *keys: str) -> Any:
    if view is None:
        return None
    raw = getattr(view, "raw", None)
    for key in keys:
        if hasattr(view, key):
            val = getattr(view, key)
            if val is not None and val != "":
                return val
        if isinstance(raw, dict) and raw.get(key) is not None:
            return raw.get(key)
    return None


def _map_node_from_view(view: Any) -> str:
    """A current listener proves the map; its paired loaded view can refine it."""
    from mm_assist.life_execute.ui_logic import _stack_from_view
    from mm_assist.memory.readers.ui_stack import MAP_LISTENER_COMPANIONS
    if view is None or getattr(view, "fresh", True) is False or getattr(view, "pid_alive", True) is False:
        return ""
    try:
        age = time.time() - float(_view_field(view, "ui_stack_ts") or 0)
    except (TypeError, ValueError):
        return ""
    if not -0.05 <= age <= 1.5 or _view_field(view, "ui_stack_error"):
        return ""
    identity = _view_field(view, "map_identity")
    if (not isinstance(identity, (tuple, list)) or len(identity) != 3
            or any(type(p) is not int or p < 0x10000 for p in identity)):
        return ""
    names, ready = _stack_from_view(view)
    if not (ready and "MapUIInstance" in (names or []) and set(names or []) <= MAP_LISTENER_COMPANIONS):
        return ""
    from mm_assist.memory.readers.ui_stack import selected_map_view
    fact = selected_map_view(_view_field(view, "map_view"))
    return fact["node"] if fact else "map"


def _selected_map_view(view: Any) -> dict[str, Any] | None:
    from mm_assist.memory.readers.ui_stack import selected_map_view
    return selected_map_view(_view_field(view, "map_view")) if _map_node_from_view(view) else None


def _craft_items_from_view(frame: Any, view: Any) -> bool:
    """A child crafting window covers the shell's left tabs entirely."""
    from mm_assist.life_execute.ui_logic import _stack_from_view
    from mm_assist.perception.insect_obtain_sense import locate, items_stack_allowed
    if (view is None or frame is None or getattr(view,'fresh',True) is False
            or getattr(view,'pid_alive',True) is False or _view_field(view,'ui_stack_error')):
        return False
    try:
        age = time.time()-float(_view_field(view,'ui_stack_ts') or 0)
    except (TypeError,ValueError):
        return False
    names,ready = _stack_from_view(view)
    if (not ready or not -.05 <= age <= 1.5
            or not {'CraftingHubPopupUIInstance','CraftingFacilityPopupUIInstance'} <= set(names or [])
            or not items_stack_allowed(names)):
        return False
    return (locate(frame,'items_title') is not None and _match_rel(frame,
        'knowledge/templates/facility/back.png',roi=(0,0,.16,.16),threshold=.72) is not None)


def _bag_insect_card_from_view(frame: Any, view: Any) -> str:
    from mm_assist.life_execute.ui_logic import _stack_from_view
    from mm_assist.perception.insect_obtain_sense import (
        PREFIX, bag_card_stack_allowed, bag_card_visible, phase,
    )
    if (view is None or frame is None or getattr(view, 'fresh', True) is False
            or getattr(view, 'pid_alive', True) is False or _view_field(view, 'ui_stack_error')):
        return ''
    try:
        age = time.time() - float(_view_field(view, 'ui_stack_ts') or 0)
    except (TypeError, ValueError):
        return ''
    names, ready = _stack_from_view(view)
    if not ready or not -.05 <= age <= 1.5 or not bag_card_stack_allowed(names):
        return ''
    matches = [mat for mat in PREFIX
               if bag_card_visible(frame, mat) and phase(frame, mat) == 'card']
    return 'item_info:' + matches[0] if len(matches) == 1 else ''


def _craft_insect_overlay_from_view(frame: Any, view: Any) -> str:
    """Current potion recipe and material card cover the crafting shell tabs."""
    from mm_assist.life_execute.ui_logic import _stack_from_view
    from mm_assist.perception.insect_obtain_sense import PREFIX, items_stack_allowed, phase
    if (view is None or frame is None or getattr(view, 'fresh', True) is False
            or getattr(view, 'pid_alive', True) is False or _view_field(view, 'ui_stack_error')):
        return ''
    try:
        age = time.time() - float(_view_field(view, 'ui_stack_ts') or 0)
    except (TypeError, ValueError):
        return ''
    names, ready = _stack_from_view(view)
    names = set(names or [])
    required = {'CraftingHubPopupUIInstance', 'CraftingFacilityPopupUIInstance',
                'CraftingPanelPopupUIInstance'}
    if (not ready or not -.05 <= age <= 1.5 or not required <= names
            or not items_stack_allowed(names - {'CraftingPanelPopupUIInstance', 'DetailCardUIInstance'})):
        return ''
    card = 'DetailCardUIInstance' in names
    matches = [mat for mat in PREFIX if phase(frame, mat) == ('card' if card else 'recipe')]
    if len(matches) != 1:
        return ''
    if card:
        return 'item_info:' + matches[0]
    from mm_assist.perception.insect_obtain_sense import material_xy, locate
    if (material_xy(frame, matches[0], inventory=False) is None
            or locate(frame, 'items_title') is None
            or _match_rel(frame, 'knowledge/templates/facility/back.png',
                          roi=(0, 0, .16, .16), threshold=.75) is None):
        return ''
    return 'craft_item_recipe'


def _craft_node_from_view(view: Any) -> str:
    """Only the current active shell's keeper may bypass page images."""
    from mm_assist.life_execute.ui_logic import _stack_from_view
    from mm_assist.memory.readers.ui_stack import craft_node_from_current
    if view is None or getattr(view, "fresh", True) is False or getattr(view, "pid_alive", True) is False:
        return ""
    try:
        age = time.time() - float(_view_field(view, "ui_stack_ts") or 0)
    except (TypeError, ValueError):
        return ""
    if not -0.05 <= age <= 1.5 or _view_field(view, "ui_stack_error"):
        return ""
    identity = _view_field(view, "craft_identity")
    if (not isinstance(identity, (tuple, list)) or len(identity) != 5
            or any(type(p) is not int or p < 0x10000 for p in identity)):
        return ""
    names, ready = _stack_from_view(view)
    return craft_node_from_current(names, ready=ready, category=_view_field(view, "craft_current_category"))


def _profile_node_from_view(view: Any) -> str:
    """A fresh, active, independently paired profile fact; never a retained UI label."""
    from mm_assist.life_execute.ui_logic import _stack_from_view
    from mm_assist.memory.readers.ui_stack import profile_node_from_category
    if view is None or getattr(view, "fresh", True) is False or getattr(view, "pid_alive", True) is False:
        return ""
    try:
        age = time.time() - float(_view_field(view, "ui_stack_ts") or 0)
    except (TypeError, ValueError):
        return ""
    if not -0.05 <= age <= 1.5 or _view_field(view, "ui_stack_error"):
        return ""
    identity = _view_field(view, "profile_identity")
    if (not isinstance(identity, (tuple, list)) or len(identity) != 2
            or any(type(p) is not int or p < 0x10000 for p in identity)):
        return ""
    names, ready = _stack_from_view(view)
    return profile_node_from_category(names, ready=ready, category=_view_field(view, "profile_category"))


def compute_inspect(
    frame: Any,
    *,
    spot: str = "",
    view: Any = None,
    stack_after: list[str] | None = None,
) -> dict[str, Any]:
    """先 UI 栈定页；仅清主界面再读 stp；最后才给「界面：」。

    stack_after：抓屏后再读的栈名。传入则与当前 view 对拍；不一致本拍不认页。
    """
    from mm_assist.life_execute.ui_logic import _stack_from_view
    from mm_assist.memory.readers.ui_stack import REPAIR_NAMES, stack_compare_key, stack_node

    t0 = time.perf_counter()
    _reset_inspect_cost()
    v = _view_or_state(view)
    names, ready = _stack_from_view(v) if v is not None else (None, False)
    if stack_after is not None:
        after = [str(x) for x in stack_after]
        if stack_compare_key(list(names or [])) != stack_compare_key(after):
            return {
                "node": "",
                "xy": None,
                "stack_names": after,
                "stack_ready": bool(ready),
                "source": "pending",
                "skew": True,
                "stp": None,
                "cost": _inspect_cost(t0),
            }
    cat = _view_field(v, "craft_hub_category", "craft_category")
    try:
        cat_i = int(cat) if cat is not None else None
    except (TypeError, ValueError):
        cat_i = None
    empty_ok = bool(_view_field(v, "ui_stack_empty_ok"))
    sn = stack_node(
        list(names or []),
        ready=bool(ready),
        craft_category=cat_i,
        empty_ok=empty_ok,
    )
    _inspect_stack_key.set(stack_compare_key(list(names or [])))
    _inspect_sn.set(str(sn or ""))
    tpl = False
    ocr = False
    profile_looked = False
    life_looked = False
    profile_mem = _profile_node_from_view(v)
    map_mem = _map_node_from_view(v)
    craft_mem = _craft_node_from_view(v)
    insect_overlay = (_craft_insect_overlay_from_view(frame, v)
                      if sn in {'fine:facility_detail', 'fine:craft_hub', 'craft_shell'} else '')
    bag_insect_card = _bag_insect_card_from_view(frame, v) if sn == 'item_info' else ''
    craft_how = ""
    hub_cards = False
    hud_menu_icon = False
    town_canvas = False
    map_chrome = False
    if (sn in {"main_hud", "overlay"} and ready
            and set(names or []) <= {"HudUIStepState", "TutorialTooltipUIState"}
            and _usable_inspect_frame(frame)):
        from mm_assist.life_execute.repair_dialog import npc_dialog_visible

        # These dialogs can leave a ready empty/HUD stack. Their registered
        # current pixels must take precedence over HUD/menu-shaped scenery.
        if npc_dialog_visible(frame):
            sn = "dialog"
            tpl = True
    if sn == "main_hud" and set(names or []) <= {"HudUIStepState", "TutorialTooltipUIState"}:
        from mm_assist.perception.map_header import coarse_map_visible
        # MapUI can leave only the HUD shell while its optional component is
        # absent. Independent current pixels establish coarse map identity;
        # neither the breadcrumb wording nor this shell establishes a tier.
        map_chrome = coarse_map_visible(frame)
        if map_chrome:
            sn = "map"
            tpl = True
    if sn == "main_hud" and not list(names or []):
        from mm_assist.nav.map_localize import known_town_canvas_matches
        # Some town maps leave a genuinely empty UI stack. Independently
        # registered terrain proves that frame is still a map, not the HUD.
        town_canvas = known_town_canvas_matches(frame)
        if town_canvas:
            sn = "map"
            tpl = True
        else:
            hit = (frame is not None
                   and locate_spot("main_hud", "菜单", frame) is not None)
            if hit:
                hud_menu_icon = True
                tpl = True
            elif not empty_ok:
                sn = ""
    if not ready or not sn:
        node = ""
    elif insect_overlay:
        node = insect_overlay
        tpl = True
    elif profile_mem:
        node = profile_mem
        profile_looked = True
        life_looked = node == "life_grid"
    elif craft_mem:
        node = craft_mem
        craft_how = "mem"
    elif map_mem:
        node = map_mem
    elif sn == "map":
        node = "map" if town_canvas or map_chrome else map_node_from_type(_view_field(v, "map_type"))
        if node == "map" and frame is not None and not (town_canvas or map_chrome):
            node, how = _inspect_map_how(frame)
            if how == "tpl":
                tpl = True
            elif how == "ocr":
                ocr = True
    elif sn == "profile":
        node = "profile"
        if frame is not None:
            profile_looked = True
            if _gather_detail_visible(frame):
                node = _gather_detail_named(frame) or "gather_detail"
                tpl = True
                life_looked = True
            elif _life_grid_confirmed(frame):
                node = "life_grid"
                tpl = True
                life_looked = True
            else:
                non_life = _profile_non_life_tab(frame)
                if non_life:
                    node = f"profile:{non_life}"
                    tpl = True
                else:
                    sid = _detect_gather_skill_sid(frame)
                    if sid:
                        node = f"gather_skill:{sid}"
                        tpl = True
                        life_looked = True
                    else:
                        tab = detect_profile_top_tab(frame)
                        life_page = _life_skills_page_visible(frame)
                        if tab in ("info", "combat", "career"):
                            node = f"profile:{tab}"
                            tpl = True
                        elif life_page:
                            node = "life_grid"
                            tpl = True
                            life_looked = True
    elif sn == "life_grid":
        node = "life_grid"
        if frame is not None:
            life_looked = True
            if _gather_detail_visible(frame):
                node = _gather_detail_named(frame) or "gather_detail"
                tpl = True
            else:
                sid = _detect_gather_skill_sid(frame)
                if sid:
                    node = f"gather_skill:{sid}"
                    tpl = True
    elif sn in ("item_info", "obtain_method"):
        node = sn
        if sn == 'item_info' and _skill_detail_card_visible(frame, names):
            node = _gather_detail_named(frame) or 'gather_detail'
            tpl = True
            life_looked = True
        elif bag_insect_card:
            node = bag_insect_card
            tpl = True
        elif _usable_inspect_frame(frame):
            named = (
                _detect_item_info(frame)
                if sn == "item_info"
                else _detect_obtain_method(frame)
            )
            if str(named or "").startswith(sn):
                node = named
                ocr = named != sn
    elif sn in _DIRECT_STACK or sn.startswith("stack:"):
        node = sn
    elif sn in _GRAB_STACK:
        overlay = ""
        if sn == "inventory" and frame is not None:
            overlay = _inventory_overlay_node(frame)
        if overlay:
            node = overlay
            tpl = True
        else:
            node = sn
    elif sn in {"craft_shell","fine:craft_hub"} and _craft_items_from_view(frame,v):
        node = "craft_items"
        tpl = True
    elif sn == "overlay" and any(n in REPAIR_NAMES for n in names or []):
        # The repair window remains an active overlay even when its State
        # disappears. Background HUD-shaped pixels never authorize a menu click.
        node = "overlay"
    elif sn.startswith("fine:") or sn == "overlay" or sn == "craft_shell":
        refine_inspect_node.last_hub_cards = False
        refined = (
            refine_inspect_node(frame, category=cat_i, sn=sn) if frame is not None else ""
        )
        if refined:
            node = refined
            tpl = True
            if refined == "craft_hub":
                hub_cards = bool(getattr(refine_inspect_node, "last_hub_cards", False))
            if refined in _CRAFT_CAT_NODES.values() and frame is not None:
                _n, how = _craft_shell_how(frame, category=cat_i)
                if _n == refined:
                    craft_how = how
                    if how == "mem":
                        tpl = False
                    elif how == "ocr":
                        tpl = False
                        ocr = True
        else:
            if sn == "overlay":
                node = "overlay"
            elif sn.startswith("fine:"):
                node = _cached_facility_node(sn, list(names or [])) or _fine_fallback(sn)
            else:
                node = "craft_shell"
    else:
        node = sn
    xy = None
    if spot and frame is not None:
        xy = locate_spot(node, spot, frame)
    stp_i: int | None = None
    if bool(ready) and node == "main_hud":
        stp_raw = _view_field(v, "stp")
        try:
            stp_i = int(stp_raw) if stp_raw is not None else None
        except (TypeError, ValueError):
            stp_i = None
    return {
        "node": node,
        "xy": xy,
        "stack_names": list(names or []),
        "stack_ready": bool(ready),
        "source": "stack+mem" if (profile_mem and node == profile_mem) or map_mem or craft_mem else _inspect_source(node, sn=sn, tpl=tpl, ocr=ocr),
        "profile_mem": bool(profile_mem and node == profile_mem),
        "craft_mem": bool(craft_mem),
        "map_mem": bool(map_mem),
        "map_view": _selected_map_view(v) if map_mem else None,
        "stack_sampled_ts": _view_field(v, "ui_stack_ts"),
        "stp": stp_i,
        "profile_looked": profile_looked,
        "life_looked": life_looked,
        "craft_how": craft_how,
        "hub_cards": hub_cards,
        "hud_menu_icon": hud_menu_icon,
        "empty_ok": empty_ok,
        "cost": _inspect_cost(t0),
    }


def _stack_sn_from_mem(ms: Any) -> str:
    """内存栈 → 节点；空栈须 empty_ok 才认主界面。"""
    from mm_assist.memory.readers.ui_stack import stack_node

    names = list(getattr(ms, "ui_stack_names", ()) or []) if ms is not None else []
    ready = bool(getattr(ms, "ui_stack_ready", False)) if ms is not None else False
    empty_ok = bool(getattr(ms, "ui_stack_empty_ok", False)) if ms is not None else False
    map_node = _map_node_from_view(ms)
    if map_node:
        return map_node
    return _profile_node_from_view(ms) or _craft_node_from_view(ms) or stack_node(names, ready=ready, empty_ok=empty_ok)


def attach_node_inspect(
    parent: Any,
    *,
    tk_mod: Any,
    get_win: Any,
    grab: Any,
    mark: Any,
    interval_ms: int = 400,
    stop: Any = None,
    post_ui: Any = None,
) -> Any:
    """界面名 + 可点下拉；选中只画红圈。抓屏/认页在后台，不堵状态栏。

    返回 apply_from_mem(ms, snap)，与内存行同拍更新「界面：」。
    """
    ttk = __import__("tkinter.ttk", fromlist=["ttk"])
    if post_ui is None:
        from mm_assist.platform.tk_mailbox import TkMailbox

        mailbox = TkMailbox(parent, stop=stop if stop is not None else threading.Event())
        post_ui = mailbox.post
    row = tk_mod.Frame(parent)
    row.pack(fill="x", padx=8, pady=(0, 4))
    node_var = tk_mod.StringVar(value="界面：—")
    tk_mod.Label(row, textvariable=node_var, font=("Segoe UI", 9)).pack(side="left")
    empty = "（选可点位置）"
    spot_var = tk_mod.StringVar(value=empty)
    box = ttk.Combobox(row, textvariable=spot_var, width=22, state="readonly")
    box.pack(side="left", padx=(12, 0))
    box["values"] = (empty,)
    state = {
        "node": "",
        "spot": "",
        "inflight": False,
        "kick": False,
        "held": None,
        "hud_streak": 0,
        "mark_xy": None,
    }

    def _hide() -> None:
        try:
            mark.hide()
        except Exception:
            pass

    def _stopped() -> bool:
        return bool(stop is not None and getattr(stop, "is_set", lambda: False)())

    def _start_work(*, urgent: bool = False) -> None:
        if _stopped():
            return
        from mm_assist.life_plan.snapshot import snapshot_in_progress

        if snapshot_in_progress():
            return
        if state["inflight"]:
            if urgent:
                state["kick"] = True
            return
        state["inflight"] = True
        state["kick"] = False
        threading.Thread(target=_work, name="node-inspect", daemon=True).start()

    def _apply(payload: dict[str, Any]) -> None:
        state["inflight"] = False
        held, streak = hold_inspect_payload(
            state.get("held") if isinstance(state.get("held"), dict) else None,
            payload,
            hud_streak=int(state.get("hud_streak") or 0),
        )
        state["held"] = held
        state["hud_streak"] = streak
        node = str(held.get("node") or "")
        ready = bool(held.get("stack_ready"))
        stp_raw = held.get("stp")
        try:
            stp_i = int(stp_raw) if stp_raw is not None else None
        except (TypeError, ValueError):
            stp_i = None
        zh = inspect_page_zh(node, stack_ready=ready, stp=stp_i)
        node_var.set("界面：" + zh)
        stack_s = ",".join(str(x) for x in (held.get("stack_names") or []))
        zh_key = (zh, stack_s, ready, node)
        if state.get("zh_log") != zh_key:
            state["zh_log"] = zh_key
            cost = held.get("cost") if isinstance(held.get("cost"), dict) else {}
            extra = ""
            if cost:
                extra = " tpl=%s ocr=%s ms=%.0f" % (
                    cost.get("tpl_n", 0),
                    cost.get("ocr_n", 0),
                    float(cost.get("ms") or 0.0),
                )
            log.info(
                "界面：%s ready=%s node=%s stack=%s%s",
                zh,
                ready,
                node or "-",
                stack_s or "-",
                extra,
            )
        try:
            from mm_assist.life_execute.ui_logic import write_inspect_page

            write_inspect_page(
                node if ready else "",
                zh=zh,
                stack_names=list(held.get("stack_names") or []),
                stack_ready=ready,
                source=str(held.get("source") or ""),
                cost=held.get("cost") if isinstance(held.get("cost"), dict) else None,
            )
        except Exception:
            pass
        names = spots_for_node(node)
        values = (empty,) + names
        try:
            box.configure(values=values)
        except Exception:
            box["values"] = values
        stale_spot = bool(state["spot"] and state["spot"] not in names and state["spot"] != empty)
        if state["node"] != node or stale_spot:
            state["node"] = node
            state["spot"] = ""
            state["mark_xy"] = None
            spot_var.set(empty)
        else:
            state["node"] = node
        win = None
        try:
            win = get_win() if get_win else None
        except Exception:
            win = None
        spot = state["spot"]
        xy = payload.get("xy")
        mark_key = (node, spot)
        if (
            spot
            and spot != empty
            and isinstance(xy, (tuple, list))
            and len(xy) == 2
        ):
            state["mark_xy"] = (mark_key, (int(xy[0]), int(xy[1])))
        elif (
            spot
            and spot != empty
            and isinstance(state.get("mark_xy"), tuple)
            and state["mark_xy"][0] == mark_key
        ):
            xy = state["mark_xy"][1]
        else:
            if not spot or spot == empty:
                state["mark_xy"] = None
        if not spot or spot == empty:
            _hide()
        elif (
            not isinstance(xy, (tuple, list))
            or len(xy) != 2
            or win is None
            or mark is None
        ):
            _hide()
        else:
            try:
                mark.show_client(
                    (int(win.left), int(win.top)), int(xy[0]), int(xy[1])
                )
            except Exception:
                _hide()
        if state["kick"]:
            _start_work()

    def _work() -> None:
        from mm_assist.life_plan.snapshot import heavy_read_lock

        with heavy_read_lock:
            _work_body()

    def _work_body() -> None:
        spot = state["spot"]
        if spot == empty:
            spot = ""
        payload: dict[str, Any] = {
            "node": "",
            "xy": None,
            "stack_names": [],
            "stack_ready": False,
            "source": "pending",
            "stp": None,
        }
        try:
            from mm_assist.life_execute.ui_logic import _stack_from_view
            from mm_assist.memory.readers.ui_stack import stack_node

            view = _view_or_state(None)
            names, ready = _stack_from_view(view) if view is not None else (None, False)
            empty_ok = bool(_view_field(view, "ui_stack_empty_ok")) if view is not None else False
            sn = stack_node(
                list(names or []),
                ready=bool(ready),
                empty_ok=empty_ok,
            )
            need_grab = bool(spot) or (
                bool(ready) and inspect_needs_grab(sn, view=view)
            ) or (bool(ready) and empty_ok and not list(names or []))
            frame = None
            if need_grab:
                try:
                    frame = grab()
                except Exception:
                    frame = None
            if need_grab:
                view_after = _view_or_state(None)
                names_after, _ready_after = (
                    _stack_from_view(view_after)
                    if view_after is not None
                    else (None, False)
                )
                payload = compute_inspect(
                    frame,
                    spot=spot,
                    view=view,
                    stack_after=list(names_after or []),
                )
            else:
                payload = compute_inspect(frame, spot=spot, view=view)
        except Exception:
            payload = {
                "node": "",
                "xy": None,
                "stack_names": [],
                "stack_ready": False,
                "source": "pending",
                "stp": None,
            }
        if _stopped():
            state["inflight"] = False
            return
        try:
            post_ui('inspect', _apply, payload)
        except Exception:
            state["inflight"] = False

    def _schedule() -> None:
        if _stopped():
            return
        _start_work()
        try:
            parent.after(interval_ms, _schedule)
        except Exception:
            pass

    def _on_pick(_evt: Any = None) -> None:
        state["spot"] = str(spot_var.get() or "")
        _start_work(urgent=True)

    box.bind("<<ComboboxSelected>>", _on_pick)

    try:
        import numpy as np
        from mm_assist.perception.ocr import ocr_roi

        ocr_roi(
            np.zeros((24, 48, 3), dtype=np.uint8),
            (0.0, 0.0, 1.0, 1.0),
            min_confidence=0.1,
            backend="cht",
        )
    except Exception:
        pass

    def apply_from_mem(ms: Any, snap: Any) -> None:
        names = list(getattr(ms, "ui_stack_names", ()) or []) if ms is not None else []
        ready = bool(getattr(ms, "ui_stack_ready", False)) if ms is not None else False
        params = str(getattr(snap, "ui_params", "") or "") if snap is not None else ""
        stp_i: int | None = None
        if "stp=1" in params:
            stp_i = 1
        elif "stp=0" in params:
            stp_i = 0
        sn = _stack_sn_from_mem(ms)
        profile_mem = bool(_profile_node_from_view(ms))
        map_mem = bool(_map_node_from_view(ms))
        craft_mem = bool(_craft_node_from_view(ms))
        held_n = str(state.get("node") or "")
        if not (profile_mem or map_mem or craft_mem) and not mem_should_apply_stack(sn, held_n, inflight=bool(state["inflight"])):
            return
        node = sn if ready else ""
        if str(node).startswith("fine:"):
            node = _fine_fallback(node)
        _apply(
            {
                "node": node,
                "xy": None,
                "stack_names": names,
                "stack_ready": ready,
                "source": "stack+mem" if profile_mem or map_mem or craft_mem else "stack",
                "profile_mem": profile_mem,
                "craft_mem": craft_mem,
                "craft_how": "mem" if craft_mem else "",
                "map_mem": map_mem,
                "map_view": _selected_map_view(ms) if map_mem else None,
                "profile_looked": profile_mem,
                "life_looked": profile_mem and node == "life_grid",
                "stack_sampled_ts": getattr(ms, "ui_stack_ts", 0.0),
                "stp": stp_i if node == "main_hud" else None,
            }
        )

    try:
        parent.after(200, _schedule)
    except Exception:
        pass
    return apply_from_mem
