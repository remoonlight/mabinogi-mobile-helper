"""按 ui_transition_review / ui_transition_graph 允许边切换。禁止边不走。"""

from __future__ import annotations

from mm_assist.life_plan.game_stop import operation_monotonic, operation_wall_time

import json
import math
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from mm_assist.config import ROOT
from mm_assist.life_execute.ui_logic import (
    craft_hub_visible,
    graph_node,
    hud_page_pending,
    is_craft_shell,
    observe,
    Observation,
    recognized_click_xy,
    read_inspect_record,
    resolve_navigation,
    skip_click,
    skip_wait_ui_params,
    ui_param_key,
    ui_param_stamps,  # Compatibility export used by diagnostic-file regression tests.
)
from mm_assist.life_plan.types import ExecuteResult

GRAPH_REL = Path("knowledge") / "ui" / "contracts" / "ui_transition_graph.json"
NAV_WAIT_S = 2.0
HUD_CONTROL_WAIT_S = 60.0
INPUT_LAYER_WAIT_S = 60.0
VERIFY_POLL_S = 0.15
VERIFY_TOTAL_S = 6.0
# 与 skill_nearby.MAX_CLICKS / life_click_list「找不到下翻最多 2 次」一致
_ITEM_LIST_SCROLL_MAX = 2
_LAST_ITEM_MISS: dict[str, Any] | None = None
# HUD 背包贴钓鱼/K；HUD 製作可能是协助钮。10h 切页走菜单。
_SKIP_EDGE_IDS = frozenset(
    {"main_hud_bag_inventory", "main_hud_craft_craft_hub"}
)

_EDGE_TEMPLATES: dict[str, Path] = {
    "craft_item_recipe_back_craft_items": Path("knowledge") / "templates" / "facility" / "back.png",
    "craft_items_back_craft_make": Path("knowledge") / "templates" / "facility" / "back.png",
    "cash_shop_close_main_hud": Path("knowledge") / "templates" / "menu" / "icon_close_x.png",
    "overlay_x_main_hud": Path("knowledge") / "templates" / "menu" / "icon_close_x.png",
    "system_menu_role_profile": Path("knowledge") / "templates" / "menu" / "icon_character.png",
    "system_menu_craft_craft_hub": Path("knowledge") / "templates" / "menu" / "icon_crafting.png",
    "system_menu_bag_inventory": Path("knowledge") / "templates" / "menu" / "icon_inventory.png",
    "system_menu_x_main_hud": Path("knowledge") / "templates" / "menu" / "icon_close_x.png",
    "profile_life_tab_life_grid": Path("knowledge") / "templates" / "profile" / "life_tab.png",
    "profile_close_main_hud": Path("knowledge") / "templates" / "profile" / "close.png",
    "life_grid_close_main_hud": Path("knowledge") / "templates" / "profile" / "close.png",
    "gather_detail_close_life_grid": Path("knowledge") / "templates" / "gather_detail" / "close.png",
    "gather_skill_close_life_grid": Path("knowledge") / "templates" / "gather_detail" / "close.png",
    "item_info_back_inventory": Path("knowledge") / "templates" / "menu" / "icon_close_x.png",
    "item_info_close_gather_skill": Path("knowledge") / "templates" / "menu" / "icon_close_x.png",
    "obtain_method_back_item_info": Path("knowledge") / "templates" / "menu" / "icon_close_x.png",
    "facility_back_craft_hub": Path("knowledge") / "templates" / "facility" / "back.png",
    "craft_hub_x_main_hud": Path("knowledge") / "templates" / "craft_hub" / "close_x.png",
    "craft_shell_x_main_hud": Path("knowledge") / "templates" / "craft_hub" / "close_x.png",
    "inventory_close_main_hud": Path("knowledge") / "templates" / "inventory" / "close.png",
    "map_dunbarton_x_main_hud": Path("knowledge") / "templates" / "map" / "close_x.png",
}

_HUD_IDS = {
    "main_hud_menu_system_menu": "menu",
    "main_hud_bag_inventory": "bag",
    "main_hud_craft_craft_hub": "craft",
    "main_hud_map_dunbarton": "map",
}

_EDGE_ROI: dict[str, tuple[float, float, float, float]] = {
    "craft_item_recipe_back_craft_items": (0.0, 0.0, 0.16, 0.16),
    "craft_items_back_craft_make": (0.0, 0.0, 0.16, 0.16),
    "cash_shop_close_main_hud": (0.90, 0.00, 0.10, 0.10),
    "overlay_x_main_hud": (0.88, 0.00, 0.12, 0.12),
    "craft_shell_x_main_hud": (0.88, 0.00, 0.12, 0.12),
    "system_menu_craft_craft_hub": (0.80, 0.12, 0.16, 0.22),
    "system_menu_bag_inventory": (0.72, 0.12, 0.20, 0.20),
    "system_menu_role_profile": (0.66, 0.08, 0.14, 0.28),
    "system_menu_x_main_hud": (0.88, 0.00, 0.12, 0.12),
    "facility_back_craft_hub": (0.00, 0.00, 0.10, 0.12),
    "profile_close_main_hud": (0.88, 0.00, 0.12, 0.12),
    "life_grid_close_main_hud": (0.88, 0.00, 0.12, 0.12),
    "profile_life_tab_life_grid": (0.00, 0.08, 0.20, 0.55),  # 待标定
    "life_grid_skill_gather_detail": (0.04, 0.08, 0.92, 0.82),  # 待标定
    "gather_detail_close_life_grid": (0.88, 0.00, 0.12, 0.12),  # 待标定
    "gather_skill_close_life_grid": (0.88, 0.00, 0.12, 0.12),  # 待标定
    "item_info_back_inventory": (0.88, 0.00, 0.12, 0.12),
    "item_info_close_gather_skill": (0.88, 0.00, 0.12, 0.12),
    "obtain_method_back_item_info": (0.88, 0.00, 0.12, 0.12),
    "craft_hub_x_main_hud": (0.88, 0.00, 0.12, 0.12),  # 待标定
    "inventory_close_main_hud": (0.88, 0.00, 0.12, 0.12),  # 待标定
    "map_dunbarton_x_main_hud": (0.88, 0.00, 0.12, 0.12),  # 待标定
}

_EDGE_MEMO: dict[str, tuple[str, float]] = {}
_EDGE_MEMO_S = 5.0
# Unlike the short duplicate-click memo, dispatched steps survive a timeout.
# A later independent observation can verify just that step, across plans.
_PENDING_EDGES: dict[str, tuple[str, str, float]] = {}
_PAGE_ALIAS = {
    "facility_recipe": "facility",
}

_SKILL_CELL = {
    "daily": Path("knowledge") / "templates" / "life_grid" / "daily.png",
    "herb": Path("knowledge") / "templates" / "life_grid" / "herb.png",
    "woodcut": Path("knowledge") / "templates" / "life_grid" / "woodcut.png",
    "mine": Path("knowledge") / "templates" / "life_grid" / "mine.png",
    "shear": Path("knowledge") / "templates" / "life_grid" / "shear.png",
    "harvest": Path("knowledge") / "templates" / "life_grid" / "harvest.png",
    "hoe": Path("knowledge") / "templates" / "life_grid" / "hoe.png",
    "insect": Path("knowledge") / "templates" / "life_grid" / "insect.png",
    "fish": Path("knowledge") / "templates" / "life_grid" / "fish.png",
}


@lru_cache(maxsize=1)
def _graph(root: str | None = None) -> dict[str, Any]:
    base = Path(root) if root else ROOT
    raw = json.loads((base / GRAPH_REL).read_text(encoding="utf-8"))
    allowed: list[dict[str, Any]] = []
    forbidden: set[tuple[str, str]] = set()
    for edge in raw.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        frm = str(edge.get("from") or "")
        to = str(edge.get("to") or "")
        kind = str(edge.get("kind") or "")
        if kind == "forbidden":
            forbidden.add((frm, to))
            continue
        if kind == "allowed" and frm and to:
            if str(edge.get("id") or "") in _SKIP_EDGE_IDS:
                continue
            allowed.append(edge)
    adj: dict[str, list[dict[str, Any]]] = {}
    for edge in allowed:
        adj.setdefault(str(edge["from"]), []).append(edge)
    return {"allowed": allowed, "forbidden": forbidden, "adj": adj}


def path_edges(src: str, dst: str, *, root: Path | None = None) -> list[dict[str, Any]] | None:
    """最短允许边。禁止边不出现。同源空列表。"""
    from mm_assist.life_execute.ui_logic import graph_node, facility_id_of

    src_n = graph_node(src)
    dst_n = graph_node(dst)
    if src_n == "overlay" and dst_n == "main_hud":
        return [
            {
                "id": "overlay_x_main_hud",
                "from": "overlay",
                "to": "main_hud",
                "kind": "allowed",
                "trigger": {"type": "visible_button", "label": "右上 X", "who": "agent"},
            }
        ]
    src_fac = facility_id_of(src)
    dst_fac = facility_id_of(dst)
    if src_n in {"facility", "facility_recipe"} and dst_n in {"facility", "facility_recipe"}:
        if src_fac and dst_fac and src_fac != dst_fac:
            dst_n = "craft_hub"
        elif src_n == "facility_recipe" and dst_n == "facility":
            return []
    alias = {"currency_wallet": "craft_hub"}
    src = "facility" if src_n == "facility_recipe" else alias.get(src_n, src_n)
    dst = alias.get(dst_n, dst_n)
    if src == dst:
        return []
    g = _graph(str(root or ROOT))
    adj: dict[str, list[dict[str, Any]]] = g["adj"]
    if src not in adj and src != dst:
        if src not in {str(e.get("to")) for e in g["allowed"]}:
            return None
    q = deque([(src, [])])
    seen = {src}
    while q:
        cur, hops = q.popleft()
        for edge in adj.get(cur) or []:
            nxt = str(edge["to"])
            if (cur, nxt) in g["forbidden"] or nxt in seen:
                continue
            seen.add(nxt)
            trail = hops + [edge]
            if nxt == dst:
                return trail
            q.append((nxt, trail))
    return None


def note_edge_clicked(eid: str, expect_to: str, *, stage: str = "") -> None:
    key = str(eid or "").strip()
    if not key:
        return
    sent_at = time.time()
    _EDGE_MEMO[key] = (str(expect_to or ""), sent_at)
    if stage:
        _PENDING_EDGES[stage] = (key, str(expect_to or ""), sent_at)


def edge_recently_clicked(eid: str) -> bool:
    key = str(eid or "").strip()
    rec = _EDGE_MEMO.get(key)
    if rec is None:
        return False
    _expect, ts = rec
    if time.time() - float(ts) > _EDGE_MEMO_S:
        _EDGE_MEMO.pop(key, None)
        return False
    return True


def clear_edge_memo_if_arrived(node: str) -> None:
    g = graph_node(node)
    dead = [
        eid
        for eid, (expect, _ts) in list(_EDGE_MEMO.items())
        if _page_eq(str(expect), g) or str(expect) == str(node)
    ]
    for eid in dead:
        _EDGE_MEMO.pop(eid, None)


def reset_edge_memo_for_tests() -> None:
    global _LAST_ITEM_MISS
    _EDGE_MEMO.clear()
    _PENDING_EDGES.clear()
    _LAST_ITEM_MISS = None


def _page_eq(a: str, b: str) -> bool:
    ga, gb = graph_node(a), graph_node(b)
    if ga == gb:
        return True
    return _PAGE_ALIAS.get(ga, ga) == _PAGE_ALIAS.get(gb, gb)


def _hud_xy(frame: Any, entry_id: str, *, root: Path | None) -> tuple[int, int] | None:
    from mm_assist.perception.hud_entry_locator import locate_hud_entry
    from mm_assist.life_execute.repair_dialog import npc_dialog_visible

    if npc_dialog_visible(frame):
        return None

    cap = json.loads(
        ((root or ROOT) / "knowledge" / "ui" / "contracts" / "main_hud_entry_capture.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next((e for e in cap.get("entries") or [] if e.get("id") == entry_id), None)
    if not isinstance(entry, dict):
        return None
    loc = locate_hud_entry(entry, frame, root=root)
    if loc.get("feature_ok") is not True:
        return None
    return recognized_click_xy(loc)


def _finite_unit(n: object) -> float | None:
    if isinstance(n, bool) or not isinstance(n, (int, float)):
        return None
    try:
        v = float(n)
    except (OverflowError, ValueError, TypeError):
        return None
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        return None
    return v


def _parse_locator_roi(raw: object) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    vals: list[float] = []
    for n in raw:
        v = _finite_unit(n)
        if v is None:
            return None
        vals.append(v)
    x, y, w, h = vals
    if w <= 0.0 or h <= 0.0:
        return None
    if x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
        return None
    return (x, y, w, h)


def _system_menu_roi(edge: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """菜单入口读 graph locator.roi；缺省用旧ROI，显式无效则拒绝。"""
    fallback = _EDGE_ROI[str(edge.get('id') or '')]
    trig = edge.get("trigger")
    if not isinstance(trig, dict):
        return fallback
    if "locator" not in trig:
        return fallback
    loc = trig.get("locator")
    if not isinstance(loc, dict):
        return None
    if "roi" not in loc:
        return fallback
    return _parse_locator_roi(loc.get("roi"))


def _match(
    frame: Any,
    rel: Path,
    *,
    root: Path | None,
    threshold: float = 0.75,
    roi: tuple[float, float, float, float] | None = None,
) -> Any:
    from mm_assist.platform.match import match_template

    path = (root or ROOT) / rel
    if frame is None or not path.is_file():
        return None
    return match_template(frame, path, threshold=threshold, roi=roi)


def locate_edge(
    edge: dict[str, Any],
    frame: Any,
    *,
    root: Path | None = None,
    skill_id: str = "",
    facility: str = "",
    item_name: str = "",
) -> tuple[int, int] | None:
    eid = str(edge.get("id") or "")
    frm = str(edge.get("from") or "")
    to = str(edge.get("to") or "")
    if eid == "overlay_x_main_hud":
        rel = _EDGE_TEMPLATES[eid]
        return recognized_click_xy(
            _match(frame, rel, root=root, threshold=0.72, roi=_EDGE_ROI[eid]),
            min_score=0.72,
        )
    if eid in _HUD_IDS:
        return _hud_xy(frame, _HUD_IDS[eid], root=root)
    if eid in {"system_menu_craft_craft_hub", "system_menu_role_profile", "system_menu_bag_inventory"}:
        # The legacy by_label helpers also accept weak templates and fixed grid
        # positions. During menu animation those can point into an empty frame.
        # Wait for the actual icon instead; _try_click_edge bounds this wait.
        rel = _EDGE_TEMPLATES[eid]
        roi = _EDGE_ROI[eid]
        if eid in {"system_menu_craft_craft_hub", "system_menu_bag_inventory"}:
            roi = _system_menu_roi(edge)
            if roi is None:
                return None
        threshold = 0.75 if eid == 'system_menu_bag_inventory' else 0.72
        return recognized_click_xy(_match(frame, rel, root=root, threshold=threshold, roi=roi))
    if eid == "facility_back_craft_hub":
        rel = _EDGE_TEMPLATES.get(eid)
        xy = recognized_click_xy(
            _match(frame, rel, root=root, threshold=0.50, roi=_EDGE_ROI.get(eid)),
            min_score=0.50,
        )
        if xy is not None:
            return xy
        return recognized_click_xy(
            _match(
                frame,
                Path("knowledge") / "templates" / "menu" / "icon_close_x.png",
                root=root,
                threshold=0.72,
                roi=(0.88, 0.00, 0.12, 0.10),
            ),
            min_score=0.72,
        )
    if eid in {"profile_close_main_hud", "life_grid_close_main_hud"}:
        rel = _EDGE_TEMPLATES.get(eid)
        xy = recognized_click_xy(
            _match(frame, rel, root=root, threshold=0.50, roi=_EDGE_ROI.get(eid)),
            min_score=0.50,
        )
        if xy is not None:
            return xy
        return recognized_click_xy(
            _match(
                frame,
                Path("knowledge") / "templates" / "menu" / "icon_close_x.png",
                root=root,
                threshold=0.55,
                roi=(0.88, 0.00, 0.12, 0.12),
            ),
            min_score=0.55,
        )
    if eid == "life_grid_skill_gather_detail":
        rel = _SKILL_CELL.get(str(skill_id or "").strip())
        roi = _EDGE_ROI.get(eid)
        if rel is None or roi is None:
            return None
        xy = recognized_click_xy(
            _match(
                frame,
                rel,
                root=root,
                threshold=0.52,
                roi=roi,
            ),
            min_score=0.52,
        )
        if xy is not None:
            return xy
        return None
    if eid == "gather_skill_item_gather_detail" or eid == "gather_skill_item_item_info" or (
        frm == "gather_skill" and to in {"gather_detail", "item_info"}
    ):
        from mm_assist.perception.item_sense import locate_skill_item

        global _LAST_ITEM_MISS
        hit = locate_skill_item(frame, skill_id=skill_id, name=item_name, root=root)
        xy = hit.get("click_client") if hit.get("ok") else None
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            _LAST_ITEM_MISS = None
            return int(xy[0]), int(xy[1])
        _LAST_ITEM_MISS = hit if isinstance(hit, dict) else None
        return None
    if eid == "inventory_item_item_info" or (frm == "inventory" and to == "item_info"):
        from mm_assist.life_plan.node_inspect import locate_spot

        return locate_spot("inventory", item_name or "樹液", frame, root=root)
    if eid == "facility_item_item_info" or (
        frm in {"facility", "facility_recipe"} and to == "item_info"
    ):
        from mm_assist.life_plan.node_inspect import locate_spot

        fid = str(facility or skill_id or "wood").strip() or "wood"
        for node in (
            f"facility:{fid}:高級木材",
            f"facility:{fid}",
            "facility",
            "facility_recipe",
        ):
            xy = locate_spot(node, item_name or "樹液", frame, root=root)
            if xy is not None:
                return xy
        return None
    if eid in {"item_info_open_obtain_method", "gather_detail_open_obtain_method"} or (
        frm in {"item_info", "gather_detail"} and to == "obtain_method"
    ):
        from mm_assist.life_plan.node_inspect import locate_spot

        xy = locate_spot("item_info", "获得方法", frame, root=root)
        if xy is not None:
            return xy
        return locate_spot("gather_detail", "获得方法", frame, root=root)
    if eid == "facility_open_recipe" or (frm == "facility" and to == "facility_recipe"):
        from mm_assist.perception.process_sense import locate_process_recipe, recipe_click_variant

        recipe = str((edge.get("trigger") or {}).get("label") or "").strip()
        if not recipe or recipe in {"配方名", "可见按钮"}:
            recipe = item_name
        hit = locate_process_recipe(
            frame,
            facility=facility,
            name=recipe,
            variant=recipe_click_variant(recipe),
            root=root,
        )
        xy = hit.get("click_client") if hit.get("ok") else None
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            return int(xy[0]), int(xy[1])
        return None
    lab = str((edge.get("trigger") or {}).get("label") or "")
    if lab in {"加工", "快速制作", "制作", "炼金术"} and (
        frm.startswith("craft_") or to.startswith("craft_")
    ):
        from mm_assist.life_plan.node_inspect import craft_tab_xy

        return craft_tab_xy(frame, lab)
    if eid == "craft_hub_card_facility" or (frm == "craft_hub" and to == "facility"):
        from mm_assist.perception.process_sense import locate_hub_card

        return recognized_click_xy(locate_hub_card(frame, facility, root=root))
    rel = _EDGE_TEMPLATES.get(eid)
    if rel is None:
        return None
    roi = _EDGE_ROI.get(eid)
    if roi is None:
        return None
    return recognized_click_xy(
        _match(frame, rel, root=root, threshold=0.75, roi=roi)
    )


def _is_skill_item_edge(edge: dict[str, Any]) -> bool:
    eid = str(edge.get("id") or "")
    frm = str(edge.get("from") or "")
    to = str(edge.get("to") or "")
    return eid in {"gather_skill_item_gather_detail", "gather_skill_item_item_info"} or (
        frm == "gather_skill" and to in {"gather_detail", "item_info"}
    )


def _is_sap_nav(item_name: str) -> bool:
    from mm_assist.perception.item_sense import fold_zh

    return fold_zh(item_name) == fold_zh("樹液")


def _sap_item_edge(frm: str) -> dict[str, Any]:
    return {
        "id": "facility_item_item_info" if frm != "inventory" else "inventory_item_item_info",
        "from": frm,
        "to": "item_info",
        "trigger": {"type": "visible_button", "label": "樹液"},
    }


def _sap_recipe_edge() -> dict[str, Any]:
    return {
        "id": "facility_open_recipe",
        "from": "facility",
        "to": "facility_recipe",
        "trigger": {"type": "visible_button", "label": "高級木材"},
    }


def _sap_hops(
    node: str,
    target: str,
    *,
    root: Path | None,
    frame: Any | None = None,
) -> list[dict[str, Any]] | None:
    """樹液第1次：菜单→製作→加工→木材→高級木材→樹液卡。仅背包已能 locate 才走物品。"""
    from mm_assist.life_execute.ui_logic import facility_id_of

    g = graph_node(node)
    t = graph_node(target)
    if t not in {"item_info", "obtain_method"}:
        return None
    if t == "obtain_method" and g == "item_info" and "樹液" in str(node):
        return path_edges(node, "obtain_method", root=root)
    if t == "item_info" and g == "obtain_method":
        return path_edges(node, "item_info", root=root)
    if g == "item_info" and "樹液" not in str(node):
        return path_edges(node, "life_grid", root=root) or path_edges(node, "craft_hub", root=root)
    if g == "item_info":
        return []
    if g in {"gather_skill", "gather_detail"}:
        return path_edges(node, "life_grid", root=root)
    if g in {"life_grid", "profile"}:
        return path_edges(node, "main_hud", root=root)
    if g == "inventory":
        xy = None
        if frame is not None:
            from mm_assist.life_plan.node_inspect import locate_spot

            xy = locate_spot("inventory", "樹液", frame, root=root)
        if xy is not None:
            return [_sap_item_edge("inventory")]
        return path_edges(node, "facility", root=root)
    fid = facility_id_of(node)
    if g == "facility_recipe":
        if fid and fid != "wood":
            return path_edges(node, "craft_hub", root=root)
        if "高級木材" in str(node) or not fid or fid == "wood":
            return [_sap_item_edge(g)]
        return path_edges(node, "craft_hub", root=root)
    if g == "facility":
        if fid and fid != "wood":
            return path_edges(node, "craft_hub", root=root)
        return [_sap_recipe_edge()]
    hops = path_edges(node, "facility", root=root)
    return hops


def _production_item_list_drag(x0: int, y0: int, x1: int, y1: int) -> None:
    from mm_assist.config import load_config
    from mm_assist.platform.window import find_window
    from mm_assist.safety.input import drag_client

    cfg = load_config()
    win = find_window(cfg.window.title_substr) or find_window("Mobile") or find_window("瑪奇")
    if win is None:
        return
    if not drag_client(win, int(x0), int(y0), int(x1), int(y1), end_hold_s=0.4):
        raise RuntimeError('drag_rejected: 翻页输入未发出')


def edge_stage(edge: dict[str, Any], *, label: str = "") -> str:
    """换页用边 id；投料/领取/清界面保留中文。边 id 不上状态栏。"""
    lab = str(label or (edge.get("trigger") or {}).get("label") or "")
    eid = str(edge.get("id") or "")
    if "投料" in lab:
        return "投料"
    if "領取" in lab or "领取" in lab:
        return "领取"
    if any(k in lab for k in ("关闭", "退出", "返回", "右上")):
        return "清界面"
    if eid:
        return eid
    from mm_assist.life_execute.ui_logic import is_open_menu_action

    if is_open_menu_action(lab) or "打开菜单" in lab:
        return "进入菜单"
    return lab or "切页"


def _mark_arrived(stage: str) -> None:
    from mm_assist.safety.autofarm_gates import apply_stage_verify

    apply_stage_verify(stage, True)
    _PENDING_EDGES.pop(stage, None)


def _verify_late_edges(where: Any, *, frame_started_at: float) -> None:
    """Only a new frame plus a ready, post-input stack verifies a prior step."""
    if not getattr(where, "stack_ready", False):
        return
    from mm_assist.memory.readers.ui_stack import input_blocking_layers

    if input_blocking_layers(getattr(where, "stack_names", ())):
        return
    sampled = float(getattr(where, "stack_sampled_ts", 0.0) or 0.0)
    now = time.time()
    if sampled <= 0 or not 0 <= now - sampled <= 16.0:
        return
    node = str(getattr(where, "node", "") or "")
    if graph_node(node) in {"unknown", "overlay", ""}:
        return
    from mm_assist.life_execute.skill_list import verify_closed_list

    verify_closed_list(where, frame_started_at=frame_started_at)
    for stage, (eid, expect, sent_at) in list(_PENDING_EDGES.items()):
        if frame_started_at <= sent_at or sampled <= sent_at:
            continue
        if graph_node(node) == graph_node(expect):
            _mark_arrived(stage)
            _EDGE_MEMO.pop(eid, None)


def _nav_node(where: Any) -> str:
    """menu=1 时「界面：」若仍写主界面，按菜单层走，禁止再点 HUD 菜单。"""
    names = list(getattr(where, "stack_names", ()) or ())
    if any("DungeonList" in n or "DungeonEntry" in n for n in names):
        return "overlay"
    node = graph_node(str(getattr(where, "node", "") or ""))
    if (node == 'facility' and getattr(where, 'stack_ready', False)
            and 'AlteringPanelPopupUIInstance' in names):
        # Recipe text can be unrecognized while its independent popup stack
        # is known. Closing that popup is a verified step, not an unchanged
        # facility. Keep the recipe name unknown rather than inventing one.
        node = 'facility_recipe'
    try:
        menu = int(getattr(where, "menu")) if getattr(where, "menu", None) is not None else None
    except (TypeError, ValueError):
        menu = None
    if menu == 1 and node == "main_hud":
        return "system_menu"
    return node


def ensure_node(
    target: str,
    *,
    grab: Callable[[], Any],
    click: Callable[[int, int], bool],
    on_progress: Callable[[str], None] | None = None,
    root: Path | None = None,
    skill_id: str = "",
    facility: str = "",
    item_name: str = "",
    already_ok: Callable[[Any], bool] | None = None,
    leave_wrong_page: Callable[[Any], bool] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    resolve: Callable[..., Any] | None = None,
    drag_fn: Callable[[int, int, int, int], Any] | None = None,
) -> ExecuteResult:
    """从当前任意已知页沿允许边走到 target。单击后即时确认；认不到不点。"""
    sleeper = sleep_fn or __import__("time").sleep
    resolver = resolve or resolve_navigation
    from mm_assist.perception.item_sense import GATHER_SKILL_IDS, fold_zh

    if _is_sap_nav(item_name):
        if skill_id and skill_id not in GATHER_SKILL_IDS:
            if not facility:
                facility = skill_id
            skill_id = ""
        elif fold_zh(skill_id) == fold_zh("woodcut"):
            skill_id = ""
            facility = facility or "wood"

    def _prog(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    def _path_dst() -> str:
        t = graph_node(target)
        if t == "facility_recipe" and facility:
            return f"facility:{facility}:{item_name}" if item_name else f"facility:{facility}"
        if t == "facility" and facility:
            return f"facility:{facility}"
        if t == "item_info" and item_name:
            return f"item_info:{item_name}"
        if t == "obtain_method" and item_name:
            return f"obtain_method:{item_name}"
        if t == "gather_detail" and item_name:
            return f"gather_detail:{item_name}"
        return target

    def _cancelled() -> ExecuteResult | None:
        try:
            from mm_assist.life_execute.ui_act import pump_aborted

            if pump_aborted():
                return ExecuteResult(ok=False, status="cancel", message="已停止操作")
        except Exception:
            pass
        return None

    def _sleep(sec: float) -> ExecuteResult | None:
        hit = _cancelled()
        if hit is not None:
            return hit
        if sleep_fn is not None:
            sleeper(float(sec))
            return _cancelled()
        end = operation_wall_time() + float(sec)
        while operation_wall_time() < end:
            hit = _cancelled()
            if hit is not None:
                return hit
            remaining = end - operation_wall_time()
            if remaining <= 0:
                break
            sleeper(min(0.15, remaining))
        return _cancelled()

    def _wrong_skill(where: Any) -> ExecuteResult | None:
        node = str(getattr(where, "node", "") or "")
        t = graph_node(target)
        skill_pages = {"gather_skill", "item_info", "gather_detail", "obtain_method"}
        if t == "gather_skill" and skill_id and node == "gather_skill":
            return ExecuteResult(False, "unknown_skill", "尚未确认技能名称，不把列表当作目标已到达")
        if t in skill_pages and skill_id and node.startswith("gather_skill:"):
            actual = node.split(":", 1)[1]
            if actual != skill_id:
                return ExecuteResult(False, "wrong_skill", f"当前技能为{actual}，目标为{skill_id}；不在错误列表点击")
        return None

    def _item_token(node: str) -> bool:
        if not item_name:
            return True
        parts = str(node or "").split(":")
        if len(parts) < 2:
            return False
        want = fold_zh(item_name)
        return any(fold_zh(p) == want for p in parts[1:])

    def _wrong_named_detail(full: str) -> bool:
        t = graph_node(target)
        g = graph_node(full)
        if not item_name or t not in {"item_info", "gather_detail", "obtain_method"}:
            return False
        if g not in {"gather_detail", "item_info", "obtain_method"}:
            return False
        if ":" not in full:
            return True
        return not _item_token(full)

    def _at_dest(node: str, frame: Any, where: Any = None) -> bool:
        from mm_assist.life_execute.ui_logic import facility_id_of

        g = graph_node(node)
        t = graph_node(target)
        if t == "craft_hub" and g != "craft_hub":
            return False
        if t == "gather_detail":
            return g in {"gather_detail", "item_info"} and _item_token(node)
        if t == "item_info":
            return g in {"item_info", "gather_detail"} and _item_token(node)
        if t == "obtain_method":
            return g == "obtain_method" and _item_token(node)
        if t == "facility_recipe":
            return g == "facility_recipe" and _item_token(node) and (
                not facility or facility_id_of(node) == facility
            )
        if t == "facility" and g in {"facility", "facility_recipe"}:
            if already_ok is not None:
                pass
            elif facility:
                return facility_id_of(node) == facility
            else:
                return True
        if g != t:
            return False
        if already_ok is None:
            if t == "craft_hub":
                if (getattr(where, "source", "") == "direct+mem"
                        and getattr(where, "stack_ready", False)
                        and -0.05 <= time.time() - float(getattr(where, "stack_sampled_ts", 0) or 0) <= 1.5):
                    return True
                return bool(craft_hub_visible(frame, root=root))
            return True
        if t == "facility" and g not in {"facility", "facility_recipe"}:
            return False
        if t == "gather_detail" and g not in {"gather_detail", "gather_skill"}:
            return False
        return bool(already_ok(frame))

    def _judge_page(
        nxt_node: str, after: Any, edge: dict[str, Any], from_node: str, stage: str,
        before_where: Any = None, after_where: Any = None,
    ) -> ExecuteResult | None | bool:
        """True=继续轮询；ExecuteResult=结束；None=到了边终点、外层接下一条。"""
        expect_to = str(edge.get("to") or "")
        # A detail/ingredient popup may cover a recipe rather than a skill list.
        # Verify the actual top layer disappeared before choosing the next path.
        if graph_node(from_node) in {"item_info", "gather_detail", "obtain_method"}:
            before_names = tuple(getattr(before_where, "stack_names", ()) or ())
            after_names = tuple(getattr(after_where, "stack_names", ()) or ())
            layer = next((name for name in ("IngredientGuideUIInstance", "DetailCardUIInstance")
                          if name in before_names), None)
            if (layer and getattr(before_where, "stack_ready", False)
                    and getattr(after_where, "stack_ready", False)
                    and layer not in after_names
                    and graph_node(nxt_node) in {"item_info", "gather_detail", "inventory",
                        "gather_skill", "life_grid", "facility", "facility_recipe", "main_hud",
                        "craft_item_recipe"}
                    and stage == "清界面"):
                _mark_arrived(stage)
                _EDGE_MEMO.pop(str(edge.get("id") or ""), None)
                return None
        if (graph_node(from_node) == 'craft_item_recipe'
                and graph_node(nxt_node) == 'craft_make'
                and getattr(after_where, 'stack_ready', False)
                and 'CraftingPanelPopupUIInstance' not in getattr(after_where, 'stack_names', ())):
            # The global back control can close the entire child items window.
            # Only the independently recognized make page proves this outcome.
            _mark_arrived(stage)
            _EDGE_MEMO.pop(str(edge.get('id') or ''), None)
            return None
        if target == "craft_hub" and nxt_node == "currency_wallet":
            return ExecuteResult(
                ok=False,
                status="wrong_page",
                message="点製作进了货币页，不是加工六卡",
            )
        if _at_dest(nxt_node, after, after_where):
            _mark_arrived(stage)
            clear_edge_memo_if_arrived(nxt_node)
            return ExecuteResult(ok=True, status="at_node", message=target)
        detail_pages = {"item_info", "gather_detail"}
        if graph_node(expect_to) in detail_pages and graph_node(nxt_node) in detail_pages:
            _mark_arrived(stage)
            clear_edge_memo_if_arrived(nxt_node)
            return None
        if _page_eq(nxt_node, expect_to):
            _mark_arrived(stage)
            clear_edge_memo_if_arrived(nxt_node)
            return None
        if graph_node(expect_to) == "life_grid" and graph_node(nxt_node) == "gather_skill":
            _mark_arrived(stage)
            clear_edge_memo_if_arrived(nxt_node)
            return None
        if (
            graph_node(from_node) == "facility_recipe"
            and graph_node(expect_to) == "craft_hub"
            and graph_node(nxt_node) == "facility"
        ):
            # 配方页左上←先关配方回到设施；外层再←出枢纽。勿与设施页判成同一页而空等。
            _mark_arrived(stage)
            _EDGE_MEMO.pop(str(edge.get("id") or ""), None)
            return None
        if (
            graph_node(from_node) == "facility"
            and graph_node(expect_to) == "craft_hub"
            and graph_node(nxt_node) == "main_hud"
        ):
            # 布料等设施页无左上←，回退点右上 X 会关到主界面；外层从 HUD 再进枢纽。
            _mark_arrived(stage)
            _EDGE_MEMO.pop(str(edge.get("id") or ""), None)
            return None
        if _page_eq(nxt_node, from_node):
            return True
        if (graph_node(expect_to) == "craft_hub" and is_craft_shell(nxt_node)
                and not is_craft_shell(from_node)):
            # 点製作常停在快速制作；算进了壳，外层再点「加工」。勿判错页去退回主界面。
            _mark_arrived(stage)
            clear_edge_memo_if_arrived(nxt_node)
            return None
        if (
            is_craft_shell(from_node)
            and is_craft_shell(expect_to)
            and is_craft_shell(nxt_node)
        ):
            # 壳内四 Tab 互切：滞后内存可能先报另一壳 Tab，继续验到目标或超时。
            return True
        return ExecuteResult(
            ok=False,
            status="wrong_page",
            message=f"点后到了{nxt_node}，不是{expect_to}",
        )

    def verify_arrived(
        edge: dict[str, Any], from_node: str, stage: str, label: str, before_where: Any = None
    ) -> ExecuteResult | None:
        _ = label
        elapsed = 0.0
        _prog("等新页")
        if sleep_fn is None and resolve is not None:
            from mm_assist.life_execute.ui_logic import wait_ui_params_refresh

            if not wait_ui_params_refresh(
                root=root, cancelled=lambda: _cancelled() is not None
            ):
                hit = _cancelled()
                if hit is not None:
                    return hit
                return ExecuteResult(
                    ok=False,
                    status="stale_ui",
                    message="点后界面参数未刷新，不继续",
                )
            elapsed = NAV_WAIT_S
        elif sleep_fn is not None:
            hit = _sleep(NAV_WAIT_S)
            if hit is not None:
                return hit
            elapsed = NAV_WAIT_S
        while True:
            started_read = operation_monotonic()
            after = grab()
            nxt_where = resolver(after)
            mismatch = _wrong_skill(nxt_where)
            if mismatch is not None:
                if mismatch.status == "wrong_skill":
                    return None
                return mismatch
            nxt_node = _nav_node(nxt_where)
            nxt_full = str(getattr(nxt_where, "node", "") or "") or nxt_node
            if sleep_fn is None:
                elapsed += operation_monotonic() - started_read
            pending = nxt_node == "unknown" or (nxt_node == "profile" and graph_node(str(edge.get("to"))) in {"life_grid", "gather_skill"})
            judged = True if pending else _judge_page(nxt_full, after, edge, from_node, stage,
                                                      before_where, nxt_where)
            if judged is not True:
                return judged
            if elapsed >= VERIFY_TOTAL_S:
                return ExecuteResult(
                    ok=False,
                    status="nav_verify_timeout",
                    message=f"未及时确认{edge.get('to')}（最后观测{nxt_node}），不再点",
                )
            step = min(VERIFY_POLL_S, VERIFY_TOTAL_S - elapsed)
            hit = _sleep(step)
            if hit is not None:
                return hit
            elapsed += step

    def _try_click_edge(
        edge: dict[str, Any],
        frame: Any,
        node: str,
        label: str,
        *,
        stamps: tuple[float, float] | None,
        param_key: tuple[str, str] | None,
        before_where: Any = None,
    ) -> ExecuteResult | None | bool:
        """True=四参内容变了重观测；ExecuteResult=结束；None=验页后继续下一条边。"""
        _ = stamps
        eid = str(edge.get("id") or "")
        if eid and edge_recently_clicked(eid):
            return ExecuteResult(
                ok=False,
                status="nav_no_move",
                message=f"已点过{label}，认页前不再点",
            )
        xy = locate_edge(edge, frame, root=root, skill_id=skill_id, facility=facility, item_name=item_name)
        scrolled = False
        if xy is None and _is_skill_item_edge(edge):
            miss = _LAST_ITEM_MISS
            plan = miss.get("scroll_plan") if isinstance(miss, dict) else None
            clipped = isinstance(miss, dict) and (
                miss.get("reason") == "clipped_row" or miss.get("error") == "clipped_row"
            )
            can_scroll = isinstance(plan, dict) and plan.get("direction") == "down"
            if can_scroll or clipped:
                from mm_assist.perception.item_sense import item_list_scroll_drag_points

                scrolled = True
                n_drag = 0
                while n_drag < _ITEM_LIST_SCROLL_MAX:
                    miss = _LAST_ITEM_MISS
                    plan = miss.get("scroll_plan") if isinstance(miss, dict) else None
                    clipped = isinstance(miss, dict) and (
                        miss.get("reason") == "clipped_row" or miss.get("error") == "clipped_row"
                    )
                    can_scroll = isinstance(plan, dict) and plan.get("direction") == "down"
                    if not (can_scroll or clipped):
                        break
                    if frame is None or getattr(frame, "size", 1) == 0:
                        break
                    try:
                        (x0, y0), (x1, y1) = item_list_scroll_drag_points(frame, step=n_drag)
                    except (AttributeError, TypeError, ValueError):
                        break
                    _prog("下翻详情物")
                    if drag_fn is not None:
                        drag_fn(int(x0), int(y0), int(x1), int(y1))
                    elif sleep_fn is None:
                        _production_item_list_drag(int(x0), int(y0), int(x1), int(y1))
                    n_drag += 1
                    hit = _sleep(0.6)
                    if hit is not None:
                        return hit
                    frame = grab()
                    current = resolver(frame)
                    if _nav_node(current) != node:
                        return True
                    xy = locate_edge(
                        edge, frame, root=root, skill_id=skill_id, facility=facility, item_name=item_name
                    )
                    if xy is not None:
                        break
                if xy is None:
                    miss = _LAST_ITEM_MISS
                    if isinstance(miss, dict) and (
                        miss.get("reason") == "clipped_row" or miss.get("error") == "clipped_row"
                    ):
                        return ExecuteResult(
                            ok=False,
                            status="clipped_row",
                            message="列表行被裁切，不点击",
                        )
        # The stack can arrive before its controls finish animating. Reobserve,
        # never repeat an input or reuse a coordinate from the preceding page.
        # Crossing regions can leave a ready empty stack throughout loading.
        # A missing HUD control is a read-only readiness wait, not a failed
        # click. Keep ordinary popup animation budgets and input gates intact.
        control_wait = HUD_CONTROL_WAIT_S if node == 'main_hud' and eid in _HUD_IDS else 2.0
        control_deadline = operation_monotonic() + control_wait
        control_elapsed = 0.0
        if xy is None and control_wait == HUD_CONTROL_WAIT_S:
            _prog('等待页面就绪：主界面控件尚未出现')
        # prepare_to passes the live resolver explicitly; it must not disable
        # HUD readiness. Injected simulation resolvers keep ordinary controls'
        # existing behavior. Count simulated sleeps without a real minute wait.
        wait_controls = resolve is None or control_wait == HUD_CONTROL_WAIT_S
        while (xy is None and not scrolled and wait_controls
               and operation_monotonic() < control_deadline and control_elapsed < control_wait):
            hit = _sleep(VERIFY_POLL_S)
            if hit is not None:
                return hit
            if sleep_fn is not None:
                control_elapsed += VERIFY_POLL_S
            frame = grab()
            current = resolver(frame)
            if _nav_node(current) != node:
                return True
            xy = locate_edge(edge, frame, root=root, skill_id=skill_id, facility=facility, item_name=item_name)
        if xy is None:
            return ExecuteResult(ok=False, status=label, message=f"找不到{label}")
        if resolve is None and node == "system_menu":
            # The menu slides horizontally. A valid icon can move into another
            # cell during evidence capture; require stable observed coordinates.
            stable_deadline = operation_monotonic() + 2.0
            while operation_monotonic() < stable_deadline:
                hit = _sleep(0.08)
                if hit is not None:
                    return hit
                frame = grab()
                if _nav_node(resolver(frame)) != node:
                    return True
                latest_xy = locate_edge(edge, frame, root=root, skill_id=skill_id, facility=facility, item_name=item_name)
                stable = (latest_xy is not None and xy is not None
                    and max(abs(latest_xy[0] - xy[0]), abs(latest_xy[1] - xy[1])) <= 4)
                xy = latest_xy
                if stable:
                    break
            else:
                return ExecuteResult(False, "control_unstable", f"{label}仍在移动，不点击")
        if resolve is not None and param_key is not None and not skip_wait_ui_params():
            if ui_param_key(root=root) != param_key:
                return True
        stage = edge_stage(edge, label=label)
        from mm_assist.safety.autofarm_gates import stage_click, _BEFORE_DOWN_ERRORS

        hit = _cancelled()
        if hit is not None:
            return hit
        try:
            sent = stage_click(stage,click,xy[0],xy[1])
        except Exception as exc:
            if eid and str(exc).split(':', 1)[0] not in _BEFORE_DOWN_ERRORS:
                # The stage reservation survives an ambiguous dispatch error.
                # Keep its destination too, so later independent evidence can
                # verify it without another click.
                note_edge_clicked(eid, str(edge.get('to') or ''), stage=stage)
            raise
        if sent is None:
            return ExecuteResult(False,"need_verify",f"{stage}已点两次，需先验画面")
        if not sent:
            hit = _cancelled()
            if hit is not None:
                return hit
            return ExecuteResult(ok=False, status="click_yield", message=f"{label}未点出")
        if eid:
            note_edge_clicked(eid, str(edge.get("to") or ""), stage=stage)
        return verify_arrived(edge, node, stage, label, before_where)

    last_stage = ""

    for _ in range(10):
        hit = _cancelled()
        if hit is not None:
            return hit
        frame_started_at = time.time()
        frame = grab()
        obs = (observe(frame=frame, root=root, resolve=resolver) if resolve is not None
               else Observation(where=resolver(frame), stamps=(0.0, 0.0), frame=frame))
        from mm_assist.memory.readers.ui_stack import (
            input_blocking_layers, facility_travel_popup_has_only_own_shield,
        )

        blocked_elapsed = 0.0
        while input_blocking_layers(getattr(obs.where, "stack_names", ())):
            if facility_travel_popup_has_only_own_shield(getattr(obs.where, "stack_names", ())):
                break  # Handle the foreground popup before waiting for covered controls.
            _prog("等待系统弹层退出，不点底层")
            if blocked_elapsed >= INPUT_LAYER_WAIT_S:
                return ExecuteResult(False, "click_ui_blocked", "系统弹层未退出，不点底层")
            started_wait = operation_monotonic()
            hit = _sleep(VERIFY_POLL_S)
            if hit is not None:
                return hit
            frame_started_at = time.time()
            frame = grab()
            obs = Observation(where=resolver(frame), stamps=(0.0, 0.0), frame=frame)
            blocked_elapsed += (VERIFY_POLL_S if sleep_fn is not None else operation_monotonic() - started_wait)
        pending_elapsed = 0.0
        while resolve is None and _nav_node(obs.where) == "unknown" and pending_elapsed < VERIFY_TOTAL_S:
            started_wait = operation_monotonic()
            hit = _sleep(VERIFY_POLL_S)
            if hit is not None:
                return hit
            frame_started_at = time.time()
            frame = grab()
            obs = Observation(where=resolver(frame), stamps=(0.0, 0.0), frame=frame)
            pending_elapsed += (VERIFY_POLL_S if sleep_fn is not None else operation_monotonic() - started_wait)
        where = obs.where
        if hud_page_pending(where):
            return ExecuteResult(False, "input_layer_wait", "等待HUD页面就绪，不关闭未识别画面")
        _verify_late_edges(where, frame_started_at=frame_started_at)
        names = list(getattr(where, "stack_names", ()) or ())
        from mm_assist.life_execute.ui_act import (
            confirm_go_facility_popup,
            dismiss_facility_result,
            stack_has_common_popup,
            stack_has_facility_result,
        )

        if stack_has_common_popup(names=names or None):
            go = confirm_go_facility_popup(
                click, grab, on_progress=on_progress, root=root, names=names or None
            )
            if go is not None:
                return go
            continue
        if stack_has_facility_result():
            dismissed = dismiss_facility_result(click, grab, on_progress=on_progress, root=root)
            if dismissed is not None:
                return dismissed
            continue
        mismatch = _wrong_skill(where)
        if mismatch is not None:
            if mismatch.status == "wrong_skill":
                # Close the identified skill popup and verify the grid before
                # selecting another skill. Never click an item in the wrong list.
                back = ensure_node("life_grid", grab=grab, click=click,
                    on_progress=on_progress, root=root, sleep_fn=sleep_fn, resolve=resolve)
                if not back.ok:
                    return back
                continue
            return mismatch
        node = _nav_node(where)
        full_node = str(getattr(where, "node", "") or "") or node
        if _at_dest(full_node, frame, where):
            _mark_arrived(last_stage)
            clear_edge_memo_if_arrived(node)
            return ExecuteResult(ok=True, status="at_node", message=target)
        if node == target:
            if already_ok is None and graph_node(target) == "craft_hub":
                # The same frame/where just failed _at_dest above. Do not
                # repeat recognition or treat rejection as permission to click.
                return ExecuteResult(
                    ok=False,
                    status="wrong_tab",
                    message="不是加工档",
                )
            if already_ok is None:
                from mm_assist.life_execute.ui_logic import facility_id_of as _fid

                fac_mismatch = (
                    graph_node(target) == "facility"
                    and bool(facility)
                    and _fid(full_node) != facility
                )
                if not fac_mismatch and not (
                    item_name and graph_node(target) in {"gather_detail", "facility_recipe", "item_info", "obtain_method"}
                ):
                    _mark_arrived(last_stage)
                    clear_edge_memo_if_arrived(node)
                    return ExecuteResult(ok=True, status="at_node", message=target)
            parent = {"facility": "craft_hub", "gather_detail": "life_grid", "facility_recipe": "craft_hub", "item_info": "gather_skill", "obtain_method": "item_info"}.get(target)
            if parent is None:
                _mark_arrived(last_stage)
                clear_edge_memo_if_arrived(node)
                return ExecuteResult(ok=True, status="at_node", message=target)
            if graph_node(node) == "facility" and graph_node(target) == "facility":
                from mm_assist.life_execute.ui_logic import facility_id_of as _fid2

                seen_facility = _fid2(full_node)
                known_wrong = bool(facility and seen_facility) and seen_facility != facility
                if not known_wrong and (leave_wrong_page is None or not leave_wrong_page(frame)):
                    return ExecuteResult(
                        ok=False,
                        status="need_verify",
                        message="设施身份未确认，不退页",
                    )
            hops = path_edges(node, parent, root=root)
            if not hops:
                return ExecuteResult(ok=False, status="wrong_page", message=f"已在{target}但不是目标页")
            edge = hops[0]
            label = str((edge.get("trigger") or {}).get("label") or edge.get("id") or "切界面")
            _prog(f"离开错误页 {label}")
            moved = _try_click_edge(
                edge, frame, node, label, stamps=obs.stamps, param_key=ui_param_key(root=root) if resolve is not None else None,
                before_where=where,
            )
            if moved is True:
                continue
            if moved is not None:
                return moved
            continue
        if node == "unknown" and resolve is not None:
            rec = read_inspect_record(root=root)
            zh = str((rec or {}).get("zh") or "")
            if "主界面-待机" in zh or "主界面-动作中" in zh:
                node = "main_hud"
            elif "菜单" in zh:
                node = "system_menu"
        if node in {"unknown", "dialog"}:
            bits = where.bits() if callable(getattr(where, "bits", None)) else ""
            return ExecuteResult(
                ok=False,
                status="unknown_ui",
                message=f"未认出界面，不点（{bits}）",
            )
        if node == 'currency_wallet':
            # A currency popup is not the crafting hub even though an older
            # path alias treats it as one. Let the existing recovery verify
            # its exit; never locate or click a covered facility card.
            return ExecuteResult(False, 'wrong_page', '当前是持有的貨幣，先退出，不点其它入口')
        # menu=1 的滞后 HUD 已 remap 成 system_menu；寻路必须跟 remap，
        # 否则会走出「打开菜单」再被 skip_click 空转。设施细名仍用 full_node。
        src = full_node if node == graph_node(full_node) else node
        if _wrong_named_detail(full_node):
            hops = path_edges(src, "life_grid", root=root)
            if not hops:
                hops = path_edges(src, "gather_skill", root=root)
            if not hops:
                return ExecuteResult(
                    ok=False,
                    status="wrong_page",
                    message=f"已在{full_node}但不是目标{target}",
                )
        elif node == "overlay" and graph_node(target) not in {"overlay", "main_hud"}:
            hops = path_edges("overlay", "main_hud", root=root)
        elif _is_sap_nav(item_name) and graph_node(target) in {"item_info", "obtain_method"}:
            hops = _sap_hops(src, target, root=root, frame=frame)
        elif graph_node(target) == "inventory":
            gsrc = graph_node(src)
            if gsrc in {"facility", "facility_recipe"}:
                hops = path_edges(src, "craft_hub", root=root)
            elif gsrc in {
                "gather_skill",
                "gather_detail",
                "life_grid",
                "profile",
                "craft_hub",
            }:
                hops = path_edges(src, "main_hud", root=root)
            else:
                hops = path_edges(src, _path_dst(), root=root)
        elif (
            graph_node(target) == "facility_recipe"
            and graph_node(src) in {"facility", "facility_recipe"}
            and facility
        ):
            from mm_assist.life_execute.ui_logic import facility_id_of as _fid_nav

            src_fid = _fid_nav(src) or _fid_nav(full_node)
            if src_fid and src_fid != facility:
                hops = path_edges(src, "craft_hub", root=root)
            else:
                hops = path_edges(src, _path_dst(), root=root)
        else:
            hops = path_edges(src, _path_dst(), root=root)
        if hops is None:
            return ExecuteResult(
                ok=False,
                status="no_path",
                message=f"{node}→{target} 无允许边（见 ui_transition_review）",
            )
        if not hops:
            if _at_dest(full_node, frame, where):
                _mark_arrived(last_stage)
                clear_edge_memo_if_arrived(node)
                return ExecuteResult(ok=True, status="at_node", message=target)
            leave = {
                "gather_detail": "life_grid",
                "item_info": "gather_skill",
                "obtain_method": "item_info",
            }.get(graph_node(node), "craft_hub")
            hops = path_edges(node, leave, root=root)
            if not hops:
                return ExecuteResult(
                    ok=False,
                    status="wrong_page",
                    message=f"已在{node}但不是目标{target}",
                )
        edge = hops[0]
        label = str((edge.get("trigger") or {}).get("label") or edge.get("id") or "切界面")
        if skip_click(node, label, menu=getattr(where, "menu", None)):
            continue
        _prog(label)
        moved = _try_click_edge(
            edge, frame, node, label, stamps=obs.stamps, param_key=ui_param_key(root=root) if resolve is not None else None,
            before_where=where,
        )
        if moved is True:
            continue
        if moved is not None:
            return moved
    return ExecuteResult(ok=False, status="nav_timeout", message=f"未到达{target}")
