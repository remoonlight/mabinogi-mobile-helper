"""界面切换的计划、方法 A 验收与证据登记。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Callable

ProgressFn = Callable[[str], None]

from mm_assist.config import ROOT, load_config
from mm_assist.jobs.transitions import (
    assess_transition_captures_v2,
    edge_by_id,
    load_ui_transition_graph,
    require_allowed_edge,
)
from mm_assist.knowledge_paths import TRANSITION_ACCEPT_JSON, TRANSITION_GRAPH_JSON

ACCEPT_ENV = "MM_TRANSITION_ACCEPT"
SUITE_GATHER_PLUS_MENU = "gather_plus_menu"
ALLOWED_SKILLS = frozenset(
    {"woodcut", "mine", "herb", "shear", "harvest", "hoe", "insect", "fish", "daily"}
)

# 状态栏 / on_progress 只用人话步骤名，不暴露 edge_id / locator / fail_tags。
EDGE_STEP_ZH: dict[str, str] = {
    "main_hud_menu_system_menu": "打开菜单",
    "system_menu_role_profile": "打开角色",
    "profile_close_main_hud": "关闭人物",
    "system_menu_bag_inventory": "从菜单打开背包",
    "inventory_close_main_hud": "关闭背包",
    "main_hud_bag_inventory": "打开背包",
    "main_hud_avatar_profile": "打开人物",
    "profile_life_tab_life_grid": "打开生活技能",
    "life_grid_skill_gather_detail": "打开采集物品列表",
    "gather_detail_close_life_grid": "关闭采集详情",
    "life_grid_close_main_hud": "关闭生活技能",
    "system_menu_x_main_hud": "关闭菜单",
}
NODE_ZH: dict[str, str] = {
    "main_hud": "主界面",
    "system_menu": "系统菜单",
    "inventory": "背包",
    "profile": "人物",
    "life_grid": "生活技能",
    "gather_detail": "采集详情",
    "craft_hub": "制作",
}
_LOCATOR_STOP_ZH: dict[str, str] = {
    "main_hud_bag_inventory": "未能找到背包入口",
    "main_hud_menu_system_menu": "未能找到菜单入口",
    "main_hud_avatar_profile": "未能找到人物入口",
    "system_menu_bag_inventory": "未能找到菜单里的背包",
    "system_menu_role_profile": "未能找到角色入口",
    "profile_life_tab_life_grid": "未能找到生活技能",
    "life_grid_skill_gather_detail": "未能找到采集技能",
    "inventory_close_main_hud": "未能找到关闭按钮",
    "profile_close_main_hud": "未能找到关闭按钮",
    "gather_detail_close_life_grid": "未能找到关闭按钮",
    "life_grid_close_main_hud": "未能找到关闭按钮",
    "system_menu_x_main_hud": "未能找到关闭按钮",
}


def zh_step_label(edge_id: str) -> str:
    eid = str(edge_id or "").strip()
    return EDGE_STEP_ZH.get(eid, "这一步")


def zh_node_label(node_id: str | None) -> str:
    nid = str(node_id or "").strip()
    return NODE_ZH.get(nid, "当前界面")


def format_progress_running(idx: int, total: int, edge_id: str) -> str:
    return f"运行中 {idx}/{total}：{zh_step_label(edge_id)}"


def format_progress_done(idx: int, total: int, edge_id: str) -> str:
    return f"已完成 {idx}/{total}：{zh_step_label(edge_id)}"


def format_progress_locator_stop(edge_id: str) -> str:
    eid = str(edge_id or "").strip()
    why = _LOCATOR_STOP_ZH.get(eid, "未能找到入口")
    return f"已停止：{why}"


def format_progress_fail_stop(edge_id: str, fail_tags: list[str] | None = None) -> str:
    tags = [str(t) for t in (fail_tags or [])]
    if any("locator_reject" in t for t in tags):
        return format_progress_locator_stop(edge_id)
    if any("not_on_from" in t for t in tags):
        return "已停止：当前界面不对"
    if any("click_failed" in t for t in tags):
        return "已停止：点击没有生效"
    return f"已停止：{zh_step_label(edge_id)}没有完成"

# 界面动画/切页：宁慢勿抢（实机反馈切太快会导致 locator_reject）
_SETTLE_S = 8.0
_ARRIVE_S = 8.0
_POST_CLICK_S = 0.85
_BRIDGE_POST_CLICK_S = 1.0
_BRIDGE_GOAL_S = 6.0
_POLL_S = 0.35


class TransitionAcceptError(RuntimeError):
    pass


def node_from_inspect(
    *, root: Path | None = None
) -> tuple[str | None, str, str, str]:
    """栏上「界面：」折成图节点。zh 只给人看，不参与判定。"""
    from mm_assist.life_execute import ui_logic

    rec = ui_logic.read_inspect_record(root=root)
    node = str(rec.get("node") or "").strip()
    zh = str(rec.get("zh") or "").strip()
    stack_ready = bool(rec.get("stack_ready"))
    if not rec or not stack_ready or not node:
        return None, node, zh, "inspect_not_ready"
    folded = ui_logic.graph_node(node)
    if folded not in ui_logic.KNOWN_NODES:
        return None, node, zh, "node_unregistered"
    return folded, node, zh, ""


def require_inspect_readable(*, root: Path | None = None) -> dict[str, Any]:
    """live 开跑前：状态栏「界面：」须可读。读不到则 fail-closed。"""
    from mm_assist.life_execute.ui_logic import read_inspect_record

    rec = read_inspect_record(root=root)
    if not rec:
        raise TransitionAcceptError(
            "读不到状态栏「界面：」。请在状态栏面板点【开始】后再跑切换验收。"
        )
    return rec


def apply_inspect_arrival(
    *,
    want_to: str,
    key_before: tuple[str, str] | None,
    assess: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """到达主判：graph_node == edge['to']，且 ui_param_key 相对点击前有变化。"""
    from mm_assist.life_execute.ui_logic import ui_param_key

    graph_n, fine, zh, why = node_from_inspect(root=root)
    out = dict(assess or {})
    tags = list(out.get("fail_tags") or [])
    out["node_id"] = graph_n
    out["node_fine"] = fine
    out["node_zh_bar"] = zh
    out["inspect_why"] = why

    def _fail(tag: str) -> dict[str, Any]:
        out["pair_ok"] = False
        if tag not in tags:
            tags.append(tag)
        out["fail_tags"] = tags
        return out

    if why == "inspect_not_ready":
        return _fail("transition:inspect_not_ready")
    if why == "node_unregistered":
        return _fail(f"transition:node_unregistered:{fine}")
    if graph_n != str(want_to):
        return _fail(f"transition:wrong_node:{fine}")
    after = ui_param_key(root=root)
    if key_before is not None and after == key_before:
        return _fail("transition:no_change")
    out["pair_ok"] = True
    out["fail_tags"] = []
    return out


def accept_env_enabled() -> bool:
    return os.environ.get(ACCEPT_ENV, "").strip().lower() in {"1", "true", "yes"}


def manifest_path(*, root: Path | None = None) -> Path:
    if root is None:
        return TRANSITION_ACCEPT_JSON
    return Path(root) / "knowledge" / "ui" / "contracts" / "transition_accept_manifest.json"


def load_manifest(*, root: Path | None = None) -> dict[str, Any]:
    data = json.loads(manifest_path(root=root).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TransitionAcceptError("manifest 须为 object")
    if str(data.get("schema") or "") != "transition_accept_manifest.v1":
        raise TransitionAcceptError("未知 manifest schema")
    order = data.get("order")
    edges = data.get("edges")
    if not isinstance(order, list) or not order:
        raise TransitionAcceptError("manifest.order 须为非空 list")
    if not isinstance(edges, dict) or not edges:
        raise TransitionAcceptError("manifest.edges 须为 object")
    for eid in order:
        if str(eid) not in edges:
            raise TransitionAcceptError(f"order 引用未知 edge={eid}")
    return data


def suite_edge_ids(manifest: dict[str, Any], suite: str) -> list[str]:
    name = str(suite or "").strip() or SUITE_GATHER_PLUS_MENU
    if name != SUITE_GATHER_PLUS_MENU:
        raise TransitionAcceptError(f"未知 suite={suite!r}")
    if str(manifest.get("suite") or "") != name:
        raise TransitionAcceptError("manifest.suite 与请求不一致")
    return [str(x) for x in manifest["order"]]


def resolve_skill(skill: str | None, manifest: dict[str, Any]) -> str:
    s = str(skill or manifest.get("default_skill") or "woodcut").strip()
    if s not in ALLOWED_SKILLS:
        raise TransitionAcceptError(f"未知 skill={s!r}；允许 {sorted(ALLOWED_SKILLS)}")
    return s


def resolve_locator(
    edge_id: str,
    *,
    skill: str,
    manifest: dict[str, Any],
    root: Path | None = None,
) -> dict[str, Any]:
    spec = (manifest.get("edges") or {}).get(edge_id)
    if not isinstance(spec, dict):
        raise TransitionAcceptError(f"manifest 无 edge={edge_id}")
    loc = spec.get("locator") if isinstance(spec.get("locator"), dict) else {}
    kind = str(loc.get("kind") or "").strip()
    base = root or ROOT
    required = [
        str(x).replace("{skill}", skill).replace("\\", "/")
        for x in (spec.get("required_templates") or [])
    ]
    if kind == "hud_entry":
        return {
            "kind": kind,
            "entry_id": str(loc.get("entry_id") or "").strip(),
            "template": None,
            "threshold": None,
            "required_templates": required,
            "resolved": bool(str(loc.get("entry_id") or "").strip()),
        }
    tmpl = str(loc.get("template") or "").replace("{skill}", skill).replace("\\", "/")
    path = Path(tmpl)
    if not path.is_absolute():
        path = base / tmpl
    if kind == "skill_cell" and tmpl.endswith("skill_cell.png"):
        raise TransitionAcceptError("禁止泛用 skill_cell.png 盲点")
    if kind not in {"template", "skill_cell"}:
        raise TransitionAcceptError(f"edge {edge_id} 未知 locator.kind={kind!r}")
    fb = loc.get("fallback_client_px")
    fallback_client_px = fb if isinstance(fb, dict) else None
    return {
        "kind": kind,
        "entry_id": None,
        "template": tmpl,
        "template_path": str(path),
        "template_exists": path.is_file(),
        "threshold": float(loc.get("threshold") or 0.78),
        "required_templates": required,
        "skill": skill if kind == "skill_cell" else None,
        "resolved": path.is_file(),
        "roi": loc.get("roi"),
        "fallback_client_px": fallback_client_px,
    }


def plan_suite(
    *,
    suite: str = SUITE_GATHER_PLUS_MENU,
    skill: str | None = None,
    root: Path | None = None,
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    g = graph if graph is not None else load_ui_transition_graph(root=root)
    man = load_manifest(root=root)
    sk = resolve_skill(skill, man)
    steps: list[dict[str, Any]] = []
    for eid in suite_edge_ids(man, suite):
        edge = require_allowed_edge(g, eid)
        loc = resolve_locator(eid, skill=sk, manifest=man, root=root)
        steps.append(
            {
                "edge_id": eid,
                "from": edge["from"],
                "to": edge["to"],
                "label": str((edge.get("trigger") or {}).get("label") or ""),
                "zh": zh_step_label(eid),
                "locator": loc,
            }
        )
    return {"suite": suite, "skill": sk, "enough": False, "steps": steps}


def suite_start_index(
    boot_node: str | None,
    steps: list[dict[str, Any]],
    graph: dict[str, Any],
) -> int:
    """已在某 from 节点时从对应步开始，避免刚开菜单后内存仍报主界面。"""
    node = str(boot_node or "")
    if node == "craft_hub":
        node = "system_menu"
    if not node:
        return 0
    for i, st in enumerate(steps):
        ed = edge_by_id(graph, str(st["edge_id"]))
        if str(ed.get("from") or "") == node:
            return i
    return 0


def format_dry_run(plan: dict[str, Any]) -> str:
    lines = [
        "# transition-accept dry-run",
        f"suite={plan.get('suite')} skill={plan.get('skill')} enough=False",
    ]
    for i, step in enumerate(plan.get("steps") or [], 1):
        loc = step.get("locator") if isinstance(step.get("locator"), dict) else {}
        target = loc.get("entry_id") or loc.get("template")
        lines.append(
            f"{i:02d} {step['edge_id']}  {step['from']}->{step['to']}  "
            f"click={step.get('label')}  locator={loc.get('kind')}:{target}  "
            f"resolved={loc.get('resolved')}"
        )
    return "\n".join(lines) + "\n"


def _alias_locator(loc: dict[str, Any]) -> dict[str, Any]:
    out = dict(loc)
    if "detector" in out and "detector" not in out:
        out["detector"] = out["detector"]
    elif "detector" in out and "detector" not in out:
        out["detector"] = out["detector"]
    if "maturity" in out and "maturity" not in out:
        out["maturity"] = out["maturity"]
    elif "maturity" in out and "maturity" not in out:
        out["maturity"] = out["maturity"]
    if "min_margin" in out and "min_margin" not in out:
        out["min_margin"] = out["min_margin"]
    elif "min_margin" in out and "min_margin" not in out:
        out["min_margin"] = out["min_margin"]
    return out


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _job_from_probe(
    *,
    role: str,
    edge_id: str,
    capture_id: str,
    probe: dict[str, Any],
    template_hits: dict[str, bool],
    raw_sha256: str,
    method: str,
) -> dict[str, Any]:
    icons = [{"name": "hit"}] if (any(template_hits.values()) or role == "source") else []
    return {
        "method": method,
        "action": "transition_accept",
        "args": {
            "pair_role": role,
            "edge_id": edge_id,
            "capture_id": capture_id,
            "pair_id": capture_id,
        },
        "result": {
            "icons": icons,
            "icon_errors": [],
            "panel_probe": probe,
            "template_hits": template_hits,
            "raw_sha256": raw_sha256,
        },
    }


def match_node_signals(probe: dict[str, Any], signals: dict[str, Any]) -> list[str]:
    misses: list[str] = []
    for key, want in (signals or {}).items():
        got = probe.get(key)
        if got is None or bool(got) is not bool(want):
            misses.append(f"{key}:want={want!r} got={got!r}")
    return misses


def signals_match(probe: dict[str, Any], signals: dict[str, Any]) -> bool:
    return not match_node_signals(probe, signals)


def apply_visual_menu_open(
    probe: dict[str, Any],
    want: dict[str, Any],
    template_hits: dict[str, bool],
) -> dict[str, Any]:
    """策略 B：模板命中不定四参，只原样返回探测值。"""
    _ = want
    _ = template_hits
    return dict(probe)


# 证明「已经在该节点」的可见模板（内存位滞后时用）
_NODE_PRESENCE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "system_menu": ("knowledge/templates/menu/icon_close_x.png",),
    "inventory": ("knowledge/templates/inventory/close.png",),
    "profile": ("knowledge/templates/profile/close.png",),
    "life_grid": ("knowledge/templates/life_grid/close.png",),
    "gather_detail": ("knowledge/templates/gather_detail/close.png",),
}
_PRESENCE_ROI: dict[str, tuple[float, float, float, float]] = {
    "system_menu": (0.88, 0.0, 0.12, 0.14),
    "profile": (0.88, 0.0, 0.12, 0.14),
    "inventory": (0.88, 0.0, 0.12, 0.14),
    "life_grid": (0.88, 0.0, 0.12, 0.14),
    "gather_detail": (0.88, 0.0, 0.12, 0.14),
}
_PROFILE_TAB_ROI = (0.0, 0.3, 0.12, 0.12)
_LIFE_GRID_CELLS_ROI = (0.08, 0.15, 0.85, 0.75)
_GATHER_DETAIL_ACTION_ROI = (0.0, 0.38, 1.0, 0.60)
_GATHER_DETAIL_MARKER_TEMPLATES = (
    "knowledge/templates/gather_detail/find_nearby.png",
)
_PROFILE_SHELL_VISUAL_ORDER = ("gather_detail", "profile", "life_grid")


def _gather_detail_markers_visible(frame: Any, *, root: Path | None = None) -> bool:
    """详情页独有：须见「尋找附近位置」。不用 help（概览页会 fuzzy 误命中）。"""
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    rel = _GATHER_DETAIL_MARKER_TEMPLATES[0]
    if template_hit(
        frame,
        rel,
        root=root,
        roi=_GATHER_DETAIL_ACTION_ROI,
        threshold=0.72,
    ):
        return True
    try:
        from mm_assist.perception.item_sense import locate_find_nearby

        return bool(locate_find_nearby(frame, root=root).get("ok"))
    except Exception:
        return False


def _skill_overview_panel_visible(
    frame: Any,
    *,
    skill: str | None = None,
    root: Path | None = None,
) -> bool:
    """点技能格后、未点详情物：右侧 tiers 列表已出（非纯九门格）。"""
    if frame is None or _gather_detail_markers_visible(frame, root=root):
        return False
    sid = str(skill or "").strip()
    if not sid:
        return False
    from mm_assist.perception.item_sense import items_for_skill, locate_skill_item

    catalog = items_for_skill(sid, root=root)
    for name in catalog[:5]:
        if locate_skill_item(frame, skill_id=sid, name=name, root=root).get("ok"):
            return True
    return False


def default_gather_item_for_skill(skill: str, *, root: Path | None = None) -> str:
    from mm_assist.perception.item_sense import default_gather_item_name

    return default_gather_item_name(skill, root=root)


def _click_gather_item_entry(
    frame: Any,
    *,
    skill: str,
    item_name: str | None = None,
    root: Path | None = None,
    screen_origin: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """生活技能概览 → 点指定详情物进入详情页（只读定位坐标）。"""
    from mm_assist.perception.item_sense import locate_skill_item

    item = str(item_name or "").strip() or default_gather_item_for_skill(skill, root=root)
    out = locate_skill_item(frame, skill_id=skill, name=item, root=root)
    if not out.get("ok"):
        return {
            "accepted": False,
            "feature_ok": False,
            "status": "item_not_found",
            "item": item,
            "detail": out,
            "click": None,
        }
    cc = out.get("click_client") or [0, 0]
    sx, sy = screen_origin
    return {
        "accepted": True,
        "feature_ok": True,
        "status": str(out.get("src") or "item"),
        "item": item,
        "click": {
            "client": [int(cc[0]), int(cc[1])],
            "screen": [int(sx + int(cc[0])), int(sy + int(cc[1]))],
        },
    }


def _ensure_gather_detail_page(
    win: Any,
    *,
    skill: str,
    item_name: str | None = None,
    method: str,
    origin: tuple[int, int],
    root: Path | None,
    run_one_click: Any,
    grab_client: Any,
    stop_event: Any | None = None,
    deadline_s: float | None = None,
) -> bool:
    """生活技能概览 → 点 tiers 详情物，直到「尋找附近」可见。"""
    sk = str(skill or "woodcut").strip()
    target_item = str(item_name or "").strip() or None
    deadline = time.time() + float(deadline_s or _ARRIVE_S)
    item_tried = False
    while time.time() < deadline:
        if _aborted(stop_event):
            return False
        frame_mid = grab_client(win)
        if _gather_detail_markers_visible(frame_mid, root=root):
            return True
        if not item_tried and _skill_overview_panel_visible(
            frame_mid, skill=sk, root=root
        ):
            item_loc = _click_gather_item_entry(
                frame_mid,
                skill=sk,
                item_name=target_item,
                root=root,
                screen_origin=origin,
            )
            item_tried = True
            if item_loc.get("accepted") and isinstance(item_loc.get("click"), dict):
                ic = item_loc["click"].get("client") or [0, 0]
                clicked = run_one_click(
                    win,
                    int(ic[0]),
                    int(ic[1]),
                    method=method,
                    need_ui="life_grid",
                    stage="ensure_gather_detail:item",
                    fail_count=0,
                )
                if not clicked.get("ok"):
                    return False
                time.sleep(_POST_CLICK_S)
                continue
        time.sleep(_POLL_S)
    frame_mid = grab_client(win)
    return _gather_detail_markers_visible(frame_mid, root=root)


def recover_to_main_hud(
    win: Any,
    *,
    graph: dict[str, Any],
    manifest: dict[str, Any],
    method: str,
    root: Path | None,
    run_one_click: Any,
    grab_client: Any,
    read_state_ui: Any,
    skill: str = "woodcut",
    on_note: Callable[[str], None] | None = None,
    max_rounds: int = 10,
) -> bool:
    """多 skill 串跑前尽力关层回主界面（只点 manifest.recover 允许的边）。"""
    origin = (int(win.left), int(win.top))
    recover_ids = list((manifest.get("recover") or {}).get("to_main_hud") or [])
    for _ in range(max(1, int(max_rounds))):
        frame = grab_client(win)
        view = read_state_ui(root=root)
        probe = probe_from_state_ui(view)
        if confirm_node_from_frame(frame, "main_hud", root=root):
            return True
        node = resolve_node_id(probe, frame, graph, root=root) or "main_hud"
        if node == "main_hud":
            return True
        acted = False
        for eid in recover_ids:
            edge = edge_by_id(graph, str(eid))
            if str(edge.get("from") or "") != node:
                continue
            loc = resolve_locator(str(eid), skill=skill, manifest=manifest)
            bl = locate_click_point(loc, frame, root=root, screen_origin=origin)
            bclick = bl.get("click") if isinstance(bl.get("click"), dict) else None
            if not bclick or not (bl.get("accepted") or bl.get("feature_ok")):
                continue
            bc = bclick.get("client") or [0, 0]
            if on_note:
                on_note(f"回收：{zh_step_label(str(eid))}")
            run_one_click(
                win,
                int(bc[0]),
                int(bc[1]),
                method=method,
                need_ui=str(edge.get("from") or ""),
                stage=f"recover:{eid}",
                fail_count=0,
            )
            acted = True
            time.sleep(_POST_CLICK_S)
            break
        if not acted:
            break
    frame = grab_client(win)
    return confirm_node_from_frame(frame, "main_hud", root=root)


def _life_grid_cells_visible(frame: Any, *, root: Path | None = None) -> bool:
    """生活技能九门大格区域是否可见。"""
    if frame is None:
        return False
    for sid in sorted(ALLOWED_SKILLS):
        rel = f"knowledge/templates/life_grid/{sid}.png"
        if template_hit(
            frame, rel, root=root, roi=_LIFE_GRID_CELLS_ROI, threshold=0.70
        ):
            return True
    return False


def _profile_page_visible(frame: Any, *, root: Path | None = None) -> bool:
    """人物壳 · 我的資訊 Tab：九门格与详情均不可见。"""
    if frame is None:
        return False
    if _gather_detail_markers_visible(frame, root=root):
        return False
    if _life_grid_cells_visible(frame, root=root):
        return False
    return template_hit(
        frame,
        "knowledge/templates/profile/close.png",
        root=root,
        threshold=0.72,
    )


def _gather_detail_page_visible(frame: Any, *, root: Path | None = None) -> bool:
    """技能详情：须见详情页动作控件，单靠右上 close 不够。"""
    return _gather_detail_markers_visible(frame, root=root)


def _life_grid_page_visible(frame: Any, *, root: Path | None = None) -> bool:
    """生活技能 Tab / 九门格；详情页动作控件可见则否。"""
    if frame is None:
        return False
    if _gather_detail_markers_visible(frame, root=root):
        return False
    if _life_grid_cells_visible(frame, root=root):
        return True
    life_tab = template_hit(
        frame,
        "knowledge/templates/profile/life_tab.png",
        root=root,
        roi=_PROFILE_TAB_ROI,
        threshold=0.72,
    )
    if not life_tab:
        return False
    return template_hit(
        frame,
        "knowledge/templates/profile/close.png",
        root=root,
        threshold=0.72,
    )


def confirm_node_from_frame(
    frame: Any,
    node_id: str,
    *,
    root: Path | None = None,
) -> bool:
    """可见模板佐证；避免内存滞后把主界面误判成系统菜单。"""
    nid = str(node_id or "")
    if nid == "system_menu":
        return _system_menu_visible(frame, root=root)
    if nid == "profile":
        return _profile_page_visible(frame, root=root)
    if nid == "life_grid":
        return _life_grid_page_visible(frame, root=root)
    if nid == "gather_detail":
        return _gather_detail_page_visible(frame, root=root)
    if nid == "gather_skill":
        from mm_assist.life_execute.ui_logic import detect_gather_skill_id
        return bool(detect_gather_skill_id(frame, root=root))
    rels = _NODE_PRESENCE_TEMPLATES.get(nid) or ()
    if not rels or frame is None:
        if nid == "main_hud":
            return not any(
                confirm_node_from_frame(frame, other, root=root)
                for other in _NODE_PRESENCE_TEMPLATES
            )
        return nid not in _NODE_PRESENCE_TEMPLATES
    roi = _PRESENCE_ROI.get(nid)
    return any(template_hit(frame, rel, root=root, roi=roi) for rel in rels)


def _hud_menu_entry_visible(frame: Any, *, root: Path | None = None) -> bool:
    """主界面 ESC 三横条仍在 → 系统菜单未开。"""
    _ = root
    if frame is None:
        return False
    from mm_assist.platform.match import find_hamburger_bars

    hit = find_hamburger_bars(frame, roi=(0.83, 0.0, 0.14, 0.12))
    if hit is None:
        return False
    width = int(frame.shape[1])
    return width > 0 and (float(hit.cx) / float(width)) >= 0.83


def _esc_grid_icon_visible(frame: Any, *, root: Path | None = None) -> bool:
    """系统菜单四宫格：角色或背包图标（拒绝主界面左上头像）。"""
    if frame is None:
        return False
    from mm_assist.perception.menu_icons import ROI_ESC_CHAR
    from mm_assist.platform.match import match_icon_fuzzy, match_template

    candidates = (
        ("knowledge/templates/menu/icon_character.png", ROI_ESC_CHAR),
        ("knowledge/templates/menu/icon_inventory.png", (0.68, 0.08, 0.28, 0.36)),
    )
    for rel, roi in candidates:
        path = (root or ROOT) / rel
        if not path.is_file():
            continue
        hit = match_template(frame, path, roi=roi, threshold=0.55, min_margin=0.02)
        if hit is None:
            hit = match_icon_fuzzy(frame, path, roi=roi, threshold=0.52)
        if hit is None:
            continue
        width = int(frame.shape[1])
        if width > 0 and (float(hit.cx) / float(width)) >= 0.62:
            return True
    return False


def _system_menu_visible(frame: Any, *, root: Path | None = None) -> bool:
    """系统菜单：关闭 X + 四宫格图标。仅关闭 X 不够（主界面也会误命中）。"""
    if not _menu_close_visible(frame, root=root):
        return False
    if not _esc_grid_icon_visible(frame, root=root):
        return False
    # 三横条仍清晰可见 → 更像主界面假关闭钮
    if _hud_menu_entry_visible(frame, root=root):
        return False
    return True


def _menu_close_visible(frame: Any, *, root: Path | None = None) -> bool:
    rel = "knowledge/templates/menu/icon_close_x.png"
    path = (root or ROOT) / rel
    if frame is None or not path.is_file():
        return False
    roi = _PRESENCE_ROI["system_menu"]
    from mm_assist.platform.match import match_icon_fuzzy, match_template

    hit = match_template(frame, path, roi=roi, threshold=0.72, min_margin=0.02)
    if hit is None:
        hit = match_icon_fuzzy(frame, path, roi=roi, threshold=0.62)
    if hit is None:
        return False
    width = int(frame.shape[1])
    return width > 0 and (float(hit.cx) / float(width)) >= 0.85


def _bridge_path(
    steps: list[dict[str, Any]],
    graph: dict[str, Any],
    start: str,
    goal: str,
    *,
    max_hops: int = 3,
) -> list[dict[str, Any]]:
    from collections import deque

    by_from: dict[str, list[dict[str, Any]]] = {}
    for st in steps:
        ed = edge_by_id(graph, str(st["edge_id"]))
        by_from.setdefault(str(ed.get("from") or ""), []).append(st)
    q: deque[tuple[str, list[dict[str, Any]]]] = deque([(start, [])])
    seen = {start}
    while q:
        cur, path = q.popleft()
        if cur == goal:
            return path
        if len(path) >= max_hops:
            continue
        for st in by_from.get(cur, []):
            ed = edge_by_id(graph, str(st["edge_id"]))
            nxt = str(ed.get("to") or "")
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, path + [st]))
    return []


def _infer_node(
    probe: dict[str, Any],
    frame: Any,
    graph: dict[str, Any],
    *,
    root: Path | None = None,
) -> str | None:
    node = resolve_node_id(probe, frame, graph, root=root)
    if node:
        return node
    return None


def _on_from_node(
    probe: dict[str, Any],
    frame: Any,
    want_from: str,
    graph: dict[str, Any],
    *,
    root: Path | None = None,
) -> bool:
    got = resolve_node_id(probe, frame, graph, prefer=want_from, root=root)
    return got == want_from


def _execute_bridge_steps(
    bridge_steps: list[dict[str, Any]],
    *,
    win: Any,
    graph: dict[str, Any],
    method: str,
    origin: tuple[int, int],
    root: Path | None,
    run_one_click: Any,
    read_state_ui: Any,
    grab_client: Any,
    _note: Any,
    goal: str | None = None,
    skill: str | None = None,
    stop_event: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    frame = grab_client(win)
    probe_src: dict[str, Any] = {}
    sk = str(skill or "woodcut").strip()
    for bst in bridge_steps:
        bed = edge_by_id(graph, str(bst["edge_id"]))
        _note(f"先{zh_step_label(str(bed.get('id')))}…")
        frame = grab_client(win)
        bl = locate_click_point(
            bst["locator"], frame, root=root, screen_origin=origin
        )
        bclick = bl.get("click") if isinstance(bl.get("click"), dict) else None
        if not bclick or not (bl.get("accepted") or bl.get("feature_ok")):
            _note("找不到补路入口，跳过")
            break
        bc = bclick.get("client") or [0, 0]
        run_one_click(
            win,
            int(bc[0]),
            int(bc[1]),
            method=method,
            need_ui=str(bed.get("from") or ""),
            stage=str(bed.get("id") or "bridge"),
            fail_count=0,
        )
        time.sleep(_BRIDGE_POST_CLICK_S)
        if str(bed.get("to") or "") == "gather_detail":
            _ensure_gather_detail_page(
                win,
                skill=sk,
                method=method,
                origin=origin,
                root=root,
                run_one_click=run_one_click,
                grab_client=grab_client,
                stop_event=stop_event,
            )
    view = read_state_ui(root=root)
    frame = grab_client(win)
    probe_src = probe_from_state_ui(view)
    if goal == "gather_detail":
        _ensure_gather_detail_page(
            win,
            skill=sk,
            method=method,
            origin=origin,
            root=root,
            run_one_click=run_one_click,
            grab_client=grab_client,
            stop_event=stop_event,
        )
        frame = grab_client(win)
        view = read_state_ui(root=root)
        probe_src = probe_from_state_ui(view)
    if goal:
        deadline = time.time() + _BRIDGE_GOAL_S
        while time.time() < deadline:
            if _on_from_node(probe_src, frame, goal, graph, root=root):
                time.sleep(_POST_CLICK_S * 0.5)  # 到位后再稳一拍再交给定位
                frame = grab_client(win)
                view = read_state_ui(root=root)
                probe_src = probe_from_state_ui(view)
                break
            time.sleep(_POLL_S)
            view = read_state_ui(root=root)
            probe_src = probe_from_state_ui(view)
            frame = grab_client(win)
    return frame, probe_src


def enrich_probe_for_node(
    probe: dict[str, Any],
    frame: Any,
    node_id: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """策略 B：画面不定四参，探测值原样返回。"""
    _ = frame
    _ = node_id
    _ = root
    return dict(probe)


def probe_from_state_ui(view: Any) -> dict[str, Any]:
    if view is None:
        return {}
    family = str(getattr(view, "family", "") or "")
    menu = getattr(view, "menu_open", None)
    names = getattr(view, "ui_stack_names", None)
    if names is None and isinstance(getattr(view, "raw", None), dict):
        names = view.raw.get("ui_stack_names")
    from mm_assist.memory.readers.ui_stack import stack_inv

    inv = stack_inv([str(x) for x in (names or [])]) if names is not None else family == "inventory"
    dlg = getattr(view, "dialog_active", None)
    mp = getattr(view, "map_open", None)
    prf = getattr(view, "profile_open", None)
    if prf is None and family == "prf":
        prf = True
    if menu is None:
        menu = family in {"menu", "prf", "inventory", "map", "craft_hub"}
    return {
        "menu_open": bool(menu),
        "inventory_open": bool(inv),
        "dialog_active": bool(dlg) if dlg is not None else False,
        "map_open": bool(mp) if mp is not None else False,
        "profile_open": bool(prf) if prf is not None else False,
    }


def detect_node_id(
    probe: dict[str, Any],
    graph: dict[str, Any],
    *,
    prefer: str | None = None,
) -> str | None:
    """信号全中的节点；多节点同时中时优先 ``prefer``（边的 from），否则信号条数最多。

    craft_hub 的 signals 是 system_menu 的超集（多一个 map_open=false），
    不能单靠条数把系统菜单误判成製作枢纽。
    """
    hits: list[tuple[int, str]] = []
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        sig = node.get("signals") if isinstance(node.get("signals"), dict) else {}
        if match_node_signals(probe, sig):
            continue
        nid = str(node.get("id") or "")
        if prefer and nid == prefer:
            return nid
        hits.append((len(sig), nid))
    if not hits:
        return None
    ids = {nid for _, nid in hits}
    if prefer is None and "system_menu" in ids:
        return "system_menu"
    hits.sort(key=lambda x: x[0], reverse=True)
    return hits[0][1]


def resolve_node_id(
    probe: dict[str, Any],
    frame: Any,
    graph: dict[str, Any],
    *,
    prefer: str | None = None,
    root: Path | None = None,
) -> str | None:
    """策略 B：先画面节点，再四参粗族。不改写 probe。"""
    if prefer and confirm_node_from_frame(frame, prefer, root=root):
        return prefer
    shell_prefer = prefer in _PROFILE_SHELL_VISUAL_ORDER
    visual_order: tuple[str, ...] = _PROFILE_SHELL_VISUAL_ORDER + (
        "inventory",
        "system_menu",
    )
    visual: str | None = None
    for nid in visual_order:
        if nid == prefer and not shell_prefer:
            continue
        if confirm_node_from_frame(frame, nid, root=root):
            visual = nid
            break
    if visual:
        return visual
    if probe.get("inventory_open"):
        return "inventory"
    if probe.get("dialog_active"):
        return None
    if not probe.get("menu_open") and not probe.get("profile_open"):
        if confirm_node_from_frame(frame, "main_hud", root=root):
            return "main_hud"
        return "main_hud"
    return detect_node_id(probe, graph, prefer=prefer or "system_menu")


def node_signals(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    for node in graph.get("nodes") or []:
        if isinstance(node, dict) and str(node.get("id")) == node_id:
            sig = node.get("signals")
            return sig if isinstance(sig, dict) else {}
    return {}


def _locator_roi(loc: dict[str, Any], path: Path) -> tuple[float, float, float, float] | None:
    raw = loc.get("roi")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    name = path.name.lower()
    if "character" in name:
        from mm_assist.perception.menu_icons import ROI_ESC_CHAR

        return ROI_ESC_CHAR
    if any(tok in name for tok in ("inventory", "bag", "craft", "exchange")):
        return (0.68, 0.08, 0.28, 0.36)
    return None


def _ocr_esc_click(
    frame: Any,
    path: Path,
    screen_origin: tuple[int, int],
) -> dict[str, Any] | None:
    """菜单四宫格文字辅判；只点已认出的格子中心。"""
    name = path.name.lower()
    if "character" not in name and "inventory" not in name and "bag" not in name:
        return None
    from mm_assist.perception.menu_icons import (
        ESC_CHAR_NX,
        ESC_GRID_COLS,
        ESC_GRID_NY,
        ROI_ESC_CHAR,
        _ocr_blob,
    )

    h, w = frame.shape[:2]
    if "character" in name:
        cx = int(ESC_CHAR_NX * w)
        cy = int(ESC_GRID_NY * h)
        roi = ROI_ESC_CHAR
        needles = ("角色", "人物")
    else:
        cx = int(ESC_GRID_COLS[1] * w)
        cy = int(ESC_GRID_NY * h)
        roi = (0.68, 0.08, 0.28, 0.36)
        needles = ("背包", "物品")
    blob, _conf = _ocr_blob(frame, roi)
    if not any(n in (blob or "") for n in needles):
        return None
    sx, sy = screen_origin
    return {
        "accepted": True,
        "feature_ok": True,
        "status": "ocr_esc",
        "click": {"client": [cx, cy], "screen": [int(sx + cx), int(sy + cy)]},
        "score": 0.0,
    }


def _esc_grid_fallback_click(
    path: Path,
    frame: Any,
    screen_origin: tuple[int, int],
) -> dict[str, Any] | None:
    """系统菜单已开但模板 miss 时，点四宫格网格中心。"""
    if not _menu_close_visible(frame):
        return None
    name = path.name.lower()
    from mm_assist.perception.menu_icons import ESC_CHAR_NX, ESC_GRID_COLS, ESC_GRID_NY

    h, w = frame.shape[:2]
    if "character" in name:
        cx = int(ESC_CHAR_NX * w)
    elif any(tok in name for tok in ("inventory", "bag")):
        cx = int(ESC_GRID_COLS[1] * w)
    else:
        return None
    cy = int(ESC_GRID_NY * h)
    if cx / float(w) < 0.62:
        return None
    sx, sy = screen_origin
    return {
        "accepted": True,
        "feature_ok": True,
        "status": "esc_grid",
        "click": {"client": [cx, cy], "screen": [int(sx + cx), int(sy + cy)]},
        "score": 0.0,
    }


def _esc_icon_hit_ok(path: Path, hit: Any, frame: Any) -> bool:
    """角色/背包在系统菜单右侧；拒绝主界面左上头像误匹配。"""
    name = path.name.lower()
    if not any(tok in name for tok in ("character", "inventory", "bag")):
        return True
    try:
        width = int(frame.shape[1])
        cx = int(getattr(hit, "cx", -1))
    except Exception:
        return False
    return width > 0 and (cx / float(width)) >= 0.62


def _profile_side_tab_hit_ok(path: Path, hit: Any, frame: Any) -> bool:
    """人物页左侧 Tab；拒绝右上关闭钮等误匹配。"""
    name = path.name.lower()
    if not any(tok in name for tok in ("life_tab",)):
        return True
    try:
        width = int(frame.shape[1])
        height = int(frame.shape[0])
        cx = int(getattr(hit, "cx", -1))
        cy = int(getattr(hit, "cy", -1))
    except Exception:
        return False
    if width <= 0 or height <= 0:
        return False
    if (cx / float(width)) >= 0.75:
        return False
    if (cy / float(height)) <= 0.12:
        return False
    return True


def _client_px_fallback_click(
    loc: dict[str, Any],
    screen_origin: tuple[int, int],
) -> dict[str, Any] | None:
    """manifest 显式 fallback_client_px；仅在该边配置时启用。"""
    raw = loc.get("fallback_client_px")
    if not isinstance(raw, dict):
        return None
    try:
        x = int(raw.get("x", 0))
        y = int(raw.get("y", 0))
        w = int(raw.get("w", 0))
        h = int(raw.get("h", 0))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    cx, cy = x + w // 2, y + h // 2
    sx, sy = screen_origin
    return {
        "accepted": True,
        "feature_ok": True,
        "status": "client_px_fallback",
        "click": {"client": [cx, cy], "screen": [int(sx + cx), int(sy + cy)]},
        "score": 0.0,
    }


def template_hit(
    frame: Any,
    rel: str,
    *,
    root: Path | None = None,
    threshold: float = 0.78,
    roi: tuple[float, float, float, float] | None = None,
) -> bool:
    from mm_assist.platform.match import match_icon_fuzzy, match_template

    path = Path(rel)
    if not path.is_absolute():
        path = (root or ROOT) / rel
    if frame is None or not path.is_file():
        return False
    if match_template(
        frame, path, roi=roi, threshold=threshold, min_margin=0.02
    ) is not None:
        return True
    name = path.name.lower()
    fuzzy_roi = roi
    fuzzy_thr = min(0.65, float(threshold))
    if name in {"find_nearby.png"}:
        fuzzy_roi = roi or _GATHER_DETAIL_ACTION_ROI
    elif "close" in name:
        fuzzy_roi = roi or (0.88, 0.0, 0.12, 0.14)
        fuzzy_thr = min(0.58, float(threshold))
    else:
        return False
    return (
        match_icon_fuzzy(frame, path, roi=fuzzy_roi, threshold=fuzzy_thr) is not None
    )


def locate_click_point(
    loc: dict[str, Any],
    frame: Any,
    *,
    root: Path | None = None,
    screen_origin: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    kind = str(loc.get("kind") or "")
    if kind == "hud_entry":
        from mm_assist.jobs.hud_entries import entry_by_id, load_main_hud_catalog
        from mm_assist.perception.hud_entry_locator import locate_hud_entry

        catalog = load_main_hud_catalog(root=root)
        entry = dict(entry_by_id(catalog, str(loc.get("entry_id") or "")))
        raw_loc = entry.get("locator") if isinstance(entry.get("locator"), dict) else {}
        entry["locator"] = _alias_locator(raw_loc)
        return locate_hud_entry(
            entry, frame, root=root or ROOT, screen_origin=screen_origin
        )
    if kind in {"template", "skill_cell"}:
        from mm_assist.platform.match import match_icon_fuzzy, match_template

        rel = str(loc.get("template_path") or loc.get("template") or "")
        path = Path(rel)
        if not path.is_absolute():
            path = (root or ROOT) / rel
        if not path.is_file():
            return {
                "accepted": False,
                "feature_ok": False,
                "click": None,
                "status": "missing_template",
            }
        thr = float(loc.get("threshold") or 0.78)
        roi = _locator_roi(loc, path)
        hit = match_template(
            frame,
            path,
            roi=roi,
            threshold=thr,
            min_margin=0.02,
        )
        if hit is None:
            hit = match_icon_fuzzy(
                frame, path, roi=roi, threshold=min(0.58, thr)
            )
        if hit is not None and not _esc_icon_hit_ok(path, hit, frame):
            hit = None
        if hit is not None and not _profile_side_tab_hit_ok(path, hit, frame):
            hit = None
        if hit is None:
            ocr_hit = _ocr_esc_click(frame, path, screen_origin)
            if ocr_hit is not None:
                return ocr_hit
            grid_hit = _esc_grid_fallback_click(path, frame, screen_origin)
            if grid_hit is not None:
                return grid_hit
            fb_hit = _client_px_fallback_click(loc, screen_origin)
            if fb_hit is not None:
                return fb_hit
            return {
                "accepted": False,
                "feature_ok": False,
                "click": None,
                "status": "no_match",
            }
        sx, sy = screen_origin
        return {
            "accepted": True,
            "feature_ok": True,
            "status": "matched",
            "click": {
                "client": [int(hit.cx), int(hit.cy)],
                "screen": [int(sx + hit.cx), int(sy + hit.cy)],
            },
            "score": float(hit.score),
        }
    return {
        "accepted": False,
        "feature_ok": False,
        "click": None,
        "status": "unknown_kind",
    }


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def archive_dir(run_id: str, edge_id: str, *, root: Path | None = None) -> Path:
    base = (root or ROOT) / "knowledge" / "captures" / "transitions" / run_id / edge_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def _report_row_note(edge_id: str, pair_ok: bool, fail_tags: list[str] | None) -> str:
    if pair_ok:
        return "-"
    return format_progress_fail_stop(
        str(edge_id), list(fail_tags or [])
    ).removeprefix("已停止：")


def _report_stopped_on(run: dict[str, Any]) -> str:
    stopped = run.get("stopped_on")
    if stopped == "user_abort":
        return "已强制结束"
    last = (run.get("edges") or [None])[-1] or {}
    return format_progress_fail_stop(
        str(stopped), list(last.get("fail_tags") or [])
    )


def write_report(run: dict[str, Any], *, root: Path | None = None) -> Path:
    run_id = str(run.get("run_id") or "unknown")
    logs = (root or ROOT) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"transition_accept_{run_id}.md"
    lines = [
        "# transition-accept",
        f"run_id={run_id}",
        f"suite={run.get('suite')} skill={run.get('skill')}",
        f"pair_ok_count={run.get('pair_ok_count')} / {run.get('total')}",
        f"executed={len(run.get('edges') or [])}",
        "enough=False",
        "",
        "| 步骤 | 结果 | 说明 | 栏上界面： | 证据 |",
        "|---|---|---|---|---|",
    ]
    for row in run.get("edges") or []:
        eid = str(row.get("edge_id") or "")
        pair_ok = bool(row.get("pair_ok"))
        result = "通过" if pair_ok else "未通过"
        note = _report_row_note(eid, pair_ok, row.get("fail_tags"))
        evid = str(row.get("evidence_dir") or "-")
        bar = str(row.get("node_zh_bar") or "-")
        lines.append(
            f"| {zh_step_label(eid)} | {result} | {note} | {bar} | `{evid}` |"
        )
    if run.get("stopped_on"):
        lines.extend(["", _report_stopped_on(run)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = logs / f"transition_accept_{run_id}.json"
    json_path.write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def graph_json_path(*, root: Path | None = None) -> Path:
    if root is None:
        return TRANSITION_GRAPH_JSON
    return Path(root) / "knowledge" / "ui" / "contracts" / "ui_transition_graph.json"


def promote_edge(
    run_id: str,
    edge_id: str,
    *,
    i_confirmed: bool,
    root: Path | None = None,
) -> dict[str, Any]:
    if not i_confirmed:
        raise TransitionAcceptError("promote 须 --i-confirmed")
    base = root or ROOT
    report = base / "logs" / f"transition_accept_{run_id}.json"
    if not report.is_file():
        raise TransitionAcceptError(f"无报告 {report}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    row = next(
        (
            item
            for item in (payload.get("edges") or [])
            if str(item.get("edge_id")) == edge_id
        ),
        None,
    )
    if row is None:
        raise TransitionAcceptError(f"run 中无 edge={edge_id}")
    if not row.get("pair_ok"):
        raise TransitionAcceptError(f"edge {edge_id} pair_ok=false，拒绝 promote")
    path = graph_json_path(root=root)
    data = json.loads(path.read_text(encoding="utf-8"))
    found = False
    for edge in data.get("edges") or []:
        if str(edge.get("id")) != edge_id:
            continue
        if str(edge.get("kind")) != "allowed":
            raise TransitionAcceptError("禁止边不可 promote")
        evid = edge.get("evidence") if isinstance(edge.get("evidence"), dict) else {}
        evid["status"] = "verified"
        evid["params_ok"] = True
        evid["icon_ok"] = True
        paths = list(evid.get("paths") or [])
        evid_dir = str(row.get("evidence_dir") or "")
        if evid_dir and evid_dir not in paths:
            paths.append(evid_dir)
        evid["paths"] = paths
        edge["evidence"] = evid
        found = True
        break
    if not found:
        raise TransitionAcceptError(f"graph 无 edge={edge_id}")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "edge_id": edge_id, "enough": False, "status": "verified"}


def _aborted(stop_event: Any) -> bool:
    return bool(stop_event is not None and getattr(stop_event, "is_set", lambda: False)())


def run_suite(
    *,
    suite: str = SUITE_GATHER_PLUS_MENU,
    skill: str | None = None,
    method: str = "sendinput",
    dry_run: bool = False,
    live: bool = False,
    root: Path | None = None,
    stop_event: Any = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    def _note(msg: str) -> None:
        logging.getLogger(__name__).info("transition-accept: %s", msg)
        if on_progress is not None:
            on_progress(msg)

    plan = plan_suite(suite=suite, skill=skill, root=root)
    if dry_run or not live:
        return {**plan, "dry_run": True, "enough": False}
    if not accept_env_enabled():
        raise TransitionAcceptError(f"FAIL-CLOSED: 未设 {ACCEPT_ENV}=1")
    from mm_assist.safety.readonly import live_benchmark_env_enabled

    if not live_benchmark_env_enabled():
        raise TransitionAcceptError("FAIL-CLOSED: 未设 MM_LIVE_BENCHMARK=1")
    if _aborted(stop_event):
        _note("已强制结束")
        return {
            **plan,
            "run_id": None,
            "enough": False,
            "edges": [],
            "pair_ok_count": 0,
            "total": len(plan.get("steps") or []),
            "stopped_on": "user_abort",
        }

    import cv2

    from mm_assist.benchmarks.live_click import run_one_click
    from mm_assist.jobs.bar import require_state_watch_running
    from mm_assist.life_execute.ui_logic import ui_param_key
    from mm_assist.perception.state_ui import read_state_ui
    from mm_assist.platform.capture import grab_client
    from mm_assist.platform.window import bring_to_foreground, find_window, is_foreground

    require_state_watch_running(root=root)
    require_inspect_readable(root=root)
    cfg = load_config()
    win = find_window(cfg.window.title_substr)
    if win is None:
        raise TransitionAcceptError("game_window_missing")
    _note("正在把游戏切到前台…")
    bring_to_foreground(int(win.hwnd), force=True)
    if not is_foreground(int(win.hwnd)):
        raise TransitionAcceptError("游戏窗口未前台：点开始后请勿挡住游戏")

    graph = load_ui_transition_graph(root=root)
    run_id = new_run_id()
    out: dict[str, Any] = {
        "run_id": run_id,
        "suite": plan["suite"],
        "skill": plan["skill"],
        "enough": False,
        "edges": [],
        "pair_ok_count": 0,
        "total": len(plan["steps"]),
        "stopped_on": None,
    }
    origin = (int(win.left), int(win.top))
    total = int(out["total"])
    _note(f"运行中 0/{total}：已就绪")
    prev_to: str | None = None
    steps = list(plan["steps"])
    run_skill = str(plan.get("skill") or "woodcut")
    boot_node, _boot_fine, _boot_zh, _boot_why = node_from_inspect(root=root)
    start_i = suite_start_index(boot_node, steps, graph)
    if start_i:
        _note(f"已在{zh_node_label(boot_node)}，从第 {start_i + 1} 步开始")
    try:
        for idx, step in enumerate(steps[start_i:], start_i + 1):
            if _aborted(stop_event):
                out["stopped_on"] = "user_abort"
                _note("已强制结束")
                break
            eid = str(step["edge_id"])
            _note(format_progress_running(idx, total, eid))
            edge = edge_by_id(graph, eid)
            loc = step["locator"]
            want_from = str(edge.get("from") or "")
            settle_deadline = time.time() + _SETTLE_S
            frame = grab_client(win)
            view = read_state_ui(root=root)
            probe_src = probe_from_state_ui(view)
            while time.time() < settle_deadline:
                if _on_from_node(probe_src, frame, want_from, graph, root=root):
                    break
                if prev_to == want_from and confirm_node_from_frame(
                    frame, want_from, root=root
                ):
                    break
                time.sleep(_POLL_S)
                view = read_state_ui(root=root)
                probe_src = probe_from_state_ui(view)
                frame = grab_client(win)
            node = _infer_node(probe_src, frame, graph, root=root)
            if not _on_from_node(probe_src, frame, want_from, graph, root=root):
                cur = node or "main_hud"
                if cur == want_from:
                    cur = "main_hud"
                bridge_steps = _bridge_path(steps, graph, cur, want_from)
                if bridge_steps:
                    frame, probe_src = _execute_bridge_steps(
                        bridge_steps,
                        win=win,
                        graph=graph,
                        method=method,
                        origin=origin,
                        root=root,
                        run_one_click=run_one_click,
                        read_state_ui=read_state_ui,
                        grab_client=grab_client,
                        _note=_note,
                        goal=want_from,
                        skill=run_skill,
                        stop_event=stop_event,
                    )
                    probe_src = enrich_probe_for_node(
                        probe_src, frame, want_from, root=root
                    )
            if (
                not _on_from_node(probe_src, frame, want_from, graph, root=root)
                and want_from == "gather_detail"
            ):
                if _ensure_gather_detail_page(
                    win,
                    skill=run_skill,
                    method=method,
                    origin=origin,
                    root=root,
                    run_one_click=run_one_click,
                    grab_client=grab_client,
                    stop_event=stop_event,
                ):
                    frame = grab_client(win)
                    view = read_state_ui(root=root)
                    probe_src = enrich_probe_for_node(
                        probe_from_state_ui(view), frame, want_from, root=root
                    )
            if not _on_from_node(probe_src, frame, want_from, graph, root=root):
                node = _infer_node(probe_src, frame, graph, root=root)
                out["edges"].append(
                    {
                        "edge_id": eid,
                        "pair_ok": False,
                        "fail_tags": [f"transition:not_on_from:{node}"],
                        "evidence_dir": "",
                    }
                )
                out["stopped_on"] = eid
                _note(
                    f"已停止：现在在{zh_node_label(node)}，这一步需要{zh_node_label(want_from)}"
                )
                break
            # 系统菜单入口：再等一拍，避免动画未完就定位角色/背包
            if want_from == "system_menu":
                time.sleep(_POST_CLICK_S)
                frame = grab_client(win)
                view = read_state_ui(root=root)
                probe_src = probe_from_state_ui(view)
                if not _on_from_node(probe_src, frame, want_from, graph, root=root):
                    node = _infer_node(probe_src, frame, graph, root=root)
                    out["edges"].append(
                        {
                            "edge_id": eid,
                            "pair_ok": False,
                            "fail_tags": [f"transition:not_on_from:{node}"],
                            "evidence_dir": "",
                        }
                    )
                    out["stopped_on"] = eid
                    _note("已停止：菜单还没稳定")
                    break
            ok, buf = cv2.imencode(".png", frame)
            raw = buf.tobytes() if ok else b""
            located = locate_click_point(loc, frame, root=root, screen_origin=origin)
            click = located.get("click") if isinstance(located.get("click"), dict) else None
            accepted = bool(located.get("accepted") or located.get("feature_ok"))
            # 主界面误判成菜单时：强制补开菜单再定位一次
            if (
                (click is None or not accepted)
                and want_from == "system_menu"
                and not confirm_node_from_frame(frame, "system_menu", root=root)
            ):
                _note("找不到入口，先打开菜单再试…")
                bridge_steps = _bridge_path(steps, graph, "main_hud", want_from)
                if bridge_steps:
                    frame, probe_src = _execute_bridge_steps(
                        bridge_steps,
                        win=win,
                        graph=graph,
                        method=method,
                        origin=origin,
                        root=root,
                        run_one_click=run_one_click,
                        read_state_ui=read_state_ui,
                        grab_client=grab_client,
                        _note=_note,
                        goal=want_from,
                        skill=run_skill,
                        stop_event=stop_event,
                    )
                    time.sleep(_POST_CLICK_S)
                    frame = grab_client(win)
                    ok, buf = cv2.imencode(".png", frame)
                    raw = buf.tobytes() if ok else b""
                    located = locate_click_point(
                        loc, frame, root=root, screen_origin=origin
                    )
                    click = (
                        located.get("click")
                        if isinstance(located.get("click"), dict)
                        else None
                    )
                    accepted = bool(located.get("accepted") or located.get("feature_ok"))
            if click is None or not accepted:
                evid = archive_dir(run_id, eid, root=root)
                if raw:
                    (evid / "source.png").write_bytes(raw)
                (evid / "locator.json").write_text(
                    json.dumps(located, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
                out["edges"].append(
                    {
                        "edge_id": eid,
                        "pair_ok": False,
                        "fail_tags": ["transition:locator_reject"],
                        "locator": located,
                        "evidence_dir": str(evid.relative_to(root or ROOT)).replace("\\", "/"),
                    }
                )
                out["stopped_on"] = eid
                _note(format_progress_locator_stop(eid))
                break
            client = click.get("client") or [0, 0]
            cx, cy = int(client[0]), int(client[1])
            capture_id = f"{eid}_{uuid.uuid4().hex[:8]}"
            key_before = ui_param_key(root=root)
            clicked = run_one_click(
                win,
                cx,
                cy,
                method=method,
                need_ui=str(edge.get("from")),
                stage=eid,
                fail_count=0,
            )
            if not clicked.get("ok"):
                out["edges"].append(
                    {
                        "edge_id": eid,
                        "pair_ok": False,
                        "fail_tags": ["transition:click_failed"],
                        "evidence_dir": "",
                    }
                )
                out["stopped_on"] = eid
                break
            time.sleep(_POST_CLICK_S)
            item_click: dict[str, Any] | None = None
            if eid == "life_grid_close_main_hud":
                hud_deadline = time.time() + _ARRIVE_S
                close_retries = 0
                while time.time() < hud_deadline:
                    if _aborted(stop_event):
                        break
                    frame_mid = grab_client(win)
                    if confirm_node_from_frame(frame_mid, "main_hud", root=root):
                        break
                    if close_retries < 2:
                        run_one_click(
                            win,
                            cx,
                            cy,
                            method=method,
                            need_ui="main_hud",
                            stage=f"{eid}:close_retry",
                            fail_count=0,
                        )
                        close_retries += 1
                        time.sleep(_POST_CLICK_S)
                        continue
                    time.sleep(_POLL_S)
            if str(edge.get("to") or "") == "gather_detail":
                item_ready = _ensure_gather_detail_page(
                    win,
                    skill=run_skill,
                    method=method,
                    origin=origin,
                    root=root,
                    run_one_click=run_one_click,
                    grab_client=grab_client,
                    stop_event=stop_event,
                )
                item_click = None
                if _aborted(stop_event):
                    out["stopped_on"] = "user_abort"
                    _note("已强制结束")
                    break
                if not item_ready:
                    out["edges"].append(
                        {
                            "edge_id": eid,
                            "pair_ok": False,
                            "fail_tags": ["transition:item_not_found"],
                            "evidence_dir": "",
                        }
                    )
                    out["stopped_on"] = eid
                    _note("已停止：未能找到详情物")
                    break
            deadline = time.time() + _ARRIVE_S
            probe_dst = dict(probe_src)
            frame2 = None
            want_to = str(edge["to"])
            req = list(loc.get("required_templates") or [])
            hits: dict[str, bool] = {}
            dst_node: str | None = None
            while time.time() < deadline:
                if _aborted(stop_event):
                    break
                time.sleep(_POLL_S)
                view2 = read_state_ui(root=root)
                probe_dst = probe_from_state_ui(view2)
                frame2 = grab_client(win)
                hits = {rel: template_hit(frame2, rel, root=root) for rel in req}
                dst_node, _fine, _zh, why = node_from_inspect(root=root)
                if why == "inspect_not_ready":
                    continue
                if dst_node == want_to:
                    time.sleep(_POST_CLICK_S * 0.4)
                    frame2 = grab_client(win)
                    hits = {rel: template_hit(frame2, rel, root=root) for rel in req}
                    break
            if frame2 is None:
                frame2 = grab_client(win)
                hits = {rel: template_hit(frame2, rel, root=root) for rel in req}
            if _aborted(stop_event):
                out["stopped_on"] = "user_abort"
                _note("已强制结束")
                break
            ok2, buf2 = cv2.imencode(".png", frame2)
            raw2 = buf2.tobytes() if ok2 else b""
            src_job = _job_from_probe(
                role="source",
                edge_id=eid,
                capture_id=capture_id,
                probe=probe_src,
                template_hits={},
                raw_sha256=_sha256_bytes(raw) if raw else "",
                method="A",
            )
            dst_job = _job_from_probe(
                role="target",
                edge_id=eid,
                capture_id=capture_id,
                probe=probe_dst,
                template_hits=hits,
                raw_sha256=_sha256_bytes(raw2) if raw2 else "",
                method="A",
            )
            assess = assess_transition_captures_v2(
                edge, src_job, dst_job, graph=graph, required_templates=req
            )
            assess = apply_inspect_arrival(
                want_to=want_to,
                key_before=key_before,
                assess=assess,
                root=root,
            )
            dst_node = assess.get("node_id")
            evid = archive_dir(run_id, eid, root=root)
            if raw:
                (evid / "source.png").write_bytes(raw)
            if raw2:
                (evid / "target.png").write_bytes(raw2)
            rec = {
                "edge_id": eid,
                "pair_ok": bool(assess.get("pair_ok")),
                "enough": False,
                "assess": assess,
                "fail_tags": list(assess.get("fail_tags") or []),
                "node_id": dst_node,
                "node_fine": assess.get("node_fine") or "",
                "node_zh_bar": assess.get("node_zh_bar") or "",
                "evidence_dir": str(evid.relative_to(root or ROOT)).replace("\\", "/"),
                "click": click,
                "item_click": item_click,
            }
            (evid / "assess.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            out["edges"].append(rec)
            if rec["pair_ok"]:
                out["pair_ok_count"] += 1
                prev_to = str(edge.get("to") or "")
                _note(format_progress_done(idx, total, eid))
            else:
                out["stopped_on"] = eid
                _note(f"已停止：{zh_step_label(eid)}没有完成")
                break
    finally:
        try:
            from mm_assist.safety.input import release_mouse_buttons

            release_mouse_buttons()
        except Exception:
            pass
    out["report"] = str(write_report(out, root=root))
    if out.get("stopped_on") == "user_abort":
        _note("已强制结束")
    elif out.get("stopped_on"):
        last = (out.get("edges") or [None])[-1] or {}
        _note(
            format_progress_fail_stop(
                str(out.get("stopped_on")),
                list(last.get("fail_tags") or []),
            )
        )
    else:
        _note(f"结束 {out.get('pair_ok_count')}/{total}")
    return out
