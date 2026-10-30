"""详情物与详情页动作控件离线/实机基准。只读 catalog；实机 scroll 须 MM_LIVE_BENCHMARK=1。"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from mm_assist.config import ROOT
from mm_assist.perception.item_sense import (
    FIND_NEARBY_REL,
    GATHER_SKILL_IDS,
    HELP_REL,
    MAX_SCROLL_STEPS,
    ITEM_LIST_ROI,
    item_list_scroll_drag_points,
    iter_tier_items,
    item_template_path,
    items_for_skill,
    locate_find_nearby,
    locate_help_button,
    locate_skill_item,
)

LEDGER_DIR = ROOT / "logs" / "benchmarks" / "items"
SCHEMA = "item_sense_benchmark.v2"


def run_catalog_check(*, root: Path | None = None) -> dict[str, Any]:
    """40 项模板存在性检查（无帧）。"""
    base = root or ROOT
    rows: list[dict[str, Any]] = []
    missing = 0
    for row in iter_tier_items(root=base):
        sid = row["skill_id"]
        name = row["name"]
        path = item_template_path(sid, name, root=base)
        ok = path.is_file()
        if not ok:
            missing += 1
        rows.append(
            {
                "skill_id": sid,
                "name": name,
                "template": str(path.relative_to(base)).replace("\\", "/"),
                "template_ok": ok,
            }
        )
    action_rows = []
    for rel, label in ((FIND_NEARBY_REL, "find_nearby"), (HELP_REL, "help")):
        path = base / rel
        ok = path.is_file()
        if not ok:
            missing += 1
        action_rows.append(
            {
                "label": label,
                "template": rel,
                "template_ok": ok,
            }
        )
    total = len(rows) + len(action_rows)
    return {
        "schema": SCHEMA,
        "mode": "catalog",
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "total": total,
        "missing": missing,
        "items": rows,
        "actions": action_rows,
        "ok": missing == 0,
    }


def _load_frame(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        return None
    return img


def probe_skill_items_on_frame(
    frame: np.ndarray,
    skill_id: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """单帧扫描一门技能全部详情物。"""
    sid = str(skill_id or "").strip()
    catalog = items_for_skill(sid, root=root)
    rows: list[dict[str, Any]] = []
    for name in catalog:
        out = locate_skill_item(frame, skill_id=sid, name=name, root=root)
        rows.append(
            {
                "skill_id": sid,
                "name": name,
                "ok": bool(out.get("ok")),
                "src": out.get("src"),
                "error": out.get("error"),
                "scroll_plan": out.get("scroll_plan"),
            }
        )
    visible = sum(1 for r in rows if r.get("ok"))
    return {
        "skill_id": sid,
        "total": len(catalog),
        "visible_ok": visible,
        "rows": rows,
        "ok": visible == len(catalog),
    }


def run_frame_batch(
    frame: np.ndarray,
    *,
    skill_id: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """单帧批测：一门或九门全部详情物。"""
    skills = (str(skill_id),) if skill_id else GATHER_SKILL_IDS
    probes = [probe_skill_items_on_frame(frame, sk, root=root) for sk in skills]
    total = sum(p["total"] for p in probes)
    ok_count = sum(p["visible_ok"] for p in probes)
    return {
        "schema": SCHEMA,
        "mode": "frame_batch",
        "skills": list(skills),
        "total_items": total,
        "visible_ok": ok_count,
        "probes": probes,
        "ok": ok_count == total,
    }


def run_frame_benchmark(
    frame: np.ndarray,
    *,
    skill_id: str,
    item_name: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """单帧：详情物 + 两动作控件定位。"""
    item = locate_skill_item(frame, skill_id=skill_id, name=item_name, root=root)
    nearby = locate_find_nearby(frame, root=root)
    help_btn = locate_help_button(frame, root=root)
    return {
        "skill_id": skill_id,
        "item_name": item_name,
        "item": item,
        "find_nearby": nearby,
        "help": help_btn,
    }


def _require_live_benchmark() -> None:
    if os.environ.get("MM_LIVE_BENCHMARK") != "1":
        raise RuntimeError("实机 scroll/点击须 MM_LIVE_BENCHMARK=1")


def _scroll_item_list_down(
    win: Any,
    frame: np.ndarray,
    *,
    method: str,
    skill_id: str = "",
    drag_fn: Callable[..., bool] | None = None,
    root: Path | None = None,
) -> bool:
    """尝试滚动；须保持同 skill overview，且列表 ROI 像素有变化。"""
    _require_live_benchmark()
    from mm_assist.benchmarks.live_click import grab_live_client, run_one_drag, run_one_wheel
    from mm_assist.benchmarks.transition_accept import _skill_overview_panel_visible

    sid = str(skill_id or "").strip()

    def _roi_sig(img: np.ndarray) -> bytes:
        h, w = int(img.shape[0]), int(img.shape[1])
        rx, ry, rw, rh = ITEM_LIST_ROI
        x0, y0 = int(w * rx), int(h * ry)
        x1, y1 = int(w * (rx + rw)), int(h * (ry + rh))
        crop = img[y0:y1, x0:x1]
        small = cv2.resize(crop, (48, 64), interpolation=cv2.INTER_AREA)
        return small.tobytes()

    before_sig = _roi_sig(frame)
    (x0, y0), (x1, y1) = item_list_scroll_drag_points(frame)
    if drag_fn is not None:
        return bool(drag_fn(win, x0, y0, x1, y1, method=method))

    h, w = int(frame.shape[0]), int(frame.shape[1])
    rx, ry, rw, rh = ITEM_LIST_ROI
    cx = int(w * (rx + rw * 0.55))
    cy = int(h * (ry + rh * 0.50))
    stage_base = sid or "list"

    def _still_ok(after_frame: np.ndarray) -> bool:
        if sid and not _skill_overview_panel_visible(after_frame, skill=sid, root=root):
            return False
        return _roi_sig(after_frame) != before_sig

    for wheel_method in ("sendinput", "postmessage"):
        wheel = run_one_wheel(
            win,
            cx,
            cy,
            method=wheel_method,
            need_ui="life_grid",
            stage=f"item_probe:wheel:{stage_base}",
            fail_count=0,
            notches=5,
            delta=-240,
        )
        if not wheel.get("ok"):
            continue
        time.sleep(0.35)
        after = grab_live_client(win)
        if _still_ok(after):
            return True
    for drag_method in ("sendinput", "postmessage"):
        out = run_one_drag(
            win,
            x0,
            y0,
            x1,
            y1,
            method=drag_method,
            need_ui="life_grid",
            stage=f"item_probe:scroll:{stage_base}",
            fail_count=0,
        )
        if not out.get("ok"):
            continue
        time.sleep(0.35)
        after = grab_live_client(win)
        if _still_ok(after):
            return True
    return False


def _find_active_skill_overview(
    frame: np.ndarray,
    *,
    root: Path | None = None,
) -> str | None:
    from mm_assist.benchmarks.transition_accept import (
        _gather_detail_markers_visible,
        _skill_overview_panel_visible,
    )

    if frame is None or _gather_detail_markers_visible(frame, root=root):
        return None
    for sid in GATHER_SKILL_IDS:
        if _skill_overview_panel_visible(frame, skill=sid, root=root):
            return sid
    return None


def _close_skill_overview(
    win: Any,
    skill_id: str,
    *,
    method: str,
    root: Path | None,
    grab_fn: Callable[[Any], np.ndarray],
    click_fn: Callable[..., dict[str, Any]] | None = None,
) -> bool:
    """关掉右侧 tiers 弹窗（toggle 同格）。"""
    sid = str(skill_id or "").strip()
    frame = grab_fn(win)
    from mm_assist.benchmarks.transition_accept import _skill_overview_panel_visible

    if not _skill_overview_panel_visible(frame, skill=sid, root=root):
        return True
    if not _click_skill_cell(
        win, frame, sid, method=method, root=root, click_fn=click_fn
    ):
        return False
    time.sleep(0.35)
    frame = grab_fn(win)
    return not _skill_overview_panel_visible(frame, skill=sid, root=root)


def _reset_to_life_grid_for_probe(
    win: Any,
    *,
    method: str,
    root: Path | None,
    grab_fn: Callable[[Any], np.ndarray],
    click_fn: Callable[..., dict[str, Any]] | None = None,
) -> bool:
    """关掉详情/弹窗，回到九门格。"""
    from mm_assist.benchmarks.transition_accept import (
        _gather_detail_markers_visible,
        _life_grid_cells_visible,
    )

    for _ in range(4):
        frame = grab_fn(win)
        for sk in GATHER_SKILL_IDS:
            _close_gather_detail_to_life_grid(
                win, skill_id=sk, method=method, root=root, grab_fn=grab_fn
            )
        frame = grab_fn(win)
        active = _find_active_skill_overview(frame, root=root)
        if active:
            _close_skill_overview(
                win,
                active,
                method=method,
                root=root,
                grab_fn=grab_fn,
                click_fn=click_fn,
            )
            time.sleep(0.25)
            continue
        frame = grab_fn(win)
        if _life_grid_cells_visible(frame, root=root) and not _gather_detail_markers_visible(
            frame, root=root
        ):
            return True
        time.sleep(0.2)
    frame = grab_fn(win)
    return _life_grid_cells_visible(frame, root=root)


def _open_skill_overview(
    win: Any,
    skill_id: str,
    *,
    method: str,
    root: Path | None,
    grab_fn: Callable[[Any], np.ndarray],
    click_fn: Callable[..., dict[str, Any]] | None = None,
) -> bool:
    """回到九门格并打开指定 skill 的 tiers 弹窗。"""
    from mm_assist.benchmarks.transition_accept import _skill_overview_panel_visible

    sid = str(skill_id or "").strip()
    if not _reset_to_life_grid_for_probe(
        win, method=method, root=root, grab_fn=grab_fn, click_fn=click_fn
    ):
        return False
    for attempt in range(2):
        frame = grab_fn(win)
        if _skill_overview_panel_visible(frame, skill=sid, root=root):
            return True
        if not _click_skill_cell(
            win, frame, sid, method=method, root=root, click_fn=click_fn
        ):
            return False
        time.sleep(0.45)
        deadline = time.time() + 4.0
        while time.time() < deadline:
            frame = grab_fn(win)
            if _skill_overview_panel_visible(frame, skill=sid, root=root):
                return True
            time.sleep(0.15)
        _close_skill_overview(
            win, sid, method=method, root=root, grab_fn=grab_fn, click_fn=click_fn
        )
        time.sleep(0.3)
    return False


def _click_skill_cell(
    win: Any,
    frame: np.ndarray,
    skill_id: str,
    *,
    method: str,
    root: Path | None,
    click_fn: Callable[..., dict[str, Any]] | None = None,
) -> bool:
    from mm_assist.benchmarks.transition_accept import locate_click_point, resolve_locator

    manifest_path = (root or ROOT) / "knowledge/ui/contracts/transition_accept_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loc = resolve_locator(
        "life_grid_skill_gather_detail",
        skill=skill_id,
        manifest=manifest,
        root=root,
    )
    origin = (int(win.left), int(win.top))
    if click_fn is not None:
        out = click_fn(loc, frame, root=root, screen_origin=origin)
    else:
        out = locate_click_point(loc, frame, root=root, screen_origin=origin)
    if not out.get("accepted") or not isinstance(out.get("click"), dict):
        return False
    cc = out["click"].get("client") or [0, 0]
    if click_fn is None:
        from mm_assist.benchmarks.live_click import run_one_click

        clicked = run_one_click(
            win,
            int(cc[0]),
            int(cc[1]),
            method=method,
            need_ui="life_grid",
            stage=f"item_probe:skill_cell:{skill_id}",
            fail_count=0,
        )
        return bool(clicked.get("ok"))
    return True


def probe_skill_items_live(
    win: Any,
    skill_id: str,
    *,
    method: str = "sendinput",
    root: Path | None = None,
    grab_fn: Callable[[Any], np.ndarray] | None = None,
    click_skill: bool = True,
    scroll: bool = True,
    drag_fn: Callable[..., bool] | None = None,
    click_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """实机：一门技能全部详情物（含向下滚动）。"""
    from mm_assist.benchmarks.live_click import grab_live_client

    sid = str(skill_id or "").strip()
    grab = grab_fn or grab_live_client
    frame = grab(win)
    from mm_assist.benchmarks.transition_accept import _skill_overview_panel_visible

    if click_skill:
        _require_live_benchmark()
        if not _open_skill_overview(
            win,
            sid,
            method=method,
            root=root,
            grab_fn=grab,
            click_fn=click_fn,
        ):
            return {
                "skill_id": sid,
                "ok": False,
                "error": "skill_overview_not_visible",
                "rows": [],
            }
    elif not _skill_overview_panel_visible(frame, skill=sid, root=root):
        return {
            "skill_id": sid,
            "ok": False,
            "error": "skill_overview_not_visible",
            "rows": [],
        }

    catalog = items_for_skill(sid, root=root)
    rows: list[dict[str, Any]] = []
    pending = list(catalog)
    found_map: dict[str, dict[str, Any]] = {}
    total_scrolls = 0
    for round_idx in range(MAX_SCROLL_STEPS + 1):
        frame = grab(win)
        still: list[str] = []
        for name in pending:
            detail = locate_skill_item(frame, skill_id=sid, name=name, root=root)
            if detail.get("ok"):
                found_map[name] = {
                    **detail,
                    "scroll_attempts": total_scrolls,
                }
            else:
                still.append(name)
        pending = still
        if not pending:
            break
        if not scroll or round_idx >= MAX_SCROLL_STEPS:
            break
        if not _scroll_item_list_down(
            win, frame, method=method, skill_id=sid, drag_fn=drag_fn, root=root
        ):
            break
        total_scrolls += 1
        time.sleep(0.45)
    for name in catalog:
        hit = found_map.get(name)
        if hit:
            rows.append(
                {
                    "skill_id": sid,
                    "name": name,
                    "ok": True,
                    "src": hit.get("src"),
                    "error": None,
                    "scroll_attempts": hit.get("scroll_attempts", 0),
                }
            )
        else:
            rows.append(
                {
                    "skill_id": sid,
                    "name": name,
                    "ok": False,
                    "src": None,
                    "error": "not_found",
                    "scroll_attempts": total_scrolls,
                }
            )
    visible = sum(1 for r in rows if r.get("ok"))
    return {
        "skill_id": sid,
        "total": len(catalog),
        "visible_ok": visible,
        "scroll_used": total_scrolls,
        "rows": rows,
        "ok": visible == len(catalog),
    }


def _ensure_life_grid_for_probe(
    win: Any,
    *,
    method: str,
    root: Path | None,
    grab_fn: Callable[[Any], np.ndarray],
) -> bool:
    """尽力导航到 life_grid（生活技能 Tab → 九门格）。"""
    from mm_assist.benchmarks.live_click import run_one_click
    from mm_assist.benchmarks.transition_accept import (
        _life_grid_cells_visible,
        edge_by_id,
        load_manifest,
        load_ui_transition_graph,
        locate_click_point,
        resolve_locator,
    )

    base = root or ROOT
    manifest = load_manifest(root=base)
    graph = load_ui_transition_graph(root=base)
    origin = (int(win.left), int(win.top))
    grab = grab_fn
    frame = grab(win)
    if _life_grid_cells_visible(frame, root=base):
        return True

    from mm_assist.benchmarks.transition_accept import confirm_node_from_frame

    if confirm_node_from_frame(frame, "profile", root=base):
        nav_edges = ("profile_life_tab_life_grid",)
    elif confirm_node_from_frame(frame, "system_menu", root=base):
        nav_edges = ("system_menu_role_profile", "profile_life_tab_life_grid")
    else:
        nav_edges = (
            "main_hud_menu_system_menu",
            "system_menu_role_profile",
            "profile_life_tab_life_grid",
        )
    for eid in nav_edges:
        frame = grab(win)
        if _life_grid_cells_visible(frame, root=base):
            return True
        edge = edge_by_id(graph, eid)
        loc = resolve_locator(eid, skill="woodcut", manifest=manifest, root=base)
        bl = locate_click_point(loc, frame, root=base, screen_origin=origin)
        bclick = bl.get("click") if isinstance(bl.get("click"), dict) else None
        if not bclick or not (bl.get("accepted") or bl.get("feature_ok")):
            if eid == "profile_life_tab_life_grid":
                continue
            return False
        bc = bclick.get("client") or [0, 0]
        clicked = run_one_click(
            win,
            int(bc[0]),
            int(bc[1]),
            method=method,
            need_ui=str(edge.get("from") or ""),
            stage=f"item_probe:nav:{eid}",
            fail_count=0,
        )
        if not clicked.get("ok"):
            return False
        time.sleep(0.4)
    frame = grab(win)
    if _life_grid_cells_visible(frame, root=base):
        return True
    # 已在人物壳但未切 Tab：再试 life_tab
    loc = resolve_locator(
        "profile_life_tab_life_grid", skill="woodcut", manifest=manifest, root=base
    )
    bl = locate_click_point(loc, frame, root=base, screen_origin=origin)
    bclick = bl.get("click") if isinstance(bl.get("click"), dict) else None
    if bclick and (bl.get("accepted") or bl.get("feature_ok")):
        bc = bclick.get("client") or [0, 0]
        run_one_click(
            win,
            int(bc[0]),
            int(bc[1]),
            method=method,
            need_ui="profile",
            stage="item_probe:nav:life_tab_retry",
            fail_count=0,
        )
        time.sleep(0.4)
    frame = grab(win)
    return _life_grid_cells_visible(frame, root=base)


def _close_gather_detail_to_life_grid(
    win: Any,
    *,
    skill_id: str,
    method: str,
    root: Path | None,
    grab_fn: Callable[[Any], np.ndarray],
) -> None:
    """详情页 → 生活技能概览（点 gather_detail close）。"""
    from mm_assist.benchmarks.live_click import run_one_click
    from mm_assist.benchmarks.transition_accept import (
        _gather_detail_markers_visible,
        locate_click_point,
        load_manifest,
        resolve_locator,
    )

    frame = grab_fn(win)
    if not _gather_detail_markers_visible(frame, root=root):
        return
    manifest = load_manifest(root=root)
    origin = (int(win.left), int(win.top))
    loc = resolve_locator(
        "gather_detail_close_life_grid",
        skill=skill_id,
        manifest=manifest,
        root=root,
    )
    bl = locate_click_point(loc, frame, root=root, screen_origin=origin)
    bclick = bl.get("click") if isinstance(bl.get("click"), dict) else None
    if not bclick:
        return
    bc = bclick.get("client") or [0, 0]
    run_one_click(
        win,
        int(bc[0]),
        int(bc[1]),
        method=method,
        need_ui="gather_detail",
        stage=f"item_probe:close_detail:{skill_id}",
        fail_count=0,
    )
    time.sleep(0.35)


def run_live_full_catalog_probe(
    win: Any,
    *,
    method: str = "sendinput",
    root: Path | None = None,
    skills: tuple[str, ...] | None = None,
    scroll: bool = True,
) -> dict[str, Any]:
    """实机九门 × 全部详情物。须 MM_LIVE_BENCHMARK=1；自动导航到 life_grid。"""
    _require_live_benchmark()
    from mm_assist.benchmarks.live_click import grab_live_client

    grab = grab_live_client
    if not _ensure_life_grid_for_probe(
        win, method=method, root=root, grab_fn=grab
    ):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return {
            "schema": SCHEMA,
            "mode": "live_full",
            "ts": ts,
            "ok": False,
            "error": "life_grid_not_ready",
            "total_items": 0,
            "visible_ok": 0,
            "probes": [],
        }
    skill_list = skills or GATHER_SKILL_IDS
    probes = []
    for sk in skill_list:
        probe = probe_skill_items_live(
            win,
            sk,
            method=method,
            root=root,
            click_skill=True,
            scroll=scroll,
        )
        probes.append(probe)
        _close_gather_detail_to_life_grid(
            win, skill_id=sk, method=method, root=root, grab_fn=grab
        )
        _close_skill_overview(
            win,
            sk,
            method=method,
            root=root,
            grab_fn=grab,
        )
        if not probe.get("ok"):
            _reset_to_life_grid_for_probe(
                win, method=method, root=root, grab_fn=grab
            )
    total = sum(p.get("total") or 0 for p in probes)
    ok_count = sum(p.get("visible_ok") or 0 for p in probes)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {
        "schema": SCHEMA,
        "mode": "live_full",
        "ts": ts,
        "skills": list(skill_list),
        "total_items": total,
        "visible_ok": ok_count,
        "probes": probes,
        "ok": ok_count == total and total > 0,
    }
    ledger = LEDGER_DIR
    ledger.mkdir(parents=True, exist_ok=True)
    path = ledger / f"item_sense_live_{ts}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out["ledger_path"] = str(path.relative_to(root or ROOT)).replace("\\", "/")
    return out


def run_offline_benchmark(
    *,
    root: Path | None = None,
    frame_path: Path | str | None = None,
    skill_id: str = "woodcut",
    item_name: str = "樹木",
    out_dir: Path | None = None,
    batch: bool = False,
) -> dict[str, Any]:
    """catalog 必跑；frame_path 时可单条或 batch 感知定位。"""
    base = root or ROOT
    out = run_catalog_check(root=base)
    out["frame_probe"] = None
    out["frame_batch"] = None
    fp = Path(frame_path) if frame_path else None
    if fp is not None:
        frame = _load_frame(fp)
        if frame is not None:
            if batch:
                batch_out = run_frame_batch(frame, skill_id=skill_id or None, root=base)
                out["frame_batch"] = batch_out
                out["ok"] = bool(out.get("ok")) and bool(batch_out.get("ok"))
            else:
                out["frame_probe"] = run_frame_benchmark(
                    frame, skill_id=skill_id, item_name=item_name, root=base
                )
    ledger = out_dir or LEDGER_DIR
    ledger.mkdir(parents=True, exist_ok=True)
    ts = out["ts"]
    path = ledger / f"item_sense_{ts}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out["ledger_path"] = str(path.relative_to(base)).replace("\\", "/")
    return out


def format_report(result: dict[str, Any]) -> str:
    mode = result.get("mode")
    lines = [
        "# item-sense benchmark",
        f"mode={mode} ok={result.get('ok')} missing={result.get('missing', '—')}/{result.get('total', '—')}",
    ]
    if result.get("ledger_path"):
        lines.append(f"ledger={result['ledger_path']}")
    if mode == "live_full":
        lines.append(
            f"visible_ok={result.get('visible_ok')}/{result.get('total_items')}"
        )
        for probe in result.get("probes") or []:
            sid = probe.get("skill_id")
            vis = probe.get("visible_ok")
            tot = probe.get("total")
            lines.append(f"skill {sid}: {vis}/{tot} ok={probe.get('ok')}")
            for row in probe.get("rows") or []:
                if not row.get("ok"):
                    lines.append(
                        f"  FAIL {row.get('name')} err={row.get('error')}"
                    )
        return "\n".join(lines) + "\n"
    for row in result.get("items") or []:
        if not row.get("template_ok"):
            lines.append(f"MISS item {row.get('skill_id')}/{row.get('name')}")
    for row in result.get("actions") or []:
        if not row.get("template_ok"):
            lines.append(f"MISS action {row.get('label')}")
    batch = result.get("frame_batch")
    if isinstance(batch, dict):
        lines.append(
            f"frame_batch visible_ok={batch.get('visible_ok')}/{batch.get('total_items')}"
        )
        for probe in batch.get("probes") or []:
            if not probe.get("ok"):
                lines.append(f"skill {probe.get('skill_id')} partial {probe.get('visible_ok')}/{probe.get('total')}")
    probe = result.get("frame_probe")
    if isinstance(probe, dict):
        for key in ("item", "find_nearby", "help"):
            sub = probe.get(key) if isinstance(probe.get(key), dict) else {}
            lines.append(f"{key}: ok={sub.get('ok')} src={sub.get('src')} err={sub.get('error')}")
    return "\n".join(lines) + "\n"
