"""加工设施页 locate / 受控导航采图。可见 ≠ 可投料。禁止点「加工」投料。"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from mm_assist.config import ROOT, AppConfig
from mm_assist.jobs.menu_buttons import process_page_buttons, save_menu_button_crop
from mm_assist.perception.process_sense import (
    CLAIM_ALL_ROI,
    FACILITY_LIST_ROI,
    HUB_CARD_IDS,
    PROCESS_BUTTON_ROI,
    RECIPE_FACILITIES,
    detect_facility_page,
    locate_hub_card,
    locate_claim_all,
    locate_process_button,
    locate_process_recipe,
    recipes_for_facility,
)
from mm_assist.platform.grab import prepare_grabber, refresh_and_grab
from mm_assist.platform.match import match_template, roi_slice
from mm_assist.platform.window import WindowInfo

LEDGER_DIR = ROOT / "logs" / "benchmarks" / "process"
SCHEMA = "process_sense_live.v1"


def _refresh_and_grab_live(cfg: AppConfig, win: WindowInfo):
    from mm_assist.benchmarks.live_click import prepare_live_view

    prepare_live_view(win)
    return refresh_and_grab(cfg, win)


def run_live_facility_probe(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    prepare_grabber(cfg)
    _, frame = _refresh_and_grab_live(cfg, win)
    base = root or ROOT
    page = detect_facility_page(frame, root=base)
    btn = locate_process_button(frame, root=base)
    rows: list[dict[str, Any]] = []
    fid = str(page.get("facility") or "")
    if page.get("ok") and fid in RECIPE_FACILITIES:
        for rec in recipes_for_facility(fid, root=base):
            name = str(rec.get("name") or "")
            variant = rec.get("variant")
            var = str(variant) if variant else None
            hit = locate_process_recipe(
                frame, facility=fid, name=name, variant=var, root=base
            )
            rows.append(
                {
                    "id": rec.get("id"),
                    "name": name,
                    "variant": var,
                    "ok": bool(hit.get("ok")),
                    "src": hit.get("src"),
                    "error": hit.get("error"),
                    "gray": hit.get("gray"),
                }
            )
    found = sum(1 for r in rows if r.get("ok"))
    out = {
        "schema": SCHEMA,
        "mode": "process-live",
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "page": page,
        "process_btn": {
            "ok": bool(btn.get("ok")),
            "src": btn.get("src"),
            "error": btn.get("error"),
        },
        "recipes": rows,
        "found": found,
        "total": len(rows),
        "ok": bool(page.get("ok")) and bool(btn.get("ok")) and found == len(rows) and bool(rows),
        "note": "可见 locate ≠ 可投料",
    }
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    path = LEDGER_DIR / f"process_live_{out['ts']}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out["path"] = str(path)
    return out


def _roi_rect(frame: np.ndarray, roi: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    _, x0, y0 = roi_slice(frame, roi)
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    x1 = min(w, int((x + rw) * w))
    y1 = min(h, int((y + rh) * h))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def expand_row_box(
    frame: np.ndarray,
    bbox: dict[str, int],
    roi: tuple[float, float, float, float],
    *,
    include_icon: bool = True,
) -> tuple[int, int, int, int]:
    x = int(bbox["x"])
    y = int(bbox["y"])
    w = int(bbox["w"])
    h = int(bbox["h"])
    pad = max(4, h // 6)
    # 右上是卡片：图标在文字上方，不是左侧行图标。
    if include_icon:
        x0 = x - pad
        y0 = y - int(max(h, 18) * 3.4)
        x1 = x + w + pad
        y1 = y + h + pad
    else:
        x0 = x - pad
        y0 = y - pad
        x1 = x + w + pad
        y1 = y + h + pad
    rx, ry, rw, rh = _roi_rect(frame, roi)
    x0 = max(rx, x0)
    y0 = max(ry, y0)
    x1 = min(rx + rw, x1)
    y1 = min(ry + rh, y1)
    return x0, y0, max(4, x1 - x0), max(4, y1 - y0)


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"PNG 编码失败 {path}")
    tmp = path.with_suffix(".png.tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(path)


def _annotate(frame: np.ndarray, boxes: list[tuple[str, tuple[int, int, int, int]]]) -> np.ndarray:
    out = frame.copy()
    for label, (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 80), 2)
        cv2.putText(
            out,
            label[:24],
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 80),
            1,
            cv2.LINE_AA,
        )
    return out


def _btn_is_here(hit: dict[str, Any]) -> bool:
    """文案已是「加工」= 在设施前，禁止再点。"""
    if not hit.get("ok"):
        return False
    if hit.get("label") == "加工":
        return True
    if hit.get("at_facility") is True:
        return True
    return str(hit.get("variant_id") or "").endswith("_here")


def _btn_is_away_go(hit: dict[str, Any]) -> bool:
    if not hit.get("ok"):
        return False
    if _btn_is_here(hit):
        return False
    if hit.get("label") == "前往加工":
        return True
    if hit.get("at_facility") is False:
        return True
    return str(hit.get("variant_id") or "").endswith("_away")


def _wait_process_here(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    root: Path | None,
    timeout_s: float = 70.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """走路期间菜单可能关掉；等到再弹出且判定在设施前。"""
    deadline = time.time() + float(timeout_s)
    frame: np.ndarray | None = None
    hit: dict[str, Any] = {}
    while time.time() < deadline:
        _, frame = _refresh_and_grab_live(cfg, win)
        hit = locate_process_button(frame, root=root, prefer_ocr=True)
        if _btn_is_here(hit):
            return frame, hit
        time.sleep(1.0)
    if frame is None:
        _, frame = _refresh_and_grab_live(cfg, win)
        hit = locate_process_button(frame, root=root, prefer_ocr=True)
    return frame, hit


def _persist_process_btn_crop(
    frame: np.ndarray,
    btn_hit: dict[str, Any],
    specs: dict[str, Any],
    boxes: list[tuple[str, tuple[int, int, int, int]]],
    base: Path,
) -> dict[str, Any]:
    btn_row: dict[str, Any] = {
        "id": "process_btn",
        "ok": False,
        "src": btn_hit.get("src"),
        "error": btn_hit.get("error"),
    }
    vid = str(btn_hit.get("variant_id") or "")
    if not vid:
        return btn_row
    btn_spec = specs.get(vid) or specs.get("process_btn")
    if btn_spec is not None and btn_hit.get("ok") and isinstance(btn_hit.get("bbox"), dict):
        box = expand_row_box(frame, btn_hit["bbox"], PROCESS_BUTTON_ROI, include_icon=False)
        crop = save_menu_button_crop(frame, btn_spec, box, root=base)
        Path(crop["json"]).write_text(
            json.dumps(
                {
                    **json.loads(Path(crop["json"]).read_text(encoding="utf-8")),
                    "pending_review": True,
                    "locate_src": btn_hit.get("src"),
                    "variant_id": btn_hit.get("variant_id"),
                    "at_facility": btn_hit.get("at_facility"),
                    "materials_ok": btn_hit.get("materials_ok"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        boxes.append(("加工", box))
        btn_row.update(
            {
                "ok": True,
                "png": crop["png"],
                "box": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]},
                "variant_id": btn_hit.get("variant_id"),
                "at_facility": btn_hit.get("at_facility"),
                "materials_ok": btn_hit.get("materials_ok"),
            }
        )
    return btn_row


def run_auto_crop_current_page(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    root: Path | None = None,
    walk_to_facility: bool = False,
) -> dict[str, Any]:
    """当前客户区套裁配方行/加工钮。walk_to_facility 时只点「前往加工」走路，不点投料/领取。"""
    prepare_grabber(cfg)
    _, frame = _refresh_and_grab_live(cfg, win)
    base = root or ROOT
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    review_dir = LEDGER_DIR / f"auto_{ts}"
    page = detect_facility_page(frame, root=base)
    specs = {s.id: s for s in process_page_buttons(root=base)}
    saved: list[dict[str, Any]] = []
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    fid = str(page.get("facility") or "")
    if page.get("ok") and fid in RECIPE_FACILITIES:
        for rec in recipes_for_facility(fid, root=base):
            rid = str(rec.get("id") or "")
            spec = specs.get(rid)
            name = str(rec.get("name") or "")
            variant = rec.get("variant")
            var = str(variant) if variant else None
            hit = locate_process_recipe(
                frame, facility=fid, name=name, variant=var, root=base, prefer_ocr=True
            )
            row: dict[str, Any] = {
                "id": rid,
                "ok": False,
                "src": hit.get("src"),
                "error": hit.get("error"),
            }
            if spec is None:
                row["error"] = "missing_spec"
                saved.append(row)
                continue
            if not hit.get("ok") or not isinstance(hit.get("bbox"), dict):
                saved.append(row)
                continue
            box = expand_row_box(frame, hit["bbox"], FACILITY_LIST_ROI, include_icon=True)
            crop = save_menu_button_crop(frame, spec, box, root=base)
            crop["pending_review"] = True
            crop["locate_src"] = hit.get("src")
            Path(crop["json"]).write_text(
                json.dumps({**json.loads(Path(crop["json"]).read_text(encoding="utf-8")), "pending_review": True, "locate_src": hit.get("src")}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            boxes.append((str(spec.label), box))
            row.update({"ok": True, "png": crop["png"], "box": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]}, "click_client": hit.get("click_client")})
            saved.append(row)
    if (
        os.environ.get("MM_LIVE_BENCHMARK", "").strip() in {"1", "true", "yes"}
        and page.get("ok")
        and fid in RECIPE_FACILITIES
    ):
        pick = next(
            (r for r in saved if "鐵錠_iron_ore" in str(r.get("id") or "") and r.get("ok") and r.get("click_client")),
            None,
        ) or next((r for r in saved if r.get("ok") and r.get("click_client")), None)
        if pick:
            from mm_assist.benchmarks.live_click import run_one_click

            xy = pick["click_client"]
            clicked = run_one_click(
                win,
                int(xy[0]),
                int(xy[1]),
                method="sendinput",
                need_ui="facility",
                stage=f"process_nav:select_recipe:{fid}",
                fail_count=0,
            )
            if clicked.get("ok"):
                time.sleep(0.8)
                _, frame = _refresh_and_grab_live(cfg, win)
    btn_hit = locate_process_button(frame, root=base, prefer_ocr=True)
    saved.append(_persist_process_btn_crop(frame, btn_hit, specs, boxes, base))
    walked = False
    walk_err: str | None = None
    live = os.environ.get("MM_LIVE_BENCHMARK", "").strip() in {"1", "true", "yes"}
    if live and walk_to_facility:
        from mm_assist.benchmarks.live_click import run_one_click

        if _btn_is_here(btn_hit):
            walked = True
        elif _btn_is_away_go(btn_hit):
            xy = btn_hit.get("click_client") or [0, 0]
            for fail_count in range(2):
                now = locate_process_button(frame, root=base, prefer_ocr=True)
                if _btn_is_here(now):
                    btn_hit = now
                    walked = True
                    break
                if not _btn_is_away_go(now):
                    frame, btn_hit = _wait_process_here(cfg, win, root=base, timeout_s=70.0)
                    walked = _btn_is_here(btn_hit)
                    if walked:
                        break
                    if fail_count >= 1:
                        walk_err = "walk_timeout"
                        break
                    continue
                xy = now.get("click_client") or xy
                clicked = run_one_click(
                    win,
                    int(xy[0]),
                    int(xy[1]),
                    method="sendinput",
                    need_ui="facility",
                    stage="process_nav:walk_to_facility",
                    fail_count=fail_count,
                )
                if not clicked.get("ok"):
                    walk_err = str(clicked.get("error") or "walk_click_failed")
                    break
                time.sleep(1.2)
                frame, btn_hit = _wait_process_here(cfg, win, root=base, timeout_s=70.0)
                if _btn_is_here(btn_hit):
                    walked = True
                    break
            if walked:
                saved.append(_persist_process_btn_crop(frame, btn_hit, specs, boxes, base))
        else:
            walk_err = "no_goto_process_btn"
    claim = locate_claim_all(frame, root=base, prefer_ocr=True)
    claim_row: dict[str, Any] = {
        "id": "claim_all",
        "ok": False,
        "src": claim.get("src"),
        "error": claim.get("error"),
        "at_facility_ui_same": True,
    }
    claim_spec = specs.get("claim_all")
    if claim_spec is not None and claim.get("ok") and isinstance(claim.get("bbox"), dict):
        box = expand_row_box(frame, claim["bbox"], CLAIM_ALL_ROI, include_icon=False)
        crop = save_menu_button_crop(frame, claim_spec, box, root=base)
        Path(crop["json"]).write_text(
            json.dumps(
                {
                    **json.loads(Path(crop["json"]).read_text(encoding="utf-8")),
                    "pending_review": True,
                    "at_facility_ui_same": True,
                    "note": "在/不在设施前 UI 无差",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        boxes.append(("全部領取", box))
        claim_row.update({"ok": True, "png": crop["png"], "box": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]}})
    saved.append(claim_row)
    materials = ""
    try:
        from mm_assist.perception.process_sense import HOLD_NEED_ROI, _blob, _ocr_lines

        materials = _blob(_ocr_lines(frame, HOLD_NEED_ROI))
    except Exception:
        materials = ""
    dismissed = False
    if os.environ.get("MM_LIVE_BENCHMARK", "").strip() in {"1", "true", "yes"}:
        go = locate_process_button(frame, root=base, prefer_ocr=True)
        if go.get("ok"):
            from mm_assist.benchmarks.live_click import run_one_click

            h, w = int(frame.shape[0]), int(frame.shape[1])
            ox, oy = int(w * 0.26), int(h * 0.48)
            clicked = run_one_click(
                win,
                ox,
                oy,
                method="sendinput",
                need_ui="facility",
                stage=f"process_nav:dismiss_detail:{fid or 'x'}",
                fail_count=0,
            )
            if clicked.get("ok"):
                time.sleep(0.45)
                _, after = _refresh_and_grab_live(cfg, win)
                gone = not locate_process_button(after, root=base, prefer_ocr=True).get("ok")
                dismissed = gone
                if not gone:
                    clicked2 = run_one_click(
                        win,
                        ox,
                        oy,
                        method="sendinput",
                        need_ui="facility",
                        stage=f"process_nav:dismiss_detail:{fid or 'x'}",
                        fail_count=1,
                    )
                    if clicked2.get("ok"):
                        time.sleep(0.45)
                        _, after = _refresh_and_grab_live(cfg, win)
                        dismissed = not locate_process_button(
                            after, root=base, prefer_ocr=True
                        ).get("ok")
    frame_path = review_dir / "frame.png"
    anno_path = review_dir / "annotated.png"
    _write_png(frame_path, frame)
    _write_png(anno_path, _annotate(frame, boxes))
    ok_n = sum(1 for r in saved if r.get("ok"))
    out = {
        "schema": "process_sense_auto_crop.v1",
        "mode": "process-auto",
        "ts": ts,
        "page": page,
        "saved": saved,
        "ok_count": ok_n,
        "total": len(saved),
        "frame": str(frame_path),
        "annotated": str(anno_path),
        "materials_ocr": materials,
        "dismissed": dismissed,
        "walked": walked,
        "walk_error": walk_err,
        "note": (
            "OCR 套裁须人工验收 PNG；可见 ≠ 可投料。"
            + (" 已点「前往加工」走路采设施前钮态；未点「加工」投料、未点全部領取。" if walk_to_facility else " 未点前往加工。")
        ),
        "ok": bool(page.get("ok")) and ok_n > 0,
    }
    review_dir.mkdir(parents=True, exist_ok=True)
    ledger = review_dir / "crop.json"
    ledger.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out["path"] = str(ledger)
    return out


def _hub_visible(frame: np.ndarray, *, root: Path | None = None) -> bool:
    """设施页里的设备立绘会撞上枢纽卡图，不能单靠卡匹配。"""
    if detect_facility_page(frame, root=root).get("ok"):
        return False
    return any(
        bool(locate_hub_card(frame, cid, root=root).get("ok")) for cid in HUB_CARD_IDS
    )


def _xy_template(
    frame: np.ndarray,
    rel: str,
    roi: tuple[float, float, float, float] | None,
    *,
    root: Path | None = None,
    threshold: float = 0.72,
) -> tuple[int, int] | None:
    path = (root or ROOT) / rel
    if not path.is_file():
        return None
    hit = match_template(frame, path, roi=roi, threshold=threshold, min_margin=0.02)
    if hit is None:
        return None
    return int(hit.cx), int(hit.cy)


def _click_verify(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    need_ui: str,
    stage: str,
    method: str,
    locate: Callable[[np.ndarray], tuple[int, int] | None],
    ok_fn: Callable[[np.ndarray], bool],
) -> tuple[np.ndarray, bool]:
    from mm_assist.benchmarks.live_click import run_one_click

    _, frame = _refresh_and_grab_live(cfg, win)
    if ok_fn(frame):
        return frame, True
    for fail_count in range(2):
        xy = locate(frame)
        if xy is None:
            return frame, False
        clicked = run_one_click(
            win,
            int(xy[0]),
            int(xy[1]),
            method=method,
            need_ui=need_ui,
            stage=stage,
            fail_count=fail_count,
        )
        if not clicked.get("ok"):
            _, frame = _refresh_and_grab_live(cfg, win)
            return frame, False
        time.sleep(0.8)
        _, frame = _refresh_and_grab_live(cfg, win)
        if ok_fn(frame):
            return frame, True
    return frame, False


def run_auto_nav_and_crop(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    method: str = "sendinput",
    root: Path | None = None,
    walk_to_facility: bool = False,
) -> dict[str, Any]:
    """菜单→製作枢纽→设施：点开后套裁。walk_to_facility 时只点「前往加工」走路，不点投料。"""
    if os.environ.get("MM_LIVE_BENCHMARK", "").strip() not in {"1", "true", "yes"}:
        raise RuntimeError("导航采图须 MM_LIVE_BENCHMARK=1")
    from mm_assist.benchmarks.transition_accept import (
        confirm_node_from_frame,
        load_manifest,
        locate_click_point,
        resolve_locator,
    )

    base = root or ROOT
    prepare_grabber(cfg)
    log: list[str] = []
    crops: list[dict[str, Any]] = []

    def grab() -> np.ndarray:
        _, img = _refresh_and_grab_live(cfg, win)
        return img

    frame = grab()
    origin = (int(win.left), int(win.top))
    manifest = load_manifest(root=base)

    def open_menu(img: np.ndarray) -> tuple[int, int] | None:
        loc = resolve_locator(
            "main_hud_menu_system_menu", skill="woodcut", manifest=manifest, root=base
        )
        bl = locate_click_point(loc, img, root=base, screen_origin=origin)
        bclick = bl.get("click") if isinstance(bl.get("click"), dict) else None
        if not bclick or not (bl.get("accepted") or bl.get("feature_ok")):
            return None
        bc = bclick.get("client") or [0, 0]
        return int(bc[0]), int(bc[1])

    def open_craft(img: np.ndarray) -> tuple[int, int] | None:
        return _xy_template(
            img,
            "knowledge/templates/menu/icon_crafting.png",
            (0.68, 0.08, 0.28, 0.36),
            root=base,
            threshold=0.62,
        )

    def back_xy(img: np.ndarray) -> tuple[int, int] | None:
        return _xy_template(
            img,
            "knowledge/templates/facility/back.png",
            (0.0, 0.0, 0.18, 0.16),
            root=base,
            threshold=0.72,
        )

    if detect_facility_page(frame, root=base).get("ok"):
        log.append("boot=facility")
    elif _hub_visible(frame, root=base):
        log.append("boot=hub")
    elif confirm_node_from_frame(frame, "system_menu", root=base):
        log.append("boot=system_menu")
        frame, ok = _click_verify(
            cfg,
            win,
            need_ui="system_menu",
            stage="process_nav:system_menu_craft_craft_hub",
            method=method,
            locate=open_craft,
            ok_fn=lambda im: _hub_visible(im, root=base),
        )
        if not ok:
            page = detect_facility_page(frame, root=base)
            return {
                "ok": False,
                "error": page.get("error") or "craft_hub_not_reached",
                "log": log,
                "note": "製作边未 verified；若进了 K 协助则停",
            }
        log.append("opened_hub")
    else:
        log.append("boot=main_or_other")
        frame, ok = _click_verify(
            cfg,
            win,
            need_ui="main_hud",
            stage="process_nav:main_hud_menu_system_menu",
            method=method,
            locate=open_menu,
            ok_fn=lambda im: confirm_node_from_frame(im, "system_menu", root=base),
        )
        if not ok:
            return {"ok": False, "error": "system_menu_not_reached", "log": log}
        frame, ok = _click_verify(
            cfg,
            win,
            need_ui="system_menu",
            stage="process_nav:system_menu_craft_craft_hub",
            method=method,
            locate=open_craft,
            ok_fn=lambda im: _hub_visible(im, root=base),
        )
        if not ok:
            page = detect_facility_page(frame, root=base)
            return {
                "ok": False,
                "error": page.get("error") or "craft_hub_not_reached",
                "log": log,
                "note": "製作边未 verified；若进了 K 协助则停",
            }
        log.append("opened_hub")

    want_facilities: tuple[str, ...] = RECIPE_FACILITIES
    if walk_to_facility:
        page0 = detect_facility_page(frame, root=base)
        fid0 = str(page0.get("facility") or "")
        if page0.get("ok") and fid0 in RECIPE_FACILITIES:
            want_facilities = (fid0,)
        else:
            want_facilities = ("metal",)

    for fid in want_facilities:
        frame = grab()
        page = detect_facility_page(frame, expected=fid, root=base)
        if not page.get("ok"):
            if detect_facility_page(frame, root=base).get("ok"):
                frame, ok = _click_verify(
                    cfg,
                    win,
                    need_ui="facility",
                    stage=f"process_nav:facility_back:{fid}",
                    method=method,
                    locate=back_xy,
                    ok_fn=lambda im: _hub_visible(im, root=base),
                )
                if not ok:
                    return {"ok": False, "error": "facility_back_failed", "log": log, "crops": crops}
            if not _hub_visible(frame, root=base):
                return {"ok": False, "error": "hub_missing_before_card", "log": log, "crops": crops}

            def card_xy(img: np.ndarray, card=fid) -> tuple[int, int] | None:
                hit = locate_hub_card(img, card, root=base)
                if not hit.get("ok"):
                    return None
                xy = hit.get("click_client") or [0, 0]
                return int(xy[0]), int(xy[1])

            frame, ok = _click_verify(
                cfg,
                win,
                need_ui="craft_hub",
                stage=f"process_nav:craft_hub_card:{fid}",
                method=method,
                locate=card_xy,
                ok_fn=lambda im, want=fid: bool(
                    detect_facility_page(im, expected=want, root=base).get("ok")
                ),
            )
            if not ok:
                return {
                    "ok": False,
                    "error": f"facility_not_reached:{fid}",
                    "log": log,
                    "crops": crops,
                    "page": detect_facility_page(frame, root=base),
                }
        crop = run_auto_crop_current_page(cfg, win, root=base, walk_to_facility=walk_to_facility)
        crops.append({"facility": fid, **{k: crop.get(k) for k in ("ok", "ok_count", "total", "path", "annotated", "walked", "walk_error")}})
        log.append(f"crop:{fid}:{crop.get('ok_count')}/{crop.get('total')}")
        if walk_to_facility:
            log.append("stop_after_walk")
            break
        frame, ok = _click_verify(
            cfg,
            win,
            need_ui="facility",
            stage=f"process_nav:facility_back_after:{fid}",
            method=method,
            locate=back_xy,
            ok_fn=lambda im: _hub_visible(im, root=base),
        )
        if not ok:
            log.append(f"back_after_{fid}_failed")
            break

    ok_all = all(c.get("ok") for c in crops) and (
        len(crops) == (1 if walk_to_facility else 3)
    )
    return {
        "ok": ok_all,
        "log": log,
        "crops": crops,
        "note": (
            "可见 locate ≠ 可投料。"
            + (" 已点前往加工走路；未点加工投料/全部領取。" if walk_to_facility else " 未点加工钮。")
        ),
    }


def format_auto_report(payload: dict[str, Any]) -> str:
    if payload.get("crops") is not None:
        lines = [
            f"process-auto-nav ok={payload.get('ok')} error={payload.get('error') or '-'}",
            f"log={payload.get('log')}",
            str(payload.get("note") or ""),
        ]
        for row in payload.get("crops") or []:
            lines.append(
                f"  {row.get('facility')} saved={row.get('ok_count')}/{row.get('total')} path={row.get('path')}"
            )
        return "\n".join(lines) + "\n"
    page = payload.get("page") or {}
    lines = [
        f"process-auto ok={payload.get('ok')} page={page.get('facility') or page.get('error')} "
        f"saved={payload.get('ok_count')}/{payload.get('total')}",
        f"annotated={payload.get('annotated')}",
        str(payload.get("note") or ""),
    ]
    for row in payload.get("saved") or []:
        tag = "ok" if row.get("ok") else (row.get("error") or "miss")
        lines.append(f"  {row.get('id')} {tag}")
    return "\n".join(lines) + "\n"


def format_report(payload: dict[str, Any]) -> str:
    page = payload.get("page") or {}
    btn = payload.get("process_btn") or {}
    lines = [
        f"process-live ok={payload.get('ok')} page={page.get('facility') or page.get('error')} "
        f"btn={btn.get('src') or btn.get('error')} recipes={payload.get('found')}/{payload.get('total')}",
        f"path={payload.get('path')}",
        str(payload.get("note") or ""),
    ]
    for row in payload.get("recipes") or []:
        tag = "ok" if row.get("ok") else (row.get("error") or "miss")
        src = row.get("src") or ""
        gray = " gray" if row.get("gray") else ""
        lines.append(f"  {row.get('id')} {tag} {src}{gray}")
    return "\n".join(lines) + "\n"
