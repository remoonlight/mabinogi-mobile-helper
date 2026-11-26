"""主界面入口与右下角状态控件的方法 B 采证。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from mm_assist.config import ROOT
from mm_assist.perception.hud_sense import TRAVEL_ROI
from mm_assist.perception.rois import HUD_SEEDS
from mm_assist.perception.hud_entry_locator import locate_hud_entry, safe_capture_id
from mm_assist.jobs.pairs import load_pair_job_json
from mm_assist.jobs.bar import open_test_job

from mm_assist.knowledge_paths import HUD_ENTRY_JSON, HUD_ENTRY_REVIEW

CATALOG_REL = HUD_ENTRY_JSON
REVIEW_REL = HUD_ENTRY_REVIEW
ENTRY_KINDS = frozenset({"page_entry", "state_control", "observe_only"})
FRAME_ROLES = frozenset({"source", "target"})
ENTRY_ORDER = ("menu", "bag", "avatar", "craft", "map", "combat", "stop")
# 契约用语义名；实机 job.icons.key 为短名 esc/inv/lb
_ICON_KEYS: dict[str, tuple[str, ...]] = {
    "menu_esc": ("menu_esc", "esc"),
    "inventory": ("inventory", "inv"),
    "left_bottom": ("left_bottom", "lb"),
    "stop": ("stop",),
}


def catalog_path(*, root: Path | None = None) -> Path:
    return (root or ROOT) / CATALOG_REL


def review_path(*, root: Path | None = None) -> Path:
    return (root or ROOT) / REVIEW_REL


def load_main_hud_catalog(*, root: Path | None = None) -> dict[str, Any]:
    data = json.loads(catalog_path(root=root).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("main_hud_entry_capture 须为 object")
    validate_main_hud_catalog(data)
    return data


def validate_main_hud_catalog(data: dict[str, Any]) -> None:
    if str(data.get("schema") or "") != "main_hud_entry_capture.v1":
        raise ValueError("未知 schema")
    src = data.get("source")
    if not isinstance(src, dict) or str(src.get("id")) != "main_hud":
        raise ValueError("source 须为 main_hud")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries 须为非空 list")
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("entry 须为 object")
        eid = str(item.get("id") or "").strip()
        if not eid:
            raise ValueError("entry 缺 id")
        if eid in seen:
            raise ValueError(f"重复 entry={eid!r}")
        seen.add(eid)
        kind = str(item.get("kind") or "")
        if kind not in ENTRY_KINDS:
            raise ValueError(f"entry {eid} 未知 kind={kind!r}")
        locator = item.get("locator")
        if not isinstance(locator, dict):
            raise ValueError(f"entry {eid} 缺 locator")
        detector = str(locator.get("detector") or "")
        if detector not in {"template", "stop_shape"}:
            raise ValueError(f"entry {eid} 未知 locator.detector={detector!r}")
        maturity = str(locator.get("maturity") or "")
        if maturity not in {"calibration", "accepted"}:
            raise ValueError(f"entry {eid} 未知 locator.maturity={maturity!r}")
        if detector == "template" and not str(locator.get("template") or "").strip():
            raise ValueError(f"entry {eid} template locator 缺 template")
        if detector == "template" and maturity == "accepted":
            template_path = Path(str(locator["template"]))
            if not template_path.is_absolute():
                template_path = ROOT / template_path
            if not template_path.is_file():
                raise ValueError(
                    f"entry {eid} accepted template 不存在: {template_path}"
                )
        tgt = item.get("target")
        if not isinstance(tgt, dict) or not tgt.get("signals"):
            raise ValueError(f"entry {eid} 缺 target.signals")
        if kind == "page_entry" and not str(item.get("edge_id") or "").strip():
            raise ValueError(f"page_entry {eid} 缺 edge_id")
    required = {"menu", "bag", "avatar", "craft", "map", "combat", "stop"}
    missing = required - seen
    if missing:
        raise ValueError(f"缺入口 {sorted(missing)}")


def entry_by_id(catalog: dict[str, Any], entry_id: str) -> dict[str, Any]:
    eid = str(entry_id or "").strip()
    for item in catalog.get("entries") or []:
        if isinstance(item, dict) and str(item.get("id")) == eid:
            return item
    raise KeyError(f"未知入口={eid!r}")


def next_entry_id(
    entry_id: str,
    queue: list[str] | tuple[str, ...] | None = None,
) -> str | None:
    """入口标定顺序中的下一项；已是最后一项则 None。"""
    eid = str(entry_id or "").strip()
    order: tuple[str, ...] | list[str]
    if queue:
        order = [str(x).strip() for x in queue if str(x).strip()]
    else:
        order = list(ENTRY_ORDER)
    if eid not in order:
        return None
    idx = order.index(eid)
    if idx + 1 >= len(order):
        return None
    return order[idx + 1]


def roi_by_id(roi_id: str | None) -> tuple[float, float, float, float] | None:
    rid = str(roi_id or "").strip()
    if rid == "hud.travel":
        return tuple(float(x) for x in TRAVEL_ROI)  # type: ignore[return-value]
    for seed in HUD_SEEDS:
        if str(seed.get("id")) == rid:
            roi = seed.get("roi")
            if isinstance(roi, list) and len(roi) == 4:
                return float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3])
    return None


def _utc_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def hud_entry_job_specs(
    entry_id: str,
    *,
    catalog: dict[str, Any] | None = None,
    capture_id: str | None = None,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    cat = catalog if catalog is not None else load_main_hud_catalog(root=root)
    entry = entry_by_id(cat, entry_id)
    cid = str(capture_id).strip() if capture_id else _utc_id(str(entry["id"]))
    src = cat["source"]
    tgt = entry["target"]
    frames = (
        ("source", src, f"主界面源态·{entry.get('label')}", src.get("need_ui")),
        ("target", tgt, f"目标态·{entry.get('label')}（人手已点）", tgt.get("need_ui")),
    )
    out: list[dict[str, Any]] = []
    for role, node, title, need in frames:
        out.append(
            {
                "action": "readonly_ui_fine",
                "method": "B",
                "need_ui": str(need),
                "title": title,
                "args": {
                    "pair_id": cid,
                    "pair_role": role,
                    "capture_id": cid,
                    "entry_id": str(entry["id"]),
                    "node_id": str(node["id"]),
                    "icon_key": entry.get("icon_key"),
                    "roi_id": entry.get("roi_id"),
                    "sense": "vision",
                    "locator_mode": "roi_center",
                },
            }
        )
    return out


def open_hud_entry_frame(
    entry_id: str,
    role: Literal["source", "target"],
    *,
    capture_id: str | None = None,
    path: Path | None = None,
    root: Path | None = None,
    catalog: dict[str, Any] | None = None,
    extra_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    role_s = str(role or "").strip().lower()
    if role_s not in FRAME_ROLES:
        raise ValueError(f"role 须为 source|target，收到 {role!r}")
    specs = hud_entry_job_specs(
        entry_id, catalog=catalog, capture_id=capture_id, root=root
    )
    spec = next(s for s in specs if str(s["args"]["pair_role"]) == role_s)
    args = dict(spec["args"])
    if extra_args:
        args.update(extra_args)
    return open_test_job(
        "readonly_ui_fine",
        method="B",
        need_ui=str(spec["need_ui"]),
        title=str(spec["title"]),
        args=args,
        path=path,
    )


def freeze_hud_entry_shot(
    win: Any,
    *,
    capture_id: str,
    role: str = "source",
    root: Path | None = None,
    capture_method: str | None = None,
) -> dict[str, Any]:
    """在点【开始采集】的同一时刻抓游戏客户区并落盘，禁止延后补截。"""
    from mm_assist.platform.capture import grab_client
    from mm_assist.platform.window import refresh_window

    role_s = str(role or "source").strip().lower()
    if role_s not in FRAME_ROLES:
        raise ValueError(f"role 须为 source|target，收到 {role!r}")
    cid = safe_capture_id(capture_id)
    current = refresh_window(win)
    img = grab_client(current, method=capture_method)
    if img is None or getattr(img, "size", 0) == 0:
        raise RuntimeError("按下采集时抓图为空")
    repo = root or ROOT
    dest = repo / "knowledge" / "captures" / "hud_entries" / cid / f"{role_s}.raw.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("冻结帧 PNG 编码失败")
    data = encoded.tobytes()
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    rel = str(dest.relative_to(repo)).replace("\\", "/")
    return {
        "path": rel,
        "sha256": hashlib.sha256(data).hexdigest(),
        "origin": [int(current.left), int(current.top)],
        "client": {"w": int(img.shape[1]), "h": int(img.shape[0])},
        "frozen_at_unix": datetime.now(timezone.utc).timestamp(),
    }


def run_hud_entry_vision(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """界面切换标定：只用按下采集时冻结的帧 + ROI OCR，禁止补截。"""
    from mm_assist.perception.hud_entry_locator import _normalized_roi
    from mm_assist.perception.ocr import ocr_roi

    entry_id = str(args.get("entry_id") or "").strip()
    if not entry_id:
        return {"ok": False, "error": "hud-entry vision 缺 entry_id"}
    frozen = str(args.get("frozen_raw") or "").strip()
    if not frozen:
        return {"ok": False, "error": "缺按下采集时的冻结帧；未在点【开始采集】时截图"}
    frozen_path = Path(frozen)
    if not frozen_path.is_absolute():
        frozen_path = (ROOT / frozen_path)
    if not frozen_path.is_file():
        return {"ok": False, "error": f"冻结帧不存在: {frozen}"}
    raw_bytes = np.fromfile(str(frozen_path), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if img is None or getattr(img, "size", 0) == 0:
        return {"ok": False, "error": "冻结帧无法解码"}
    catalog = load_main_hud_catalog()
    entry = entry_by_id(catalog, entry_id)
    locator = entry.get("locator") if isinstance(entry.get("locator"), dict) else {}
    roi = _normalized_roi(locator)
    ocr_rows: list[dict[str, Any]] = []
    if roi is not None:
        try:
            lines = ocr_roi(img, roi, min_confidence=0.35)
        except Exception as exc:
            lines = []
            ocr_rows.append({"error": str(exc)})
        else:
            ocr_rows = [
                {"text": str(line.text), "confidence": float(line.confidence)}
                for line in lines
                if str(line.text or "").strip()
            ]
    capture_id = str(args.get("capture_id") or args.get("pair_id") or "").strip()
    role = str(args.get("pair_role") or "source").strip().lower()
    detail = {
        "sense": "vision",
        "locator_mode": "roi_center",
        "ocr_roi": ocr_rows,
        "pair_id": str(args.get("pair_id") or capture_id),
        "pair_role": role,
        "entry_id": entry_id,
        "confidence": "low",
        "client": {"w": int(img.shape[1]), "h": int(img.shape[0])},
    }
    origin = args.get("frozen_origin")
    if isinstance(origin, list) and len(origin) == 2:
        screen_origin = (int(origin[0]), int(origin[1]))
    else:
        screen_origin = (
            int(getattr(getattr(ctx, "win", None), "left", 0) or 0),
            int(getattr(getattr(ctx, "win", None), "top", 0) or 0),
        )
    detail["frozen_raw"] = frozen.replace("\\", "/")
    evidence = archive_hud_entry_frame(
        entry,
        img,
        capture_id=capture_id,
        role=role,  # type: ignore[arg-type]
        detail=detail,
        screen_origin=screen_origin,
        prefer_roi_center=True,
    )
    detail["hud_entry_evidence"] = evidence
    detail["locator"] = evidence["locator"]
    return {"ok": True, "detail": detail}


def record_capture_box(
    entry_id: str,
    frame: np.ndarray,
    client_rect: tuple[int, int, int, int] | list[int],
    *,
    capture_id: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """按获取框裁出图案，写入模板并更新契约 ROI。"""
    from mm_assist.platform.hint_overlay import client_rect_to_norm, crop_client_pattern

    eid = str(entry_id or "").strip()
    cid = safe_capture_id(capture_id)
    repo = root or ROOT
    fh, fw = frame.shape[:2]
    x, y, w, h = (int(v) for v in client_rect)
    patch = crop_client_pattern(frame, (x, y, w, h))
    if patch is None:
        raise ValueError("获取框无法裁出图案")
    catalog = load_main_hud_catalog(root=repo)
    entry = entry_by_id(catalog, eid)
    locator = entry.get("locator") if isinstance(entry.get("locator"), dict) else {}
    rel = str(locator.get("template") or f"knowledge/templates/hud_entries/{eid}.png")
    template_path = Path(rel)
    if not template_path.is_absolute():
        template_path = repo / template_path
    template_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", patch)
    if not ok:
        raise RuntimeError("图案 PNG 编码失败")
    data = encoded.tobytes()
    tmp = template_path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(template_path)
    cap_path = repo / "knowledge" / "captures" / "hud_entries" / cid / "source.pattern.png"
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    cap_tmp = cap_path.with_suffix(".tmp")
    cap_tmp.write_bytes(data)
    cap_tmp.replace(cap_path)
    roi = client_rect_to_norm((x, y, w, h), frame_w=fw, frame_h=fh)
    box_payload = {
        "entry_id": eid,
        "capture_id": cid,
        "client_rect": [x, y, w, h],
        "roi": [round(v, 6) for v in roi],
        "template": str(template_path.relative_to(repo)).replace("\\", "/"),
        "pattern_path": str(cap_path.relative_to(repo)).replace("\\", "/"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": [int(patch.shape[1]), int(patch.shape[0])],
    }
    box_path = cap_path.with_name("source.box.json")
    box_tmp = box_path.with_suffix(".tmp")
    box_tmp.write_text(
        json.dumps(box_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    box_tmp.replace(box_path)
    return box_payload


def archive_hud_entry_frame(
    entry: dict[str, Any],
    frame: np.ndarray,
    *,
    capture_id: str,
    role: Literal["source", "target"],
    detail: dict[str, Any],
    root: Path | None = None,
    screen_origin: tuple[int, int] = (0, 0),
    prefer_roi_center: bool = False,
) -> dict[str, Any]:
    """保存未叠加红点的原始帧、定位结果与哈希绑定。"""
    cid = safe_capture_id(capture_id)
    role_s = str(role or "").strip().lower()
    if role_s not in FRAME_ROLES:
        raise ValueError(f"role 须为 source|target，收到 {role!r}")
    repo = root or ROOT
    dest_dir = repo / "knowledge" / "captures" / "hud_entries" / cid
    dest_dir.mkdir(parents=True, exist_ok=True)

    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("hud-entry 原始帧 PNG 编码失败")
    raw_bytes = encoded.tobytes()
    raw_path = dest_dir / f"{role_s}.raw.png"
    raw_tmp = raw_path.with_suffix(".tmp")
    raw_tmp.write_bytes(raw_bytes)
    raw_tmp.replace(raw_path)
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    locator = locate_hud_entry(
        entry,
        frame,
        root=repo,
        screen_origin=screen_origin,
        prefer_roi_center=prefer_roi_center,
    )
    payload = {
        "schema": "hud_entry_evidence.v1",
        "capture_id": cid,
        "entry_id": str(entry.get("id") or ""),
        "role": role_s,
        "raw_path": str(raw_path.relative_to(repo)).replace("\\", "/"),
        "raw_sha256": raw_sha256,
        "locator": locator,
        "detail": detail,
    }
    result_path = dest_dir / f"{role_s}.result.json"
    result_tmp = result_path.with_suffix(".tmp")
    result_tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result_tmp.replace(result_path)
    return {
        "entry_id": payload["entry_id"],
        "capture_id": cid,
        "role": role_s,
        "raw_path": payload["raw_path"],
        "raw_sha256": raw_sha256,
        "result_path": str(result_path.relative_to(repo)).replace("\\", "/"),
        "locator": locator,
    }


def archive_hud_entry_job(
    job: dict[str, Any],
    *,
    root: Path | None = None,
) -> Path:
    """把完成后及确认更新后的完整 job 固化到同一证据目录。"""
    args = job.get("args") if isinstance(job.get("args"), dict) else {}
    entry_id = str(args.get("entry_id") or "").strip()
    role = str(args.get("pair_role") or "").strip().lower()
    capture_id = safe_capture_id(
        str(args.get("capture_id") or args.get("pair_id") or "")
    )
    if not entry_id:
        raise ValueError("hud-entry job 缺 entry_id")
    if role not in FRAME_ROLES:
        raise ValueError(f"hud-entry job role 非法：{role!r}")
    if str(job.get("method") or "").upper() != "B":
        raise ValueError("hud-entry job 必须 method=B")
    if str(job.get("action") or "") != "readonly_ui_fine":
        raise ValueError("hud-entry job action 必须 readonly_ui_fine")

    repo = root or ROOT
    dest = (
        repo
        / "knowledge"
        / "captures"
        / "hud_entries"
        / capture_id
        / f"{role}.job.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(job, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(dest)
    return dest


def _job_probe(job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    probe = result.get("panel_probe")
    return probe if isinstance(probe, dict) else {}


def _job_icons(job: dict[str, Any]) -> list[dict[str, Any]]:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    icons = result.get("icons")
    if isinstance(icons, list):
        return [x for x in icons if isinstance(x, dict)]
    return []


def _match_signals(probe: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    misses: list[str] = []
    for key, want in expected.items():
        got = probe.get(key)
        if got is None:
            misses.append(f"{key}:want={want!r} got={got!r}")
            continue
        if isinstance(want, bool):
            matched = isinstance(got, bool) and got is want
        else:
            matched = got == want
        if not matched:
            misses.append(f"{key}:want={want!r} got={got!r}")
    return misses


def _icon_row(icons: list[dict[str, Any]], key: str | None) -> dict[str, Any] | None:
    if not key:
        return None
    aliases = _ICON_KEYS.get(key, (key,))
    names = {key, *aliases}
    for row in icons:
        got = str(row.get("key") or row.get("name") or "")
        if got in names:
            return row
    return None


def _combat_icon_row(icons: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in icons:
        if str(row.get("state") or "") == "Combat":
            return row
    return None


def clickable_status(entry: dict[str, Any], source_job: dict[str, Any]) -> dict[str, Any]:
    """源态是否允许请用户点。不可点则 gap，不强行验收点击。"""
    probe = _job_probe(source_job)
    result = source_job.get("result") if isinstance(source_job.get("result"), dict) else {}
    icons = _job_icons(source_job)
    icon_key = entry.get("icon_key")
    row = _icon_row(icons, str(icon_key) if icon_key else None)
    roi = roi_by_id(str(entry.get("roi_id") or "") or None)
    kind = str(entry.get("kind"))
    if kind == "state_control" and str(entry.get("id")) == "stop":
        stop_vis = result.get("stop_visible")
        if stop_vis is None:
            stop_vis = probe.get("stop_visible")
        if stop_vis is not True:
            return {
                "allow_click": False,
                "reason": "stop_not_visible",
                "visible": False,
                "interactable": False,
                "roi": roi,
            }
        interactable = True
        if row is not None and row.get("interactable") is False:
            interactable = False
        return {
            "allow_click": interactable,
            "reason": "stop_visible" if interactable else "stop_not_interactable",
            "visible": True,
            "interactable": interactable,
            "roi": roi,
        }
    if kind == "state_control" and str(entry.get("id")) == "combat":
        crow = row or _combat_icon_row(icons)
        visible = None
        interactable = None
        if crow is not None:
            visible = str(crow.get("state") or "") in {"Show", "Combat"} or crow.get("visible") is True
            interactable = bool(crow.get("interactable") is True)
        if interactable is True:
            return {
                "allow_click": True,
                "reason": "combat_interactable",
                "visible": bool(visible),
                "interactable": True,
                "roi": roi,
                "icon": crow,
            }
        return {
            "allow_click": False,
            "reason": "combat_observe_only",
            "visible": visible,
            "interactable": interactable,
            "roi": roi,
            "icon": crow,
        }
    if row is not None and row.get("interactable") is False:
        return {
            "allow_click": False,
            "reason": "icon_not_interactable",
            "visible": True,
            "interactable": False,
            "roi": roi,
        }
    if roi is None and row is None:
        return {
            "allow_click": False,
            "reason": "no_roi_or_icon",
            "visible": None,
            "interactable": None,
            "roi": None,
        }
    return {
        "allow_click": True,
        "reason": "roi_or_icon_ok",
        "visible": True,
        "interactable": True if row is None else bool(row.get("interactable") is True),
        "roi": roi,
        "icon": row,
    }


def assess_hud_entry_captures(
    entry: dict[str, Any],
    source_job: dict[str, Any],
    target_job: dict[str, Any] | None,
    *,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    src_sig = catalog.get("source", {}).get("signals") or {}
    tgt_sig = (entry.get("target") or {}).get("signals") or {}
    pa = _job_probe(source_job)
    src_miss = _match_signals(pa, src_sig)
    click = clickable_status(entry, source_job)
    fail_tags: list[str] = []
    if src_miss:
        fail_tags.append("hud:source_not_clear")
    if not click.get("allow_click"):
        fail_tags.append(f"hud:{click.get('reason')}")
    tgt_miss: list[str] = []
    if target_job is None:
        if click.get("allow_click") and not src_miss:
            fail_tags.append("hud:target_missing")
    else:
        pb = _job_probe(target_job)
        tgt_miss = _match_signals(pb, tgt_sig)
        if tgt_miss:
            fail_tags.append("hud:target_signal_mismatch")
    pair_ok = bool(
        not src_miss
        and click.get("allow_click")
        and target_job is not None
        and not tgt_miss
    )
    source_result = (
        source_job.get("result") if isinstance(source_job.get("result"), dict) else {}
    )
    locator = (
        source_result.get("locator")
        if isinstance(source_result.get("locator"), dict)
        else {}
    )
    overlay_evidence = (
        source_result.get("overlay_evidence")
        if isinstance(source_result.get("overlay_evidence"), dict)
        else {}
    )
    overlay_confirmation = (
        source_result.get("overlay_confirmation")
        if isinstance(source_result.get("overlay_confirmation"), dict)
        else {}
    )
    provenance_fail: list[str] = []

    def _check_job_provenance(job: dict[str, Any], role: str) -> tuple[str, str]:
        if str(job.get("status") or "") != "ok":
            provenance_fail.append(f"{role}:status")
        if str(job.get("method") or "").upper() != "B":
            provenance_fail.append(f"{role}:method")
        if str(job.get("action") or "") != "readonly_ui_fine":
            provenance_fail.append(f"{role}:action")
        job_args = job.get("args") if isinstance(job.get("args"), dict) else {}
        if str(job_args.get("entry_id") or "") != str(entry.get("id") or ""):
            provenance_fail.append(f"{role}:entry_id")
        if str(job_args.get("pair_role") or "") != role:
            provenance_fail.append(f"{role}:pair_role")
        capture_id = str(
            job_args.get("capture_id") or job_args.get("pair_id") or ""
        ).strip()
        if not capture_id:
            provenance_fail.append(f"{role}:capture_id")
        job_result = (
            job.get("result") if isinstance(job.get("result"), dict) else {}
        )
        evidence = (
            job_result.get("hud_entry_evidence")
            if isinstance(job_result.get("hud_entry_evidence"), dict)
            else {}
        )
        raw_hash = str(evidence.get("raw_sha256") or "").strip()
        if not raw_hash:
            provenance_fail.append(f"{role}:raw_sha256")
        return capture_id, raw_hash

    source_capture_id, _ = _check_job_provenance(source_job, "source")
    target_capture_id = ""
    if target_job is None:
        provenance_fail.append("target:missing")
    else:
        target_capture_id, _ = _check_job_provenance(target_job, "target")
    if source_capture_id and target_capture_id and source_capture_id != target_capture_id:
        provenance_fail.append("capture_id:mismatch")
    provenance_ok = not provenance_fail
    feature_ok = bool(locator.get("feature_ok") is True)
    position_ok = bool(locator.get("position_ok") is True)
    detector_accepted = bool(locator.get("accepted") is True)
    overlay_evidence_ok = bool(overlay_evidence.get("ok") is True)
    explicit_confirmation_ok = str(overlay_confirmation.get("decision") or "") == "correct"
    click_accept_ok = bool(
        pair_ok
        and provenance_ok
        and feature_ok
        and position_ok
        and detector_accepted
        and overlay_evidence_ok
        and explicit_confirmation_ok
    )
    if not detector_accepted:
        fail_tags.append("hud:detector_not_accepted")
    if not provenance_ok:
        fail_tags.append("hud:provenance_invalid")
    if not overlay_evidence_ok:
        fail_tags.append("hud:overlay_evidence_missing")
    if not explicit_confirmation_ok:
        fail_tags.append("hud:overlay_unconfirmed")
    return {
        "pair_ok": pair_ok,
        "click_accept_ok": click_accept_ok,
        "enough": False,
        "click": click,
        "locator": locator,
        "feature_ok": feature_ok,
        "position_ok": position_ok,
        "detector_accepted": detector_accepted,
        "overlay_evidence_ok": overlay_evidence_ok,
        "explicit_confirmation_ok": explicit_confirmation_ok,
        "provenance_ok": provenance_ok,
        "provenance_fail": provenance_fail,
        "source_miss": src_miss,
        "target_miss": tgt_miss,
        "fail_tags": fail_tags,
        "note": "只读人工点击验收；够用仍否",
    }


def validate_hud_entry(
    entry_id: str,
    *,
    source_path: Path,
    target_path: Path | None = None,
    root: Path | None = None,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cat = catalog if catalog is not None else load_main_hud_catalog(root=root)
    out: dict[str, Any] = {
        "ok": False,
        "entry_id": entry_id,
        "enough": False,
        "error": None,
    }
    try:
        entry = entry_by_id(cat, entry_id)
        source_job = load_pair_job_json(Path(source_path))
        target_job = load_pair_job_json(Path(target_path)) if target_path else None
        assess = assess_hud_entry_captures(
            entry, source_job, target_job, catalog=cat
        )
        repo = (root or ROOT).resolve()
        integrity_fail: list[str] = []
        verified_images: dict[str, np.ndarray] = {}

        def _verify_png(label: str, rel_path: Any, expected_hash: Any) -> None:
            rel = str(rel_path or "").strip()
            expected = str(expected_hash or "").strip().lower()
            if not rel or not expected:
                integrity_fail.append(f"{label}:missing")
                return
            try:
                candidate = (repo / rel).resolve()
                candidate.relative_to(repo)
            except (OSError, ValueError):
                integrity_fail.append(f"{label}:outside_root")
                return
            if not candidate.is_file():
                integrity_fail.append(f"{label}:not_found")
                return
            data = candidate.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                integrity_fail.append(f"{label}:sha256_mismatch")
                return
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3 or image.shape[2] != 3:
                integrity_fail.append(f"{label}:invalid_png")
                return
            verified_images[label] = image

        source_result = (
            source_job.get("result")
            if isinstance(source_job.get("result"), dict)
            else {}
        )
        target_result = (
            target_job.get("result")
            if target_job is not None and isinstance(target_job.get("result"), dict)
            else {}
        )
        source_evidence = (
            source_result.get("hud_entry_evidence")
            if isinstance(source_result.get("hud_entry_evidence"), dict)
            else {}
        )
        target_evidence = (
            target_result.get("hud_entry_evidence")
            if isinstance(target_result.get("hud_entry_evidence"), dict)
            else {}
        )
        overlay_evidence = (
            source_result.get("overlay_evidence")
            if isinstance(source_result.get("overlay_evidence"), dict)
            else {}
        )
        confirmation = (
            source_result.get("overlay_confirmation")
            if isinstance(source_result.get("overlay_confirmation"), dict)
            else {}
        )
        source_args = (
            source_job.get("args") if isinstance(source_job.get("args"), dict) else {}
        )
        target_args = (
            target_job.get("args")
            if target_job is not None and isinstance(target_job.get("args"), dict)
            else {}
        )
        capture_id = str(
            source_args.get("capture_id") or source_args.get("pair_id") or ""
        )
        try:
            safe_capture_id(capture_id)
        except ValueError:
            integrity_fail.append("capture_id:unsafe")

        def _expect_context(
            label: str,
            payload: dict[str, Any],
            *,
            role: str | None = None,
        ) -> None:
            if str(payload.get("entry_id") or "") != str(entry.get("id") or ""):
                integrity_fail.append(f"{label}:entry_id")
            if str(payload.get("capture_id") or "") != capture_id:
                integrity_fail.append(f"{label}:capture_id")
            if role is not None and str(payload.get("role") or "") != role:
                integrity_fail.append(f"{label}:role")
            if str(payload.get("raw_sha256") or "") != str(
                source_evidence.get("raw_sha256") if role != "target" else target_evidence.get("raw_sha256")
            ):
                integrity_fail.append(f"{label}:raw_sha256")

        _expect_context("source_manifest", source_evidence, role="source")
        _expect_context("target_manifest", target_evidence, role="target")
        _expect_context("overlay_manifest", overlay_evidence)
        _expect_context("confirmation", confirmation)

        _verify_png(
            "source_raw",
            source_evidence.get("raw_path"),
            source_evidence.get("raw_sha256"),
        )
        _verify_png(
            "target_raw",
            target_evidence.get("raw_path"),
            target_evidence.get("raw_sha256"),
        )
        _verify_png(
            "overlay",
            overlay_evidence.get("path"),
            overlay_evidence.get("sha256"),
        )
        stored_locator = (
            source_result.get("locator")
            if isinstance(source_result.get("locator"), dict)
            else {}
        )
        click = (
            stored_locator.get("click")
            if isinstance(stored_locator.get("click"), dict)
            else {}
        )
        click_client = click.get("client")
        if not isinstance(click_client, list) or len(click_client) != 2:
            integrity_fail.append("locator:click_client")
        else:
            expected_click = [int(click_client[0]), int(click_client[1])]
            if overlay_evidence.get("click_client") != expected_click:
                integrity_fail.append("overlay_manifest:click_client")
            if confirmation.get("click_client") != expected_click:
                integrity_fail.append("confirmation:click_client")
            client_box = overlay_evidence.get("client_box")
            if not isinstance(client_box, list) or len(client_box) != 4:
                integrity_fail.append("overlay_manifest:client_box")
            else:
                locator_screen = click.get("screen")
                if (
                    not isinstance(locator_screen, list)
                    or len(locator_screen) != 2
                    or [
                        int(client_box[0]) + expected_click[0],
                        int(client_box[1]) + expected_click[1],
                    ]
                    != [int(locator_screen[0]), int(locator_screen[1])]
                ):
                    integrity_fail.append("overlay_manifest:screen_point")

        source_image = verified_images.get("source_raw")
        target_image = verified_images.get("target_raw")
        overlay_image = verified_images.get("overlay")
        if source_image is not None:
            client = (
                source_result.get("client")
                if isinstance(source_result.get("client"), dict)
                else {}
            )
            if [source_image.shape[1], source_image.shape[0]] != [
                int(client.get("w") or 0),
                int(client.get("h") or 0),
            ]:
                integrity_fail.append("source_raw:client_size")
            rerun = locate_hud_entry(entry, source_image, root=repo)
            if (
                rerun.get("feature_ok") is not True
                or rerun.get("position_ok") is not True
                or rerun.get("accepted") is not True
                or rerun.get("feature_rect_client")
                != stored_locator.get("feature_rect_client")
                or (rerun.get("click") or {}).get("client") != click_client
            ):
                integrity_fail.append("locator:rerun_mismatch")
        if (
            source_image is not None
            and target_image is not None
            and source_image.shape != target_image.shape
        ):
            integrity_fail.append("target_raw:client_size")
        if source_image is not None and overlay_image is not None and isinstance(
            click_client, list
        ) and len(click_client) == 2:
            from mm_assist.platform.hint_overlay import red_marker_delta_ok

            if not red_marker_delta_ok(
                source_image,
                overlay_image,
                client_xy=(int(click_client[0]), int(click_client[1])),
            ):
                integrity_fail.append("overlay:red_delta_missing")

        if str(target_args.get("capture_id") or target_args.get("pair_id") or "") != capture_id:
            integrity_fail.append("target:capture_id")
        integrity_ok = not integrity_fail
        assess["integrity_ok"] = integrity_ok
        assess["integrity_fail"] = integrity_fail
        if not integrity_ok:
            assess["click_accept_ok"] = False
            tags = list(assess.get("fail_tags") or [])
            if "hud:evidence_integrity_invalid" not in tags:
                tags.append("hud:evidence_integrity_invalid")
            assess["fail_tags"] = tags
        out.update(
            {
                "ok": True,
                "pair_ok": bool(assess.get("pair_ok")),
                "enough": False,
                "assess": assess,
                "fail_tags": list(assess.get("fail_tags") or []),
            }
        )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}:{exc}"
    return out


def format_validate_hud_entry_report(result: dict[str, Any]) -> str:
    lines = ["# validate-hud-entry", f"entry_id={result.get('entry_id')}"]
    if result.get("error"):
        lines.append(f"error: {result['error']}")
        return "\n".join(lines) + "\n"
    assess = result.get("assess") if isinstance(result.get("assess"), dict) else {}
    click = assess.get("click") if isinstance(assess.get("click"), dict) else {}
    lines.extend(
        [
            f"pair_ok={result.get('pair_ok')} enough=False",
            f"click_accept_ok={assess.get('click_accept_ok')} "
            f"provenance_ok={assess.get('provenance_ok')} "
            f"integrity_ok={assess.get('integrity_ok')}",
            f"feature_ok={assess.get('feature_ok')} "
            f"position_ok={assess.get('position_ok')} "
            f"overlay_ok={assess.get('overlay_evidence_ok')} "
            f"confirmed={assess.get('explicit_confirmation_ok')}",
            f"allow_click={click.get('allow_click')} reason={click.get('reason')}",
        ]
    )
    tags = result.get("fail_tags") or []
    if tags:
        lines.append("fail_tags: " + ", ".join(str(t) for t in tags))
    return "\n".join(lines) + "\n"
