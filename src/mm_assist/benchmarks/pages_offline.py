"""离线真帧认页尺子：`benchmark perception --mode pages-offline`。

语料：`knowledge/captures/transitions/*/{source,target}.png`。
真值来自转移图边 id（目录名）的 from / to。OCR 默认打桩为空，只计量模板（与诊断同口径）。
缺制作壳/设施真帧时不得声称细页已量过。不得宣称感知够用 / C2。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mm_assist.benchmarks.ui import LEDGER_DIR, _percentile
from mm_assist.config import ROOT
from mm_assist.knowledge_paths import TRANSITION_GRAPH_JSON

MISSING_FAMILIES = ("craft_hub", "craft_quick", "craft_make", "craft_alchemy", "facility")
KNOWN_FAMILIES = (
    "main_hud",
    "system_menu",
    "inventory",
    "profile",
    "life_grid",
    "gather_detail",
    "gather_skill",
)


def _family(node: str) -> str:
    return str(node or "").split(":", 1)[0]


def sn_for_expected(node: str) -> str:
    fam = _family(node)
    if fam in ("craft_hub", "craft_quick", "craft_make", "craft_alchemy"):
        return "fine:craft_hub"
    if fam == "facility":
        return "fine:facility"
    if fam == "facility_detail":
        return "fine:facility_detail"
    if fam == "item_info":
        return "item_info"
    if fam in ("life_grid", "gather_skill", "gather_detail"):
        return "fine:life_grid"
    return ""


def node_matches(got: str, expected: str) -> bool:
    g, e = str(got or ""), str(expected or "")
    if not e:
        return False
    if g == e:
        return True
    return g.startswith(e + ":") or e.startswith(g + ":")


def load_graph(*, root: Path | None = None) -> dict[str, Any]:
    path = TRANSITION_GRAPH_JSON if root is None else Path(root) / "knowledge/ui/contracts/ui_transition_graph.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _edge_index(graph: dict[str, Any]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for row in graph.get("edges") or []:
        eid = str(row.get("id") or "")
        if not eid:
            continue
        out[eid] = (str(row.get("from") or ""), str(row.get("to") or ""))
    return out


def parse_pair_dir(name: str, graph: dict[str, Any]) -> tuple[str, str]:
    hit = _edge_index(graph).get(name)
    if hit:
        return hit
    nodes = sorted(
        (str(n.get("id") or "") for n in (graph.get("nodes") or []) if n.get("id")),
        key=len,
        reverse=True,
    )
    for src in nodes:
        prefix = src + "_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        for dst in nodes:
            suffix = "_" + dst
            if rest.endswith(suffix) or rest == dst:
                return src, dst
    return "", ""


def _read_bgr(path: Path) -> np.ndarray | None:
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return img


def iter_page_frames(
    captures: Path,
    graph: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not captures.is_dir():
        return rows
    for run in sorted(p for p in captures.iterdir() if p.is_dir()):
        if run.name.startswith("."):
            continue
        for pair in sorted(p for p in run.iterdir() if p.is_dir()):
            src_n, dst_n = parse_pair_dir(pair.name, graph)
            for role, expected in (("source", src_n), ("target", dst_n)):
                png = pair / f"{role}.png"
                if not png.is_file() or not expected:
                    continue
                rows.append(
                    {
                        "run": run.name,
                        "edge_id": pair.name,
                        "role": role,
                        "expected": expected,
                        "path": png,
                    }
                )
    return rows


def _stub_ocr_roi(*_a: Any, **_k: Any) -> list[Any]:
    return []


def _eval_refine(
    frame: np.ndarray,
    *,
    sn: str,
    warmup: int,
    repeats: int,
    refine,
    reset_cost,
    inspect_cost,
) -> dict[str, Any]:
    times: list[float] = []
    node = ""
    cost: dict[str, Any] = {}
    rounds = max(0, int(warmup)) + max(1, int(repeats))
    for i in range(rounds):
        reset_cost()
        t0 = time.perf_counter()
        node = str(refine(frame, sn=sn) or "")
        cost = inspect_cost(t0)
        if i >= max(0, int(warmup)):
            times.append(float(cost.get("ms") or 0.0))
    return {
        "node": node,
        "tpl_n": int(cost.get("tpl_n") or 0),
        "ocr_n": int(cost.get("ocr_n") or 0),
        "p95_ms": round(_percentile(times, 95), 3),
        "sn": sn,
    }


def run_pages_offline(
    *,
    warmup: int = 0,
    repeats: int = 1,
    root: Path | None = None,
    out_dir: Path | None = None,
    stub_ocr: bool = True,
) -> dict[str, Any]:
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.perception import ocr as ocr_mod

    base = Path(root) if root is not None else ROOT
    captures = base / "knowledge" / "captures" / "transitions"
    graph = load_graph(root=base)
    frames = iter_page_frames(captures, graph)

    prev_ocr = ocr_mod.ocr_roi
    results: list[dict[str, Any]] = []
    try:
        if stub_ocr:
            ocr_mod.ocr_roi = _stub_ocr_roi  # type: ignore[assignment]
        for spec in frames:
            img = _read_bgr(Path(spec["path"]))
            if img is None or img.size == 0:
                continue
            expected = str(spec["expected"])
            full = _eval_refine(
                img,
                sn="",
                warmup=warmup,
                repeats=repeats,
                refine=ni.refine_inspect_node,
                reset_cost=ni._reset_inspect_cost,
                inspect_cost=ni._inspect_cost,
            )
            routed = _eval_refine(
                img,
                sn=sn_for_expected(expected),
                warmup=warmup,
                repeats=repeats,
                refine=ni.refine_inspect_node,
                reset_cost=ni._reset_inspect_cost,
                inspect_cost=ni._inspect_cost,
            )
            results.append(
                {
                    "run": spec["run"],
                    "edge_id": spec["edge_id"],
                    "role": spec["role"],
                    "expected": expected,
                    "path": str(Path(spec["path"]).as_posix()),
                    "full": full,
                    "routed": routed,
                    "match_full": node_matches(str(full.get("node") or ""), expected),
                    "match_routed": node_matches(str(routed.get("node") or ""), expected),
                    "h": int(img.shape[0]),
                    "w": int(img.shape[1]),
                }
            )
    finally:
        if stub_ocr:
            ocr_mod.ocr_roi = prev_ocr

    seen = sorted({_family(str(r.get("expected") or "")) for r in results if r.get("expected")})
    missing = [f for f in MISSING_FAMILIES if f not in seen]
    n = len(results)
    n_full = sum(1 for r in results if r.get("match_full"))
    n_routed = sum(1 for r in results if r.get("match_routed"))
    p95s = [float((r.get("full") or {}).get("p95_ms") or 0.0) for r in results]
    payload = {
        "schema": "pages_offline.v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "warmup": warmup,
        "repeats": repeats,
        "ocr_stub": bool(stub_ocr),
        "captures": str(captures.as_posix()),
        "frames": n,
        "match_full_n": n_full,
        "match_routed_n": n_routed,
        "p95_ms_mean": round(sum(p95s) / max(len(p95s), 1), 3),
        "families_seen": seen,
        "families_missing": missing,
        "enough": False,
        "ok": n > 0,
        "note": (
            "真帧 refine 尺子；OCR 打桩只算模板。"
            "缺制作壳/六卡/设施/配方真帧，不得声称细页认页已量过。"
            "不得宣称感知够用 / C2。"
        ),
        "results": results,
    }
    ledger = Path(out_dir) if out_dir is not None else LEDGER_DIR
    ledger.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ledger / f"pages_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jsonl = ledger / "pages.jsonl"
    with jsonl.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"at": payload["at"], "path": str(path.as_posix()), "frames": n}, ensure_ascii=False)
            + "\n"
        )
    payload["ledger"] = str(path.as_posix())
    return payload


def format_pages_table(payload: dict[str, Any]) -> str:
    lines = [
        "edge role expected got_full match_full tpl_n p95_ms got_routed match_routed tpl_n_r p95_r"
    ]
    for r in payload.get("results") or []:
        full = r.get("full") or {}
        routed = r.get("routed") or {}
        lines.append(
            f"{r.get('edge_id')} {r.get('role')} {r.get('expected')} "
            f"{full.get('node')} {int(bool(r.get('match_full')))} {full.get('tpl_n')} {full.get('p95_ms')} "
            f"{routed.get('node')} {int(bool(r.get('match_routed')))} {routed.get('tpl_n')} {routed.get('p95_ms')}"
        )
    miss = ",".join(payload.get("families_missing") or []) or "-"
    lines.append(
        f"frames={payload.get('frames')} match_full={payload.get('match_full_n')}/{payload.get('frames')} "
        f"match_routed={payload.get('match_routed_n')}/{payload.get('frames')} "
        f"p95_ms_mean={payload.get('p95_ms_mean')} missing={miss} enough=False"
    )
    lines.append(str(payload.get("note") or ""))
    return "\n".join(lines) + "\n"
