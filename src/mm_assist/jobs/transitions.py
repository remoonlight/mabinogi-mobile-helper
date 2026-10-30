"""核心界面只读关系图：节点/边校验与方法 B 成对任务。

不发送键鼠。forbidden 边不可开 job。enough 永不因本模块为真。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mm_assist.config import ROOT
from mm_assist.jobs.pairs import load_pair_job_json
from mm_assist.jobs.bar import open_test_job

from mm_assist.knowledge_paths import TRANSITION_GRAPH_JSON

GRAPH_REL = TRANSITION_GRAPH_JSON

NODE_IDS = frozenset(
    {
        "main_hud",
        "stack:CashShopUIState",
        "system_menu",
        "inventory",
        "map_dunbarton",
        "map_dunbarton_nerys",
        "profile",
        "life_grid",
        "gather_skill",
        "gather_detail",
        "item_info",
        "obtain_method",
        "dialog_skip",
        "craft_shell",
        "craft_items",
        "craft_item_recipe",
        "craft_hub",
        "craft_quick",
        "craft_make",
        "craft_alchemy",
        "facility",
        "facility_recipe",
    }
)
EDGE_KINDS = frozenset({"allowed", "forbidden"})
EVIDENCE_STATUSES = frozenset({"unverified", "verified", "forbidden", "gap"})
FRAME_ROLES = frozenset({"source", "target"})
TRIGGER_WHOS = frozenset({"user_manual", "agent"})
def graph_path(*, root: Path | None = None) -> Path:
    return (root or ROOT) / GRAPH_REL


def load_ui_transition_graph(*, root: Path | None = None) -> dict[str, Any]:
    p = graph_path(root=root)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("ui_transition_graph 须为 object")
    validate_ui_transition_graph(data)
    return data


def validate_ui_transition_graph(data: dict[str, Any]) -> None:
    if str(data.get("schema") or "") != "ui_transition_graph.v1":
        raise ValueError("未知 schema")
    nodes = data.get("nodes")
    edges = data.get("edges")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("nodes 须为非空 list")
    if not isinstance(edges, list) or not edges:
        raise ValueError("edges 须为非空 list")
    seen_nodes: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("node 须为 object")
        nid = str(node.get("id") or "").strip()
        if nid not in NODE_IDS:
            raise ValueError(f"未知 node={nid!r}")
        if nid in seen_nodes:
            raise ValueError(f"重复 node={nid!r}")
        if not str(node.get("name") or "").strip():
            raise ValueError(f"node {nid} 缺 name")
        if not str(node.get("need_ui") or "").strip():
            raise ValueError(f"node {nid} 缺 need_ui")
        sig = node.get("signals")
        if not isinstance(sig, dict) or not sig:
            raise ValueError(f"node {nid} 缺 signals")
        seen_nodes.add(nid)
    missing = NODE_IDS - seen_nodes
    if missing:
        raise ValueError(f"缺核心 node: {sorted(missing)}")

    seen_edges: set[str] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("edge 须为 object")
        eid = str(edge.get("id") or "").strip()
        if not eid:
            raise ValueError("edge 缺 id")
        if eid in seen_edges:
            raise ValueError(f"重复 edge={eid!r}")
        seen_edges.add(eid)
        kind = str(edge.get("kind") or "").strip()
        if kind not in EDGE_KINDS:
            raise ValueError(f"edge {eid} 未知 kind={kind!r}")
        src = str(edge.get("from") or "").strip()
        dst = str(edge.get("to") or "").strip()
        if src not in seen_nodes or dst not in seen_nodes:
            raise ValueError(f"edge {eid} 引用未知 node {src}->{dst}")
        trig = edge.get("trigger") if isinstance(edge.get("trigger"), dict) else {}
        who = str(trig.get("who") or "").strip()
        if who not in TRIGGER_WHOS:
            raise ValueError(
                f"edge {eid} trigger.who 须为 {sorted(TRIGGER_WHOS)!r}，收到 {who!r}"
            )
        if not str(trig.get("label") or "").strip():
            raise ValueError(f"edge {eid} 缺 trigger.label")
        ev = edge.get("evidence") if isinstance(edge.get("evidence"), dict) else {}
        st = str(ev.get("status") or "").strip()
        if st not in EVIDENCE_STATUSES:
            raise ValueError(f"edge {eid} 未知 evidence.status={st!r}")
        if kind == "forbidden":
            if st != "forbidden":
                raise ValueError(f"forbidden 边 {eid} 须 evidence.status=forbidden")
            if not str(edge.get("reason") or "").strip():
                raise ValueError(f"forbidden 边 {eid} 缺 reason")
        elif st == "forbidden":
            raise ValueError(f"allowed 边 {eid} 不得 status=forbidden")


def node_by_id(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    nid = str(node_id or "").strip()
    for node in graph.get("nodes") or []:
        if isinstance(node, dict) and str(node.get("id")) == nid:
            return node
    raise KeyError(f"未知 node={nid!r}")


def edge_by_id(graph: dict[str, Any], edge_id: str) -> dict[str, Any]:
    eid = str(edge_id or "").strip()
    for edge in graph.get("edges") or []:
        if isinstance(edge, dict) and str(edge.get("id")) == eid:
            return edge
    raise KeyError(f"未知 edge={eid!r}")


def require_allowed_edge(graph: dict[str, Any], edge_id: str) -> dict[str, Any]:
    edge = edge_by_id(graph, edge_id)
    if str(edge.get("kind")) == "forbidden":
        raise ValueError(
            f"禁止边 {edge_id}：{edge.get('reason') or '不可开采证任务'}"
        )
    return edge


def _utc_transition_id(edge_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{edge_id}_{stamp}"


def transition_job_specs(
    edge_id: str,
    *,
    graph: dict[str, Any] | None = None,
    transition_id: str | None = None,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    g = graph if graph is not None else load_ui_transition_graph(root=root)
    edge = require_allowed_edge(g, edge_id)
    src = node_by_id(g, str(edge["from"]))
    dst = node_by_id(g, str(edge["to"]))
    tid = str(transition_id).strip() if transition_id else _utc_transition_id(str(edge["id"]))
    label = str((edge.get("trigger") or {}).get("label") or "")
    frames = (
        ("source", src, f"源态·{src.get('name')}", src.get("need_ui")),
        (
            "target",
            dst,
            f"目标态·{dst.get('name')}（人手已点 {label}）",
            dst.get("need_ui"),
        ),
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
                    "pair_id": tid,
                    "pair_role": role,
                    "transition_id": tid,
                    "edge_id": str(edge["id"]),
                    "node_id": str(node["id"]),
                },
            }
        )
    return out


def open_transition_frame(
    edge_id: str,
    role: Literal["source", "target"],
    *,
    transition_id: str | None = None,
    path: Path | None = None,
    root: Path | None = None,
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    role_s = str(role or "").strip().lower()
    if role_s not in FRAME_ROLES:
        raise ValueError(f"role 须为 source|target，收到 {role!r}")
    specs = transition_job_specs(
        edge_id, graph=graph, transition_id=transition_id, root=root
    )
    spec = next(s for s in specs if str(s["args"]["pair_role"]) == role_s)
    return open_test_job(
        "readonly_ui_fine",
        method="B",
        need_ui=str(spec["need_ui"]),
        title=str(spec["title"]),
        args=dict(spec["args"]),
        path=path,
    )


def _job_probe(job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    probe = result.get("panel_probe")
    if isinstance(probe, dict):
        return probe
    return {}


def _match_signals(probe: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    misses: list[str] = []
    for key, want in expected.items():
        got = probe.get(key)
        if bool(got) is not bool(want) or got is None:
            misses.append(f"{key}:want={want!r} got={got!r}")
    return misses


def _icon_ok(job: dict[str, Any]) -> bool:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    icons = result.get("icons") if isinstance(result.get("icons"), list) else []
    errs = [str(e) for e in (result.get("icon_errors") or [])]
    if any("wall_timeout" in e for e in errs):
        return False
    probe = _job_probe(job)
    if icons:
        return True
    if probe.get("craft_icon_hits") or probe.get("craft_open_signal"):
        return True
    return False


def assess_transition_captures(
    edge: dict[str, Any],
    source_job: dict[str, Any],
    target_job: dict[str, Any],
    *,
    graph: dict[str, Any],
) -> dict[str, Any]:
    src_node = node_by_id(graph, str(edge["from"]))
    dst_node = node_by_id(graph, str(edge["to"]))
    pa = _job_probe(source_job)
    pb = _job_probe(target_job)
    src_miss = _filter_shell_probe_misses(
        edge, _match_signals(pa, src_node.get("signals") or {})
    )
    dst_miss = _filter_shell_probe_misses(
        edge, _match_signals(pb, dst_node.get("signals") or {})
    )
    fail_tags: list[str] = []
    if src_miss:
        fail_tags.append("transition:source_signal_mismatch")
    if dst_miss:
        fail_tags.append("transition:target_signal_mismatch")
    params_ok = not src_miss and not dst_miss
    icon_src = _icon_ok(source_job)
    icon_dst = _icon_ok(target_job)
    icon_ok = icon_src and icon_dst
    if not icon_ok:
        fail_tags.append("transition:icon_or_panel_missing")
    pair_ok = bool(params_ok and icon_ok)
    return {
        "pair_ok": pair_ok,
        "enough": False,
        "params_ok": params_ok,
        "icon_ok": icon_ok,
        "source_miss": src_miss,
        "target_miss": dst_miss,
        "fail_tags": fail_tags,
        "note": "只读成对差分；够用仍否",
    }


def _template_hits_ok(job: dict[str, Any], required: list[str]) -> tuple[bool, list[str]]:
    if not required:
        return True, []
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    hits = result.get("template_hits") if isinstance(result.get("template_hits"), dict) else {}
    misses: list[str] = []
    for rel in required:
        key = str(rel).replace("\\", "/")
        ok = bool(hits.get(key) or hits.get(rel))
        if not ok:
            misses.append(key)
    return not misses, misses


def _provenance_ok(job: dict[str, Any], *, role: str, capture_id: str) -> list[str]:
    fails: list[str] = []
    method = str(job.get("method") or "").strip().upper()
    if method not in {"A", "B"}:
        fails.append(f"{role}:method")
    action = str(job.get("action") or "")
    if action and action not in {"readonly_ui_fine", "transition_accept"}:
        fails.append(f"{role}:action")
    args = job.get("args") if isinstance(job.get("args"), dict) else {}
    cid = str(args.get("capture_id") or args.get("pair_id") or "").strip()
    if not cid:
        fails.append(f"{role}:capture_id")
    elif capture_id and cid != capture_id:
        fails.append(f"{role}:capture_id_mismatch")
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    sha = str(result.get("raw_sha256") or "").strip()
    if not sha:
        evid = result.get("hud_entry_evidence") if isinstance(result.get("hud_entry_evidence"), dict) else {}
        sha = str(evid.get("raw_sha256") or "").strip()
    if not sha:
        fails.append(f"{role}:raw_sha256")
    return fails


_PROFILE_SHELL_NODES = frozenset({"profile", "life_grid", "gather_detail"})
# 人物壳内 Tab 切换时内存 profile_open 常滞后，画面已到位仍报 miss
_SHELL_PROBE_EXEMPT = frozenset({"profile_open"})


def _filter_shell_probe_misses(
    edge: dict[str, Any], misses: list[str]
) -> list[str]:
    if not _profile_shell_edge(edge):
        return misses
    out: list[str] = []
    for row in misses:
        key = str(row).split(":", 1)[0]
        if key in _SHELL_PROBE_EXEMPT:
            continue
        out.append(row)
    return out


def _profile_shell_edge(edge: dict[str, Any]) -> bool:
    """人物壳内 Tab/详情切换不改变四参，须豁免 signal_delta。"""
    return (
        str(edge.get("from") or "") in _PROFILE_SHELL_NODES
        and str(edge.get("to") or "") in _PROFILE_SHELL_NODES
    )


def profile_shell_edge(edge: dict[str, Any]) -> bool:
    """供 harness 判断 arrival 是否须画面确认。"""
    return _profile_shell_edge(edge)


def _signal_delta(src: dict[str, Any], dst: dict[str, Any]) -> int:
    keys = set(src) | set(dst)
    n = 0
    for key in keys:
        if bool(src.get(key)) != bool(dst.get(key)):
            n += 1
    return n


def assess_transition_captures_v2(
    edge: dict[str, Any],
    source_job: dict[str, Any],
    target_job: dict[str, Any],
    *,
    graph: dict[str, Any],
    required_templates: list[str] | None = None,
) -> dict[str, Any]:
    """加强成对验收：信号 + 目标模板 + 出处 + 至少一处信号变化。enough 仍为 False。"""
    base = assess_transition_captures(edge, source_job, target_job, graph=graph)
    fail_tags = list(base.get("fail_tags") or [])
    req = [str(x) for x in (required_templates or []) if str(x).strip()]
    tmpl_ok, tmpl_miss = _template_hits_ok(target_job, req)
    if req and not tmpl_ok:
        fail_tags.append("transition:required_template_miss")
        icon_ok = False
    else:
        icon_ok = bool(base.get("icon_ok")) and tmpl_ok
        if not icon_ok and "transition:icon_or_panel_missing" not in fail_tags:
            fail_tags.append("transition:icon_or_panel_missing")
    src_args = source_job.get("args") if isinstance(source_job.get("args"), dict) else {}
    capture_id = str(src_args.get("capture_id") or src_args.get("pair_id") or "").strip()
    prov_fail = _provenance_ok(source_job, role="source", capture_id=capture_id)
    prov_fail.extend(_provenance_ok(target_job, role="target", capture_id=capture_id))
    provenance_ok = not prov_fail
    if not provenance_ok:
        fail_tags.append("transition:provenance_invalid")
    pa = _job_probe(source_job)
    pb = _job_probe(target_job)
    delta = _signal_delta(pa, pb)
    shell_edge = _profile_shell_edge(edge)
    pre_post_delta = delta >= 1 or shell_edge
    if not pre_post_delta:
        fail_tags.append("transition:no_signal_delta")
    elif shell_edge:
        fail_tags = [t for t in fail_tags if t != "transition:no_signal_delta"]
    params_ok = bool(base.get("params_ok"))
    pair_ok = bool(params_ok and icon_ok and provenance_ok and pre_post_delta)
    return {
        "pair_ok": pair_ok,
        "enough": False,
        "params_ok": params_ok,
        "icon_ok": icon_ok,
        "provenance_ok": provenance_ok,
        "pre_post_delta": pre_post_delta,
        "signal_delta": delta,
        "template_miss": tmpl_miss,
        "provenance_fail": prov_fail,
        "source_miss": list(base.get("source_miss") or []),
        "target_miss": list(base.get("target_miss") or []),
        "fail_tags": fail_tags,
        "note": "v2 成对验收；够用仍否",
    }


def validate_transition(
    edge_id: str,
    *,
    source_path: Path,
    target_path: Path,
    root: Path | None = None,
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    g = graph if graph is not None else load_ui_transition_graph(root=root)
    out: dict[str, Any] = {
        "ok": False,
        "edge_id": edge_id,
        "enough": False,
        "error": None,
    }
    try:
        edge = require_allowed_edge(g, edge_id)
        source_job = load_pair_job_json(Path(source_path))
        target_job = load_pair_job_json(Path(target_path))
        assess = assess_transition_captures(edge, source_job, target_job, graph=g)
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


def format_validate_transition_report(result: dict[str, Any]) -> str:
    lines = [
        "# validate-transition",
        f"edge_id={result.get('edge_id')}",
    ]
    if result.get("error"):
        lines.append(f"error: {result['error']}")
        return "\n".join(lines) + "\n"
    assess = result.get("assess") if isinstance(result.get("assess"), dict) else {}
    lines.extend(
        [
            f"pair_ok={result.get('pair_ok')} enough=False（硬闸）",
            f"params_ok={assess.get('params_ok')} icon_ok={assess.get('icon_ok')}",
        ]
    )
    tags = result.get("fail_tags") or []
    if tags:
        lines.append("fail_tags: " + ", ".join(str(t) for t in tags))
    return "\n".join(lines) + "\n"


