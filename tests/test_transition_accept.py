"""transition-accept：dry-run、v2 验收、promote 门闩。不跑 LIVE。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_assist.benchmarks.transition_accept import (
    EDGE_STEP_ZH,
    TransitionAcceptError,
    apply_inspect_arrival,
    apply_visual_menu_open,
    format_dry_run,
    format_progress_done,
    format_progress_fail_stop,
    format_progress_locator_stop,
    format_progress_running,
    load_manifest,
    locate_click_point,
    node_from_inspect,
    plan_suite,
    promote_edge,
    require_inspect_readable,
    resolve_locator,
    run_suite,
    signals_match,
    write_report,
    zh_step_label,
)
from mm_assist.cli import ALLOWED_CMDS, main
from mm_assist.jobs.transitions import (
    assess_transition_captures_v2,
    load_ui_transition_graph,
    require_allowed_edge,
)


def test_cli_allows_transition_accept():
    assert "transition-accept" in ALLOWED_CMDS
    assert "gather" not in ALLOWED_CMDS


def test_detect_node_prefers_system_menu_over_craft_hub_superset():
    from mm_assist.benchmarks.transition_accept import detect_node_id
    from mm_assist.jobs.transitions import load_ui_transition_graph

    g = load_ui_transition_graph()
    probe = {
        "menu_open": True,
        "inventory_open": False,
        "dialog_active": False,
        "map_open": False,
        "profile_open": False,
    }
    assert detect_node_id(probe, g) == "system_menu"
    assert detect_node_id(probe, g, prefer="system_menu") == "system_menu"
    assert detect_node_id(probe, g, prefer="craft_hub") == "craft_hub"


def test_plan_suite_eleven_allowed_edges():
    plan = plan_suite(suite="gather_plus_menu", skill="woodcut")
    assert plan["enough"] is False
    assert len(plan["steps"]) == 11
    assert plan["steps"][0]["edge_id"] == "main_hud_menu_system_menu"
    assert plan["steps"][3]["locator"]["kind"] == "skill_cell"
    assert plan["steps"][3]["locator"]["template"].endswith("woodcut.png")
    assert all(step["locator"]["resolved"] for step in plan["steps"])
    close_life = next(s for s in plan["steps"] if s["edge_id"] == "life_grid_close_main_hud")
    assert close_life["locator"]["template"].endswith("profile/close.png")
    menu_close = next(s for s in plan["steps"] if s["edge_id"] == "system_menu_x_main_hud")
    assert menu_close["locator"]["roi"] == [0.88, 0.0, 0.12, 0.14]
    text = format_dry_run(plan)
    assert "enough=False" in text
    assert "life_grid_skill_gather_detail" in text


def test_dry_run_cli(capsys):
    rc = main(["transition-accept", "run", "--dry-run", "--suite", "gather_plus_menu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "main_hud_menu_system_menu" in out


def test_forbidden_edge_not_in_suite():
    plan = plan_suite()
    ids = {s["edge_id"] for s in plan["steps"]}
    assert "facility_avatar_profile_forbidden" not in ids
    g = load_ui_transition_graph()
    with pytest.raises(ValueError, match="禁止边"):
        require_allowed_edge(g, "facility_avatar_profile_forbidden")


def test_unknown_skill_rejected():
    with pytest.raises(TransitionAcceptError, match="skill"):
        plan_suite(skill="alchemy")


def _pair_jobs(*, capture_id: str = "cap1") -> tuple[dict, dict]:
    src = {
        "method": "A",
        "action": "transition_accept",
        "args": {"capture_id": capture_id, "pair_role": "source"},
        "result": {
            "icons": [{"name": "menu"}],
            "icon_errors": [],
            "panel_probe": {
                "menu_open": False,
                "inventory_open": False,
                "dialog_active": False,
                "map_open": False,
                "profile_open": False,
            },
            "template_hits": {},
            "raw_sha256": "a" * 64,
        },
    }
    dst = {
        "method": "A",
        "action": "transition_accept",
        "args": {"capture_id": capture_id, "pair_role": "target"},
        "result": {
            "icons": [{"name": "x"}],
            "icon_errors": [],
            "panel_probe": {
                "menu_open": True,
                "inventory_open": False,
                "dialog_active": False,
            },
            "template_hits": {
                "knowledge/templates/menu/icon_close_x.png": True,
            },
            "raw_sha256": "b" * 64,
        },
    }
    return src, dst


def test_assess_v2_pair_ok_enough_false():
    g = load_ui_transition_graph()
    edge = require_allowed_edge(g, "main_hud_menu_system_menu")
    src, dst = _pair_jobs()
    out = assess_transition_captures_v2(
        edge,
        src,
        dst,
        graph=g,
        required_templates=["knowledge/templates/menu/icon_close_x.png"],
    )
    assert out["pair_ok"] is True
    assert out["enough"] is False
    assert out["params_ok"] is True
    assert out["icon_ok"] is True
    assert out["provenance_ok"] is True
    assert out["pre_post_delta"] is True


def test_assess_v2_missing_template_fails():
    g = load_ui_transition_graph()
    edge = require_allowed_edge(g, "main_hud_menu_system_menu")
    src, dst = _pair_jobs()
    dst["result"]["template_hits"] = {}
    out = assess_transition_captures_v2(
        edge,
        src,
        dst,
        graph=g,
        required_templates=["knowledge/templates/menu/icon_close_x.png"],
    )
    assert out["pair_ok"] is False
    assert out["enough"] is False
    assert "transition:required_template_miss" in out["fail_tags"]


def test_user_abort_before_live_loop(monkeypatch):
    import threading

    ev = threading.Event()
    ev.set()
    monkeypatch.setenv("MM_TRANSITION_ACCEPT", "1")
    monkeypatch.setenv("MM_LIVE_BENCHMARK", "1")

    def boom(*_a, **_k):
        raise AssertionError("abort 后不应碰到 watch")

    monkeypatch.setattr("mm_assist.jobs.bar.require_state_watch_running", boom)
    out = run_suite(live=True, stop_event=ev)
    assert out["stopped_on"] == "user_abort"
    assert out["enough"] is False
    assert out["pair_ok_count"] == 0

    monkeypatch.delenv("MM_TRANSITION_ACCEPT", raising=False)
    monkeypatch.delenv("MM_LIVE_BENCHMARK", raising=False)
    with pytest.raises(TransitionAcceptError, match="FAIL-CLOSED"):
        run_suite(live=True, dry_run=False)


def test_promote_requires_confirmation(tmp_path: Path):
    with pytest.raises(TransitionAcceptError, match="i-confirmed"):
        promote_edge("nope", "main_hud_menu_system_menu", i_confirmed=False, root=tmp_path)


def test_promote_writes_verified(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    run_id = "unit1"
    (logs / f"transition_accept_{run_id}.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "edges": [
                    {
                        "edge_id": "main_hud_menu_system_menu",
                        "pair_ok": True,
                        "evidence_dir": "knowledge/captures/transitions/unit1/main_hud_menu_system_menu",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    contracts = tmp_path / "knowledge" / "ui" / "contracts"
    contracts.mkdir(parents=True)
    from mm_assist.knowledge_paths import TRANSITION_GRAPH_JSON

    graph = json.loads(TRANSITION_GRAPH_JSON.read_text(encoding="utf-8"))
    (contracts / "ui_transition_graph.json").write_text(
        json.dumps(graph, ensure_ascii=False), encoding="utf-8"
    )
    out = promote_edge(
        run_id, "main_hud_menu_system_menu", i_confirmed=True, root=tmp_path
    )
    assert out["status"] == "verified"
    assert out["enough"] is False
    saved = json.loads((contracts / "ui_transition_graph.json").read_text(encoding="utf-8"))
    edge = next(e for e in saved["edges"] if e["id"] == "main_hud_menu_system_menu")
    assert edge["evidence"]["status"] == "verified"


def test_signals_match_and_visual_does_not_rewrite_params():
    want = {"menu_open": True, "inventory_open": False, "dialog_active": False}
    lag = {"menu_open": False, "inventory_open": False, "dialog_active": False}
    assert signals_match(lag, want) is False
    patched = apply_visual_menu_open(
        lag, want, {"knowledge/templates/menu/icon_close_x.png": True}
    )
    assert patched["menu_open"] is False
    assert signals_match(patched, want) is False


def test_locate_template_falls_back_to_fuzzy(monkeypatch, tmp_path):
    from types import SimpleNamespace

    png = tmp_path / "icon_character.png"
    png.write_bytes(b"x")
    frame = SimpleNamespace(shape=(1440, 1920, 3))

    monkeypatch.setattr(
        "mm_assist.platform.match.match_template",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mm_assist.platform.match.match_icon_fuzzy",
        lambda *a, **k: SimpleNamespace(cx=1400, cy=200, score=0.7),
    )
    monkeypatch.setattr(
        "mm_assist.perception.menu_icons.ROI_ESC_CHAR",
        (0.66, 0.08, 0.14, 0.28),
        raising=False,
    )
    out = locate_click_point(
        {
            "kind": "template",
            "template_path": str(png),
            "threshold": 0.78,
        },
        frame,
        screen_origin=(10, 20),
    )
    assert out["accepted"] is True
    assert out["click"]["client"] == [1400, 200]
    assert out["click"]["screen"] == [1410, 220]


def test_locate_rejects_left_hud_avatar_as_character(monkeypatch, tmp_path):
    from types import SimpleNamespace

    png = tmp_path / "icon_character.png"
    png.write_bytes(b"x")
    frame = SimpleNamespace(shape=(1440, 1920, 3))
    monkeypatch.setattr(
        "mm_assist.platform.match.match_template",
        lambda *a, **k: SimpleNamespace(cx=343, cy=86, score=0.8),
    )
    monkeypatch.setattr(
        "mm_assist.platform.match.match_icon_fuzzy",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._ocr_esc_click",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._menu_close_visible",
        lambda *a, **k: False,
    )
    out = locate_click_point(
        {"kind": "template", "template_path": str(png), "threshold": 0.62},
        frame,
    )
    assert out["accepted"] is False
    assert out["status"] == "no_match"


def test_resolve_node_uses_close_x_without_rewriting_probe(monkeypatch):
    from mm_assist.benchmarks.transition_accept import (
        enrich_probe_for_node,
        resolve_node_id,
    )

    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept.confirm_node_from_frame",
        lambda frame, nid, root=None: nid == "system_menu",
    )
    probe = {
        "menu_open": False,
        "inventory_open": False,
        "dialog_active": False,
        "map_open": False,
        "profile_open": False,
    }
    out = enrich_probe_for_node(probe, object(), "system_menu")
    assert out["menu_open"] is False
    g = load_ui_transition_graph()
    assert resolve_node_id(out, object(), g) == "system_menu"


def test_resolve_node_prefer_beats_false_gather_detail(monkeypatch):
    """关闭钮误命中 gather_detail 时，prefer=system_menu 仍认系统菜单。"""
    from mm_assist.benchmarks.transition_accept import resolve_node_id

    def _confirm(frame, nid, root=None):
        return nid in {"system_menu", "gather_detail"}

    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept.confirm_node_from_frame",
        _confirm,
    )
    g = load_ui_transition_graph()
    probe = {
        "menu_open": False,
        "inventory_open": False,
        "dialog_active": False,
        "map_open": False,
        "profile_open": False,
    }
    assert resolve_node_id(probe, object(), g) == "gather_detail"
    assert (
        resolve_node_id(probe, object(), g, prefer="system_menu") == "system_menu"
    )


def test_suite_order_role_before_bag_close():
    """开菜单后先人物，关人物壳后再补开菜单点背包。"""
    plan = plan_suite()
    ids = [s["edge_id"] for s in plan["steps"]]
    assert ids.index("main_hud_menu_system_menu") < ids.index("system_menu_role_profile")
    assert ids.index("system_menu_role_profile") < ids.index("life_grid_close_main_hud")
    assert ids.index("life_grid_close_main_hud") < ids.index("system_menu_bag_inventory")
    assert ids.index("system_menu_bag_inventory") < ids.index("inventory_close_main_hud")


def test_system_menu_not_confirmed_when_hamburger_visible(monkeypatch):
    from mm_assist.benchmarks.transition_accept import confirm_node_from_frame

    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._hud_menu_entry_visible",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._menu_close_visible",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._esc_grid_icon_visible",
        lambda *a, **k: True,
    )
    assert confirm_node_from_frame(object(), "system_menu") is False


def test_system_menu_needs_close_and_esc_grid(monkeypatch):
    from mm_assist.benchmarks.transition_accept import confirm_node_from_frame

    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._hud_menu_entry_visible",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._menu_close_visible",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._esc_grid_icon_visible",
        lambda *a, **k: False,
    )
    assert confirm_node_from_frame(object(), "system_menu") is False
    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._esc_grid_icon_visible",
        lambda *a, **k: True,
    )
    assert confirm_node_from_frame(object(), "system_menu") is True


def test_suite_starts_from_system_menu_when_already_open():
    from mm_assist.benchmarks.transition_accept import detect_node_id, suite_start_index

    g = load_ui_transition_graph()
    probe = {
        "menu_open": True,
        "inventory_open": False,
        "dialog_active": False,
        "map_open": False,
        "profile_open": False,
    }
    node = detect_node_id(probe, g)
    plan = plan_suite()
    assert suite_start_index(node, plan["steps"], g) == 1
    assert suite_start_index("main_hud", plan["steps"], g) == 0


def test_write_report_uses_chinese_step_labels(tmp_path: Path):
    run_id = "report_zh"
    payload = {
        "run_id": run_id,
        "suite": "gather_plus_menu",
        "skill": "woodcut",
        "pair_ok_count": 1,
        "total": 2,
        "edges": [
            {
                "edge_id": "main_hud_menu_system_menu",
                "pair_ok": True,
                "fail_tags": [],
                "node_zh_bar": "菜单",
                "evidence_dir": "knowledge/captures/transitions/report_zh/main_hud_menu_system_menu",
            },
            {
                "edge_id": "main_hud_bag_inventory",
                "pair_ok": False,
                "fail_tags": ["transition:locator_reject"],
                "evidence_dir": "knowledge/captures/transitions/report_zh/main_hud_bag_inventory",
            },
        ],
        "stopped_on": "main_hud_bag_inventory",
    }
    path = write_report(payload, root=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "| 步骤 | 结果 | 说明 | 栏上界面： | 证据 |" in text
    assert "| 打开菜单 | 通过 | - | 菜单 |" in text
    assert "| 打开背包 | 未通过 | 未能找到背包入口 | - |" in text
    assert "已停止：未能找到背包入口" in text
    for line in text.splitlines():
        if line.startswith("| ") and not line.startswith("| ---"):
            assert "pair_ok" not in line
            step_col = line.split("|")[1].strip()
            assert "main_hud" not in step_col
            assert "system_menu" not in step_col
    assert "stopped_on=" not in text


def test_write_report_user_abort_stopped_on(tmp_path: Path):
    payload = {
        "run_id": "abort_zh",
        "suite": "gather_plus_menu",
        "skill": None,
        "pair_ok_count": 0,
        "total": 11,
        "edges": [],
        "stopped_on": "user_abort",
    }
    path = write_report(payload, root=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "已强制结束" in text
    assert "user_abort" not in text


def test_progress_copy_is_plain_chinese_without_edge_ids():
    plan = plan_suite(suite="gather_plus_menu", skill="woodcut")
    ids = [s["edge_id"] for s in plan["steps"]]
    assert len(ids) == 11
    assert set(ids) <= set(EDGE_STEP_ZH)
    assert [s["zh"] for s in plan["steps"]] == [zh_step_label(i) for i in ids]
    bag = "main_hud_bag_inventory"
    assert format_progress_running(10, 11, bag) == "运行中 10/11：打开背包"
    assert format_progress_done(10, 11, bag) == "已完成 10/11：打开背包"
    assert format_progress_locator_stop(bag) == "已停止：未能找到背包入口"
    leaked = format_progress_fail_stop(bag, ["transition:locator_reject"])
    assert leaked == "已停止：未能找到背包入口"
    assert "edge" not in leaked.lower()
    assert "locator" not in leaked.lower()
    assert "main_hud" not in leaked
    for eid in ids:
        zh = zh_step_label(eid)
        assert zh and eid not in zh
        run = format_progress_running(1, 11, eid)
        assert eid not in run
        assert "locator" not in run


def test_profile_side_tab_rejects_top_right_hit():
    from types import SimpleNamespace

    from mm_assist.benchmarks.transition_accept import _profile_side_tab_hit_ok

    path = Path("knowledge/templates/profile/life_tab.png")
    frame = SimpleNamespace(shape=(1440, 1920))
    bad = SimpleNamespace(cx=1862, cy=57)
    good = SimpleNamespace(cx=96, cy=511)
    assert _profile_side_tab_hit_ok(path, bad, frame) is False
    assert _profile_side_tab_hit_ok(path, good, frame) is True


def test_life_grid_close_uses_profile_close_template():
    from mm_assist.benchmarks.transition_accept import load_manifest, resolve_locator

    man = load_manifest()
    loc = resolve_locator("life_grid_close_main_hud", skill="woodcut", manifest=man)
    assert loc["template"].endswith("profile/close.png")
    assert loc["required_templates"] == ["knowledge/templates/hud_entries/menu.png"]


def test_gather_detail_page_requires_action_markers(monkeypatch):
    from mm_assist.benchmarks.transition_accept import (
        _gather_detail_page_visible,
        confirm_node_from_frame,
    )

    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._gather_detail_markers_visible",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept.template_hit",
        lambda *a, **k: "close" in str(a[1] if len(a) > 1 else ""),
    )
    assert _gather_detail_page_visible(object()) is False
    assert confirm_node_from_frame(object(), "gather_detail") is False

    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._gather_detail_markers_visible",
        lambda *a, **k: True,
    )
    assert _gather_detail_page_visible(object()) is True


def test_gather_detail_close_requires_skill_cell_not_panel_close():
    man = load_manifest()
    loc = resolve_locator(
        "gather_detail_close_life_grid", skill="woodcut", manifest=man
    )
    assert "life_grid/woodcut.png" in loc["required_templates"][0]
    assert "life_grid/close" not in loc["required_templates"][0]


def test_open_skill_list_does_not_require_item_detail_button():
    man = load_manifest()
    loc = resolve_locator(
        "life_grid_skill_gather_detail", skill="woodcut", manifest=man
    )
    assert loc["required_templates"] == []


def test_profile_page_visible_rejects_life_grid_cells(monkeypatch):
    from mm_assist.benchmarks.transition_accept import (
        _life_grid_page_visible,
        _profile_page_visible,
        confirm_node_from_frame,
    )

    def _cells(frame, root=None):
        return True

    def _close(frame, rel, root=None, roi=None, threshold=0.78):
        return "profile/close.png" in rel

    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept._life_grid_cells_visible", _cells
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept.template_hit", _close
    )
    assert _profile_page_visible(object()) is False
    assert _life_grid_page_visible(object()) is True
    assert confirm_node_from_frame(object(), "profile") is False
    assert confirm_node_from_frame(object(), "life_grid") is True


def test_resolve_node_profile_before_life_grid(monkeypatch):
    from mm_assist.benchmarks.transition_accept import resolve_node_id
    from mm_assist.jobs.transitions import load_ui_transition_graph

    calls: list[str] = []

    def _confirm(frame, nid, root=None):
        calls.append(nid)
        return nid == "profile"

    monkeypatch.setattr(
        "mm_assist.benchmarks.transition_accept.confirm_node_from_frame",
        _confirm,
    )
    g = load_ui_transition_graph()
    probe = {"menu_open": True, "profile_open": True}
    assert resolve_node_id(probe, object(), g, prefer="profile") == "profile"
    assert calls[0] == "profile"


def test_profile_shell_edge_exempts_signal_delta():
    from mm_assist.jobs.transitions import assess_transition_captures_v2, load_ui_transition_graph

    g = load_ui_transition_graph()
    edge = next(e for e in g["edges"] if e["id"] == "profile_life_tab_life_grid")
    probe = {"menu_open": True, "profile_open": True}
    src = {"probe": probe, "method": "A", "action": "transition_accept", "args": {"capture_id": "x"}, "result": {"raw_sha256": "a"}}
    dst = {**src, "result": {"raw_sha256": "b", "template_hits": {
        "knowledge/templates/profile/life_tab.png": True,
        "knowledge/templates/life_grid/woodcut.png": True,
    }}}
    out = assess_transition_captures_v2(
        edge,
        src,
        dst,
        graph=g,
        required_templates=[
            "knowledge/templates/life_grid/woodcut.png",
        ],
    )
    assert out["pre_post_delta"] is True
    assert "transition:no_signal_delta" not in (out.get("fail_tags") or [])


def test_plan_suite_all_skills_resolved():
    from mm_assist.benchmarks.transition_accept import ALLOWED_SKILLS, plan_suite

    for sk in sorted(ALLOWED_SKILLS):
        plan = plan_suite(skill=sk)
        assert plan["skill"] == sk
        step = next(s for s in plan["steps"] if s["edge_id"] == "life_grid_skill_gather_detail")
        assert step["locator"]["resolved"] is True
        assert sk in step["locator"]["template"]


def test_skill_overview_not_gather_detail_on_live_capture():
    """125429Z 失败帧：help fuzzy 误当详情页，修复后概览须可点详情物。"""
    import cv2
    from pathlib import Path

    from mm_assist.benchmarks.transition_accept import (
        _click_gather_item_entry,
        _gather_detail_markers_visible,
        _skill_overview_panel_visible,
        confirm_node_from_frame,
    )

    cap = (
        Path(__file__).resolve().parents[1]
        / "knowledge/captures/transitions/20260824T125429Z/life_grid_skill_gather_detail/target.png"
    )
    if not cap.is_file():
        pytest.skip("live capture missing")
    frame = cv2.imread(str(cap))
    assert frame is not None and frame.size > 0
    assert _gather_detail_markers_visible(frame) is False
    assert _skill_overview_panel_visible(frame, skill="woodcut") is True
    assert confirm_node_from_frame(frame, "gather_detail") is False
    click = _click_gather_item_entry(frame, skill="woodcut")
    assert click.get("accepted") is True
    assert click.get("click") is not None


def test_pure_life_grid_not_skill_overview():
    """纯九门格（未点技能）不算技能概览，避免抢先点详情物。"""
    import cv2
    from pathlib import Path

    from mm_assist.benchmarks.transition_accept import (
        _life_grid_cells_visible,
        _skill_overview_panel_visible,
    )

    cap = Path(__file__).resolve().parents[1] / "logs/_probe_live2.png"
    if not cap.is_file():
        pytest.skip("live grid probe missing")
    frame = cv2.imread(str(cap))
    assert frame is not None and frame.size > 0
    assert _life_grid_cells_visible(frame) is True
    assert _skill_overview_panel_visible(frame, skill="woodcut") is False


def _write_inspect(
    tmp_path: Path,
    node: str,
    *,
    stack_ready: bool = True,
    zh: str = "",
    pid: int | None = None,
) -> None:
    import os
    import time

    from mm_assist.life_execute.ui_logic import inspect_page_path

    path = inspect_page_path(root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts_unix": time.time(),
        "pid": int(os.getpid() if pid is None else pid),
        "node": node,
        "zh": zh,
        "stack_ready": stack_ready,
        "stack_names": ["HudUIStepState"] if stack_ready else [],
        "source": "test",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_state_ui(tmp_path: Path, *, stp: int = 0) -> None:
    import time

    path = tmp_path / "logs" / "state_watch_ui.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts_unix": time.time(),
        "ui_params": f"stp={stp}",
        "stp": stp,
        "ui_stack_ready": True,
        "ui_stack_names": ["HudUIStepState"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_target_node_from_inspect_graph_node(tmp_path: Path):
    _write_inspect(tmp_path, "gather_skill:woodcut", zh="技能详情")
    graph, fine, _zh, why = node_from_inspect(root=tmp_path)
    assert why == ""
    assert graph == "gather_skill"
    assert fine == "gather_skill:woodcut"
    judged = apply_inspect_arrival(
        want_to="gather_detail",
        key_before=("stp=0|", "life_grid"),
        root=tmp_path,
    )
    assert judged["pair_ok"] is False
    assert "transition:wrong_node:gather_skill:woodcut" in judged["fail_tags"]


def test_inspect_empty_node_is_not_main_hud(tmp_path: Path):
    _write_inspect(tmp_path, "", stack_ready=False, zh="—")
    graph, fine, _zh, why = node_from_inspect(root=tmp_path)
    assert graph is None
    assert why == "inspect_not_ready"
    assert graph != "main_hud"
    judged = apply_inspect_arrival(
        want_to="main_hud",
        key_before=("stp=0|", ""),
        root=tmp_path,
    )
    assert judged["pair_ok"] is False
    assert "transition:inspect_not_ready" in judged["fail_tags"]
    _write_inspect(tmp_path, "", stack_ready=True, zh="—")
    graph2, _fine2, _zh2, why2 = node_from_inspect(root=tmp_path)
    assert graph2 is None
    assert why2 == "inspect_not_ready"


def test_inspect_overlay_not_registered(tmp_path: Path):
    _write_inspect(tmp_path, "overlay", zh="离开主界面")
    graph, fine, _zh, why = node_from_inspect(root=tmp_path)
    assert graph is None
    assert fine == "overlay"
    assert why == "node_unregistered"
    judged = apply_inspect_arrival(
        want_to="main_hud",
        key_before=("stp=0|", "overlay"),
        root=tmp_path,
    )
    assert judged["pair_ok"] is False
    assert "transition:node_unregistered:overlay" in judged["fail_tags"]


def test_inspect_foreign_pid_fails_closed(tmp_path: Path):
    import os

    _write_inspect(tmp_path, "main_hud", pid=os.getpid() + 1)
    with pytest.raises(TransitionAcceptError, match="状态栏") as ei:
        require_inspect_readable(root=tmp_path)
    assert "界面：" in str(ei.value)


def test_accept_requires_change_from_before_key(tmp_path: Path):
    from mm_assist.life_execute.ui_logic import ui_param_key

    _write_inspect(tmp_path, "system_menu", zh="菜单")
    _write_state_ui(tmp_path, stp=0)
    before = ui_param_key(root=tmp_path)
    judged = apply_inspect_arrival(
        want_to="system_menu",
        key_before=before,
        root=tmp_path,
    )
    assert judged["pair_ok"] is False
    assert "transition:no_change" in judged["fail_tags"]


def test_main_hud_stp_change_is_not_node_change(tmp_path: Path):
    from mm_assist.life_execute.ui_logic import ui_param_key

    _write_inspect(tmp_path, "main_hud", zh="主界面-待机")
    _write_state_ui(tmp_path, stp=0)
    g0, fine0, _zh0, why0 = node_from_inspect(root=tmp_path)
    k0 = ui_param_key(root=tmp_path)
    _write_state_ui(tmp_path, stp=1)
    _write_inspect(tmp_path, "main_hud", zh="主界面-动作中")
    g1, fine1, _zh1, why1 = node_from_inspect(root=tmp_path)
    k1 = ui_param_key(root=tmp_path)
    assert why0 == why1 == ""
    assert g0 == g1 == "main_hud"
    assert fine0 == fine1 == "main_hud"
    assert k0 != k1
    judged = apply_inspect_arrival(
        want_to="main_hud",
        key_before=k0,
        root=tmp_path,
    )
    assert judged["node_id"] == "main_hud"
    assert "transition:wrong_node" not in "".join(judged["fail_tags"])
