from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from mm_assist.config import load_config
from mm_assist.safety.lease import assert_protocol_ready

ALLOWED_CMDS = (
    "doctor",
    "watch",
    "spawn-hidden",
    "capture",
    "validate",
    "benchmark",
    "mouse-unlock",
    "transition-accept",
)


def _setup_log(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "mm_assist.log", encoding="utf-8"),
        ],
    )


def _require_window(cfg):
    from mm_assist.platform.window import enable_dpi_aware, find_window, is_foreground

    enable_dpi_aware()
    win = find_window(cfg.window.title_substr)
    if not win:
        raise SystemExit("game_window_missing")
    logging.getLogger(__name__).info(
        "window=%r client=%sx%s foreground=%s",
        win.title,
        win.width,
        win.height,
        is_foreground(win.hwnd),
    )
    return win


def cmd_doctor(cfg, *, as_json: bool = False) -> int:
    from mm_assist.config import ROOT
    from mm_assist.knowledge_paths import FRAME_JSON, SCENE_JSON
    from mm_assist.safety.lease import PROTOCOL_VERSION
    from mm_assist.platform.window import find_window

    checks: dict[str, object] = {
        "profile": "readonly_method_b",
        "protocol": PROTOCOL_VERSION,
        "config": str(ROOT / "config.yaml"),
        "read_only_only": bool(getattr(cfg, "read_only_only", True)),
        "knowledge_scene": SCENE_JSON.is_file(),
        "knowledge_frame": FRAME_JSON.is_file(),
        "rule": (ROOT / ".cursor/rules/10-live-test.mdc").is_file(),
    }
    try:
        assert_protocol_ready()
        checks["protocol_ok"] = True
    except Exception as exc:
        checks["protocol_ok"] = False
        checks["protocol_error"] = str(exc)
    win = find_window(cfg.window.title_substr)
    checks["game_window"] = bool(win)
    if not win:
        checks["game_window_error"] = "game_window_missing"
    if as_json:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
    else:
        for k, v in checks.items():
            print(f"{k}={v}")
    ok = bool(checks.get("protocol_ok")) and bool(checks.get("knowledge_scene"))
    return 0 if ok else 1


def cmd_watch(
    cfg,
    *,
    interval_s: float = 0.25,
    duration_s: float | None = None,
    restart: bool = False,
    detach: bool = False,
    install_shortcut: bool = False,
) -> int:
    from mm_assist.jobs.state_watch import run_state_watch
    from mm_assist.jobs.watch_lifecycle import install_watch_shortcut, stop_existing_state_watch

    if install_shortcut:
        path = install_watch_shortcut()
        print(f"shortcut={path}")
        if not restart and not detach:
            return 0
    if restart:
        info = stop_existing_state_watch()
        print(
            "stopped "
            f"pid={info.get('heartbeat_pid')} "
            f"closed={info.get('closed_windows')} "
            f"terminated={info.get('terminated')}"
        )
        from mm_assist.jobs.bar import _pid_alive

        hb = int(info.get("heartbeat_pid") or 0)
        leftover = [int(p) for p in (info.get("still_alive") or []) if int(p) > 0]
        deadline = time.time() + 8.0
        while time.time() < deadline:
            live = []
            if hb > 0 and _pid_alive(hb):
                live.append(hb)
            live.extend(p for p in leftover if _pid_alive(p))
            if not live:
                break
            time.sleep(0.2)
    if detach:
        import os
        from types import SimpleNamespace

        from mm_assist.config import ROOT

        env = os.environ.copy()
        src = str(ROOT / "src")
        env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else src + os.pathsep + env["PYTHONPATH"]
        os.environ.update({k: env[k] for k in ("PYTHONPATH",)})
        args = SimpleNamespace(argv=["watch", "--interval", str(interval_s)])
        return cmd_spawn_hidden(args)

    assert_protocol_ready()
    win = _require_window(cfg)
    return run_state_watch(cfg, win, interval_s=interval_s, duration_s=duration_s, lean=True)


def cmd_capture_pair(args) -> int:
    from mm_assist.jobs.pairs import open_perceive_sample, perceive_pair_job_specs

    specs = perceive_pair_job_specs(str(args.kind), pair_id=args.pair_id)
    role = str(args.role or "").strip()
    spec = None
    for item in specs:
        if str(item["args"]["pair_role"]) == role:
            spec = item
            break
    if spec is None:
        roles = [str(s["args"]["pair_role"]) for s in specs]
        print(f"未知 role={role!r}；可选 {roles}", file=sys.stderr)
        return 2
    job = open_perceive_sample(
        pair_id=str(spec["args"]["pair_id"]),
        pair_role=str(spec["args"]["pair_role"]),
        need_ui=str(spec["need_ui"]),
        title=str(spec["title"]),
        probe_menu=bool(spec["args"].get("probe_menu")),
    )
    print(json.dumps(job, ensure_ascii=False, indent=2))
    return 0


def cmd_capture_transition(args) -> int:
    from mm_assist.jobs.transitions import open_transition_frame

    edge = str(args.edge or "").strip()
    role = str(args.role or "").strip()
    if not edge:
        print("capture transition 须 --edge", file=sys.stderr)
        return 2
    if role not in {"source", "target"}:
        print("capture transition 须 --role source|target", file=sys.stderr)
        return 2
    job = open_transition_frame(edge, role, transition_id=args.pair_id)  # type: ignore[arg-type]
    print(json.dumps(job, ensure_ascii=False, indent=2))
    return 0


def cmd_validate_transition(args) -> int:
    from mm_assist.jobs.transitions import (
        format_validate_transition_report,
        validate_transition,
    )

    edge = str(args.edge or "").strip()
    if not edge or not args.source or not args.target:
        print("validate transition 须 --edge --source --target", file=sys.stderr)
        return 2
    out = validate_transition(
        edge,
        source_path=Path(args.source),
        target_path=Path(args.target),
    )
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(format_validate_transition_report(out), end="")
    return 0 if out.get("ok") else 1


def cmd_capture_hud_entry(args) -> int:
    from mm_assist.jobs.hud_entries import open_hud_entry_frame

    entry = str(args.entry or "").strip()
    role = str(args.role or "").strip()
    if not entry:
        print("capture hud-entry 须 --entry", file=sys.stderr)
        return 2
    if role not in {"source", "target"}:
        print("capture hud-entry 须 --role source|target", file=sys.stderr)
        return 2
    job = open_hud_entry_frame(
        entry,
        role,
        capture_id=args.pair_id,
        extra_args=(
            {"entry_queue": [p.strip() for p in str(args.queue).split(",") if p.strip()]}
            if str(getattr(args, "queue", "") or "").strip()
            else None
        ),
    )  # type: ignore[arg-type]
    print(json.dumps(job, ensure_ascii=False, indent=2))
    return 0


def cmd_capture_menu_button(cfg) -> int:
    from mm_assist.jobs.menu_buttons import run_menu_button_capture

    try:
        out = run_menu_button_capture(cfg, interactive=True)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_capture_ui_button(cfg, args) -> int:
    from mm_assist.jobs.menu_buttons import run_page_button_capture

    ids = [p.strip() for p in str(getattr(args, "queue", "") or "").split(",") if p.strip()]
    try:
        out = run_page_button_capture(cfg, interactive=True, ids=ids or None)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_validate_hud_entry(args) -> int:
    from mm_assist.jobs.hud_entries import (
        format_validate_hud_entry_report,
        validate_hud_entry,
    )

    entry = str(args.entry or "").strip()
    if not entry or not args.source:
        print("validate hud-entry 须 --entry --source", file=sys.stderr)
        return 2
    out = validate_hud_entry(
        entry,
        source_path=Path(args.source),
        target_path=Path(args.target) if args.target else None,
    )
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(format_validate_hud_entry_report(out), end="")
    assess = out.get("assess") if isinstance(out.get("assess"), dict) else {}
    return 0 if out.get("ok") and assess.get("click_accept_ok") is True else 1


def cmd_validate_pair(args) -> int:
    from mm_assist.jobs.pairs import format_validate_pair_report, validate_sample_pair

    out = validate_sample_pair(
        str(args.kind),
        pair_id=args.pair_id,
        closed_path=args.closed,
        open_path=args.open,
    )
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(format_validate_pair_report(out), end="")
    return 0 if out.get("ok") else 1


def cmd_benchmark_perception(cfg, args) -> int:
    mode = str(getattr(args, "mode", "") or "offline")
    out_dir = Path(args.out) if getattr(args, "out", None) else None
    if mode in {"items", "items-live", "items-batch"}:
        if mode == "items-live":
            from mm_assist.benchmarks.item_sense_bench import (
                format_report,
                run_live_full_catalog_probe,
            )
            from mm_assist.jobs.bar import require_state_watch_running
            from mm_assist.platform.window import bring_to_foreground, find_window, is_foreground

            require_state_watch_running()
            win = _require_window(cfg)
            bring_to_foreground(int(win.hwnd), force=True)
            if not is_foreground(int(win.hwnd)):
                print("FAIL: 游戏窗口未前台", file=sys.stderr)
                return 1
            skill_arg = str(getattr(args, "skill", "") or "all").strip()
            skills = None
            if skill_arg and skill_arg.lower() != "all":
                skills = tuple(s.strip() for s in skill_arg.split(",") if s.strip())
            out = run_live_full_catalog_probe(
                win,
                method=str(getattr(args, "method", "") or "sendinput"),
                skills=skills,
                scroll=not bool(getattr(args, "no_scroll", False)),
            )
            print(format_report(out), end="")
            if args.json:
                print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0 if out.get("ok") else 1

        from mm_assist.benchmarks.item_sense_bench import format_report, run_offline_benchmark

        out = run_offline_benchmark(
            frame_path=getattr(args, "frame", None),
            skill_id=str(getattr(args, "skill", "") or "woodcut"),
            item_name=str(getattr(args, "item", "") or "樹木"),
            out_dir=out_dir,
            batch=mode == "items-batch",
        )
        print(format_report(out), end="")
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out.get("ok") else 1

    if mode in {"process-live", "process-auto", "process-walk"}:
        from mm_assist.benchmarks.process_sense_bench import (
            format_auto_report,
            format_report as format_process_report,
            run_auto_nav_and_crop,
            run_live_facility_probe,
        )
        from mm_assist.jobs.bar import require_state_watch_running
        from mm_assist.platform.window import bring_to_foreground, is_foreground

        require_state_watch_running()
        win = _require_window(cfg)
        if mode in {"process-auto", "process-walk"}:
            bring_to_foreground(int(win.hwnd), force=True)
            if not is_foreground(int(win.hwnd)):
                print("FAIL: 游戏窗口未前台", file=sys.stderr)
                return 1
            out = run_auto_nav_and_crop(
                cfg,
                win,
                method=str(getattr(args, "method", "") or "sendinput"),
                walk_to_facility=mode == "process-walk",
            )
            print(format_auto_report(out), end="")
        else:
            out = run_live_facility_probe(cfg, win)
            print(format_process_report(out), end="")
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out.get("ok") else 1

    if mode == "pages-offline":
        from mm_assist.benchmarks.pages_offline import format_pages_table, run_pages_offline

        out = run_pages_offline(
            warmup=int(getattr(args, "warmup", 0) or 0),
            repeats=int(getattr(args, "repeats", 1) or 1),
            out_dir=out_dir,
        )
        print(format_pages_table(out), end="")
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out.get("ok") else 1

    from mm_assist.benchmarks.ui import (
        format_capture_rank_table,
        format_ocr_rank_table,
        run_live_capture_benchmark,
        run_offline_benchmark,
    )

    if mode == "capture":
        win = _require_window(cfg)
        out = run_live_capture_benchmark(win, out_dir=out_dir)
        print(format_capture_rank_table(out), end="")
    else:
        out = run_offline_benchmark(
            warmup=int(getattr(args, "warmup", 1) or 1),
            repeats=int(getattr(args, "repeats", 3) or 3),
            out_dir=out_dir,
        )
        print(format_ocr_rank_table(out), end="")
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_benchmark_memory(cfg, args) -> int:
    from mm_assist.benchmarks.memory import format_rank_table, wait_job_done
    from mm_assist.jobs.bar import open_test_job, require_state_watch_running

    _require_window(cfg)
    require_state_watch_running()
    job = open_test_job(
        "benchmark_memory",
        method="B",
        need_ui="游戏在主界面静止（只读内存基准，零键鼠）",
        title="memory-benchmark",
        args={
            "warmup": int(getattr(args, "warmup", 2) or 2),
            "repeats": int(getattr(args, "repeats", 8) or 8),
            "include_heap": True,
            "usb_safe": True,
        },
    )
    print("method=B：请在状态栏点【开始采集】", flush=True)
    done = wait_job_done()
    detail = done.get("result") if isinstance(done.get("result"), dict) else None
    if args.json:
        print(json.dumps(done, ensure_ascii=False, indent=2, default=str))
    elif isinstance(detail, dict):
        print(format_rank_table(detail), end="")
        print(f"path={detail.get('path')}")
    else:
        print(json.dumps(done, ensure_ascii=False, indent=2, default=str))
    return 0 if str(done.get("status")) == "ok" else 1


def cmd_mouse_unlock(_cfg) -> int:
    from mm_assist.safety.input import release_mouse_buttons

    release_mouse_buttons()
    print("mouse-unlock: released")
    return 0


def cmd_transition_accept(args) -> int:
    from mm_assist.benchmarks.transition_accept import (
        TransitionAcceptError,
        format_dry_run,
        promote_edge,
        run_suite,
        write_report,
    )

    action = str(getattr(args, "action", "") or "").strip()
    try:
        if action == "run":
            all_skills = bool(getattr(args, "all_skills", False))
            if all_skills:
                from mm_assist.benchmarks.live_click import run_one_click
                from mm_assist.benchmarks.transition_accept import (
                    ALLOWED_SKILLS,
                    load_manifest,
                    recover_to_main_hud,
                )
                from mm_assist.config import load_config
                from mm_assist.jobs.bar import require_state_watch_running
                from mm_assist.jobs.transitions import load_ui_transition_graph
                from mm_assist.perception.state_ui import read_state_ui
                from mm_assist.platform.capture import grab_client
                from mm_assist.platform.window import find_window

                last: dict[str, Any] = {}
                rc = 0
                live = bool(getattr(args, "live", False))
                dry = bool(getattr(args, "dry_run", False))
                method = str(getattr(args, "method", "") or "sendinput")
                suite = str(getattr(args, "suite", "") or "gather_plus_menu")
                for i, sk in enumerate(sorted(ALLOWED_SKILLS)):
                    if live and not dry and i > 0:
                        require_state_watch_running()
                        cfg = load_config()
                        win = find_window(cfg.window.title_substr)
                        if win is not None:
                            graph = load_ui_transition_graph()
                            manifest = load_manifest()
                            recover_to_main_hud(
                                win,
                                graph=graph,
                                manifest=manifest,
                                method=method,
                                root=None,
                                run_one_click=run_one_click,
                                grab_client=grab_client,
                                read_state_ui=read_state_ui,
                                skill=sk,
                            )
                    out = run_suite(
                        suite=suite,
                        skill=sk,
                        method=method,
                        dry_run=dry,
                        live=live,
                    )
                    last = out
                    if out.get("dry_run"):
                        print(format_dry_run(out), end="")
                    else:
                        print(
                            json.dumps(
                                {
                                    k: out[k]
                                    for k in (
                                        "run_id",
                                        "skill",
                                        "pair_ok_count",
                                        "total",
                                        "stopped_on",
                                        "enough",
                                        "report",
                                    )
                                    if k in out
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                    if out.get("stopped_on"):
                        rc = 1
                return rc if not last.get("dry_run") else 0
            out = run_suite(
                suite=str(getattr(args, "suite", "") or "gather_plus_menu"),
                skill=getattr(args, "skill", None),
                method=str(getattr(args, "method", "") or "sendinput"),
                dry_run=bool(getattr(args, "dry_run", False)),
                live=bool(getattr(args, "live", False)),
            )
            if out.get("dry_run"):
                print(format_dry_run(out), end="")
            else:
                print(json.dumps({k: out[k] for k in ("run_id", "pair_ok_count", "total", "stopped_on", "enough", "report") if k in out}, ensure_ascii=False, indent=2))
            return 0 if out.get("dry_run") or not out.get("stopped_on") else 1
        if action == "report":
            run_id = str(getattr(args, "run_id", "") or "").strip()
            if not run_id:
                print("transition-accept report 须 --run-id", file=sys.stderr)
                return 2
            payload = {
                "run_id": run_id,
                "suite": "",
                "skill": "",
                "pair_ok_count": 0,
                "total": 0,
                "edges": [],
            }
            from mm_assist.config import ROOT

            src = ROOT / "logs" / f"transition_accept_{run_id}.json"
            if src.is_file():
                payload = json.loads(src.read_text(encoding="utf-8"))
            path = write_report(payload)
            print(f"report={path}")
            return 0
        if action == "promote":
            run_id = str(getattr(args, "run_id", "") or "").strip()
            edge = str(getattr(args, "edge", "") or "").strip()
            if not run_id or not edge:
                print("transition-accept promote 须 --run-id --edge", file=sys.stderr)
                return 2
            out = promote_edge(
                run_id,
                edge,
                i_confirmed=bool(getattr(args, "i_confirmed", False)),
            )
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0
    except TransitionAcceptError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("transition-accept 须 run|report|promote|ui", file=sys.stderr)
    return 2


def cmd_spawn_hidden(args) -> int:
    import os

    from mm_assist.config import ROOT
    from mm_assist.platform.win_spawn import build_spawn_argv, infer_log_stem, spawn_logged

    try:
        cmd = build_spawn_argv(list(getattr(args, "argv", None) or []))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    stem = infer_log_stem(cmd)
    log_dir = ROOT / "logs"
    proc = spawn_logged(
        cmd,
        cwd=ROOT,
        env=os.environ.copy(),
        stdout_path=log_dir / f"{stem}.out.log",
        stderr_path=log_dir / f"{stem}.err.log",
    )
    print(f"pid={proc.pid} stem={stem}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mm_assist")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_doc = sub.add_parser("doctor")
    p_doc.add_argument("--json", action="store_true")

    p_memory = sub.add_parser('memory-audit', help='只读版本及字段布局核对，不登记支持版本')
    p_memory.add_argument('kind', choices=['build', 'layout', 'runtime'])
    p_memory.add_argument('--client-dir')
    p_memory.add_argument('--dump')
    p_memory.add_argument('--out')
    p_memory.add_argument('--pid', type=int)

    p_watch = sub.add_parser("watch")
    p_watch.add_argument("--interval", type=float, default=0.25)
    p_watch.add_argument("--duration", type=float, default=0.0)
    p_watch.add_argument("--restart", dest="restart", action="store_true")
    p_watch.add_argument("--detach", action="store_true")
    p_watch.add_argument("--install-shortcut", action="store_true")

    sub.add_parser("spawn-hidden")

    p_cap = sub.add_parser("capture")
    p_cap.add_argument("kind", choices=["pair", "transition", "hud-entry", "menu-button", "ui-button"])
    p_cap.add_argument("--pair-id", dest="pair_id", default=None)
    p_cap.add_argument("--role", default="")
    p_cap.add_argument("--kind-pair", dest="pair_kind", default="craft")
    p_cap.add_argument("--edge", default="")
    p_cap.add_argument("--entry", default="")
    p_cap.add_argument(
        "--queue",
        default="",
        help="ui-button 队列：items / process / gather-anchor / craft-tabs / facility-titles，或逗号分隔 id",
    )

    p_val = sub.add_parser("validate")
    p_val.add_argument("kind", choices=["pair", "transition", "hud-entry"])
    p_val.add_argument("--json", action="store_true")
    p_val.add_argument("--kind-pair", dest="pair_kind", default="craft")
    p_val.add_argument("--pair-id", dest="pair_id", default=None)
    p_val.add_argument("--closed", default=None)
    p_val.add_argument("--open", default=None)
    p_val.add_argument("--edge", default="")
    p_val.add_argument("--entry", default="")
    p_val.add_argument("--source", default=None)
    p_val.add_argument("--target", default=None)

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("kind", choices=["memory", "perception"])
    p_bench.add_argument("--json", action="store_true")
    p_bench.add_argument(
        "--mode",
        default="offline",
        help="offline / pages-offline / capture / items / items-live / items-batch / process-live / process-auto / process-walk",
    )
    p_bench.add_argument("--frame", default=None, help="items/items-batch 模式：详情页基准帧路径")
    p_bench.add_argument("--skill", default="woodcut", help="items 模式：技能 id；items-live 可用 all 或逗号分隔")
    p_bench.add_argument("--item", default="樹木", help="items 模式：详情物名")
    p_bench.add_argument("--method", default="sendinput", help="items-live 输入方法")
    p_bench.add_argument("--no-scroll", dest="no_scroll", action="store_true", help="items-live 禁用列表滚动")
    p_bench.add_argument("--warmup", type=int, default=1)
    p_bench.add_argument("--repeats", type=int, default=3)
    p_bench.add_argument("--out", default=None)

    sub.add_parser("mouse-unlock")

    p_ta = sub.add_parser("transition-accept")
    p_ta.add_argument("action", choices=["run", "report", "promote"])
    p_ta.add_argument("--suite", default="gather_plus_menu")
    p_ta.add_argument("--skill", default="woodcut")
    p_ta.add_argument("--all-skills", dest="all_skills", action="store_true")
    p_ta.add_argument("--method", default="sendinput")
    p_ta.add_argument("--dry-run", action="store_true")
    p_ta.add_argument("--live", action="store_true")
    p_ta.add_argument("--run-id", dest="run_id", default="")
    p_ta.add_argument("--edge", default="")
    p_ta.add_argument("--i-confirmed", action="store_true")

    args, rest = parser.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    args.argv = rest
    cfg = load_config()
    _setup_log(cfg.resolve(cfg.paths.logs))
    if args.cmd == "doctor":
        return cmd_doctor(cfg, as_json=bool(getattr(args, "json", False)))
    if args.cmd == 'memory-audit':
        from mm_assist.memory.layout_audit import main as audit_main
        audit_args = [args.kind]
        for name in ('client_dir', 'dump', 'out', 'pid'):
            value = getattr(args, name)
            if value is not None:
                audit_args.extend(['--'+name.replace('_', '-'), str(value)])
        return audit_main(audit_args)
    if args.cmd == "watch":
        dur = float(args.duration) if args.duration and args.duration > 0 else None
        return cmd_watch(
            cfg,
            interval_s=args.interval,
            duration_s=dur,
            restart=bool(getattr(args, "restart", False)),
            detach=bool(getattr(args, "detach", False)),
            install_shortcut=bool(getattr(args, "install_shortcut", False)),
        )
    if args.cmd == "spawn-hidden":
        args.argv = rest
        return cmd_spawn_hidden(args)
    if args.cmd == "capture":
        if args.kind == "pair":
            args.kind = getattr(args, "pair_kind", "craft")
            return cmd_capture_pair(args)
        if args.kind == "transition":
            return cmd_capture_transition(args)
        if args.kind == "menu-button":
            return cmd_capture_menu_button(cfg)
        if args.kind == "ui-button":
            return cmd_capture_ui_button(cfg, args)
        return cmd_capture_hud_entry(args)
    if args.cmd == "validate":
        if args.kind == "pair":
            args.kind = getattr(args, "pair_kind", "craft")
            return cmd_validate_pair(args)
        if args.kind == "transition":
            return cmd_validate_transition(args)
        return cmd_validate_hud_entry(args)
    if args.cmd == "benchmark":
        if args.kind == "memory":
            return cmd_benchmark_memory(cfg, args)
        return cmd_benchmark_perception(cfg, args)
    if args.cmd == "mouse-unlock":
        return cmd_mouse_unlock(cfg)
    if args.cmd == "transition-accept":
        return cmd_transition_accept(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
