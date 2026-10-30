"""状态栏内的单场景实机验收；复用正式执行器，一次租约、一次执行、不自动重试。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from mm_assist.config import ROOT
from mm_assist.life_plan.recipes import GATHER_NODE, RECIPES, RETIRED_ITEMS, SKILL_GATHER, TRAINING_UPGRADES, can_afford
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem, SkillGrowth, SLOT_CAP
from mm_assist.perception.item_sense import fold_zh

GATHER_VERIFY_S = 90.0
SNAP_VERIFY_BUDGET_S = 3.0
START_STP_WAIT_S = 90.0
STABLE_IDLE_S = 0.8


def scenarios() -> dict[str, PlanItem]:
    out = {f"gather:{mat}": PlanItem("test", "gather", f"采集 {mat}", recipe=mat, skill=skill, detail=detail)
           for mat, (skill, detail) in GATHER_NODE.items()}
    out.update({f"feed:{name}": PlanItem("test", "feed", f"投料 {name}", recipe=name, facility=rec.facility)
                for name, rec in RECIPES.items()})
    out.update({f"claim:{fac}": PlanItem("test", "claim", f"领取 {fac}", facility=fac)
                for fac in ("metal", "wood", "cloth", "medicine")})
    upgrades = ((skill, detail) for skill, tiers in TRAINING_UPGRADES.items()
                for _, detail in tiers)
    for skill, detail in (*SKILL_GATHER, *upgrades):
        out[f"train:{detail}"] = PlanItem("test", "gather", f"练采集 {detail}", recipe=detail, skill=skill, detail=detail, gather_mode="skill")
    out["repair"] = PlanItem("test", "repair", "一键修理")
    from mm_assist.perception.item_sense import GATHER_SKILL_IDS, iter_tier_items

    nav_targets = [
        "main_hud", "system_menu", "profile", "life_grid", "inventory",
        "craft_hub", "craft_quick", "craft_make", "craft_alchemy", "insect_items",
        "obtain_method:wood:樹液",
        "item_info:wood:樹液",
        "facility:metal", "facility:wood", "facility:cloth", "facility:medicine",
        "facility_recipe:metal:鋼錠", "facility_recipe:wood:木材",
        "facility_recipe:cloth:布料", "facility_recipe:medicine:淨淨蘑菇孢子",
    ]
    nav_targets.extend(f"gather_skill:{sid}" for sid in GATHER_SKILL_IDS)
    nav_targets.extend(f"item_info:{row['skill_id']}:{row['name']}" for row in iter_tier_items()
                       if row['name'] not in RETIRED_ITEMS)
    for target in nav_targets:
        out[f"nav:{target}"] = PlanItem("test", "gather", f"仅切页 {target}", detail=target)
    return out


_NAV_FACILITY_IDS = frozenset({"metal", "wood", "cloth", "medicine", "leather", "food"})


def parse_nav_detail(detail: str) -> tuple[str, str, str, str]:
    """切页目标 → (ensure_node 节点, skill_id, facility, 物名/配方名)。"""
    parts = [p for p in str(detail or "").split(":") if p != ""]
    target = parts[0] if parts else ""
    rest = parts[1:]
    if target == "facility_recipe":
        return target, "", rest[0] if rest else "", rest[1] if len(rest) > 1 else ""
    if target in {"gather_detail", "item_info", "obtain_method"}:
        mid = rest[0] if rest else ""
        name = rest[1] if len(rest) > 1 else ""
        if mid in _NAV_FACILITY_IDS:
            return target, "", mid, name
        return target, mid, "", name
    if target == "facility":
        return target, "", rest[0] if rest else "", ""
    if target == "gather_skill":
        return target, rest[0] if rest else "", "", ""
    return target, rest[0] if rest else "", "", ""


def _token_in_node(node: str, token: str) -> bool:
    if not token:
        return True
    from mm_assist.perception.item_sense import fold_zh

    parts = str(node or "").split(":")
    if len(parts) < 2:
        return False
    want = fold_zh(token)
    return any(fold_zh(p) == want for p in parts[1:])


def insect_items_navigation_node(frame, where, *, captured_at: float) -> str:
    """Independent current-frame proof for the page-only calibration scene."""
    from mm_assist.life_execute.ui_logic import graph_node
    from mm_assist.perception.insect_obtain_sense import phase
    if (not getattr(where, 'stack_ready', False)
            or getattr(where, 'stack_sampled_ts', 0) <= captured_at
            or graph_node(where.node) not in {'craft_shell', 'craft_make', 'craft_items'}):
        return ''
    return 'insect_items' if phase(frame, '滴答螢火蟲') == 'items' else ''


def nav_page_matched(observed: str, detail: str) -> bool:
    from mm_assist.life_execute.ui_logic import graph_node

    node = str(observed or "")
    want = str(detail or "")
    if node == want:
        return True
    target, skill, facility, item = parse_nav_detail(want)
    if target == "gather_detail":
        return graph_node(node) in {"gather_detail", "item_info"} and _token_in_node(node, item)
    if target == "item_info":
        return graph_node(node) in {"item_info", "gather_detail"} and _token_in_node(node, item)
    if target == "obtain_method":
        return graph_node(node) == target and _token_in_node(node, item)
    if target == "facility_recipe":
        return graph_node(node) == "facility_recipe" and _token_in_node(node, item) and (
            not facility or facility in node.split(":")
        )
    if not item and not skill and not facility:
        return graph_node(node) == want
    if target == "gather_skill":
        return node == want
    if target == "facility":
        return graph_node(node) == "facility" and facility in node
    return False


def precondition(item: PlanItem, snap: LifeSnapshot) -> str:
    if not snap.ok:
        return snap.error or "snapshot_unreadable"
    if item.gather_mode == 'skill':
        from mm_assist.life_plan.planner import training_target_allowed

        if not training_target_allowed(item, snap):
            return 'training_level_unverified'
    slots = snap.facilities.get(item.facility, ())
    if item.kind == "feed":
        if len(slots) >= SLOT_CAP:
            return "slots_full"
        if not can_afford(item.recipe, snap.items):
            return "no_mats"
    if item.kind == "claim" and not any(s.done for s in slots):
        return "nothing_to_claim"
    if item.recipe == "裝水的瓶子" and snap.items.get("空瓶", 0) <= 0:
        return "no_empty_bottle"
    return ""


def _skill_named(snap: LifeSnapshot, name: str) -> SkillGrowth | None:
    want = str(name or "")
    for row in snap.skills:
        if row.name == want:
            return row
    return None


def _snapshot_kw(item: PlanItem) -> dict[str, bool]:
    return {
        "include_tools": item.kind == "repair",
        "include_skills": item.gather_mode == "skill",
    }


def _slot_recipe_matches(slot_name: str, plan_name: str) -> bool:
    """设施槽名可带括号档，如 鐵錠(鐵礦石) 对计划 鐵錠。"""
    a, b = fold_zh(slot_name), fold_zh(plan_name)
    if not a or not b:
        return False
    if a == b:
        return True
    return a.startswith(b + "(") or b.startswith(a + "(")


_GATHER_RETRY = frozenset({
    "no_item_gain",
    "no_skill_gain",
    "skill_unreadable",
    "skill_name_missing",
    "mem_budget",
    "after_unreadable",
})


def verdict(item: PlanItem, before: LifeSnapshot, after: LifeSnapshot, result: ExecuteResult) -> tuple[str, str]:
    if not result.ok:
        return "failed", result.status
    if not after.ok:
        return "unverified", after.error or "after_unreadable"
    if item.kind == "gather":
        if item.gather_mode == "skill":
            if "skills_unreadable" in (after.error or "") or not after.skills:
                return "unverified", "skill_unreadable"
            prev = _skill_named(before, item.skill)
            nxt = _skill_named(after, item.skill)
            if prev is None or nxt is None:
                return "unverified", "skill_name_missing"
            if nxt.level > prev.level or nxt.exp > prev.exp:
                return "passed", f"skill_delta={item.skill}:{prev.level}.{prev.exp}->{nxt.level}.{nxt.exp}"
            return "failed", "no_skill_gain"
        delta = after.items.get(item.recipe, 0) - before.items.get(item.recipe, 0)
        return ("passed", f"item_delta={delta}") if delta > 0 else ("failed", "no_item_gain")
    if item.kind == "feed":
        rec = RECIPES[item.recipe]
        consumed = all(before.items.get(k, 0) - after.items.get(k, 0) >= v for k, v in rec.inputs.items())
        def count(snap):
            return sum(
                _slot_recipe_matches(s.recipe, item.recipe)
                for s in snap.facilities.get(item.facility, ())
            )
        queued = count(after) > count(before)
        return ("passed", "materials_consumed_and_queued") if consumed and queued else ("failed", "feed_not_verified")
    if item.kind == "claim":
        done = [s for s in before.facilities.get(item.facility, ()) if s.done]
        gain = any(
            after.items.get(s.recipe, 0) > before.items.get(s.recipe, 0)
            or any(
                _slot_recipe_matches(s.recipe, key) and after.items.get(key, 0) > before.items.get(key, 0)
                for key in after.items
            )
            for s in done
        )
        fewer = sum(s.done for s in after.facilities.get(item.facility, ())) < len(done)
        return ("passed", "claimed_items_received") if gain and fewer else ("failed", "claim_not_verified")
    if item.kind == "repair":
        def _dur_map(snap: LifeSnapshot) -> dict[str, float]:
            return {f"{t.name}:{i}": float(t.durability) for i, t in enumerate(snap.tools)}
        if result.status == "skip":
            return "unverified", "repair_skipped"
        prev, nxt = _dur_map(before), _dur_map(after)
        if prev and nxt and any(nxt.get(k, old) > old for k, old in prev.items()):
            return "passed", "durability_rose"
        if not after.tools:
            return "unverified", "tools_unreadable"
        return "failed", "no_durability_gain"
    return "unverified", "independent_repair_evidence_required"


def _gather_idle(rec: dict) -> bool:
    zh = str((rec or {}).get("zh") or "")
    if "动作中" in zh:
        return False
    try:
        stp = (rec or {}).get("stp")
        if stp is not None and int(stp) == 1:
            return False
    except (TypeError, ValueError):
        pass
    return "主界面-待机" in zh


def wait_hud_idle(
    read_ui,
    *,
    timeout_s: float,
    stable_s: float = STABLE_IDLE_S,
    sleep=time.sleep,
    clock=time.time,
    cancelled=None,
) -> dict:
    """等到主界面不在动作中并连稳 stable_s。超时返回空 dict。"""
    deadline = float(clock()) + float(timeout_s)
    stable_since: float | None = None
    while clock() < deadline:
        if cancelled is not None and cancelled():
            return {}
        rec = read_ui() or {}
        zh = str(rec.get("zh") or "")
        try:
            stp_now = rec.get("stp")
            walking = stp_now is not None and int(stp_now) == 1
        except (TypeError, ValueError):
            walking = False
        node = str(rec.get("node") or "")
        hud_busy = node == "main_hud" and (walking or "动作中" in zh)
        if hud_busy or node in {"unknown", "", "None"}:
            stable_since = None
            sleep(0.4)
            continue
        if stable_since is None:
            stable_since = float(clock())
        if float(clock()) - stable_since >= float(stable_s):
            return rec
        sleep(0.4)
    return {}


def snapshot_after_execute(
    item: PlanItem,
    before: LifeSnapshot,
    result: ExecuteResult,
    *,
    include_tools: bool,
    take_snapshot,
    read_ui,
    clock=None,
    sleep=None,
    timeout_s: float | None = None,
    include_skills: bool = False,
) -> LifeSnapshot:
    """gather 非 nav：等主界面待机后再读材料/技能，3s 快照可重试到净增或超时。"""
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    if not result.ok:
        return LifeSnapshot(ok=False, error="execution_failed")
    budget = 12.0 if include_skills else SNAP_VERIFY_BUDGET_S
    snap_kw = {"include_tools": include_tools, "include_skills": include_skills}
    if item.kind == "claim":
        deadline = clock() + 8.0
        after = take_snapshot(budget_s=budget, **snap_kw)
        status, reason = verdict(item, before, after, result)
        while clock() < deadline and status != "passed":
            if reason != "claim_not_verified":
                break
            sleep(0.4)
            after = take_snapshot(budget_s=budget, **snap_kw)
            status, reason = verdict(item, before, after, result)
        return after
    if item.kind != "gather":
        return take_snapshot(budget_s=budget, **snap_kw)
    deadline = clock() + float(GATHER_VERIFY_S if timeout_s is None else timeout_s)
    while clock() < deadline:
        if _gather_idle(read_ui() or {}):
            break
        sleep(0.2)
    after = take_snapshot(budget_s=budget, **snap_kw)
    status, reason = verdict(item, before, after, result)
    while clock() < deadline and status != "passed":
        if reason not in _GATHER_RETRY:
            break
        sleep(0.5)
        if clock() < deadline and not _gather_idle(read_ui() or {}):
            while clock() < deadline and not _gather_idle(read_ui() or {}):
                sleep(0.2)
        after = take_snapshot(budget_s=budget, **snap_kw)
        status, reason = verdict(item, before, after, result)
    return after


@dataclass
class Evidence:
    folder: Path
    win: object
    scenario: str
    started: float = field(default_factory=time.monotonic)
    progress: str = ""
    events: list = field(default_factory=list)
    stages: list = field(default_factory=list)
    progress_log: list = field(default_factory=list)
    _stage: str = ""
    _stage_t0: float = 0.0
    before_input: Callable[[], float | None] | None = None

    def capture(self, label: str, xy=None) -> None:
        from mm_assist.platform.capture import grab_client, save_capture

        row = {"ts": time.time(), "elapsed_s": round(time.monotonic() - self.started, 3),
               "label": label, "progress": self.progress, "xy": xy}
        for name in ("ui_inspect.json", "state_watch_ui.json"):
            try:
                row[name] = json.loads((ROOT / "logs" / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                row[name] = None
        try:
            row["image"] = str(save_capture(grab_client(self.win), self.folder, prefix=f"{len(self.events):03d}"))
        except Exception as exc:
            row["capture_error"] = str(exc)
        self.events.append(row)
        (self.folder / "events.json").write_text(json.dumps(self.events, ensure_ascii=False, indent=2), encoding="utf-8")

    def note(self, name: str, **extra) -> dict:
        now = time.monotonic()
        if self._stage:
            self.stages.append({
                "name": self._stage,
                "t0": round(self._stage_t0 - self.started, 3),
                "t1": round(now - self.started, 3),
                "elapsed_s": round(now - self._stage_t0, 3),
            })
        self._stage = name
        self._stage_t0 = now
        row = {"ts": time.time(), "elapsed_s": round(now - self.started, 3), "stage": name}
        row.update(extra)
        self.progress_log.append(row)
        return row


def bounded_stacks(*, max_threads: int = 8, max_frames: int = 6) -> list[dict]:
    rows = []
    frames = sys._current_frames()
    for ident, frame in list(frames.items())[:max_threads]:
        stack = traceback.format_stack(frame, limit=max_frames)
        rows.append({"ident": ident, "stack": [ln.strip() for ln in stack[-max_frames:]]})
    return rows


_OPERATOR_EXES = {
    "python.exe",
    "pythonw.exe",
    "cursor.exe",
    "code.exe",
}


def classify_focus(*, is_game: bool, fg_exe: str, fg_title: str, same_pid: bool) -> str:
    """game=游戏；operator=本机操作台/桌面/IDE（允许一次前置）；user=用户切到其它程序。"""
    if is_game:
        return "game"
    if same_pid:
        return "statusbar"
    exe = str(fg_exe or "").strip().lower()
    title = str(fg_title or "")
    if exe in _OPERATOR_EXES or title == "Program Manager":
        return "operator"
    if exe or title:
        return "user"
    return "unknown"


def probe_focus(game_hwnd: int) -> dict:
    out = {
        "game_hwnd": int(game_hwnd or 0),
        "fg_hwnd": 0,
        "fg_exe": "",
        "fg_title": "",
        "is_game": False,
        "kind": "unknown",
    }
    try:
        import ctypes
        import win32gui

        fg = int(win32gui.GetForegroundWindow() or 0)
        out["fg_hwnd"] = fg
        out["fg_title"] = str(win32gui.GetWindowText(fg) or "")[:80]
        pid = ctypes.c_ulong(0)
        if fg:
            ctypes.windll.user32.GetWindowThreadProcessId(int(fg), ctypes.byref(pid))
        from mm_assist.platform.window import _hwnd_exe_basename

        out["fg_exe"] = _hwnd_exe_basename(fg) if fg else ""
        out["is_game"] = bool(game_hwnd) and fg == int(game_hwnd)
        out["kind"] = classify_focus(
            is_game=bool(out["is_game"]),
            fg_exe=str(out["fg_exe"]),
            fg_title=str(out["fg_title"]),
            same_pid=int(pid.value or 0) == int(os.getpid()),
        )
    except Exception as exc:
        out["error"] = str(exc)
    return out


_ACTIVE: Evidence | None = None


def record_input(win, kind: str, coords: list[int]) -> float | None:
    from mm_assist.life_plan.execution_records import input_requested
    input_requested(kind, coords, [getattr(win, 'width', 0), getattr(win, 'height', 0)])
    if _ACTIVE is not None and int(win.hwnd) == int(_ACTIVE.win.hwnd):
        _ACTIVE.capture(kind, coords)
        if _ACTIVE.before_input is not None:
            return _ACTIVE.before_input()
    return None


def check_input_deadline(deadline: float | None) -> None:
    if deadline is None:
        return
    elapsed = time.monotonic() - (deadline - 1.0)
    if _ACTIVE is not None and _ACTIVE.events:
        _ACTIVE.events[-1]["focus_to_dispatch_s"] = round(elapsed, 4)
    if elapsed >= 1.0:
        raise RuntimeError("focus_click_timeout")


def code_digest() -> str:
    h = hashlib.sha256()
    for path in sorted((ROOT / "src" / "mm_assist").rglob("*.py")):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


_LOADED_CODE = code_digest()


def wait_start_ui(read, *, stopped=lambda: False, timeout_s=10.0, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout_s
    while not stopped():
        row = read()
        if row and row.get("stack_ready") and str(row.get("node") or "") not in {"", "unknown"}:
            return row
        if clock() >= deadline:
            return {}
        sleep(0.1)
    return {}


def inspect_nav_watch_ready(
    *,
    root: Path | None = None,
    now: float | None = None,
    expected_pid: int | None = None,
) -> tuple[dict, str]:
    """读 state-watch 就绪快照。不睡、不扫内存、不替代 resolve_navigation。"""
    from mm_assist.jobs.state_watch import load_watch_ready, watch_ready_refuse_reason

    row = load_watch_ready(root=root or ROOT, now=now)
    reason = watch_ready_refuse_reason(
        row, now=now, expected_pid=expected_pid, for_page_nav=True
    )
    return row, reason


def run_scenario(cfg, win, scenario: str, on_hint=lambda _: None) -> dict:
    """只能由状态栏工作线程调用。报告区分条件不足、流程失败和真实验收通过。"""
    from mm_assist.jobs.bar import require_state_watch_running
    from mm_assist.life_plan.execute import try_execute
    from mm_assist.life_plan.snapshot import take_snapshot
    from mm_assist.life_plan.controller import set_current_action
    from mm_assist.safety.lease import bounded_test_lease
    from mm_assist.tool_repair import clear_op_cancel, request_op_cancel

    global _ACTIVE
    item = scenarios()[scenario]
    navigation_only = scenario.startswith("nav:")
    require_state_watch_running()
    if win is None:
        raise RuntimeError("无游戏窗口")
    run_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    folder = ROOT / "logs" / "life_tests" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "scenario": scenario, "pid": os.getpid(), "code": _LOADED_CODE,
              "started_at": time.time(), "status": "preparing", "need_ui": "recognized_current_page"}
    def save():
        data = json.dumps(report, ensure_ascii=False, indent=2)
        (folder / "report.json").write_text(data, encoding="utf-8")
        latest = ROOT / "logs" / "life_test_latest.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        tmp = latest.with_suffix(".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(latest)
    save()
    evidence = Evidence(folder, win, scenario)
    clear_op_cancel()
    try:
        if navigation_only:
            ready, ready_reason = inspect_nav_watch_ready(root=ROOT)
            report["watch_ready"] = ready
            save()
            if ready_reason:
                report.update(status="blocked", reason=ready_reason)
                return report
        evidence.note("snapshot")
        before = (
            LifeSnapshot(ok=True)
            if navigation_only
            else take_snapshot(budget_s=45, **_snapshot_kw(item))
        )
        report["before"] = asdict(before)
        report["stages"] = evidence.stages
        save()
        reason = precondition(item, before)
        if reason:
            report.update(status="blocked", reason=reason)
            return report
        from mm_assist.life_execute.ui_logic import LAST_UI_WAIT, read_inspect_record
        from mm_assist.tool_repair import op_cancelled
        evidence.note("wait_start_ui")
        def navigation_record():
            from mm_assist.life_execute.ui_logic import resolve_navigation
            from mm_assist.platform.capture import grab_client
            t0 = time.monotonic()
            captured_at = time.time()
            frame = grab_client(win)
            capture_ms = round((time.monotonic() - t0) * 1000, 1)
            where = resolve_navigation(frame)
            node = (insect_items_navigation_node(frame, where, captured_at=captured_at)
                    if scenario == 'nav:insect_items' else '') or where.node
            return {"node": node, "stack_ready": where.stack_ready,
                    "cost_ms": {"capture": capture_ms, **getattr(where, "cost_ms", {})}}
        current = wait_start_ui(navigation_record if navigation_only else read_inspect_record, stopped=op_cancelled)
        if op_cancelled():
            report.update(status="failed", reason="cancel")
            return report
        if not current or str(current.get("node")) in {"unknown", "", "None"}:
            evidence.capture("unrecognized_start")
            report.update(status="blocked", reason="unknown_start_ui")
            return report
        report["need_ui"] = current.get("node")
        if navigation_only:
            report["navigation_before"] = current
        if (not navigation_only and current.get("node") == "main_hud"
                and (current.get("stp") == 1 or "动作中" in str(current.get("zh") or ""))):
            evidence.note("wait_stp")
            rec = wait_hud_idle(
                read_inspect_record,
                timeout_s=START_STP_WAIT_S,
                cancelled=op_cancelled,
            )
            if op_cancelled():
                report.update(status="failed", reason="cancel")
                return report
            if not rec:
                evidence.capture("busy_start")
                report.update(status="blocked", reason="stp_busy")
                return report
            current = rec
        from mm_assist.safety.autofarm_gates import reset_stage_gates
        reset_stage_gates()
        from mm_assist.platform.window import focus_window
        evidence.note("focus")
        focus0 = probe_focus(int(win.hwnd))
        report["focus_start"] = focus0
        focused = False
        if not navigation_only:
            if focus0.get("kind") == "user":
                report.update(status="failed", reason="user_switched_away")
                report["focus_fail"] = focus0
                return report
            if not focus0.get("is_game"):
                focused = bool(focus_window(win.hwnd, force=True))
                focus1 = probe_focus(int(win.hwnd))
                report["focus_after_one"] = focus1
                report["focus_window_ok"] = focused
                if focus1.get("kind") == "user":
                    report.update(status="failed", reason="user_switched_away")
                    report["focus_fail"] = focus1
                    return report
                if not focus1.get("is_game"):
                    report.update(status="failed", reason="focus_failed")
                    report["focus_fail"] = focus1
                    return report
        else:
            focus_attempted = False
            def focus_at_input():
                nonlocal focus_attempted
                if focus_attempted:
                    return None
                focus_attempted = True
                if op_cancelled():
                    raise RuntimeError("cancel")
                # Capture and recognition finish in the background before this point.
                if not focus_window(win.hwnd, force=True):
                    raise RuntimeError("focus_failed")
                focused_at = time.monotonic()
                report["focus_window_ok"] = True
                report["focus_after_one"] = probe_focus(int(win.hwnd))
                if not report["focus_after_one"].get("is_game"):
                    raise RuntimeError("focus_failed")
                return focused_at + 1.0
            evidence.before_input = focus_at_input
        _ACTIVE = evidence
        evidence.capture("before")
        def progress(text):
            evidence.progress = str(text)
            evidence.note("progress", text=str(text), ui_wait=dict(LAST_UI_WAIT))
            set_current_action("测试 " + str(text))
            on_hint("测试 " + str(text))
            report["progress"] = str(text)
            report["stages"] = list(evidence.stages)
            report["progress_log"] = evidence.progress_log[-20:]
            save()
        report["status"] = "running"
        save()
        if not navigation_only or scenario == 'nav:insect_items':
            evidence.note("wait_stp_pre_lease")
            def start_record():
                rec = read_inspect_record()
                if scenario == 'nav:insect_items':
                    from mm_assist.life_plan.action_finish import read_completion_stp
                    sample = read_completion_stp()
                    if not sample.fresh or sample.stp != 0:
                        return {'node':'unknown'}
                return rec
            rec = wait_hud_idle(
                start_record,
                timeout_s=START_STP_WAIT_S,
                cancelled=op_cancelled,
            )
            if op_cancelled():
                report.update(status="failed", reason="cancel")
                return report
            if not rec:
                evidence.capture("busy_start")
                report.update(status="blocked", reason="stp_busy")
                return report
        evidence.note("execute")
        if navigation_only:
            with bounded_test_lease(hwnd=win.hwnd, intent=item.title, lease_s=60):
                from mm_assist.life_execute.ui_nav import ensure_node
                from mm_assist.platform.capture import grab_client
                from mm_assist.safety.input import click_client, drag_client
                target, skill, facility, item_name = parse_nav_detail(item.detail)
                if scenario == 'nav:insect_items':
                    from mm_assist.life_execute.insect_obtain import execute_insect_obtain
                    result = execute_insect_obtain(cfg, win, mat='滴答螢火蟲', entry_only=True,
                        grab=lambda: grab_client(win),
                        do_click=lambda x, y: click_client(win, x, y, hold_s=0.08),
                        on_progress=progress)
                else:
                    result = ensure_node(
                        target, skill_id=skill, facility=facility, item_name=item_name,
                        grab=lambda: grab_client(win),
                        click=lambda x, y: click_client(win, x, y, hold_s=0.08),
                        drag_fn=lambda x0, y0, x1, y1: drag_client(win, x0, y0, x1, y1),
                        on_progress=progress,
                    )
        elif item.kind == "repair":
            input_deadline = time.monotonic() + 60.0
            with bounded_test_lease(hwnd=win.hwnd, intent=item.title, lease_s=60):
                result = try_execute(
                    item, cfg=cfg, win=win, items=before.items,
                    facilities=before.facilities, on_progress=progress, phase="nav",
                )
                if result.status == "walking":
                    evidence.note("wait_repair_arrive")
                    from mm_assist.nav.map_atlas import wait_stop_then_dialog

                    arrived, _st, detail = wait_stop_then_dialog(
                        cfg, win, timeout_s=90.0, idle_abort_s=0.0, settle_s=0.0
                    )
                    if op_cancelled():
                        report.update(status="failed", reason="cancel")
                        return report
                    result = (
                        ExecuteResult(True, "at_dialog") if arrived
                        else ExecuteResult(False, "arrive_wait", detail or "到站等待失败")
                    )
                if result.status == "at_dialog" and time.monotonic() >= input_deadline:
                    result = ExecuteResult(False, "scenario_timeout", "输入租约已到期，不再执行修理对话")
                elif result.status == "at_dialog":
                    result = try_execute(
                        item, cfg=cfg, win=win, items=before.items,
                        facilities=before.facilities, on_progress=progress, phase="dialog",
                    )
        else:
            with bounded_test_lease(hwnd=win.hwnd, intent=item.title, lease_s=60):
                result = try_execute(item, cfg=cfg, win=win, items=before.items,
                                 facilities=before.facilities, on_progress=progress)
        evidence.note("after")
        evidence.capture("after")
        if navigation_only:
            # An independent final read may land on a short stack rescan even
            # after navigation confirmed arrival. Wait for fresh evidence only;
            # never repeat the successful click or reuse its previous reading.
            observed = wait_start_ui(navigation_record, stopped=op_cancelled, timeout_s=6.0)
            matched = nav_page_matched(str(observed.get("node") or ""), item.detail)
            report.update(status="passed" if result.ok and matched else "failed",
                          reason="page_matched" if result.ok and matched else result.status if not result.ok else "page_not_verified",
                          result=asdict(result), navigation_after=observed, navigation_only=True)
            return report
        after = snapshot_after_execute(
            item, before, result, include_tools=item.kind == "repair",
            include_skills=item.gather_mode == "skill",
            take_snapshot=take_snapshot, read_ui=read_inspect_record,
        )
        status, reason = verdict(item, before, after, result)
        report.update(status=status, reason=reason, result=asdict(result), after=asdict(after))
        return report
    except Exception as exc:
        evidence.capture("exception")
        report["focus_fail"] = probe_focus(int(getattr(win, "hwnd", 0) or 0))
        report["stuck_stacks"] = bounded_stacks()
        report.update(status="failed", reason=str(exc))
        return report
    finally:
        evidence.note("end")
        _ACTIVE = None
        request_op_cancel()
        report["elapsed_s"] = round(time.monotonic() - evidence.started, 3)
        report["finished_at"] = time.time()
        report["evidence"] = str(folder)
        report["stages"] = evidence.stages
        report["progress_log"] = evidence.progress_log
        if report.get("status") == "failed" and "stuck_stacks" not in report:
            if report.get("reason") not in {"user_switched_away", "focus_failed"}:
                report["stuck_stacks"] = bounded_stacks()
            report.setdefault("focus_fail", probe_focus(int(getattr(win, "hwnd", 0) or 0)))
        save()
        on_hint(f"测试 {report['status']}：{report.get('reason', '')}")
