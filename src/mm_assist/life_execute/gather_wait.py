"""采集结束后只读观察：净增只表示有产出，不能单独完成当前项。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable

from mm_assist.life_plan.types import ExecuteResult
from mm_assist.life_plan.execution_records import trace_wait
from mm_assist.perception.state_ui import DEFAULT_MAX_AGE_S

WAIT_GATHER_S = 90.0
HOLD_IDLE_S = 2.0
POLL_S = 0.05
HUD_GRAB_PERIOD_S = 0.25


@dataclass(frozen=True)
class GatherObservation:
    stp: int | None
    page: str
    fresh: bool


@dataclass(frozen=True)
class GatherPage:
    node: str
    fresh: bool


def _page_of(read_page: Callable[[], Any] | None) -> GatherPage:
    if read_page is None:
        return GatherPage(node="unknown", fresh=False)
    raw = read_page()
    if raw is None:
        return GatherPage(node="unknown", fresh=False)
    if isinstance(raw, GatherPage):
        return raw
    node = str(getattr(raw, "node", "") or (raw.get("node") if isinstance(raw, dict) else "") or "unknown")
    fresh = bool(getattr(raw, "fresh", None) if not isinstance(raw, dict) else raw.get("fresh"))
    return GatherPage(node=node or "unknown", fresh=fresh)


def observation_from_parts(
    *,
    read_stp: Callable[[], int | None] | None = None,
    read_page: Callable[[], Any] | None = None,
) -> GatherObservation:
    stp = read_stp() if read_stp is not None else None
    page = _page_of(read_page)
    return GatherObservation(stp=stp, page=page.node, fresh=page.fresh)


def _payload(view: Any) -> dict[str, Any]:
    raw = getattr(view, "raw", None)
    return raw if isinstance(raw, dict) else {}


def _inspect_confirms_main(view: Any, *, root: Any, now: float | None) -> bool:
    from mm_assist.life_execute.ui_logic import inspect_page_path

    path = inspect_page_path(root)
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not isinstance(rec, dict):
        return False
    try:
        ts = float(rec.get("ts_unix") or 0.0)
    except (TypeError, ValueError):
        return False
    if ts <= 0.0:
        return False
    if now is None:
        import time

        age = max(0.0, time.time() - ts)
    else:
        age = max(0.0, float(now) - ts)
    if age > float(DEFAULT_MAX_AGE_S):
        return False
    try:
        rec_pid = int(rec.get("pid"))
        view_pid = int(getattr(view, "pid", 0) or 0)
    except (TypeError, ValueError):
        return False
    if rec_pid <= 0 or view_pid <= 0 or rec_pid != view_pid:
        return False
    payload = _payload(view)
    rec_names = [str(x) for x in (rec.get("stack_names") or [])]
    view_names = [str(x) for x in (payload.get("ui_stack_names") or [])]
    if rec_names != view_names:
        return False
    if "stp" in rec and rec.get("stp") is not None:
        try:
            if int(rec.get("stp")) != int(getattr(view, "stp")):
                return False
        except (TypeError, ValueError):
            return False
    node = str(rec.get("node") or "").strip()
    zh = str(rec.get("zh") or "").strip()
    return node == "main_hud" or "主界面-待机" in zh or "主界面-动作中" in zh


def _unknown_is_confirmed_main(view: Any, *, root: Any, now: float | None) -> bool:
    from mm_assist.memory.readers.ui_stack import stack_left_overlay, stack_node

    payload = _payload(view)
    names = [str(x) for x in (payload.get("ui_stack_names") or [])]
    ready = bool(payload.get("ui_stack_ready"))
    if not ready:
        return False
    if getattr(view, "menu_open", None) is True:
        return False
    if getattr(view, "dialog_active", None) is True:
        return False
    if getattr(view, "map_open", None) is True:
        return False
    try:
        hub = int(getattr(view, "craft_hub_category")) if getattr(view, "craft_hub_category", None) is not None else None
    except (TypeError, ValueError):
        hub = None
    if hub in (1, 2, 3):
        return False
    if stack_left_overlay(names):
        return False
    empty_ok = bool(payload.get("ui_stack_empty_ok"))
    line = str(payload.get("ui_stack") or "").replace(" ", "").replace("\u3000", "")
    if line == "主界面":
        empty_ok = True
    if stack_node(names, ready=True, empty_ok=empty_ok) == "main_hud":
        return True
    return _inspect_confirms_main(view, root=root, now=now)


def _page_from_view(view: Any, *, root: Any, now: float | None) -> str:
    fam = str(getattr(view, "family", "") or "")
    if fam == "main_hud":
        return "main_hud"
    if fam not in {"", "unknown"}:
        return fam
    if _unknown_is_confirmed_main(view, root=root, now=now):
        return "main_hud"
    return "unknown"


def _finite_ts(value: Any) -> float | None:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ts):
        return None
    return ts


def _unpack_grab(raw: Any, now: float) -> tuple[Any, float | None, bool]:
    """(frame, captured_at, time_ok). 显式非法时间 time_ok=False。"""
    if raw is None:
        return None, None, True
    if isinstance(raw, dict):
        img = raw.get("frame", raw.get("image"))
        if "captured_at" in raw:
            ts = _finite_ts(raw.get("captured_at"))
            return img, ts, ts is not None
        if "ts" in raw and raw.get("ts") is not None:
            ts = _finite_ts(raw.get("ts"))
            return img, ts, ts is not None
        return img, now, True
    if isinstance(raw, (tuple, list)) and len(raw) >= 1:
        img = raw[0]
        if len(raw) < 2:
            return img, now, True
        ts = _finite_ts(raw[1])
        return img, ts, ts is not None
    return raw, now, True


def _hud_image_ok(frame: Any, *, root: Any) -> bool:
    if frame is None:
        return False
    try:
        if getattr(frame, "size", 1) == 0:
            return False
    except Exception:
        return False
    try:
        from mm_assist.life_execute.ui_logic import _main_hud_confirmed

        return bool(_main_hud_confirmed(frame, root=root))
    except Exception:
        return False


def _confirm_idle_hud(
    *,
    grab: Callable[[], Any] | None,
    now: float,
    root: Any,
    hud_gate: dict[str, Any] | None,
    force: bool,
) -> bool:
    if grab is None:
        return False
    if (
        not force
        and hud_gate is not None
        and hud_gate.get("checked")
        and now - float(hud_gate.get("at") or 0.0) < HUD_GRAB_PERIOD_S
    ):
        return bool(hud_gate.get("ok"))
    try:
        raw = grab()
    except Exception:
        if hud_gate is not None:
            hud_gate["checked"] = True
            hud_gate["at"] = now
            hud_gate["ok"] = False
        return False
    frame, captured_at, time_ok = _unpack_grab(raw, now)
    if not time_ok or captured_at is None:
        ok = False
    elif float(captured_at) > float(now):
        ok = False
    elif now - float(captured_at) > float(DEFAULT_MAX_AGE_S):
        ok = False
    else:
        ok = _hud_image_ok(frame, root=root)
    if hud_gate is not None:
        hud_gate["checked"] = True
        hud_gate["at"] = now
        hud_gate["ok"] = ok
    return ok


def read_gather_observation(
    *,
    root: Any = None,
    now: float | None = None,
    grab: Callable[[], Any] | None = None,
    hud_gate: dict[str, Any] | None = None,
    force_hud: bool = False,
) -> GatherObservation:
    from mm_assist.perception.state_ui import read_state_ui

    view = read_state_ui(root=root, now=now)
    if view is None or not bool(getattr(view, "fresh", False)) or not bool(getattr(view, "pid_alive", False)):
        return GatherObservation(stp=None, page="unknown", fresh=False)
    raw = getattr(view, "raw", None)
    payload = raw if isinstance(raw, dict) else {}
    if "stp" not in payload or payload.get("stp") is None:
        stp: int | None = getattr(view, "stp", None)
        if stp is not None:
            try:
                stp = int(stp)
            except (TypeError, ValueError):
                stp = None
    else:
        try:
            stp = int(payload.get("stp"))
        except (TypeError, ValueError):
            stp = None
    page = _page_from_view(view, root=root, now=now)
    if page == "main_hud" and stp == 0:
        import time

        hud_now = float(now if now is not None else time.time())
        if not _confirm_idle_hud(
            grab=grab,
            now=hud_now,
            root=root,
            hud_gate=hud_gate,
            force=bool(force_hud),
        ):
            page = "unknown"
    return GatherObservation(stp=stp, page=page, fresh=True)


def is_gather_idle(obs: GatherObservation) -> bool:
    return bool(obs.fresh) and obs.page == "main_hud" and obs.stp == 0


@trace_wait('等待采集待机')
def wait_gather_idle(
    *,
    observe: Callable[[], GatherObservation] | None = None,
    read_stp: Callable[[], int | None] | None = None,
    read_page: Callable[[], Any] | None = None,
    now_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    wait_s: float | None = None,
    hold_s: float | None = None,
    read_items: Callable[[], Any] | None = None,
    message_ok: str = "主界面待机",
    already_started: bool = False,
    issued: bool = False,
    root: Any = None,
    grab: Callable[[], Any] | None = None,
) -> ExecuteResult:
    import time

    now = now_fn or time.time
    sleep = sleep_fn or time.sleep
    window = float(WAIT_GATHER_S if wait_s is None else wait_s)
    hold = float(HOLD_IDLE_S if hold_s is None else hold_s)

    hud_gate: dict[str, Any] = {}

    def _obs() -> GatherObservation:
        if observe is not None:
            return observe()
        if read_stp is not None or read_page is not None:
            return observation_from_parts(read_stp=read_stp, read_page=read_page)
        return read_gather_observation(root=root, now=now(), grab=grab, hud_gate=hud_gate)

    deadline = now() + window
    saw_stp1 = bool(already_started)
    idle_since = 0.0
    while now() < deadline:
        if cancelled is not None and cancelled():
            return ExecuteResult(
                ok=False, status="cancel", message="已停止操作",
                action_issued=issued, action_started=saw_stp1,
            )
        if read_items is not None:
            try:
                read_items()
            except Exception:
                pass
        obs = _obs()
        if obs.stp == 1:
            saw_stp1 = True
            idle_since = 0.0
        elif saw_stp1 and is_gather_idle(obs):
            t = now()
            if idle_since <= 0.0:
                idle_since = t
            if t - idle_since >= hold:
                return ExecuteResult(
                    ok=True, status="done", message=message_ok,
                    action_issued=issued, action_started=saw_stp1,
                )
        else:
            idle_since = 0.0
        sleep(POLL_S)
    if saw_stp1:
        return ExecuteResult(
            ok=False,
            status="gather_pending",
            message="采集仍在进行或界面未知，等待结束，不换项",
            action_issued=issued,
            action_started=saw_stp1,
        )
    if issued:
        return ExecuteResult(
            ok=False,
            status="gather_watch",
            message="采集已发出但未见停钮，等待结束，不换项",
            action_issued=True,
            action_started=saw_stp1,
        )
    return ExecuteResult(
        ok=False,
        status="no_stp",
        message="寻找附近后未见停钮(stp=1)，不假装完成",
        action_issued=False,
        action_started=saw_stp1,
    )
