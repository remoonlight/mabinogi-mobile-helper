"""只读切片仅保留製作成对归档评估。"""

from __future__ import annotations

from typing import Any, Callable


def _craft_pair_probe_from_job(job: dict[str, Any]) -> dict[str, Any]:
    res = job.get("result") if isinstance(job.get("result"), dict) else {}
    probe = res.get("panel_probe") if isinstance(res.get("panel_probe"), dict) else None
    if probe is None:
        probe = res.get("combat_probe") if isinstance(res.get("combat_probe"), dict) else {}
    return dict(probe or {})


def assess_craft_pair_captures(
    closed_job: dict[str, Any],
    open_job: dict[str, Any],
) -> dict[str, Any]:
    pa = _craft_pair_probe_from_job(closed_job)
    pb = _craft_pair_probe_from_job(open_job)
    rb = open_job.get("result") if isinstance(open_job.get("result"), dict) else {}

    closed_hub = pa.get("craft_hub_category")
    open_hub = pb.get("craft_hub_category")
    try:
        hub_open_ok = open_hub is not None and int(open_hub) in (1, 2, 3)
    except (TypeError, ValueError):
        hub_open_ok = False

    def _signal(probe: dict[str, Any]) -> dict[str, Any]:
        existing = probe.get("craft_open_signal")
        if isinstance(existing, dict) and existing.get("source"):
            return dict(existing)
        return {"open": False, "trusted": False, "source": ""}

    sig_a = _signal(pa)
    sig_b = _signal(pb)
    visual_open_ok = bool(sig_b.get("open") and sig_b.get("trusted"))
    open_ok = bool(hub_open_ok or visual_open_ok)

    obj_a = (pa.get("obj_n") or {}) if isinstance(pa.get("obj_n"), dict) else {}
    obj_b = (pb.get("obj_n") or {}) if isinstance(pb.get("obj_n"), dict) else {}
    hub_n_closed = obj_a.get("CraftingHubUIState")
    hub_n_open = obj_b.get("CraftingHubUIState")
    try:
        hub_obj_grew = (
            hub_n_closed is not None
            and hub_n_open is not None
            and int(hub_n_open) > int(hub_n_closed)
        )
    except (TypeError, ValueError):
        hub_obj_grew = False

    fail_tags: list[str] = []
    menu_flip = pa.get("menu_open") is False and pb.get("menu_open") is True
    if not menu_flip:
        fail_tags.append("pair:menu_not_flipped")
    if not open_ok:
        fail_tags.append("pair:open_hub_not_1_2_3")
        if open_hub is None and hub_obj_grew:
            fail_tags.append("pair:hub_null_despite_obj_growth")
        if not pb.get("craft_open_signal") and not pb.get("craft_icon_hits"):
            fail_tags.append("pair:no_visual_fallback_in_archive")

    icons_b = rb.get("icons") if isinstance(rb.get("icons"), list) else []
    icon_errs = [str(e) for e in (rb.get("icon_errors") or [])]
    hud_icon_timeout = any("wall_timeout" in e for e in icon_errs) or not icons_b
    if hud_icon_timeout:
        fail_tags.append("pair:icon_wall_timeout_or_empty")

    pair_ok = bool(
        menu_flip
        and open_ok
        and (visual_open_ok or not hud_icon_timeout)
        and "pair:closed_visual_still_open" not in fail_tags
    )
    return {
        "pair_ok": pair_ok,
        "enough": False,
        "menu_flip": menu_flip,
        "closed_hub": closed_hub,
        "open_hub": open_hub,
        "hub_open_ok": hub_open_ok,
        "visual_open_ok": visual_open_ok,
        "open_ok": open_ok,
        "open_signal": sig_b,
        "closed_signal": sig_a,
        "fail_tags": fail_tags,
        "gaps": [],
        "note": "只读切片离线对拍；够用仍否",
    }


_stage_counts: dict[str, int] = {}
_stage_need_verify: dict[str, bool] = {}

_HUD_OCCLUDED_PEAK = 0.35


def allow_stage_action(stage: str) -> bool:
    """同 stage 未验证前最多 2 次动作。"""
    key = str(stage or "").strip() or "切页"
    if _stage_need_verify.get(key):
        return False
    n = int(_stage_counts.get(key, 0) or 0)
    if n >= 2:
        _stage_need_verify[key] = True
        return False
    _stage_counts[key] = n + 1
    if _stage_counts[key] >= 2:
        _stage_need_verify[key] = True
    return True


_BEFORE_DOWN_ERRORS = frozenset({'click_pointer_moved','click_focus_lost',
    'focus_click_timeout','click_ui_blocked','click_ui_not_ready'})


def stage_click(stage: str, click: Callable[[int,int],bool], x: int, y: int) -> bool | None:
    """Reserve an attempt; refund only a proven refusal before mouse down.

    None means the gate is closed. Unknown exceptions retain the reservation,
    since input may already have reached the game. This never retries a click.
    """
    if not allow_stage_action(stage):
        return None
    key = str(stage or '').strip() or '切页'
    def refund() -> None:
        _stage_counts[key] = max(0,_stage_counts.get(key,0)-1)
        _stage_need_verify[key] = _stage_counts[key] >= 2
    try:
        sent = bool(click(x,y))
    except RuntimeError as exc:
        if str(exc).split(':',1)[0] in _BEFORE_DOWN_ERRORS:
            refund()
        raise
    if not sent:
        refund()
    return sent


def apply_stage_verify(stage: str, ok: bool) -> None:
    """验画面后清零或标记需再验。"""
    key = str(stage or "").strip() or "切页"
    if ok:
        _stage_counts[key] = 0
        _stage_need_verify[key] = False
        return
    _stage_need_verify[key] = True


def reset_stage_gates() -> None:
    """脱困/开停泵后清零，避免同一 stage 一直挡点击。"""
    _stage_counts.clear()
    _stage_need_verify.clear()


def reset_stage_gates_for_tests() -> None:
    reset_stage_gates()


def diagnose_hud_menu_miss(*, peak_score: float | None = None) -> str:
    """只读诊断。peak 过低 → hud_occluded，否则 menu_icon_miss。"""
    if peak_score is None:
        return "menu_icon_miss"
    try:
        peak = float(peak_score)
    except (TypeError, ValueError):
        return "menu_icon_miss"
    if peak < _HUD_OCCLUDED_PEAK:
        return "hud_occluded"
    return "menu_icon_miss"
