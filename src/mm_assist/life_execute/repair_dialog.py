"""到达奈麗絲后的对话修理链（模板+OCR；禁名牌旁路）。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from mm_assist.config import ROOT, AppConfig
from mm_assist.nav.compat import in_test_mode, timed_action
from mm_assist.perception.list_ocr import normalize_text
from mm_assist.perception.ocr import OcrLine
from mm_assist.platform.match import MatchHit, match_template
from mm_assist.platform.window import WindowInfo, refresh_window
from mm_assist.safety.input import click_client
from mm_assist.tool_repair import (
    REPAIR_AFTER_ARRIVE,
    REPAIR_ENTRY_DEBT,
    REPAIR_ENTRY_ICON_MISS,
    op_cancelled,
    plan_repair_dialog_entry,
    text_blob,
)

log = logging.getLogger(__name__)

POST_CLICK_UI_WAIT_S = 1.0
FOCUS_BEFORE_CLICK_S = 1.0
DIALOG_ESCAPE_RESTART_MAX = 2
REPAIR_SKIP_REVEAL_MAX = 2
_AWAY_NX, _AWAY_NY = 0.42, 0.28

SKIP_DIALOG_NEEDLES: tuple[str, ...] = (
    "跳過對話",
    "跳过对话",
    "跳過对話",
    "跳過対話",
    "跳對话",
    "跳對話",
    "跳对话",
    "跳過對",
    "跳过對",
)
DIALOG_REPAIR_ENTRY: tuple[str, ...] = ("修理", "修復", "修复", "維修", "维修")
REPAIR_ENTRY_TEMPLATE_RELS: tuple[str, ...] = (
    "knowledge/ui/repair/02d_repair_entry_bar.png",
    "knowledge/ui/repair/02c_repair_anvil.png",
    "knowledge/ui/repair/02b_repair_btn_crop.png",
    "knowledge/ui/repair/02_repair_menu.png",
)
REPAIR_ENTRY_ROI = (0.00, 0.70, 0.55, 0.16)
REPAIR_ENTRY_TEMPLATE_THRESHOLD = 0.62
REPAIR_ENTRY_TEMPLATE_MIN_MARGIN = 0.0
REPAIR_ENTRY_TEMPLATE_SCALES = (1.0, 0.90, 1.10, 0.80, 1.20, 0.70)
REPAIR_ENTRY_OCR_MIN_NY = 0.72
REPAIR_ENTRY_OCR_MAX_NY = 0.88
REPAIR_ENTRY_SHOP_DX_NX = 0.0765
REPAIR_ENTRY_ICON_MIN_NY = 0.72
REPAIR_ENTRY_ICON_MAX_NY = 0.88
REPAIR_SKIP_TEMPLATE_RELS = ("knowledge/ui/repair/01_skip_dialog.png",)
REPAIR_SKIP_ROI = (0.70, 0.00, 0.30, 0.28)
REPAIR_SKIP_TEMPLATE_THRESHOLD = 0.70
REPAIR_END_TEMPLATE_RELS = ("knowledge/ui/repair/05_end_dialog.png",
                            "knowledge/ui/repair/05b_end_dialog_label.png")
REPAIR_END_ROI = (0.45, 0.68, 0.55, 0.32)
REPAIR_END_TEMPLATE_THRESHOLD = 0.70
REPAIR_EQUIPPED_TEMPLATE_RELS = (
    "knowledge/ui/repair/03b_equipped_left.png",
    "knowledge/ui/repair/03_repair_equipped.png",
)
REPAIR_EQUIPPED_ROI = (0.02, 0.70, 0.96, 0.30)
REPAIR_EQUIPPED_TEMPLATE_THRESHOLD = 0.60
EQUIPPED_REPAIR_NEEDLES: tuple[str, ...] = (
    "修理已配戴的裝備",
    "修理已配戴的装备",
    "修理已配戴的装備",
    "修理已配戴的装",
    "修理已佩戴的裝備",
    "修理已佩戴的装备",
    "修理已配戴",
    "修理已佩戴",
    "修理所有已穿戴裝備",
    "修理所有已穿戴装备",
    "修理所有已穿戴",
)
CONFIRM_REPAIR_NEEDLES: tuple[str, ...] = ("修理",)


def repair_dialog_templates_ready() -> bool:
    needed = (
        REPAIR_SKIP_TEMPLATE_RELS[0],
        REPAIR_ENTRY_TEMPLATE_RELS[0],
        REPAIR_EQUIPPED_TEMPLATE_RELS[0],
        *REPAIR_END_TEMPLATE_RELS,
    )
    return all((ROOT / rel).is_file() for rel in needed)


def _center(L: OcrLine) -> tuple[int, int] | None:
    if not L.box:
        return None
    xs = [p[0] for p in L.box]
    ys = [p[1] for p in L.box]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def _template_paths(rels: tuple[str, ...]) -> list[Path]:
    return [p for rel in rels if (p := ROOT / rel).is_file()]


def _frame_height(win: WindowInfo, img: np.ndarray | None = None) -> int:
    if img is not None and getattr(img, "shape", None) is not None:
        return max(1, int(img.shape[0]))
    return max(1, int(getattr(win, "height", 0) or 1))


def _peek_dialog_active() -> bool | None:
    try:
        from mm_assist.memory.readers.scene import Il2CppSceneReader

        rdr = Il2CppSceneReader()
        try:
            ms = rdr.read(allow_scan=False)
        finally:
            rdr.close()
        if ms is None:
            return None
        return bool(ms.dialog_active) if ms.dialog_active is not None else None
    except Exception:
        return None


def _snap_frame(cfg: AppConfig, win: WindowInfo) -> tuple[WindowInfo, np.ndarray, list[OcrLine]]:
    from mm_assist.nav.map_atlas import snap

    win2, img, lines, _ = snap(cfg, win, min_conf=0.28)
    return win2, img, lines


def _grab_frame(cfg: AppConfig, win: WindowInfo) -> tuple[WindowInfo, np.ndarray]:
    from mm_assist.platform.capture import grab_client

    win = refresh_window(win) or win
    _ = cfg
    img = grab_client(win)
    return win, img


def _escape_then_retry(_cfg, win, _out, reason: str = "") -> tuple[Any, bool]:
    log.info("dialog escape stub skip restart reason=%s", reason)
    return win, False


def _focus_then_click(win: WindowInfo, cx: int, cy: int, *, hold_s: float = 0.10) -> bool:
    from mm_assist.platform.window import focus_window

    focus_window(win.hwnd, force=True)
    if not in_test_mode():
        time.sleep(FOCUS_BEFORE_CLICK_S)
    win = refresh_window(win) or win
    return bool(click_client(win, cx, cy, method="sendinput", refocus=True, hold_s=hold_s))


def click_away(win: WindowInfo) -> None:
    cx = int(win.width * _AWAY_NX)
    cy = int(win.height * _AWAY_NY)
    click_client(win, cx, cy, method="sendinput", refocus=True, hold_s=0.06)


def npc_entry_menu_visible(lines: list[OcrLine] | None, img: np.ndarray | None) -> bool:
    """奈麗絲六钮入口还在：禁止点空白，否则永远进不了已配戴修理页。"""
    blob = normalize_text(text_blob(lines or []))
    if "特化" in blob and ("修理" in blob or "解除詛咒" in blob or "解除诅咒" in blob):
        return True
    if img is not None and find_repair_entry_icon(img) is not None:
        return True
    return False


def _click_first_needle(
    win: WindowInfo,
    lines: list[OcrLine],
    needles: tuple[str, ...],
    *,
    prefer_longest: bool = True,
    ban_substr: tuple[str, ...] = (),
    min_ny: float | None = None,
    max_ny: float | None = None,
    prefer_high_ny: bool = False,
    max_text_len: int | None = None,
    img: np.ndarray | None = None,
) -> str | None:
    h = _frame_height(win, img)
    cands: list[tuple[float, int, int, int, str]] = []
    for L in lines:
        t = normalize_text(L.text or "")
        if not t:
            continue
        if max_text_len is not None and len(t) > int(max_text_len):
            continue
        if ban_substr and any(normalize_text(b) in t for b in ban_substr if b):
            continue
        c = _center(L)
        if not c:
            continue
        ny = c[1] / h
        if min_ny is not None and ny < min_ny:
            continue
        if max_ny is not None and ny > max_ny:
            continue
        for n in needles:
            nn = normalize_text(n)
            if nn and nn in t:
                rank = -len(nn) if prefer_longest else len(t)
                y_key = -c[1] if prefer_high_ny else c[1]
                cands.append((float(rank), y_key, c[1], c[0], n))
                break
    if not cands:
        return None
    cands.sort()
    _rank, _yk, cy, cx, needle = cands[0]
    _focus_then_click(win, cx, cy, hold_s=0.10)
    log.info("repair click %r @%d,%d", needle, cx, cy)
    return needle


def repair_entry_template_paths() -> list[Path]:
    return _template_paths(REPAIR_ENTRY_TEMPLATE_RELS)


def _repair_entry_hit_on_bottom_bar(hit: MatchHit, img: np.ndarray) -> bool:
    h = max(1, int(img.shape[0]))
    ny = float(hit.cy) / h
    return REPAIR_ENTRY_ICON_MIN_NY <= ny <= REPAIR_ENTRY_ICON_MAX_NY


def find_repair_entry_icon(img: np.ndarray) -> MatchHit | None:
    best: MatchHit | None = None
    for path in repair_entry_template_paths():
        for sc in REPAIR_ENTRY_TEMPLATE_SCALES:
            hit = match_template(
                img,
                path,
                roi=REPAIR_ENTRY_ROI,
                threshold=0.0,
                min_margin=REPAIR_ENTRY_TEMPLATE_MIN_MARGIN,
                scale=float(sc),
                check_margin=False,
            )
            if hit is None or not _repair_entry_hit_on_bottom_bar(hit, img):
                continue
            if hit.score < REPAIR_ENTRY_TEMPLATE_THRESHOLD:
                continue
            if best is None or hit.score > best.score:
                best = hit
    return best


def click_repair_entry_icon(win: WindowInfo, img: np.ndarray) -> str | None:
    hit = find_repair_entry_icon(img)
    if hit is None:
        return None
    cx = int(hit.x + max(10, hit.w // 4))
    cy = int(hit.cy)
    click_client(win, cx, cy, method="sendinput", refocus=True, hold_s=0.08)
    return "修理图标"


def locate_repair_entry_click(
    win: WindowInfo, lines: list[OcrLine], img: np.ndarray | None
) -> tuple[int, int, str] | None:
    """点六钮「修理」行（约 ny=0.77）。商店/結束對話在 ny≈0.93，旧阈值会点到那一行。"""
    if img is not None:
        hit = find_repair_entry_icon(img)
        if hit is not None:
            return int(hit.x + max(10, hit.w // 4)), int(hit.cy), "修理图标"
    h = _frame_height(win, img)
    best: tuple[int, int] | None = None
    for L in lines or []:
        t = normalize_text(L.text or "")
        c = _center(L)
        if not t or not c or len(t) > 8:
            continue
        if any(k in t for k in ("所有", "全部", "已佩戴", "已穿戴", "已配戴", "商店", "結束", "结束")):
            continue
        ny = c[1] / h
        if ny < REPAIR_ENTRY_OCR_MIN_NY or ny > REPAIR_ENTRY_OCR_MAX_NY:
            continue
        if any(normalize_text(n) in t for n in DIALOG_REPAIR_ENTRY if n):
            best = (int(c[0]), int(c[1]))
            break
    if best is None:
        return None
    return best[0], best[1], "修理胶囊"


def locate_end_dialog_click(
    win: WindowInfo, lines: list[OcrLine], img: np.ndarray | None
) -> tuple[int, int] | None:
    """結束對話在商店行（ny≈0.93），禁止用会配到铁砧的结束模板。"""
    h = _frame_height(win, img)
    for L in lines or []:
        t = normalize_text(L.text or "")
        c = _center(L)
        if not t or not c:
            continue
        if "結束對話" not in t and "结束对话" not in t:
            continue
        if c[1] / h < 0.88:
            continue
        return int(c[0]), int(c[1])
    return None


def _hud_menu_xy(frame) -> tuple[int, int] | None:
    """告别近景仍是主界面栈，菜单图标未回到 ROI 则不能点。"""
    if frame is None:
        return None
    from mm_assist.perception.hud_entry_locator import locate_hud_entry
    from mm_assist.life_execute.ui_logic import recognized_click_xy

    try:
        cap = json.loads(
            (ROOT / "knowledge" / "ui" / "contracts" / "main_hud_entry_capture.json").read_text(
                encoding="utf-8"
            )
        )
        entry = next((e for e in cap.get("entries") or [] if e.get("id") == "menu"), None)
        if not isinstance(entry, dict):
            return None
        loc = locate_hud_entry(entry, frame, root=ROOT)
        if loc.get("feature_ok") is not True:
            return None
        return recognized_click_xy(loc)
    except Exception:
        return None


def npc_goodbye_visible(frame) -> bool:
    """告别原图模板主判；OCR 文案不能独立授权点击。"""
    return locate_npc_goodbye_click(frame) is not None


def npc_end_dialog_visible(frame) -> bool:
    """Independent complete end label and repair entry identify the NPC menu."""
    return find_end_dialog_btn(frame) is not None and find_repair_entry_icon(frame) is not None


def npc_dialog_visible(frame) -> bool:
    """The ready HUD shell can still contain the repair menu or goodbye."""
    return npc_end_dialog_visible(frame) or npc_goodbye_visible(frame)


def locate_npc_goodbye_click(frame) -> tuple[int, int] | None:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    if find_repair_entry_icon(frame) is not None:
        return None
    try:
        spec = json.loads((ROOT / 'knowledge/ui/repair/06_goodbye.json').read_text(encoding='utf-8'))
        hit = match_template(frame, ROOT / spec['template'], roi=tuple(spec['roi']),
                             threshold=spec['threshold'], min_margin=spec['min_margin'])
        if hit is None:
            return None
        nx, ny = spec['click_in_template']
        return round(hit.x + hit.w * nx), round(hit.y + hit.h * ny)
    except Exception:
        return None


def wait_hud_after_npc_dismiss(click, grab, sleep_fn=None, *, timeout_s=6.0,
                              verify_hud=None) -> bool:
    """识别告别气泡后只点一次；独立 HUD 返回才完成。"""
    sleeper = sleep_fn or time.sleep
    grabber = grab
    if grabber is None:
        return False
    clicked = False
    final_read = False
    from mm_assist.safety.autofarm_gates import apply_stage_verify, stage_click
    stage = 'npc_goodbye_dismiss'
    deadline = time.monotonic() + max(0.0, timeout_s)
    for attempt in range(25):
        if op_cancelled():
            return False
        if attempt == 24 or time.monotonic() >= deadline:
            if not clicked or final_read:
                return False
            # An in-budget dispatch can consume the remaining time. Observe
            # its result once, without authorizing another input or wait loop.
            final_read = True
        last = grabber()
        if op_cancelled():
            return False
        if last is None:
            if final_read:
                return False
            sleeper(0.25)
            continue
        if find_repair_entry_icon(last) is not None:
            if final_read:
                return False
            sleeper(0.25)
            continue
        if _hud_menu_xy(last) is not None:
            verified = verify_hud is None or verify_hud(last)
            if op_cancelled():
                return False
            if verified:
                apply_stage_verify(stage, True)
                apply_stage_verify('npc_end_dialog', True)
                return True
        if final_read:
            return False
        if not clicked:
            xy = locate_npc_goodbye_click(last)
            if xy is not None:
                if op_cancelled() or time.monotonic() >= deadline:
                    return False
                if not stage_click(stage, click, *xy):
                    return False
                clicked = True
        sleeper(0.25)
    return False


def dismiss_world_npc_menu(click, frame, *, grab=None, sleep_fn=None) -> bool:
    """栈认成主界面时，奈麗絲六钮仍盖住菜单。点結束對話，不点铁砧。"""
    if op_cancelled() or grab is None or not npc_end_dialog_visible(frame):
        return False
    hit = find_end_dialog_btn(frame)
    if hit is None or op_cancelled():
        return False
    from mm_assist.safety.autofarm_gates import stage_click

    if not stage_click('npc_end_dialog', click, hit.cx, hit.cy):
        return False
    return wait_hud_after_npc_dismiss(click, grab, sleep_fn=sleep_fn)


def click_repair_entry_via_neighbors(
    win: WindowInfo, lines: list[OcrLine], *, img: np.ndarray | None = None
) -> str | None:
    h = _frame_height(win, img)
    w = max(1, int(getattr(win, "width", 0) or (img.shape[1] if img is not None else 1)))
    anchors: list[tuple[float, float, str]] = []
    for L in lines or []:
        t = normalize_text(L.text or "")
        c = _center(L)
        if not t or not c:
            continue
        ny = c[1] / h
        if ny < REPAIR_ENTRY_OCR_MIN_NY or ny > REPAIR_ENTRY_OCR_MAX_NY:
            continue
        if any(k in t for k in ("商店", "特化", "結束對話", "结束对话", "隨便聊聊")):
            anchors.append((float(c[0]), float(c[1]), t))
    if not anchors:
        return None
    shop = next((a for a in anchors if "商店" in a[2]), None)
    if shop is not None:
        cx, cy = int(shop[0] - REPAIR_ENTRY_SHOP_DX_NX * w), int(shop[1])
    else:
        left = min(anchors, key=lambda a: a[0])
        cx, cy = int(left[0] - REPAIR_ENTRY_SHOP_DX_NX * w), int(left[1])
    click_client(win, cx, cy, method="sendinput", refocus=True, hold_s=0.08)
    return "修理邻钮"


def find_skip_dialog_icon(img: np.ndarray) -> MatchHit | None:
    best: MatchHit | None = None
    for path in _template_paths(REPAIR_SKIP_TEMPLATE_RELS):
        hit = match_template(
            img, path, roi=REPAIR_SKIP_ROI, threshold=REPAIR_SKIP_TEMPLATE_THRESHOLD, min_margin=0.0, check_margin=False
        )
        if hit is None:
            continue
        if best is None or hit.score > best.score:
            best = hit
    return best


def click_skip_dialog(
    win: WindowInfo, lines: list[OcrLine], img: np.ndarray | None, *, max_ny: float = 0.30
) -> str | None:
    hit = _click_first_needle(win, lines, SKIP_DIALOG_NEEDLES, prefer_longest=True, max_ny=max_ny, img=img)
    if hit:
        return hit
    if img is None:
        return None
    icon = find_skip_dialog_icon(img)
    if icon is None:
        return None
    click_client(win, icon.cx, icon.cy, method="sendinput", refocus=True, hold_s=0.08)
    return "跳過對話图标"


def _has_skip_dialog(img: np.ndarray | None, lines: list[OcrLine], win: WindowInfo) -> bool:
    if img is not None and find_skip_dialog_icon(img) is not None:
        return True
    h = _frame_height(win, img)
    for L in lines or []:
        t = normalize_text(L.text or "")
        if not t or not any(normalize_text(n) in t for n in SKIP_DIALOG_NEEDLES if n):
            continue
        c = _center(L)
        if c is None or c[1] / h <= 0.45:
            return True
    return False


def _has_short_repair_ocr(lines: list[OcrLine], win: WindowInfo, *, img: np.ndarray | None = None) -> bool:
    h = _frame_height(win, img)
    ban = ("所有", "全部", "已佩戴", "已穿戴", "已配戴", "跳過", "跳过", "結束", "结束")
    for L in lines or []:
        t = normalize_text(L.text or "")
        if not t or len(t) > 8 or any(normalize_text(b) in t for b in ban if b):
            continue
        c = _center(L)
        if c is None:
            continue
        ny = c[1] / h
        if ny < REPAIR_ENTRY_OCR_MIN_NY or ny > REPAIR_ENTRY_OCR_MAX_NY:
            continue
        if any(normalize_text(n) in t for n in DIALOG_REPAIR_ENTRY if n):
            return True
    return False


def find_end_dialog_btn(img: np.ndarray) -> MatchHit | None:
    if not isinstance(img, np.ndarray) or img.size == 0:
        return None
    try:
        spec = json.loads((ROOT/'knowledge/ui/repair/05b_end_dialog_label.json').read_text(encoding='utf-8'))
        candidates = [spec] + [variant for variant in spec.get('template_variants', [])
            if variant.get('source_size') == [img.shape[1], img.shape[0]]]
        for candidate in candidates:
            hit = match_template(img, ROOT/candidate['template'], roi=tuple(spec['roi']),
                                 threshold=spec['threshold'], min_margin=spec['min_margin'])
            if hit is not None:
                return hit
        return None
    except (OSError, ValueError, KeyError):
        return None


def click_end_dialog(win: WindowInfo, img: np.ndarray | None) -> str | None:
    if img is None:
        return None
    btn = find_end_dialog_btn(img)
    if btn is None:
        return None
    from mm_assist.safety.autofarm_gates import stage_click
    if not stage_click('npc_end_dialog',
                       lambda x,y: click_client(win,x,y,method='sendinput',refocus=True,hold_s=.08),
                       btn.cx, btn.cy):
        return None
    return "結束對話按钮"


def _wait_and_end_dialog(cfg: AppConfig, win: WindowInfo, *, timeout_s=6.0):
    """Result/menu animation is read-only; recognize and dispatch end once."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    for _ in range(24):
        if op_cancelled() or time.monotonic() >= deadline:
            return win, None
        win, img = _grab_frame(cfg, win)
        if op_cancelled() or time.monotonic() >= deadline:
            return win, None
        if find_end_dialog_btn(img) is not None:
            return win, click_end_dialog(win, img)
        time.sleep(.25)
    return win, None


def find_repair_equipped_icon(img: np.ndarray) -> MatchHit | None:
    best: MatchHit | None = None
    for path in _template_paths(REPAIR_EQUIPPED_TEMPLATE_RELS):
        hit = match_template(
            img,
            path,
            roi=REPAIR_EQUIPPED_ROI,
            threshold=REPAIR_EQUIPPED_TEMPLATE_THRESHOLD,
            min_margin=0.0,
            check_margin=False,
        )
        if hit is None:
            continue
        if best is None or hit.score > best.score:
            best = hit
    return best


def click_repair_equipped_icon(win: WindowInfo, img: np.ndarray) -> str | None:
    hit = find_repair_equipped_icon(img)
    if hit is None:
        return None
    cx = int(hit.x + max(12, hit.w // 4))
    cy = int(hit.cy)
    _focus_then_click(win, cx, cy, hold_s=0.10)
    return "修理已配戴图标"


def click_full_repair_button(
    win: WindowInfo, lines: list[OcrLine], *, img: np.ndarray | None = None
) -> str | None:
    with timed_action("click_full_repair") as ctx:
        hit = _click_equipped_ocr(win, lines, img=img)
        if hit:
            ctx["ok"] = True
            return hit
        if img is not None:
            thit = click_repair_equipped_icon(win, img)
            if thit:
                ctx["ok"] = True
                return thit
        ctx["ok"] = False
        return None


def _click_equipped_ocr(
    win: WindowInfo, lines: list[OcrLine], *, img: np.ndarray | None = None
) -> str | None:
    """底栏绿钮：点 OCR 文字中心，仅上移 24px 落到绿色本体。"""
    cands: list[tuple[int, int, int, str]] = []
    for L in lines:
        t = normalize_text(L.text or "")
        if not t or "全部" in t:
            continue
        c = _center(L)
        if not c:
            continue
        for n in EQUIPPED_REPAIR_NEEDLES:
            nn = normalize_text(n)
            if nn and nn in t:
                cands.append((c[1], c[0], -len(nn), n))
                break
    if not cands:
        return None
    cy, cx, _neg, needle = min(cands, key=lambda x: (x[2], -x[0]))
    cy = max(0, int(cy) - 24)
    _focus_then_click(win, cx, cy, hold_s=0.10)
    log.info("repair click equipped %r @%d,%d", needle, cx, cy)
    return needle


def _wait_leave_dialog(cfg: AppConfig, win: WindowInfo, *, timeout_s: float = 6.0, away_clicks: int = 3) -> tuple[WindowInfo, bool]:
    from mm_assist.life_execute.ui_logic import resolve_navigation

    def grab():
        nonlocal win
        win, img = _grab_frame(cfg, win)
        return img

    def click(x, y):
        return bool(click_client(win, x, y, method='sendinput', refocus=True, hold_s=.06))

    # The standalone legacy scene flag can outlive the dialog. Verify the same
    # HUD against the current paired UI stack before clearing either gate.
    def verified_hud(frame):
        from types import SimpleNamespace
        from mm_assist.memory.readers.ui_stack import shared_ui_stack_reader, stack_hud_only

        reader = shared_ui_stack_reader()
        invalid = False
        def read(**kwargs):
            nonlocal invalid
            snap = reader.read(**kwargs)
            hud = (stack_hud_only(snap.names)
                   or (not snap.names and snap.empty_confirmed))
            sample_age = time.time() - float(snap.sampled_ts or 0)
            invalid |= (not snap.ready or bool(snap.error) or not hud
                        or not -.5 <= sample_age <= 3)
            return snap
        where = resolve_navigation(frame, reader=SimpleNamespace(read=read))
        if op_cancelled():
            return False
        age = time.time() - float(where.stack_sampled_ts or 0)
        return (not invalid and where.node == 'main_hud'
                and where.stack_ready and -.5 <= age <= 3)

    _ = away_clicks  # Compatibility only: no blind clicks or repeated end action.
    left = wait_hud_after_npc_dismiss(click, grab, timeout_s=timeout_s,
                                    verify_hud=verified_hud)
    return win, left


def run_repair_dialog_chain(cfg: AppConfig, win: WindowInfo, **_k: Any) -> tuple[WindowInfo, dict]:
    if not repair_dialog_templates_ready():
        return win, {
            "ok": False,
            "stage": "dialog_templates_missing",
            "error": "修理对话模板未入库（knowledge/ui/repair/*.png）",
        }
    cur_lines, cur_img = _k.get("lines"), _k.get("img")
    restarts = 0
    while True:
        win, out = _repair_dialog_chain_once(cfg, win, lines=cur_lines, img=cur_img,
                                            on_action_issued=_k.get('on_action_issued'))
        cur_lines, cur_img = None, None
        if out.get("ok") or out.get("stage") == "cancel":
            return win, out
        if out.get("restart_from_head") and restarts < DIALOG_ESCAPE_RESTART_MAX:
            restarts += 1
            continue
        return win, out


def _repair_dialog_chain_once(
    cfg: AppConfig,
    win: WindowInfo,
    *,
    lines: list[OcrLine] | None = None,
    img: np.ndarray | None = None,
    on_action_issued: Callable[[], None] | None = None,
) -> tuple[WindowInfo, dict]:
    out: dict = {"ok": False, "stage": "dialog"}
    if op_cancelled():
        out["error"] = "已停止操作"
        out["stage"] = "cancel"
        return win, out
    if lines is None or img is None:
        win, img, lines = _snap_frame(cfg, win)
    try:
        from mm_assist.platform.window import focus_window

        focus_window(win.hwnd, force=True)
        time.sleep(0.12)
        win = refresh_window(win) or win
    except Exception:
        pass

    skip_attempted = False
    dlg_open = _peek_dialog_active()

    def _entry_plan() -> dict:
        equipped = img is not None and find_repair_equipped_icon(img) is not None
        if not equipped:
            equipped = any(
                normalize_text(n) in normalize_text(text_blob(lines)) for n in EQUIPPED_REPAIR_NEEDLES
            )
        has_icon = img is not None and find_repair_entry_icon(img) is not None
        has_ocr = _has_short_repair_ocr(lines, win, img=img)
        has_neighbors = False
        if not has_icon and not has_ocr:
            h = _frame_height(win, img)
            for L in lines or []:
                t = normalize_text(L.text or "")
                c = _center(L)
                if not c:
                    continue
                ny = c[1] / h
                if ny < REPAIR_ENTRY_OCR_MIN_NY or ny > REPAIR_ENTRY_OCR_MAX_NY:
                    continue
                if any(k in t for k in ("商店", "特化", "結束對話", "结束对话")):
                    has_neighbors = True
                    break
        return plan_repair_dialog_entry(
            arrived=True,
            dialog_open=dlg_open,
            has_skip=_has_skip_dialog(img, lines, win),
            has_repair_icon=has_icon or has_neighbors,
            has_repair_ocr=has_ocr,
            has_equipped_icon=equipped,
            skip_attempted=skip_attempted,
        )

    plan = _entry_plan()
    if plan.get("op") == "skip_then_find_entry":
        for _si in range(REPAIR_SKIP_REVEAL_MAX):
            if op_cancelled():
                out["error"] = "已停止操作"
                out["stage"] = "cancel"
                return win, out
            skip = click_skip_dialog(win, lines, img, max_ny=0.45)
            skip_attempted = True
            out["skipped"] = skip
            time.sleep(POST_CLICK_UI_WAIT_S)
            win, img, lines = _snap_frame(cfg, win)
            dlg_open = _peek_dialog_active()
            plan = _entry_plan()
            if plan.get("op") != "skip_then_find_entry" or not skip:
                break

    if op_cancelled():
        out["error"] = "已停止操作"
        out["stage"] = "cancel"
        return win, out

    if plan.get("op") != "click_equipped":
        loc = locate_repair_entry_click(win, lines, img)
        entry = None
        if loc is not None:
            click_client(win, loc[0], loc[1], method="sendinput", refocus=True, hold_s=0.08)
            entry = loc[2]
        if not entry:
            entry = _click_first_needle(
                win,
                lines,
                DIALOG_REPAIR_ENTRY,
                prefer_longest=False,
                ban_substr=("所有", "全部", "已佩戴", "已穿戴", "跳過", "跳过", "結束", "结束"),
                min_ny=REPAIR_ENTRY_OCR_MIN_NY,
                max_ny=REPAIR_ENTRY_OCR_MAX_NY,
                max_text_len=8,
                img=img,
            )
        if not entry:
            entry = click_repair_entry_via_neighbors(win, lines, img=img)
        if not entry and img is not None:
            entry = click_repair_entry_icon(win, img)
        out["entry"] = entry
        if not entry:
            miss = plan_repair_dialog_entry(arrived=True, dialog_open=dlg_open, skip_attempted=True)
            out["error"] = str(miss.get("error") or "对话入口「修理」图标未找到")
            out["stage"] = "entry"
            out["fail_tag"] = str(miss.get("fail_tag") or REPAIR_ENTRY_ICON_MISS)
            out["debt"] = str(miss.get("debt") or REPAIR_ENTRY_DEBT)
            win, do_restart = _escape_then_retry(cfg, win, out, reason="entry_not_found")
            if do_restart:
                out["restart_from_head"] = True
            return win, out
        time.sleep(0.45)
        win, img, lines = _snap_frame(cfg, win)
        has_eq = img is not None and find_repair_equipped_icon(img) is not None
        if not has_eq:
            has_eq = any(
                normalize_text(n) in normalize_text(text_blob(lines)) for n in EQUIPPED_REPAIR_NEEDLES
            )
        if not has_eq:
            skip2 = click_skip_dialog(win, lines, img, max_ny=0.30)
            if skip2:
                out["skipped"] = out.get("skipped") or skip2
                time.sleep(0.50)
                win, img, lines = _snap_frame(cfg, win)
                has_eq = img is not None and find_repair_equipped_icon(img) is not None
                if not has_eq:
                    has_eq = any(
                        normalize_text(n) in normalize_text(text_blob(lines)) for n in EQUIPPED_REPAIR_NEEDLES
                    )
        if not has_eq:
            if npc_entry_menu_visible(lines, img):
                time.sleep(0.70)
                win, img, lines = _snap_frame(cfg, win)
            else:
                click_away(win)
                out["blank_after_entry"] = True
                time.sleep(0.45)
                win, img, lines = _snap_frame(cfg, win)

    hit = click_full_repair_button(win, lines, img=img)
    if not hit:
        win, img, lines = _snap_frame(cfg, win)
        hit = click_full_repair_button(win, lines, img=img)
    out["equipped"] = hit
    if not hit:
        out["stage"] = "equipped"
        out["error"] = "修理已配戴按钮未找到"
        out["blob_sample"] = text_blob(lines)[:160]
        return win, out

    time.sleep(POST_CLICK_UI_WAIT_S)
    win, img, lines = _snap_frame(cfg, win)
    confirm = _click_first_needle(
        win,
        lines,
        CONFIRM_REPAIR_NEEDLES,
        prefer_longest=False,
        prefer_high_ny=True,
        min_ny=0.70,
        ban_substr=("已佩戴", "已配戴", "已穿戴", "所有", "全部", "跳過", "跳过", "結束", "结束"),
        img=img,
    )
    if not confirm and not in_test_mode():
        confirm = _click_first_needle(
            win,
            lines,
            CONFIRM_REPAIR_NEEDLES,
            prefer_longest=False,
            prefer_high_ny=True,
            min_ny=0.55,
            ban_substr=("已佩戴", "已配戴", "已穿戴", "所有", "全部"),
            img=img,
        )
    out["confirm"] = confirm
    if not confirm:
        out["stage"] = "confirm"
        out["error"] = "确认修理绿钮未找到"
        out["blob_sample"] = text_blob(lines)[:160]
        return win, out

    if on_action_issued is not None:
        on_action_issued()
    time.sleep(POST_CLICK_UI_WAIT_S)
    win, ended = _wait_and_end_dialog(cfg, win)
    out["ended"] = ended
    if not ended:
        out["stage"] = "end"
        out["error"] = "结束对话完整文字未确认或输入未发出"
        return win, out

    win, left = _wait_leave_dialog(cfg, win, timeout_s=6.0, away_clicks=3)
    out["dlg_left"] = left
    if not left:
        out["ok"] = False
        out["stage"] = "leave_dialog"
        out["error"] = "结束对话后 dlg 未归 0"
        return win, out
    out["ok"] = True
    out["stage"] = "done"
    _ = REPAIR_AFTER_ARRIVE
    return win, out
