"""离线/实机 UI 感知效率基准：同输入比较候选，写 JSONL 账本。"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mm_assist.platform.capture import Grabber, _looks_black
from mm_assist.config import ROOT
from mm_assist.platform.match import match_template
from mm_assist.perception.ocr import ocr_image
from mm_assist.benchmarks.fixtures import (
    SIMP_INTRUDERS,
    TRAD_MARKERS,
    load_manifest,
    load_sample_bgr,
)
from mm_assist.platform.window import WindowInfo

log = logging.getLogger(__name__)

LEDGER_DIR = ROOT / "logs" / "benchmarks"

PreprocessFn = Callable[[np.ndarray], np.ndarray]


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ys) - 1)
    if lo == hi:
        return float(ys[lo])
    return float(ys[lo] + (ys[hi] - ys[lo]) * (k - lo))


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def char_error_rate(got: str, expected: str) -> float:
    if not expected:
        return 0.0 if not got else 1.0
    return levenshtein(got, expected) / max(len(expected), 1)


def trad_fidelity_ok(blob: str, needles: list[str]) -> bool:
    for n in needles:
        for m, s in zip(TRAD_MARKERS, SIMP_INTRUDERS, strict=True):
            if m in n and s in blob and m not in blob:
                return False
    return True


def preprocess_none(img: np.ndarray) -> np.ndarray:
    return img


def preprocess_gray3(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def preprocess_clahe(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(g), cv2.COLOR_GRAY2BGR)


def preprocess_up2(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def preprocess_thresh(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)


PREPROCESSORS: dict[str, PreprocessFn] = {
    "none": preprocess_none,
    "gray3": preprocess_gray3,
    "clahe": preprocess_clahe,
    "up2": preprocess_up2,
    "thresh": preprocess_thresh,
}


def ocr_backend_rapid_v5(img: np.ndarray) -> str:
    import os

    os.environ["MM_OCR_BACKEND"] = "v5"
    return "".join(l.text for l in ocr_image(img, min_confidence=0.3))


def ocr_backend_rapid_cht(img: np.ndarray) -> str:
    import os

    os.environ["MM_OCR_BACKEND"] = "cht"
    return "".join(l.text for l in ocr_image(img, min_confidence=0.3))


def ocr_backend_legacy(img: np.ndarray) -> str:
    import os

    os.environ["MM_OCR_BACKEND"] = "legacy"
    return "".join(l.text for l in ocr_image(img, min_confidence=0.3))


def ocr_backend_easyocr_ch_tra(img: np.ndarray) -> str:
    import easyocr  # type: ignore

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reader = _easy_reader()
    out = reader.readtext(rgb, detail=0)
    return "".join(str(x) for x in out)


def ocr_backend_tesseract_chi_tra(img: np.ndarray) -> str:
    import pytesseract  # type: ignore

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return str(pytesseract.image_to_string(rgb, lang="chi_tra") or "")


def ocr_backend_winrt(img: np.ndarray) -> str:
    del img
    raise RuntimeError("WinOCR 未接线")


_easy = None


def _easy_reader():
    global _easy
    if _easy is None:
        import easyocr  # type: ignore

        _easy = easyocr.Reader(["ch_tra"], gpu=False)
    return _easy


OCR_BACKENDS: dict[str, Callable[[np.ndarray], str]] = {
    "rapid_v5": ocr_backend_rapid_v5,
    "rapid_cht": ocr_backend_rapid_cht,
    "rapid_legacy": ocr_backend_legacy,
    "easyocr_ch_tra": ocr_backend_easyocr_ch_tra,
    "tesseract_chi_tra": ocr_backend_tesseract_chi_tra,
    "winrt": ocr_backend_winrt,
}


def capture_mss_rect(left: int, top: int, w: int, h: int) -> np.ndarray:
    g = Grabber(method="mss")
    try:
        return g.grab_rect(left, top, w, h)
    finally:
        g.close()


def capture_printwindow(win: WindowInfo) -> np.ndarray:
    g = Grabber(method="printwindow")
    try:
        return g.grab_client(win)
    finally:
        g.close()


def capture_gdi_bitblt(win: WindowInfo) -> np.ndarray:
    g = Grabber(method="bitblt")
    try:
        return g.grab_client(win)
    finally:
        g.close()


def capture_dxcam(region: tuple[int, int, int, int] | None, backend: str) -> np.ndarray:
    import dxcam  # type: ignore

    cam = dxcam.create(backend=backend, output_color="BGR")
    try:
        frame = cam.grab(region=region, new_frame_only=False)
        if frame is None:
            raise RuntimeError("dxcam returned None")
        return np.ascontiguousarray(frame)
    finally:
        try:
            cam.release()
        except Exception:
            pass


CAPTURE_LIVE = (
    "mss_client",
    "printwindow",
    "gdi_bitblt",
    "dxcam_dxgi",
    "dxcam_winrt",
)


@dataclass
class VariantResult:
    name: str
    layer: str
    times_ms: list[float] = field(default_factory=list)
    correct: bool = True
    skipped: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        xs = list(self.times_ms)
        return {
            "name": self.name,
            "layer": self.layer,
            "skipped": self.skipped,
            "correct": self.correct,
            "n": len(xs),
            "p50_ms": round(_percentile(xs, 50), 3),
            "p95_ms": round(_percentile(xs, 95), 3),
            "mean_ms": round(float(statistics.fmean(xs)), 3) if xs else 0.0,
            "detail": self.detail,
        }


def _time_call(fn: Callable[[], Any], warmup: int, repeats: int) -> tuple[list[float], Any]:
    last = None
    for _ in range(max(0, warmup)):
        last = fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times, last


def eval_ocr_sample(
    sample: dict,
    img: np.ndarray,
    backend: str,
    preprocess: str,
    *,
    warmup: int,
    repeats: int,
) -> VariantResult:
    name = f"{backend}+{preprocess}"
    vr = VariantResult(name=name, layer="ocr")
    fn_ocr = OCR_BACKENDS.get(backend)
    fn_pre = PREPROCESSORS.get(preprocess)
    if fn_ocr is None or fn_pre is None:
        vr.skipped = "unknown_backend"
        vr.correct = False
        return vr
    needles = [str(x) for x in sample.get("needles") or []]

    def run() -> str:
        return fn_ocr(fn_pre(img))

    try:
        times, blob = _time_call(run, warmup, repeats)
    except Exception as exc:
        vr.skipped = type(exc).__name__ + ": " + str(exc)[:160]
        vr.correct = False
        return vr
    vr.times_ms = times
    blob_s = str(blob or "")
    hits = [n for n in needles if n and n in blob_s]
    miss_rate = 0.0 if not needles else 1.0 - (len(hits) / len(needles))
    fid = trad_fidelity_ok(blob_s, needles)
    word_ok = bool(needles) and len(hits) == len(needles)
    vr.correct = word_ok and fid
    vr.detail = {
        "hits": hits,
        "needles": needles,
        "miss_rate": round(miss_rate, 4),
        "trad_fidelity": fid,
        "blob_preview": blob_s[:80],
        "sample": sample.get("id"),
        "split": sample.get("split"),
    }
    return vr


def eval_template_sample(
    sample: dict,
    img: np.ndarray,
    *,
    warmup: int,
    repeats: int,
    tmp_dir: Path,
) -> VariantResult:
    vr = VariantResult(name="match_gray_ccoeff", layer="match")
    roi = sample.get("template_roi")
    if not roi:
        vr.skipped = "no_template_roi"
        return vr
    x, y, w, h = (int(v) for v in roi)
    crop = img[y : y + h, x : x + w]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tpl = tmp_dir / f"{sample['id']}_tpl.png"
    cv2.imwrite(str(tpl), crop)

    def run():
        return match_template(img, tpl, threshold=0.75, check_margin=False)

    times, hit = _time_call(run, warmup, repeats)
    vr.times_ms = times
    cx, cy = (int(hit.cx), int(hit.cy)) if hit else (-1, -1)
    in_click = False
    in_forb = False
    for box in sample.get("clickable") or []:
        x0, y0, x1, y1 = (int(v) for v in box)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            in_click = True
    for box in sample.get("forbidden") or []:
        x0, y0, x1, y1 = (int(v) for v in box)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            in_forb = True
    allow_empty = not sample.get("clickable")
    vr.correct = (not in_forb) and (in_click or allow_empty or hit is not None)
    if sample.get("family") == "combat" and hit is not None:
        vr.correct = True
    vr.detail = {
        "hit": None if hit is None else {"cx": cx, "cy": cy, "score": hit.score},
        "in_clickable": in_click,
        "in_forbidden": in_forb,
        "sample": sample.get("id"),
    }
    return vr


def eval_ingest(img: np.ndarray, *, warmup: int, repeats: int) -> VariantResult:
    vr = VariantResult(name="numpy_copy", layer="ingest")

    def run():
        return np.ascontiguousarray(img.copy())

    times, _ = _time_call(run, warmup, repeats)
    vr.times_ms = times
    vr.detail = {"shape": list(img.shape), "black": _looks_black(img)}
    vr.correct = not _looks_black(img)
    return vr


def run_offline_benchmark(
    *,
    warmup: int = 1,
    repeats: int = 3,
    ocr_backends: list[str] | None = None,
    preprocess: list[str] | None = None,
    max_combos: int = 80,
    root: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(root=root)
    samples = list(manifest.get("samples") or [])
    backends = ocr_backends or ["rapid_v5", "rapid_cht"]
    pres = preprocess or ["none", "gray3", "clahe"]
    ledger = Path(out_dir) if out_dir is not None else LEDGER_DIR
    tmp = ledger / "_tpl"
    results: list[dict[str, Any]] = []
    combos = 0
    for sample in samples:
        img = load_sample_bgr(sample, root=root)
        ing = eval_ingest(img, warmup=warmup, repeats=repeats)
        results.append(ing.summary())
        mt = eval_template_sample(
            sample, img, warmup=warmup, repeats=repeats, tmp_dir=tmp
        )
        results.append(mt.summary())
        for b in backends:
            for p in pres:
                if combos >= max_combos:
                    break
                vr = eval_ocr_sample(
                    sample, img, b, p, warmup=warmup, repeats=repeats
                )
                results.append(vr.summary())
                combos += 1
            if combos >= max_combos:
                break
        if combos >= max_combos:
            break

    ranked_rows = [r for r in results if r["layer"] == "ocr" and not r.get("skipped")]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in ranked_rows:
        grouped.setdefault(str(r["name"]), []).append(r)
    ocr_ranked: list[dict[str, Any]] = []
    for name, rs in grouped.items():
        n_ok = sum(1 for r in rs if r["correct"])
        p95s = [float(r["p95_ms"]) for r in rs]
        ocr_ranked.append(
            {
                "name": name,
                "samples": len(rs),
                "correct_n": n_ok,
                "accuracy": round(n_ok / max(len(rs), 1), 4),
                "all_correct": n_ok == len(rs),
                "p95_ms_mean": round(sum(p95s) / max(len(p95s), 1), 3),
            }
        )
    ocr_ranked.sort(
        key=lambda r: (not r["all_correct"], -r["accuracy"], r["p95_ms_mean"])
    )
    payload = {
        "schema": "ui_benchmark_run.v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "warmup": warmup,
        "repeats": repeats,
        "rank": "accuracy desc, p95_ms_mean asc",
        "fixture": "synthetic_offline",
        "results": results,
        "ocr_ranked": ocr_ranked,
        "winner_offline_ocr": ocr_ranked[0] if ocr_ranked else None,
        "note": "合成帧赢家只作相对排序；实机黄金帧未采不得宣称感知够用。",
    }
    ledger.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ledger / f"offline_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jsonl = ledger / "offline.jsonl"
    with jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": payload["at"], "path": str(path)}, ensure_ascii=False) + "\n")
    payload["ledger"] = str(path)
    return payload


def format_capture_rank_table(payload: dict[str, Any]) -> str:
    lines = ["rank backend background_ok accuracy p95_ms"]
    for i, r in enumerate(payload.get("ranked") or [], 1):
        lines.append(
            f"{i} {r.get('name')} {int(bool(r.get('background_ok')))} "
            f"{r.get('accuracy')} {r.get('p95_ms')}"
        )
    w = payload.get("winner_capture") or {}
    lines.append(
        f"winner={w.get('name')} acc={w.get('accuracy')} p95_ms={w.get('p95_ms')} "
        f"bg={w.get('background_ok')}"
    )
    return "\n".join(lines) + "\n"


def format_ocr_rank_table(payload: dict[str, Any]) -> str:
    lines = ["rank backend accuracy correct_n/samples p95_ms_mean"]
    for i, r in enumerate(payload.get("ocr_ranked") or [], 1):
        lines.append(
            f"{i} {r.get('name')} {r.get('accuracy')} "
            f"{r.get('correct_n')}/{r.get('samples')} {r.get('p95_ms_mean')}"
        )
    w = payload.get("winner_offline_ocr") or {}
    lines.append(
        f"winner={w.get('name')} acc={w.get('accuracy')} p95_ms={w.get('p95_ms_mean')}"
    )
    return "\n".join(lines) + "\n"


# 后台可用：按 HWND 取帧，不依赖桌面合成最上层。
BG_CAPTURE_BACKENDS = frozenset({"printwindow", "gdi_bitblt"})
ACC_BG_MIN = 0.90


def accuracy_vs_ref(img: np.ndarray, ref: np.ndarray) -> float:
    """相对 PrintWindow 参考帧的准确率：1 - 归一化 MAE。尺寸不同则为 0。"""
    if img is None or ref is None or img.size == 0 or ref.size == 0:
        return 0.0
    if img.shape[:2] != ref.shape[:2]:
        return 0.0
    a = img.astype(np.float32)
    b = ref.astype(np.float32)
    mae = float(np.mean(np.abs(a - b)))
    return max(0.0, 1.0 - mae / 255.0)


def rank_capture_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """后台合格优先，再准确率降序，再 p95 升序。"""
    scored = [r for r in rows if not r.get("skipped")]
    scored.sort(
        key=lambda r: (
            not bool(r.get("background_ok")),
            -round(float(r.get("accuracy") or 0.0), 2),
            float(r.get("p95_ms") or 1e9),
        )
    )
    return scored


def _dxcam_grab_copy(cam: Any, region: tuple[int, int, int, int]) -> np.ndarray:
    frame = cam.grab(region=region, new_frame_only=False)
    if frame is None:
        raise RuntimeError("dxcam returned None")
    return np.ascontiguousarray(frame)


def run_live_capture_benchmark(
    win: WindowInfo,
    *,
    warmup: int = 1,
    repeats: int = 5,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    from mm_assist.platform.window import is_foreground

    rows: list[dict[str, Any]] = []
    last_img: dict[str, np.ndarray] = {}
    fg = bool(is_foreground(int(win.hwnd)))

    def _finish(name: str, times: list[float], img: np.ndarray) -> None:
        vr = VariantResult(name=name, layer="capture")
        vr.times_ms = times
        black = _looks_black(img)
        vr.correct = not black and img.size > 0
        vr.detail = {
            "shape": list(img.shape),
            "black": black,
            "mean": float(img.mean()) if img.size else 0.0,
        }
        last_img[name] = img
        rows.append(vr.summary())

    def time_grab(name: str, fn: Callable[[], np.ndarray]) -> None:
        vr = VariantResult(name=name, layer="capture")
        try:
            times, img = _time_call(fn, warmup, repeats)
        except Exception as exc:
            vr.skipped = type(exc).__name__ + ": " + str(exc)[:160]
            vr.correct = False
            rows.append(vr.summary())
            return
        _finish(name, times, img)

    g_pw = Grabber("printwindow")
    try:
        time_grab("printwindow", lambda: g_pw.grab_client(win))
    finally:
        g_pw.close()

    g_mss = Grabber("mss")
    try:
        time_grab("mss_client", lambda: g_mss.grab_client(win))
        rw = max(80, int(win.width * 0.2))
        rh = max(80, int(win.height * 0.15))
        time_grab(
            "mss_roi",
            lambda: g_mss.grab_rect(int(win.left), int(win.top), rw, rh),
        )
    finally:
        g_mss.close()

    time_grab("gdi_bitblt", lambda: capture_gdi_bitblt(win))

    region = (win.left, win.top, win.left + win.width, win.top + win.height)
    try:
        import dxcam  # type: ignore

        cam = dxcam.create(backend="dxgi", output_color="BGR")
        try:
            times, img = _time_call(
                lambda: _dxcam_grab_copy(cam, region),
                warmup,
                repeats,
            )
            _finish("dxcam_dxgi_reuse", times, img)
        finally:
            cam.release()
    except Exception as exc:
        vr = VariantResult(name="dxcam_dxgi_reuse", layer="capture")
        vr.skipped = type(exc).__name__ + ": " + str(exc)[:160]
        vr.correct = False
        rows.append(vr.summary())

    time_grab("dxcam_winrt", lambda: capture_dxcam(None, "winrt"))

    ref = last_img.get("printwindow")
    for r in rows:
        name = str(r.get("name") or "")
        img = last_img.get(name)
        acc = 1.0 if name == "printwindow" and img is not None else accuracy_vs_ref(img, ref) if img is not None and ref is not None else 0.0
        hwnd_bg = name in BG_CAPTURE_BACKENDS
        not_black = not bool((r.get("detail") or {}).get("black"))
        r["accuracy"] = round(acc, 4)
        r["hwnd_capture"] = hwnd_bg
        r["background_ok"] = bool(
            hwnd_bg and not r.get("skipped") and not_black and acc >= ACC_BG_MIN and r.get("correct")
        )

    ranked = rank_capture_rows(rows)
    payload = {
        "schema": "ui_benchmark_capture.v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "rank": "background_ok, accuracy desc, p95_ms asc",
        "window": {
            "w": win.width,
            "h": win.height,
            "title": win.title,
            "foreground": fg,
        },
        "note": "后台合格=PrintWindow/BitBlt 且相对 PW 准确率>=0.90。MSS/DXcam 依赖桌面合成，后台不合格，即使更快也不排前面。",
        "results": rows,
        "ranked": ranked,
        "winner_capture": ranked[0] if ranked else None,
    }
    ledger = Path(out_dir) if out_dir is not None else LEDGER_DIR
    ledger.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ledger / f"capture_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload["ledger"] = str(path)
    return payload
