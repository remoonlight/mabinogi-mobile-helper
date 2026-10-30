# -*- coding: utf-8 -*-
from __future__ import annotations

from mm_assist.benchmarks.ui import (
    char_error_rate,
    eval_template_sample,
    levenshtein,
    run_offline_benchmark,
    trad_fidelity_ok,
)
from mm_assist.benchmarks.fixtures import ensure_synthetic_fixtures, load_manifest, load_sample_bgr


def test_levenshtein_and_cer():
    assert levenshtein("採礦", "採礦") == 0
    assert char_error_rate("采礦", "採礦") > 0


def test_trad_fidelity_rejects_simplified():
    assert trad_fidelity_ok("開始採集", ["開始採集"]) is True
    assert trad_fidelity_ok("开始采集", ["開始採集"]) is False


def test_fixtures_have_holdout_and_trad_needles():
    ensure_synthetic_fixtures()
    man = load_manifest()
    assert man["locale"] == "zh-Hant"
    ids = [s["id"] for s in man["samples"]]
    assert "skill_list" in ids
    hold = [s for s in man["samples"] if s["split"] == "holdout"]
    assert hold
    blob = "".join("".join(s["needles"]) for s in man["samples"])
    for ch in ("製", "藥", "採", "礦"):
        assert ch in blob


def test_offline_benchmark_writes_ranked(tmp_path, monkeypatch):
    from mm_assist.benchmarks import ui as ub

    def fake_ocr(img):
        return "選單背包生活技能採礦採集藥草礦脈尋找附近開始採集製作藥品加工裝備鋤鐮戰鬥中"

    monkeypatch.setattr(ub, "LEDGER_DIR", tmp_path)
    monkeypatch.setitem(ub.OCR_BACKENDS, "rapid_v5", fake_ocr)
    out = run_offline_benchmark(
        warmup=0, repeats=1, ocr_backends=["rapid_v5"], preprocess=["none"], out_dir=tmp_path
    )
    assert out["schema"] == "ui_benchmark_run.v1"
    assert any(r["layer"] == "match" for r in out["results"])
    ocr_rows = [r for r in out["results"] if r["layer"] == "ocr"]
    assert ocr_rows
    assert all(r["correct"] for r in ocr_rows)
    assert (tmp_path / "offline.jsonl").is_file()


def test_template_center_not_in_forbidden():
    man = load_manifest()
    sample = next(s for s in man["samples"] if s["id"] == "hud_entries")
    img = load_sample_bgr(sample)
    vr = eval_template_sample(sample, img, warmup=0, repeats=1, tmp_dir=tmp_path_dir())
    assert vr.correct is True
    assert vr.detail.get("in_forbidden") is False


def tmp_path_dir():
    from mm_assist.config import ROOT

    p = ROOT / "logs" / "benchmarks" / "_tpl_test"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_accuracy_vs_ref_and_bg_rank():
    import numpy as np

    from mm_assist.benchmarks.ui import accuracy_vs_ref, rank_capture_rows

    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = a.copy()
    assert accuracy_vs_ref(a, b) == 1.0
    b[0, 0] = (255, 255, 255)
    assert 0.9 < accuracy_vs_ref(a, b) < 1.0
    rows = [
        {"name": "dxcam", "skipped": None, "background_ok": False, "accuracy": 0.99, "p95_ms": 3},
        {"name": "printwindow", "skipped": None, "background_ok": True, "accuracy": 1.0, "p95_ms": 50},
        {"name": "gdi_bitblt", "skipped": None, "background_ok": True, "accuracy": 0.99, "p95_ms": 20},
    ]
    ranked = rank_capture_rows(rows)
    assert [r["name"] for r in ranked] == ["printwindow", "gdi_bitblt", "dxcam"]


def test_cli_has_benchmark():
    from mm_assist.cli import ALLOWED_CMDS

    assert "benchmark" in ALLOWED_CMDS
    assert "watch" in ALLOWED_CMDS
