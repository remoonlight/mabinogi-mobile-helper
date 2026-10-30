# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mm_assist.benchmarks.item_sense_bench import (
    probe_skill_items_on_frame,
    run_catalog_check,
    run_frame_batch,
    run_frame_benchmark,
)
from mm_assist.perception.item_sense import (
    FIND_NEARBY_OCR_NEEDLES,
    HELP_OCR_NEEDLES,
    iter_tier_items,
    locate_find_nearby,
    locate_help_button,
    ocr_needle_matches,
)
from mm_assist.platform.match import MatchHit


def test_iter_tier_items_count():
    rows = iter_tier_items()
    assert len(rows) >= 40
    assert len(rows) == len({(r["skill_id"], r["name"]) for r in rows})
    assert any(r["skill_id"] == "daily" and r["name"] == "水井" for r in rows)


def test_catalog_check_ok_when_templates_present():
    rows = iter_tier_items()
    out = run_catalog_check()
    assert out["mode"] == "catalog"
    assert len(out["items"]) == len(rows)
    assert len(out["actions"]) == 2
    labels = {a["label"] for a in out["actions"]}
    assert labels == {"find_nearby", "help"}
    assert all(a["template_ok"] for a in out["actions"])
    missing_items = [i["name"] for i in out["items"] if not i["template_ok"]]
    assert missing_items == []
    assert out["ok"] is True
    assert out["missing"] == 0


def test_probe_skill_items_on_blank_frame():
    from mm_assist.perception.item_sense import items_for_skill

    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    out = probe_skill_items_on_frame(img, "shear")
    assert out["total"] == len(items_for_skill("shear"))
    assert out["visible_ok"] == 0
    assert out["ok"] is False


def test_run_frame_batch_smoke():
    from mm_assist.perception.item_sense import items_for_skill

    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    out = run_frame_batch(img, skill_id="daily")
    assert out["mode"] == "frame_batch"
    assert out["total_items"] == len(items_for_skill("daily"))
    assert out["visible_ok"] == 0


def test_ocr_needle_matches_find_nearby():
    assert ocr_needle_matches("尋找附近位置", FIND_NEARBY_OCR_NEEDLES)
    assert ocr_needle_matches("寻找附近", FIND_NEARBY_OCR_NEEDLES)
    assert not ocr_needle_matches("開始採集", FIND_NEARBY_OCR_NEEDLES)


def test_locate_find_nearby_ocr_fallback(monkeypatch):
    from mm_assist.perception import item_sense as mod

    img = np.zeros((200, 400, 3), dtype=np.uint8)

    class _Line:
        def __init__(self, text: str):
            self.text = text
            self.box = [(10, 150), (120, 150), (120, 170), (10, 170)]

    monkeypatch.setattr(mod, "ocr_roi", lambda *a, **k: [_Line("尋找附近位置")])
    out = locate_find_nearby(img)
    assert out["ok"] is True
    assert out["src"] == "ocr_cht"


def test_locate_find_nearby_template(monkeypatch, tmp_path: Path):
    from mm_assist.perception import item_sense as mod

    rel = mod.FIND_NEARBY_REL
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")

    img = np.zeros((200, 400, 3), dtype=np.uint8)
    hit = MatchHit(score=0.9, cx=200, cy=180, x=180, y=160, w=40, h=40)
    monkeypatch.setattr(mod, "match_template", lambda *a, **k: hit)
    out = locate_find_nearby(img, root=tmp_path)
    assert out["ok"] is True
    assert out["src"] == "template"


def test_locate_help_ocr_fallback(monkeypatch):
    from mm_assist.perception import item_sense as mod

    img = np.zeros((200, 400, 3), dtype=np.uint8)

    class _Line:
        def __init__(self, text: str):
            self.text = text
            self.box = [(300, 150), (320, 150), (320, 170), (300, 170)]

    monkeypatch.setattr(mod, "ocr_roi", lambda *a, **k: [_Line("?")])
    out = locate_help_button(img)
    assert out["ok"] is True
    assert out["src"] == "ocr_cht"
    assert ocr_needle_matches("?", HELP_OCR_NEEDLES)


def test_run_frame_benchmark_smoke(monkeypatch, tmp_path: Path):
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "mm_assist.benchmarks.item_sense_bench.locate_skill_item",
        lambda *a, **k: {"ok": False, "error": "not_found"},
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.item_sense_bench.locate_find_nearby",
        lambda *a, **k: {"ok": True, "src": "ocr_cht"},
    )
    monkeypatch.setattr(
        "mm_assist.benchmarks.item_sense_bench.locate_help_button",
        lambda *a, **k: {"ok": False, "error": "missing_template"},
    )
    out = run_frame_benchmark(img, skill_id="woodcut", item_name="樹木", root=tmp_path)
    assert out["find_nearby"]["ok"] is True
    assert out["help"]["ok"] is False
