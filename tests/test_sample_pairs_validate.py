"""sample_pairs 成对归档离线校验（零键鼠）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_assist.jobs.pairs import (
    format_validate_pair_report,
    load_craft_pair_jobs,
    validate_craft_pair_jobs,
    validate_sample_pair,
)


def test_validate_craft_pair_r33_fixture_paths():
    root = Path(__file__).resolve().parent / "fixtures" / "craft_pair_r33"
    closed = root / "craft_closed.json"
    open_ = root / "craft_open.json"
    out = validate_sample_pair(
        "craft",
        closed_path=closed,
        open_path=open_,
    )
    assert out["ok"] is True
    assert out["pair_ok"] is False
    assert out["enough"] is False
    assert "pair:open_hub_not_1_2_3" in out["fail_tags"]
    assert "pair:no_visual_fallback_in_archive" in out["fail_tags"]
    report = format_validate_pair_report(out)
    assert "pair_ok=False" in report
    assert "enough=False" in report


def test_validate_sample_pair_missing_dir():
    out = validate_sample_pair("craft", pair_id="craft_nonexistent_unit_xyz")
    assert out["ok"] is False
    assert out["error"]
    assert "FileNotFoundError" in out["error"]


def test_load_craft_pair_jobs_and_assess():
    root = Path(__file__).resolve().parent / "fixtures" / "craft_pair_r33"
    closed_job, open_job = load_craft_pair_jobs(
        "",
        closed_path=root / "craft_closed.json",
        open_path=root / "craft_open.json",
    )
    assess = validate_craft_pair_jobs(closed_job, open_job)
    assert assess["menu_flip"] is True
    assert assess["visual_open_ok"] is False
    assert assess["enough"] is False


def test_validate_sample_pair_unknown_kind():
    with pytest.raises(ValueError, match="未知 pair_kind"):
        validate_sample_pair("stop")
