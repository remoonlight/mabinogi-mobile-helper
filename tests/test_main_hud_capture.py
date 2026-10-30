"""主界面入口只读采证契约。"""

from __future__ import annotations

import json
import hashlib
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import cv2

from mm_assist.jobs.hud_entries import (
    archive_hud_entry_frame,
    archive_hud_entry_job,
    assess_hud_entry_captures,
    clickable_status,
    entry_by_id,
    load_main_hud_catalog,
    next_entry_id,
    open_hud_entry_frame,
    record_capture_box,
    roi_by_id,
    validate_hud_entry,
    validate_main_hud_catalog,
)
from mm_assist.perception.hud_entry_locator import locate_hud_entry


def test_catalog_loads_seven_entries():
    cat = load_main_hud_catalog()
    ids = [e["id"] for e in cat["entries"]]
    assert ids == ["menu", "bag", "avatar", "craft", "map", "combat", "stop"]
    validate_main_hud_catalog(cat)
    for entry in cat["entries"]:
        locator = entry["locator"]
        assert locator["detector"] in {"template", "stop_shape"}
        assert locator["maturity"] in {"calibration", "accepted"}
    for eid in ("menu", "bag", "craft", "map"):
        assert entry_by_id(cat, eid)["kind"] == "page_entry"
    assert entry_by_id(cat, "avatar")["kind"] == "observe_only"


def test_next_entry_id_walks_seven_in_order():
    assert next_entry_id("menu") == "bag"
    assert next_entry_id("stop") is None
    assert next_entry_id("unknown") is None
    assert next_entry_id("menu", queue=["menu", "craft", "combat", "stop"]) == "craft"
    assert next_entry_id("stop", queue=["menu", "craft", "combat", "stop"]) is None


def test_freeze_hud_entry_shot_writes_png_immediately(tmp_path: Path, monkeypatch):
    from mm_assist.jobs.hud_entries import freeze_hud_entry_shot
    from mm_assist.platform.window import WindowInfo

    fake = np.zeros((20, 30, 3), dtype=np.uint8)
    fake[0, 0] = (1, 2, 3)
    monkeypatch.setattr(
        "mm_assist.platform.capture.grab_client",
        lambda _win, method=None: fake,
    )
    monkeypatch.setattr(
        "mm_assist.platform.window.refresh_window",
        lambda win: win,
    )
    shot = freeze_hud_entry_shot(
        WindowInfo(1, "t", 10, 20, 30, 20),
        capture_id="fz1",
        root=tmp_path,
    )
    path = tmp_path / shot["path"]
    assert path.is_file()
    assert shot["origin"] == [10, 20]
    saved = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert saved is not None
    assert int(saved[0, 0, 2]) == 3


def test_vision_requires_frozen_frame():
    from mm_assist.jobs.hud_entries import run_hud_entry_vision

    out = run_hud_entry_vision(None, {"entry_id": "combat", "pair_role": "source"})
    assert out["ok"] is False
    assert "冻结帧" in str(out.get("error") or "")


def test_record_capture_box_saves_pattern_and_client_rect(tmp_path: Path):
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[4:12, 10:18] = (3, 5, 9)
    out = record_capture_box(
        "menu",
        frame,
        (10, 4, 8, 8),
        capture_id="box1",
        root=tmp_path,
    )
    assert out["client_rect"] == [10, 4, 8, 8]
    assert out["size"] == [8, 8]
    pattern = tmp_path / out["pattern_path"]
    assert pattern.is_file()
    saved = cv2.imdecode(np.fromfile(str(pattern), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert saved is not None
    assert int(saved[0, 0, 2]) == 9


def test_catalog_rejects_entry_without_locator():
    cat = load_main_hud_catalog()
    cat["entries"][0].pop("locator")
    with pytest.raises(ValueError, match="locator"):
        validate_main_hud_catalog(cat)


def test_catalog_rejects_accepted_missing_template():
    cat = load_main_hud_catalog()
    cat["entries"][0]["locator"]["maturity"] = "accepted"
    cat["entries"][0]["locator"]["template"] = "knowledge/templates/hud_entries/missing.png"
    with pytest.raises(ValueError, match="accepted template 不存在"):
        validate_main_hud_catalog(cat)


def test_roi_menu_and_travel():
    menu = roi_by_id("hud.menu_esc")
    travel = roi_by_id("hud.travel")
    assert menu is not None and len(menu) == 4
    assert travel == (0.70, 0.78, 1.0, 1.0)
    assert roi_by_id("hud.missing") is None


def test_open_hud_entry_method_b(tmp_path: Path):
    job = open_hud_entry_frame(
        "menu",
        "source",
        capture_id="unit_menu",
        path=tmp_path / "j.json",
    )
    assert job["status"] == "await_start"
    assert job["method"] == "B"
    assert job["action"] == "readonly_ui_fine"
    assert job["args"]["entry_id"] == "menu"
    assert job["args"]["pair_role"] == "source"
    assert job["args"]["node_id"] == "main_hud"


def test_validate_menu_fixture_pair_ok_enough_false():
    root = Path(__file__).resolve().parent / "fixtures" / "main_hud_menu"
    out = validate_hud_entry(
        "menu",
        source_path=root / "source.json",
        target_path=root / "target.json",
    )
    assert out["ok"] is True
    assert out["pair_ok"] is True
    assert out["enough"] is False
    assert out["assess"]["click"]["allow_click"] is True
    assert out["assess"]["click_accept_ok"] is False


def test_click_accept_requires_accepted_locator_overlay_evidence_and_confirmation():
    root = Path(__file__).resolve().parent / "fixtures" / "main_hud_menu"
    source = json.loads((root / "source.json").read_text(encoding="utf-8"))
    target = json.loads((root / "target.json").read_text(encoding="utf-8"))
    source["result"]["locator"] = {
        "feature_ok": True,
        "position_ok": True,
        "accepted": True,
    }
    source["result"]["overlay_evidence"] = {"ok": True}
    source["result"]["overlay_confirmation"] = {"decision": "correct"}
    source.update(
        {
            "status": "ok",
            "method": "B",
            "action": "readonly_ui_fine",
            "args": {
                "entry_id": "menu",
                "pair_role": "source",
                "capture_id": "menu_accept_1",
            },
        }
    )
    source["result"]["hud_entry_evidence"] = {"raw_sha256": "source-hash"}
    target.update(
        {
            "status": "ok",
            "method": "B",
            "action": "readonly_ui_fine",
            "args": {
                "entry_id": "menu",
                "pair_role": "target",
                "capture_id": "menu_accept_1",
            },
        }
    )
    target["result"]["hud_entry_evidence"] = {"raw_sha256": "target-hash"}
    catalog = load_main_hud_catalog()
    result = assess_hud_entry_captures(
        entry_by_id(catalog, "menu"),
        source,
        target,
        catalog=catalog,
    )

    assert result["pair_ok"] is True
    assert result["provenance_ok"] is True
    assert result["click_accept_ok"] is True


def test_click_accept_rejects_mismatched_capture_ids():
    root = Path(__file__).resolve().parent / "fixtures" / "main_hud_menu"
    source = json.loads((root / "source.json").read_text(encoding="utf-8"))
    target = json.loads((root / "target.json").read_text(encoding="utf-8"))
    for job, role, cid in (
        (source, "source", "capture-a"),
        (target, "target", "capture-b"),
    ):
        job.update(
            {
                "status": "ok",
                "method": "B",
                "action": "readonly_ui_fine",
                "args": {
                    "entry_id": "menu",
                    "pair_role": role,
                    "capture_id": cid,
                },
            }
        )
        job["result"]["hud_entry_evidence"] = {"raw_sha256": f"{role}-hash"}
    source["result"]["locator"] = {
        "feature_ok": True,
        "position_ok": True,
        "accepted": True,
    }
    source["result"]["overlay_evidence"] = {"ok": True}
    source["result"]["overlay_confirmation"] = {"decision": "correct"}
    catalog = load_main_hud_catalog()

    result = assess_hud_entry_captures(
        entry_by_id(catalog, "menu"),
        source,
        target,
        catalog=catalog,
    )

    assert result["pair_ok"] is True
    assert result["provenance_ok"] is False
    assert result["click_accept_ok"] is False


def test_map_target_requires_exact_map_type_not_truthiness():
    catalog = load_main_hud_catalog()
    entry = entry_by_id(catalog, "map")
    source = {
        "result": {
            "panel_probe": {
                "menu_open": False,
                "inventory_open": False,
                "dialog_active": False,
                "map_open": False,
                "profile_open": False,
            },
            "icons": [],
        }
    }
    target = {
        "result": {
            "panel_probe": {
                "menu_open": True,
                "map_open": True,
                "profile_open": False,
                "map_type": 1,
            }
        }
    }

    result = assess_hud_entry_captures(entry, source, target, catalog=catalog)

    assert result["pair_ok"] is False
    assert "map_type:want=2 got=1" in result["target_miss"]


def test_validate_click_accept_verifies_archived_file_hashes(tmp_path: Path):
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "main_hud_menu"
    source = json.loads((fixture_root / "source.json").read_text(encoding="utf-8"))
    target = json.loads((fixture_root / "target.json").read_text(encoding="utf-8"))
    source_raw = tmp_path / "source.raw.png"
    target_raw = tmp_path / "target.raw.png"
    overlay = tmp_path / "source.overlay.png"
    template_path = tmp_path / "menu_template.png"
    template = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.rectangle(template, (1, 1), (18, 18), (240, 240, 240), 2)
    cv2.line(template, (3, 10), (16, 10), (20, 180, 250), 3)
    source_image = np.zeros((100, 200, 3), dtype=np.uint8)
    source_image[40:60, 90:110] = template
    target_image = np.zeros((100, 200, 3), dtype=np.uint8)
    overlay_image = source_image.copy()
    cv2.circle(overlay_image, (100, 50), 9, (32, 32, 255), -1)
    assert cv2.imwrite(str(template_path), template)
    assert cv2.imwrite(str(source_raw), source_image)
    assert cv2.imwrite(str(target_raw), target_image)
    assert cv2.imwrite(str(overlay), overlay_image)

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for job, role, raw_path in (
        (source, "source", source_raw),
        (target, "target", target_raw),
    ):
        job.update(
            {
                "status": "ok",
                "method": "B",
                "action": "readonly_ui_fine",
                "args": {
                    "entry_id": "menu",
                    "pair_role": role,
                    "capture_id": "menu-hash-1",
                },
            }
        )
        job["result"]["hud_entry_evidence"] = {
            "entry_id": "menu",
            "capture_id": "menu-hash-1",
            "role": role,
            "raw_path": raw_path.name,
            "raw_sha256": _sha(raw_path),
        }
        job["result"]["client"] = {"w": 200, "h": 100}
    catalog = copy.deepcopy(load_main_hud_catalog())
    menu_entry = entry_by_id(catalog, "menu")
    menu_entry["locator"] = {
        "detector": "template",
        "template": template_path.name,
        "threshold": 0.8,
        "min_margin": 0.02,
        "maturity": "accepted",
    }
    source["result"]["locator"] = locate_hud_entry(
        menu_entry,
        source_image,
        root=tmp_path,
    )
    source["result"]["overlay_evidence"] = {
        "ok": True,
        "path": overlay.name,
        "sha256": _sha(overlay),
        "entry_id": "menu",
        "capture_id": "menu-hash-1",
        "raw_sha256": _sha(source_raw),
        "click_client": [100, 50],
        "client_box": [0, 0, 200, 100],
    }
    source["result"]["overlay_confirmation"] = {
        "decision": "correct",
        "entry_id": "menu",
        "capture_id": "menu-hash-1",
        "raw_sha256": _sha(source_raw),
        "click_client": [100, 50],
    }
    source_path = tmp_path / "source.job.json"
    target_path = tmp_path / "target.job.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    target_path.write_text(json.dumps(target), encoding="utf-8")
    accepted = validate_hud_entry(
        "menu",
        source_path=source_path,
        target_path=target_path,
        root=tmp_path,
        catalog=catalog,
    )
    source["result"]["overlay_confirmation"]["capture_id"] = "other-capture"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    wrong_confirmation = validate_hud_entry(
        "menu",
        source_path=source_path,
        target_path=target_path,
        root=tmp_path,
        catalog=catalog,
    )
    source["result"]["overlay_confirmation"]["capture_id"] = "menu-hash-1"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    source_raw.write_bytes(b"tampered")
    tampered = validate_hud_entry(
        "menu",
        source_path=source_path,
        target_path=target_path,
        root=tmp_path,
        catalog=catalog,
    )

    assert accepted["assess"]["integrity_ok"] is True
    assert accepted["assess"]["click_accept_ok"] is True
    assert wrong_confirmation["assess"]["click_accept_ok"] is False
    assert tampered["assess"]["integrity_ok"] is False
    assert tampered["assess"]["click_accept_ok"] is False


def test_cli_exit_code_tracks_final_click_acceptance(monkeypatch):
    from mm_assist import cli
    from mm_assist.jobs import hud_entries

    args = SimpleNamespace(
        entry="menu",
        source="source.json",
        target="target.json",
        json=False,
    )
    monkeypatch.setattr(
        hud_entries,
        "format_validate_hud_entry_report",
        lambda _out: "report\n",
    )
    monkeypatch.setattr(
        hud_entries,
        "validate_hud_entry",
        lambda *_args, **_kwargs: {
            "ok": True,
            "assess": {"click_accept_ok": False},
        },
    )
    assert cli.cmd_validate_hud_entry(args) == 1

    monkeypatch.setattr(
        hud_entries,
        "validate_hud_entry",
        lambda *_args, **_kwargs: {
            "ok": True,
            "assess": {"click_accept_ok": True},
        },
    )
    assert cli.cmd_validate_hud_entry(args) == 0


def test_stop_not_visible_blocks_click():
    cat = load_main_hud_catalog()
    entry = entry_by_id(cat, "stop")
    job = {
        "result": {
            "stop_visible": False,
            "icons": [],
            "panel_probe": {
                "menu_open": False,
                "inventory_open": False,
                "dialog_active": False,
                "map_open": False,
                "profile_open": False,
                "stop_visible": False,
            },
        }
    }
    click = clickable_status(entry, job)
    assert click["allow_click"] is False
    assert click["reason"] == "stop_not_visible"


def test_combat_observe_only_without_interactable():
    cat = load_main_hud_catalog()
    entry = entry_by_id(cat, "combat")
    job = {
        "result": {
            "icons": [{"key": "esc", "state": "Show", "interactable": True}],
            "panel_probe": {"menu_open": False},
        }
    }
    click = clickable_status(entry, job)
    assert click["allow_click"] is False
    assert click["reason"] == "combat_observe_only"


def test_cli_has_hud_entry_commands():
    from mm_assist.cli import ALLOWED_CMDS

    assert "capture" in ALLOWED_CMDS
    assert "validate" in ALLOWED_CMDS


def test_unknown_entry_rejected():
    cat = load_main_hud_catalog()
    with pytest.raises(KeyError, match="未知入口"):
        entry_by_id(cat, "nerys")


def test_archive_hud_entry_frame_binds_raw_hash_locator_and_result(tmp_path: Path):
    cat = load_main_hud_catalog()
    entry = entry_by_id(cat, "menu")
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    evidence = archive_hud_entry_frame(
        entry,
        frame,
        capture_id="menu_unit",
        role="source",
        detail={"ui_params": "inv=0 menu=0 dlg=0"},
        root=tmp_path,
        screen_origin=(300, 200),
    )

    assert evidence["raw_sha256"]
    assert evidence["locator"]["position_ok"] is True
    assert evidence["locator"]["feature_ok"] is False
    raw_path = tmp_path / evidence["raw_path"]
    result_path = tmp_path / evidence["result_path"]
    assert raw_path.is_file()
    assert result_path.is_file()
    assert evidence["raw_sha256"] in result_path.read_text(encoding="utf-8")


def test_archive_hud_entry_job_persists_confirmation_updates(tmp_path: Path):
    job = {
        "id": "job-1",
        "status": "ok",
        "method": "B",
        "action": "readonly_ui_fine",
        "args": {
            "capture_id": "menu_unit",
            "pair_id": "menu_unit",
            "pair_role": "source",
            "entry_id": "menu",
        },
        "result": {
            "locator": {"accepted": True},
            "overlay_evidence": {"ok": True},
            "overlay_confirmation": {"decision": "correct"},
        },
    }

    path = archive_hud_entry_job(job, root=tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["id"] == "job-1"
    assert saved["result"]["overlay_confirmation"]["decision"] == "correct"
    assert path.name == "source.job.json"
