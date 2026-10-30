from __future__ import annotations

import numpy as np

from mm_assist.platform.hint_overlay import (
    overlay_marker_from_job,
    red_marker_delta_ok,
    red_marker_center_ok,
    screen_rect_to_client,
    crop_client_pattern,
)


def _source_job(locator: dict) -> dict:
    return {
        "id": "job-1",
        "action": "readonly_ui_fine",
        "method": "B",
        "status": "ok",
        "args": {
            "entry_id": "menu",
            "pair_role": "source",
            "capture_id": "menu-1",
        },
        "result": {"locator": locator},
    }


def test_source_job_with_position_shows_click_through_marker():
    marker = overlay_marker_from_job(
        _source_job(
            {
                "feature_ok": True,
                "position_ok": True,
                "accepted": True,
                "click": {"screen": [1500, 80], "client": [1200, 40]},
            }
        )
    )

    assert marker["visible"] is True
    assert marker["screen"] == [1500, 80]
    assert marker["entry_id"] == "menu"
    assert marker["capture_id"] == "menu-1"
    assert marker["accepted"] is True
    assert marker["provisional"] is False
    assert marker["client_rect"] == [1176, 16, 48, 48]
    assert marker["screen_rect"] == [1476, 56, 48, 48]


def test_calibration_roi_may_show_but_is_never_accepted():
    marker = overlay_marker_from_job(
        _source_job(
            {
                "feature_ok": False,
                "position_ok": True,
                "accepted": False,
                "provisional": True,
                "click": {"screen": [1500, 80]},
            }
        )
    )

    assert marker["visible"] is True
    assert marker["accepted"] is False
    assert marker["provisional"] is True


def test_target_or_failed_job_hides_marker():
    target = _source_job({"position_ok": True, "click": {"screen": [1, 2]}})
    target["args"]["pair_role"] = "target"
    failed = _source_job({"position_ok": True, "click": {"screen": [1, 2]}})
    failed["status"] = "fail"

    assert overlay_marker_from_job(target) == {"visible": False}
    assert overlay_marker_from_job(failed) == {"visible": False}


def test_overlay_evidence_requires_red_pixels_at_expected_center():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[36:45, 46:55] = (32, 32, 255)

    assert red_marker_center_ok(image, client_xy=(50, 40)) is True
    assert red_marker_center_ok(image, client_xy=(10, 10)) is False


def test_hint_overlay_show_deiconifies_withdrawn_window():
    import tkinter as tk

    from mm_assist.platform.hint_overlay import HintOverlay

    root = tk.Tk()
    root.withdraw()
    overlay = HintOverlay(root, size=24)
    try:
        overlay.show_rect(68, 78, 24, 24)
        assert str(overlay._window.state()) != "withdrawn"
        geom = overlay._window.geometry()
        assert geom.startswith("24x24")
        assert "+68+78" in geom
    finally:
        overlay.destroy()
        root.destroy()


def test_overlay_evidence_requires_new_red_pixels_not_game_red():
    raw = np.zeros((80, 100, 3), dtype=np.uint8)
    raw[36:45, 46:55] = (32, 32, 255)
    unchanged = raw.copy()
    with_marker = raw.copy()
    with_marker[30:51, 40:61] = (20, 20, 255)

    assert red_marker_delta_ok(raw, unchanged, client_xy=(50, 40)) is False
    assert red_marker_delta_ok(raw, with_marker, client_xy=(50, 40)) is True


def test_screen_rect_maps_into_game_client():
    assert screen_rect_to_client(
        (1060, 420, 40, 30),
        origin=(1000, 400),
        frame_w=1920,
        frame_h=1440,
    ) == (60, 20, 40, 30)


def test_crop_client_pattern_keeps_patch_pixels():
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    frame[10:18, 20:28] = (9, 8, 7)
    patch = crop_client_pattern(frame, (20, 10, 8, 8))
    assert patch is not None
    assert patch.shape == (8, 8, 3)
    assert int(patch[0, 0, 2]) == 7
