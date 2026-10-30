from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.perception.hud_sense import (
    STOP_TEMPLATE_PATH,
    detect_stop_icon,
    match_stop_template,
)
from mm_assist.perception.ui_fine import merge_stop_visible, status_bar_stop_visible


def test_merge_stop_visible_mem_first_then_image():
    assert merge_stop_visible(True, True) == (True, "mem")
    assert merge_stop_visible(False, False) == (False, "mem")
    assert merge_stop_visible(True, False) == (True, "mem")
    assert merge_stop_visible(False, True) == (False, "mem")
    assert merge_stop_visible(True, None) == (True, "mem")
    assert merge_stop_visible(None, False) == (False, "image")
    assert merge_stop_visible(None, None) == (None, "none")
    assert status_bar_stop_visible(True, False) == (True, "mem")


def test_stop_template_self_match_in_search_roi():
    path = Path(STOP_TEMPLATE_PATH)
    if not path.is_file():
        pytest.skip("缺 knowledge/templates/hud_entries/stop.png")
    tmpl = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert tmpl is not None
    th, tw = tmpl.shape[:2]
    frame = np.zeros((1440, 1920, 3), dtype=np.uint8)
    x0, y0 = 1727, 1248
    frame[y0 : y0 + th, x0 : x0 + tw] = tmpl
    assert match_stop_template(frame) is True
    assert detect_stop_icon(frame, ocr_space=False) is True


def test_stop_template_miss_on_blank():
    blank = np.zeros((1440, 1920, 3), dtype=np.uint8)
    if not Path(ROOT / "knowledge/templates/hud_entries/stop.png").is_file():
        pytest.skip("缺 stop.png")
    assert match_stop_template(blank) is False
    assert detect_stop_icon(blank, ocr_space=False) is False
