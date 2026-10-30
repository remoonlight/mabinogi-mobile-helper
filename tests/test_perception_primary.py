# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

from mm_assist.platform.capture import Grabber
from mm_assist.perception.ocr import ocr_image
from mm_assist.platform.window import WindowInfo


def _win() -> WindowInfo:
    return WindowInfo(hwnd=1, title="x", left=0, top=0, width=64, height=48)


def test_auto_uses_rendered_frame_instead_of_foreground_client_surface(monkeypatch):
    monkeypatch.setattr('mm_assist.platform.window.is_foreground',lambda _:True)
    g = Grabber("auto")
    calls: list[str] = []
    good = np.full((48, 64, 3), 80, dtype=np.uint8)

    def bitblt(_win):
        calls.append("bitblt")
        return good

    def pw(_win):
        calls.append("pw")
        return good

    monkeypatch.setattr(g, "_bitblt_client", bitblt)
    monkeypatch.setattr(g, "_printwindow_client", pw)
    out = g.grab_client(_win())
    assert out is good
    assert calls == ["pw"]
    assert g._last_source == "printwindow"


def test_auto_render_does_not_depend_on_black_client_surface(monkeypatch):
    monkeypatch.setattr('mm_assist.platform.window.is_foreground',lambda _:True)
    g = Grabber("auto")
    black = np.zeros((48, 64, 3), dtype=np.uint8)
    good = np.full((48, 64, 3), 90, dtype=np.uint8)
    monkeypatch.setattr(g, "_bitblt_client", lambda _w: black)
    monkeypatch.setattr(g, "_printwindow_client", lambda _w: good)
    out = g.grab_client(_win())
    assert out is good
    assert g._last_source == "printwindow"


def test_ocr_up2_primary_clahe_backup(monkeypatch):
    img = np.full((16, 16, 3), 40, dtype=np.uint8)
    seen: list[tuple[int, int]] = []

    class Fake:
        def __call__(self, im):
            h, w = im.shape[:2]
            seen.append((h, w))
            if (h, w) == (32, 32):
                class Out:
                    boxes = [[[0, 0], [2, 0], [2, 2], [0, 2]]]
                    txts = ("採礦",)
                    scores = (0.9,)

                return Out()

            class Out2:
                boxes = [[[0, 0], [2, 0], [2, 2], [0, 2]]]
                txts = ("藥",)
                scores = (0.9,)

            return Out2()

    monkeypatch.setenv("MM_OCR_BACKEND", "v5")
    monkeypatch.setenv("MM_OCR_PREPROCESS", "up2,clahe")
    monkeypatch.setattr("mm_assist.perception.ocr._engine", lambda _b: ("rapid", Fake()))
    lines = ocr_image(img, min_confidence=0.5)
    assert [L.text for L in lines] == ["採礦"]
    assert seen[0] == (32, 32)


def test_ocr_fallback_when_up2_empty(monkeypatch):
    img = np.full((16, 16, 3), 40, dtype=np.uint8)

    class Fake:
        def __call__(self, im):
            h = im.shape[0]

            class Out:
                boxes = [] if h > 16 else [[[0, 0], [1, 0], [1, 1], [0, 1]]]
                txts = () if h > 16 else ("裝備",)
                scores = () if h > 16 else (0.95,)

            return Out()

    monkeypatch.setenv("MM_OCR_PREPROCESS", "up2,clahe")
    monkeypatch.setattr("mm_assist.perception.ocr._engine", lambda _b: ("rapid", Fake()))
    lines = ocr_image(img, min_confidence=0.5)
    assert [L.text for L in lines] == ["裝備"]


def test_ocr_skip_up2_on_large_crop(monkeypatch):
    img = np.full((400, 700, 3), 40, dtype=np.uint8)
    seen: list[tuple[int, int]] = []

    class Fake:
        def __call__(self, im):
            h, w = im.shape[:2]
            seen.append((h, w))

            class Out:
                boxes = [[[0, 0], [2, 0], [2, 2], [0, 2]]]
                txts = ("礦",)
                scores = (0.9,)

            return Out()

    monkeypatch.setenv("MM_OCR_BACKEND", "v5")
    monkeypatch.setenv("MM_OCR_PREPROCESS", "up2,clahe")
    monkeypatch.setattr("mm_assist.perception.ocr._engine", lambda _b: ("rapid", Fake()))
    lines = ocr_image(img, min_confidence=0.5)
    assert [L.text for L in lines] == ["礦"]
    assert seen[0] == (400, 700)
    assert (800, 1400) not in seen
