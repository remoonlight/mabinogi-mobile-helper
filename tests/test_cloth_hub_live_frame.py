from pathlib import Path

import cv2

from mm_assist.life_execute import ui_nav
from mm_assist.perception import process_sense
from mm_assist.perception.ocr import OcrLine


FIX = Path(__file__).parent / "fixtures/ui_benchmark/cloth_hub_click_no_move.png"


def test_cloth_hub_uses_observed_caption_without_model_offset(monkeypatch):
    """14:03 click above the caption at (1649,258) did not open a page."""
    frame = cv2.imread(str(FIX))
    assert frame is not None
    monkeypatch.setattr(process_sense, "_match_card", lambda *a, **k: None)
    caption = OcrLine("布料加工", .99943,
                      [[1580, 384], [1718, 384], [1718, 421], [1580, 421]])
    monkeypatch.setattr(process_sense, "_ocr_lines", lambda *a, **k: [caption])
    edge = {"id": "craft_hub_card_facility", "from": "craft_hub", "to": "facility"}
    point = ui_nav.locate_edge(edge, frame, facility="cloth")
    assert point == (1649, 402)
    assert 1580 <= point[0] <= 1718 and 384 <= point[1] <= 421
