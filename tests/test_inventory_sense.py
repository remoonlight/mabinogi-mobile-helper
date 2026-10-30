import cv2
import numpy as np

from mm_assist.config import ROOT
from mm_assist.perception.inventory_sense import locate_props_tab


def test_original_equipment_frame_locates_props_tab():
    frame = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/sap_inventory_equipment.png'))
    assert locate_props_tab(frame) == (1434, 1381)


def test_missing_props_label_has_no_coordinate_fallback():
    assert locate_props_tab(np.zeros((1442, 1912, 3), np.uint8)) is None
    assert locate_props_tab(None) is None


def test_selected_props_tab_is_independently_recognized():
    from mm_assist.perception.inventory_sense import props_tab_selected

    selected = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/sap_inventory_props_selected.png'))
    equipment = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/sap_inventory_equipment.png'))
    assert props_tab_selected(selected)
    assert not props_tab_selected(equipment)
    assert not props_tab_selected(np.zeros_like(selected))
