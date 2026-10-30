"""The real inventory icon must fit inside the configured search region."""
import cv2
from mm_assist.config import ROOT
from mm_assist.life_execute import ui_nav


def test_actual_menu_inventory_icon_is_not_clipped():
    frame = cv2.imread(str(ROOT/'tests/fixtures/process_stance/menu_bag_clipped.png'))
    edge = ui_nav.path_edges('system_menu','inventory')[0]
    xy = ui_nav.locate_edge(edge,frame)
    assert xy is not None and abs(xy[0]-1446) < 3 and abs(xy[1]-465) < 3


def test_explicit_invalid_inventory_roi_does_not_fall_back():
    frame = cv2.imread(str(ROOT/'tests/fixtures/process_stance/menu_bag_clipped.png'))
    edge = dict(ui_nav.path_edges('system_menu','inventory')[0])
    edge['trigger'] = {'locator':{'roi':[0,0,0,0]}}
    assert ui_nav.locate_edge(edge,frame) is None
