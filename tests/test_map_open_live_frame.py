"""Map entry must use observed controls, never the stale minimap coordinate."""
import cv2
import numpy as np
import pytest
from mm_assist.config import ROOT, AppConfig
from mm_assist.nav import map_atlas as maps
from mm_assist.platform.window import WindowInfo


def frame():
    img = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/map_open_stale_point.png'))
    assert img is not None
    return img


def test_failed_hud_frame_locates_hotkey_button_without_terrain_template():
    img = frame()
    win = WindowInfo(0, 'fixture', 0, 0, img.shape[1], img.shape[0])
    assert maps._find_map_open_btn_xy(img, win) == (1870, 204)


def test_no_map_control_never_falls_back_to_calibrated_click(monkeypatch):
    img = np.zeros_like(frame())
    win = WindowInfo(0, 'fixture', 0, 0, img.shape[1], img.shape[0])
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **k: (win, img))
    monkeypatch.setattr(maps, '_focus_for_map_open', lambda *a: True)
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr(maps, '_click_map_open', lambda *a, **k: pytest.fail('no observed control, no click'))
    out = maps.open_map(AppConfig(), win, skip_precheck=True)
    assert out.state == 'map_button_missing'


def test_observed_map_control_is_clicked_once_and_checked(monkeypatch):
    img = frame()
    win = WindowInfo(0, 'fixture', 0, 0, img.shape[1], img.shape[0])
    clicks = []
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **k: (win, img))
    monkeypatch.setattr(maps, '_focus_for_map_open', lambda *a: True)
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr(maps, '_click_map_open', lambda win, x, y, **k: clicks.append((x,y,k)) or 'ok')
    monkeypatch.setattr(maps, '_wait_action_effect', lambda *a, **k: (win,img,False,0))
    monkeypatch.setattr(maps, '_snap_map_light', lambda *a, **k: (win,img,[]))
    monkeypatch.setattr(maps, 'snap', lambda *a, **k: (win,img,[],None))
    monkeypatch.setattr(maps, '_map_already_open', lambda *a: None)
    maps.open_map(AppConfig(), win, skip_precheck=True)
    assert clicks == [(1870,204,{'hold_s':.10,'cal':False})]
