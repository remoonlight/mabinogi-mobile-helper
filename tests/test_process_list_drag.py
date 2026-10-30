"""Recipe-list paging must drag vertically and inspect the next frame."""
import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import medicine_process as process
from mm_assist.platform.window import WindowInfo


@pytest.fixture
def run_list(monkeypatch):
    image = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/medicine_standalone.png'))
    assert image is not None
    monkeypatch.setattr('mm_assist.platform.window.is_foreground', lambda _: True)
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr('mm_assist.life_execute.ui_act.stack_has_facility_result', lambda: False)
    monkeypatch.setattr(process, '_recipe_detail_open', lambda *a: False)
    monkeypatch.setattr('mm_assist.perception.process_sense.locate_process_button', lambda *a, **kw: {'ok': False})

    def run(*, scale=1, misses=1, next_error=None, cancel=False, drag_ok=True, root=None):
        frame = cv2.resize(image, None, fx=scale, fy=scale)
        win = WindowInfo(0, 'offline', 0, 0, frame.shape[1], frame.shape[0])
        drags, clicks, observations = [], [], []
        cancelled = [False]
        def locate(current, **kwargs):
            observations.append(current)
            if next_error and drags:
                return {'ok': False, 'error': next_error}
            if len(drags) < misses:
                return {'ok': False, 'error': 'not_found'}
            return {'ok': True, 'click_client': [100, 100]}
        def drag(window, x0, y0, x1, y1, **kwargs):
            assert kwargs['end_hold_s'] == .4
            drags.append((x0, y0, x1, y1))
            cancelled[0] = cancel
            return drag_ok
        monkeypatch.setattr('mm_assist.perception.process_sense.locate_process_recipe', locate)
        monkeypatch.setattr('mm_assist.life_execute.ui_act.pump_aborted', lambda: cancelled[0])
        monkeypatch.setattr('mm_assist.safety.input.drag_client', drag)
        monkeypatch.setattr('mm_assist.safety.input.click_client', lambda window, x, y: clicks.append((x, y)) or False)
        result = process.execute_medicine_feed(win=win, recipe='咻咻蘑菇汁液',
            items={'咻咻蘑菇孢子': 100, '裝水的瓶子': 100}, facilities={'medicine': ()},
            grab=lambda: frame.copy(), sleep_fn=lambda _: None, read_stp=lambda: 0, root=root)
        return result, win, drags, clicks, observations
    return run


@pytest.mark.parametrize('scale', [1, .5])
def test_real_feed_path_drags_up_to_page_down_and_relocates(run_list, scale):
    result, win, drags, clicks, observed = run_list(scale=scale)
    assert len(drags) == 1
    x0, y0, x1, y1 = drags[0]
    assert x0 == x1 and y0 > y1
    assert .38 * win.width < x0 < .98 * win.width
    assert .06 * win.height < y1 < y0 < .84 * win.height
    assert len(observed) == 2 and observed[0] is not observed[1]
    assert clicks == [(100, 100)] and not result.ok  # mocked recipe click refused


def test_next_frame_wrong_page_stops_without_second_drag(run_list):
    result, _, drags, clicks, observed = run_list(misses=3, next_error='wrong_facility')
    assert not result.ok and len(drags) == 1 and not clicks
    assert len(observed) == 2


def test_pause_after_drag_stops_before_next_recipe_action(run_list):
    result, _, drags, clicks, _ = run_list(cancel=True)
    assert result.status == 'cancel' and len(drags) == 1 and not clicks


def test_refused_drag_is_not_retried_or_reported_as_paging(run_list):
    result, _, drags, clicks, observed = run_list(misses=3, drag_ok=False)
    assert result.status == 'drag_rejected' and len(drags) == 1
    assert not clicks and len(observed) == 1


def test_list_never_drags_more_than_existing_two_action_limit(run_list):
    result, _, drags, clicks, observed = run_list(misses=3)
    assert not result.ok and len(drags) == 2 and not clicks
    assert len(observed) == 3


def test_config_preserves_original_layout_source_and_has_no_missing_config_fallback(tmp_path):
    import hashlib
    import json
    image_path = ROOT / 'tests/fixtures/process_stance/medicine_standalone.png'
    frame = cv2.imread(str(image_path))
    spec = json.loads((ROOT / process.PROCESS_NAVIGATION_CONFIG).read_text(encoding='utf-8'))['recipe_list_down']
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == spec['source_sha256']
    assert [frame.shape[1], frame.shape[0]] == spec['source_dimensions']
    with pytest.raises(FileNotFoundError):
        process._recipe_list_down_points(frame, root=tmp_path)
