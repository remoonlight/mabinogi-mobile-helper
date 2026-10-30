"""Same foreground HWND: observed BitBlt surface is an old dungeon result."""
import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.platform import capture, window
from mm_assist.life_plan.node_inspect import compute_inspect
from types import SimpleNamespace
import time


@pytest.fixture
def scene(monkeypatch):
    def image(name):
        return cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/life_faults_20261023' / name,
                                      dtype=np.uint8), cv2.IMREAD_COLOR)
    stale, rendered = image('dungeon_result.png'), image('wood_rendered.png')
    assert stale is not None and rendered is not None
    grabber = object.__new__(capture.Grabber)
    grabber.method, grabber._last_source = 'auto', None
    calls = []
    def surface(*a, **kw):
        calls.append('bitblt')
        return stale
    def render(*a, **kw):
        calls.append('printwindow')
        return rendered
    monkeypatch.setattr(grabber, '_bitblt_client', surface)
    monkeypatch.setattr(grabber, '_printwindow_client', render)
    monkeypatch.setattr(window, 'is_foreground', lambda _: True)
    win = window.WindowInfo(1, 'game', 0, 0, 1919, 1440)
    return grabber, win, stale, rendered, calls


def test_auto_foreground_frame_preserves_actual_wood_facility_identity(scene):
    grabber, win, stale, rendered, calls = scene
    result = grabber.grab_client(win)
    view = SimpleNamespace(ui_stack_names=['AlteringFacilityPopupUIInstance'],
        ui_stack_ready=True, ui_stack_ts=time.time())
    assert compute_inspect(result, view=view)['node'] == 'facility:wood'
    assert result is rendered and calls == ['printwindow']


@pytest.mark.parametrize('bad', [None, 'black'])
def test_failed_render_does_not_use_old_nonblack_foreground_surface(scene, monkeypatch, bad):
    grabber, win, stale, rendered, calls = scene
    monkeypatch.setattr(grabber, '_printwindow_client',
        lambda _: None if bad is None else np.zeros_like(rendered))
    with pytest.raises(RuntimeError):
        grabber.grab_client(win)
    assert not calls  # the only recording stub left is BitBlt


@pytest.mark.parametrize('fallback', [True, False])
def test_auto_foreground_roi_never_uses_retained_surface(scene, fallback):
    grabber, win, stale, rendered, calls = scene
    result = grabber.grab_client_norm_box(win, (.0, .0, .36, .09), fallback_full=fallback)
    if fallback:
        assert np.array_equal(result, rendered[:int(1440*.09), :int(1919*.36)])
        assert calls == ['printwindow']
    else:
        assert result is None and not calls


def test_explicit_bitblt_configuration_is_unchanged(scene):
    grabber, win, stale, rendered, calls = scene
    grabber.method = 'bitblt'
    assert grabber.grab_client(win) is stale and calls == ['bitblt']
