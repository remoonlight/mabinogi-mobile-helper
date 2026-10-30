"""Real background client surfaces differ from the game's rendered frame."""
import cv2
import numpy as np
import pytest
from mm_assist.config import ROOT
from mm_assist.platform import window
from mm_assist.platform.capture import Grabber


@pytest.fixture
def scene(monkeypatch):
    folder=ROOT/'tests/fixtures/life_faults_20261020'
    wrong=cv2.imread(str(folder/'capture_occluded.png'))
    rendered=cv2.imread(str(folder/'capture_rendered.png'))
    assert wrong is not None and rendered is not None
    g=object.__new__(Grabber);g.method='auto';g._last_source=None
    calls=[]
    def bitblt(*a,**kw):
        calls.append('bitblt')
        return wrong
    def printwindow(*a,**kw):
        calls.append('printwindow')
        return rendered
    monkeypatch.setattr(g,'_bitblt_client',bitblt)
    monkeypatch.setattr(g,'_printwindow_client',printwindow)
    monkeypatch.setattr(window,'is_foreground',lambda _:False)
    win=window.WindowInfo(1,'test',0,0,1920,1440)
    return g,win,wrong,rendered,calls


def test_background_capture_preserves_real_skill_detail_recognition(scene):
    from test_herb_detail_current import native_view
    from mm_assist.life_plan.node_inspect import compute_inspect
    g,win,wrong,rendered,calls=scene
    assert not np.array_equal(wrong,rendered)
    output=g.grab_client(win)
    assert compute_inspect(output,view=native_view())['node']=='gather_detail:淨淨蘑菇'
    assert output is rendered and calls==['printwindow']


@pytest.mark.parametrize('bad',[None,'black'])
def test_background_render_failure_never_falls_back_to_occluded_surface(scene,monkeypatch,bad):
    g,win,wrong,rendered,calls=scene
    monkeypatch.setattr(g,'_printwindow_client',lambda _:None if bad is None else np.zeros_like(rendered))
    with pytest.raises(RuntimeError):
        g.grab_client(win)
    assert 'bitblt' not in calls


def test_foreground_change_during_render_keeps_rendered_frame(scene,monkeypatch):
    g,win,wrong,rendered,calls=scene
    foreground=[True]
    monkeypatch.setattr(window,'is_foreground',lambda _:foreground[0])
    original=g._printwindow_client
    def render(*a,**kw):
        foreground[0]=False
        return original(*a,**kw)
    monkeypatch.setattr(g,'_printwindow_client',render)
    assert g.grab_client(win) is rendered
    assert not foreground[0] and calls==['printwindow']


@pytest.mark.parametrize('fallback',[True,False])
def test_background_roi_never_uses_visible_surface(scene,fallback):
    g,win,wrong,rendered,calls=scene
    box=(.32,.26,.5,.34)
    out=g.grab_client_norm_box(win,box,fallback_full=fallback)
    if fallback:
        x,y=int(1920*.32),int(1440*.26)
        w,h=int(1920*(.5-.32)),int(1440*(.34-.26))
        assert np.array_equal(out,rendered[y:y+h,x:x+w]) and calls==['printwindow']
    else:
        assert out is None and not calls


def test_explicit_surface_backend_keeps_its_explicit_semantics(scene):
    g,win,wrong,rendered,calls=scene
    g.method='bitblt'
    assert g.grab_client(win) is wrong and calls==['bitblt']
