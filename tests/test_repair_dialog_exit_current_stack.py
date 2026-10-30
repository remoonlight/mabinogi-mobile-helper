"""Actual post-repair HUD must agree with the current stack, never a legacy flag."""
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import AppConfig
from mm_assist.life_execute import repair_dialog as dialog
from mm_assist.memory.readers import ui_stack
from mm_assist.platform.window import WindowInfo


@pytest.mark.parametrize('mode,expected', [('hud',True),('late',True),('dialog',False),
    ('stale',False),('unready',False),('blocked',False)])
def test_real_failed_hud_uses_current_independent_stack(monkeypatch,mode,expected):
    frame=cv2.imdecode(np.fromfile(Path(__file__).parent/'fixtures/repair/leave_dialog_hud_20261014.png',np.uint8),1)
    win=WindowInfo(0,'offline',0,0,frame.shape[1],frame.shape[0])
    reads=0
    def read(**kwargs):
        nonlocal reads
        reads+=1
        names=['TutorialTooltipUIState']
        if mode=='dialog' or (mode=='late' and reads <= 2): names+=['DialogueInteractionUIState']
        if mode=='blocked': names+=['ClickShieldUIState']
        return ui_stack.UiStackSnap(names=names,ready=mode!='unready',
            sampled_ts=time.time()-(10 if mode=='stale' else 0))
    monkeypatch.setattr(ui_stack,'shared_ui_stack_reader',lambda:SimpleNamespace(read=read))
    monkeypatch.setattr(dialog,'_grab_frame',lambda *a:(win,frame))
    monkeypatch.setattr(dialog,'_peek_dialog_active',lambda:True)
    monkeypatch.setattr(dialog,'click_client',lambda *a,**kw:pytest.fail('post-repair verification is read-only'))
    monkeypatch.setattr(dialog.time,'sleep',lambda _:None)
    assert dialog._wait_leave_dialog(AppConfig(),win)[1] is expected
