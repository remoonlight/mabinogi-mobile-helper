"""Actual items layer must back out before exposing the hidden process tab."""
from types import SimpleNamespace as NS
import time
import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_plan import node_inspect as inspect
from mm_assist.life_execute import ui_logic,ui_nav
from mm_assist.memory.readers.ui_stack import UiStackSnap
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

NAMES=['CraftingHubPopupUIInstance','CraftingFacilityPopupUIInstance']


def pixels():
    return cv2.imread(str(ROOT/'tests/fixtures/process_stance/craft_process_tab_failed_20261014.png'))


def view(**extra):
    values=dict(ui_stack_names=NAMES,ui_stack_ready=True,ui_stack_ts=time.time(),ui_stack_error='')
    values.update(extra)
    return NS(**values)


def test_retained_failure_is_items_layer_not_a_shell_with_clickable_process_tab(monkeypatch):
    monkeypatch.setattr('mm_assist.perception.ocr.ocr_roi',lambda *a,**k:pytest.fail('pixel identity only'))
    row=inspect.compute_inspect(pixels(),view=view())
    assert row['node']=='craft_items'
    assert '道具' in inspect.format_node_zh(row['node'])
    def snap():return UiStackSnap(names=NAMES,ready=True,sampled_ts=time.time())
    where=ui_logic.resolve_navigation(pixels(),reader=NS(read=lambda **kw:snap()))
    assert where.node=='craft_items' and where.stack_ready
    assert inspect.craft_tab_xy(pixels(),'加工') is None
    assert inspect.locate_spot('craft_items','返回',pixels())==(60,57)


@pytest.mark.parametrize('extra',[{'ui_stack_ready':False},{'ui_stack_ts':0},
    {'ui_stack_ts':time.time()-10},{'ui_stack_error':'stack_read_reused'},
    {'ui_stack_names':['CraftingHubPopupUIInstance']},
    {'ui_stack_names':NAMES+['DetailCardUIInstance']},
    {'ui_stack_names':NAMES+['UnknownPopup']}])
def test_items_pixels_do_not_override_unknown_stale_or_covered_stack(extra):
    assert inspect.compute_inspect(pixels(),view=view(**extra))['node']!='craft_items'


def test_items_exit_path_verifies_make_then_clicks_process(monkeypatch):
    from mm_assist.life_execute.ui_act import prepare_to
    reset_stage_gates_for_tests();ui_nav._EDGE_MEMO.clear();ui_nav._PENDING_EDGES.clear()
    items=pixels()
    make=cv2.imread(str(ROOT/'tests/fixtures/process_stance/craft_make_20261014.png'))
    process=cv2.imread(str(ROOT/'tests/fixtures/process_stance/craft_process_gate_20261005.png'))
    assert make is not None
    current=[0];clicks=[]
    def resolve(frame):
        if current[0]==0:
            snap=lambda:UiStackSnap(names=NAMES,ready=True,sampled_ts=time.time())
            return ui_logic.resolve_navigation(frame,reader=NS(read=lambda **kw:snap()))
        return ui_logic.UiWhere('craft_make' if current[0]==1 else 'craft_hub',1,0,0,
            'fixture',stack_names=('CraftingHubPopupUIInstance',),stack_ready=True,
            stack_sampled_ts=time.time())
    def click(x,y):
        clicks.append((x,y));current[0]+=1;return True
    monkeypatch.setattr('mm_assist.life_execute.ui_act.pump_aborted',lambda:False)
    try:
        result=prepare_to('craft_hub',grab=lambda:[items,make,process][min(2,current[0])],
            resolve=resolve,click=click,sleep_fn=lambda _:None)
        assert result.ok and len(clicks)==2
        assert clicks[0][0]<100 and clicks[0][1]<100
        assert 580<clicks[1][1]<740
    finally:
        reset_stage_gates_for_tests();ui_nav._EDGE_MEMO.clear();ui_nav._PENDING_EDGES.clear()


def test_stack_change_after_items_pixels_is_rejected():
    snaps=iter([UiStackSnap(names=NAMES,ready=True,sampled_ts=time.time()),
        UiStackSnap(names=['CraftingHubPopupUIInstance'],ready=True,sampled_ts=time.time())])
    where=ui_logic.resolve_navigation(pixels(),reader=NS(read=lambda **kw:next(snaps)))
    assert where.node=='unknown' and where.source=='direct_transition'
