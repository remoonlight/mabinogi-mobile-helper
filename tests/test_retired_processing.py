"""Retired processing has no entry; pre-existing slots remain occupied."""
from importlib.util import find_spec
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_plan import planner
from mm_assist.life_plan.elixir_reserve import MOUNTAIN_NAME
from mm_assist.life_plan.live_test import scenarios
from mm_assist.life_plan.types import LifeSnapshot, PlanItem, QueueSlot

RETIRED = ('魔力起爆劑','蘊含魔力的石頭','魔力藥草','閃閃發亮的苔蘚')


def stock():
    return {'合金鋼錠':100,'高級木材':100,'高級布料':100,MOUNTAIN_NAME:0,
        '淨淨蘑菇孢子':10000,'裝水的瓶子':10000}


def test_retired_routes_and_registered_inputs_are_absent():
    from mm_assist.perception.process_sense import recipes_for_facility
    for name in ('life_plan.detonator','life_execute.detonator_obtain','perception.detonator_sense'):
        assert find_spec('mm_assist.'+name) is None
    assert not any(name in key for key in scenarios() for name in RETIRED)
    assert not any(row['name'] in RETIRED for row in recipes_for_facility('medicine'))


@pytest.mark.parametrize('name',RETIRED)
def test_old_action_requests_cannot_send_input(name):
    from mm_assist.life_plan.execute import try_execute
    clicks=[]
    kind='feed' if name==RETIRED[0] else 'gather'
    result=try_execute(PlanItem('old',kind,name,facility='medicine',recipe=name,
        skill='採集藥草',detail=name),
        items=dict.fromkeys(RETIRED,10000),do_click=lambda *xy:clicks.append(xy))
    assert not result.ok and not result.action_issued and not clicks


@pytest.mark.parametrize('done',[False,True])
def test_existing_unregistered_slots_keep_capacity_and_claim_threshold(done):
    plan=planner.plan_life(LifeSnapshot(ok=True,items=stock(),
        facilities={'medicine':(QueueSlot(RETIRED[0],done),)*2}))
    assert not any(p.kind=='claim' for p in plan)
    assert [p.recipe for p in plan if p.kind=='feed']==['淨淨蘑菇汁液']*3


def test_three_completed_legacy_slots_can_be_claimed_before_filling():
    plan=planner.plan_life(LifeSnapshot(ok=True,items=stock(),
        facilities={'medicine':(QueueSlot(RETIRED[0],True),)*3}))
    assert plan[0].kind=='claim'
    assert [p.recipe for p in plan if p.kind=='feed']==['淨淨蘑菇汁液']*5


def test_unscanned_legacy_feeds_remain_occupied_using_archived_recipe_data():
    plan=planner.plan_life(LifeSnapshot(ok=True,items=stock()),reserved_feeds={RETIRED[0]:2})
    assert [p.recipe for p in plan if p.kind=='feed']==['淨淨蘑菇汁液']*3


def test_registered_unscanned_feeds_and_other_recipes_are_not_overbooked():
    plan=planner.plan_life(LifeSnapshot(ok=True,items=stock(),
        facilities={'medicine':(QueueSlot('淨淨蘑菇汁液',False),)*2}),
        reserved_feeds={'淨淨蘑菇孢子':2})
    assert [p.recipe for p in plan if p.kind=='feed']==['淨淨蘑菇汁液']


def test_retained_native_pixels_no_longer_register_a_recipe_or_input_target(monkeypatch):
    from mm_assist.life_plan import node_inspect
    from mm_assist.perception.process_sense import locate_process_recipe
    registration=json.loads((ROOT/'knowledge/templates/detonator/catalog.json').read_text('utf-8'))
    frame=cv2.imdecode(np.frombuffer((ROOT/'tests/fixtures/detonator/recipe.png').read_bytes(),np.uint8),1)
    view=SimpleNamespace(ui_stack_names=registration['sources']['recipe']['stack']['names'],
        ui_stack_ready=True,ui_stack_ts=time.time(),ui_stack_error='',fresh=True,pid_alive=True)
    monkeypatch.setattr(node_inspect,'_ocr_blob',lambda *a,**kw:RETIRED[0])
    node=node_inspect.compute_inspect(frame,view=view)['node']
    assert not any(name in node for name in RETIRED)
    assert not locate_process_recipe(frame,facility='medicine',name=RETIRED[0])['ok']
