"""Maps register as listeners rather than stacking players; reject retained objects."""
import time
from types import SimpleNamespace as NS

import pytest

from mm_assist.life_execute import ui_logic
from mm_assist.life_plan import node_inspect
from mm_assist.memory.readers import ui_stack as stack


@pytest.fixture
def memory(monkeypatch):
    reader = stack.Il2CppUiStackReader()
    reader._h, reader._ga = object(), 0x90000
    ptrs = {0x900b8:0xa0000, 0xa0050:0x10000, 0x10020:0x20000,
            0x20010:0x30000, 0x30020:0x50000, 0x50088:0x60000,
            0x60010:0x70000, 0x50080:0x80000}
    values = {0x20018:34, 0x2001c:1, 0x30018:2, 0x500f8:1, 0x500f9:1, 0x70020:0}
    names = {0x10000:'UIStackingService', 0x50000:'MapUIInstance',
             0x60000:'MapResourceScope', 0x70000:'UIResourceScope', 0x80000:'HudUIStepState'}
    monkeypatch.setattr(stack, 'resolve_klass', lambda *a:0x90000)
    monkeypatch.setattr(stack, 'read_ptr', lambda h,p:ptrs.get(p))
    monkeypatch.setattr(stack, 'read_u32', lambda h,p:values.get(p))
    monkeypatch.setattr(stack, 'read_u8', lambda h,p:values.get(p))
    monkeypatch.setattr(stack, 'klass_simple_name', lambda h,p,o:names.get(p,''))
    yield reader, ptrs, values, names
    reader._h = None  # Mock handle never reaches Windows cleanup.


def test_live_map_listener_needs_current_registration_and_live_resource(memory):
    reader, _, values, _ = memory
    assert reader._active_map_listener(0x10000) == (0x50000,0x60000,0x70000)
    values[0x500f9] = 0
    assert reader._active_map_listener(0x10000) is None
    values[0x500f9], values[0x500f8] = 1,0
    assert reader._active_map_listener(0x10000) is None
    values[0x500f8], values[0x70020] = 1,1
    assert reader._active_map_listener(0x10000) is None


@pytest.mark.parametrize('fault', ['root', 'scope', 'resource', 'hud_step', 'list',
                                  'size', 'duplicate', 'wrong_type', 'version_change',
                                  'resource_change', 'unregister_during_read'])
def test_map_listener_rejects_invalid_or_changing_associations(memory, monkeypatch, fault):
    reader, ptrs, values, names = memory
    if fault == 'root': ptrs[0xa0050] = 0xb0000
    if fault == 'scope': ptrs[0x50088] = 0
    if fault == 'resource': ptrs[0x60010] = 0
    if fault == 'hud_step': ptrs[0x50080] = 0
    if fault == 'list': ptrs[0x20010] = 0
    if fault == 'size': values[0x2001c] = 3
    if fault == 'duplicate':
        values[0x2001c] = 2
        ptrs[0x30028] = 0x51000
        names[0x51000] = 'MapUIInstance'
    if fault == 'wrong_type': names[0x70000] = 'UnknownResource'
    address = {'version_change':0x20018, 'resource_change':0x60010,
               'unregister_during_read':0x500f9}.get(fault)
    if address:
        reads = 0
        def changing(h,p):
            nonlocal reads
            if p == address:
                reads += 1
                if reads > 1:
                    return 0
            return (ptrs if fault == 'resource_change' else values).get(p)
        monkeypatch.setattr(stack, 'read_ptr' if fault == 'resource_change' else
                            'read_u32' if fault == 'version_change' else 'read_u8', changing)
    assert reader._active_map_listener(0x10000) is None


def test_reader_to_navigation_uses_listener_map_without_image_or_ocr(memory, monkeypatch):
    reader, _, _, _ = memory
    monkeypatch.setattr(reader, '_ensure', lambda:True)
    reader._state_stacks = [0x12000]
    monkeypatch.setattr(reader, '_ensure_state_stacks', lambda:None)
    monkeypatch.setattr(reader, '_names_from_state_lists', lambda:(['TutorialTooltipUIState'],None))
    monkeypatch.setattr(reader, '_players_for_read', lambda *a,**k:([],0x10000))
    for method in ('_tmpl','_ocr_blob','_inspect_map_how'):
        monkeypatch.setattr(node_inspect, method,
                            lambda *a,**kw:pytest.fail('active map needs no visual recognition'))
    snap = reader.read(force=True)
    assert 'MapUIInstance' in snap.names and snap.map_identity == (0x50000,0x60000,0x70000)
    where = ui_logic.resolve_navigation(object(), reader=reader)
    assert where.node == 'map' and where.stack_ready
    view = NS(ui_stack_names=snap.names, ui_stack_ready=True, ui_stack_ts=time.time(),
              map_identity=snap.map_identity)
    result = node_inspect.compute_inspect(None, view=view)
    assert result['node'] == 'map' and result['source'] == 'stack+mem'
    assert result['cost']['tpl_n'] == result['cost']['ocr_n'] == 0
    assert not node_inspect.inspect_needs_grab('map', view)


@pytest.mark.parametrize('cover', ['SideNavigationUIInstance','NewProfileUIInstance',
                                 'ClickShieldUIState','UnknownPopup'])
def test_map_listener_never_promotes_covering_stack(memory, monkeypatch, cover):
    reader, _, _, _ = memory
    monkeypatch.setattr(reader, '_ensure', lambda:True)
    reader._state_stacks = [0x12000]
    monkeypatch.setattr(reader, '_ensure_state_stacks', lambda:None)
    monkeypatch.setattr(reader, '_names_from_state_lists', lambda:([cover],None))
    monkeypatch.setattr(reader, '_players_for_read', lambda *a,**k:([],0x10000))
    snap = reader.read(force=True)
    assert 'MapUIInstance' not in snap.names and snap.map_identity is None


@pytest.mark.parametrize('fault', ['stale','future','unready','reused','invalid','covered'])
def test_invalid_map_fact_keeps_existing_visual_path(fault):
    view = NS(ui_stack_names=['MapUIInstance'], ui_stack_ready=True,
              ui_stack_ts=time.time(), map_identity=(0x50000,0x60000,0x70000))
    if fault == 'stale': view.ui_stack_ts -= 10
    if fault == 'future': view.ui_stack_ts += 10
    if fault == 'unready': view.ui_stack_ready = False
    if fault == 'reused': view.ui_stack_error = 'stack_read_reused'
    if fault == 'invalid': view.map_identity = (0,1,2)
    if fault == 'covered': view.ui_stack_names += ['UnknownPopup']
    assert not node_inspect._map_node_from_view(view)
    assert node_inspect.inspect_needs_grab('map',view)


def test_map_fact_clears_on_close_or_unready_stack():
    from mm_assist.memory.readers.scene import MemSceneInfo
    from mm_assist.perception.scene import apply_stack_bits
    from mm_assist.perception.state_ui import build_state_ui_payload
    ms = MemSceneInfo()
    live = stack.UiStackSnap(names=['MapUIInstance'], ready=True,
        map_identity=(0x50000,0x60000,0x70000), sampled_ts=time.time())
    apply_stack_bits(ms,live)
    assert build_state_ui_payload(NS(),ms)['map_identity'] == live.map_identity
    for closed in (stack.UiStackSnap(names=['TutorialTooltipUIState'],ready=True),
                   stack.UiStackSnap(ready=False),
                   stack.UiStackSnap(names=live.names,ready=True,error='stack_read_reused')):
        apply_stack_bits(ms,closed)
        assert ms.map_identity is None


@pytest.mark.parametrize('map_type,node', [(None,'map'),(1,'map'),(2,'map'),(3,'map')])
def test_listener_never_upgrades_generic_map_enum_to_specific_region(map_type,node,monkeypatch):
    monkeypatch.setattr(node_inspect,'_inspect_map_how',lambda *a:pytest.fail('no visual map refinement'))
    view = NS(ui_stack_names=['MapUIInstance'],ui_stack_ready=True,ui_stack_ts=time.time(),
              map_identity=(0x50000,0x60000,0x70000),map_type=map_type)
    result = node_inspect.compute_inspect(object(),view=view)
    assert result['node'] == node and result['source'] == 'stack+mem' and result['map_mem']
    assert node_inspect._stack_sn_from_mem(view) == node
    assert ui_logic.node_from_mem(view)[0] == node


def test_navigation_rejects_map_closed_or_reopened_during_read():
    def snap(identity):
        return stack.UiStackSnap(names=['MapUIInstance'],ready=True,map_identity=identity,sampled_ts=time.time())
    for identity in (None,(0x51000,0x60000,0x70000)):
        samples = iter([snap((0x50000,0x60000,0x70000)),snap(identity)])
        where = ui_logic.resolve_navigation(object(),reader=NS(read=lambda **kw:next(samples)))
        assert where.source == 'direct_transition' and where.node == 'unknown'


def test_older_image_job_cannot_overwrite_current_map_memory_tick():
    live = {'node':'map','map_mem':True,'stack_sampled_ts':100,
            'stack_names':['MapUIInstance'],'stack_ready':True,'source':'stack+mem'}
    old = {'node':'main_hud','stack_sampled_ts':99,'stack_names':[],
           'stack_ready':True,'source':'stack+tpl'}
    held, _ = node_inspect.hold_inspect_payload(live,old,hud_streak=1)
    assert held == live
    newer = dict(old,stack_sampled_ts=101,hud_menu_icon=True)
    held, streak = node_inspect.hold_inspect_payload(live,newer,hud_streak=0)
    assert held['node'] == 'map' and streak == 1
    held, _ = node_inspect.hold_inspect_payload(held,dict(newer,stack_sampled_ts=102),hud_streak=streak)
    assert held['node'] == 'main_hud'


def test_current_memory_map_wins_over_retained_hud_display(monkeypatch):
    view = NS(ui_stack_names=['MapUIInstance'],ui_stack_ready=True,ui_stack_ts=time.time(),
              map_identity=(0x50000,0x60000,0x70000),raw={})
    monkeypatch.setattr(ui_logic,'read_inspect_page',lambda **k:'main_hud')
    monkeypatch.setattr(ui_logic,'read_inspect_record',lambda **k:{
        'node':'main_hud','zh':'主界面-待机','stack_names':[]})
    assert ui_logic.resolve_ui_node(view=view).node == 'map'
