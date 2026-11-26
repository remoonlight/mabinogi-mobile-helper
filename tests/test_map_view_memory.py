"""Active displayed map, independent of player location and breadcrumb font size."""
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from mm_assist.life_plan import node_inspect as ni
from mm_assist.life_execute import ui_logic
from mm_assist.memory.readers import ui_stack as stack
from test_map_memory import memory


@pytest.mark.parametrize('label,node,title', [('current','map_dunbarton','杜巴頓'),
    ('continent','map_continent','歐拉大陸'),('laert','map','萊爾特丘陵')])
def test_recorded_selected_map_uses_loaded_resource_not_retained_title(label,node,title):
    row = json.loads((Path(__file__).parent/'fixtures/map_memory'/f'{label}_20261013.json').read_text(encoding='utf-8'))
    before, after = row['before'],row['after']
    assert before['viewing'] == after['viewing'] and before['resource_name'] == after['resource_name']
    fact = {**before['viewing'],'resource_name':before['resource_name'],
            'titles':{k:v['raw'] for k,v in before['titles'].items()}}
    view = NS(ui_stack_names=['MapUIInstance'],ui_stack_ready=True,ui_stack_ts=time.time(),
        map_identity=before['map_identity'],map_view=fact,map_type=2,geo='杜巴頓')
    result = ni.compute_inspect(None,view=view)
    assert result['node'] == node
    assert result['map_view']['selected_title'] == title
    assert result['cost']['tpl_n'] == result['cost']['ocr_n'] == 0
    assert ui_logic.node_from_mem(view)[0] == node
    assert ni._stack_sn_from_mem(view) == node


def test_generic_map_enum_never_identifies_a_specific_region():
    for value in (None,1,2,3,4,'bad'):
        assert ni.map_node_from_type(value) == 'map'


@pytest.fixture
def mapped(memory,monkeypatch):
    reader,ptrs,values,names = memory
    # Retained 2026-10-13 layout; current configuration follows the newer client.
    reader._off.update({'GameStageEntityComponents.CurrentViewingMapUIComponent':0x6a8,
        'MapImageUI.MapModel.Meta':0x70,'MapButtonUI.UISlot':0x40,
        'MapButtonUISlot.TitleWorldText':0x58,'MapButtonUISlot.TitleTownText':0x60,
        'MapButtonUISlot.TitleDungeonText':0x68})
    reader._off.pop('MarqueeTextLabelSlot.SourceText', None)
    ptrs.update({0x50010:0xb0000,0xb0010:0xc0000,0xc0018:0xd0000,
        0xd06a8:0xe0000,0x50040:0xf0000,0xf0070:0x110000,
        0x110010:0x120000,0x50058:0x130000,0x130040:0x140000,
        0x140058:0x150000,0x140060:0x160000,0x140068:0x170000,
        0x1500e0:0x180000,0x1600e0:0x190000,0x1700e0:0x1a0000})
    names.update({0xb0000:'UIInstanceContext`1',0xc0000:'MapUIInstanceComponentBuilder',
        0xd0000:'GameStageEntityComponents',0xe0000:'CurrentViewingMapUIComponent',
        0xf0000:'MapImageUI',0x110000:'MapModelMeta',0x130000:'MapButtonUI',
        0x140000:'MapButtonUISlot',0x150000:'UIText',0x160000:'UIText',0x170000:'UIText'})
    values.update({0xf0020:0,**{0xe0000+o:1 for o in range(0x20,0x40,4)}})
    strings = {0x120000:'WorldMap@v1_5',0x180000:'<custom=0/>歐拉大陸',
        0x190000:'<noheight><size=24>杜巴頓</size></noheight>',0x1a0000:''}
    monkeypatch.setattr(stack,'read_il2cpp_string',lambda h,p,**kw:strings.get(p))
    return reader,ptrs,values,names,strings


@pytest.fixture
def marquee(mapped):
    reader,ptrs,values,names,strings = mapped
    reader._off.update({'GameStageEntityComponents.CurrentViewingMapUIComponent':0x6c0,
        'MapImageUI.MapModel.Meta':0x78,'MapButtonUI.UISlot':0x38,
        'MapButtonUISlot.WorldMapMarqueeLabel':0x38,'MapButtonUISlot.TownMapMarqueeLabel':0x40,
        'MapButtonUISlot.DungeonMapMarqueeLabel':0x48,'MarqueeTextLabelSlot.SourceText':0x48})
    for address in (0x140058,0x140060,0x140068):
        ptrs.pop(address, None)
    for old,new in [(0xd06a8,0xd06c0),(0xf0070,0xf0078),(0x130040,0x130038)]:
        ptrs[new] = ptrs.pop(old)
    for offset, label, text in [(0x38,0x210000,0x150000),(0x40,0x220000,0x160000),(0x48,0x230000,0x170000)]:
        ptrs[0x140000+offset], ptrs[label+0x48] = label, text
        names[label] = 'MarqueeTextLabelSlot'
    return mapped


def test_current_marquee_map_title_preserves_independent_resource_identity(marquee):
    reader,*_ = marquee
    fact = reader._active_map_view(0x10000,(0x50000,0x60000,0x70000))
    assert fact['resource_name'] == 'WorldMap@v1_5'
    assert stack.selected_map_view(fact)['node'] == 'map_continent'
    assert '杜巴頓' in fact['titles']['town']


@pytest.mark.parametrize('fault', ['missing_label','wrong_label','missing_text','wrong_text','changed_label'])
def test_current_marquee_map_title_rejects_missing_wrong_or_changing_links(marquee,monkeypatch,fault):
    reader,ptrs,values,names,strings = marquee
    if fault == 'missing_label': ptrs[0x140038] = 0
    if fault == 'wrong_label': names[0x210000] = 'OtherSlot'
    if fault == 'missing_text': ptrs[0x210048] = 0
    if fault == 'wrong_text': names[0x150000] = 'OtherText'
    if fault == 'changed_label':
        original = stack.read_ptr
        calls = [0]
        def changed(h,p):
            if p == 0x140038:
                calls[0] += 1
                if calls[0] > 1: return 0
            return original(h,p)
        monkeypatch.setattr(stack,'read_ptr',changed)
    assert reader._active_map_view(0x10000,(0x50000,0x60000,0x70000)) is None


def test_map_view_reads_active_association_and_resource(mapped):
    reader,*_ = mapped
    fact = reader._active_map_view(0x10000,(0x50000,0x60000,0x70000))
    assert fact['map_type'] == 1 and fact['resource_name'] == 'WorldMap@v1_5'


def test_real_reader_scene_payload_and_navigation_use_active_map_view(mapped,monkeypatch):
    from mm_assist.memory.readers.scene import MemSceneInfo
    from mm_assist.perception.scene import apply_stack_bits
    from mm_assist.perception.state_ui import build_state_ui_payload
    reader,*_ = mapped
    monkeypatch.setattr(reader,'_ensure',lambda:True)
    reader._state_stacks = [0x12000]
    monkeypatch.setattr(reader,'_ensure_state_stacks',lambda:None)
    monkeypatch.setattr(reader,'_names_from_state_lists',lambda:(['TutorialTooltipUIState'],None))
    monkeypatch.setattr(reader,'_players_for_read',lambda *a,**kw:([],0x10000))
    snap = reader.read(force=True)
    ms = MemSceneInfo()
    apply_stack_bits(ms,snap)
    view = NS(raw=build_state_ui_payload(NS(),ms))
    result = ni.compute_inspect(None,view=view)
    assert result['node'] == 'map_continent' and result['map_view']['selected_title'] == '歐拉大陸'
    assert not ni.inspect_needs_grab('map',view)
    assert ui_logic.resolve_navigation(object(),reader=reader).node == 'map_continent'


@pytest.mark.parametrize('kind,resource,title',[(1,'OtherWorld','歐拉大陸'),
    (1,'WorldMap@v1_5','其他大陸'),(2,'Gairech','杜巴頓'),(2,'Dunbarton','萊爾特丘陵')])
def test_resource_type_and_title_must_agree_for_specific_map(kind,resource,title):
    fact = {'map_type':kind,'resource_name':resource,'titles':{'world':title,'town':title}}
    assert stack.selected_map_view(fact)['node'] == 'map'


@pytest.mark.parametrize('fault',['loading','bad_class','null','resource_change','type_change','closed'])
def test_map_view_rejects_loading_and_concurrent_changes(mapped,monkeypatch,fault):
    reader,ptrs,values,names,strings = mapped
    if fault == 'loading': values[0xf0020] = 1
    if fault == 'bad_class': names[0xd0000] = 'OldGameStage'
    if fault == 'null': ptrs[0xd06a8] = 0
    address = {'resource_change':0xf0070,'type_change':0xe0020,'closed':0x500f9}.get(fault)
    if address:
        count = 0
        def changing(h,p):
            nonlocal count
            if p == address:
                count += 1
                if count > 1: return 0
            return (ptrs if fault == 'resource_change' else values).get(p)
        monkeypatch.setattr(stack,'read_ptr' if fault == 'resource_change' else
            'read_u32' if fault == 'type_change' else 'read_u8',changing)
    assert reader._active_map_view(0x10000,(0x50000,0x60000,0x70000)) is None


def test_scene_payload_clears_map_view_with_closed_stack():
    from mm_assist.memory.readers.scene import MemSceneInfo
    from mm_assist.perception.scene import apply_stack_bits
    from mm_assist.perception.state_ui import build_state_ui_payload
    ms = MemSceneInfo()
    fact = {'map_type':2,'resource_name':'Gairech'}
    apply_stack_bits(ms,stack.UiStackSnap(names=['MapUIInstance'],ready=True,
        map_identity=(0x50000,0x60000,0x70000),map_view=fact,sampled_ts=time.time()))
    assert build_state_ui_payload(NS(),ms)['map_view'] == fact
    apply_stack_bits(ms,stack.UiStackSnap(ready=False))
    assert ms.map_view is None


def test_navigation_rejects_same_instance_map_switch():
    identity = (0x50000,0x60000,0x70000)
    samples = iter(stack.UiStackSnap(names=['MapUIInstance'],ready=True,map_identity=identity,
        map_view=fact,sampled_ts=time.time()) for fact in (
            {'map_type':1,'resource_name':'WorldMap@v1_5','titles':{'world':'歐拉大陸'}},
            {'map_type':2,'resource_name':'Gairech','titles':{'town':'萊爾特丘陵'}}))
    where = ui_logic.resolve_navigation(object(),reader=NS(read=lambda **kw:next(samples)))
    assert where.node == 'unknown' and where.source == 'direct_transition'
