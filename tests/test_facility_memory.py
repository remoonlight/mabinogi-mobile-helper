"""Active facility widgets replace repeated OCR without relaxing claim gates."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from mm_assist.perception import process_sense as sense

FIXTURES=Path(__file__).parent/'fixtures/facility_memory'
HISTORICAL_NAMES=dict(zip(
    (115861,115862,115863,108184,106130,108159,115865,115866,108182,115867,
     108179,108149,108150,115868,108083,108152,108151,115869,115870,115871),
    ('uiTitle','backButtonClick','canGoBack','scrollToRecipeIndex','currencyGroup',
     'filterButtonGroup','onSearchButtonClicked','isSearchApplied','currentFacilityName',
     'facilityDescription','facilityImagePath','isUpgradeVisible','isUpgradable','isGetAll',
     'upgradeButton','moveToFacilityButton','isMoveToFacilityVisible','getAllButton',
     'processingQueue','itemRecipeList')))

def sample(name='metal_away'):
    row=json.loads((FIXTURES/(name+'_20261013.json')).read_text(encoding='utf-8'))
    fact=row['before']['facilities'][0]
    fact['render']['root_attributes']=fact['items']
    for item in fact['items']:
        item['name']=HISTORICAL_NAMES[item['name_id']]
    return fact

def view():
    from mm_assist.memory.readers.facility_ui import select_facility_view
    row=sample()
    return select_facility_view(row['render'],row['facility_type'])

@pytest.mark.parametrize('kind,point', [('go_equip',[223,305]),('claim_all',[187,1087])])
def test_live_widgets_locate_without_templates_or_ocr(monkeypatch,kind,point):
    def forbidden(*a,**k):pytest.fail('memory widget must not run image/OCR')
    monkeypatch.setattr(sense,'_ocr_lines',forbidden)
    monkeypatch.setattr(sense,'match_template',forbidden)
    fn=sense.locate_go_equip if kind=='go_equip' else sense.locate_claim_all
    hit=fn(np.zeros((1440,1920,3),np.uint8),memory_view=view(),facility='metal')
    assert hit['ok'] and hit['src']=='memory'
    assert hit['click_client']==point

def test_actual_active_window_supplies_facility_identity_and_two_widgets():
    fact=view()
    assert fact['facility']=='metal'
    assert fact['buttons']['claim_all']['usable'] is True
    assert fact['buttons']['go_equip']['present'] is True

def test_actual_wood_facility_does_not_borrow_metal_identity(monkeypatch):
    from mm_assist.memory.readers.facility_ui import select_facility_view
    row=sample('wood_away');fact=select_facility_view(row['render'],row['facility_type'])
    assert fact['facility']=='wood'
    frame=np.zeros((1440,1920,3),np.uint8)
    monkeypatch.setattr(sense,'_ocr_lines',lambda *a,**k:pytest.fail('OCR ran'))
    assert sense.locate_go_equip(frame,memory_view=fact,facility='wood')['ok']
    assert not sense.locate_claim_all(frame,memory_view=fact,facility='wood')['ok']
    assert not sense.locate_go_equip(frame,memory_view=fact,facility='metal')['ok']

@pytest.mark.parametrize('change', ['disposed','dirty','wrong_window','cycle','duplicate','wrong_title','closing','destroyed'])
def test_malformed_or_changing_widget_does_not_authorize_memory_click(change):
    from mm_assist.memory.readers.facility_ui import select_facility_view
    row=sample();r=row['render'];elements=r['elements']
    text=next(e for e in elements if e.get('text')=='前往設備')
    if change=='disposed':text['disposed']=1
    if change=='dirty':text['layout_dirty']=1
    if change=='wrong_window':text['window']+=1
    if change=='cycle':text['parent']=text['ptr']
    if change=='duplicate':elements.append(copy.deepcopy(text))
    if change=='wrong_title':next(e for e in elements if e.get('text')=='金屬加工')['text']='木材加工'
    if change=='closing':r['closing']=1
    if change=='destroyed':r['resource_destroyed']=1
    assert select_facility_view(r,row['facility_type']) is None

@pytest.mark.parametrize('mutation', ['disabled','not_interactable','can_execute_false','can_execute_unknown'])
def test_claim_widget_presence_is_not_claimable(mutation):
    from mm_assist.memory.readers.facility_ui import select_facility_view
    row=sample();button=next(e for e in row['render']['elements'] if e['class']=='MMMLButtonElement' and e['key']==49)
    if mutation=='disabled':button['enabled']=0
    if mutation=='not_interactable':button['interactable']=0
    if mutation=='can_execute_false':button['can_execute']='0100'
    if mutation=='can_execute_unknown':button['can_execute']='0000'
    fact=select_facility_view(row['render'],0)
    hit=sense.locate_claim_all(np.zeros((1440,1920,3),np.uint8),memory_view=fact,facility='metal')
    assert not hit['ok'] and hit['src']=='memory'

def test_other_facility_and_wrong_client_aspect_are_rejected_without_ocr(monkeypatch):
    monkeypatch.setattr(sense,'_ocr_lines',lambda *a,**k:pytest.fail('wrong memory page must stop'))
    fact=view()
    assert not sense.locate_go_equip(np.zeros((1440,1920,3),np.uint8),memory_view=fact,facility='wood')['ok']
    assert not sense.locate_claim_all(np.zeros((1000,1920,3),np.uint8),memory_view=fact,facility='metal')['ok']

@pytest.mark.parametrize('name,go,claim', [('metal_away',True,True),('metal_arrived',False,True),('metal_empty',False,False)])
def test_three_actual_scenes_use_model_permission_instead_of_retained_control(name,go,claim):
    from mm_assist.memory.readers.facility_ui import select_facility_view
    row=sample(name);fact=select_facility_view(row['render'],0)
    assert fact['buttons']['go_equip']['present'] is go
    assert fact['buttons']['claim_all']['usable'] is claim
    if name=='metal_empty':
        # Actual empty frame retains label, enabled control and canExecute=true.
        assert fact['buttons']['claim_all']['present'] is True
        button=next(e for e in row['render']['elements'] if e['ptr']==fact['buttons']['claim_all']['element'])
        assert button['enabled']==1 and button['can_execute']=='0101'

@pytest.mark.parametrize('mutation',['missing','renamed','wrong_type','unknown_bool','extra'])
def test_unregistered_model_schema_or_boolean_is_unknown(mutation):
    from mm_assist.memory.readers.facility_ui import select_facility_view
    row=sample();attrs=row['render']['root_attributes']
    e=next(e for e in attrs if e['name_id']==115868)
    if mutation=='missing':attrs.remove(e)
    if mutation=='renamed':e['name']='unregistered_attribute'
    if mutation=='wrong_type':e['type']=3
    if mutation=='unknown_bool':e['i32']=0
    if mutation=='extra':attrs.append(copy.deepcopy(e))
    assert select_facility_view(row['render'],0) is None

def arena(row,monkeypatch):
    """Only injected bytes; no external handles, process lookup or game reads."""
    import struct
    from mm_assist.memory.readers import facility_ui as mod
    off={k:int(v,16) for k,v in mod._config()['offsets'].items()}
    data={};cursor=0x1000000000;classes={}
    def put(a,raw):data.update({a+i:b for i,b in enumerate(raw)})
    def alloc():
        nonlocal cursor
        cursor+=0x1000;put(cursor,b'\0'*1024);return cursor
    def pointer(a,value):put(a,struct.pack('<Q',value))
    def field(a,key,value,fmt='Q'):put(a+off[key],struct.pack('<'+fmt,value))
    def obj(cls,address=None):
        if cls not in classes:
            k=alloc();s=alloc();put(s,cls.encode()+b'\0'*128);field(k,'klass.name',s);classes[cls]=k
        a=address or alloc();pointer(a,classes[cls]);return a
    def string(s):
        if s is None:return 0
        a=alloc();field(a,'String.length',len(s),'I');put(a+off['String.data'],s.encode('utf-16-le'));return a
    def array(values,fmt):
        a=alloc();field(a,'Array.length',len(values),'I');put(a+off['Array.data'],struct.pack('<'+fmt*len(values),*values));return a
    def dictionary(pairs):
        d=alloc();field(d,'Dict.count',len(pairs),'I')
        field(d,'Dict.keys',array([k for k,v in pairs],'I'));field(d,'Dict.values',array([v for k,v in pairs],'Q'));return d
    r=row['render'];stack=obj('UIStackingService');api=alloc();sf=alloc();field(api,'klass.static_fields',sf);field(sf,'UIAPI.Stacking',stack)
    inst=obj('AlteringFacilityPopupUIInstance',row['instance']);ui=obj('MMMLUIHandle',row['handle'])
    vmc=obj('AlteringFacilityPopupUIViewModelComponent',row['vmc']);vm=obj('MMMLUIViewModel`2',row['vm'])
    root=obj('XForest',row['forest']);service=obj('MMMLService',r['service']);window=obj('UnityMMMLWindow',r['window'])
    scope=obj('UIResourceScope',r['resource_scope']);field(scope,'Scope.destroyed',0,'B')
    player=obj('UIStackingPlayer');field(player,'Player.instance',inst)
    players=alloc();field(players,'MMList.items',array([player],'Q'));field(players,'MMList.size',1,'I');field(players,'MMList.version',12,'I')
    field(stack,'Stacking.players',players)
    for key,val in [('Facility.type',row['facility_type']),('Facility.base_handle',ui),('Facility.handle',ui),('Facility.vm',vm),('Facility.vmc',vmc)]:field(inst,key,val,'I' if key=='Facility.type' else 'Q')
    field(vmc,'VMC.handle',ui);field(ui,'Handle.vm',vm);field(ui,'Handle.active',1,'B')
    field(ui,'Handle.root',root);field(ui,'Handle.root_id',row['element_id'],'I');field(ui,'Handle.service',service)
    put(ui+off['Handle.window_id'],bytes.fromhex(r['window_id_nullable']));wid=struct.unpack_from('<I',bytes.fromhex(r['window_id_nullable']),4)[0]
    field(service,'Service.windows',dictionary([(wid,window)]));put(service+off['Service.screen'],bytes.fromhex(r['screen_size']))
    field(window,'Window.id',wid,'I');field(window,'Window.root',root);field(window,'Window.root_id',row['element_id'],'I')
    field(window,'Window.scope',scope);field(window,'Window.closing',0,'B')
    attrs=alloc();items=row['items'];packed_list=struct.unpack('<Q',struct.pack('<iHH',0,len(items),len(items)))[0]
    field(attrs,'Attrs.table',dictionary([(row['element_id'],packed_list)]));ar=alloc();field(ar,'Array.length',len(items),'I')
    put(ar+off['Array.data'],b''.join(struct.pack('<iBBHi',e['name_id'],e['type'],e['aux_high'],e['aux_low'],e['i32']) for e in items))
    field(attrs,'Attrs.array',ar);field(attrs,'Attrs.version',0,'I');field(root,'Forest.attributes',attrs)
    for e in r['elements']:
        a=obj(e['class'],e['ptr']);field(a,'Element.id',e['id'],'I');field(a,'Element.window',window);field(a,'Element.parent',e['parent'])
        field(a,'Element.disposed',e['disposed'],'B');field(a,'Element.dirty',e['layout_dirty'],'B');put(a+off['Element.layout'],struct.pack('<ffff',*e['layout']))
        if e['class']=='MMMLUITextElement':
            t=obj('UIText');field(a,'Text.raw',t);field(a,'Text.text',string(e.get('text')));field(t,'UIText.text',string(e.get('raw_text')))
        if e['class']=='MMMLButtonElement':
            c=obj('ButtonControl',e['control']);field(a,'Button.control',c);field(a,'Button.enabled',e['enabled'],'B')
            field(c,'Control.enabled',e['control_enabled'],'B');field(c,'Control.interactable',e['interactable'],'B');put(c+off['Control.can_execute'],bytes.fromhex(e['can_execute']))
    field(window,'Window.elements',dictionary([(e['key'],e['ptr']) for e in r['elements']]))
    calls=[]
    def read(h,a,n):
        calls.append((a,n));return bytes(data.get(a+i,0) for i in range(n))
    obj('UniqueStringErinn');name_class=classes['UniqueStringErinn'];name_static=alloc();name_array=alloc()
    field(name_class,'klass.static_fields',name_static)
    pointer(name_static+0x10,name_array)
    put(name_static+0x8,struct.pack('<I',200000));field(name_array,'Array.length',200000,'I')
    name_strings={}
    for item in items:
        addr=string(item['name']);pointer(name_array+off['Array.data']+item['name_id']*8,addr)
        name_strings[item['name']]=addr
    monkeypatch.setattr(mod,'read_mem',read)
    monkeypatch.setattr(mod,'resolve_klass',lambda h,ga,rva:name_class if rva==0x1083DA68 else api)
    return mod,read,put,field,off,{'ui':ui,'window':window,'players':players,'attrs':ar,'scope':scope,'inst':inst,
        'name_class':name_class,'name_static':name_static,'name_array':name_array,'name_strings':name_strings},calls

@pytest.mark.parametrize('name', ['metal_away','metal_arrived','metal_empty'])
def test_production_reader_through_locators_on_actual_widget_bytes(monkeypatch,name):
    mod,*_=arena(sample(name),monkeypatch)
    fact=mod.read_active_facility(1,1,1,budget_s=2)
    assert fact is not None and fact['facility']=='metal'
    frame=np.zeros((1440,1920,3),np.uint8)
    assert sense.locate_claim_all(frame,memory_view=fact,facility='metal')['ok'] is (name!='metal_empty')
    assert sense.locate_go_equip(frame,memory_view=fact,facility='metal')['ok'] is (name=='metal_away')

@pytest.mark.parametrize('mutation',['closed','handle','root','scope','size','overlay','changed','budget'])
def test_native_reader_rejects_invalid_or_changing_associations(monkeypatch,mutation):
    mod,read,put,field,off,roots,calls=arena(sample(),monkeypatch)
    if mutation=='closed':field(roots['window'],'Window.closing',1,'B')
    if mutation=='handle':field(roots['ui'],'Handle.active',0,'B')
    if mutation=='root':field(roots['window'],'Window.root_id',999,'I')
    if mutation=='scope':field(roots['scope'],'Scope.destroyed',1,'B')
    if mutation=='size':field(roots['players'],'MMList.size',999,'I')
    if mutation=='overlay':field(roots['players'],'MMList.size',2,'I')
    if mutation=='changed':
        address=roots['ui']+off['Handle.active'];seen=0
        def racing(h,a,n):
            nonlocal seen
            raw=read(h,a,n)
            if a<=address<a+n:
                seen+=1
                if seen>1:
                    changed=bytearray(raw);changed[address-a]=0;return bytes(changed)
            return raw
        monkeypatch.setattr(mod,'read_mem',racing)
    assert mod.read_active_facility(1,1,1,budget_s=0 if mutation=='budget' else 2) is None

@pytest.mark.parametrize('scene,success', [('metal_arrived',True),('metal_empty',False)])
@pytest.mark.parametrize('result_layer',[True,False])
def test_real_claim_chain_uses_memory_and_retains_independent_result_check(monkeypatch,scene,success,result_layer):
    from mm_assist.memory.readers.facility_ui import select_facility_view
    from mm_assist.life_execute import medicine_process as process,ui_act
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests
    reset_stage_gates_for_tests()
    row=sample(scene);fact=select_facility_view(row['render'],0);clicks=[]
    # The recorded fixture is the facility overview, regardless of the live
    # status bar's current page. Keep offline tests independent of its files.
    monkeypatch.setattr(ui_act,'stack_has_facility_recipe',lambda:False)
    monkeypatch.setattr(sense,'_ocr_lines',lambda *a,**k:pytest.fail('OCR ran'))
    monkeypatch.setattr(sense,'detect_facility_page',lambda *a,**k:pytest.fail('image identity ran'))
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:False)
    monkeypatch.setattr(ui_act,'stack_has_common_popup',lambda **k:False)
    monkeypatch.setattr(ui_act,'stack_has_facility_result',lambda **k:result_layer)
    monkeypatch.setattr(ui_act,'dismiss_facility_result',lambda *a,**k:None)
    out=process._run_process(None,facility='metal',recipe='',mode='claim',items={},
        do_click=lambda x,y:clicks.append((x,y)) or True,read_items=None,grab=lambda:np.zeros((1440,1920,3),np.uint8),
        on_progress=None,root=None,sleep_fn=lambda s:None,read_stp=lambda:0,defer_completion=True,memory_reader=lambda:copy.deepcopy(fact))
    assert len(clicks)==int(success)
    assert out.ok is (success and result_layer)
    if success and not result_layer:assert out.status=='claim_result_missing'

@pytest.mark.parametrize('change',['empty','missing','other_facility','pid_changed'])
def test_claim_rechecks_after_recognition_and_does_not_click_old_memory(monkeypatch,change):
    from mm_assist.memory.readers.facility_ui import select_facility_view
    from mm_assist.life_execute import medicine_process as process,ui_act
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests
    reset_stage_gates_for_tests();r=sample('metal_arrived');initial=select_facility_view(r['render'],0)
    monkeypatch.setattr(ui_act,'stack_has_facility_recipe',lambda:False)
    later=copy.deepcopy(initial)
    if change=='empty':later['buttons']['claim_all']['usable']=False
    if change=='missing':later=None
    if change=='other_facility':later['facility']='wood'
    if change=='pid_changed':later['game_pid']=99
    reads=0;clicks=[]
    def memory():
        nonlocal reads
        reads+=1
        if reads<=4:return copy.deepcopy(initial)
        if change=='pid_changed':return copy.deepcopy(initial if reads==5 else later)
        return copy.deepcopy(later)
    monkeypatch.setattr(sense,'_ocr_lines',lambda *a,**k:pytest.fail('changed memory must not run OCR'))
    monkeypatch.setattr(sense,'detect_facility_page',lambda *a,**k:pytest.fail('changed memory must not use old pixels'))
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:False)
    monkeypatch.setattr(ui_act,'stack_has_common_popup',lambda **k:False)
    out=process._run_process(None,facility='metal',recipe='',mode='claim',items={},
        do_click=lambda *p:clicks.append(p) or True,read_items=None,grab=lambda:np.zeros((1440,1920,3),np.uint8),
        on_progress=None,root=None,sleep_fn=lambda s:None,read_stp=lambda:0,defer_completion=True,memory_reader=memory)
    assert not out.ok and not clicks

def test_shared_serial_reader_timeout_does_not_start_scan_or_open_handle():
    from types import SimpleNamespace
    from mm_assist.memory.readers.ui_stack import Il2CppUiStackReader
    reader=object.__new__(Il2CppUiStackReader)
    reader._read_gate=SimpleNamespace(acquire=lambda **k:False)
    reader._ensure=lambda:pytest.fail('busy shared reader must not open/scan')
    assert reader.read_facility_ui() is None

def test_memory_unavailable_never_degrades_business_buttons_to_ocr(monkeypatch):
    from mm_assist.life_execute import medicine_process as process,ui_act
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests
    reset_stage_gates_for_tests();clicks=[]
    monkeypatch.setattr(sense,'_ocr_lines',lambda *a,**k:pytest.fail('business OCR fallback ran'))
    monkeypatch.setattr(sense,'detect_facility_page',lambda *a,**k:{'ok':True,'facility':'metal'})
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:False)
    monkeypatch.setattr(ui_act,'stack_has_facility_recipe',lambda **k:False)
    monkeypatch.setattr(ui_act,'stack_has_common_popup',lambda **k:False)
    out=process._run_process(None,facility='metal',recipe='',mode='claim',items={},
        do_click=lambda *p:clicks.append(p) or True,read_items=None,grab=lambda:np.zeros((1440,1920,3),np.uint8),
        on_progress=None,root=None,sleep_fn=lambda s:None,read_stp=lambda:0,defer_completion=True,memory_reader=lambda:None)
    assert not out.ok and not clicks

def test_memory_claim_travels_first_then_uses_new_arrival_view(monkeypatch):
    from itertools import count
    from mm_assist.memory.readers.facility_ui import select_facility_view
    from mm_assist.life_execute import medicine_process as process,ui_act,ui_logic
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests
    reset_stage_gates_for_tests()
    away=sample();arrived=sample('metal_arrived')
    facts=[select_facility_view(away['render'],0),select_facility_view(arrived['render'],0)]
    state={'arrived':False,'result':False,'stp':0};clicks=[]
    monkeypatch.setattr(ui_act,'stack_has_facility_recipe',lambda:False)
    def click(x,y):
        clicks.append((x,y))
        if (x,y)==(223,305):state['arrived']=True
        elif (x,y)==(187,1087):
            assert state['arrived'];state['result']=True
        else:pytest.fail('unexpected coordinate')
        return True
    def stp():
        if not state['arrived']:
            return 0
        state['stp']+=1
        return 1 if state['stp']==1 else 0
    monkeypatch.setattr(sense,'_ocr_lines',lambda *a,**k:pytest.fail('OCR ran'))
    monkeypatch.setattr(sense,'detect_facility_page',lambda *a,**k:pytest.fail('image identity ran'))
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:False)
    monkeypatch.setattr(ui_act,'stack_has_common_popup',lambda **k:False)
    monkeypatch.setattr(ui_act,'confirm_go_facility_popup',lambda *a,**k:None)
    monkeypatch.setattr(ui_act,'stack_has_facility_result',lambda **k:state['result'])
    monkeypatch.setattr(ui_act,'dismiss_facility_result',lambda *a,**k:None)
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.skip_wait_ui_params',lambda:False)
    # Exercise the real two-beat gate with injected samples; never depend on
    # a running state-watch advancing its production files during this test.
    samples=count(1)
    monkeypatch.setattr(ui_logic,'ui_param_stamps',lambda **k:(float(next(samples)),1.0))
    monkeypatch.setattr(ui_logic,'_ui_stack_sample_ts',lambda **k:float(next(samples)))
    monkeypatch.setattr(ui_logic,'ui_param_key',lambda **k:('stp=0|facility',
        'facility_result' if state['result'] else 'facility:metal'))
    out=process._run_process(None,facility='metal',recipe='',mode='claim',items={},
        do_click=click,read_items=None,grab=lambda:np.zeros((1440,1920,3),np.uint8),
        on_progress=None,root=None,sleep_fn=lambda s:None,read_stp=stp,defer_completion=True,
        memory_reader=lambda:copy.deepcopy(facts[int(state['arrived'])]))
    assert out.ok and clicks==[(223,305),(187,1087)]
    assert ui_logic.LAST_UI_WAIT['outcome']=='two_beat'
