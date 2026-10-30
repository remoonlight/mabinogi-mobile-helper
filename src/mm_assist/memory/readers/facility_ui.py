"""Bounded active MMML facility widgets; no heap search, cached UI or input."""
from __future__ import annotations

import json
import math
import struct
import time
from functools import lru_cache
from mm_assist.knowledge_paths import ui_config
from mm_assist.memory.il2cpp import parse_int, resolve_klass
from mm_assist.memory.process import read_mem

FACILITIES = {0: ('metal','金屬加工'), 1: ('wood','木材加工'), 2: ('cloth','布料加工'),
              3: ('leather','皮革加工'), 4: ('medicine','藥品加工'), 5: ('food','食材加工')}
LABELS = {'go_equip':'前往設備', 'claim_all':'全部領取'}


def _element_tree(render):
    """One window, one connected root, and one owner per native control."""
    elements=render['elements']; by={e['ptr']:e for e in elements}
    if not elements or len(by)!=len(elements):return None
    roots=[]; controls=set()
    for e in elements:
        if (type(e['ptr']) is not int or e['ptr']<=0
                or type(e['parent']) is not int or e['parent']<0
                or e['window']!=render['window']):return None
        if not e['parent']:roots.append(e['ptr'])
        elif e['parent'] not in by:return None
        if e['class']=='MMMLButtonElement':
            control=e['control']
            if type(control) is not int or control<=0 or control in controls:return None
            controls.add(control)
    if len(roots)!=1:return None
    connected={roots[0]}
    for e in elements:
        current=e['ptr']; path=set()
        while current not in connected:
            if current in path:return None
            path.add(current);current=by[current]['parent']
        connected.update(path)
    return by,roots[0]


def _stable_ancestors(element, by, window):
    chain=[];seen=set()
    for _ in range(24):
        if (element['ptr'] in seen or element['window']!=window
                or element['disposed']!=0 or element['layout_dirty']!=0):return None
        seen.add(element['ptr']);chain.append(element)
        if not element['parent']:return chain
        element=by.get(element['parent'])
        if element is None:return None
    return None


def _model_flags(items):
    spec=_config()['semantic_model']
    schema=sorted(tuple(p) for p in spec['schema'])
    if sorted((e.get('name',''),e['type']) for e in items)!=schema:return None
    flags={}
    for kind,key in (('claim_all','isGetAll'),('go_equip','isMoveToFacilityVisible')):
        value=next(e for e in items if e['name']==key)
        if value['aux_high']!=0 or value['aux_low']!=0 or value['i32'] not in (1,2):return None
        flags[kind]=value['i32']==1
    return flags


def select_facility_view(render: dict, facility_type: int) -> dict | None:
    """Resolve unique text descendants to their actual active button ancestors."""
    try:
        facility,title=FACILITIES[facility_type]
        flags=_model_flags(render['root_attributes'])
        if flags is None:return None
        if (render['window_class']!='UnityMMMLWindow' or render['closing']!=0
                or render['resource_class']!='UIResourceScope' or render['resource_destroyed']!=0):
            return None
        screen=struct.unpack('<ff',bytes.fromhex(render['screen_size']))
        if any(not math.isfinite(v) or v<100 or v>16384 for v in screen):return None
        elements=render['elements'];tree=_element_tree(render)
        if tree is None:return None
        by,root_ptr=tree
        titles=[e for e in elements if e.get('text')==title and e.get('raw_text')==title]
        if len(titles)!=1:return None
        if _stable_ancestors(titles[0],by,render['window']) is None:return None
        buttons={}
        for kind,label in LABELS.items():
            texts=[e for e in elements if e.get('text')==label]
            if len(texts)>1:return None
            if not texts:
                if flags[kind]:return None
                buttons[kind]={'present':False,'usable':False};continue
            if texts[0].get('raw_text')!=label:return None
            chain=_stable_ancestors(texts[0],by,render['window'])
            if chain is None or chain[-1]['ptr']!=root_ptr:return None
            button=next((e for e in chain if e['class']=='MMMLButtonElement'),None)
            root=chain[-1]['layout']
            if (len(root)!=4 or abs(root[0])>1 or abs(root[2])>1
                    or abs(root[1]-screen[0])>1 or abs(root[3]-screen[1])>1):return None
            if button is None or button['control_class']!='ButtonControl':return None
            x,w,y,h=button['layout']
            if (any(not math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<4
                    or min(x,y)<0 or x+w>screen[0]+1 or y+h>screen[1]+1):return None
            usable=button['enabled']==1 and button['control_enabled']==1 and button['interactable']==1
            if kind=='go_equip' and not flags[kind]:return None
            if kind=='claim_all':usable=usable and button['can_execute']=='0101' and flags[kind]
            buttons[kind]={'present':True,'usable':usable,'rect':[x,y,w,h],
                           'element':button['ptr'],'control':button['control']}
        return {'facility':facility,'screen_size':list(screen),'buttons':buttons,
                'window':render['window'],'view_root':list(render['view_root'])}
    except (KeyError,TypeError,ValueError,struct.error,IndexError):return None


@lru_cache(maxsize=1)
def _config():
    return json.loads(ui_config('il2cpp_facility_ui.json').read_text('utf-8'))


def _recheck_fields(handle, checked, spans, deadline, diagnostics=None):
    """Coalesce only previously read ranges; compare every actual sampled field.

    Prefetch padding can share a read, but never becomes a validation signal.
    No new address gap is crossed and no chunk exceeds the original raw cap.
    """
    ranges=[(a,a+n,None) for a,n in spans]
    ranges.extend((a,a+n,(a,n,data)) for (a,n),data in checked.items())
    groups=[]
    for start,end,field in sorted(ranges,key=lambda r:(r[0],r[1])):
        if (groups and start<=groups[-1][1]
                and max(end,groups[-1][1])-groups[-1][0]<=32768):
            groups[-1][1]=max(end,groups[-1][1])
            if field is not None:groups[-1][2].append(field)
        else:
            groups.append([start,end,[] if field is None else [field]])
    for start,end,fields in groups:
        if time.perf_counter()>deadline:
            if diagnostics is not None:diagnostics['error']='budget:recheck'
            return False
        current=read_mem(handle,start,end-start)
        if current is None or len(current)!=end-start:
            if diagnostics is not None:diagnostics['error']=f'unreadable:recheck:{start:#x}/{end-start}'
            return False
        for a,n,old in fields:
            if current[a-start:a-start+n]!=old:
                if diagnostics is not None:diagnostics['error']=f'changed:{a:#x}/{n}'
                return False
    return time.perf_counter()<=deadline


def read_active_craft_render(handle, ga: int, uiapi_rva: int, *, budget_s: float=0.18, diagnostics=None) -> dict | None:
    """Diagnostic render/selector only: not a page or click permission.

    CraftingFacilityPopupUIInstance.handle@0x50 is verified in dump.cs
    537979-537982; VM and component fields share 0x58/0x60 with Altering.
    Selector links/struct: 687755, 689494, 687957-60, 3492409-13.
    Raw subgroup IDs/dropdown indices have no registered label mapping.
    """
    return read_active_facility(handle, ga, uiapi_rva, budget_s=budget_s,
        diagnostics=diagnostics, _craft_render=True)


def read_active_facility(handle, ga: int, uiapi_rva: int, *, budget_s: float=0.18, diagnostics=None,
        _craft_render: bool=False) -> dict | None:
    """Read and recheck all sampled links/content within one bounded serial call."""
    off={k:parse_int(v) for k,v in _config()['offsets'].items()}
    started=time.perf_counter();deadline=started+budget_s; checked={}; class_names={}
    stage='players';valid=False
    if diagnostics is not None:diagnostics.clear()
    slabs=[];spans=[]
    def raw(addr,size):
        if not addr or size<0 or size>32768 or time.perf_counter()>deadline:raise ValueError('budget/address')
        if (addr,size) in checked:return checked[(addr,size)]
        data=None
        for start,block in slabs:
            if start<=addr and addr+size<=start+len(block):
                data=block[addr-start:addr-start+size];break
        if data is None:data=read_mem(handle,addr,size)
        if data is None or len(data)!=size:raise ValueError(f'unreadable {addr:#x} size={size}')
        old=checked.setdefault((addr,size),data)
        if old!=data:raise ValueError('changed')
        return data
    def prefetch(obj,end):
        if not obj or not 0<end<=32768 or time.perf_counter()>deadline:raise ValueError('budget/address')
        block=read_mem(handle,obj,end)
        if block is None or len(block)!=end:raise ValueError('unreadable object')
        slabs.append((obj,block));spans.append((obj,end))
    def p(obj,field):return struct.unpack('<Q',raw(obj+off[field],8))[0] if obj else 0
    def u(obj,field):return struct.unpack('<I',raw(obj+off[field],4))[0] if obj else None
    def b(obj,field):return raw(obj+off[field],1)[0] if obj else None
    def name(obj):
        if not obj:return ''
        klass=struct.unpack('<Q',raw(obj,8))[0]
        if klass not in class_names:
            address=p(klass,'klass.name')
            class_names[klass]=raw(address,128).split(b'\0',1)[0].decode('ascii',errors='strict')
        return class_names[klass]
    def text(obj):
        if not obj:return None
        length=u(obj,'String.length')
        if length is None or not 0<=length<=150:raise ValueError('string bound')
        if length==0:return ''
        return raw(obj+off['String.data'],length*2).decode('utf-16-le')
    def model_names(attributes):
        spec=_config()['semantic_model']
        klass=resolve_klass(handle,ga,parse_int(spec['typeinfo_rva']))
        if not klass:raise ValueError('name table class')
        label=raw(p(klass,'klass.name'),128).split(b'\0',1)[0]
        if label!=b'UniqueStringErinn':raise ValueError('name table class')
        static=p(klass,'klass.static_fields')
        table=p(static,'NameTable.strings');count=u(static,'NameTable.next_index')
        length=u(table,'Array.length')
        if not count or length is None or not 0<count<=length<=1000000:raise ValueError('name table bounds')
        for item in attributes:
            index=item['name_id']
            if not 0<index<count:raise ValueError('name index')
            pointer=struct.unpack('<Q',raw(table+off['Array.data']+index*8,8))[0]
            value=text(pointer)
            if not value:raise ValueError('name missing')
            item['name']=value
    def craft_category(inst,vmc):
        hub=p(inst,'Crafting.hub')
        if name(hub)!='CraftingHubUIComponent' or p(vmc,'CraftingVMC.hub')!=hub:return None
        selector=p(hub,'CraftingHub.selector')
        if name(selector)!='CraftingHubCategorySelector':return None
        main=p(selector,'CraftingSelector.main')
        has_sub=b(selector,'CraftingSelector.has_sub')
        if name(main)!='String' or has_sub not in (0,1):return None
        return {'main_name':text(main),
            'sub_display_group':struct.unpack('<i',raw(selector+off['CraftingSelector.group'],4))[0],
            'has_sub_selection':bool(has_sub),
            'dropdown_index':struct.unpack('<i',raw(selector+off['CraftingSelector.dropdown'],4))[0],
            'identity':[hub,selector]}
    def pairs(d,limit):
        count=u(d,'Dict.count');keys=p(d,'Dict.keys');values=p(d,'Dict.values')
        if count is None or not 0<=count<=limit:raise ValueError('dict bound')
        if count==0:return []
        if min(u(keys,'Array.length') or 0,u(values,'Array.length') or 0)<count:raise ValueError('dict arrays')
        ks=struct.unpack('<'+'I'*count,raw(keys+off['Array.data'],count*4))
        vs=struct.unpack('<'+'Q'*count,raw(values+off['Array.data'],count*8))
        if len(set(ks))!=count:raise ValueError('dict keys')
        return list(zip(ks,vs))
    try:
        api=resolve_klass(handle,ga,uiapi_rva); sf=p(api,'klass.static_fields')
        stacking=p(sf,'UIAPI.Stacking')
        if name(stacking)!='UIStackingService':return None
        players=p(stacking,'Stacking.players');arr=p(players,'MMList.items');size=u(players,'MMList.size')
        u(players,'MMList.version')
        if size is None or not 0<size<=8 or (u(arr,'Array.length') or 0)<size:return None
        ps=struct.unpack('<'+'Q'*size,raw(arr+off['Array.data'],size*8))
        instances=[]
        target_class='CraftingFacilityPopupUIInstance' if _craft_render else 'AlteringFacilityPopupUIInstance'
        for player in ps:
            if name(player)!='UIStackingPlayer':return None
            inst=p(player,'Player.instance');nm=name(inst)
            if nm not in ('CraftingHubPopupUIInstance',target_class):return None
            if nm==target_class:instances.append(inst)
        if len(instances)!=1:return None
        stage='instance'
        inst=instances[0];ft=None if _craft_render else u(inst,'Facility.type')
        ui=p(inst,'Crafting.handle' if _craft_render else 'Facility.handle')
        vmc=p(inst,'Facility.vmc');vm=p(inst,'Facility.vm')
        vmc_class='CraftingFacilityPopupUIViewModelComponent' if _craft_render else 'AlteringFacilityPopupUIViewModelComponent'
        if (name(ui)!='MMMLUIHandle' or name(vmc)!=vmc_class
                or name(vm)!='MMMLUIViewModel`2' or p(vmc,'VMC.handle')!=ui
                or p(inst,'Facility.base_handle')!=ui or p(ui,'Handle.vm')!=vm or b(ui,'Handle.active')!=1):return None
        root=p(ui,'Handle.root');root_id=u(ui,'Handle.root_id');service=p(ui,'Handle.service')
        if name(root)!='XForest' or name(service)!='MMMLService':return None
        stage='attributes'
        attrs=p(root,'Forest.attributes');u(attrs,'Attrs.version')
        table=pairs(p(attrs,'Attrs.table'),4096);array=p(attrs,'Attrs.array')
        lists=[v for k,v in table if k==root_id]
        if len(lists)!=1:return None
        offset,capacity,count=struct.unpack('<iHH',struct.pack('<Q',lists[0]))
        if offset<0 or not 0<count<=capacity<=256 or offset+count>(u(array,'Array.length') or 0):return None
        attributes=[]
        for key,typ,high,low,value in struct.iter_unpack('<iBBHi',raw(array+off['Array.data']+12*offset,12*count)):
            attributes.append({'name_id':key,'type':typ,'aux_high':high,'aux_low':low,'i32':value})
        nullable=raw(ui+off['Handle.window_id'],8)
        stage='window'
        if nullable[0]!=1:return None
        wid=struct.unpack_from('<I',nullable,4)[0]
        windows=pairs(p(service,'Service.windows'),128)
        window=next((v for k,v in windows if k==wid),0)
        if (name(window)!='UnityMMMLWindow' or u(window,'Window.id')!=wid
                or p(window,'Window.root')!=root or u(window,'Window.root_id')!=root_id):return None
        scope=p(window,'Window.scope')
        render={'window':window,'window_class':name(window),'closing':b(window,'Window.closing'),
                'resource_class':name(scope),'resource_destroyed':b(scope,'Scope.destroyed'),
                'screen_size':raw(service+off['Service.screen'],8).hex(),
                'view_root':[root,root_id],'elements':[],'root_attributes':attributes}
        common_end=max(off['Element.window']+8,off['Element.id']+4,
            off['Element.parent']+8,off['Element.layout']+16,
            off['Element.dirty']+1,off['Element.disposed']+1)
        text_end=max(common_end,off['Text.raw']+8,off['Text.text']+8)
        button_end=max(common_end,off['Button.control']+8,off['Button.enabled']+1)
        stage='elements'
        for key,e in pairs(p(window,'Window.elements'),1024):
            slabs.clear()
            prefetch(e,common_end)
            cls=name(e)
            if not cls.startswith('MMML'):raise ValueError('element type')
            # Common element fields are one bounded snapshot, rather than six
            # separate RPM calls. The final check still compares each field.
            if cls in {'MMMLUITextElement','MMMLButtonElement'}:
                prefetch(e,text_end if cls=='MMMLUITextElement' else button_end)
            if u(e,'Element.id')!=key:raise ValueError('element key')
            row={'ptr':e,'class':cls,'window':p(e,'Element.window'),'parent':p(e,'Element.parent'),
                 'disposed':b(e,'Element.disposed'),'layout_dirty':b(e,'Element.dirty'),
                 'layout':list(struct.unpack('<ffff',raw(e+off['Element.layout'],16)))}
            if cls=='MMMLUITextElement':
                t=p(e,'Text.raw')
                prefetch(t,off['UIText.text']+8)
                if name(t)!='UIText':raise ValueError('text type')
                row.update(text=text(p(e,'Text.text')),raw_text=text(p(t,'UIText.text')))
            if cls=='MMMLButtonElement':
                ctrl=p(e,'Button.control')
                prefetch(ctrl,max(off['Control.enabled']+1,off['Control.interactable']+1,off['Control.can_execute']+2))
                ctrl_class=name(ctrl)
                if ctrl_class!='ButtonControl':raise ValueError('control type')
                row.update(control=ctrl,control_class=ctrl_class,enabled=b(e,'Button.enabled'),
                           control_enabled=b(ctrl,'Control.enabled'),interactable=b(ctrl,'Control.interactable'),
                           can_execute=raw(ctrl+off['Control.can_execute'],2).hex())
            render['elements'].append(row)
        stage='selector'
        if _craft_render:
            if (render['closing']!=0 or render['resource_class']!='UIResourceScope'
                    or render['resource_destroyed']!=0):return None
            screen=struct.unpack('<ff',bytes.fromhex(render['screen_size']))
            if any(not math.isfinite(v) or v<100 or v>16384 for v in screen):return None
            if any(e['window']!=window or e['disposed']!=0 or e['layout_dirty']!=0
                    for e in render['elements']):return None
            if _element_tree(render) is None:return None
            # Keep unknown distinct from All/closed. This raw diagnostic does
            # not identify a 強化 submenu or authorize a filter/potion click.
            fact={'render':render,'category_state':craft_category(inst,vmc)}
        else:
            model_names(attributes)
            fact=select_facility_view(render,ft)
        if fact is None:return None
        stage='recheck'
        if not _recheck_fields(handle,checked,spans,deadline,diagnostics):return None
        valid=True
        return {**fact,'identity':[inst,ui,root,root_id,window]}
    except (OSError,ValueError,TypeError,struct.error,UnicodeError) as exc:
        if diagnostics is not None:diagnostics['error']=str(exc)
        return None
    finally:
        if diagnostics is not None:
            diagnostics.update(stage=stage,checked=len(checked),elapsed_ms=round((time.perf_counter()-started)*1000,1))
            diagnostics.setdefault('error','' if valid else stage)
