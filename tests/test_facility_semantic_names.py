"""Process-local Name numbers must not invalidate semantic facility facts."""
import copy
import json
import struct
from pathlib import Path

import pytest

from test_facility_memory import arena,sample


def test_retained_current_native_widgets_and_semantic_table_reach_real_locator(monkeypatch):
    import numpy as np
    from mm_assist.perception import process_sense
    native=json.loads((Path(__file__).parent/'fixtures/facility_memory/semantic_names_20261014.json').read_text('utf-8'))
    assert native['names']['115910']=='isGetAll' and native['names']['108193']=='isMoveToFacilityVisible'
    row=sample()
    row['render']={**row['render'],**native['render']}
    row['items']=copy.deepcopy(native['render']['root_attributes'])
    for item in row['items']:item['name']=native['names'][str(item['name_id'])]
    row['render']['root_attributes']=row['items']
    row['forest'],row['element_id']=row['render']['view_root']
    # The retained production render omits dictionary IDs after validating
    # them. Synthetic associations supply IDs; widget content stays native.
    for index,element in enumerate(row['render']['elements'],1):
        element['id']=element['key']=index
    mod,*_=arena(row,monkeypatch)
    fact=mod.read_active_facility(1,1,1,budget_s=2)
    assert fact is not None and fact['facility']=='metal'
    monkeypatch.setattr(process_sense,'_ocr_lines',lambda *a,**kw:pytest.fail('no OCR'))
    monkeypatch.setattr(process_sense,'match_template',lambda *a,**kw:pytest.fail('no image button fallback'))
    frame=np.zeros((1440,1920,3),np.uint8)
    assert process_sense.locate_claim_all(frame,memory_view=fact,facility='metal')['ok']


@pytest.mark.parametrize('shift',[42,321])
@pytest.mark.parametrize('scene',['metal_away','metal_arrived','metal_empty'])
def test_new_process_ids_preserve_real_claim_and_travel_permissions(monkeypatch,shift,scene):
    row=sample(scene)
    for item in row['items']:item['name_id']+=shift
    mod,*_=arena(row,monkeypatch)
    fact=mod.read_active_facility(1,1,1,budget_s=2)
    assert fact is not None
    assert fact['buttons']['claim_all']['usable'] is (scene!='metal_empty')
    assert fact['buttons']['go_equip']['present'] is (scene=='metal_away')


@pytest.mark.parametrize('fault',['class','table','index','name','changed_pointer','changed_text'])
def test_semantic_identity_is_paired_and_unknown_table_stops(monkeypatch,fault):
    row=sample();mod,read,put,field,off,roots,_=arena(row,monkeypatch)
    address=roots['name_array']+off['Array.data']+115868*8
    if fault=='class':field(roots['name_class'],'klass.name',0)
    if fault=='table':put(roots['name_static']+0x10,b'\0'*8)
    if fault=='index':put(roots['attrs']+off['Array.data'],struct.pack('<i',999999))
    if fault=='name':
        target=roots['name_strings']['isGetAll']
        put(target+off['String.data'],'wrongKey'.encode('utf-16-le'))
    if fault in {'changed_pointer','changed_text'}:
        if fault=='changed_text':address=roots['name_strings']['isGetAll']+off['String.data']
        seen=0
        def changing(h,start,n):
            nonlocal seen
            value=read(h,start,n)
            if start<=address<start+n:
                seen+=1
                if seen>1:
                    value=bytearray(value);value[address-start]^=1;value=bytes(value)
            return value
        monkeypatch.setattr(mod,'read_mem',changing)
    assert mod.read_active_facility(1,1,1,budget_s=2) is None
