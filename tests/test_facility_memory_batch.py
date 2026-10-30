"""Keep all sampled-field guards while fitting the existing 180 ms budget."""
import pytest
from test_facility_memory import arena, sample


@pytest.mark.parametrize('scene',['metal_away','metal_arrived','metal_empty'])
@pytest.mark.parametrize('rpm_cost',[0.00005,0.00016])
def test_native_facility_read_fits_budget_with_cost_per_rpm(monkeypatch,scene,rpm_cost):
    mod,read,*rest=arena(sample(scene),monkeypatch)
    calls=rest[-1];clock=[0.0]
    def timed(h,address,size):
        clock[0]+=rpm_cost
        return read(h,address,size)
    monkeypatch.setattr(mod,'read_mem',timed)
    monkeypatch.setattr(mod.time,'perf_counter',lambda:clock[0])
    result=mod.read_active_facility(1,1,1)  # Original 180 ms wall budget.
    assert result is not None and result['facility']=='metal'
    assert len(calls)<2500 and clock[0]<=0.18


@pytest.mark.parametrize('field,size',[('Element.window',8),('Element.id',4),
    ('Element.parent',8),('Element.layout',16),('Element.dirty',1),('Element.disposed',1)])
def test_batched_recheck_still_rejects_each_changed_common_field(monkeypatch,field,size):
    row=sample();mod,read,put,write,off,roots,calls=arena(row,monkeypatch)
    element=next(e['ptr'] for e in row['render']['elements'] if e['class']=='MMMLButtonElement')
    address=element+off[field];seen=0
    def changing(h,start,n):
        nonlocal seen
        data=read(h,start,n)
        if start<=address and address+size<=start+n:
            seen+=1
            if seen>1:
                data=bytearray(data);data[address-start]^=1;data=bytes(data)
        return data
    monkeypatch.setattr(mod,'read_mem',changing)
    assert mod.read_active_facility(1,1,1,budget_s=2) is None


def test_unused_padding_is_not_mistaken_for_a_sampled_field(monkeypatch):
    row=sample();mod,read,put,write,off,roots,calls=arena(row,monkeypatch)
    element=next(e['ptr'] for e in row['render']['elements'] if e['class']=='MMMLButtonElement')
    padding=element+0x12;seen=0
    def changing(h,start,n):
        nonlocal seen
        data=read(h,start,n)
        if start<=padding<start+n:
            seen+=1
            if seen>1:
                data=bytearray(data);data[padding-start]^=1;data=bytes(data)
        return data
    monkeypatch.setattr(mod,'read_mem',changing)
    assert mod.read_active_facility(1,1,1,budget_s=2) is not None


def test_recheck_does_not_cross_unread_address_gaps(monkeypatch):
    from mm_assist.memory.readers import facility_ui as mod
    checked={(100,4):b'abcd',(112,4):b'wxyz'};calls=[]
    def read(h,a,n):
        calls.append((a,n));return checked[(a,n)]
    monkeypatch.setattr(mod,'read_mem',read)
    monkeypatch.setattr(mod.time,'perf_counter',lambda:0.)
    assert mod._recheck_fields(1,checked,[],1.)
    assert calls==[(100,4),(112,4)]


def test_recheck_keeps_original_chunk_cap(monkeypatch):
    from mm_assist.memory.readers import facility_ui as mod
    checked={(100,20000):b'a'*20000,(20100,20000):b'b'*20000};calls=[]
    def read(h,a,n):
        calls.append((a,n));return checked[(a,n)]
    monkeypatch.setattr(mod,'read_mem',read)
    monkeypatch.setattr(mod.time,'perf_counter',lambda:0.)
    assert mod._recheck_fields(1,checked,[],1.)
    assert calls==[(100,20000),(20100,20000)]


def test_recheck_deadline_rejects_without_another_read(monkeypatch):
    from mm_assist.memory.readers import facility_ui as mod
    monkeypatch.setattr(mod,'read_mem',lambda *a:pytest.fail('expired read'))
    monkeypatch.setattr(mod.time,'perf_counter',lambda:2.)
    assert not mod._recheck_fields(1,{(100,4):b'abcd'},[],1.)


@pytest.mark.parametrize('role',['unrelated','claim'])
def test_valid_empty_ui_string_does_not_invalidate_other_buttons(monkeypatch,role):
    row=sample();mod,read,put,write,off,roots,calls=arena(row,monkeypatch)
    protected={'金屬加工','前往設備','全部領取'}
    element=next(e for e in row['render']['elements'] if e['class']=='MMMLUITextElement'
        and (e.get('text')=='全部領取' if role=='claim' else e.get('text') not in protected))
    string=int.from_bytes(read(1,element['ptr']+off['Text.text'],8),'little')
    assert string
    write(string,'String.length',0,'I')
    # Native read_mem(size=0) returns None; a byte-arena slice returns b'' and
    # would hide the production bug unless this boundary is represented.
    monkeypatch.setattr(mod,'read_mem',lambda h,a,n:None if n==0 else read(h,a,n))
    fact=mod.read_active_facility(1,1,1,budget_s=2)
    if role=='claim':assert fact is None
    else:assert fact is not None and fact['buttons']['claim_all']['usable']


@pytest.mark.parametrize('role',['text','control'])
def test_typed_object_prefetch_keeps_runtime_class_checks(monkeypatch,role):
    row=sample();mod,read,put,write,off,roots,calls=arena(row,monkeypatch)
    cls='MMMLUITextElement' if role=='text' else 'MMMLButtonElement'
    element=next(e for e in row['render']['elements'] if e['class']==cls)
    field='Text.raw' if role=='text' else 'Button.control'
    obj=int.from_bytes(read(1,element['ptr']+off[field],8),'little')
    put(obj,read(1,roots['ui'],8))  # MMMLUIHandle cannot stand in for either type.
    assert mod.read_active_facility(1,1,1,budget_s=2) is None


def test_diagnostics_distinguish_budget_and_selector_without_changing_facts(monkeypatch):
    row=sample();mod,read,put,write,off,roots,calls=arena(row,monkeypatch)
    diag={}
    assert mod.read_active_facility(1,1,1,budget_s=0,diagnostics=diag) is None
    assert diag['error']=='budget/address' and diag['stage']=='players'
    assert mod.read_active_facility(1,1,1,budget_s=2,diagnostics=diag) is not None
    assert not diag['error'] and diag['stage']=='recheck'
    write(roots['scope'],'Scope.destroyed',1,'B')
    assert mod.read_active_facility(1,1,1,budget_s=2,diagnostics=diag) is None
    assert diag['error']=='selector'


def test_recheck_diagnostic_reports_exact_changed_field(monkeypatch):
    from mm_assist.memory.readers import facility_ui as mod
    monkeypatch.setattr(mod,'read_mem',lambda *a:b'abce')
    monkeypatch.setattr(mod.time,'perf_counter',lambda:0.)
    diag={}
    assert not mod._recheck_fields(1,{(100,4):b'abcd'},[],1.,diag)
    assert diag['error']=='changed:0x64/4'
