"""Bounded craft-window sampling; synthetic bytes are not live calibration."""
import struct
import pytest
from test_facility_memory import arena, sample


def craft_arena(monkeypatch):
    mod, read, put, field, off, roots, calls = arena(sample(), monkeypatch)
    def rename(obj, value):
        klass = struct.unpack('<Q', read(1, obj, 8))[0]
        address = struct.unpack('<Q', read(1, klass + off['klass.name'], 8))[0]
        put(address, value.encode().ljust(128, b'\0'))
    rename(roots['inst'], 'CraftingFacilityPopupUIInstance')
    vmc = struct.unpack('<Q', read(1, roots['inst'] + off['Facility.vmc'], 8))[0]
    rename(vmc, 'CraftingFacilityPopupUIViewModelComponent')
    put(roots['inst'] + 0x50, struct.pack('<Q', roots['ui']))
    # Synthetic stable crafting render. The source metal fixture contains dirty
    # padding spacers; do not weaken production rejection to make it pass.
    for e in sample()['render']['elements']:
        field(e['ptr'], 'Element.dirty', 0, 'B')
    # dump.cs 537979/687755/689494/687957 and value struct 3492409-13.
    # Explicit synthetic names/IDs: these do not register a live 強化 mapping.
    hub, selector, main = 0xA000000000, 0xA000001000, 0xA000002000
    def obj(address, cls, index):
        klass, name = 0xA000010000 + index*0x2000, 0xA000011000 + index*0x2000
        put(address, b'\0'*256)
        put(address, struct.pack('<Q', klass))
        put(klass + off['klass.name'], struct.pack('<Q', name))
        put(name, cls.encode().ljust(128, b'\0'))
    obj(hub, 'CraftingHubUIComponent', 0)
    obj(selector, 'CraftingHubCategorySelector', 1)
    obj(main, 'String', 2)
    field(main, 'String.length', 2, 'I')
    put(main + off['String.data'], '其他'.encode('utf-16-le'))
    put(roots['inst'] + 0x48, struct.pack('<Q', hub))
    put(vmc + 0x20, struct.pack('<Q', hub))
    put(hub + 0x20, struct.pack('<Q', selector))
    put(selector + 0x10, struct.pack('<QiB', main, 12345, 1))
    put(selector + 0x24, struct.pack('<i', -1))
    roots.update(hub=hub, selector=selector, main=main, vmc=vmc)
    return mod, read, put, field, off, roots, calls


def test_current_craft_window_exposes_rechecked_render_without_facility_guess(monkeypatch):
    mod, *_ = craft_arena(monkeypatch)
    result = mod.read_active_craft_render(1, 1, 1, budget_s=2)
    assert result is not None
    assert len(result['identity']) == 5
    assert result['render']['window'] == result['identity'][4]
    assert result['render']['elements']
    assert 'facility' not in result


def test_craft_reader_never_reinterprets_altering_facility(monkeypatch):
    mod, *_ = arena(sample(), monkeypatch)
    assert mod.read_active_craft_render(1, 1, 1, budget_s=2) is None


@pytest.mark.parametrize('mutation', ['closed', 'inactive', 'root', 'overlay', 'dirty', 'changed', 'budget'])
def test_craft_render_rejects_stale_or_unbound_native_facts(monkeypatch, mutation):
    mod, read, put, field, off, roots, calls = craft_arena(monkeypatch)
    if mutation == 'closed': field(roots['window'], 'Window.closing', 1, 'B')
    if mutation == 'inactive': field(roots['ui'], 'Handle.active', 0, 'B')
    if mutation == 'root': field(roots['window'], 'Window.root_id', 999, 'I')
    if mutation == 'overlay': field(roots['players'], 'MMList.size', 2, 'I')
    if mutation == 'dirty': field(sample()['render']['elements'][0]['ptr'], 'Element.dirty', 1, 'B')
    if mutation == 'changed':
        address = roots['ui'] + off['Handle.active']
        seen = [0]
        def changing(h, a, n):
            data = read(h, a, n)
            if a <= address < a + n:
                seen[0] += 1
                if seen[0] > 1:
                    data = bytearray(data); data[address-a] = 0; data = bytes(data)
            return data
        monkeypatch.setattr(mod, 'read_mem', changing)
    assert mod.read_active_craft_render(1, 1, 1, budget_s=0 if mutation == 'budget' else 2) is None


def test_busy_shared_craft_reader_does_not_open_handle_or_scan():
    from types import SimpleNamespace
    from mm_assist.memory.readers.ui_stack import Il2CppUiStackReader
    reader = object.__new__(Il2CppUiStackReader)
    reader._read_gate = SimpleNamespace(acquire=lambda **kw: False)
    reader._ensure = lambda: pytest.fail('busy read must not attach or scan')
    assert reader.read_craft_render() is None


def test_craft_diagnostic_includes_current_selector_and_raw_subgroup(monkeypatch):
    mod, _, _, _, _, roots, _ = craft_arena(monkeypatch)
    result = mod.read_active_craft_render(1, 1, 1, budget_s=2)
    assert result['category_state'] == {
        'main_name': '其他', 'sub_display_group': 12345,
        'has_sub_selection': True, 'dropdown_index': -1,
        'identity': [roots['hub'], roots['selector']],
    }
    # A raw group ID is not a registered 強化 label or click permission.
    assert 'selected_label' not in result['category_state']
    assert 'buttons' not in result


@pytest.mark.parametrize('mutation', ['vmc_hub', 'hub_class', 'selector_class', 'null_main', 'bad_bool', 'unreadable'])
def test_craft_selector_unknown_is_not_all_or_closed(monkeypatch, mutation):
    mod, read, put, _, off, roots, _ = craft_arena(monkeypatch)
    if mutation == 'vmc_hub': put(roots['vmc'] + 0x20, struct.pack('<Q', roots['hub']+8))
    if mutation == 'hub_class': put(roots['hub'], read(1, roots['selector'], 8))
    if mutation == 'selector_class': put(roots['selector'], read(1, roots['hub'], 8))
    if mutation == 'null_main': put(roots['selector'] + 0x10, struct.pack('<Q', 0))
    if mutation == 'bad_bool': put(roots['selector'] + 0x1C, b'\x02')
    if mutation == 'unreadable':
        address = roots['selector'] + 0x24
        monkeypatch.setattr(mod, 'read_mem', lambda h,a,n: None if a<=address<a+n else read(h,a,n))
    result = mod.read_active_craft_render(1, 1, 1, budget_s=2)
    assert result is None or result['category_state'] is None


@pytest.mark.parametrize('part', ['hub', 'selector', 'main', 'group', 'sub_flag', 'dropdown'])
def test_craft_selector_change_during_sampling_rejects_whole_snapshot(monkeypatch, part):
    mod, read, _, _, _, roots, _ = craft_arena(monkeypatch)
    addresses = {'hub': roots['inst']+0x48, 'selector': roots['hub']+0x20,
                 'main': roots['main']+0x14, 'group': roots['selector']+0x18,
                 'sub_flag': roots['selector']+0x1C, 'dropdown': roots['selector']+0x24}
    address = addresses[part]; seen = [0]
    def changing(h, a, n):
        data = read(h, a, n)
        if a<=address<a+n:
            seen[0] += 1
            if seen[0]>1:
                data=bytearray(data); data[address-a] ^= 1; data=bytes(data)
        return data
    monkeypatch.setattr(mod, 'read_mem', changing)
    assert mod.read_active_craft_render(1, 1, 1, budget_s=2) is None
