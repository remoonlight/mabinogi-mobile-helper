"""Reject stable but unrelated controls, and bindings changed during sampling."""
import copy
import threading

import numpy as np
import pytest

from test_facility_memory import arena, sample
from test_crafting_items_memory import craft_arena


@pytest.mark.parametrize('fault', ['foreign_title', 'detached_title', 'second_root', 'shared_control'])
def test_unrelated_widget_cannot_reach_a_production_locator(monkeypatch, fault):
    from mm_assist.memory.readers.facility_ui import select_facility_view
    from mm_assist.perception.process_sense import locate_claim_all
    row = sample()
    r = row['render']; es = r['elements']
    title = next(e for e in es if e.get('text') == '金屬加工')
    if fault == 'foreign_title': title['window'] += 8
    if fault == 'detached_title': title['parent'] = 0
    if fault == 'second_root':
        root = copy.deepcopy(next(e for e in es if not e['parent']))
        root['ptr'] += 0x100000000000
        es.append(root)
        title['parent'] = root['ptr']
    if fault == 'shared_control':
        buttons = [e for e in es if e['class'] == 'MMMLButtonElement']
        claim = next(e for e in buttons if e['key'] == 49)
        other = next(e for e in buttons if e['ptr'] != claim['ptr'])
        other['control'] = claim['control']
    fact = select_facility_view(r, row['facility_type'])
    assert fact is None
    # Explicit failure must retain the no-image/OCR fallback at the consumer.
    monkeypatch.setattr('mm_assist.perception.process_sense._ocr_lines',
                        lambda *a, **k: pytest.fail('OCR authorized a broken binding'))
    assert not locate_claim_all(np.zeros((1440, 1920, 3), np.uint8),
        memory_view={'error': 'control_binding'}, facility='metal')['ok']


@pytest.mark.parametrize('fault', ['cycle', 'missing_parent', 'second_root', 'duplicate_element'])
def test_craft_diagnostic_requires_one_connected_element_tree(monkeypatch, fault):
    mod, read, put, field, off, roots, _ = craft_arena(monkeypatch)
    elements = sample()['render']['elements']
    e = next(e for e in elements if e['parent'])
    if fault == 'cycle': field(e['ptr'], 'Element.parent', e['ptr'])
    if fault == 'missing_parent': field(e['ptr'], 'Element.parent', 0x9999999990)
    if fault == 'second_root': field(e['ptr'], 'Element.parent', 0)
    if fault == 'duplicate_element':
        # Duplicate the same pointer under two keys while keeping the second
        # key's ID consistent; uniqueness must not rely on dictionary keys.
        root = next(e for e in elements if not e['parent'])
        field(e['ptr'], 'Element.parent', root['ptr'])
        table = int.from_bytes(read(1, roots['window']+off['Window.elements'], 8), 'little')
        values = int.from_bytes(read(1, table+off['Dict.values'], 8), 'little')
        index = next(i for i, x in enumerate(elements) if x['ptr'] == e['ptr'])
        other = next(i for i, x in enumerate(elements) if i != index and x['parent'])
        put(values+off['Array.data']+other*8, e['ptr'].to_bytes(8, 'little'))
    assert mod.read_active_craft_render(1, 1, 1, budget_s=2) is None


@pytest.mark.parametrize('craft', [False, True])
@pytest.mark.parametrize('fault', ['identity_changed', 'identity_error', 'selected_pid_changed'])
def test_shared_mmml_reader_rechecks_process_before_publishing(monkeypatch, craft, fault):
    from mm_assist.memory.readers import ui_stack, facility_ui
    reader = object.__new__(ui_stack.Il2CppUiStackReader)
    reader._read_gate = threading.RLock()
    reader._h, reader._pid, reader._ga = 1, 101, 1000
    reader._rva = {'UIAPI': 1}
    reader._ensure = lambda: True
    reader.close = lambda: None
    identity = {'pid': 101, 'birth': 10, 'build_id': 'offline'}
    state = {'current': identity, 'pid': 101, 'error': False}
    def bound(*args):
        if state['error']: raise OSError('process expired')
        return state['current']
    monkeypatch.setattr(ui_stack, 'read_identity', bound)
    monkeypatch.setattr(ui_stack, 'find_game_pid', lambda: state['pid'])
    def read(*a, **kw):
        if fault == 'identity_changed': state['current'] = {**identity, 'birth': 11}
        if fault == 'identity_error': state['error'] = True
        if fault == 'selected_pid_changed': state['pid'] = 102
        return {'identity': [1, 2, 3, 4, 5], 'facility': 'metal'}
    monkeypatch.setattr(facility_ui, 'read_active_craft_render' if craft else 'read_active_facility', read)
    diag = {}
    result = reader.read_craft_render(diagnostics=diag) if craft else reader.read_facility_ui(diagnostics=diag)
    assert result is None
    assert diag['error']


@pytest.mark.parametrize('craft', [False, True])
def test_shared_mmml_reader_keeps_stable_process_provenance(monkeypatch, craft):
    from mm_assist.memory.readers import ui_stack, facility_ui
    reader = object.__new__(ui_stack.Il2CppUiStackReader)
    reader._read_gate = threading.RLock()
    reader._h, reader._pid, reader._ga = 1, 101, 1000
    reader._rva = {'UIAPI': 1}; reader._ensure = lambda: True
    identity = {'pid': 101, 'birth': 10, 'build_id': 'offline'}
    monkeypatch.setattr(ui_stack, 'read_identity', lambda *a: identity)
    monkeypatch.setattr(ui_stack, 'find_game_pid', lambda: 101)
    monkeypatch.setattr(facility_ui, 'read_active_craft_render' if craft else 'read_active_facility',
                        lambda *a, **k: {'identity': [1, 2, 3, 4, 5]})
    result = reader.read_craft_render() if craft else reader.read_facility_ui()
    assert result['process_identity'] == identity and result['game_pid'] == 101


def test_late_old_reader_does_not_publish_or_close_the_new_binding(monkeypatch):
    from mm_assist.memory.readers import ui_stack, facility_ui
    reader = object.__new__(ui_stack.Il2CppUiStackReader)
    reader._read_gate = threading.RLock()
    reader._h, reader._pid, reader._ga = 1, 101, 1000
    reader._rva = {'UIAPI': 1}; reader._ensure = lambda: True
    reader.close = lambda: pytest.fail('closed the new binding')
    monkeypatch.setattr(ui_stack, 'read_identity', lambda *a: {'pid': 101, 'birth': 10})
    monkeypatch.setattr(ui_stack, 'find_game_pid', lambda: 101)
    def read(*a, **kw):
        reader._h, reader._pid, reader._ga = 2, 102, 2000
        return {'identity': [1, 2, 3, 4, 5]}
    monkeypatch.setattr(facility_ui, 'read_active_facility', read)
    assert reader.read_facility_ui() is None
    assert (reader._h, reader._pid, reader._ga) == (2, 102, 2000)
