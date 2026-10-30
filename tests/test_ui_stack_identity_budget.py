"""Keep two lifecycle checks around a serialized read, without a third in ensure."""
import pytest

from mm_assist.memory.readers import ui_stack


@pytest.fixture
def reader(monkeypatch):
    value = ui_stack.Il2CppUiStackReader()
    value._h, value._pid, value._ga = 7, 123, 456
    value._state_stacks = [789]
    monkeypatch.setattr(ui_stack, 'process_handle_alive', lambda _: True)
    monkeypatch.setattr(value, '_ensure_state_stacks', lambda: None)
    monkeypatch.setattr(value, '_names_from_state_lists', lambda: (['TutorialTooltipUIState'], None))
    monkeypatch.setattr(value, '_names_from_players', lambda: ([], 0))
    monkeypatch.setattr(value, '_active_map_listener', lambda _: None)
    monkeypatch.setattr(value, 'close', lambda: setattr(value, '_h', None))
    return value


def test_current_stack_read_checks_identity_exactly_before_and_after(reader, monkeypatch):
    calls = []
    monkeypatch.setattr(ui_stack, 'read_identity', lambda h, pid:
                        calls.append((h, pid)) or {'pid': pid, 'birth': 1})
    result = reader.read(force=True)
    assert result.ready and result.names == ['TutorialTooltipUIState']
    assert calls == [(7, 123), (7, 123)]
    assert result.process_identity == {'pid': 123, 'birth': 1}


@pytest.mark.parametrize('case', ['changed', 'refused'])
def test_final_identity_failure_discards_the_read(reader, monkeypatch, case):
    calls = []
    def identity(h, pid):
        calls.append((h, pid))
        if len(calls) == 1:
            return {'pid': pid, 'birth': 1}
        if case == 'refused':
            raise OSError('client_file_changed')
        return {'pid': pid, 'birth': 2}
    monkeypatch.setattr(ui_stack, 'read_identity', identity)
    result = reader.read(force=True)
    assert not result.ready and not result.names
    assert result.error == ('client_file_changed' if case == 'refused' else 'process_binding_changed')
    assert calls == [(7, 123), (7, 123)] and reader._h is None


def test_identity_cost_is_included_in_reported_total(reader, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(ui_stack.time, 'perf_counter', lambda: clock[0])
    def identity(h, pid):
        clock[0] += .3
        return {'pid': pid, 'birth': 1}
    monkeypatch.setattr(ui_stack, 'read_identity', identity)
    result = reader.read(force=True)
    assert result.ready
    assert result.cost_ms['total_ms'] == pytest.approx(600.)


def test_later_cached_read_still_rechecks_identity_on_both_sides(reader, monkeypatch):
    calls = []
    monkeypatch.setattr(ui_stack, 'read_identity', lambda h, pid:
                        calls.append((h, pid)) or {'pid': pid, 'birth': 1})
    assert reader.read(force=True).ready
    assert reader.read(force=False).ready
    assert calls == [(7, 123)] * 4


def test_binding_changed_during_initial_check_cannot_read_memory(reader, monkeypatch):
    def identity(h, pid):
        reader._ga = 457
        return {'pid': pid, 'birth': 1}
    monkeypatch.setattr(ui_stack, 'read_identity', identity)
    monkeypatch.setattr(reader, '_names_from_state_lists', lambda: pytest.fail('changed binding must not read'))
    result = reader.read(force=True)
    assert not result.ready and not result.names and result.error == 'process_binding_changed'
