"""Offline contracts for client versions, read provenance and diagnostics."""
import json
from types import SimpleNamespace

import pytest


def test_repository_configuration_matches_explicit_registration():
    """Checkout formatting must preserve the already verified byte fingerprint."""
    from mm_assist.memory import client_build as build

    registry = json.loads(build.ui_config('il2cpp_client_builds.json').read_text(encoding='utf-8'))
    assert build.configuration_digest() in {row['config_sha256'] for row in registry['builds']}


def test_layout_config_checkout_preserves_registered_bytes():
    import subprocess
    from mm_assist.config import ROOT

    relative = 'knowledge/ui/config/il2cpp_layout_checks.json'
    # Exercise Git's real checkout conversion, including core.autocrlf=true.
    checked_out = subprocess.check_output(
        ['git', '-c', 'core.autocrlf=true', 'cat-file', '--filters', f'HEAD:{relative}'],
        cwd=ROOT,
    )
    assert checked_out == (ROOT / relative).read_bytes()
    assert b'\r\n' not in checked_out


def test_build_requires_exact_files_and_configuration(tmp_path):
    from mm_assist.memory.client_build import fingerprint_files, select_build, BuildRejected
    files = {name: tmp_path / name for name in ('GameAssembly.dll', 'UnityPlayer.dll', 'global-metadata.dat')}
    for name, path in files.items():
        path.write_bytes(name.encode())
    observed = fingerprint_files(files)
    profile = {'id': 'fixture', 'files': observed, 'config_sha256': 'a' * 64, 'evidence': 'fixture only'}
    assert select_build(observed, 'a' * 64, [profile]) == 'fixture'
    with pytest.raises(BuildRejected, match='configuration_changed'):
        select_build(observed, 'b' * 64, [profile])
    files['GameAssembly.dll'].write_bytes(b'updated')
    with pytest.raises(BuildRejected, match='client_build_unregistered'):
        select_build(fingerprint_files(files), 'a' * 64, [profile])
    with pytest.raises(BuildRejected, match='client_build_unregistered'):
        select_build(observed, 'a' * 64, [])


def test_hash_cancellation_and_file_mutation_are_not_valid_builds(tmp_path):
    from mm_assist.memory.client_build import fingerprint_files, BuildRejected
    path = tmp_path / 'dll'
    path.write_bytes(b'0' * (2 * 1024 * 1024))
    calls = []
    def stop():
        calls.append(1)
        if len(calls) == 3:
            path.write_bytes(b'changed')
        return False
    with pytest.raises(BuildRejected, match='file_changed'):
        fingerprint_files({'dll': path}, should_stop=stop)
    with pytest.raises(BuildRejected, match='cancelled'):
        fingerprint_files({'dll': path}, should_stop=lambda: True)


def test_module_paths_must_belong_to_the_same_installation(tmp_path):
    from mm_assist.memory.client_build import client_files, BuildRejected
    with pytest.raises(BuildRejected, match='module_installation_mismatch'):
        client_files({'GameAssembly.dll': (1, str(tmp_path/'one/GameAssembly.dll')),
                      'UnityPlayer.dll': (2, str(tmp_path/'two/UnityPlayer.dll'))})


def test_denied_read_preserves_original_error_when_diagnostic_fails(monkeypatch):
    from mm_assist.memory import process
    from mm_assist.memory import access_diagnostic as access
    monkeypatch.setattr(process.kernel32, 'OpenProcess', lambda *a: 0)
    monkeypatch.setattr(process.ctypes, 'get_last_error', lambda: 5)
    monkeypatch.setattr(access, 'collect_access', lambda pid: (_ for _ in ()).throw(RuntimeError('query failed')))
    with pytest.raises(access.ProcessReadDenied) as caught:
        process.open_process(123)
    error = caught.value
    assert error.winerror == 5 and error.diagnostic['target_pid'] == 123
    assert error.diagnostic['access'] == 0x410 and error.diagnostic['diagnostic_error']


def test_snapshot_rejects_changed_process_and_keeps_component_timings(monkeypatch):
    from mm_assist.life_plan import snapshot
    monkeypatch.setattr(snapshot, 'find_game_pid', lambda: 123)
    monkeypatch.setattr(snapshot, 'open_process', lambda pid: 7)
    monkeypatch.setattr(snapshot, 'close_handle', lambda h: None)
    identity = iter([{'pid': 123, 'birth': 1, 'build_id': 'x'}, {'pid': 123, 'birth': 2, 'build_id': 'x'}])
    monkeypatch.setattr(snapshot, 'read_identity', lambda *a: next(identity), raising=False)
    monkeypatch.setattr(snapshot, 'read_consumable_counts', lambda *a: {'羊毛': 12})
    monkeypatch.setattr(snapshot, 'read_facility_queues', lambda *a: {})
    snap = snapshot.take_snapshot(include_tools=False)
    assert not snap.ok and snap.error == 'process_binding_changed'
    assert snap.items == {} and snap.evidence['components']['items']['elapsed_s'] >= 0


def test_root_changes_cannot_be_published_as_component_evidence():
    from mm_assist.memory.evidence import ReadEvidence, bind_evidence, root_observation
    pointers = {0x10000: 0x20000, 0x10010: 0x30000}
    reader = lambda h, addr: pointers.get(addr)
    evidence = ReadEvidence({'pid': 1, 'birth': 3})
    with bind_evidence(evidence), root_observation('items', 7, 0x10000, (0, 0x10), reader=reader):
        pointers[0x10010] = 0x40000
    assert evidence.error == 'component_binding_changed:items'
    assert not evidence.recheck(7, reader=reader)


def test_scan_diagnostic_retained_but_hidden_from_statusbar():
    from mm_assist.jobs.state_watch import format_mem_panel
    snap = SimpleNamespace(mem_status='resolving', mem_xy=None, ui_params='',
        memory_diagnostic={'reason': 'coverage_incomplete', 'scan': {'bytes_read': 1048576, 'regions_visited': 2, 'candidate_count': 3, 'coverage_complete': False}})
    text = format_mem_panel(snap)
    # User requested hiding scan telemetry from the status bar. The retained
    # diagnostic still describes incomplete coverage; presentation cannot gate.
    assert snap.memory_diagnostic['reason'] == 'coverage_incomplete'
    assert snap.memory_diagnostic['scan'] == {'bytes_read': 1048576,
        'regions_visited': 2, 'candidate_count': 3, 'coverage_complete': False}
    assert '掃描未完成' not in text and 'MiB' not in text and '候選' not in text


def test_scan_diagnostic_reports_partial_and_complete_coverage(monkeypatch):
    from mm_assist.memory import il2cpp as il
    from mm_assist.memory.diagnostics import latest_scan
    monkeypatch.setattr(il, '_hot_key', lambda h: h)
    monkeypatch.setattr(il, '_load_hot_regions', lambda *a: None)
    monkeypatch.setattr(il, '_save_hot_regions', lambda *a: None)
    monkeypatch.setattr(il, '_list_heap_regions', lambda *a, **k: [(0x10000, 32, 4, 0x20000)])
    monkeypatch.setattr(il, 'read_chunk', lambda *a: (bytearray(32), 32))
    assert il.scan_klass_objects(123, 0x90000) == []
    row = latest_scan(123)
    assert row['coverage_complete'] and row['bytes_read'] == 32 and row['regions_visited'] == 1
    assert il.scan_klass_objects(123, 0x90000, should_stop=lambda: True) == []
    row = latest_scan(123)
    assert not row['coverage_complete'] and row['reason'] == 'cancelled'


def test_layout_audit_reports_missing_ambiguous_type_and_type_mismatch(tmp_path):
    from mm_assist.memory.layout_audit import audit_dump
    dump = tmp_path/'dump.cs'
    dump.write_text('// Namespace: A\npublic class Foo // TypeDefIndex: 1\n{\n public int count; // 0x10\n}\n'
                    '// Namespace: B\npublic class Foo // TypeDefIndex: 2\n{\n public long count; // 0x18\n}\n', encoding='utf-8')
    checks = [{'type': 'Foo', 'field': 'count', 'offset': 16},
              {'type': 'A.Foo', 'field': 'count', 'offset': 16, 'field_type': 'long'},
              {'type': 'B.Foo', 'field': 'missing', 'offset': 16}]
    rows = audit_dump(dump, checks)['checks']
    assert [r['status'] for r in rows] == ['ambiguous_type', 'type_mismatch', 'field_missing']


def test_layout_audit_never_promotes_an_offset_match_to_client_support(tmp_path):
    from mm_assist.memory.layout_audit import audit_dump
    dump = tmp_path/'dump.cs'
    dump.write_text('// Namespace: A\npublic class Foo // TypeDefIndex: 1\n{\n public int count; // 0x10\n}\n', encoding='utf-8')
    report = audit_dump(dump, [{'type': 'A.Foo', 'field': 'count', 'offset': 16, 'field_type': 'int'}])
    assert report['ok'] and report['checks'][0]['status'] == 'match'
    assert report['ready_for_input'] is False and report['client_build_verified'] is False


@pytest.fixture
def registered_client(tmp_path, monkeypatch):
    from mm_assist.memory import client_build as build, process
    for name in ('_CACHE', '_LATEST', '_BINDINGS', '_LIFETIMES', '_FILE_HASH_CACHE'):
        monkeypatch.setattr(build, name, {})
    config = tmp_path/'config'
    config.mkdir()
    cfg = config/'il2cpp_scene.json'
    cfg.write_text('{"fixture":1}')
    monkeypatch.setattr(build, 'ui_config', lambda name: config/name)
    metadata = tmp_path/'MabinogiMobile_Data/il2cpp_data/Metadata/global-metadata.dat'
    metadata.parent.mkdir(parents=True)
    files = {'GameAssembly.dll': tmp_path/'GameAssembly.dll',
             'UnityPlayer.dll': tmp_path/'UnityPlayer.dll', 'global-metadata.dat': metadata}
    for name, path in files.items():
        path.write_text(name)
    hashes = build.fingerprint_files(files)
    registry = config/'il2cpp_client_builds.json'
    registry.write_text(json.dumps({'builds': [{'id': 'fixture', 'files': hashes,
        'config_sha256': build.configuration_digest(), 'evidence': 'test-only'}]}))
    modules = {'GameAssembly.dll': (0x10000, str(files['GameAssembly.dll'])),
               'UnityPlayer.dll': (0x20000, str(files['UnityPlayer.dll']))}
    lifetime = {'birth': 10, 'alive': True}
    monkeypatch.setattr(process, 'process_birth', lambda h: lifetime['birth'])
    monkeypatch.setattr(process, 'process_handle_alive', lambda h: lifetime['alive'])
    monkeypatch.setattr(process, 'process_modules', lambda pid: dict(modules))
    monkeypatch.setattr(build, '_SESSION_SIGNATURE', build._session_signature())
    return build, process, files, cfg, registry, modules, lifetime


@pytest.mark.parametrize('mutation', ['birth', 'exit', 'module', 'config', 'file_deleted', 'config_added'])
def test_bound_client_cannot_survive_identity_or_file_changes(registered_client, mutation):
    build, _, files, cfg, _, modules, lifetime = registered_client
    identity = build.verify_client(7, 123)
    assert identity == build.bound_identity(7, 123)
    if mutation == 'birth': lifetime['birth'] = 11
    if mutation == 'exit': lifetime['alive'] = False
    if mutation == 'module': modules['GameAssembly.dll'] = (0x90000, str(files['GameAssembly.dll']))
    if mutation == 'config': cfg.write_text('{"fixture":2}')
    if mutation == 'file_deleted': files['UnityPlayer.dll'].unlink()
    if mutation == 'config_added': (cfg.parent/'il2cpp_new.json').write_text('{}')
    with pytest.raises(build.BuildRejected):
        build.bound_identity(7, 123)
    assert not build.build_diagnostic(123)['verified']


def test_unknown_version_hashes_are_cached_but_never_enrolled(registered_client, monkeypatch):
    build, _, files, _, registry, _, _ = registered_client
    registry.write_text('{"builds":[]}')
    monkeypatch.setattr(build, '_SESSION_SIGNATURE', build._session_signature())
    reads = []
    original = build.fingerprint_files
    def read(files, **kwargs):
        reads.extend(name for name in files if name.endswith('.dll') or name.endswith('.dat'))
        return original(files, **kwargs)
    monkeypatch.setattr(build, 'fingerprint_files', read)
    for _ in range(2):
        with pytest.raises(build.BuildRejected, match='unregistered'):
            build.verify_client(7, 123)
    assert sorted(reads) == sorted(files)
    assert json.loads(registry.read_text()) == {'builds': []}
    assert build.build_diagnostic(123)['files'] == original(files)


def test_open_closes_rejected_handle_and_forgets_accepted_handle(registered_client, monkeypatch):
    build, process, _, _, registry, _, _ = registered_client
    closed = []
    monkeypatch.setattr(process.kernel32, 'OpenProcess', lambda *a: 7)
    monkeypatch.setattr(process.kernel32, 'CloseHandle', closed.append)
    assert process.open_process(123) == 7
    process.close_handle(7)
    assert closed == [7] and not build._BINDINGS
    registry.write_text('{"builds":[]}')
    with pytest.raises(build.BuildRejected):
        process.open_process(123)
    assert closed == [7, 7]


def test_closed_old_lifetime_still_invalidates_discovery_cache(registered_client, monkeypatch):
    build, _, _, _, _, _, lifetime = registered_client
    from mm_assist.memory import il2cpp
    resets = []
    monkeypatch.setattr(il2cpp, 'reset_scan_cache', resets.append)
    build.verify_client(7, 123)
    build.forget_handle(7)
    lifetime['birth'] += 1
    build.verify_client(8, 123)
    assert resets == [8]


def test_scan_late_results_and_reset_do_not_overwrite_newer_status():
    from mm_assist.memory.diagnostics import begin_scan, publish_scan, latest_scan, invalidate_scans
    old = begin_scan(456)
    current = begin_scan(456)
    publish_scan(456, current, {'reason': 'coverage_incomplete', 'bytes_read': 4})
    publish_scan(456, old, {'reason': 'coverage_complete', 'bytes_read': 99})
    assert latest_scan(456)['bytes_read'] == 4
    invalidate_scans(456)
    publish_scan(456, current, {'reason': 'coverage_complete'})
    assert latest_scan(456) == {}


def test_runtime_audit_rechecks_metadata_and_never_proves_field_type():
    from mm_assist.memory.layout_audit import audit_runtime
    klass, table = 0x100000100, 0x30000
    cells = {0x10100: klass.to_bytes(8, 'little'), (klass+16): (0x40000).to_bytes(8, 'little'),
             (klass+24): (0x41000).to_bytes(8, 'little'), (klass+0x80): table.to_bytes(8, 'little'),
             0x40000: b'Foo\0'.ljust(256, b'\0'), 0x41000: b'A\0'.ljust(256, b'\0'),
             0x42000: b'count\0'.ljust(256, b'\0'), table+32: bytes(32),
             table: (0x42000).to_bytes(8, 'little')+bytes(8)+klass.to_bytes(8, 'little')+(16).to_bytes(4, 'little')+bytes(4)}
    check = [{'type': 'A.Foo', 'field': 'count', 'offset': 16, 'typeinfo_rva': 0x100, 'field_type': 'int'}]
    report = audit_runtime(lambda addr, size: cells.get(addr), 0x10000, check)
    assert report['ok'] and report['checks'][0]['field_type_verified'] is False
    calls = {}
    def changed(addr, size):
        calls[addr] = calls.get(addr, 0)+1
        value = cells.get(addr)
        return b'Changed\0'.ljust(256, b'\0') if addr == 0x40000 and calls[addr] > 1 else value
    report = audit_runtime(changed, 0x10000, check)
    assert not report['ok'] and report['error'] == 'runtime_layout_changed'
    assert report['ready_for_input'] is False


def test_build_refusal_blocks_page_navigation_even_when_warm_counts_are_full():
    from mm_assist.jobs.state_watch import watch_ready_refuse_reason
    row = {'pid': 7, 'updated_at_unix': 100, 'tick_completed': True, 'warm_loaded': 20, 'warm_total': 20,
           'memory_diagnostic': {'build': {'verified': False, 'reason': 'client_build_unregistered'}}}
    assert watch_ready_refuse_reason(row, now=100, expected_pid=7, for_page_nav=True) == 'client_build_unregistered'


def test_stack_cannot_publish_a_ready_result_from_an_old_process(monkeypatch):
    from mm_assist.memory.readers.ui_stack import Il2CppUiStackReader, UiStackSnap
    from mm_assist.memory.readers import ui_stack
    reader = Il2CppUiStackReader()
    reader._h, reader._pid = 7, 123
    identities = iter([{'pid': 123, 'birth': 1}, {'pid': 123, 'birth': 2}])
    monkeypatch.setattr(ui_stack, 'read_identity', lambda *a: next(identities))
    monkeypatch.setattr(reader, '_read_current', lambda **k: UiStackSnap(ready=True, names=['MapUIInstance']))
    monkeypatch.setattr(reader, 'close', lambda: setattr(reader, '_h', None))
    snap = reader.read(force=True)
    assert not snap.ready and not snap.names and snap.error == 'process_binding_changed'


def test_snapshot_keeps_successful_root_provenance_and_final_rechecks(monkeypatch):
    from mm_assist.life_plan import snapshot
    from mm_assist.memory import process
    from mm_assist.memory.evidence import root_observation
    monkeypatch.setattr(snapshot, 'find_game_pid', lambda: 123)
    monkeypatch.setattr(snapshot, 'open_process', lambda pid: 7)
    monkeypatch.setattr(snapshot, 'close_handle', lambda h: None)
    cells = {0x10000: 0x20000, 0x10010: 0x30000}
    monkeypatch.setattr(process, 'read_ptr', lambda h, addr: cells[addr])
    def items(*a):
        with root_observation('items', 7, 0x10000, (0, 0x10)):
            return {'羊毛': 12}
    monkeypatch.setattr(snapshot, 'read_consumable_counts', items)
    monkeypatch.setattr(snapshot, 'read_facility_queues', lambda *a: {})
    snap = snapshot.take_snapshot(include_tools=False)
    assert snap.ok and snap.evidence['roots'][0]['instance'] == 0x10000
    def queues(*a):
        cells[0x10010] = 0x90000
        return {}
    monkeypatch.setattr(snapshot, 'read_facility_queues', queues)
    snap = snapshot.take_snapshot(include_tools=False)
    assert not snap.ok and snap.items == {} and snap.error == 'component_binding_changed:items'


def test_permission_queries_are_read_only_and_release_partial_results(monkeypatch):
    import win32api
    import win32security as security
    from mm_assist.memory.access_diagnostic import collect_access
    closed, rights = [], []
    class Handle:
        def __init__(self, name): self.name = name
        def Close(self): closed.append(self.name)
    class Sid:
        def GetSubAuthorityCount(self): return 1
        def GetSubAuthority(self, index): return 0x3000
    monkeypatch.setattr(win32api, 'GetCurrentProcess', lambda: Handle('current'))
    def open_process(access, inherit, pid):
        rights.append(('process', access, inherit, pid))
        return Handle('target')
    monkeypatch.setattr(win32api, 'OpenProcess', open_process)
    def token(process, access):
        rights.append(('token', access))
        return Handle(process.name+'-token')
    monkeypatch.setattr(security, 'OpenProcessToken', token)
    def query(token, kind):
        if kind == security.TokenElevation:
            raise OSError('denied elevation query')
        assert kind == security.TokenIntegrityLevel
        return Sid(), 0
    monkeypatch.setattr(security, 'GetTokenInformation', query)
    row = collect_access(123)
    assert row['tool_elevated'] is None and row['target_elevated'] is None
    assert row['tool_integrity_rid'] == row['target_integrity_rid'] == 0x3000
    assert rights == [('token', 8), ('process', 0x1000, False, 123), ('token', 8)]
    assert sorted(closed) == ['current-token', 'target', 'target-token']


def test_audit_cli_layout_is_offline_and_does_not_change_config(tmp_path, monkeypatch, capsys):
    from mm_assist.memory import layout_audit as audit, process
    dump = tmp_path/'dump.cs'
    dump.write_text('// Namespace: A\npublic class Foo // TypeDefIndex: 1\n{\n public int count; // 0x10\n}\n')
    monkeypatch.setattr(audit, 'configured_checks', lambda: [{'type': 'A.Foo', 'field': 'count', 'offset': 16}])
    monkeypatch.setattr(process, 'open_process', lambda *a: pytest.fail('offline opened process'))
    out = tmp_path/'result.json'
    assert audit.main(['layout', '--dump', str(dump), '--out', str(out)]) == 0
    report = json.loads(out.read_text(encoding='utf-8'))
    assert report['ok'] and not report['client_build_verified']


@pytest.mark.parametrize('failure', ['missing', 'changed', 'budget'])
def test_runtime_layout_invalid_or_unready_evidence_is_never_input_permission(failure):
    from mm_assist.memory.layout_audit import audit_runtime
    check = [{'type': 'A.Foo', 'field': 'count', 'offset': 16, 'typeinfo_rva': 0x100}]
    read = (lambda *a: None) if failure == 'missing' else lambda *a: bytes(8)
    report = audit_runtime(read, 0x10000, check, max_wall_s=0 if failure == 'budget' else 2)
    assert not report['ok'] and not report['ready_for_input'] and not report['client_build_verified']


def test_scan_hit_limit_does_not_mean_full_candidate_coverage(monkeypatch):
    from mm_assist.memory import il2cpp as il
    from mm_assist.memory.diagnostics import latest_scan
    monkeypatch.setattr(il, '_hot_key', lambda h: 123)
    monkeypatch.setattr(il, '_load_hot_regions', lambda *a: None)
    monkeypatch.setattr(il, '_save_hot_regions', lambda *a: None)
    monkeypatch.setattr(il, '_list_heap_regions', lambda *a, **k: [(0x10000, 32, 4, 0x20000)])
    raw = (0x90000).to_bytes(8, 'little') * 4
    monkeypatch.setattr(il, 'read_chunk', lambda *a: (raw, 32))
    assert il.scan_klass_objects(123, 0x90000, max_hits=1) == [0x10000]
    row = latest_scan(123)
    assert not row['coverage_complete'] and row['reason'] == 'target_limit_reached'
    assert row['candidate_count'] == 1


def test_late_klass_read_is_namespaced_by_the_lifetime_it_started_in(monkeypatch):
    from mm_assist.memory import il2cpp as il, client_build
    monkeypatch.setattr(il, '_KLASS_CACHE', {})
    monkeypatch.setattr(il, '_hot_key', lambda h: 123)
    lifetime = [1]
    monkeypatch.setattr(client_build, 'discovery_token', lambda h: (123, lifetime[0], 0x10000, 'fixture'))
    reads = []
    def read(*a):
        reads.append(lifetime[0])
        lifetime[0] = 2
        return 0x100000001 if len(reads) == 1 else 0x100000002
    monkeypatch.setattr(il, 'read_ptr', read)
    assert il.resolve_klass(7, 0x10000, 0x100) == 0x100000001
    assert il.resolve_klass(7, 0x10000, 0x100) == 0x100000002
    assert reads == [1, 2]


def test_configuration_reregistration_still_requires_reopening_the_reader_process(registered_client):
    build, _, files, cfg, registry, _, _ = registered_client
    build.verify_client(7, 123)
    cfg.write_text('{"fixture":99}')
    registry.write_text(json.dumps({'builds': [{'id': 'fixture', 'files': build.fingerprint_files(files),
        'config_sha256': build.configuration_digest(), 'evidence': 'new-fixture'}]}))
    # Existing reader objects still hold their old parsed configuration.
    with pytest.raises(build.BuildRejected, match='client_session_changed'):
        build.verify_client(8, 123)


def test_fingerprint_pair_rejects_an_earlier_file_changing_while_hashing_a_later_file(tmp_path):
    from mm_assist.memory.client_build import fingerprint_files, BuildRejected
    first, second = tmp_path/'first', tmp_path/'second'
    first.write_bytes(b'a')
    second.write_bytes(b'b')
    calls = []
    def check():
        calls.append(1)
        if len(calls) == 3:
            first.write_bytes(b'changed after first hash')
        return False
    with pytest.raises(BuildRejected, match='file_changed'):
        fingerprint_files({'a': first, 'b': second}, should_stop=check)


def test_monitor_diagnostic_uses_current_stack_error_instead_of_old_last_error(monkeypatch):
    from mm_assist.perception.scene import SceneMonitor
    from mm_assist.memory import diagnostics
    mon = object.__new__(SceneMonitor)
    mon._last_win = None
    mon._stack_reader = SimpleNamespace(last_error=None)
    monkeypatch.setattr(diagnostics, 'memory_diagnostic', lambda pid, **k: k)
    assert mon.memory_diagnostic(stack_error='stack_scan_pending')['stack_error'] == 'stack_scan_pending'


def test_rejected_candidate_cannot_poison_an_otherwise_valid_snapshot():
    from mm_assist.memory.evidence import ReadEvidence, bind_evidence, root_observation
    evidence = ReadEvidence({'pid': 123, 'birth': 1})
    with bind_evidence(evidence), root_observation('tools', 7, 0x10000, (0, 0x10), reader=lambda *a: None) as candidate:
        candidate.accepted = False  # A ghost/empty optional tools component is not a returned fact.
    assert evidence.error == '' and evidence.roots == []


def _mock_modules(monkeypatch, rows):
    from mm_assist.memory import process
    source = iter(rows)
    closed = []
    def advance(handle, entry):
        row = next(source, None)
        if row is None:
            return False
        name, base, path = row
        entry._obj.szModule = name
        entry._obj.modBaseAddr = base
        entry._obj.szExePath = path
        return True
    monkeypatch.setattr(process.kernel32, 'CreateToolhelp32Snapshot', lambda *a: 77)
    monkeypatch.setattr(process.kernel32, 'Module32FirstW', advance)
    monkeypatch.setattr(process.kernel32, 'Module32NextW', advance)
    monkeypatch.setattr(process.kernel32, 'CloseHandle', closed.append)
    return process, closed


def test_unrelated_duplicate_dll_does_not_block_exact_client_module_binding(monkeypatch):
    process, closed = _mock_modules(monkeypatch, [
        ('GameAssembly.dll', 0x10000, 'D:/game/GameAssembly.dll'),
        ('MachineIdLib.dll', 0x20000, 'D:/game/sdk1/MachineIdLib.dll'),
        ('MachineIdLib.dll', 0x30000, 'D:/game/sdk2/MachineIdLib.dll'),
        ('UnityPlayer.dll', 0x40000, 'D:/game/UnityPlayer.dll')])
    modules = process.process_modules(123)
    assert modules['GameAssembly.dll'] == (0x10000, 'D:/game/GameAssembly.dll')
    assert modules['UnityPlayer.dll'] == (0x40000, 'D:/game/UnityPlayer.dll')
    assert modules['MachineIdLib.dll'] == (0, '')  # Neither duplicate is silently selected.
    assert closed == [77]


@pytest.mark.parametrize('name', ['GameAssembly.dll', 'UnityPlayer.dll'])
def test_duplicate_critical_client_module_still_refuses_and_closes_snapshot(monkeypatch, name):
    process, closed = _mock_modules(monkeypatch, [
        (name, 0x10000, 'D:/one/'+name), (name, 0x20000, 'D:/two/'+name)])
    with pytest.raises(OSError, match='ambiguous loaded module'):
        process.process_modules(123)
    assert closed == [77]


def test_lookup_of_ambiguous_unrelated_module_never_selects_arbitrary_copy(monkeypatch):
    process, closed = _mock_modules(monkeypatch, [
        ('MachineIdLib.dll', 0x20000, 'D:/game/sdk1/MachineIdLib.dll'),
        ('MachineIdLib.dll', 0x30000, 'D:/game/sdk2/MachineIdLib.dll')])
    assert process.find_module_base(123, 'MachineIdLib.dll') is None
    assert closed == [77]
