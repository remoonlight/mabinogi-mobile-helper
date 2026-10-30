"""Explicit field checks against a dump or a supplied read-only runtime reader.

Reports are diagnostic evidence only; they never edit offsets or approve builds.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

_DECL = re.compile(r'\b(?:class|struct|enum)\s+([^\s:{]+).*// TypeDefIndex:')
_FIELD = re.compile(r'^\s*(?:public|private|protected|internal)\s+(.*?)\s+([^\s;=]+);\s*// (0x[0-9a-fA-F]+)\s*$')
_MODIFIERS = re.compile(r'\b(?:static|readonly|const|volatile)\s+')


def _number(value) -> int:
    return int(value, 0) if isinstance(value, str) else int(value)


def _field_result(check: dict, fields: dict) -> dict:
    row = dict(check)
    actual = fields.get(check['field'])
    if actual is None:
        row['status'] = 'field_missing'
    else:
        row.update(actual_offset=actual['offset'], actual_type=actual.get('field_type'))
        row['status'] = ('offset_mismatch' if actual['offset'] != _number(check['offset'])
                         else 'type_mismatch' if check.get('field_type') and check['field_type'] != actual.get('field_type')
                         else 'match')
    return row


def audit_dump(path: Path, checks: list[dict]) -> dict:
    """Stream large dumps and retain only requested types. Ambiguity is explicit."""
    path = Path(path)
    wanted = {c['type'] for c in checks}
    namespace, current = '', None
    types = {}
    digest = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as source:
        for raw in source:
            digest.update(raw)
            line = raw.decode('utf-8', errors='strict').rstrip()
            if line.startswith('// Namespace:'):
                namespace = line.split(':', 1)[1].strip()
                current = None
            match = _DECL.search(line)
            if match:
                simple = match[1]
                full = namespace + '.' + simple if namespace else simple
                current = {} if simple in wanted or full in wanted else None
                if current is not None:
                    types.setdefault(full, []).append(current)
                    if full != simple:
                        types.setdefault(simple, []).append(current)
            elif current is not None:
                field = _FIELD.match(line)
                if field:
                    current[field[2]] = {'offset': int(field[3], 16),
                                         'field_type': _MODIFIERS.sub('', field[1])}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('dump_changed_during_audit')
    rows = []
    for check in checks:
        candidates = types.get(check['type'], [])
        if len(candidates) != 1:
            rows.append({**check, 'status': 'type_missing' if not candidates else 'ambiguous_type'})
        else:
            rows.append(_field_result(check, candidates[0]))
    return {'ok': bool(rows) and all(r['status'] == 'match' for r in rows), 'checks': rows,
            'dump_sha256': digest.hexdigest(), 'dump': str(path.resolve()),
            'ready_for_input': False, 'client_build_verified': False}


def audit_runtime(read, ga_base: int, checks: list[dict], *, max_wall_s: float = 2.0) -> dict:
    """No heap scan. Fixed class ABI derives from this repo's il2cpp.h (x64).

    Every byte read is compared again before accepting the report. FieldInfo is
    32 bytes (name, type, parent, offset, token); parent bounds the field table.
    Runtime type names are not inferred from field offsets. Unsupported ABI fails.
    """
    deadline = time.monotonic() + max_wall_s
    observed = {}
    def exact(address, size):
        if time.monotonic() >= deadline or address < 0x10000 or not 0 < size <= 8192:
            raise ValueError('runtime_audit_budget_or_range')
        data = read(address, size)
        if data is None or len(data) != size:
            raise ValueError('runtime_audit_short_read')
        previous = observed.setdefault((address, size), bytes(data))
        if previous != data:
            raise ValueError('runtime_layout_changed')
        return data
    def ptr(address):
        return int.from_bytes(exact(address, 8), 'little')
    def text(address):
        raw = exact(address, 256)
        end = raw.find(b'\0')
        if end < 0:
            raise ValueError('runtime_string_unterminated')
        return raw[:end].decode('utf-8')
    rows = []
    try:
        layouts = {}
        for check in checks:
            rva = _number(check['typeinfo_rva'])
            if rva not in layouts:
                klass = ptr(ga_base+rva)
                if klass < 0x100000000 or klass % 8:
                    raise ValueError('typeinfo_uninitialized')
                simple, namespace = text(ptr(klass+0x10)), text(ptr(klass+0x18))
                actual_type = namespace + '.' + simple if namespace else simple
                table = ptr(klass+0x80)
                fields = {}
                for index in range(256):
                    field = exact(table+index*32, 32)
                    if int.from_bytes(field[16:24], 'little') != klass:
                        break
                    name = text(int.from_bytes(field[:8], 'little'))
                    if name in fields:
                        raise ValueError('runtime_field_ambiguous')
                    fields[name] = {'offset': int.from_bytes(field[24:28], 'little', signed=True)}
                else:
                    raise ValueError('runtime_field_limit')
                layouts[rva] = (actual_type, fields)
            actual_type, fields = layouts[rva]
            if check['type'] != actual_type:
                rows.append({**check, 'status': 'type_mismatch', 'actual_type': actual_type})
            else:
                # Field type reflection is deliberately not guessed from Il2CppType.
                row = _field_result({k: v for k, v in check.items() if k != 'field_type'}, fields)
                row['field_type_verified'] = False
                rows.append(row)
        for (address, size), value in list(observed.items()):
            if exact(address, size) != value:
                raise ValueError('runtime_layout_changed')
    except (ValueError, OSError, UnicodeError) as exc:
        return {'ok': False, 'error': str(exc), 'checks': rows,
                'ready_for_input': False, 'client_build_verified': False}
    return {'ok': bool(rows) and all(r['status'] == 'match' for r in rows), 'checks': rows,
            'ready_for_input': False, 'client_build_verified': False,
            'scope': 'explicit_class_names_and_field_offsets_only'}


def configured_checks() -> list[dict]:
    import json
    from mm_assist.knowledge_paths import ui_config
    manifest = json.loads(ui_config('il2cpp_layout_checks.json').read_text(encoding='utf-8'))
    checks = []
    for entry in manifest['checks']:
        config = json.loads(ui_config(entry['config']).read_text(encoding='utf-8'))
        checks.append({**entry, 'offset': _number(config['offsets'][entry['key']]),
                       'typeinfo_rva': config['typeinfo_rva'][entry['typeinfo_key']]})
    return checks


def main(argv=None) -> int:
    import argparse
    import json
    from mm_assist.config import ROOT
    from mm_assist.memory.client_build import fingerprint_files, configuration_digest
    parser = argparse.ArgumentParser(description='Read-only memory layout/client audit; never registers builds')
    parser.add_argument('kind', choices=['build', 'layout', 'runtime'])
    parser.add_argument('--client-dir', type=Path)
    parser.add_argument('--dump', type=Path, default=ROOT/'_ref/il2cpp_dump/dump.cs')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--pid', type=int, help='runtime only: exact target game PID')
    args = parser.parse_args(argv)
    if args.kind == 'build':
        if args.client_dir is None:
            parser.error('build requires --client-dir')
        directory = args.client_dir.resolve()
        files = {'GameAssembly.dll': directory/'GameAssembly.dll', 'UnityPlayer.dll': directory/'UnityPlayer.dll',
                 'global-metadata.dat': directory/'MabinogiMobile_Data/il2cpp_data/Metadata/global-metadata.dat'}
        report = {'ok': True, 'files': fingerprint_files(files), 'config_sha256': configuration_digest(),
                  'client_dir': str(directory), 'source': 'installation_files_not_running_modules',
                  'client_build_verified': False, 'ready_for_input': False}
    elif args.kind == 'layout':
        report = audit_dump(args.dump, configured_checks())
    else:
        if not args.pid:
            parser.error('runtime requires --pid (no automatic process selection)')
        from mm_assist.memory.process import (kernel32, process_birth, process_modules,
                                              process_handle_alive, read_mem, find_game_pid)
        from mm_assist.memory.client_build import client_files
        from mm_assist.safety.lease import assert_protocol_ready
        from mm_assist.jobs.bar import require_state_watch_running
        assert_protocol_ready()
        heartbeat = require_state_watch_running()
        if find_game_pid() != args.pid:
            raise ValueError('runtime_audit_target_is_not_current_game')
        # Explicit read-only diagnostic exception: opening this handle never enables
        # production reads, scans or inputs and cannot register a supported build.
        handle = kernel32.OpenProcess(0x410, False, args.pid)
        if not handle:
            import ctypes
            from mm_assist.memory.access_diagnostic import ProcessReadDenied
            raise ProcessReadDenied(args.pid, ctypes.get_last_error())
        try:
            identity = (args.pid, process_birth(handle))
            modules = process_modules(args.pid)
            files = client_files(modules)
            hashes = fingerprint_files(files)
            stamps = {n: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for n, p in files.items()}
            report = audit_runtime(lambda address, size: read_mem(handle, address, size),
                                   modules['GameAssembly.dll'][0], configured_checks())
            if (not process_handle_alive(handle) or identity != (args.pid, process_birth(handle))
                    or modules != process_modules(args.pid) or find_game_pid() != args.pid
                    or stamps != {n: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for n, p in files.items()}):
                report.update(ok=False, error='runtime_audit_binding_changed')
            report.update(process={'pid': args.pid, 'birth': identity[1]}, files=hashes,
                          config_sha256=configuration_digest(), watch_pid=heartbeat['pid'])
        finally:
            kernel32.CloseHandle(handle)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded+'\n', encoding='utf-8')
    print(encoded)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
