"""Bind memory layouts to explicitly verified client files. No auto-enrolment."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from mm_assist.knowledge_paths import ui_config


class BuildRejected(OSError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


_LOCK = threading.RLock()
_CACHE: dict[tuple, dict] = {}
_LATEST: dict[int, dict] = {}
_BINDINGS: dict[int, dict] = {}
_LIFETIMES: dict[int, tuple] = {}
_FILE_HASH_CACHE: dict[tuple, str] = {}


def _stat(path: Path) -> tuple:
    s = path.stat()
    return (str(path.resolve()), s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def fingerprint_files(files: dict[str, Path], *, should_stop=None) -> dict[str, str]:
    out = {}
    initial = {name: _stat(Path(path)) for name, path in files.items()}
    for name, raw_path in sorted(files.items()):
        path = Path(raw_path)
        before = _stat(path)
        digest = hashlib.sha256()
        with path.open('rb') as source:
            while True:
                if should_stop and should_stop():
                    raise BuildRejected('cancelled')
                block = source.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        if before != _stat(path):
            raise BuildRejected('client_file_changed')
        out[name] = digest.hexdigest()
    if initial != {name: _stat(Path(path)) for name, path in files.items()}:
        raise BuildRejected('client_file_changed')
    return out


def client_files(modules: dict[str, tuple[int, str]]) -> dict[str, Path]:
    try:
        ga = Path(modules['GameAssembly.dll'][1]).resolve(strict=True)
        unity = Path(modules['UnityPlayer.dll'][1]).resolve(strict=True)
    except (KeyError, OSError) as exc:
        # Check installation association even when a diagnostic fixture has no files.
        if 'GameAssembly.dll' in modules and 'UnityPlayer.dll' in modules:
            if Path(modules['GameAssembly.dll'][1]).parent != Path(modules['UnityPlayer.dll'][1]).parent:
                raise BuildRejected('module_installation_mismatch') from exc
        raise BuildRejected('client_files_missing') from exc
    if ga.parent != unity.parent:
        raise BuildRejected('module_installation_mismatch')
    return {'GameAssembly.dll': ga, 'UnityPlayer.dll': unity,
            'global-metadata.dat': ga.parent/'MabinogiMobile_Data/il2cpp_data/Metadata/global-metadata.dat'}


def configuration_files() -> dict[str, Path]:
    return {p.name: p for p in sorted(ui_config('il2cpp_scene.json').parent.glob('il2cpp_*.json'))
            if p.name != 'il2cpp_client_builds.json'}


def configuration_digest() -> str:
    content = fingerprint_files(configuration_files())
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _session_signature() -> tuple:
    try:
        files = configuration_files()
        return tuple(_stat(p) for p in (*files.values(), ui_config('il2cpp_client_builds.json')))
    except OSError:
        return ('configuration_unavailable',)


def _cached_fingerprints(files: dict[str, Path]) -> dict[str, str]:
    out = {}
    for name, path in files.items():
        key = _stat(path)
        value = _FILE_HASH_CACHE.get(key)
        if value is None:
            value = fingerprint_files({name: path})[name]
            if key != _stat(path):
                raise BuildRejected('client_file_changed')
            _FILE_HASH_CACHE[key] = value
            while len(_FILE_HASH_CACHE) > 64:
                _FILE_HASH_CACHE.pop(next(iter(_FILE_HASH_CACHE)))
        out[name] = value
    return out


def select_build(files: dict[str, str], config_sha256: str, builds: list[dict]) -> str:
    candidates = [b for b in builds if b.get('files') == files and b.get('id') and b.get('evidence')]
    if not candidates:
        raise BuildRejected('client_build_unregistered')
    matched = [b for b in candidates if b.get('config_sha256') == config_sha256]
    if len(matched) != 1:
        raise BuildRejected('configuration_changed' if not matched else 'client_build_ambiguous')
    return str(matched[0]['id'])


def verify_client(handle, pid: int) -> dict:
    """First attach hashes files; subsequent attaches use stat-keyed, process-bound results."""
    from mm_assist.memory.process import process_birth, process_handle_alive, process_modules
    with _LOCK:
        candidate = {}
        try:
            if _SESSION_SIGNATURE != _session_signature():
                # Reader objects may retain parsed offsets. Even an explicitly
                # re-registered pair must not hot-load over those old objects.
                raise BuildRejected('client_session_changed')
            if not process_handle_alive(handle):
                raise BuildRejected('process_binding_changed')
            birth = process_birth(handle)
            modules = process_modules(pid)
            files = client_files(modules)
            cfg_files = configuration_files()
            registry = ui_config('il2cpp_client_builds.json')
            stamps = tuple(_stat(p) for p in (*files.values(), *cfg_files.values(), registry))
            base = int(modules['GameAssembly.dll'][0])
            key = (pid, birth, base, stamps)
            row = _CACHE.get(key)
            if row is None:
                _LATEST[pid] = {'pid': pid, 'reason': 'client_build_hashing', 'verified': False}
                hashes = _cached_fingerprints(files)
                config_hash = configuration_digest()
                candidate = {'files': hashes, 'config_sha256': config_hash, 'birth': birth, 'module_base': base}
                builds = json.loads(registry.read_text(encoding='utf-8'))['builds']
                build_id = select_build(hashes, config_hash, builds)
                if stamps != tuple(_stat(p) for p in (*files.values(), *cfg_files.values(), registry)):
                    raise BuildRejected('client_file_changed')
                row = {'pid': int(pid), 'birth': birth, 'module_base': base,
                       'build_id': build_id, 'config_sha256': config_hash,
                       'module_paths': {n: (int(modules[n][0]), str(Path(modules[n][1]).resolve()))
                                        for n in ('GameAssembly.dll', 'UnityPlayer.dll')},
                       'files': hashes, 'verified': True, 'reason': 'verified'}
                _CACHE.clear()  # One active client; never retain unbounded historical processes.
                _CACHE[key] = row
            current_modules = process_modules(pid)
            if (not process_handle_alive(handle) or process_birth(handle) != birth
                    or any(current_modules.get(n) != modules.get(n) for n in ('GameAssembly.dll', 'UnityPlayer.dll'))):
                raise BuildRejected('process_binding_changed')
            _LATEST[pid] = {**row, 'checked_at': time.time()}
            # All handles of the same lifetime share discovery hints. A new lifetime
            # must not inherit initialized klass pointers or scan cursors by PID alone.
            lifetime = (birth, base, row['build_id'], row['config_sha256'])
            if pid in _LIFETIMES and _LIFETIMES[pid] != lifetime:
                from mm_assist.memory.il2cpp import reset_scan_cache
                reset_scan_cache(handle)
            _LIFETIMES[pid] = lifetime
            while len(_LIFETIMES) > 8:
                _LIFETIMES.pop(next(iter(_LIFETIMES)))
            _BINDINGS[int(handle)] = {'identity': dict(row), 'stamps': stamps,
                                     'paths': tuple((*files.values(), *cfg_files.values(), registry)),
                                     'config_names': tuple(cfg_files)}
            return dict(row)
        except Exception as exc:
            reason = exc.reason if isinstance(exc, BuildRejected) else 'client_build_unreadable'
            _LATEST.clear()
            _LATEST[pid] = {**candidate, 'pid': pid, 'reason': reason, 'verified': False, 'detail': str(exc)}
            if isinstance(exc, BuildRejected):
                raise
            raise BuildRejected(reason) from exc


def bound_identity(handle, pid: int) -> dict:
    """Recheck the already bound handle and file stamps without hashing or heap scans."""
    from mm_assist.memory.process import process_birth, process_handle_alive, process_modules
    with _LOCK:
        binding = _BINDINGS.get(int(handle))
        if binding is None:
            _LATEST[pid] = {'pid': pid, 'reason': 'process_binding_missing', 'verified': False}
            raise BuildRejected('process_binding_missing')
        row = binding['identity']
        if (row['pid'] != pid or not process_handle_alive(handle)
                or process_birth(handle) != row['birth']):
            _LATEST[pid] = {'pid': pid, 'reason': 'process_binding_changed', 'verified': False}
            raise BuildRejected('process_binding_changed')
        modules = process_modules(pid)
        if any(name not in modules or (int(modules[name][0]), str(Path(modules[name][1]).resolve())) != expected
               for name, expected in row['module_paths'].items()):
            _LATEST[pid] = {'pid': pid, 'reason': 'process_binding_changed', 'verified': False}
            raise BuildRejected('process_binding_changed')
        try:
            files_changed = (binding['stamps'] != tuple(_stat(p) for p in binding['paths'])
                             or binding['config_names'] != tuple(configuration_files()))
        except OSError:
            files_changed = True
        if files_changed:
            _LATEST[pid] = {'pid': pid, 'reason': 'client_file_changed', 'verified': False}
            raise BuildRejected('client_file_changed')
        return dict(row)


def forget_handle(handle) -> None:
    with _LOCK:
        _BINDINGS.pop(int(handle), None)


def discovery_token(handle) -> tuple:
    """Cache namespace only. No process facts or readiness are inferred from it."""
    binding = _BINDINGS.get(int(handle))
    if binding is None:
        return ('unbound', int(handle))
    row = binding['identity']
    return (row['pid'], row['birth'], row['module_base'], row['build_id'], row['config_sha256'])


def build_diagnostic(pid: int | None = None) -> dict:
    # A status read must stay nonblocking while a first attach hashes large files.
    rows = dict(_LATEST)
    return dict(rows.get(pid, {}) if pid is not None else (list(rows.values())[-1] if rows else {}))


# Imported by the shared IL2CPP layer before reader objects parse their layouts.
_SESSION_SIGNATURE = _session_signature()
