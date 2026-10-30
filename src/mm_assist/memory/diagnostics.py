"""Bounded, read-only status evidence. No scans or process opens on UI reads."""
from __future__ import annotations

import threading
import time
from copy import deepcopy

_LOCK = threading.Lock()
_SCAN = {}
_ACCESS = {}
_SERIAL = 0


def begin_scan(pid: int) -> int:
    global _SERIAL
    with _LOCK:
        _SERIAL += 1
        _SCAN[pid] = {'sequence': _SERIAL, 'pid': pid, 'reason': 'heap_read_busy',
                      'coverage_complete': False, 'bytes_read': 0, 'regions_visited': 0,
                      'candidate_count': 0, 'sampled_at': time.time()}
        while len(_SCAN) > 8:
            _SCAN.pop(next(iter(_SCAN)))
        return _SERIAL


def publish_scan(pid: int, sequence: int, row: dict) -> None:
    with _LOCK:
        if _SCAN.get(pid, {}).get('sequence') == sequence:
            _SCAN[pid] = {**row, 'sequence': sequence, 'pid': pid, 'sampled_at': time.time()}


def latest_scan(pid: int) -> dict:
    with _LOCK:
        return deepcopy(_SCAN.get(pid, {}))


def invalidate_scans(pid=None) -> None:
    with _LOCK:
        if pid is None:
            _SCAN.clear()
        else:
            _SCAN.pop(pid, None)


def note_access_error(pid: int, row: dict) -> None:
    with _LOCK:
        _ACCESS.clear()
        _ACCESS[pid] = {**row, 'sampled_at': time.time()}


def memory_diagnostic(pid: int | None = None, *, stack_error: str = '') -> dict:
    from mm_assist.memory.client_build import build_diagnostic
    build = build_diagnostic(pid)
    pid = int(pid or build.get('pid') or 0)
    with _LOCK:
        access = deepcopy(_ACCESS.get(pid, {}))
    # Access errors expire; a later successful build verification supersedes them.
    if (build.get('verified') and build.get('checked_at', 0) > access.get('sampled_at', 0)
            or time.time() - access.get('sampled_at', 0) > 12):
        access = {}
    scan = latest_scan(pid)
    fresh_scan = 0 <= time.time() - scan.get('sampled_at', 0) <= 12
    reason = (access.get('reason') or (build.get('reason') if not build.get('verified') else '')
              or stack_error or (scan.get('reason') if fresh_scan else '') or '')
    return {'pid': pid, 'reason': reason, 'build': build, 'access': access,
            'scan': scan, 'scan_stale': not fresh_scan}
