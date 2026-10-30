"""Per-read provenance; never a cache of business values or input permission."""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


_CURRENT = ContextVar('memory_read_evidence', default=None)


@dataclass
class RootCandidate:
    accepted: bool = True


@dataclass
class ReadEvidence:
    process: dict
    components: dict = field(default_factory=dict)
    roots: list = field(default_factory=list)
    error: str = ''

    def recheck(self, handle, *, reader=None) -> bool:
        if reader is None:
            from mm_assist.memory.process import read_ptr
            reader = read_ptr
        for root in self.roots:
            if any(reader(handle, address) != value for address, value in root['links']):
                self.error = 'component_binding_changed:' + root['component']
        return not self.error

    def to_dict(self) -> dict:
        return {'process': dict(self.process), 'components': dict(self.components),
                'roots': list(self.roots), 'error': self.error,
                'atomic_business_snapshot': False}


@contextmanager
def bind_evidence(evidence: ReadEvidence):
    token = _CURRENT.set(evidence)
    try:
        yield evidence
    finally:
        _CURRENT.reset(token)


@contextmanager
def root_observation(component: str, handle, root: int, offsets: tuple[int, ...], *, reader=None, accepted=True):
    candidate = RootCandidate(accepted=accepted)
    evidence = _CURRENT.get()
    if evidence is None:
        yield candidate
        return
    if reader is None:
        from mm_assist.memory.process import read_ptr
        reader = read_ptr
    links = [(root + off, reader(handle, root + off)) for off in offsets]
    try:
        yield candidate
    finally:
        if candidate.accepted:
            if not root or any(value is None or reader(handle, address) != value for address, value in links):
                evidence.error = 'component_binding_changed:' + component
            else:
                evidence.roots.append({'component': component, 'instance': int(root),
                                       'sampled_at': time.time(), 'links': links})
