"""Formal action completion: two continuous two-second stp=0 segments."""
from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from mm_assist.life_plan.types import ExecuteResult
from mm_assist.perception.state_ui import DEFAULT_MAX_AGE_S
from mm_assist.life_plan.execution_records import trace_stp

SEGMENT_S = 2.0
SEGMENTS = 2
_STOP_SOURCE = None


@contextmanager
def completion_stp_source(reader):
    """Bind the watch's existing sampler across its worker threads; no game I/O."""
    global _STOP_SOURCE
    previous = _STOP_SOURCE
    _STOP_SOURCE = reader
    try:
        yield
    finally:
        if _STOP_SOURCE is reader:
            _STOP_SOURCE = previous


@dataclass(frozen=True)
class StpSample:
    stp: int | None
    timestamp: float
    pid: int
    fresh: bool
    generation: int = 0


@trace_stp
def read_completion_stp() -> StpSample:
    reader = _STOP_SOURCE
    if reader is not None:
        try:
            row = reader()
            stamp = float(row['timestamp'])
            value = row['stp']
            generation = int(row['generation'])
            fresh = (value in (0, 1) and math.isfinite(stamp) and stamp > 0
                     and 0 <= time.time() - stamp <= DEFAULT_MAX_AGE_S)
            return StpSample(value if fresh else None, stamp, os.getpid(), fresh, generation)
        except Exception:
            # A bound source is authoritative, including unknown/closed/error.
            return StpSample(None, 0, os.getpid(), False)
    from mm_assist.perception.state_ui import read_state_ui
    view = read_state_ui()
    if view is None:
        return StpSample(None, 0, 0, False)
    raw = getattr(view, 'raw', None)
    raw = raw if isinstance(raw, dict) else {}
    try:
        stamp = float(raw.get('ts_unix') or 0)
    except (ValueError, TypeError):
        stamp = 0
    pid = getattr(view, 'pid', 0)
    age = time.time() - stamp
    fresh = bool(getattr(view, 'fresh', False) and getattr(view, 'pid_alive', False)
                 and pid == os.getpid() and math.isfinite(stamp) and stamp > 0
                 and 0 <= age <= DEFAULT_MAX_AGE_S)
    return StpSample(getattr(view, 'stp', None), stamp, pid, fresh)


@dataclass
class IdleSegments:
    zero_since: float | None = None
    last_at: float | None = None
    last_stamp: float = 0
    pid: int = 0
    generation: int = 0

    def reset(self) -> None:
        self.zero_since = None
        self.last_at = None
        self.last_stamp = 0
        self.pid = 0
        self.generation = 0

    def observe(self, sample: StpSample, now: float) -> bool:
        if (not math.isfinite(now) or not sample.fresh or sample.stp != 0 or not math.isfinite(sample.timestamp)
                or sample.timestamp <= 0 or sample.pid <= 0):
            self.reset()
            return False
        if self.last_at is not None and (
            now < self.last_at or now - self.last_at > DEFAULT_MAX_AGE_S
            or sample.pid != self.pid or sample.timestamp < self.last_stamp
            or sample.generation != self.generation
        ):
            self.reset()
        if sample.timestamp == self.last_stamp:
            return False
        if self.zero_since is None:
            self.zero_since = now
        self.last_at = now
        self.last_stamp = sample.timestamp
        self.pid = sample.pid
        self.generation = sample.generation
        return now - self.zero_since >= SEGMENT_S * SEGMENTS

    def completed_segments(self, now: float) -> int:
        return 0 if self.zero_since is None else min(SEGMENTS, int((now-self.zero_since)/SEGMENT_S))


@dataclass
class PendingAction:
    item_id: str
    run_started_at: float
    result: ExecuteResult
    actions_complete: bool = False
    idle: IdleSegments = field(default_factory=IdleSegments)
