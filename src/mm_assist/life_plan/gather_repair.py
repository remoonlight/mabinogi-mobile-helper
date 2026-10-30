"""Read-only repair interrupt scoped to the formal gathering executor."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import time

from mm_assist.perception.state_ui import DEFAULT_MAX_AGE_S

REPAIR_MENU = 'IngredientGuideUIInstance'
REPAIR_REQUIRED = 'gather_repair_required'
_WATCH: ContextVar[GatherRepairWatch | None] = ContextVar('gather_repair_watch', default=None)


@dataclass
class GatherRepairWatch:
    win: object
    names: tuple[str, ...] = ()
    sampled_ts: float = 0.0

    def check(self) -> bool:
        if self.names:
            return True
        if not int(getattr(self.win, 'hwnd', 0) or 0):
            return False
        from mm_assist.memory.readers.ui_stack import shared_ui_stack_reader
        try:
            snap = shared_ui_stack_reader().read(force=True)
            stamp = float(snap.sampled_ts)
            if (snap.ready and math.isfinite(stamp) and stamp > 0
                    and 0 <= time.time() - stamp <= DEFAULT_MAX_AGE_S
                    and REPAIR_MENU in snap.names):
                self.names = tuple(snap.names)
                self.sampled_ts = stamp
                return True
        except Exception:
            pass
        return False


@contextmanager
def watch_gather_repair(win):
    watch = GatherRepairWatch(win)
    token = _WATCH.set(watch)
    try:
        yield watch
    finally:
        _WATCH.reset(token)


def gathering_interrupted() -> bool:
    watch = _WATCH.get()
    return watch is not None and watch.check()
