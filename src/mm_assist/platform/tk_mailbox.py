"""后台只投递数据；所有 Tk 调用由创建者线程执行。"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


class TkMailbox:
    def __init__(self, parent, *, stop, interval_ms: int = 100):
        self._parent = parent
        self._stop = stop
        self._interval_ms = interval_ms
        self._lock = threading.Lock()
        self._pending = {}
        parent.after(interval_ms, self._drain)

    def post(self, key, callback, *args) -> None:
        with self._lock:
            if not self._stop.is_set():
                # 界面阻塞时每种更新只留最新一份，不堆积旧画面。
                self._pending.pop(key, None)
                self._pending[key] = (callback, args)

    def _drain(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
        if self._stop.is_set():
            return
        for callback, args in pending.values():
            if self._stop.is_set():
                return
            try:
                callback(*args)
            except Exception:
                log.exception('状态栏界面更新失败')
        self._parent.after(self._interval_ms, self._drain)
