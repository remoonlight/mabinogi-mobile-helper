from __future__ import annotations

import logging

import pytest

from mm_assist.safety.input import AbortRequested, check_abort
from mm_assist.safety.lease import reset_guard_state, signal_user_cancel
from mm_assist.session_log import LATEST_NAME, start_watch_session_log


def test_check_abort_raises_after_cancel():
    reset_guard_state()
    check_abort()
    signal_user_cancel("unit")
    with pytest.raises(AbortRequested):
        check_abort()
    reset_guard_state()
    check_abort()


def test_start_watch_session_log_writes_latest(tmp_path, monkeypatch):
    monkeypatch.setattr("mm_assist.session_log.ROOT", tmp_path)
    monkeypatch.setattr("mm_assist.session_log._SESSION_ATTACHED", False)
    path = start_watch_session_log()
    latest = tmp_path / "logs" / LATEST_NAME
    assert path.is_file()
    assert latest.is_file()
    text = latest.read_text(encoding="utf-8")
    assert "watch session log" in text
    pack = logging.getLogger("mm_assist")
    for h in list(pack.handlers):
        if isinstance(h, logging.FileHandler) and (
            str(h.baseFilename).endswith(LATEST_NAME) or "watch_" in str(h.baseFilename)
        ):
            pack.removeHandler(h)
            h.close()
    monkeypatch.setattr("mm_assist.session_log._SESSION_ATTACHED", False)
