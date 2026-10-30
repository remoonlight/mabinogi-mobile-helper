"""Permission evidence only; never enables privileges or changes elevation."""
from __future__ import annotations

import os


def _token_info(process, prefix: str, out: dict) -> None:
    import win32security as security
    token = None
    try:
        token = security.OpenProcessToken(process, security.TOKEN_QUERY)
    except Exception as exc:
        out[prefix + '_query_error'] = type(exc).__name__ + ':' + str(exc)
        return
    try:
        try:
            out[prefix + '_elevated'] = bool(security.GetTokenInformation(token, security.TokenElevation))
        except Exception as exc:
            out[prefix + '_elevation_error'] = type(exc).__name__ + ':' + str(exc)
        try:
            sid = security.GetTokenInformation(token, security.TokenIntegrityLevel)
            if isinstance(sid, tuple):
                sid = sid[0]
            out[prefix + '_integrity_rid'] = int(sid.GetSubAuthority(sid.GetSubAuthorityCount()-1))
        except Exception as exc:
            out[prefix + '_integrity_error'] = type(exc).__name__ + ':' + str(exc)
    finally:
        if token is not None:
            try:
                token.Close()
            except Exception as exc:
                out[prefix + '_close_error'] = type(exc).__name__ + ':' + str(exc)


def collect_access(pid: int) -> dict:
    import win32api
    out = {'tool_elevated': None, 'target_elevated': None,
           'tool_integrity_rid': None, 'target_integrity_rid': None}
    _token_info(win32api.GetCurrentProcess(), 'tool', out)
    target = None
    try:
        target = win32api.OpenProcess(0x1000, False, int(pid))
        _token_info(target, 'target', out)
    except Exception as exc:
        out['target_query_error'] = type(exc).__name__ + ':' + str(exc)
    finally:
        if target is not None:
            try:
                target.Close()
            except Exception as exc:
                out['target_close_error'] = type(exc).__name__ + ':' + str(exc)
    return out


class ProcessReadDenied(OSError):
    def __init__(self, pid: int, error: int):
        super().__init__(error, f'OpenProcess read-only failed pid={pid} err={error}')
        self.winerror = int(error)
        self.diagnostic = {}
        try:
            self.diagnostic.update(collect_access(pid))
        except Exception as exc:
            self.diagnostic['diagnostic_error'] = type(exc).__name__ + ':' + str(exc)
        # Ancillary queries cannot replace the original failed request.
        self.diagnostic.update(target_pid=int(pid), tool_pid=os.getpid(), access=0x410,
                               win_error=int(error), reason='process_read_denied' if error == 5 else 'process_read_failed')
