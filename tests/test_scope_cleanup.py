"""Retired controls/inputs disappear; remote repair navigation keeps its guards."""
from types import SimpleNamespace

import pytest

from mm_assist.config import AppConfig
from mm_assist.nav import map_atlas as maps
from mm_assist.safety import input as inputs
from mm_assist.safety.lease import GuardError


def test_retired_control_command_is_rejected_before_dispatch(monkeypatch):
    from mm_assist import cli
    monkeypatch.setattr(cli, 'cmd_transition_accept', lambda *a: pytest.fail('retired UI dispatched'))
    with pytest.raises(SystemExit) as exc:
        cli.main(['transition-accept', 'ui'])
    assert exc.value.code == 2


def test_removed_controls_and_dye_are_not_exported():
    from importlib.util import find_spec
    from mm_assist.benchmarks import transition_accept
    assert not hasattr(transition_accept, 'attach_accept_controls')
    assert not hasattr(transition_accept, 'run_control_panel')
    assert find_spec('mm_assist.dye_highlight') is None
    for name in ('right_drag_client_dye', 'right_arc_client_dye', 'wheel_client_dye', 'drag_client_dye'):
        assert not hasattr(inputs, name)


def test_dye_gesture_path_rejected_without_loading_deleted_dependencies():
    with pytest.raises(RuntimeError, match='未放行'):
        inputs._require_right_wheel_path('dye_palette_pose')


@pytest.mark.parametrize('outcome', ['arrived', 'abort', 'guard'])
def test_remote_navigation_keeps_results_cancel_and_release_without_fake_ack(monkeypatch, outcome):
    from mm_assist.perception import game_frame
    monkeypatch.setattr(game_frame, 'ensure_game_frame', lambda *a, **k: None)
    released, visited = [], []
    monkeypatch.setattr(inputs, 'release_mouse_buttons', lambda *a, **k: released.append(True))
    def check():
        if outcome == 'abort':
            raise inputs.AbortRequested('stopped')
    monkeypatch.setattr(maps, 'check_abort', check)
    from mm_assist.nav import compat
    monkeypatch.setattr(compat, 'check_abort', check)
    monkeypatch.setattr(compat, 'in_production_mode', lambda: False)
    def navigate(cfg, win, **kwargs):
        visited.append(kwargs)
        if outcome == 'guard':
            raise GuardError('blocked')
        return {'ok': True, 'stage': 'arrived', 'verified': True}
    monkeypatch.setattr(maps, '_map_goto_once', navigate)
    result = maps.map_goto(AppConfig(), SimpleNamespace(), region='杜巴頓', npc='奈麗絲')
    assert released
    assert 'acknowledged' not in result and 'valid' not in result
    assert result['ok'] == (outcome == 'arrived')
    if outcome == 'abort':
        assert not visited and result['stage'] == 'aborted'
    else:
        assert len(visited) == 1
        assert visited[0]['region'] == '杜巴頓' and visited[0]['npc'] == '奈麗絲'
        if outcome == 'guard':
            assert result['stage'] == 'guard' and not result['verified']
