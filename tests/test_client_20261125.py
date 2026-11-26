"""The current client must match its recorded layout and exact registry pair."""
import json
from pathlib import Path

import pytest


def test_current_client_layout_and_registry():
    from mm_assist.knowledge_paths import ui_config
    from mm_assist.memory import client_build

    fixture = json.loads((Path(__file__).parent / 'fixtures/client_20261125_layout.json').read_text(encoding='utf-8'))
    for filename, key, _, _, offset, _ in fixture['fields']:
        cfg = json.loads(ui_config(filename).read_text(encoding='utf-8'))
        assert int(cfg['offsets'][key], 0) == offset, (filename, key)
    for filename, key, address, _ in fixture['typeinfo']:
        cfg = json.loads(ui_config(filename).read_text(encoding='utf-8'))
        assert int(cfg['typeinfo_rva'][key], 0) == address, (filename, key)
    cfg = json.loads(ui_config('il2cpp_facility_ui.json').read_text(encoding='utf-8'))
    assert int(cfg['semantic_model']['typeinfo_rva'], 0) == fixture['facility_name_table_rva']
    registry = json.loads(ui_config('il2cpp_client_builds.json').read_text(encoding='utf-8'))['builds']
    digest = client_build.configuration_digest()
    assert client_build.select_build(fixture['files'], digest, registry) == 'tw-20261125-acfcc855'
    changed = {**fixture['files'], 'GameAssembly.dll': '0' * 64}
    with pytest.raises(client_build.BuildRejected, match='client_build_unregistered'):
        client_build.select_build(changed, digest, registry)
    with pytest.raises(client_build.BuildRejected, match='configuration_changed'):
        client_build.select_build(fixture['files'], '0' * 64, registry)
