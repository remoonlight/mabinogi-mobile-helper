"""Replay the HUD migration and check the current game's recorded layout."""
from pathlib import Path
import json

def test_hud_action_uses_control_key_after_root_object(monkeypatch):
    from mm_assist.memory.readers import hud_action as hud
    from mm_assist.memory import process

    reader = hud.Il2CppHudActionReader()
    obj, base, button, layer = 0x200000000, 0x300000000, 0x400000000, 0x500000000
    reader._h, reader._ga, reader._pid = 7, 8, 9
    monkeypatch.setattr(reader, '_ensure', lambda: True)
    monkeypatch.setattr(reader, '_klass', lambda n: (1 if n == 'HudActionUIComponent' else 2 if n == 'BasicHudActionButtonControl' else None, None))
    monkeypatch.setattr(hud, 'scan_klass_objects', lambda h,k,**kw: [layer] if k == 1 else [obj] if k == 2 else [])
    pointers = {obj+0x18: base, base+0x38: button}
    monkeypatch.setattr(hud, 'read_ptr', lambda h,p: pointers.get(p))
    monkeypatch.setattr(hud, 'read_u8', lambda h,p: 1)
    monkeypatch.setattr(hud, 'read_u32', lambda h,p: {obj+0x38: 0xf1234567, obj+0x40: 12}.get(p))
    monkeypatch.setattr(process, 'read_floats', lambda h,p,n: [7.5] if p == layer+0x70 else [99.0])
    try:
        frame = reader.capture(need_rect=False)
        assert frame.buttons[0].control_key_id == 12
        assert frame.ui_components[0]['cooldownRemaining'] == 7.5
    finally:
        reader._h = None


def _current_layout():
    path = Path(__file__).parent / 'fixtures/client_20261125_layout.json'
    return json.loads(path.read_text(encoding='utf-8'))


def test_current_configuration_matches_recorded_metadata_fields():
    from mm_assist.knowledge_paths import ui_config
    from mm_assist.memory import client_build

    fixture = _current_layout()
    for row in fixture['fields']:
        filename, key, type_name, field_name, offset, field_type = row
        cfg = json.loads(ui_config(filename).read_text(encoding='utf-8'))
        assert int(cfg['offsets'][key], 0) == offset, (filename, key, type_name, field_name, field_type)
    for filename,key,address,_ in fixture['typeinfo']:
        cfg = json.loads(ui_config(filename).read_text(encoding='utf-8'))
        assert int(cfg['typeinfo_rva'][key], 0) == address, (filename,key)
    facility = json.loads(ui_config('il2cpp_facility_ui.json').read_text(encoding='utf-8'))
    assert int(facility['semantic_model']['typeinfo_rva'], 0) == fixture['facility_name_table_rva']
    builds = json.loads(ui_config('il2cpp_client_builds.json').read_text(encoding='utf-8'))['builds']
    assert client_build.select_build(
        fixture['files'], client_build.configuration_digest(), builds,
    ) == 'tw-20261125-acfcc855'


def test_removed_channel_map_strings_are_not_restored_from_defaults(tmp_path):
    from mm_assist.memory.readers.scene import load_config

    path = tmp_path/'scene.json'
    path.write_text('{"offsets": {}}', encoding='utf-8')
    offsets = load_config(path)['offsets']
    assert 'ChannelSheetData.MapImagePath' not in offsets
    assert 'ChannelSheetData.MapBackgroundImagePath' not in offsets


def test_reader_loads_registered_framework_type_tokens():
    from mm_assist.memory.readers.scene import load_config

    tokens = load_config().get('framework_type_tokens', {})
    expected = _current_layout()['framework_type_tokens']
    assert {key: int(spec['token'], 0) for key, spec in tokens.items()} == expected
    assert tokens['GameSpaceAPI']['namespace'] == 'MM.Client.Framework.GeographyResources'
