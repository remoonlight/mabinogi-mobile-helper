from mm_assist.memory.readers.scene import MemSceneInfo
from mm_assist.perception.scene import skip_fullframe_ocr_on_main_hud


def test_skip_fullframe_ocr_when_menu_closed():
    ms = MemSceneInfo(menu_open=False)
    assert skip_fullframe_ocr_on_main_hud(ms) is True


def test_allow_fullframe_ocr_when_menu_open():
    ms = MemSceneInfo(menu_open=True)
    assert skip_fullframe_ocr_on_main_hud(ms) is False


def test_allow_fullframe_ocr_when_menu_unknown():
    ms = MemSceneInfo(menu_open=None)
    assert skip_fullframe_ocr_on_main_hud(ms) is False
