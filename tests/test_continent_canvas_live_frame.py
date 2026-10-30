"""A city chip remains on the continent canvas; prove pixels before navigation."""
import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT, AppConfig
from mm_assist.nav import map_atlas as maps, map_localize as localize
from mm_assist.perception.ocr import OcrLine
from mm_assist.platform.window import WindowInfo


def frame(name='continent_city_chip.png'):
    img = cv2.imread(str(ROOT / 'tests/fixtures/process_stance' / name))
    assert img is not None
    return img


def test_original_failure_canvas_matches_independent_atlas_regions():
    atlas, _ = localize.load_continent_atlas(ROOT / 'knowledge/maps/ulara')
    assert localize.continent_canvas_matches(frame(), atlas)


@pytest.mark.parametrize('fixture', ['cloth_standalone.png', 'wood_standalone.png', 'metal_standalone.png'])
def test_facility_cannot_be_promoted_to_continent(fixture):
    atlas, _ = localize.load_continent_atlas(ROOT / 'knowledge/maps/ulara')
    img = cv2.imread(str(ROOT / 'tests/fixtures/process_stance' / fixture))
    assert not localize.continent_canvas_matches(img, atlas)


def test_blank_or_unavailable_pixels_never_prove_overview():
    atlas, _ = localize.load_continent_atlas(ROOT / 'knowledge/maps/ulara')
    for img in (None, np.zeros_like(frame())):
        assert not localize.continent_canvas_matches(img, atlas)
    assert not localize.continent_canvas_matches(frame(), None)


def test_laerte_split_title_failure_requires_ascent_before_atlas(monkeypatch):
    img = frame('laerte_region_split_title.png')
    win = WindowInfo(0, 'offline', 0, 0, img.shape[1], img.shape[0])
    lines = [OcrLine(text=text, confidence=1, box=[[x, 35], [x+width, 35],
            [x+width, 65], [x, 65]]) for text,x,width in
            [('歐拉大陸', 80, 140), ('萊爾特丘陵', 270, 190)]]
    calls = []
    def ascent(*args):
        calls.append('ascent')
        return win, lines, 'region'
    monkeypatch.setattr(maps, 'click_ulara_text_and_ocr', ascent)
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **k: (win, img))
    monkeypatch.setattr(localize, 'drag_until_near', lambda **k: pytest.fail('region cannot use atlas drag'))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    result = maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=lines, img=img)
    assert calls == ['ascent']
    assert not result['ok'] and result['stage'] == 'continent_region_stuck'


def test_new_tir_canvas_at_atlas_border_is_located_and_proven():
    img = frame('continent_tir_border.png')
    atlas, meta = localize.load_continent_atlas(ROOT / 'knowledge/maps/ulara')
    loc = localize.localize_viewport_zoom_search(img, atlas, min_score=.60)
    assert loc.ok
    assert loc.zoom == pytest.approx(.547, abs=.01)
    assert loc.match_x == pytest.approx(480, abs=10)
    # The north extension shifts the old coordinate origin; terrain stays put.
    shift_y = meta.get('atlas_update', {}).get('coordinate_shift', [0, 0])[1]
    assert loc.match_y == pytest.approx(703 + shift_y, abs=10)
    assert localize.continent_canvas_matches(img, atlas)


def test_unrelated_synthetic_pixels_have_no_feature_fallback():
    img = np.random.default_rng(7).integers(0, 255, frame().shape, dtype=np.uint8)
    atlas, _ = localize.load_continent_atlas(ROOT / 'knowledge/maps/ulara')
    assert not localize.localize_viewport_zoom_search(img, atlas, min_score=.60).ok


def test_verified_canvas_skips_false_breadcrumb_ascent(monkeypatch):
    img = frame()
    win = WindowInfo(0, 'fixture', 0, 0, img.shape[1], img.shape[0])
    lines = [OcrLine(text='歐拉大陸／庫漢', confidence=1,
                     box=[[80, 40], [370, 40], [370, 65], [80, 65]])]
    monkeypatch.setattr(maps, 'click_ulara_text_and_ocr',
                        lambda *a, **k: pytest.fail('proven canvas needs no ascent'))
    monkeypatch.setattr(localize, 'drag_until_near',
                        lambda **k: (_ for _ in ()).throw(RuntimeError('atlas reached after independent proof')))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    with pytest.raises(RuntimeError, match='atlas reached after independent proof'):
        maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=lines, img=img)


def test_updated_north_continent_matches_without_lowering_thresholds():
    atlas, meta = localize.load_continent_atlas(ROOT / 'knowledge/maps/ulara')
    img = frame('continent_laerte_update.png')
    loc = localize.localize_viewport_zoom_search(img, atlas, min_score=.60)
    assert loc.ok
    assert localize.continent_canvas_matches(img, atlas, min_score=.60, min_patch_score=.65)
    # The existing city anchor must still project onto its real castle icon.
    target = localize.find_place_xy(meta, '杜巴頓')
    nx, ny = localize.atlas_xy_to_client_norm(loc, *target, img.shape[1], img.shape[0])
    assert nx * img.shape[1] == pytest.approx(610, abs=12)
    assert ny * img.shape[0] == pytest.approx(1270, abs=12)


def test_updated_north_canvas_does_not_click_unchanged_breadcrumb(monkeypatch):
    img = frame('continent_laerte_update.png')
    win = WindowInfo(0, 'fixture', 0, 0, img.shape[1], img.shape[0])
    lines = [OcrLine(text='歐拉大陸／萊爾特丘陵', confidence=1,
                     box=[[80, 40], [480, 40], [480, 65], [80, 65]])]
    monkeypatch.setattr(maps, 'click_ulara_text_and_ocr',
                        lambda *a, **k: pytest.fail('independent north canvas needs no ascent'))
    monkeypatch.setattr(localize, 'drag_until_near',
                        lambda **k: (_ for _ in ()).throw(RuntimeError('north atlas reached')))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    with pytest.raises(RuntimeError, match='north atlas reached'):
        maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=lines, img=img)


def test_current_overview_preserves_pixel_scale_and_castle_projection():
    atlas, meta = localize.load_continent_atlas(ROOT / 'knowledge/maps/ulara')
    assert atlas.shape[:2] == (meta['height'], meta['width'])
    assert meta['atlas_update']['coordinate_scale'] == 1
    img = frame('continent_north_current.png')
    loc = localize.localize_viewport_zoom_search(img, atlas, min_score=.60)
    assert loc.ok
    assert localize.continent_canvas_matches(img, atlas, min_score=.60, min_patch_score=.65)
    target = localize.find_place_xy(meta, '杜巴頓')
    nx, ny = localize.atlas_xy_to_client_norm(loc, *target, img.shape[1], img.shape[0])
    assert nx * img.shape[1] == pytest.approx(675, abs=12)
    assert ny * img.shape[0] == pytest.approx(1063, abs=12)


def test_upper_bankor_landmarks_extend_only_observed_atlas():
    atlas, meta = localize.load_continent_atlas(ROOT / 'knowledge/maps/ulara')
    img = frame('continent_bankor_upper.png')
    # New northern landmarks were outside the old atlas. Validate the pixels
    # using the offline pose fitted to the pre-existing southern overlap.
    transform = np.array(meta['atlas_update']['upper_source_to_atlas'])
    assert transform[0, 0] == pytest.approx(.547, abs=.001)
    bankor = transform[:, :2] @ (699, 354) + transform[:, 2]
    assert bankor[0] == pytest.approx(872, abs=4)
    shift_y = meta['atlas_update']['coordinate_shift'][1]
    assert bankor[1] == pytest.approx(-267 + shift_y, abs=4)
    for center in ((699, 354), (602, 461), (575, 825), (565, 1190)):
        target = transform[:, :2] @ center + transform[:, 2]
        # Compare spatially separate city/mine/cloud/Laerte patches; do not
        # infer runtime page recognition from capture registration alone.
        x, y = np.round(target).astype(int)
        patch_transform = transform.copy()
        patch_transform[:, 2] -= (x - 24, y - 24)
        expected = cv2.warpAffine(img, patch_transform, (48, 48))
        actual = atlas[y - 24:y + 24, x - 24:x + 24]
        assert actual.shape == expected.shape
        score = cv2.matchTemplate(actual, expected, cv2.TM_CCOEFF_NORMED)[0, 0]
        assert score > .90


@pytest.mark.parametrize('fixture', ['continent_bankor_upper.png', 'continent_bankor_failure.png'])
def test_bankor_overview_has_distributed_localization_and_independent_canvas(fixture):
    img = frame(fixture)
    atlas, _ = localize.load_continent_atlas(ROOT / 'knowledge/maps/ulara')
    loc = localize.localize_viewport_zoom_search(img, atlas, min_score=.60)
    assert loc.ok, loc.note
    assert loc.zoom == pytest.approx(.547, abs=.005)
    assert localize.continent_canvas_matches(img, atlas, min_score=.60, min_patch_score=.65)


def test_bankor_failure_reaches_city_navigation_without_breadcrumb_click(monkeypatch):
    img = frame('continent_bankor_failure.png')
    win = WindowInfo(0, 'fixture', 0, 0, img.shape[1], img.shape[0])
    lines = [OcrLine(text='歐拉大陸／班克爾', confidence=1,
                     box=[[80, 40], [480, 40], [480, 65], [80, 65]])]
    monkeypatch.setattr(maps, 'click_ulara_text_and_ocr',
                        lambda *a, **k: pytest.fail('proven Bankor canvas needs no ascent'))
    monkeypatch.setattr(localize, 'drag_until_near',
                        lambda **k: (_ for _ in ()).throw(RuntimeError('Bankor atlas reached')))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    with pytest.raises(RuntimeError, match='Bankor atlas reached'):
        maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=lines, img=img)


def test_bankor_town_failure_is_not_continent_and_ascends_once(monkeypatch):
    import json
    town = frame('bankor_town_map.png')
    overview = frame('continent_bankor_upper.png')
    row = json.loads((ROOT/'tests/fixtures/process_stance/bankor_town_map.json').read_text(encoding='utf-8'))
    lines = [OcrLine(**line) for line in row['lines']]
    win = WindowInfo(0, 'fixture', 0, 0, town.shape[1], town.shape[0])
    assert maps._map_level(maps._blob(lines), lines, win) == 'town'
    atlas, _ = localize.load_continent_atlas(ROOT/'knowledge/maps/ulara')
    assert not localize.continent_canvas_matches(town, atlas)
    calls = []
    def ascend(*args, **kwargs):
        calls.append(1)
        assert len(calls) == 1, 'fresh overview needs no second breadcrumb input'
        return win, lines, 'region'
    ascend.last_changed = True
    monkeypatch.setattr(maps, 'click_ulara_text_and_ocr', ascend)
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **k: (win, overview))
    def drag(**kwargs):
        assert calls == [1], 'town must ascend before continent navigation'
        raise RuntimeError('Bankor town correctly ascended')
    monkeypatch.setattr(localize, 'drag_until_near', drag)
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    with pytest.raises(RuntimeError, match='correctly ascended'):
        maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=lines, img=town)


def test_bankor_town_landmarks_are_independent_of_text_and_reject_wrong_pages():
    town = frame('bankor_town_map.png')
    assert localize.known_town_canvas_matches(town)
    for name in ('continent_bankor_upper.png', 'continent_bankor_failure.png', 'wood_standalone.png'):
        assert not localize.known_town_canvas_matches(frame(name))
    # Two individually valid landmarks under conflicting translations cannot
    # identify the town view. Keeping one real patch is insufficient.
    moved = town.copy()
    moved[600:760, 600:830] = 0
    moved[650:810, 600:830] = town[600:760, 600:830]
    assert not localize.known_town_canvas_matches(moved)


def test_bankor_town_empty_stack_is_map_in_display_and_direct_navigation(monkeypatch):
    from types import SimpleNamespace
    from mm_assist.life_plan.node_inspect import compute_inspect
    from mm_assist.life_execute.ui_logic import resolve_navigation
    from mm_assist.memory.readers.ui_stack import UiStackSnap
    town = frame('bankor_town_map.png')
    view = SimpleNamespace(ui_stack_names=[], ui_stack_ready=True, ui_stack_empty_ok=True)
    monkeypatch.setattr('mm_assist.life_plan.node_inspect._ocr_blob',
                        lambda *a, **k: pytest.fail('town terrain must not need OCR'))
    assert compute_inspect(town, view=view)['node'] == 'map'
    class Reader:
        def read(self, **kwargs):
            return UiStackSnap(names=[], ready=True, empty_confirmed=True)
    assert resolve_navigation(town, reader=Reader()).node == 'map'
    view.ui_stack_ready = False
    assert compute_inspect(town, view=view)['node'] == ''


def test_resume_from_bankor_town_closes_verified_map_before_hud(monkeypatch):
    from mm_assist.life_execute import ui_logic, ui_nav
    from mm_assist.memory.readers.ui_stack import UiStackSnap
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests
    town = frame('bankor_town_map.png')
    class Reader:
        def read(self, **kwargs):
            return UiStackSnap(names=[], ready=True, empty_confirmed=True)
    state = {'closed': False}
    def resolve(img):
        if state['closed']:
            return ui_logic.UiWhere('main_hud', 0, 0, 0, 'fixture', stack_ready=True)
        return ui_logic.resolve_navigation(img, reader=Reader())
    clicks = []
    def click(x, y):
        clicks.append((x, y))
        assert x > 1800 and y < 100, 'must recognize map X, never click HUD menu on map'
        state['closed'] = True
        return True
    monkeypatch.setattr(ui_nav, 'resolve_navigation', resolve)
    reset_stage_gates_for_tests()
    try:
        result = ui_nav.ensure_node('main_hud', grab=lambda: town, click=click, sleep_fn=lambda _: None)
        assert result.ok, result
        assert len(clicks) == 1
    finally:
        reset_stage_gates_for_tests()


def test_current_nerys_portrait_rejects_player_pin_and_requires_fresh_target():
    img = cv2.imread(str(ROOT/'tests/fixtures/repair/dunbarton_nerys_daisy.png'))
    win = WindowInfo(0,'fixture',0,0,img.shape[1],img.shape[0])
    hit = maps.registered_nerys_avatar(img,win)
    assert hit is not None
    assert hit.nx*win.width == pytest.approx(925,abs=3)
    assert hit.ny*win.height == pytest.approx(638,abs=3)
    img[590:685,880:970] = 0
    assert maps.registered_nerys_avatar(img,win) is None  # Daisy is still visible.
    wrong = cv2.imread(str(ROOT/'tests/fixtures/repair/dunbarton_player_pin_daisy.png'))
    assert maps.registered_nerys_avatar(wrong,win) is None


def test_map_location_panel_instance_cannot_be_promoted_to_hud():
    from mm_assist.memory.readers.ui_stack import stack_node, left_main_hud
    names = ['MapLocationPanelUIInstance']
    assert stack_node(names,ready=True) == 'map'
    assert left_main_hud(names)


@pytest.mark.parametrize('vanished', [False,True])
def test_town_nerys_click_uses_fresh_portrait_and_never_player_projection(monkeypatch,vanished):
    img = cv2.imread(str(ROOT/'tests/fixtures/repair/dunbarton_nerys_daisy.png'))
    win = WindowInfo(0,'fixture',0,0,img.shape[1],img.shape[0])
    lost = img.copy()
    lost[590:685,880:970] = 0
    frames = iter([img,img,lost if vanished else img])
    monkeypatch.setattr('mm_assist.perception.game_frame.ensure_game_frame',lambda *a,**k: None)
    monkeypatch.setattr(maps,'_grab_map_frame',lambda *a,**k:(win,next(frames)))
    monkeypatch.setattr(maps,'_snap_map_light',lambda *a,**k:(win,img,[]))
    lines = [OcrLine('奈麗絲',1,[[1200,1000],[1300,1000],[1300,1030],[1200,1030]]),
             OcrLine('前往此處',1,[[1400,1300],[1500,1300],[1500,1350],[1400,1350]])]
    monkeypatch.setattr(maps,'snap',lambda *a,**k:(win,img,lines,None))
    monkeypatch.setattr(maps,'find_npc_avatar',lambda *a,**k: None)
    monkeypatch.setattr(localize,'load_town_atlas',lambda *a:pytest.fail('visible target needs no atlas drag'))
    monkeypatch.setattr(maps,'_drag_map',lambda *a,**k:pytest.fail('no blind drag'))
    monkeypatch.setattr(maps,'_wait_action_effect',lambda *a,**k:(win,img,True,10))
    clicks = []
    monkeypatch.setattr(maps,'_click_norm',lambda w,x,y:clicks.append((x*w.width,y*w.height)))
    result = maps.town_pixel_click_npc(AppConfig(),win)
    if vanished:
        assert result['stage'] == 'npc_avatar_unverified' and not result['ok']
        assert not clicks
    else:
        assert result['ok'],result
        assert len(clicks) == 1
        assert clicks[0] == pytest.approx((925,638),abs=3)
