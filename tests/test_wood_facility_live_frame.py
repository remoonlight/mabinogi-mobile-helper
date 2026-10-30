import cv2
from mm_assist.config import ROOT
from mm_assist.life_plan import node_inspect as ni
from mm_assist.life_execute import ui_nav
from mm_assist.life_execute.ui_logic import UiWhere
from mm_assist.perception import process_sense as ps


def _frame(monkeypatch):
    frame = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/wood_standalone.png'))
    assert frame is not None
    ni._reset_facility_fid_cache()
    monkeypatch.setattr(ps, 'ocr_roi', lambda *a, **k: [])
    monkeypatch.setattr(ni, '_ocr_blob', lambda *a, **k: '')
    return frame


def test_standalone_wood_title_is_identified_without_ocr(monkeypatch):
    frame = _frame(monkeypatch)
    assert ni._detect_facility_fid(frame) == 'wood'
    page = ps.detect_facility_page(frame, expected='wood')
    assert page['ok'] and page['facility'] == 'wood'
    assert page['src'] == 'template_title'


def test_standalone_wood_is_rejected_for_metal_without_ocr(monkeypatch):
    frame = _frame(monkeypatch)
    page = ps.detect_facility_page(frame, expected='metal')
    assert not page['ok']
    assert page['error'] == 'wrong_facility' and page['facility'] == 'wood'


def test_standalone_wood_continues_without_back_navigation(monkeypatch):
    frame = _frame(monkeypatch)
    clicks = []
    def resolve(_frame):
        fid = ni._detect_facility_fid(_frame)
        return UiWhere('facility:'+fid if fid else 'facility', 1,0,0,'mem')
    result = ui_nav.ensure_node('facility', facility='wood', grab=lambda: frame,
        click=lambda x,y: clicks.append((x,y)) or True, sleep_fn=lambda _: None, resolve=resolve,
        already_ok=lambda img: ps.detect_facility_page(img,expected='wood')['ok'])
    assert result.ok and result.status == 'at_node'
    assert clicks == []


def test_named_destination_does_not_make_unknown_facility_wrong(monkeypatch):
    monkeypatch.setattr(ui_nav,'locate_edge',lambda *a,**k:(7,8))
    clicks=[]
    result=ui_nav.ensure_node('facility',facility='wood',grab=lambda:object(),
        click=lambda x,y:clicks.append((x,y)) or True,sleep_fn=lambda _:None,
        resolve=lambda _:UiWhere('facility',1,0,0,'mem'),already_ok=lambda _:False,
        leave_wrong_page=lambda _:False)
    assert result.status=='need_verify'
    assert clicks==[]


def test_facility_identity_changes_without_stack_change(monkeypatch):
    wood = _frame(monkeypatch)
    metal = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/metal_standalone.png'))
    assert metal is not None
    monkeypatch.setattr(ni, '_fid_cache_id', lambda *a: 'AlteringFacilityPopupUIInstance')
    assert ni._detect_facility_fid(wood) == 'wood'
    assert ni._detect_facility_fid(metal) == 'metal'
    assert ps.detect_facility_page(metal, expected='metal')['ok']
    assert not ps.detect_facility_page(metal, expected='wood')['ok']


def test_cached_identity_cannot_replace_fresh_facility_image(monkeypatch):
    from types import SimpleNamespace
    wood = _frame(monkeypatch)
    names = ['AlteringFacilityPopupUIInstance']
    ni._inspect_stack_key.set(ni._fid_cache_id(names))
    assert ni._detect_facility_fid(wood) == 'wood'
    view = SimpleNamespace(ui_stack_names=names, ui_stack_ready=True)
    assert ni.inspect_needs_grab('fine:facility', view=view)
    assert ni._cached_facility_node('fine:facility', names) == ''
