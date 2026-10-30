"""First visit uses potion material; later visits use the bag insect icon."""
from __future__ import annotations

import time
from typing import Any

from mm_assist.life_plan.types import ExecuteResult

INSECT_MATS = frozenset({'滴答螢火蟲', '獨角仙'})
POTIONS = {'獨角仙': '強擊強化秘藥', '滴答螢火蟲': '連擊強化秘藥'}
_visits: dict[str, int] = {}
_source_after: dict[str, float] = {}
_source_attempts: dict[str, int] = {}
_entry_attempts: dict[str, int] = {}
_entry_after: dict[str, tuple[str, float]] = {}
_items_scroll_after: dict[str, tuple[float, str, int, tuple[int, int]]] = {}
_items_scroll_used: dict[str,int] = {}


def reset_insect_visits_for_tests():
    _visits.clear()
    _source_after.clear()
    _source_attempts.clear()
    _entry_attempts.clear()
    _entry_after.clear()
    _items_scroll_after.clear()
    _items_scroll_used.clear()


def execute_insect_obtain(cfg: Any = None, win: Any = None, *, mat: str,
        items=None, grab=None, do_click=None, do_drag=None, resolve=None, sleep_fn=None,
        now_fn=None, on_progress=None, root=None, entry_only: bool=False, **kwargs) -> ExecuteResult:
    from mm_assist.life_execute.ui_act import click_spot, prepare_to, pump_aborted as native_pump_aborted
    from mm_assist.life_execute.ui_logic import graph_node, resolve_navigation
    from mm_assist.perception import insect_obtain_sense as sense
    from mm_assist.perception.inventory_sense import locate_props_tab, props_tab_selected
    from mm_assist.perception.ocr import ocr_roi
    from mm_assist.perception.item_sense import line_center, line_matches_name
    from mm_assist.safety.autofarm_gates import apply_stage_verify

    if mat not in INSECT_MATS:
        return ExecuteResult(False, 'bad_mat', '未知虫料')
    if grab is None and win is None:
        return ExecuteResult(False, 'need_context', '昆蟲採集需要游戏窗')
    now = now_fn or time.time
    from mm_assist.life_plan.game_stop import network_wait_seconds
    def budget_now():
        return now()-network_wait_seconds()
    sleep = sleep_fn or time.sleep
    observe = resolve or resolve_navigation
    armed = do_click is None
    issued = False
    from mm_assist.life_plan.controller import get_controller
    owner = get_controller()

    def pump_aborted():
        if native_pump_aborted():
            return True
        current = get_controller()
        return (current.life_started_at != owner.life_started_at
                or current.current != owner.current or current.running != owner.running
                or current.preparing != owner.preparing)

    def frame():
        if grab is not None:
            return grab()
        from mm_assist.platform.capture import grab_client
        return grab_client(win)

    def click(x, y):
        if do_click is not None:
            return bool(do_click(x, y))
        from mm_assist.safety.input import click_client
        return bool(click_client(win, x, y))

    def fail(status, message):
        return ExecuteResult(False, status, message, action_issued=issued)

    def named_xy(image, name, roi):
        key = ('make_items_label' if name == '道具' else next(
            (sense.PREFIX[material] + '_potion_label'
             for material, potion in POTIONS.items() if potion == name), None))
        if key is not None:
            xy = sense.locate(image, key, roi=roi, root=root)
            if xy is not None:
                return xy
        rows = [ln for ln in ocr_roi(image, roi) if line_matches_name(ln.text, name, (name,))]
        return line_center(rows[0]) if len(rows) == 1 else None

    def recognized(image, where):
        if not getattr(where, 'stack_ready', False) or graph_node(where.node) == 'unknown':
            return ''
        p = sense.phase(image, mat, root=root)
        node = graph_node(where.node)
        if (p == 'card' and sense.bag_card_visible(image, mat, root=root)
                and not sense.bag_card_stack_allowed(getattr(where, 'stack_names', ()))):
            return ''
        if p == 'obtain' and node == 'obtain_method':
            return p
        if p in {'card', 'recipe'} and node == 'item_info':
            return p
        if p == 'recipe' and node in {'craft_shell', 'craft_item_recipe'}:
            names = set(getattr(where,'stack_names',()) or ())
            panel = {'CraftingHubPopupUIInstance','CraftingFacilityPopupUIInstance',
                     'CraftingPanelPopupUIInstance'}
            if (panel <= names and sense.items_stack_allowed(names-{'CraftingPanelPopupUIInstance'})
                    and sense.material_xy(image,mat,inventory=False,root=root) is not None):
                return 'recipe'
        if p == 'items' and node in {'craft_make', 'craft_shell', 'craft_items'}:
            names = set(getattr(where,'stack_names',()) or ())
            if not sense.items_stack_allowed(names):
                return ''  # A covered list never authorizes a potion click.
            return p
        if node == 'inventory':
            return 'bag'
        if node == 'craft_make':
            return 'make'
        return ''

    def arrived(image, actual, expected):
        return (actual == 'bag' and props_tab_selected(image, root=root)
                if expected == 'bag_props' else actual == expected)

    def clear_entry(stage):
        _entry_after.pop(stage, None)
        _entry_attempts.pop(stage, None)
        apply_stage_verify(stage, True)

    def scroll_verified(image, where):
        after,key,y,size = _items_scroll_after[mat]
        if (getattr(where,'stack_sampled_ts',0) <= after or recognized(image,where) != 'items'
                or tuple(image.shape[:2]) != size):
            return False
        anchor = sense.locate(image,key,root=root)
        return anchor is not None and anchor[1] < y-max(20,int(size[0]*.02))

    def scroll_items(image):
        from mm_assist.safety.autofarm_gates import stage_click, _BEFORE_DOWN_ERRORS
        count = _items_scroll_used.get(mat,0)
        if count >= 3:
            return fail('insect_item_not_visible','已到三段下翻上限仍未见目标；保留当前位置，不盲拖')
        points = sense.items_down_points(image,root=root,first=count==0)
        anchor = sense.items_down_anchor(image,root=root)
        if points is None or anchor is None:
            return fail('insect_item_not_visible','目标未见且无已登记下翻依据，不拖动')
        stage = '虫料道具下翻:'+mat
        def dispatched():
            _items_scroll_used[mat] = count+1
            _items_scroll_after[mat] = (now(),anchor[0],anchor[1][1],tuple(image.shape[:2]))
        def drag(*unused):
            if do_drag is not None:
                return do_drag(*points,end_hold_s=.4)
            if not armed or win is None:
                return False
            from mm_assist.safety.input import drag_client
            return drag_client(win,*points,end_hold_s=.4)
        if on_progress:
            on_progress('道具列表下翻：寻找'+POTIONS[mat])
        try:
            sent = stage_click(stage,drag,points[0],points[1])
        except Exception as exc:
            if str(exc).split(':',1)[0] not in _BEFORE_DOWN_ERRORS:
                dispatched()  # Possible mouse-down: preserve across pause/calls.
            raise
        if not sent:
            return fail('drag_rejected','道具下翻输入未发出，不重试')
        dispatched()
        deadline = budget_now()+6
        for _ in range(30):
            if pump_aborted():
                return fail('cancel','已停止操作')
            if budget_now() >= deadline:
                break
            sleep(.2)
            fresh_started_at = now()
            fresh = frame()
            current = observe(fresh)
            if pump_aborted():
                return fail('cancel','已停止操作')
            if (getattr(current,'stack_sampled_ts',0) > fresh_started_at
                    and scroll_verified(fresh,current)):
                _items_scroll_after.pop(mat,None)
                apply_stage_verify(stage,True)
                return None
            if (getattr(current,'stack_ready',False)
                    and graph_node(current.node) not in {'unknown','craft_make','craft_shell','craft_items'}):
                return fail('insect_scroll_wrong_page','下翻后不是道具页，不继续')
        return fail('insect_scroll_unverified','下翻后未确认道具定位行上移，不重发')

    if armed:
        from mm_assist.safety.lease import production_mode, prefer_visible_clicks
        production_mode(True)
        prefer_visible_clicks(True)
    try:
        nav_done = False
        for _ in range(10):  # Original chain plus up to three verified viewports.
            if pump_aborted():
                return fail('cancel', '已停止操作')
            fresh_deadline = budget_now() + 6.0
            fresh_ready = False
            for _sample in range(30):
                if pump_aborted():
                    return fail('cancel', '已停止操作')
                frame_started_at = now()
                image = frame()
                where = observe(image)
                if pump_aborted():
                    return fail('cancel', '已停止操作')
                sampled_at = getattr(where, 'stack_sampled_ts', 0)
                if (getattr(where, 'stack_ready', False)
                        and frame_started_at < sampled_at):
                    fresh_ready = True
                    break
                if budget_now() >= fresh_deadline:
                    break
                sleep(.2)
            if not fresh_ready:
                return fail('stp_busy', '等待本次画面对应的新鲜就绪栈，不点')
            pending = _source_after.get(mat)
            if (not entry_only and pending is not None and graph_node(where.node) == 'main_hud'
                    and getattr(where, 'stack_sampled_ts', 0) > pending):
                # Independent fresh HUD verifies leaving the obtain-method page.
                apply_stage_verify('昆蟲採集:' + mat, True)
                _source_after.pop(mat, None)
                _source_attempts.pop(mat, None)
            p = recognized(image, where)
            if mat in _items_scroll_after:
                if p in {'recipe','card','obtain'} or scroll_verified(image,where):
                    _items_scroll_after.pop(mat,None)
                    apply_stage_verify('虫料道具下翻:'+mat,True)
                else:
                    return fail('insect_scroll_unverified','已发下翻未验明，保留当前位置不重发')
            for stage, (expected, after) in list(_entry_after.items()):
                if (stage.startswith('虫料入口:' + mat + ':')
                        and frame_started_at > after
                        and getattr(where, 'stack_sampled_ts', 0) > after
                        and arrived(image, p, expected)):
                    clear_entry(stage)
            if entry_only:
                if p == 'items':
                    return ExecuteResult(True, 'page_matched', '已确认道具页；未点配方或采集')
                # Page-only calibration can never enter material/source branches.
                if p != 'make':
                    p = ''
            if on_progress:
                on_progress('采集 ' + mat + '：' + (p or '核对入口'))
            if p == 'obtain':
                if _source_attempts.get(mat, 0) >= 2:
                    return fail('need_verify', '昆蟲採集已发两次，先验新界面，不重发')
                xy = sense.locate(image, 'insect_entry', root=root)
                def sent():
                    nonlocal issued
                    issued = True
                    _source_attempts[mat] = _source_attempts.get(mat, 0) + 1
                    _visits[mat] = _visits.get(mat, 0) + 1
                    _source_after[mat] = now()
                    cb = kwargs.get('on_action_issued')
                    if cb:
                        cb()
                try:
                    err = click_spot(click, xy, '昆蟲採集', on_progress=on_progress,
                        stage='昆蟲採集:' + mat, on_dispatch=sent, wait_refresh=False, root=root)
                except Exception as exc:
                    from mm_assist.safety.autofarm_gates import _BEFORE_DOWN_ERRORS
                    if not issued and str(exc).split(':', 1)[0] not in _BEFORE_DOWN_ERRORS:
                        # Unknown dispatch exceptions can follow mouse-down.
                        # Keep the possible primary input so the pump won't resend.
                        sent()
                    raise
                if err:
                    return err
                if kwargs.get('defer_completion'):
                    return fail('action_pending', '昆蟲採集已发出，等待新鲜待机及库存刷新')
                from mm_assist.life_execute.gather_wait import wait_gather_idle
                result = wait_gather_idle(issued=True, read_items=kwargs.get('read_items'),
                    observe=kwargs.get('observe'), read_stp=kwargs.get('read_stp'),
                    read_page=kwargs.get('read_page'), cancelled=pump_aborted,
                    now_fn=now, sleep_fn=sleep, root=root,
                    grab=frame if not any(kwargs.get(k) for k in ('observe', 'read_stp', 'read_page')) else None,
                    message_ok=mat + ' 主界面待机')
                return result
            if p == 'card':
                xy = sense.locate(image, 'obtain_button', root=root)
                label, expected = '獲得方法', 'obtain'
            elif p == 'recipe':
                xy = sense.material_xy(image, mat, inventory=False, root=root)
                label, expected = mat, 'card'
            elif p == 'bag':
                if not props_tab_selected(image, root=root):
                    xy = locate_props_tab(image, root=root)
                    label, expected = '道具', 'bag_props'
                else:
                    xy = sense.material_xy(image, mat, inventory=True, root=root)
                    label, expected = mat, 'card'
            elif p == 'items':
                xy = named_xy(image, POTIONS[mat], sense.region('recipe_labels', root=root))
                if xy is None:
                    error = scroll_items(image)
                    if error is not None:
                        return error
                    continue  # New frame/stack and target lookup after paging.
                sleep(.25)
                fresh_started_at = now()
                fresh = frame()
                current = observe(fresh)
                fresh_xy = named_xy(fresh,POTIONS[mat],sense.region('recipe_labels',root=root))
                if (pump_aborted() or recognized(fresh,current) != 'items'
                        or getattr(current,'stack_sampled_ts',0) <= fresh_started_at
                        or fresh_xy is None or max(abs(a-b) for a,b in zip(xy,fresh_xy)) > 4):
                    return fail('insect_items_moving','道具目标或当前页未稳定，不点')
                xy = fresh_xy
                label, expected = POTIONS[mat], 'recipe'
            elif p == 'make':
                xy = named_xy(image, '道具', sense.region('make_categories', root=root))
                label, expected = '道具', 'items'
            else:
                if not entry_only and graph_node(where.node) in {'item_info', 'obtain_method'}:
                    return fail('insect_material_unknown', '当前物品明细不是目标虫料，不点')
                if nav_done:
                    return fail('insect_entry_unknown', '未确认虫料入口，不点')
                target = 'inventory' if not entry_only and _visits.get(mat, 0) > 0 else 'craft_make'
                result = prepare_to(target, grab=frame, click=click, resolve=observe,
                    sleep_fn=sleep_fn, on_progress=on_progress, root=root, title='采集 ' + mat,
                    read_stp=kwargs.get('read_stp'))
                if not result.ok:
                    return result
                nav_done = True
                continue
            stage = '虫料入口:' + mat + ':' + label
            if _entry_attempts.get(stage, 0) >= 2:
                return fail('need_verify', label + '已发两次，先验目标页，不重发')
            dispatched = False
            def entry_sent():
                nonlocal dispatched
                dispatched = True
                _entry_attempts[stage] = _entry_attempts.get(stage, 0) + 1
                _entry_after[stage] = (expected, now())
            try:
                err = click_spot(click, xy, label, stage=stage, wait_refresh=False,
                    on_progress=on_progress, root=root, on_dispatch=entry_sent)
            except Exception as exc:
                from mm_assist.safety.autofarm_gates import _BEFORE_DOWN_ERRORS
                if not dispatched and str(exc).split(':', 1)[0] not in _BEFORE_DOWN_ERRORS:
                    entry_sent()
                raise
            if err:
                return err
            deadline = budget_now() + 6.0
            verified = False
            while budget_now() < deadline:
                if pump_aborted():
                    return fail('cancel', '已停止操作')
                sleep(0.2)
                fresh_started_at = now()
                fresh = frame()
                current = observe(fresh)
                if (fresh_started_at <= _entry_after[stage][1]
                        or getattr(current, 'stack_sampled_ts', 0) <= fresh_started_at):
                    continue
                actual = recognized(fresh, current)
                verified = arrived(fresh, actual, expected)
                if verified:
                    clear_entry(stage)
                    break
            if not verified:
                return fail('insect_refresh_unknown', label + '后未确认新鲜目标页，不重发')
        return fail('insect_entry_timeout', '虫料入口未完成，不重试')
    except Exception as exc:
        return fail('error', str(exc))
    finally:
        if armed:
            from mm_assist.safety.lease import production_mode
            from mm_assist.safety.input import release_mouse_buttons
            production_mode(False)
            release_mouse_buttons()
