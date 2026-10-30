"""逐帧查找技能列表：每段下翻独立核对，停稳后才返回可点目标。"""
from __future__ import annotations

import time
from mm_assist.life_plan.game_stop import operation_monotonic

from mm_assist.life_plan.types import ExecuteResult
from mm_assist.safety.autofarm_gates import stage_click, apply_stage_verify, _BEFORE_DOWN_ERRORS

# 已发但尚未核验的手势不能因暂停、换计划或重新调用而丢失。
_pending: dict[tuple[str, str], tuple[float, tuple, dict]] = {}


def verify_closed_list(where, *, frame_started_at):
    """Retire only a gesture whose source window has independently closed."""
    from mm_assist.life_execute.ui_logic import graph_node
    from mm_assist.memory.readers.ui_stack import PROFILE_NAMES

    now = time.time()
    sampled = float(getattr(where, 'stack_sampled_ts', 0) or 0)
    if (not getattr(where, 'stack_ready', False)
            or not 0 <= now - frame_started_at <= 16
            or not frame_started_at < sampled <= now):
        return
    names = set(getattr(where, 'stack_names', ()) or ())
    shell = {'HudUIStepState', 'TutorialTooltipUIState'}
    node = graph_node(where.node)
    if node == 'main_hud':
        closed = names <= shell
    else:
        closed = (node in {'profile', 'life_grid'} and 'NewProfileUIInstance' in names
            and names <= (PROFILE_NAMES | shell | {'UILayerHideState', 'SizeAdjustmentSender'}))
    if not closed:
        return
    for key, (sent_at, _, _) in list(_pending.items()):
        if sent_at < frame_started_at:
            _pending.pop(key, None)
            apply_stage_verify('详情物下翻:' + key[0] + ':' + key[1], True)


def seek_item(frame, item, *, skill_id, name, grab, drag=None, validate=None,
              captured_at=None, cancelled=lambda: False, sleep=time.sleep, on_scroll=lambda:None, root=None):
    from mm_assist.perception.item_sense import (
        locate_skill_item, item_list_row_positions, item_list_scroll_drag_points,
        items_for_skill, MAX_SCROLL_STEPS,
    )
    from mm_assist.life_execute.ui_logic import recognized_click_xy

    key = skill_id, name
    stage = '详情物下翻:' + skill_id + ':' + name
    last_capture_started = 0.0
    def fail(status, message):
        return frame, item, ExecuteResult(False, status, message)
    def read(deadline=None):
        nonlocal last_capture_started
        if deadline is None:
            deadline = operation_monotonic() + 6
        for sample in range(30):
            if cancelled():
                return None, None, ExecuteResult(False, 'cancel', '已停止操作')
            started = time.time()
            last_capture_started = started
            fresh = grab()
            if cancelled():
                return fresh, None, ExecuteResult(False, 'cancel', '已停止操作')
            valid = validate is None or validate(fresh, started)
            # Slow stack reads may return after pause/new-start. Never accept
            # that result or discard the pending drag on the old caller's behalf.
            if cancelled():
                return fresh, None, ExecuteResult(False, 'cancel', '已停止操作')
            if valid:
                break
            if sample == 29 or operation_monotonic() >= deadline:
                return fresh, None, ExecuteResult(False, 'item_list_unknown', '当前新鲜技能列表未确认，不继续')
            sleep(.2)
        hit = locate_skill_item(fresh, skill_id=skill_id, name=name, root=root)
        if cancelled():
            return fresh, None, ExecuteResult(False, 'cancel', '已停止操作')
        return fresh, hit, None
    def stable():
        nonlocal frame, item
        deadline = operation_monotonic() + 6
        previous = recognized_click_xy(item)
        for _ in range(12):
            if cancelled():
                return ExecuteResult(False, 'cancel', '已停止操作')
            sleep(.25)
            frame, item, error = read(deadline)
            if error is not None:
                return error
            current = recognized_click_xy(item)
            if (previous is not None and current is not None
                    and max(abs(a-b) for a,b in zip(previous,current)) <= 4):
                return None
            previous = current
            if operation_monotonic() >= deadline:
                break
        return ExecuteResult(False, 'item_list_moving', '目标未在两张新画面中停稳，不点')
    def progressed(pending):
        sent_at, size, before = pending
        if (frame is None or tuple(frame.shape[:2]) != size or not before
                or last_capture_started <= sent_at):
            return False
        # validate() requires a post-capture stack, and this capture must follow
        # the issued gesture. The actual frame is never reused as two samples.
        after = item_list_row_positions(frame, skill_id, root=root)
        common = set(before) & set(after)
        if any(after[n] < before[n]-max(20, int(size[0]*.02)) for n in common):
            return True
        catalog = items_for_skill(skill_id, root=root)
        old = [catalog.index(n) for n in before if n in catalog]
        new = [catalog.index(n) for n in after if n in catalog]
        return bool(not common and old and new and min(new) > max(old))
    def clear():
        _pending.pop(key, None)
        apply_stage_verify(stage, True)

    if validate is not None:
        if captured_at is None:
            frame, item, error = read()
            if error is not None:
                return frame, item, error
        elif not validate(frame, captured_at):
            # The first paired stack can straddle the list opening. Discard
            # that frame/target and use the same bounded read-only path as
            # later samples; an unknown page never authorizes any gesture.
            frame, item, error = read()
            if error is not None:
                return frame, item, error
    for step in range(MAX_SCROLL_STEPS+1):
        if cancelled():
            return fail('cancel', '已停止操作')
        if key in _pending and (frame is None or tuple(frame.shape[:2]) != _pending[key][1]):
            return fail('item_scroll_unverified', '下翻后窗口尺寸改变，不核销或继续')
        if item.get('ok'):
            error = stable()
            if error is None and (key not in _pending or last_capture_started > _pending[key][0]):
                clear()
            elif error is None:
                return fail('item_scroll_unverified', '画面早于已发下翻，不核销')
            return frame, item, error
        if key in _pending:
            # Re-observe after a pause/exception; never verify with the old
            # pre-gesture frame supplied by an interrupted caller.
            deadline = operation_monotonic() + 6
            for sample in range(30):
                frame, item, error = read(deadline)
                if error is not None:
                    return frame, item, error
                if item.get('ok') or progressed(_pending[key]):
                    break
                # Virtualized rows can paint their names before their icons.
                # A ready stack alone cannot verify movement, nor does one
                # empty row match justify leaving and poisoning later plans.
                if sample == 29 or operation_monotonic() >= deadline:
                    return fail('item_scroll_unverified', '已发下翻尚未确认行上移，不重发')
                sleep(.2)
            if item.get('ok'):
                continue
            if cancelled():
                return fail('cancel', '已停止操作')
            clear()
        plan = item.get('scroll_plan') or {}
        if step >= MAX_SCROLL_STEPS or plan.get('direction') != 'down':
            return fail('item_not_visible', f'未见{name}或无新画面向下依据，不拖动')
        on_scroll()
        if drag is not None:
            before = item_list_row_positions(frame, skill_id, root=root)
            if not before:
                return fail('item_not_visible', '缺少当前完整行图依据，不拖动')
            size = tuple(frame.shape[:2])
            deadline = operation_monotonic()+6
            for _ in range(12):
                sleep(.25)
                frame, item, error = read(deadline)
                if error is not None:
                    return frame, item, error
                if tuple(frame.shape[:2]) != size:
                    return fail('item_list_unknown', '定位期间窗口尺寸改变，不拖动')
                if item.get('ok'):
                    break  # The target moved into view; verify it, never drag it away.
                current = item_list_row_positions(frame, skill_id, root=root)
                common = set(before)&set(current)
                if common and all(abs(before[n]-current[n]) <= 4 for n in common):
                    before = current
                    break
                before = current
                if operation_monotonic() >= deadline:
                    return fail('item_list_moving', '列表完整行图尚未停稳，不拖动')
            else:
                return fail('item_list_moving', '列表完整行图尚未停稳，不拖动')
            if item.get('ok'):
                continue
            if not before or (item.get('scroll_plan') or {}).get('direction') != 'down':
                return fail('item_not_visible', '新画面无完整行图或向下依据，不拖动')
            (x,y0),(_,y1) = item_list_scroll_drag_points(frame, step=step)
            try:
                sent = stage_click(stage, lambda *_: drag(x,y0,x,y1), x,y0)
            except Exception as exc:
                if str(exc).split(':',1)[0] not in _BEFORE_DOWN_ERRORS:
                    _pending[key] = time.time(), size, before
                raise
            if not sent:
                return fail('drag_rejected', '翻页输入未发出或待验，已停止')
            _pending[key] = time.time(), size, before
        sleep(.6)
        frame, item, error = read()
        if error is not None:
            return frame, item, error
    return fail('item_not_visible', '达到下翻上限，未点目标')
