"""Independent fatal dialogs and a read-only transient network wait for Life.

These native game overlays can leave the managed UI stack reporting HUD/stp=0.
Use two registered pixel anchors per dialog until a verified memory identity exists.
"""
from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from functools import lru_cache, wraps
import hashlib
import json
import time

import cv2
import numpy as np

from mm_assist.config import ROOT

ASSETS = ROOT / 'knowledge/templates/game_stop'
_WATCH = ContextVar('life_game_stop_watch', default=None)
_LOCATED_RESULT = ContextVar('life_network_located_result', default=None)
LABELS = {'runtime': '遊戲已執行過久，請重新啟動遊戲',
          'disconnect': '與遊戲伺服器的連線已中斷'}
FRAME_WAIT_S = 2.0
FRAME_WAIT_SAMPLES = 24
# One resident input owner; retain a capture failure until fresh pixels verify
# the game again. A new run/item cannot inherit this failure.
_FRAME_FAILURE = None


class FrameUnavailable(ValueError):
    """Invalid pixels cannot establish presence or absence of a dialog."""


def _gray(frame):
    # Retain antialiased strokes at smaller client sizes; do not binarize text.
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(3,3),.5)


@lru_cache(maxsize=1)
def _templates():
    catalog = json.loads((ASSETS / 'catalog.json').read_text('utf-8'))
    threshold = float(catalog['threshold'])
    if catalog['source_size'] != [1920,1440] or not .88 <= threshold <= 1:
        raise ValueError('invalid game stop registration')
    dialogs = []
    for dialog in catalog['dialogs']:
        if dialog['id'] not in LABELS or len(dialog['anchors']) != 2:
            raise ValueError('invalid game stop identity')
        anchors = []
        for anchor in dialog['anchors']:
            path = ASSETS / anchor['file']
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != anchor['sha256']:
                raise ValueError('game stop template fingerprint changed')
            image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            x1,y1,x2,y2 = anchor['crop']
            if image is None or image.shape[:2] != (y2-y1,x2-x1):
                raise ValueError('invalid game stop crop')
            image = cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
            anchors.append((_gray(image),x1/2,y1/2))
        dialogs.append((dialog['id'],anchors))
    if {d[0] for d in dialogs} != set(LABELS):
        raise ValueError('incomplete game stop registration')
    return threshold,dialogs


def detect_game_stop(frame) -> tuple[str,...]:
    """Fresh client pixels, height-normalized; both title and button must match.

    Search only registered regions and require consistent anchor displacement.
    A common 確認/結束 button or HUD memory is never a positive/negative identity.
    Invalid/black frames raise rather than become an empty detection.
    """
    if (not isinstance(frame,np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
            or frame.dtype != np.uint8 or min(frame.shape[:2]) < 240
            or max(frame.shape[:2]) > 16384 or np.max(frame) < 5):
        raise FrameUnavailable('game stop frame unavailable')
    scale = 720/frame.shape[0]
    image = cv2.resize(frame,(round(frame.shape[1]*scale),720),interpolation=cv2.INTER_AREA)
    image = _gray(image)
    threshold,dialogs = _templates()
    found = []
    for identity,anchors in dialogs:
        offsets = []
        for template,x,y in anchors:
            x += image.shape[1]/2-480
            th,tw = template.shape
            x1,y1 = max(0,int(x)-16),max(0,int(y)-16)
            x2,y2 = min(image.shape[1],int(x)+tw+17),min(720,int(y)+th+17)
            region = image[y1:y2,x1:x2]
            if region.shape[0] < th or region.shape[1] < tw:
                break
            _,score,_,point = cv2.minMaxLoc(cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED))
            if not np.isfinite(score) or score < threshold:
                break
            offsets.append((x1+point[0]-x,y1+point[1]-y))
        if len(offsets)==2 and max(abs(offsets[0][i]-offsets[1][i]) for i in (0,1)) <= 4:
            found.append(identity)
    return tuple(found)


def _capture(cfg,win):
    from mm_assist.perception.game_frame import ensure_game_frame
    from mm_assist.nav.map_atlas import _grab_navigation_client
    frame = ensure_game_frame(cfg,win,persist=False)
    image = _grab_navigation_client(cfg,frame.to_window())
    if image.shape[:2] != (frame.height,frame.width):
        raise ValueError('game stop client size changed')
    return image


class GameStopWatch:
    def __init__(self,cfg,win,state):
        self.cfg,self.win,self.state = cfg,win,state
        self.stopped = False
        self.checked = float('-inf')
        self.network_wait_s = 0.0
        self.network_wait_started = None
        self.network_before_frame = None
        self.network_after_frame = None
        self.network_cleared_at = None
        self.retry_point = None

    def active(self):
        from mm_assist.life_plan.controller import get_controller
        from mm_assist.tool_repair import op_cancelled
        from mm_assist.safety.lease import cancel_requested
        if op_cancelled() or cancel_requested():
            return False
        current = get_controller()
        return (current.running and not current.paused
                and current.life_started_at == self.state.life_started_at
                and current.current == self.state.current)

    def _retry_allowed(self):
        from mm_assist.safety.lease import in_test_mode, current_lease
        lease = current_lease() if in_test_mode() else None
        return self.active() and (not in_test_mode() or lease is not None and lease.alive())

    def _retry_note(self,frame,tag,extra):
        from mm_assist.life_plan import anomalies
        from mm_assist.platform.capture import save_capture
        try:
            shot = save_capture(frame,anomalies.ROOT/anomalies.SHOT_REL,prefix='network_retry')
        except (OSError,cv2.error):
            shot = '截圖保存失敗'
        anomalies.log_life_anomaly(plan_title=self.state.current.title,plan_id=self.state.current.id,
            kind=self.state.current.kind,tag=tag,stage='network_retry',
            extra=f'{extra}；原图={shot}',run_started_at=self.state.life_started_at)

    def _retry_network(self,frame):
        """User-authorized five-second retries, each from independent pixels.

        Keep this invocation suspended; on disappearance abandon its old UI
        targets. The next pump retains pending business actions and reidentifies.
        """
        from mm_assist.perception.network_retry import locate_network_retry, WAIT_S
        from mm_assist.life_plan.controller import set_current_action, observe_action_completion
        from mm_assist.life_plan.action_finish import StpSample
        from mm_assist.safety.input import click_client, release_mouse_buttons, foreground_input_scope
        from mm_assist.safety.lease import in_test_mode
        from mm_assist.nav.compat import production_click_method
        from mm_assist.safety.autofarm_gates import stage_click, apply_stage_verify

        release_mouse_buttons()
        observe_action_completion(StpSample(None,0,0,False),now=time.monotonic(),
            item_id=self.state.current.id,run_started_at=self.state.life_started_at)
        if not self._retry_allowed():
            return
        self._retry_note(frame,'network_retry_wait','偵測「網路連線狀態不穩定」；每5秒重新驗圖並重試，直到彈窗消失')
        set_current_action('網路連線狀態不穩定：等待5秒後點「重新嘗試」',
            run_started_at=self.state.life_started_at)
        while self._retry_allowed():
            due=time.monotonic()+WAIT_S
            while self._retry_allowed() and time.monotonic()<due:
                time.sleep(min(.05,max(0.,due-time.monotonic())))
            if not self._retry_allowed():
                return
            try:
                frame=_capture(self.cfg,self.win)
                dialogs=detect_game_stop(frame)
            except (OSError,ValueError,RuntimeError):
                continue  # Unknown renders never authorize input or disappearance.
            if not self._retry_allowed():
                return
            if dialogs:
                return frame  # Preserve this exact fatal frame for the original pause path.
            point=locate_network_retry(frame)
            if point is None:
                apply_stage_verify('網路重新嘗試',True)
                set_current_action('重新嘗試彈窗已消失，重新辨識目前畫面',
                    run_started_at=self.state.life_started_at)
                return
            if frame.shape[:2] != (self.win.height,self.win.width):
                raise RuntimeError('重新嘗試前客戶區尺寸改變')
            # Explicit user request: keep retrying this exact dialog. Each new
            # full-title/button observation is after the previous input; this
            # verifies the retry target, not a successful reconnect/business act.
            apply_stage_verify('網路重新嘗試',True)
            self.retry_point=point
            try:
                method='sendinput' if in_test_mode() else production_click_method(self.cfg)[0]
                with foreground_input_scope(self.win.hwnd,win=self.win):
                    sent=stage_click('網路重新嘗試',
                        lambda x,y:click_client(self.win,x,y,method=method),*point)
            except RuntimeError as exc:
                # Native focus/lease/pointer gates still refuse input. Only
                # fresh pixels on a later iteration may authorize another try.
                if self._retry_allowed():
                    self._retry_note(frame,'network_retry_refused',str(exc))
                continue
            finally:
                self.retry_point=None
                release_mouse_buttons()
            if self._retry_allowed() and sent:
                self._retry_note(frame,'network_retry_clicked',f'已點「重新嘗試」；客戶區座標={point}；不重發主要動作')

    def _wait_network(self,frame):
        """Keep this exact executor call suspended; never restart an action."""
        self.network_before_frame = frame.copy()
        self.network_after_frame = None
        self.network_wait_started = time.monotonic()
        try:
            return self._wait_network_body(frame)
        finally:
            self.network_wait_s += max(0.0,time.monotonic()-self.network_wait_started)
            self.network_wait_started = None

    def _wait_network_body(self,frame):
        from mm_assist.perception.network_toast import network_unstable, POLL_INTERVAL_S
        from mm_assist.life_plan.controller import get_controller, set_current_action, observe_action_completion
        from mm_assist.life_plan.action_finish import StpSample
        from mm_assist.safety.input import release_mouse_buttons
        from mm_assist.safety.lease import in_test_mode, current_lease
        from mm_assist.life_plan.anomalies import log_life_anomaly

        previous = get_controller().current_action
        release_mouse_buttons()
        observe_action_completion(StpSample(None,0,0,False), now=time.monotonic(),
            item_id=self.state.current.id,run_started_at=self.state.life_started_at)
        set_current_action('等待網路恢復：每2秒檢查「網路不穩定。」',run_started_at=self.state.life_started_at)
        log_life_anomaly(plan_title=self.state.current.title,plan_id=self.state.current.id,
            kind=self.state.current.kind,tag='network_wait_started',stage='network_unstable',
            extra='保留当前执行位置；每2秒只读检查固定区域，不重发主要动作',run_started_at=self.state.life_started_at)
        due = time.monotonic()+POLL_INTERVAL_S
        while self.active():
            lease = current_lease() if in_test_mode() else None
            if in_test_mode() and (lease is None or not lease.alive()):
                return None
            remaining = due-time.monotonic()
            if remaining > 0:
                time.sleep(min(.05,remaining))
                continue
            due = time.monotonic()+POLL_INTERVAL_S
            frame = _capture(self.cfg,self.win)
            if not self.active():
                return None
            try:
                dialogs = detect_game_stop(frame)
            except FrameUnavailable:
                continue  # Unknown pixels do not mean the warning disappeared.
            if dialogs:
                return frame  # The original fatal-dialog path remains authoritative.
            if network_unstable(frame):
                continue
            if not self.active():
                return None
            set_current_action(previous,run_started_at=self.state.life_started_at)
            log_life_anomaly(plan_title=self.state.current.title,plan_id=self.state.current.id,
                kind=self.state.current.kind,tag='network_wait_finished',stage='network_clear',
                extra='新鲜有效画面确认提示消失；继续当前执行位置',run_started_at=self.state.life_started_at)
            self.checked = time.monotonic()
            self.network_after_frame = frame.copy()
            self.network_cleared_at = time.time()
            return frame
        return None

    def check(self,*,force=False,allow_frame_wait=False,check_network=True):
        if self.stopped:
            return True
        if not self.active():
            return True  # Cancel this old/pause scope without changing newer state.
        if not int(getattr(self.win,'hwnd',0) or 0):
            return False
        now = time.monotonic()
        if not force and now-self.checked < .2:
            return False
        self.checked = now
        frame = None
        try:
            frame = _capture(self.cfg,self.win)
            deadline = time.monotonic() + FRAME_WAIT_S
            for attempt in range(FRAME_WAIT_SAMPLES):
                if not self.active():
                    return True
                try:
                    dialogs = detect_game_stop(frame)
                    if check_network and not dialogs and self.retry_point is None:
                        from mm_assist.perception.network_toast import network_unstable
                        if network_unstable(frame):
                            frame = self._wait_network(frame)
                            if frame is None:
                                return True
                            dialogs = detect_game_stop(frame)
                    if not dialogs:
                        from mm_assist.perception.network_retry import locate_network_retry
                        retry = locate_network_retry(frame)
                        if self.retry_point is not None:
                            if check_network and retry != self.retry_point:
                                raise RuntimeError('click_ui_not_ready: 重新嘗試畫面或落點已改變')
                        elif retry is not None:
                            fatal_frame = self._retry_network(frame) if check_network else None
                            if fatal_frame is None:
                                self.stopped = True
                                return True
                            frame = fatal_frame
                            dialogs = detect_game_stop(frame)
                    break
                except FrameUnavailable:
                    # A single black render (e.g. during travel) is unknown,
                    # not a termination dialog. Block this call/input until
                    # fresh rendered pixels arrive, with a bounded readonly
                    # wait. Never use cached pixels or BitBlt as a fallback.
                    # Only pump entry/completion observations can wait. Input
                    # and drag cancellation checks must pause/release at once.
                    if not allow_frame_wait:
                        raise
                    if attempt == 0:
                        from mm_assist.life_plan.controller import observe_action_completion
                        from mm_assist.life_plan.action_finish import StpSample
                        observe_action_completion(StpSample(None,0,0,False),
                            now=time.monotonic(),item_id=self.state.current.id,
                            run_started_at=self.state.life_started_at)
                    remaining = deadline-time.monotonic()
                    if attempt+1 >= FRAME_WAIT_SAMPLES or remaining <= 0:
                        raise
                    time.sleep(min(.1,remaining))
                    if not self.active():
                        return True
                    if time.monotonic() >= deadline:
                        raise
                    frame = _capture(self.cfg,self.win)
            reason = '；'.join(LABELS[d] for d in dialogs)
        except Exception as exc:
            if self.retry_point is not None:
                raise RuntimeError('click_ui_not_ready: 重新嘗試前未能核驗目前畫面') from exc
            dialogs = ()
            reason = f'未能核验游戏阻挡弹窗（{type(exc).__name__}）'
            if isinstance(exc, FrameUnavailable) and not self.state.pause_on_failure:
                if not self.active():
                    return True
                self.stopped = True  # Abort this invocation, including drag polling.
                global _FRAME_FAILURE
                already_saved = (_FRAME_FAILURE is not None
                    and _FRAME_FAILURE[:2] == (self.state.life_started_at,self.state.current.id))
                _FRAME_FAILURE = (self.state.life_started_at, self.state.current.id, reason)
                from mm_assist.life_plan.controller import observe_action_completion, set_current_action
                from mm_assist.life_plan.action_finish import StpSample
                from mm_assist.safety.input import release_mouse_buttons
                observe_action_completion(StpSample(None,0,0,False),
                    now=time.monotonic(),item_id=self.state.current.id,
                    run_started_at=self.state.life_started_at)
                release_mouse_buttons()
                set_current_action('等待有效游戏画面，暂不输入；恢复画面后处理失败项',
                                   run_started_at=self.state.life_started_at)
                if not already_saved:
                    self._evidence(frame,dialogs,reason,self.state)
                return True
        if not self.active():
            return True
        if not reason:
            return False
        self.stopped = True
        from mm_assist.life_plan.controller import pause_life
        # Atomic generation/item checks; release input and retain issued facts.
        paused = pause_life(run_started_at=self.state.life_started_at,
                            item_id=self.state.current.id,reason=reason)
        if (not paused.paused or paused.life_started_at != self.state.life_started_at
                or paused.current != self.state.current):
            return True
        self._evidence(frame,dialogs,reason,paused)
        return True

    def _evidence(self,frame,dialogs,reason,paused):
        from mm_assist.life_plan import anomalies
        from mm_assist.perception.state_ui import read_state_ui
        from mm_assist.platform.capture import save_capture
        shot = ''
        try:
            # Preserve this exact detection frame; never recapture or prune old evidence.
            if frame is not None:
                shot = str(save_capture(frame,anomalies.ROOT/anomalies.SHOT_REL,prefix='game_stop'))
                view = read_state_ui()
                record = {'ts':time.time(),'dialogs':dialogs,'reason':reason,
                    'run_started_at':paused.life_started_at,'plan_id':paused.current.id,
                    'hwnd':getattr(self.win,'hwnd',0),'ui':view.raw if view else None,
                    'action_issued':paused.pending_action is not None or bool(paused.gather_pending_id),
                    'pending_action':repr(paused.pending_action),'input_events':0,
                    'catalog_sha256':hashlib.sha256((ASSETS/'catalog.json').read_bytes()).hexdigest()}
                from pathlib import Path
                Path(shot).with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n','utf-8')
        except Exception:
            pass  # Evidence I/O cannot unpause or allow input.
        anomalies.log_life_anomaly(plan_title=paused.current.title,plan_id=paused.current.id,
            kind=paused.current.kind,tag='game_stop_paused' if paused.paused else 'game_frame_wait',
            stage=','.join(dialogs) or 'unverified',
            extra=f'{reason}；原图={shot}；' + ('不确认、不脱困、不跳过、不自动恢复'
                if paused.paused else '未识别为弹窗；有效新图前不输入、不推进队列'),
            run_started_at=paused.life_started_at)


def network_wait_seconds() -> float:
    """Only business wait budgets exclude this scope's network hold."""
    watch = _WATCH.get()
    if watch is None:
        return 0.0
    active = (max(0.0,time.monotonic()-watch.network_wait_started)
              if watch.network_wait_started is not None else 0.0)
    return watch.network_wait_s+active


def operation_monotonic() -> float:
    return time.monotonic()-network_wait_seconds()


def operation_wall_time() -> float:
    return time.time()-network_wait_seconds()


def network_input_unchanged(point) -> bool:
    """A boundary that waited may only reuse an unchanged visible target."""
    watch = _WATCH.get()
    if watch is None:
        return False
    if watch.retry_point is not None:
        return watch.active() and tuple(point or ()) == watch.retry_point
    located = _LOCATED_RESULT.get()
    if located is not None:
        return _located_result_current(point)
    before,after = watch.network_before_frame,watch.network_after_frame
    if before is None or after is None or before.shape != after.shape:
        return False
    h,w = before.shape[:2]
    regions = [(0,0,round(.36*w),round(.09*h))]
    if point is not None:
        x,y = point
        dx,dy = round(128*h/1440),round(48*h/1440)
        regions.append((max(0,x-dx),max(0,y-dy),min(w,x+dx+1),min(h,y+dy+1)))
    for x1,y1,x2,y2 in regions:
        old,new = before[y1:y2,x1:x2],after[y1:y2,x1:x2]
        if old.size == 0 or old.shape != new.shape:
            return False
        a,b = _gray(old),_gray(new)
        if np.array_equal(a,b):
            continue
        if np.std(a) < 1 or np.std(b) < 1:
            return False
        score = cv2.matchTemplate(b,a,cv2.TM_CCOEFF_NORMED)[0,0]
        if not np.isfinite(score) or score < .98:
            return False
    return True


def _located_result_current(point):
    located = _LOCATED_RESULT.get()
    watch = _WATCH.get()
    return bool(located is not None and located[0] is watch
        and tuple(point or ()) == located[1] and network_wait_seconds() == located[2]
        and 0 <= time.time()-located[3] <= 2.5 and watch.active())


@contextmanager
def located_result_input(frame,point,*,captured_at,network_revision):
    """Bind only a newly located result-confirm point to this network generation.

    This is not an input permit: the native boundary must still force-read the
    current result stack, lease, foreground, geometry and cancellation gates.
    """
    watch = _WATCH.get()
    located = None
    if (watch is not None and network_revision > 0
            and network_revision == network_wait_seconds()
            and watch.network_cleared_at is not None
            and captured_at >= watch.network_cleared_at
            and 0 <= time.time()-captured_at <= 2.5
            and isinstance(frame,np.ndarray) and frame.dtype == np.uint8
            and watch.network_after_frame is not None
            and frame.shape == watch.network_after_frame.shape and np.max(frame) >= 5):
        located = (watch,tuple(point),network_revision,captured_at)
    token = _LOCATED_RESULT.set(located)
    try:
        yield
    finally:
        _LOCATED_RESULT.reset(token)


def network_located_result_stack_current(names,point,hwnd):
    located = _LOCATED_RESULT.get()
    if located is None:
        return True
    watch = _WATCH.get()
    return (_located_result_current(point) and int(hwnd or 0) == int(watch.win.hwnd)
        and {'FacilityResultUIInstance','FacilityResultUIState'} <= set(names))


def game_stop_scope_cancelled():
    """Fast cancellation recheck after this input's fresh frame/stack guard."""
    watch = _WATCH.get()
    return watch is not None and (watch.stopped or not watch.active())


def game_stop_interrupted(*,force=False,allow_frame_wait=False,check_network=True):
    watch = _WATCH.get()
    return watch is not None and watch.check(force=force,allow_frame_wait=allow_frame_wait,
                                              check_network=check_network)


def guard_life_pump(function):
    @wraps(function)
    def guarded(cfg,win,**kwargs):
        global _FRAME_FAILURE
        from mm_assist.life_plan.controller import get_controller
        state = get_controller()
        if not state.running or state.paused or state.current is None:
            return
        watch = GameStopWatch(cfg,win,state)
        token = _WATCH.set(watch)
        try:
            if not watch.check(force=True,allow_frame_wait=True):
                failure = _FRAME_FAILURE
                if failure is not None:
                    _FRAME_FAILURE = None
                    if failure[:2] == (state.life_started_at,state.current.id):
                        # Only this new valid capture can release the hold.
                        # Completed action chains retain their issued fact and
                        # resume observation. Failed chains are skipped through
                        # the existing normal-mode policy, never executed again.
                        pending = get_controller().pending_action
                        if pending is None or not pending.actions_complete:
                            from mm_assist.life_plan.bar_ui import _note_fail_and_continue
                            _note_fail_and_continue(win=win,item=state.current,
                                reason='game_frame_unavailable',extra=failure[2],
                                run_started_at=state.life_started_at)
                            return
                result = function(cfg,win,**kwargs)
                return result
        finally:
            _WATCH.reset(token)
    return guarded
