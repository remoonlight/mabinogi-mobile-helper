"""Read-only rolling execution timings. Never reads the game or drives input."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import json
import logging
import math
import os
from pathlib import Path
import threading
import time

from mm_assist.config import ROOT

KEEP_S = 12 * 3600
PATH = ROOT / 'logs/life_execution_12h.json'
_CONTEXT = ContextVar('life_execution_record', default=None)
_RECORDER = None
_INIT_LOCK = threading.Lock()
_LAST_ERROR = -float('inf')


class ExecutionRecorder:
    def __init__(self, path: Path, *, wall=time.time, mono=time.monotonic, pid=None):
        self.path, self.wall, self.mono = Path(path), wall, mono
        self.pid = os.getpid() if pid is None else pid
        self.lock = threading.RLock()
        self.rows = []
        self.current = None
        self.last_mono = 0.
        self.last_save = -float('inf')
        try:
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            if saved.get('schema') == 'life_execution.v1':
                self.rows = [r for r in saved['plans'] if _valid_row(r)]
                for row in self.rows:
                    if row.get('status') in {'running', 'paused'}:
                        row.update(status='interrupted', ended_at=row['updated_at'],
                                   result='进程结束；计时止于最后一次落盘')
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            self.rows = []
        self._prune()

    def _key(self, row):
        return row['pid'], row['run_started_at'], row['plan_id']

    def _tick(self):
        if self.current is None:
            return
        row = self.current
        now, stamp = self.mono(), self.wall()
        delta = max(0., now - self.last_mono)
        self.last_mono = now
        segment = row['segments'][-1]
        segment['duration_s'] += delta
        segment['ended_at'] = stamp
        row['elapsed_s'] += delta
        row['updated_at'] = stamp
        field = 'paused_s' if segment['kind'] == 'paused' else 'active_s'
        row[field] += delta
        if segment['kind'] == 'stp_wait':
            row['stp_wait_s'] += delta

    def _segment(self, text, kind, node):
        stamp = self.wall()
        self.current['segments'].append(dict(started_at=stamp, ended_at=stamp,
            duration_s=0., phase=str(text), kind=kind, node=node))

    def begin(self, run, item):
        with self.lock:
            key = self.pid, float(run), item.id
            if self.current is not None and self._key(self.current) == key:
                return key
            if self.current is not None:
                self.finish(self._key(self.current), 'interrupted', '切换计划或开始新一轮')
            stamp = self.wall()
            self.current = dict(pid=self.pid, run_started_at=float(run), plan_id=item.id,
                plan=item.title, kind=item.kind, started_at=stamp, updated_at=stamp,
                ended_at=None, status='running', result='', elapsed_s=0., active_s=0.,
                paused_s=0., stp_wait_s=0., navigation_s=0., segments=[], events=[])
            self.rows.append(self.current)
            self.last_mono = self.mono()
            self._segment('准备执行', 'execute', 'unknown')
            self.checkpoint(force=True)
            return key

    def matches(self, key, *, running=True):
        return (self.current is not None and self._key(self.current) == key
                and (not running or self.current['status'] == 'running'))

    def phase(self, key, text, kind='execute'):
        with self.lock:
            if not self.matches(key):
                return
            self._tick()
            old = self.current['segments'][-1]
            if (old['phase'], old['kind']) != (text, kind):
                self._segment(text, kind, old['node'])
            self.checkpoint()

    def phase_snapshot(self, key):
        with self.lock:
            if self.matches(key):
                segment = self.current['segments'][-1]
                return segment['phase'], segment['kind']
        return None

    def event(self, key, event, **data):
        with self.lock:
            if not self.matches(key):
                return
            self._tick()
            if event == 'page':
                old = self.current['segments'][-1]
                if old['node'] != data['node']:
                    self._segment(old['phase'], old['kind'], data['node'])
                self.current['navigation_s'] += data.pop('elapsed_s', 0.)
            row = dict(event=event, **data)
            # Store changes, not every polling sample. The live segment keeps growing.
            previous = next((e for e in reversed(self.current['events']) if e['event'] == event), None)
            if event == 'input_requested' or previous is None or {k:v for k,v in previous.items() if k!='ts'} != row:
                self.current['events'].append(dict(ts=self.wall(), **row))
            self.checkpoint()

    def suspend(self, key):
        with self.lock:
            if self.matches(key):
                self._tick()
                self._segment('人工暂停', 'paused', self.current['segments'][-1]['node'])
                self.current['status'] = 'paused'
                self.checkpoint(force=True)

    def resume(self, key):
        with self.lock:
            if self.matches(key, running=False) and self.current['status'] == 'paused':
                self._tick()
                self.current['status'] = 'running'
                self._segment('恢复执行', 'execute', self.current['segments'][-1]['node'])
                self.checkpoint(force=True)

    def finish(self, key, status, result=''):
        with self.lock:
            if not self.matches(key, running=False):
                return
            self._tick()
            self.current.update(status=status, result=str(result), ended_at=self.wall())
            self.current = None
            self.checkpoint(force=True)

    def _prune(self):
        cutoff = self.wall() - KEEP_S
        self.rows = [r for r in self.rows if r is self.current or r['updated_at'] >= cutoff]
        for row in self.rows:
            row['events'] = [e for e in row['events'] if e.get('ts', 0) >= cutoff]
            row['segments'] = [s for s in row['segments'] if s.get('ended_at', 0) >= cutoff
                               or (row is self.current and s is row['segments'][-1])]

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f'.{self.pid}.tmp')
        temporary.write_text(json.dumps(dict(schema='life_execution.v1', keep_s=KEEP_S,
            updated_at=self.wall(), plans=self.rows), ensure_ascii=False), encoding='utf-8')
        temporary.replace(self.path)

    def checkpoint(self, *, force=False):
        with self.lock:
            self._tick()
            self._prune()
            if force or self.mono() - self.last_save >= 2.:
                self._save()
                self.last_save = self.mono()

    def read(self):
        with self.lock:
            self._tick()
            self._prune()
            return deepcopy(self.rows)


def _valid_row(row):
    """A damaged row must not hide the other plans or break gameplay hooks."""
    if not isinstance(row, dict):
        return False
    numbers = ('pid', 'run_started_at', 'started_at', 'updated_at', 'elapsed_s',
               'active_s', 'paused_s', 'stp_wait_s', 'navigation_s')
    if not all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in numbers):
        return False
    if not all(isinstance(row.get(k), str) for k in ('plan_id', 'plan', 'kind', 'status', 'result')):
        return False
    segments, events = row.get('segments'), row.get('events')
    return (isinstance(segments, list) and isinstance(events, list)
            and all(isinstance(s, dict) and all(isinstance(s.get(k), str)
                    for k in ('phase', 'kind', 'node'))
                    and all(isinstance(s.get(k), (int, float)) and math.isfinite(s[k])
                    for k in ('started_at', 'ended_at', 'duration_s')) for s in segments)
            and all(isinstance(e, dict) and isinstance(e.get('event'), str)
                    and isinstance(e.get('ts'), (int, float)) and math.isfinite(e['ts']) for e in events))


def _get():
    global _RECORDER
    if _RECORDER is None and not os.environ.get('PYTEST_CURRENT_TEST'):
        with _INIT_LOCK:
            if _RECORDER is None:
                _RECORDER = _safe(lambda: ExecutionRecorder(PATH))
    return _RECORDER


def _safe(callback):
    global _LAST_ERROR
    try:
        return callback()
    except Exception:
        now = time.monotonic()
        if now - _LAST_ERROR >= 60:
            _LAST_ERROR = now
            logging.getLogger(__name__).warning('执行记录写入失败；保留原调度行为', exc_info=True)
        return None


@contextmanager
def bind(key):
    token = _CONTEXT.set(key)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def _bound(callback):
    key = _CONTEXT.get()
    if key is not None:
        _safe(lambda: callback(_get(), key) if _get() is not None else None)


def phase(text, *, kind=None):
    text = str(text)
    if kind is None:
        kind = ('stp_wait' if any(s in text for s in ('等待停钮', '等待采集结束',
                '等待当前游戏动作', '等待移动或当前游戏动作')) else
                'page_wait' if ('等' in text and any(s in text for s in ('界面','页面','刷新'))) else 'execute')
    _bound(lambda rec, key: rec.phase(key, text, kind))


def navigation(where, elapsed_s):
    _bound(lambda rec, key: rec.event(key, 'page', node=str(getattr(where,'node','unknown')),
        ready=bool(getattr(where,'stack_ready',False)),
        stack=list(getattr(where,'stack_names',())), elapsed_s=elapsed_s))


def stp_sample(sample):
    _bound(lambda rec, key: rec.event(key, 'stp', value=sample.stp,
        fresh=bool(sample.fresh), source_pid=sample.pid))


def input_requested(kind, coords, client_size):
    _bound(lambda rec, key: rec.event(key, 'input_requested', kind=kind,
        coords=list(coords), client_size=list(client_size)))


def pointer_trace(**data):
    _bound(lambda rec, key: rec.event(key, 'pointer_trace', **data))


def _lifecycle(method, run, item, *args):
    rec = _get()
    if rec is not None:
        _safe(lambda: getattr(rec, method)((rec.pid, float(run), item), *args))


def finish(run, item, status, result=''):
    _lifecycle('finish', run, item, status, result)


def suspend(run, item):
    _lifecycle('suspend', run, item)


def resume(run, item):
    _lifecycle('resume', run, item)


def interrupt_current():
    rec = _get()
    if rec is not None and rec.current is not None:
        _safe(lambda: rec.finish(rec._key(rec.current), 'interrupted', '开始新一轮'))


def checkpoint():
    # The existing UI timer also flushes a blocked worker. No new worker or game read.
    rec = _RECORDER
    if rec is not None:
        _safe(rec.checkpoint)


def trace_pump(function):
    @wraps(function)
    def traced(*args, **kwargs):
        from mm_assist.life_plan.controller import get_controller
        state = get_controller()
        rec = _get()
        if rec is None or not state.running or state.current is None:
            return function(*args, **kwargs)
        key = _safe(lambda: rec.begin(state.life_started_at, state.current))
        with bind(key):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                _safe(lambda: rec.event(key, 'exception', message=str(exc)))
                raise
            finally:
                _safe(lambda: rec.checkpoint())
    return traced


def trace_navigation(function):
    @wraps(function)
    def traced(*args, **kwargs):
        started = time.monotonic()
        result = function(*args, **kwargs)
        navigation(result, time.monotonic()-started)
        return result
    return traced


def trace_stp(function):
    @wraps(function)
    def traced(*args, **kwargs):
        result = function(*args, **kwargs)
        stp_sample(result)
        return result
    return traced


def trace_wait(label):
    """Time an existing blocking stp wait, restoring the caller's stage on exit."""
    def decorate(function):
        @wraps(function)
        def traced(*args, **kwargs):
            key = _CONTEXT.get()
            rec = _get() if key is not None else None
            old = _safe(lambda: rec.phase_snapshot(key)) if rec is not None else None
            if old is not None:
                _safe(lambda: rec.phase(key, label, 'stp_wait'))
            try:
                return function(*args, **kwargs)
            finally:
                if old is not None:
                    _safe(lambda: rec.phase(key, *old))
        return traced
    return decorate


def show_records(parent):
    """Native read-only viewer; no game capture, focus or input."""
    import tkinter as tk
    from tkinter import ttk
    win = tk.Toplevel(parent)
    win.title('生活备料 · 最近12小时执行记录')
    win.geometry('1000x680')
    columns = ('time','plan','status','total','stp','pause')
    listing = ttk.Frame(win)
    listing.pack(fill='both', expand=True)
    listing.rowconfigure(0, weight=1)
    listing.columnconfigure(0, weight=1)
    tree = ttk.Treeview(listing, columns=columns, show='headings', height=12)
    for name, label, width in zip(columns, ('开始时间','计划','状态','总耗时/秒','等待stp/秒','暂停/秒'),
                                   (150,280,90,110,110,100)):
        tree.heading(name,text=label);tree.column(name,width=width)
    tree.grid(row=0, column=0, sticky='nsew')
    list_scroll = ttk.Scrollbar(listing, orient='vertical', command=tree.yview)
    list_scroll.grid(row=0, column=1, sticky='ns')
    horizontal = ttk.Scrollbar(listing, orient='horizontal', command=tree.xview)
    horizontal.grid(row=1, column=0, sticky='ew')
    tree.configure(yscrollcommand=list_scroll.set, xscrollcommand=horizontal.set)
    details = ttk.Frame(win)
    details.pack(fill='both', expand=True)
    detail = tk.Text(details, height=18, wrap='word', state='disabled')
    detail_scroll = ttk.Scrollbar(details, orient='vertical', command=detail.yview)
    detail_scroll.pack(side='right', fill='y')
    detail.configure(yscrollcommand=detail_scroll.set)
    detail.pack(side='left', fill='both', expand=True)
    rows = {}
    def select(_=None):
        selected = tree.selection()
        if not selected or selected[0] not in rows:
            return
        row = rows[selected[0]]
        detail.configure(state='normal');detail.delete('1.0','end')
        detail.insert('end', format_detail(row));detail.configure(state='disabled')
    def refresh():
        rec = _get()
        data = _safe(rec.read) if rec is not None else []
        rows.clear()
        selection = tree.selection()
        tree.delete(*tree.get_children())
        for row in reversed(data or []):
            key = f"{row['pid']}:{row['run_started_at']}:{row['plan_id']}"
            rows[key] = row
            status = {'running':'运行中','paused':'暂停','completed':'调度完成',
                      'skipped':'已跳过','interrupted':'已中断'}.get(row['status'],row['status'])
            tree.insert('', 'end', iid=key, values=(time.strftime('%m-%d %H:%M:%S',time.localtime(row['started_at'])),
                row['plan_id']+' '+row['plan'],status,*(f"{row[k]:.2f}" for k in ('elapsed_s','stp_wait_s','paused_s'))))
        if selection and selection[0] in rows:
            tree.selection_set(selection[0]);select()
    ttk.Button(win,text='刷新记录',command=refresh).pack(anchor='e')
    tree.bind('<<TreeviewSelect>>',select)
    refresh()


def format_detail(row):
    lines = [f"进程：{row['pid']}　运行代：{row['run_started_at']}　结果：{row['result'] or '—'}",
             '总耗时包含暂停；等待stp仅计实际等待分支。界面为最后一次独立观测，未知保持unknown；阶段耗时并不证明画面始终未变。输入坐标是请求，不能证明已点击或业务成功。',
             '保留最近12小时更新的计划及明细；跨窗口运行的计划总计从其首次执行累积。',
             '阶段 / 最后观测界面 / 耗时：']
    lines += [f"{s['phase']} / {s['node']} / {s['duration_s']:.3f}秒 ({s['kind']})" for s in row['segments']]
    lines.append('界面与位置事件：')
    lines += [time.strftime('%H:%M:%S',time.localtime(e['ts']))+' '+json.dumps(e,ensure_ascii=False)
              for e in row['events']]
    return '\n'.join(lines)
