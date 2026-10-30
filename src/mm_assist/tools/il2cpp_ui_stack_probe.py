"""探测当前 UI 面板栈（只读，无键鼠）。

用法：
  python -m mm_assist.tools.il2cpp_ui_stack_probe
  python -m mm_assist.tools.il2cpp_ui_stack_probe --watch 0.5 --count 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from mm_assist.memory.readers.ui_stack import Il2CppUiStackReader


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="IL2CPP UI 面板栈探测（只读）")
    ap.add_argument("--watch", type=float, default=0, help="秒间隔；0=只读一次")
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    r = Il2CppUiStackReader()
    ok_once = False
    n = 0
    try:
        while True:
            snap = r.read()
            if args.json:
                print(json.dumps(snap.to_dict(), ensure_ascii=False))
            else:
                err = f" err={snap.error}" if snap.error else ""
                print(f"OK line={snap.line!r} n={snap.n} names={snap.names}{err}")
            if not snap.error or snap.line not in ("—", ""):
                ok_once = True
            n += 1
            if not args.watch:
                break
            if args.count and n >= args.count:
                break
            time.sleep(args.watch)
    finally:
        r.close()
    return 0 if ok_once else 2


if __name__ == "__main__":
    sys.exit(main())
