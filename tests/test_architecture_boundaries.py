# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
from pathlib import Path

from mm_assist.config import ROOT
from mm_assist.knowledge_paths import FRAME_JSON, SCENE_JSON, TRANSITION_GRAPH_JSON
from mm_assist.memory.process import PROCESS_QUERY_INFORMATION, PROCESS_VM_READ
from mm_assist.cli import ALLOWED_CMDS

SRC = ROOT / "src" / "mm_assist"
FORBIDDEN_IMPORTS = {
    "mm_assist.map_atlas",
    "mm_assist.runtime",
    "mm_assist.intent_hud",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_no_forbidden_modules():
    hits: list[str] = []
    for path in SRC.rglob("*.py"):
        mods = _imports(path)
        bad = mods & FORBIDDEN_IMPORTS
        if bad:
            hits.append(f"{path.relative_to(ROOT)}: {sorted(bad)}")
    assert hits == []


def test_openprocess_is_readonly():
    write = 0x0008 | 0x0020  # PROCESS_VM_WRITE | PROCESS_VM_OPERATION
    assert PROCESS_VM_READ == 0x0010
    assert PROCESS_QUERY_INFORMATION == 0x0400
    assert (PROCESS_VM_READ | PROCESS_QUERY_INFORMATION) & write == 0


def test_cli_surface_is_readonly():
    assert "watch" in ALLOWED_CMDS
    assert "capture" in ALLOWED_CMDS
    assert "benchmark" in ALLOWED_CMDS
    forbidden = {"gather", "craft", "map-goto", "map-town-click", "probe-backend"}
    assert not (forbidden & set(ALLOWED_CMDS))


def test_knowledge_runtime_files_exist():
    assert SCENE_JSON.is_file()
    assert FRAME_JSON.is_file()
    assert TRANSITION_GRAPH_JSON.is_file()
