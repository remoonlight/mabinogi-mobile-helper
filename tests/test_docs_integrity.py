# -*- coding: utf-8 -*-
"""文档链接与运行时 knowledge 路径完整性。"""
from __future__ import annotations

import re
from pathlib import Path

from mm_assist.config import ROOT
from mm_assist import knowledge_paths as kp

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
AUTHORITY = [
    ROOT / "README.md",
    ROOT / "docs" / "PRODUCT.md",
    ROOT / "docs" / "STATUS.md",
    ROOT / "docs" / "TESTING.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "RUNTIME.md",
    ROOT / "docs" / "KNOWLEDGE.md",
    ROOT / "docs" / "HISTORY.md",
]
ALLOW_MISSING_SUBSTR = (
    "外部主仓",
    "历史缺失",
    "资料缺口",
    "切片缺口",
)
ALLOW_MISSING_PREFIXES = (
    "RESULT_",
    "knowledge/playbooks/",
    "knowledge/items/",
    "knowledge/references/",
    "docs/sessions/",
    "ui_mode_findings/",
    "TW_ITEMS",
    "il2cpp_position.md",
)
SKIP_SCHEME = ("http://", "https://", "mailto:")


def _strip_target(raw: str) -> str:
    t = raw.strip().strip("<>").split()[0]
    t = t.split("#", 1)[0]
    return t.strip("'\"")


def _allowed_missing(rel: str, context: str) -> bool:
    if any(s in context for s in ALLOW_MISSING_SUBSTR):
        return True
    norm = rel.replace("\\", "/")
    return any(p in norm for p in ALLOW_MISSING_PREFIXES)


def test_authority_files_exist():
    missing = [str(p.relative_to(ROOT)) for p in AUTHORITY if not p.is_file()]
    assert missing == []


def test_knowledge_paths_exist():
    missing = []
    for name, val in vars(kp).items():
        if not isinstance(val, Path) or name != name.upper():
            continue
        if not val.exists():
            missing.append(f"{name}: {val.relative_to(ROOT)}")
    assert missing == []


def test_markdown_repo_links_resolve():
    broken: list[str] = []
    roots = [ROOT / "docs", ROOT / "knowledge", ROOT / ".cursor"]
    files = list(ROOT.glob("*.md"))
    for base in roots:
        if base.is_dir():
            files.extend(p for p in base.rglob("*.md") if ".venv" not in p.parts)
            files.extend(p for p in base.rglob("*.mdc") if ".venv" not in p.parts)
    for path in files:
        text = path.read_text(encoding="utf-8")
        for m in MD_LINK.finditer(text):
            target = _strip_target(m.group(1))
            if not target or target.startswith(SKIP_SCHEME) or target.startswith("#"):
                continue
            dest = (path.parent / target).resolve()
            try:
                dest.relative_to(ROOT.resolve())
            except ValueError:
                continue
            candidates = [dest]
            if dest.suffix == ".json" and path.parent.name == "ui":
                candidates.append((path.parent / "config" / dest.name).resolve())
                candidates.append((path.parent / "contracts" / dest.name).resolve())
            if path.parent.name == "reports":
                candidates.append((path.parent.parent / dest.name).resolve())
                candidates.append((path.parent.parent / "contracts" / dest.name).resolve())
            if any(c.exists() for c in candidates):
                continue
            rel = str(Path(target)).replace("\\", "/")
            ctx_start = max(0, m.start() - 160)
            ctx = text[ctx_start : m.end() + 80]
            if _allowed_missing(rel, ctx):
                continue
            broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert broken == [], "broken links:\n" + "\n".join(broken[:40])


def test_life_loop_keeps_data_index_and_recipe_catalogs():
    leftover = list((ROOT / "knowledge" / "life_loop").glob("*.md"))
    names = {p.name for p in leftover}
    assert names <= {"README.md"}
    # Cleanup removes empty archive indexes and obsolete plans; the two
    # versioned recipe catalogs remain sources for the retained JSON data.
    catalog_dir = ROOT / "knowledge" / "archive" / "life_loop"
    assert {p.name for p in catalog_dir.glob("*.md")} == {
        "craft_catalog.md", "process_catalog.md",
    }
