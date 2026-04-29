#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify relative markdown links in selected documentation paths (repo root)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Markdown: [text](url) — skip images for optional separate check; we validate same.
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

ROOT = Path(__file__).resolve().parents[1]

SCAN_GLOBS = [
    "docs/**/*.md",
    "README.md",
    "INSTALL.md",
    "examples/README.md",
    "apps/*/README.md",
]


def strip_fenced_code(text: str) -> str:
    """Remove fenced code blocks so example URLs in snippets are not checked."""
    return re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)


def collect_markdown_files() -> list[Path]:
    out: list[Path] = []
    for pattern in SCAN_GLOBS:
        out.extend(ROOT.glob(pattern))
    return sorted(set(p.resolve() for p in out if p.is_file()))


def check_file(md_path: Path) -> list[tuple[str, str]]:
    """Return list of (link_target, error_message) for broken links."""
    bad: list[tuple[str, str]] = []
    text = strip_fenced_code(md_path.read_text(encoding="utf-8", errors="replace"))
    base = md_path.parent
    for m in LINK_RE.finditer(text):
        raw = m.group(1).strip()
        if not raw or raw.startswith(("#", "mailto:")):
            continue
        if raw.startswith("http://") or raw.startswith("https://"):
            continue
        path_part = raw.split("#", 1)[0].split("?", 1)[0]
        if not path_part:
            continue
        if path_part.startswith("/"):
            target = (ROOT / path_part.lstrip("/")).resolve()
        else:
            target = (base / path_part).resolve()
        try:
            target.relative_to(ROOT)
        except ValueError:
            bad.append((raw, "escapes repository root"))
            continue
        if not target.exists():
            bad.append((raw, f"missing: {target.relative_to(ROOT)}"))
    return bad


def main() -> int:
    all_bad: list[tuple[Path, str, str]] = []
    for md in collect_markdown_files():
        for link, err in check_file(md):
            all_bad.append((md, link, err))

    if not all_bad:
        print("check_doc_links: OK (no broken relative links in scanned markdown)")
        return 0

    print("check_doc_links: broken relative links\n", file=sys.stderr)
    for md, link, err in all_bad:
        rel_md = md.relative_to(ROOT)
        print(f"  {rel_md}: [{link}] -> {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
