#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Remove **bold** from prose markdown outside fenced code blocks (conservative)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

BOLD = re.compile(r"\*\*([^*]+)\*\*")


def strip_bold_outside_fences(text: str) -> str:
    parts = text.split("```")
    out: list[str] = []
    for i, chunk in enumerate(parts):
        if i % 2 == 0:
            out.append(BOLD.sub(r"\1", chunk))
        else:
            out.append(chunk)
    return "```".join(out)


def main() -> int:
    roots = [Path("docs")]
    for name in (
        "README.md",
        "CONTRIBUTING.md",
        "INSTALL.md",
        "examples/README.md",
        "scripts/README.md",
        "tests/benchmarks/README.md",
        ".github/workflows/README.md",
    ):
        p = Path(name)
        if p.is_file():
            roots.append(p)
    roots.extend(sorted(Path("apps").glob("*/README.md")))
    changed = 0
    for root in roots:
        if root.is_file():
            files = [root]
        else:
            files = sorted(root.rglob("*.md"))
        for path in files:
            if "docs/image-prompts.md" in str(path):
                continue
            raw = path.read_text(encoding="utf-8")
            new = strip_bold_outside_fences(raw)
            if new != raw:
                path.write_text(new, encoding="utf-8")
                changed += 1
    print(f"unbold_markdown_docs: updated {changed} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
