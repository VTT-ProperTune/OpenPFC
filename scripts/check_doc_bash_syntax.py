#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Syntax-check ```bash fenced blocks under docs/ (bash -n)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def bash_blocks(text: str) -> list[str]:
    """Extract fenced bash blocks (```bash or ```sh)."""
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        fence = lines[i].strip()
        if fence.startswith("```") and fence[3:].strip() in ("bash", "sh"):
            i += 1
            chunk: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                chunk.append(lines[i])
                i += 1
            body = "\n".join(chunk).strip()
            if body:
                out.append(body)
        i += 1
    return out


def check_block(script: str, path: Path, index: int) -> bool:
    r = subprocess.run(
        ["bash", "-n"],
        input=script.encode(),
        capture_output=True,
    )
    if r.returncode != 0:
        err = (r.stderr or b"").decode(errors="replace").strip()
        print(f"check_doc_bash_syntax: FAIL {path} block {index + 1}: {err}", file=sys.stderr)
        return False
    return True


def main() -> int:
    md_files = sorted(DOCS.rglob("*.md"))
    failed = False
    total = 0
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        blocks = bash_blocks(text)
        for j, block in enumerate(blocks):
            total += 1
            if not check_block(block + "\n", path, j):
                failed = True
    if failed:
        print("check_doc_bash_syntax: FAILED", file=sys.stderr)
        return 1
    print(f"check_doc_bash_syntax: OK ({total} bash blocks in docs/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
