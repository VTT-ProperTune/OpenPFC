#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ensure docs/examples_catalog.md lists the same executables as examples/CMakeLists.txt."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CMAKE = ROOT / "examples" / "CMakeLists.txt"
CATALOG = ROOT / "docs" / "examples_catalog.md"

ADD_EXE_RE = re.compile(r"^\s*add_executable\(\s*([^\s\)]+)")


def targets_from_cmake() -> set[str]:
    text = CMAKE.read_text(encoding="utf-8", errors="replace")
    out: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = ADD_EXE_RE.match(line)
        if m:
            out.add(m.group(1))
    return out


def targets_from_catalog() -> set[str]:
    text = CATALOG.read_text(encoding="utf-8", errors="replace")
    in_full = False
    names: list[str] = []
    for line in text.splitlines():
        if line.startswith("## Full catalog"):
            in_full = True
            continue
        if in_full and line.startswith("## ") and "Full catalog" not in line:
            break
        if not in_full or not line.startswith("|"):
            continue
        m = re.match(r"\|\s*`([^`]+)`\s*\|", line)
        if m:
            names.append(m.group(1))
    return set(names)


def main() -> int:
    if not CMAKE.is_file() or not CATALOG.is_file():
        print("check_examples_catalog: missing CMakeLists or catalog", file=sys.stderr)
        return 2
    cm = targets_from_cmake()
    doc = targets_from_catalog()
    only_cmake = sorted(cm - doc)
    only_doc = sorted(doc - cm)
    if not only_cmake and not only_doc:
        print("check_examples_catalog: OK (CMake targets match docs/examples_catalog.md)")
        return 0
    print("check_examples_catalog: mismatch\n", file=sys.stderr)
    if only_cmake:
        print("  In CMakeLists.txt but not catalog:", ", ".join(only_cmake), file=sys.stderr)
    if only_doc:
        print("  In catalog but not CMakeLists.txt:", ", ".join(only_doc), file=sys.stderr)
    print("  Update docs/examples_catalog.md or examples/CMakeLists.txt.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
