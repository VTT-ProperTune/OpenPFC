#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ensure the examples reference lists the executables from examples/CMakeLists.txt."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CMAKE = ROOT / "examples" / "CMakeLists.txt"
CATALOG = ROOT / "docs" / "reference" / "examples_catalog.md"

ADD_EXE_RE = re.compile(r"^\s*add_executable\(\s*([^\s\)]+)")


def targets_from_cmake() -> set[str]:
    text = CMAKE.read_text(encoding="utf-8", errors="replace")
    out: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ADD_EXE_RE.match(line)
        if match:
            out.add(match.group(1))
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
        match = re.match(r"\|\s*`([^`]+)`\s*\|", line)
        if match:
            names.append(match.group(1))
    return set(names)


def main() -> int:
    if not CMAKE.is_file() or not CATALOG.is_file():
        print(
            "check_examples_catalog: missing examples/CMakeLists.txt or "
            "docs/reference/examples_catalog.md",
            file=sys.stderr,
        )
        return 2

    cmake_targets = targets_from_cmake()
    documented_targets = targets_from_catalog()
    only_cmake = sorted(cmake_targets - documented_targets)
    only_documented = sorted(documented_targets - cmake_targets)

    if not only_cmake and not only_documented:
        print(
            "check_examples_catalog: OK "
            "(CMake targets match docs/reference/examples_catalog.md)"
        )
        return 0

    print("check_examples_catalog: mismatch\n", file=sys.stderr)
    if only_cmake:
        print(
            "  In examples/CMakeLists.txt but not catalog: "
            + ", ".join(only_cmake),
            file=sys.stderr,
        )
    if only_documented:
        print(
            "  In catalog but not examples/CMakeLists.txt: "
            + ", ".join(only_documented),
            file=sys.stderr,
        )
    print(
        "  Update docs/reference/examples_catalog.md or examples/CMakeLists.txt.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
