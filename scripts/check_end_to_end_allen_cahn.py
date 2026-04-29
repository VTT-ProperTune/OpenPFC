#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ensure Allen–Cahn numeric CLI args in end_to_end_visualization match apps/allen_cahn/README."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "apps" / "allen_cahn" / "README.md"
END_TO_END = ROOT / "docs" / "tutorials" / "end_to_end_visualization.md"

# Seven tokens: nx ny n_steps dt M epsilon driving_force
ARGS_RE = re.compile(
    r"allen_cahn(?:\.exe)?\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
    re.IGNORECASE,
)


def extract_args(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = ARGS_RE.search(text)
    if not m:
        return None
    return " ".join(m.groups())


def main() -> int:
    if not README.is_file() or not END_TO_END.is_file():
        print("check_end_to_end_allen_cahn: missing README or tutorial", file=sys.stderr)
        return 2
    a = extract_args(README)
    b = extract_args(END_TO_END)
    if a is None:
        print("check_end_to_end_allen_cahn: could not parse apps/allen_cahn/README.md", file=sys.stderr)
        return 2
    if b is None:
        print("check_end_to_end_allen_cahn: could not parse end_to_end_visualization.md", file=sys.stderr)
        return 2
    if a != b:
        print(
            "check_end_to_end_allen_cahn: Allen–Cahn numeric args differ:\n"
            f"  README:    {a}\n"
            f"  Tutorial:  {b}\n"
            "  Align docs/tutorials/end_to_end_visualization.md with apps/allen_cahn/README.md.",
            file=sys.stderr,
        )
        return 1
    print("check_end_to_end_allen_cahn: OK (Allen–Cahn args match README example)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
