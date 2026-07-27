#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
DIAGNOSTIC_RE = re.compile(r"\b(?:WARNING|ERROR):")
GENERATED_API_MARKERS = (
    "/docs/api/generated/",
    "docs/api/generated/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Sphinx diagnostics with a generated-API warning budget."
    )
    parser.add_argument("log", type=Path, help="Sphinx build log")
    parser.add_argument(
        "--generated-api-budget",
        type=int,
        default=int(os.environ.get("OPENPFC_GENERATED_API_WARNING_BUDGET", "50")),
        help="maximum accepted diagnostics from Exhale-generated pages",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.log.is_file():
        print(f"check_sphinx_log: missing log: {args.log}", file=sys.stderr)
        return 2

    text = ANSI_RE.sub("", args.log.read_text(encoding="utf-8", errors="replace"))
    generated: list[str] = []
    maintained: list[str] = []

    for line in text.replace("\r", "\n").splitlines():
        if not DIAGNOSTIC_RE.search(line):
            continue
        if any(marker in line for marker in GENERATED_API_MARKERS):
            generated.append(line)
        else:
            maintained.append(line)

    if generated:
        print(
            "check_sphinx_log: "
            f"{len(generated)} generated API diagnostic(s) "
            f"(budget {args.generated_api_budget})"
        )
        for line in generated:
            print(f"  {line}")
    else:
        print("check_sphinx_log: no generated API diagnostics")

    if maintained:
        print(
            "check_sphinx_log: maintained documentation diagnostics found",
            file=sys.stderr,
        )
        for line in maintained:
            print(f"  {line}", file=sys.stderr)
        return 1

    if len(generated) > args.generated_api_budget:
        print(
            "check_sphinx_log: generated API diagnostic budget exceeded "
            f"({len(generated)} > {args.generated_api_budget})",
            file=sys.stderr,
        )
        return 1

    print("check_sphinx_log: maintained documentation is warning-free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
