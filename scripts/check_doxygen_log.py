#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

WARNING_RE = re.compile(r"\bwarning:\s*(?P<message>.*)$")

# Baseline captured when the unified Sphinx/Breathe site was introduced.
# These are legacy source-comment defects, not acceptable targets for new code.
BASELINE = {
    "duplicate_example": 60,
    "parameter_drift": 36,
    "unresolved_reference": 30,
    "missing_example_include": 19,
    "malformed_markup": 4,
    "parameter_section_without_arguments": 1,
}


def classify(message: str) -> str | None:
    if "already documented. Ignoring documentation found here." in message:
        return "duplicate_example"
    if "included file" in message and "is not found" in message:
        return "missing_example_include"
    if (
        "unable to resolve reference" in message
        or "explicit link request" in message
    ):
        return "unresolved_reference"
    if (
        "argument '" in message
        or "The following parameter" in message
        or "The following parameters" in message
    ):
        return "parameter_drift"
    if (
        "end of comment block while expecting command" in message
        or "tag without matching" in message
    ):
        return "malformed_markup"
    if "has @param documentation sections but no arguments" in message:
        return "parameter_section_without_arguments"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Doxygen warnings against the legacy debt baseline."
    )
    parser.add_argument("log", type=Path, help="Doxygen warning log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.log.is_file():
        print(f"check_doxygen_log: missing log: {args.log}", file=sys.stderr)
        return 2

    counts: Counter[str] = Counter()
    unknown: list[str] = []

    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = WARNING_RE.search(line)
        if not match:
            continue
        message = match.group("message")
        category = classify(message)
        if category is None:
            unknown.append(line)
        else:
            counts[category] += 1

    print("check_doxygen_log: legacy diagnostic baseline")
    for category, limit in BASELINE.items():
        print(f"  {category}: {counts[category]} / {limit}")

    failed = False
    for category, count in counts.items():
        limit = BASELINE.get(category)
        if limit is None or count > limit:
            failed = True
            print(
                f"check_doxygen_log: {category} exceeds baseline "
                f"({count} > {limit})",
                file=sys.stderr,
            )

    if unknown:
        failed = True
        print("check_doxygen_log: unrecognized warnings:", file=sys.stderr)
        for line in unknown:
            print(f"  {line}", file=sys.stderr)

    if failed:
        return 1

    total = sum(counts.values())
    print(
        "check_doxygen_log: OK "
        f"({total} known legacy warning(s), no category increased)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
