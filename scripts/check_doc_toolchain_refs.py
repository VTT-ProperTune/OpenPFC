#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "vtt-propertune.github.io/OpenPFC/dev/": "use the integrated docs/api reference",
    "mkdocs.yml": "use docs/conf.py and docs/index.md",
    "build_mkdocs.sh": "use scripts/build_docs.sh",
    "mkdocs_preview.md": "use docs/development/sphinx_preview.md",
}


def source_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files.extend((ROOT / "docs").rglob("*.md"))
    files.extend(
        [
            ROOT / ".github" / "workflows" / "docs.yml",
            ROOT / "scripts" / "build_docs.sh",
        ]
    )
    return sorted(path for path in files if path.is_file())


def main() -> int:
    findings: list[str] = []
    for path in source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for token, replacement in FORBIDDEN.items():
                if token in line:
                    rel = path.relative_to(ROOT)
                    findings.append(
                        f"{rel}:{line_number}: contains {token!r}; {replacement}"
                    )

    if findings:
        print("check_doc_toolchain_refs: retired references found", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1

    print("check_doc_toolchain_refs: OK (no retired documentation references)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
