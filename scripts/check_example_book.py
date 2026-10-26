#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check that rendered lesson chapters contain code, citations and local resources."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Chapter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_code = False
        self.code: list[str] = []
        self.links: list[str] = []
        self.has_references = False

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "code" and "cpp" in attrs.get("class", "").split():
            self.in_code = True
            self.code.append("")
        self.has_references |= attrs.get("id") == "refs"
        for key in ("src", "href"):
            if attrs.get(key):
                self.links.append(attrs[key])

    def handle_endtag(self, tag):
        if tag == "code":
            self.in_code = False

    def handle_data(self, data):
        if self.in_code:
            self.code[-1] += data


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_example_book.py <rendered-lesson-directory>")
    directory = Path(sys.argv[1]).resolve()
    errors = []
    for source in sorted((ROOT / "examples").glob("*/*/index.qmd")):
        page = directory / source.relative_to(ROOT / "examples").with_suffix(".html")
        if not page.is_file():
            errors.append(f"Missing rendered chapter: {page}")
            continue
        chapter = Chapter()
        chapter.feed(page.read_text())
        if "lesson.zip" not in chapter.links:
            errors.append(f"Missing runnable lesson download: {page}")
        if not chapter.code or any(len(block.strip()) < 20 for block in chapter.code):
            errors.append(f"Missing or empty C++ source excerpts: {page}")
        if any("// start snippet" in block or "// end snippet" in block for block in chapter.code):
            errors.append(f"Unresolved snippet selection included source markers: {page}")
        # Geometry/API lessons need not attach an unrelated scientific citation.
        front_matter = source.read_text().split("---", 2)[1]
        if "bibliography:" in front_matter and not chapter.has_references:
            errors.append(f"Missing bibliography: {page}")
        for link in chapter.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                continue
            target = page.parent / unquote(url.path)
            if not target.exists():
                errors.append(f"Missing resource in {page}: {link}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Example chapters contain nonempty C++ excerpts, bibliographies and local resources.")


if __name__ == "__main__":
    main()
