# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export visible, rendered handbook/API text for the website's shared search."""

from html.parser import HTMLParser
import json
from pathlib import Path


class ArticleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = []
        self.text = []
        self.in_title = False
        self.in_article = False
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag == "title" and not self.title:
            self.in_title = True
        if tag == "article":
            self.in_article = True
        if tag in {"script", "style", "nav", "svg"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag == "article":
            self.in_article = False
        if tag in {"script", "style", "nav", "svg"}:
            self.skip -= 1

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        if self.in_article and not self.skip:
            self.text.append(data)


def export_index(app, exception):
    if exception or app.builder.format != "html":
        return
    root = Path(app.outdir)
    documents = []
    for path in sorted(root.rglob("*.html")):
        relative = path.relative_to(root)
        if relative.parts[0] in {"_static", "_downloads", "genindex", "search"}:
            continue
        article = ArticleText()
        article.feed(path.read_text(encoding="utf-8"))
        text = " ".join(" ".join(article.text).split())
        if not text:
            continue
        documents.append(
            {
                "href": relative.as_posix(),
                "title": "".join(article.title).removesuffix(
                    " - OpenPFC documentation"
                ),
                "text": text,
                "section": "C++ API" if relative.parts[0] == "api" else "Handbook",
            }
        )
    (root / "search-index.json").write_text(
        json.dumps(documents, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def setup(app):
    app.connect("build-finished", export_index)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
