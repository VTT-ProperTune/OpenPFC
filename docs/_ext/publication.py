# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read the explicit source allowlist for the public handbook."""

from pathlib import Path, PurePosixPath
from html import escape
import json


def public_sources(repository_root: Path) -> set[Path]:
    """Resolve explicitly selected Markdown sources, rejecting missing entries."""
    root = repository_root.resolve()
    manifest = root / "docs/public_pages.txt"
    sources = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        relative = PurePosixPath(name)
        path = (root / name).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(root)
        ):
            raise ValueError(f"Invalid public handbook source: {name}")
        if path.suffix != ".md" or not path.is_file():
            raise ValueError(f"Missing Markdown source in public handbook list: {name}")
        sources.add(path)
    return sources


def public_redirects(repository_root: Path) -> dict[str, str]:
    """Map former handbook docnames to selected public guides."""
    manifest = repository_root / "docs/public_redirects.json"
    return json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}


def redirect_pages(app):
    """Keep old URLs useful without rendering their operational source text."""
    for old, target in public_redirects(Path(app.srcdir).parent).items():
        url = escape(app.builder.get_relative_uri(old, target), quote=True)
        yield old, {
            "title": "OpenPFC guide moved",
            "body": (
                '<h1>OpenPFC guide moved</h1>'
                f'<p>Continue to the <a href="{url}">public OpenPFC guide</a>.</p>'
            ),
            "metatags": (
                '<meta name="openpfc-public-redirect" content="true">'
                f'<link rel="canonical" href="{url}">'
                f'<meta http-equiv="refresh" content="0; url={url}">'
            ),
            "sourcename": "",
        }, "page.html"


def setup(app):
    app.connect("html-collect-pages", redirect_pages)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
