# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote, unquote

from sphinx.application import Sphinx

_LINK_RE = re.compile(
    r"(?P<prefix>!?\[[^\]]*\]\()"
    r"(?P<target><[^>]+>|[^)\s]+)"
    r"(?P<suffix>(?:\s+[\"'][^\"']*[\"'])?\))"
)


def _rewrite_repository_links(app: Sphinx, docname: str, source: list[str]) -> None:
    source_path = Path(app.env.doc2path(docname, base=True)).resolve()
    docs_dir = Path(app.srcdir).resolve()
    repository_root = docs_dir.parent
    repository = os.environ.get("OPENPFC_REPOSITORY", "VTT-ProperTune/OpenPFC")
    revision = os.environ.get("OPENPFC_REVISION", "master")

    def replace(match: re.Match[str]) -> str:
        raw_target = match.group("target")
        bracketed = raw_target.startswith("<") and raw_target.endswith(">")
        target = raw_target[1:-1] if bracketed else raw_target

        if target.startswith(("#", "/", "mailto:", "http://", "https://")):
            return match.group(0)

        path_part, separator, fragment = target.partition("#")
        resolved = (source_path.parent / unquote(path_part)).resolve()

        try:
            resolved.relative_to(docs_dir)
            return match.group(0)
        except ValueError:
            pass

        try:
            relative = resolved.relative_to(repository_root)
        except ValueError:
            return match.group(0)

        if not resolved.exists():
            return match.group(0)

        kind = "tree" if resolved.is_dir() else "blob"
        url = (
            f"https://github.com/{repository}/{kind}/{revision}/"
            f"{quote(relative.as_posix(), safe='/')}"
        )
        if separator:
            url = f"{url}#{fragment}"
        if bracketed:
            url = f"<{url}>"

        return f"{match.group('prefix')}{url}{match.group('suffix')}"

    source[0] = _LINK_RE.sub(replace, source[0])


def setup(app: Sphinx) -> dict[str, object]:
    app.connect("source-read", _rewrite_repository_links)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
