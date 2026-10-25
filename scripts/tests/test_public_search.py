# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The shared index contains rendered article text, not chrome or source files."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "public_search", Path(__file__).resolve().parents[2] / "docs/_ext/public_search.py"
)
public_search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public_search)


def test_export_visible_articles(tmp_path):
    page = tmp_path / "api/field/index.html"
    page.parent.mkdir(parents=True)
    page.write_text(
        "<title>Fields - OpenPFC documentation</title><nav>Navigation</nav>"
        "<svg><title>Menu</title></svg><article><h1>Fields</h1>"
        "<p>Host &amp; device</p><code>Field&lt;double&gt;</code>"
        "<script>privateMarker</script><svg><title>Icon</title></svg></article>"
    )
    (tmp_path / "README.md").write_text("Source marker")
    for excluded in ["search", "genindex", "_static"]:
        target = tmp_path / excluded / "index.html"
        target.parent.mkdir()
        target.write_text("<article>Excluded marker</article>")
    app = SimpleNamespace(outdir=tmp_path, builder=SimpleNamespace(format="html"))
    public_search.export_index(app, None)
    documents = json.loads((tmp_path / "search-index.json").read_text())
    assert documents == [
        {
            "href": "api/field/index.html",
            "title": "Fields",
            "text": "Fields Host & device Field<double>",
            "section": "C++ API",
        }
    ]


def test_failed_build_does_not_publish_an_index(tmp_path):
    app = SimpleNamespace(outdir=tmp_path, builder=SimpleNamespace(format="html"))
    public_search.export_index(app, RuntimeError("build failed"))
    assert not (tmp_path / "search-index.json").exists()
