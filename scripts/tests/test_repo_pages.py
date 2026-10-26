# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which repository guides the handbook publishes, and under which names."""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/_ext"))

# repo_pages imports Sphinx only for a type annotation.
sys.modules.setdefault("sphinx", types.ModuleType("sphinx"))
sys.modules.setdefault("sphinx.application", types.SimpleNamespace(Sphinx=object))
spec = importlib.util.spec_from_file_location(
    "repo_pages", ROOT / "docs/_ext/repo_pages.py"
)
repo_pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repo_pages)


def write(path, text="# Title\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def approve(root, *sources):
    write(root / "docs/public_pages.txt", "\n".join(sources) + "\n")


def test_published_pages(tmp_path):
    write(tmp_path / "INSTALL.md")
    write(tmp_path / "CHANGELOG.md")
    write(tmp_path / "apps/README.md")
    write(tmp_path / "apps/tungsten/README.md")
    write(tmp_path / "apps/tungsten/inputs_json/README.md")
    write(tmp_path / "apps/kobayashi/docs/measurements.md")
    write(tmp_path / "examples/flow/README.md")
    write(tmp_path / "apps/tungsten/build/README.md")
    approve(tmp_path, "INSTALL.md", "CHANGELOG.md", "apps/README.md",
            "apps/tungsten/README.md", "apps/tungsten/inputs_json/README.md",
            "examples/flow/README.md")
    pages = {
        p.relative_to(tmp_path).as_posix(): d
        for p, d in repo_pages.published_pages(tmp_path).items()
    }
    assert pages == {
        "INSTALL.md": "project/install",
        "CHANGELOG.md": "project/changelog",
        "apps/README.md": "apps/index",
        "apps/tungsten/README.md": "apps/tungsten/index",
        "apps/tungsten/inputs_json/README.md": "apps/tungsten/inputs_json/index",
        "examples/flow/README.md": "examples/flow/index",
    }


def test_repository_guides_are_published():
    pages = {p.relative_to(ROOT).as_posix() for p in repo_pages.published_pages(ROOT)}
    assert {
        "INSTALL.md",
        "CONTRIBUTING.md",
        "apps/tungsten/README.md",
    } <= pages


def test_unlisted_readme_is_not_implicitly_published(tmp_path):
    write(tmp_path / "apps/reviewed/README.md")
    write(tmp_path / "apps/internal/README.md", "Internal execution record\n")
    approve(tmp_path, "apps/reviewed/README.md")
    pages = repo_pages.published_pages(tmp_path)
    assert (tmp_path / "apps/reviewed/README.md").resolve() in pages
    assert (tmp_path / "apps/internal/README.md").resolve() not in pages
    assert (tmp_path / "apps/internal/README.md").read_text() == "Internal execution record\n"


def test_publication_rejects_missing_sources_and_repository_escape(tmp_path):
    import pytest
    from publication import public_sources

    approve(tmp_path, "docs/missing.md")
    with pytest.raises(ValueError, match="Missing Markdown source"):
        public_sources(tmp_path)
    approve(tmp_path, "../outside.md")
    with pytest.raises(ValueError, match="Invalid public handbook source"):
        public_sources(tmp_path)


def test_generated_source_links_use_original_files(tmp_path, monkeypatch):
    write(tmp_path / "INSTALL.md")
    write(tmp_path / "apps/example model/README.md")
    approve(tmp_path, "INSTALL.md", "apps/example model/README.md")
    app = types.SimpleNamespace(srcdir=tmp_path / "docs")
    context = {
        "theme_source_repository": "https://github.com/example/OpenPFC/",
        "theme_source_branch": "master",
    }
    monkeypatch.setenv("OPENPFC_REVISION", "reviewed-source")
    repo_pages._source_page_context(
        app, "apps/example model/index", "page.html", context, None
    )
    assert context["theme_source_view_link"] == (
        "https://github.com/example/OpenPFC/blob/reviewed-source/apps/example%20model/README.md?plain=true"
    )
    assert context["theme_source_edit_link"] == (
        "https://github.com/example/OpenPFC/edit/master/apps/example%20model/README.md"
    )
    repo_pages._source_page_context(app, "project/install", "page.html", context, None)
    assert context["theme_source_view_link"].endswith(
        "/reviewed-source/INSTALL.md?plain=true"
    )


def test_regular_handbook_source_links_remain_theme_owned(tmp_path):
    approve(tmp_path)
    context = {"theme_source_repository": "https://github.com/example/OpenPFC/"}
    repo_pages._source_page_context(
        types.SimpleNamespace(srcdir=tmp_path / "docs"),
        "concepts/architecture",
        "page.html",
        context,
        None,
    )
    assert "theme_source_view_link" not in context
