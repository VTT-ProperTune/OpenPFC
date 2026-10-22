# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which repository guides the handbook publishes, and under which names."""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# repo_pages imports Sphinx only for a type annotation.
sys.modules.setdefault("sphinx", types.ModuleType("sphinx"))
sys.modules.setdefault("sphinx.application", types.SimpleNamespace(Sphinx=object))
spec = importlib.util.spec_from_file_location("repo_pages", ROOT / "docs/_ext/repo_pages.py")
repo_pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repo_pages)


def write(path, text="# Title\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_published_pages(tmp_path):
    write(tmp_path / "INSTALL.md")
    write(tmp_path / "CHANGELOG.md")
    write(tmp_path / "apps/README.md")
    write(tmp_path / "apps/tungsten/README.md")
    write(tmp_path / "apps/tungsten/inputs_json/README.md")
    write(tmp_path / "apps/kobayashi/docs/measurements.md")
    write(tmp_path / "examples/flow/README.md")
    write(tmp_path / "apps/tungsten/build/README.md")
    pages = {p.relative_to(tmp_path).as_posix(): d for p, d in repo_pages.published_pages(tmp_path).items()}
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
    assert {"INSTALL.md", "CHANGELOG.md", "CONTRIBUTING.md", "apps/tungsten/README.md"} <= pages
