# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(DOCS_DIR / "_ext"))

project = "OpenPFC"
author = "OpenPFC contributors"
copyright = "2026, VTT Technical Research Centre of Finland Ltd"

_cmake_project = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
_version_match = re.search(r"project\(OpenPFC VERSION ([0-9.]+)", _cmake_project)
release = _version_match.group(1) if _version_match else "development"
version = release

extensions = [
    "myst_parser",
    "breathe",
    "sphinxcontrib.mermaid",
    "sphinx.ext.githubpages",
    "repo_links",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
primary_domain = "cpp"

exclude_patterns = [
    "README.md",
    ".venv/**",
    "_build/**",
    "Thumbs.db",
    ".DS_Store",
]
suppress_warnings = ["toc.not_included"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "substitution",
    "tasklist",
]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 5

_doxygen_xml = Path(
    os.environ.get(
        "OPENPFC_DOXYGEN_XML",
        REPOSITORY_ROOT / "build" / "docs" / "xml",
    )
).resolve()
breathe_projects = {"OpenPFC": str(_doxygen_xml)}
breathe_default_project = "OpenPFC"
breathe_domain_by_extension = {
    "h": "cpp",
    "hpp": "cpp",
    "c": "c",
    "cc": "cpp",
    "cpp": "cpp",
}

html_theme = "furo"
html_title = "OpenPFC documentation"
html_logo = "img/logo.png"
html_favicon = "img/logo.png"
html_theme_options = {
    "source_repository": "https://github.com/VTT-ProperTune/OpenPFC/",
    "source_branch": "master",
    "source_directory": "docs/",
}

html_show_sourcelink = True
html_copy_source = False
html_last_updated_fmt = "%Y-%m-%d"
