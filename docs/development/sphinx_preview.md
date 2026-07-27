<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Sphinx documentation site

OpenPFC publishes one documentation site. MyST renders the Markdown under
`docs/`, Doxygen extracts the C++ API as XML, and Breathe places that generated
reference inside the same Sphinx navigation and search index.

Configuration lives in [`conf.py`](../conf.py). Python dependencies are managed
with `uv` in `docs/pyproject.toml` and `docs/uv.lock`. Doxygen XML settings live
in [`docs/CMakeLists.txt`](../CMakeLists.txt).

## Prerequisites

Install Doxygen and Ninja in addition to Python and `uv`. On Debian or Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y doxygen ninja-build
```

The standalone documentation CMake project does not configure or build OpenPFC,
MPI, HeFFTe, CUDA, HIP, or a TeX distribution.

## Build locally

Run from the repository root:

```bash
uv sync --project docs --locked
bash scripts/build_docs.sh build
```

The command performs two steps:

1. configure `docs/CMakeLists.txt` and generate `build/docs/xml/` with Doxygen;
2. build the complete Sphinx site into `site/`, then validate its diagnostics.

Maintained Markdown, navigation, and MyST links must be warning-free. Generated
API pages are curated to stable public concepts rather than mirroring every
header and implementation helper.

Serve the result with any static HTTP server, for example:

```bash
python3 -m http.server --directory site 8000
```

Then open `http://127.0.0.1:8000/`.

## Interactive preview

Generate the API XML and start `sphinx-autobuild` with:

```bash
uv sync --project docs --locked
bash scripts/build_docs.sh serve
```

The preview server rebuilds Markdown and configuration changes automatically.
When public C++ comments or declarations change, restart the command so the
Doxygen XML is regenerated before Sphinx reloads the API pages.

## Link validation

The documentation uses complementary checks:

1. `scripts/check_doc_links.py` verifies repository-relative Markdown targets,
   including files outside `docs/`;
2. Sphinx validates the rendered document tree, MyST cross-references, anchors,
   Breathe input, and navigation;
3. `docs/_ext/repo_links.py` rewrites valid links outside `docs/` to
   commit-pinned GitHub URLs in the rendered site.

This keeps source links useful when Markdown is read directly on GitHub without
requiring duplicate copies of `INSTALL.md`, application inputs, headers, or
scripts inside the documentation source tree.

## API generation

Doxygen is not a second presentation layer. It only generates XML:

```bash
cmake -S docs -B build/docs -GNinja
cmake --build build/docs --target openpfc-doxygen-xml
```

Breathe reads `build/docs/xml/` and renders the curated declarations from
[`api/index.md`](../api/index.md). The compatibility target `docs` still points
to `openpfc-doxygen-xml` for existing local commands.

OpenPFC has pre-existing source-comment defects in its public headers. They are
classified in `scripts/check_doxygen_log.py` by category and exact count. CI
fails if a known category grows or an unrecognized warning appears. When a
cleanup reduces a category, lower its recorded baseline in the same change.

## Published layout

| URL area | Content |
|----------|---------|
| Pages root | Unified Sphinx site with prose and API |
| `/api/` | Breathe-rendered C++ API inside the main site |
| `/dev/` | Compatibility redirect to `/api/` |

The Documentation workflow uploads one HTML artifact. A push to `master`
publishes that artifact through GitHub Pages.

## Navigation

[`index.md`](../index.md) contains the captioned Sphinx toctrees. Major
subdirectories retain local `README.md` indexes so the source tree remains
understandable when browsed without rendering.

Add a new page to its nearest local index. Add it to `index.md` when it belongs
in the primary site navigation.

## See also

- [Documentation index](../index.md)
- [Contributing to documentation](contributing-docs.md)
- [Documentation versioning](documentation_versioning.md)
