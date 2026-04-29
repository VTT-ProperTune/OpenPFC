<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# MkDocs preview (Material)

The repository ships an optional **MkDocs + Material** site for browsing `docs/**/*.md` in a browser. Configuration lives at **`mkdocs.yml`** in the **repository root**; Python dependencies are managed with **`uv`** under this directory (`pyproject.toml`, `uv.lock`).

## Commands (from repository root)

Material prints an optional **MkDocs 2.0** banner unless you set **`NO_MKDOCS_2_WARNING=1`** (see upstream `material/templates/__init__.py`). Use the wrapper or export the variable:

```bash
# Install deps once (creates docs/.venv)
uv sync --project docs

# Live preview — http://127.0.0.1:8000/
NO_MKDOCS_2_WARNING=1 uv run --project docs mkdocs serve

# Static HTML in ./site/ (gitignored); strict = fail on warnings
./scripts/build_mkdocs.sh build --strict
```

Or without the wrapper:

```bash
NO_MKDOCS_2_WARNING=1 uv run --project docs mkdocs build --strict
```

## Notes

- **Relative links** to files **outside** `docs/` (for example `../INSTALL.md`, `../apps/...`) are valid in GitHub-style browsing but MkDocs will **warn** during `build`/`serve` because those paths are not part of the MkDocs `docs_dir`. The HTML output still lists pages; some links may be broken in the static site unless you mirror those files under `docs/` or replace links with absolute repository URLs.
- **Navigation** in `mkdocs.yml` is a **curated subset**; every other markdown file under `docs/` is still built and reachable via **search** (unless excluded).
- This site **does not replace** Doxygen HTML (`OpenPFC_BUILD_DOCUMENTATION`) or the published API reference — it is a **prose** companion.

## See also

- [`contributing-docs.md`](contributing-docs.md) — link checks and doc workflow  
- [`README.md`](../README.md) — full prose index  
