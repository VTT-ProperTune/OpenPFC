<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Contributing to documentation

Repository-wide pointers (tests, changelog expectations): **[`../CONTRIBUTING.md`](../CONTRIBUTING.md)**. The doc index **[`README.md`](README.md)** lists **`CONTRIBUTING.md`** and **`CHANGELOG.md`** under *Where to go first* and *Other*.

## Before you open a PR

1. **Relative links** — From a file under **`docs/`**, paths like **`../INSTALL.md`** reach the repo root. Nested files (e.g. **`docs/getting_started/01-basics/README.md`**) need enough **`..`** segments. Run the checker (below).
2. **SPDX** — When you edit a file that already has **`SPDX-FileCopyrightText`**, update the year to the current calendar year (see project rules).
3. **Cross-link new guides** — Add a row to **[`README.md`](README.md)** (*Where to go first* and the right topic table—**Guides by topic** or **Tutorials**), and wire from **[`quickstart.md`](quickstart.md)** *Next steps* and/or **[`getting_started/README.md`](getting_started/README.md)** when the audience is learners. For discoverability, add a line to the **[`faq.md`](faq.md)** *Documentation map* if the page answers a common “where do I find…?” question.
4. **Topic-specific hooks** — Examples: extension / **`App`** docs should link **[`app_pipeline.md`](app_pipeline.md)** and **[`class_tour.md`](class_tour.md)**; validation UX should mention **[`parameter_validation.md`](parameter_validation.md)** and the root **`README.md`** validation section.

## Check markdown links locally

From the repository root:

```bash
python3 scripts/check_doc_links.py
```

The script scans **`docs/**/*.md`**, **`README.md`**, **`INSTALL.md`**, **`examples/README.md`**, and **`apps/*/README.md`**. It resolves **relative** links and paths starting with **`/`** (repository root). **http(s)** links are not fetched.

The same check runs in **CI** as part of the Documentation workflow (`.github/workflows/docs.yml`).

## Style

- Prefer **tables** and **short sections** over long unstructured prose.
- Link to **`architecture.md`** for layering instead of duplicating the kernel/runtime story.
- For C++ behavior, point to **headers** and **`examples/`** rather than copying signatures.

## See also

- **[`styleguide.md`](styleguide.md)** — code and header style (also informs doc examples)
