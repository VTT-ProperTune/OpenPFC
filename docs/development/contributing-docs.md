<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Contributing to documentation

Repository-wide pointers (tests, changelog expectations): [`../CONTRIBUTING.md`](../../CONTRIBUTING.md). The doc index [`README.md`](../README.md) lists onboarding under *Where to go first*; contributor-only topics (roadmap, scalability experiment doc) sit in **Contributors and project internals**.

## Before you open a PR

1. Relative links — From a file under `docs/`, paths like `../INSTALL.md` reach the repo root. Nested files (e.g. `docs/getting_started/01-basics/README.md`) need enough `..` segments. Run the checker (below).
2. SPDX — When you edit a file that already has `SPDX-FileCopyrightText`, update the year to the current calendar year (see project rules).
3. Cross-link new guides — Add a row to [`README.md`](../README.md) (*First-time onboarding*, *Where to go first*, and the right topic table—Guides by topic or Tutorials). Short **goal-oriented** pages belong under [`recipes/`](../recipes) with an index row in [`recipes/README.md`](../recipes/README.md). List narrative tutorials in [`tutorials/README.md`](../tutorials/README.md), update [`personas.md`](personas.md) if a new **role** entry fits, and wire from [`quickstart.md`](../quickstart.md) and [`start_here_15_minutes.md`](../start_here_15_minutes.md) when the audience is new users. For discoverability, add a line to the [`faq.md`](../faq.md) *Documentation map* if the page answers a common “where do I find…?” question. **ADRs** for architectural decisions go under [`adr/`](../adr) with an index row in [`adr/README.md`](../adr/README.md).
4. **User-visible behavior** — If the PR changes how people build, configure, or run OpenPFC, update at least one of: [`INSTALL.md`](../../INSTALL.md), a **tutorial** or **recipe**, [`spectral_app_config_reference.md`](../reference/spectral_app_config_reference.md), or [`CHANGELOG.md`](../../CHANGELOG.md) (as appropriate). Purely internal refactors may skip prose if the contract is unchanged.
5. Topic-specific hooks — Examples: extension / `App` docs should link [`app_pipeline.md`](../user_guide/app_pipeline.md) and [`class_tour.md`](../reference/class_tour.md); validation UX should mention [`parameter_validation.md`](../user_guide/parameter_validation.md) and the root `README.md` validation section; GPU CMake flags should point at [`tutorials/gpu_app_quickstart.md`](../tutorials/gpu_app_quickstart.md) and [`build_cpu_gpu.md`](../hpc/build_cpu_gpu.md); test-related CMake knobs belong in [`testing.md`](testing.md) as well as `build_options.md`.

## Check markdown links locally

From the repository root:

```bash
python3 scripts/check_doc_links.py
```

The script scans `docs/**/*.md`, `README.md`, `INSTALL.md`, `examples/README.md`, and `apps/*/README.md`. It resolves relative links and paths starting with `/` (repository root). http(s) links are not fetched.

The same check runs in CI as part of the Documentation workflow (`.github/workflows/docs.yml`).

## Catalog and tutorial consistency (CI)

These catch drift between CMake, shipped README examples, and prose:

```bash
python3 scripts/check_examples_catalog.py
python3 scripts/check_end_to_end_allen_cahn.py
```

They run in the Documentation workflow after the link checker.

## Bash fenced blocks (CI)

` ```bash ` / ` ```sh ` snippets under `docs/` are checked with `bash -n`:

```bash
python3 scripts/check_doc_bash_syntax.py
```

Keep examples syntactically valid; use comments for non-literal lines if needed.

## Optional printable handbook

Maintainers can build a concatenated PDF/HTML — see [`handbook_build.md`](handbook_build.md) and [`scripts/build_handbook.sh`](../../scripts/build_handbook.sh).

## MkDocs + Material (browser preview)

Optional static site for prose under `docs/`:

```bash
./scripts/build_mkdocs.sh build --strict   # from repository root; output ./site/
NO_MKDOCS_2_WARNING=1 uv run --project docs mkdocs serve
```

Details and caveats (links outside `docs/`): [`mkdocs_preview.md`](mkdocs_preview.md).

## Style

- Prefer tables and short sections over long unstructured prose.
- Link to [`architecture.md`](../concepts/architecture.md) for layering instead of duplicating the kernel/runtime story.
- For C++ behavior, point to headers and `examples/` rather than copying signatures.
- Prose: avoid bold except rarely; do not use emojis in technical docs. Optional workspace rule: `.cursor/rules/documentation-prose-style.mdc`. Optional bulk cleanup: `scripts/unbold_markdown_docs.py` (inspect the diff).

## See also

- [`styleguide.md`](styleguide.md) — code and header style (also informs doc examples)
