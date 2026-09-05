<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Contributing to documentation

Repository-wide contribution rules, commit messages, tests, and changelog
expectations are in [`CONTRIBUTING.md`](../../CONTRIBUTING.md). This page covers
the documentation-specific workflow.

## Documentation surfaces

OpenPFC has two source surfaces and one rendered product:

| Source | Purpose | Rendered by |
|--------|---------|-------------|
| Markdown under `docs/` | Tutorials, concepts, operations, and stable reference guides | Sphinx with MyST |
| Public headers | Exact C++ declarations and source-level comments | Doxygen XML through Breathe |

The root [`README.md`](../../README.md) is a stable project landing page, not a
second manual. Sphinx combines prose and generated API material into one site,
one navigation tree, and one search index.

## Choose the right location

| Content | Location |
|---------|----------|
| Smallest first successful run | `docs/start_here_15_minutes.md` |
| Broad build/run/link overview | `docs/quickstart.md` |
| Sequenced route for a user role | `docs/learning_paths.md` |
| Task-shaped copy-paste answer | `docs/recipes/` |
| Multi-step hands-on walkthrough | `docs/tutorials/` |
| Concept or architecture explanation | `docs/concepts/` |
| Cluster and performance operations | `docs/hpc/` or `docs/lumi_slurm/` |
| Scientific interpretation and limitations | `docs/science/` |
| Lookup-oriented stable contract | `docs/reference/` |
| Curated generated C++ declarations | Public headers, exposed through `docs/api/` |
| Maintainer workflow or implementation note | `docs/development/` |
| Architecture decision | `docs/adr/` |
| Finished 0.2 planning documents | `docs/archive/` (not a source of current work) |

Large subtrees have their own `README.md` index. Add new pages to the nearest
local index and to `docs/index.md` when they belong in primary navigation. Only
add a root README link when the page is an important entry point for a broad
audience.

## Before opening a pull request

1. Preserve the SPDX header and update its year when project policy requires
   it.
2. Use relative Markdown links so pages work both on GitHub and in Sphinx.
3. Update the nearest local index and, when appropriate, `docs/index.md`.
4. Keep user-visible commands aligned with scripts, CMake targets, and
   packaging tests rather than copying an unverified approximation.
5. Update behavior documentation when a change affects installation, CMake
   options, configuration keys, output formats, or runtime operation.
6. Prefer links to headers and runnable examples over copied signatures or
   large code listings that will drift.
7. Put exact declarations and parameter comments next to public C++ code; add
   narrative explanation to Markdown only when users need context or workflow.
8. Do not add new Doxygen warnings. When touching an already warned comment,
   reduce or remove the corresponding baseline category when practical.

Topic-specific cross-links are still useful:

- extension and `App` pages should link
  [`app_pipeline.md`](../user_guide/app_pipeline.md) and
  [`class_tour.md`](../reference/class_tour.md);
- GPU build flags should link
  [`gpu_app_quickstart.md`](../tutorials/gpu_app_quickstart.md) and
  [`build_cpu_gpu.md`](../hpc/build_cpu_gpu.md);
- test-related CMake options belong in both
  [`testing.md`](testing.md) and
  [`build_options.md`](../reference/build_options.md);
- configuration validation belongs in
  [`parameter_validation.md`](../user_guide/parameter_validation.md), not as a
  second tutorial in the root README.

## Run source checks

From the repository root:

```bash
python3 scripts/check_doc_links.py
python3 scripts/check_doc_toolchain_refs.py
python3 scripts/check_examples_catalog.py
python3 scripts/check_end_to_end_allen_cahn.py
python3 scripts/check_doc_bash_syntax.py
```

These checks cover repository-relative Markdown links, retired toolchain
references, the examples catalog, the Allen-Cahn end-to-end documentation
contract, and shell syntax in fenced `bash` or `sh` blocks. External HTTP links
are not fetched by the repository link checker.

## Build the complete site

Install Doxygen and Ninja, synchronize the locked Python environment, and run
the one canonical build command:

```bash
uv sync --project docs --locked
bash scripts/build_docs.sh build
```

The command generates Doxygen XML and then renders the complete Sphinx site.
The rendered site is written to `site/` and includes both prose and the curated
C++ API reference. Maintained Sphinx sources must be warning-free.

For an interactive preview:

```bash
uv sync --project docs --locked
bash scripts/build_docs.sh serve
```

See [`sphinx_preview.md`](sphinx_preview.md) for environment, link-rewriting,
and publication details.

## API documentation policy

Doxygen is a parser, not a separately published site. Public API changes should
update comments near declarations. Breathe imports the resulting XML into the
curated pages under [`../api/`](../api/index.md).

The repository currently contains a classified legacy Doxygen-comment debt.
`scripts/check_doxygen_log.py` records the exact count in each known category
and fails when a category grows or an unknown warning appears. The baseline is
therefore a ceiling, not an accepted quality target. Reducing a category should
be followed by lowering its baseline in the checker.

Avoid long narrative Doxygen blocks when a prose concept or tutorial page is
the clearer home; link the two surfaces instead.

## Optional printable handbook

Maintainers can build the concatenated handbook described in
[`handbook_build.md`](handbook_build.md):

```bash
bash scripts/build_handbook.sh
```

## Style

- Prefer short sections, focused tables, and explicit outcomes.
- Explain concepts in prose; use reference pages for exhaustive lookup.
- Keep commands executable and code snippets minimal.
- Avoid implementation-roadmap details in user-facing type tours.
- Link to [`architecture.md`](../concepts/architecture.md) instead of repeating
  the kernel/runtime/frontend layering on multiple pages.
- Avoid decorative emphasis and emojis in technical documentation.

## Review checklist

- [ ] The page has one clear audience and purpose.
- [ ] Commands and CMake targets match the current repository.
- [ ] New pages are linked from the correct local index.
- [ ] Primary pages are represented in `docs/index.md`.
- [ ] Relative links and fenced shell blocks pass their checkers.
- [ ] `bash scripts/build_docs.sh build` succeeds.
- [ ] Maintained Sphinx documentation is warning-free.
- [ ] Doxygen diagnostics stay within the recorded baseline and add no unknown warning.
- [ ] The root README remains a landing page rather than a duplicate manual.

## See also

- [`styleguide.md`](styleguide.md) — code and public-header style
- [`documentation_versioning.md`](documentation_versioning.md) — development
  docs versus release tags
- [`testing.md`](testing.md) — code and MPI test workflows
