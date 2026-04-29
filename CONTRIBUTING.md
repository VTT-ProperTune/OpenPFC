<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Contributing to OpenPFC

## Documentation

- [`docs/README.md`](docs/README.md) — index of all guides (architecture, HPC, tutorials). **API reference (HTML)** vs prose: see the opening table there.
- [`docs/learning_paths.md`](docs/learning_paths.md) — run / extend / integrate tracks.
- [`docs/tutorials/README.md`](docs/tutorials/README.md) — step-by-step tutorials (VTK, HeFFTe, spectral examples, GPU, …).
- [`docs/personas.md`](docs/development/personas.md) — short entry points by role (cluster runner, model developer, integrator).
- [`docs/tutorials/add_catch2_test.md`](docs/tutorials/add_catch2_test.md) — minimal Catch2 / `ctest` pattern.
- [`docs/showcase.md`](docs/user_guide/showcase.md) — figures mapped to apps and examples.
- [`docs/testing.md`](docs/development/testing.md) — `ctest`, `openpfc-tests`, MPI test CMake options.
- [`docs/contributing-docs.md`](docs/development/contributing-docs.md) — link checks, SPDX headers, where to add cross-links in the doc index.
- Run from the repo root: `python3 scripts/check_doc_links.py`
- Style for code and headers: [`docs/styleguide.md`](docs/development/styleguide.md)
- Extending the library (models, `App`, validation): [`docs/extending_openpfc/README.md`](docs/extending_openpfc/README.md), [`docs/class_tour.md`](docs/reference/class_tour.md), [`docs/tutorials/custom_app_minimal.md`](docs/tutorials/custom_app_minimal.md), [`docs/parameter_validation.md`](docs/user_guide/parameter_validation.md)

## Build and test

Follow [`INSTALL.md`](INSTALL.md) for MPI, HeFFTe, and CMake. Run tests with your configured build (e.g. `ctest` or the project’s test targets) after `OpenPFC_BUILD_TESTS=ON`.

## CI (GitHub Actions)

Pull requests run workflows under [`.github/workflows/`](.github/workflows): **`ci.yml`** (main build/test matrix on Ubuntu 24.04), **`docs.yml`** (markdown link check via `scripts/check_doc_links.py`, Doxygen when enabled), **`coverage.yml`**, **`asan.yml`**, **`clang-tidy.yml`**. Doc-only edits under `docs/**` still trigger the **Documentation** workflow’s link job—run `python3 scripts/check_doc_links.py` locally before pushing.

## Changelog

User-visible and developer-facing changes are recorded in [`CHANGELOG.md`](CHANGELOG.md). Add a note under `[Unreleased]` when your change affects behavior, CMake options, or config file keys.

## Questions

Use [GitHub Issues](https://github.com/VTT-ProperTune/OpenPFC/issues) for bugs and feature discussion.
