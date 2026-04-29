<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Contributing to OpenPFC

## Documentation

- **[`docs/README.md`](docs/README.md)** — index of all guides (architecture, HPC, tutorials).
- **[`docs/testing.md`](docs/testing.md)** — **`ctest`**, **`openpfc-tests`**, MPI test CMake options.
- **[`docs/contributing-docs.md`](docs/contributing-docs.md)** — link checks, SPDX headers, where to add cross-links in the doc index.
- Run from the repo root: **`python3 scripts/check_doc_links.py`**
- **Style** for code and headers: **[`docs/styleguide.md`](docs/styleguide.md)**
- **Extending the library** (models, **`App`**, validation): **[`docs/extending_openpfc/README.md`](docs/extending_openpfc/README.md)**, **[`docs/class_tour.md`](docs/class_tour.md)**, **[`docs/tutorials/custom_app_minimal.md`](docs/tutorials/custom_app_minimal.md)**, **[`docs/parameter_validation.md`](docs/parameter_validation.md)**

## Build and test

Follow **[`INSTALL.md`](INSTALL.md)** for MPI, HeFFTe, and CMake. Run tests with your configured build (e.g. **`ctest`** or the project’s test targets) after **`OpenPFC_BUILD_TESTS=ON`**.

## Changelog

User-visible and developer-facing changes are recorded in **[`CHANGELOG.md`](CHANGELOG.md)**. Add a note under **`[Unreleased]`** when your change affects behavior, CMake options, or config file keys.

## Questions

Use **[GitHub Issues](https://github.com/VTT-ProperTune/OpenPFC/issues)** for bugs and feature discussion.
