<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Agent and contributor notes (OpenPFC)

Short orientation for people and automated agents working in this repository. For full prose documentation, start from [`docs/README.md`](docs/README.md) and the root [`README.md`](README.md).

## Install and build

- **Canonical install guide (source build, MPI, HeFFTe, optional CUDA/HIP):** [`INSTALL.md`](INSTALL.md) — treat this as the source of truth for toolchain and CMake.
- **After install:** [`docs/quickstart.md`](docs/quickstart.md) (configure → run examples or apps → or `find_package(OpenPFC)`).
- **Fastest linear path (clone → build → one `mpirun`):** [`docs/start_here_15_minutes.md`](docs/start_here_15_minutes.md).
- **Cluster-specific:** e.g. [`docs/hpc/INSTALL.tohtori.md`](docs/hpc/INSTALL.tohtori.md), [`docs/hpc/INSTALL.LUMI.md`](docs/hpc/INSTALL.LUMI.md); HPC overview in [`docs/hpc/operator_guide.md`](docs/hpc/operator_guide.md).
- **CMake options reference:** [`docs/reference/build_options.md`](docs/reference/build_options.md).
- **When builds fail:** [`docs/troubleshooting.md`](docs/troubleshooting.md).

**HeFFTe:** build and install **outside** the OpenPFC clone (typical prefixes under `$HOME/opt/heffte/…`). Do not vendor HeFFTe sources next to `CMakeLists.txt`; details are in `INSTALL.md` (HeFFTe section).

## Workspace conventions (this project)

- **CMake build trees:** keep them under a top-level **`builds/`** directory (e.g. `builds/debug`, `builds/release`, `builds/cpu`, `builds/gpu`). Example configure: `cmake -S . -B builds/debug`. This keeps the source tree clean and matches how we want local work organized.
- **Simulation output:** write runtime artifacts (fields, VTK, logs, checkpoints, etc.) under a top-level **`results/`** directory (e.g. per run or per case in subfolders). App configs or job scripts should prefer paths under `results/` so outputs stay out of Git and out of `docs/`.

The root [`.gitignore`](.gitignore) ignores common build and output paths (`build`, `builds`, `results`, …) so these directories are not committed by mistake.

## Repository map

| Area | Role |
|------|------|
| [`include/openpfc/`](include/openpfc/) | Public C++ API (headers). |
| [`src/`](src/) | Library implementation sources. |
| [`apps/`](apps/) | Runnable programs (JSON/TOML-driven spectral apps, demos). |
| [`examples/`](examples/) | Small programs illustrating APIs and workflows. |
| [`tests/`](tests/) | Unit and integration tests (Catch2, `ctest`). |
| [`docs/`](docs/) | User and developer guides (not a substitute for Doxygen API HTML). |
| [`cmake/`](cmake/) | CMake modules, presets, toolchains. |

**Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md) — tests, CI expectations, changelog. **Tests:** [`docs/development/testing.md`](docs/development/testing.md).

## Language and tooling

- **C++ standard:** C++20 (`cmake/CompilerSettings.cmake`). Prefer modern idioms when touching code; see [`.cursor/rules/prefer-cxx20-idioms.mdc`](.cursor/rules/prefer-cxx20-idioms.mdc) if present.
- **IDE / clangd:** configure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` after the correct modules/compilers are loaded (`INSTALL.md` discusses stale caches and `compile_commands.json`).

## Cursor / agent rules

Project-specific guidance for Cursor lives under [`.cursor/rules/`](.cursor/rules/) (build system, documentation expectations, SPDX year updates, cluster module notes, etc.). Read the relevant rule when changing builds, docs, or public API.

## Published API reference

HTML generated from headers: <https://vtt-propertune.github.io/OpenPFC/dev/> — pair with the `docs/` tree for tutorials and cluster operations.
