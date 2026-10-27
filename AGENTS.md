<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Agent and contributor notes (OpenPFC)

> Process rules come from the latest approved
> [OodiHandbook](https://github.com/ahojukka5/OodiHandbook) tag (AUTH-007).
> Start with its `guides/local-development.qmd` and `guides/contributing.qmd`.
> This file adds only this repository's facts (ENG-001, ENG-028).

Technical orientation for agents. Human contributors start from
[`CONTRIBUTING.md`](CONTRIBUTING.md); prose documentation starts from
[`docs/README.md`](docs/README.md) and the root [`README.md`](README.md).

## Repositories

`ahojukka5/OpenPFC` is the source of truth for development, issues, pull
requests and CI. `VTT-ProperTune/OpenPFC` mirrors the reviewed code and
publishes the website: replicate canonical commits there without a separate
history, and keep its `master` at the canonical `master` commit.

## Repository map

| Area | Role |
|------|------|
| [`include/openpfc/`](include/openpfc/) | Public C++ API (headers). |
| [`src/`](src/) | Library implementation sources. |
| [`apps/`](apps/) | Runnable programs (JSON/TOML-driven spectral apps, demos). |
| [`examples/`](examples/) | Small programs illustrating APIs and workflows. |
| [`tests/`](tests/) | Unit and integration tests (Catch2, `ctest`). |
| [`docs/`](docs/) | User and developer guides (not a substitute for Doxygen API HTML). Historical 0.2 plans are under [`docs/archive/`](docs/archive/README.md); do not treat them as current architecture. |
| [`cmake/`](cmake/) | CMake modules, presets, toolchains. |

## Install and build

Use [`scripts/build.sh`](scripts/build.sh) with the local toolchain;
`--machine=portable` preserves an ordinary workstation environment. Direct
CMake/CTest commands are appropriate when the script does not support the
available environment. Report missing dependencies and unrun checks (ENG-011).

```bash
./scripts/build.sh --help
./scripts/build.sh --machine=portable --cpu     # local workstation
./scripts/build.sh                              # auto-detects Tohtori vs LUMI
./scripts/build.sh --machine=tohtori --with-cuda
./scripts/build.sh --machine=lumi --with-rocm   # default LUMI path (HIP)
```

- **Cluster modes** load the compiler, MPI and HeFFTe modules. Tohtori
  defaults to a Release CPU build; `--with-cuda` selects the CUDA toolchain
  and matching HeFFTe. LUMI defaults to HIP/ROCm (no CUDA there): configure
  runs where FetchContent has network access, then compile and `ctest` run
  on a GPU allocation (`--partition`) so AMD devices are available.
- **Build trees** stay out of the source tree: on workstations under a
  top-level `builds/` (`builds/debug`, `builds/release`), one per worktree.
- **HeFFTe** is built and installed outside the clone or loaded as a module;
  never vendor its sources next to `CMakeLists.txt`.
- **Guides:** [`INSTALL.md`](INSTALL.md) (manual toolchain, HeFFTe, stale
  `compile_commands.json`), [`docs/hpc/INSTALL.tohtori.md`](docs/hpc/INSTALL.tohtori.md),
  [`docs/hpc/INSTALL.LUMI.md`](docs/hpc/INSTALL.LUMI.md),
  [`docs/quickstart.md`](docs/quickstart.md),
  [`docs/start_here_15_minutes.md`](docs/start_here_15_minutes.md),
  [`docs/reference/build_options.md`](docs/reference/build_options.md),
  [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Tests

The Catch2 suite builds as `openpfc-tests`; details and every CTest name are
in [`docs/development/testing.md`](docs/development/testing.md). Run the
smallest test that exercises the change, and the full `ctest` suite when the
change is wide (ENG-015, Handbook `guides/testing.qmd`). In OpenPFC, shared
state is wide: a field type, a halo exchanger, decomposition, FFT or a CMake
option can break tests that never appear in the diff.

```bash
./openpfc-tests '[halo]'            # one Catch2 tag or test name
./openpfc-tests '~[MPI]'            # all non-MPI tests (CTest: openpfc-all-tests)
ctest -N                            # list registered tests
ctest -R mpi_2procs_all --output-on-failure
ctest -L verification               # guard checks such as deprecated symbols
ctest --output-on-failure           # full suite
```

- MPI suites register only with `OpenPFC_RUN_MPI_SUITES=ON`; GPU tests
  labelled `required-device` need real hardware. CPU tests do not validate
  GPU or MPI execution; report such checks as unrun when no device is
  available.
- A PR that adds an entry point (binary, script, CMake target) must make the
  suite invoke it; helper unit tests are not enough.
- Doc changes: run `python3 scripts/check_doc_links.py` from the root.

## CI tiers

Workflows live in [`.github/workflows/`](.github/workflows) (see its README);
do not change them unasked (ENG-009). A merge waits only for checks an active
ruleset requires (ENG-007, ENG-022).

- **Pull request:** `ci.yml` selects components from the base/head diff.
  Doc-only changes run REUSE and documentation checks without a C++ build;
  shared or unclassified changes run library tests, SDK contracts and
  consumer smoke checks, with the fallback reason logged.
- **Push to master:** adds the full routine CTest suite, GCC 13 Debug with
  3/4/6-rank multigrid transfers, benchmark compile, clang-18 compile,
  packaging smoke and CUDA/HIP compile-only (no GPU). Device and GPU
  spectral/HeFFTe validation remain allocated cluster checks.
- **Monthly:** `compatibility.yml` (GCC 11–14, Clang 18, Ubuntu 22.04/24.04,
  MPICH, no HeFFTe), then coverage, clang-tidy and AddressSanitizer; all
  also `workflow_dispatch`.

## Language and style

- **C++20** (`cmake/CompilerSettings.cmake`). Generic C++/CUDA/HIP advice:
  Handbook `guides/coding-cpp.qmd`; naming: SWE-015 and `guides/naming.qmd`.
- OpenPFC code and header style: [`docs/development/styleguide.md`](docs/development/styleguide.md),
  root `.clang-format` (clang-format 20, as in CI) and `.clang-tidy`.
- For clangd, configure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` after the
  correct modules and compilers are loaded.
- Extending the library: [`docs/extending_openpfc/README.md`](docs/extending_openpfc/README.md),
  [`docs/reference/class_tour.md`](docs/reference/class_tour.md).

## Outputs and records

- Runtime artifacts (fields, VTK, logs, checkpoints) go under a top-level
  `results/`; app configs and job scripts prefer paths there. `.gitignore`
  covers `build`, `builds`, `results` and similar paths.
- This package is the library: no campaign records, measurement dumps or
  pointers to research results (PUB-008). Launchers already under
  `apps/*/slurm` and `docs/lumi_slurm` are legacy; they do not grow.
- Changelog: user-visible and developer-facing changes to behaviour, CMake
  options or config keys go in [`CHANGELOG.md`](CHANGELOG.md) under
  `[Unreleased]`, in the same commit as the change.

## Published API reference

HTML generated from headers: <https://vtt-propertune.github.io/OpenPFC/dev/>.
Only sources listed in `docs/public_pages.txt` are published; a new `docs/`
file stays out of the site until it is reviewed and listed.
