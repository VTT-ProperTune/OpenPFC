<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Agent and contributor notes (OpenPFC)

Short orientation for people and automated agents working in this repository. For full prose documentation, start from [`docs/README.md`](docs/README.md) and the root [`README.md`](README.md).

## Install and build

**Always build and test through [`scripts/build.sh`](scripts/build.sh).** Do not
invoke `cmake`, `cmake --build`, or `ctest` by hand for routine work. The
script loads the correct Lmod stack (compiler, MPI, HeFFTe), picks the
machine toolchain, configures, builds, and runs tests.

```bash
./scripts/build.sh --help
./scripts/build.sh                              # auto-detects Tohtori vs LUMI
./scripts/build.sh --machine=tohtori --with-cuda
./scripts/build.sh --machine=lumi --with-rocm   # default LUMI path (HIP)
./scripts/build.sh --machine=lumi --partition=standard-g --wait
```

- **Tohtori:** default is a Release CPU build in `builds/release`. `--with-cuda` loads `cuda/13.1` and the matching HeFFTe prefix/module.
- **LUMI:** default is HIP/ROCm. The script loads `LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath` and the `heffte-rocm` module from `$HOME/privatemodules`. Configure runs on the login node (FetchContent needs the network). Compile and `ctest` are submitted to a GPU partition so AMD devices are available: **`standard-g`** (default) or **`dev-g`**. CUDA is not available on LUMI — use Tohtori for NVIDIA. Build trees go under `/flash/project_462001519/juaho/build/`, not inside the git clone (inode quota). Job logs go to `/scratch/project_462001519/juaho/logs/`. Slurm account is **`project_462001519`** (do not submit as `project_462001245` — those jobs never start).
- **Canonical install guide** (manual toolchain / HeFFTe details): [`INSTALL.md`](INSTALL.md). Cluster notes: [`docs/hpc/INSTALL.tohtori.md`](docs/hpc/INSTALL.tohtori.md), [`docs/hpc/INSTALL.LUMI.md`](docs/hpc/INSTALL.LUMI.md).
- **After install:** [`docs/quickstart.md`](docs/quickstart.md). **15-minute path:** [`docs/start_here_15_minutes.md`](docs/start_here_15_minutes.md).
- **CMake options:** [`docs/reference/build_options.md`](docs/reference/build_options.md). **When builds fail:** [`docs/troubleshooting.md`](docs/troubleshooting.md).

**HeFFTe:** build and install **outside** the OpenPFC clone. On Tohtori typical prefixes are `$HOME/opt/heffte/…`. On LUMI load `heffte-rocm` (do not vendor HeFFTe sources next to `CMakeLists.txt`).

## Workspace conventions (this project)

- **CMake build trees:** on workstations/Tohtori keep them under a top-level **`builds/`** directory (e.g. `builds/debug`, `builds/release`). On **LUMI** use flash via `scripts/build.sh` (default `/flash/project_462001519/juaho/build/openpfc-lumi-…`). Do not configure or compile inside the git clone on LUMI.
- **Simulation output:** write runtime artifacts (fields, VTK, logs, checkpoints, etc.) under a top-level **`results/`** directory (e.g. per run or per case in subfolders). App configs or job scripts should prefer paths under `results/` so outputs stay out of Git and out of `docs/`. On LUMI, large job I/O belongs under `/scratch/project_462001519/juaho/`.

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

**Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md) — tests, CI expectations, commit scope, commit messages, PR/merge workflow (rebase not squash), changelog. **Tests:** [`docs/development/testing.md`](docs/development/testing.md).

## Language and tooling

- **C++ standard:** C++20 (`cmake/CompilerSettings.cmake`). Prefer modern idioms when touching code; see [`.cursor/rules/prefer-cxx20-idioms.mdc`](.cursor/rules/prefer-cxx20-idioms.mdc) if present.
- **IDE / clangd:** configure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` after the correct modules/compilers are loaded (`INSTALL.md` discusses stale caches and `compile_commands.json`).

## Cursor / agent rules

Project-specific guidance for Cursor lives under [`.cursor/rules/`](.cursor/rules/) (build system, documentation expectations, SPDX year updates, cluster module notes, etc.). Read the relevant rule when changing builds, docs, or public API.

### Limina gate agents

When you are a **Limina gate role** (especially `implement` / `code_review`)
driven by `scripts/run_*.sh` and the living prompt from `limina-rpc`, that
**living prompt wins** for board and git workflow on the prepared work
branch. Implementers may commit, rebase onto the base branch, squash/amend
for a clean history, and `git push --force-with-lease` **only that work
branch** when history was rewritten. Never rewrite or force-push `master` /
`main`. Interactive Cursor chats (human-driven, no Limina Mode A) still
follow normal “commit/push only when asked” habits unless the human says
otherwise.

## Published API reference

HTML generated from headers: <https://vtt-propertune.github.io/OpenPFC/dev/> — pair with the `docs/` tree for tutorials and cluster operations.
