<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Scripts Directory

This directory contains utility scripts for OpenPFC development and workflow automation.

## Cluster build (tohtori)

### build_tohtori.sh

End-to-end CPU build matching [INSTALL.md](../INSTALL.md) (§1 modules, §3 HeFFTe under `$HOME/opt/heffte/2.4.1-cpu`, §5 OpenPFC) and [cmake/toolchains/tohtori-gcc11-openmpi.cmake](../cmake/toolchains/tohtori-gcc11-openmpi.cmake).

```bash
sh ./scripts/build_tohtori.sh --help
sh ./scripts/build_tohtori.sh
```

The script sources Lmod when needed and runs `module load gcc/11.2.0` itself (you do not need to load it first). Unless you pass `--build-openmpi`, it also loads `openmpi/4.1.1`. Invoking with `sh` re-executes under bash so `module` and the rest of the script work.

HeFFTe is skipped if already installed at `HEFFTE_PREFIX`; use `--clean-heffte` to reconfigure the HeFFTe build tree, or remove the install prefix and re-run to reinstall. `OpenPFC_ENABLE_HDF5=ON` is set so profiling HDF5 export (e.g. experiments/scalability_tohtori) works. Override `OPENPFC_BUILD_DIR` if you do not want `build/tohtori-release`.

### User-built OpenMPI (Slurm `srun` / PMI)

Site OpenMPI is sometimes built without Slurm PMI; `srun` then fails at `MPI_Init`. Building OpenMPI with `--with-slurm` (as in your own module recipe) fixes that. This repo supports an opt-in source build:

```bash
sh ./scripts/build_tohtori.sh --build-openmpi
# or only OpenMPI:
sh ./scripts/build_tohtori.sh --openmpi-only
```

Defaults: `OPENMPI_VER=5.0.10`, prefix `$HOME/opt/openmpi/$OPENMPI_VER`. CMake uses it when `OPENMPI_ROOT` points at that prefix (`cmake/toolchains/tohtori-gcc11-openmpi.cmake` picks `$OPENMPI_ROOT/bin/mpicc` when present).

UCX: `--build-ucx` builds UCX (default `UCX_VER=1.20.0`, `$HOME/opt/ucx/$UCX_VER`) and sets `UCX_HOME` before Open MPI `configure`, so the MPI stack links against your UCX (useful when the site UCX and site Open MPI do not match). Example:

```bash
sh ./scripts/build_tohtori.sh --build-ucx --build-openmpi --openmpi-only
```

After installing, sanity-check inter-node bandwidth with `scripts/mpi_inter_node_bw/` (see that directory’s `README.md`).

Configure needs Slurm development files on the build host; if the login node lacks them, submit the same script via Slurm (below).

### Queue build with `sbatch`

For long compiles, use `submit_build_tohtori_sbatch.sh` (wraps `sbatch_build_tohtori.slurm`):

```bash
./scripts/submit_build_tohtori_sbatch.sh
SBATCH_PARTITION=gen05_epyc SBATCH_TIME=08:00:00 SBATCH_CPUS_PER_TASK=32 \
 BUILD_TOHTORI_EXTRA_ARGS='--build-openmpi --clean-openmpi' \
 ./scripts/submit_build_tohtori_sbatch.sh
```

`OPENPFC_REPO` defaults to the repository root; override if you submit from elsewhere. Optional `SBATCH_ACCOUNT` is forwarded to `sbatch --account`.

### Build + two-node MPI test (Slurm)

`submit_build_test_tohtori_sbatch.sh` submits two jobs so you do not pay for a second node during compiles:

1. `sbatch_tohtori_build_stack.slurm` — one node, runs `build_tohtori.sh` (default `--build-ucx --build-openmpi`, full OpenPFC unless you override `BUILD_TOHTORI_EXTRA_ARGS`).
2. `sbatch_mpi_inter_node_bw.slurm` — two nodes, Slurm `--dependency=afterok:` the build job, then builds and runs `scripts/mpi_inter_node_bw`.

```bash
SBATCH_PARTITION=gen05_epyc SBATCH_BUILD_TIME=08:00:00 SBATCH_BW_TIME=00:30:00 \
 ./scripts/submit_build_test_tohtori_sbatch.sh
```

## Development Scripts

### pre-commit-hook

Purpose: Automatically check code formatting before commits to prevent CI failures.

Installation (Required for all developers):

```bash
# From the project root directory (preferred: tracked hooks under .githooks/)
git config core.hooksPath .githooks
```

Alternatively, copy the entrypoint only:

```bash
cp scripts/pre-commit-hook .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

What it does:

- Runs automatically before each `git commit`
- Checks C++ files (`.cpp`, `.hpp`, `.h`, `.cc`, `.cxx`) for formatting issues
- Uses `clang-format` to verify code style compliance
- Blocks commits if formatting issues are found
- Then runs [`pre-commit-clang-tidy-hook`](pre-commit-clang-tidy-hook) (see below)

Requirements:

- `clang-format` (version 17+ recommended, minimum version 9.0)
- On Rocky Linux 8: `dnf install clang` (provides clang-format 19+)

Why this is mandatory:
The CI/CD pipeline runs `clang-format` checks and will fail if code is not properly formatted. Installing this hook ensures you catch formatting issues locally before pushing, saving CI time and preventing failed builds.

If formatting check fails:

```bash
# Fix all staged files automatically
clang-format -i path/to/file.cpp

# Or let the hook tell you which files need fixing
# It will provide the exact command to run
```

Testing the hook:

```bash
# Stage a C++ file with formatting issues
git add tests/unit/fft/test_fft.cpp

# Try to commit (hook will check formatting)
git commit -m "test"

# If it passes, you're good!
# If it fails, run the suggested clang-format command
```

Bypassing the hook (not recommended):

```bash
# Only use this if you have a very good reason
git commit --no-verify -m "emergency fix"
```

### pre-commit-clang-tidy-hook

Purpose: When a CMake build directory with `compile_commands.json` is present (same layout as [`run-clang-tidy.sh`](run-clang-tidy.sh): default `build-tidy`, or `OPENPFC_TIDY_BUILD_DIR`), run `clang-tidy` on staged `.cpp` and `.hpp` files only. If there is no compilation database, the hook does nothing (no error).

Invocation: Not run standalone; `scripts/pre-commit-hook` `exec`s it after `clang-format` succeeds.

Requirements: `clang-tidy` on `PATH` whenever `compile_commands.json` exists in the chosen build directory.

Notes: Uses the same default `clang-tidy` flags as `run-clang-tidy.sh` without `--fail-fast` (`-p`, `-header-filter='include/openpfc/.*'`; failures follow `.clang-tidy` WarningsAsErrors only). Skips the same CPU-only GPU entrypoints as the full tidy script.

### run-clang-tidy.sh

Purpose: Run `clang-tidy` on OpenPFC `.cpp` files the same way as CI (`.clang-tidy`, `-header-filter='include/openpfc/.*'`, and the same file list). GitHub Actions calls `bash scripts/run-clang-tidy.sh --build-dir=build-tidy` after configuring; do not duplicate the `find`/`clang-tidy` loop in workflow YAML—extend this script instead.

Excluded sources (CPU-only analysis): a small set of GPU-only tungsten tests and entrypoints that use `#error` or HIP/CUDA headers when CUDA/HIP is off. The list lives in `list_cpp_for_tidy()` inside the script; keep it in sync if you add similar TUs.

Requirements: `clang-tidy`, Ninja, CMake, project dependencies (MPI, HeFFTe, etc.). On tohtori, load `gcc/11.2.0` and `openmpi/4.1.1` before configuring if you are not using the pinned toolchain paths only.

Typical workflow (tohtori):

```bash
module load gcc/11.2.0
module load openmpi/4.1.1
./scripts/run-clang-tidy.sh --configure # once: CMake + compile_commands.json
./scripts/run-clang-tidy.sh # run analysis (can take a long time)
```

`--configure` passes `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` (and the tohtori toolchain file [`cmake/toolchains/tohtori-gcc11-openmpi.cmake`](../cmake/toolchains/tohtori-gcc11-openmpi.cmake) also forces it) so `compile_commands.json` is generated for this script and for `clangd`/IDEs.

Options:

- `--configure` / `-c` — configure `build-tidy` (or `OPENPFC_TIDY_BUILD_DIR`) with the tohtori toolchain and OpenPFC test/example/app targets, then exit.
- `--build-dir=NAME` — build directory under the repo root (default: `build-tidy`).
- `--fail-fast` — pass **`--warnings-as-errors=*` (every tidy diagnostic fails) and stop after the first failing `.cpp` (useful to fix issues one file at a time: run, fix, commit, repeat).
- `--file=PATH` — run `clang-tidy` on a single `.cpp` relative to the repo root (e.g. after a fix, before a full sweep).

## Build Scripts

### build_cuda.sh

Purpose: Automated build script for OpenPFC with optional CUDA support.

Usage**:
```bash
# Build with CUDA support (default)
./scripts/build_cuda.sh

# Build without CUDA
./scripts/build_cuda.sh --no-cuda

# Build Release with custom job count
./scripts/build_cuda.sh --build-type Release --jobs 16

# Build without cleaning first
./scripts/build_cuda.sh --no-clean
```

What it does:
- Automatically loads required modules (`cuda`, `openmpi/4.1.1`)
- Sets correct compiler paths (GCC 11.2.0)
- Selects appropriate HeFFTe version (2.4.1 for CPU, 2.4.1-cuda for GPU)
- Configures CMake with correct options
- Builds with specified number of parallel jobs
- Provides clear status messages and error handling

Key Features:
- Handles module loading automatically
- Correctly sets compiler paths (fixes CMake auto-detection issues)
- Selects HeFFTe version based on CUDA enablement
- Cleans build directory by default (use `--no-clean` to skip)
- Works on both AMD (no CUDA) and NVIDIA (with CUDA) systems

Options:
- `--build-type TYPE`: Debug or Release (default: Debug)
- `--cuda` / `--no-cuda`: Enable/disable CUDA (default: enabled)
- `--jobs N, -j N`: Number of parallel build jobs (default: 8)
- `--no-clean`: Don't clean build directory before building
- `--help, -h`: Show help message

Examples:
```bash
# Standard CUDA build
./scripts/build_cuda.sh

# CPU-only build
./scripts/build_cuda.sh --no-cuda

# Release build with 16 jobs
./scripts/build_cuda.sh --build-type Release -j 16
```

## Other Scripts

### check_kernel_no_frontend_includes.sh

CI and local guard: fails if any kernel source under `include/openpfc/kernel` or `src/openpfc/kernel` contains `#include` of `openpfc/frontend/...` (see [`docs/architecture.md`](../docs/architecture.md) — *Include audit*). Requires `rg` (ripgrep); exits 0 if `rg` is missing (for minimal sandboxes).

```bash
bash scripts/check_kernel_no_frontend_includes.sh
```

### xdmfgen.py

Generate XDMF files for visualization with ParaView.

### pvrender.py

Render images from ParaView for batch visualization.

---

## Contributing

When adding new development scripts:

1. Add them to this directory
2. Make them executable: `chmod +x script_name`
3. Add SPDX license headers
4. Document them in this README
5. If they're part of the required workflow, mark them as such
