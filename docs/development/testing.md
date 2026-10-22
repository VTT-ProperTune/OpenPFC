<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Building and running tests

OpenPFC ships a Catch2-based suite under `tests/`, built as the `openpfc-tests` executable. This page complements the one-line `OpenPFC_BUILD_TESTS` entry in [`build_options.md`](../reference/build_options.md).

## Enable / disable

| CMake option | Default | Effect |
|--------------|---------|--------|
| `OpenPFC_BUILD_TESTS` | ON | `enable_testing()`, `find_package(Catch2)`, `add_subdirectory(tests)`. The `openpfc-tests` binary stays in the build tree; it is not installed. |
| `OpenPFC_BUILD_BENCHMARKS` | OFF | Adds slow benchmarks from `tests/benchmarks/` into the same `openpfc-tests` binary when ON. |

Defined in `cmake/BuildOptions.cmake`.

## Quick commands

From your build directory after a successful configure:

```bash
cmake --build . -j"$(nproc)"
ctest --output-on-failure
```

To run the Catch2 executable directly (faster iteration):

```bash
./openpfc-tests          # or build/openpfc-tests depending on generator
./openpfc-tests '~[MPI]' # same filter as the default CTest target below
```

## What CTest registers

| CTest name | What it runs |
|------------|----------------|
| `openpfc-all-tests` | `openpfc-tests "~[MPI]"` — all non-MPI Catch2 tests in one process (fast). |
| `mpi_2procs_all` | Only if `OpenPFC_RUN_MPI_SUITES` is ON: MPI tests on 2 ranks (Catch tags `[MPI]` excluding some sub-suites). |
| `mpi_3procs_ring` | If `OpenPFC_RUN_MPI_SUITES` and the configure host has enough logical CPUs (or you force registration — see below). |
| `mpi_4procs_grid_multiple` | Same, for 4 ranks. |

MPI CTest targets are optional because many laptops and CI runners cannot launch multi-rank jobs reliably. Options in `tests/CMakeLists.txt`:

- `OpenPFC_RUN_MPI_SUITES` — master switch for `mpi_*` tests.  
- `OpenPFC_MPI_TEST_REGISTER_HIGH_RANK_ALWAYS` — register 3- and 4-rank tests even when the configure host reports few CPUs (e.g. configure on a login node, run tests on a compute node).  
- `OpenPFC_MPI_TEST_MAX_WORLD_SIZE` — cap registered MPI world sizes (useful on CI).

VTK writer tests and other targets may `add_test` separately under `tests/unit/frontend/io/` — use `ctest -N` to list everything after configure.

## Application tests

When `OpenPFC_BUILD_APPS=ON`, some apps register extra CTest entries (e.g. tungsten / aluminum checks). Those live next to each app’s `CMakeLists.txt`. `tungsten-golden-4rank` and `aluminum-golden-4rank` follow `OpenPFC_MPI_TEST_MAX_WORLD_SIZE` (CI uses 2, so they are omitted on GitHub-hosted runners).

Minimal new-test walkthrough: [`tutorials/add_catch2_test.md`](../tutorials/add_catch2_test.md).

## See also

- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — changelog and review expectations  
- [`build_options.md`](../reference/build_options.md) — all CMake switches  
- [`INSTALL.md`](../../INSTALL.md) — Catch2 / MPI / toolchain prerequisites  
- [`troubleshooting.md`](../troubleshooting.md) — configure and link failures  

## What runs in CI vs on clusters (audit 11 / PN)

| Check | GitHub Actions CI | Cluster (tohtori / LUMI) |
|-------|-------------------|--------------------------|
| CPU unit + integration (`ctest`) | pull requests: gcc-13 Release; master: also gcc-13 Debug; monthly: gcc-11–14, clang-18, MPICH, no HeFFTe | yes |
| MPI suites | 2 ranks; master also runs the multigrid plan on 3/4/6 ranks | 2/3/4 ranks (`build.sh --mpi-tests`) |
| `find_package(OpenPFC)` packaging smoke | yes (`packaging-smoke` job) | `tests/packaging/consumer` |
| CUDA/HIP **compile** (no GPU) | required `cuda-compile` / `hip-compile` jobs on every master push | `build.sh --with-cuda --no-test`, `build.sh --with-rocm --no-test` |
| CUDA/HIP **run** (parity, VTK, IC residency) | not possible (no GPU) | tohtori (CUDA), LUMI (HIP) |

### Fail-closed Dirichlet GMG numerical device qualification

OpenPFC's CUDA/HIP GitHub jobs compile and link
`test_nodal_multigrid_cuda` and `test_nodal_multigrid_hip`; they
**do not run** the device numerical tests. A Catch2 SKIP is not
evidence of GPU correctness.

On an *already approved/allocated* GPU system, use the normal
`scripts/build.sh` workflow with:

```text
--cmake-arg=-DOpenPFC_REQUIRE_GPU_DIRICHLET_TESTS=ON --mpi-tests --test
```

Select `--machine=tohtori --with-cuda` for NVIDIA or
`--machine=lumi --with-rocm` for AMD, and follow the usual cluster
partition/allocation rules. The opt-in mode registers strict
`DirichletGPU_{cuda,hip}_{1rank,2ranks,4ranks}_Required` CTests
(subject to the configured world-size cap). Each runs the **entire**
`[nodal_dirichlet][device]` numerical acceptance: operator,
halo, transfer, two-/three-/four-level Q1 V-cycle, preconditioned
true-residual CG, and independent manufactured-source physical
`L²` refinement order. `OPENPFC_REQUIRE_GPU_TESTS=1` rejects
missing/unbindable devices on *any rank*, rather than returning a
green SKIP.

Store actual machine/driver/GPU binding, MPI/ROCm/CUDA versions,
requested and achieved rank worlds, exact source SHA, true residuals,
iterations, errors and any negative outcomes with the **owning
DiscretizationLab HPC campaign evidence**. No performance claim or
repeated HPC batch is implied by these software test definitions.

| Golden trajectories / performance baselines | CPU checksum pins (`tungsten-cpu-golden`, `allen-cahn-cpu-golden`, `wave2d-cpu-golden`) | GPU parity and perf JSON (see `tests/baselines/`) |

The authoritative GPU compile check today is `scripts/build.sh --with-cuda --no-test`
(and `--with-rocm`) on a machine with the toolchain; the GPU-run checks
(`test_tungsten_cpu_vs_cuda`, `test_tungsten_app_gpu_ic`, etc.) must be run on a
GPU node. See `tests/baselines/BASELINES.md` for the scientific/performance
baselines and which comparisons are bitwise vs tolerance-based.

## Existing toolchains and portable consumers

`./scripts/build.sh --machine=portable --cpu --no-test --jobs=2` builds a
package using the caller's compilers, MPI and CMake package search paths.
Set `CC`, `CXX` and `CMAKE_PREFIX_PATH`, or pass explicit `--cmake-arg`
options for dependencies and install prefixes. This mode performs no module
loading and injects no Tohtori/LUMI toolchain. Configure and build remain
canonical; run `cmake --install <build> --prefix <prefix>` to install.
Without `--no-test`, the ordinary Python and CTest suites still run.
CUDA/HIP SDKs and device allocation remain the caller's responsibility.
