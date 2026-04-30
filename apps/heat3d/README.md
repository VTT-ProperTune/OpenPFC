<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Heat3D — four single-purpose drivers

MPI CPU drivers for the **3D heat equation** \(\partial u/\partial t = D \nabla^2 u\) on a uniform periodic brick \([0,N)^3\) with spacing 1. Each method ships as its **own executable** so each binary is a focused educational example:

| Binary | Discretisation | What it shows |
|---|---|---|
| **`heat3d_fd`** | Compact finite differences (orders 2–20) | The "production" path: three lines compose `pfc::sim::stacks::FdCpuStack` + `pfc::field::FdGradient<HeatGrads>` + `pfc::sim::steppers::EulerStepper`. Physics is hidden inside the kernel. |
| **`heat3d_fd_manual`** | Laboratory-style FD (2nd order, 7-point) | The **same physics, hand-written**. The driver spells out the stencil, the non-blocking comm/compute overlap, and the explicit Euler update. Plumbing (MPI / decomposition / linear-index arithmetic) is hidden behind `pfc::field::PaddedBrick<T>`, `pfc::PaddedHaloExchanger<T>`, and the `for_each_owned/inner/border` lambda iterators. |
| **`heat3d_spectral`** | Implicit Euler in Fourier space | HeFFTe-backed forward + backward FFT per step (`heat3d::SpectralHeatPropagator`). |
| **`heat3d_spectral_pointwise`** | Explicit Euler with point-wise spectral RHS | `pfc::field::SpectralGradient<HeatGrads>` materialises `(u_xx, u_yy, u_zz)` and the same `HeatModel::rhs` is applied cell-by-cell. |

The compact driver and the manual driver compute the **same thing** — the manual one just *lets you read the loop*. They produce identical interior L2 (verified by a unit test). All four binaries share `HeatModel` for physics and `heat3d::report` for the `method / timing / l2_error` summary line.

Initial condition matches the diffusion examples: \(u(\mathbf{x},0)=\exp(-|\mathbf{x}|^2/(4D))\) with origin at \((0,0,0)\).

## Build

Enabled with `OpenPFC_BUILD_APPS=ON` (default). Requires HeFFTe (the spectral binaries use the same FFT stack as the library). When CMake finds OpenMP for C++, all four binaries link it so the FD paths can use hybrid **MPI + OpenMP**: the compact driver's interior loop runs under a single `omp parallel for collapse(2)` inside `pfc::field::for_each_interior`; the manual driver opts into the same parallelisation explicitly via `pfc::field::for_each_inner_omp` and `pfc::field::for_each_owned_omp`. Set `OMP_NUM_THREADS` to control the thread count.

### Source layout

Shared physics, IC, propagator, parser, and reporting headers (live in `include/heat3d/`):

- **[`include/heat3d/heat_grads.hpp`](include/heat3d/heat_grads.hpp)** — the **per-point grads aggregate** the model consumes. A three-`double` struct (`xx, yy, zz`) drawn from the OpenPFC catalog so the kernel's templated evaluators (`pfc::field::FdGradient<G>`, `pfc::field::SpectralGradient<G>`) compute exactly those second derivatives and nothing else. See [`docs/extending_openpfc/per_point_grads.md`](../../docs/extending_openpfc/per_point_grads.md) for the contract.
- **[`include/heat3d/heat_model.hpp`](include/heat3d/heat_model.hpp)** — the **physics**. A small self-contained `heat3d::HeatModel` struct with the diffusion coefficient, an initial-condition lambda, an optional boundary-value provider, and the per-point right-hand side `D * (g.xx + g.yy + g.zz)`. **OpenPFC-free** (only `<cmath>` and `<functional>` are included); this is the only file a physicist edits to define a new heat problem.
- **[`include/heat3d/spectral_heat_propagator.hpp`](include/heat3d/spectral_heat_propagator.hpp)** — heat-specific **implicit-Euler-in-Fourier-space** propagator. Builds the `1 / (1 - dt·D·k²)` symbol table once from the FFT layout and exposes `step(LocalField&)`. Backend-agnostic (takes `pfc::fft::IFFT&`).
- **[`include/heat3d/cli.hpp`](include/heat3d/cli.hpp)** — `RunConfig` plus the slim per-binary parsers `parse_fd` / `parse_spectral` and their `_or_print_usage` wrappers. Each binary already knows its own discretisation, so the parsers do **not** consume an `argv[1]` discriminator. Header-only, MPI-free, OpenPFC-free; trivially unit-testable.
- **[`include/heat3d/reporting.hpp`](include/heat3d/reporting.hpp)** — `analytic_gaussian` (closed-form reference solution on \(\mathbb{R}^3\)), `fd_extra_metadata` (FD/OpenMP info string), and the rank-0 `report` template that prints the canonical `method` / `timing` / `l2_error` triplet, shared by all four binaries.

Per-binary drivers (live in `src/cpu/`):

- **[`src/cpu/heat3d_fd.cpp`](src/cpu/heat3d_fd.cpp)** — compact FD driver. The time loop is three lines: `stack.exchange_halos(); t = stepper.step(t, stack.u().vec());`. Builds `pfc::sim::stacks::FdCpuStack` + `pfc::field::FdGradient<HeatGrads>` + `pfc::sim::steppers::EulerStepper`.
- **[`src/cpu/heat3d_fd_manual.cpp`](src/cpu/heat3d_fd_manual.cpp)** — **laboratory-style** 2nd-order central FD driver. The hot loop reads:

  ```cpp
  halo.start_halo_exchange(u.data(), u.size());
  field::for_each_inner_omp(u, hw, stencil_step);   // overlap
  halo.finish_halo_exchange();
  field::for_each_border (u, hw, stencil_step);     // touches halos
  field::for_each_owned_omp(u, [&](int i, int j, int k) {
    u(i, j, k) += cfg.dt * du(i, j, k);             // explicit Euler
  });
  ```

  with a single `stencil_step` lambda calling `model.rhs(0.0, HeatGrads{xx, yy, zz})`. Each stage is wrapped in `pfc::runtime::tic(timer, "...")` / `toc(timer, "...")` and `print_timing_summary(timer, 0)` prints a sorted breakdown on rank 0 at the end.
- **[`src/cpu/heat3d_spectral.cpp`](src/cpu/heat3d_spectral.cpp)** — implicit-Euler spectral driver. Calls `heat3d::SpectralHeatPropagator::step(stack.u())` once per step (forward FFT → diagonal multiply in k-space → inverse FFT).
- **[`src/cpu/heat3d_spectral_pointwise.cpp`](src/cpu/heat3d_spectral_pointwise.cpp)** — point-wise spectral RHS: materialises `(u_xx, u_yy, u_zz)` per cell with `pfc::field::SpectralGradient<HeatGrads>` and runs the same `EulerStepper` the FD driver uses.

Tests:

- **[`tests/test_heat3d.cpp`](tests/test_heat3d.cpp)** — Catch2 unit tests covering the model in isolation (defaults, IC tracking `D`, IC override, the `rhs` formula), the slim per-binary CLI parsers (happy paths + every rejection case), the analytic reference solution, two single-rank integration tests against `LocalField` / `EulerStepper`, **and** two single-rank cases for the manual driver — a smoke + L2-vs-analytic test, and a side-by-side cross-check that the manual `PaddedBrick + PaddedHaloExchanger` loop produces the same interior L2 as the compact `FdCpuStack` path. Built into the `test_heat3d` executable and registered with CTest as `heat3d-all-tests` whenever `OpenPFC_BUILD_TESTS=ON` and Catch2 is available (set `HEAT3D_ENABLE_TESTS=OFF` to skip).

### Where OpenMP runs (FD)

In **`heat3d_fd.cpp`** the per-step interior loop is delegated to `pfc::field::for_each_interior` (`include/openpfc/kernel/simulation/for_each_interior.hpp`), which runs a single `#pragma omp parallel for collapse(2) schedule(static)` over local interior \((i_z, i_y)\) and walks the inner \(x\)-line serially. Each cell calls the user lambda, which builds the per-point `HeatGrads` via `pfc::field::FdGradient<HeatGrads>` (per-axis stencil application from `kernel/field/fd_apply.hpp`) and stores the explicit-Euler update in place.

In **`heat3d_fd_manual.cpp`** the same parallelisation is applied explicitly: `pfc::field::for_each_inner_omp` parallelises the inner-region stencil over `(k, j)`, `pfc::field::for_each_owned_omp` parallelises the Euler update; the (cheap) border pass stays serial. **Not parallel** in either driver: the MPI halo exchange (`SeparatedFaceHaloExchanger` / `PaddedHaloExchanger`).

So `htop`/`top` can look **single-threaded** when wall time is dominated by **MPI** (many ranks, thin subdomains), or when the launcher has pinned the process to a **narrow CPU mask** (OpenMP then schedules all threads inside that mask).

**Single MPI rank on Linux:** `heat3d` calls `pfc::runtime::reset_cpu_affinity_if_single_mpi_rank` (which uses `sched_setaffinity`) after `MPI_Init` to allow **all online logical CPUs** for that process, so OpenMP can use the machine without extra `mpirun` flags. Set **`OPENPFC_NO_RESET_AFFINITY`** (any value) to keep the launcher’s mask unchanged (e.g. if your site policy forbids overriding binding).

**Several MPI ranks on one node:** the reset is **not** applied (`MPI_COMM_WORLD` size \(>1\)); use launcher options so each rank gets a disjoint CPU set and set `OMP_NUM_THREADS` per rank accordingly, for example:

```bash
export OMP_NUM_THREADS=4
mpirun --bind-to none -n 4 ./apps/heat3d/heat3d fd 256 25 1e-6 1.0 12
# or: export OMPI_MCA_hwloc_base_binding_policy=none
```

Rank 0 prints `omp_max_threads` and **`omp_get_num_procs()`** in the summary line. After a single-rank run, **`omp_get_num_procs()`** should match the number of CPUs OpenMP may use; if it stays at 1, check binding and whether `OPENPFC_NO_RESET_AFFINITY` is set.

## Usage

```text
heat3d_fd                  <N> <n_steps> <dt> <D> <fd_order>
heat3d_fd_manual           <N> <n_steps> <dt> <D>
heat3d_spectral            <N> <n_steps> <dt> <D>
heat3d_spectral_pointwise  <N> <n_steps> <dt> <D>
```

- `fd_order` (compact FD only): even integers **2 through 20**. Halo width is `fd_order/2` per face. Each local subdomain must be **wider** than `fd_order` in every dimension (the driver aborts with `MPI_Abort` if the interior slab is empty).
- The **manual** driver hard-codes 2nd-order central (halo width 1) — extend `stencil_step` if you want a different order; the goal is to keep the showcase code uncluttered.
- **Stability (FD)**: explicit Euler requires a sufficiently small `dt` (CFL); higher spatial order does not remove this limit.
- **Spectral**: each step applies \(\hat u \leftarrow \hat u / (1 + \Delta t\,D|\mathbf{k}|^2)\) in complex Fourier space (implicit Euler for the linear diffusion operator).

## Example

From your build directory:

```bash
mpirun -n 4 ./apps/heat3d/heat3d_fd 64 200 0.001 1.0 4
mpirun -n 4 ./apps/heat3d/heat3d_fd_manual 64 200 0.001 1.0
mpirun -n 4 ./apps/heat3d/heat3d_spectral 64 200 0.001 1.0
mpirun -n 4 ./apps/heat3d/heat3d_spectral_pointwise 64 200 0.001 1.0
```

The compact and manual FD binaries report identical `l2_error_vs_R3_analytic_rms` (the manual binary additionally prints a per-section breakdown — `inner / halo_wait / border / euler` — from `pfc::runtime::print_timing_summary`).

Rank 0 prints `timing_s`, `avg_step_time_s` (MPI max across ranks), `omp_max_threads` when OpenMP is enabled, and an RMS L2 error against the **infinite-domain** Gaussian reference

\[
u(\mathbf{x},t)=(1+t)^{-3/2}\exp\!\left(-\frac{|\mathbf{x}|^2}{4D(1+t)}\right)
\]

The domain is **periodic**; for a localized Gaussian the mismatch at boundaries contributes to the reported error even when the numerics are consistent.

## Benchmarks (FD)

All numbers below are **`avg_step_time_s`** from rank 0 (same as **MPI_MAX** across ranks for these runs). Build: **Release**, **GCC 11.2.0**, **Open MPI 4.1.1**, HeFFTe CPU prefix on `CMAKE_PREFIX_PATH`. Host: login node used for development (timing varies with load; repeat for publication-quality data).

### Higher-order FD and MPI (`OMP_NUM_THREADS=1`)

Fixed case: **`N=256`**, **`n_steps=25`**, **`dt=1e-6`**, **`D=1.0`**. Higher orders use wider stencils (more flops per step) but still scale down with rank count on this grid.

| fd_order | ranks=1 | ranks=2 | ranks=4 | ranks=8 |
|----------|---------|---------|---------|---------|
| 8 | 0.651413 | 0.313773 | 0.156707 | 0.0852177 |
| 10 | 0.920717 | 0.460287 | 0.198479 | 0.100299 |
| 12 | 0.972677 | 0.542606 | 0.264795 | 0.155862 |
| 14 | 1.03012 | 0.58818 | 0.310875 | 0.151631 |
| 16 | 1.21346 | 0.617872 | 0.331 | 0.164514 |
| 18 | 1.42205 | 0.682012 | 0.36659 | 0.18127 |
| 20 | 1.51928 | 0.839491 | 0.474414 | 0.190416 |

Reproduce (after `module load gcc/11.2.0 openmpi/4.1.1` and configuring with that toolchain):

```bash
export OMP_NUM_THREADS=1
mpirun -n 8 --mca btl tcp,self --mca oob tcp ./apps/heat3d/heat3d_fd 256 25 1e-6 1.0 20
```

Sweep example:

```bash
export OMP_NUM_THREADS=1
for ranks in 1 2 4 8; do
  for ord in 8 10 12 14 16 18 20; do
    mpirun -n "$ranks" --mca btl tcp,self --mca oob tcp ./apps/heat3d/heat3d_fd 256 25 1e-6 1.0 "$ord"
  done
done
```

### OpenMP scaling (`fd_order=12`, one MPI rank)

Same grid **`256³`**, **`n_steps=25`**, **`dt=1e-6`**, **`mpirun -n 1`**.

**Correctness:** For this case, **`l2_error_vs_R3_analytic_rms` is identical** for `OMP_NUM_THREADS` in `{1,2,4,8}` (bit-for-bit in repeated runs), so the OpenMP region is **not** changing the numerics—only scheduling.

**Scaling (2026-04-29, shared login node):** Real FD run, **`mpirun -n 1`**, after the in-process **single-rank CPU mask reset** on Linux so `omp_get_num_procs()` matches the node (here **32**). **`l2_error_vs_R3_analytic_rms=9.91404e-21`** was **identical** for \(\{1,2,4,8\}\) threads. Speedup is **sub-linear** (memory traffic per step + fork/join each step), but **8 threads \(\approx\) 5.3\(\times\)** faster than 1 thread on this capture—consistent with a healthy OpenMP Laplacian loop rather than a pinned single-core mask.

| OMP_NUM_THREADS | avg_step_time_s |
|-----------------|-----------------|
| 1 | 2.86396 |
| 2 | 1.53138 |
| 4 | 0.883718 |
| 8 | 0.535633 |

```bash
for t in 1 2 4 8; do
  export OMP_NUM_THREADS=$t
  mpirun -n 1 --mca btl tcp,self --mca oob tcp ./apps/heat3d/heat3d_fd 256 25 1e-6 1.0 12
done
```

## Hybrid MPI + OpenMP (FD)

Use **fewer MPI ranks** and **`OMP_NUM_THREADS`** matching usable cores so halo exchange stays smaller while threads parallelize over interior \((i_y,i_z)\) lines (the `omp parallel for collapse(2)` lives in `pfc::field::for_each_interior` for the compact driver and in `pfc::field::for_each_inner_omp` / `for_each_owned_omp` for the manual driver).

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close   # optional; site-dependent
mpirun --bind-to none -n 4 --mca btl tcp,self ./apps/heat3d/heat3d_fd 256 25 1e-6 1.0 12
```

The **spectral** binaries are unchanged (MPI over ranks only).

## See also

- [`examples/15_finite_difference_heat.cpp`](../../examples/15_finite_difference_heat.cpp) — FD + halos pattern
- [`examples/diffusion_model.hpp`](../../examples/diffusion_model.hpp) — spectral diffusion IC and operator
