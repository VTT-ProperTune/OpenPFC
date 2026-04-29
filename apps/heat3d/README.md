<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Heat3D (`heat3d`)

MPI CPU driver for the **3D heat equation** \(\partial u/\partial t = D \nabla^2 u\) on a uniform periodic brick \([0,N)^3\) with spacing 1, comparing:

- **Finite differences**: explicit Euler with separated face halos and **even spatial orders 2–20** (central Laplacian; see `laplacian_even_order_interior_separated` in `include/openpfc/kernel/field/finite_difference.hpp`).
- **Spectral**: HeFFTe-backed real FFT, semi-implicit step in Fourier space (same pattern as `examples/diffusion_model.hpp`).

Initial condition matches the diffusion examples: \(u(\mathbf{x},0)=\exp(-|\mathbf{x}|^2/(4D))\) with origin at \((0,0,0)\).

## Build

Enabled with `OpenPFC_BUILD_APPS=ON` (default). Requires HeFFTe (spectral path uses the same FFT stack as the library). When CMake finds OpenMP for C++, `heat3d` links it so the FD path can use hybrid **MPI + OpenMP**: a **single** `omp parallel for collapse(2)` over interior \((i_y,i_z)\) lines, each line calling `laplacian_even_order_interior_separated_xy_row` (set `OMP_NUM_THREADS`).

### Where OpenMP runs (FD)

Only **`apps/heat3d/src/cpu/heat3d.cpp`**: the Laplacian each step is a `collapse(2)` loop over local interior `iz` and `iy`; each iteration updates one \(x\)-line in `laplacian_even_order_interior_separated_xy_row` (`finite_difference.hpp`). **Not parallel:** `SeparatedFaceHaloExchanger::exchange_halos` (MPI), clearing `lap`, and the explicit Euler update over the full brick.

So `htop`/`top` can look **single-threaded** when wall time is dominated by **MPI** (many ranks, thin subdomains), or when the launcher has pinned the process to a **narrow CPU mask** (OpenMP then schedules all threads inside that mask).

**Single MPI rank on Linux:** `heat3d` calls `sched_setaffinity` after `MPI_Init` to allow **all online logical CPUs** for that process, so OpenMP can use the machine without extra `mpirun` flags. Set **`HEAT3D_NO_RESET_AFFINITY`** (any value) to keep the launcher’s mask unchanged (e.g. if your site policy forbids overriding binding).

**Several MPI ranks on one node:** the reset is **not** applied (`MPI_COMM_WORLD` size \(>1\)); use launcher options so each rank gets a disjoint CPU set and set `OMP_NUM_THREADS` per rank accordingly, for example:

```bash
export OMP_NUM_THREADS=4
mpirun --bind-to none -n 4 ./apps/heat3d/heat3d fd 256 25 1e-6 1.0 12
# or: export OMPI_MCA_hwloc_base_binding_policy=none
```

Rank 0 prints `omp_max_threads` and **`omp_get_num_procs()`** in the summary line. After a single-rank run, **`omp_get_num_procs()`** should match the number of CPUs OpenMP may use; if it stays at 1, check binding and whether `HEAT3D_NO_RESET_AFFINITY` is set.

### OpenMP dummy benchmark (isolates threading)

Set **`HEAT3D_FD_OMP_DUMMY=1`** to replace the real Laplacian with synthetic FP work in the **same** `omp parallel for collapse(2)` over interior `(iy,iz)`. The timed loop **skips halo exchange and the Euler update** so wall time is dominated by that parallel region (not the stencil or MPI).

Optional **`HEAT3D_FD_DUMMY_ITERS`** (default `12000`): scalar inner iterations per `(iy,iz)` line; increase for a heavier load.

Rank 0 prints affinity-related environment variables and, on Linux, **`Cpus_allowed_list:`** from `/proc/self/status`. Example sweep (same binary, same grid):

```bash
# Without mpirun: threads should spread (subject to OS)
for t in 1 2 4 8; do
  export OMP_NUM_THREADS=$t
  HEAT3D_FD_OMP_DUMMY=1 HEAT3D_FD_DUMMY_ITERS=6000 ./apps/heat3d/heat3d fd 32 10 1e-6 1.0 12
done

# mpirun -n 1: heat3d resets affinity on Linux → omp_get_num_procs() should match hardware
HEAT3D_FD_OMP_DUMMY=1 HEAT3D_FD_DUMMY_ITERS=6000 OMP_NUM_THREADS=8 mpirun -n 1 ./apps/heat3d/heat3d fd 32 10 1e-6 1.0 12

# Same with affinity reset disabled (mimics old default pin): expect omp_get_num_procs()==1
HEAT3D_NO_RESET_AFFINITY=1 HEAT3D_FD_OMP_DUMMY=1 HEAT3D_FD_DUMMY_ITERS=6000 OMP_NUM_THREADS=8 mpirun -n 1 ./apps/heat3d/heat3d fd 32 10 1e-6 1.0 12
```

**What we observed on a login node (illustrative):** before the in-process affinity reset, `mpirun -n 1` often left `Cpus_allowed_list` on a single CPU and `omp_get_num_procs()=1` while `omp_max_threads=8`. After reset (default), `omp_get_num_procs()` matches the node and dummy `avg_step_time_s` drops in line with `mpirun --bind-to none` on the same case. With **`HEAT3D_NO_RESET_AFFINITY`**, the old pinned behaviour returns.

## Usage

```text
heat3d fd <N> <n_steps> <dt> <D> <fd_order>
heat3d spectral <N> <n_steps> <dt> <D>
```

- `fd_order`: even integers **2 through 20**. Halo width is `fd_order/2` per face. Each local subdomain must be **wider** than `fd_order` in every dimension (the driver aborts with `MPI_Abort` if the interior slab is empty).
- **Stability (FD)**: explicit Euler requires a sufficiently small `dt` (CFL); higher spatial order does not remove this limit.
- **Spectral**: each step applies \(\hat u \leftarrow \hat u / (1 + \Delta t\,D|\mathbf{k}|^2)\) in complex Fourier space (implicit Euler for the linear diffusion operator).

## Example

From your build directory:

```bash
mpirun -n 4 ./apps/heat3d/heat3d fd 64 200 0.001 1.0 4
mpirun -n 4 ./apps/heat3d/heat3d spectral 64 200 0.001 1.0
```

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
mpirun -n 8 --mca btl tcp,self --mca oob tcp ./apps/heat3d/heat3d fd 256 25 1e-6 1.0 20
```

Sweep example:

```bash
export OMP_NUM_THREADS=1
for ranks in 1 2 4 8; do
  for ord in 8 10 12 14 16 18 20; do
    mpirun -n "$ranks" --mca btl tcp,self --mca oob tcp ./apps/heat3d/heat3d fd 256 25 1e-6 1.0 "$ord"
  done
done
```

### OpenMP scaling (`fd_order=12`, one MPI rank)

Same grid **`256³`**, **`n_steps=25`**, **`dt=1e-6`**, **`mpirun -n 1`**.

**Correctness:** For this case, **`l2_error_vs_R3_analytic_rms` is identical** for `OMP_NUM_THREADS` in `{1,2,4,8}` (bit-for-bit in repeated runs), so the OpenMP region is **not** changing the numerics—only scheduling.

**Scaling (2026-04-29, shared login node):** Real FD run (no `HEAT3D_FD_OMP_DUMMY`), **`mpirun -n 1`**, after the in-process **single-rank CPU mask reset** on Linux so `omp_get_num_procs()` matches the node (here **32**). **`l2_error_vs_R3_analytic_rms=9.91404e-21`** was **identical** for \(\{1,2,4,8\}\) threads. Speedup is **sub-linear** (memory traffic per step + fork/join each step), but **8 threads \(\approx\) 5.3\(\times\)** faster than 1 thread on this capture—consistent with a healthy OpenMP Laplacian loop rather than a pinned single-core mask.

| OMP_NUM_THREADS | avg_step_time_s |
|-----------------|-----------------|
| 1 | 2.86396 |
| 2 | 1.53138 |
| 4 | 0.883718 |
| 8 | 0.535633 |

```bash
for t in 1 2 4 8; do
  export OMP_NUM_THREADS=$t
  mpirun -n 1 --mca btl tcp,self --mca oob tcp ./apps/heat3d/heat3d fd 256 25 1e-6 1.0 12
done
```

## Hybrid MPI + OpenMP (FD)

Use **fewer MPI ranks** and **`OMP_NUM_THREADS`** matching usable cores so halo exchange stays smaller while threads parallelize over interior \((i_y,i_z)\) lines (`laplacian_even_order_interior_separated_xy_row` in `heat3d.cpp`; the kernel in `finite_difference.hpp` stays serial along \(x\)).

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close   # optional; site-dependent
mpirun --bind-to none -n 4 --mca btl tcp,self ./apps/heat3d/heat3d fd 256 25 1e-6 1.0 12
```

The **spectral** path is unchanged (MPI over ranks only).

## See also

- [`examples/15_finite_difference_heat.cpp`](../../examples/15_finite_difference_heat.cpp) — FD + halos pattern
- [`examples/diffusion_model.hpp`](../../examples/diffusion_model.hpp) — spectral diffusion IC and operator
