<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Kobayashi dendritic growth (manual FD)

Coupled **phase field** \(\phi\) and **temperature** \(T\) after Kobayashi (Physica D, 1993), using the same explicit finite-difference layout as the historical Julia script `kobayashi_v1` (Biner-style terms, periodic torus in **x** and **y**).

## Binaries

| Target | Description |
|--------|-------------|
| `kobayashi_fd_manual` | Two-pass explicit Euler per step on `PaddedBrick<double>`; periodic **MPI halos** (`nz = 1`). |
| `kobayashi_fd_openmp` | Same discrete splitting on a **single full grid**; periodic **torus via index wrapping** (no halos, no MPI); **OpenMP** `collapse(2)` over the two passes per step. Requires OpenMP at build time. |
| `kobayashi_fd_cuda` | Same physics as **`kobayashi_fd_manual`**, **two CUDA kernels per step**. **`MPI_COMM_WORLD` size 1** uses **device periodic halos** (`KOBAYASHI_CUDA_HALO_MODE=device_periodic_local`) — no MPI in the timestep halo path (avoids CPU busy-wait + extra CUDA sync). **`nproc > 1`** uses **`pfc::cuda::PaddedDeviceHaloExchanger`**: GPU-aware MPI when **`OpenPFC_MPI_CUDA_AWARE`** + `MPIX_Query_cuda_support`, else packed faces (**`OPENPFC_CUDA_FORCE_PACKED_HALO=1`**). Rank 0 prints **`KOBAYASHI_CUDA_HALO_MODE`**. Build with **`-DOpenPFC_ENABLE_CUDA=ON`**. |
| `kobayashi_fd_hip` | Same MPI + halo pattern as **`kobayashi_fd_manual`**, with **two HIP kernels per step** (host-staged halos: `hipMemcpy` + `PaddedHaloExchanger`, portable without GPU-aware MPI). Each MPI rank calls **`hipSetDevice(local_rank % device_count)`** where `local_rank` is the **shared-memory** rank (`MPI_COMM_TYPE_SHARED`). Build with **`-DOpenPFC_ENABLE_HIP=ON`**. |

## Equations (discrete layout matches Julia)

- \(\tau \partial_t \phi = \partial_y(\epsilon\epsilon'\partial_x\phi) - \partial_x(\epsilon\epsilon'\partial_y\phi) + \epsilon^2 \nabla^2\phi + \phi(1-\phi)(\phi-\tfrac12+M(T))\)
- \(\partial_t T = \nabla^2 T + \kappa \partial_t \phi\) — latent heat term implemented as \(\kappa(\phi^{n+1}-\phi^n)\) after updating \(\phi\), matching the Julia script.

Material constants and output cadence (`nprint`, `nsave`) live in [`include/kobayashi/defaults.hpp`](include/kobayashi/defaults.hpp).

## Usage (`kobayashi_fd_manual`)

```bash
# Defaults = Julia script (256³ grid, 2000 steps, dt=1e-4, dx=0.03, results/kobayashi_v1/)
mpirun -n 4 ./kobayashi_fd_manual

# Explicit grid and output directory
mpirun -n 4 ./kobayashi_fd_manual 128 128 500 1.0e-4 0.03 results/my_run
```

Writes grayscale PNGs of \(\phi\) on rank 0 (`phi_0000.png`, then every `nsave` steps, plus **`phi_final.png`** after the last time step). Directory is created under `output_dir`.

## Usage (`kobayashi_fd_openmp`)

Single process; parallelism is **threads only**.

```bash
export OMP_NUM_THREADS=16
./kobayashi_fd_openmp 512 512 5000 1e-4 0.03 results/kobayashi_openmp_run

# Optional explicit thread count (8th argument); requires argv output_dir as 7th argument:
./kobayashi_fd_openmp 512 512 5000 1e-4 0.03 results/kobayashi_openmp_run 16
```

PNG cadence matches the MPI driver. **`KOBAYASHI_VERIFY`** reports **`nthreads=`** and **`wall_loop_max_s`** (single-node timer); **`KOBAYASHI_VERIFY_HEX`** uses the same global field reductions as the MPI driver — for identical `(Nx, Ny, steps, dt, dx)` it matches **`kobayashi_fd_manual`** with **`nproc=1`** (bitwise, same toolchain).

Unit test: **`test_kobayashi_fd_openmp`** (Catch2) checks **1 vs 4 threads** field equality on a small grid.

Slurm thread scaling (partition **`gen05_epyc`**): [`slurm/kobayashi_openmp_scaling_gen05_epyc.sbatch`](slurm/kobayashi_openmp_scaling_gen05_epyc.sbatch) plus **`summarize_openmp_scaling.py`** / **`plot_strong_scaling.py`** on the resulting **`summary.tsv`** (first column **`nthreads`**).

### Verification / timing lines (stdout)

After the time-step loop, rank 0 prints **`KOBAYASHI_VERIFY`** (decimal scalars + **`wall_loop_max_s`**, the MPI_MAX of per-rank loop timers) and **`KOBAYASHI_VERIFY_HEX`** (`sum_phi`, `sumsq_phi`, `sum_T`, `sumsq_T` in `%a` form). Global sums use a **gather + fixed \((g_x,g_y)\) accumulation order** so results match across MPI rank counts when the physics is bitwise deterministic.

Environment toggles:

- **`OPENPFC_KOBAYASHI_SKIP_PNG=1`** — skip all PNG I/O (useful for scaling studies).
- **`OPENPFC_KOBAYASHI_QUIET=1`** — suppress per-`nprint` progress lines.
- **`OPENPFC_KOBAYASHI_PERF=1`** — after the timestep loop, rank 0 prints **`KOBAYASHI_PERF_LOOP`** (CPU `MPI_Wtime` around halo driver calls, **`kobayashi_stage_a_cuda`**, **`kobayashi_stage_b_cuda`**, PNG saves inside the loop, and **unaccounted** wall vs **`wall_loop_max_s`**). With **`nproc > 1`**, also sets **`OPENPFC_CUDA_PROFILE_HALO=1`** unless already set, then prints **`OPENPFC_CUDA_PROFILE_HALO_SUMMARY`** (MPI_MAX across ranks of halo internals: pre-stream sync, MPI, post CUDA sync, packed D2H / wait / H2D if used). Override buckets only with **`OPENPFC_CUDA_PROFILE_HALO=0`** in the environment if you want the loop breakdown without halo detail.

Slurm strong-scaling driver (partition **`gen05_epyc`**) and log summariser: [`slurm/README.md`](slurm/README.md).

## Usage (`kobayashi_fd_cuda`)

Same CLI as **`kobayashi_fd_manual`** (`Nx Ny n_steps dt dx [output_dir]`). Requires CUDA runtime per rank.

```bash
mpirun -np 1 ./kobayashi_fd_cuda 512 512 5000 1e-4 0.03 results/kobayashi_cuda
mpirun -np 2 ./kobayashi_fd_cuda 512 512 5000 1e-4 0.03 results/kobayashi_cuda_2gpu
```

NVIDIA H100 Slurm workflow (rebuild + 1 vs 2 GPU scaling, partition **`nvidia_h100`**): [`slurm/kobayashi_rebuild_openpfc_cuda_h100.sbatch`](slurm/kobayashi_rebuild_openpfc_cuda_h100.sbatch), [`slurm/kobayashi_cuda_scaling_h100.sbatch`](slurm/kobayashi_cuda_scaling_h100.sbatch). Override **`CMAKE_CUDA_ARCHITECTURES`** if your queue uses a different GPU (default **`90`** = H100). The rebuild script always uses **`${OPENPFC_REPO}/builds/kobayashi-cuda-h100`** unless you set **`KOBAYASHI_CUDA_OPENPFC_BUILD_DIR`** (so a leftover **`OPENPFC_BUILD_DIR`** from HIP does not clobber the CUDA tree).

**Performance (CUDA):** Halos no longer use **full-volume** device↔host copies each step. **`PaddedDeviceHaloExchanger`** either posts MPI directly on **device pointers** (CUDA-aware stack) or exchanges **face slabs only** via pack/unpack kernels + pinned host buffers. Remaining costs: six fields exchange per step, **`cudaDeviceSynchronize`** around MPI when GPU-aware (required for correctness with typical Open MPI builds), and PNG paths still stage \(\phi\) on the host. The **`nvidia_h100`** Slurm rebuild script (**`kobayashi_rebuild_openpfc_cuda_h100.sbatch`**) defaults to **`OpenPFC_MPI_CUDA_AWARE=ON`**; set **`KOBAYASHI_REBUILD_CUDA_MPI_AWARE=0`** there to force a packed-only build. Thread blocks: **`OPENPFC_KOBAYASHI_CUDA_BLOCK`** (default **32×32**); see **`INSTALL.md`** for MPI stack notes.

## Usage (`kobayashi_fd_hip`)

Same CLI as **`kobayashi_fd_manual`** (`Nx Ny n_steps dt dx [output_dir]`). Requires ROCm/HIP at runtime per rank.

```bash
# One rank → one visible GPU (typical Slurm: --gpus-per-task=1)
mpirun -np 1 ./kobayashi_fd_hip 512 512 5000 1e-4 0.03 results/kobayashi_hip

# Two ranks on one dual-GPU node → rank 0 uses GPU 0, rank 1 uses GPU 1
mpirun -np 2 ./kobayashi_fd_hip 512 512 5000 1e-4 0.03 results/kobayashi_hip_2gpu
```

AMDGPU Slurm workflow (rebuild + 1 vs 2 GPU scaling): [`slurm/kobayashi_rebuild_openpfc_amdgpu.sbatch`](slurm/kobayashi_rebuild_openpfc_amdgpu.sbatch), [`slurm/kobayashi_hip_scaling_amdgpu.sbatch`](slurm/kobayashi_hip_scaling_amdgpu.sbatch).

**Note:** GPU reductions may differ slightly from CPU/MPI-only builds; compare **`KOBAYASHI_VERIFY_HEX`** against **`kobayashi_fd_manual`** only when validating the same floating-point path.

## See also

- [`apps/wave2d/README.md`](../wave2d/README.md) — similar manual FD + coupled fields pattern on a slab
- [`docs/user_guide/applications.md`](../../docs/user_guide/applications.md)
