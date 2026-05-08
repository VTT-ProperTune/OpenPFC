<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Kobayashi CUDA on NVIDIA H100 — scaling notes

This note summarizes **strong scaling** and **profiling** experiments for `kobayashi_fd_cuda` on partition **`nvidia_h100`** (Open MPI **5.0.10**, CUDA **13.x**, GCC **11.2**). Slurm drivers live under [`slurm/`](../slurm/).

## Scripts and artifacts

| Purpose | Script |
|--------|--------|
| Configure + build `kobayashi_fd_cuda` on a GPU node | [`kobayashi_rebuild_openpfc_cuda_h100.sbatch`](../slurm/kobayashi_rebuild_openpfc_cuda_h100.sbatch) |
| Timed run, **2 ranks / 2 GPUs**, default **512²**, **5000** steps | [`kobayashi_fd_cuda_h100_np2_1cpu.sbatch`](../slurm/kobayashi_fd_cuda_h100_np2_1cpu.sbatch) |
| Same with **Nsight Systems** (`nsys profile …`), default **512²**, **200** steps | [`kobayashi_fd_cuda_h100_np2_nsys.sbatch`](../slurm/kobayashi_fd_cuda_h100_np2_nsys.sbatch) |
| **A/B nsys:** GPU-aware vs **packed** halos (two profiles, one job) | [`kobayashi_fd_cuda_h100_np2_nsys_halo_path_compare.sbatch`](../slurm/kobayashi_fd_cuda_h100_np2_nsys_halo_path_compare.sbatch) |

Environment knobs used by the timed/nsys drivers:

- **`OMPI_MCA_mtl=^ofi`** unless **`KOBAYASHI_OMPI_KEEP_OFI=1`** — avoids the Libfabric MTL path that has been problematic with CUDA buffers on some clusters.
- **`SLURM_MPI_TYPE=pmix`** — Slurm / Open MPI launch consistency.
- **Nsight artifacts:** the **`np2_nsys`** and **`np2_nsys_halo_path_compare`** scripts write **`*.nsys-rep`** under **`/WRKDIR/$USER/openpfc`** when that directory exists (fast NVMe); otherwise under **`SLURM_SUBMIT_DIR`**. Override with **`KOBAYASHI_NSYS_ARTIFACT_BASE`** (see [`pick_nsys_artifact_base.bash`](../slurm/pick_nsys_artifact_base.bash)).

Large-grid nsys overrides (example):

```bash
export KOBAYASHI_NSYS_NX=8192 KOBAYASHI_NSYS_NY=4096 KOBAYASHI_NSYS_STEPS=200
sbatch apps/kobayashi/slurm/kobayashi_fd_cuda_h100_np2_nsys.sbatch
```

For **MPI binding parity** with [`kobayashi_fd_cuda_h100_np2_1cpu.sbatch`](../slurm/kobayashi_fd_cuda_h100_np2_1cpu.sbatch) (which requests **1 CPU per task**), submit the nsys script with e.g. **`sbatch --cpus-per-task=1 …`** so `mpirun --bind-to core` sees the same slot layout.

## What the logs measure

With **`OPENPFC_KOBAYASHI_PERF=1`** (set in the Slurm scripts), rank 0 prints:

- **`KOBAYASHI_PERF_LOOP`** — `MPI_Wtime` buckets: halo **exchange driver**, **`stage_a` / `stage_b`** CUDA kernels, PNG inside the loop, vs **`wall_loop_max_s`** (MPI_MAX over ranks).
- **`OPENPFC_CUDA_PROFILE_HALO_SUMMARY`** — MPI_MAX over ranks of halo internals (`pre_stream_sync`, **`gpu_aware_mpi`**, `post_exchange_cuda_sync`, packed-face stages if used).

**`KOBAYASHI_CUDA_HALO_MODE`** reports the active path (e.g. **`gpu_aware_mpi`** vs **`packed_faces_pcie`**).

## Two regimes for **`gpu_aware_mpi`**

Jobs **1236633** and **1236626** were **identical** on paper: same binary tree, same **`512²`** problem, same **`KOBAYASHI_CUDA_HALO_MODE=gpu_aware_mpi`**, same Slurm script class.

| Job | `exchange_per_step_avg_s_max` (approx.) | Interpretation |
|-----|----------------------------------------|----------------|
| **1236633** | **~6.0×10⁻⁴** s/step | Expected healthy GPU-aware halo |
| **1236626** | **~0.176** s/step | Pathological wait inside **`gpu_aware_mpi`** |

So there is **no separate “secret environment”** visible in those logs — the difference is **intermittent runtime / network / UCX behavior** on the same mode string. When the stack cooperates, nearly all halo wall time remains in the **`gpu_aware_mpi`** bucket but stays **sub‑millisecond per step**; when it does not, that same bucket grows by **~300×** while physics kernels stay cheap.

**Mitigations to prefer** (before declaring a regression in application code):

1. Keep the Slurm script defaults above (**`mtl=^ofi`**, **pmix**).
2. Build with **`mpicxx`** as `CMAKE_CXX_COMPILER` so OpenPFC’s [**GPU-aware MPI** probe](../../../../cmake/OpenPFCGpuAwareMpi.cmake) runs against real Open MPI headers (documented in the rebuild sbatch).
3. Align CPU resources / binding between timed and nsys jobs when comparing numbers (**`--cpus-per-task=1`** for parity).
4. Enable **NVIDIA persistence** on shared nodes if initialization warnings appear (reduces noise; separate from the orders-of-magnitude **`gpu_aware_mpi`** gap).

If **`gpu_aware_mpi`** stays unstable, force **`packed_faces_pcie`** via **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`** or rebuild with **`KOBAYASHI_REBUILD_CUDA_MPI_AWARE=0`** (see rebuild script).

## Why GPU-aware halos can show millions of `cuMemcpyAsync` calls

**Application code** on the GPU-aware path (`PaddedDeviceHaloExchanger::exchange_gpu_aware_` in [`padded_device_halo_exchange.hpp`](../../../include/openpfc/runtime/cuda/padded_device_halo_exchange.hpp)) does **not** call `cudaMemcpy` / `cudaMemcpyAsync`. It passes the **device** base pointer of the padded brick to `MPI_Irecv` / `MPI_Isend` with **MPI derived types** built by **`MPI_Type_create_subarray`** ([`halo_mpi_types.hpp`](../../../include/openpfc/kernel/decomposition/halo_mpi_types.hpp)). Each face is a **3-D subarray** of that brick; most faces are **not** a single contiguous byte range in memory (the slowest index in `MPI_ORDER_C` is **z**, then **y**, then **x**).

The MPI runtime (here **Open MPI** + **UCX** with CUDA) must still **materialize** those non-contiguous device regions for the network path. Nsight’s **`cuda_api_sum`** then often shows **tens of millions** of **`cuMemcpyAsync`** and matching **`cuStreamSynchronize`** calls: that is **driver / transport** work, not a loop in OpenPFC that copies “one double per call” from C++. The **packed** fallback explicitly **gathers** each face with `padded_pack_face_kernel`, then does **one** `cudaMemcpyAsync` per face to pinned host, **contiguous** MPI, then **one** H2D per face — so **`Num Calls`** for `cuMemcpyAsync` stays **O(faces × fields × steps)**, not O(face elements).

**Sanity check** (e.g. jobs **1236668** / **1236776**): with **6** halo rounds per step and **200** steps, the app does **1200** `exchange_halos_device` calls per rank. If Nsight reports **~39.3×10⁶** `cuMemcpyAsync` calls total, that is **~3.3×10⁴** per exchange — on the order of **one small copy per owned line** along the face normal (e.g. **4096** doubles × a few logical transfers), which matches “subarray realized as many tiny copies” more than “we exchange the full volume once per step.”

### A/B experiment (confirm in one Slurm job)

[`kobayashi_fd_cuda_h100_np2_nsys_halo_path_compare.sbatch`](../slurm/kobayashi_fd_cuda_h100_np2_nsys_halo_path_compare.sbatch) defaults to **8192×4096**, **200** steps (override with **`KOBAYASHI_NSYS_NX`**, **`KOBAYASHI_NSYS_NY`**, **`KOBAYASHI_NSYS_STEPS`**). Requests **4 hours** — two large `nsys` captures + report export can be slow.

```bash
cd "$OPENPFC_REPO"
sbatch --cpus-per-task=1 \
  apps/kobayashi/slurm/kobayashi_fd_cuda_h100_np2_nsys_halo_path_compare.sbatch
# Quick 512² smoke: add --export=ALL,KOBAYASHI_NSYS_NX=512,KOBAYASHI_NSYS_NY=512
```

The job log prints **`cuda_api_sum`** twice. **Expectation:** **A_gpu_aware** shows **very large** `Num Calls` for `cuMemcpyAsync` / `cuStreamSynchronize`; **B_packed_forced** shows **orders of magnitude fewer** `cuMemcpyAsync` calls (and **`KOBAYASHI_CUDA_HALO_MODE=packed_faces_pcie`** in the app log). Compare wall time separately: packed trades PCIe staging vs fine-grained GPU-aware copies.

## Problem size vs halo (kernels vs wall clock)

**Small grid nsys** (e.g. job **1236673**, **512²**, 200 steps): Nsight **`cuda_gpu_kern_sum`** showed **~91%** of GPU kernel time in face **pack/unpack** and **~9%** in **`stage_a` / `stage_b`**.

**Large grid nsys** (job **1236668**, **8192×4096**, 200 steps): the same report showed **~70%** pack/unpack and **~30%** stages — consistent with **volume work growing faster than surface work** *among CUDA kernels*.

However, **`KOBAYASHI_PERF_LOOP`** on **1236668** still had **`exchange_per_step_avg_s_max ~ 1.2` s** vs **`stage_*`** totals **~20 ms** over the whole run — i.e. **host/MPI wait** dominated **wall time**, the same *qualitative* failure mode as **1236626**. Interpreting “halo vs compute” for multi-GPU therefore requires **healthy `gpu_aware_mpi`** (or a packed pipeline) **and** checking **`KOBAYASHI_PERF_LOOP`**, not CUDA kernel percentages alone.

## Reproduction checklist (large grid, post-rebuild)

After a fresh CUDA build on an H100 node:

```bash
cd "$OPENPFC_REPO"
export OPENPFC_BUILD_DIR="$OPENPFC_REPO/builds/kobayashi-cuda-h100"
sbatch apps/kobayashi/slurm/kobayashi_rebuild_openpfc_cuda_h100.sbatch   # optional if tree is current
# Then (example dependency chain used in development):
# sbatch --dependency=afterok:<rebuild_id> --cpus-per-task=1 \
#   --export=ALL,KOBAYASHI_NSYS_NX=8192,KOBAYASHI_NSYS_NY=4096,KOBAYASHI_NSYS_STEPS=200 \
#   apps/kobayashi/slurm/kobayashi_fd_cuda_h100_np2_nsys.sbatch
```

Inspect **`KOBAYASHI_PERF_LOOP`** first: **`exchange_per_step_avg_s_max`** should be **~10⁻³** s/step or smaller at **512²** when the stack is healthy; **~1 s/step** at large grid indicates the slow path is still active.

## See also

- **Lessons + job 1236819 analysis (GPU-aware vs packed, Nsight + timers):** [`cuda_halo_lessons_h100.md`](cuda_halo_lessons_h100.md)
- [`../README.md`](../README.md) — binaries, env toggles, verification lines
- [`../slurm/README.md`](../slurm/README.md) — queue-specific notes
