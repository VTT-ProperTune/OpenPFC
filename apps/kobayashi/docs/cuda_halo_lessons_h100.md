<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# CUDA halos on H100: lessons from GPU-aware vs packed (`kobayashi_fd_cuda`)

This note records **what we ran**, **what the numbers showed**, and **how to read them**, using job **1236819** on partition **`nvidia_h100`** (Open MPI **5.0.10**, CUDA **13.0**).

## What we actually changed (the experiment)

We did **not** rewrite the physics or the halo *semantics*. We compared two **existing** code paths in `pfc::cuda::PaddedDeviceHaloExchanger` ([`padded_device_halo_exchange.hpp`](../../../include/openpfc/runtime/cuda/padded_device_halo_exchange.hpp)):

| Leg | Environment | Mode printed by the app |
|-----|-------------|-------------------------|
| **A** | `OPENPFC_CUDA_FORCE_PACKED_HALO` **unset** (default when CUDA-aware probe succeeds) | `gpu_aware_mpi` |
| **B** | `OPENPFC_CUDA_FORCE_PACKED_HALO=1` | `packed_faces_pcie` |

Both legs used the **same** binary, grid **8192×4096**, **nz = 1**, **200** time steps, **2** MPI ranks / **2** GPUs, **`halo_batch=off`**, **`halo_extended=off`**. The Slurm driver is [`slurm/kobayashi_fd_cuda_h100_np2_nsys_halo_path_compare.sbatch`](../slurm/kobayashi_fd_cuda_h100_np2_nsys_halo_path_compare.sbatch).

**Leg A** posts **`MPI_Irecv` / `MPI_Isend`** on the **device** pointer with **MPI derived types** (`MPI_Type_create_subarray` face slabs). Application code does **not** call `cudaMemcpyAsync` on that path.

**Leg B** runs **`exchange_packed_fallback_`**: **pack** face into contiguous device scratch → **`cudaMemcpyAsync` D2H** (one per face) → **MPI on pinned host** → **`cudaMemcpyAsync` H2D** → **unpack** kernel.

So “what we did” for performance investigation was: **force the packed PCIe path** with **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`** and compare against **GPU-aware MPI** on an otherwise identical run.

Artifacts for **1236819**:

- `kobayashi_fd_cuda_h100_nsys_halo_ab_1236819/A_gpu_aware/kob_nsys.nsys-rep`
- `kobayashi_fd_cuda_h100_nsys_halo_ab_1236819/B_packed_forced/kob_nsys.nsys-rep`
- Slurm log: `kobayashi_fd_cuda_h100_np2_nsys_halo_ab_1236819.out`

---

## Nsight Systems: `cuda_api_sum` (driver call counts)

From **`nsys stats -q --report cuda_api_sum`** (embedded in the Slurm `.out`):

| API | Leg **A** (`gpu_aware_mpi`) | Leg **B** (`packed_faces_pcie`) |
|-----|---------------------------|--------------------------------|
| **`cuMemcpyAsync`** — **Num Calls** | **39,321,600** | **28,800** |
| **`cuStreamSynchronize`** (driver `cu*` row) | **39,321,600** | **33,600** |
| **`cudaStreamSynchronize`** (runtime) | 4,800 | *(small vs driver row)* |

The **~39.3M** pairs on **A** line up with **MPI/UCX realizing non-contiguous device datatypes** as a storm of **tiny driver-level copies and syncs**, not with moving “one double per application statement.” On **B**, memcpy calls scale like **O(faces × fields × steps)** with **contiguous** slabs — here **28,800** `cudaMemcpyAsync` calls for the profiled region.

On **B**, **`cudaStreamSynchronize`** also dominates **CUDA API time** in this report (**~90%** of the summed API duration), with **`cuMemcpyAsync`** at **~0.4%** — fewer copies, but each exchange still **serializes** the stream often (pack → D2H → … → H2D → unpack).

**Lessons:**

1. **Huge `Num Calls` on GPU-aware does not mean we exchange the full volume each step** — total halo **bytes** are still \(O(\text{face area})\); the problem is **how** the stack implements **`MPI_Type_subarray` on GPU memory**.
2. **Packed mode collapses the memcpy storm** at the price of **explicit staging** and **many stream synchronizations** (**33,600** `cudaStreamSynchronize` calls vs **39.3M** driver-level **`cuStreamSynchronize`** pairs on **A**) — still a vastly smaller **memcpy** count (**28,800** vs **39.3M**).

---

## Digging into `kob_nsys.sqlite`: MPI sizes (leg **B**, job **1236819**)

Nsight exports **`MPI_P2P_EVENTS`** into `B_packed_forced/kob_nsys.sqlite`. Aggregating **`size`** (bytes):

| `size` | Count | What it is |
|--------|-------|------------|
| **32,768** | **19,200** | **Thin** faces (**±X**, **±Y** with **local nz = 1**): **4096 doubles** × 8 B |
| **134,217,728** | **9,600** | **±Z** “faces”: **4096×4096 doubles** ≈ **128 MiB** **per MPI operation** |

**28,800** rows total = **12** MPI ops per rank per **`exchange_halos_device`** × **1200** calls × **2** ranks.

So the **“~32 KiB halo”** mental model applies only to the **four** small faces. For **`nz = 1`**, **±Z** slabs in the **3D padded brick** still have extent **nx×ny×hw** — geometrically the **full XY plane**, not a thin strip. That is expected from [`padded_halo_mpi_types.hpp`](../../../include/openpfc/kernel/decomposition/padded_halo_mpi_types.hpp); the surprise is what the **packed** driver did with periodic **same-rank** neighbors.

### Why **~0.32 s/step** was still absurd (packed leg **B**, pre-fix)

**`exchange_packed_fallback_`** used to **`MPI_Irecv` / `MPI_Isend` every face**, including **`m_neighbors[i] == m_rank`**. For periodic **±Z** that meant **MPI to self** moving **~128 MiB** **four times per `exchange_halos_device` call** (two Z faces × recv/send); **×1200** calls × **2** ranks ⇒ **9600** rows at **128 MiB** in **`MPI_P2P_EVENTS`**. **`KOBAYASHI_PERF_LOOP`** showed **`exchange_per_step_avg ≈ 0.319 s`** largely because of **those self moves + waits**, not because **32 KiB** peer halos are hard.

**GPU-aware** mode already skipped MPI for same-rank faces (device pack/unpack first); **packed** mode did **not**, until **`PaddedDeviceHaloExchanger`** was aligned with the same rule ([`padded_device_halo_exchange.hpp`](../../../include/openpfc/runtime/cuda/padded_device_halo_exchange.hpp)).

After rebuilding with that fix, re-run [`kobayashi_fd_cuda_h100_np2_nsys_halo_path_compare.sbatch`](../slurm/kobayashi_fd_cuda_h100_np2_nsys_halo_path_compare.sbatch): **`MPI_P2P_EVENTS`** should show **only 32 KiB** sizes for **true inter-rank** faces, and **`packed_mpi_waitall`** should drop sharply.

### OS runtime (`osrt_sum`)

Leg **B** **`osrt_sum`** is dominated by **`epoll_wait`**, **`poll`**, **`sem_wait`**, **`pthread_cond_timedwait`** — typical **MPI progress / threading**, alongside CUDA sync. That complements (does not replace) the MPI-size analysis above.

---

## Where wall time goes (`KOBAYASHI_PERF_LOOP` + halo buckets)

Same problem and step count; **`OPENPFC_KOBAYASHI_PERF=1`** prints **`MPI_Wtime`** buckets (`MPI_MAX` over ranks where noted).

### Leg **A** — GPU-aware

| Bucket | Wall (slow rank, approx.) | Share of ~246 s loop |
|--------|---------------------------|----------------------|
| **`exchange_driver_wall_s_max`** | **245.76 s** | **~99.97%** |
| **`stage_a` + `stage_b`** | ~0.010 s | &lt; 0.01% |
| **`gpu_aware_mpi`** (inside halo summary) | **245.52 s** | Matches exchange driver |

So on this cluster + grid + driver stack, **almost the entire timestep loop** is **waiting inside GPU-aware MPI**, not in the Kobayashi CUDA kernels.

### Leg **B** — Packed PCIe

| Bucket | Wall (slow rank, approx.) | Notes |
|--------|-----------------------------|--------|
| **`wall_loop_max_s`** | **63.73 s** | **~3.86× faster** than leg **A** (~246 s) |
| **`exchange_driver_wall_s_max`** | **63.73 s** | Still dominates the loop |
| **`packed_mpi_waitall`** | **46.04 s** | Largest **single** halo sub-bucket — host MPI progress / peer |
| **`packed_pack_d2h_sync`** | **9.74 s** | Pack + D2H + stream sync per face |
| **`packed_h2d_unpack_sync`** | **8.27 s** | H2D + sync + unpack |
| **`stage_a` + `stage_b`** | ~0.004 s | Still negligible |

**Lessons:**

3. **Forcing packed halos fixed the catastrophic GPU-aware path** on this case (~246 s → ~64 s per 200 steps), but **the loop stayed halo-bound** — physics stages remain **~10⁻⁵ s/step** each.
4. On the **1236819** packed binary, **`packed_mpi_waitall`** (~46 s / ~71% of halo CPU summary) was inflated by **MPI-to-self at 128 MiB** on **±Z** (see SQLite histogram above). That is **not** “32 KiB halos are intrinsically slow”; it was a **packed-path omission** now fixed in **`PaddedDeviceHaloExchanger`**. Expect **`exchange_per_step_avg`** to fall a lot after rebuild + re-profile.

---

## How this relates to “healthy” GPU-aware on small grids

On **512²**, the same **`gpu_aware_mpi`** mode sometimes shows **sub-millisecond** exchange per step (e.g. job **1236633**), while another run (**1236626**) showed **~0.18 s/step** with identical flags — **intermittent stack behavior**. The **8192×4096** legs **A** here (**~1.23 s/step**) match the **slow** regime: **large subarrays + GPU-aware** appears especially prone to bad scaling even when “CUDA-aware” is enabled.

---

## Practical checklist

1. **Diagnose:** Use **`KOBAYASHI_PERF_LOOP`** + **`OPENPFC_CUDA_PROFILE_HALO_SUMMARY`**; if **`gpu_aware_mpi`** dominates with huge **`exchange_per_step_avg`**, compare **`nsys stats cuda_api_sum`** for **`cuMemcpyAsync`** counts.
2. **Mitigate:** Try **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`** (runtime) or rebuild with GPU-aware off — trade PCIe staging + host MPI for avoiding derived-type GPU transfers. Ensure you run a build that includes **packed same-rank face handling** (device pack/unpack, **no** MPI-to-self on huge **±Z** slabs when **nz = 1**).
3. **Reduce MPI rounds:** **`KOBAYASHI_HALO_BATCH=1`** or **`KOBAYASHI_HALO_EXTENDED=1`** (see [`kobayashi_fd_cuda.cpp`](../src/cuda/kobayashi_fd_cuda.cpp)) lower the number of **`exchange_halos_device`** calls per step.
4. **Keep cluster MPI knobs:** e.g. **`OMPI_MCA_mtl=^ofi`**, **`SLURM_MPI_TYPE=pmix`** (already in the Slurm scripts).

---

## Why **±Z** halos run at all on a **2D** Kobayashi slab (`nz = 1`)

The world is still a **3D** `PaddedBrick`: [`kobayashi_fd_cuda.cpp`](../src/cuda/kobayashi_fd_cuda.cpp) builds `GridSize({Nx, Ny, 1})`, so **local `nz = 1`** with padding **`nzp = 1 + 2·hw`**. Generic six-face machinery ([`padded_halo_mpi_types.hpp`](../../../include/openpfc/kernel/decomposition/padded_halo_mpi_types.hpp), [`PaddedDeviceHaloExchanger`](../../../include/openpfc/runtime/cuda/padded_device_halo_exchange.hpp)) always defines **±Z** faces with cross-section **nx×ny** — those “faces” are **entire XY planes**, not thin **O(perimeter)** strips.

The CUDA physics kernels ([`kobayashi_fd_cuda_kernels.cu`](../src/cuda/kobayashi_fd_cuda_kernels.cu)) use **`constexpr int iz = 0`** and only index **ix±1, iy±1** at fixed **k**. They **never read ghost cells at `k±1`**. So **±Z halo data is not needed for the stencil** — it exists because the **storage layout is 3D** and the exchanger is **axis-aligned 6-face**, not because the equations use **z** neighbours.

**Single rank** already avoids MPI on the step loop and applies **x/y + z** periodicity on device via **`kobayashi_periodic_halos_xy_cuda`** (including **`kobayashi_periodic_halos_z_edges_hw1_kernel`**). **Multi-rank** still drives **all six faces** through the generic exchanger; **±Z** should be **same-rank periodic only** (now handled without MPI-to-self after the packed fix), but work is still **O(nx·ny)** pack/unpack **per field per exchange** instead of a **cheap local z-wrap kernel**.

**Not best possible for this app:** an optimal **2D slab + MPI** path would **MPI-exchange only directions that split the domain** (here typically **±X** with a **1×2** rank grid, **±Y** self-periodic like today) and fill **±Z ghosts only with the existing device z-edge kernel** — **no** six-face **nx×ny** “halo” path for **z**. That would require either a **nz==1-aware** exchanger flag or a Kobayashi-specific exchange wrapper.

### **Done:** `Axes2D()` halo direction set switch

[`kobayashi_fd_cuda.cpp`](../src/cuda/kobayashi_fd_cuda.cpp) now constructs both **`PaddedDeviceHaloExchanger`** and **`BatchedPaddedDeviceHalo`** with **`pfc::halo::presets::Axes2D()`** (4 directions: **±X, ±Y**) instead of the historical implicit `Axes3D()` (6 directions). With this set:

- The exchangers no longer iterate over the **±Z** slots in either the **GPU-aware** or **packed** branches; no MPI calls, no device pack/unpack, and no scratch allocations are issued for **±Z**.
- The **128 MiB** **±Z** entries that previously dominated **`MPI_P2P_EVENTS`** in leg **B** are gone — the histogram should now contain only the **32,768-byte** thin-face messages.
- The legacy **`kobayashi_periodic_halos_z_edges_hw1_kernel`** is no longer needed in the multi-rank path either (kernels read only **iz = 0**); cleanup of that kernel in the single-rank path is tracked separately.

The change is opt-in via the new ctor: callers that want the old behaviour pass `pfc::halo::presets::Axes3D()` explicitly. See [`docs/concepts/halo_exchange.md` § 5.4 Direction sets and presets](../../../docs/concepts/halo_exchange.md) for the preset table and per-rank selector hook.

### Should we use **`FullPaddedDeviceHalo`** (26 neighbours / corner-filled)?

**No** for current Kobayashi FD: [`full_padded_device_halo.hpp`](../../../include/openpfc/runtime/cuda/full_padded_device_halo.hpp) exists for stencils that need **corners** (mixed **∂²/∂x∂y**, 27-point neighbours, etc.). Kobayashi’s kernels are **axis-aligned five-point in x/y only**; [`docs/concepts/halo_exchange.md`](../../../docs/concepts/halo_exchange.md) states six-face exchange is **sufficient**. **`FullPaddedDeviceHalo`** would add **three passes** with extra **`cudaDeviceSynchronize`** between passes and **still** treat **z** as a full axis — it does **not** replace a dedicated **“2D MPI + local z wrap”** optimization.

The **“8 vs 26”** ghost regions are a **topological** count (2D perimeter pieces vs 3D shell); OpenPFC’s CUDA path implements **faces** (6 or widened passes), not **8 separate MPI buffers** for 2D.

---

## See also

- Overview and script index: [`scalability_cuda_h100.md`](scalability_cuda_h100.md)
- Conceptual halo documentation: [`docs/concepts/halo_exchange.md`](../../../docs/concepts/halo_exchange.md)
