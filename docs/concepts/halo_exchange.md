<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Halo Exchange: Status and Roadmap

This document consolidates halo (ghost cell) exchange in OpenPFC: design goals (spectral + real-space), halo policies, implementation status, and evolution.

---

## 1. Purpose

Halo exchange enables real-space stencil operations (e.g. finite difference Laplacian) on distributed domains by synchronizing ghost cells between neighboring ranks. Without it, real-space operations that need neighbor data would have to go through global FFTs. With it we can support:

- Finite difference derivatives and diffusion
- Hybrid spectral + real-space methods
- Future adaptive or local operations

---

## 2. Design goals: spectral + real-space

HeFFTe / FFT expects each rank to hold a contiguous block of physical samples for its subdomain (`fft::get_inbox` / `decomposition::get_subworld`), with no ghost layers in that layout.

In-place halos (traditional FD): ghost values are written into the boundary slabs of the same `nx×ny×nz` array used for the owned grid. After exchange, those boundary samples are not in general the same as the purely owned global-grid values at those indices on a multi-rank periodic domain. Therefore you must not use that same buffer for distributed FFT after a halo fill unless you know the physics allows it (e.g. single-rank).

Recommended for FFT + FD coexistence: keep a core buffer of size exactly the subdomain (`nx×ny×nz`) for spectral work, and store received ghost data in separate face buffers. Exchange sends from faces of the core (MPI derived types) and receives into contiguous slabs. `SeparatedFaceHaloExchanger` and `field::fd::laplacian_periodic_separated<Order>` (or the interior-only `field::fd::laplacian_interior<Order>` when the iteration is restricted to `[hw, n-hw)`) implement this path.

Minimal hybrid timestep (conceptual, periodic FD):

1. Spectral substep: `fft.forward` / `backward` on core only.
2. Before FD: `SeparatedFaceHaloExchanger::exchange_halos(core, halos)` (or in-place path if FFT is not used on that field).
3. FD: `laplacian_periodic_separated<Order>(core, halos, lap, …)` (full owned domain), or `laplacian_interior<Order>(core, lap, …)` (interior slab only).
4. Update core (and/or `lap`) as required by the scheme.

---

## 3. Halo policies

Policies describe where ghost data lives and what is safe for FFT. They are documented here; see `include/openpfc/kernel/decomposition/halo_policy.hpp` for the `enum class HaloPolicy` used in API/docs cross-references.

| Policy | Storage | FFT on same buffer | FD / stencils |
|--------|---------|-------------------|---------------|
| None | Core `nx×ny×nz` only | Yes | N/A |
| InPlace | One array; ghosts in boundary slabs of that array | No (multi-rank) after halo fill | `HaloExchanger` + `laplacian_interior<Order>` |
| **PaddedBrick** | Single contiguous `(nx+2hw)×(ny+2hw)×(nz+2hw)` buffer with a real ghost ring; owned core at `[hw, hw+n)` per axis | No (FFT does not see padded layout) | `PaddedHaloExchanger` + manual stencil over `pfc::field::PaddedBrick<T>`; see `apps/heat3d/src/cpu/heat3d_fd_scratch.cpp` (raw pointer arithmetic, `u_ptr[lin ± stride]`) and `apps/heat3d/src/cpu/heat3d_fd_manual.cpp` (lambda-iterator wrapper) |
| Separated | Core + six face halo buffers | Yes on core only | `SeparatedFaceHaloExchanger` + `laplacian_periodic_separated<Order>` (or `laplacian_interior<Order>` for interior-only) |
| Mixed / hybrid | Core has no aliased ghosts; sidecar holds all ghost data | Core only | Same as Separated; extra sync/copy steps are explicit, slower path |

Note: “No halos in the FFT block” means no ghost layers stored inside that array, not “no periodicity.” Periodicity still comes from `Decomposition` / `World`.

---

## 4. Out-of-band (“OOB”) ghost model

Ghosts are not resolved by per-point MPI or maps in hot loops. The pattern (which faces, which indices) is fixed at setup—same idea as `halo_pattern.hpp`. Each step runs batched communication, then stencils read pre-filled face buffers with closed-form indexing (see `finite_difference.hpp` for separated layout).

---

## 5. Current status

### 5.1 Components

| Component | Location | Description |
|-----------|----------|-------------|
| Halo policy enum | `include/openpfc/kernel/decomposition/halo_policy.hpp` | `HaloPolicy`: None, InPlace, Separated, MixedHybrid (documentation-oriented). |
| Face halo sizes | `include/openpfc/kernel/decomposition/halo_face_layout.hpp` | Per-face element counts and `std::vector<T>` allocation for six face buffers (order +X, −X, +Y, −Y, +Z, −Z). |
| Neighbor discovery | `decomposition_neighbors.hpp` | Face / all neighbors; periodic only. |
| Halo patterns | `halo_pattern.hpp` | `create_send_halo`, `create_recv_halo`, `create_halo_patterns`. Recv indices target in-place positions in `[0, nx·ny·nz)` today. |
| SparseVector + gather/scatter | `sparse_vector.hpp`, `sparse_vector_ops.hpp` | Pack path for halos. |
| MPI exchange | `exchange.hpp` | `isend_face` / `irecv_face` (derived types), pack `isend_data` / `irecv_data`, etc. |
| Face MPI types | `halo_mpi_types.hpp` | `create_face_types_6` for send/recv subarrays in a single `[nx,ny,nz]` buffer. |
| Padded face MPI types | `padded_halo_mpi_types.hpp` | `create_padded_face_types_6` — same idea but the outer extents are `(nx+2hw, ny+2hw, nz+2hw)` and recv subarrays target the dedicated halo ring rather than the outermost owned cells. |
| In-place driver | `halo_exchange.hpp` | `HaloExchanger<T>`: recv into core boundary slabs (traditional). |
| Padded brick driver | `padded_halo_exchange.hpp` | `PaddedHaloExchanger<T>`: in-place non-blocking face exchange on a `pfc::field::PaddedBrick<T>` so `u(i±hw, j, k)` legitimately reaches the ghost ring. Same `start_/finish_halo_exchange` API as `HaloExchanger`. |
| Padded brick, **device buffer** (axis-aligned 6-face) | `runtime/cuda/padded_device_halo_exchange.hpp` | `pfc::cuda::PaddedDeviceHaloExchanger`: same MPI face derived types as `PaddedHaloExchanger<double>`, but the base pointer is a **CUDA device** allocation. Uses **GPU-aware MPI** when `OpenPFC_MPI_CUDA_AWARE` and `MPIX_Query_cuda_support()` succeed; otherwise **pack/unpack face slabs** (kernels in `src/openpfc/runtime/cuda/padded_halo_faces.cu`) + MPI on pinned host. Those kernels are **linked into `kobayashi_fd_cuda`** (not `libopenpfc_gpu_kernels`) so CUDA separable compilation device-links with the executable. Env **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`** forces the packed path. **Periodic faces whose neighbor rank equals the caller** (e.g. ±Z when the MPI process grid is one cell thick in Z) use **device pack/unpack** instead of GPU-aware MPI to self, which is slow or problematic on some stacks. **`kobayashi_fd_cuda`** with **one MPI rank** skips this in the timestep loop and applies **periodic halos on device** (`kobayashi_periodic_halos_xy_cuda` in `apps/kobayashi/`) instead of MPI + `cudaStreamSynchronize` / `cudaDeviceSynchronize` per exchange. **Fills only the 6 face halo strips — corners and edges are *not* populated**, so this exchanger is correct for axis-aligned stencils (5/7-point Laplacians, gradients along x/y/z) and **incorrect for any stencil that reads diagonal neighbours** such as the mixed second derivatives `u_xy` / `u_xz` / `u_yz`. Use `FullPaddedDeviceHalo` (below) for general second-order PDE kernels. |
| Padded brick, **device buffer** (full 26-direction) | `runtime/cuda/full_padded_device_halo.hpp` | `pfc::cuda::FullPaddedDeviceHalo`: corner- and edge-correct **multi-field** halo exchange built on the same MPI derived types and the same `padded_pack_face_kernel` as `PaddedDeviceHaloExchanger`, but in **3 widening passes** (X → Y-with-X-halos → Z-with-XY-halos) so every cell of the halo ring `[-hw, 0)` and `[n, n+hw)` on every axis is populated, including the 12 edge strips and 8 corner cubes. After the call, every padded cell equals the **periodic-equivalent global value** at that offset — verified bit-identically by `tests/integration/scenarios/parallel_scaling/test_full_padded_device_halo.cpp` on 1/2/4 ranks. Self-axis loops (proc-grid extent 1 along an axis) use device pack/unpack with widened slabs and the **correct** periodic direction (the +axis halo receives the rank's *first* `hw` owned cells, not its *last* `hw`); the self-pack in `PaddedDeviceHaloExchanger` does not implement this wrap correctly and remains in place only for backwards compatibility on small models that do not stress the self-axis. Cost: **3 syncs per call** vs. 2 for the single-pass exchanger; total MPI message count is unchanged. Env **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`** falls back to a per-field axis-aligned `PaddedDeviceHaloExchanger` (corners *not* filled), useful as a sanity check against the same env switch in the older exchanger. **Foundation layer for the planned PDE-kernel codegen path** (`f(t, u, ∂_i u, ∂_i ∂_j u)`): see [`docs/development/refactoring_roadmap.md`](../development/refactoring_roadmap.md) once the rest of the stack lands. |
| Padded brick storage | `field/padded_brick.hpp` | `pfc::field::PaddedBrick<T>` — single contiguous owned + halo-ring buffer; `T &operator()(int i, int j, int k)` valid for any `i, j, k in [-hw, n+hw)`. |
| Brick iteration | `field/brick_iteration.hpp` | `for_each_owned`, `for_each_inner(brick, r, fn)`, `for_each_border(brick, r, fn)` yielding `(int i, int j, int k)` triples — drives the laboratory-style FD loop. OMP-parallel `_omp` variants over `(k, j)`. |
| Separated driver | `separated_halo_exchange.hpp` | `SeparatedFaceHaloExchanger<T>`: send from core, recv into separate face buffers. |
| Persistent halos | `halo_persistent.hpp` | `PersistentHaloExchanger` for in-place six-face path. |
| FD primitives | `field/fd_apply.hpp`, `field/fd_stencils.hpp` | `apply_d1_along<Axis, Stencil>`, `apply_d2_along<Axis, Stencil>`, `apply_tensor_d<Mx, My, Mz, ...>`, `EvenCentralD1<Order>` (orders 2..14), `EvenCentralD2<Order>` (orders 2..20) |
| Generic stencils (custom: Sobel, CNN, anisotropic) | `field/stencil_apply.hpp` | `apply_1d_along<Axis>(coeffs, half_width, ...)`, `apply_separable(cx, Hx, cy, Hy, cz, Hz, ...)`, `apply_dense<Nz, Ny, Nx>(weights, ...)` — runtime-coefficient primitives that accept arbitrary kernels (laboratory layer for custom evaluators built atop the same halo plumbing). |
| FD bricks | `field/finite_difference.hpp` | `laplacian_interior<Order>`, `laplacian_periodic_separated<Order>`, `laplacian2d_xy_interior<Order>`, `laplacian2d_xy_periodic_separated<Order>`, runtime-order `laplacian_interior(int order, ...)` |
| FD point evaluator (CPU) | `field/fd_gradient.hpp` | `pfc::field::FdGradient<G>` + factory `pfc::field::create<G>(LocalField, order)` — populates `g.value / g.x / g.y / g.z / g.xx / g.yy / g.zz` according to `pfc::field::has_*<G>`. Mixed seconds (`xy / xz / yz`) are `static_assert`-rejected pending the corner-filled CPU exchanger. |
| FD point evaluator (GPU) | `runtime/cuda/fd_gradient_device.hpp` | `pfc::cuda::FdGradientDevice<G>` — host-side wrapper around a trivially-copyable `FdGradientDevicePOD` (padded device pointer + pre-scaled weight rows). `__device__ G evaluate_fd_grad<G>(POD, i, j, k)` does the per-point work. Mirrors the CPU twin's pruning + diagnostic-throw behaviour. |
| GPU `for_each_interior_device` | `runtime/cuda/for_each_interior_device.hpp` | `pfc::sim::cuda::for_each_interior_device(model, eval_pod, du_padded, t, nx, ny, nz, stream)` — single-field driver that launches a templated kernel calling `model.rhs(t, g)` for every owned cell of a padded device buffer. The model's `rhs` must carry the portable `OPENPFC_HD` annotation from `kernel/data/host_device.hpp` so it is callable from `__device__`. |
| Examples | `examples/15_finite_difference_heat.cpp` | Separated halos + heat equation; core is FFT-safe. |

Design choice: Indices for the pack path are exchanged once at setup; only values move each step.

Index semantics (in-place):

- Send: Local linear indices of the boundary layer to send.
- Recv: Local linear indices where received data is written—inside the same `nx×ny×nz` array (boundary slabs).

Separated recv: No scatter into core; MPI receives into contiguous face buffers whose element order matches the same face traversal as `create_recv_halo` (and MPI subarray layout).

### 5.2 Tests

- Unit: `tests/unit/kernel/decomposition/test_halo_pattern.cpp`, `test_halo_face_layout.cpp`, `test_padded_halo_mpi_types.cpp` (per-face subarray geometry, MPI_Sendrecv on `MPI_COMM_SELF`).
- Unit: `tests/unit/kernel/field/test_padded_brick.cpp`, `test_brick_iteration.cpp` — padded indexing, owned/inner/border iteration counts.
- Integration: `test_halo_patterns.cpp`, `test_halo_exchange_driver.cpp`, `test_padded_halo_exchange.cpp` (1/2/4-rank periodic wrap, axis-aligned 6-face), `test_fd_heat_mpi.cpp` — MPI parity (in-place vs separated where applicable).
- Integration (CUDA): `test_full_padded_device_halo.cpp` — bit-identical full 26-direction fill on 1, 2 (`2x1x1`), and 4 (`2x2x1`) ranks; every padded cell, including all 12 edges and 8 corners, is checked against `hash(periodic_global_coord)`. Single-rank `hw=2` case also covered.
- Integration (CUDA): `test_fd_gradient_device.cu` — end-to-end `for_each_interior_device(model, eval.pod(), du, t, ...)` against an analytic polynomial RHS (`u = a + b x + c x² + d y + e y² + f z + g z²`, `rhs = value + ∂_i u + ∂_i^2 u`). Confirms every owned cell matches the closed form to within `1e-9` and that halo cells of `du` stay untouched. Constructor diagnostic-throw test for unsupported D1 orders also covered.

### 5.3 Documentation references

- Architecture: `docs/architecture.md`
- Design history: `llm/user-stories/0009-implement-halo-exchange-layer.md`, `llm/IMPLEMENTATION_HALO_PATTERN.md`, `llm/IMPLEMENTATION_SPARSE_VECTOR.md`, `llm/design/finite_difference_gradient_design.md`

---

## 6. Architecture (data flow)

- Decomposition defines global domain and per-rank boxes.
- Patterns map directions to send/recv index sets (in-place) or drive pack/unpack sizes (separated).
- In-place flow: `gather` → MPI → `scatter` into core, or zero-copy `isend_face`/`irecv_face` on core.
- Separated flow: `isend_face` from core + `MPI_Irecv` into face buffer (or pack path: gather from core, `irecv_data` into SparseVector, memcpy to face buffer).

---

## 7. State of the art and improvements

| Aspect | OpenPFC | Notes |
|--------|---------|--------|
| Index vs data | Setup once | Same as common practice |
| Six-face path | Derived types on send from core | Separated path: contiguous recv into slabs |
| Fewer than six dirs | Pack path | Separated: gather/scatter to face buffers |
| Overlap | `start_` / `finish_halo_exchange` | Same idea for separated (future split API) |
| GPU / aware MPI | See exchange docs | Separated CPU path first |

---

## 8. Gaps and limitations

- Persistent separated exchanger not implemented yet (mirror `PersistentHaloExchanger`).
- Overlap API for separated exchanger: can add `start_` / `finish_` mirroring `HaloExchanger`.
- Wider stencils / edges / corners on the **CPU** padded-brick path: `PaddedHaloExchanger` is axis-aligned only. The CUDA path has `FullPaddedDeviceHalo` (3-pass widening, 26-direction); a CPU equivalent could be cloned from it if a finite-difference kernel ever needs the full Hessian on host.
- Self-axis periodic wrap in `PaddedDeviceHaloExchanger` writes the rank's *last* `hw` owned cells into the +axis halo (rather than the *first*). This is masked on small/short runs but is incorrect for periodic boundary conditions when the dendrite or front reaches the boundary on a single-rank-per-axis grid; `FullPaddedDeviceHalo` implements the correct wrap.
- Orchestration: Optional thin `exchange_if_needed(HaloPolicy, …)` can be added when multiple call sites need it; policies are documented first.

---

## 9. Next steps

- GPU / persistent variants for `SeparatedFaceHaloExchanger`.
- Optional `DataBlock` / gradient abstractions per `llm/design/finite_difference_gradient_design.md`.
- Derived types or tuning for pack-heavy decompositions.

---

## 10. Working examples

- **Separated layout** (FFT-safe core + face buffers):
  `examples/15_finite_difference_heat.cpp` — `mpirun -np P ./15_finite_difference_heat` runs the heat equation with `SeparatedFaceHaloExchanger` and `laplacian_periodic_separated<2>`. The core field can be passed to `fft.forward` / `backward` on the same decomposition (comment in source).
- **Padded brick layout, version 0** (minimum-OpenPFC consumer of `PaddedHaloExchanger`):
  `apps/heat3d/src/cpu/heat3d_fd_scratch.cpp` — `mpirun -np P ./apps/heat3d/heat3d_fd_scratch N n_steps dt` runs the same heat equation with the only OpenPFC piece in the hot loop being `pfc::PaddedHaloExchanger<double>::exchange_halos`. Everything else is bare triple loops over `[0, n)`, manual padded `lin = (i+hw)*sx + (j+hw)*sy + (k+hw)*sz`, raw pointer arithmetic, and a plain `std::vector<double>` for the per-step Laplacian (no halo). Read this driver to see what the higher-level layouts hide.
- **Padded brick layout, laboratory style** (in-place ghost ring + comm/compute overlap):
  `apps/heat3d/src/cpu/heat3d_fd_manual.cpp` — `mpirun -np P ./apps/heat3d/heat3d_fd_manual N n_steps dt` runs the same heat equation against `pfc::field::PaddedBrick<double>` + `pfc::PaddedHaloExchanger<double>` with `HeatModel::rhs` and the `for_each_inner / for_each_border / for_each_owned` lambda iterators. The driver shows the explicit non-blocking overlap (`start_halo_exchange` → `for_each_inner_omp` → `finish_halo_exchange` → `for_each_border` → Euler) and per-section `pfc::runtime::tic/toc` timers; see [`apps/heat3d/README.md`](../../apps/heat3d/README.md) for the side-by-side comparison with `heat3d_fd_scratch` and the compact `heat3d_fd`.

---

## 11. References

External

| Topic | Reference |
|-------|-----------|
| Non-blocking halo | Irecv → Isend → Waitall |
| GPU-aware MPI | CUDA/HIP-aware MPI when available |
| MPI derived types | `MPI_Type_create_subarray`, `MPI_Type_vector` |

---

*Single source of truth for halo exchange under `docs/`.*
