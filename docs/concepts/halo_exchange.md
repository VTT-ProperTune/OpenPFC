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

Recommended for FFT + FD coexistence: keep a core buffer of size exactly the subdomain (`nx×ny×nz`) for spectral work, and store received ghost data in separate face buffers. Exchange is driven by `pfc::SparseHaloExchanger<T>` ([`include/openpfc/kernel/decomposition/sparse_halo_exchange.hpp`](../../include/openpfc/kernel/decomposition/sparse_halo_exchange.hpp)) — fully sparse, grid-agnostic, accepts an arbitrary `std::vector<halo::RemoteHalo<T>>`. For the standard structured face exchange, `pfc::halo::make_structured_halos<T>(decomp, rank, hw, dirs = Axes3D())` builds the `RemoteHalo` list from a `HaloDirectionSet`. After the exchange, `pfc::halo::copy_to_face_layout(ex, face_halos)` (see [`halo_face_layout.hpp`](../../include/openpfc/kernel/decomposition/halo_face_layout.hpp)) refills the `std::array<std::vector<T>, 6>` layout that `field::fd::laplacian_periodic_separated<Order>` (or the interior-only `field::fd::laplacian_interior<Order>` when the iteration is restricted to `[hw, n-hw)`) expects.

Minimal hybrid timestep (conceptual, periodic FD):

1. Spectral substep: `fft.forward` / `backward` on core only.
2. Before FD: `ex.exchange_halos(core, core_size); halo::copy_to_face_layout(ex, face_halos);` (or use the padded-brick path if FFT is not used on that field).
3. FD: `laplacian_periodic_separated<Order>(core, face_halos, lap, …)` (full owned domain), or `laplacian_interior<Order>(core, lap, …)` (interior slab only).
4. Update core (and/or `lap`) as required by the scheme.

### Which exchanger when

Use `pfc::field::PaddedBrick<T>` + `pfc::communication::PaddedHaloExchanger<T>` (or the CUDA `pfc::cuda::PaddedDeviceHaloExchanger`) for **classical FD** with a single contiguous `(nx+2hw)*(ny+2hw)*(nz+2hw)` array and negative halo indexing. This is the **default** for FD-only apps and is the lowest-overhead path: ghost width is baked into storage, and the inner stencil reads `u(i±hw, j, k)` directly with no face-buffer indirection.

Use `pfc::field::LocalField<T>` (unpadded `nx*ny*nz`) + `pfc::SparseHaloExchanger<T>` (typically built via `pfc::halo::make_structured_halos`) for:

- **Mixed FD + spectral** — the FFT does not tolerate halo regions in the data block, so the core stays unpadded.
- **Non-axis halos** (edges / corners) without baking a 3-pass widening into the FD path — the user supplies any `HaloDirectionSet` (or any `RemoteHalo` list at all).
- **Arbitrary peer / index patterns** that have no face geometry — multi-block grids, hopping over distance, mixed `(peer_rank, send_indices, recv_indices, send_tag, recv_tag)` tuples.
- **Future unstructured / FEM** halo communication — the API already accepts arbitrary `RemoteHalo` entries; only a `make_fem_halos(...)` builder needs to land on top.

### Stage preparation protocol

`PaddedHaloExchanger` / `HaloExchanger` / `SparseHaloExchanger` are **transport**: they move ghost faces given buffers and a decomposition. They do not interpret integrator stage flags or boundary-condition timing.

`pfc::communication::StagePreparationService` ([`stage_preparation.hpp`](../../include/openpfc/kernel/decomposition/stage_preparation.hpp)) is the **protocol** layered on that transport for CPU/MPI padded bricks:

- Consumes `StagePreparationRequirements` (`needs_halo_exchange`, `needs_boundary_update`, `region_kind`, `BoundaryHaloOrder`), typically via `pfc::integrator::requirements_from(StageContext)`.
- When halo is required, calls existing `pfc::communication::exchange` on named, bound `PaddedHaloExchanger`s.
- When boundary update is required, runs an injectable boundary hook. Default order is **boundary then halo** so updated owned faces are published to neighbors before evaluation.
- `prepare` is **pre-evaluation** only. Post-evaluation BC enforcement after writing new owned values stays a separate driver responsibility outside `prepare`.
- Rejection / retry does not roll back halo buffers: re-prepare from the accepted owned core.

Method and operator layers should request `prepare` rather than embedding ad-hoc MPI at each evaluation site. Raw exchanger calls remain valid for drivers that manage timing themselves.

---

## 3. Halo policies

Policies describe where ghost data lives and what is safe for FFT. They are documented here; see `include/openpfc/kernel/decomposition/halo_policy.hpp` for the `enum class HaloPolicy` used in API/docs cross-references.

| Policy | Storage | FFT on same buffer | FD / stencils |
|--------|---------|-------------------|---------------|
| None | Core `nx×ny×nz` only | Yes | N/A |
| InPlace | One array; ghosts in boundary slabs of that array | No (multi-rank) after halo fill | `HaloExchanger` + `laplacian_interior<Order>` |
| **PaddedBrick** | Single contiguous `(nx+2hw)×(ny+2hw)×(nz+2hw)` buffer with a real ghost ring; owned core at `[hw, hw+n)` per axis | No (FFT does not see padded layout) | `pfc::communication::PaddedHaloExchanger` + manual stencil over `pfc::field::PaddedBrick<T>`; see `apps/heat3d/src/cpu/heat3d_fd_scratch.cpp` (raw pointer arithmetic, `u_ptr[lin ± stride]`), `apps/heat3d/src/cpu/heat3d_fd_manual.cpp` (lambda-iterator wrapper), and `apps/heat3d/src/cpu/heat3d_fd.cpp` (`pfc::communication::exchange` or `start_exchange` / `finish_exchange` + `pfc::gradient::FDGradient` + `pfc::field::for_each` on the same padded layout) |
| Separated | Core + six face halo buffers | Yes on core only | `pfc::SparseHaloExchanger<T>` built by `halo::make_structured_halos<T>` + `halo::copy_to_face_layout` + `laplacian_periodic_separated<Order>` (or `laplacian_interior<Order>` for interior-only) |
| Mixed / hybrid | Core has no aliased ghosts; sidecar holds all ghost data | Core only | Same as Separated; extra sync/copy steps are explicit, slower path |
| **Sparse / arbitrary** | Any user-supplied `(peer, send_indices, recv_indices)` tuples; no grid concept | Yes on core only | `pfc::SparseHaloExchanger<T>` directly with a hand-built `std::vector<halo::RemoteHalo<T>>` (FEM / non-axis / multi-block patterns) |

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
| Device SparseVector MPI | `runtime/cuda/exchange_cuda.hpp`, `runtime/hip/exchange_hip.hpp` | `SparseVector<CudaTag\|HipTag>` overloads. **Non-blocking** `isend_data` / `irecv_data` require GPU-aware MPI (`OpenPFC_MPI_CUDA_AWARE` / `OpenPFC_MPI_HIP_AWARE` **and** runtime `MPIX_Query_cuda_support()` / `MPIX_Query_hip_support()`); otherwise they throw `std::runtime_error` (never a silent no-op with `MPI_REQUEST_NULL`). **Blocking** `send_data` / `receive_data` / `send` host-stage via `cudaMemcpy` / `hipMemcpy` when unaware (same runtime gate as `PaddedDeviceHaloExchanger`). |
| Face MPI types | `halo_mpi_types.hpp` | `create_face_types_6` for send/recv subarrays in a single `[nx,ny,nz]` buffer. |
| Padded face MPI types | `padded_halo_mpi_types.hpp` | `create_padded_face_types_6` — same idea but the outer extents are `(nx+2hw, ny+2hw, nz+2hw)` and recv subarrays target the dedicated halo ring rather than the outermost owned cells. |
| In-place driver | `halo_exchange.hpp` | `HaloExchanger<T>`: recv into core boundary slabs (traditional). |
| Padded brick driver | `padded_halo_exchange.hpp` | `PaddedHaloExchanger<T>`: in-place non-blocking face exchange on a `pfc::field::PaddedBrick<T>` so `u(i±hw, j, k)` legitimately reaches the ghost ring. Same `start_/finish_halo_exchange` API as `HaloExchanger`. **Face-only** — corners/edges are not filled. |
| Padded brick, **host buffer** (full 26-direction) | `full_padded_halo_exchange.hpp` | `pfc::communication::FullPaddedHaloExchanger<T>`: host twin of `FullPaddedDeviceHalo` — **3 widening passes** (X → Y-with-X-halos → Z-with-XY-halos) so every halo cell (6 faces + 12 edges + 8 corners) equals the periodic-equivalent neighbour value. Self-periodic axes (proc-grid extent 1) use host pack/unpack of widened slabs (never MPI-to-self). Default direction set is `Full3D()`. Verified by `tests/integration/scenarios/parallel_scaling/test_full_padded_halo_exchange.cpp` on 1/2/4 ranks. |
| Padded brick, **device buffer** (axis-aligned 6-face) | `runtime/cuda/padded_device_halo_exchange.hpp` | `pfc::cuda::PaddedDeviceHaloExchanger`: same MPI face derived types as `PaddedHaloExchanger<double>`, but the base pointer is a **CUDA device** allocation. Uses **GPU-aware MPI** when `OpenPFC_MPI_CUDA_AWARE` and `MPIX_Query_cuda_support()` succeed; otherwise **pack/unpack face slabs** (kernels in `src/openpfc/runtime/cuda/padded_halo_faces.cu`) + MPI on pinned host. Those kernels are **linked into `kobayashi_fd_cuda`** (not `libopenpfc_gpu_kernels`) so CUDA separable compilation device-links with the executable. Env **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`** forces the packed path. **Periodic faces whose neighbor rank equals the caller** (e.g. ±Z when the MPI process grid is one cell thick in Z) use **device pack/unpack** instead of GPU-aware **or packed MPI** to self — important when **local nz = 1**, because ±Z face slabs are **nx×ny** elements (~128 MiB per message at 4096²) and must not be staged through **`MPI_Irecv`/`Isend` to self**. **`kobayashi_fd_cuda`** with **one MPI rank** skips this in the timestep loop and applies **periodic halos on device** (`kobayashi_periodic_halos_xy_cuda` in `apps/kobayashi/`) instead of MPI + `cudaStreamSynchronize` / `cudaDeviceSynchronize` per exchange. **Fills only the 6 face halo strips — corners and edges are *not* populated**, so this exchanger is correct for axis-aligned stencils (5/7-point Laplacians, gradients along x/y/z) and **incorrect for any stencil that reads diagonal neighbours** such as the mixed second derivatives `u_xy` / `u_xz` / `u_yz`. Use `FullPaddedDeviceHalo` (below) for general second-order PDE kernels. |
| Padded brick, **device buffer** (full 26-direction) | `runtime/cuda/full_padded_device_halo.hpp` | `pfc::cuda::FullPaddedDeviceHalo`: corner- and edge-correct **multi-field** halo exchange built on the same MPI derived types and the same `padded_pack_face_kernel` as `PaddedDeviceHaloExchanger`, but in **3 widening passes** (X → Y-with-X-halos → Z-with-XY-halos) so every cell of the halo ring `[-hw, 0)` and `[n, n+hw)` on every axis is populated, including the 12 edge strips and 8 corner cubes. After the call, every padded cell equals the **periodic-equivalent global value** at that offset — verified bit-identically by `tests/integration/scenarios/parallel_scaling/test_full_padded_device_halo.cpp` on 1/2/4 ranks. Self-axis loops (proc-grid extent 1 along an axis) use device pack/unpack with widened slabs and the **correct** periodic direction (the +axis halo receives the rank's *first* `hw` owned cells, not its *last* `hw`); the self-pack in `PaddedDeviceHaloExchanger` does not implement this wrap correctly and remains in place only for backwards compatibility on small models that do not stress the self-axis. Cost: **3 syncs per call** vs. 2 for the single-pass exchanger; total MPI message count is unchanged. Env **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`** falls back to a per-field axis-aligned `PaddedDeviceHaloExchanger` (corners *not* filled), useful as a sanity check against the same env switch in the older exchanger. **Foundation layer for the planned PDE-kernel codegen path** (`f(t, u, ∂_i u, ∂_i ∂_j u)`): see [`docs/development/refactoring_roadmap.md`](../development/refactoring_roadmap.md) once the rest of the stack lands. |
| Padded brick storage | `field/padded_brick.hpp` | `pfc::field::PaddedBrick<T>` — single contiguous owned + halo-ring buffer; `T &operator()(int i, int j, int k)` valid for any `i, j, k in [-hw, n+hw)`. |
| Brick iteration | `field/brick_iteration.hpp` | `for_each_owned`, `for_each_inner(brick, r, fn)`, `for_each_border(brick, r, fn)` yielding `(int i, int j, int k)` triples — drives the laboratory-style FD loop. OMP-parallel `_omp` variants over `(k, j)`. |
| Sparse driver | `sparse_halo_exchange.hpp` | `pfc::SparseHaloExchanger<T>`: fully sparse, grid-agnostic non-blocking exchanger. Accepts any `std::vector<halo::RemoteHalo<T>>` (peer + send/recv indices + tags). For structured exchanges use `pfc::halo::make_structured_halos<T>(decomp, rank, hw, dirs = Axes3D())` and `pfc::halo::copy_to_face_layout` to refill the array-of-six face buffers consumed by the templated periodic-separated FD Laplacians. |
| Persistent halos | `halo_persistent.hpp` | `PersistentHaloExchanger` for in-place six-face path. `start_exchange` / `wait_exchange` fail closed on `MPI_Startall` / `MPI_Waitall` errors via `pfc::mpi::throw_on_mpi_error`. |
| FD primitives | `field/fd_apply.hpp`, `field/fd_stencils.hpp` | `apply_d1_along<Axis, Stencil>`, `apply_d2_along<Axis, Stencil>`, `apply_tensor_d<Mx, My, Mz, ...>`, `EvenCentralD1<Order>` (orders 2..14), `EvenCentralD2<Order>` (orders 2..20) |
| Generic stencils (custom: Sobel, CNN, anisotropic) | `field/stencil_apply.hpp` | `apply_1d_along<Axis>(coeffs, half_width, ...)`, `apply_separable(cx, Hx, cy, Hy, cz, Hz, ...)`, `apply_dense<Nz, Ny, Nx>(weights, ...)` — runtime-coefficient primitives that accept arbitrary kernels (laboratory layer for custom evaluators built atop the same halo plumbing). |
| FD bricks | `field/finite_difference.hpp` | `laplacian_interior<Order>`, `laplacian_periodic_separated<Order>`, `laplacian2d_xy_interior<Order>`, `laplacian2d_xy_periodic_separated<Order>`, runtime-order `laplacian_interior(int order, ...)` |
| FD point evaluator (CPU) | `field/fd_gradient.hpp` | `pfc::field::FdGradient<G>` + factory `pfc::field::create<G>(LocalField, order)` — populates `g.value / g.x / g.y / g.z / g.xx / g.yy / g.zz` according to `pfc::field::has_*<G>`. Host 26-fill exists via `pfc::communication::FullPaddedHaloExchanger`; mixed seconds (`xy / xz / yz`) remain `static_assert`-rejected pending a **follow-up enablement** of those members after Catch2 proves corners (wiring, not missing exchanger). |
| FD point evaluator (GPU) | `runtime/cuda/fd_gradient_device.hpp` | `pfc::cuda::FdGradientDevice<G>` — host-side wrapper around a trivially-copyable `FdGradientDevicePOD` (padded device pointer + pre-scaled weight rows). `__device__ G evaluate_fd_grad<G>(POD, i, j, k)` does the per-point work. Mirrors the CPU twin's pruning + diagnostic-throw behaviour. |
| GPU `for_each_interior_device` | `runtime/cuda/for_each_interior_device.hpp` | `pfc::sim::cuda::for_each_interior_device` — single-field (`double *du_padded`) and multi-field (`DevicePtrPack2`/`3`/`4`) drivers that launch a templated kernel calling `model.rhs(t, g)` for every owned cell of a padded device buffer. Multi-field launches take a `CompositeGradientDevicePOD` (or `CompositeGradientDevice` host wrapper) plus explicit catalog `PerFieldGrads...`; device scatter uses named-member `DeviceIncN` / `scatter_device` (no device `std::get` / `std::apply` / `std::forward_as_tuple`). The model's `rhs` must carry the portable `OPENPFC_HD` annotation from `kernel/data/host_device.hpp` so it is callable from `__device__`. |
| Examples | `examples/15_finite_difference_heat.cpp` | Separated halos + heat equation; core is FFT-safe. |

Design choice: Indices for the pack path are exchanged once at setup; only values move each step.

Setup-phase size envelope: `exchange::send` on CPU (`exchange.hpp`), CUDA (`exchange_cuda.hpp`), and HIP (`exchange_hip.hpp`) always posts exactly one `MPI_UNSIGNED_LONG_LONG` size word before any index/data messages — including value `0` when the `SparseVector` is empty — matching shared `exchange::receive`, which always `MPI_Recv`s that single size word. A prior GPU empty path that used a zero-count `MPI_Send(nullptr, 0, …)` was incorrect and could hang or skew tags against CPU peers.

Index semantics (in-place):

- Send: Local linear indices of the boundary layer to send.
- Recv: Local linear indices where received data is written—inside the same `nx×ny×nz` array (boundary slabs).

Separated recv: No scatter into core; MPI receives into contiguous face buffers whose element order matches the same face traversal as `create_recv_halo` (and MPI subarray layout).

### 5.2 Tests

- Unit: `tests/unit/kernel/decomposition/test_halo_pattern.cpp`, `test_halo_face_layout.cpp`, `test_padded_halo_mpi_types.cpp` (per-face subarray geometry, MPI_Sendrecv on `MPI_COMM_SELF`).
- Unit: `tests/unit/kernel/field/test_padded_brick.cpp`, `test_brick_iteration.cpp` — padded indexing, owned/inner/border iteration counts.
- Integration: `test_halo_patterns.cpp`, `test_halo_exchange_driver.cpp`, `test_padded_halo_exchange.cpp` (1/2/4-rank periodic wrap, axis-aligned 6-face), `test_full_padded_halo_exchange.cpp` (host full 26-direction fill on 1/2/4 ranks, edges + corners), `test_fd_heat_mpi.cpp` — MPI parity (in-place vs separated where applicable).
- Integration (CUDA): `test_full_padded_device_halo.cpp` — bit-identical full 26-direction fill on 1, 2 (`2x1x1`), and 4 (`2x2x1`) ranks; every padded cell, including all 12 edges and 8 corners, is checked against `hash(periodic_global_coord)`. Single-rank `hw=2` case also covered.
- Integration (CUDA): `test_fd_gradient_device.cu` — end-to-end `for_each_interior_device(model, eval.pod(), du, t, ...)` against an analytic polynomial RHS (`u = a + b x + c x² + d y + e y² + f z + g z²`, `rhs = value + ∂_i u + ∂_i^2 u`). Confirms every owned cell matches the closed form to within `1e-9` and that halo cells of `du` stay untouched. Constructor diagnostic-throw test for unsupported D1 orders also covered.
- Integration (CUDA multi-field): `test_multi_field_device.cu` / `test_composite_gradient_pod_size.cu` — `DevicePtrPackN` + `CompositeGradientDevice` path for 2-field (wave2d-style catalog `UGrads`/`VGrads`, kobayashi-style `phi`/`tempr`) and 3-field synthetic kernels; GPU owned-cell increments match CPU `for_each_interior` within `1e-12`; `scatter_device` and `evaluate_fd_grad_composite` covered; `sizeof(CompositeGradientDevicePOD) == 2088`.

### 5.3 Documentation references

- Architecture: `docs/architecture.md`
- Design history: `llm/user-stories/0009-implement-halo-exchange-layer.md`, `llm/IMPLEMENTATION_HALO_PATTERN.md`, `llm/IMPLEMENTATION_SPARSE_VECTOR.md`, `llm/design/finite_difference_gradient_design.md`

---

## 5.4 Direction sets and presets

Every face exchanger above accepts an explicit **`pfc::halo::HaloDirectionSet`** (in `include/openpfc/kernel/decomposition/halo_directions.hpp`) so callers can shrink the active direction list — most commonly to skip ±Z on a 2D slab problem. The set is a deduplicated, validated list of unit `Int3` vectors (each component in `{-1, 0, 1}`, never `{0,0,0}`); presets cover the canonical cases.

| Preset      | Size | Members                                      | Use for |
|-------------|------|----------------------------------------------|---------|
| `Axes2D()`  |   4  | `±X, ±Y`                                     | 2D slab problems (`nz == 1`) with axis-aligned stencils. |
| `Full2D()`  |   8  | axes + 4 XY corners                          | 2D problems with diagonal reads (`u_xy`, 9-point Laplacian). |
| `Axes3D()`  |   6  | `±X, ±Y, ±Z`                                 | Default 3D — historical 6-face exchange (7-point Laplacian). |
| `Full3D()`  |  26  | axes + 12 edges + 8 corners                  | 3D mixed second derivatives; default for `FullPaddedHaloExchanger` and `FullPaddedDeviceHalo`. |

Public ctor pattern, applied uniformly to every face exchanger:

```cpp
Exchanger(decomp, rank, hw, comm,
          pfc::halo::HaloDirectionSet dirs = presets::Axes3D(),
          int base_tag = 0,
          pfc::halo::HaloDirectionSelector per_rank = {});
```

If `per_rank` is provided, the exchanger calls `per_rank(rank)` for its own rank and uses that result; otherwise it uses the uniform `dirs`. Exchangers that historically defaulted to a different connectivity (`FullPaddedHaloExchanger` / `FullPaddedDeviceHalo` ⇒ `Full3D()`) keep their old default after the change. Custom sets that mix faces with diagonals are tolerated by face-only exchangers (the diagonals are silently ignored — they cannot be expressed as one of the 6 canonical face slots); for full corner/edge fill use `pfc::communication::FullPaddedHaloExchanger` (host) or `pfc::cuda::FullPaddedDeviceHalo` (device) and feed it `Full3D()` (or a smaller preset to subset its widening passes).

`HaloExchanger` and `PaddedHaloExchanger` use the zero-copy MPI subarray fast path **iff** every face slot is in the active set; subsetting via direction set falls back to the gather/scatter pack path. `PaddedDeviceHaloExchanger` and `BatchedPaddedDeviceHalo` skip excluded slots in both their GPU-aware and packed-fallback branches; same-rank periodic faces *inside* the active set still use device pack/unpack (no MPI-to-self) — this is the lever that turns off the `nx*ny*hw` ±Z self transfers when `local nz == 1`.

`FullPaddedDeviceHalo` and `FullPaddedHaloExchanger` share the same `axis_active` / `axis_widen` interpretation of diagonal directions:

- **Pass `a` is enabled** iff at least one of `±a` is in the set.
- **Pass `a` widens** the slab cross-section over previously-filled axes iff the set contains a direction with `d[a] != 0` and `d[b] != 0` for some `b < a`. With `Full3D()` this is exactly the original 3-pass widening; with `Axes3D()` every pass uses narrow slabs (face-only); with `Axes2D()` the Z pass is skipped entirely.

For 2D slab apps (`apps/kobayashi/src/cuda/kobayashi_fd_cuda.cpp` is the canonical example), pass `presets::Axes2D()` to both `PaddedDeviceHaloExchanger` and `BatchedPaddedDeviceHalo` to remove all ±Z communication / self-pack work without changing the rest of the driver.

> **Inter-rank consistency:** CPU exchangers that accept a `HaloDirectionSet` /
> `HaloDirectionSelector` (`HaloExchanger`, `PaddedHaloExchanger`,
> `PersistentHaloExchanger`) call
> `pfc::halo::validate_neighbour_direction_agreement` immediately after
> `resolve_direction_set`. That helper `MPI_Allgather`s a canonical encoding of
> each rank's resolved set and checks **paired-boundary** agreement: every
> active direction `d` toward neighbour `n` requires `-d` in `n`'s set (global
> set identity is not required). A mismatch throws `std::runtime_error` at
> construction — before any exchange posts — rather than hanging on an unmatched
> Waitall. **Follow-up:** CUDA/HIP `PaddedDeviceHaloExchanger` /
> `FullPaddedDeviceHalo` (and app-local `BatchedPaddedDeviceHalo`) dirs/selector
> constructors do not yet call the same helper.

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
- Face-only vs full-26 on the **CPU** padded-brick path: `PaddedHaloExchanger` remains the axis-aligned 6-face default. Full corner/edge fill is available via `pfc::communication::FullPaddedHaloExchanger` (3-pass widening, host twin of `FullPaddedDeviceHalo`). Wiring mixed seconds into `FDGradient` / `apply_tensor_d` is still a follow-up after corners are proven.
- Self-axis periodic wrap in `PaddedDeviceHaloExchanger` writes the rank's *last* `hw` owned cells into the +axis halo (rather than the *first*). This is masked on small/short runs but is incorrect for periodic boundary conditions when the dendrite or front reaches the boundary on a single-rank-per-axis grid; `FullPaddedDeviceHalo` implements the correct wrap.
- Orchestration: Optional thin `exchange_if_needed(HaloPolicy, …)` can be added when multiple call sites need it; policies are documented first.

---

## 9. Next steps

- GPU / persistent variants for `pfc::SparseHaloExchanger`.
- `make_fem_halos(...)` helper on top of `pfc::SparseHaloExchanger` for unstructured / FEM neighbour discovery.
- Optional `DataBlock` / gradient abstractions per `llm/design/finite_difference_gradient_design.md`.
- Derived types or tuning for pack-heavy decompositions.

---

## 10. Working examples

- **Separated layout** (FFT-safe core + face buffers):
  `examples/15_finite_difference_heat.cpp` — `mpirun -np P ./15_finite_difference_heat` runs the heat equation with `pfc::SparseHaloExchanger<double>` (configured by `pfc::halo::make_structured_halos<double>(...)`, default `Axes3D()`) and `laplacian_periodic_separated<2>`. After the exchange the example calls `pfc::halo::copy_to_face_layout` to refill the array-of-six face buffers the Laplacian consumes. The core field can be passed to `fft.forward` / `backward` on the same decomposition (comment in source).
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
