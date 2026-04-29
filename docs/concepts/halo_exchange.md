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

Recommended for FFT + FD coexistence: keep a core buffer of size exactly the subdomain (`nx×ny×nz`) for spectral work, and store received ghost data in separate face buffers. Exchange sends from faces of the core (MPI derived types) and receives into contiguous slabs. `SeparatedFaceHaloExchanger` and `field::fd::laplacian_7point_interior_separated` implement this path.

Minimal hybrid timestep (conceptual):

1. Spectral substep: `fft.forward` / `backward` on core only.
2. Before FD: `SeparatedFaceHaloExchanger::exchange_halos(core, halos)` (or in-place path if FFT is not used on that field).
3. FD: `laplacian_7point_interior_separated(core, halos, lap, …)`.
4. Update core (and/or `lap`) as required by the scheme.

---

## 3. Halo policies

Policies describe where ghost data lives and what is safe for FFT. They are documented here; see `include/openpfc/kernel/decomposition/halo_policy.hpp` for the `enum class HaloPolicy` used in API/docs cross-references.

| Policy | Storage | FFT on same buffer | FD / stencils |
|--------|---------|-------------------|---------------|
| None | Core `nx×ny×nz` only | Yes | N/A |
| InPlace | One array; ghosts in boundary slabs of that array | No (multi-rank) after halo fill | `HaloExchanger` + `laplacian_7point_interior` |
| Separated | Core + six face halo buffers | Yes on core only | `SeparatedFaceHaloExchanger` + `laplacian_7point_interior_separated` |
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
| In-place driver | `halo_exchange.hpp` | `HaloExchanger<T>`: recv into core boundary slabs (traditional). |
| Separated driver | `separated_halo_exchange.hpp` | `SeparatedFaceHaloExchanger<T>`: send from core, recv into separate face buffers. |
| Persistent halos | `halo_persistent.hpp` | `PersistentHaloExchanger` for in-place six-face path. |
| FD (in-place) | `field/finite_difference.hpp` | `laplacian_7point_interior` |
| FD (separated) | `field/finite_difference.hpp` | `laplacian_7point_interior_separated` |
| Examples | `examples/15_finite_difference_heat.cpp` | Separated halos + heat equation; core is FFT-safe. |

Design choice: Indices for the pack path are exchanged once at setup; only values move each step.

Index semantics (in-place):

- Send: Local linear indices of the boundary layer to send.
- Recv: Local linear indices where received data is written—inside the same `nx×ny×nz` array (boundary slabs).

Separated recv: No scatter into core; MPI receives into contiguous face buffers whose element order matches the same face traversal as `create_recv_halo` (and MPI subarray layout).

### 5.2 Tests

- Unit: `tests/unit/kernel/decomposition/test_halo_pattern.cpp`, `test_halo_face_layout.cpp` — pattern sizes vs face layout counts.
- Integration: `test_halo_patterns.cpp`, `test_halo_exchange_driver.cpp`, `test_fd_heat_mpi.cpp` — MPI parity (in-place vs separated where applicable).

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
- Wider stencils / edges / corners: may need extra buffers beyond six faces.
- Orchestration: Optional thin `exchange_if_needed(HaloPolicy, …)` can be added when multiple call sites need it; policies are documented first.

---

## 9. Next steps

- GPU / persistent variants for `SeparatedFaceHaloExchanger`.
- Optional `DataBlock` / gradient abstractions per `llm/design/finite_difference_gradient_design.md`.
- Derived types or tuning for pack-heavy decompositions.

---

## 10. Working example

`examples/15_finite_difference_heat.cpp`: `mpirun -np P ./15_finite_difference_heat` — separated face halos, `SeparatedFaceHaloExchanger`, `laplacian_7point_interior_separated`. The core field can be passed to `fft.forward` / `backward` on the same decomposition (comment in source).

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
