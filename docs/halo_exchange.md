# Halo Exchange: Status and Roadmap

This document consolidates the current state of halo (ghost cell) exchange in OpenPFC, the design decisions in place, and the path to a **working multi-rank finite difference example** (e.g. heat equation). It is the single place under `docs/` for halo exchange strategy and status.

---

## 1. Purpose

Halo exchange enables **real-space stencil operations** (e.g. finite difference Laplacian) on distributed domains by synchronizing ghost cells between neighboring ranks. Without it, real-space operations that need neighbor data would have to go through global FFTs. With it we can support:

- Finite difference derivatives and diffusion
- Hybrid spectral + real-space methods
- Future adaptive or local operations

**Target outcome:** A runnable example that solves a simple problem (e.g. heat equation) with **multiple MPI ranks** using **finite differences** and the existing halo building blocks.

---

## 2. Current Status

### 2.1 What Exists (Implemented)

| Component | Location | Description |
|-----------|----------|-------------|
| **Neighbor discovery** | `include/openpfc/kernel/decomposition/decomposition_neighbors.hpp` | `get_neighbor_rank(decomp, rank, direction)`, `find_face_neighbors(decomp, rank)` (6), `find_all_neighbors(decomp, rank)` (26). Periodic boundaries only. |
| **Halo patterns** | `include/openpfc/kernel/decomposition/halo_pattern.hpp` | `create_send_halo(decomp, rank, direction, halo_width)`, `create_recv_halo(...)`, `create_halo_patterns(decomp, rank, Connectivity::Faces \| All, halo_width)`. Return `SparseVector<BackendTag, size_t>` of **local** indices. |
| **SparseVector** | `include/openpfc/kernel/decomposition/sparse_vector.hpp` | Index + value buffers; indices sorted for contiguous access. Used as halo “pattern” (indices) and value buffer. |
| **Gather / scatter** | `include/openpfc/kernel/decomposition/sparse_vector_ops.hpp` | `gather(sparse, source)` and `scatter(sparse, dest)`. **CPU only**; CUDA paths throw. |
| **MPI exchange** | `include/openpfc/kernel/decomposition/exchange.hpp` | **Setup:** `send()` / `receive()` (indices + data). **Runtime:** `send_data()` / `receive_data()` (data only); **non-blocking:** `isend_data()`, `irecv_data()`, `wait_all()`. **Zero-copy faces:** `sendrecv_face()`, `isend_face()`, `irecv_face()` with MPI derived types. CPU and CUDA (CUDA copies to host for MPI unless GPU-aware). |
| **Face MPI types** | `include/openpfc/kernel/decomposition/halo_mpi_types.hpp` | `create_face_type()`, `create_face_types_6()` for zero-copy send/recv; RAII `MPI_Type_guard`. |
| **Halo driver** | `include/openpfc/kernel/decomposition/halo_exchange.hpp` | `HaloExchanger<T>`: builds patterns and (when 6 faces) face types; `exchange_halos(field_ptr, size)` uses zero-copy for 6 faces, pack path fallback otherwise. Non-blocking (Irecv then Isend then Waitall). |

**Design choice:** Indices are exchanged **once** at setup; only **values** are transferred each step.

**Index semantics:**

- **Send halo:** Local linear indices of the boundary layer to send (e.g. for +X: rightmost `halo_width` slices in local space).
- **Recv halo:** Local linear indices where received data is written (e.g. for +X: leftmost `halo_width` slices). Both send and recv indices lie in `[0, local_total)` for the current implementation (boundary layers of the same local domain).

### 2.2 Tests

- **Unit:** `tests/unit/kernel/decomposition/test_halo_pattern.cpp` — create send/recv halos, sizes match, gather from local field.
- **Integration:** `tests/integration/scenarios/parallel_scaling/test_halo_patterns.cpp` — MPI; checks halo send/recv sizes vs expected face areas.

### 2.3 Documentation References

- **Architecture:** `docs/architecture.md` — lists kernel/decomposition (exchange, halo_pattern).
- **Design and history:**  
  - `llm/user-stories/0009-implement-halo-exchange-layer.md` — full user story, constraints (zero-copy, MPI datatypes, periodic BCs).  
  - `llm/IMPLEMENTATION_HALO_PATTERN.md` — implementation summary for halo patterns.  
  - `llm/IMPLEMENTATION_SPARSE_VECTOR.md` — SparseVector and two-phase exchange.  
  - `llm/design/finite_difference_gradient_design.md` — FD design, DataBlock with halos, `HaloExchangePattern` (not yet in kernel).

---

## 3. Architecture (Current Design)

- **Decomposition** defines the global domain and per-rank local boxes (from `decomposition::get_subworld`).
- **Neighbors** are found from the decomposition grid; periodic BCs imply every rank has 6 (faces) or 26 (all) neighbors.
- **Halo patterns** turn (decomp, rank, direction, halo_width) into two index sets: send (indices to read from local field) and recv (indices to write into local field).
- **SparseVector** holds these indices and, at runtime, the values at those indices.
- **Exchange** does point-to-point MPI: setup sends indices + data once; runtime sends only data.
- **Gather** fills SparseVector value buffer from the dense local field; **scatter** writes SparseVector values back into the local field.

End-to-end flow (conceptually):

1. Build patterns: `create_halo_patterns(decomp, rank, Faces, halo_width)`.
2. For each neighbor: create value SparseVectors from pattern indices; run `send()`/`receive()` once (setup).
3. Each step: for each neighbor, `gather(send_halo_values, local_field)` → `send_data()` / `receive_data()` → `scatter(recv_halo_values, local_field)`.

There is **no** high-level “halo exchanger” or “field with halos” type in the kernel yet; the design in `llm/design/finite_difference_gradient_design.md` (DataBlock, HaloExchangePattern) is future work.

---

## 4. State of the Art and Possible Improvements

### 4.1 How the current approach compares

The current implementation is **correct** and sufficient for a first working multi-rank FD example. The comparison below is for **performance and scalability** and for alignment with the stricter requirements in user story 0009 (zero-copy, no pack in hot loop).

| Aspect | Current OpenPFC | Common best practice / state of the art |
|--------|-----------------|----------------------------------------|
| **Index vs data** | Indices exchanged once, only data each step. | Same idea is standard (pattern setup once, then data-only). |
| **Data path** | **Pack**: gather from field into contiguous buffer → MPI_Send/Recv → scatter back. | **Zero-copy**: MPI derived types (`MPI_Type_create_subarray`, `MPI_Type_vector`) describe faces; MPI reads/writes directly from field. User story 0009 explicitly requires this to avoid pack overhead in the hot loop. |
| **Blocking** | Blocking `MPI_Send` / `MPI_Recv` per neighbor. | **Non-blocking**: Post all `MPI_Irecv`, then all `MPI_Isend`, then `MPI_Waitall`. Avoids deadlock and enables overlap. |
| **Overlap** | None: full exchange (gather → send/recv → scatter) then compute. | **Overlap**: Post halo exchange, compute on interior (no halo), wait for halos, then compute on boundary. Hides latency when interior is large enough. |
| **GPU** | CUDA path copies to host, then MPI (not GPU-aware). | **GPU-aware MPI**: Pass device pointers to MPI; library uses GPUDirect RDMA where available. |
| **Requests** | New send/recv each call. | **Persistent requests**: `MPI_Send_init` / `MPI_Recv_init` once, `MPI_Start`/`MPI_Startall` each step, `MPI_Waitall`. Less overhead per step. |

So: the **two-phase** (indices once, data every step) and **pattern-from-decomposition** design are sound. The main gaps versus “state of the art” and versus user story 0009 are: **pack in the hot path** (instead of zero-copy with derived types), **blocking** (no non-blocking or overlap), and **no GPU-aware MPI**.

### 4.2 Zero-copy vs pack

- **Current (pack):** For each face we gather selected indices into a contiguous SparseVector buffer and send that. Flexible (arbitrary index sets, edges/corners), but every step we pay gather + scatter cost and extra memory traffic.
- **Zero-copy (desired):** For **face** halos, each face is a contiguous or strided region in the 3D array. With row-major `[nx, ny, nz]`:
  - Faces normal to Z are contiguous (full `nx*ny` per layer).
  - Faces normal to X or Y are strided; `MPI_Type_create_subarray` or `MPI_Type_vector` can describe them so MPI sends/receives directly from the field, with no separate pack buffer or gather/scatter in the hot path.
- **Trade-off:** Zero-copy aligns with 0009 and reduces hot-path cost; pack is already implemented and is sufficient for a first working example and for non-face connectivity (edges/corners) unless we add derived types for those too.

### 4.3 Recommended improvement directions (after a working example)

1. **Non-blocking exchange and safe ordering**  
   Post all receives, then all sends, then wait. Implement in the halo driver so all ranks follow the same order; no persistent requests needed initially.

2. **MPI derived types for face halos (zero-copy)**  
   For the 6 faces, create `MPI_Datatype` (e.g. via `MPI_Type_create_subarray`) once per face per rank; use them in Send/Recv (or Isend/Irecv) with the field base pointer. Removes gather/scatter for face data in the hot path. Requires a clear convention for field layout (e.g. with or without halo padding).

3. **GPU-aware MPI**  
   When running on GPU, pass device pointers to MPI and use CUDA-aware MPI build so that no host staging copy is used. Depends on environment (GPUDirect, etc.).

4. **Communication/computation overlap**  
   After non-blocking is in place: post halo exchange, compute on interior points that do not depend on halos, wait for halo completion, then compute boundary. Most beneficial when interior dominates.

5. **Persistent requests**  
   Optional: create persistent send/recv requests at setup and reuse them each step to reduce MPI overhead.

---

## 5. Gaps and Limitations

- **Orchestration:** `HaloExchanger` provides single-call `exchange_halos()` for 6 face neighbors with non-blocking MPI and (when 6 faces) zero-copy via MPI derived types. Pack path used when fewer than 6 directions.
- **Overlap:** `HaloExchanger` exposes `start_halo_exchange(field_ptr, size)` and `finish_halo_exchange()` so callers can post exchange, compute interior, then wait and scatter (see §4.3). `exchange_halos()` is equivalent to start + finish.
- **Gather/scatter on GPU:** CUDA gather and scatter for `double` are implemented in `sparse_vector_ops.cu`; other types throw. Pack path can run on device when CUDA is enabled.
- **GPU-aware MPI:** When `OpenPFC_MPI_CUDA_AWARE` is defined (CMake option `-DOpenPFC_MPI_CUDA_AWARE=ON` with CUDA enabled), exchange uses device pointers in MPI_Send/Recv and Isend/Irecv; otherwise CUDA path copies to host first.
- **Recv layout:** Recv halo indices assume the same local layout as the send side (boundary layers in the same array). No explicit halo-padded layout or offset documented in the API.
- **No FD or field integration:** No Laplacian/stencil or field type that uses halos; no example application (heat equation example still to do).

---

## 6. Next Steps (Toward a Working Example)

Prioritized to reach a **multi-rank finite difference heat equation** (or similar) example:

1. **Halo exchange driver (CPU)**  
   - Implement a small helper or class that, given `Decomposition`, rank, and `halo_width`:  
     - Builds all face halo patterns (6 directions).  
     - Creates send/recv value SparseVectors per direction.  
     - Runs setup once (exchange indices + initial data).  
     - Exposes one call e.g. `exchange_halos(local_field)` that: gather → send_data/receive_data (in safe order, e.g. even/odd or use non-blocking) → scatter.  
   - Use only **blocking** point-to-point in a deadlock-free order (e.g. send to +X, recv from -X, etc., or paired send/recv per neighbor).  
   - Keeps implementation in kernel or in the example first; no need for DataBlock yet.

2. **Heat equation (or simple diffusion) example**  
   - Small executable or example under `examples/` or `apps/`:  
     - Domain and decomposition (e.g. 2D or 3D, 2×2 or 2×2×2 ranks).  
     - Local field per rank (dense, matching decomposition).  
     - 3-point or 5-point Laplacian in real space (no FFT).  
     - Each step: (1) halo exchange using the driver above, (2) interior update using the stencil.  
   - Verify: constant or known initial condition, compare with single-rank or known solution (e.g. decay rate).

3. **Tests**  
   - Integration test: run the example with 2+ ranks; check that a known solution (e.g. constant field, or symmetric decay) matches expectations.

4. **Optional follow-ups**  
   - Document recv index semantics and, if needed, halo-padded layout in `halo_pattern.hpp`.  
   - Non-blocking exchange and communication/computation overlap (§4.3).  
   - CUDA gather/scatter and GPU-aware exchange (if needed for GPU FD).  
   - Later: zero-copy face exchange via MPI derived types (§4.2, §4.3).

---

## 7. Target: Working Example (Heat Equation)

**Goal:** One runnable example that:

- Uses **multiple MPI ranks** (e.g. `mpirun -np 4`).
- Uses **finite difference** Laplacian (no FFT).
- Solves a simple PDE (e.g. ∂u/∂t = α ∇²u) for a few steps.
- Relies on the **existing** building blocks: `decomposition`, `halo_pattern`, `exchange`, `gather`/`scatter`.

**Suggested steps inside the example:**

1. Create world and decomposition (e.g. 64³ or 32³, 2×2×1 or 2×2×2).
2. Get local box and allocate 1 double field per rank (local size).
3. Initialize field (e.g. 1.0 everywhere, or a simple bump).
4. Build halo patterns (faces, halo_width = 1 for 3-point stencil).
5. Setup: create send/recv value SparseVectors; one full exchange (indices + data).
6. Time loop:  
   (a) Halo exchange (gather → send_data/receive_data → scatter).  
   (b) Apply FD Laplacian on interior points (using updated halos).  
   (c) Update u += dt * alpha * laplacian(u).
7. Optional: reduce max difference vs single-rank or analytical solution.

This gives a clear “where we are” and “what we do next” and validates the halo stack end-to-end.

---

## 8. References

**Internal**

| Document | Content |
|----------|---------|
| `docs/architecture.md` | Kernel layout; decomposition, exchange, halo_pattern listed. |
| `llm/user-stories/0009-implement-halo-exchange-layer.md` | Full halo layer user story, constraints, desired API (zero-copy, MPI datatypes). |
| `llm/IMPLEMENTATION_HALO_PATTERN.md` | Halo pattern implementation summary and usage. |
| `llm/IMPLEMENTATION_SPARSE_VECTOR.md` | SparseVector and two-phase exchange. |
| `llm/design/finite_difference_gradient_design.md` | FD design, DataBlock, HaloExchangePattern, gradient abstraction. |

**External (state of the art / best practices)**

| Topic | Reference |
|-------|-----------|
| Non-blocking halo exchange | Post Irecv first, then Isend, then Waitall to avoid deadlock and allow overlap. |
| GPU-aware MPI | Use device pointers in MPI when library is CUDA-aware; avoid host staging. |
| MPI derived types for halos | `MPI_Type_create_subarray`, `MPI_Type_vector` for zero-copy face send/recv. |
| Kokkos + MPI halo | [Kokkos MPI Halo Exchange](https://kokkos.org/kokkos-core-wiki/usecases/MPI-Halo-Exchange.html). |
| Halo exchange libraries | e.g. [Tausch](https://github.com/luspi/tausch) (generic C++ halo exchange). |

---

*Last updated: 2025-03. Single source of truth for halo exchange status under `docs/`.*
