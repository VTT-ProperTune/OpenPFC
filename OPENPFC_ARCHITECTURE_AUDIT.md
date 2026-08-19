<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# OpenPFC Architecture Audit

**Scope:** full repository at commit `e86e65ce` (v0.1.4 + ~7 months of unreleased work).
**Method:** independent senior C++/HPC architecture review. Every claim below was verified against the implementation, not the documentation. Eight parallel subsystem deep-dives (core data model; execution/memory; MPI/decomposition/halo; spectral/FD; model/simulator/steppers; applications; frontend/config/I/O; build/tests/CI/docs) plus direct verification of all critical findings.
**Convention:** findings are tagged **[C]** immediate correctness defect, **[AB]** architectural blocker, **[SD]** significant technical debt, **[LC]** local cleanup — and given a verdict: *retain / repair / redesign / replace / remove*.

> **This is a historical snapshot as of commit `e86e65ce`.** It is not updated as the refactor progresses — `OPENPFC_REFACTORING_EXECUTION_PLAN.md` tracks current milestone-by-milestone status (see its "Current status" section), and `docs/development/0.2_migration_map.md` (in-repo) tracks which 0.1 types have been replaced. As of the last cross-check (2026-08-01, `master` @ `fa35197c`): Pre-M0 (all §4/§11 defects below) is fixed and released as `v0.1.5`; M0 (ADRs, layering enforcement) is complete; the "ten field-container types" / "six index-box types" count in §1/§13.2 has shrunk on the box/domain axis — `Box3D`, `World`'s templated `CoordinateSystem`, and the dead `world_types` strong-type layer are all deleted, with `Box3i`/`Domain` now canonical (M1, nearly done) — but on the field-container axis the count has *not* yet shrunk: `LocalField`, `PaddedBrick`, `DiscreteField`, and `Array` (§2, §4.10, §13.3) all still exist alongside the new `pfc::data::Field<T, MemorySpace>` (M2, roughly a third to half done). Everything in §5 (AB-2 through AB-6) and §13.4 onward remains as described.

---

## 1. Executive summary

OpenPFC is in much better shape than a typical "first substantial C++ project": the test suite is large and scientifically meaningful (1030 test cases, real convergence-order and CPU-vs-GPU parity pins), the documentation unusually describes the code that exists rather than an aspiration, MPI communication code is defensively written and recently hardened, and several individual components — the FD stencil tables (orders 2–20), the gradient-evaluator concept, the collective MPI-IO writer, the `Time` transaction API — are genuinely well designed.

The architecture as a whole, however, is **not yet a sound foundation for the 0.2 goals**, for one dominant reason and several supporting ones:

**The dominant problem is unfinished migration, not wrong design.** Three architecture generations coexist with no bridge between them:

- **Gen‑1** (virtual `Model::step` + `Simulator` + `ui::App`, string-keyed non-owning field registry) carries *all* production physics (tungsten, aluminumNew) and the entire JSON frontend;
- **Gen‑2** (World/Decomposition/PaddedBrick/halo exchangers + hand-rolled loops) carries the FD demo apps;
- **Gen‑3** (concept-based steppers, solver contracts, gradient evaluators, adaptive-control components, checkpoint primitives) is the *intended* architecture — and has **zero production users**. It is validated only on toy physics, only on CPU.

Every cross-cutting concern — boundary conditions, checkpointing, save scheduling, multi-field coupling, device residency — is consequently solved zero, one, or two divergent times. The same disease repeats at every level: **10 field-container types, 6 index-box types, 9 halo-exchanger classes (4 with no production users), 5 step-attempt protocols, 2 `StageContext` types, 2 workspace types, 2 integrator-method enums, and 2 GPU portability stacks (one of which is a façade whose "CUDA" `parallel_for` is a serial host loop).**

The second structural problem is the **CUDA/HIP vendor fork**: ~85% of the device runtime is literal copy-paste (two 635-line halo exchangers differing by 4 lines), and the divergent 15% is feature skew, not vendor necessity — HIP lacks the multi-field driver, autotuning, the composite gradient, and the frontend backend enum, and the HIP apps use algorithmically inferior host-staged paths even where a device path exists in the library. On LUMI (Cray MPICH), GPU-aware MPI detection is compile-time dead, so halos silently host-stage.

There is also one **critical correctness defect verified during this audit**: the shipped `tungsten_cuda`/`tungsten_hip` binaries driven through `ui::App` never copy JSON initial conditions from the registered CPU mirror to the device field — the framework has no residency protocol, and the only correct GPU driver is a bespoke app-side loop.

**Verdict:** the project's Gen‑3 direction is correct and most of its parts deserve to survive. What 0.2 requires is not new invention but *consolidation*: pick one winner per concept, finish the Gen‑3 vertical slice on a production PFC model (GPU + ETD + multi-field + restart), single-source the GPU runtime, and delete the losers. The big-bang breaking refactor the project anticipates is justified and feasible; the regression surface needed to do it safely mostly exists, with a few identified gaps (multi-rank tungsten golden baseline, GPU CI).

---

## 2. Current architecture as actually implemented

**Layering.** `include/openpfc/` is split into `kernel/` (~31.7k LOC: data, decomposition, execution, fft, field, integrator, mpi, profiling, simulation, checkpoint), `runtime/` (~7.8k LOC: cpu/cuda/hip/common backend code), and `frontend/` (~5.2k LOC: ui wiring, io, utils), with a small compiled library (`src/`, 18 TUs, built as two OBJECT libs merged into one installed target `openpfc`). The kernel→frontend boundary is real and verified (no kernel header includes frontend); kernel→runtime and runtime→frontend are unenforced, and there are inversions *inside* kernel (§5).

**Domain model.** `World<CartesianTag>` = global index bounds + coordinate system (spacing, origin, periodicity flags). `Decomposition` (value type, HeFFTe `split_world`-based brick split) stores the global world plus *all* ranks' subworlds — reusing `World` as the subdomain-box type. Fields are, depending on generation: `std::vector<double>` aliases (`pfc::Field`), `DiscreteField<T,D>` over `Array<T,D>`, `pfc::field::Field<T>`, `LocalField<T>`, `PaddedBrick<T>`, or `DataBuffer<Tag,T>`/`View` on the execution side.

**Execution.** Two stacks: a Kokkos-facsimile (`View`, memory/execution spaces, `parallel_for`, `deep_copy`, `create_mirror`) that no application uses and whose device `parallel_for` is a serial host loop (`runtime/cuda/parallel_cuda.hpp:26–45`); and the real device layer — `DataBuffer<CudaTag/HipTag>`, hand-written kernels, device halo exchangers — duplicated per vendor.

**Spectral/FD.** FFT: `IFFT` interface → `FFT_Impl<BackendTag>` HeFFTe wrapper (shared CPU/GPU) → per-backend aliases and factories; `Box3i` firewalls HeFFTe types out of public headers. FD: stencil tables for even central D2 (orders 2–20) and D1 (2–14) in compile-time and runtime forms; `FDGradient<G>` / `SpectralGradient<G>` / `CompositeGradient` evaluators consumed by a duck-typed `for_each_interior`; halo width fail-closed against stencil half-width.

**Simulation.** Gen‑1: `Simulator` owns writer map + IC/BC `FieldModifier` lists, orchestrates `Model::step`; `ui::App<Model>` drives a 9-phase JSON pipeline hardwired to a `SpectralCpuStack` (concrete `CpuFft`). Gen‑3: `FdCpuStack`/`SpectralCpuStack` RAII bundles, explicit/embedded RK, IMEX, ETD1 steppers, `SolveFunction` solver contract with `SpectralDiagonalSolver`, `Time` attempt transactions, adaptive-control components, checkpoint capture/publish primitives — none of it wired end-to-end.

**Apps.** tungsten (Gen‑1; CPU/CUDA/HIP model triplets + bespoke GPU driver), aluminumNew (Gen‑1, CPU-only, tungsten's diverged ancestor), kobayashi (Gen‑2 FD; app-local 531-line batched device halo subsystem), heat3d/wave2d/allen_cahn (Gen‑2/3 validation ladders — effectively framework tests, and good ones).

---

## 3. Architectural strengths worth preserving

These should survive 0.2 largely intact — several are better than what peer research codes have:

1. **The gradient-evaluator concept** (`kernel/field/grad_concepts.hpp`, `fd_gradient.hpp`, `spectral_gradient.hpp`, `composite_gradient.hpp`, `kernel/simulation/for_each_interior.hpp`). Physics exposes `rhs(t, G)` over a self-declared gradients aggregate; FD and spectral evaluators are drop-in interchangeable, and `CompositeGradient` mixes methods per field within one model. This *is* the "select numerical method independently of physics" requirement, already working for the explicit path. **Retain and extend** (to device, and toward implicit solves).
2. **The FD stencil layer** (`kernel/field/fd_stencils.hpp`, `fd_apply.hpp`): orders 2–20, dual compile-time/runtime dispatch, correct coefficients, halo width fail-closed against order (`fd_cpu_stack.hpp:82–89`, `fd_gradient.hpp:362–367`). The "8th/10th-order stencils" requirement is already met. **Retain.**
3. **Halo-exchange engineering quality**: Irecv-before-Isend everywhere, RAII `MPI_Type_guard`/`MPI_File_guard`, fail-closed count checks, GPU-aware-MPI probing with pinned-staging fallback, self-neighbor short-circuits, construction-time neighbor-agreement validation. The *engineering* is strong even where the *strategy* (too many classes, no overlap in practice) is weak. **Retain the machinery, consolidate the classes.**
4. **`BinaryWriter`** (`frontend/io/binary_writer.hpp`): collective MPI-IO with subarray views, communicator-wide fail-closed validation via `MPI_Allreduce` *before* any collective (deadlock-proof), short-write detection. Exemplary. **Retain.**
5. **`Time`** (`kernel/simulation/time.hpp`): accepted-time vs candidate-dt separation, attempt/commit/reject transactions with save-point clipping, RAII rollback guard. The best-designed single class audited. **Retain.**
6. **The FFT wrapper's include hygiene**: one shared `FFT_Impl<BackendTag>` for CPU/CUDA/HIP, `Box3i` HeFFTe firewall, consistent scaling convention (forward unscaled, backward full-scale → exact round trip). **Retain** (fix its interface honesty, §4/§7).
7. **`Decomposition` as a value type** storing `World` by value after a real dangling-reference bug — the codebase learns from its mistakes. **Retain.**
8. **Frontend discipline**: strict tree-shaped includes, every wiring step callable piecemeal on a bare `Simulator`, catalog-based DI for field modifiers, verified frontend→kernel-only dependency direction. **Retain** the structure while generalizing what it wires (§10).
9. **Test and docs culture**: 1030 test cases; convergence-order pins with explicit ratio windows; CPU-vs-CUDA/HIP parity tests at 1e-10; docs-consistency CI scripts; ADRs; a refactoring roadmap that honestly tracks debt. **Retain.**
10. **`SparseVector` + sparse halo exchange** (`kernel/decomposition/sparse_vector*.hpp`): a legitimate index-set communication primitive, the declared path to FEM/unstructured coupling. **Retain** (fix its GPU story, §9).

---

## 4. Critical correctness and lifetime findings

Ranked by severity. Items 1–3 warrant fixes before, or independent of, any refactor.

1. **[C, critical] App-driven GPU tungsten integrates from a device buffer that never receives the initial condition.** Verified directly: `TungstenCUDA::allocate()` registers only the CPU mirror `m_psi_cpu` as `"psi"` (`apps/tungsten/include/tungsten/cuda/tungsten_model.hpp:203–208`), with `m_cpu_buffer_valid = false` and the device `psi` freshly allocated. `Simulator` applies JSON ICs to the registered CPU vector; the ADL `step()` hook (`apps/tungsten/include/tungsten/cuda/tungsten.hpp:24–32`) calls `m.step(t)` with no sync; and **no code in `include/` or `src/` calls `prepare_for_field_modifiers`/`sync_cpu_to_gpu`** — grep is empty. The shipped `tungsten_cuda`/`tungsten_hip` binaries (`apps/tungsten/src/cuda/tungsten.cpp:12`) therefore evolve an unseeded device field. The only correct GPU driver is the bespoke `run_tungsten_gpu_vtk.hpp`, which brackets every modifier application manually (lines 157–159, 169–171). This is the clearest single proof of framework inadequacy: the flagship frontend cannot correctly drive the flagship model on GPU. *Verdict: repair immediately (framework-level device-residency protocol, §13), and until then remove or guard the App-driven GPU entry points.*
2. **[C] `parallel_for<Cuda/HIP>` is a serial host loop over device memory.** `runtime/cuda/parallel_cuda.hpp:26–45` and the HIP twin implement the device policies as plain host `for` loops; `View<…,CudaSpace>::operator()` dereferences the raw device pointer. The advertised pattern (device View + device policy) segfaults or reads garbage, with no compile-time or runtime guard. Nothing in production uses it — it is a loaded trap, not an active bug. *Verdict: neutralize now (static_assert host-accessibility), then resolve via §6.*
3. **[C] Silent no-op in the runtime FD dispatcher.** `laplacian_interior(int order, …)` returns without writing anything for unsupported orders (`kernel/field/finite_difference.hpp:166`, `default: return;`) — inconsistent with the fail-closed philosophy everywhere else and capable of producing silently wrong science. *Verdict: repair (throw).*
4. **[C] Declared periodicity is silently discarded.** `world::create(GridSize, PhysicalOrigin, GridSpacing)` uses the periodicity argument to compute spacing, then constructs the coordinate system with default `{true,true,true}` (`world_helpers.hpp:114–137`, `src/…/world.cpp:45–57`). Nothing currently *reads* the flags (they have zero consumers), so this is latent — but it becomes live the moment non-periodic support lands. *Verdict: repair or delete the parameter.*
5. **[C] Subworld physical bounds are wrong.** `get_lower_bounds`/`get_upper_bounds` (`world_queries.hpp:581–604`) ignore `m_lower`, so any subworld with a nonzero offset reports the *global* origin as its lower bound — a direct consequence of reusing `World` as the subdomain-box type. *Verdict: repair; root cause addressed by the World/Box split (§13).*
6. **[C] Two coordinate→index conventions.** `world::to_indices` documents rounding but `csys::to_index` truncates (`csys.hpp:311`) while `DiscreteField::map_coordinates_to_indices` rounds (`discrete_field.hpp:427–428`). Interpolation through different paths picks different cells for the same point. *Verdict: repair (one convention).*
7. **[C] Unchecked MPI in the GPU packed-halo fallback.** `MPI_Irecv`/`MPI_Isend` posted without error checking (`runtime/cuda/padded_device_halo_exchange.hpp:521, 548` and HIP twin), the one spot missed by the recent error-handling sweep; also `decomposition_factory.cpp:23–24` (unchecked `MPI_Comm_rank/size`). *Verdict: repair.*
8. **[C, latent] Undefined-but-declared functions.** `utils::compute_upper_bounds`/`compute_spacing` are declared with no definition anywhere (`world_types.hpp:65–77`); the strong-type constructors calling them are guaranteed link errors. Dead layer, demoed by `examples/01_hello_world/world.cpp:144–148`. *Verdict: remove.*
9. **[C-risk] Implicit HeFFTe box-ordering invariant.** `get_neighbor_rank` (`decomposition_neighbors.hpp:65–95`) assumes rank *i* occupies grid coordinate `(i % gx, …)` x-fastest — i.e., that `heffte::split_world` enumerates boxes in exactly that order. Untested directly; a HeFFTe version change would silently corrupt halos. *Verdict: repair (assert the correspondence at `Decomposition` construction).*
10. **Lifetime hazards.** `pfc::field::Field<T>` holds `const World&` (`field.hpp:66`) — the exact dangling pattern `Decomposition` was already bitten by and fixed; steppers and stacks traffic in these. `Model` holds `const World&` and a registry of non-owning `std::vector&` references whose lifetimes are entirely the caller's problem (`model.hpp:109–114`, `model_field_registry.hpp:33–37`). `SpectralGradient` and `ScaledField` hold raw pointers with comment-enforced contracts. `spectral_cpu_stack.hpp:19–27` documents a required member-initialization order (and its comment is stale). *Verdict: repair by value-semantics for geometry PODs and an owning state registry (§13).*
11. **[C-policy] Inconsistent fail-closed destructors.** `mpi::environment` and `MPI_Type_guard` destructors are `noexcept(false)` and throw (terminate during unwinding), while `MPI_Type_guard`'s move-assignment silently ignores the same failure (`halo_mpi_types.hpp:42–66`). One policy, applied consistently. *Verdict: repair.*
12. **[C, minor] Duplicated save-point arithmetic.** Tungsten's GPU driver computes `save_interval = round(saveat/dt)` (`run_tungsten_gpu_vtk.hpp:166–192`) while `Time::do_save()` uses float-modulo with 1e-6 tolerance — divergent results when `dt` does not divide `saveat`. *Verdict: repair by having one scheduler.*

---

## 5. Major architectural blockers

Ranked by impact on the 0.2 goals.

**AB‑1: Three coexisting architecture generations, no bridge.** Gen‑1 carries all production physics and the entire frontend; Gen‑3 (the intended architecture) has zero production users and cannot yet express the production use case: `Etd1Stepper` accepts only real `std::vector<double>` state, while tungsten's ETD update operates on the *complex* spectral field (`tungsten_model.hpp:342–345`, `TODO(remove-tungsten-etd-workspace)… after #169` in four files). There is no `Model`→`StageFunction` adapter, `Simulator` cannot drive a concept stepper (the `step_with_physics` seam exists but is unused), and the frontend cannot select an integrator even though the parser exists (`from_json_integrator_method.hpp`, consumed only by tests). Consequences: every new feature must choose a generation, doubling or trebling work; the intended architecture accumulates seam headers (there are ~2,500 lines of adaptive-stepping scaffolding with no controller connecting them) without ever being proven on real physics. *Verdict: redesign is already underway — what is missing is a decision to finish it. Declare Gen‑3 the target, ship one complete vertical slice on a production PFC model, then delete Gen‑1 (§13, §15).*

**AB‑2: The CUDA/HIP fork.** Token-normalized diffs: `padded_device_halo_exchange.hpp` 4 changed lines of 635; `full_padded_device_halo.hpp` 1 include of 537; `sparse_vector_ops` 10/238; `databuffer` 18/376. ~2,300 lines maintained twice, and the divergent remainder is *feature skew*: HIP lacks the multi-field interior driver (CUDA 443 lines vs HIP 195), `CompositeGradientDevice`, autotune hooks, and `fft::Backend`/`backend_from_string` entries. Recent history shows every GPU fix landing per-vendor. Worse, the fork buys nothing: HIP apps bypass the library's own HIP device exchanger and host-stage instead (kobayashi HIP: blocking `hipMemcpy` around the *CPU* exchanger, `kobayashi_fd_hip.cpp:134–141`, while an identical 635-line device exchanger sits unused). And on Cray MPICH (LUMI — the actual AMD target) GPU-aware detection is gated on `defined(OPEN_MPI)`, so the device exchanger *always* host-stages regardless of `MPICH_GPU_SUPPORT_ENABLED` (`runtime/hip/padded_device_halo_exchange.hpp:57–60, 120–123`). "Efficient execution on both NVIDIA and AMD GPUs" is currently false in practice on AMD. *Verdict: replace with a single-source device layer (vendor shim or hipify-at-build); treat HIP as generated, never hand-edited.*

**AB‑3: No canonical field or box type.** Ten grid-data containers across four generations (§2 list; full table in the data-model analysis, `report1`), with at least four independent row-major linearizations that agree on "x fastest" by convention only; six index-box types. Any new capability — GPU mirrors, halo policies, I/O, checkpointing — must be implemented N times or arbitrarily privilege one container; this is precisely why device residency (§4.1) fell through the cracks. *Verdict: redesign around one owning field + one view + one box (§13).*

**AB‑4: The polymorphic FFT interface lies about GPU.** `fft::create_with_backend(..., Backend::CUDA)` returns a `unique_ptr<IFFT>` whose *every virtual method throws* — GPU transforms are non-virtual member templates on the concrete type (`fft_heffte_backend.hpp:146–170, 189–192, 246–249`). An interface that cannot be substituted by its implementations; polymorphic call sites (e.g. `SpectralGradient` holding `IFFT*`) compile against GPU and fail at runtime. Also: `Backend` enum omits HIP entirely. Consequence: GPU models construct their own FFTs inside the model while the `Model` base drags a mandatory host `CpuFft` the app driver literally names `dummy_fft` — every GPU run pays for a full host HeFFTe plan it never uses. *Verdict: redesign the FFT surface (§13); honest factories.*

**AB‑5: The physics interface conflates physics with numerical scheme, storage, and wiring.** `Model::step(t)` hard-codes the integrator inside the physics class; `initialize(dt)` bakes `dt` into precomputed propagators, structurally forbidding adaptive stepping for every legacy model; the class also owns FFT access, MPI rank logic, and field registration. Tungsten's `step()` performs FFTs, nonlinear evaluation, and the ETD combine in one method. Meanwhile ~80% of tungsten's ~5,000 non-test lines are backend dispatch, duplicated model shells, host/device mirroring and a bespoke driver — the app is compensating for the framework at scale, and aluminumNew is a diverged copy-paste ancestor of the same skeleton (same fields, same step shape, pre-cache ETD weights, hand-rolled JSON). *Verdict: replace `Model` with concept-based physics (RHS/operator evaluation) + owning simulation state; the framework absorbs the "spectral ETD pseudo-spectral model" skeleton so a model supplies only `physics_for_mode` + nonlinearity.*

**AB‑6: Stepper protocol fragmentation.** The seven leaves return `StepAttemptResult` / `MultiStepAttemptResult`; in-place `step()` is attempt+commit. `IntegratorBase` and the on-hold result DTO are deleted. `pfc::sim::StageContext` aliases `pfc::integrator::StageContext`; `StageWorkspace<T>` aliases `integrator::Workspace<T>`. `Time` stores `RKIntegratorMethod` (the `time.hpp` method enum is gone). `MultiStageFunction` / `MultiEtd1Stepper` are N-ary and accept `Scalar` (default `double`). All seven single-field leaves accept host `Field<double>` via `vec()` and `Scalar` / `Field<Scalar>` (IMEX included). MultiEuler / MultiExplicitRK / MultiImex / MultiEtd1 take `Scalar` packs. The vector path remains. Heterogeneous packs are still open. Host field overloads use `HostFieldState`. Device ETD combine is `apply_etd1_update_{cuda,hip}`. *Verdict: protocol port done; remaining M6 is heterogeneous packs / `state_concepts` on multi-field leaves.*

---

## 6. Backend and memory model analysis

**What is sound.** The memory-space *model* — backend tags in kernel, `DataBuffer<Tag,T>` with real CUDA/HIP specializations injected from `runtime/` headers, explicit size-checked transfers — is coherent, complete across CPU/CUDA/HIP, and appropriately explicit for HPC. Backend selection is compile-time per-binary (each app ships `src/{cpu,cuda,hip}/` TUs), which is consistent and defensible. The one genuinely shared GPU file, `gpu_autotune.hpp`, shows the single-source pattern works.

**What is not.**
- **The Kokkos facsimile is ornamental.** `View`/`create_mirror`/`deep_copy`/`parallel_for`/policies (~1,100 lines) have zero production consumers; the device `parallel_for` is a host loop (§4.2); `deep_copy(view, scalar)` on device round-trips an O(n) host vector (`deep_copy.hpp:137–147`); no `parallel_reduce`, subviews, atomics, or async semantics. Its stated purpose — easing later Kokkos adoption — is undermined by nothing using it. *Verdict: decide (open decision §17): adopt Kokkos for real, or delete everything above `DataBuffer`. Do not keep the half-implementation.*
- **Storage triplication:** `GPUVector` (CUDA-only, throws at runtime on non-CUDA builds — the anti-pattern the tag system was built to fix), `DataBuffer` (the workhorse), `View` (tests only), plus demo kernels (`kernels_simple.cu`) wired into the autotuner. *Verdict: `DataBuffer` is the keeper; remove `GPUVector` + demo kernels.*
- **Vendor fork:** see AB‑2. Includes deliberate asymmetries (HIP checks `hipFree` in destructors, CUDA ignores `cudaFree`) and `.cu`/`.hip` sources living under `include/` and installed as headers (`LibraryConfiguration.cmake:186–246`), while `src/…/padded_halo_faces.cu` is in *no* library and recompiled per consumer. *Verdict: single-source; move sources to `src/`.*
- **No residency/mirroring support at all.** Host↔device coherence is app-side hand-rolled state machines (`m_cpu_buffer_valid`, `sync_*` — duplicated per vendor per app). This is the root cause of the §4.1 critical bug. *Verdict: the 0.2 field abstraction must own placement and mirroring (§13).*
- **`OPENPFC_HD` keys only on `__CUDACC__`** (`host_device.hpp:54–60`); works under hipcc by accident. *[LC] repair.*

---

## 7. Spectral and finite-difference architecture analysis

**Spectral.** Two real layers over HeFFTe, shared across CPU/CUDA/HIP; consistent normalization; correct r2c k-space folding helpers (`kspace.hpp`). Defects: the `IFFT` GPU dishonesty (AB‑4); eager dual-precision GPU workspace allocation wasting ~33% device memory on double-only runs (`fft_heffte_backend.hpp:102–106`) *[SD, repair]*; the k-space triple loop re-implemented 5+ times because `kspace.hpp` offers only per-scalar helpers — tungsten ×3, aluminumNew (which still hand-rolls `pi = atan(1)*4`), and `SpectralGradient` itself ignores the helpers (`spectral_gradient.hpp:111–143`) *[SD, repair: add a `for_each_kpoint(outbox, world, fn)` iterator]*; Nyquist mode not zeroed for odd-derivative spectral operators *[C-low, repair]*; **no dealiasing option anywhere** — cubic nonlinearities transformed with no 2/3-rule mask, admitting aliasing-driven error at marginal resolution *[C, model-dependent; repair: optional diagonal mask + document]*.

**Finite difference.** The stencil layer is the strongest-designed corner of the codebase (§3.2) — the 8th/10th-order requirement is met today. Defects: capability inversion — CPU `FDGradient` `static_assert`-rejects mixed derivatives (`xy/xz/yz`) that the GPU twin already ships (`fd_gradient.hpp:306–314` vs `fd_gradient_device.hpp:214–252`) *[SD, repair]*; three overlapping FD evaluation APIs with back-compat shims marked for removal *[SD, consolidate]*; the silent no-op dispatcher (§4.3); **periodic-only boundary handling** — no non-periodic FD boundary treatment exists, and physical BC machinery lives on the spectral model path *[SD, latent; must be addressed for FD-first-class status]*.

**Method orthogonality.** For explicit point-wise stepping, FD/spectral selection is genuinely orthogonal to physics (§3.1) — the best idea in the codebase. But the unification stops there: implicit-Fourier and ETD paths (`SpectralDiagonalSolver`, `spectral_exp_coefficients` — CPU-only, unusable by GPU tungsten) are spectral-only silos, and production models hand-write their spectral stepping. That is partly physics reality (PFC is stiff and spectral-friendly), but the solver-contract layer (`solver_contract.hpp` — descriptor-based, no virtual `LinearSolver`, genuinely solver-agnostic in shape) is the right convergence point and should grow toward the evaluator concept rather than remaining a tests-only artifact. There is **no implicit FD solve at all** — acceptable for 0.2 as long as the solver contract doesn't preclude one.

**Stacks.** `FdCpuStack`/`SpectralCpuStack` are convenience RAII bundles, not a method-selection abstraction — and exist only for CPU. GPU apps assemble everything by hand. *Verdict: after the GPU layer is single-sourced, provide device-capable stacks; method choice should be a constructor-level decision, which the JSON layer can then expose.*

---

## 8. Physics, coupled fields, and time-integration analysis

**Physics interface:** see AB‑1/AB‑5. Two model notions coexist (virtual `Model` vs duck-typed `rhs(t, G)`), no adapter. The registry path is stringly-typed, non-owning, allocates a `std::string` per lookup, and couples by convention inside `step()`.

**Coupled fields:** bolted on twice, in opposite ways. Legacy: string-keyed reference maps, coupling by convention. New: compile-time tuple packs with inconsistent arity caps (§AB‑6) and `std::vector`-only state. `wave2d` proves the tuple path works for a genuinely coupled 2-field system with real convergence tests — but nothing beyond N=2 is exercised, and per-field dt/tolerances exist only in an unconsumed config struct (`adaptive_control_config.hpp:71–73`). *Verdict: generalize the typed path over N and over field types; keep the name registry as an I/O-naming layer only.*

**Time integration:** `Time` itself is excellent (§3.5). Adaptive stepping exists as components (embedded RK error vectors → MPI-aware `error_evidence` → `AdaptiveControlConfig` → `Time` transactions) with **no controller connecting them and zero non-test users** — ~2,500 lines of scaffolding without an end-to-end path. Legacy models can't adapt anyway (dt baked into `initialize`). *Verdict: build the missing controller as the next increment; stop adding evidence/config types until one adaptive run exists.*

**Boundary conditions:** `FixedBC`/`MovingBC` are not general BCs — they are the tungsten/aluminum directional-solidification setup (sigmoid density band, hard-coded x-direction, front tracking with `MPI_Reduce`) living in the framework namespace. BC-as-FieldModifier is honest for the spectral-periodic penalty regime but cannot express ghost-cell/operator-level BCs for FD; meanwhile two more embryonic BC mechanisms exist unimplemented (`ExecutionService::prepare_boundaries`, `StageContext` BC flags) — three parallel BC concepts, one working. Also: `Simulator` silently *drops* modifiers whose target field is missing (rank-0 warning only). *Verdict: relocate FixedBC/MovingBC to the apps; design BC handling for the stepper stack around stage preparation; make registration failures loud.*

**Checkpoint/restart:** good primitives, no owner (§10).

---

## 9. MPI, decomposition, communication, and scalability analysis

**Decomposition.** HeFFTe-brick split as a value type; FD and spectral share one decomposition and one ownership map — coherent. Issues: compile-time HeFFTe dependency even for FD-only use (`decomposition.cpp:4`) *[SD]*; the implicit box-ordering invariant (§4.9); periodicity hard-wired into neighbor wrap (`decomposition_neighbors.hpp:84–88`) *[SD — must gain per-axis flags]*; O(P) subworlds on every rank *[LC, fine to ~10⁵ ranks]*; no MPI topology hints.

**Halo exchange: nine library classes + one app-local, of which four have zero production users** (`HaloExchanger`, `PersistentHaloExchanger`, `FullPaddedHaloExchanger`, both `FullPaddedDeviceHalo`s). The unpadded in-place `HaloExchanger` overwrites outermost *owned* cells — a documented semantic trap. Meanwhile the most scalable exchanger in the repository — `BatchedPaddedDeviceHalo`, real multi-field message aggregation — lives inside the kobayashi app as a self-described "Phase 1 workaround". *Verdict: two blessed paths (`PaddedHaloExchanger` for structured bricks, `SparseHaloExchanger` for general index sets), backend-templated; fold persistent-request mode in; promote batching to the library; remove/quarantine the rest.*

**Scalability posture: correct but latency-bound at scale.** Everything is deadlock-free and defensively checked, but: persistent requests implemented and used by nothing; `start_/finish_` overlap split exists on all host exchangers and every production caller uses the blocking form; **GPU exchangers are blocking-only — no overlap API exists on device at all** (the main scalability ceiling for a GPU-first stencil code); aggregation only app-local; one message per face per field per step with hand-spaced tag offsets (kobayashi CPU: six exchangers at tags 0,20,40…); construction-time `MPI_Allgather` per exchanger. GPU-aware path uses non-contiguous subarray datatypes on device pointers, which UCX degrades to many tiny copies — the pack-to-contiguous fix exists only as an env toggle. And the sparse path's `RemoteHalo` is hard-bound to `CpuTag`, so GPU apps using it copy the **entire field D2H every step** (`apps/allen_cahn/src/cuda/allen_cahn.cpp:100–116`). Plus the LUMI/Cray-MPICH dead detection (AB‑2). *Verdict: this is where "distributed-memory scalability" is won or lost — consolidate, then make the consolidated exchangers persistent + split-phase + batched + device-resident by default.*

**Communicators.** `pfc::mpi::communicator` correctly wraps arbitrary comms; `MPI_COMM_WORLD` appears only as default arguments in the simulation layer (residual hardcoding confined to frontend utils and apps). Subcommunicator-based external coupling is architecturally possible today. One gap: wrapped comms are aliased, never duplicated — tag collisions with a host application are possible *[SD for coupling; repair with optional `MPI_Comm_dup`]*.

---

## 10. Application, configuration, I/O, and external coupling analysis

**Configuration/wiring.** The much-fragmented `frontend/ui` layer is actually disciplined: strict include tree, single-purpose files, piecemeal-callable wiring steps, catalog DI. Its problems are *reach*, not rot: the session is hardwired to spectral+CPU (`SpectralCpuStack::m_fft` is a concrete `CpuFft`); there is no FD session despite `FdCpuStack` existing (wave2d rolls its own driver); JSON cannot select the integrator; three overload families per wiring function. Per-app config is worse: tungsten defines `from_json` three times (once per backend class) over the same 21 fields plus a 200-line validator block; aluminumNew hand-rolls a different idiom; four FD apps each carry private `cli.hpp`/reporting helpers (~1,000 deletable lines). *Verdict: retain the wiring architecture; generalize the session (type-erase or template the FFT/stack axis); one declarative parameter schema per model, not per backend-class.*

**I/O.** `BinaryWriter` exemplary; `VTKWriter` solid but **not registered in the default writer catalog** — JSON `"writer": "vtk"` warns and silently produces nothing (`results_writer_catalog.hpp:66–74`, `simulation_wiring_writers.hpp:83–89`), so tungsten ships its own VTK path *[repair: register it; make unknown writers a hard error]*. `ResultsWriter` is filename-template-based and receives the FFT-inbox-shaped domain — implicitly coupled to the spectral decomposition and unable to express stream/in-memory sinks (relevant to coupling). **No HDF5 field output** (HDF5 exists only for profiling) — the largest user-facing I/O gap given headerless raw bricks with a sidecar spec.

**Checkpoint/restart: write-only.** `kernel/checkpoint/` primitives are high quality (validate-before-mutate capture, atomic stage→rename publication) but there is **no loader**: `CheckpointMetadata` has `to_json` and no `from_json`; grep finds no reader of a published bundle anywhere; publication is `std::ofstream`, not MPI-collective, with undefined multi-rank semantics; `ui::App` never triggers a checkpoint. Actual restart is a manual three-key ritual (`from_file` IC + `result_counter` + `increment` + hand-edited `t0`). *Verdict: [SD, high user impact] finish the read/resume side and give checkpointing an owner before adding any more publish features.*

**External coupling.** No interface exists. Viable seams already present: `FieldModifier` (per-step hook with comm + field access), the free-function integration loop (an external orchestrator could own it), subcommunicator support, `SparseVector` for non-matching data. Missing: a stable field-handle export (name + extents + spacing + origin in one struct — currently scattered across World/Decomposition/FFT inbox), a time-negotiation contract, and coupling-coordinated restart. *Verdict: no redesign needed; specify a thin coupling adapter once the 0.2 state/field model exists (§13).*

---

## 11. Build system, packaging, tests, and CI analysis

**Targets/layering.** Header-dominant library; two OBJECT libs merged into one installed target; layer boundaries enforced only by a one-direction grep script (kernel→frontend) — kernel→runtime and runtime→frontend unenforced, header-only code unenforced by the build. **Packaging is the most defective area** because nothing consumes `find_package(OpenPFC)` in CI: **[C-build]** `--coverage` is applied PUBLIC and defaults ON regardless of build type — default Release builds and the *installed, exported package* force gcov instrumentation on consumers (cluster presets patch around it, which is the tell); **[C-build]** `openpfc_hip_kernels` is missing from the install/export set while CUDA's twin is exported (`Installation.cmake:37–43`) — HIP installs are broken; `OpenPFCConfig.cmake.in` lacks `find_dependency` for CUDAToolkit/HDF5/hip; `OpenPFC_ENABLE_CUDA/HIP` are directory-scope definitions (not exported) while the `_SPECTRAL` variants are target-PUBLIC (exported) — downstream header behavior can differ from in-tree. Also: `.cu/.hip` sources under `include/` installed as headers; `padded_halo_faces.cu` in no library, recompiled per consumer; cmake floor of 3.15 is fiction (HIP needs 3.21); default build type Debug+coverage for an HPC code. *Verdict: a one-day repair pass on the four High items plus a CI job that installs OpenPFC and builds hello-world against it closes most of the risk.*

**Tests.** Strong: 1030 cases; convergence-order pins with ratio windows; CPU-vs-CUDA/HIP parity at 1e-10 (tungsten 32³/10 ETD steps; allen_cahn; wave2d); ETD-weight provenance pins (`spectral_exp_cache_matches_legacy_etd_weights`); golden norms for aluminum; hexfloat checksums for kobayashi; manual-vs-stack equivalence pins for heat3d/wave2d. Gaps: **no GPU tests run in CI at all**; heat3d's `vs_legacy_step` is a self-described structural stub; no multi-rank multi-step tungsten golden-field baseline; MPI suites capped at 2 ranks in CI; all non-MPI tests run as one monolithic ctest invocation.

**CI.** Ubuntu-only, gcc-only merge gate; clang-tidy and clang-format non-blocking; ASan manual-only; **no GPU CI, not even compile-only** — the HIP export bug proves GPU paths rot silently between manual cluster builds. Docs CI (link check, examples-catalog consistency, Doxygen warnings-as-errors) is unusually good. **Benchmarks:** two micro-benchmarks + app-side slurm harnesses; **no tracked performance baselines** anywhere; stale one-off numbers embedded in a README. *Verdict: add compile-only CUDA and clang jobs; capture machine-tagged baseline JSON via the existing profiling exporter.*

---

## 12. Documentation versus implementation discrepancies

Unusually for a project of this history, the documentation **largely describes reality**: `docs/concepts/architecture.md`'s kernel/runtime/frontend tables match the tree file-for-file; the refactoring roadmap's "Done" bullets check out against the build; docs-consistency scripts run in CI. Real discrepancies found:

- `docs/concepts/architecture.md` claims "no `#ifdef OpenPFC_ENABLE_CUDA/HIP` in kernel or frontend" — false: `kernel/fft/detail/fft_heffte_backend.hpp:31,35` and `frontend/ui/app.hpp:99,115`.
- `model.hpp` Doxygen is aspirational/confused: `@since v2.0` claims in a 0.1.4 project; examples show APIs (`register_real_field`) that don't exist under those names; the deprecated virtual `get_field()` is retained for out-of-tree code that may not exist.
- `world_queries.hpp` documents rounding where the code truncates; `spectral_cpu_stack.hpp` comments describe a `Decomposition` reference that is now a value; `csys.hpp` docs show methods that are free functions; stale `pfc::core`/`core/` path references throughout (the directory no longer exists).
- Names overpromise: `vs_legacy_step` tests for heat3d/wave2d are structural stubs, not numerical baselines; checkpoint doc-comments advertise "checkpointing and restart" while no restart path exists.
- `CHANGELOG.md` and the CMake version are frozen at 0.1.4 (2025‑12) versus seven months of substantive commits.

*Verdict: minor in volume, but fix the overpromising items (checkpoint, vs_legacy_step naming) before 0.2 planning relies on them.*

---

## 13. Proposed OpenPFC 0.2 target architecture

The proposal below is deliberately the *simplest* architecture that satisfies the stated goals. It keeps the codebase's proven ideas (gradient concepts, stencil tables, HeFFTe wrapper, Time, catalog wiring) and eliminates the generational parallelism. Compile-time composition dominates; runtime dispatch appears only at configuration boundaries and plugin seams.

### 13.1 Layer 0 — process environment
`mpi::environment`, `mpi::communicator` (gaining optional `MPI_Comm_dup` isolation). Every framework object takes a communicator; `MPI_COMM_WORLD` remains only a default at the outermost driver. *(Survives from 0.1 nearly unchanged.)*

### 13.2 Layer 1 — domain and layout
- **`Box3i`** — the single index-box POD (unify `Box3D`, `fft::Box3i`, World-as-box, `IndexBounds`; validated or removed redundant `size` member).
- **`Domain`** — replaces `World<CartesianTag>`: global grid size, spacing, origin, **per-axis periodicity that is actually consumed**. Plain Cartesian struct; the coordinate-system template parameter is deleted until a second coordinate system exists (it has exactly one instantiation today and half its query API is already Cartesian-concrete).
- **`Decomposition`** — maps `Domain` → per-rank `Box3i` (owned box) + neighbor topology honoring periodicity. Keeps the HeFFTe-compatible brick split but: asserts the box-ordering invariant, moves the HeFFTe call behind an interface so FD-only builds don't link HeFFTe, and hands out boxes, not Worlds.
- Strong types `GridSize/PhysicalOrigin/GridSpacing` survive; the other five and the dead `world_types` layer are deleted.

### 13.3 Layer 2 — storage and fields
- **`DataBuffer<MemorySpace, T>`** survives as the single storage primitive (CPU/CUDA/HIP specializations, explicit transfers). `GPUVector`, `Array`, and the Kokkos-facsimile `View` machinery are deleted (or, if the Kokkos decision in §17 goes the other way, replaced wholesale by `Kokkos::View`).
- **`Field<T, MemorySpace>`** — the *one* owning field: `DataBuffer` + layout metadata (owned `Box3i`, halo width, global `Domain` handle, geometry POD by value — no `const World&` members). Absorbs `LocalField` (its shape is the winner) and `PaddedBrick` (halo width 0 = unpadded); one linearization function; `apply(f(x,y,z))` defined once.
- **`FieldView<T>`** — the one non-owning view (today's `state_access.hpp` shape), used at kernel boundaries, I/O, and the external-coupling surface. `ScaledField` survives as a transient expression proxy.
- **Residency protocol**: a field knows its memory space; mirrors are explicit (`mirror_host(field)`) but *tracked* — the framework (not app models) brackets host-side operations (modifiers, writers) with validity flags. This is the structural fix for the §4.1 critical bug.
- **`SimulationState`** — owns fields by value, keyed by name for I/O/wiring and by typed handle for hot paths. Replaces `ModelFieldRegistry`'s non-owning references. This is the checkpoint/restart unit and the coupling export unit ("field handle" = name + `FieldView` + geometry, in one struct).

### 13.4 Layer 3 — communication
Two exchangers, both templated on memory space, single-sourced across vendors:
- **`HaloExchange`** (from `PaddedHaloExchanger` + device twins): structured face/edge/corner exchange sized by stencil requirements; modes folded in rather than sibling classes — persistent requests, split `start()/finish()` for overlap, multi-field batching (promoted from kobayashi's `BatchedPaddedDeviceHalo`), packed-contiguous device transport by default, corrected GPU-aware detection (incl. Cray MPICH).
- **`SparseExchange`** (from `SparseVector` + `SparseHaloExchanger`): general index-set exchange, backend-templated `RemoteHalo` (kills the per-step full-field D2H), the substrate for future FEM/unstructured coupling.
Deleted: `HaloExchanger` (in-place trap), standalone `PersistentHaloExchanger`, standalone FullPadded variants (26-direction support becomes a mode of `HaloExchange`).

### 13.5 Layer 4 — execution backends
One `runtime/gpu/` source tree parameterized over a ~20-line vendor shim (`gpuMalloc`/`gpuStream_t`/`GPU_CHECK`); HIP is never hand-edited. CPU path keeps OpenMP-collapse `for_each_interior`. `for_each_interior_device` gains parity (multi-field, composite, autotune) automatically by being single-sourced. Backend remains a compile-time axis per translation unit; the string/enum runtime seam (`Backend`, `backend_from_string`) covers all three backends honestly and is used only at configuration time.

### 13.6 Layer 5 — spatial operators and transforms
- **Gradient evaluators** (`FDGradient`, `SpectralGradient`, `CompositeGradient` + `grad_concepts`) are *the* operator abstraction — retained, extended to device via layer 4, CPU mixed-derivative gap closed. Stencil order remains a template parameter with the runtime-view escape hatch; halo width continues to be derived from the declared gradient aggregate, fail-closed.
- **FFT**: keep the HeFFTe `FFT_Impl<BackendTag>` core. The public surface splits honestly: a host interface over host containers and a device interface over `DataBuffer` (or one interface templated on buffer family — §17). Factories never return objects that throw on every method. Workspace precision is lazy or templated. A `for_each_kpoint(outbox, domain, fn)` iterator becomes the single k-space loop; an optional 2/3-rule dealiasing mask is provided as a standard diagonal.

### 13.7 Layer 6 — physics models and coupled equations
A **model is data + concept-conforming callables, not a base class**:
- declares its fields (names, types, memory space) → `SimulationState` allocates;
- declares parameters via one declarative schema (single `from_json` per model, not per backend);
- provides either `rhs(t, G)` (point-wise, method-agnostic — today's Gen‑3 shape) and/or spectral-diagonal descriptors (`physics_for_mode`: linear symbol + nonlinearity) for stiff PFC models.
The framework owns the "pseudo-spectral ETD model skeleton" (fields, k-loop, transform choreography, ETD combine) that tungsten and aluminumNew currently each hand-write; a PFC model shrinks to its physics (~300 lines, one file, all backends). Multi-field coupling is first-class: typed field packs generalized over N (no hard-coded arity-2), with the name registry retained only for I/O and wiring.

### 13.8 Layer 7 — time integration and solvers
- **One step protocol**: attempt/commit (`StepAttemptResult` was clearly the intended shape). Explicit Euler/RK, embedded RK, IMEX, and ETD all conform; `IntegratorBase`, `IntegratorResult`, the stranded `time.hpp` enum, and the duplicate `StageContext`/workspace types are deleted.
- **State type**: steppers operate on field/pack concepts (real *and complex* — the ETD/complex gap is the #169 blocker), not raw `std::vector<double>`.
- **Solvers**: the descriptor-based `SolveFunction` contract survives as the implicit seam; `SpectralDiagonalSolver` is its first implementation; the contract must not assume diagonality so an FD implicit solver (or an external library) can slot in later.
- **Adaptivity**: one controller closing the existing chain (embedded error → `error_evidence` → `AdaptiveControlConfig` → `Time` transactions). `Time` survives as-is and becomes the *only* save-point scheduler.

### 13.9 Layer 8 — orchestration, conditions, I/O, coupling
- **One `Simulator`** (thin): owns `SimulationState`, `Time`, a stepper, condition lists, writer/checkpoint services; its loop is a free function an external orchestrator can own. The Gen‑1 `Simulator`/`Model` pair is deleted after migration; `step_with_physics` is the transitional bridge.
- **Initial conditions**: `FieldModifier` + catalog survives (it is good). **Boundary conditions** split by nature: penalty/frame modifiers stay FieldModifiers (and `FixedBC`/`MovingBC` move into the PFC apps); operator-level BCs become stage preparation owned by the stepper/evaluator layer (one mechanism, replacing today's three half-mechanisms). Registration failures are hard errors.
- **Output**: `ResultsWriter` catalog with `vtk` registered and an HDF5/XDMF writer added; the writer contract narrows so non-file sinks are possible; domain metadata comes from `SimulationState`, not the FFT inbox.
- **Checkpoint/restart**: the existing capture/publish primitives gain the read side (metadata `from_json`, bundle loader through `BinaryReader`) and an owner (`CheckpointService` used by the Simulator), MPI-collective publication, and a `restart_from` config key that restores accepted time.
- **External coupling**: a small type-erased surface — field-handle export from `SimulationState`, the free-function loop, communicator injection, `SparseExchange` for data motion. Runtime polymorphism is justified exactly here (plugin/ABI boundary) and in the writer/modifier catalogs; everywhere else composition stays static.

### 13.10 Configuration
The existing JSON wiring layer survives structurally, generalized along the axes it currently hardcodes: session parameterized by (backend, method-stack) instead of spectral-CPU-only; integrator method selectable (the parser already exists); no dead host FFT for GPU runs.

---

## 14. Concepts to retain, replace, merge, or remove

**Retain (as-is or with local repair):** `mpi::environment`/`communicator`; `Decomposition` (value semantics); `Box3i` (as *the* box); strong types `GridSize/PhysicalOrigin/GridSpacing`; `DataBuffer`; FD stencil tables + `fd_apply`; gradient evaluators + `grad_concepts` + `for_each_interior`; `FFT_Impl` HeFFTe core + `FFTLayout`; `SparseVector` + sparse exchange; `Time`; `SolveFunction` contract + `SpectralDiagonalSolver`; `FieldModifier` + catalogs; `BinaryWriter`/`BinaryReader`/`VTKWriter`; checkpoint capture/publish primitives; the JSON wiring tree; profiling; test suite; docs infrastructure.

**Merge:** `LocalField` + `PaddedBrick` + `field::Field<T>` + `DiscreteField` → one `Field<T,MemorySpace>`; the nine halo exchangers → `HaloExchange` (+modes) and `SparseExchange`; five step protocols → attempt/commit; two `StageContext`s, two workspaces, two method enums → one each; CUDA+HIP runtime trees → one single-sourced device layer; per-backend tungsten model triplets → one templated model; `World` global-vs-subworld roles → `Domain` + `Box3i`.

**Replace:** virtual `Model` → concept-based physics + `SimulationState` (owning); `IFFT`'s dishonest GPU surface → split/templated FFT interfaces; app-side residency hand-rolling → framework residency protocol; hand-rolled per-app JSON/validators → declarative parameter schema.

**Remove:** the Kokkos-facsimile above `DataBuffer` (unless Kokkos is adopted outright — §17); `GPUVector`; `kernels_simple`; `Array`; `Box3D`; the dead `world_types` strong-type layer and its undefined functions; the `CoordinateSystem` template machinery; unused strong types (`LocalOffset`, `GlobalOffset`, `PhysicalCoords`, `IndexBounds`, `PhysicalBounds`); `HaloExchanger` (in-place trap) and the standalone persistent/full-padded siblings after their modes are folded in; `IntegratorBase`, `IntegratorResult`, `IntegratorMethod`-in-`time.hpp`; `legacy_adapter.hpp`; `Model::get_field()` and the `pfc::Field` vector alias; `FixedBC`/`MovingBC` from the framework namespace (relocate to apps); commented-out constructor graveyard in `world.cpp`; `.cu/.hip` sources from `include/`.

**Responsibilities to make independent:** device residency (framework, not models); save scheduling (Time only); BC application (stage preparation, not physics); checkpoint orchestration (service, not app wrappers); k-space iteration (library helper, not per-model loops); parameter validation (schema, not per-backend `from_json`).

**Dispatch policy:** templates/concepts for physics, steppers, evaluators, memory spaces (zero-overhead hot paths); runtime dispatch only at configuration boundaries (catalogs, backend-from-string, writer/modifier factories) and the external-coupling/plugin surface (type-erased field handles, free-function loop).

---

## 15. Recommended refactoring phases

Phases, not milestones — each leaves the tree green and scientifically validated. Detailed execution planning is explicitly out of scope for this audit.

- **Phase 0 — Baseline hardening (before any structural change).** Fix the §4.1 residency bug (or disable App-driven GPU entry points); fix §4.2/4.3/4.7 traps; the §11 packaging four (coverage leak, HIP export, find_dependency, definition propagation); add compile-only CUDA CI + a find_package smoke test; capture the missing golden baselines (§16). Nothing else proceeds until the safety net exists.
- **Phase 1 — Core data model.** `Domain`/`Box3i` split (fixes §4.4–4.6 by construction); `Field<T,MemorySpace>` + `FieldView` + `SimulationState`; migrate Gen‑2/Gen‑3 consumers; delete the four losing containers and dead type layers. Kokkos decision (§17) is taken at the start of this phase because it determines `Field`'s storage substrate.
- **Phase 2 — Single-source GPU runtime + communication consolidation.** Vendor shim; HIP parity (multi-field driver, composite, autotune, enum/string seam, Cray MPICH detection); fold halo modes (persistent/split/batched) into the two blessed exchangers; backend-templated `RemoteHalo`; delete unused exchangers.
- **Phase 3 — Stepper/solver unification.** One attempt/commit protocol; complex-capable ETD (unblocks #169); N-generalized field packs; the adaptive controller; honest FFT interface split. Validated against the existing convergence and ETD-provenance pins.
- **Phase 4 — The production vertical slice.** Tungsten rebuilt as one backend-templated concept model on the framework ETD skeleton + `SimulationState` + residency protocol + device stacks, validated bit-comparably (within documented tolerances) against the Phase‑0 golden baselines on CPU, CUDA, HIP, multi-rank. aluminumNew follows on the same skeleton. This phase is the go/no-go gate for deleting Gen‑1.
- **Phase 5 — Orchestration and frontend generalization.** Thin `Simulator`; session parameterized by backend/method; integrator selection from JSON; checkpoint read side + `restart_from`; VTK/HDF5 writers in catalog; BC-as-stage-prep; `FixedBC`/`MovingBC` relocated.
- **Phase 6 — Deletion and release.** Remove `Model`/legacy `Simulator` path, compat shims, deprecated APIs; layering enforcement (per-layer targets or bidirectional grep); docs sweep; version bump to 0.2.

Compatibility adapters recommended *only* where they reduce risk during phases 3–4: the `Model`→physics-concept shim and `step_with_physics` bridge, both deleted in Phase 6. No permanent parallel legacy/modern systems.

---

## 16. Risks and required scientific validation

**Baselines that already exist and must be preserved as the refactor contract:**
- CPU↔CUDA↔HIP parity: tungsten (32³, 10 ETD steps, ≤1e‑10), allen_cahn, wave2d.
- ETD provenance: `spectral_exp_cache_matches_legacy_etd_weights` and the spectral-operator edge-case suite (`test_tungsten.cpp:415–604`) — the key pin for the ETD/stepper migration.
- Convergence orders: RK2/RK3/RK4 ratio-window tests (integration scenarios + wave2d); temporal-convergence and mass-conservation scenarios.
- Golden values: aluminum 5-step field norms; kobayashi hexfloat checksums with bitwise OpenMP thread-count parity.
- Equivalence pins: heat3d manual-vs-stack L2 equality; wave2d manual-vs-separated step.

**Baselines that must be created before Phase 1:**
1. A **multi-rank (≥4), multi-step (≥100) tungsten golden-field trajectory** (checksummed binary output, CPU) — the flagship currently has no long-horizon distributed regression.
2. Equivalent aluminum trajectory (the only other production physics).
3. CPU-runnable golden-field captures for each per-app CPU-vs-GPU test's CPU side, so CPU CI can detect refactor regressions even without GPUs.
4. **Performance baselines**: machine-tagged JSON (the profiling schema-v2 exporter exists) for tungsten strong/weak scaling and halo-exchange microtimings on at least one NVIDIA and one AMD system — otherwise "no performance regression" is unfalsifiable.
5. A restart-equivalence test (run N+M steps ≡ run N, restart, run M) the moment the checkpoint read side lands.

**Principal risks:**
- *Scientific drift during the ETD migration* (highest): mitigated by the provenance pins and trajectory baselines; any tolerance widening must be justified in writing.
- *GPU regressions invisible to CI*: mitigated by Phase 0 compile-only GPU CI plus scheduled cluster runs of the parity suites; the LUMI GPU-aware-MPI fix must be validated with `verify_gpu_aware_mpi` on the real machine.
- *Floating-point non-determinism* when consolidating kernels (reduction order, FMA): declare which baselines are bitwise (kobayashi checksums) vs tolerance-based, per backend, before Phase 2.
- *Big-bang scope creep*: the phase gates (especially Phase 4 as go/no-go for deleting Gen‑1) are the control; if Phase 4 fails validation, Gen‑1 remains shippable throughout.
- *HeFFTe coupling*: the box-ordering assertion (§4.9) must land before any HeFFTe version bump.

---

## 17. Open architectural decisions

Decisions the audit deliberately leaves to the maintainer, with a recommendation each:

1. **Adopt Kokkos vs keep the minimal homegrown layer.** Adopting buys real device `parallel_for`/`parallel_reduce`, views, and vendor portability for free, at the cost of a heavyweight dependency alongside HeFFTe and less control over halo/MPI interplay (which stays homegrown either way). Keeping means `DataBuffer` + single-sourced kernels and deleting the facsimile. *Recommendation: keep the minimal layer for 0.2 — the kernel inventory is small and stencil-shaped, the halo machinery (the hard part) is already yours, and the facsimile's only argued benefit (easy later adoption) is preserved by deleting rather than half-maintaining it. Revisit if kernel diversity grows (reductions, scans, unstructured).*
2. **FFT interface shape**: one interface templated on buffer family vs split `IHostFFT`/`IDeviceFFT`. *Recommendation: split interfaces — simpler, and the config layer already knows the backend when it constructs the session.*
3. **Precision policy**: `double`-only (current reality) vs `RealType` templating end-to-end (tungsten GPU already templates). *Recommendation: template the new `Field`/steppers from day one — retrofitting is the expensive direction — but instantiate/test only `double` in 0.2.*
4. **Decomposition ownership**: keep HeFFTe's splitter behind an interface vs write the ~30-line min-surface splitter and make HeFFTe purely an FFT dependency. *Recommendation: the latter, once the box-ordering assertion exists; it decouples FD-only builds.*
5. **Field output format strategy**: raw bricks + sidecar (current) vs HDF5/XDMF as the blessed format. *Recommendation: add HDF5/XDMF behind the catalog, keep raw for performance paths; decide the checkpoint bundle format at the same time.*
6. **Error taxonomy**: keep `std::runtime_error`-with-good-messages vs introduce a `pfc::Error` hierarchy. *Recommendation: defer; only needed if library embedding/coupling users demand typed catching.*
7. **In-tree apps' destiny**: tungsten/aluminum as in-tree reference models vs separate downstream repos consuming `find_package(OpenPFC)`. *Recommendation: keep in-tree through 0.2 (they are the validation vehicle), reconsider after the packaging surface is CI-tested.*

---

## 18. Final assessment

**Is the current architecture a sound foundation for the long-term goals? Not yet — but its newest third is.**

The codebase contains, side by side: a legacy architecture that runs all the science but cannot reach the goals (virtual `Model` welded to spectral stepping, no adaptivity, no device residency, string-keyed non-owning state); and a newer architecture that points squarely *at* the goals (method-agnostic operator evaluation, high-order FD as first-class, attempt/commit stepping, solver contracts, explicit memory spaces) but has never been asked to do production work. Around both sits infrastructure of genuinely mixed maturity: communication code that is tactically excellent and strategically underused, a GPU story that is duplicated where it exists and bypassed where it matters, and a packaging surface that would fail its first external consumer.

The failure mode to avoid is the current trajectory's natural endpoint: incremental header-splitting and seam-adding (visible in the roadmap's Phases A–E) that polishes both generations indefinitely without ever collapsing them. The audit's central recommendation is therefore a *decision*, not a design: *declare the Gen‑3 concepts the only architecture, prove them on tungsten with GPUs and restart, and delete everything they replace* — roughly the sequence in §15. The design work required is modest because most winning components already exist; the discipline required is in the deletions.

Severity-ranked top ten actions: (1) fix the GPU-tungsten IC residency defect; (2) capture the missing golden/performance baselines and minimal GPU CI; (3) single-source the CUDA/HIP runtime and fix Cray-MPICH GPU-aware detection; (4) unify the field/box data model; (5) unify the stepper protocol with complex-state ETD (#169); (6) rebuild tungsten as the Gen‑3 vertical slice; (7) fold halo modes into two exchangers and promote batching; (8) generalize the frontend session beyond spectral-CPU and wire integrator selection; (9) finish checkpoint/restart end-to-end; (10) repair the packaging/export defects.

Done in that order, OpenPFC 0.2 can honestly claim what 0.1 aspires to: one architecture, two discretization methods, three backends, N coupled fields — with the science pinned before, during, and after the change.
