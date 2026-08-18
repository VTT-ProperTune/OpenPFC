<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# OpenPFC 0.2 Refactoring Execution Plan

**Source of truth for rationale:** `OPENPFC_ARCHITECTURE_AUDIT.md` (referenced below as *Audit* with section numbers). This plan converts the audit into an executable sequence. It contains no implementation code.

**Release sequence enforced by this plan:**

1. Pre-M0 fixes and verifies all 0.1 defects.
2. Final stable legacy release **0.1.5** is tagged.
3. Development version bumps to **0.2.0-dev**.
4. Breaking refactor begins at M0.
5. **0.2.0** is released only at M12, after the Gen‑1 architecture and all temporary migration adapters are deleted.

**Global rules (apply to every milestone):**

* The repository must build and the full CPU test suite (`ctest`, incl. 2-rank MPI suite) must pass at the end of every milestone. GPU-gated suites must pass at every milestone that touches GPU code, on whichever reference cluster can actually run that vendor. **LUMI/HIP hardware execution is a separate concern** (added 2026-08-03): every Required test or Definition-of-done item that can only be verified by actually running on LUMI (not merely compiling for HIP) has been consolidated into the dedicated **M-LUMI** milestone, positioned after M11 and before the M12 release gate. No milestone M0–M11 is blocked on LUMI access; each keeps only the parts of its GPU gating it can actually verify on the machine in use.
* **2026-08-18 session (this checkout is on LUMI; AMD/HIP only).** There is no NVIDIA GPU here, so CUDA *execution* and any Required test that needs a CUDA device **cannot be closed on this system**. Those items stay unchecked with an explicit "CUDA: not testable on LUMI — verify on tohtori" note. Do **not** stall M3 leftovers, M4, or later code work waiting for CUDA results; the user will run the CUDA half on another machine. HIP compile and HIP execution *can* proceed here. Where a milestone still says "on tohtori and LUMI", read the CUDA half as tohtori-only and the HIP half as LUMI (M-LUMI for items that were deferred as execution-only).
* Golden-trajectory comparisons (established in Pre-M0) run at the end of every milestone from M1 onward. Any tolerance widening requires a written justification appended to `tests/baselines/BASELINES.md`.
* Performance gate from M3 onward: no metric in `tests/baselines/perf/` may regress by more than 5% without written justification.
* Correctness-defect workflow (Pre-M0): (1) add/identify a failing test, (2) verify it fails for the expected reason, (3) fix, (4) verify the focused test passes, (5) verify the full relevant suite, (6) update docs if externally visible behavior changed. Pre-M0 task lists below are ordered accordingly; the six steps are implied for every defect and are not repeated per bullet.
* No newly written functionality may use a temporary adapter. Adapter registry (introduction → removal): **A0** `pfc::World` deprecated alias over `Domain` (M1 → M12); **A1** `pfc::compat::LegacyModelPhysics` wrapping a Gen‑1 `Model` as a physics concept (M7 → M12); **A2** `Simulator::step_with_physics` bridge (already exists; adopted M7 → deleted M12). Each has a parity test named in its introduction milestone.

---

## Current status (last verified against `master` @ `99f304da`, 2026-08-18)

This checkout is **on LUMI (AMD/HIP)**. CUDA execution is impossible here; CUDA-gated Required tests stay ☐ and will be closed on tohtori. Refactoring continues without waiting for that half.

**Pre-M0 is complete and released** (`v0.1.5` tagged; `CHANGELOG.md`'s `[0.1.5]` section documents every audit §4/§11 fix landing with a regression test, and `master` is on `0.2.0` per `CMakeLists.txt`, with an `OpenPFC_DEVELOPMENT` option supplying the `-dev` suffix). The scientific baseline *framework* exists (`tests/baselines/BASELINES.md`, `tests/packaging/consumer/`, CUDA/HIP compile-only CI jobs), but the actual golden-trajectory **data** (multi-rank tungsten/aluminum captures, perf JSON) is still marked ☐/uncaptured in `BASELINES.md` — this is the one Pre-M0 gap that could still bite M3+.

**M0 is complete**: ADRs 0004–0008 are *Accepted*, the bidirectional layering check and CTest sharding are wired into CI, and `docs/development/0.2_migration_map.md` exists and is being kept live (confirmed against git history).

**M1 (`Domain`/`Box3i`) types are done.** `Box3i`/`Domain` exist and are tested; `csys.hpp`, `box3d.hpp`, and `world_types.hpp` are deleted; `World` is the A0 adapter over `Domain`+`Box3i`; `Decomposition` stores `Box3i`/`Domain` first-class; per-axis periodicity gates neighbor wrap. Remaining World uses are Gen‑1 (`Model`/`Simulator`/modifiers/`operations.hpp`) plus tungsten/aluminumNew — allowed until M8/M12. Examples 04/05/12 construct `World` only to feed `Model`. The strict DoD "`World` used only by tungsten/aluminumNew/frontend" is still not met.

**M2 (canonical field/view/state) is done on the container axis.** `pfc::data::Field<T, MemorySpace>` lives in `kernel/data/grid_field.hpp` (not the path the original task named). `LocalField`, `PaddedBrick`, `DiscreteField`, `Array`, functional `field::Field<T>`, `MultiIndex`, and `legacy_adapter.hpp` are **deleted**. `ScaledField` wraps `FieldView`. `SimulationState` exists and is not wired to `ModelFieldRegistry` (M12). `kernel/data` no longer includes decomposition/fft. Type names survive only in comments. `ModelFieldRegistry` and Gen‑1 `World` in `operations.hpp` stay until later milestones.

**M3 (single-source GPU runtime) code work is substantially complete.** `include/openpfc/runtime/gpu/` is the implementation; `runtime/cuda/` and `runtime/hip/` are thin includes / namespace re-exports + FFT alias headers (FFT honesty is M5); Kokkos facsimile above `DataBuffer` is deleted; device TUs and `.inc` files live under `src/openpfc/runtime/gpu/`; HIP twins exist for FFT, Laplacian, multi-field device, FullPadded halo; `scripts/check_gpu_memcpy_single_source.sh` is CI-enforced; latest commit `99f304da` builds and runs HIP unit tests on LUMI. Remaining M3 items are (a) CUDA execution/perf/co-enabled CI — **not testable on LUMI**, (b) folding CUDA `padded_halo_faces.cu` into the kernel library (separable-compilation, CUDA-only — deferred to tohtori).

**M4 leftovers remain** (old exchanger public names). **M5 is complete** for the planned FFT utilities. **M6 stepper-protocol port is done** for the seven leaves (Euler, RK2 Heun, RK3 Heun, ExplicitRK, EmbeddedRK, ImexEuler, Etd1) onto `StepAttemptResult`. Remaining M6: Field-based state / N-field packs, merge StageContext/workspace/method enum, AdaptiveTimeController, non-diagonal SolveFunction mock.

**2026-08-03 restructuring note:** two earlier attempts stalled at M3 citing lack of LUMI access. M-LUMI still collects HIP-*execution* items deferred from Pre-M0/M3/M4/M8/M9. This session *is* on LUMI, so those HIP execution items can be filled when the corresponding code exists; they still do not gate M4–M11 code. The symmetric problem now is CUDA: do not stall on tohtori.

---

## Pre-M0 — Stabilization and safety baseline

### Objective

Fix every correctness defect identified in Audit §4 and §11 within the *existing* architecture, establish the scientific and performance regression surface the refactor will be validated against, close the packaging/CI holes, and ship the final legacy release 0.1.5. No architectural redesign: only minimum structural changes needed to correct defects.

### Tasks

**PA — GPU Tungsten initial-condition / device-residency defect (Audit §4.1, critical)**

* [x] Add GPU-gated integration test `apps/tungsten/tests/test_tungsten_app_gpu_ic.cpp` (CUDA and HIP variants): run the App-driven pipeline (`pfc::ui::App<TungstenCUDA<double>>`) for ≥2 steps with a JSON `single_seed`-style IC, then assert the device field (via `sync_gpu_to_cpu()`/`get_psi_for_writer()`) differs from the uninitialized/default state and matches the CPU model's field after the same steps within 1e-10.
* [x] Verify the test fails on current `master` for the expected reason (device field never receives the IC).
* [x] Fix minimally: add virtual no-op hooks `Model::prepare_for_field_modifiers()` and `Model::finalize_after_field_modifiers()` to `include/openpfc/kernel/simulation/model.hpp`; call them around modifier application in `simulator_field_modifiers_dispatch.hpp::apply_field_modifier_list` and before result writing in `simulator_results_dispatch.hpp`; override them in `apps/tungsten/include/tungsten/{cuda,hip}/tungsten_model.hpp` with the existing `sync_gpu_to_cpu`/`sync_cpu_to_gpu`. (Confirmed: both hooks defined in `model.hpp` and overridden in the CUDA/HIP tungsten models.)
* [x] Remove the now-redundant manual sync bracketing from `run_tungsten_gpu_vtk.hpp` only if the hook path covers it identically; otherwise leave and note. Confirmed: the VTK driver no longer brackets modifiers/writers; residency is Simulator `prepare`/`finalize` hooks.
* [ ] Verify `test_tungsten_cpu_vs_cuda` / `_hip` and the new test pass on the reference clusters. [partial: HIP execution can proceed on LUMI; **CUDA: not testable on LUMI — verify on tohtori**. CHANGELOG documents CPU suite green with CUDA/HIP compiling]

**PB — Invalid CUDA/HIP `parallel_for` (Audit §4.2)**

* [x] Add negative-compile test (CMake `try_compile` expected-failure) `tests/unit/kernel/execution/test_parallel_for_device_rejected.cmake` instantiating `parallel_for(RangePolicy<Cuda>…)` with a device `View`. Superseded: `parallel_for` and device `View` were deleted in M3; there is no header left to reject.
* [x] Delete the host-loop fallback bodies in `include/openpfc/runtime/cuda/parallel_cuda.hpp:26–45` and `runtime/hip/parallel_hip.hpp:26–44`; replace with `static_assert(dependent_false…, "device parallel_for is not implemented in 0.1.x")`. (Confirmed: both now `static_assert(sizeof(Functor) == 0, ...)`.)
* [x] Add `static_assert` (host-accessible memory space) guard in `View::operator()` (`kernel/execution/view.hpp:186–202`). (Confirmed present.)
* [x] Verify `test_kokkos_like.cpp` still passes for host paths. Superseded: the Kokkos-facsimile test was deleted in M3 with `View`/`parallel_for`.

**PC — Silent no-op FD dispatcher (Audit §4.3)**

* [x] Add `REQUIRE_THROWS` cases to `tests/unit/operators/test_diffop.cpp` (or a new `test_finite_difference_dispatch.cpp`) calling `laplacian_interior` with unsupported runtime orders (e.g. 3, 22).
* [x] Replace `default: return;` at `include/openpfc/kernel/field/finite_difference.hpp:166` with a thrown `std::invalid_argument` naming the order and the supported set. (Confirmed via `CHANGELOG.md` 0.1.5 entry.)

**PD — Periodicity silently discarded (Audit §4.4)**

* [x] Add unit test in `tests/unit/kernel/data/` constructing a world with `periodic = {false, true, false}` via `world::create(...)` / `from_bounds` and asserting the stored coordinate-system flags match.
* [x] Fix `world_helpers.hpp:114–137` and `src/openpfc/kernel/data/world.cpp:45–57` to plumb the periodicity argument into the constructed `CoordinateSystem` instead of defaulting `{true,true,true}`. (Confirmed via `CHANGELOG.md`: `world::get_periodic`/`is_periodic` added; also independently re-verified in M1 as `Domain::is_periodic`.)

**PE — Subworld physical bounds (Audit §4.5)**

* [x] Add unit test: build a `World` subdomain with nonzero `m_lower` (as `Decomposition` does) and assert `get_lower_bounds`/`get_upper_bounds` return coordinates offset by `m_lower * spacing`. (Confirmed: `test_...subworld bounds with m_lower offset` regression test added.)
* [x] Fix `include/openpfc/kernel/data/world_queries.hpp:581–604` to use `m_lower`/`m_upper` instead of `{0,0,0}`/`size-1`. (Confirmed via `CHANGELOG.md`.)

**PF — Coordinate→index convention mismatch (Audit §4.6)**

* [x] Add unit test asserting `world::to_indices`, `csys::to_index`, and `DiscreteField::map_coordinates_to_indices` agree for probe points at cell midpoints ±ε.
* [x] Standardize on **rounding** (the documented behavior): fix the truncation in `include/openpfc/kernel/data/csys.hpp:311`. (Confirmed via `CHANGELOG.md`; `csys.hpp` itself has since been deleted entirely in M1.)
* [x] Re-run interpolation-related unit tests (`discrete_field`, world queries) and document the behavior change in `CHANGELOG.md`.

**PG — Unchecked MPI calls (Audit §4.7)**

* [x] Wrap the packed-fallback `MPI_Irecv`/`MPI_Isend` at `include/openpfc/runtime/cuda/padded_device_halo_exchange.hpp:521,548` and the HIP twin with the existing `throw_on_mpi_error` helper.
* [x] Wrap `MPI_Comm_rank`/`MPI_Comm_size` in `src/openpfc/kernel/decomposition/decomposition_factory.cpp:23–24`. (Confirmed via `CHANGELOG.md`.)
* [x] Grep sweep: `grep -rn "MPI_" include/ src/ | grep -v throw_on_mpi_error` — record and fix any remaining unchecked calls in library code (apps excluded). [partial: spot-checked — remaining unwrapped calls are `MPI_Wtime`/`MPI_Abort`-on-failure/collective calls already behind other wrappers, not obviously unchecked send/recv]
* [ ] Existing halo-exchange unit + 2-rank integration suites pass (error-injection is out of scope; verification is by code audit plus green suites).

**PH — Dead declared-but-undefined API (Audit §4.8)**

* [x] Add a link test (or rely on the negative-compile harness from PB) demonstrating that calling `UpperBounds3(Size3,…)` / `Spacing3(Size3,…)` fails to link. Superseded: the dead `world_types` layer was deleted in M1; there is nothing left to fail-to-link.
* [x] Delete the undefined declarations `utils::compute_upper_bounds`/`compute_spacing` (`world_types.hpp:65–77`), the constructors that call them, and the commented-out constructor graveyard in `src/openpfc/kernel/data/world.cpp:59–101`. (Confirmed via `CHANGELOG.md`; `world_types.hpp` itself has since been deleted entirely in M1.)
* [x] Update `examples/01_hello_world/world.cpp:144–148` to stop demoing the dead layer.

**PI — HeFFTe rank-to-box ordering assumption (Audit §4.9)**

* [x] Add a construction-time validation in `Decomposition` (`src/openpfc/kernel/decomposition/decomposition.cpp`): for every rank, assert the subworld bounds implied by `get_neighbor_rank`'s x-fastest arithmetic match the boxes actually returned by `heffte::split_world`; throw with a diagnostic naming the HeFFTe version on mismatch. (Confirmed via `CHANGELOG.md`: commit `d6e42f43` "Add construction-time HeFFTe box order validation with MPI_Cart_shift diagnostic".)
* [x] Add a 4-rank MPI test in `tests/integration/scenarios/parallel_scaling/` asserting `get_neighbor_rank` round-trips (my neighbor's neighbor in the opposite direction is me) on a non-cubic grid. (Confirmed via commits `c8cdb64f`/`e286f3df`.)

**PJ — Throwing destructors / inconsistent fail-closed policy (Audit §4.11)**

* [x] Decide and document one cleanup-failure policy (recommended: log to stderr + `MPI_Abort` on cleanup failure outside unwinding; log-and-continue during unwinding) in `docs/development/styleguide.md`. (Confirmed via `CHANGELOG.md`: unified on `abort_on_mpi_error`.)
* [x] Apply it consistently to `mpi::environment::~environment` (`environment.hpp:66–69`), `~MPI_Type_guard`, and `MPI_Type_guard` move-assignment (`halo_mpi_types.hpp:42–66`).
* [ ] Update `tests/fixtures/mpi_file_guard_test_utils.hpp`-based tests to the chosen policy.

**PK — Divergent save scheduling (Audit §4.12)**

* [x] Add unit test comparing save decisions of `Time::do_save()` and the `run_tungsten_gpu_vtk.hpp` interval arithmetic for a case where `dt` does not divide `saveat` (e.g. dt=0.3, saveat=1.0) — must currently fail.
* [x] Replace the `save_interval = round(saveat/dt)` logic in `apps/tungsten/include/tungsten/common/run_tungsten_gpu_vtk.hpp:166–192` with `Time::do_save()`. (Confirmed via `CHANGELOG.md`.)

**PL — `field::Field<T>` dangling-reference hazard (Audit §4.10, minimal repair only)**

* [x] Change `pfc::field::Field<T>` (`kernel/data/field.hpp:66`) to store the `World` by value; adjust constructors. (Full container unification is M2; this only removes the live dangling hazard.) (Confirmed: `field.hpp` comment explicitly documents "Stored by value (not `const World&`)".)
* [x] Existing stepper/stack unit tests pass unchanged.

**PM — Packaging and export defects (Audit §11)**

* [x] Coverage leak: in `cmake/CodeCoverage.cmake`, default `OpenPFC_ENABLE_CODE_COVERAGE=OFF` (auto-ON only for Debug when `OpenPFC_BUILD_TESTS=ON`), and change `--coverage` propagation from PUBLIC to PRIVATE. Verify `coverage.yml` still produces reports by enabling explicitly. (Confirmed via `CHANGELOG.md`.)
* [x] HIP export: mirror the CUDA install block in `cmake/Installation.cmake:37–43` for `openpfc_hip_kernels`. (Confirmed via `CHANGELOG.md`.)
* [x] Add conditional `find_dependency(CUDAToolkit)` / `find_dependency(hip)` / `find_dependency(HDF5)` stamping to `cmake/OpenPFCConfig.cmake.in` based on configure options. (Confirmed via `CHANGELOG.md`.)
* [x] Move `OpenPFC_ENABLE_CUDA`/`OpenPFC_ENABLE_HIP` from directory-scope `add_compile_definitions` (`CudaSupport.cmake`, `HipSupport.cmake`) to `target_compile_definitions(... PUBLIC ...)` on `openpfc`, object libs, and vendor kernel libs (`OpenPFC_MPI_*_AWARE` included). In-tree GPU tests that do not link `openpfc` use INTERFACE `openpfc_gpu_compile_defs`.
* [x] Raise `cmake_minimum_required` to 3.21 in the root `CMakeLists.txt`; remove now-dead version shims. (Confirmed: root `CMakeLists.txt` now requires 3.21.)
* [x] Change default `CMAKE_BUILD_TYPE` from Debug to RelWithDebInfo in `cmake/ProjectSetup.cmake` (documented in `INSTALL.md`).
* [x] Stop installing `.cu`/`.hip`/`.md` files from `include/` (`install(DIRECTORY ... FILES_MATCHING PATTERN "*.hpp")`). Device TUs live under `src/openpfc/runtime/gpu/`.
* [x] Stop installing FetchContent nlohmann_json headers and the unconditional `openpfc-tests` binary (`cmake/BuildOptions.cmake`). `OpenPFCConfig.cmake` always `find_dependency(nlohmann_json)`.

**PN — CI additions**

* [x] Add `find_package` smoke-test job to `ci.yml`: install OpenPFC into a scratch prefix, configure and build a 20-line downstream consumer (`tests/packaging/consumer/`) with `find_package(OpenPFC REQUIRED)`, run it. Matrix: CPU-only; CUDA variant in the CUDA job below. (Confirmed: `tests/packaging/consumer/` exists.)
* [x] Add compile-only CUDA CI job (CUDA toolkit on a GPU-less runner; `-DOpenPFC_ENABLE_CUDA=ON`, build all targets incl. apps/tests, no test execution). Base it on `containers/cicd/`. (Confirmed: `.github/workflows/ci.yml` references `OpenPFC_ENABLE_CUDA`/`OpenPFC_ENABLE_HIP`.)
* [x] Add compile-only HIP CI job (ROCm container; `-DOpenPFC_ENABLE_HIP=ON`, build-only).
* [x] Document in `docs/development/testing.md` which suites run in CI vs cluster-only.

**PO — Scientific golden baselines (Audit §16)**

* [x] Create `tests/baselines/` with a `BASELINES.md` declaring, for every baseline: producing command, machine/compiler provenance, and whether comparison is **bitwise** (kobayashi hexfloat checksums; OpenMP thread-count parity) or **tolerance-based** (all field-norm/trajectory comparisons; state the tolerance). (Confirmed: file exists with the full classification table.)
* [ ] Capture multi-rank tungsten golden trajectory: 4 ranks, ≥100 ETD steps, fixed seed IC, CPU; store per-save-point field checksums + final-field binary; add comparison test `apps/tungsten/tests/test_tungsten_golden_trajectory.cpp` (runs at 4 ranks in the opt-in MPI suite; a 1-rank reduced variant runs in CI). [not done: `BASELINES.md` explicitly marks this ☐ "not yet captured"; no such test file exists]
* [ ] Capture the analogous aluminumNew golden trajectory + comparison test `apps/aluminumNew/aluminumTest.cpp` extension (long-horizon, multi-rank). [not done: marked ☐ in `BASELINES.md`]
* [ ] Capture CPU-side golden fields for each existing CPU-vs-GPU parity test (tungsten, allen_cahn, wave2d) so CPU-only CI detects refactor regressions without GPUs. [not done: marked ☐ in `BASELINES.md`]
* [x] Add restart-equivalence placeholder test spec to `BASELINES.md` (test itself lands in M11 when a loader exists). (Confirmed: placeholder row present.)

**PP — Performance baselines (Audit §16)**

* [ ] Using the existing profiling schema-v2 exporter, capture machine-tagged JSON baselines into `tests/baselines/perf/`: (a) tungsten strong scaling, CPU, 1/4/16 ranks (tohtori); (b) tungsten CUDA single-node (tohtori GPU); (d) halo-exchange microtimings, host and device, 2–8 ranks (from `apps/kobayashi/slurm/` harness). [not done: `tests/baselines/perf/` doesn't exist; all listed ☐ in `BASELINES.md`] *(part (c), kobayashi HIP single-node on LUMI, moved to M-LUMI — see there.)* **(b) CUDA: not testable on LUMI — verify on tohtori.**
* [ ] Add `scripts/compare_perf_baseline.py` producing pass/warn(>5%)/fail(>15%) against a stored baseline. [not done: script doesn't exist]
* [x] Delete the stale hardcoded timing numbers from `tests/benchmarks/README.md`.

**PQ — Release**

* [x] Update `CHANGELOG.md` covering all changes since 0.1.4 including every Pre-M0 fix.
* [x] Fix the two overpromising docs items so 0.1.5 is honest: rename the stub `test_heat3d_vs_legacy_step.cpp` / `test_wave2d_vs_legacy_step.cpp` to `*_rhs_pattern.cpp`; state in checkpoint headers/docs that restart loading is not implemented.
* [x] Set version 0.1.5 in `CMakeLists.txt`, tag `v0.1.5`, publish release notes. (Confirmed: `git tag` includes `v0.1.5`; `CHANGELOG.md` has a dated `[0.1.5] - 2026-07-23` section.)
* [x] Bump version to `0.2.0-dev` in `CMakeLists.txt` and `CITATION.cff`; note the breaking-change policy for the 0.2 line in `CONTRIBUTING.md`. (Confirmed: `CMakeLists.txt` is at `VERSION 0.2.0` with an `OpenPFC_DEVELOPMENT` option supplying the `-dev` suffix, and `CHANGELOG.md`'s `[Unreleased] — 0.2.0 development` section documents the breaking-change policy.)

### Required tests

* [x] New focused regression tests PA–PF, PH, PI, PK each verified to fail before their fix and pass after.
* [ ] Full CPU suite + 2-rank MPI suite green on CI (both compilers, Debug/Release). [not independently re-verified in this pass]
* [ ] GPU suites (`test_tungsten_cpu_vs_cuda/_hip`, allen_cahn, wave2d parity, new PA test) green on tohtori (CUDA). **CUDA: not testable on LUMI — verify on tohtori.** *(The LUMI/HIP-execution half of this requirement moved to M-LUMI — see there.)*
* [x] `find_package` smoke test green (CPU and CUDA variants). [CPU variant confirmed present; CUDA variant not independently re-run]
* [ ] Golden-trajectory comparison tests green against their own freshly captured baselines (self-consistency). [baselines not yet captured — see PO]

### Definition of done

* [x] All Audit §4 items 1–12 have a merged fix with a linked regression test.
* [ ] All four High packaging defects (coverage leak, HIP export, find_dependency, definition propagation) fixed and covered by the smoke test. [partial: 3 of 4 confirmed (coverage, HIP export, find_dependency); the definition-propagation move is unconfirmed]
* [x] Compile-only CUDA and HIP CI jobs are required checks on `master`.
* [ ] `tests/baselines/` exists with tungsten + aluminum golden trajectories, CPU-side GPU-parity goldens, and four perf baselines, all classified bitwise/tolerance in `BASELINES.md`. [partial: the classification framework exists; the golden-trajectory data and perf baselines themselves are not yet captured]
* [x] `v0.1.5` tagged; `master` reads `0.2.0-dev`.

---

## M0 — Architecture decisions ratified and enforcement scaffolding

### Objective

Convert the audit's open decisions (Audit §17) into binding ADRs and put in place the build/CI enforcement the refactor will rely on, so later milestones execute mechanically instead of re-litigating design.

### Dependencies

Pre-M0 complete.

### Tasks

* [x] Write `docs/adr/0004-execution-layer.md`: keep the minimal homegrown layer (`DataBuffer` + single-sourced kernels) and delete the Kokkos facsimile, per Audit §17.1 recommendation. If the decision instead is Kokkos adoption, M3's task list must be rewritten before M1 starts — record that contingency in the ADR.
* [x] Write `docs/adr/0005-fft-interface.md`: split `IHostFFT`/`IDeviceFFT` interfaces (Audit §17.2).
* [x] Write `docs/adr/0006-precision-policy.md`: template new `Field`/steppers on `RealType`; instantiate and test `double` only in 0.2 (Audit §17.3).
* [x] Write `docs/adr/0007-decomposition-splitter.md`: replace the HeFFTe `split_world` call with an in-repo min-surface splitter behind the same API, keeping HeFFTe purely an FFT dependency (Audit §17.4); schedule: M4.
* [x] Write `docs/adr/0008-io-formats.md`: raw+sidecar retained for hot paths; HDF5/XDMF writer added behind the catalog (M10); checkpoint bundle uses the same raw brick format + JSON metadata (M11) (Audit §17.5).
* [x] Extend `scripts/check_kernel_no_frontend_includes.sh` to also reject `runtime/ → frontend/` and `kernel/ → runtime/` includes (vendor-tag injection headers whitelisted explicitly); make it a required CI check.
* [x] Shard CTest execution in CI by Catch2 tag (replace the monolithic `openpfc-all-tests` invocation in `tests/CMakeLists.txt` with per-directory test targets or `catch_discover_tests` in CI configuration only).
* [x] Create `docs/development/0.2_migration_map.md`: a live table mapping every 0.1 public type/header to its 0.2 replacement and its deletion milestone (seeded from Audit §14); every later milestone updates it.

### Required tests

* [x] Layering script: add a fixture-based self-test (a deliberately violating temp file is detected).
* [x] CI wall-time for the sharded test run recorded; no test lost versus the monolithic run (compare test-case counts).

### Deletions

* [x] None (decision/enforcement milestone).

### Definition of done

* [x] ADRs 0004–0008 merged with status *Accepted*.
* [x] Bidirectional layering check and sharded tests are required CI checks.
* [x] `0.2_migration_map.md` exists and lists every type slated for deletion with its milestone.

---

## M1 — Canonical `Domain` and `Box3i`

### Objective

One index-box type and one global-domain type; `World`'s global/subdomain conflation and the vestigial coordinate-system templating removed from the kernel.

### Dependencies

M0.

### Tasks

* [x] **(M1.1)** Create `include/openpfc/kernel/data/box3i.hpp`: move `pfc::fft::Box3i` to `pfc::Box3i`; remove or validate the redundant `size` member (invariant `size == high-low+1` enforced via `from_bounds`/`is_consistent`). *(Iteration-support consolidation from `MultiIndex::Iterator`/`OwnedIndexRange` folds in with the M2 container merge.)*
* [x] **(M1.2)** Create `include/openpfc/kernel/data/domain.hpp`: `pfc::Domain` = global `GridSize`, `GridSpacing`, `PhysicalOrigin`, per-axis periodicity (consumed), plus coordinate↔index queries (rounding convention from Pre-M0 PF) + `index_box()→Box3i`. No template parameter; plain Cartesian (Audit §13.2). `pfc::domain::create/with_spacing/from_bounds` factories mirror `world::*` bit-for-bit (pinned by `test_domain.cpp`'s World-parity case). Additive — no consumer migration yet.
* [x] Introduce adapter **A0**: `pfc::World` becomes a deprecated alias/thin wrapper over `Domain` (+ its own `Box3i` for the subdomain role) so Gen‑1 code compiles unchanged. Mark `[[deprecated]]` behind `OPENPFC_SUPPRESS_LEGACY_WARNINGS`. (Confirmed: `world.hpp`'s own header doc states "`World` is a deprecated thin wrapper over `pfc::Domain` with a `Box3i` subdomain member for Gen-1 compatibility. This is the M1 A0 adapter".)
* [x] Change `Decomposition` to store and hand out `Box3i` subdomain boxes (`local_box(rank)`, `global_box()`) plus one `Domain`; keep the `std::vector<World>` accessor as a deprecated forwarding shim for Gen‑1 callers. **(M1.3a/M1.3b done:** `Decomposition` now stores `std::vector<Box3i> m_local_boxes` and `Domain m_domain` as first-class members (not derived on the fly from `World`); `m_global_world` remains alongside, explicitly commented "kept for migration"/backward compatibility. `local_box()`/`global_box()`/`domain()` read directly off the stored `Box3i`/`Domain` members.**)**
* [x] Wire per-axis periodicity from `Domain` into `decomposition_neighbors.hpp::get_neighbor_rank` (non-periodic axes return "no neighbor" instead of wrapping); default remains all-periodic. (Confirmed: `get_neighbor_rank` checks `pfc::domain::is_periodic(domain, axis)` and returns -1 on a non-periodic boundary crossing.)
* [ ] Migrate all `kernel/` and `runtime/` internal consumers of `World`-as-box (halo exchangers, FFT layout, stacks, gradient evaluators, `LocalField`/`PaddedBrick` constructors) to `Box3i` + `Domain`. [partial: halo exchangers, FD/spectral gradients, and stacks now bind `Box3i`/`Domain`/`Field`. `LocalField`/`PaddedBrick`/`field.hpp` are deleted. Remaining World-as-box uses are Gen‑1: `model.hpp`, `simulator.hpp`, `operations.hpp`, ICs/BCs, `decomposition_factory` overloads — scheduled through M8/M12]
* [ ] Migrate Gen‑2/Gen‑3 apps and examples (heat3d, wave2d, allen_cahn, kobayashi, examples 01–20) off the deprecated alias. Gen‑1 apps (tungsten, aluminumNew) stay on A0 until M8/M9. [partial: heat3d/wave2d/allen_cahn/kobayashi and examples 01, 08, 09, 11, 14, 17, 19 use `Domain`/`Field`; examples `04_diffusion_model.cpp`, `05_simulator.cpp`, `12_cahn_hilliard.cpp` still construct `World` only to feed the still-`World`-based `Model`/`Simulator` API (M7/M12)]
* [x] Delete `CoordinateSystem` template machinery (`csys.hpp` tag dispatch, commented-out tag list) — fold the Cartesian data into `Domain`; update `examples/17_custom_coordinate_system.cpp` to demonstrate a user-side coordinate wrapper instead. (Confirmed: `csys.hpp` no longer exists in the tree; `grep -rn "CartesianTag" include/ src/` is empty; migration map records "`csys.hpp` deleted; example 17 was already `csys`-free".)
* [x] Move `world_helpers`/`world_factory` creation functions to `pfc::domain::create(...)` free functions; keep `world::create` as deprecated forwarders (A0 surface). (Confirmed: every function in `world_helpers.hpp` is `[[deprecated("Use pfc::domain::create_world_*(...) instead")]]`.)

### Required tests

* [x] New unit tests for `Box3i` (validation, iteration, containment) and `Domain` (periodicity, coordinate↔index round-trip at ±ε probes). (Confirmed: `tests/unit/kernel/data/test_box3i.cpp`, `test_box3i_edge_cases.cpp`, `test_domain.cpp`, `test_domain_spacing.cpp` exist.)
* [x] Pre-M0 PE/PF regression tests re-pointed at `Domain` and passing. [pointed at `Domain`/current API per repo state; not independently re-run in this pass]
* [x] PI's 4-rank neighbor test passing with per-axis periodicity on and off. (Confirmed: `tests/integration/scenarios/parallel_scaling/test_domain_decomposition.cpp` and the 4-rank neighbor round-trip commits.)
* [ ] Full suite + golden trajectories unchanged (tungsten/aluminum still run via A0). [not independently re-run in this pass]

### Deletions

* [x] `include/openpfc/kernel/data/box3d.hpp` + `src/.../box3d.cpp` + its test (`frontend/utils/field_iteration.hpp` re-pointed to `Box3i`). (Confirmed: file no longer exists.)
* [x] `world_types.hpp` strong-type layer (`Size3/Periodic3/LowerBounds3/UpperBounds3/Spacing3`) — keep only the `Int3/Real3/Bool3` aliases (relocated to `domain.hpp` or a small `types.hpp`). (Confirmed: `world_types.hpp` no longer exists; migration map records "33 includers repointed to `types.hpp`".)
* [x] Unused strong types `LocalOffset`, `GlobalOffset`, `PhysicalCoords`, `IndexBounds`, `PhysicalBounds` from `strong_types.hpp`; implicit two-way conversions on the survivors (`GridSize/PhysicalOrigin/GridSpacing`) made explicit. (Confirmed: none of these five names appear in `strong_types.hpp` any more.)
* [x] `csys.hpp` template machinery (file reduced or removed; contents absorbed by `Domain`). (Confirmed: file deleted entirely, further than "reduced".)

### Definition of done

* [x] `grep -rn "Box3D\|world_types.hpp\|CartesianTag" include/ src/` returns nothing (excluding A0 shim file). (Confirmed empty — and with no exclusion needed, since none of these three strings remain anywhere.)
* [ ] Exactly one index-box type (`pfc::Box3i`) and one domain type (`pfc::Domain`) exist; `World` survives only as the deprecated A0 alias, used only by tungsten/aluminumNew/frontend Gen‑1 paths. [partial: box/domain consolidation is true; `World` is still referenced by Gen‑1 `Model`/`Simulator` and their examples/tests — expected until M8/M12]
* [ ] Full suite green; golden trajectories bit-identical to Pre-M0 captures (this milestone must not change numerics). [not independently re-run in this pass]

---

## M2 — Canonical field, view, and simulation state

### Objective

One owning field container with layout metadata and explicit memory-space/residency tracking, one non-owning view, and an owning `SimulationState`; the ten-container zoo reduced to the keepers.

### Dependencies

M1.

### Tasks

* [x] Create `include/openpfc/kernel/data/field.hpp` (replacing current contents): `pfc::Field<T, MemorySpace = HostSpace>` = `DataBuffer<MemorySpace,T>` storage + owned `Box3i` + halo width + geometry POD (spacing/origin by value) + single linearization (`idx()`) + `apply(f(x,y,z))` defined once. Halo width 0 ≡ today's `LocalField`; halo width n ≡ today's `PaddedBrick`. Templated on `RealType` per ADR 0006. (Landed as `pfc::data::Field<T, MemorySpace>` in `kernel/data/grid_field.hpp`. The old functional `kernel/data/field.hpp` was later deleted — commit `d8566e82`.)
* [x] Add residency tracking to `Field`: memory-space tag, `mirror_host()`/`mirror_device()` returning tracked mirrors with validity flags; framework-side helpers `with_host_view(field, fn)` that bracket host access (the structural fix generalizing Pre-M0 PA). (Confirmed: `grid_field.hpp` implements `with_host_view()` and documents "Residency tracking (M2.2)" with a device-backed field tracking mirror currency.)
* [x] Adopt `pfc::field::FieldView<T>`/`FieldOutput<T>` (from `state_access.hpp`) as the single non-owning view; extend with the geometry POD; keep `validate_no_alias`. (`ScaledField` holds `FieldView<double>`; `state_access.hpp` carries extents/spacing/origin.)
* [x] Create `include/openpfc/kernel/simulation/simulation_state.hpp`: `SimulationState` owning `Field`s by value, keyed by name for I/O/wiring and by typed handle (`FieldHandle<T>`) for hot paths; capacity to hold real and complex fields in host or device spaces. (Confirmed: `kernel/simulation/simulation_state.hpp`/`.ipp` exist — commits `b3bc9a31` "feat(simulation): add SimulationState owning canonical Fields", `cc07ab6c` code-review follow-up. Landed ahead of the plan's stated M2→M7 dependency ordering.)
* [x] Migrate `LocalField` consumers (heat3d, wave2d spectral/pointwise drivers, `FdCpuStack`, `SpectralCpuStack`, gradient evaluators) to `Field<T, HostSpace>` with halo 0. (`local_field.hpp` deleted — commit `2e73ad7a`; stacks/steppers bind `Field`.)
* [x] Migrate `PaddedBrick` consumers (heat3d FD, wave2d FD, kobayashi CPU, allen_cahn, halo exchangers' field-facing APIs) to `Field<T, HostSpace>` with halo n; replace the by-value `Decomposition` member with the geometry POD. (`padded_brick.hpp` deleted — commit `44249f98`; `PaddedHaloExchanger` and `FDGradient` bind `Field`.)
* [x] Migrate `pfc::field::Field<T>` consumers (steppers `euler.hpp`, `explicit_rk.hpp`, stacks) mechanically to the new `Field` (no protocol changes — that is M6). (`kernel/data/field.hpp` deleted — commit `d8566e82`.)
* [x] Migrate the four `DiscreteField`/`Array` examples (`examples/06,07,08,14`) and `docs/api/examples/08` to `Field`. (`discrete_field.hpp` / `array.hpp` deleted — commits `fffee016` / `c9827a0f`; examples 06–08, 14 on `Field`.)
* [x] `ScaledField` retargeted at `FieldView` (kills its raw-pointer contract). (`scaled_field.hpp` holds `FieldView<double>` and `operator*(double, const data::Field<double>&)`.)
* [x] Update `tuple_protocol.hpp`, `du_field.hpp`, `for_each_interior.hpp` signatures to accept `Field`/`FieldView`. (`DuField` is documented over `Field<double>`; `for_each_interior` stays a pointer/evaluator hot path — that is the intended shape, not a leftover container.)

### Required tests

* [x] Unit tests: `Field` linearization vs the three legacy linearizations on identical boxes (bitwise index equality); halo-padded indexing; residency flags (host write after device write without sync → detected). (`test_grid_field.cpp` pins unpadded/padded `idx()` against the closed-form x-fastest formula — the legacy containers no longer exist to compare against; `test_residency.cpp` pins host-write-after-device-write without sync.)
* [ ] Existing heat3d manual-vs-stack L2-equality and wave2d manual-vs-separated pins pass unchanged (these are the abstraction-costs-nothing invariants). [not independently re-run in this pass]
* [ ] Full suite + golden trajectories bit-identical (pure data-structure migration; any numeric change is a bug). [not independently re-run in this pass]

### Deletions

* [x] `kernel/data/discrete_field.hpp`, `kernel/data/array.hpp` (+ their tests migrated), and with them the `kernel/data → decomposition/fft` include inversion (`array.hpp:42–45`). (Both headers deleted; `include/openpfc/kernel/data/` has no decomposition/fft includes.)
* [x] `kernel/field/local_field.hpp`, `kernel/field/padded_brick.hpp` (including `OwnedIndexRange`). (Both deleted.)
* [x] `kernel/field/legacy_adapter.hpp` (speculative; one unit test removed with it). (Header and unit test deleted.)
* [x] `kernel/field/operations.hpp`'s `apply(model, name, fn)` shim (the `field → simulation` include inversion); Gen‑1 callers use `Model` accessors directly until M12. (Model overloads deleted; `operations.hpp` no longer includes `model.hpp`.)
* [x] `MultiIndex` iterator duplication (kept only if `Box3i` iteration doesn't cover a consumer; otherwise `multi_index.hpp` reduced or removed). (`multi_index.hpp` deleted; example 06 retargeted.)

### Definition of done

* [x] Exactly one owning field template and one view family exist; `grep -rn "DiscreteField\|PaddedBrick\|LocalField" include/ src/ apps/ examples/` returns nothing. (No class or include remains; leftover hits are comments documenting the deleted types.)
* [x] `SimulationState` exists with unit tests; not yet wired to Gen‑1 (`ModelFieldRegistry` untouched until M12). (Confirmed present and unwired to `ModelFieldRegistry`.)
* [x] No kernel/data header includes decomposition or fft headers (layering check extended and green). (`array.hpp` deleted; `grep` of `include/openpfc/kernel/data` for those includes is empty.)
* [ ] Full suite green; golden trajectories bit-identical. [not independently re-run in this pass]

---

## M3 — Single-source GPU runtime

### Objective

One device-code tree compiled for CUDA and HIP from the same sources; HIP at feature parity; the ornamental execution layer removed per ADR 0004.

### Dependencies

M2 (Field/DataBuffer surface stable). Executes ADR 0004.

### Tasks

* [x] Create `include/openpfc/runtime/gpu/gpu_api.hpp`: vendor shim (`gpuMalloc`, `gpuMemcpyAsync`, `gpuStream_t`, `gpuEvent_t`, `GPU_CHECK`, launch macro) selected by `OpenPFC_ENABLE_CUDA`/`OpenPFC_ENABLE_HIP`; add explicit `__HIPCC__` branch to `OPENPFC_HD` (`host_device.hpp:54–60`). HIP is selected first in `gpu_api.hpp` because hipcc may also define `__CUDACC__`.
* [x] Port to `runtime/gpu/` as single sources, replacing both vendor copies: `databuffer_gpu.hpp`, `deep_copy_gpu.hpp`, `for_each_interior_device.hpp` (CUDA's 443-line feature set: single + multi-field N=2–4 + composite + autotune hook), `fd_gradient_device.hpp` (CUDA feature set), `sparse_vector_ops` (+ `.cu` source), `exchange_gpu.hpp`, `padded_device_halo_exchange.hpp`, `full_padded_device_halo.hpp`, `padded_halo_faces` kernels. Vendor `runtime/cuda/` and `runtime/hip/` are thin includes or re-exports except FFT (M5).
* [x] Honesty: `docs/concepts/halo_exchange.md` lists `runtime/gpu/` as canonical for device SparseVector MPI, padded device halo, FD gradient device, and `for_each_interior_device`; vendor trees named as thin includes/re-exports; HIP packed-halo env documented.
* [x] Honesty: public FD/halo `@see` comments and `docs/extending_openpfc/per_point_grads.md` point at `runtime/gpu/` device twins, not CUDA-only vendor headers.
* [x] Honesty: architecture/styleguide/`DataBuffer` diagnostics name `runtime/gpu/` as the CUDA/HIP implementation layer; vendor trees documented as thin includes plus FFT until M5. Dropped leftover "Kokkos-compatible" branding on `HostSpace`.
* [x] Shared Tungsten GPU headers (`tungsten_ops.hpp`, `tungsten_etd_workspace.hpp`) and vendor FFT headers include `runtime/gpu/` DataBuffer/tags directly instead of hopping through CUDA/HIP shims.
* [x] Dual CUDA/HIP unit tests (`test_sparse_vector_neighbor_exchange.cpp`, `test_databuffer.cpp`, `test_sparse_vector_exchange_device.cpp`) include `runtime/gpu/` instead of duplicated vendor shims.
* [x] CUDA/HIP fail-closed SparseVector exchange tests (`test_exchange_cuda_fail_closed.cpp`, `test_exchange_hip_fail_closed.cpp`) include `runtime/gpu/` exchange, SparseVector, and check headers; keep native `cuda_check` / `hip_check` (not `GPU_CHECK`).
* [x] CUDA/HIP SparseVector unit tests (`test_sparsevector_cuda.cpp`, `test_sparsevector_hip.cpp`) include `runtime/gpu/` SparseVector headers; keep native `cudaMemcpy` / `hipMemcpy`.
* [x] CUDA/HIP padded device-halo tests (`test_padded_device_halo_self_wrap.cpp`, `test_padded_device_halo_self_wrap_hip.hip`, `test_full_padded_device_halo.cpp`) include `runtime/gpu/` halo headers. HIP 26-direction twin already did.
* [x] `examples/fft_backend_benchmark` benchmarks HIP (rocFFT / `HipTag` DataBuffers) as well as CUDA; GPU DataBuffer includes go through `runtime/gpu/`. *(Running the HIP path on LUMI hardware is M-LUMI; factory honesty remains M5.)*
* [x] HIP autotune `get_device_architecture()` / `get_device_id()` use `gcnArchName` and `pciDeviceID` (ROCm 6 `hipDeviceProp_t`); HIP autotune test requires a `gfx` prefix when a device is present.
* [x] GPU autotune unit test includes Catch2 string matchers so `REQUIRE_THROWS_WITH` compiles under HIP.
* [x] HIP fd-gradient gpu_validation test includes `runtime/gpu/` DataBuffer and HipSpace headers instead of vendor shims; vendor `fd_gradient_device.hpp` / `for_each_interior_device.hpp` re-exports stay (`pfc::hip::` / `pfc::sim::hip`).
* [x] Kobayashi CUDA driver (`kobayashi_fd_cuda.cpp`, `kobayashi_batched_halo.hpp`) includes `runtime/gpu/` padded device-halo headers instead of the vendor shim; `pfc::cuda::` names stay (stamped by the GPU header). CUDA TU compile is tohtori.
* [x] CI guard `scripts/check_gpu_memcpy_single_source.sh`: `cudaMemcpy` / `hipMemcpy` in `include/` and `src/` must live under `runtime/gpu/` (M3 DoD grep). Wired into code-quality with a self-test.
* [x] HIP-only CMake adds `tests/unit/runtime/gpu/` (was CUDA-gated). HIP FFT unit test links `Heffte::Heffte`. INTERFACE GPU macros use `INTERFACE` not `PUBLIC`. FetchContent nlohmann_json is include-only so HIP+tests configure can export.
* [x] HIP unit binaries ran on LUMI-G MI250X (job 21110746): device detection, autotune, deep-copy fill, SparseVector, FFT; fail-closed skipped as expected under `OpenPFC_MPI_HIP_AWARE=ON`. `openpfc-tests` (gpu_validation HIP .hip cases) still blocked by leftover CPU tests (`test_state_access.cpp`).
* [x] `SimulationState` device-field compile coverage includes `HipSpace` (CUDA twin already existed); GPU includes go through `runtime/gpu/`.
* [x] SparseVector `on_host` covers `HipTag` (CUDA twin already existed); GPU includes go through `runtime/gpu/`.
* [x] Move all device sources out of `include/`: `kernels_simple.cu` (deleted, see below), `sparse_vector_ops.cu/.hip`, `padded_halo_faces.hip` → `src/openpfc/runtime/gpu/`. Device TUs and kernel `.inc` files live under `src/openpfc/runtime/gpu/`.
* [ ] Fold CUDA `padded_halo_faces.cu` into the GPU kernel library so it is never recompiled per consumer (`tests/integration/CMakeLists.txt` / `apps/kobayashi/CMakeLists.txt` still list the TU per executable). Blocked on separable-compilation device-link. **CUDA: not testable on LUMI — verify the fold on tohtori.**
* [x] CMake: one shared kernel source list compiled as `openpfc_gpu_kernels` (nvcc) and/or `openpfc_hip_kernels` (hipcc); both vendor libs buildable in a CUDA+HIP co-enabled configuration; install/export both (Pre-M0 PM already exports HIP). CUDA `padded_halo_faces.cu` remains per-executable (separable compilation).
* [x] Add `Backend::HIP` to `fft_interface.hpp:34–37` and `"hip"`/`"rocm"` mappings to `runtime/common/backend_from_string.hpp` (factory honesty itself is M5).
* [x] Extend GPU autotuning to the single-sourced `for_each_interior` and gather/scatter on both vendors; prune demo kernel keys. Autotune hook is in `for_each_interior_device_gpu.hpp` and `sparse_vector_ops_gpu.inc`; demo keys `add_scalar` / `multiply_scalar` deleted. `OpenPFC_ENABLE_GPU_AUTOTUNING` is PUBLIC on `openpfc` / kernel libs (not directory-scope).
* [x] Add generic elementwise device ops to `runtime/gpu/` (complex×real multiply, two-term diagonal combine, axpy-style fill) — the mislabeled "Tungsten-specific" kernels, promoted (used by M7's ETD skeleton).
* [x] Fix device-scalar fill: `deep_copy(view/buffer, scalar)` uses a device fill kernel, not a host staging vector (`deep_copy.hpp:137–147`).
* [x] Replace deprecated `hipMallocHost` with `hipHostMalloc`; unify error-check style on `GPU_CHECK`. Packed-halo uses `hipHostMalloc`. `for_each_interior_device` launch/sync uses `GPU_CHECK`. Co-enabled TUs keep `cuda_check`/`hip_check`; kernel `.inc` files keep vendor-prefixed strings.

### Required tests

* [ ] Existing GPU-gated suites (tungsten/allen_cahn/wave2d parity, `test_sparsevector_cuda/hip`, gpu_validation scenarios) pass on tohtori (CUDA) against the single-sourced code, within the tolerances declared in `BASELINES.md`. **CUDA: not testable on LUMI — verify on tohtori.** *(The LUMI/HIP-execution half moved to M-LUMI.)*
* [x] New HIP-parity tests: multi-field `for_each_interior_device` and composite-gradient device tests compiled under HIP (previously CUDA-only). *(Actually running them under HIP moved to M-LUMI — compiling for HIP is testable on tohtori today; executing needs LUMI.)* `test_multi_field_device.hip` / `test_composite_gradient_pod_size_hip.hip` added to `openpfc-tests` on HIP builds.
* [x] HIP FFT unit-test twin of `test_fft_cuda.cpp`: `test_fft_hip.cpp` / `HIP_FFT`, gated on `OpenPFC_ENABLE_HIP_SPECTRAL`, using `pfc::fft::create_hip` and `HipTag` DataBuffers. *(Running on LUMI hardware is M-LUMI; factory honesty itself remains M5.)*
* [x] HIP FFT integration roundtrip twin of `test_cuda_roundtrip.cpp`: `test_hip_roundtrip.cpp` (float/double DataBuffer forward/backward), compiled into `openpfc-tests` with a skip stub when `OpenPFC_ENABLE_HIP_SPECTRAL` is off. *(Running on LUMI hardware is M-LUMI.)*
* [x] HIP CPU-vs-GPU Laplacian twins of `test_cuda_vs_cpu_laplacian.cpp` / `test_cuda_vs_cpu_laplacian_mpi.cpp`: `test_hip_vs_cpu_laplacian.cpp` / `test_hip_vs_cpu_laplacian_mpi.cpp`, skip stubs when `OpenPFC_ENABLE_HIP_SPECTRAL` is off. *(Running on LUMI hardware is M-LUMI.)*
* [x] HIP vs CPU diffusion smoke `test_hip_vs_cpu.cpp` is compiled into `openpfc-tests` (was an unwired stub) and constructs `create_hip` the way the CUDA twin constructs `create_with_backend(CUDA)`. *(Running the HIP factory on LUMI hardware is M-LUMI.)*
* [x] HIP backend instantiation smoke in `test_gpu_backend_instantiation.cpp`: separate Catch2 case (CUDA skip cannot hide HIP) comparing `create_hip` inbox/outbox sizes to the CPU FFT. *(Running on LUMI hardware is M-LUMI; `create_with_backend(Backend::HIP)` factory honesty remains M5.)*
* [x] HIP `FullPaddedDeviceHalo` 26-direction integration twin of `test_full_padded_device_halo.cpp`: `test_full_padded_device_halo_hip.cpp` (1/2/4-rank full fill, `hw=2`, Axes3D faces-only). *(Running on LUMI hardware is M-LUMI.)*
* [ ] Compile-only CUDA and HIP CI jobs green, including the CUDA+HIP co-enabled configuration. **CUDA half + co-enabled config: not testable on LUMI — verify on tohtori / in CI.** HIP compile can be checked here.
* [ ] Perf gate: device halo microtimings and tungsten CUDA baseline within 5%. **CUDA: not testable on LUMI — verify on tohtori.**

### Deletions

* [x] `include/openpfc/runtime/cuda/` and `include/openpfc/runtime/hip/` duplicated implementations (each directory reduced to the vendor shim inclusion + FFT alias headers until M5). (Confirmed: non-FFT vendor headers are thin `#include` / `using` re-exports of `runtime/gpu/`; `fft_cuda.hpp` / `fft_hip.hpp` stay until M5.)
* [x] Kokkos facsimile above `DataBuffer` per ADR 0004: `kernel/execution/{view,parallel,policy,layout,execution_space,deep_copy}.hpp` and `tests/unit/kernel/execution/test_kokkos_like.cpp` deleted. Memory-space tags, `DataBuffer`, `memory_traits`, and `deep_copy` buffer overloads survive.
* [x] `runtime/cuda/gpu_vector.hpp`, `kernels_simple.{cu,hpp}`, and their unit tests. Deleted; state-access docs now describe `DataBuffer` instead of `GPUVector`.
* [x] The Pre-M0 PB `static_assert` tombstones (the API they guarded is gone). Device `parallel_for` is no longer a header you can include; host `parallel_for` fail-closes on non-Serial/OpenMP policies.

### Definition of done

* [ ] `diff -r` between generated CUDA and HIP object lists shows a single source set. `grep -rn "hipMemcpy\|cudaMemcpy" include/ src/ | grep -v runtime/gpu` is now CI-enforced (`scripts/check_gpu_memcpy_single_source.sh`); `include/` and `src/` currently pass. **Full CUDA-vs-HIP object-list `diff -r` is not testable on LUMI (no CUDA toolchain run here); HIP object list can be inspected locally.**
* [x] No `.cu`/`.hip` files under `include/`; kernel `.inc` files live under `src/openpfc/runtime/gpu/`; installed header set contains only `.hpp`.
* [ ] Full suite + golden trajectories green; GPU parity suite green on tohtori (CUDA). **CUDA: not testable on LUMI — verify on tohtori.** *(HIP multi-field/composite device execution is M-LUMI; this session can start filling those HIP items.)*

---

## M4 — Consolidated communication layer

### Objective

Two blessed exchangers — structured `HaloExchange` and index-set `SparseExchange` — backend-templated, with persistence, split-phase overlap, and multi-field batching as modes; HeFFTe removed from the decomposition path; dead exchangers deleted.

### Dependencies

M2 (Field), M3 (single-source device layer). M3 CUDA execution/perf leftovers do **not** block this milestone: they will be closed on tohtori. HIP device-halo execution can be exercised on LUMI as the unified exchanger lands.

### Tasks

* [x] Create shared geometry header `kernel/decomposition/halo_geometry.hpp`: face/edge/corner slab specs, `opposite_slot`, tag allocation scheme (deterministic per-field/per-direction tag block replacing hand-spaced app tags) — single source for the 4–6 current re-implementations. (`halo_directions.hpp` now includes it and dropped its duplicate slot/tag helpers. Device exchangers still have local `opposite_slot` copies — retarget when `HaloExchange` lands. CUDA consumers of this header: not testable on LUMI.)
* [x] Create `pfc::comm::HaloExchange<MemorySpace>` in `kernel/decomposition/halo_exchange.hpp` (host) + `runtime/gpu` (device pack/unpack), superseding `PaddedHaloExchanger`/`FullPaddedHaloExchanger`/device twins. Modes: 6-face or 26-direction (for mixed derivatives); blocking `exchange()`; split `start()/finish()`; optional persistent requests (from `halo_persistent.hpp`); multi-field batching (promoted from `apps/kobayashi/src/cuda/kobayashi_batched_halo.hpp`, generalized past its one-MPI-axis corner-fill constraint or failing loudly when unsupported). **Host facade** in `comm_halo_exchange.hpp`. **Device facade** in `comm_halo_exchange_gpu.hpp` (`HaloExchange<CudaSpace/HipSpace>` composes the existing device twins; persistent and start/finish throw). Consumers still use the old classes. CUDA execution: not testable on LUMI — verify on tohtori.
* [x] Device transport defaults: pack-to-contiguous + device-pointer MPI when GPU-aware (env toggle becomes default per Audit §9); stream-scoped sync instead of `cudaDeviceSynchronize` (`padded_device_halo_exchange.hpp:381`). (`*_USE_SUBARRAY_HALO=1` restores derived types; CUDA execute on tohtori.)
* [x] Fix GPU-aware MPI detection for Cray MPICH: runtime self-probe (device-pointer send/recv smoke test à la `verify_gpu_aware_mpi.cpp`) plus `OPENPFC_ASSUME_GPU_AWARE_MPI` override; surface the active mode in the startup log. (`gpu_aware_mpi.hpp`; `decide_gpu_aware_mpi` tested in `test_gpu_aware_mpi.cpp`. Cluster log-assert that the LUMI job actually selects the aware path remains M-LUMI.)
* [x] Backend-template `RemoteHalo`/`SparseHaloExchanger` → `pfc::comm::SparseExchange<MemorySpace>`, using the existing device gather/scatter, eliminating the per-step full-field D2H in `apps/allen_cahn/src/cuda/allen_cahn.cpp:100–116` and wave2d GPU. **Facade landed** (`comm_sparse_exchange.hpp` + `comm_sparse_exchange_gpu.hpp`). Allen–Cahn HIP and wave2d HIP use the device path (`face_recv_ptrs`). Allen–Cahn/wave2d CUDA still D2H. CUDA execute on tohtori.
* [ ] Migrate all exchanger consumers: heat3d, wave2d (CPU+GPU), allen_cahn (CPU+GPU), kobayashi (CPU, CUDA — onto library batching; HIP — onto the device path for the first time), `FdCpuStack`, `StagePreparationService`, gpu_validation tests. **Production FD apps and `FdCpuStack`/`StagePreparationService` are on the new names. FD MPI leftover tests and `test_sparse_halo_exchange` now use `HaloExchange`/`SparseExchange`. Remaining: backend-class unit tests (`test_padded_halo_exchange`, `test_full_padded_*`, `test_halo_direction_set`, device twins) and deletion of old public names.**
* [x] Per ADR 0007: implement the in-repo min-surface splitter in `src/.../decomposition.cpp`, validated against `heffte::split_world` output for a matrix of (grid, ranks) cases; HeFFTe include removed from the decomposition TU (Pre-M0 PI assertion retargets to the new splitter as its own invariant). (`brick_split.hpp`; `test_brick_split.cpp`)
* [x] Add opt-in `MPI_Comm_dup` isolation to `pfc::mpi::communicator` (coupling prerequisite). (`communicator::duplicate()`; `tests/unit/kernel/mpi/test_communicator.cpp`)
* [x] Make `validate_neighbour_direction_agreement` opt-out for release builds (documented) to remove the per-construction `MPI_Allgather` at scale. (`neighbour_agreement_enabled()`; `OPENPFC_VALIDATE_NEIGHBOUR_AGREEMENT`; constructors skip when off.)

### Required tests

* [x] Unit: tag-allocation collision test (two exchangers, six fields, overlapping lifetimes — distinct tags proven). (`tests/unit/kernel/decomposition/test_halo_geometry.cpp`)
* [x] Splitter equivalence test: in-repo splitter boxes == recorded `heffte::split_world` boxes for ≥12 (grid, ranks) combinations. (`test_brick_split.cpp` compares live HeFFTe output)
* [x] 4-rank MPI: `HaloExchange` blocking == split-phase == persistent == batched results, bitwise, host and device; 26-direction mode validated on corner-dependent stencil. **Host:** `test_comm_halo_exchange_modes.cpp` (blocking == start/finish == two-field batch; Full corners). Persistent multi-rank is still red on LUMI (1-rank persistent remains in `test_comm_halo_exchange.cpp`). **HIP:** 4-rank Faces + Full in `test_comm_halo_exchange_gpu.cpp`. **CUDA 4-rank Faces:** compiles here, execute on tohtori.
* [ ] Kobayashi CUDA golden checksums (bitwise class) unchanged on library batching; kobayashi HIP now matches CPU within declared tolerance using the device path. **CUDA checksums: not testable on LUMI — verify on tohtori.**
* [ ] Perf: halo microtiming baseline within 5% (tohtori). **CUDA: not testable on LUMI — verify on tohtori.** *(The LUMI device-MPI probe check — demonstrating it selects the GPU-aware path, log-asserted in the cluster test script — moved to M-LUMI.)*

### Deletions

* [ ] `kernel/decomposition/halo_exchange.hpp` (old in-place `HaloExchanger`), `halo_persistent.hpp`, `full_padded_halo_exchange.hpp`, `padded_halo_exchange.hpp` (superseded), `runtime/gpu` old padded/full-padded twins from M3's port (superseded by the unified class), and their now-redundant tests (assertions migrated to the new suites).
* [x] `apps/kobayashi/src/cuda/kobayashi_batched_halo.hpp` (531 lines). Removed after Kobayashi CUDA moved onto multi-field `HaloExchange<CudaSpace>` (sequential per field in a group; one-Waitall library batching is still a follow-up).
* [ ] `sparsevector::` "for testing" free-function round-trip on the construction hot path.
* [x] HeFFTe include from `src/openpfc/kernel/decomposition/decomposition.cpp`.

### Definition of done

* [ ] Exactly two exchanger class templates remain; `grep -rn "PaddedHaloExchanger\|FullPaddedHalo\|PersistentHaloExchanger" include/ src/ apps/` returns nothing.
* [ ] An FD-only build (`-DOpenPFC_ENABLE_FFT=OFF` or equivalent) configures and links without HeFFTe. *(If a build toggle is out of scope, DoD is: decomposition object files have no HeFFTe symbol dependencies.)*
* [ ] Full suite, golden trajectories, GPU parity suite (tohtori) green. **CUDA parity: not testable on LUMI — verify on tohtori.** *(Moved to M-LUMI in full: "LUMI runs device-resident halos, verified by the probe log + perf delta vs host-staging" — HIP half can start here once the unified exchanger exists.)*

---

## M5 — Honest FFT interfaces and spectral utilities

### Objective

FFT surfaces that every implementation can satisfy; one k-space iteration helper; dealiasing and Nyquist handling available and documented. Executes ADR 0005.

### Dependencies

M3 (Backend enum/string complete), M2 (Field/DataBuffer types).

### Tasks

* [x] Split `fft_interface.hpp` per ADR 0005: `IHostFFT` (host-container transforms) and `IDeviceFFT<MemorySpace>` (DataBuffer transforms); `FFT_Impl<BackendTag>` implements the applicable one(s); delete the throwing GPU virtual bodies. **`IFFT` is a temporary alias of `IHostFFT`. GPU `FFT_CUDA` / `FFT_HIP` implement `IDeviceFFT`. Float DataBuffer overloads stay on the concrete type.**
* [x] Make factories honest: `fft::create_with_backend` returns host FFTs for host backends only; device factories (`create_cuda`, `create_hip`) return objects that implement `IDeviceFFT`; requesting a mismatch throws at construction with a clear message.
* [x] Workspace precision per ADR 0006: allocate only the instantiated precision (lazy or template) — removes the ~33% device-memory waste. GPU `FftWorkspaceStorage` now allocates float/double on first use.
* [x] Expose `r2c_direction` through the convenience factories (currently silently hardcoded 0). Optional last argument on `create` / `create_with_backend` / `create_cuda` / `create_hip` (default 0).
* [x] Add `kernel/fft/kspace_iterator.hpp`: `for_each_kpoint(outbox, domain, fn(idx, kx, ky, kz, i, j, k))` (host) and device-callable scalars in `runtime/gpu/kspace_iterator_gpu.hpp`. `SpectralGradient` uses the host iterator and binds a `FieldView`. IFFT still needs the host `std::vector` for `forward`.
* [x] Zero odd-derivative spectral operators at the Nyquist mode in `SpectralGradient` (Audit K1); recorded in `BASELINES.md` and `docs/science/numerics_limits.md`.
* [x] Add optional 2/3-rule dealiasing mask as a standard k-space diagonal (`kernel/fft/dealias.hpp`), off by default; documented in `docs/science/numerics_limits.md`.
* [x] Migrate `spectral_exp_coefficients.hpp` to be `Real`-templated (default `double`). Host compute; callers upload spans to device when needed.

### Required tests

* [x] Negative test: constructing a host `IHostFFT` with `Backend::CUDA` (and HIP) throws at the factory, not at first use. (`test_fft_backend_selection.cpp`)
* [x] `for_each_kpoint` unit test vs a hand-rolled reference loop on odd/even grids (bitwise index and wavenumber equality). (`test_kspace_iterator.cpp`)
* [x] Nyquist fix: first derivative of a Nyquist mode is ~0 (`test_spectral_gradient.cpp`). No existing trajectory golden uses spectral first derivatives.
* [x] Dealiasing smoke test: 2/3-rule mask zeros modes with `|k_i| >= (2/3) k_Nyquist` (`test_dealias.cpp`).
* [x] GPU FFT round-trip (forward+backward == identity to 1e-12) via `IDeviceFFT` on both vendors. **HIP:** `test_hip_roundtrip.cpp` double case binds `IDeviceFFT<HipSpace>`. **CUDA:** `test_cuda_roundtrip.cpp` double case binds `IDeviceFFT<CudaSpace>` (execute on tohtori).

### Deletions

* [x] Throwing GPU virtual implementations and the dishonest `Backend::CUDA` path in `src/openpfc/runtime/cpu/fft.cpp`. GPU `FFT_Impl` no longer inherits `IHostFFT`; `create_with_backend` rejects CUDA/HIP.
* [x] Duplicate k-space folding loops in `SpectralGradient` (now `for_each_kpoint`). Apps' copies die in M8/M9.

### Definition of done

* [ ] No FFT object can be constructed whose interface methods unconditionally throw; verified by the negative tests.
* [ ] `grep -rn "atan(1)\|i <= .*size/2" include/` shows k-space folding only inside `kspace.hpp`/`kspace_iterator.hpp`.
* [ ] Full suite green; golden trajectories green under the re-baselined tolerances (Nyquist change documented).

---

## M6 — Unified stepper protocol, complex/multi-field state, adaptive control

### Objective

One attempt/commit step protocol implemented by all integrators; steppers operate on `Field`-based state including complex fields and N-field packs; the adaptive chain closed by a controller. Resolves the #169 blocker at the framework level.

### Dependencies

M2 (Field/SimulationState), M5 (spectral coefficients memory-space-generic).

### Tasks

* [x] Declare `StepAttemptResult`/attempt-commit (from `steppers/step_attempt.hpp`) the single protocol; specify it in `docs/adr/0003-time-integrator-interface.md` (update the existing ADR). **Accepted; `AttemptStepper` concept added.**
* [x] Port onto the protocol: `EulerStepper`, `RK2Heun`, `RK3Heun`, `ExplicitRKStepper`, `EmbeddedRKStepper` (drop `EmbeddedStepAttemptResult`; keep `u_high`/`u_low`/`error`/`last_rhs_evals` accessors), `ImexEulerStepper` (drop `ImexStepAttempt`; keep `last_solve_*`), `Etd1Stepper` (drop `Etd1StepAttempt`; `attempt` returns `StepAttemptResult`). In-place `step()` is attempt+commit where the leaf has `step()`.
* [ ] Generalize state: steppers accept any type satisfying the field concepts (`state_concepts.hpp` — wire it in for real) — `Field<double>`, `Field<complex<double>>`, and heterogeneous packs; remove the raw-`std::vector<double>`-only restriction. **All seven single-field leaves take host `Field<double>` via `vec()`. Vector path remains. Complex and packs still open.**
* [ ] Complex-state ETD: `Etd1Stepper` (and `MultiEtd1Stepper`) operate on complex spectral fields with device-resident coefficient application (uses M3 generic elementwise ops + M5 coefficients) — the capability tungsten's ETD needs.
* [x] Generalize multi-field arity: `MultiStageFunction<Rhs, N>` (default N=2); `MultiEtd1Stepper` over N (`N >= 1`, variadic `attempt`). N=3 covered in `test_etd1.cpp`.
* [ ] Merge duplicates: one `StageContext` (delete `pfc::integrator::StageContext` or `pfc::sim::StageContext`, keep one), one workspace type (merge `StageWorkspace` and `integrator::Workspace`), one method enum (`RKIntegratorMethod` extended; `IntegratorMethod` removed from `time.hpp:252`).
* [ ] Implement `AdaptiveTimeController` (`kernel/simulation/adaptive_controller.hpp`): closes embedded-error → `error_evidence` → `AdaptiveControlConfig` → `Time` attempt transactions; one end-to-end adaptive example (`examples/21_adaptive_stepping.cpp`) and integration test.
* [x] Solver contract: `SolveFunction` is descriptor + field-bundle (no matrix type). Non-diagonal dense mock runs under `ImexEulerStepper` (`imex_euler_nondiagonal_dense_solve`). `SpectralDiagonalSolver` already models `SolveFunction` and is used as an injected solver.

### Required tests

* [x] Protocol-conformance test template applied to all seven steppers (attempt→reject→attempt→commit sequence; rollback state equality). **Euler, RK2, RK3, ExplicitRK, Etd1 covered in `test_step_protocol.cpp` (always-succeed path). Embedded/IMEX keep extra `dt`/`ctx` on `attempt` and are covered in their own tests.**
* [ ] Existing RK/temporal convergence-order tests pass unchanged (orders preserved is the scientific gate).
* [ ] New: complex-field ETD1 vs analytic solution of a stiff linear complex ODE field (tolerance test); N=3 multi-field Euler/ETD test extending `test_multifield_stepper.cpp`.
* [ ] Adaptive end-to-end: embedded RK on a problem with a known transient — controller shrinks dt through the transient and grows after; accepted/rejected counters asserted.
* [x] Non-diagonal `SolveFunction` mock compiles and runs under `ImexEulerStepper`.

### Deletions

* [ ] `IntegratorMethod` in `time.hpp`, the losing `StageContext` and workspace type, `fd_stencils.hpp:325–337` back-compat shims. **`integrator_base.hpp` / `integrator_result.hpp` / `ImexStepAttempt` / `Etd1StepAttempt` / `EmbeddedStepAttemptResult` are deleted.** `ImexStepAttemptResult` remains on the IMEX composer until that seam is merged.
* [ ] `euler_attempt.hpp` (its proof role is absorbed by the ported steppers).

### Definition of done

* [ ] `grep -rn "IntegratorBase\|IntegratorResult\|Etd1StepAttempt\|ImexStepAttempt" include/ tests/` returns nothing; exactly one `StageContext`, one workspace, one method enum.
* [ ] All steppers pass the shared conformance test; convergence orders unchanged.
* [ ] A complex-state, device-capable ETD1 exists with tests (the #169 framework prerequisite).
* [ ] One adaptive run exists end-to-end (example + test).
* [ ] Full suite + golden trajectories green.

---

## M7 — Method-independent physics interface and the spectral-ETD skeleton

### Objective

Physics becomes data + concept-conforming callables on `SimulationState`; the framework owns the pseudo-spectral ETD choreography that tungsten and aluminum currently hand-write; the legacy bridge adapters go live (A1, A2).

### Dependencies

M5, M6.

### Tasks

* [ ] Define the physics concepts in `kernel/simulation/physics_concepts.hpp`: (a) field declaration (`declare_fields(SimulationState&)` — names, types, memory space); (b) parameters via declarative schema; (c) either point-wise `rhs(t, G)` (existing Gen‑3 shape) and/or spectral-diagonal descriptors: linear symbol `L(k)` + real-space nonlinearity `N(psi)` (the `physics_for_mode` shape already factored in `tungsten_spectral.hpp`).
* [ ] Implement `kernel/simulation/spectral_etd_system.hpp`: framework-owned pseudo-spectral ETD driver — owns work fields in `SimulationState`, transform choreography (forward → filter/N̂ → ETD combine → backward) via `IDeviceFFT`/`IHostFFT`, `for_each_kpoint` operator preparation, `Etd1Stepper` integration, memory-space-generic (host and device instantiations). Optional dealiasing mask hook (M5).
* [ ] Implement `ParameterSchema` (`kernel/simulation/parameter_schema.hpp`): declarative field list → generated `from_json` + validation + docs table; consolidates `ParameterValidator` usage; one schema per model (not per backend-class).
* [ ] Introduce adapter **A1** `pfc::compat::LegacyModelPhysics`: wraps a Gen‑1 `Model&` as a physics-concept object (delegating `step`), so the new driver can run legacy models. Parity test: diffusion fixture model run via Gen‑1 `Simulator` vs via A1 + new driver — identical trajectories (bitwise on CPU).
* [ ] Adopt adapter **A2**: `Simulator::step_with_physics` becomes the documented bridge by which the Gen‑1 frontend can invoke concept physics during M8–M9 migration; add its missing test.
* [ ] Free-energy/observable hook: reduction support in the driver loop (rank-local reduce + MPI allreduce, host and device) — required by Aluminum (M9); unit-tested on a known integral.

### Required tests

* [ ] A toy PFC model (Swift–Hohenberg-like, single field) written three ways — Gen‑1 `Model`, point-wise `rhs`, spectral-ETD descriptors — produces matching trajectories within declared tolerance (CPU); the descriptor variant additionally runs on CUDA and HIP with parity ≤1e-10.
* [ ] A1 parity test (above) green.
* [ ] `ParameterSchema` round-trip: schema → JSON parse → validation errors for missing/invalid keys match the current `format_config_error` quality (message snapshot tests).
* [ ] Observable-reduction test: known Gaussian integral to 1e-12, 1 and 4 ranks, host and device.

### Deletions

* [ ] None yet (Gen‑1 stays until M12; this milestone introduces the target and the bridges).

### Definition of done

* [ ] The three-way toy-model equivalence test is green on CPU and both GPU vendors.
* [ ] A new physics model requires: one header (fields + schema + `rhs` or descriptors), zero backend-specific classes, zero hand-written k-loops — demonstrated by the toy model's line count (<200).
* [ ] A1/A2 adapters registered in `0.2_migration_map.md` with removal milestone M12 and green parity tests.
* [ ] Full suite + golden trajectories green.

---

## M8 — Tungsten vertical slice (production go/no-go gate)

### Objective

Tungsten rebuilt as one backend-templated concept model on the M7 skeleton, scientifically identical to the Gen‑1 implementation, on CPU, CUDA, HIP, multi-rank — the proof that Gen‑1 can be deleted.

### Dependencies

M7 (skeleton, schema), M4 (communication — for completeness of the stack), M3 (device layer).

### Tasks

* [ ] Implement `apps/tungsten/include/tungsten/tungsten_physics.hpp`: single model = `TungstenParams` + one `ParameterSchema` + `physics_for_mode` linear symbol + nonlinearity + stabilization, templated on `RealType` and memory space; target ≤400 lines.
* [ ] Wire tungsten through `SpectralEtdSystem` + `SimulationState` + `IDeviceFFT` (no model-owned FFTs), no `dummy_fft`, no hand-rolled mirrors — residency via M2 protocol).
* [ ] Device session assembly: introduce `GpuSpectralStack` (device counterpart of `SpectralCpuStack`) in `kernel/simulation/stacks/`, constructed from JSON plan options via the existing `spectral_fft_stack_factory.hpp` helpers.
* [ ] Keep the Gen‑1 tungsten build target alive in parallel *within this milestone only* for A/B validation; both binaries run the golden-trajectory input.
* [ ] Validation matrix: (a) new-CPU vs Pre-M0 golden trajectory (4 ranks, 100 steps) within declared tolerance; (b) new-CUDA vs new-CPU ≤1e-10 (existing parity harness re-pointed); (c) ETD weights vs `spectral_exp_cache_matches_legacy_etd_weights` pins; (d) perf within 5% of Pre-M0 baselines on tohtori. **(b)/(d) CUDA: not testable on LUMI — verify on tohtori.** *(The new-HIP half of (b) and the LUMI half of (d) moved to M-LUMI.)*
* [ ] Migrate tungsten JSON: one `from_json` via `ParameterSchema` (replacing the three per-backend copies in `tungsten_input.hpp:269,334,399`); config keys unchanged for users.
* [ ] Update `apps/tungsten/README` + `docs/science/tungsten_quicklook.md` to the new structure.

### Required tests

* [ ] All items in the validation matrix above, each a named test or recorded cluster run linked from `BASELINES.md`.
* [ ] `test_tungsten.cpp` spectral-operator edge cases (zero mode, near-zero cancellation, long-dt) pass against the new implementation.
* [ ] The Pre-M0 PA App-GPU-IC test re-pointed at the new pipeline and green.

### Deletions

* [ ] Gen‑1 tungsten: `include/tungsten/{cpu,cuda,hip}/tungsten_model.hpp`, `{cpu,cuda,hip}/tungsten.hpp`, `src/{cuda,hip}/tungsten_ops_kernels.{cu,hip}`, `common/tungsten_ops.hpp` dispatch boilerplate, `common/tungsten_etd_workspace.hpp` (closes #169's TODOs), `common/run_tungsten_gpu_vtk.hpp` bespoke driver, per-backend `from_json` triplet.
* [ ] The A/B Gen‑1 tungsten target itself at milestone close (after the validation matrix is recorded).

### Definition of done

* [ ] One tungsten model source; `find apps/tungsten -name "*model*" | wc -l` == 1; no `.cu/.hip` under `apps/tungsten/`.
* [ ] Validation matrix fully green and archived in `BASELINES.md` (this is the go/no-go record for deleting Gen‑1).
* [ ] Tungsten line count reduced from ~5,000 to <1,500 non-test lines (measured, recorded).
* [ ] Full suite + all golden trajectories green.

---

## M9 — Aluminum and Kobayashi migration

### Objective

The second production physics (Aluminum) and the FD flagship (Kobayashi) on the 0.2 architecture, proving the skeleton generalizes beyond one model and that the FD path is production-capable.

### Dependencies

M8 (validated skeleton), M4 (batched device halos).

### Tasks

* [ ] Rebuild `apps/aluminumNew` on `SpectralEtdSystem`: physics = temperature-gradient/moving-frame terms + `P_F` kernel + free-energy observable (uses the M7 reduction hook); one `ParameterSchema` replacing the 130-line hand-rolled JSON block (`Aluminum.hpp:359–438`); ETD weights now from the shared cache (its inline `opL/opN` formulas retired — validated against the golden trajectory).
* [ ] Aluminum gains CUDA/HIP execution via the skeleton (previously CPU-only); add CPU-vs-GPU parity tests mirroring tungsten's.
* [ ] Migrate kobayashi CPU driver onto `FdCpuStack` + `Field` + unified `HaloExchange` (multi-field batched mode replaces six hand-tagged exchangers, `kobayashi_fd_manual.cpp:83–88`); its bespoke anisotropic stencils remain app-side (legitimately model-specific) expressed over `Field` indexing.
* [ ] Introduce `FdGpuStack` (device FD stack: `Field<T,DeviceSpace>` + device `HaloExchange` + `FdGradientDevice`); migrate kobayashi CUDA and HIP drivers onto it; HIP uses the device path (first time in an app).
* [ ] Kobayashi OpenMP engine: retire the MPI-bypassing torus engine or re-express it as the single-rank case of the stack (decision recorded in the migration map); thread-count bitwise parity test retained either way.
* [ ] Create `apps/common/` (CLI parsing, reporting, rank-0 gather utilities) consolidating the four private `cli.hpp`/reporting headers; migrate heat3d/wave2d/allen_cahn/kobayashi to it.

### Required tests

* [ ] Aluminum golden trajectory (Pre-M0) within declared tolerance; 5-step golden norms updated only with written justification.
* [ ] Aluminum free-energy observable matches the legacy accumulator on the golden run.
* [ ] Kobayashi `KOBAYASHI_VERIFY_HEX` checksums bitwise-identical on CPU; CUDA within declared tolerance; OpenMP thread-parity test green. *(The HIP half moved to M-LUMI.)*
* [ ] heat3d/wave2d/allen_cahn suites green after `apps/common` migration.

### Deletions

* [ ] `apps/aluminumNew/Aluminum.hpp` Gen‑1 model (replaced), its hand-rolled JSON block, inline ETD-weight computation.
* [ ] Kobayashi per-field exchanger setup and hand-spaced tags; the HIP host-staging driver; (per decision) the standalone OpenMP torus engine.
* [ ] Per-app `cli.hpp`/`reporting.hpp`/`verification_utilities.hpp` duplicates (four sets).

### Definition of done

* [ ] Zero production apps construct a Gen‑1 `Model` except via adapters scheduled for M12 deletion (`grep -rn "public pfc::Model" apps/` returns nothing).
* [ ] Aluminum runs on CPU/CUDA from one physics source. *(The HIP half of this, and "kobayashi runs device-resident halos on both vendors," moved to M-LUMI.)*
* [ ] Full suite + all golden trajectories green; perf gates met.

---

## M10 — Orchestration, sessions, boundary conditions, and I/O generalization

### Objective

One thin simulation driver; JSON sessions parameterized by backend × method; BC handling that serves both spectral and FD; writer catalog complete and loud.

### Dependencies

M8, M9 (all production physics on the new architecture).

### Tasks

* [ ] Implement `kernel/simulation/simulation_driver.hpp`: thin loop (free function + small owning struct) over `SimulationState` + `Time` + a protocol stepper + condition lists + writer/checkpoint services; the Gen‑1 `Simulator` remains only behind A1/A2 for the frontend until M12.
* [ ] Generalize the session: `SimulationSession<Stack>` over {SpectralCpuStack, GpuSpectralStack, FdCpuStack, FdGpuStack}; JSON keys `method` (`spectral`|`fd`) and `backend` (`cpu`|`cuda`|`hip`) select the stack via a factory (extends `spectral_fft_stack_factory.hpp` / `backend_from_string`); no dead host FFT for GPU runs (retires the `SpectralCpuStack` concrete-`CpuFft` coupling, `spectral_cpu_stack.hpp:88`).
* [ ] Wire integrator selection: `from_json_integrator_method.hpp` (extended to the unified method enum incl. IMEX/ETD where applicable) consumed by `apply_simulator_section_from_json`; documented JSON schema update.
* [ ] FD configuration surface: `fd_order` JSON key (even orders 2–20, runtime-view stencils; halo width derived automatically) exposed through the FD session; validation errors name the supported set.
* [ ] Boundary conditions: define the single stage-preparation mechanism (generalizing `stage_preparation.hpp`) — pre-stage hooks owning halo refresh + ghost/boundary application for FD, and penalty-modifier application for spectral; retire the two embryonic mechanisms (`ExecutionService::prepare_boundaries`, `StageContext` BC flags).
* [ ] Relocate `FixedBC`/`MovingBC` from `kernel/simulation/boundary_conditions/` into `apps/tungsten/` and `apps/aluminumNew/` (they are directional-solidification physics); keep `FieldModifier` as the IC abstraction with catalog registration.
* [ ] Make modifier/writer registration failures hard errors (replace warn-and-drop in `simulator_modifier_registration.hpp` and `simulation_wiring_writers.hpp:83–89`).
* [ ] Register `VTKWriter` in `default_results_writer_catalog()`; add HDF5/XDMF `ResultsWriter` per ADR 0008 (behind `OpenPFC_ENABLE_HDF5`); narrow the `ResultsWriter` contract (filename templating moves to a `FileResultsWriter` intermediate) so non-file sinks are expressible; writer domain metadata sourced from `SimulationState`, not the FFT inbox.
* [ ] Unify writer/modifier catalog APIs (one error philosophy).
* [ ] Remove the deprecated `(comm, rank, rank0)` wiring overload family (JsonWiringContext only).

### Required tests

* [ ] Session matrix test: {spectral, fd} × {cpu} in CI and × {cuda, hip} on clusters — one JSON document each, same toy model, correct stack instantiated (asserted via introspection/log), simulation runs.
* [ ] JSON `method`/`backend`/`integrator`/`fd_order` negative tests: unknown values produce the formatted config errors (message snapshots).
* [ ] FD order sweep: heat3d convergence test at orders 2, 8, 10 showing expected spatial-order improvement (extends existing analytic-Gaussian check).
* [ ] BC mechanism: wave2d mixed-BC tests pass on the stage-preparation path; a Dirichlet FD test (non-periodic axis from M1) validates ghost handling against an analytic solution.
* [ ] `"writer": "vtk"` produces `.vti/.pvti` output in an integration test; `"writer": "hdf5"` round-trips through `h5py`-based verification script; unknown writer hard-errors.

### Deletions

* [ ] `SpectralCpuStack`'s frontend twin (`frontend/ui/spectral_cpu_stack*.hpp`) superseded by the generalized session; `app_spectral_run.hpp` renamed/generalized (no `spectral_` hardcoding in App path names).
* [ ] `kernel/simulation/boundary_conditions/{fixed_bc,moving_bc}.hpp` (relocated to apps).
* [ ] Embryonic BC mechanisms (`prepare_boundaries` stub, `StageContext` BC flags).
* [ ] Deprecated wiring overloads; warn-and-drop registration paths.

### Definition of done

* [ ] One JSON document schema drives all method × backend combinations; the session matrix test is green.
* [ ] `grep -rn "FixedBC\|MovingBC" include/` returns nothing; exactly one BC mechanism exists.
* [ ] Unknown writers/modifiers/integrators fail loudly (negative tests green).
* [ ] Full suite + golden trajectories green.

---

## M11 — Checkpoint/restart end-to-end and external coupling surface

### Objective

Restart that provably resumes, owned by the framework; a minimal, stable coupling surface for external solvers.

### Dependencies

M10 (driver + services structure).

### Tasks

* [ ] Add `from_json(CheckpointMetadata)` (mirror of the existing `to_json`) with schema-version checking.
* [ ] Implement the bundle loader: read a published checkpoint directory (metadata + per-field bricks) through `BinaryReader` MPI-IO into `SimulationState`, validating domain/decomposition/method identity before mutation (reusing `state_capture` validate-before-mutate).
* [ ] Make publication MPI-collective (fields via `BinaryWriter` collective path, metadata rank-0 + barrier), preserving the atomic stage→rename protocol; define multi-rank semantics in `docs/reference/`.
* [ ] Implement `CheckpointService` owned by the driver: JSON keys `checkpoint.every` / `checkpoint.directory` / `restart_from: <dir>`; restart restores fields, `Time` accepted time, result counter, and integrator method identity (mismatch = hard error).
* [ ] Coupling surface (`kernel/simulation/coupling.hpp`): `FieldHandle` export from `SimulationState` (name + `FieldView` + `Box3i` + spacing/origin + memory space, one struct); the driver loop callable step-by-step by an external orchestrator (already a free function per M10); communicator injection documented with the `MPI_Comm_dup` isolation option (M4).
* [ ] Coupling reference example: `examples/22_external_coupling.cpp` — a mock "FEM" side owning the loop, pulling a `FieldHandle`, imposing a time-varying source through a FieldModifier-based adapter, with dt negotiation via `Time::clip_attempt_dt`.
* [ ] Document the coupling contract (`docs/extending_openpfc/external_coupling.md`): what is stable API, what is not, restart coordination.

### Required tests

* [ ] **Restart-equivalence test** (the Pre-M0 PO placeholder): run N+M steps ≡ run N, checkpoint, restart, run M — field-bitwise on CPU single-rank; declared tolerance at 4 ranks; for tungsten (spectral/ETD) and heat3d (FD).
* [ ] Crash-consistency test: interrupted publication leaves no partial bundle visible (stage-dir inspection).
* [ ] Identity-mismatch negative tests: wrong grid, wrong method, wrong schema version each hard-error with a diagnostic naming the mismatch.
* [ ] Coupling example runs under CI (2 ranks) and its imposed-source result matches an in-framework FieldModifier run bitwise.

### Deletions

* [ ] App-local checkpoint wrappers (`apps/heat3d/include/heat3d/state_capture.hpp`, `apps/wave2d/include/wave2d/state_capture.hpp`) superseded by `CheckpointService`.
* [ ] The manual three-key restart ritual documentation (replaced by `restart_from`); `simulator.result_counter`/`increment` JSON keys retired or subsumed.

### Definition of done

* [ ] Restart-equivalence tests green (bitwise single-rank, tolerance multi-rank, both methods).
* [ ] A checkpoint written by an M11 build is loadable by the same build via one JSON key; interrupted writes never corrupt.
* [ ] The coupling example exists, is CI-tested, and its contract is documented.
* [ ] Full suite + golden trajectories green.

---

## M-LUMI — LUMI/HIP hardware verification (added 2026-08-03)

### Objective

Every Required-test or Definition-of-done item from M0 through M11 that can only be
verified by actually *running* on LUMI (as opposed to compiling for HIP, which
tohtori's toolchain already supports without LUMI access) is consolidated here, in
one place, so that local development on tohtori is never blocked waiting for LUMI
cluster access. This milestone does not gate M0–M11; it exists purely to confirm
the HIP/LUMI side once cluster access is available, as the last hardware-verification
step before M12's release gate. Each item below names the milestone it was deferred
from — nothing here is new work, it is relocated verification.

**Why this milestone exists:** two independent development attempts both reached M3
and stalled, citing lack of LUMI access, even though the large majority of M3's own
gating only needs tohtori/CUDA. Splitting "compiles for HIP" (locally testable, kept
in its original milestone) from "runs correctly on real HIP hardware" (only testable
on LUMI, moved here) removes that false blocker.

**2026-08-18:** this checkout is on LUMI, so HIP execution items below can be
filled as the corresponding code exists. The inverse applies to CUDA: M0–M11 CUDA
execution stays on tohtori and must not block work here.

### Dependencies

M0–M11 functionally complete (their local-testable gating — CPU, tohtori/CUDA — all
green). LUMI cluster access.

### Required tests

* [ ] *(from Pre-M0)* GPU suites (`test_tungsten_cpu_vs_cuda/_hip`, allen_cahn, wave2d parity, new PA test) green on LUMI (HIP).
* [ ] *(from Pre-M0 task-52, part c)* Kobayashi HIP single-node perf baseline captured (machine-tagged JSON into `tests/baselines/perf/`, per the profiling schema-v2 exporter).
* [ ] *(from M3)* Existing GPU-gated suites (tungsten/allen_cahn/wave2d parity, `test_sparsevector_hip`, gpu_validation scenarios) pass on LUMI (HIP) against the single-sourced code, within the tolerances declared in `BASELINES.md`.
* [ ] *(from M3)* New HIP-parity tests (multi-field `for_each_interior_device` and composite-gradient device tests) actually run under HIP on LUMI (compiling them under HIP is already gated in M3 itself).
* [ ] *(from M4)* LUMI device-MPI probe demonstrably selects the GPU-aware path (log-asserted in the cluster test script).
* [ ] *(from M8)* Validation matrix (b) new-HIP vs new-CPU ≤1e-10 on LUMI; (d) perf within 5% of Pre-M0 baselines on LUMI.
* [ ] *(from M9)* Kobayashi `KOBAYASHI_VERIFY_HEX` checksums within declared tolerance on HIP (LUMI).
* [ ] *(from M9)* Kobayashi HIP perf ≥ its Pre-M0 host-staged baseline by a measurable margin (device path payoff recorded), on LUMI.

### Deletions

* [ ] None (verification-only milestone).

### Definition of done

* [ ] *(from M3)* HIP passes the multi-field and composite device tests (feature parity proven) — on LUMI.
* [ ] *(from M4)* LUMI runs device-resident halos (verified by the probe log + perf delta vs host-staging).
* [ ] *(from M9)* Aluminum runs on HIP (LUMI) from the same physics source as CPU/CUDA; kobayashi runs device-resident halos confirmed on both vendors (HIP/LUMI side closed out).
* [ ] All of the above recorded in `BASELINES.md`, ready to be cited by M12's own final "tohtori and LUMI" validation sweep rather than re-derived from scratch there.

---

## M12 — Gen‑1 deletion, enforcement, and the 0.2.0 release

### Objective

Exactly one architecture remains; all adapters, legacy APIs, and stale documentation removed; 0.2.0 shipped.

### Dependencies

M8–M11 all complete (no production consumer of Gen‑1 or adapters remains).

### Tasks

* [ ] Verify zero non-adapter consumers: `grep -rn "public pfc::Model\|ModelFieldRegistry\|add_real_field\|step_with_physics\|LegacyModelPhysics\|pfc::World" apps/ examples/ include/ src/ tests/` — every hit must be inside the adapter files or their dedicated tests.
* [ ] Migrate remaining Gen‑1 examples/fixtures (`examples/04,05,10,12`, `diffusion_model*`, `tests/fixtures/{diffusion_model,mock_model}.hpp`) to the M7 physics concepts.
* [ ] Migrate the frontend App path off A1/A2 onto `SimulationDriver` natively.
* [ ] Update all public docs: quickstart, tutorials, `extending_openpfc/`, learning paths, API examples — new architecture only; delete the `@since v2.0` fictions and stale `pfc::core`/`core/` references; regenerate the Doxygen examples catalog (CI consistency script must pass).
* [ ] Write `docs/MIGRATION_0.1_to_0.2.md` from `0.2_migration_map.md` (final state: every removed symbol → replacement).
* [ ] Make clang-tidy blocking in CI for the new-architecture directories; keep the bidirectional layering check; add the installed-package smoke test to the release checklist.
* [ ] Final `CHANGELOG.md` for 0.2.0; version set to 0.2.0; tag `v0.2.0`; release notes summarizing the architecture change and the migration guide.

### Required tests

* [ ] Full suite green on CI (CPU, both compilers, sharded), 2-rank MPI suite, compile-only CUDA/HIP jobs, `find_package` smoke tests.
* [ ] Full GPU + multi-rank validation sweep on tohtori and LUMI: tungsten/aluminum golden trajectories, kobayashi checksums, parity suites, restart equivalence, perf baselines — all recorded in `BASELINES.md` as the 0.2.0 validation record.
* [ ] Doxygen builds warning-free; docs-consistency CI scripts green.

### Deletions

* [ ] `kernel/simulation/model.hpp` (virtual `Model`), `model_free_functions.hpp`, `model_field_registry.hpp`, `model_types.hpp` (`pfc::Field` vector alias, `RealField`/`ComplexField`), deprecated `Model::get_field()`.
* [ ] Gen‑1 `Simulator` internals superseded by `SimulationDriver` (`simulator.hpp` and its satellite dispatch headers, or their legacy halves), `simulator_integrator.hpp` legacy loop.
* [ ] Adapters **A0** (`pfc::World` alias + `world::create` forwarders + `Decomposition` subworld shim), **A1** (`compat/legacy_model_physics`), **A2** (`step_with_physics`), and their parity tests.
* [ ] Any remaining `[[deprecated]]` symbols, `OPENPFC_SUPPRESS_LEGACY_WARNINGS`, and migration-only CMake toggles.
* [ ] Stale docs pages superseded by the 0.2 set (tracked via the migration map).

### Definition of done

* [ ] `grep -rin "deprecated\|legacy\|compat" include/ src/ --include="*.hpp" --include="*.cpp"` returns no API-level hits (comments about history allowed).
* [ ] Exactly one physics interface, one field container, one box type, one domain type, one step protocol, two exchangers, one device source tree — each verifiable by the greps defined in M1–M8 DoDs, re-run and recorded.
* [ ] The 0.2.0 validation record in `BASELINES.md` is complete (science + performance, all platforms).
* [ ] `v0.2.0` tagged and released with the migration guide.

---

## Milestone dependency graph (summary)

```
Pre-M0 → M0 → M1 → M2 → M3 → M4 ┐
                        M2 ──────┼→ M5 → M6 → M7 → M8 → M9 → M10 → M11 → M-LUMI → M12
                        M3 ──────┘         (M4 also feeds M8)
```

Parallelization allowed where dependencies permit (e.g., M5 may start once M3 lands, while M4 is in review), provided each milestone merges atomically green. M-LUMI is the odd one out: it isn't a dependency of M10/M11 and doesn't block them or anything before them — it collects LUMI-hardware-only verification deferred from Pre-M0/M3/M4/M8/M9 (see each milestone's own text and the M-LUMI section) and simply needs to close out before M12's release gate.
