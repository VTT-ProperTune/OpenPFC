<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Changelog

## [Unreleased] — 0.2.0 development

Breaking architecture refactor (see `OPENPFC_REFACTORING_EXECUTION_PLAN.md`,
milestones M0–M12). 0.2.0 will be released only after the Gen-1 architecture and
all temporary migration adapters are removed. Expect breaking API changes; 0.1.x
source compatibility is explicitly not a goal.

### Added

- `scripts/build.sh --machine=lumi` — LUMI HIP/ROCm path: loads `LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath` and `heffte-rocm`, configures on the login node, then submits compile + ctest to `dev-g` or `standard-g`. CUDA is rejected on LUMI.
- `pfc::comm::SparseExchange<HostSpace>` — host index-set facade over `SparseHaloExchanger` (structured `make_structured_halos` or a custom `RemoteHalo` list; `exchange`/`start`/`finish`). Device `SparseExchange<CudaSpace/HipSpace>` gathers, posts device-pointer MPI, and scatters without a full-field D2H. CUDA execution: verify on tohtori.
- Device halo default transport is pack-to-contiguous + device-pointer MPI when GPU-aware (`PaddedDeviceHaloExchanger` / `FullPaddedDeviceHalo`). `OPENPFC_{CUDA,HIP}_USE_SUBARRAY_HALO=1` restores derived-type MPI; `*_FORCE_PACKED_HALO=1` still host-stages. Post-exchange sync is stream-scoped, not `cudaDeviceSynchronize` / `hipDeviceSynchronize`.
- `pfc::comm::HaloExchange<CudaSpace/HipSpace>` — device facade in `runtime/gpu/comm_halo_exchange_gpu.hpp` (Faces / Full, blocking `exchange()`, residency sync). Persistent and split-phase fail closed (device exchangers are blocking-only). CUDA execution: verify on tohtori.
- `pfc::gpu::runtime_mpi_gpu_aware()` — shared GPU-aware MPI decision (assume override, Open MPI MPIX query, Cray `MPICH_GPU_SUPPORT_ENABLED=1`, optional device-pointer probe). Halo and SparseVector exchange use this instead of an Open-MPI-only query.
- `pfc::mpi::communicator::duplicate()` — opt-in `MPI_Comm_dup` isolation.
- `pfc::comm::HaloExchange<HostSpace>` — unified host halo facade (Faces/Full, `exchange`/`start`/`finish`, optional persistent, multi-field tag blocks). Device specialization is remaining M4 work.
- `pfc::halo` geometry helpers in `kernel/decomposition/halo_geometry.hpp` (M4): face slots, `opposite_slot` / `opposite_direction`, per-field MPI tag blocks, and padded send/recv slabs. `halo_directions.hpp` uses these instead of its own slot/tag copies.
- `pfc::Domain` — canonical coordinate/geometry type replacing the templated `World`
- `pfc::Box3i` — single canonical inclusive integer index box
- `pfc::data::Field<T, MemorySpace>` — canonical owning field container unifying LocalField/PaddedBrick
- `include/openpfc/runtime/gpu/gpu_api.hpp` — vendor shim (`gpuMalloc`, `gpuMemcpyAsync`, `gpuStream_t`, `GPU_CHECK`, `GPU_LAUNCH_KERNEL`) selected by CUDA vs HIP (`OPENPFC_HD` already covers `__HIPCC__` in `host_device.hpp`)
- `include/openpfc/runtime/gpu/databuffer_gpu.hpp` — single-source GPU `DataBuffer` for CUDA and HIP; `databuffer_cuda.hpp` / `databuffer_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/deep_copy_gpu.hpp` — single-source GPU `deep_copy(buffer, scalar)` fill; `deep_copy_cuda.hpp` / `deep_copy_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/memory_space_gpu.hpp` — single-source `CudaSpace` / `HipSpace`; `memory_space_cuda.hpp` / `memory_space_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/backend_tags_gpu.hpp` — single-source `CudaTag` / `HipTag`; `backend_tags_cuda.hpp` / `backend_tags_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/memory_traits_gpu.hpp` — single-source GPU `backend_traits`; `memory_traits_cuda.hpp` / `memory_traits_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/gpu_check.hpp` — single-source `cuda_check` / `hip_check`; `cuda_check.hpp` / `hip_check.hpp` are thin includes
- `include/openpfc/runtime/gpu/exchange_gpu.hpp` — single-source GPU SparseVector MPI exchange; `exchange_cuda.hpp` / `exchange_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/fd_gradient_device_gpu.hpp` — single-source GPU FD gradient evaluator (CUDA composite + HIP padded-Field factory); vendor `fd_gradient_device.hpp` re-export into `pfc::cuda` / `pfc::hip`
- `include/openpfc/runtime/gpu/for_each_interior_device_gpu.hpp` — single-source GPU interior driver (single-field + multi-field N=2–4 + autotune hook); vendor headers re-export into `pfc::sim::cuda` / `pfc::sim::hip`
- `include/openpfc/runtime/gpu/sparse_vector_gpu.hpp` — single-source GPU SparseVector copy-to-device; `sparse_vector_cuda.hpp` / `sparse_vector_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/sparse_vector_ops_gpu.hpp` — single-source GPU SparseVector gather/scatter; vendor `sparse_vector_ops.hpp` is a thin include; kernels live in `src/openpfc/runtime/gpu/sparse_vector_ops_gpu.inc` compiled from `sparse_vector_ops.cu` / `.hip`
- `include/openpfc/runtime/gpu/padded_device_halo_exchange_gpu.hpp` — single-source GPU 6-face padded device halo exchanger (HIP Field overloads stamped for CUDA); vendor `padded_device_halo_exchange.hpp` are thin includes; env/timer names stay `OPENPFC_CUDA_*` / `OPENPFC_HIP_*`
- `include/openpfc/runtime/gpu/full_padded_device_halo_gpu.hpp` — single-source GPU 26-direction padded device halo (CUDA `m_use_full_widening` stamped for HIP); vendor `full_padded_device_halo.hpp` are thin includes
- `src/openpfc/runtime/gpu/padded_halo_faces_gpu.inc` — single-source padded face pack/unpack kernels; compiled from `padded_halo_faces.cu` and `.hip`
- `include/openpfc/runtime/gpu/elementwise_ops_gpu.hpp` — generic device elementwise ops (complex×real multiply, two-term diagonal combine, axpy-style fill `out = alpha * x + beta`); compiled from `src/openpfc/runtime/gpu/elementwise_ops.cu` / `.hip`
- HIP-parity gpu_validation tests: `test_multi_field_device.hip` and `test_composite_gradient_pod_size_hip.hip` (HIP twins of the CUDA-only multi-field `for_each_interior_device` and composite-gradient POD layout cases)
- HIP FFT unit test `tests/unit/runtime/gpu/test_fft_hip.cpp` (`HIP_FFT`), gated on `OpenPFC_ENABLE_HIP_SPECTRAL` — twin of CUDA `test_fft_cuda.cpp` using `pfc::fft::create_hip` and `HipTag` DataBuffers
- HIP FFT integration roundtrip `tests/integration/scenarios/gpu_validation/test_hip_roundtrip.cpp` — twin of CUDA `test_cuda_roundtrip.cpp` (float/double DataBuffer forward/backward)
- HIP CPU-vs-GPU Laplacian integration tests `test_hip_vs_cpu_laplacian.cpp` and `test_hip_vs_cpu_laplacian_mpi.cpp` — twins of the CUDA Laplacian gpu_validation scenarios
- HIP vs CPU diffusion smoke `test_hip_vs_cpu.cpp` is compiled into `openpfc-tests` and constructs `create_hip` (previously an unwired stub)
- HIP backend instantiation smoke in `test_gpu_backend_instantiation.cpp` — separate Catch2 case so a CUDA skip cannot hide HIP; compares `create_hip` inbox/outbox sizes to the CPU FFT
- `examples/fft_backend_benchmark` benchmarks HIP (rocFFT) as well as CUDA, using `runtime/gpu/` DataBuffer/tags
- HIP `FullPaddedDeviceHalo` 26-direction integration twin `test_full_padded_device_halo_hip.cpp` of the CUDA `test_full_padded_device_halo.cpp` cases
- `scripts/check_gpu_memcpy_single_source.sh` — CI guard that `cudaMemcpy` / `hipMemcpy` in `include/` and `src/` stay under `runtime/gpu/`
- `pfc::fft::Backend::HIP` and `backend_from_string("hip"` / `"rocm")` when `OpenPFC_ENABLE_HIP_SPECTRAL` is on; JSON `from_json<fft::Backend>` and `create_with_backend` accept HIP the same way they already accept CUDA

### Fixed

- HIP GPU autotune device queries used the deprecated `gcnArch` field (an `int` on ROCm 6, so `std::string(prop.gcnArch)` was not an architecture name) and `pciDeviceId` (the struct spells `pciDeviceID`). They now use `gcnArchName` and `pciDeviceID`; the HIP autotune test requires a `gfx` prefix when a device is present.
- GPU autotune unit test includes Catch2 string matchers so `REQUIRE_THROWS_WITH` compiles.
- HIP FFT unit test did not link HeFFTe, so `heffte.h` was not on the include path. Both CUDA and HIP FFT unit-test binaries now link `Heffte::Heffte`.
- HIP configure failed: `openpfc_gpu_compile_defs` is an INTERFACE library, but GPU macros were applied with `PUBLIC` (`target_compile_definitions` only allows `INTERFACE` on INTERFACE targets).
- HIP configure failed at export: FetchContent `nlohmann_json` is not in `OpenPFCTargets`. The library now uses that header-only tree via `BUILD_INTERFACE` includes instead of linking the FetchContent target. GPU kernel libraries get the same include path so autotune JSON parses under HIP.
- Removed `tests/unit/kernel/data/test_field.cpp`; it included the deleted Gen-1 `kernel/data/field.hpp`. Canonical coverage is `test_grid_field.cpp`.
- Removed `tests/unit/kernel/data/test_multi_index.cpp`; `multi_index.hpp` was deleted in M2.
- **Sticky CUDA/HIP error from a handled allocation failure:** `DataBuffer`'s CUDA/HIP specializations checked `cudaMalloc`/`hipMalloc`'s return value but never called `cudaGetLastError()`/`hipGetLastError()` to clear the driver's sticky error flag before throwing. A deliberately-triggered allocation failure (e.g. in a resize-failure test) left that flag poisoned for the rest of the process, later misattributed to an unrelated kernel launch elsewhere as a false "out of memory".
- **`FullPaddedDeviceHalo` skipped its corner/edge fill without GPU-aware MPI:** the 3-pass widening algorithm was gated entirely behind GPU-aware MPI availability, even for self-only periodic axes that never touch MPI device pointers at all. It now runs the full algorithm whenever no active axis has a real (non-self) neighbor, or GPU-aware MPI is genuinely available; only real cross-rank axes without GPU-aware MPI fall back to the face-only path.
- **`test_stage_preparation.cpp` checked halo cells `PaddedHaloExchanger` never fills:** its comparison helpers walked the full padded range on orthogonal axes, but the exchanger is documented face-only (corners/edges untouched). Restricted the checks to the owned range, matching the already-correct pattern in `test_padded_halo_exchange.cpp`.
- Missing `pfc::ui::from_json<Domain>` specialization (declared, never defined) caused a link error in any CUDA/HIP app driver calling it directly instead of through `SpectralSimulationSession`.
- `TungstenCUDA`/`TungstenHIP` had no constructor matching the generic `(fft::IFFT&, const World&, MPI_Comm)` session-wiring signature, only failing to compile when a CUDA/HIP app target was actually built.

### Changed

- Example `09_parallel_fft_high_level` stores the FFT outbox in `pfc::data::Field<std::complex<double>>` instead of legacy `Array`.
- `pfc::field::for_each*` in `brick_iteration.hpp` no longer has `PaddedBrick` overloads; padded `pfc::data::Field` is the only container (tests already used Field).
- `pfc::communication::PaddedHaloExchanger` no longer binds `PaddedBrick`; padded `pfc::data::Field` (or explicit `Box3i` + `Domain`) is the only container binding. Callers already used the Field constructors.
- HIP `pfc::hip::FdGradientDevice` factory binds `pfc::data::Field` (any memory space) instead of legacy `PaddedBrick`, matching the CUDA twin. Unpadded Fields (`storage_halo == 0`) are rejected; padded `Field<double, HipSpace>` is the device path.
- CPU `pfc::gradient::FDGradient` no longer has a `PaddedBrick` constructor or factory; padded `pfc::data::Field` is the only container binding (`test_multi_field_device.cu` migrated).
- `scripts/build.sh` (Tohtori) auto-detects a custom CUDA-aware Open MPI build (see `scripts/build_tohtori.sh --cuda`) and uses it in place of the site `openmpi/5.0.10` module when present, defaulting `MPI_CUDA_AWARE` to `ON` in that case. Without it, the default stays `OFF`: the site module links a UCX built without `--with-cuda`, and passing device pointers to it segfaults despite Open MPI's own `MPIX_Query_cuda_support()` probe claiming support.
- HIP packed-halo pinned host buffers use `hipHostMalloc` / `hipHostFree` instead of the deprecated `hipMallocHost` / `hipFreeHost`.
- `OpenPFC_ENABLE_CUDA` / `OpenPFC_ENABLE_HIP` and `OpenPFC_MPI_CUDA_AWARE` / `OpenPFC_MPI_HIP_AWARE` are PUBLIC compile definitions on `openpfc` (and the vendor kernel libraries) instead of directory-scope `add_compile_definitions`, so `find_package(OpenPFC)` consumers see the same macros as the in-tree build.
- Device kernel TUs (`sparse_vector_ops.cu/.hip`, `padded_halo_faces.cu/.hip`) live under `src/openpfc/runtime/gpu/` instead of `include/` (and instead of `src/openpfc/runtime/cuda/` for the CUDA halo-face TU). CUDA `padded_halo_faces.cu` remains linked per executable because of separable-compilation device-link.
- `deep_copy(buffer, scalar)` for CUDA/HIP `DataBuffer` runs a device fill kernel (`runtime/gpu/fill_gpu`) instead of staging a host vector. Device scalar fill supports `float` and `double`; include `deep_copy_gpu.hpp` (or the vendor shim). Device `View` fill and View-to-View device copies are not provided. GPU fill tests cover `DataBuffer` and raw `fill_*_impl` pointers.
- Tungsten CUDA/HIP `multiply_complex_real` and `apply_time_integration` call `runtime/gpu/elementwise_ops_gpu` instead of duplicating those kernels. Nonlinear and stabilization kernels stay in the Tungsten TUs.
- `cmake --install` no longer ships FetchContent `nlohmann_json` headers or the `openpfc-tests` binary. `find_package(OpenPFC)` always `find_dependency(nlohmann_json)`.
- `OpenPFC_ENABLE_GPU_AUTOTUNING` is a PUBLIC compile definition on `openpfc` (and the vendor kernel libraries) instead of directory-scope `add_compile_definitions`.
- Heat3D/Wave2D structural `rhs()` tests renamed off `vs_legacy_step` (they were never numerical vs-legacy baselines). Checkpoint headers/docs state that restart loading is not implemented.
- Bare `cmake` on a single-config generator defaults to `RelWithDebInfo`, not Debug (`cmake/ProjectSetup.cmake`; documented in `INSTALL.md`).
- `tests/benchmarks/README.md` no longer lists machine-specific nanosecond/millisecond claims; measure locally in Release.
- State-access design docs describe GPU storage as `pfc::core::DataBuffer` (`CudaTag`/`HipTag`); `pfc::gpu::GPUVector` is gone.
- `NAN_CHECK_ENABLED` is a PUBLIC compile definition on `openpfc` when Debug is selected or `OpenPFC_ENABLE_NAN_CHECK=ON`, instead of directory-scope `add_compile_definitions`.
- GPU SparseVector host-to-device copy failures report `"HIP copy failed: …"` with the runtime string, matching CUDA; unused duplicate `sparse_vector_ops_cuda.hpp` / `sparse_vector_ops_hip.hpp` shims removed (`sparse_vector_ops.hpp` remains).
- `for_each_interior_device` launch/sync failures use `GPU_CHECK` (`"GPU error: …"`) instead of a per-overload hand-rolled check. Kernel `.inc` files still prefix CUDA/HIP; co-enabled TUs still use `cuda_check` / `hip_check`.
- CUDA `openpfc_gpu_kernels` and HIP `openpfc_hip_kernels` share one CMake source list (`sparse_vector_ops`, `fill`, `elementwise_ops`). HIP still adds `padded_halo_faces.hip`; CUDA halo-face kernels stay linked per executable.
- GPU kernel `.inc` sources live under `src/openpfc/runtime/gpu/` next to the vendor TUs that include them (not under `include/`; not installed).
- Architecture, styleguide, and `DataBuffer` diagnostics name `runtime/gpu/` as the CUDA/HIP implementation layer; vendor `runtime/cuda` / `runtime/hip` trees are documented as thin includes plus FFT until M5.
- Shared Tungsten GPU headers and vendor FFT headers include `runtime/gpu/` DataBuffer/tags directly instead of hopping through CUDA/HIP shims.
- Dual CUDA/HIP unit tests include `runtime/gpu/` SparseVector exchange and DataBuffer headers instead of duplicated vendor shims (`test_sparse_vector_exchange_device.cpp` included).
- CUDA/HIP fail-closed SparseVector exchange tests include `runtime/gpu/` exchange, SparseVector, and check headers instead of vendor shims (keep native `cuda_check` / `hip_check` calls).
- CUDA/HIP SparseVector unit tests (`test_sparsevector_cuda.cpp`, `test_sparsevector_hip.cpp`) include `runtime/gpu/` SparseVector headers instead of vendor shims (keep native `cudaMemcpy` / `hipMemcpy`).
- CUDA/HIP padded device-halo tests (`test_padded_device_halo_self_wrap.cpp`, `test_padded_device_halo_self_wrap_hip.hip`, `test_full_padded_device_halo.cpp`) include `runtime/gpu/` halo headers instead of vendor shims
- HIP fd-gradient gpu_validation test includes `runtime/gpu/` DataBuffer and HipSpace headers instead of vendor shims; vendor `fd_gradient_device.hpp` / `for_each_interior_device.hpp` re-exports stay (call sites use `pfc::hip::` / `pfc::sim::hip`)
- Kobayashi CUDA driver includes `runtime/gpu/` padded device-halo headers instead of the vendor shim (`pfc::cuda::PaddedDeviceHaloExchanger` stays; the GPU header already stamps it)
- `SimulationState` device-field compile coverage includes `HipSpace` (CUDA twin already existed) and includes `runtime/gpu/` memory-space headers.
- SparseVector `on_host` coverage includes `HipTag` (CUDA twin already existed) and includes `runtime/gpu/` SparseVector headers.
- Halo-exchange concept docs list canonical GPU sources under `runtime/gpu/` (vendor CUDA/HIP headers are thin includes); HIP packed-halo env and kernel-library split are documented alongside CUDA.
- FD/halo Doxygen `@see` comments and per-point-gradient docs point at `runtime/gpu/` device twins, not CUDA-only vendor headers.

### Removed

- `pfc::field::LocalField` (`kernel/field/local_field.hpp`). Use `pfc::data::Field` with `field_from_subdomain_unpadded` (unpadded storage) or `field_from_inbox` for spectral inboxes. Stepper factories already bound `Field`.
- `pfc::field::PaddedBrick` (`kernel/field/padded_brick.hpp`). Use padded `pfc::data::Field` via `field_from_subdomain(decomp, rank, halo)`. Halo exchangers, FDGradient, and `brick_iteration` already bind Field.
- `pfc::DiscreteField` (`kernel/data/discrete_field.hpp`) and `pfc::interpolate`. Use `pfc::data::Field` with `coords()` / `apply()`. The quarantined DiscreteField unit tests were deleted with the type.
- `pfc::Array` (`kernel/data/array.hpp`). Use `pfc::data::Field`. The Array unit tests were deleted with the type.
- `pfc::field::Field<T>` (`kernel/data/field.hpp`). Use `pfc::data::Field`. Steppers, stacks, and factories already bound the canonical type; the functional container had no remaining callers.
- `pfc::field::make_legacy_modifier` (`kernel/field/legacy_adapter.hpp`). Wrap a lambda in a `FieldModifier` and call `pfc::field::apply` instead.
- `pfc::field::apply(Model&, name, fn)` and the matching `apply_with_time` / `apply_inplace*` Model overloads. Call `apply(get_real_field(m, name), get_world(m), get_fft(m), fn)` instead so `operations.hpp` no longer includes the simulation layer.
- `pfc::gpu::GPUVector` (`runtime/cuda/gpu_vector.hpp`), `kernels_simple` (`add_scalar` / `multiply_scalar`), and their CUDA unit tests. Use `pfc::core::DataBuffer` (or `pfc::data::Field`) for device storage.
- `pfc::create_mirror` / `pfc::create_mirror_view` (`kernel/execution/create_mirror.hpp`). Host `View` copies use `deep_copy`; device storage is `DataBuffer`.
- GPU View execution-space mapping (`runtime/gpu/view_gpu.hpp` and vendor `view_cuda.hpp` / `view_hip.hpp`). `View` is host-only; device storage is `DataBuffer`.
- GPU `parallel_for` / `fence` (`runtime/gpu/parallel_gpu.hpp` and vendor shims) and `Cuda`/`HIP` execution-space tags (`execution_space_gpu.hpp`). Host `parallel_for` remains Serial/OpenMP-only.
- Kokkos-facsimile `View`, host `parallel_for`/`fence`, `RangePolicy`/`MDRangePolicy`, layouts, `deep_copy` View overloads, `Serial`/`OpenMP` execution-space tags, and `tests/unit/kernel/execution/test_kokkos_like.cpp`. Device storage is `DataBuffer`; `deep_copy(buffer, scalar)` remains.
- GPU autotune demo keys `add_scalar` / `multiply_scalar` (registry + fallback defaults). Remaining defaults are `for_each_interior_3d`, `gather`, and `scatter`.

## [0.1.5] - 2026-07-23

Final stable 0.1.x release: a correctness and packaging pass ("Pre-M0
stabilization") completed before the breaking OpenPFC 0.2 architecture refactor.
This is the last release on the 0.1 architecture.

### Fixed (Pre-M0 stabilization — final 0.1.x correctness pass before the 0.2 refactor)

Each fix has a regression test; the full CPU suite (26 ctest batches + Python)
passes, and CUDA/HIP compile cleanly. GPU *runtime* behavior for the device-only
fixes is marked `// TODO: not tested` pending a GPU-node run.

- **FD dispatcher fail-closed (audit §4.3):** `field::fd::laplacian_interior(int order, …)` now throws `std::invalid_argument` on an unsupported order instead of silently doing nothing.
- **Periodicity honored (audit §4.4):** `world::create` / `from_bounds` now store the requested per-axis periodicity (was always all-periodic); added `world::get_periodic` / `world::is_periodic`.
- **Subdomain bounds (audit §4.5):** `world::get_lower_bounds` / `get_upper_bounds` now respect a subdomain's index offset instead of reporting the global origin.
- **Coordinate→index convention (audit §4.6):** `csys::to_index` now rounds (matching the documented `to_indices` contract and `DiscreteField`), not truncates.
- **Dead API removed (audit §4.8):** deleted the undefined `utils::compute_upper_bounds/compute_spacing` declarations and the constructors that called them.
- **Dangling reference removed (audit §4.10):** `field::Field<T>` stores `World` by value.
- **Checked MPI + fail-closed cleanup (audit §4.7, §4.11):** checked `MPI_Comm_size` in the decomposition factory and the GPU packed-halo `Irecv/Isend`; unified destructor/move-assign cleanup on a single `abort_on_mpi_error` (log + `MPI_Abort`, never throw from a destructor).
- **Device `parallel_for` trap (audit §4.2):** CUDA/HIP `parallel_for` is now a compile-time error instead of silently running device work on the host.
- **GPU initial-condition residency (audit §4.1):** `Model` gained `prepare/finalize_for_field_modifiers` hooks that the `Simulator` calls around modifier application and result writing, so App-driven GPU runs seed the device field (previously integrated from an unseeded buffer).
- **Single save scheduler (audit §4.12):** the Tungsten GPU driver uses `Time::do_save()` instead of a divergent `round(saveat/dt)` rule.
- **HeFFTe box-order invariant (audit §4.9):** `Decomposition` asserts at construction that `heffte::split_world` box order matches the x-fastest neighbor convention.

### Fixed — packaging / build (audit §11)

- Code-coverage instrumentation no longer defaults ON and is `PRIVATE`, so it cannot leak into Release builds or the installed/exported package.
- The HIP kernel library is now part of the install/export set; `OpenPFCConfig.cmake` declares its transitive dependencies (CUDAToolkit / hip / HDF5 / nlohmann_json); backend-enable definitions are exported; installs ship headers only. Guarded by a new `find_package(OpenPFC)` packaging smoke test in CI (`tests/packaging/consumer`).

### Documentation

- **Onboarding spine:** `docs/start_here_15_minutes.md`, `docs/spectral_stack.md`, `docs/recipes/`, `docs/gpu_path_decision.md`, `docs/hpc_operator_guide.md`.  
- **Quality / teaching:** `docs/when_not_to_use_openpfc.md` (fit + FD vs spectral direction), `docs/documentation_versioning.md`, `docs/from_paper_to_run.md`, `docs/workshop/`, `docs/adr/`, `docs/operator_playbooks.md`, `docs/science_numerics_limits.md`, optional printable handbook (`docs/handbook_build.md`, `scripts/build_handbook.sh`).  
- **MkDocs (optional):** `uv` project under `docs/` (`mkdocs`, Material theme); root `mkdocs.yml` builds a browsable prose site — see `docs/mkdocs_preview.md`.  
- **CI:** `scripts/check_doc_bash_syntax.py` validates fenced `bash`/`sh` blocks under `docs/`.  
- **Binary MPI-IO fields:** `docs/binary_field_io_spec.md` (layout, filename templates, collectives).  
- **Spectral `App` config keys:** `docs/spectral_app_config_reference.md` (world, time, `plan_options`, `fields`, IC/BC).  
- **HPC:** `docs/tutorials/hpc_slurm_day_one.md`, `docs/mpi_io_layout_checklist.md`.  
- **Science notes:** `docs/science_tungsten_quicklook.md`, `docs/science_cahn_hilliard_vs_allen_cahn.md`.  
- Indexes and cross-links updated (`docs/README.md`, `learning_paths.md`, `tutorials/README.md`, …).

**Upgrade discipline (maintainers):** When a change alters **CMake options**, **JSON/TOML keys**, **default writers**, or **on-disk formats**, add an explicit **migration** bullet under `[Unreleased]` (usually `### Changed` or `### Removed`) with what to do instead and a link to the relevant `docs/*.md`.

### Added

- **Sparse, grid-agnostic halo exchanger** (`include/openpfc/kernel/decomposition/sparse_halo_exchange.hpp`): new `pfc::SparseHaloExchanger<T>` and `pfc::halo::RemoteHalo<T>` accept arbitrary `(peer_rank, send_indices, recv_indices, send_tag, recv_tag)` tuples — no grid, axis, or face semantics. Drives one `MPI_Isend`/`MPI_Irecv` pair per `RemoteHalo` over the existing `core::SparseVector` + `exchange::isend_data` / `irecv_data` plumbing; supports optional `core::scatter` after the wait. The new `pfc::halo::make_structured_halos<T>(decomp, rank, hw, dirs = Axes3D())` builds the `RemoteHalo` list for the standard structured face/edge/corner exchanges driven by a `HaloDirectionSet`, and `pfc::halo::copy_to_face_layout` (in `halo_face_layout.hpp`) refills the `std::array<std::vector<T>, 6>` layout that `field::fd::laplacian_periodic_separated<Order>` expects. Foundation for future FEM / unstructured / multi-block patterns. See [`docs/concepts/halo_exchange.md`](docs/concepts/halo_exchange.md).
- **Customizable halo direction sets** (`include/openpfc/kernel/decomposition/halo_directions.hpp`): new `pfc::halo::HaloDirectionSet` and named presets **`Axes2D` (4) / `Full2D` (8) / `Axes3D` (6) / `Full3D` (26)**, plus a per-rank `HaloDirectionSelector` callback. Every face exchanger gained a new ctor that accepts a `HaloDirectionSet` (default preserves historical behaviour: `Axes3D()` for face exchangers, `Full3D()` for `FullPaddedDeviceHalo`); excluded slots are skipped in both GPU-aware and packed branches. **`apps/kobayashi/src/cuda/kobayashi_fd_cuda.cpp`** now uses **`Axes2D()`** so the 2D slab driver no longer touches **±Z** halos. See [`docs/concepts/halo_exchange.md` § 5.4](docs/concepts/halo_exchange.md) and [`apps/kobayashi/docs/cuda_halo_lessons_h100.md`](apps/kobayashi/docs/cuda_halo_lessons_h100.md).
- **CUDA padded halos**: `pfc::cuda::PaddedDeviceHaloExchanger` (`runtime/cuda/padded_device_halo_exchange.hpp`) and face pack/unpack kernels (`src/openpfc/runtime/gpu/padded_halo_faces.cu`, linked into **`kobayashi_fd_cuda`**) — same MPI derived types as `PaddedHaloExchanger<double>` with a **device** base pointer; GPU-aware MPI when supported, else narrow face slabs + pinned host. Env **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`** forces the packed path.
- **Applications**: `kobayashi_fd_manual` (`apps/kobayashi/`) — Kobayashi phase-field + temperature coupling on a periodic 2D slab, explicit finite differences matching the historical Julia `kobayashi_v1` layout; PNG snapshots of \(\phi\); optional **`KOBAYASHI_VERIFY`** / **`KOBAYASHI_VERIFY_HEX`** stdout lines and **`OPENPFC_KOBAYASHI_SKIP_PNG`** / **`OPENPFC_KOBAYASHI_QUIET`** env toggles; Slurm scaling under `apps/kobayashi/slurm/` (`gen05_epyc`) with **`summarize_scaling.py`** and **`plot_strong_scaling.py`** (SVG strong-scaling figure from **`summary.tsv`**). **`kobayashi_fd_openmp`** — same numerics on one node with periodic **index wrapping** (no MPI halos) and **OpenMP** parallelism; Catch2 **`test_kobayashi_fd_openmp`**; Slurm **`kobayashi_openmp_scaling_gen05_epyc.sbatch`** and **`summarize_openmp_scaling.py`**.
- **Documentation / clusters:** Kobayashi **`apps/kobayashi/slurm/`** README and **`kobayashi_rebuild_openpfc_gen05_epyc.sbatch`** for rebuilding against **`openmpi/5.0.10`** (Slurm PMI/`srun`).
- **HeFFTe builds**: Optional vendoring of HeFFTe 2.4.1 via CMake FetchContent (`FetchHeffte.cmake`), HeFFTe discovery hints, and a pinned GCC 11 + OpenMPI toolchain preset for fixed cluster layouts (e.g. tohtori).
- **HIP / AMD GPUs**: CMake HIP/ROCm detection, HeFFTe ROCm backend when HIP is enabled, rocFFT-backed `fft::create_hip`, and Tungsten on HIP (model, kernels, applications, scalability and VTK-focused tests).
- **MPI halos**: `PersistentHaloExchanger` for six-face persistent MPI halo exchange (with integration coverage against the non-persistent path).
- **Profiling**: Kernel `ProfilingSession` library, wiring through the JSON/TOML `App`, MPI path instrumentation, and documentation plus Tungsten input/schema alignment for performance runs.
- **Kokkos-like API**: Experimental View types, execution and memory-space tags, `parallel_for` / `fence`, and host–device copy/mirror helpers for Kokkos-style structuring of numerical code.
- **Field modifiers**: Multi-field initial/boundary condition targets on `FieldModifier`, with `Simulator` validating each name against `has_field`.
- **GPU MPI**: CMake option `OpenPFC_MPI_CUDA_AWARE` for GPU-aware MPI when using CUDA.
- **Documentation**: Developer style guide (`docs/styleguide.md`), separate CPU vs CUDA/HIP build layout guide (`docs/build_cpu_gpu.md`), LUMI-G build notes, and expanded performance/profiling material.

### Changed

- **CUDA GPU-aware MPI (configure + Slurm):** `find_package(MPI REQUIRED COMPONENTS C CXX)`; configure probe runs **`try_run`** on **`cmake/openpfc_mpix_cuda_probe.c`** (links **`MPI::MPI_C`**) so **`MPIX_Query_cuda_support()`** is validated on the configure host. **`kobayashi_rebuild_openpfc_cuda_h100.sbatch`** defaults to **`-DOpenPFC_MPI_CUDA_AWARE=ON`** and uses **`mpicxx`** as **`CMAKE_CXX_COMPILER`**; **`KOBAYASHI_REBUILD_CUDA_MPI_AWARE=0`** forces a packed-only compile path.
- **`PaddedDeviceHaloExchanger` (GPU-aware):** periodic face neighbors that map to **the same MPI rank** (e.g. ±Z when the process grid is **1** deep in Z, as in **`nz = 1`** Kobayashi slabs) no longer use **`MPI_Irecv` / `MPI_Isend` on device buffers to self**; those faces use **device pack/unpack** into a small **`m_d_scratch`** buffer instead, avoiding pathological stalls / low GPU utilization on some Open MPI + UCX builds.
- **`kobayashi_fd_cuda`**: **`MPI_COMM_WORLD` size 1** uses **device-only periodic halos** (`device_periodic_local`) instead of **`PaddedDeviceHaloExchanger`** in the timestep loop (avoids MPI progress on the host and redundant global CUDA sync). **`nproc > 1`** still uses **`PaddedDeviceHaloExchanger`**; rank 0 prints **`KOBAYASHI_CUDA_HALO_MODE`**.
- **CUDA build**: `padded_halo_faces.cu` is compiled into **`kobayashi_fd_cuda`** instead of **`libopenpfc_gpu_kernels`** so separable CUDA compilation registers correctly at the final device link (static archives + mixed host link previously produced undefined `__cudaRegisterLinkedBinary_*` symbols).
- **Clusters (tohtori):** Default Open MPI **5.0.10** in **`cmake/toolchains/tohtori-gcc11-openmpi.cmake`**, **`CMakePresets.json`**, and **`scripts/build_tohtori.sh`**; **`INSTALL.md`**, **`cmake/README.md`**, **`scripts/README.md`**, dependency matrix, and related docs aligned (override with **`OPENMPI_ROOT`** when needed).
- **Profiling**: `ProfilingSession` is frame-only generic (`begin_frame`, `set_frame_metric`, `set_frame_metric_elapsed_since_begin`, `end_frame`); OpenPFC step/MPI/memory wiring uses **`openpfc_frame_metrics.hpp`**. JSON **`frame_metric_names`** use **`heap_secondary_bytes`** instead of **`fft_heap_bytes`** for the second heap column.
- **Profiling export**: JSON and HDF5 use **schema version 2** with a per-MPI-rank hierarchy (`ranks[]` in JSON; `openpfc/profiling/ranks/<id>/` in HDF5). See **`docs/profiling_export_schema.md`**. **`ProfilingPrintOptions::wall_denominator_metric`** configures the %tot denominator (default **`wall_step`**). **`print_profiling_timer(std::ostream &, MPI_Comm, …)`** with **`mpi_aggregate_stdout`** prints a rank-0 table combining per-rank timer totals (**`mpi_aggregate_stat`**: mean/sum/min/max/median). **`App`** enables this when **`profiling.print_report`** is true (all ranks participate in the gather).
- **Profiling export (schema v3)**: Optional **`ProfilingExportOptions::run_id`** and **`export_metadata`**; **`App`** reads **`profiling.run_id`**, **`profiling.export_metadata`**, and environment (**`SLURM_JOB_ID`**, **`OPENPFC_PROFILING_RUN_ID`**, domain sizes, Slurm layout). When **`run_id`** is set, HDF5/JSON use a merge-friendly layout under **`openpfc/profiling/runs/<id>/`**. **`experiments/scalability/`** documents a Slurm driver for scaling studies (site/workload profiles, **`scala`** CLI).
- **Layout & includes**: Clearer **kernel / runtime / frontend** layering, consistent `<openpfc/...>` includes across the library and unit tests, and FFT layout helpers split into `fft_layout.hpp`.
- **SparseVector / MPI**: Zero-copy face exchange and non-blocking MPI paths for sparse halo communication where applicable.
- **CI**: GitHub Actions runners pinned to **Ubuntu 24.04 LTS** (main matrix, coverage, docs, clang-tidy, code quality). LLVM apt repos use **noble** for Clang 14/16; removed the gcc-13 toolchain PPA. The coverage job runs tests via **CTest** like the main workflow; workflow README aligned with HeFFTe 2.4.1.
- **UI**: `list_valid_field_modifiers()` reads registered names from `FieldModifierRegistry` (sorted) instead of a duplicated literal list.

### Fixed

- **MPI `SparseVector` neighbor tests** (`tests/unit/kernel/decomposition/test_sparse_vector_neighbor_exchange.cpp`): rank-two-only cases now **skip** when `MPI_Comm_size != 2` (they previously called matching `receive` on ranks ≥2 and hung). Ring / multi-neighbor exchanges use **safe blocking order** (odd/even send–recv), the **2×2 grid** case exchanges horizontal then vertical halves without deadlock, and **multiple-neighbor** recv peers / expected values were corrected (left-going payload is received from the **right** neighbor). Eliminates long **`mpirun -n 4 … '[MPI]'`** hangs / SIGTERM from these tests.
- **`PersistentHaloExchanger`**: persistent requests now use the same **MPI tag pairing** as `HaloExchanger` zero-copy (`MPI_Recv_init` uses **opposite face slot**, `MPI_Send_init` uses the **local slot**), and requests are registered **all recv then all send** like `start_halo_exchange`. Integration parity runs on **`MPI_Comm_size == 2`** with **`{1,1,2}`** Z-splitting so ±Z neighbors are never **self** (avoids fragile persistent self-message ordering); **`mpi_4procs_grid_multiple`** no longer includes this case (it remains under **`mpi_2procs_all`**).
- **`[MPI]` tag hygiene:** `test_halo_exchange_driver.cpp` (hard-wired `{2,1,1}` / two subdomains) and the **`[profiling][MPI]`** JSON/timer assertions now **return early unless `MPI_Comm_size == 2`**, so a broad filter like **`mpirun -n 4 ./openpfc-tests '[MPI]'`** no longer runs them with **world size ≠ decomposition domains** or **≠ 2 assumed ranks**.
- **Decomposition**: Halo loops aligned with **inclusive** `World` bounds so face exchanges match the intended domain.
- **MPI timer**: `pfc::mpi::timer::toc()` no longer reads uninitialized state when called before `tic()`; misuse now throws `std::logic_error`, and `reset()` clears an in-progress lap.
- **Logging**: If `gmtime_r` / `gmtime_s` fails, log lines use a `<time-unavailable>` placeholder instead of formatting an uninitialized `tm`.
- **Memory reporter**: `get_system_memory_bytes()` only uses parsed `MemTotal` kB when stream extraction succeeds, avoiding read of uninitialized `mem_kb` on malformed lines.
- **`PaddedDeviceHaloExchanger` (packed fallback):** same-rank periodic faces (e.g. ±Z when **local nz = 1**) no longer use **`MPI_Irecv` / `MPI_Isend` to self** on **nx×ny** face buffers (~128 MiB per message at 4096²); they use **device pack/unpack** like the GPU-aware path, avoiding multi‑second-per-step stalls when **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`**.

### Removed

- **`pfc::SeparatedFaceHaloExchanger<T>`** and `include/openpfc/kernel/decomposition/separated_halo_exchange.hpp`. The face-only exchanger has been superseded by the fully sparse `pfc::SparseHaloExchanger<T>` plus `pfc::halo::make_structured_halos<T>` for the structured shortcut. **Migration:** replace `SeparatedFaceHaloExchanger<T> ex(decomp, rank, hw, comm);` + `ex.exchange_halos(u.data(), u.size(), face_halos);` with `SparseHaloExchanger<T> ex(comm, rank, halo::make_structured_halos<T>(decomp, rank, hw));` + `ex.exchange_halos(u.data(), u.size()); halo::copy_to_face_layout(ex, face_halos);`. `pfc::halo::FaceHaloCounts`, `face_halo_counts`, and `allocate_face_halos` are unchanged. **Drive-by fix:** `pfc::halo::Connectivity::Edges` no longer aliases `Faces` in `create_halo_patterns` — it now correctly returns the 18-direction faces+edges subset (corners excluded).
- **Nix / flake support**: Removed `flake.nix`, `flake.lock`, and the `nix/` packaging tree; dropped the Nix job from CI. Use CMake and `INSTALL.md` for builds.

## [0.1.4] - 2025-12-18

### Added

- **GPU/CUDA Support**: Complete CUDA implementation enabling GPU-accelerated PFC simulations.
  Added `DataBuffer` for backend-agnostic memory management with CPU/GPU memory traits,
  CUDA FFT integration via HeFFTe, GPU kernels for element-wise operations, and `GPUVector`
  RAII container. Implemented full Tungsten model on GPU with optimized kernel launches and
  CPU-GPU synchronization for FieldModifiers and VTK output. Runtime backend selection API
  allows choosing between CPU and CUDA FFT backends via configuration. Comprehensive test
  coverage includes GPU device detection, memory allocation, FFT operations, and CPU vs CUDA
  result comparison. Build system supports optional `OpenPFC_ENABLE_CUDA` flag.
- **VTK Output**: New VTK ImageData writer in `include/openpfc/results/vtk_writer.hpp` and
  `src/results/vtk_writer.cpp` for parallel visualization output. Generates `.vti` files
  for each rank and `.pvti` parallel metadata files for ParaView/VisIt. Includes comprehensive
  test suite with MPI-aware tests and single-invocation test model to prevent cleanup races.
- **TOML Configuration**: Added TOML config file support alongside JSON. New
  `feat(utils): Add TOML to JSON conversion utility` enables `.toml` input files with
  automatic conversion. Integrated tomlplusplus library via CMake find module. All example
  configurations converted to TOML format. Unit tests validate conversion accuracy.
- **Modular CMake Architecture**: Refactored monolithic CMakeLists.txt into 12 focused modules
  in `cmake/` directory: ProjectSetup, CompilerSettings, CudaSupport, Dependencies,
  LibraryConfiguration, BuildOptions, CodeCoverage, Installation, PackageConfig, BuildSummary.
  Improves maintainability and reusability. Documented in `cmake/README.md`.
- **CI/CD Pipelines**: Comprehensive GitHub Actions workflows for build matrix (GCC/Clang,
  multiple OS), documentation deployment, code coverage analysis with Codecov integration,
  and REUSE license compliance. Status badges added to README. Documentation includes
  workflow descriptions and troubleshooting guides.
- **Parameter Validation System**: New UI subsystem for configuration validation with
  `ParameterMetadata`, `ParameterValidator`, and `ValidationResult` classes. Supports nested
  path validation, finite checks, type validation, and helpful error messages. Integrated
  into Tungsten app with comprehensive test coverage (300+ assertions).
- **FFT Backend Selection**: Runtime FFT backend selection API allowing users to choose
  between available HeFFTe backends (FFTW, MKL, cuFFT) via configuration. New
  `examples/fft_backend_benchmark.cpp` demonstrates performance comparison. Backend field
  added to config schema with parsing and validation.
- **SparseVector & MPI Exchange**: New `SparseVector` container with halo exchange patterns
  for domain decomposition. Includes gather/scatter operations, neighbor exchange with MPI,
  and halo pattern creation utilities. Comprehensive test suite validates exchange correctness.
- **Testing Infrastructure**: First integration test suite for diffusion model validating
  complete simulation pipeline against analytical solutions (4 test cases, 331 assertions).
  Added benchmark subdirectory with microbenchmarks for World coordinate operations.
  Comprehensive unit tests for UI validation (300+ assertions), VTK writer (MPI-aware),
  DataBuffer, GPUVector, and SparseVector. Switched to single-invocation test model to
  prevent MPI initialization issues. Test coverage improvements across all modules.
- **World API**: Type-safe World construction using strong types from `strong_types.hpp`.
  Added new `create(GridSize, PhysicalOrigin, GridSpacing)` overload preventing parameter
  confusion at compile time. Old `create(Int3, Real3, Real3)` API deprecated. Zero overhead -
  strong types compile away completely. Updated all examples and helper functions. Test suite
  with 71 assertions covering type safety, zero overhead, and backward compatibility.
- **Documentation**: Added 10 comprehensive API examples (World, FFT, Simulator, Time,
  Decomposition, ResultsWriter, FieldModifier, DiscreteField, Model, custom field initializer).
  Added CITATION.cff for standardized citations. Improved Doxygen configuration. README
  sections on configuration validation, FFT backend selection, and extending OpenPFC.
- **Research Tools**: Added power consumption benchmarks for FFT operations (CPU and GPU),
  multi-GPU HeFFTe examples, and scalability testing applications for Tungsten model.

### Changed

- **CMake Structure**: Root `project()` moved to top-level CMakeLists.txt. Build options
  reorganized into logical modules. Test discovery switched to single-invocation model.
  Benchmark compilation now optional via `OpenPFC_BUILD_BENCHMARKS`.
- **Tungsten Structure**: Split monolithic tungsten code into modular headers and separate
  JSON inputs into `inputs_json/` subdirectory. Restructured JSON schema to nested format.
  Renamed 'origo' field to 'origin' for consistency.
- **UI Module**: Split monolithic `ui.hpp` into modular components. Made `plan_options`
  optional in app config. Added error formatting utilities for better user messages.
- **World Module**: Split `world.hpp` into modular headers. Added query helper examples.
  Updated coordinate benchmark documentation.
- **Test Organization**: Split monolithic parameter validation tests. Serialize VTK writer
  tests to prevent cleanup races. Make `MPI_Worker` static to persist MPI per process.
  Normalize test commands under single-invocation model.
- **Build Warnings**: Enabled additional compiler warnings for code quality in Debug builds.
  Added `-Werror=format-security`. Made GCC-specific warnings conditional. Format check
  warns instead of fails in Nix builds.
- **Dependencies**: Updated nixpkgs from 23.11 to 24.05. Added git and tomlplusplus to
  Nix build dependencies. Integrated Catch2 test discovery.

### Fixed

- **Build System**: Fixed CMake warnings by moving `project()` to root. Fixed Catch2 test
  discovery and optional MPI suites. Made documentation comment posting optional in CI.
  Cleaned up clang-format artifacts before REUSE checks. Improved error reporting in Nix tests.
- **Test Fixes**: Fixed narrowing conversions in sparse vector tests. Fixed GridSpacing
  initializers in FFT tests. Fixed syntax errors in world benchmark and CUDA tests. Added
  missing `pfc` namespace qualifiers. Suppressed unused variable/parameter warnings with
  `[[maybe_unused]]`. Fixed incorrectly converted `world::create` calls.
- **Application Fixes**: Fixed missing `set_fft()` call in diffusion example causing runtime
  errors. Removed unused fields (verbose in Diffusion, m_first in Aluminum). Fixed array
  initialization in SeedFCC. Added MPI-aware main to tungsten CPU vs CUDA test.
- **MPI Fixes**: Fixed `MPI_Worker` to be safe for test frameworks. Query current MPI
  rank/size when generating PVTI instead of using stale values. Synchronize ranks before
  cleanup in VTK writer test to prevent races.
- **Memory Safety**: Initialize all params struct members in aluminum to prevent undefined
  behavior. Add explicit template instantiation for World constructor. Fix CPU FFT
  `std::vector` interface to call HeFFTe directly.
- **Code Quality**: Removed redundant const qualifiers. Added missing override keywords.
  Fixed variable shadowing in multiple files. Removed variable shadowing in timing collection.
  Fixed clang-format violations across codebase.
- **CI/CD**: Removed ubuntu-20.04 from test matrix. Removed Cachix binary cache step.
  Initialized git submodules in all workflows. Made clang-format check warning instead of
  error. Used forked clang-format-action with fail-on-error option.
- **Documentation**: Removed internal tracking references from code. Added SPDX headers for
  REUSE compliance to all test READMEs. Fixed Doxygen file headers for better doc generation.

### Deprecated

- **World API**: Old `world::create(Int3, Real3, Real3)` deprecated in favor of type-safe
  `create(GridSize, PhysicalOrigin, GridSpacing)`. Migration guide in documentation.

### Breaking Changes

None - all deprecated APIs remain functional with warnings.

## [0.1.3] - 2025-11-25

### Added

- **Examples**: Custom coordinate system example in `examples/17_custom_coordinate_system.cpp`
  demonstrating OpenPFC's extensibility via ADL (Argument-Dependent Lookup). Implements
  complete polar (2D: r, θ) and spherical (3D: r, θ, φ) coordinate systems with coordinate
  transformations (`polar_to_coords()`, `polar_to_indices()`, `spherical_to_coords()`,
  `spherical_to_indices()`). Includes comprehensive Doxygen documentation (615 lines),
  round-trip transformation verification, and 4-step recipe showing users how to add
  custom coordinate systems without modifying OpenPFC source code. Embodies "Laboratory,
  Not Fortress" philosophy - users can extend with cylindrical, spherical, or custom
  geometries through tag-based dispatch and free functions. Example compiles cleanly
  with zero warnings and demonstrates working coordinate conversions with correct output.
- **Documentation**: Comprehensive API documentation for top 10 most-used public
  APIs with detailed @example blocks and usage patterns. Enhanced documentation
  covers World (domain creation and coordinate transforms), Model (physics
  implementation), Simulator (time integration orchestration), FFT (spectral
  transforms), Time (time stepping), Decomposition (parallel decomposition),
  ResultsWriter (output formats), FieldModifier (IC/BC extensibility), and
  DiscreteField (coordinate-aware fields). Added 10 standalone example programs
  (4,570+ lines) demonstrating complete usage workflows from basic setup to
  production PFC simulations. Includes build system integration via
  docs/api/examples/CMakeLists.txt with BUILD_API_EXAMPLES option. Documentation
  warnings reduced from 9 to 1 (89% improvement). All examples validated and
  test suite confirms no regressions (73 test cases, 5,836 assertions passing).
- **FFT**: K-space helper functions in `include/openpfc/fft/kspace.hpp` providing
  zero-cost abstractions for wave vector calculations in spectral methods.
  Added 4 inline helper functions: `k_frequency_scaling(world)` for computing
  frequency scaling factors (2π/L), `k_component(index, size, freq_scale)` for
  wave vector components with Nyquist folding, `k_laplacian_value(ki, kj, kk)`
  for computing -k² Laplacian operator, and `k_squared_value(ki, kj, kk)` for
  magnitude squared. Eliminates 120+ lines of duplicated k-space calculation
  code across examples (04_diffusion_model.cpp, 12_cahn_hilliard.cpp, tungsten.cpp,
  etc.). All functions are inline, noexcept, and compile to identical machine
  code as manual implementation (zero runtime overhead). Comprehensive test
  coverage (177 assertions in 6 test cases). Example program added in
  `examples/fft_kspace_helpers_example.cpp` demonstrating before/after comparison.
- **DiscreteField**: Converted `interpolate()` from member function to free function
  `pfc::interpolate(field, coords)` aligning with OpenPFC's "structs + free functions"
  design philosophy. Added both mutable and const overloads for type safety. Member
  function deprecated with `[[deprecated]]` attribute for v1.x backward compatibility
  (will be removed in v2.0). Free function enables ADL-based extension allowing users
  to provide custom interpolation schemes without modifying OpenPFC. Updated all
  11 call sites across tests, examples, and documentation to use new API. Added
  comprehensive test coverage (95+ new test lines) including mutable/const overloads,
  ADL lookup verification, and nearest-neighbor rounding behavior tests. All 222
  assertions pass. Zero runtime overhead maintained (inline functions).

## [0.1.2] - 2025-11-21

### Added

- **Core**: World construction helper functions in `include/openpfc/core/world.hpp`
  providing ergonomic, zero-cost abstractions for common grid creation patterns.
  Added 5 inline helper functions: `uniform(int)` and `uniform(int, double)` for
  N³ grids, `from_bounds(...)` for automatic spacing computation from physical
  bounds (periodic/non-periodic aware), `with_spacing(...)` for custom spacing
  with default origin, and `with_origin(...)` for custom origin with unit spacing.
  All helpers include input validation with clear error messages. Reduces
  boilerplate from `world::create({64,64,64}, {0,0,0}, {1,1,1})` to
  `world::uniform(64)`. Backward compatible - existing `create()` API unchanged.
  Comprehensive test coverage (32 new assertions). Example program added in
  `examples/world_helpers_example.cpp`.
- **Core**: Mathematical constants in `include/openpfc/constants.hpp` for
  compile-time evaluation with zero runtime overhead. Added 12 constants: π,
  2π, π/2, π/4, 1/π, √π, √2, √3, e, ln(2), ln(10), and φ (golden ratio).
  All constants are `constexpr double` with 16+ decimal digits precision.
  Comprehensive Doxygen documentation included. Constants accessible via both
  `pfc::constants::pi` and `pfc::pi` namespaces. API matches C++20
  `std::numbers` for future migration.
- **Testing**: Comprehensive test suite for mathematical constants in
  `tests/unit/core/test_constants.cpp` with 13 test cases and 41 assertions
  covering precision verification, derived constants, compile-time evaluation,
  and integration scenarios (FFT wave numbers, crystal geometry).
- **Testing**: Pre-commit hook for automatic clang-format checking to prevent
  formatting issues before pushing to CI. Hook available in `scripts/pre-commit-hook`
  with installation instructions in `scripts/README.md`.
- **Testing**: Comprehensive test coverage improvements achieving 90.7% line
  coverage and 94.8% function coverage. Added tests for `utils.hpp`,
  `world.cpp`, and `fixed_bc.hpp`.
- **Build system**: Added `-Werror=format-security` compiler flag to catch
  format string vulnerabilities locally before CI, matching CI behavior.
- **Documentation**: Added SPDX license headers to test README files
  (`tests/`, `tests/benchmarks/`, `tests/fixtures/`, `tests/integration/`,
  `tests/unit/`) for REUSE compliance (174/174 files now compliant).
- **Documentation**: Added comprehensive `@file` documentation tags to all 41
  public header files in `include/openpfc/` achieving 100% coverage. Each header
  now includes brief description, detailed explanation, practical usage examples,
  and cross-references to related components. Reduced Doxygen @file warnings
  from 47 to 0. Improves API discoverability for new users and enables better
  IDE/LLM assistance.

### Fixed

- **Examples**: Replaced runtime pi calculation (`std::atan(1.0) * 4.0`) with
  compile-time `pfc::constants::pi` in `diffusion_model.hpp`,
  `12_cahn_hilliard.cpp`, and `05_simulator.cpp` for zero runtime overhead in
  FFT wave number calculations. Removed unused global PI constants.
- **CMake build system**: Fixed Catch2 detection in `FindCatch2.cmake` by
  explicitly setting `Catch2_FOUND` variable after `FetchContent_MakeAvailable`.
  This enables the test suite to build when `OpenPFC_BUILD_TESTS=ON`.
- **CMake build system**: Fixed HeFFTe detection in `FindHeffte.cmake` by
  setting `Heffte_FOUND=TRUE` after FetchContent to prevent fatal errors when
  HeFFTe is downloaded instead of using system-installed package.
- **tungsten application**: Added explicit `find_package(Heffte REQUIRED)` and
  corrected target link to `Heffte::Heffte` to ensure proper linkage with
  separately installed HeFFTe v2.4.1.
- **Code quality**: Fixed format-security compiler error in `utils.hpp` by
  adding overload for `string_format()` with no variadic arguments.
- **Code formatting**: Removed trailing whitespace in `test_fft.cpp` to pass
  clang-format checks.

### Breaking Changes

- **Model::rank0 is now private**: The public member variable `rank0` has been
  moved to private section and renamed to `m_rank0`. Use the `Model::is_rank0()`
  method instead.
  - **Migration**: Replace `model.rank0` with `model.is_rank0()` in your code
  - **Reason**: Better encapsulation and consistent API with other query methods
    like `get_world()` and `get_fft()`
  - **Impact**: All examples and applications updated to use the new API
  - **Note**: The method `is_rank0()` is now `const` and `inline` for zero overhead

## [0.1.1] - 2024-06-13

- Make some changes to tungsten and aluminum models to be more consistent with
  the use of minus signs in different operators: move minus sign from peak
  function to opCk operator (commits 8685f7a and b4392b3).
- Bug fixes and changes in CMakeLists.txt: conditionally install nlohmann_json
  headers (issue #16), do not add RPATH to binaries when installing them,
  (commit 6c91de3) and also install binaries to INSTALL_PREFIX/bin (issue #14).
- Start using clang-format in the project (ci pipeline). (Issue #43)
- Add possibility to add initial and boundary conditions to fields with other
  name than "default". (Commit c65fb23)
- Add schema file for the input file. (Commit 6eeeab9)
- Fix license headers in source files, add license header checker to GH Action
  and in general improve licensing information. (Issues #25, #39, #40)
- Replace `#pragma once` with a proper include guard in all header files. (Issue
  #48)
- Fix bug with clang-tidy configuration preventing compilation. (Issue #52)
- Major updates to README.md: update citing information, add description of
  application structure, add new images, scalability results, and add example
  simulation of Cahn-Hilliard equation. (Issues #5, #19, #22, #23, #27, #28,
  #40)

## [0.1.0] - 2023-08-17

- Initial release.
