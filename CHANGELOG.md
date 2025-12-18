<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Changelog

## [Unreleased]

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
