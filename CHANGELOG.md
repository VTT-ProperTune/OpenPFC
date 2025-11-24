<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Changelog

## [Unreleased]

### Added

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
  (User Story #0048)

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
  `examples/world_helpers_example.cpp`. (User Story #0030)
- **Core**: Mathematical constants in `include/openpfc/constants.hpp` for
  compile-time evaluation with zero runtime overhead. Added 12 constants: π,
  2π, π/2, π/4, 1/π, √π, √2, √3, e, ln(2), ln(10), and φ (golden ratio).
  All constants are `constexpr double` with 16+ decimal digits precision.
  Comprehensive Doxygen documentation included. Constants accessible via both
  `pfc::constants::pi` and `pfc::pi` namespaces. API matches C++20
  `std::numbers` for future migration. (User Story #0049)
- **Testing**: Comprehensive test suite for mathematical constants in
  `tests/unit/core/test_constants.cpp` with 13 test cases and 41 assertions
  covering precision verification, derived constants, compile-time evaluation,
  and integration scenarios (FFT wave numbers, crystal geometry). (User Story #0049)
- **Testing**: Pre-commit hook for automatic clang-format checking to prevent
  formatting issues before pushing to CI. Hook available in `scripts/pre-commit-hook`
  with installation instructions in `scripts/README.md`.
- **Testing**: Comprehensive test coverage improvements achieving 90.7% line
  coverage and 94.8% function coverage. Added tests for `utils.hpp`,
  `world.cpp`, and `fixed_bc.hpp`. (User Story #0044)
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
  IDE/LLM assistance. (User Story #0010)

### Fixed

- **Examples**: Replaced runtime pi calculation (`std::atan(1.0) * 4.0`) with
  compile-time `pfc::constants::pi` in `diffusion_model.hpp`,
  `12_cahn_hilliard.cpp`, and `05_simulator.cpp` for zero runtime overhead in
  FFT wave number calculations. Removed unused global PI constants. (User Story #0049)
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
