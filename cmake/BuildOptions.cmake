# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Build options for apps, examples, and tests. OpenPFC_BUILD_BENCHMARKS
# toggles sources under tests/benchmarks/ via tests/CMakeLists.txt.

option(OpenPFC_BUILD_APPS "Build OpenPFC applications" ON)
option(OpenPFC_BUILD_EXAMPLES "Build OpenPFC examples" ON)
option(OpenPFC_BUILD_TESTS "Build OpenPFC tests" ON)
option(OpenPFC_BUILD_BENCHMARKS "Build performance benchmarks (slow tests)" OFF)

if(OpenPFC_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 REQUIRED)
  message(STATUS "✅ Catch2 v${Catch2_VERSION} found at ${Catch2_DIR}")
  if(OpenPFC_ENABLE_HEFFTE)
    message(STATUS "🔍 Building tests")
    add_subdirectory(tests)
  else()
    message(STATUS
      "⏭️  Skipping tests/: OpenPFC_ENABLE_HEFFTE=OFF (many Catch2 TUs include fft_fftw.hpp). "
      "FD app tests (e.g. heat3d) still register.")
  endif()
endif()

if(OpenPFC_BUILD_APPS)
  message(STATUS "📦 Building applications")
  add_subdirectory(apps)
endif()

if(OpenPFC_BUILD_EXAMPLES)
  if(OpenPFC_ENABLE_HEFFTE)
    message(STATUS "📚 Building examples")
    add_subdirectory(examples)
  else()
    message(STATUS "⏭️  Skipping examples: OpenPFC_ENABLE_HEFFTE=OFF")
  endif()
endif()
