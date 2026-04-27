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
  message(STATUS "🔍 Building tests")
  enable_testing()
  find_package(Catch2 REQUIRED)
  message(STATUS "✅ Catch2 v${Catch2_VERSION} found at ${Catch2_DIR}")
  add_subdirectory(tests)
  message(STATUS "Installing openpfc-tests binary")
  install(TARGETS openpfc-tests DESTINATION bin)
endif()

if(OpenPFC_BUILD_APPS)
  message(STATUS "📦 Building applications")
  add_subdirectory(apps)
endif()

if(OpenPFC_BUILD_EXAMPLES)
  message(STATUS "📚 Building examples")
  add_subdirectory(examples)
endif()
