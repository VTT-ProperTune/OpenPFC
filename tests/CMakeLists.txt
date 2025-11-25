# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#top

find_package(Catch2 REQUIRED)

# Create test executable with just the runner initially
add_executable(openpfc-tests runtests.cpp)

# Add tests/ directory to include path so fixtures/ can be found
target_include_directories(openpfc-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Add test subdirectories (they will add their sources via target_sources)
add_subdirectory(unit)
add_subdirectory(integration)

# Benchmarks are optional (can be slow)
if(OpenPFC_BUILD_BENCHMARKS)
  message(STATUS "⏱️  Including performance benchmarks in test suite")
  add_subdirectory(benchmarks)
else()
  message(STATUS "⏭️  Skipping performance benchmarks (enable with -DOpenPFC_BUILD_BENCHMARKS=ON)")
endif()

# Link against OpenPFC and Catch2
target_link_libraries(openpfc-tests PRIVATE OpenPFC Catch2::Catch2)

# Attach Catch2 include dirs manually for compile_commands.json
get_target_property(_catch2_include_dirs Catch2::Catch2 INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Catch2 includes: ${_catch2_include_dirs}")
target_include_directories(openpfc-tests PRIVATE ${_catch2_include_dirs})
foreach(dir IN LISTS _catch2_include_dirs)
  target_compile_options(openpfc-tests PRIVATE "-isystem" "${dir}")
endforeach()

if(OpenPFC_ENABLE_CODE_COVERAGE)
  # Enable code coverage if OpenPFC_ENABLE_CODE_COVERAGE is set
  message(STATUS "Enabling code coverage for tests")
  target_compile_options(openpfc-tests PRIVATE --coverage)
  target_link_options(openpfc-tests PRIVATE --coverage)
endif()

# Disable automatic test discovery during the build
# Comment out or remove the following line:
# catch_discover_tests(openpfc-tests)

# Optionally, add a custom target to run tests manually
add_custom_target(run-tests COMMAND openpfc-tests)
