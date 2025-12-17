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

# Add GPU tests if CUDA is enabled AND available
# This will be skipped on AMD systems or systems without CUDA
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    add_subdirectory(unit/gpu)
endif()

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

# Run all non-MPI tests in a single invocation to avoid per-test overhead
# This is much faster than catch_discover_tests which runs each test separately
add_test(NAME openpfc-all-tests
         COMMAND openpfc-tests "~[MPI]")
set_tests_properties(openpfc-all-tests PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

# Register MPI tests explicitly with mpiexec wrappers
option(OpenPFC_RUN_MPI_SUITES "Enable explicit MPI multi-rank test suites" OFF)
find_package(MPI REQUIRED)
if(OpenPFC_RUN_MPI_SUITES AND MPI_CXX_FOUND)
  # 2-process MPI suite: runs MPI tests suitable for 2 ranks
  add_test(NAME mpi_2procs_all
       COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
           $<TARGET_FILE:openpfc-tests>
           "[MPI]~[grid]~[ring]~[multiple]"
  )
  set_tests_properties(mpi_2procs_all PROPERTIES TIMEOUT 120 ENVIRONMENT "OPENPFC_TEST_MPI_INIT=1")

  # 3-process ring topology
  add_test(NAME mpi_3procs_ring
       COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
           $<TARGET_FILE:openpfc-tests>
           "[MPI][ring]"
  )
  set_tests_properties(mpi_3procs_ring PROPERTIES TIMEOUT 120 ENVIRONMENT "OPENPFC_TEST_MPI_INIT=1")

  # 4-process tests: grid (requires exactly 4) and multiple neighbors (>=4)
  add_test(NAME mpi_4procs_grid_multiple
       COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
           $<TARGET_FILE:openpfc-tests>
           "[MPI][grid],[MPI][multiple]"
  )
  set_tests_properties(mpi_4procs_grid_multiple PROPERTIES TIMEOUT 180 ENVIRONMENT "OPENPFC_TEST_MPI_INIT=1")
endif()

# Optionally, add a custom target to run tests manually
add_custom_target(run-tests COMMAND openpfc-tests)
