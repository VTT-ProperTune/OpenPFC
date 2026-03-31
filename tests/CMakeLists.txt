# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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

# Link against OpenPFC and Catch2, and add HeFFTe if available
target_link_libraries(openpfc-tests PRIVATE OpenPFC Catch2::Catch2)
if(TARGET Heffte::Heffte)
  target_link_libraries(openpfc-tests PRIVATE Heffte::Heffte)
elseif(TARGET Heffte)
  target_link_libraries(openpfc-tests PRIVATE Heffte)
elseif(TARGET heffte)
  target_link_libraries(openpfc-tests PRIVATE heffte)
endif()

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
# When ON, register mpi_3procs_ring / mpi_4procs_grid_multiple even if this machine
# reports fewer logical CPUs at configure time (e.g. configure on a login node, run
# tests on a compute node with mpirun --oversubscribe).
option(OpenPFC_MPI_TEST_REGISTER_HIGH_RANK_ALWAYS
       "Always register 3- and 4-rank MPI CTest targets regardless of configure host CPU count"
       OFF)
set(OpenPFC_MPI_TEST_MAX_WORLD_SIZE
    "0"
    CACHE STRING
    "If > 0, omit MPI CTest targets that need more ranks than this (e.g. 2 on runners where OpenMPI rejects mpirun -n 3+). 0 = no cap from this option."
)
find_package(MPI REQUIRED)
if(OpenPFC_RUN_MPI_SUITES AND MPI_CXX_FOUND)
  cmake_host_system_information(RESULT _openpfc_mpi_configure_cores
                                QUERY NUMBER_OF_LOGICAL_CORES)
  if(NOT _openpfc_mpi_configure_cores GREATER 0)
    set(_openpfc_mpi_configure_cores 2)
    message(
      STATUS
        "MPI CTest: NUMBER_OF_LOGICAL_CORES unavailable; assuming ${_openpfc_mpi_configure_cores} for multirank registration"
    )
  endif()
  set(_openpfc_mpi_register_3rank TRUE)
  set(_openpfc_mpi_register_4rank TRUE)
  if(NOT OpenPFC_MPI_TEST_REGISTER_HIGH_RANK_ALWAYS)
    if(_openpfc_mpi_configure_cores LESS 3)
      set(_openpfc_mpi_register_3rank FALSE)
      set(_openpfc_mpi_register_4rank FALSE)
      message(
        STATUS
          "MPI CTest: configure host reports ${_openpfc_mpi_configure_cores} logical CPU(s); "
          "skipping mpi_3procs_ring and mpi_4procs_grid_multiple (OpenMPI default slot limit). "
          "Reconfigure with -DOpenPFC_MPI_TEST_REGISTER_HIGH_RANK_ALWAYS=ON to register them anyway."
      )
    elseif(_openpfc_mpi_configure_cores LESS 4)
      set(_openpfc_mpi_register_4rank FALSE)
      message(
        STATUS
          "MPI CTest: configure host reports ${_openpfc_mpi_configure_cores} logical CPU(s); "
          "skipping mpi_4procs_grid_multiple. "
          "Use -DOpenPFC_MPI_TEST_REGISTER_HIGH_RANK_ALWAYS=ON to register it anyway."
      )
    endif()
  endif()

  # Optional hard cap (CI): logical CPU count can exceed OpenMPI slot limits on the same host
  if(OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0)
    if(OpenPFC_MPI_TEST_MAX_WORLD_SIZE LESS 3)
      set(_openpfc_mpi_register_3rank FALSE)
      set(_openpfc_mpi_register_4rank FALSE)
      message(
        STATUS
          "MPI CTest: OpenPFC_MPI_TEST_MAX_WORLD_SIZE=${OpenPFC_MPI_TEST_MAX_WORLD_SIZE} — "
          "skipping mpi_3procs_ring and mpi_4procs_grid_multiple")
    elseif(OpenPFC_MPI_TEST_MAX_WORLD_SIZE LESS 4)
      set(_openpfc_mpi_register_4rank FALSE)
      message(
        STATUS
          "MPI CTest: OpenPFC_MPI_TEST_MAX_WORLD_SIZE=${OpenPFC_MPI_TEST_MAX_WORLD_SIZE} — "
          "skipping mpi_4procs_grid_multiple")
    endif()
  endif()

  # 2-process MPI suite: runs MPI tests suitable for 2 ranks
  add_test(NAME mpi_2procs_all
       COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
           $<TARGET_FILE:openpfc-tests>
           "[MPI]~[grid]~[ring]~[multiple]"
  )
  set_tests_properties(
    mpi_2procs_all
    PROPERTIES TIMEOUT 120
               ENVIRONMENT "OPENPFC_TEST_MPI_INIT=1"
               RUN_SERIAL TRUE)

  # 3-process ring topology (skipped at configure time if host has < 3 logical CPUs)
  if(_openpfc_mpi_register_3rank)
    add_test(NAME mpi_3procs_ring
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
             $<TARGET_FILE:openpfc-tests>
             "[MPI][ring]"
    )
    set_tests_properties(
      mpi_3procs_ring
      PROPERTIES TIMEOUT 120
                 ENVIRONMENT "OPENPFC_TEST_MPI_INIT=1"
                 RUN_SERIAL TRUE)
  endif()

  # 4-process tests: grid (requires exactly 4) and multiple neighbors (>=4)
  if(_openpfc_mpi_register_4rank)
    add_test(NAME mpi_4procs_grid_multiple
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
             $<TARGET_FILE:openpfc-tests>
             "[MPI][grid],[MPI][multiple],[MPI][fd][persistent]"
    )
    set_tests_properties(
      mpi_4procs_grid_multiple
      PROPERTIES TIMEOUT 180
                 ENVIRONMENT "OPENPFC_TEST_MPI_INIT=1"
                 RUN_SERIAL TRUE)
  endif()
endif()

# Optionally, add a custom target to run tests manually
add_custom_target(run-tests COMMAND openpfc-tests)
