# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Header-only helpers shared by FD apps and PFC solidification BCs.
add_library(openpfc_apps_common INTERFACE)
target_include_directories(openpfc_apps_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(openpfc_apps_common INTERFACE OpenPFC)

# Tests for the header-only pieces that carry real numerics rather than
# plumbing. Needs the spectral stack, so it is gated on HeFFTe exactly like the
# spectral apps in apps/CMakeLists.txt.
option(APPS_COMMON_ENABLE_TESTS "Enable building apps/common tests" ON)

# plasma_dispersion.hpp is pure arithmetic: standard library only, no MPI, no
# FFT, no OpenPFC. It is deliberately registered OUTSIDE the HeFFTe gate
# below, because it is the oracle every linear plasma benchmark is measured
# against and a build without the spectral stack should still be able to
# check that the oracle itself is right.
if(APPS_COMMON_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2WithMain)
    add_executable(test_plasma_dispersion tests/test_plasma_dispersion.cpp)
    target_include_directories(test_plasma_dispersion PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(test_plasma_dispersion PRIVATE Catch2::Catch2WithMain)
    include(CTest)
    add_test(NAME apps-common-plasma-dispersion COMMAND test_plasma_dispersion)
    set_tests_properties(apps-common-plasma-dispersion PROPERTIES TIMEOUT 300)
  endif()
endif()

if(APPS_COMMON_ENABLE_TESTS AND OpenPFC_BUILD_TESTS AND OpenPFC_ENABLE_HEFFTE)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    set(_apps_common_heffte_tgt "")
    if(TARGET Heffte::Heffte)
      set(_apps_common_heffte_tgt Heffte::Heffte)
    elseif(TARGET Heffte)
      set(_apps_common_heffte_tgt Heffte)
    elseif(TARGET heffte)
      set(_apps_common_heffte_tgt heffte)
    endif()

    add_executable(test_microelasticity tests/test_microelasticity.cpp)
    target_link_libraries(test_microelasticity PRIVATE OpenPFC openpfc_apps_common
                          Catch2::Catch2 ${_apps_common_heffte_tgt})
    include(CTest)
    add_test(NAME apps-common-microelasticity COMMAND test_microelasticity)
    set_tests_properties(apps-common-microelasticity PROPERTIES TIMEOUT 300)

    if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
       (OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR
        OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
      # Same assertions, two ranks: the golden numbers in the [mpi] case are
      # what make this a 1-vs-N comparison rather than a smoke test.
      add_test(NAME apps-common-microelasticity-mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:test_microelasticity> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(apps-common-microelasticity-mpi PROPERTIES
        TIMEOUT 300 PROCESSORS 2)
    endif()
  else()
    message(STATUS "apps/common: Catch2 not found, not building tests")
  endif()
endif()
