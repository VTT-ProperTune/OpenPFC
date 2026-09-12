# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Cahn–Hilliard: JSON SpectralETDSession (CPU + HIP) plus host-only
# coherent-elastic binary on the existing Green operator.

option(CAHN_HILLIARD_ENABLE_TESTS "Enable building Cahn–Hilliard tests" ON)

set(CAHN_HILLIARD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_cahn_hilliard_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_cahn_hilliard_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_cahn_hilliard_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_cahn_hilliard_heffte_tgt heffte)
endif()

function(cahn_hilliard_cpu_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${CAHN_HILLIARD_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_cahn_hilliard_heffte_tgt})
endfunction()

function(cahn_hilliard_hip_executable target)
  add_executable(${target} ${ARGN} src/gpu/cahn_hilliard_pointwise.hip)
  target_include_directories(${target} PRIVATE ${CAHN_HILLIARD_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_cahn_hilliard_heffte_tgt}
                        hip::host openpfc_hip_kernels)
  set_target_properties(${target} PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
endfunction()

cahn_hilliard_cpu_executable(cahn_hilliard src/cahn_hilliard.cpp)
install(TARGETS cahn_hilliard DESTINATION bin)

# Coherent Vegard misfit on the existing Green operator (host-only, #157).
# Not a seventeenth application: same Fe–Cr model, extra binary like
# thin_film_nonlinear.
cahn_hilliard_cpu_executable(cahn_hilliard_elastic src/cahn_hilliard_elastic.cpp)
install(TARGETS cahn_hilliard_elastic DESTINATION bin)

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  cahn_hilliard_hip_executable(cahn_hilliard_hip src/hip/cahn_hilliard.cpp)
  install(TARGETS cahn_hilliard_hip DESTINATION bin)
  message(STATUS "cahn_hilliard_hip (MPI + HIP spectral) enabled")
  if(CAHN_HILLIARD_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
    include(CTest)
    add_test(NAME cahn-hilliard-hip-smoke
             COMMAND $<TARGET_FILE:cahn_hilliard_hip>
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/fe_cr_hip_smoke.json)
    set_tests_properties(cahn-hilliard-hip-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "SPECTRAL_CHECKSUM"
      TIMEOUT 180)
  endif()
endif()

if(CAHN_HILLIARD_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    cahn_hilliard_cpu_executable(test_cahn_hilliard tests/test_cahn_hilliard.cpp)
    target_link_libraries(test_cahn_hilliard PRIVATE Catch2::Catch2)
    include(CTest)
    add_test(NAME cahn-hilliard-all-tests COMMAND test_cahn_hilliard)
    if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
       (OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
      add_test(NAME cahn-hilliard-mpi-tests
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:test_cahn_hilliard> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(cahn-hilliard-mpi-tests PROPERTIES TIMEOUT 120 PROCESSORS 2)
    endif()

    if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
      cahn_hilliard_hip_executable(test_cahn_hilliard_hip
        tests/test_cahn_hilliard_gpu.cpp)
      target_include_directories(test_cahn_hilliard_hip PRIVATE
          ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
      target_link_libraries(test_cahn_hilliard_hip PRIVATE Catch2::Catch2 MPI::MPI_CXX)
      add_test(NAME HIP_CahnHilliardETD COMMAND test_cahn_hilliard_hip)
    endif()
  else()
    message(STATUS "Cahn–Hilliard: Catch2 not found, not building tests")
  endif()
endif()
