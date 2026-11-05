# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Cahn–Hilliard: JSON SpectralETDSession (CPU + HIP), the host coherent-elastic
# binary, and the HIP coherent-elastic binary on the device Green solver.

option(CAHN_HILLIARD_ENABLE_TESTS "Enable building Cahn–Hilliard tests" ON)

set(CAHN_HILLIARD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

function(cahn_hilliard_cpu_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${CAHN_HILLIARD_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC::openpfc
                        nlohmann_json::nlohmann_json )
endfunction()

function(cahn_hilliard_hip_executable target)
  add_executable(${target} ${ARGN} src/gpu/cahn_hilliard_pointwise.hip)
  target_include_directories(${target} PRIVATE ${CAHN_HILLIARD_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC::openpfc
                        nlohmann_json::nlohmann_json
                        hip::host openpfc_hip_kernels)
  set_target_properties(${target} PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
endfunction()

cahn_hilliard_cpu_executable(cahn_hilliard src/cahn_hilliard.cpp)
install(TARGETS cahn_hilliard DESTINATION bin)

# Coherent Vegard misfit on the existing Green operator (host-only, #157).
# Not a separate application: same Fe–Cr model, extra binary like
# thin_film_nonlinear.
cahn_hilliard_cpu_executable(cahn_hilliard_elastic src/cahn_hilliard_elastic.cpp)
install(TARGETS cahn_hilliard_elastic DESTINATION bin)

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  cahn_hilliard_hip_executable(cahn_hilliard_hip src/hip/cahn_hilliard.cpp)
  install(TARGETS cahn_hilliard_hip DESTINATION bin)
  message(STATUS "cahn_hilliard_hip (MPI + HIP spectral) enabled")
  if(TARGET openpfc_microelasticity_hip)
    add_executable(cahn_hilliard_elastic_hip
      src/hip/cahn_hilliard_elastic.cpp
      src/gpu/elastic_coupled_kernels.hip)
    target_include_directories(cahn_hilliard_elastic_hip PRIVATE
      ${CAHN_HILLIARD_INCLUDE_DIR})
    target_link_libraries(cahn_hilliard_elastic_hip PRIVATE OpenPFC::openpfc
      nlohmann_json::nlohmann_json hip::host openpfc_hip_kernels
      openpfc_microelasticity_hip)
    target_compile_definitions(cahn_hilliard_elastic_hip PRIVATE
      OpenPFC_ENABLE_HIP_SPECTRAL)
    set_target_properties(cahn_hilliard_elastic_hip PROPERTIES
      HIP_STANDARD 20
      HIP_STANDARD_REQUIRED ON)
    install(TARGETS cahn_hilliard_elastic_hip DESTINATION bin)
    message(STATUS "cahn_hilliard_elastic_hip (device Green) enabled")
    if(CAHN_HILLIARD_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
      add_test(NAME cahn-hilliard-elastic-hip-smoke
               COMMAND $<TARGET_FILE:cahn_hilliard_elastic_hip>
                       ${CMAKE_CURRENT_SOURCE_DIR}/tests/fe_cr_elastic_hip_smoke.json)
      set_tests_properties(cahn-hilliard-elastic-hip-smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "CAHN_HILLIARD_ELASTIC_HIP"
        TIMEOUT 180)
    endif()
  endif()
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
    cahn_hilliard_cpu_executable(test_cahn_hilliard
      tests/test_cahn_hilliard.cpp tests/test_bounded_axis.cpp
      tests/test_elastic_coupled_host.cpp)
    target_link_libraries(test_cahn_hilliard PRIVATE Catch2::Catch2)
    include(CTest)
    add_test(NAME cahn-hilliard-all-tests COMMAND test_cahn_hilliard)
    add_test(NAME cahn-hilliard-bounded COMMAND test_cahn_hilliard "[bounded]")
    set_tests_properties(cahn-hilliard-bounded PROPERTIES TIMEOUT 120)
    add_test(NAME cahn-hilliard-elastic-coupled
             COMMAND test_cahn_hilliard "[elastic_coupled]")
    set_tests_properties(cahn-hilliard-elastic-coupled PROPERTIES TIMEOUT 180)
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
      if(TARGET openpfc_microelasticity_hip)
        add_executable(test_cahn_hilliard_elastic_hip
          tests/test_elastic_coupled_hip.cpp
          src/gpu/elastic_coupled_kernels.hip)
        target_include_directories(test_cahn_hilliard_elastic_hip PRIVATE
          ${CAHN_HILLIARD_INCLUDE_DIR}
          ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
        target_link_libraries(test_cahn_hilliard_elastic_hip PRIVATE
          OpenPFC::openpfc nlohmann_json::nlohmann_json hip::host
          openpfc_hip_kernels openpfc_microelasticity_hip Catch2::Catch2
          MPI::MPI_CXX)
        target_compile_definitions(test_cahn_hilliard_elastic_hip PRIVATE
          OpenPFC_ENABLE_HIP_SPECTRAL)
        set_target_properties(test_cahn_hilliard_elastic_hip PROPERTIES
          HIP_STANDARD 20
          HIP_STANDARD_REQUIRED ON)
        add_test(NAME HIP_CahnHilliardElastic
                 COMMAND test_cahn_hilliard_elastic_hip)
        set_tests_properties(HIP_CahnHilliardElastic PROPERTIES
          TIMEOUT 180
          SKIP_RETURN_CODE 4)
      endif()
    endif()
  else()
    message(STATUS "Cahn–Hilliard: Catch2 not found, not building tests")
  endif()
endif()
