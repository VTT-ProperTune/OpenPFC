# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Higher-order PFC: one physics header, one JSON SpectralETDSession, CPU + HIP.

option(HIGHER_ORDER_PFC_ENABLE_TESTS "Enable building higher-order PFC tests" ON)

set(HIGHER_ORDER_PFC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_higher_order_pfc_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_higher_order_pfc_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_higher_order_pfc_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_higher_order_pfc_heffte_tgt heffte)
endif()

function(higher_order_pfc_cpu_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${HIGHER_ORDER_PFC_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_higher_order_pfc_heffte_tgt})
endfunction()

function(higher_order_pfc_hip_executable target)
  add_executable(${target} ${ARGN} src/gpu/higher_order_pfc_pointwise.hip)
  target_include_directories(${target} PRIVATE ${HIGHER_ORDER_PFC_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_higher_order_pfc_heffte_tgt}
                        hip::host openpfc_hip_kernels)
  set_target_properties(${target} PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
endfunction()

higher_order_pfc_cpu_executable(higher_order_pfc src/higher_order_pfc.cpp)
install(TARGETS higher_order_pfc DESTINATION bin)

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  higher_order_pfc_hip_executable(higher_order_pfc_hip src/hip/higher_order_pfc.cpp)
  install(TARGETS higher_order_pfc_hip DESTINATION bin)
  message(STATUS "higher_order_pfc_hip (MPI + HIP spectral) enabled")
  if(HIGHER_ORDER_PFC_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
    include(CTest)
    add_test(NAME higher-order-pfc-hip-smoke
             COMMAND $<TARGET_FILE:higher_order_pfc_hip>
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/two_mode_hip_smoke.json)
    set_tests_properties(higher-order-pfc-hip-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "SPECTRAL_CHECKSUM"
      TIMEOUT 180)
  endif()
endif()

if(HIGHER_ORDER_PFC_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    higher_order_pfc_cpu_executable(test_higher_order_pfc
      tests/test_higher_order_pfc.cpp
      tests/test_order_parameter.cpp)
    target_link_libraries(test_higher_order_pfc PRIVATE Catch2::Catch2)
    include(CTest)
    add_test(NAME higher-order-pfc-all-tests COMMAND test_higher_order_pfc)
    if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
       (OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
      add_test(NAME higher-order-pfc-mpi-tests
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:test_higher_order_pfc> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(higher-order-pfc-mpi-tests PROPERTIES TIMEOUT 120 PROCESSORS 2)
    endif()
  else()
    message(STATUS "higher_order_pfc: Catch2 not found, not building tests")
  endif()
endif()
