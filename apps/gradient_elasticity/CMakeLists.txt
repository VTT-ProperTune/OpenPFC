# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Strain-gradient elasticity: one-shot spectral 2x2 Helmholtz-Navier, CPU + HIP.

option(GRADIENT_ELASTICITY_ENABLE_TESTS "Enable building gradient-elasticity tests" ON)

set(GRADIENT_ELASTICITY_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_gradient_elasticity_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_gradient_elasticity_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_gradient_elasticity_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_gradient_elasticity_heffte_tgt heffte)
endif()

function(gradient_elasticity_cpu_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${GRADIENT_ELASTICITY_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_gradient_elasticity_heffte_tgt})
endfunction()

function(gradient_elasticity_hip_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${GRADIENT_ELASTICITY_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_gradient_elasticity_heffte_tgt}
                        hip::host openpfc_hip_kernels)
  set_target_properties(${target} PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
endfunction()

gradient_elasticity_cpu_executable(gradient_elasticity src/gradient_elasticity.cpp)
install(TARGETS gradient_elasticity DESTINATION bin)

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  gradient_elasticity_hip_executable(gradient_elasticity_hip
                                     src/hip/gradient_elasticity.cpp)
  install(TARGETS gradient_elasticity_hip DESTINATION bin)
  message(STATUS "gradient_elasticity_hip (MPI + HIP spectral) enabled")
  if(GRADIENT_ELASTICITY_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
    include(CTest)
    add_test(NAME gradient-elasticity-hip-smoke
             COMMAND $<TARGET_FILE:gradient_elasticity_hip>
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/gradient_elasticity_hip_smoke.json)
    set_tests_properties(gradient-elasticity-hip-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "SPECTRAL_CHECKSUM"
      TIMEOUT 180)
  endif()
endif()

if(GRADIENT_ELASTICITY_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    gradient_elasticity_cpu_executable(test_gradient_elasticity
                                       tests/test_gradient_elasticity.cpp)
    target_link_libraries(test_gradient_elasticity PRIVATE Catch2::Catch2)
    include(CTest)
    add_test(NAME gradient-elasticity-all-tests COMMAND test_gradient_elasticity)

    if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
      gradient_elasticity_hip_executable(test_gradient_elasticity_hip
                                         tests/test_gradient_elasticity_gpu.cpp)
      target_include_directories(test_gradient_elasticity_hip PRIVATE
          ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
      target_link_libraries(test_gradient_elasticity_hip PRIVATE Catch2::Catch2 MPI::MPI_CXX)
      add_test(NAME HIP_GradientElasticity COMMAND test_gradient_elasticity_hip)
    endif()
  else()
    message(STATUS "gradient elasticity: Catch2 not found, not building tests")
  endif()
endif()
