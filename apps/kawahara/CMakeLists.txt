# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Kawahara capillary–gravity waves: complex ik^3/ik^5 spectral ETD, CPU + HIP.

option(KAWAHARA_ENABLE_TESTS "Enable building kawahara tests" ON)

set(KAWAHARA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_kawahara_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_kawahara_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_kawahara_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_kawahara_heffte_tgt heffte)
endif()

function(kawahara_cpu_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${KAWAHARA_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_kawahara_heffte_tgt})
endfunction()

function(kawahara_hip_executable target)
  add_executable(${target} ${ARGN} src/gpu/kawahara_pointwise.hip)
  target_include_directories(${target} PRIVATE ${KAWAHARA_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_kawahara_heffte_tgt}
                        hip::host openpfc_hip_kernels)
  set_target_properties(${target} PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
endfunction()

kawahara_cpu_executable(kawahara src/kawahara.cpp)
install(TARGETS kawahara DESTINATION bin)

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  kawahara_hip_executable(kawahara_hip src/hip/kawahara.cpp)
  install(TARGETS kawahara_hip DESTINATION bin)
  message(STATUS "kawahara_hip (MPI + HIP spectral) enabled")
  if(KAWAHARA_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
    include(CTest)
    add_test(NAME kawahara-hip-smoke
             COMMAND $<TARGET_FILE:kawahara_hip>
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/kawahara_hip_smoke.json)
    set_tests_properties(kawahara-hip-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "SPECTRAL_CHECKSUM"
      TIMEOUT 180)
  endif()
endif()

if(KAWAHARA_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    kawahara_cpu_executable(test_kawahara tests/test_kawahara.cpp)
    target_link_libraries(test_kawahara PRIVATE Catch2::Catch2)
    include(CTest)
    add_test(NAME kawahara-all-tests COMMAND test_kawahara)

    if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
      kawahara_hip_executable(test_kawahara_hip tests/test_kawahara_gpu.cpp)
      target_include_directories(test_kawahara_hip PRIVATE
          ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
      target_link_libraries(test_kawahara_hip PRIVATE Catch2::Catch2 MPI::MPI_CXX)
      add_test(NAME HIP_KawaharaETD COMMAND test_kawahara_hip)
    endif()
  else()
    message(STATUS "Kawahara: Catch2 not found, not building tests")
  endif()
endif()
