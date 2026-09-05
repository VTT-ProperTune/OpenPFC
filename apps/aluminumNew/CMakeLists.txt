# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Aluminum: one physics header (`aluminum_physics.hpp`), one generic session
# (`pfc::ui::SpectralETDSession`), three backends. The only device code is the
# instantiation of the pointwise nonlinearity in `src/gpu/aluminum_pointwise.inc`
# (stamped into `.cu` / `.hip`), compiled into every GPU target.

cmake_minimum_required(VERSION 3.21)
project(Aluminum VERSION 0.2.0 LANGUAGES CXX)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    find_package(OpenPFC REQUIRED)
    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
    find_package(nlohmann_json REQUIRED)
    set(CMAKE_CXX_STANDARD 20)
endif()

set(ALUMINUM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/apps/common/include)

if(TARGET Heffte::Heffte)
  set(_aluminum_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_aluminum_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_aluminum_heffte_tgt heffte)
else()
  set(_aluminum_heffte_tgt "")
endif()

function(aluminum_cpu_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${ALUMINUM_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_aluminum_heffte_tgt})
endfunction()

function(aluminum_cuda_executable target)
  add_executable(${target} ${ARGN} src/gpu/aluminum_pointwise.cu)
  target_include_directories(${target} PRIVATE ${ALUMINUM_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_aluminum_heffte_tgt}
                        CUDA::cudart openpfc_gpu_kernels)
  set_target_properties(${target} PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
endfunction()

function(aluminum_hip_executable target)
  add_executable(${target} ${ARGN} src/gpu/aluminum_pointwise.hip)
  target_include_directories(${target} PRIVATE ${ALUMINUM_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_aluminum_heffte_tgt}
                        hip::host openpfc_hip_kernels)
  set_target_properties(${target} PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
endfunction()

# ---- production binaries ---------------------------------------------------
aluminum_cpu_executable(aluminumNew src/aluminum.cpp)
install(TARGETS aluminumNew DESTINATION bin)
# `aluminum_etd` is the historical name of the 0.2 binary; same source.
aluminum_cpu_executable(aluminum_etd src/aluminum.cpp)
install(TARGETS aluminum_etd DESTINATION bin)

if(OpenPFC_ENABLE_CUDA_SPECTRAL)
  aluminum_cuda_executable(aluminum_etd_cuda src/aluminum_cuda.cpp)
  install(TARGETS aluminum_etd_cuda DESTINATION bin)
  message(STATUS "aluminum_etd_cuda enabled")
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  aluminum_hip_executable(aluminum_etd_hip src/aluminum_hip.cpp)
  install(TARGETS aluminum_etd_hip DESTINATION bin)
  message(STATUS "aluminum_etd_hip enabled")
endif()

# ---- tests -----------------------------------------------------------------
option(ALUMINUM_ENABLE_TESTS "Enable building Aluminum tests" ON)

if(ALUMINUM_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2WithMain)
    aluminum_cpu_executable(aluminumTest aluminumTest.cpp tests/test_aluminum_physics.cpp)
    target_link_libraries(aluminumTest PRIVATE Catch2::Catch2WithMain)
    include(CTest)
    # Run all aluminum tests in a single invocation to avoid per-test MPI overhead
    add_test(NAME aluminum-all-tests COMMAND aluminumTest)
    add_test(NAME aluminum-etd-cpu-golden COMMAND aluminumTest
        "[seed_grid_fcc][session]")
    if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
      if(OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0 AND
         OpenPFC_MPI_TEST_MAX_WORLD_SIZE LESS 4)
        message(STATUS
          "Skipping aluminum-golden-4rank: OpenPFC_MPI_TEST_MAX_WORLD_SIZE="
          "${OpenPFC_MPI_TEST_MAX_WORLD_SIZE}")
      else()
        add_test(NAME aluminum-golden-4rank
          COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                  $<TARGET_FILE:aluminumTest> "[golden][MPI]")
        set_tests_properties(aluminum-golden-4rank PROPERTIES
          TIMEOUT 180
          RUN_SERIAL TRUE)
      endif()
    endif()

    if(OpenPFC_ENABLE_CUDA_SPECTRAL)
      aluminum_cuda_executable(test_aluminum_etd_cuda tests/test_aluminum_etd_gpu.cpp)
      target_include_directories(test_aluminum_etd_cuda PRIVATE
          ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
      target_link_libraries(test_aluminum_etd_cuda PRIVATE Catch2::Catch2 MPI::MPI_CXX)
      target_compile_definitions(test_aluminum_etd_cuda PRIVATE
          OPENPFC_TEST_ALUMINUM_ETD_CUDA)
      add_test(NAME CUDA_AluminumETD COMMAND test_aluminum_etd_cuda)
    endif()

    if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
      aluminum_hip_executable(test_aluminum_etd_hip tests/test_aluminum_etd_gpu.cpp)
      target_include_directories(test_aluminum_etd_hip PRIVATE
          ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
      target_link_libraries(test_aluminum_etd_hip PRIVATE Catch2::Catch2 MPI::MPI_CXX)
      target_compile_definitions(test_aluminum_etd_hip PRIVATE
          OPENPFC_TEST_ALUMINUM_ETD_HIP)
      add_test(NAME HIP_AluminumETD COMMAND test_aluminum_etd_hip)
    endif()
  else()
    message(STATUS "Aluminum: Catch2 not found, not building tests")
  endif()
endif()
