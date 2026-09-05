# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Tungsten: one physics header (`tungsten_physics.hpp`), one generic session
# (`pfc::ui::SpectralETDSession`), three backends. The only device code is the
# instantiation of the pointwise nonlinearity in `src/gpu/tungsten_pointwise.inc`
# (stamped into `.cu` / `.hip`), compiled into every GPU target that runs the
# device `SpectralETDSystem`.

set(TUNGSTEN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/apps/common/include)

set(_tungsten_heffte_tgt $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>)
if(NOT TARGET Heffte::Heffte AND NOT TARGET heffte)
  find_package(Heffte REQUIRED)
endif()

function(tungsten_cpu_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${TUNGSTEN_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_tungsten_heffte_tgt})
endfunction()

function(tungsten_cuda_executable target)
  add_executable(${target} ${ARGN} src/gpu/tungsten_pointwise.cu)
  target_include_directories(${target} PRIVATE ${TUNGSTEN_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_tungsten_heffte_tgt}
                        CUDA::cudart openpfc_gpu_kernels)
  set_target_properties(${target} PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
endfunction()

function(tungsten_hip_executable target)
  add_executable(${target} ${ARGN} src/gpu/tungsten_pointwise.hip)
  target_include_directories(${target} PRIVATE ${TUNGSTEN_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_tungsten_heffte_tgt}
                        hip::host openpfc_hip_kernels)
  set_target_properties(${target} PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
endfunction()

# ---- production binaries ---------------------------------------------------
tungsten_cpu_executable(tungsten src/cpu/tungsten.cpp)
install(TARGETS tungsten DESTINATION bin)
# `tungsten_etd` is the historical name of the 0.2 binary; same source.
tungsten_cpu_executable(tungsten_etd src/cpu/tungsten.cpp)
install(TARGETS tungsten_etd DESTINATION bin)

if(OpenPFC_ENABLE_CUDA_SPECTRAL)
  tungsten_cuda_executable(tungsten_cuda src/cuda/tungsten.cpp)
  install(TARGETS tungsten_cuda DESTINATION bin)
  tungsten_cuda_executable(tungsten_etd_cuda src/cuda/tungsten.cpp)
  install(TARGETS tungsten_etd_cuda DESTINATION bin)
  message(STATUS "Tungsten CUDA version enabled")
endif()
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE AND NOT OpenPFC_ENABLE_CUDA_SPECTRAL)
  message(STATUS
    "Tungsten CUDA version disabled: CUDA toolkit is available, but CUDA-enabled "
    "HeFFTe is missing (spectral CUDA support OFF).")
endif()

# HIP: GPU-aware MPI probe builds whenever HIP is available (no rocFFT HeFFTe required).
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(verify_gpu_aware_mpi src/verify_gpu_aware_mpi.cpp)
  target_link_libraries(verify_gpu_aware_mpi PRIVATE MPI::MPI_CXX hip::host)
  set_target_properties(verify_gpu_aware_mpi PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
  )
  install(TARGETS verify_gpu_aware_mpi DESTINATION bin)
  message(STATUS "verify_gpu_aware_mpi enabled (HIP + MPI device buffer test)")
  if(OpenPFC_BUILD_TESTS AND OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME HIP_VerifyGpuAwareMpi
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     $<TARGET_FILE:verify_gpu_aware_mpi>)
    set_tests_properties(HIP_VerifyGpuAwareMpi PROPERTIES
      PASS_REGULAR_EXPRESSION
      "OK: device-buffer MPI_Send/Recv succeeded"
      ENVIRONMENT
      "MPICH_GPU_SUPPORT_ENABLED=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  tungsten_hip_executable(tungsten_hip src/hip/tungsten.cpp)
  install(TARGETS tungsten_hip DESTINATION bin)
  tungsten_hip_executable(tungsten_etd_hip src/hip/tungsten.cpp)
  install(TARGETS tungsten_etd_hip DESTINATION bin)
  message(STATUS "Tungsten HIP version enabled")
elseif(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND NOT OpenPFC_ENABLE_HIP_SPECTRAL)
  message(STATUS
    "Tungsten HIP spectral targets disabled: ROCm HeFFTe (rocFFT) missing (HIP spectral support OFF).")
endif()

# ---- tests -----------------------------------------------------------------
option(TUNGSTEN_ENABLE_TESTS "Enable building Tungsten tests" ON)

if(TUNGSTEN_ENABLE_TESTS)
  find_package(Catch2 REQUIRED)
  if(TARGET Catch2::Catch2)
    tungsten_cpu_executable(test_tungsten tests/test_tungsten.cpp
                                          tests/test_tungsten_physics.cpp)
    target_link_libraries(test_tungsten PRIVATE Catch2::Catch2)
    include(CTest)
    # Run all tungsten tests in a single invocation to avoid per-test MPI overhead
    add_test(NAME tungsten-all-tests COMMAND test_tungsten)
    add_test(NAME tungsten-cpu-golden COMMAND test_tungsten "[etd_cpu_golden]")
    add_test(NAME tungsten-etd-cpu-golden COMMAND test_tungsten "[etd_cpu_golden]")
    if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
      if(OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0 AND
         OpenPFC_MPI_TEST_MAX_WORLD_SIZE LESS 4)
        message(STATUS
          "Skipping tungsten-golden-4rank: OpenPFC_MPI_TEST_MAX_WORLD_SIZE="
          "${OpenPFC_MPI_TEST_MAX_WORLD_SIZE}")
      else()
        add_test(NAME tungsten-golden-4rank
          COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                  $<TARGET_FILE:test_tungsten> "[golden][MPI]")
        set_tests_properties(tungsten-golden-4rank PROPERTIES
          TIMEOUT 180
          RUN_SERIAL TRUE)
      endif()
    endif()

    if(OpenPFC_ENABLE_CUDA_SPECTRAL)
      tungsten_cuda_executable(test_tungsten_etd_cuda tests/test_tungsten_etd_gpu.cpp)
      target_include_directories(test_tungsten_etd_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
      target_link_libraries(test_tungsten_etd_cuda PRIVATE Catch2::Catch2 MPI::MPI_CXX)
      target_compile_definitions(test_tungsten_etd_cuda PRIVATE
        OPENPFC_TEST_TUNGSTEN_ETD_CUDA)
      add_test(NAME CUDA_TungstenETD COMMAND test_tungsten_etd_cuda)
      add_test(NAME tungsten-etd-cpu-vs-cuda COMMAND test_tungsten_etd_cuda "[parity32]")
      add_test(NAME tungsten-cpu-vs-cuda-tests COMMAND test_tungsten_etd_cuda "[parity32]")

      tungsten_cuda_executable(test_tungsten_cuda_vtk tests/test_tungsten_cuda_vtk.cpp)
      message(STATUS "Tungsten CUDA VTK test program enabled")
    endif()

    if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
      tungsten_hip_executable(test_tungsten_etd_hip tests/test_tungsten_etd_gpu.cpp)
      target_include_directories(test_tungsten_etd_hip PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
      target_link_libraries(test_tungsten_etd_hip PRIVATE Catch2::Catch2 MPI::MPI_CXX)
      target_compile_definitions(test_tungsten_etd_hip PRIVATE
        OPENPFC_TEST_TUNGSTEN_ETD_HIP)
      add_test(NAME HIP_TungstenETD COMMAND test_tungsten_etd_hip)

      tungsten_hip_executable(test_tungsten_hip_vtk tests/test_tungsten_hip_vtk.cpp)
      message(STATUS "Tungsten HIP VTK test program enabled")
    endif()
  else()
    message(WARNING "Tungsten: Catch2 not found, not building tests")
  endif()
endif()

# ---- scalability study -----------------------------------------------------
if(OpenPFC_ENABLE_CUDA_SPECTRAL)
  tungsten_cuda_executable(tungsten_scalability src/tungsten_scalability.cpp)
elseif(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  tungsten_hip_executable(tungsten_scalability src/tungsten_scalability.cpp)
  target_compile_definitions(tungsten_scalability PRIVATE OpenPFC_ENABLE_HIP_SPECTRAL)
else()
  tungsten_cpu_executable(tungsten_scalability src/tungsten_scalability.cpp)
endif()
target_link_libraries(tungsten_scalability PRIVATE MPI::MPI_CXX)
install(TARGETS tungsten_scalability DESTINATION bin)
