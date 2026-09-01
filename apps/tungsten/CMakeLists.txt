# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(TUNGSTEN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/apps/common/include)

option(TUNGSTEN_REUSE_ARRAYS "Reuse Tungsten work arrays to reduce memory use" OFF)

function(tungsten_apply_options target_name)
  if(TUNGSTEN_REUSE_ARRAYS)
    target_compile_definitions(${target_name} PUBLIC TUNGSTEN_REUSE_ARRAYS)
  endif()
endfunction()

add_executable(tungsten src/cpu/tungsten.cpp)
target_include_directories(tungsten PRIVATE ${TUNGSTEN_INCLUDE_DIR})
# Link HeFFTe if available; fallback to find_package
if(TARGET Heffte::Heffte)
  target_link_libraries(tungsten PRIVATE OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte)
elseif(TARGET heffte)
  target_link_libraries(tungsten PRIVATE OpenPFC nlohmann_json::nlohmann_json heffte)
else()
  find_package(Heffte REQUIRED)
  target_link_libraries(tungsten PRIVATE OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte)
endif()

if(TUNGSTEN_REUSE_ARRAYS)
  message(STATUS "Tungsten: reusing arrays")
endif()
tungsten_apply_options(tungsten)

install(TARGETS tungsten DESTINATION bin)

# Alias of production `tungsten` (0.2 ETD session).
add_executable(tungsten_etd src/cpu/tungsten_etd.cpp)
target_include_directories(tungsten_etd PRIVATE ${TUNGSTEN_INCLUDE_DIR})
if(TARGET Heffte::Heffte)
  target_link_libraries(tungsten_etd PRIVATE OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte)
elseif(TARGET heffte)
  target_link_libraries(tungsten_etd PRIVATE OpenPFC nlohmann_json::nlohmann_json heffte)
else()
  find_package(Heffte REQUIRED)
  target_link_libraries(tungsten_etd PRIVATE OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte)
endif()
install(TARGETS tungsten_etd DESTINATION bin)

# CUDA spectral version (requires CUDA toolkit and CUDA-enabled HeFFTe)
if(OpenPFC_ENABLE_CUDA_SPECTRAL)
  add_executable(tungsten_cuda src/cuda/tungsten.cpp)
  target_include_directories(tungsten_cuda PRIVATE ${TUNGSTEN_INCLUDE_DIR})

  target_link_libraries(tungsten_cuda PRIVATE
    OpenPFC
    nlohmann_json::nlohmann_json
    $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
    CUDA::cudart
    openpfc_gpu_kernels
  )

  tungsten_apply_options(tungsten_cuda)

  install(TARGETS tungsten_cuda DESTINATION bin)
  message(STATUS "Tungsten CUDA version enabled")

  add_executable(tungsten_etd_cuda src/cuda/tungsten_etd.cpp)
  target_include_directories(tungsten_etd_cuda PRIVATE ${TUNGSTEN_INCLUDE_DIR})
  target_link_libraries(tungsten_etd_cuda PRIVATE
    OpenPFC
    nlohmann_json::nlohmann_json
    $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
    CUDA::cudart
    openpfc_gpu_kernels
  )
  install(TARGETS tungsten_etd_cuda DESTINATION bin)

  # Test program for CUDA with VTK output
  add_executable(test_tungsten_cuda_vtk tests/test_tungsten_cuda_vtk.cpp)
  target_include_directories(test_tungsten_cuda_vtk PRIVATE ${TUNGSTEN_INCLUDE_DIR})
  target_link_libraries(test_tungsten_cuda_vtk PRIVATE
    OpenPFC
    nlohmann_json::nlohmann_json
    $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
    CUDA::cudart
    openpfc_gpu_kernels
  )
  message(STATUS "Tungsten CUDA VTK test program enabled")
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
endif()

# HIP spectral Tungsten (requires ROCm-enabled HeFFTe — same rule as CUDA spectral).
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  add_executable(tungsten_hip src/hip/tungsten.cpp)
  target_include_directories(tungsten_hip PRIVATE ${TUNGSTEN_INCLUDE_DIR})

  target_link_libraries(tungsten_hip PRIVATE
    OpenPFC
    nlohmann_json::nlohmann_json
    $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
    hip::host
    openpfc_hip_kernels
  )

  set_target_properties(tungsten_hip PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
  )

  tungsten_apply_options(tungsten_hip)

  install(TARGETS tungsten_hip DESTINATION bin)
  message(STATUS "Tungsten HIP version enabled")

  add_executable(tungsten_etd_hip src/hip/tungsten_etd.cpp)
  target_include_directories(tungsten_etd_hip PRIVATE ${TUNGSTEN_INCLUDE_DIR})
  target_link_libraries(tungsten_etd_hip PRIVATE
    OpenPFC
    nlohmann_json::nlohmann_json
    $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
    hip::host
    openpfc_hip_kernels
  )
  set_target_properties(tungsten_etd_hip PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
  )
  install(TARGETS tungsten_etd_hip DESTINATION bin)
  message(STATUS "tungsten_etd_hip enabled")

  add_executable(test_tungsten_hip_vtk tests/test_tungsten_hip_vtk.cpp)
  target_include_directories(test_tungsten_hip_vtk PRIVATE ${TUNGSTEN_INCLUDE_DIR})
  target_link_libraries(test_tungsten_hip_vtk PRIVATE
    OpenPFC
    nlohmann_json::nlohmann_json
    $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
    hip::host
    openpfc_hip_kernels
  )
  set_target_properties(test_tungsten_hip_vtk PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
  )
  message(STATUS "Tungsten HIP VTK test program enabled")
elseif(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND NOT OpenPFC_ENABLE_HIP_SPECTRAL)
  message(STATUS
    "Tungsten HIP spectral targets disabled: ROCm HeFFTe (rocFFT) missing (HIP spectral support OFF).")
endif()

option(TUNGSTEN_ENABLE_TESTS "Enable building Tungsten tests" ON)

if(TUNGSTEN_ENABLE_TESTS)
    find_package(Catch2 REQUIRED)
    if(TARGET Catch2::Catch2)
        add_executable(test_tungsten tests/test_tungsten.cpp
                                    tests/test_tungsten_physics.cpp)
        target_include_directories(test_tungsten PRIVATE ${TUNGSTEN_INCLUDE_DIR})
        if(TARGET Heffte::Heffte)
          target_link_libraries(test_tungsten PRIVATE OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte Catch2::Catch2)
        elseif(TARGET heffte)
          target_link_libraries(test_tungsten PRIVATE OpenPFC nlohmann_json::nlohmann_json heffte Catch2::Catch2)
        else()
          find_package(Heffte REQUIRED)
          target_link_libraries(test_tungsten PRIVATE OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte Catch2::Catch2)
        endif()
        include(CTest)
        # Run all tungsten tests in a single invocation to avoid per-test MPI overhead
        add_test(NAME tungsten-all-tests COMMAND test_tungsten)
        add_test(NAME tungsten-cpu-golden COMMAND test_tungsten "[cpu_golden]")
        add_test(NAME tungsten-etd-cpu-golden COMMAND test_tungsten "[etd_cpu_golden]")
        if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
          add_test(NAME tungsten-golden-4rank
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    $<TARGET_FILE:test_tungsten> "[golden][MPI]")
          set_tests_properties(tungsten-golden-4rank PROPERTIES
            TIMEOUT 180
            RUN_SERIAL TRUE)
        endif()

        # CPU vs CUDA comparison test (only if CUDA spectral support is enabled)
        if(OpenPFC_ENABLE_CUDA_SPECTRAL)
            add_executable(test_tungsten_cpu_vs_cuda tests/test_tungsten_cpu_vs_cuda.cpp)
            target_include_directories(test_tungsten_cpu_vs_cuda PRIVATE ${TUNGSTEN_INCLUDE_DIR})
            target_link_libraries(test_tungsten_cpu_vs_cuda PRIVATE
              OpenPFC
              nlohmann_json::nlohmann_json
              $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
              Catch2::Catch2
              CUDA::cudart
              openpfc_gpu_kernels
            )
            target_sources(test_tungsten_cpu_vs_cuda PRIVATE
                src/cuda/tungsten_ops_kernels.cu
            )
            set_target_properties(test_tungsten_cpu_vs_cuda PROPERTIES
                CUDA_STANDARD 20
                CUDA_STANDARD_REQUIRED ON
                CUDA_SEPARABLE_COMPILATION ON
            )
            include(CTest)
            # Run all CPU vs CUDA tests in a single invocation
            add_test(NAME tungsten-cpu-vs-cuda-tests COMMAND test_tungsten_cpu_vs_cuda)
            message(STATUS "Tungsten CPU vs CUDA comparison test enabled")

            add_executable(test_tungsten_etd_cuda tests/test_tungsten_etd_gpu.cpp)
            target_include_directories(test_tungsten_etd_cuda PRIVATE
              ${TUNGSTEN_INCLUDE_DIR}
              ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
            target_link_libraries(test_tungsten_etd_cuda PRIVATE
              OpenPFC
              nlohmann_json::nlohmann_json
              $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
              Catch2::Catch2
              CUDA::cudart
              openpfc_gpu_kernels
              MPI::MPI_CXX
            )
            target_compile_definitions(test_tungsten_etd_cuda PRIVATE
              OPENPFC_TEST_TUNGSTEN_ETD_CUDA)
            add_test(NAME CUDA_TungstenETD COMMAND test_tungsten_etd_cuda)
            add_test(NAME tungsten-etd-cpu-vs-cuda COMMAND test_tungsten_etd_cuda
                "[parity32]")
        endif()

        # CPU vs HIP comparison test (HIP spectral + rocFFT HeFFTe)
        if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
            add_executable(test_tungsten_cpu_vs_hip tests/test_tungsten_cpu_vs_hip.cpp)
            target_include_directories(test_tungsten_cpu_vs_hip PRIVATE ${TUNGSTEN_INCLUDE_DIR})
            target_link_libraries(test_tungsten_cpu_vs_hip PRIVATE
              OpenPFC
              nlohmann_json::nlohmann_json
              $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
              Catch2::Catch2
              hip::host
              openpfc_hip_kernels
            )
            target_sources(test_tungsten_cpu_vs_hip PRIVATE
                src/hip/tungsten_ops_kernels.hip
            )
            set_target_properties(test_tungsten_cpu_vs_hip PROPERTIES
                HIP_STANDARD 20
                HIP_STANDARD_REQUIRED ON
            )
            include(CTest)
            add_test(NAME tungsten-cpu-vs-hip-tests COMMAND test_tungsten_cpu_vs_hip)
            message(STATUS "Tungsten CPU vs HIP comparison test enabled")

            add_executable(test_tungsten_etd_hip tests/test_tungsten_etd_gpu.cpp)
            target_include_directories(test_tungsten_etd_hip PRIVATE
              ${TUNGSTEN_INCLUDE_DIR}
              ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
            target_link_libraries(test_tungsten_etd_hip PRIVATE
              OpenPFC
              nlohmann_json::nlohmann_json
              $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
              Catch2::Catch2
              hip::host
              openpfc_hip_kernels
              MPI::MPI_CXX
            )
            target_compile_definitions(test_tungsten_etd_hip PRIVATE
              OPENPFC_TEST_TUNGSTEN_ETD_HIP)
            set_target_properties(test_tungsten_etd_hip PROPERTIES
              HIP_STANDARD 20
              HIP_STANDARD_REQUIRED ON
            )
            add_test(NAME HIP_TungstenETD COMMAND test_tungsten_etd_hip)
        endif()
    else()
        message(WARNING "Tungsten: Catch2 not found, not building tests")
    endif()
endif()

# Scalability study application
add_executable(tungsten_scalability src/tungsten_scalability.cpp)
target_include_directories(tungsten_scalability PRIVATE ${TUNGSTEN_INCLUDE_DIR})
target_link_libraries(tungsten_scalability PRIVATE
  OpenPFC
  nlohmann_json::nlohmann_json
  $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
  MPI::MPI_CXX
)

if(OpenPFC_ENABLE_CUDA_SPECTRAL)
    target_link_libraries(tungsten_scalability PRIVATE CUDA::cudart openpfc_gpu_kernels)
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
    target_link_libraries(tungsten_scalability PRIVATE hip::host openpfc_hip_kernels)
    set_target_properties(tungsten_scalability PROPERTIES
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
    )
    target_compile_definitions(tungsten_scalability PRIVATE OpenPFC_ENABLE_HIP_SPECTRAL)
endif()

install(TARGETS tungsten_scalability DESTINATION bin)
