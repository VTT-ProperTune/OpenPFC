# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(TUNGSTEN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

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

option(TUNGSTEN_REUSE_ARRAYS "Reuse some arrays" OFF)

if(TUNGSTEN_REUSE_ARRAYS)
  message(STATUS "Tungsten: reusing arrays")
  target_compile_definitions(tungsten PUBLIC MAHTI_HACK)
endif()

install(TARGETS tungsten DESTINATION bin)

# CUDA version (only if CUDA is enabled)
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  add_executable(tungsten_cuda
    src/cuda/tungsten.cpp
    src/cuda/tungsten_ops_kernels.cu
  )
  target_include_directories(tungsten_cuda PRIVATE ${TUNGSTEN_INCLUDE_DIR})

  target_link_libraries(tungsten_cuda PRIVATE
    OpenPFC
    nlohmann_json::nlohmann_json
    $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
    CUDA::cudart
  )

  # Set CUDA properties
  set_target_properties(tungsten_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )

  if(TUNGSTEN_REUSE_ARRAYS)
    target_compile_definitions(tungsten_cuda PUBLIC MAHTI_HACK)
  endif()

  install(TARGETS tungsten_cuda DESTINATION bin)
  message(STATUS "Tungsten CUDA version enabled")

  # Test program for CUDA with VTK output
  add_executable(test_tungsten_cuda_vtk
    tests/test_tungsten_cuda_vtk.cpp
    src/cuda/tungsten_ops_kernels.cu
  )
  target_include_directories(test_tungsten_cuda_vtk PRIVATE ${TUNGSTEN_INCLUDE_DIR})
  target_link_libraries(test_tungsten_cuda_vtk PRIVATE
    OpenPFC
    nlohmann_json::nlohmann_json
    $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
    CUDA::cudart
  )
  set_target_properties(test_tungsten_cuda_vtk PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  message(STATUS "Tungsten CUDA VTK test program enabled")
endif()

# HIP version (only if HIP is enabled)
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(tungsten_hip
    src/hip/tungsten.cpp
    src/hip/tungsten_ops_kernels.hip
  )
  target_include_directories(tungsten_hip PRIVATE ${TUNGSTEN_INCLUDE_DIR})

  target_link_libraries(tungsten_hip PRIVATE
    OpenPFC
    nlohmann_json::nlohmann_json
    $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
    hip::host
  )

  set_target_properties(tungsten_hip PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
  )

  if(TUNGSTEN_REUSE_ARRAYS)
    target_compile_definitions(tungsten_hip PUBLIC MAHTI_HACK)
  endif()

  install(TARGETS tungsten_hip DESTINATION bin)
  message(STATUS "Tungsten HIP version enabled")

  # Standalone GPU-aware MPI probe (device pointers; needs MPICH_GPU_SUPPORT_ENABLED=1 on LUMI)
  add_executable(verify_gpu_aware_mpi src/verify_gpu_aware_mpi.cpp)
  target_link_libraries(verify_gpu_aware_mpi PRIVATE MPI::MPI_CXX hip::host)
  set_target_properties(verify_gpu_aware_mpi PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
  )
  install(TARGETS verify_gpu_aware_mpi DESTINATION bin)
  message(STATUS "verify_gpu_aware_mpi enabled (HIP + MPI device buffer test)")

  # Test program for HIP with VTK output
  add_executable(test_tungsten_hip_vtk
    tests/test_tungsten_hip_vtk.cpp
    src/hip/tungsten_ops_kernels.hip
  )
  target_include_directories(test_tungsten_hip_vtk PRIVATE ${TUNGSTEN_INCLUDE_DIR})
  target_link_libraries(test_tungsten_hip_vtk PRIVATE
    OpenPFC
    nlohmann_json::nlohmann_json
    $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
    hip::host
  )
  set_target_properties(test_tungsten_hip_vtk PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
  )
  message(STATUS "Tungsten HIP VTK test program enabled")
endif()

option(TUNGSTEN_ENABLE_TESTS "Enable building Tungsten tests" ON)

if(TUNGSTEN_ENABLE_TESTS)
    find_package(Catch2 REQUIRED)
    if(TARGET Catch2::Catch2)
        add_executable(test_tungsten tests/test_tungsten.cpp)
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

        # CPU vs CUDA comparison test (only if CUDA is enabled)
        if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
            add_executable(test_tungsten_cpu_vs_cuda tests/test_tungsten_cpu_vs_cuda.cpp)
            target_include_directories(test_tungsten_cpu_vs_cuda PRIVATE ${TUNGSTEN_INCLUDE_DIR})
            target_link_libraries(test_tungsten_cpu_vs_cuda PRIVATE
              OpenPFC
              nlohmann_json::nlohmann_json
              $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
              Catch2::Catch2
              CUDA::cudart
            )
            target_sources(test_tungsten_cpu_vs_cuda PRIVATE
                src/cuda/tungsten_ops_kernels.cu
            )
            set_target_properties(test_tungsten_cpu_vs_cuda PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
                CUDA_SEPARABLE_COMPILATION ON
            )
            include(CTest)
            # Run all CPU vs CUDA tests in a single invocation
            add_test(NAME tungsten-cpu-vs-cuda-tests COMMAND test_tungsten_cpu_vs_cuda)
            message(STATUS "Tungsten CPU vs CUDA comparison test enabled")
        endif()

        # CPU vs HIP comparison test (only if HIP is enabled)
        if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
            add_executable(test_tungsten_cpu_vs_hip tests/test_tungsten_cpu_vs_hip.cpp)
            target_include_directories(test_tungsten_cpu_vs_hip PRIVATE ${TUNGSTEN_INCLUDE_DIR})
            target_link_libraries(test_tungsten_cpu_vs_hip PRIVATE
              OpenPFC
              nlohmann_json::nlohmann_json
              $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
              Catch2::Catch2
              hip::host
            )
            target_sources(test_tungsten_cpu_vs_hip PRIVATE
                src/hip/tungsten_ops_kernels.hip
            )
            set_target_properties(test_tungsten_cpu_vs_hip PROPERTIES
                HIP_STANDARD 17
                HIP_STANDARD_REQUIRED ON
            )
            include(CTest)
            add_test(NAME tungsten-cpu-vs-hip-tests COMMAND test_tungsten_cpu_vs_hip)
            message(STATUS "Tungsten CPU vs HIP comparison test enabled")
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

# Add CUDA support to scalability app if available
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    target_link_libraries(tungsten_scalability PRIVATE CUDA::cudart)
    target_sources(tungsten_scalability PRIVATE
        src/cuda/tungsten_ops_kernels.cu
    )
    set_target_properties(tungsten_scalability PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Add HIP support to scalability app if available
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
    target_link_libraries(tungsten_scalability PRIVATE hip::host)
    target_sources(tungsten_scalability PRIVATE
        src/hip/tungsten_ops_kernels.hip
    )
    set_target_properties(tungsten_scalability PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
    )
endif()

install(TARGETS tungsten_scalability DESTINATION bin)
