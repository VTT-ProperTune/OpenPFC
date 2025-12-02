# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(tungsten tungsten.cpp)
find_package(Heffte REQUIRED)
target_link_libraries(tungsten PRIVATE OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte)

option(TUNGSTEN_REUSE_ARRAYS "Reuse some arrays" OFF)

if(TUNGSTEN_REUSE_ARRAYS)
  message(STATUS "Tungsten: reusing arrays")
  target_compile_definitions(tungsten PUBLIC MAHTI_HACK)
endif()

install(TARGETS tungsten DESTINATION bin)

# CUDA version (only if CUDA is enabled)
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  add_executable(tungsten_cuda 
    tungsten_cuda.cpp
    tungsten_ops_kernels.cu
  )
  
  target_link_libraries(tungsten_cuda PRIVATE 
    OpenPFC 
    nlohmann_json::nlohmann_json 
    Heffte::Heffte 
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
endif()

option(TUNGSTEN_ENABLE_TESTS "Enable building Tungsten tests" ON)

if(TUNGSTEN_ENABLE_TESTS)
    find_package(Catch2 REQUIRED)
    if(TARGET Catch2::Catch2)
        add_executable(test_tungsten test_tungsten.cpp)
        target_link_libraries(test_tungsten PRIVATE OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte Catch2::Catch2)
        include(CTest)
        include(Catch)
        catch_discover_tests(test_tungsten)
        
        # CPU vs CUDA comparison test (only if CUDA is enabled)
        if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
            add_executable(test_tungsten_cpu_vs_cuda test_tungsten_cpu_vs_cuda.cpp)
            target_link_libraries(test_tungsten_cpu_vs_cuda PRIVATE 
                OpenPFC 
                nlohmann_json::nlohmann_json 
                Heffte::Heffte 
                Catch2::Catch2
                CUDA::cudart
            )
            target_sources(test_tungsten_cpu_vs_cuda PRIVATE
                tungsten_ops_kernels.cu
            )
            set_target_properties(test_tungsten_cpu_vs_cuda PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
                CUDA_SEPARABLE_COMPILATION ON
            )
            include(CTest)
            include(Catch)
            catch_discover_tests(test_tungsten_cpu_vs_cuda)
            message(STATUS "Tungsten CPU vs CUDA comparison test enabled")
        endif()
    else()
        message(WARNING "Tungsten: Catch2 not found, not building tests")
    endif()
endif()

# Scalability study application
add_executable(tungsten_scalability tungsten_scalability.cpp)
target_link_libraries(tungsten_scalability PRIVATE 
    OpenPFC 
    nlohmann_json::nlohmann_json 
    Heffte::Heffte
    MPI::MPI_CXX
)

# Add CUDA support to scalability app if available
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    target_link_libraries(tungsten_scalability PRIVATE CUDA::cudart)
    target_sources(tungsten_scalability PRIVATE
        tungsten_ops_kernels.cu
    )
    set_target_properties(tungsten_scalability PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

install(TARGETS tungsten_scalability DESTINATION bin)
