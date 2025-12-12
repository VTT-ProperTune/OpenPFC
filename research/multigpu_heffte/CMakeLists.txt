# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.19)

project(HeffteMultiGPUResearch LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find HeFFTe from our custom installation
# Option to use version without GPU-aware MPI for testing
# Set environment variable: export HEFFTE_NO_GPU_AWARE=1
if(DEFINED ENV{HEFFTE_NO_GPU_AWARE})
    if($ENV{HEFFTE_NO_GPU_AWARE} EQUAL 1)
        set(Heffte_DIR "$ENV{HOME}/opt/heffte/2.4.1-cuda-no-gpuaware/lib64/cmake/Heffte")
        message(STATUS "Using HeFFTe WITHOUT GPU-aware MPI (for testing)")
    else()
        set(Heffte_DIR "$ENV{HOME}/opt/heffte/2.4.1-cuda/lib64/cmake/Heffte")
        message(STATUS "Using HeFFTe WITH GPU-aware MPI (default)")
    endif()
else()
    set(Heffte_DIR "$ENV{HOME}/opt/heffte/2.4.1-cuda/lib64/cmake/Heffte")
    message(STATUS "Using HeFFTe WITH GPU-aware MPI (default)")
endif()
find_package(Heffte 2.4.1 REQUIRED PATHS ${Heffte_DIR})

# Print information about found HeFFTe
message(STATUS "Found HeFFTe version: ${Heffte_VERSION}")
message(STATUS "HeFFTe include directory: ${Heffte_INCLUDE_DIRS}")
message(STATUS "HeFFTe library: ${Heffte_LIBRARIES}")

# Check if CUDA backend is available
if(Heffte_CUDA_FOUND)
    message(STATUS "HeFFTe CUDA backend is enabled")
    enable_language(CUDA)
else()
    message(FATAL_ERROR "HeFFTe CUDA backend is not available. Please rebuild HeFFTe with -DHeffte_ENABLE_CUDA=ON")
endif()

# Add executable for CUDA FFT example
# Mark the source file as CUDA language so kernels are compiled correctly
add_executable(cuda_fft_example cuda_fft_example.cpp)
set_source_files_properties(cuda_fft_example.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(cuda_fft_example 
    PRIVATE 
    Heffte::Heffte
    MPI::MPI_CXX
    CUDA::cudart
)

# Enable CUDA language support for the target
set_target_properties(cuda_fft_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA architecture if not already set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "OFF" CACHE STRING "CUDA architectures to compile")
endif()

# Add FFT power benchmark executable
add_executable(fft_power_benchmark fft_power_benchmark.cpp)
set_source_files_properties(fft_power_benchmark.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(fft_power_benchmark 
    PRIVATE 
    Heffte::Heffte
    MPI::MPI_CXX
    CUDA::cudart
)

# Enable CUDA language support for the target
set_target_properties(fft_power_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Try to find NVML for power monitoring (optional)
find_library(NVML_LIBRARY nvidia-ml PATHS /usr/lib64 /usr/lib)
if(NVML_LIBRARY)
    target_link_libraries(fft_power_benchmark PRIVATE ${NVML_LIBRARY})
    target_compile_definitions(fft_power_benchmark PRIVATE NVML_FOUND)
    message(STATUS "NVML found: ${NVML_LIBRARY}")
else()
    message(STATUS "NVML not found - power monitoring will use CUDA APIs")
endif()

# Add direct cuFFT benchmark executable (no MPI, no HeFFTe)
add_executable(cufft_direct_benchmark cufft_direct_benchmark.cpp)
set_source_files_properties(cufft_direct_benchmark.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(cufft_direct_benchmark 
    PRIVATE 
    CUDA::cudart
    CUDA::cufft
)

# Enable CUDA language support for the target
set_target_properties(cufft_direct_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add NVML support if available
if(NVML_LIBRARY)
    target_link_libraries(cufft_direct_benchmark PRIVATE ${NVML_LIBRARY})
    target_compile_definitions(cufft_direct_benchmark PRIVATE NVML_FOUND)
endif()

# Add high-power cuFFT benchmark (batch processing)
add_executable(cufft_high_power_benchmark cufft_high_power_benchmark.cpp)
set_source_files_properties(cufft_high_power_benchmark.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(cufft_high_power_benchmark 
    PRIVATE 
    CUDA::cudart
    CUDA::cufft
)

set_target_properties(cufft_high_power_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add CPU profiling tool
add_executable(profile_cpu_usage profile_cpu_usage.cpp)
set_source_files_properties(profile_cpu_usage.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(profile_cpu_usage 
    PRIVATE 
    Heffte::Heffte
    MPI::MPI_CXX
    CUDA::cudart
)

set_target_properties(profile_cpu_usage PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
