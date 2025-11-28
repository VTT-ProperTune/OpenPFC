cmake_minimum_required(VERSION 3.19)

project(HeffteMultiGPUResearch LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find HeFFTe from our custom installation
# The HeffteConfig.cmake is located at ~/opt/heffte/2.4.1-cuda/lib64/cmake/Heffte/
set(Heffte_DIR "$ENV{HOME}/opt/heffte/2.4.1-cuda/lib64/cmake/Heffte")
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
