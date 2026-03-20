# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Library target creation and configuration

option(BUILD_SHARED_LIBS "Build OpenPFC as a shared library" OFF)

# Create library
add_library(openpfc
    src/openpfc/kernel/data/world.cpp
    src/openpfc/kernel/data/box3d.cpp
    src/openpfc/kernel/decomposition/decomposition.cpp
    src/openpfc/kernel/decomposition/decomposition_factory.cpp
    src/openpfc/runtime/cpu/fft.cpp
    src/openpfc/frontend/utils/logging.cpp
    src/openpfc/frontend/ui/ui_errors.cpp
    src/openpfc/frontend/io/vtk_writer.cpp
    $<$<BOOL:${OpenPFC_ENABLE_CUDA}>:src/openpfc/runtime/cuda/fft_cuda.cpp>
    $<$<BOOL:${OpenPFC_ENABLE_HIP}>:src/openpfc/runtime/hip/fft_hip.cpp>
)

add_library(OpenPFC ALIAS openpfc)

# Set the library version
set_target_properties(openpfc PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "openpfc"  # ensures the filename is lowercase
)

# Public API (headers)
target_include_directories(openpfc
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Options
option(OpenPFC_ENABLE_MPI "Enable MPI support" ON)
option(OpenPFC_ENABLE_HEFFTE "Enable HeFFTe FFT support" ON)

# Conditionally find MPI
if(OpenPFC_ENABLE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(openpfc PUBLIC MPI::MPI_CXX)
endif()

# Conditionally find HeFFTe
if(OpenPFC_ENABLE_HEFFTE)
  # Prefer already-fetched target; fall back to find_package if needed
  if(TARGET Heffte::Heffte)
    target_link_libraries(openpfc PRIVATE Heffte::Heffte)
    get_target_property(_heffte_inc Heffte::Heffte INTERFACE_INCLUDE_DIRECTORIES)
  elseif(TARGET heffte)
    target_link_libraries(openpfc PRIVATE heffte)
    get_target_property(_heffte_inc heffte INTERFACE_INCLUDE_DIRECTORIES)
  else()
    find_package(Heffte REQUIRED)
    target_link_libraries(openpfc PRIVATE Heffte::Heffte)
    get_target_property(_heffte_inc Heffte::Heffte INTERFACE_INCLUDE_DIRECTORIES)
  endif()
  if(_heffte_inc)
    # Propagate HeFFTe headers since our public API includes <heffte.h>
    target_include_directories(openpfc PUBLIC $<BUILD_INTERFACE:${_heffte_inc}>)
  endif()
endif()

# GPU kernel library (only when CUDA is enabled)
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    add_library(openpfc_gpu_kernels
        include/openpfc/runtime/cuda/kernels_simple.cu
        include/openpfc/runtime/cuda/sparse_vector_ops.cu
    )
    
    target_include_directories(openpfc_gpu_kernels
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    
    target_link_libraries(openpfc_gpu_kernels
        PUBLIC
        CUDA::cudart
    )
    
    set_target_properties(openpfc_gpu_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Link GPU kernels to main library (private - implementation detail)
    # Users can still use the kernels via headers, but don't need to link the library
    target_link_libraries(openpfc PRIVATE openpfc_gpu_kernels)
    
    message(STATUS "✅ GPU kernel library enabled")
endif()

# Add tomlplusplus include directory (header-only library)
# Since it's header-only (often from FetchContent), we just need the include path
# Use $<BUILD_INTERFACE:...> to avoid export issues with build directory paths
if(DEFINED TOMLPLUSPLUS_SOURCE_DIR)
  target_include_directories(openpfc PUBLIC $<BUILD_INTERFACE:${TOMLPLUSPLUS_SOURCE_DIR}/include>)
elseif(DEFINED tomlplusplus_SOURCE_DIR)
  target_include_directories(openpfc PUBLIC $<BUILD_INTERFACE:${tomlplusplus_SOURCE_DIR}/include>)
elseif(TARGET tomlplusplus::tomlplusplus)
  get_target_property(TOML_INCLUDE_DIR tomlplusplus::tomlplusplus INTERFACE_INCLUDE_DIRECTORIES)
  if(TOML_INCLUDE_DIR)
    target_include_directories(openpfc PUBLIC $<BUILD_INTERFACE:${TOML_INCLUDE_DIR}>)
  endif()
elseif(TARGET tomlplusplus_tomlplusplus)
  get_target_property(TOML_INCLUDE_DIR tomlplusplus_tomlplusplus INTERFACE_INCLUDE_DIRECTORIES)
  if(TOML_INCLUDE_DIR)
    target_include_directories(openpfc PUBLIC $<BUILD_INTERFACE:${TOML_INCLUDE_DIR}>)
  endif()
endif()

# Require C++17
target_compile_features(openpfc PUBLIC cxx_std_17)

# GCC 8.x: std::filesystem is in libstdc++fs and must be linked explicitly
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
  target_link_libraries(openpfc PUBLIC stdc++fs)
endif()
