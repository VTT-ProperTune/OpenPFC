# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Library target creation and configuration

option(BUILD_SHARED_LIBS "Build OpenPFC as a shared library" OFF)

# Profiling: OpenPFC_PROFILING_LEVEL==0 strips OPENPFC_PROFILE / PFC_PROFILE_SCOPE to no-ops
# in profile_scope_macro.hpp; >0 enables those macros. Runtime ProfilingSession is unchanged.
set(OpenPFC_PROFILING_LEVEL "2" CACHE STRING
    "Compile-time profiling level (0=off, 1=wall/MPI stats, 2=+scoped regions)")
set_property(CACHE OpenPFC_PROFILING_LEVEL PROPERTY STRINGS "0" "1" "2")
if(NOT OpenPFC_PROFILING_LEVEL MATCHES "^[012]$")
  message(FATAL_ERROR "OpenPFC_PROFILING_LEVEL must be 0, 1, or 2 (got '${OpenPFC_PROFILING_LEVEL}')")
endif()

# Create library
add_library(openpfc
    src/openpfc/kernel/data/world.cpp
    src/openpfc/kernel/data/box3d.cpp
    src/openpfc/kernel/decomposition/decomposition.cpp
    src/openpfc/kernel/decomposition/decomposition_factory.cpp
    src/openpfc/kernel/profiling/session.cpp
    src/openpfc/kernel/profiling/timer_report.cpp
    src/openpfc/runtime/cpu/fft.cpp
    src/openpfc/frontend/utils/logging.cpp
    src/openpfc/frontend/ui/ui_errors.cpp
    src/openpfc/frontend/io/vtk_writer.cpp
    $<$<BOOL:${OpenPFC_ENABLE_CUDA}>:src/openpfc/runtime/cuda/fft_cuda.cpp>
    $<$<BOOL:${OpenPFC_ENABLE_HIP}>:src/openpfc/runtime/hip/fft_hip.cpp>
)

add_library(OpenPFC ALIAS openpfc)

# Layering: openpfc is the compiled implementation; public include path is on the
# target above. HeFFTe stays PRIVATE (see block below) so only TUs that include
# fft_fftw.hpp need to link HeFFTe.

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
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
)

# Options (MPI option is declared in ProjectSetup.cmake before Dependencies.cmake)
option(OpenPFC_ENABLE_HEFFTE "Enable HeFFTe FFT support" ON)

if(OpenPFC_ENABLE_MPI)
  target_link_libraries(openpfc PUBLIC MPI::MPI_CXX)
endif()

target_link_libraries(openpfc PRIVATE nlohmann_json::nlohmann_json)

if(OpenPFC_ENABLE_HDF5)
  if(TARGET HDF5::HDF5)
    target_link_libraries(openpfc PRIVATE HDF5::HDF5)
  elseif(DEFINED HDF5_LIBRARIES)
    if(DEFINED HDF5_INCLUDE_DIRS)
      target_include_directories(openpfc PRIVATE ${HDF5_INCLUDE_DIRS})
    elseif(DEFINED HDF5_INCLUDE_DIR)
      target_include_directories(openpfc PRIVATE ${HDF5_INCLUDE_DIR})
    endif()
    target_link_libraries(openpfc PRIVATE ${HDF5_LIBRARIES})
  else()
    message(FATAL_ERROR "HDF5 enabled but HDF5::HDF5 target and HDF5_LIBRARIES not set")
  endif()
  target_compile_definitions(openpfc PRIVATE OPENPFC_HAS_HDF5=1)
endif()

# Conditionally find HeFFTe
if(OpenPFC_ENABLE_HEFFTE)
  # Prefer already-fetched target; fall back to find_package if needed
  if(TARGET Heffte::Heffte)
    target_link_libraries(openpfc PRIVATE Heffte::Heffte)
  elseif(TARGET Heffte)
    target_link_libraries(openpfc PRIVATE Heffte)
  elseif(TARGET heffte)
    target_link_libraries(openpfc PRIVATE heffte)
  else()
    find_package(Heffte REQUIRED)
    target_link_libraries(openpfc PRIVATE Heffte::Heffte)
  endif()
  # HeFFTe is linked PRIVATE only; public headers that need <heffte.h> live in
  # fft_fftw.hpp (include explicitly or use openpfc.hpp). Downstream TUs must link
  # HeFFTe themselves if they include fft_fftw.hpp.
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

target_compile_definitions(openpfc PUBLIC
    "OPENPFC_PROFILING_LEVEL=${OpenPFC_PROFILING_LEVEL}")
target_compile_definitions(openpfc PRIVATE
    "OPENPFC_PROFILING_BUILD_VERSION=\"${PROJECT_VERSION}\"")

# GCC 8.x: std::filesystem is in libstdc++fs and must be linked explicitly
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
  target_link_libraries(openpfc PUBLIC stdc++fs)
endif()
