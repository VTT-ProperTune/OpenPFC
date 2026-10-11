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

# Split compiled sources into object libraries (kernel/runtime vs frontend) for
# clearer layering and faster incremental rebuilds; link as one `openpfc` for
# install/export (see docs/archive/refactoring_roadmap.md Phase D).
set(OpenPFC_HAS_FFTW3 OFF)
find_library(OpenPFC_FFTW3_LIBRARY NAMES fftw3)
if(OpenPFC_FFTW3_LIBRARY)
  set(OpenPFC_HAS_FFTW3 ON)
endif()

set(_openpfc_kernel_obj_sources
    src/openpfc/kernel/decomposition/decomposition.cpp
    src/openpfc/kernel/decomposition/decomposition_factory.cpp
    src/openpfc/kernel/profiling/session.cpp
    src/openpfc/kernel/profiling/detail/session_merge_json.cpp
    $<$<BOOL:${OpenPFC_ENABLE_HDF5}>:src/openpfc/kernel/profiling/detail/session_profiling_hdf5.cpp>
    src/openpfc/kernel/profiling/timer_report.cpp
    src/openpfc/kernel/utils/logging.cpp
)
if(OpenPFC_HAS_FFTW3)
  list(APPEND _openpfc_kernel_obj_sources src/openpfc/kernel/fft/chebyshev.cpp
       src/openpfc/kernel/fft/real_transform.cpp)
endif()
if(OpenPFC_ENABLE_HEFFTE)
  list(APPEND _openpfc_kernel_obj_sources src/openpfc/runtime/cpu/fft.cpp)
  if(OpenPFC_HAS_FFTW3)
    list(APPEND _openpfc_kernel_obj_sources
         src/openpfc/runtime/cpu/mixed_transform.cpp)
  endif()
endif()
if(OpenPFC_ENABLE_CUDA_SPECTRAL OR OpenPFC_ENABLE_HIP_SPECTRAL)
  list(APPEND _openpfc_kernel_obj_sources src/openpfc/runtime/gpu/fft_gpu.cpp)
endif()

add_library(openpfc_kernel_obj OBJECT ${_openpfc_kernel_obj_sources})

set(_openpfc_frontend_obj_sources
    src/openpfc/frontend/ui/app_profiling.cpp
    src/openpfc/frontend/ui/ui_errors.cpp
    src/openpfc/frontend/io/vtk_writer.cpp
    src/openpfc/frontend/io/vtk_writer_validate.cpp
    src/openpfc/frontend/io/png_writer.cpp
    $<$<BOOL:${OpenPFC_ENABLE_HDF5}>:src/openpfc/frontend/io/hdf5_writer.cpp>
)
add_library(openpfc_frontend_obj OBJECT ${_openpfc_frontend_obj_sources})

# Single static/shared library target for installation and find_package
add_library(openpfc
    $<TARGET_OBJECTS:openpfc_kernel_obj>
    $<TARGET_OBJECTS:openpfc_frontend_obj>
)

add_library(OpenPFC ALIAS openpfc)
if(OpenPFC_HAS_FFTW3)
  # Private: the static archive records the FFTW dependency for its
  # consumers without exporting the absolute library path.
  target_link_libraries(openpfc PRIVATE ${OpenPFC_FFTW3_LIBRARY})
endif()

# Object libraries need the same include paths / standards as the merged `openpfc`
# target so their translation units compile identically to the old monolithic lib.
target_include_directories(openpfc_kernel_obj
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
)
target_include_directories(openpfc_frontend_obj
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/stb>
)
target_compile_features(openpfc_kernel_obj PUBLIC cxx_std_20)
target_compile_features(openpfc_frontend_obj PUBLIC cxx_std_20)
target_compile_definitions(openpfc_kernel_obj PUBLIC
    "OPENPFC_PROFILING_LEVEL=${OpenPFC_PROFILING_LEVEL}")
target_compile_definitions(openpfc_kernel_obj PRIVATE
    "OPENPFC_PROFILING_BUILD_VERSION=\"${PROJECT_VERSION}\"")
target_compile_definitions(openpfc_frontend_obj PUBLIC
    "OPENPFC_PROFILING_LEVEL=${OpenPFC_PROFILING_LEVEL}")
target_compile_definitions(openpfc_frontend_obj PRIVATE
    "OPENPFC_PROFILING_BUILD_VERSION=\"${PROJECT_VERSION}\"")

# GPU backend-enable flags as PUBLIC usage requirements (audit 11 / PM / M3).
# Directory-scope add_compile_definitions is not part of the installed
# INTERFACE, so OpenPFC::openpfc consumers must get the same macros as in-tree
# TUs. Apply them on the exported library, the object libs that compile it,
# and the vendor kernel libs (PUBLIC so tests that link only the kernel lib
# still see OpenPFC_ENABLE_*).
function(openpfc_gpu_public_definitions tgt)
  if(NOT TARGET ${tgt})
    return()
  endif()
  get_target_property(_openpfc_gpu_defs_type ${tgt} TYPE)
  if(_openpfc_gpu_defs_type STREQUAL "INTERFACE_LIBRARY")
    set(_openpfc_gpu_defs_scope INTERFACE)
  else()
    set(_openpfc_gpu_defs_scope PUBLIC)
  endif()
  if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    target_compile_definitions(${tgt} ${_openpfc_gpu_defs_scope} OpenPFC_ENABLE_CUDA)
    if(OpenPFC_MPI_CUDA_AWARE)
      target_compile_definitions(${tgt} ${_openpfc_gpu_defs_scope} OpenPFC_MPI_CUDA_AWARE)
    endif()
  endif()
  if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
    target_compile_definitions(${tgt} ${_openpfc_gpu_defs_scope} OpenPFC_ENABLE_HIP)
    if(OpenPFC_MPI_HIP_AWARE)
      target_compile_definitions(${tgt} ${_openpfc_gpu_defs_scope} OpenPFC_MPI_HIP_AWARE)
    endif()
  endif()
  if(OpenPFC_ENABLE_GPU_AUTOTUNING)
    target_compile_definitions(${tgt} ${_openpfc_gpu_defs_scope} OpenPFC_ENABLE_GPU_AUTOTUNING)
  endif()
endfunction()

openpfc_gpu_public_definitions(openpfc)
openpfc_gpu_public_definitions(openpfc_kernel_obj)
openpfc_gpu_public_definitions(openpfc_frontend_obj)

# Shared device-kernel TUs under src/openpfc/runtime/gpu/. CUDA and HIP compile
# the same basenames with vendor extensions. Both vendors add padded_halo_faces
# as an extra source (CUDA via CUDA_RESOLVE_DEVICE_SYMBOLS on the archive).
# openpfc_add_gpu_kernel_library(<tgt> <ext> <runtime_lib> [extra sources...])
function(openpfc_add_gpu_kernel_library tgt ext runtime_lib)
  add_library(${tgt}
      src/openpfc/runtime/gpu/sparse_vector_ops.${ext}
      src/openpfc/runtime/gpu/fill.${ext}
      src/openpfc/runtime/gpu/elementwise_ops.${ext}
      ${ARGN}
  )
  target_include_directories(${tgt}
      PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${tgt} PUBLIC ${runtime_lib})
  target_link_libraries(openpfc PRIVATE ${tgt})
  openpfc_gpu_public_definitions(${tgt})
  if(DEFINED nlohmann_json_SOURCE_DIR)
    target_include_directories(${tgt} PRIVATE
        $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/single_include>)
  endif()
endfunction()

# In-tree GPU unit tests that do not link openpfc (device detection, autotune).
add_library(openpfc_gpu_compile_defs INTERFACE)
openpfc_gpu_public_definitions(openpfc_gpu_compile_defs)

if(OpenPFC_NAN_CHECK_ACTIVE)
  target_compile_definitions(openpfc PUBLIC NAN_CHECK_ENABLED)
  target_compile_definitions(openpfc_kernel_obj PUBLIC NAN_CHECK_ENABLED)
  target_compile_definitions(openpfc_frontend_obj PUBLIC NAN_CHECK_ENABLED)
endif()

# Layering: openpfc is the compiled implementation; public include path is on the
# target above. When HeFFTe is enabled, openpfc links it PUBLIC so an installed
# consumer that includes a spectral header does not name Heffte itself.

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
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/stb>
)

# Options (MPI / HeFFTe options are declared in ProjectSetup.cmake).

if(OpenPFC_ENABLE_MPI)
  target_link_libraries(openpfc PUBLIC MPI::MPI_CXX)
  target_link_libraries(openpfc_kernel_obj PUBLIC MPI::MPI_CXX)
  target_link_libraries(openpfc_frontend_obj PUBLIC MPI::MPI_CXX)
endif()

# Header-only FetchContent json must not appear in OpenPFCTargets (CMake 3.28
# export then requires a nlohmann_json target that is not in the export set).
# Installed consumers get nlohmann_json via find_dependency in OpenPFCConfig.
# In-tree: PUBLIC BUILD_INTERFACE so targets that include public headers
# (checkpoint_service.hpp, frontend JSON) get the FetchContent include path.
if(DEFINED nlohmann_json_SOURCE_DIR)
  target_include_directories(openpfc PUBLIC
      $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/single_include>)
  target_include_directories(openpfc_kernel_obj PRIVATE
      $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/single_include>)
  target_include_directories(openpfc_frontend_obj PRIVATE
      $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/single_include>)
else()
  target_link_libraries(openpfc PRIVATE nlohmann_json::nlohmann_json)
  target_link_libraries(openpfc_kernel_obj PRIVATE nlohmann_json::nlohmann_json)
  target_link_libraries(openpfc_frontend_obj PRIVATE nlohmann_json::nlohmann_json)
endif()

if(OpenPFC_ENABLE_HDF5)
  if(TARGET HDF5::HDF5)
    target_link_libraries(openpfc PUBLIC HDF5::HDF5)
    target_link_libraries(openpfc_kernel_obj PRIVATE HDF5::HDF5)
    target_link_libraries(openpfc_frontend_obj PRIVATE HDF5::HDF5)
  elseif(DEFINED HDF5_LIBRARIES)
    if(DEFINED HDF5_INCLUDE_DIRS)
      target_include_directories(openpfc PUBLIC ${HDF5_INCLUDE_DIRS})
      target_include_directories(openpfc_kernel_obj PRIVATE ${HDF5_INCLUDE_DIRS})
      target_include_directories(openpfc_frontend_obj PRIVATE ${HDF5_INCLUDE_DIRS})
    elseif(DEFINED HDF5_INCLUDE_DIR)
      target_include_directories(openpfc PUBLIC ${HDF5_INCLUDE_DIR})
      target_include_directories(openpfc_kernel_obj PRIVATE ${HDF5_INCLUDE_DIR})
      target_include_directories(openpfc_frontend_obj PRIVATE ${HDF5_INCLUDE_DIR})
    endif()
    target_link_libraries(openpfc PUBLIC ${HDF5_LIBRARIES})
    target_link_libraries(openpfc_kernel_obj PRIVATE ${HDF5_LIBRARIES})
    target_link_libraries(openpfc_frontend_obj PRIVATE ${HDF5_LIBRARIES})
  else()
    message(FATAL_ERROR "HDF5 enabled but HDF5::HDF5 target and HDF5_LIBRARIES not set")
  endif()
  target_compile_definitions(openpfc PUBLIC OPENPFC_HAS_HDF5=1)
  target_compile_definitions(openpfc_kernel_obj PRIVATE OPENPFC_HAS_HDF5=1)
  target_compile_definitions(openpfc_frontend_obj PRIVATE OPENPFC_HAS_HDF5=1)
endif()

# HeFFTe is optional. Spectral FFT sources (cpu/fft.cpp, gpu/fft_gpu.cpp)
# are added only when ON. Decomposition no longer depends on it.
if(OpenPFC_ENABLE_HEFFTE)
  target_compile_definitions(openpfc PUBLIC OpenPFC_ENABLE_HEFFTE)
  target_compile_definitions(openpfc_kernel_obj PUBLIC OpenPFC_ENABLE_HEFFTE)
  target_compile_definitions(openpfc_frontend_obj PUBLIC OpenPFC_ENABLE_HEFFTE)
  # Prefer already-fetched target; fall back to find_package if needed
  if(TARGET Heffte::Heffte)
    target_link_libraries(openpfc PUBLIC Heffte::Heffte)
    target_link_libraries(openpfc_kernel_obj PRIVATE Heffte::Heffte)
    target_link_libraries(openpfc_frontend_obj PRIVATE Heffte::Heffte)
  elseif(TARGET Heffte)
    target_link_libraries(openpfc PUBLIC Heffte)
    target_link_libraries(openpfc_kernel_obj PRIVATE Heffte)
    target_link_libraries(openpfc_frontend_obj PRIVATE Heffte)
  elseif(TARGET heffte)
    target_link_libraries(openpfc PUBLIC heffte)
    target_link_libraries(openpfc_kernel_obj PRIVATE heffte)
    target_link_libraries(openpfc_frontend_obj PRIVATE heffte)
  else()
    find_package(Heffte REQUIRED)
    target_link_libraries(openpfc PUBLIC Heffte::Heffte)
    target_link_libraries(openpfc_kernel_obj PRIVATE Heffte::Heffte)
    target_link_libraries(openpfc_frontend_obj PRIVATE Heffte::Heffte)
  endif()
  # Public spectral headers include <heffte.h>, and the static library needs
  # HeFFTe symbols at final link. PUBLIC propagates both. OpenPFCConfig finds
  # Heffte before the exported targets, so a consumer links only OpenPFC::openpfc.
  # The headers still name HeFFTe types; this does not hide that include.
endif()

if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  target_link_libraries(openpfc_kernel_obj PRIVATE CUDA::cudart)
  if(OpenPFC_ENABLE_CUDA_SPECTRAL)
    target_compile_definitions(openpfc_kernel_obj PUBLIC OpenPFC_ENABLE_CUDA_SPECTRAL)
    target_compile_definitions(openpfc PUBLIC OpenPFC_ENABLE_CUDA_SPECTRAL)
  endif()
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  target_link_libraries(openpfc_kernel_obj PRIVATE hip::host)
  target_compile_definitions(openpfc_kernel_obj PUBLIC OpenPFC_ENABLE_HIP_SPECTRAL)
  target_compile_definitions(openpfc PUBLIC OpenPFC_ENABLE_HIP_SPECTRAL)
endif()

# GPU kernel libraries. Shared TUs: sparse_vector_ops, fill, elementwise_ops.
# Both vendors compile padded_halo_faces into the kernel lib (not per executable).
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    openpfc_add_gpu_kernel_library(openpfc_gpu_kernels cu CUDA::cudart
        src/openpfc/runtime/gpu/padded_halo_faces.cu
        src/openpfc/runtime/gpu/grain_tracking.cu)
    if(OpenPFC_ENABLE_CUDA_SPECTRAL AND OpenPFC_HAS_FFTW3)
      # Keep these objects in openpfc with the shared axis-contract definitions:
      # a reverse dependency from the separate kernel archive breaks static links.
      target_sources(openpfc_kernel_obj PRIVATE
          src/openpfc/runtime/gpu/batched_transform.cu)
      set_target_properties(openpfc_kernel_obj PROPERTIES
          CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
      target_link_libraries(openpfc_kernel_obj PRIVATE CUDA::cufft)
      target_link_libraries(openpfc PRIVATE CUDA::cufft)
    endif()
    # Pin CUDA language standard: project C++20 would otherwise map to "CUDA20" on
    # some CMake versions that do not know the matching nvcc flags (e.g. CMake 3.22).
    # Device-link the static archive. Without this, nvcc relocatable objects
    # in the .a call __cudaRegisterLinkedBinary_* that CMake only emits when
    # the consumer's own .cu files are dlinked — CUDA executables then fail
    # at host link with an undefined register symbol (seen on Tohtori/nvcc 13.1).
    set_target_properties(openpfc_gpu_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    message(STATUS "✅ GPU kernel library enabled")
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
    openpfc_add_gpu_kernel_library(openpfc_hip_kernels hip hip::host
        src/openpfc/runtime/gpu/padded_halo_faces.hip
        src/openpfc/runtime/gpu/grain_tracking.hip)
    if(OpenPFC_ENABLE_HIP_SPECTRAL AND OpenPFC_HAS_FFTW3)
      find_package(rocfft REQUIRED CONFIG)
      set_source_files_properties(src/openpfc/runtime/gpu/batched_transform.hip
          PROPERTIES LANGUAGE HIP)
      target_sources(openpfc_kernel_obj PRIVATE
          src/openpfc/runtime/gpu/batched_transform.hip)
      set_target_properties(openpfc_kernel_obj PROPERTIES
          HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
      target_link_libraries(openpfc_kernel_obj PRIVATE roc::rocfft)
      target_link_libraries(openpfc PRIVATE roc::rocfft)
    endif()
    set_target_properties(openpfc_hip_kernels PROPERTIES
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
    )
    message(STATUS "✅ HIP kernel library enabled")
endif()

# Periodic microelasticity device kernels. A solver target, not part of the
# generic runtime library and not an application.
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  set_source_files_properties(
      src/openpfc/solvers/microelasticity/microelasticity_hip_kernels.hip
      PROPERTIES LANGUAGE HIP)
  add_library(openpfc_microelasticity_hip
      src/openpfc/solvers/microelasticity/microelasticity_hip_kernels.hip)
  target_include_directories(openpfc_microelasticity_hip
      PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_link_libraries(openpfc_microelasticity_hip PUBLIC OpenPFC hip::host)
  target_compile_definitions(openpfc_microelasticity_hip PUBLIC
      OpenPFC_ENABLE_HIP OpenPFC_ENABLE_HIP_SPECTRAL)
  set_target_properties(openpfc_microelasticity_hip PROPERTIES
      HIP_STANDARD 17
      HIP_STANDARD_REQUIRED ON)
  message(STATUS "openpfc_microelasticity_hip (periodic microelasticity) enabled")
endif()

# Periodic incompressible rotational step. One backend per build: HIP and
# CUDA must not both define these symbols in one link.
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL
   AND OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE AND OpenPFC_ENABLE_CUDA_SPECTRAL)
  message(FATAL_ERROR
    "openpfc_incompressible device kernels support one spectral backend per build")
endif()
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  set_source_files_properties(
      src/openpfc/runtime/gpu/incompressible_device_kernels.hip
      PROPERTIES LANGUAGE HIP)
  add_library(openpfc_incompressible_hip
      src/openpfc/runtime/gpu/incompressible_device_kernels.hip)
  target_include_directories(openpfc_incompressible_hip
      PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_link_libraries(openpfc_incompressible_hip PUBLIC OpenPFC hip::host)
  target_compile_definitions(openpfc_incompressible_hip PUBLIC
      OpenPFC_ENABLE_HIP OpenPFC_ENABLE_HIP_SPECTRAL)
  set_target_properties(openpfc_incompressible_hip PROPERTIES
      HIP_STANDARD 20
      HIP_STANDARD_REQUIRED ON)
  message(STATUS "openpfc_incompressible_hip (periodic rotational step) enabled")
endif()
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE AND OpenPFC_ENABLE_CUDA_SPECTRAL)
  set_source_files_properties(
      src/openpfc/runtime/gpu/incompressible_device_kernels.cu
      PROPERTIES LANGUAGE CUDA)
  add_library(openpfc_incompressible_cuda
      src/openpfc/runtime/gpu/incompressible_device_kernels.cu)
  target_include_directories(openpfc_incompressible_cuda
      PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_link_libraries(openpfc_incompressible_cuda PUBLIC OpenPFC CUDA::cudart)
  target_compile_definitions(openpfc_incompressible_cuda PUBLIC
      OpenPFC_ENABLE_CUDA OpenPFC_ENABLE_CUDA_SPECTRAL)
  set_target_properties(openpfc_incompressible_cuda PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON)
  message(STATUS "openpfc_incompressible_cuda (periodic rotational step) enabled")
endif()

# Add tomlplusplus include directory (header-only library)
# Since it's header-only (often from FetchContent), we just need the include path
# Use $<BUILD_INTERFACE:...> to avoid export issues with build directory paths
if(DEFINED TOMLPLUSPLUS_SOURCE_DIR)
  target_include_directories(openpfc PUBLIC $<BUILD_INTERFACE:${TOMLPLUSPLUS_SOURCE_DIR}/include>)
  target_include_directories(openpfc_kernel_obj PUBLIC $<BUILD_INTERFACE:${TOMLPLUSPLUS_SOURCE_DIR}/include>)
  target_include_directories(openpfc_frontend_obj PUBLIC $<BUILD_INTERFACE:${TOMLPLUSPLUS_SOURCE_DIR}/include>)
elseif(DEFINED tomlplusplus_SOURCE_DIR)
  target_include_directories(openpfc PUBLIC $<BUILD_INTERFACE:${tomlplusplus_SOURCE_DIR}/include>)
  target_include_directories(openpfc_kernel_obj PUBLIC $<BUILD_INTERFACE:${tomlplusplus_SOURCE_DIR}/include>)
  target_include_directories(openpfc_frontend_obj PUBLIC $<BUILD_INTERFACE:${tomlplusplus_SOURCE_DIR}/include>)
elseif(TARGET tomlplusplus::tomlplusplus)
  get_target_property(TOML_INCLUDE_DIR tomlplusplus::tomlplusplus INTERFACE_INCLUDE_DIRECTORIES)
  if(TOML_INCLUDE_DIR)
    target_include_directories(openpfc PUBLIC $<BUILD_INTERFACE:${TOML_INCLUDE_DIR}>)
    target_include_directories(openpfc_kernel_obj PUBLIC $<BUILD_INTERFACE:${TOML_INCLUDE_DIR}>)
    target_include_directories(openpfc_frontend_obj PUBLIC $<BUILD_INTERFACE:${TOML_INCLUDE_DIR}>)
  endif()
elseif(TARGET tomlplusplus_tomlplusplus)
  get_target_property(TOML_INCLUDE_DIR tomlplusplus_tomlplusplus INTERFACE_INCLUDE_DIRECTORIES)
  if(TOML_INCLUDE_DIR)
    target_include_directories(openpfc PUBLIC $<BUILD_INTERFACE:${TOML_INCLUDE_DIR}>)
    target_include_directories(openpfc_kernel_obj PUBLIC $<BUILD_INTERFACE:${TOML_INCLUDE_DIR}>)
    target_include_directories(openpfc_frontend_obj PUBLIC $<BUILD_INTERFACE:${TOML_INCLUDE_DIR}>)
  endif()
endif()

# Require C++20
target_compile_features(openpfc PUBLIC cxx_std_20)

target_compile_definitions(openpfc PUBLIC
    "OPENPFC_PROFILING_LEVEL=${OpenPFC_PROFILING_LEVEL}")
target_compile_definitions(openpfc PRIVATE
    "OPENPFC_PROFILING_BUILD_VERSION=\"${PROJECT_VERSION}\"")

# GCC 8.x: std::filesystem is in libstdc++fs and must be linked explicitly
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
  target_link_libraries(openpfc PUBLIC stdc++fs)
endif()

# Compile-only CUDA Field residency API. Not a Catch2 test and not linked into
# openpfc-tests: a global constructor in this TU used to run CUDA at process
# start (CTest discovery on GPU-less runners).
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  add_library(openpfc_cuda_field_residency_compile OBJECT
      ${PROJECT_SOURCE_DIR}/tests/unit/kernel/data/cuda_field_residency_compile.cpp)
  target_link_libraries(openpfc_cuda_field_residency_compile PRIVATE
      openpfc openpfc_gpu_compile_defs CUDA::cudart)
  target_compile_features(openpfc_cuda_field_residency_compile PRIVATE cxx_std_20)
endif()
