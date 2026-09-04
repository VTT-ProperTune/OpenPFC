# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# HIP (ROCm) support detection and configuration

option(OpenPFC_ENABLE_HIP "Enable HIP (ROCm) support" OFF)

set(OpenPFC_HIP_AVAILABLE FALSE)

# Site ROCm modules (Tohtori rocm/7.2.1) often set PATH + LD_LIBRARY_PATH but
# not CMAKE_PREFIX_PATH / ROCM_PATH, so find_package(HIP) misses hip-config.cmake.
function(openpfc_rocm_has_hip_config root)
  if(EXISTS "${root}/lib/cmake/hip/hip-config.cmake"
     OR EXISTS "${root}/lib64/cmake/hip/hip-config.cmake"
     OR EXISTS "${root}/lib/cmake/hip/HIPConfig.cmake")
    set(openpfc_rocm_has_hip_config_result TRUE PARENT_SCOPE)
  else()
    set(openpfc_rocm_has_hip_config_result FALSE PARENT_SCOPE)
  endif()
endfunction()

function(openpfc_guess_rocm_root out_var)
  set(_root "")
  foreach(_env_name IN ITEMS ROCM_PATH HIP_PATH EBROOTROCM ROCM_ROOT)
    if(DEFINED ENV{${_env_name}})
      openpfc_rocm_has_hip_config("$ENV{${_env_name}}")
      if(openpfc_rocm_has_hip_config_result)
        set(_root "$ENV{${_env_name}}")
        break()
      endif()
    endif()
  endforeach()
  if(NOT _root)
    foreach(_cand IN ITEMS /opt/rocm-7.2.1 /opt/rocm)
      openpfc_rocm_has_hip_config("${_cand}")
      if(openpfc_rocm_has_hip_config_result)
        set(_root "${_cand}")
        break()
      endif()
    endforeach()
  endif()
  if(NOT _root)
    find_program(_openpfc_hipcc hipcc)
    if(_openpfc_hipcc)
      get_filename_component(_cursor "${_openpfc_hipcc}" REALPATH)
      get_filename_component(_cursor "${_cursor}" DIRECTORY)
      foreach(_i RANGE 1 5)
        get_filename_component(_cursor "${_cursor}" DIRECTORY)
        openpfc_rocm_has_hip_config("${_cursor}")
        if(openpfc_rocm_has_hip_config_result)
          set(_root "${_cursor}")
          break()
        endif()
      endforeach()
    endif()
  endif()
  set(${out_var} "${_root}" PARENT_SCOPE)
endfunction()

function(openpfc_prepend_prefix_path root)
  if(root AND EXISTS "${root}")
    list(PREPEND CMAKE_PREFIX_PATH "${root}")
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
  endif()
endfunction()

if(OpenPFC_ENABLE_HIP)
  openpfc_guess_rocm_root(_openpfc_rocm_root)
  openpfc_prepend_prefix_path("${_openpfc_rocm_root}")
  if(_openpfc_rocm_root AND NOT DEFINED ENV{ROCM_PATH})
    set(ENV{ROCM_PATH} "${_openpfc_rocm_root}")
  endif()

  if(_openpfc_rocm_root)
    foreach(_hip_cfg IN ITEMS
        "${_openpfc_rocm_root}/lib/cmake/hip"
        "${_openpfc_rocm_root}/lib64/cmake/hip")
      if(EXISTS "${_hip_cfg}/hip-config.cmake" OR EXISTS "${_hip_cfg}/HIPConfig.cmake")
        set(hip_DIR "${_hip_cfg}")
        break()
      endif()
    endforeach()
  endif()

  # ROCm 5 used HIPConfig.cmake (package HIP); ROCm 6/7 ship hip-config.cmake (package hip).
  find_package(hip CONFIG QUIET)
  if(NOT hip_FOUND)
    find_package(HIP QUIET)
  endif()

  # Host C++20 headers: ROCm clang does not search GCC libstdc++ unless told.
  if(DEFINED ENV{OPENPFC_GCC_ROOT} AND EXISTS "$ENV{OPENPFC_GCC_ROOT}")
    if(NOT CMAKE_HIP_FLAGS MATCHES "gcc-toolchain")
      string(APPEND CMAKE_HIP_FLAGS
        " --gcc-toolchain=$ENV{OPENPFC_GCC_ROOT} -stdlib=libstdc++")
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS}" CACHE STRING
        "HIP compile flags (OpenPFC)" FORCE)
      message(STATUS "   CMAKE_HIP_FLAGS += --gcc-toolchain=$ENV{OPENPFC_GCC_ROOT} -stdlib=libstdc++")
    endif()
  endif()

  # CMake 3.21–3.24 reject the hipcc wrapper as CMAKE_HIP_COMPILER; they want
  # the ROCm Clang that compiles HIP (HeFFTe on this cluster uses
  # /opt/rocm/lib/llvm/bin/clang++).
  if(CMAKE_HIP_COMPILER MATCHES "hipcc")
    unset(CMAKE_HIP_COMPILER CACHE)
    unset(CMAKE_HIP_COMPILER)
  endif()
  if(NOT CMAKE_HIP_COMPILER AND _openpfc_rocm_root)
    find_program(CMAKE_HIP_COMPILER clang++
      PATHS "${_openpfc_rocm_root}/lib/llvm/bin"
            "${_openpfc_rocm_root}/llvm/bin"
            "${_openpfc_rocm_root}/bin"
      NO_DEFAULT_PATH)
  endif()

  if(NOT hip_FOUND AND NOT HIP_FOUND)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
      enable_language(HIP)
      if(CMAKE_HIP_COMPILER)
        set(HIP_FOUND TRUE)
      endif()
    endif()
  endif()

  if(hip_FOUND OR HIP_FOUND)
    set(OpenPFC_HIP_AVAILABLE TRUE)
    # OpenPFC_ENABLE_HIP / OpenPFC_MPI_HIP_AWARE are PUBLIC usage requirements
    # on openpfc (and kernel libs) in LibraryConfiguration.cmake — same reason
    # as the CUDA block in CUDASupport.cmake.

    option(OpenPFC_MPI_HIP_AWARE "Use GPU-aware MPI with HIP (device pointers in MPI_Send/Recv)" ON)
    if(OpenPFC_MPI_HIP_AWARE)
      message(STATUS "   OpenPFC_MPI_HIP_AWARE=ON (MPI uses device pointers)")
    endif()

    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
      enable_language(HIP)
    endif()

    # Ubuntu 24.04 links executables as PIE. ROCm Clang does not add -fPIC to
    # HIP TUs by default, so host ld fails with R_X86_64_32 against .rodata.
    if(NOT CMAKE_HIP_FLAGS MATCHES "-fPIC" AND NOT CMAKE_HIP_FLAGS MATCHES "-fPIE")
      string(APPEND CMAKE_HIP_FLAGS " -fPIC")
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS}" CACHE STRING
        "HIP compile flags (OpenPFC)" FORCE)
    endif()

    if(_openpfc_rocm_root)
      message(STATUS "✅ HIP enabled (found HIP; ROCm root ${_openpfc_rocm_root})")
    else()
      message(STATUS "✅ HIP enabled (found HIP)")
    endif()
  else()
    message(FATAL_ERROR
      "OpenPFC_ENABLE_HIP=ON but HIP was not found.\n"
      "  Load a ROCm module so hipcc is on PATH, or set CMAKE_PREFIX_PATH / ROCM_PATH\n"
      "  to the ROCm prefix (the directory that contains lib/cmake/hip/hip-config.cmake).\n"
      "  Example: -DCMAKE_PREFIX_PATH=/opt/rocm or /opt/rocm-7.2.1\n"
      "  scripts/build.sh --with-rocm locates ROCM_PATH after loading ROCM_MODULE.")
  endif()
else()
  message(STATUS "HIP disabled (use -DOpenPFC_ENABLE_HIP=ON to enable)")
endif()

# Note: HIP and CUDA can both be enabled in the same build (different executables
# e.g. tungsten_hip vs tungsten_cuda) when both toolkits are present. scripts/build.sh
# still treats --with-cuda and --with-rocm as exclusive because each machine node
# typically has one vendor stack and one HeFFTe prefix.
