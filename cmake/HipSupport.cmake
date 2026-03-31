# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# HIP (ROCm) support detection and configuration

option(OpenPFC_ENABLE_HIP "Enable HIP (ROCm) support" OFF)

set(OpenPFC_HIP_AVAILABLE FALSE)

if(OpenPFC_ENABLE_HIP)
  # Try to find HIP (ROCm). Users may set CMAKE_PREFIX_PATH to e.g. /opt/rocm
  find_package(HIP QUIET)

  if(HIP_FOUND)
    set(OpenPFC_HIP_AVAILABLE TRUE)
    add_compile_definitions(OpenPFC_ENABLE_HIP)

    option(OpenPFC_MPI_HIP_AWARE "Use GPU-aware MPI with HIP (device pointers in MPI_Send/Recv)" ON)
    if(OpenPFC_MPI_HIP_AWARE)
      add_compile_definitions(OpenPFC_MPI_HIP_AWARE)
      message(STATUS "   OpenPFC_MPI_HIP_AWARE=ON (MPI uses device pointers)")
    endif()

    # Enable HIP language for .hip sources (CMake 3.21+)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
      enable_language(HIP)
    endif()

    message(STATUS "✅ HIP enabled (found HIP)")
  else()
    message(WARNING "⚠️  OpenPFC_ENABLE_HIP=ON but HIP not found. HIP support disabled.")
    message(WARNING "   Install ROCm or set CMAKE_PREFIX_PATH (e.g. -DCMAKE_PREFIX_PATH=/opt/rocm).")
    set(OpenPFC_ENABLE_HIP OFF)
  endif()
else()
  message(STATUS "HIP disabled (use -DOpenPFC_ENABLE_HIP=ON to enable)")
endif()

# Note: HIP and CUDA can both be enabled in the same build (different executables
# e.g. tungsten_hip vs tungsten_cuda). We do not force mutual exclusivity.
