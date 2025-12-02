# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# CUDA support detection and configuration

option(OpenPFC_ENABLE_CUDA "Enable CUDA support" OFF)

set(OpenPFC_CUDA_AVAILABLE FALSE)

if(OpenPFC_ENABLE_CUDA)
    # Try to find CUDA, but don't fail if not found
    enable_language(CUDA OPTIONAL)
    find_package(CUDAToolkit QUIET)
    
    if(CUDAToolkit_FOUND)
        set(OpenPFC_CUDA_AVAILABLE TRUE)
        add_compile_definitions(OpenPFC_ENABLE_CUDA)
        
        # Set CUDA architectures (required by CMake policy CMP0104)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures to compile for" FORCE)
        endif()
        
        message(STATUS "✅ CUDA enabled (found CUDAToolkit)")
        message(STATUS "   CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(WARNING "⚠️  OpenPFC_ENABLE_CUDA=ON but CUDAToolkit not found. CUDA support disabled.")
        message(WARNING "   Install CUDA toolkit or set CUDAToolkit_ROOT to enable CUDA.")
        set(OpenPFC_ENABLE_CUDA OFF)  # Disable if not found
    endif()
else()
    message(STATUS "CUDA disabled (use -DOpenPFC_ENABLE_CUDA=ON to enable)")
endif()
