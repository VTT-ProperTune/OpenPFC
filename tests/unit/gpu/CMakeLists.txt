# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# GPU unit tests - only build if CUDA is enabled AND found
if(OPENPFC_ENABLE_CUDA AND OPENPFC_CUDA_AVAILABLE)
    # Test 1: Device detection
    add_executable(test_gpu_device
        test_device.cpp
    )
    
    target_include_directories(test_gpu_device PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    
    target_link_libraries(test_gpu_device PRIVATE
        Catch2::Catch2
    )
    
    target_link_libraries(test_gpu_device PRIVATE
        CUDA::cudart
    )
    add_test(NAME GPU_DeviceDetection COMMAND test_gpu_device)
    
    # Test 2: GPUVector
    add_executable(test_gpu_vector
        test_gpu_vector.cpp
    )
    
    target_include_directories(test_gpu_vector PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    
    target_link_libraries(test_gpu_vector PRIVATE
        Catch2::Catch2
    )
    
    target_link_libraries(test_gpu_vector PRIVATE
        CUDA::cudart
    )
    add_test(NAME GPU_Vector COMMAND test_gpu_vector)
    
    message(STATUS "✅ GPU tests enabled")
elseif(OPENPFC_ENABLE_CUDA)
    message(STATUS "⚠️  GPU tests disabled (CUDA not found)")
else()
    message(STATUS "ℹ️  GPU tests disabled (OPENPFC_ENABLE_CUDA=OFF)")
endif()
