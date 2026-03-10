# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# GPU unit tests - only build if CUDA is enabled AND found
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    add_executable(test_gpu_device test_device.cpp)
    target_include_directories(test_gpu_device PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_gpu_device PRIVATE Catch2::Catch2 CUDA::cudart)
    add_test(NAME GPU_DeviceDetection COMMAND test_gpu_device)

    add_executable(test_gpu_vector test_gpu_vector.cpp)
    target_include_directories(test_gpu_vector PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_gpu_vector PRIVATE Catch2::Catch2 CUDA::cudart)
    add_test(NAME GPU_Vector COMMAND test_gpu_vector)

    add_executable(test_gpu_kernels test_kernels_simple.cpp)
    target_include_directories(test_gpu_kernels PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_gpu_kernels PRIVATE Catch2::Catch2 openpfc_gpu_kernels CUDA::cudart)
    add_test(NAME GPU_Kernels COMMAND test_gpu_kernels)

    add_executable(test_gpu_fft test_fft_cuda.cpp)
    target_include_directories(test_gpu_fft PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_gpu_fft PRIVATE Catch2::Catch2 openpfc CUDA::cudart MPI::MPI_CXX)
    add_test(NAME GPU_FFT COMMAND test_gpu_fft)

    message(STATUS "✅ GPU tests enabled")
elseif(OpenPFC_ENABLE_CUDA)
    message(STATUS "⚠️  GPU tests disabled (CUDA not found)")
else()
    message(STATUS "ℹ️  GPU tests disabled (OpenPFC_ENABLE_CUDA=OFF)")
endif()
