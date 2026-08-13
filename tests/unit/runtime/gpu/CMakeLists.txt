# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# GPU unit tests - only build if CUDA is enabled AND found
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    add_executable(test_gpu_device test_device.cpp)
    target_include_directories(test_gpu_device PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_gpu_device PRIVATE Catch2::Catch2 CUDA::cudart
                                                openpfc_gpu_compile_defs)
    add_test(NAME GPU_DeviceDetection COMMAND test_gpu_device)

    add_executable(test_gpu_fft test_fft_cuda.cpp)
    target_include_directories(test_gpu_fft PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_gpu_fft PRIVATE Catch2::Catch2 openpfc CUDA::cudart MPI::MPI_CXX)
    add_test(NAME GPU_FFT COMMAND test_gpu_fft)

    add_executable(test_exchange_cuda_fail_closed test_exchange_cuda_fail_closed.cpp)
    target_include_directories(test_exchange_cuda_fail_closed PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_exchange_cuda_fail_closed PRIVATE
        Catch2::Catch2 openpfc CUDA::cudart MPI::MPI_CXX)
    add_test(NAME CUDA_ExchangeFailClosed COMMAND test_exchange_cuda_fail_closed)
    # This test's assertions only compile in when OpenPFC_MPI_CUDA_AWARE is
    # OFF; under the default ON build it SKIPs its one TEST_CASE, and Catch2
    # returns exit code 4 for an all-skipped run. Tell CTest that's a skip,
    # not a failure.
    set_tests_properties(CUDA_ExchangeFailClosed PROPERTIES SKIP_RETURN_CODE 4)

    message(STATUS "✅ GPU tests enabled")
elseif(OpenPFC_ENABLE_CUDA)
    message(STATUS "⚠️  GPU tests disabled (CUDA not found)")
else()
    message(STATUS "ℹ️  GPU tests disabled (OpenPFC_ENABLE_CUDA=OFF)")
endif()

# GPU autotuning tests - only build if GPU autotuning is enabled
if(OpenPFC_ENABLE_GPU_AUTOTUNING)
    add_executable(test_gpu_autotune test_gpu_autotune.cpp)
    target_include_directories(test_gpu_autotune PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_gpu_autotune PRIVATE
        Catch2::Catch2
        nlohmann_json::nlohmann_json
        openpfc_gpu_compile_defs
    )
    if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
        target_link_libraries(test_gpu_autotune PRIVATE CUDA::cudart)
    endif()
    if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
        target_link_libraries(test_gpu_autotune PRIVATE hip::host)
    endif()
    add_test(NAME GPU_Autotune COMMAND test_gpu_autotune)
    message(STATUS "✅ GPU autotuning tests enabled")
endif()

# HIP unit tests - only build if HIP is enabled AND found
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
    add_executable(test_hip_device test_device.cpp)
    target_include_directories(test_hip_device PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_hip_device PRIVATE Catch2::Catch2 hip::host
                                               openpfc_gpu_compile_defs)
    add_test(NAME HIP_DeviceDetection COMMAND test_hip_device)

    add_executable(test_exchange_hip_fail_closed test_exchange_hip_fail_closed.cpp)
    target_include_directories(test_exchange_hip_fail_closed PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_exchange_hip_fail_closed PRIVATE
        Catch2::Catch2 openpfc hip::host MPI::MPI_CXX)
    add_test(NAME HIP_ExchangeFailClosed COMMAND test_exchange_hip_fail_closed)
    # See CUDA_ExchangeFailClosed above: SKIPs (exit code 4) under the
    # default OpenPFC_MPI_HIP_AWARE=ON build; that's a skip, not a failure.
    set_tests_properties(HIP_ExchangeFailClosed PROPERTIES SKIP_RETURN_CODE 4)

    message(STATUS "✅ HIP tests enabled")
elseif(OpenPFC_ENABLE_HIP)
    message(STATUS "⚠️  HIP tests disabled (HIP not found)")
else()
    message(STATUS "ℹ️  HIP tests disabled (OpenPFC_ENABLE_HIP=OFF)")
endif()

# SparseVector tests - build based on available backends
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    add_executable(test_sparsevector_cuda
        ${CMAKE_SOURCE_DIR}/tests/unit/operators/test_sparsevector_cuda.cpp)
    target_include_directories(test_sparsevector_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/operators
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_sparsevector_cuda PRIVATE Catch2::Catch2 openpfc_gpu_kernels CUDA::cudart)
    add_test(NAME CUDA_SparseVector COMMAND test_sparsevector_cuda)

    add_executable(test_deep_copy_fill_cuda test_deep_copy_fill.cpp)
    target_include_directories(test_deep_copy_fill_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_deep_copy_fill_cuda PRIVATE Catch2::Catch2 openpfc_gpu_kernels CUDA::cudart)
    target_compile_definitions(test_deep_copy_fill_cuda PRIVATE OPENPFC_TEST_DEEP_COPY_CUDA)
    add_test(NAME CUDA_DeepCopyFill COMMAND test_deep_copy_fill_cuda)
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
    add_executable(test_sparsevector_hip
        ${CMAKE_SOURCE_DIR}/tests/unit/operators/test_sparsevector_hip.cpp)
    target_include_directories(test_sparsevector_hip PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/operators
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_sparsevector_hip PRIVATE Catch2::Catch2 openpfc_hip_kernels hip::host)
    add_test(NAME HIP_SparseVector COMMAND test_sparsevector_hip)

    add_executable(test_deep_copy_fill_hip test_deep_copy_fill.cpp)
    target_include_directories(test_deep_copy_fill_hip PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_deep_copy_fill_hip PRIVATE Catch2::Catch2 openpfc_hip_kernels hip::host)
    target_compile_definitions(test_deep_copy_fill_hip PRIVATE OPENPFC_TEST_DEEP_COPY_HIP)
    add_test(NAME HIP_DeepCopyFill COMMAND test_deep_copy_fill_hip)
endif()
