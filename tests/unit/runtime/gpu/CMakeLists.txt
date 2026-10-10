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

    if(OpenPFC_ENABLE_CUDA_SPECTRAL)
    add_executable(test_gpu_fft test_fft_cuda.cpp)
    target_include_directories(test_gpu_fft PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_gpu_fft PRIVATE Catch2::Catch2 openpfc CUDA::cudart MPI::MPI_CXX Heffte::Heffte)
    add_test(NAME GPU_FFT COMMAND test_gpu_fft)

    # Issue #382: one-rank single-device FFT vs the heFFTe backend.
    add_executable(test_fft_single_device_cuda test_fft_single_device.cpp)
    target_include_directories(test_fft_single_device_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_fft_single_device_cuda PRIVATE Catch2::Catch2 openpfc
        openpfc_gpu_kernels CUDA::cudart CUDA::cufft MPI::MPI_CXX Heffte::Heffte)
    add_test(NAME GPU_FFT_SingleDevice COMMAND test_fft_single_device_cuda)

    # One device-vs-host ETD parity target (plain, mean-field, moving-frame toys);
    # the toy pointwise functors are instantiated for the device in the .cu TU.
    add_executable(test_spectral_etd_system_cuda
        test_spectral_etd_system_gpu.cpp
        spectral_etd_toys_pointwise.cu)
    add_executable(test_spectral_flux_cuda
        test_spectral_flux_gpu.cpp
        spectral_flux_pointwise.cu)
    target_include_directories(test_spectral_etd_system_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_spectral_etd_system_cuda PRIVATE
        Catch2::Catch2 openpfc openpfc_gpu_kernels CUDA::cudart MPI::MPI_CXX
        Heffte::Heffte)
    target_compile_definitions(test_spectral_etd_system_cuda PRIVATE
        OPENPFC_TEST_SPECTRAL_ETD_CUDA)
    set_target_properties(test_spectral_etd_system_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON)
    add_test(NAME CUDA_SpectralETD COMMAND test_spectral_etd_system_cuda)

    target_include_directories(test_spectral_flux_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_spectral_flux_cuda PRIVATE
        Catch2::Catch2 openpfc openpfc_gpu_kernels CUDA::cudart MPI::MPI_CXX
        Heffte::Heffte)
    target_compile_definitions(test_spectral_flux_cuda PRIVATE
        OPENPFC_TEST_SPECTRAL_FLUX_CUDA)
    set_target_properties(test_spectral_flux_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON)
    add_test(NAME CUDA_SpectralFlux COMMAND test_spectral_flux_cuda)

    add_executable(test_gpu_spectral_stack_cuda test_gpu_spectral_stack.cpp)
    target_include_directories(test_gpu_spectral_stack_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_gpu_spectral_stack_cuda PRIVATE
        Catch2::Catch2 openpfc nlohmann_json::nlohmann_json Heffte::Heffte
        CUDA::cudart MPI::MPI_CXX)
    target_compile_definitions(test_gpu_spectral_stack_cuda PRIVATE
        OPENPFC_TEST_GPU_STACK_CUDA)
    add_test(NAME CUDA_GPUSpectralStack COMMAND test_gpu_spectral_stack_cuda)
    add_test(NAME session-matrix-cuda COMMAND test_gpu_spectral_stack_cuda
        "[session][json]")
    endif()

    add_executable(test_fd_gpu_stack_cuda test_fd_gpu_stack.cpp)
    target_include_directories(test_fd_gpu_stack_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_fd_gpu_stack_cuda PRIVATE
        Catch2::Catch2 openpfc nlohmann_json::nlohmann_json
        CUDA::cudart MPI::MPI_CXX openpfc_gpu_kernels)
    set_target_properties(test_fd_gpu_stack_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_definitions(test_fd_gpu_stack_cuda PRIVATE
        OPENPFC_TEST_FD_GPU_STACK_CUDA)
    add_test(NAME CUDA_FDGPUStack COMMAND test_fd_gpu_stack_cuda)

    add_executable(test_observable_reduce_cuda test_observable_reduce_gpu.cpp)
    target_include_directories(test_observable_reduce_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_observable_reduce_cuda PRIVATE
        Catch2::Catch2 openpfc CUDA::cudart MPI::MPI_CXX)
    target_compile_definitions(test_observable_reduce_cuda PRIVATE
        OPENPFC_TEST_OBSERVABLE_CUDA)
    add_test(NAME CUDA_ObservableReduce COMMAND test_observable_reduce_cuda)

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

    add_executable(test_fd_gpu_stack_hip test_fd_gpu_stack.cpp)
    target_include_directories(test_fd_gpu_stack_hip PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_fd_gpu_stack_hip PRIVATE
        Catch2::Catch2 openpfc nlohmann_json::nlohmann_json hip::host MPI::MPI_CXX
        openpfc_hip_kernels)
    target_compile_definitions(test_fd_gpu_stack_hip PRIVATE
        OPENPFC_TEST_FD_GPU_STACK_HIP)
    add_test(NAME HIP_FDGPUStack COMMAND test_fd_gpu_stack_hip)

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

    if(OpenPFC_ENABLE_HIP_SPECTRAL)
        add_executable(test_gpu_fft_hip test_fft_hip.cpp)
        target_include_directories(test_gpu_fft_hip PRIVATE
            ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
            ${CMAKE_SOURCE_DIR}/include
        )
        target_link_libraries(test_gpu_fft_hip PRIVATE
            Catch2::Catch2 openpfc hip::host MPI::MPI_CXX Heffte::Heffte)
        add_test(NAME HIP_FFT COMMAND test_gpu_fft_hip)

        # Issue #382: one-rank single-device FFT (rocFFT) vs the heFFTe backend.
        add_executable(test_fft_single_device_hip test_fft_single_device.cpp)
        target_include_directories(test_fft_single_device_hip PRIVATE
            ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
            ${CMAKE_SOURCE_DIR}/include
        )
        target_compile_definitions(test_fft_single_device_hip PRIVATE OPENPFC_TEST_HIP)
        target_link_libraries(test_fft_single_device_hip PRIVATE Catch2::Catch2 openpfc
            openpfc_hip_kernels hip::host MPI::MPI_CXX Heffte::Heffte)
        add_test(NAME HIP_FFT_SingleDevice COMMAND test_fft_single_device_hip)

        add_executable(test_spectral_etd_system_hip
            test_spectral_etd_system_gpu.cpp
            spectral_etd_toys_pointwise.hip)
        add_executable(test_spectral_flux_hip
            test_spectral_flux_gpu.cpp
            spectral_flux_pointwise.hip)
        target_include_directories(test_spectral_etd_system_hip PRIVATE
            ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
            ${CMAKE_SOURCE_DIR}/tests
            ${CMAKE_SOURCE_DIR}/include
        )
        target_link_libraries(test_spectral_etd_system_hip PRIVATE
            Catch2::Catch2 openpfc openpfc_hip_kernels hip::host MPI::MPI_CXX
            Heffte::Heffte)
        target_compile_definitions(test_spectral_etd_system_hip PRIVATE
            OPENPFC_TEST_SPECTRAL_ETD_HIP)
        set_target_properties(test_spectral_etd_system_hip PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON)
        add_test(NAME HIP_SpectralETD COMMAND test_spectral_etd_system_hip)

        target_include_directories(test_spectral_flux_hip PRIVATE
            ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
            ${CMAKE_SOURCE_DIR}/tests
            ${CMAKE_SOURCE_DIR}/include
        )
        target_link_libraries(test_spectral_flux_hip PRIVATE
            Catch2::Catch2 openpfc openpfc_hip_kernels hip::host MPI::MPI_CXX
            Heffte::Heffte)
        target_compile_definitions(test_spectral_flux_hip PRIVATE
            OPENPFC_TEST_SPECTRAL_FLUX_HIP)
        set_target_properties(test_spectral_flux_hip PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON)
        add_test(NAME HIP_SpectralFlux COMMAND test_spectral_flux_hip)

        add_executable(test_gpu_spectral_stack_hip test_gpu_spectral_stack.cpp)
        target_include_directories(test_gpu_spectral_stack_hip PRIVATE
            ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
            ${CMAKE_SOURCE_DIR}/include
        )
        target_link_libraries(test_gpu_spectral_stack_hip PRIVATE
            Catch2::Catch2 openpfc nlohmann_json::nlohmann_json Heffte::Heffte
            hip::host MPI::MPI_CXX)
        target_compile_definitions(test_gpu_spectral_stack_hip PRIVATE
            OPENPFC_TEST_GPU_STACK_HIP)
        add_test(NAME HIP_GPUSpectralStack COMMAND test_gpu_spectral_stack_hip)
        add_test(NAME session-matrix-hip COMMAND test_gpu_spectral_stack_hip
            "[session][json]")

        add_executable(test_observable_reduce_hip test_observable_reduce_gpu.cpp)
        target_include_directories(test_observable_reduce_hip PRIVATE
            ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
            ${CMAKE_SOURCE_DIR}/include
        )
        target_link_libraries(test_observable_reduce_hip PRIVATE
            Catch2::Catch2 openpfc hip::host MPI::MPI_CXX)
        target_compile_definitions(test_observable_reduce_hip PRIVATE
            OPENPFC_TEST_OBSERVABLE_HIP)
        add_test(NAME HIP_ObservableReduce COMMAND test_observable_reduce_hip)
    endif()

    message(STATUS "✅ HIP tests enabled")
elseif(OpenPFC_ENABLE_HIP)
    message(STATUS "⚠️  HIP tests disabled (HIP not found)")
else()
    message(STATUS "ℹ️  HIP tests disabled (OpenPFC_ENABLE_HIP=OFF)")
endif()

# SparseVector tests - build based on available backends
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    add_executable(test_sparsevector_cuda
        ${CMAKE_SOURCE_DIR}/tests/unit/operators/test_sparsevector_cuda.cpp
        ${CMAKE_SOURCE_DIR}/tests/unit/operators/test_sparsevector_typed.cpp)
    target_compile_definitions(test_sparsevector_cuda PRIVATE
        OPENPFC_TEST_TYPED_GATHER_CUDA)
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

    add_executable(test_etd1_apply_cuda test_etd1_apply.cpp)
    target_include_directories(test_etd1_apply_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_etd1_apply_cuda PRIVATE Catch2::Catch2 openpfc_gpu_kernels CUDA::cudart)
    target_compile_definitions(test_etd1_apply_cuda PRIVATE OPENPFC_TEST_ETD1_APPLY_CUDA)
    add_test(NAME CUDA_ETD1Apply COMMAND test_etd1_apply_cuda)
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
    add_executable(test_sparsevector_hip
        ${CMAKE_SOURCE_DIR}/tests/unit/operators/test_sparsevector_hip.cpp
        ${CMAKE_SOURCE_DIR}/tests/unit/operators/test_sparsevector_typed.cpp)
    target_compile_definitions(test_sparsevector_hip PRIVATE
        OPENPFC_TEST_TYPED_GATHER_HIP)
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

    add_executable(test_etd1_apply_hip test_etd1_apply.cpp)
    target_include_directories(test_etd1_apply_hip PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_etd1_apply_hip PRIVATE Catch2::Catch2 openpfc_hip_kernels hip::host)
    target_compile_definitions(test_etd1_apply_hip PRIVATE OPENPFC_TEST_ETD1_APPLY_HIP)
    add_test(NAME HIP_ETD1Apply COMMAND test_etd1_apply_hip)
endif()

# Grain tracking parity uses the ordinary runtime library, without spectral FFTs.
foreach(_grain_backend IN ITEMS cuda hip)
  if(_grain_backend STREQUAL "cuda" AND OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    set(_grain_runtime openpfc_gpu_kernels CUDA::cudart)
    set(_grain_definition OPENPFC_TEST_TRACKING_CUDA)
  elseif(_grain_backend STREQUAL "hip" AND OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
    set(_grain_runtime openpfc_hip_kernels hip::host)
    set(_grain_definition OPENPFC_TEST_TRACKING_HIP)
  else()
    continue()
  endif()
  add_executable(test_grain_tracking_${_grain_backend} test_grain_tracking.cpp)
  target_include_directories(test_grain_tracking_${_grain_backend} PRIVATE
      ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(test_grain_tracking_${_grain_backend} PRIVATE
      Catch2::Catch2 ${_grain_runtime})
  target_compile_definitions(test_grain_tracking_${_grain_backend} PRIVATE
      ${_grain_definition})
  target_compile_features(test_grain_tracking_${_grain_backend} PRIVATE cxx_std_20)
  add_test(NAME GrainTracking_${_grain_backend} COMMAND test_grain_tracking_${_grain_backend})
  set_tests_properties(GrainTracking_${_grain_backend} PROPERTIES
      TIMEOUT 120 SKIP_RETURN_CODE 4)
endforeach()
