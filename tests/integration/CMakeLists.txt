# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Integration tests
target_sources(openpfc-tests PRIVATE
  test_multifield_stepper.cpp
  scenarios/parallel_scaling/test_global_coverage.cpp
  scenarios/parallel_scaling/test_halo_patterns.cpp
  scenarios/parallel_scaling/test_halo_exchange_driver.cpp
  scenarios/parallel_scaling/test_full_padded_device_halo.cpp
  scenarios/parallel_scaling/test_full_padded_device_halo_hip.cpp
  scenarios/parallel_scaling/test_device_halo_element_types.cpp
  scenarios/parallel_scaling/test_full_halo_fd6.cpp
  scenarios/parallel_scaling/test_padded_device_halo_self_wrap.cpp
  scenarios/parallel_scaling/test_comm_halo_exchange.cpp
  scenarios/parallel_scaling/test_comm_halo_exchange_modes.cpp
  scenarios/parallel_scaling/test_comm_halo_exchange_gpu.cpp
  scenarios/parallel_scaling/test_comm_sparse_exchange.cpp
  scenarios/parallel_scaling/test_comm_sparse_exchange_gpu.cpp
  scenarios/parallel_scaling/test_stage_preparation.cpp
  scenarios/parallel_scaling/test_sbp_diffusion.cpp
  scenarios/parallel_scaling/test_halo_direction_set.cpp
  scenarios/parallel_scaling/test_sparse_halo_exchange.cpp
  scenarios/parallel_scaling/test_fd_heat_mpi.cpp
  scenarios/parallel_scaling/test_fd_highorder_mpi.cpp
  scenarios/parallel_scaling/test_fd_xy_mpi.cpp
  scenarios/parallel_scaling/neighbor_round_trip_mpi.cpp
  scenarios/field_operations/test_state_contracts.cpp
  scenarios/field_operations/test_heat3d_state_access.cpp
  scenarios/field_operations/test_wave2d_state_access.cpp
  scenarios/time_integration/test_rk3_convergence.cpp
  scenarios/time_integration/test_field_accessor_concept.cpp
  scenarios/time_integration/test_loop_policy.cpp
)

if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  add_executable(test_central_derivatives_cuda central_derivatives.cu)
  target_include_directories(test_central_derivatives_cuda PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(test_central_derivatives_cuda PRIVATE OpenPFC Catch2::Catch2WithMain CUDA::cudart)
  set_target_properties(test_central_derivatives_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME CUDA_CentralDerivatives COMMAND test_central_derivatives_cuda)
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(test_central_derivatives_hip central_derivatives.hip)
  target_include_directories(test_central_derivatives_hip PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(test_central_derivatives_hip PRIVATE OpenPFC Catch2::Catch2WithMain hip::host)
  set_target_properties(test_central_derivatives_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
  add_test(NAME HIP_CentralDerivatives COMMAND test_central_derivatives_hip)
endif()
# Includes the HeFFTe backend from a .hip TU, so the HIP compiler (Clang)
# compiles FFT_Impl; regression for issue #373.
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  add_executable(test_fft_device_interface_hip fft_device_interface.hip)
  target_link_libraries(test_fft_device_interface_hip PRIVATE OpenPFC hip::host)
  set_target_properties(test_fft_device_interface_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
  add_test(NAME HIP_FFTDeviceInterface COMMAND test_fft_device_interface_hip)
endif()
if(OpenPFC_ENABLE_HEFFTE)
  target_sources(openpfc-tests PRIVATE
    test_ui_validation.cpp
    scenarios/parallel_scaling/test_domain_decomposition.cpp
    scenarios/field_operations/test_field_modifiers_integration.cpp
    scenarios/gpu_validation/test_gpu_backend_instantiation.cpp
    scenarios/gpu_validation/test_cuda_roundtrip.cpp
    scenarios/gpu_validation/test_hip_roundtrip.cpp
    scenarios/gpu_validation/test_cuda_vs_cpu_laplacian.cpp
    scenarios/gpu_validation/test_cuda_vs_cpu_laplacian_mpi.cpp
    scenarios/gpu_validation/test_hip_vs_cpu_laplacian.cpp
    scenarios/gpu_validation/test_hip_vs_cpu_laplacian_mpi.cpp
    scenarios/time_integration/test_stepper_contract_equivalence.cpp
)
endif()

# Device Faces/Full pack kernels live in `openpfc_gpu_kernels`. The test
# executable still needs CUDA::cudart and those kernels because
# `HaloExchange<CUDASpace>` calls `launch_padded_pack_face` from headers.
#
# `test_fd_gradient_device.cu` instantiates the templated
# `for_each_interior_device_kernel` for a test-local model and grads
# aggregate; the kernel must be compiled by `nvcc`, so the test source is
# itself a `.cu` file added to the test target only on CUDA builds.
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  target_sources(openpfc-tests PRIVATE
    scenarios/gpu_validation/test_rotation.cu
    scenarios/gpu_validation/test_fd_gradient_device.cu
    scenarios/gpu_validation/test_face_current.cu
    scenarios/gpu_validation/test_observable_reduce.cu
    scenarios/gpu_validation/test_composite_gradient_pod_size.cu
    scenarios/gpu_validation/test_stage_fields.cu
    scenarios/gpu_validation/test_multi_field_device.cu
  )
  target_link_libraries(openpfc-tests PRIVATE CUDA::cudart openpfc_gpu_kernels)
endif()


# HIP integration tests follow the same pattern as CUDA tests.
# test_fd_gradient_device_hip.hip and test_multi_field_device.hip instantiate
# the templated for_each_interior_device_kernel for test-local models; the
# kernel must be compiled by hipcc, so the sources are added only on HIP
# builds. test_composite_gradient_pod_size_hip.hip is the HIP twin of the
# CUDA POD-layout lock.
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  target_sources(openpfc-tests PRIVATE
    scenarios/gpu_validation/test_rotation.hip
    scenarios/gpu_validation/test_fd_gradient_device_hip.hip
    scenarios/gpu_validation/test_face_current.hip
    scenarios/gpu_validation/test_observable_reduce.hip
    scenarios/gpu_validation/test_composite_gradient_pod_size_hip.hip
    scenarios/gpu_validation/test_stage_fields.hip
    scenarios/gpu_validation/test_multi_field_device.hip
    scenarios/parallel_scaling/test_padded_device_halo_self_wrap_hip.hip
    scenarios/parallel_scaling/test_full_halo_fd6.hip
  )
  target_link_libraries(openpfc-tests PRIVATE hip::host)
endif()

if(OpenPFC_RUN_MPI_SUITES)
  foreach(_ranks 1 2 4 8)
    if(OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0 AND
       _ranks GREATER OpenPFC_MPI_TEST_MAX_WORLD_SIZE)
      continue()
    endif()
    add_test(NAME full_halo_fd6_${_ranks}ranks
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_ranks}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:openpfc-tests> "[full_halo_fd6]"
        ${MPIEXEC_POSTFLAGS})
    set_tests_properties(full_halo_fd6_${_ranks}ranks PROPERTIES
      TIMEOUT 120 RUN_SERIAL TRUE PROCESSORS ${_ranks})
  endforeach()
endif()

add_subdirectory(rk_coupled)
add_subdirectory(rk_nfield)
