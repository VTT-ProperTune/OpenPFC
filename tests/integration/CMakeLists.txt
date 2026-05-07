# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Integration tests
target_sources(openpfc-tests PRIVATE
  test_ui_validation.cpp
  test_diffusion_integration.cpp
  scenarios/complete_simulation/test_complete_diffusion_simulation.cpp
  scenarios/complete_simulation/test_mass_conservation.cpp
  scenarios/parallel_scaling/test_domain_decomposition.cpp
  scenarios/parallel_scaling/test_global_coverage.cpp
  scenarios/parallel_scaling/test_halo_patterns.cpp
  scenarios/parallel_scaling/test_halo_exchange_driver.cpp
  scenarios/parallel_scaling/test_padded_halo_exchange.cpp
  scenarios/parallel_scaling/test_full_padded_device_halo.cpp
  scenarios/parallel_scaling/test_fd_heat_mpi.cpp
  scenarios/parallel_scaling/test_fd_highorder_mpi.cpp
  scenarios/parallel_scaling/test_fd_xy_mpi.cpp
  scenarios/io_workflows/test_vtk_roundtrip.cpp
  scenarios/field_operations/test_field_modifiers_integration.cpp
  scenarios/gpu_validation/test_cuda_vs_cpu.cpp
  scenarios/gpu_validation/test_gpu_backend_instantiation.cpp
  scenarios/gpu_validation/test_cuda_roundtrip.cpp
  scenarios/gpu_validation/test_cuda_vs_cpu_laplacian.cpp
  scenarios/gpu_validation/test_cuda_vs_cpu_laplacian_mpi.cpp
  scenarios/convergence_studies/test_temporal_convergence.cpp
  scenarios/convergence_studies/test_center_peak_decay.cpp
  scenarios/convergence_studies/test_monotonic_decay.cpp
)

# `FullPaddedDeviceHalo` calls `pfc::cuda::detail::launch_padded_pack_face` /
# `launch_padded_unpack_face`, which today live in
# `src/openpfc/runtime/cuda/padded_halo_faces.cu` and are linked **per-app**
# (e.g. into `kobayashi_fd_cuda`) rather than into `libopenpfc.so`. The test
# executable consumes those symbols directly, so we add the .cu source to
# `openpfc-tests` when CUDA is enabled and ensure CUDA::cudart is linked.
#
# Likewise, `test_fd_gradient_device.cu` instantiates the templated
# `for_each_interior_device_kernel` for a test-local model and grads
# aggregate; the kernel must be compiled by `nvcc`, so the test source is
# itself a `.cu` file added to the test target only on CUDA builds.
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  target_sources(openpfc-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/openpfc/runtime/cuda/padded_halo_faces.cu
    scenarios/gpu_validation/test_fd_gradient_device.cu
  )
  target_link_libraries(openpfc-tests PRIVATE CUDA::cudart)
endif()
