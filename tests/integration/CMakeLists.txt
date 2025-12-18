# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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
