# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# FFT Backend Benchmark (requires MPI)
if(NOT TARGET grain_remapping_example)
  add_executable(grain_remapping_example grain_remapping.cpp)
  target_link_libraries(grain_remapping_example PRIVATE OpenPFC::openpfc)
endif()

add_executable(fft_backend_benchmark fft_backend_benchmark.cpp)

target_link_libraries(fft_backend_benchmark PRIVATE OpenPFC::openpfc )
if(OpenPFC_ENABLE_CUDA)
  target_compile_definitions(fft_backend_benchmark PRIVATE OpenPFC_ENABLE_CUDA)
endif()
if(OpenPFC_ENABLE_HIP)
  target_compile_definitions(fft_backend_benchmark PRIVATE OpenPFC_ENABLE_HIP)
endif()
if(OpenPFC_ENABLE_HIP_SPECTRAL)
  target_compile_definitions(fft_backend_benchmark PRIVATE OpenPFC_ENABLE_HIP_SPECTRAL)
endif()

# add_executable(01_hello_world 01_hello_world.cpp)
# target_link_libraries(01_hello_world PRIVATE OpenPFC::openpfc)

add_executable(02_domain_decomposition 02_domain_decomposition.cpp)
target_link_libraries(02_domain_decomposition PRIVATE OpenPFC::openpfc )

add_executable(03_parallel_fft 03_parallel_fft.cpp)
target_link_libraries(03_parallel_fft PRIVATE OpenPFC::openpfc )

add_executable(04_diffusion_model 04_diffusion_model.cpp)
target_link_libraries(04_diffusion_model PRIVATE OpenPFC::openpfc )

add_executable(05_simulator 05_simulator.cpp)
target_link_libraries(05_simulator PRIVATE OpenPFC::openpfc )

add_executable(06_multi_index 06_multi_index.cpp)
target_link_libraries(06_multi_index PRIVATE OpenPFC::openpfc )

add_executable(07_array 07_array.cpp)
target_link_libraries(07_array PRIVATE OpenPFC::openpfc )

add_executable(08_discrete_fields 08_discrete_fields.cpp)
target_link_libraries(08_discrete_fields PRIVATE OpenPFC::openpfc )

add_executable(09_parallel_fft_high_level 09_parallel_fft_high_level.cpp)
target_link_libraries(09_parallel_fft_high_level PRIVATE OpenPFC::openpfc )

add_executable(10_ui_register_ic 10_ui_register_ic.cpp)
target_link_libraries(10_ui_register_ic PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json )

add_executable(11_write_results 11_write_results.cpp)
target_link_libraries(11_write_results PRIVATE OpenPFC::openpfc )

add_executable(12_cahn_hilliard 12_cahn_hilliard.cpp)
target_link_libraries(12_cahn_hilliard PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json )

add_executable(fft_kspace_helpers_example fft_kspace_helpers_example.cpp)
target_link_libraries(fft_kspace_helpers_example PRIVATE OpenPFC::openpfc )

add_executable(mpi_worker mpi_worker.cpp)
target_link_libraries(mpi_worker PRIVATE OpenPFC::openpfc )

add_executable(mpi_worker_inside_class mpi_worker_inside_class.cpp)
target_link_libraries(mpi_worker_inside_class PRIVATE OpenPFC::openpfc )

add_executable(mpi_timers mpi_timers.cpp)
target_link_libraries(mpi_timers PRIVATE OpenPFC::openpfc )

add_executable(profiling_timer_report profiling_timer_report.cpp)
target_link_libraries(profiling_timer_report PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json )

add_executable(time time.cpp)
target_link_libraries(time PRIVATE OpenPFC::openpfc )

add_executable(write_results write_results.cpp)
target_link_libraries(write_results PRIVATE OpenPFC::openpfc )

add_executable(diffusion_model_with_custom_initial_condition diffusion_model_with_custom_initial_condition.cpp)
target_link_libraries(diffusion_model_with_custom_initial_condition PRIVATE OpenPFC::openpfc )

add_executable(json_read json_read.cpp)
target_link_libraries(json_read PRIVATE nlohmann_json::nlohmann_json)

add_executable(world_helpers_example world_helpers_example.cpp)
target_link_libraries(world_helpers_example PRIVATE OpenPFC::openpfc )

add_executable(14_custom_field_initializer 14_custom_field_initializer.cpp)
target_link_libraries(14_custom_field_initializer PRIVATE OpenPFC::openpfc )

add_executable(15_finite_difference_heat 15_finite_difference_heat.cpp)
target_link_libraries(15_finite_difference_heat PRIVATE OpenPFC::openpfc )

add_executable(16_strong_types_demo 16_strong_types_demo.cpp)
target_link_libraries(16_strong_types_demo PRIVATE OpenPFC::openpfc )

add_executable(17_custom_coordinate_system 17_custom_coordinate_system.cpp)
target_link_libraries(17_custom_coordinate_system PRIVATE OpenPFC::openpfc )

add_executable(18_sobel_edge_detection 18_sobel_edge_detection.cpp)
target_link_libraries(18_sobel_edge_detection PRIVATE OpenPFC::openpfc )

add_executable(world_strong_types_example world_strong_types_example.cpp)
target_link_libraries(world_strong_types_example PRIVATE OpenPFC::openpfc )

add_executable(world_query_helpers_example world_query_helpers_example.cpp)
target_link_libraries(world_query_helpers_example PRIVATE OpenPFC::openpfc )

# Explicit Euler stepper examples (FD and spectral backends)
add_executable(19_explicit_stepper_fd 19_explicit_stepper_fd.cpp)
target_link_libraries(19_explicit_stepper_fd PRIVATE OpenPFC::openpfc )

add_executable(20_explicit_stepper_spectral 20_explicit_stepper_spectral.cpp)
target_link_libraries(20_explicit_stepper_spectral PRIVATE OpenPFC::openpfc )

add_executable(21_adaptive_stepping 21_adaptive_stepping.cpp)
target_link_libraries(21_adaptive_stepping PRIVATE OpenPFC::openpfc )

add_executable(24_adaptive_etd 24_adaptive_etd.cpp)
target_link_libraries(24_adaptive_etd PRIVATE OpenPFC::openpfc )

add_executable(22_external_coupling 22_external_coupling.cpp)
target_link_libraries(22_external_coupling PRIVATE OpenPFC::openpfc )

add_executable(23_halo_microtiming 23_halo_microtiming.cpp)
target_link_libraries(23_halo_microtiming PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
if(OpenPFC_ENABLE_CUDA)
  target_compile_definitions(23_halo_microtiming PRIVATE OpenPFC_ENABLE_CUDA)
endif()
if(OpenPFC_ENABLE_HIP)
  target_compile_definitions(23_halo_microtiming PRIVATE OpenPFC_ENABLE_HIP)
endif()
if(OpenPFC_BUILD_TESTS AND MPIEXEC_EXECUTABLE)
  add_test(NAME halo-microtiming-host-2rank
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                   $<TARGET_FILE:23_halo_microtiming> --nx 16 --iters 4 --warmup 1
                   --output ${CMAKE_CURRENT_BINARY_DIR}/halo_microtiming_smoke.json)
  set_tests_properties(halo-microtiming-host-2rank PROPERTIES
                       ENVIRONMENT
                       "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
endif()

# Issue #21 research prototype under examples/, not an apps/ executable.
add_subdirectory(ns2d_vorticity)

# Issue #229 verification driver under examples/, not an apps/ executable.
add_subdirectory(taylor_green3d)

if(NOT TARGET coupled_fields_example)
  add_subdirectory(coupled_fields)
endif()
