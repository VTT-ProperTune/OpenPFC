# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# FFT Backend Benchmark (requires MPI)
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


add_executable(21_adaptive_stepping 21_adaptive_stepping.cpp)
target_link_libraries(21_adaptive_stepping PRIVATE OpenPFC::openpfc )

add_executable(24_adaptive_etd 24_adaptive_etd.cpp)
target_link_libraries(24_adaptive_etd PRIVATE OpenPFC::openpfc )

add_executable(22_external_coupling 22_external_coupling.cpp)
target_link_libraries(22_external_coupling PRIVATE OpenPFC::openpfc )

# Teaching chapters own their implementations; binary names remain stable.
add_subdirectory(differentiation)
add_subdirectory(heat)

if(OpenPFC_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME example-periodic-differentiation
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_lessons.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case periodic_differentiation)
  set_tests_properties(example-periodic-differentiation PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-periodic-differentiation-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_lessons.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case periodic_differentiation
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-periodic-differentiation-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-19-explicit-stepper-fd
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_lessons.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 19_explicit_stepper_fd)
  set_tests_properties(example-19-explicit-stepper-fd PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-19-explicit-stepper-fd-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_lessons.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 19_explicit_stepper_fd
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-19-explicit-stepper-fd-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-20-explicit-stepper-spectral
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_lessons.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 20_explicit_stepper_spectral)
  set_tests_properties(example-20-explicit-stepper-spectral PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-20-explicit-stepper-spectral-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_lessons.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 20_explicit_stepper_spectral
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-20-explicit-stepper-spectral-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-heat-exponential
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_lessons.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case heat_exponential)
  set_tests_properties(example-heat-exponential PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-heat-exponential-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_lessons.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case heat_exponential
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-heat-exponential-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
endif()

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

# Checked nonlinear Fourier/ETD2 teaching model and independent verifier.
add_executable(nonlinear_spectral_course nonlinear_spectral/allen_cahn.cpp)
target_link_libraries(nonlinear_spectral_course PRIVATE OpenPFC::openpfc)
add_subdirectory(nonlinear_spectral)
