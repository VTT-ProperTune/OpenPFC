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



add_executable(04_diffusion_model 04_diffusion_model.cpp)
target_link_libraries(04_diffusion_model PRIVATE OpenPFC::openpfc )

add_executable(05_simulator 05_simulator.cpp)
target_link_libraries(05_simulator PRIVATE OpenPFC::openpfc )





add_executable(10_ui_register_ic 10_ui_register_ic.cpp)
target_link_libraries(10_ui_register_ic PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json )

add_executable(11_write_results 11_write_results.cpp)
target_link_libraries(11_write_results PRIVATE OpenPFC::openpfc )

add_executable(12_cahn_hilliard 12_cahn_hilliard.cpp)
target_link_libraries(12_cahn_hilliard PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json )

add_executable(fft_kspace_helpers_example fft_kspace_helpers_example.cpp)
target_link_libraries(fft_kspace_helpers_example PRIVATE OpenPFC::openpfc )






add_executable(write_results write_results.cpp)
target_link_libraries(write_results PRIVATE OpenPFC::openpfc )

add_executable(diffusion_model_with_custom_initial_condition diffusion_model_with_custom_initial_condition.cpp)
target_link_libraries(diffusion_model_with_custom_initial_condition PRIVATE OpenPFC::openpfc )

add_executable(json_read json_read.cpp)
target_link_libraries(json_read PRIVATE nlohmann_json::nlohmann_json)





add_executable(17_custom_coordinate_system 17_custom_coordinate_system.cpp)
target_link_libraries(17_custom_coordinate_system PRIVATE OpenPFC::openpfc )

add_executable(18_sobel_edge_detection 18_sobel_edge_detection.cpp)
target_link_libraries(18_sobel_edge_detection PRIVATE OpenPFC::openpfc )



# Explicit Euler stepper examples (FD and spectral backends)





# Teaching chapters own their implementations; binary names remain stable.
add_subdirectory(coupling)
add_subdirectory(diagnostics)
add_subdirectory(differentiation)
add_subdirectory(fourier)
add_subdirectory(geometry)
add_subdirectory(heat)
add_subdirectory(initialization)
add_subdirectory(integration)
add_subdirectory(interpolation)
add_subdirectory(output)
add_subdirectory(parallel)
add_subdirectory(polynomials)
add_subdirectory(waves)

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
  add_test(NAME example-15-finite-difference-heat
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_lessons.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 15_finite_difference_heat)
  set_tests_properties(example-15-finite-difference-heat PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-15-finite-difference-heat-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_lessons.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 15_finite_difference_heat
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-15-finite-difference-heat-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-02-domain-decomposition
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 02_domain_decomposition)
  set_tests_properties(example-02-domain-decomposition PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-02-domain-decomposition-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 02_domain_decomposition
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-02-domain-decomposition-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-06-multi-index
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 06_multi_index)
  set_tests_properties(example-06-multi-index PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-07-array
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 07_array)
  set_tests_properties(example-07-array PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-08-discrete-fields
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 08_discrete_fields)
  set_tests_properties(example-08-discrete-fields PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-08-discrete-fields-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 08_discrete_fields
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-08-discrete-fields-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-03-parallel-fft
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 03_parallel_fft)
  set_tests_properties(example-03-parallel-fft PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-03-parallel-fft-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 03_parallel_fft
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-03-parallel-fft-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-09-parallel-fft-high-level
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 09_parallel_fft_high_level)
  set_tests_properties(example-09-parallel-fft-high-level PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-09-parallel-fft-high-level-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 09_parallel_fft_high_level
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-09-parallel-fft-high-level-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-cardinal-interpolation
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case cardinal_interpolation)
  set_tests_properties(example-cardinal-interpolation PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-periodic-collocation
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case periodic_collocation)
  set_tests_properties(example-periodic-collocation PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-periodic-collocation-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case periodic_collocation
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-periodic-collocation-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-spectral-smoothness
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case spectral_smoothness)
  set_tests_properties(example-spectral-smoothness PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-spectral-smoothness-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case spectral_smoothness
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-spectral-smoothness-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-world-helpers-example
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case world_helpers_example)
  set_tests_properties(example-world-helpers-example PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-world-query-helpers-example
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case world_query_helpers_example)
  set_tests_properties(example-world-query-helpers-example PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-16-strong-types-demo
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 16_strong_types_demo)
  set_tests_properties(example-16-strong-types-demo PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-world-strong-types-example
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case world_strong_types_example)
  set_tests_properties(example-world-strong-types-example PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-variable-transport
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_dynamics.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case variable_transport)
  set_tests_properties(example-variable-transport PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-variable-transport-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_dynamics.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case variable_transport
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-variable-transport-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-stability-regions
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_dynamics.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case stability_regions)
  set_tests_properties(example-stability-regions PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-polynomial-interpolation
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_polynomials.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case polynomial_interpolation)
  set_tests_properties(example-polynomial-interpolation PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-polynomial-nodal-polynomials
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_polynomials.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case polynomial_nodal_polynomials)
  set_tests_properties(example-polynomial-nodal-polynomials PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-polynomial-differentiation
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_polynomials.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case polynomial_differentiation)
  set_tests_properties(example-polynomial-differentiation PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-polynomial-regularity
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_polynomials.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case polynomial_regularity)
  set_tests_properties(example-polynomial-regularity PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-polynomial-integration
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_polynomials.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case polynomial_integration)
  set_tests_properties(example-polynomial-integration PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-polynomial-boundary-values
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_polynomials.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case polynomial_boundary_values)
  set_tests_properties(example-polynomial-boundary-values PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-polynomial-nonlinear-boundary
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_polynomials.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case polynomial_nonlinear_boundary)
  set_tests_properties(example-polynomial-nonlinear-boundary PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-21-adaptive-stepping
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 21_adaptive_stepping)
  set_tests_properties(example-21-adaptive-stepping PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-21-adaptive-stepping-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 21_adaptive_stepping
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-21-adaptive-stepping-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-24-adaptive-etd
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 24_adaptive_etd)
  set_tests_properties(example-24-adaptive-etd PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-24-adaptive-etd-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 24_adaptive_etd
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-24-adaptive-etd-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-time
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case time)
  set_tests_properties(example-time PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-22-external-coupling
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 22_external_coupling)
  set_tests_properties(example-22-external-coupling PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-22-external-coupling-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 22_external_coupling
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-22-external-coupling-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-runtime-coupled-fields
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case runtime_coupled_fields)
  set_tests_properties(example-runtime-coupled-fields PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-runtime-coupled-fields-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case runtime_coupled_fields
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-runtime-coupled-fields-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-checkpoint-restart
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case checkpoint_restart)
  set_tests_properties(example-checkpoint-restart PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-checkpoint-restart-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case checkpoint_restart
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-checkpoint-restart-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-mpi-worker
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case mpi_worker)
  set_tests_properties(example-mpi-worker PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-mpi-worker-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case mpi_worker
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-mpi-worker-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-profiling-timer-report
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case profiling_timer_report)
  set_tests_properties(example-profiling-timer-report PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-profiling-timer-report-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case profiling_timer_report
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-profiling-timer-report-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
  add_test(NAME example-first-domain
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_everyday.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case first_domain)
  set_tests_properties(example-first-domain PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-14-custom-field-initializer
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_everyday.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 14_custom_field_initializer)
  set_tests_properties(example-14-custom-field-initializer PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-14-custom-field-initializer-mpi
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_everyday.py" --bin-dir "${CMAKE_BINARY_DIR}/examples" --output "${PROJECT_SOURCE_DIR}/results/example-checks" --case 14_custom_field_initializer
                     --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-14-custom-field-initializer-mpi PROPERTIES PROCESSORS 2 TIMEOUT 360
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
