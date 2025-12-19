# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# FFT Backend Benchmark (requires MPI)
add_executable(fft_backend_benchmark fft_backend_benchmark.cpp)
set(_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_heffte_tgt heffte)
endif()

target_link_libraries(fft_backend_benchmark PRIVATE OpenPFC ${_heffte_tgt})
if(OpenPFC_ENABLE_CUDA)
  target_compile_definitions(fft_backend_benchmark PRIVATE OpenPFC_ENABLE_CUDA)
endif()

# add_executable(01_hello_world 01_hello_world.cpp)
# target_link_libraries(01_hello_world PRIVATE OpenPFC)

add_executable(02_domain_decomposition 02_domain_decomposition.cpp)
target_link_libraries(02_domain_decomposition PRIVATE OpenPFC ${_heffte_tgt})

add_executable(03_parallel_fft 03_parallel_fft.cpp)
target_link_libraries(03_parallel_fft PRIVATE OpenPFC ${_heffte_tgt})

add_executable(04_diffusion_model 04_diffusion_model.cpp)
target_link_libraries(04_diffusion_model PRIVATE OpenPFC ${_heffte_tgt})

add_executable(05_simulator 05_simulator.cpp)
target_link_libraries(05_simulator PRIVATE OpenPFC ${_heffte_tgt})

add_executable(06_multi_index 06_multi_index.cpp)
target_link_libraries(06_multi_index PRIVATE OpenPFC ${_heffte_tgt})

add_executable(07_array 07_array.cpp)
target_link_libraries(07_array PRIVATE OpenPFC ${_heffte_tgt})

add_executable(08_discrete_fields 08_discrete_fields.cpp)
target_link_libraries(08_discrete_fields PRIVATE OpenPFC ${_heffte_tgt})

add_executable(09_parallel_fft_high_level 09_parallel_fft_high_level.cpp)
target_link_libraries(09_parallel_fft_high_level PRIVATE OpenPFC ${_heffte_tgt})

add_executable(10_ui_register_ic 10_ui_register_ic.cpp)
target_link_libraries(10_ui_register_ic PRIVATE OpenPFC nlohmann_json::nlohmann_json ${_heffte_tgt})

add_executable(11_write_results 11_write_results.cpp)
target_link_libraries(11_write_results PRIVATE OpenPFC ${_heffte_tgt})

add_executable(12_cahn_hilliard 12_cahn_hilliard.cpp)
target_link_libraries(12_cahn_hilliard PRIVATE OpenPFC nlohmann_json::nlohmann_json ${_heffte_tgt})

add_executable(fft_kspace_helpers_example fft_kspace_helpers_example.cpp)
target_link_libraries(fft_kspace_helpers_example PRIVATE OpenPFC ${_heffte_tgt})

add_executable(mpi_worker mpi_worker.cpp)
target_link_libraries(mpi_worker PRIVATE OpenPFC ${_heffte_tgt})

add_executable(mpi_worker_inside_class mpi_worker_inside_class.cpp)
target_link_libraries(mpi_worker_inside_class PRIVATE OpenPFC ${_heffte_tgt})

add_executable(mpi_timers mpi_timers.cpp)
target_link_libraries(mpi_timers PRIVATE OpenPFC ${_heffte_tgt})

add_executable(time time.cpp)
target_link_libraries(time PRIVATE OpenPFC ${_heffte_tgt})

add_executable(write_results write_results.cpp)
target_link_libraries(write_results PRIVATE OpenPFC ${_heffte_tgt})

add_executable(diffusion_model_with_custom_initial_condition diffusion_model_with_custom_initial_condition.cpp)
target_link_libraries(diffusion_model_with_custom_initial_condition PRIVATE OpenPFC ${_heffte_tgt})

add_executable(json_read json_read.cpp)
target_link_libraries(json_read PRIVATE nlohmann_json::nlohmann_json)

add_executable(world_helpers_example world_helpers_example.cpp)
target_link_libraries(world_helpers_example PRIVATE OpenPFC ${_heffte_tgt})

add_executable(14_custom_field_initializer 14_custom_field_initializer.cpp)
target_link_libraries(14_custom_field_initializer PRIVATE OpenPFC ${_heffte_tgt})

add_executable(16_strong_types_demo 16_strong_types_demo.cpp)
target_link_libraries(16_strong_types_demo PRIVATE OpenPFC ${_heffte_tgt})

add_executable(17_custom_coordinate_system 17_custom_coordinate_system.cpp)
target_link_libraries(17_custom_coordinate_system PRIVATE OpenPFC ${_heffte_tgt})

add_executable(world_strong_types_example world_strong_types_example.cpp)
target_link_libraries(world_strong_types_example PRIVATE OpenPFC ${_heffte_tgt})
