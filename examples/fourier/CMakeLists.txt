# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(03_parallel_fft round-trip/model.cpp)
target_link_libraries(03_parallel_fft PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(03_parallel_fft PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(09_parallel_fft_high_level field-layout/model.cpp)
target_link_libraries(09_parallel_fft_high_level PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(09_parallel_fft_high_level PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(harmonic_oscillator harmonic-oscillator/model.cpp)
target_link_libraries(harmonic_oscillator PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(harmonic_oscillator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(mathieu_spectrum mathieu/model.cpp)
target_link_libraries(mathieu_spectrum PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(mathieu_spectrum PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(fft_kspace_helpers_example symbols/model.cpp)
target_link_libraries(fft_kspace_helpers_example PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(fft_kspace_helpers_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(fft_backend_benchmark backend-buffers/model.cpp)
target_link_libraries(fft_backend_benchmark PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(fft_backend_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
