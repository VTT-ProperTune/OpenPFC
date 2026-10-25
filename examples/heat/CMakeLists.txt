# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(19_explicit_stepper_fd finite-differences/model.cpp)
target_link_libraries(19_explicit_stepper_fd PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(19_explicit_stepper_fd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(20_explicit_stepper_spectral fourier/model.cpp)
target_link_libraries(20_explicit_stepper_spectral PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(20_explicit_stepper_spectral PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(heat_exponential exponential/model.cpp)
target_link_libraries(heat_exponential PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(heat_exponential PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(15_finite_difference_heat separated-halos/model.cpp)
target_link_libraries(15_finite_difference_heat PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(15_finite_difference_heat PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(04_diffusion_model implicit-euler/model.cpp)
target_link_libraries(04_diffusion_model PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(04_diffusion_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(05_simulator driver-hooks/model.cpp)
target_link_libraries(05_simulator PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(05_simulator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
