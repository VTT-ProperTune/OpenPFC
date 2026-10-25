# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(periodic_differentiation periodic/model.cpp)
target_link_libraries(periodic_differentiation PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(periodic_differentiation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(periodic_collocation collocation/model.cpp)
target_link_libraries(periodic_collocation PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(periodic_collocation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(spectral_smoothness smoothness/model.cpp)
target_link_libraries(spectral_smoothness PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(spectral_smoothness PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
