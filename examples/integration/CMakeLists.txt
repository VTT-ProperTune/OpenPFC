# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(stability_regions stability/model.cpp)
target_link_libraries(stability_regions PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(stability_regions PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(21_adaptive_stepping adaptive-rk/model.cpp)
target_link_libraries(21_adaptive_stepping PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(21_adaptive_stepping PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(24_adaptive_etd adaptive-etd/model.cpp)
target_link_libraries(24_adaptive_etd PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(24_adaptive_etd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(time output-clock/model.cpp)
target_link_libraries(time PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(time PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
