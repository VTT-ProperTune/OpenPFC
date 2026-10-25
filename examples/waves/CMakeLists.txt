# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(variable_transport variable-transport/model.cpp)
target_link_libraries(variable_transport PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(variable_transport PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(chebyshev_wave_line chebyshev-line/model.cpp)
target_link_libraries(chebyshev_wave_line PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(chebyshev_wave_line PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(chebyshev_wave_square chebyshev-square/model.cpp)
target_link_libraries(chebyshev_wave_square PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(chebyshev_wave_square PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
