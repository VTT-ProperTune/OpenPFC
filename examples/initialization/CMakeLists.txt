# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(14_custom_field_initializer patterns/model.cpp)
target_link_libraries(14_custom_field_initializer PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(14_custom_field_initializer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(10_ui_register_ic catalog/model.cpp)
target_link_libraries(10_ui_register_ic PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(10_ui_register_ic PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
