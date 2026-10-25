# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(02_domain_decomposition decomposition/model.cpp)
target_link_libraries(02_domain_decomposition PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(02_domain_decomposition PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(06_multi_index index-boxes/model.cpp)
target_link_libraries(06_multi_index PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(06_multi_index PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(07_array field-storage/model.cpp)
target_link_libraries(07_array PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(07_array PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(08_discrete_fields coordinate-fields/model.cpp)
target_link_libraries(08_discrete_fields PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(08_discrete_fields PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(world_helpers_example grid-spacing/model.cpp)
target_link_libraries(world_helpers_example PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(world_helpers_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(world_query_helpers_example domain-queries/model.cpp)
target_link_libraries(world_query_helpers_example PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(world_query_helpers_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(16_strong_types_demo strong-arguments/model.cpp)
target_link_libraries(16_strong_types_demo PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(16_strong_types_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(world_strong_types_example anisotropic-grid/model.cpp)
target_link_libraries(world_strong_types_example PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(world_strong_types_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(first_domain first-domain/model.cpp)
target_link_libraries(first_domain PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(first_domain PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(17_custom_coordinate_system polar-patterns/model.cpp)
target_link_libraries(17_custom_coordinate_system PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(17_custom_coordinate_system PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
