# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(22_external_coupling host-orchestrator/model.cpp)
target_link_libraries(22_external_coupling PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(22_external_coupling PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(runtime_coupled_fields runtime-fields/model.cpp)
target_link_libraries(runtime_coupled_fields PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(runtime_coupled_fields PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_subdirectory(fixed-tuples)

add_executable(symbolic_coupled_rhs symbolic-rhs/model.cpp)
target_link_libraries(symbolic_coupled_rhs PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(symbolic_coupled_rhs PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
