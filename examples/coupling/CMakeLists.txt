# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(22_external_coupling host-orchestrator/model.cpp)
target_link_libraries(22_external_coupling PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(22_external_coupling PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
