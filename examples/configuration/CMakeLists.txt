# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(json_read session/model.cpp)
target_link_libraries(json_read PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(json_read PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
