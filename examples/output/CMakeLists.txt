# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(checkpoint_restart checkpoint-restart/model.cpp)
target_link_libraries(checkpoint_restart PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(checkpoint_restart PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
