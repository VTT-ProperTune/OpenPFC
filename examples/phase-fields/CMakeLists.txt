# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(12_cahn_hilliard cahn-hilliard/model.cpp)
target_link_libraries(12_cahn_hilliard PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(12_cahn_hilliard PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_subdirectory(alias-free-cubic)
