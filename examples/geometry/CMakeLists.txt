# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(02_domain_decomposition decomposition/model.cpp)
target_link_libraries(02_domain_decomposition PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(02_domain_decomposition PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
