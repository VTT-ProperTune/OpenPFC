# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(cardinal_interpolation cardinal/model.cpp)
target_link_libraries(cardinal_interpolation PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(cardinal_interpolation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
