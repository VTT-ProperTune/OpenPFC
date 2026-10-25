# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(gamma_contour complex-contour/model.cpp)
target_link_libraries(gamma_contour PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(gamma_contour PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
