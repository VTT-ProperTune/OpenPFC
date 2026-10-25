# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(polynomial_interpolation interpolation/model.cpp)
target_link_libraries(polynomial_interpolation PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(polynomial_interpolation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
