# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(19_explicit_stepper_fd finite-differences/model.cpp)
target_link_libraries(19_explicit_stepper_fd PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(19_explicit_stepper_fd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
