# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(mpi_timers mpi-timing/model.cpp)
target_link_libraries(mpi_timers PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(mpi_timers PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
