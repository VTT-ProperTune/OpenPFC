# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(mpi_worker hello/model.cpp)
target_link_libraries(mpi_worker PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(mpi_worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(mpi_worker_inside_class session/model.cpp)
target_link_libraries(mpi_worker_inside_class PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(mpi_worker_inside_class PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
