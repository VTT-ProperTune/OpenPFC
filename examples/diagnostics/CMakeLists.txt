# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(mpi_timers mpi-timing/model.cpp)
target_link_libraries(mpi_timers PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(mpi_timers PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(profiling_timer_report profiling/model.cpp)
target_link_libraries(profiling_timer_report PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(profiling_timer_report PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(23_halo_microtiming halo-frames/model.cpp)
target_link_libraries(23_halo_microtiming PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(23_halo_microtiming PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
if(OpenPFC_ENABLE_CUDA)
  target_compile_definitions(23_halo_microtiming PRIVATE OpenPFC_ENABLE_CUDA)
endif()
if(OpenPFC_ENABLE_HIP)
  target_compile_definitions(23_halo_microtiming PRIVATE OpenPFC_ENABLE_HIP)
endif()
