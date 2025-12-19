# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Field modifier unit tests
target_sources(openpfc-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_field_modifier.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_ic_constant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_ic_single_seed.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_ic_random_seeds.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_ic_seed_grid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_ic_file_reader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_boundary_conditions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_bc_moving.cpp
)
