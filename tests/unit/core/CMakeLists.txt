# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Core unit tests
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_world.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_box3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_field.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_arraynd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_discrete_field.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_index.cpp
)
