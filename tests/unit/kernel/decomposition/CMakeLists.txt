# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Kernel decomposition layer unit tests
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_decomposition_comprehensive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_decomposition_lifetime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_vector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_vector_comprehensive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_vector_exchange.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_vector_neighbor_exchange.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_halo_pattern.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_halo_face_layout.cpp
)
