# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Kernel field operations unit tests
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_composite_gradient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fd_apply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fd_brick_laplacian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fd_gradient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stencil_apply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_padded_brick.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_brick_iteration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_state_access.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_state_concepts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/field_from_subdomain_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dirichlet_ghosts.cpp
)
if(OpenPFC_ENABLE_HEFFTE)
  target_sources(openpfc-tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/test_field_operations.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_field_comprehensive.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_spectral_gradient.cpp
  )
endif()
