# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_subdirectory(microelasticity)
add_subdirectory(finite_strain_fft)

target_sources(openpfc-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_krylov.cpp)
target_sources(openpfc-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_newton.cpp)
target_sources(openpfc-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_multigrid.cpp)
