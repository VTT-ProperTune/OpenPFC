# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_history.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_saint_venant_kirchhoff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simo_j2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_crystal_plasticity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_constitutive_evaluation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_quaternion_orientation.cpp
)

if(OpenPFC_ENABLE_HEFFTE)
  target_sources(openpfc-tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/test_displacement.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_equilibrium.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_crystal_cell.cpp
  )
endif()

