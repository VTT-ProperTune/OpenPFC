# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Kernel simulation layer unit tests
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simulation_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simulator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_time.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_model_fft_baseline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_model_comprehensive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_steppers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_euler_stepper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/state_concepts_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_solver_contract.cpp
)

add_subdirectory(steppers)
