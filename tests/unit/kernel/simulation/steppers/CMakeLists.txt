# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

target_sources(openpfc-tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_butcher_tableau.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_embedded_rk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_etd1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_imex_stage_composition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stage_workspace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_integrator_method.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_integrator_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stepper_validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stepper_concept.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_method_composition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_step_attempt.cpp
)

# AC-facing CTest name: `ctest -R test_integrator_base` (shared openpfc-tests binary).
add_test(NAME test_integrator_base
         COMMAND $<TARGET_FILE:openpfc-tests> "[integrator_base]")
set_tests_properties(test_integrator_base PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

# AC-facing CTest name: `ctest -R '^method_composition$'`.
add_test(NAME method_composition
         COMMAND $<TARGET_FILE:openpfc-tests> "[method_composition]")
set_tests_properties(method_composition PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

# AC-facing CTest name: `ctest -R test_imex_stage_composition` (shared openpfc-tests binary).
add_test(NAME test_imex_stage_composition
         COMMAND $<TARGET_FILE:openpfc-tests> "[imex_stage_composition]")
set_tests_properties(test_imex_stage_composition PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

# AC-facing CTest name: `ctest -R '^test_step_attempt$'` (shared openpfc-tests binary).
add_test(NAME test_step_attempt
         COMMAND $<TARGET_FILE:openpfc-tests> "[step_attempt]")
set_tests_properties(test_step_attempt PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
