# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

target_sources(openpfc-tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_butcher_tableau.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_embedded_rk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stage_workspace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_integrator_method.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_integrator_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stepper_validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stepper_concept.cpp
)

# AC-facing CTest name: `ctest -R test_integrator_base` (shared openpfc-tests binary).
add_test(NAME test_integrator_base
         COMMAND $<TARGET_FILE:openpfc-tests> "[integrator_base]")
set_tests_properties(test_integrator_base PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

