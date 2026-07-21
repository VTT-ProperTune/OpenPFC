# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

target_sources(openpfc-tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_spectral_exp_coefficients.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_error_evidence.cpp
)

# AC-facing CTest name: `ctest -R spectral_exp` (shared openpfc-tests binary).
add_test(NAME test_spectral_exp_coefficients
         COMMAND $<TARGET_FILE:openpfc-tests> "[integrator][spectral_exp]")
set_tests_properties(test_spectral_exp_coefficients PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

# Prefer running via openpfc-tests '[error_evidence]' (see plan validation).
add_test(NAME test_error_evidence
         COMMAND $<TARGET_FILE:openpfc-tests> "[error_evidence]")
set_tests_properties(test_error_evidence PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
