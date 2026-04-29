# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Frontend UI layer tests
target_sources(openpfc-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_json_wiring_context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_heffte_plan_options.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_json_helpers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_settings_loader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_ui_errors.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_toml_to_json.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_parameter_metadata.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_parameter_validator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_profiling_config.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_validation_result.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_parameter_validation.cpp
)
