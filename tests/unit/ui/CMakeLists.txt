# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# UI layer tests
target_sources(openpfc-tests PRIVATE
  test_ui_errors.cpp
  test_toml_to_json.cpp
  test_parameter_metadata.cpp
  test_parameter_validator.cpp
  test_validation_result.cpp
)
