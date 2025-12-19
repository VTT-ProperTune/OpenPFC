# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Add field operations unit tests to main test target
target_sources(openpfc-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_field_operations.cpp
)
