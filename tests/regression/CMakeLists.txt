# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Regression tests for Pre-M0 PE (periodicity) and PF (coordinate rounding) defects
# Re-pointed from World to Domain abstraction in M1

target_sources(openpfc-tests PRIVATE
  test_be_01.cpp
  test_be_02.cpp
)
