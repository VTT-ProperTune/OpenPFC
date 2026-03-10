# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Operator / sparse vector unit tests (part of main test executable)
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_diffop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sparsevector.cpp
)
