# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Kernel execution layer unit tests
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_databuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_kokkos_like.cpp
)
