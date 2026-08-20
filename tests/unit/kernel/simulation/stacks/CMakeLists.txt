# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Kernel simulation stacks unit tests
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fd_cpu_stack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fd_padded_cpu_stack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_spectral_cpu_stack.cpp
)
