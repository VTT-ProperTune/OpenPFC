# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# FFT unit tests
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fft_setting.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_kspace.cpp
)
