# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Model unit tests
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_model_fft_baseline.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/test_model_comprehensive.cpp  # TODO: Fix incomplete test
)
