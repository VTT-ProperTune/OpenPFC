# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

target_sources(openpfc-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_checkpoint_publish.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_state_capture.cpp
)
