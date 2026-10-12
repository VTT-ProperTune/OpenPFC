# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

file(GLOB GEOMETRY_UNIT_TEST_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
target_sources(openpfc-tests PRIVATE ${GEOMETRY_UNIT_TEST_SOURCES})
