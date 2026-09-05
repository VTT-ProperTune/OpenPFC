# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Kernel data layer unit tests
file(GLOB DATA_UNIT_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
target_sources(openpfc-tests PRIVATE ${DATA_UNIT_TEST_SOURCES})
# cuda_field_residency_compile.cpp is an OBJECT library in
# cmake/LibraryConfiguration.cmake; it is not part of openpfc-tests.
