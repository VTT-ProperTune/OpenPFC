# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

file(GLOB GEOMETRY_UNIT_TEST_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
# The spectral experiment needs the optional compiled ComplexLineFFT backend.
# Pure map/positive-Jacobian tests must remain available without FFTW3.
if(NOT OpenPFC_HAS_FFTW3)
  list(FILTER GEOMETRY_UNIT_TEST_SOURCES EXCLUDE REGEX
       "/test_periodic_cluster_map[.]cpp$")
endif()
target_sources(openpfc-tests PRIVATE ${GEOMETRY_UNIT_TEST_SOURCES})
