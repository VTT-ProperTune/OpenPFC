# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Double-precision FFTW SDK; no threads/global lifecycle dependency is added.
find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h)
find_library(FFTW3_LIBRARY NAMES fftw3)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3 REQUIRED_VARS FFTW3_INCLUDE_DIR FFTW3_LIBRARY)
mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY)
if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3)
  add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
  set_target_properties(FFTW3::fftw3 PROPERTIES
    IMPORTED_LOCATION "${FFTW3_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}")
endif()
