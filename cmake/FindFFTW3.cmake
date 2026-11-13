# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Double- and single-precision FFTW SDK (`fftw3` and `fftw3f`, one header);
# no threads/global lifecycle dependency is added. Both libraries are
# required: the local host FFT plans float transforms with `fftwf_*`, and
# every HeFFTe FFTW build links `fftw3f` itself.
find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h)
find_library(FFTW3_LIBRARY NAMES fftw3)
find_library(FFTW3F_LIBRARY NAMES fftw3f)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3 REQUIRED_VARS FFTW3_INCLUDE_DIR FFTW3_LIBRARY
                                                      FFTW3F_LIBRARY)
mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY FFTW3F_LIBRARY)
if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3)
  add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
  set_target_properties(FFTW3::fftw3 PROPERTIES
    IMPORTED_LOCATION "${FFTW3_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}")
endif()
if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3f)
  add_library(FFTW3::fftw3f UNKNOWN IMPORTED)
  set_target_properties(FFTW3::fftw3f PROPERTIES
    IMPORTED_LOCATION "${FFTW3F_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}")
endif()
