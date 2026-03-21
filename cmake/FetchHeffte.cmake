# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Populate HeFFTe via FetchContent when OpenPFC_FETCH_HEFFTE is ON and no
# installed package was found. Requires FFTW development libraries (and MPI).

include(FetchContent)

set(Heffte_ENABLE_FFTW ON CACHE BOOL "HeFFTe FFTW backend (OpenPFC fetch)" FORCE)
set(Heffte_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(Heffte_ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)

if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  set(Heffte_ENABLE_CUDA ON CACHE BOOL "HeFFTe CUDA backend (OpenPFC fetch)" FORCE)
endif()
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  set(Heffte_ENABLE_ROCM ON CACHE BOOL "HeFFTe ROCm backend (OpenPFC fetch)" FORCE)
endif()

FetchContent_Declare(
  openpfc_heffte
  URL https://github.com/icl-utk-edu/heffte/archive/refs/tags/v2.4.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

FetchContent_MakeAvailable(openpfc_heffte)

# Upstream only defines Heffte::Heffte when HeFFTe is the top-level project.
if(TARGET Heffte AND NOT TARGET Heffte::Heffte)
  add_library(Heffte::Heffte INTERFACE IMPORTED GLOBAL)
  target_link_libraries(Heffte::Heffte INTERFACE Heffte)
endif()
