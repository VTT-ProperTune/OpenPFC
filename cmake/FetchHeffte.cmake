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
  # Upstream v2.4.1, pinned independently of the mutable tag.
  URL https://codeload.github.com/icl-utk-edu/heffte/tar.gz/6e16996e5a3eb65c60fd62fd4d93c4b2df833033
  URL_HASH SHA256=543cc06336140d6eddbc7a0955fe795bf400440578131121fb7b03ca4922fba6
  PATCH_COMMAND "${CMAKE_COMMAND}" -DSOURCE_DIR=<SOURCE_DIR>
    -P "${CMAKE_CURRENT_LIST_DIR}/PatchHeffteFftw.cmake"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

FetchContent_MakeAvailable(openpfc_heffte)

# Upstream only defines Heffte::Heffte when HeFFTe is the top-level project.
if(TARGET Heffte AND NOT TARGET Heffte::Heffte)
  add_library(Heffte::Heffte INTERFACE IMPORTED GLOBAL)
  target_link_libraries(Heffte::Heffte INTERFACE Heffte)
endif()
