# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Toolchain for LUMI-G: Cray wrappers + cpeGNU (GCC 12.x) + cray-mpich + ROCm/HIP.
#
# Expected module stack before configure:
#   module purge
#   module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
#
# This toolchain intentionally uses Cray compiler wrappers to keep the MPI and PE
# link/include paths coherent.

set(CMAKE_C_COMPILER cc CACHE FILEPATH "Cray C wrapper")
set(CMAKE_CXX_COMPILER CC CACHE FILEPATH "Cray C++ wrapper")
set(MPI_C_COMPILER cc CACHE FILEPATH "Cray MPI C wrapper")
set(MPI_CXX_COMPILER CC CACHE FILEPATH "Cray MPI C++ wrapper")

# Find Cray MPICH include dir for HIP translation units (mpi.h under hipcc).
# Prefer environment, then known Cray installation layout.
set(_OpenPFC_lumi_mpich_inc "")
if(DEFINED ENV{MPICH_DIR} AND EXISTS "$ENV{MPICH_DIR}/include/mpi.h")
  set(_OpenPFC_lumi_mpich_inc "$ENV{MPICH_DIR}/include")
else()
  file(GLOB _OpenPFC_lumi_mpich_inc_candidates
    "/opt/cray/pe/mpich/*/ofi/gnu/*/include")
  list(REVERSE _OpenPFC_lumi_mpich_inc_candidates)
  foreach(_cand IN LISTS _OpenPFC_lumi_mpich_inc_candidates)
    if(EXISTS "${_cand}/mpi.h")
      set(_OpenPFC_lumi_mpich_inc "${_cand}")
      break()
    endif()
  endforeach()
endif()

if(NOT _OpenPFC_lumi_mpich_inc STREQUAL "")
  set(CMAKE_HIP_FLAGS_INIT "-I${_OpenPFC_lumi_mpich_inc}")
  message(STATUS "LUMI toolchain: using MPICH include for HIP: ${_OpenPFC_lumi_mpich_inc}")
else()
  message(WARNING
    "LUMI toolchain: could not auto-detect Cray MPICH include path for HIP. "
    "If HIP sources fail with 'mpi.h not found', pass "
    "-DCMAKE_HIP_FLAGS='-I/opt/cray/pe/mpich/<ver>/ofi/gnu/<ver>/include'.")
endif()

# Optional convenience for local installs.
if(NOT "$ENV{HOME}" STREQUAL "")
  set(_OpenPFC_heffte_rocm "$ENV{HOME}/opt/heffte/2.4.1-rocm")
  if(EXISTS "${_OpenPFC_heffte_rocm}")
    list(PREPEND CMAKE_PREFIX_PATH "${_OpenPFC_heffte_rocm}")
  endif()
endif()
