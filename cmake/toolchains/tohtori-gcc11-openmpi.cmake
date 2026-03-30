# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Toolchain for VTT **tohtori**: GCC 11.2.0 + OpenMPI 4.1.1 without `module load`
# in the CMake/IDE process (e.g. Cursor CMake Tools).
#
# Paths match `module show gcc/11.2.0` and `module show openmpi/4.1.1` on tohtori.
# After cluster upgrades, re-check those modules or use CMakeUserPresets.json to
# point CMAKE_TOOLCHAIN_FILE at a copy of this file with updated paths.

set(_OpenPFC_tohtori_gcc_root "/share/apps/gcc/11.2.0")
set(_OpenPFC_tohtori_ompi_root "/share/apps/OpenMPI/4.1.1")

set(CMAKE_C_COMPILER "${_OpenPFC_tohtori_gcc_root}/bin/gcc" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${_OpenPFC_tohtori_gcc_root}/bin/g++" CACHE FILEPATH "C++ compiler")
set(MPI_C_COMPILER "${_OpenPFC_tohtori_ompi_root}/bin/mpicc" CACHE FILEPATH "MPI C wrapper")
set(MPI_CXX_COMPILER "${_OpenPFC_tohtori_ompi_root}/bin/mpicxx" CACHE FILEPATH "MPI C++ wrapper")

if(NOT "$ENV{HOME}" STREQUAL "")
  set(_OpenPFC_heffte_cpu "$ENV{HOME}/opt/heffte/2.4.1-cpu")
  if(EXISTS "${_OpenPFC_heffte_cpu}")
    list(PREPEND CMAKE_PREFIX_PATH "${_OpenPFC_heffte_cpu}")
  endif()
endif()
