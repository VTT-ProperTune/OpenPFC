# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Toolchain for VTT **tohtori**: GCC 11.2.0 + OpenMPI without `module load` in CMake/IDE.
#
# GCC path matches `module show gcc/11.2.0` on tohtori. MPI wrappers default to
# `module show openmpi/4.1.1` unless **OPENMPI_ROOT** is set in the environment and
# contains bin/mpicc (see scripts/build_tohtori.sh --build-openmpi).

set(_OpenPFC_tohtori_gcc_root "/share/apps/gcc/11.2.0")
# Prefer a user-built OpenMPI (e.g. scripts/build_tohtori.sh --build-openmpi) when set.
# Note: /share/apps/OpenMPI/4.1.1 is the *site default prefix* (module openmpi/4.1.1), not an
# API/version pin — when OPENMPI_ROOT is unset, FindMPI uses that tree.
if(NOT "$ENV{OPENMPI_ROOT}" STREQUAL "" AND EXISTS "$ENV{OPENMPI_ROOT}/bin/mpicc")
  set(_OpenPFC_tohtori_ompi_root "$ENV{OPENMPI_ROOT}")
  set(_OpenPFC_tohtori_ompi_from_env TRUE)
else()
  set(_OpenPFC_tohtori_ompi_root "/share/apps/OpenMPI/4.1.1")
  set(_OpenPFC_tohtori_ompi_from_env FALSE)
endif()

set(CMAKE_C_COMPILER "${_OpenPFC_tohtori_gcc_root}/bin/gcc" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${_OpenPFC_tohtori_gcc_root}/bin/g++" CACHE FILEPATH "C++ compiler")
# Without FORCE, a prior configure (e.g. site MPI) leaves MPI_*_COMPILER in the cache and this
# toolchain line is ignored — then OPENMPI_ROOT can be set but CMake still uses old wrappers/libs.
if(_OpenPFC_tohtori_ompi_from_env)
  set(MPI_C_COMPILER "${_OpenPFC_tohtori_ompi_root}/bin/mpicc" CACHE FILEPATH "MPI C wrapper" FORCE)
  set(MPI_CXX_COMPILER "${_OpenPFC_tohtori_ompi_root}/bin/mpicxx" CACHE FILEPATH "MPI C++ wrapper" FORCE)
else()
  set(MPI_C_COMPILER "${_OpenPFC_tohtori_ompi_root}/bin/mpicc" CACHE FILEPATH "MPI C wrapper")
  set(MPI_CXX_COMPILER "${_OpenPFC_tohtori_ompi_root}/bin/mpicxx" CACHE FILEPATH "MPI C++ wrapper")
endif()
unset(_OpenPFC_tohtori_ompi_from_env)

# compile_commands.json for clang-tidy, clangd, and IDEs (matches CI -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile_commands.json (clang-tidy, clangd, IDEs)" FORCE)

if(NOT "$ENV{HOME}" STREQUAL "")
  set(_OpenPFC_heffte_cpu "$ENV{HOME}/opt/heffte/2.4.1-cpu")
  if(EXISTS "${_OpenPFC_heffte_cpu}")
    list(PREPEND CMAKE_PREFIX_PATH "${_OpenPFC_heffte_cpu}")
  endif()
endif()
