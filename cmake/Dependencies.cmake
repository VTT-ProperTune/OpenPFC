# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Find and configure all project dependencies

# MPI — required for all supported configurations (HeFFTe + distributed FFT).
if(NOT OpenPFC_ENABLE_MPI)
  message(FATAL_ERROR
    "OpenPFC_ENABLE_MPI=OFF is not supported. OpenPFC is built around MPI "
    "(and HeFFTe uses MPI). Install MPI development packages, e.g.\n"
    "  Debian/Ubuntu: sudo apt install libopenmpi-dev openmpi-bin\n"
    "  RHEL/Fedora:   sudo dnf install openmpi-devel\n"
    "Then reconfigure with the default OpenPFC_ENABLE_MPI=ON (see INSTALL.md §2).")
endif()
find_package(MPI REQUIRED)

# HeFFTe (required). Prefer an installed package; optionally fetch when missing.
option(OpenPFC_FETCH_HEFFTE
  "If no HeFFTe is found, download and build v2.4.1 with FetchContent (needs FFTW)"
  OFF)

find_package(Heffte CONFIG QUIET)

if(NOT Heffte_FOUND AND NOT TARGET Heffte::Heffte AND NOT TARGET Heffte AND NOT TARGET heffte)
  if(OpenPFC_FETCH_HEFFTE)
    message(STATUS "📥 HeFFTe not found; fetching and building with FetchContent")
    include(cmake/FetchHeffte.cmake)
  endif()
endif()

find_package(Heffte CONFIG QUIET)

if(Heffte_FOUND)
  message(STATUS "✅ HeFFTe v${Heffte_VERSION} found at ${Heffte_DIR}")

  # Verify CUDA backend if CUDA is enabled
  if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    if(NOT Heffte_CUDA_FOUND)
      message(WARNING "⚠️  CUDA-enabled HeFFTe not found. HeFFTe at ${Heffte_DIR} may not have CUDA support.")
      message(WARNING "   For CUDA builds, rebuild HeFFTe with -DHeffte_ENABLE_CUDA=ON (see INSTALL.md).")
    else()
      message(STATUS "✅ HeFFTe CUDA backend is available")
    endif()
  endif()
  # For HIP builds with find_package(Heffte), HeFFTe must have been built with ROCm (Heffte_ENABLE_ROCM=ON)
  if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
    if(DEFINED Heffte_ROCM_FOUND AND NOT Heffte_ROCM_FOUND)
      message(WARNING "⚠️  HIP-enabled HeFFTe not found. HeFFTe at ${Heffte_DIR} may not have ROCm/rocFFT support.")
      message(WARNING "   For HIP builds, use a HeFFTe built with -DHeffte_ENABLE_ROCM=ON.")
    elseif(Heffte_ROCM_FOUND)
      message(STATUS "✅ HeFFTe ROCm backend is available")
    endif()
  endif()
elseif(TARGET Heffte::Heffte OR TARGET Heffte OR TARGET heffte)
  message(STATUS "✅ HeFFTe targets available (FetchContent or add_subdirectory)")
else()
  message(FATAL_ERROR
    "HeFFTe was not found (CMake package 'Heffte').\n"
    "\n"
    "This is expected until HeFFTe is installed and CMake can see it. OpenPFC does not "
    "vendor HeFFTe in the source tree.\n"
    "\n"
    "Do this (see INSTALL.md §3 and §5.1):\n"
    "  • Build/install HeFFTe outside the repo, e.g. prefix $HOME/opt/heffte/2.4.1-cpu\n"
    "  • Then: export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cpu:$CMAKE_PREFIX_PATH\n"
    "    or: cmake ... -DHeffte_DIR=$HOME/opt/heffte/2.4.1-cpu/lib64/cmake/Heffte\n"
    "    (use the directory that contains HeffteConfig.cmake; some sites use lib/cmake/Heffte)\n"
    "\n"
    "Also ensure MPI is OpenMPI from your modules (INSTALL.md §1): module load gcc/... && "
    "module load openmpi, then CC/CXX from that environment.\n"
    "\n"
    "Optional fallback: -DOpenPFC_FETCH_HEFFTE=ON (FFTW + MPI still required; FetchContent "
    "uses build/_deps only, not the OpenPFC source tree).\n"
    "\n"
    "  → INSTALL.md §5.1 lists common configure failures (HeFFTe missing, mpi.h missing, stale cache).\n"
  )
endif()

# nlohmann_json (required)
find_package(nlohmann_json REQUIRED)

# Add toml++ for TOML file support
include(cmake/FindTomlPlusPlus.cmake)

# Documentation dependencies
option(OpenPFC_BUILD_DOCUMENTATION "Build documentation" ON)

if(OpenPFC_BUILD_DOCUMENTATION)
  message(STATUS "📚 Generating sources for documentation (build with target docs)")
  find_package(Doxygen)

  if(Doxygen_FOUND)
    message(STATUS "✅ Doxygen v${DOXYGEN_VERSION} found at ${DOXYGEN_EXECUTABLE}")
    add_subdirectory(docs)
  else()
    message(WARNING "⚠️  Doxygen not found, skipping documentation generation.")
  endif()
endif()

# Test dependencies (will be found when tests are built)
# Catch2 is found in BuildOptions.cmake when OpenPFC_BUILD_TESTS is ON
