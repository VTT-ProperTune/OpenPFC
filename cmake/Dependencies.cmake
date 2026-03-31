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
# Cray PE: FindMPI can fail to detect MPI_C while MPI::MPI_CXX is sufficient for OpenPFC.
find_package(MPI REQUIRED COMPONENTS CXX)

option(OpenPFC_ENABLE_HDF5 "Enable HDF5 export for profiling dumps (optional)" OFF)
if(OpenPFC_ENABLE_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C)
  message(STATUS "✅ HDF5 enabled for profiling")
endif()

# HeFFTe (required). Prefer CMAKE_PREFIX_PATH / Heffte_DIR; then well-known prefixes
# (see cmake/OpenPFCHeffteHints.cmake); optionally FetchContent when still missing.
option(OpenPFC_FETCH_HEFFTE
  "If no HeFFTe is found, download and build v2.4.1 with FetchContent (needs FFTW)"
  OFF)

find_package(Heffte CONFIG QUIET)

if(NOT Heffte_FOUND AND NOT TARGET Heffte::Heffte AND NOT TARGET Heffte AND NOT TARGET heffte)
  include(cmake/OpenPFCHeffteHints.cmake)
  openpfc_heffte_autodetect_from_hints()
  find_package(Heffte CONFIG QUIET)
endif()

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
  include(cmake/OpenPFCHeffteHints.cmake)
  openpfc_heffte_collect_hint_prefixes(_openpfc_heffte_hint_roots)
  set(_openpfc_heffte_hint_lines "")
  foreach(_p IN LISTS _openpfc_heffte_hint_roots)
    string(APPEND _openpfc_heffte_hint_lines "    ${_p}\n")
  endforeach()
  message(FATAL_ERROR
    "HeFFTe was not found (CMake package 'Heffte').\n"
    "\n"
    "OpenPFC searched for HeffteConfig.cmake under your CMAKE_PREFIX_PATH / Heffte_DIR first, "
    "then under common install prefixes (e.g. \\$HOME/opt/heffte/..., /opt/heffte/..., "
    "/share/apps/heffte/...). None contained a valid HeFFTe CMake package.\n"
    "\n"
    "Prefix roots considered for auto-detection (install tree must contain "
    "lib64/cmake/Heffte or lib/cmake/Heffte):\n"
    "${_openpfc_heffte_hint_lines}"
    "\n"
    "Install HeFFTe (recommended: v2.4.1) to a stable prefix — see INSTALL.md §3:\n"
    "  • Example CPU install prefix:  \\$HOME/opt/heffte/2.4.1-cpu\n"
    "  • Then either:\n"
    "      export CMAKE_PREFIX_PATH=\\$HOME/opt/heffte/2.4.1-cpu:\\$CMAKE_PREFIX_PATH\n"
    "    or pass:\n"
    "      -DHeffte_DIR=\\$HOME/opt/heffte/2.4.1-cpu/lib64/cmake/Heffte\n"
    "    (use the directory that contains HeffteConfig.cmake; some sites use lib/cmake/Heffte)\n"
    "\n"
    "Ensure the same MPI toolchain as for OpenPFC (INSTALL.md §1): e.g. module load openmpi "
    "before configuring.\n"
    "\n"
    "If the install exists but uses a different path, set Heffte_DIR or CMAKE_PREFIX_PATH "
    "explicitly, or add your site prefix to cmake/OpenPFCHeffteHints.cmake.\n"
    "\n"
    "Optional fallback: -DOpenPFC_FETCH_HEFFTE=ON (requires FFTW + MPI; downloads into "
    "build/_deps only — see INSTALL.md).\n"
    "\n"
    "  → INSTALL.md §5.1 lists common configure failures (stale CMake cache, wrong MPI).\n"
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
