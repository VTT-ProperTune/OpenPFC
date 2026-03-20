# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Find and configure all project dependencies

# MPI (required)
find_package(MPI REQUIRED)

# HeFFTe (required): must be installed separately — OpenPFC does not download it.
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
else()
  message(FATAL_ERROR
    "HeFFTe was not found (CMake package 'Heffte').\n"
    "\n"
    "OpenPFC does not download or build HeFFTe automatically. Install HeFFTe first, then point CMake at it.\n"
    "\n"
    "  → See INSTALL.md in the repository root for step-by-step instructions (recommended layout: ~/opt/heffte/2.4.1).\n"
    "\n"
    "Quick hints:\n"
    "  • Releases: https://github.com/icl-utk-edu/heffte/releases (e.g. v2.4.1)\n"
    "  • Upstream build guide: https://icl-utk-edu.github.io/heffte/md_doxygen_installation.html\n"
    "  • After install: export CMAKE_PREFIX_PATH=<heffte-prefix>:$CMAKE_PREFIX_PATH\n"
    "    or: cmake ... -DHeffte_DIR=<heffte-prefix>/lib64/cmake/Heffte\n"
    "    (some installs use lib/cmake/Heffte — use the directory that contains HeffteConfig.cmake)\n"
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
