# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Find and configure all project dependencies

# MPI (required)
find_package(MPI REQUIRED)

# Try to find HeFFTe installed on the system (e.g., installed by Nix)
find_package(Heffte CONFIG QUIET)
# If not found, fallback to downloading via FetchContent
if (Heffte_FOUND)
  message(STATUS "✅ HeFFTe v${Heffte_VERSION} found at ${Heffte_DIR}")
  
  # Verify CUDA backend if CUDA is enabled
  if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
      if(NOT Heffte_CUDA_FOUND)
          message(WARNING "⚠️  CUDA-enabled HeFFTe not found. HeFFTe at ${Heffte_DIR} may not have CUDA support.")
          message(WARNING "   For CUDA builds, use: -DHeffte_DIR=$HOME/opt/heffte/2.4.1-cuda/lib64/cmake/Heffte")
      else()
          message(STATUS "✅ HeFFTe CUDA backend is available")
      endif()
  endif()
else()
  message(WARNING "⚠️ Heffte not found via find_package(), falling back to FetchContent.")
  include(cmake/FindHeffte.cmake)
  if (NOT Heffte_FOUND)
    message(FATAL_ERROR "HeFFTe not found. Please install HeFFTe or set the Heffte_DIR variable to the location of HeffteConfig.cmake.")
  endif()
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
