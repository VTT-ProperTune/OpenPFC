# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# CMake minimum version (must be at top level, before project())
cmake_minimum_required(VERSION 3.15)

# Project definition (must be literal in root file to avoid CMake warnings)
project(OpenPFC VERSION 0.1.4 DESCRIPTION "Phase Field Crystal simulation framework")

# ============================================================================
# OpenPFC CMake Build System
# ============================================================================
# This root CMakeLists.txt orchestrates the build by including modular
# configuration files from the cmake/ directory. Each module handles a
# specific aspect of the build configuration.
# ============================================================================

# ----------------------------------------------------------------------------
# 1. Project Setup
# ----------------------------------------------------------------------------
# Project version configuration, development mode, and basic CMake configuration
# Note: project() is already called above, this file only sets up version details
include(cmake/ProjectSetup.cmake)

# ----------------------------------------------------------------------------
# 2. Compiler Settings
# ----------------------------------------------------------------------------
# C++ standard, compiler flags, warnings, and static analysis tools
include(cmake/CompilerSettings.cmake)

# ----------------------------------------------------------------------------
# 3. CUDA Support
# ----------------------------------------------------------------------------
# CUDA detection and configuration (must come before dependencies that may
# depend on CUDA availability)
include(cmake/CudaSupport.cmake)

# ----------------------------------------------------------------------------
# 4. Dependencies
# ----------------------------------------------------------------------------
# Find and configure all third-party dependencies (MPI, HeFFTe, nlohmann_json,
# toml++, Doxygen, Catch2)
include(cmake/Dependencies.cmake)

# ----------------------------------------------------------------------------
# 5. Library Configuration
# ----------------------------------------------------------------------------
# Create the main library target, configure its properties, and link dependencies
include(cmake/LibraryConfiguration.cmake)

# ----------------------------------------------------------------------------
# 6. Build Options
# ----------------------------------------------------------------------------
# Configure what gets built (apps, examples, tests, benchmarks)
include(cmake/BuildOptions.cmake)

# ----------------------------------------------------------------------------
# 7. Code Coverage
# ----------------------------------------------------------------------------
# Code coverage configuration and custom targets
include(cmake/CodeCoverage.cmake)

# ----------------------------------------------------------------------------
# 8. Installation Rules
# ----------------------------------------------------------------------------
# Installation rules for headers, libraries, and binaries
include(cmake/Installation.cmake)

# ----------------------------------------------------------------------------
# 9. Package Configuration
# ----------------------------------------------------------------------------
# Generate CMake package configuration files for downstream projects
include(cmake/PackageConfig.cmake)

# ----------------------------------------------------------------------------
# 10. Build Summary
# ----------------------------------------------------------------------------
# Print final build configuration summary
include(cmake/BuildSummary.cmake)
