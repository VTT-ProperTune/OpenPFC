# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# CMake minimum version (must be at top level, before project())
cmake_minimum_required(VERSION 3.21)

# Project definition (must be literal in root file to avoid CMake warnings)
# 0.2.0 is the first release of the post-Gen-1 architecture (PR #71).
# Configure with -DOpenPFC_DEVELOPMENT=ON for the "-dev" version suffix.
project(OpenPFC VERSION 0.2.0 DESCRIPTION "Phase Field Crystal simulation framework")

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
include(cmake/CUDASupport.cmake)

# ----------------------------------------------------------------------------
# 4. HIP Support
# ----------------------------------------------------------------------------
# HIP (ROCm) detection and configuration
include(cmake/HIPSupport.cmake)

# ----------------------------------------------------------------------------
# 5a. GPU Kernel Auto-Tuning
#----------------------------------------------------------------------------
# GPU kernel auto-tuning infrastructure (requires CUDA or HIP)
include(cmake/GPUAutotuning.cmake)


# ----------------------------------------------------------------------------
# 5. Dependencies
# ----------------------------------------------------------------------------
# Third-party dependencies — see cmake/Dependencies.cmake: MPI, HeFFTe,
# nlohmann_json, toml++; optional HDF5; Doxygen and docs/ when
# OpenPFC_BUILD_DOCUMENTATION=ON (default ON).
# Catch2 is found from BuildOptions.cmake when OpenPFC_BUILD_TESTS=ON.
include(cmake/Dependencies.cmake)

# ----------------------------------------------------------------------------
# 5b. GPU-aware MPI (configure-time probes where applicable)
# ----------------------------------------------------------------------------
include(cmake/OpenPFCGPUAwareMpi.cmake)

# ----------------------------------------------------------------------------
# 6. Library Configuration
# ----------------------------------------------------------------------------
# Create the main library target, configure its properties, and link dependencies
include(cmake/LibraryConfiguration.cmake)

# ----------------------------------------------------------------------------
# 7. Code Coverage
# ----------------------------------------------------------------------------
# Directory-scope --coverage must be set before tests/apps/examples are added.
include(cmake/CodeCoverage.cmake)

# ----------------------------------------------------------------------------
# 8. Build Options
# ----------------------------------------------------------------------------
# Configure what gets built: apps, examples, tests; benchmark sources under
# tests/ when OpenPFC_BUILD_BENCHMARKS=ON (see tests/CMakeLists.txt).
include(cmake/BuildOptions.cmake)

# ----------------------------------------------------------------------------
# 9. Installation Rules
# ----------------------------------------------------------------------------
# Installation rules for headers, libraries, and binaries
include(cmake/Installation.cmake)

# ----------------------------------------------------------------------------
# 10. Package Configuration
# ----------------------------------------------------------------------------
# Generate CMake package configuration files for downstream projects
include(cmake/PackageConfig.cmake)

# ----------------------------------------------------------------------------
# 11. Build Summary
# ----------------------------------------------------------------------------
# Print final build configuration summary
include(cmake/BuildSummary.cmake)
