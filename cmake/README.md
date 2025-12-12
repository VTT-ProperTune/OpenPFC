<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# CMake Module Structure

This directory contains modular CMake configuration files that organize the build system into logical components. The root `CMakeLists.txt` includes these modules in the correct order.

## Module Overview

### Core Configuration Modules

1. **ProjectSetup.cmake**
   - Project definition and version
   - Development mode configuration
   - Version header generation
   - Build type defaults
   - CMake module path setup

2. **CompilerSettings.cmake**
   - C++ standard (C++17)
   - Compiler flags and warnings
   - Build-type specific flags
   - Debug mode settings
   - Clang-tidy integration

3. **CudaSupport.cmake**
   - CUDA detection and configuration
   - CUDA architecture settings
   - CUDA availability checks

### Dependency Management

4. **Dependencies.cmake**
   - MPI (required)
   - HeFFTe (with CUDA backend verification)
   - nlohmann_json (required)
   - toml++ (via FindTomlPlusPlus.cmake)
   - Doxygen (optional, for documentation)
   - Catch2 (found when tests are built)

### Library and Build Configuration

5. **LibraryConfiguration.cmake**
   - Main `openpfc` library target creation
   - Library properties and versioning
   - Include directories
   - Dependency linking (MPI, HeFFTe)
   - GPU kernel library (when CUDA is enabled)
   - toml++ include directory setup

6. **BuildOptions.cmake**
   - Build options for apps, examples, tests, benchmarks
   - Catch2 finding and test setup
   - Subdirectory inclusion (apps, examples, tests)

7. **CodeCoverage.cmake**
   - Coverage tool detection (lcov, genhtml, gcov)
   - Coverage flags for the library
   - Custom targets: `coverage` and `coverage-clean`

### Installation and Packaging

8. **Installation.cmake**
   - Header installation
   - Library installation
   - GPU kernel library installation (if enabled)
   - nlohmann_json header installation (when built from source)

9. **PackageConfig.cmake**
   - CMake package configuration file generation
   - Export target configuration
   - Version file generation

### Build Summary

10. **BuildSummary.cmake**
    - Final build configuration summary
    - Displays all options, dependencies, and settings

## Module Dependencies

The modules must be included in the correct order due to dependencies:

```
ProjectSetup.cmake
  ↓
CompilerSettings.cmake
  ↓
CudaSupport.cmake
  ↓
Dependencies.cmake (may check CUDA availability)
  ↓
LibraryConfiguration.cmake (creates openpfc target)
  ↓
BuildOptions.cmake
  ↓
CodeCoverage.cmake (modifies openpfc target)
  ↓
Installation.cmake
  ↓
PackageConfig.cmake
  ↓
BuildSummary.cmake
```

## Benefits of This Structure

1. **Maintainability**: Each module has a single, clear responsibility
2. **Readability**: The root CMakeLists.txt is now ~75 lines instead of 542
3. **Modularity**: Easy to modify specific aspects without touching other parts
4. **Testability**: Individual modules can be tested in isolation
5. **Documentation**: Each module is self-contained and easier to document

## Adding New Configuration

When adding new build configuration:

1. Determine which module it belongs to (or create a new one if needed)
2. Add the configuration to the appropriate module
3. Update this README if creating a new module
4. Ensure proper ordering in the root CMakeLists.txt

## Find Modules

Additional Find modules for dependencies are also in this directory:
- `FindArgparse.cmake`
- `FindCatch2.cmake`
- `FindHeffte.cmake`
- `FindHeffteFFTWLibraries.cmake`
- `FindJson.cmake`
- `Findnlohmann_json.cmake`
- `FindTomlPlusPlus.cmake`

These are used by `Dependencies.cmake` to locate third-party packages.
