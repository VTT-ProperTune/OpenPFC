<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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

4. **HipSupport.cmake**
   - HIP (ROCm) detection and configuration
   - HIP availability checks (pairs with CUDA for optional GPU backends)

### Dependency Management

5. **Dependencies.cmake**
   - MPI (required)
   - HeFFTe: **`OpenPFCHeffteHints.cmake`** probes common install prefixes (`$HOME/opt/heffte`, `/opt`, `/share/apps`, Spack `EBROOTHEFFTE`, …) before failing; when **CUDA** or **HIP** is enabled, **`Dependencies.cmake`** warns if the found HeFFTe lacks the matching backend (CUDA or ROCm); optional **`OpenPFC_FETCH_HEFFTE=ON`** uses **FetchContent** into **`build/_deps`** (needs **FFTW** + **MPI** — see **INSTALL.md**)
   - nlohmann_json (required)
   - toml++ (via FindTomlPlusPlus.cmake)
   - Optional **HDF5** when **`OpenPFC_ENABLE_HDF5=ON`** (profiling export)
   - **Doxygen** and **`docs/`** when **`OpenPFC_BUILD_DOCUMENTATION=ON`** (default **ON**); if Doxygen is missing, documentation generation is skipped with a warning

6. **OpenPFCGpuAwareMpi.cmake** (included from the root **`CMakeLists.txt`** immediately after **Dependencies.cmake**)
   - Optional **CUDA + Open MPI** configure probe using **`MPIX_Query_cuda_support()`** when **`OpenPFC_MPI_CUDA_AWARE=ON`**
   - **HIP**: status-only reminder for runtime checks (**`verify_gpu_aware_mpi`**, **`MPICH_GPU_SUPPORT_ENABLED`** — see **INSTALL.md** §5.2.1 and **docs/INSTALL.LUMI.md**)

### Library and Build Configuration

7. **LibraryConfiguration.cmake**
   - Main `openpfc` library target creation
   - Library properties and versioning
   - Include directories
   - Dependency linking (MPI, HeFFTe)
   - GPU kernel library when **CUDA** and/or **HIP** (ROCm) GPU support is enabled
   - toml++ include directory setup

8. **BuildOptions.cmake**
   - Options **`OpenPFC_BUILD_APPS`**, **`OpenPFC_BUILD_EXAMPLES`**, **`OpenPFC_BUILD_TESTS`** (and related); **`add_subdirectory`** for **apps**, **examples**, and **tests** when enabled
   - **`OpenPFC_BUILD_BENCHMARKS`** (default **OFF**) is defined here but only affects sources under **`tests/benchmarks/`** via **`tests/CMakeLists.txt`** (see **`tests/benchmarks/README.md`**)
   - Catch2 finding and **`enable_testing()`** when tests are built

9. **CodeCoverage.cmake**
   - Coverage tool detection (lcov, genhtml, gcov)
   - Coverage flags for the library
   - Custom targets: `coverage` and `coverage-clean`

### Installation and Packaging

10. **Installation.cmake**
   - Header installation
   - Library installation
   - GPU kernel library installation (if enabled)
   - nlohmann_json header installation (when built from source)

11. **PackageConfig.cmake**
   - CMake package configuration file generation
   - Export target configuration
   - Version file generation

### Build Summary

12. **BuildSummary.cmake**
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
HipSupport.cmake
  ↓
Dependencies.cmake (may check CUDA/HIP availability)
  ↓
OpenPFCGpuAwareMpi.cmake (optional CUDA + Open MPI GPU-aware probe)
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
2. **Readability**: The root CMakeLists.txt is now ~90 lines instead of 542
3. **Modularity**: Easy to modify specific aspects without touching other parts
4. **Testability**: Individual modules can be tested in isolation
5. **Documentation**: Each module is self-contained and easier to document

## Cluster / VS Code (tohtori)

On **tohtori**, **Cursor** / **VS Code CMake Tools** may configure without **`module load`**, so MPI and GCC 11 are missing from the environment. Use CMake presets **`tohtori-debug`** and **`tohtori-release`** in the root **`CMakePresets.json`**: they set **`CMAKE_TOOLCHAIN_FILE`** to **`cmake/toolchains/tohtori-gcc11-openmpi.cmake`** and preset **`environment`** entries for **`PATH`** and **`LD_LIBRARY_PATH`** (same effect as **`gcc/11.2.0`** + **`openmpi/4.1.1`** modules). Build directories are **`build/tohtori-debug`** and **`build/tohtori-release`**.

The toolchain sets **`MPI_C_COMPILER`** / **`MPI_CXX_COMPILER`** to OpenMPI wrappers under **`/share/apps/OpenMPI/4.1.1`**. Presets **`tohtori-debug`** / **`tohtori-release`** set **`OpenPFC_ENABLE_CODE_COVERAGE=OFF`** so configure stays quiet on nodes without **`lcov`** (typical on RHEL); re-enable with **`-DOpenPFC_ENABLE_CODE_COVERAGE=ON`** after installing **`lcov`** (e.g. **EPEL** + **`dnf install lcov`** on EL8).

If the site layout changes, run **`module show`** on the cluster and edit the toolchain, or add **`CMakeUserPresets.json`** (CMake ≥ 3.19) at the **source root** that inherits **`tohtori-debug`** and overrides **`cacheVariables.CMAKE_TOOLCHAIN_FILE`** to your own file:

```json
{
  "version": 3,
  "cmakeMinimumRequired": { "major": 3, "minor": 21, "patch": 0 },
  "configurePresets": [
    {
      "name": "my-tohtori",
      "inherits": ["tohtori-debug"],
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "${sourceDir}/cmake/toolchains/my-site.cmake"
      }
    }
  ]
}
```

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
- `FindHeffte.cmake` (deprecated stub — do not include; HeFFTe is required via `find_package` after install)
- `FindHeffteFFTWLibraries.cmake` (legacy helper; unused by default)
- `FindJson.cmake`
- `Findnlohmann_json.cmake`
- `FindTomlPlusPlus.cmake`

These are used by `Dependencies.cmake` to locate third-party packages.
