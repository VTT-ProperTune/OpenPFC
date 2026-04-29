<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Troubleshooting

Quick fixes for common configure and runtime problems. The full install story is [`INSTALL.md`](../INSTALL.md).

## Configure-time

### `fatal error: 'mpi.h' file not found`

CMake or the compiler is not using the same MPI you intend. Before `cmake`:

1. Load your modules (`gcc`, `openmpi`, …) — see [`INSTALL.md`](../INSTALL.md) §1.
2. Set `CC` / `CXX` to the MPI wrappers or module `gcc` explicitly.
3. Remove a stale build tree or re-run `cmake` with `-DCMAKE_C_COMPILER=…` and `-DCMAKE_CXX_COMPILER=…` so `CMakeCache.txt` does not still point at `/usr/bin/gcc`.

HIP/Cray builds may need extra include flags for MPI in HIP translation units; see [`INSTALL.LUMI.md`](INSTALL.LUMI.md) §2.

### CMake finds the wrong GCC (e.g. GCC 8 instead of 11)

Same as above: configure after `module load`, set compilers explicitly, or delete the build directory. See [`INSTALL.md`](../INSTALL.md) (“Stale CMake cache”).

### `Could not find Heffte` / `Heffte_DIR` not set

1. Build and install HeFFTe 2.4.1 (or compatible) for your backend — [`INSTALL.md`](../INSTALL.md) §3.
2. Point CMake at the install prefix: `export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cpu:$CMAKE_PREFIX_PATH` (adjust path and variant: `-cpu`, `-cuda`, `-rocm`).
3. Do not unpack HeFFTe inside the OpenPFC clone.

### `find_package(OpenPFC)` fails in a downstream project

OpenPFC must be installed (or you must point CMake at a build tree that exports the package). Set `CMAKE_PREFIX_PATH` to the install prefix, or `-DOpenPFC_DIR=/path/to/lib/cmake/OpenPFC`. See [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md).

## Link / run-time

### Wrong MPI at run time (`mpirun` from MPICH, binary linked to Open MPI)

Build and run with the same MPI implementation. `which mpirun` should match the prefix of `mpicc` used to build HeFFTe and OpenPFC. See [`INSTALL.md`](../INSTALL.md) (MPI callout).

### Missing `examples/` or `apps/...` binaries

Examples and apps are controlled by `OpenPFC_BUILD_EXAMPLES` and `OpenPFC_BUILD_APPS` (default ON). If you configured with OFF, reconfigure with ON and rebuild. See [`quickstart.md`](quickstart.md).

### GPU / HIP job fails with GPU-aware MPI errors

For ROCm/LUMI-style stacks, `MPICH_GPU_SUPPORT_ENABLED=1` and a build with GPU-aware MPI may be required. See [`INSTALL.LUMI.md`](INSTALL.LUMI.md) and [`applications.md`](applications.md).

## Config-driven runs (`App` + JSON/TOML)

### Process exits immediately with “validation” or parameter errors

Models such as tungsten validate `model.params` at startup. Read the printed report: missing keys, out-of-range values, or wrong types. Compare your file to `apps/tungsten/inputs_json/` samples. See [`app_pipeline.md`](app_pipeline.md) for which JSON sections are consumed and [`apps/tungsten/README.md`](../apps/tungsten/README.md) for layout pointers.

### “No such file” for the config path

Paths are resolved from the current working directory (often your `build/` folder). Use a path relative to that directory, or an absolute path.

## Still stuck?

- [`faq.md`](faq.md) — short Q&A  
- [`quickstart.md`](quickstart.md) — first successful run  
- [`mpi_io_layout_checklist.md`](mpi_io_layout_checklist.md) — MPI ranks, cwd, binary I/O  
- [`learning_paths.md`](learning_paths.md) — pick a sequenced track by role  
- [`README.md`](README.md) — full documentation index  
- Issues — <https://github.com/VTT-ProperTune/OpenPFC/issues>
