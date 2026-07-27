<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Frequently asked questions

This page gives short answers to common questions. Use the
[documentation index](index.md) to find a topic, or
[Troubleshooting](troubleshooting.md) for symptom-driven diagnosis.

## Getting started

### Where do I begin after cloning the repository?

Follow [Start here](start_here_15_minutes.md). It verifies the compiler, MPI,
HeFFTe, OpenPFC build, and one single-rank example. Continue with
[Quick start](quickstart.md) after that succeeds.

### Is OpenPFC header-only?

No. OpenPFC has public headers and a compiled library. An installed downstream
project links `OpenPFC::openpfc`; see
[Link OpenPFC from your own project](quickstart.md#link-openpfc-from-your-own-project).

### Do I need MPI?

Yes for the currently supported workflows. The spectral stack, decomposition,
and documented applications use MPI. A supported serial-only configuration is
not currently provided.

### Do I need a GPU?

No. The CPU/FFTW path is the normal starting point. CUDA and HIP are optional
backends and require matching HeFFTe builds. See the
[GPU path decision guide](hpc/gpu_path_decision.md).

## Build and CMake

### `find_package(OpenPFC)` cannot find the package

Install OpenPFC, then set `CMAKE_PREFIX_PATH` to the installation prefix or set
`OpenPFC_DIR` to the directory containing `OpenPFCConfig.cmake`. The compiler,
MPI, and HeFFTe stack used by the consumer must be compatible with the OpenPFC
installation. See [Troubleshooting](troubleshooting.md#find_packageopenpfc-fails-in-a-downstream-project).

### Why does the downstream CMake project enable both C and C++?

The installed package resolves MPI components that include the C target, so the
validated minimal consumer uses:

```cmake
project(my_sim LANGUAGES C CXX)
```

The complete tested shape is shown in
[Quick start](quickstart.md#link-openpfc-from-your-own-project).

### Why are examples or applications missing from the build directory?

Check `OpenPFC_BUILD_EXAMPLES` and `OpenPFC_BUILD_APPS`. Both are enabled by
default, but a previous CMake configuration may have cached them as `OFF`. See
[CMake options](reference/build_options.md).

### Should CPU, CUDA, and HIP use the same build directory?

No. Use separate build directories and matching HeFFTe installation prefixes.
See [CPU and GPU build trees](hpc/build_cpu_gpu.md).

## Running

### Where are the example executables?

They are normally under `<build>/examples/`. Exact placement can vary with the
CMake generator. The authoritative list is the
[Examples catalog](reference/examples_catalog.md).

### Why can an application not find its JSON or TOML file?

Configuration paths are interpreted relative to the process working directory,
not necessarily the source directory. Use an absolute path or calculate the
relative path from the directory in which `mpirun` or `srun` starts the
executable.

### How do I know a run succeeded?

The process must exit with status zero. Rank zero usually prints setup and time
integration messages, and configured writers may produce files. There is no
single success string shared by every example. See
[Example run output](reference/example_run_output.md).

### Where are result formats documented?

Start with [Result files](user_guide/io_results.md). The exact raw binary
contract is in [Binary field file layout](reference/binary_field_io_spec.md).

## Extending OpenPFC

### How do I add a model or a config-driven application?

Follow the [extension guide](extending_openpfc/README.md), then use the
[minimal custom application tutorial](tutorials/custom_app_minimal.md) for an
out-of-tree `App<Model>` executable.

### How do I add an initial or boundary condition?

Use a `FieldModifier` extension point or functional field operations, depending
on whether the behavior must be selected through configuration. See
[Extending OpenPFC](extending_openpfc/README.md) and
[Functional field operations](getting_started/functional_field_ops.md).

### Where do exact configuration keys belong?

Use the [Spectral App configuration reference](reference/spectral_app_config_reference.md).
Tutorials should demonstrate a small configuration but should not duplicate the
full key reference.

### Where is the API reference?

The generated C++ reference is part of this site: [API reference](api/index.md).
Use the [Tour of main types](reference/class_tour.md) when you need to connect a
class or namespace to the larger architecture.

## Contributing

Documentation structure, preview commands, and CI checks are described in
[Contributing to documentation](development/contributing-docs.md). General code,
test, commit, and changelog rules are in [`CONTRIBUTING.md`](../CONTRIBUTING.md).
