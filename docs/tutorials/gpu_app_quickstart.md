<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: GPU-enabled applications

A GPU run needs a matching compiled application, FFT dependency, device runtime
and MPI stack. Start from a successful CPU run, then change the build and
launcher while keeping the physical input fixed. The
[CPU and GPU build guide](../hpc/build_cpu_gpu.md) explains why each backend
needs its own build directory.

## Build the matching executable

On Tohtori, build and test CUDA through the supported script:

```bash
./scripts/build.sh --machine=tohtori --with-cuda --build-dir=builds/cuda
```

This selects the CUDA toolchain and a CUDA-enabled HeFFTe install. On LUMI use
HIP/ROCm; the script configures on the login node and schedules compilation and
tests on a GPU partition:

```bash
./scripts/build.sh --machine=lumi --with-rocm --partition=standard-g --wait
```

The [installation guide](../../INSTALL.md) describes dependency setup. The
[LUMI guide](../hpc/INSTALL.LUMI.md) gives the maintained source/build paths,
account, modules and communication check. CUDA is a Tohtori path; LUMI uses AMD
hardware.

## Prepare a small case

Use the [first tungsten run](../quickstart.md#run-an-application) to copy
`tungsten_single_seed.json` and replace its site-specific output paths with
paths under `results/`. The physics and field definitions are shared by the
CPU and GPU tungsten executables.

| Executable | Compiled stack |
|---|---|
| `tungsten` | CPU FFT / HeFFTe |
| `tungsten_cuda` | CUDA spectral stack |
| `tungsten_hip` | HIP spectral stack |

These executables select their stack at compilation. A JSON `backend` value
cannot add a missing backend or change `tungsten` into `tungsten_hip`. Other
applications have their own supported binaries and input contracts; use the
[application catalog](../user_guide/applications.md).

## Run in a GPU allocation

After preparing `results/first-tungsten/input.json`, a single-rank CUDA launch
from the checkout root has this form:

```bash
mpirun -n 1 ./builds/cuda/apps/tungsten/tungsten_cuda \
  results/first-tungsten/input.json
```

Run this inside an allocation with a GPU and the compiler/MPI/runtime modules
matching the build. The build script loads modules only in its own process;
it does not configure a later launch shell. On LUMI submit with `srun` using
the [LUMI GPU job recipe](../hpc/INSTALL.LUMI.md#gpu-jobs-and-verification).

Check that the startup log identifies the expected device backend and that
output files appear in the prepared directory. Compare a small CPU and GPU
case before increasing the rank count. Floating-point agreement is a numerical
question; successful startup alone does not establish it.

## GPU-aware MPI

Multi-GPU transforms and halo exchange pass device buffers to MPI. The MPI
installation must support those buffers. On LUMI export
`MPICH_GPU_SUPPORT_ENABLED=1` and run `verify_gpu_aware_mpi` inside the allocation
before a larger run. On Tohtori use the CUDA-aware stack selected by the build
script; see [Tohtori installation](../hpc/INSTALL.tohtori.md).

## Extend a GPU application

The [minimal custom application tutorial](custom_app_minimal.md) describes the
current session and callable-physics interfaces. There is no `pfc::App<Model>`
base class in the current API. A downstream GPU application needs a compatible
GPU-enabled SDK and device compilation for its own GPU translation units;
linking a CPU consumer to another prefix is not sufficient.

The native CLI's [CPU model bundle](../user_guide/cli.md) and source-model
workflow have their own support limits. Do not infer GPU qualification from
those CPU examples.

## See also

- [GPU path decision](../hpc/gpu_path_decision.md) — choose an execution route.
- [Configuration](../user_guide/configuration.md) — application input sections.
- [Parameter validation](../user_guide/parameter_validation.md) — rejected inputs.
- [Troubleshooting](../troubleshooting.md) — compiler, MPI and device failures.
