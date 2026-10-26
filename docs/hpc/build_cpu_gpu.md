<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# CPU and GPU build trees

Keep a separate build directory for each backend. CUDA and HIP select different
translation units, compilers and dependency libraries; changing flags inside
one configured tree can leave incompatible cached paths.

Use `scripts/build.sh` from the repository root. It loads the matching modules,
selects the toolchain, builds and runs tests. The following Tohtori commands use
explicit directories so switching backends cannot reuse another backend's cache:

```bash
./scripts/build.sh --machine=tohtori --cpu --build-dir=builds/cpu
./scripts/build.sh --machine=tohtori --with-cuda --build-dir=builds/cuda
```

On LUMI the build directory belongs on flash. Configure runs on the login node;
compilation and tests run in a scheduled GPU allocation:

```bash
./scripts/build.sh --machine=lumi --with-rocm --partition=standard-g --wait
```

| Variant | Main options selected by the script | Dependency |
|---|---|---|
| CPU spectral | CUDA and HIP off; HeFFTe on | CPU HeFFTe with FFTW |
| CUDA spectral | `OpenPFC_ENABLE_CUDA=ON` | CUDA toolkit and CUDA-enabled HeFFTe |
| HIP spectral | `OpenPFC_ENABLE_HIP=ON` | ROCm and ROCm-enabled HeFFTe |
| CPU finite differences only | `--cpu --no-heffte` | MPI; spectral targets are omitted |

HeFFTe is installed outside the OpenPFC clone. To select a site-specific install,
set `HEFFTE_PREFIX` or `HEFFTE_MODULE` before invoking the script. Run
`./scripts/build.sh --help` for supported overrides, architecture choices and
machine defaults.

Use the [installation guide](../../INSTALL.md) for dependency setup,
the [LUMI guide](INSTALL.LUMI.md) for scheduled HIP builds and storage, and
the [GPU application tutorial](../tutorials/gpu_app_quickstart.md) for launching
the resulting executables. A build directory selects compiled capabilities;
a JSON `backend` value cannot enable a backend absent from that build.
