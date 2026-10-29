<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# EasyBuild recipes

Module install of released OpenPFC on LUMI, as an alternative to building from
a source checkout with [`scripts/build.sh`](../scripts/build.sh). The modules
are installed once into the project's shared EasyBuild tree, so members of
`project_462001519` load them instead of building.

| Easyconfig | What it builds |
|---|---|
| [`h/HeFFTe/HeFFTe-2.4.1-cpeGNU-25.09-fftw-planfix.eb`](easyconfigs/h/HeFFTe/HeFFTe-2.4.1-cpeGNU-25.09-fftw-planfix.eb) | HeFFTe 2.4.1 with the FFTW planning correction, FFTW backend, `cpeGNU` (LUMI-C) |
| [`h/HeFFTe/HeFFTe-2.4.1-cpeGNU-25.09-rocm-planfix.eb`](easyconfigs/h/HeFFTe/HeFFTe-2.4.1-cpeGNU-25.09-rocm-planfix.eb) | The same correction, rocFFT + FFTW, GPU-aware MPI, `gfx90a` (LUMI-G) |
| [`o/OpenPFC/OpenPFC-0.6.0-cpeGNU-25.09.eb`](easyconfigs/o/OpenPFC/OpenPFC-0.6.0-cpeGNU-25.09.eb) | OpenPFC 0.6.0, CPU applications |
| [`o/OpenPFC/OpenPFC-0.6.0-cpeGNU-25.09-rocm.eb`](easyconfigs/o/OpenPFC/OpenPFC-0.6.0-cpeGNU-25.09-rocm.eb) | OpenPFC 0.6.0, CPU and HIP applications, GPU-aware MPI |
| [`h/HeFFTe/HeFFTe-2.4.1-cpeGNU-25.09-fftw.eb`](easyconfigs/h/HeFFTe/HeFFTe-2.4.1-cpeGNU-25.09-fftw.eb), [`-rocm.eb`](easyconfigs/h/HeFFTe/HeFFTe-2.4.1-cpeGNU-25.09-rocm.eb) | Unpatched HeFFTe 2.4.1 used by the installed 0.5.0 and 0.4.0 modules |

The `-planfix` HeFFTe recipes apply
[`HeFFTe-2.4.1_fftw-real-plans.patch`](easyconfigs/h/HeFFTe/HeFFTe-2.4.1_fftw-real-plans.patch),
the same change as [`cmake/patches/heffte-2.4.1-fftw-plans.patch`](../cmake/patches/heffte-2.4.1-fftw-plans.patch).
Unpatched 2.4.1 plans FFTW real transforms in place and executes them out of
place, which can crash or give wrong float and strided real transforms
(#781). Both LUMI HeFFTe builds contain the FFTW backend, so both need it.
A new suffix leaves the unpatched modules, which older OpenPFC modules
depend on, as installed.

## Use

Set the project prefix before loading the `LUMI` stack; the stack module adds
the project's EasyBuild tree to the module path from it.

```bash
export EBU_USER_PREFIX=/projappl/project_462001519/EasyBuild

module load LUMI/25.09 partition/C
module load OpenPFC/0.6.0-cpeGNU-25.09
srun -p small -n 4 tungsten case.json
```

On LUMI-G:

```bash
export EBU_USER_PREFIX=/projappl/project_462001519/EasyBuild

module load LUMI/25.09 partition/G
module load OpenPFC/0.6.0-cpeGNU-25.09-rocm
srun -p standard-g -N 1 --ntasks-per-node=2 --gpus-per-node=2 \
  bash -c 'export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID; exec tungsten_hip case.json'
```

The modules put the shipped applications on `PATH` and add the install prefix
to `CMAKE_PREFIX_PATH`, so `find_package(OpenPFC)` works for code built
against them. The ROCm module also sets `MPICH_GPU_SUPPORT_ENABLED=1`.

Each release stays installed. The module name selects the version, and the
`-rocm` suffix selects the LUMI-G variant; `partition/C` or `partition/G`
decides which variants are visible. `module avail OpenPFC` lists them, and
loading another `OpenPFC` module replaces the loaded one.

> **Note**
> `module load` inside a shell pipeline runs in a subshell and its environment
> changes are lost. Load it on its own line.

## Build

Install into the shared tree with a group-writable umask, so other members of
the project can later install the next release:

```bash
export EBU_USER_PREFIX=/projappl/project_462001519/EasyBuild
umask 0007

module load LUMI/25.09 partition/C
module load EasyBuild-user
eb --robot="$PWD/easybuild/easyconfigs" \
   easybuild/easyconfigs/o/OpenPFC/OpenPFC-0.6.0-cpeGNU-25.09.eb

module --force purge
module load LUMI/25.09 partition/G
module load EasyBuild-user
eb --parallel=16 --robot="$PWD/easybuild/easyconfigs" \
   easybuild/easyconfigs/o/OpenPFC/OpenPFC-0.6.0-cpeGNU-25.09-rocm.eb
```

`--robot` resolves the HeFFTe dependency from this directory. Run `eb` on a
login node: the configure step fetches `nlohmann_json` and `tomlplusplus`
through CMake's FetchContent and needs outbound network. Start a long build
with `setsid nohup ... &`; a plain background job dies with the SSH session.
`--parallel=16` keeps the HIP compile reasonable on a shared login node.

## ROCm details

LUMI-EasyBuild-contrib ships HeFFTe ROCm easyconfigs for `cpeAMD` only.
OpenPFC builds with `cpeGNU`, and HeFFTe and OpenPFC must share one toolchain,
so the HeFFTe ROCm recipe here is a `cpeGNU` one mirroring
[`scripts/build_heffte_rocm.sh`](../scripts/build_heffte_rocm.sh).

`hipcc` compiles `.hip` sources without the Cray wrapper, which is what
otherwise supplies the MPICH include path. Both ROCm recipes therefore pass
`CMAKE_HIP_FLAGS=-I$MPICH_DIR/include`. Do not set `MPI_C_COMPILER=cc`
explicitly inside EasyBuild: FindMPI then interrogates the wrapper, fails to
find `MPI_C_LIB_NAMES`, and configure stops.

## Scope

What these recipes do **not** cover, deliberately:

* **The current checkout.** The recipes install the released **0.6.0** tag
  from GitHub, so a module does not include work merged after that tag. Build
  from source for that.
* **Tests.** `OpenPFC_BUILD_TESTS=OFF`, because the module build is not the
  place to run the suite. Use `scripts/build.sh`.
* **HDF5** on LUMI-G, which `scripts/build.sh` also leaves off there.

## Verified

Built into `/projappl/project_462001519/EasyBuild` and run on LUMI
(2026-10-29), with `apps/aluminumNew/inputs_json/smoke.json` from the tag:

* `HeFFTe/2.4.1-cpeGNU-25.09-{fftw,rocm}-planfix`: EasyBuild applied the
  patch, and both installed `heffte_backend_fftw.h` have SHA-256
  `c7b0d134…6788`, the corrected header.
* `OpenPFC/0.6.0-cpeGNU-25.09`: `aluminumNew` on 4 ranks on `debug`:
  `SPECTRAL_CHECKSUM sum=-24.576000000000043`, the value 0.5.0 gave.
* `OpenPFC/0.6.0-cpeGNU-25.09-rocm`: `aluminum_etd_hip` on 2 GCDs on
  `dev-g` with `mpi:from-gpu`: `sum=-24.576000000000487`, the same as
  0.5.0 there and as the 2-rank CPU and CUDA runs of 0.6.0 on Tohtori.

The 0.4.0 and 0.5.0 modules stay installed next to them, on the unpatched
HeFFTe.

With 0.5.0 and older, pre-create the case's results directory for
multi-rank runs: rank 0 creates it without a barrier (#615, fixed in 0.6.0).

## Updating for a new release

Rename the two OpenPFC easyconfigs, and change `version` and `checksums`:

```bash
curl -sSL -o openpfc.tar.gz \
  https://github.com/VTT-ProperTune/OpenPFC/archive/refs/tags/vX.Y.Z.tar.gz
sha256sum openpfc.tar.gz
```

The tag must be on VTT-ProperTune/OpenPFC; the recipes download from there.
EasyBuild's own download of that archive has failed on LUMI login nodes while
`curl` worked. Put the checksummed archive in the shared source cache first:

```bash
curl -fsSL -o /projappl/project_462001519/EasyBuild/sources/o/OpenPFC/openpfc-X.Y.Z.tar.gz \
  https://github.com/VTT-ProperTune/OpenPFC/archive/refs/tags/vX.Y.Z.tar.gz
```
If the LUMI stack moves off `25.09`, rename all files and update the
`toolchain` version in each. Keep the toolchain versions identical: HeFFTe and
OpenPFC must be built with the same one.
