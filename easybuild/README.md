<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# EasyBuild recipes

Module install of OpenPFC on LUMI, as an alternative to building from a source
checkout with [`scripts/build.sh`](../scripts/build.sh). This is the
module-productization path; the container path is issue #13 and the CLI is #41.

| Easyconfig | What it builds |
|---|---|
| [`h/HeFFTe/HeFFTe-2.4.1-cpeGNU-25.09-fftw.eb`](easyconfigs/h/HeFFTe/HeFFTe-2.4.1-cpeGNU-25.09-fftw.eb) | HeFFTe 2.4.1, FFTW backend, `cpeGNU` |
| [`o/OpenPFC/OpenPFC-0.2.0-cpeGNU-25.09.eb`](easyconfigs/o/OpenPFC/OpenPFC-0.2.0-cpeGNU-25.09.eb) | OpenPFC 0.2.0 (CPU) and its applications |

## Build

```bash
module load LUMI/25.09 partition/C
module load EasyBuild-user

eb --robot="$PWD/easybuild/easyconfigs" \
   easybuild/easyconfigs/o/OpenPFC/OpenPFC-0.2.0-cpeGNU-25.09.eb
```

`--robot` resolves the HeFFTe dependency from this directory. Both installs go
under `$HOME/EasyBuild` with the standard LUMI user setup.

## Use

```bash
module load LUMI/25.09 partition/C
module load EasyBuild-user
module load OpenPFC/0.2.0-cpeGNU-25.09

srun -n 4 tungsten case.json
```

The module puts the shipped applications on `PATH` (`tungsten`,
`aluminum_etd`, `heat3d_*`, `wave2d_*`, `kobayashi_*`, `allen_cahn`), and adds
the install prefix to `CMAKE_PREFIX_PATH` so `find_package(OpenPFC)` works for
code built against it.

> **Note**
> `module load` inside a shell pipeline runs in a subshell and its environment
> changes are lost. Load it on its own line.

## Scope

What these recipes do **not** cover, deliberately:

* **GPU.** CPU `cpeGNU` only. The HIP build stays with
  `scripts/build.sh --machine=lumi --with-rocm`; LUMI-EasyBuild-contrib ships
  ROCm HeFFTe easyconfigs for `cpeAMD` if you want to go that way.
* **The current checkout.** The recipe installs the released **0.2.0** tag from
  GitHub, so a module build does not include work merged after that tag. Build
  from source for that.
* **Tests.** `OpenPFC_BUILD_TESTS=OFF`, because the suite fetches Catch2 and
  the module build is not the place to run it. Use `scripts/build.sh`.

The build fetches `nlohmann_json` and `tomlplusplus` through CMake's
FetchContent during configure, so the machine running `eb` needs outbound
network. LUMI login nodes have it.

## Known wart: CMake 4

LUMI's `buildtools/25.09` provides CMake 4.2.3, which no longer accepts
`cmake_minimum_required` below 3.5. The `nlohmann_json` 3.11.2 that OpenPFC
pins still declares one, so the OpenPFC easyconfig passes CMake's documented
escape hatch:

```
-D CMAKE_POLICY_VERSION_MINIMUM=3.5
```

This is **not** specific to the 0.2.0 tag — the same pin is on `master`, so any
CMake 4 toolchain hits it. The real fix is to bump the fetched `nlohmann_json`,
after which the flag can be dropped here.

## Verified

Built and run on LUMI (2026-09-09):

* `HeFFTe/2.4.1-cpeGNU-25.09-fftw` — installed in 59 s.
* `OpenPFC/0.2.0-cpeGNU-25.09` — installed in 2 min 52 s; `ldd` on the
  installed `tungsten` resolves `libheffte.so.2` from the EasyBuild HeFFTe and
  FFTW from `cray-fftw`.
* `srun -n 4 tungsten case.json` on the `small` partition — a 64³ tungsten
  case ran to completion (`tungsten done t=5`).

## Updating for a new release

Change `version` and `checksums` in the OpenPFC easyconfig:

```bash
curl -sSL -o /tmp/openpfc.tar.gz \
  https://github.com/VTT-ProperTune/OpenPFC/archive/refs/tags/vX.Y.Z.tar.gz
sha256sum /tmp/openpfc.tar.gz
```

If the LUMI stack moves off `25.09`, rename both files and update the
`toolchain` version in each. Keep the two toolchain versions identical: HeFFTe
and OpenPFC must be built with the same one.
