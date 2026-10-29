<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Native runtime images

`openpfc bundle` is the packager for C++ source models and compiled catalogued
CPU app cases. It inventories ELF dependencies without executing the model,
keeps MPI and fabric libraries on the host, records hashes and GLIBC symbol
requirements, and writes an inspectable recipe before invoking Apptainer or
Singularity. No Python interpreter is required in the image or to run the CLI.

The maintained commands and compatibility limits are in the
[CLI guide](../../docs/user_guide/cli.md#package-a-cpu-model).

## Migrating a compiled case

The Python build-time baker and Python runtime entry point have been retired.
Use the native CLI from the current OpenPFC build or installation:

```bash
openpfc bundle CASE -o case.sif \
  --host-prefix /path/to/openmpi --host-prefix /path/to/ucx
./case.sif.run --version
./case.sif.run --provenance
./case.sif.run init fresh-case
mpirun -n 2 ./case.sif.run run fresh-case
```

`CASE` must already be compiled with `openpfc compile CASE --profile=local`.
Pass each required fabric library outside the prefixes with `--host-lib FILE`;
the command reports missing host declarations. System library directories must
not be bound over the image's glibc. The companion `.run` helper uses stable
library names present on the launching node. Move it together with the SIF.

`-o` now names the image file, rather than a new staging directory. The CLI
stores `bundle.def`, `payload/`, optional scratch `rootfs/`, launch helper,
manifests and build logs under `$OPENPFC_CACHE`. `--prepare-only` stages these
files without a container runtime. Without `--base`, the image contains the
host loader, required glibc libraries and `/bin/sh`; an optional pinned local
base can provide those instead. A Python base image is unnecessary.

Case images preserve `--version`, `--provenance`, `init CASE` and `run CASE`.
The native runtime is `openpfc image`, invoked by the generated runscript.
Source-model images continue to pass their arguments directly to the model.
The image and sidecars are never overwritten, and `init` requires a fresh case
directory. Case inputs are frozen in the image; copied inputs may be edited.
Restart inputs, file-based initial or boundary conditions and output paths outside
`results/` are rejected. The CLI records observed legacy compile metadata and
binary hashes; it cannot establish missing original build provenance.

## Target compatibility

The qualified native profile remains `local-cpu`. `--site=cray` supplies Cray
MPICH/libfabric/Slingshot classification and the Slurm PMI socket bind; it does
not qualify a LUMI GPU image. `tohtori-cuda` and `lumi-g` remain unqualified.
LUMI provides SingularityCE, so recipes intended for that runtime use
`--runtime=singularity` and an external `srun`. Build images on a compatible
host allowed to build SIFs; do not assume LUMI login nodes provide that ability.

Earlier Python-baker Tohtori and LUMI measurements describe the retired flow;
they do not qualify the native case runtime on LUMI. New native GPU and fast
cross-node network qualification belongs to the remaining work in #636.
Per-release distribution images remain a separate scope (#612).
