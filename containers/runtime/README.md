<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Bake a CPU case with host MPI

`scripts/bake_case.py` creates an Apptainer runtime image from an already
compiled tungsten case. The image contains its executable, input, CLI, and
non-MPI runtime dependencies. MPI, PMIx, and UCX remain on the host and are
mounted read-only by the generated `run-host.sh`. The application is not
recompiled inside the image.

Two host MPI stacks are supported, selected with `--site`:

| `--site` | stack | tested on |
|---|---|---|
| `openmpi` (default) | Open MPI / PMIx / UCX | Tohtori, single node |
| `cray` | Cray MPICH / libfabric / Slingshot | LUMI, **4 nodes / 32 ranks** |

It is a runtime image, not the development image or a GPU profile.

## Prepare and bake

Create and compile a case with the [CLI](../../docs/user_guide/cli.md). Load
the same compiler/MPI modules before baking so `ldd` resolves the libraries
used by the executable. On the tested Tohtori CPU stack:

```bash
./scripts/openpfc init results/mycase
./scripts/openpfc compile results/mycase --profile=local
module load openmpi/5.0.10

mkdir -p builds/containers
apptainer pull --disable-cache builds/containers/python.sif \
  docker://python:3.12-slim-bookworm

python3 scripts/bake_case.py results/mycase \
  --base builds/containers/python.sif \
  --output builds/containers/mycase \
  --host-prefix /share/apps/OpenMPI/5.0.10 \
  --host-prefix /share/apps/openpmix/7.0.0 \
  --host-prefix /share/apps/ucx/1.17
```

The base is an existing Python 3.8+ SIF image. Its glibc must support the
application and host MPI binaries. The example uses Python 3.12 on Debian
Bookworm with a Tohtori EL8-built application. Retain the downloaded SIF:
the registry tag can change, while the baker records the exact base SHA-256.

The output directory must not already exist. `--prepare-only` creates the
recipe and payload without invoking Apptainer. Building a definition file
requires a working site Apptainer/fakeroot setup. If a build fails after
staging, inspect the payload and retry from that directory:

```bash
cd builds/containers/mycase
apptainer build --disable-cache case.sif runtime.def
```

No `--force` is used: an existing image is not replaced. All C++ compilation
and its tests remain under `scripts/build.sh` through `openpfc compile`.

## Run the baked input

```bash
builds/containers/mycase/run-host.sh --version
builds/containers/mycase/run-host.sh --provenance
builds/containers/mycase/run-host.sh init results/baked-run

# Single-node CPU smoke with explicit shared-memory/TCP transport:
mpirun --mca pml ob1 --mca btl self,sm,tcp -n 2 \
  builds/containers/mycase/run-host.sh run results/baked-run
```

`init` copies the baked input into a new writable case. `run` uses that case
as its working directory. The image remains immutable; fields and checkpoints
go under the case's `results/`. The host launcher starts one container per
MPI rank. No MPI launcher is shipped inside the runtime image. For a singleton
check, call `run-host.sh run CASE` with the same host MPI mounts available.

`case.sif` and `run-host.sh` can be moved together to another location. On
another machine, the MPI ABI and host paths still need to match; inspect
`--provenance` and adapt the wrapper's bind paths for that site's stack.
The wrapper preserves both symlink and canonical spellings of site prefixes,
since MPI can embed original paths for plugins and help files.

The explicit Open MPI transport flags above are a smoke configuration.
They do not validate the site's RDMA path. Production networking can require
additional provider directories, `/etc/libibverbs.d`, devices, and matching
libfabric/PMI libraries. Do not infer multi-node scaling from this smoke run.

## LUMI (`--site cray`)

LUMI differs from Tohtori in four ways that each break a naive port, so they
are encoded in the site profile rather than left to the operator.

**1. The base image needs a new enough glibc.** The bind model loads *host*
libraries inside the image, so the base's glibc must be at least as new as
theirs. On LUMI, Cray MPICH, libfabric and libcxi all require `GLIBC_2.38`:

```console
$ objdump -T /opt/cray/pe/lib64/libmpi_gnu_123.so.12 | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -1
GLIBC_2.38
```

The Debian Bookworm base used for Tohtori ships glibc 2.36 and **cannot** work
here; the symptom is an unhelpful symbol error at run time. Use a Trixie or
Ubuntu 24.04 base. The baker records the measured floor as
`host_glibc_requirement` in `provenance.json`.

**2. Slingshot libraries share a directory with ordinary system libraries.**
`libcxi.so.1` and `libxpmem.so.0` live in `/usr/lib64` next to `libstdc++.so.6`
and `libz.so.1`. A prefix rule cannot separate them, and mounting the whole
directory would shadow the base image's own libraries. Name them individually
with `--host-lib`; the wrapper binds each file to `/opt/openpfc/hostlib/<soname>`
and never mounts the shared directory. Omitting one is an error, not a silent
bake of the interconnect into the image.

**3. Cray PMI needs the Slurm daemon's socket directory.** The site profile
binds `/var/spool/slurmd`. Without it `MPI_Init` fails with `job id unknown`
and every rank believes it is rank 0.

**4. LUMI cannot build images.** There is no `/etc/subuid` mapping for ordinary
users and no readable `proot`, so `singularity build` fails whether or not
`--fakeroot` is given. Bake on a build host and copy `case.sif` and
`run-host.sh` to LUMI — which is the workflow #13 describes anyway. LUMI ships
`singularity-ce`, not `apptainer`, hence `--runtime singularity`.

```bash
python3 scripts/bake_case.py CASE \
  --site cray --runtime singularity \
  --base BASE_WITH_GLIBC_2.38_OR_NEWER.sif \
  --output builds/containers/mycase \
  --host-prefix /opt/cray \
  --host-prefix /opt/rocm-6.3.4 --host-prefix /opt/amdgpu \
  --host-lib /usr/lib64/libcxi.so.1 \
  --host-lib /usr/lib64/libxpmem.so.0
```

Use `/opt/cray`, not `/opt/cray/pe/lib64`: the libraries there are symlinks into
`/opt/cray/pe/mpich/...`, and the baker resolves them before classifying.

The case directory is not bound by the wrapper, so point `SINGULARITY_BIND` at
the filesystem holding it. On LUMI that means both spellings, because
`/flash/project_*` is a symlink into `/pfs`:

```bash
export SINGULARITY_BIND=/pfs,/flash
srun -n 32 builds/containers/mycase/run-host.sh run "$CASE"
```

### Multi-node validation

`small` partition, **4 nodes x 8 ranks = 32 ranks**, tungsten 64^3 for 5 steps.
Both runs report `MPICH CH4 OFI netmod using cxi provider (domain_name=cxi0)`,
so this exercises the Slingshot RDMA path and not a TCP fallback:

| run | job | `sum` (hex) | `sumsq` (hex) |
|---|---|---|---|
| native | 21830651 | `-0x1.9e04f59f6c34p+13` | `0x1.04d0a9db35c98p+17` |
| baked bundle | 21830651 | `-0x1.9e04f59f6c34p+13` | `0x1.04d0a9db35c98p+17` |

Bit-identical. An earlier 2-node run (21830470 native, 21830489 container) agrees
the same way. This closes the "host-MPI bind on more than one node" item of #13.

The baked-bundle run above bound `payload/` into the base image rather than
using a built `case.sif`, because LUMI cannot build one (point 4). It exercises
the generated `run-host.sh` bind list, the payload/host split and the
entrypoint; the SIF packaging step itself is covered by the Tohtori path.

## Bundle contents and provenance

The staging directory contains `runtime.def`, `payload/`, `base.sif` (a symlink
to the retained base), `run-host.sh`, and the built `case.sif`. The image has:

```text
/opt/openpfc/
  bin/                  tungsten, openpfc, baked-case
  lib/                  non-MPI dependencies resolved at bake time
  case/                 baked input and portable project manifest
  share/openpfc/version
  provenance.json
```

Provenance records hashes of the executable, input, base image, bundled
libraries, and host-provided MPI stack, plus the OpenPFC version. The
`base_provided_glibc` records identify host glibc libraries intentionally
excluded from the payload; their hashes describe the build host, not the
glibc bytes in the base image. The base hash identifies the latter image.
The checkout revision/dirty flag is observed at packaging time and is
explicitly not proof of which source produced the binary. Keep the original
source revision, build logs, and toolchain records with scientific releases.

The first baker accepts self-contained constant/single-seed/seed-grid inputs
and relative output paths under `results/`. It rejects file-based initial
conditions and restart inputs because their extra files are not yet bundled.
It rejects GPU profiles and unresolved dependencies. MPI/PMIx/UCX libraries
must fall under explicit host prefixes, preventing accidental MPI-in-image
packaging. The base supplies glibc; the payload includes the compiler runtime,
HeFFTe, FFTW, and other resolved non-MPI dependencies.

## Verification and remaining work

On Tohtori g0005, Apptainer 1.5.1 built the CPU image and printed OpenPFC 0.2.0.
A two-rank 16³ / 10-step baked run produced the same field sum and
sum-of-squares as the native CLI run. Both generated VTK output and a step-10
checkpoint. The final step-10 `psi.bin` also matches the native file byte-for-byte
(SHA-256 `963b4981d6090be7e4d65eed8c1c8926b7a33c43e435a037ac53af57d573ce88`).
With automatic home/CWD mounts disabled, `ldd` resolves HeFFTe, FFTW, and the
compiler runtime from `/opt/openpfc/lib`, and MPI/PMIx/UCX from the declared
host mounts. The tested SIF is approximately 60 MiB.

Tests of the baker invoke its CLI with controlled dependency
resolution and check payload exclusion, hashes, input portability, and
non-overwrite behavior. They do not require Apptainer privileges.

Remaining issue #13 work includes a development image, exact build-source
provenance, CUDA/HIP image profiles, and baking on a host that can build the
SIF for LUMI. No container has been published to a registry, so the "download
an image" entry point of the user workflow does not exist yet.

The runtime follows Apptainer's documented
[MPI bind model](https://apptainer.org/docs/user/main/mpi.html#bind-model) and
[definition-file build workflow](https://apptainer.org/docs/user/latest/definition_files.html).
