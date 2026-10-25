<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Compile a model with the native CLI

`openpfc` is a native C++20 executable. It can compile a user-owned C++ model
against an installed OpenPFC SDK, optionally preserve inputs and run records,
or manage the existing catalogued app cases. The installed CLI and its
generated model projects require no Python interpreter. The CLI links the C++ and OpenSSL Crypto
runtimes; it does not load OpenPFC, MPI or CUDA merely to print help or scaffold a
case. Install OpenSSL development files (Crypto 1.1 or newer) when building it.

Use `openpfc --help` for the command list, then `openpfc compile --help`
or `openpfc help run` for command-specific options and defaults. Help works
without a configured SDK, compiler, MPI launcher or Python interpreter.

## Check the installed environment

After loading the compiler and MPI modules for an installed SDK, run:

```bash
openpfc doctor --sdk /path/to/openpfc-sdk
```

`doctor` uses the same CMake discovery and SDK compatibility checker as source
builds. It compiles, links and runs a small host probe, then reports the resolved
SDK, C++ compiler, MPI launcher, recorded SDK backends and compatibility result.
It leaves its probe files and logs in the printed diagnostics directory under
the cache. A compatible CPU result returns zero; a failed check returns one
with recovery instructions. Invalid usage or a disallowed execution context
returns two. It never installs or rebuilds dependencies.

SDK discovery matches `compile`: `--sdk`, `OpenPFC_DIR`, an SDK beside the
installed CLI, then CMake discovery through `CMAKE_PREFIX_PATH`. Options such as
`--cc`, `--cxx`, `--toolchain`, `--generator` and `--build-type` select the
environment to check; pass the same options when compiling the model.

Run `doctor` once outside an MPI launch. On LUMI use an allocation or permitted
build environment, since the probe invokes a compiler. A successful check
verifies the host build contract; it does not launch MPI ranks or certify GPU
execution. An SDK's recorded CUDA/HIP capabilities do not qualify those source
profiles. The compiled executable can run directly on one rank.

## Create an editable starter model

```bash
openpfc init my-model --model=diffusion
cd my-model
openpfc doctor
openpfc compile model.cpp -o model
./model input.json
```

Add `--sdk /path/to/openpfc-sdk` to doctor and compile when needed. The
starter can be initialized before installing an SDK; compilation and execution
require a compatible provisioned environment. The destination must be new, so
initialization never replaces edits. `--model` cannot be mixed with the
catalogued-case `--app` or `--preset` options.

The generated `model.cpp` uses public OpenPFC fields, periodic halo exchange,
finite differences and an owned-cell Euler stepper for the heat equation.
`input.json` sets the grid, diffusivity, timestep and step count; the model
validates its input and explicit stability bound. Edit `Diffusion::rhs` to change
the physics. The starter is a numerical example, not a calibrated material model.

The compiled `model` is an ordinary executable. It writes directly to
`results/` in the current working directory. Inspect `results/diagnostics.csv`
and `results/summary.json`: with the default input the mean stays near 1 and
the L2 norm decreases. Each MPI rank writes `results/field_rank_N.csv`.
Changing only `input.json` needs no compilation; repeat `./model input.json`.
After editing the source, compile again. Direct execution uses the model's
normal output paths, so retain distinct runs in separate working directories.
Use your own launcher for multiple ranks: `mpirun -n 2 ./model input.json`,
or `srun -n 2 ./model input.json` in a compatible Slurm allocation.
The generated README explains input limits and optional recorded execution.
For further library examples see
[`examples/heat/finite-differences/model.cpp`](../../examples/heat/finite-differences/model.cpp)
and the [extension guide](../extending_openpfc/README.md).

## Compile a user model

Load the pre-provisioned compiler/MPI environment and installed SDK, then run:

```bash
openpfc compile my_model.cpp -o model --sdk /path/to/openpfc-sdk
./model my_input.json
# Or use your MPI/scheduler launcher:
mpirun -n 2 ./model my_input.json
srun -n 2 ./model my_input.json
```

`-o FILE` and `--output FILE` select a regular executable pathname. Without
an output option, the default is the source filename without its extension,
in the current working directory. Its parent directory must already exist.
Compilation publishes a separate copy of the cached executable; moving or
removing your output does not alter the cache. Successful recompilation
atomically replaces an existing regular output file. A failed build leaves
it unchanged. Source dependencies, cache paths, symlinks and directories are
rejected as output destinations. `compile` never launches the model.

A model is a complete C++20 translation unit with `main(argc, argv)`. It owns
its command-line arguments, JSON interpretation, output paths and exit status.
The generated starter accepts the input pathname as `argv[1]`. Direct
execution uses your working directory, terminal and launcher; no OpenPFC run
record or automatic output redirection is involved. Keep the matching SDK
compiler/MPI runtime environment loaded. This executable is not a bundle of
its runtime dependencies; see packaging below for that use case.

`--sdk` accepts an install prefix or the directory containing
`OpenPFCConfig.cmake`. Without it, the CLI uses `OpenPFC_DIR`, an SDK next to the
installed CLI, or normal CMake discovery through `CMAKE_PREFIX_PATH`. A missing
or incompatible SDK is an error. Provision a compatible SDK before compiling;
this workflow never fetches or rebuilds OpenPFC, MPI, HeFFTe or GPU toolkits.
The [SDK compatibility contract](../reference/build_options.md#installed-sdk-build-manifest-and-compatibility)
explains the evidence and remaining unknowns.

The CLI generates a visible CMake project that calls
`find_package(OpenPFC 0.5 CONFIG REQUIRED)` and links `OpenPFC::openpfc`. CMake
owns dependency discovery and public usage requirements. The CLI launches tools
with argument vectors, without shell interpolation of filenames.

Source-model builds currently qualify CPU execution. CUDA/HIP source profiles
fail explicitly pending separate device/compiler/MPI qualification. Do not
interpret source portability as compatibility between GPU vendors or MPI stacks.
On LUMI, run model builds in a permitted build environment or allocation rather
than on a login node.

## Optional recorded execution

Use `openpfc run my_model.cpp my_input.json --launcher=none` when you want the
CLI to compile if needed, freeze declared inputs, launch the model in a fresh
attempt directory, and capture logs, exit status and output checksums. This
is an optional workflow; the compiled executable runs independently. Recorded
execution passes the frozen input pathname as `argv[1]` and expects output
under `results/`. The sections below describe its input and record contracts.

## Source and input dependencies

Local quoted includes within the source directory are snapshotted recursively.
Declare additional local headers, data needed at compilation, or translation
units with repeatable `--file relative/path`. Paths are relative to the primary
source directory. Set `--source-root` to a containing project directory when
using a layout such as `src/model.cpp` and `include/material.hpp`; declared
paths are then relative to that root. For example:

```bash
openpfc compile model.cpp -o model --file include/material.hpp --file material.cpp
openpfc run model.cpp input.json --input-file initial/field.bin --launcher=none
```

For optional recorded execution, `--input-file` paths are relative to the
input JSON's directory. Input bytes are preserved without rewriting JSON. Each attempt receives its own copies of the
input and declared data; relative references resolve from the attempt directory.
Output paths must be under `results/`. Path-like JSON strings that name inputs
must be declared; absolute paths, parent-directory escapes and source/input
symlinks are rejected. These rules bound the initial workflow: declare all
consumed data rather than reading unrelated mutable files from the model.
This recorded-execution contract preserves declared inputs; it is not a
sandbox for arbitrary C++ programs. Direct execution uses the model's own
input and output handling without these CLI path restrictions.

Build options include `--build-type=Release|Debug|RelWithDebInfo|MinSizeRel`,
`--generator`, `--toolchain`, `--cc`, `--cxx`, `--jobs` (default 2), and repeatable
`--define=NAME=VALUE`. The active compiler environment is retained. Use
`--arg=VALUE` to pass additional model arguments after the input pathname;
these arguments participate in the recorded run identity.

## Cache, runs and durable export

The cache uses `$OPENPFC_CACHE`, otherwise `$XDG_CACHE_HOME/openpfc`, otherwise
`~/.cache/openpfc`. The cache must be owned by the invoking user and must not be
writable by other users; new caches are private. It contains ordinary inspectable
directories:

```text
builds/openpfc-build-<sha256>/
  CMakeLists.txt
  model-files.cmake
  src/
  model
  build-manifest.json
  configure.stdout.log
  compile.stderr.log
runs/openpfc-run-<sha256>/
  model/
  inputs/
  run-manifest.json
  attempts/<attempt-id>/
    input.json
    results/
    stdout.log
    stderr.log
    status.json
```

Build fingerprints include the frozen source closure, generated CMake template,
SDK manifest and library evidence, actual compiler/toolchain identity and build
options. An unchanged build is a cache hit only after verifying the source,
project, executable digest and executable permissions. Changing a local header,
SDK/toolchain or build option changes the build identity. Changing only input
JSON changes the run identity without recompiling the model.

Run fingerprints add exact input/data bytes, arguments, MPI ranks, launcher and
recorded numerical runtime settings. Repeated requests get distinct attempt IDs;
previous output is preserved. Failed compilation leaves its staged project and
logs under `staging/`. Failed models retain stdout, stderr, partial results and
exit status. Output checksums are recorded after execution, separately from the
requested-run fingerprint. A fingerprint identifies the requested case, not a
promise of deterministic floating-point results.

The CLI records common OpenMP, GPU-device selection and MPI transport settings.
Add a numerically meaningful setting with `--record-env=NAME`; credential-like
names are rejected. Compiler environment values that participate in build
identity are hashed; the CLI does not dump the process environment.

The cache is disposable. To preserve a completed run or build:

```bash
openpfc export /path/to/cache/runs/openpfc-run-<sha256> /durable/new-record
```

Export refuses an existing destination, active attempts and damaged artifacts.
The saved `model/` directory, or exported build directory, is a reusable CMake
project. Copy it, make your development copy writable, and configure a fresh
build directory against a provisioned SDK. Its original build cache is diagnostic
and may contain paths to the original location.

## MPI execution

For source models, invoke the CLI once, outside an MPI launch:

```bash
openpfc run model.cpp input.json --ranks=2 --launcher=mpirun
```

`auto` selects `srun` inside a Slurm allocation and `mpirun` otherwise. Source runs
default to one explicitly selected rank; `--launcher=none` supports one rank.
The MPI executable discovered by CMake is preferred for `mpirun`. Load the
matching runtime modules in the invoking shell. The CLI propagates application
exit codes and records attempts even when the model fails.

## Existing catalogued app cases

The directory-based workflow remains available:

```bash
openpfc apps
openpfc init results/mycase --app=tungsten --preset=smoke
openpfc compile results/mycase --profile=local --source=/path/to/OpenPFC
openpfc run results/mycase --ranks=1
```

`init` refuses an existing directory and uses installed preset data. It creates
`input.json`, a version-1 `openpfc.json` manifest and a README. The catalog lists
available presets and build targets; it does not imply that every backend is
installed or qualified on this machine. Small presets are runtime or verification
examples, not calibrated material predictions.

Directory-case `compile` retains the existing `scripts/build.sh` route for shipped
applications. `local` selects CPU, `tohtori` selects CUDA, and `lumi` selects HIP
with the script's Slurm/flash policy. `--build-dir`, `--jobs`, `--no-test` and
`--dry-run` retain their meanings. Only successful builds publish the executable
selection to the case manifest. This source-checkout build route is separate
from the new user-source workflow against an installed SDK.

Case `run` uses the directory as its working directory. It accepts `--executable`
to select an installed app and detects an existing outer `mpirun`/`srun` launch,
executing directly in each rank. It rejects nested launchers. Reusing a legacy
case can replace its ordinary snapshot outputs; select fresh output paths to
preserve them. Source-model run records always use separate attempts.

From a source checkout, build through `scripts/build.sh` first.
`./scripts/openpfc` is a POSIX-sh convenience launcher for
`builds/release/bin/openpfc`; set `OPENPFC_CLI` for another build directory.
Only the native executable is installed as `bin/openpfc`.

## Package a CPU model

`openpfc bundle` packages the model produced by `compile` or `run`. The image
contains that executable, the shared libraries that are not part of the host
MPI stack, and a provenance file. Glibc and `/bin/sh` are copied from the host
into a scratch image, or supplied by `--base`. It does not contain a
compiler, the OpenPFC sources, SDK headers or Python. The model is not rebuilt
inside the image.

```bash
openpfc bundle model.cpp -o model.sif \
  --host-prefix /path/to/openmpi --host-prefix /path/to/ucx
```

Omit `--base` to build a scratch image. The image then copies the host dynamic
loader, the glibc libraries required by the model and by `/bin/sh`, and
`/bin/sh` itself. Apptainer and Singularity open `/bin/sh` to start the
runscript, so a scratch image without that shell does not run. Host MPI and
fabric libraries stay outside the image. The copied glibc is the build
machine's C library, so this image is for that same CPU host. The scratch
image does not include a passwd database, a locale archive, or the Open MPI
help text, and the runtime may warn about those.

Use the same `--sdk`, compiler and `--build-type` options as the compile that
should be packaged. `--profile=local-cpu` is the qualified profile (`local` is
accepted). `--profile=tohtori-cuda` and `--profile=lumi-g` are rejected; those
GPU profiles are not qualified. `--site=openmpi` is the default host stack.
`--site=cray` classifies Cray MPICH, libfabric and Slingshot libraries as host
libraries. That is not a ROCm qualification.

Open MPI, UCX and PMIx installations are selected with `--host-prefix`. A
fabric library that shares a directory with the C library, such as `libfabric`
or `libibverbs` in `/usr/lib64`, is not selected that way: binding the whole
directory would hide the C library in the image. Pass each such file with
`--host-lib`. The command prints the missing file. A provider directory beside
that file (`libfabric`, `libibverbs` or `hwloc`) is bind-mounted when it exists
and is not itself a system library directory. Other plugins opened only with
`dlopen` are not copied. Ordinary libraries from the same directory, such as
`libstdc++`, `libevent` and `libnuma`, are copied into the image.
`--host-prefix` must not be `/`, `/usr/lib64` or another system library
directory.

`--base` is optional. When it is set, it is a pinned local
Apptainer/Singularity image that supplies glibc and `/bin/sh` instead of
copying them from the host. Its glibc must be at least as new as the GLIBC
symbol versions required by the model and the host libraries. The CLI checks
that version when it builds the image.

`--prepare-only` writes the recipe under the cache and does not run a container
runtime. The printed directory contains `bundle.def`, `payload/` and
`run-host.sh`. A scratch recipe also contains `rootfs/`. Building an image
needs `apptainer` or `singularity` on a host that is allowed to build images.
A LUMI login node is not assumed to be able to
build one. Pass `--runtime=singularity` where the site provides SingularityCE.

The command refuses to overwrite `model.sif`, `model.sif.sha256` or
`model.sif.run`. The sidecar is the image digest. That digest is not written
into the image, because the image bytes would then depend on themselves.
`/opt/openpfc/provenance.json` inside the image records the content-addressed
`build_id`, SDK manifest digest, library closure and host/bundled
classification. The closure is the ELF `NEEDED` graph. A library that the model
opens only with `dlopen` is not copied.

`model.sif.run` is a POSIX `sh` helper. It bind-mounts the declared host MPI
prefixes read-only and executes the image. When `TMPDIR` or
`PMIX_SERVER_TMPDIR` is set, as it is under `mpirun`, the helper also
bind-mounts that directory so the process manager's session is visible.
`./model.sif input.json` passes
arguments through the runscript, and it works when the required libraries are
already visible to the container. Host MPI is not visible that way unless the
helper, or an equivalent bind, is used. An optional `--input FILE` stores one
frozen input in the image and records its digest. Calling the image with
arguments still passes those arguments to the model.

Catalogued app cases are a different packaging path. The
[legacy CPU runtime baker](../../containers/runtime/README.md) still packages a
compiled tungsten case with `scripts/bake_case.py`.
