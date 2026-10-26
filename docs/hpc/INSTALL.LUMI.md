<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Building OpenPFC on LUMI-G (HeFFTe + HIP / ROCm)

LUMI-G uses AMD MI250X GPUs and HIP/ROCm. Use the repository build script for
source builds; it keeps the module stack, HeFFTe backend and Slurm allocation
consistent. The general [installation guide](../../INSTALL.md) describes the
dependencies and installation options.

## Storage and account

Use your own authorized allocation and writable directories. Set these in
your submission shell; keep personal values in local configuration:

```bash
: "${HPC_ACCOUNT:?Set HPC_ACCOUNT to your authorized Slurm allocation}"
: "${FLASH:?Set FLASH to your writable flash directory}"
: "${SCRATCH:?Set SCRATCH to your writable scratch directory}"
export OPENPFC_BUILD_DIR="$FLASH/build/openpfc-rocm"
export OPENPFC_RUN_ROOT="$SCRATCH/openpfc"
mkdir -p "$OPENPFC_RUN_ROOT/logs"
```

Keep the source checkout separate from the build tree, and put large
simulation output under a run-specific directory on scratch. Existing
private execution records are not installation defaults.

## Build and test

From the maintained source checkout:

```bash
git pull --ff-only
./scripts/build.sh --machine=lumi --with-rocm \
  --account="$HPC_ACCOUNT" --partition=standard-g \
  --build-dir="$OPENPFC_BUILD_DIR" \
  --wait
```

The script configures on the login node because dependency downloads require
network access. It submits compilation and tests to `standard-g`, where AMD
devices are available. `--wait` returns when the job finishes; omit it to leave
the job queued and inspect the printed job ID and log paths. `--partition=dev-g`
selects the shorter development queue. Do not compile or run simulations on the
login node.

The script loads `LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath` and the
`heffte-rocm` module from `$HOME/privatemodules`. To inspect the launch environment:

```bash
module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
module use "$HOME/privatemodules"
module load heffte-rocm
module list
```

HeFFTe must be built with ROCm and GPU-aware MPI, installed outside the clone,
and discoverable through that module. For another install, set `HEFFTE_MODULE`
or `HEFFTE_PREFIX` as described by `./scripts/build.sh --help`. A CPU-only
HeFFTe install cannot serve a HIP spectral build.

## GPU jobs and verification

Cray MPICH needs `MPICH_GPU_SUPPORT_ENABLED=1` to accept device buffers. Load
the same stack for the run as for the build. The following single-node check
runs inside a GPU allocation and writes its log under scratch:

```bash
sbatch --account="$HPC_ACCOUNT" --partition=standard-g \
  --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=00:10:00 \
  --output="$OPENPFC_RUN_ROOT/logs/gpu-mpi-%j.out" \
  --wrap='export MPICH_GPU_SUPPORT_ENABLED=1; srun --cpu-bind=cores "$OPENPFC_BUILD_DIR/apps/tungsten/verify_gpu_aware_mpi"'
```

This assumes the modules above are loaded in the submission shell. Success
prints `[verify_gpu_aware_mpi] OK`. A failure can indicate missing GPU support,
a mismatched MPI stack or incorrect binding. The check exercises communication;
it is not a scalability or physical-model validation result.

For application runs, MI250X nodes have eight graphics compute dies (GCDs).
The maintained device applications select a device using the node-local MPI
rank. Follow the [LUMI distribution and binding guide](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/)
when changing the layout; do not mask peer GCDs independently of the
application's device-selection contract. The [first Slurm job tutorial](../tutorials/hpc_slurm_day_one.md)
explains resource requests, job status and run directories.

## Diagnosing the toolchain

If HIP translation units cannot find `mpi.h`, inspect the include path supplied
by the Cray C++ wrapper after loading the selected programming environment:

```bash
CC -E -Wp,-v -xc++ /dev/null 2>&1 | grep mpich
```

For a custom stack, pass that include directory with
`--cmake-arg=-DCMAKE_HIP_FLAGS=-I/path/to/mpich/include`. Do not copy a versioned
MPICH path from another programming environment.

Keep `lumi-CrayPath` last in the module order. If the runtime resolves a different
ROCm installation, inspect `LD_LIBRARY_PATH` and use the module stack's libraries:

```bash
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

See the [LUMI programming environment guide](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/)
for Cray wrappers, HIP and GPU-aware MPI. OpenPFC's HIP executables use rocFFT
through the compiled HIP stack; setting `backend = "cuda"` in an input does not
select AMD hardware.

## Released modules (no build)

Everything above builds OpenPFC from a source checkout, which is the path for
development. Project members who only run a release load it as a module from
the project's shared EasyBuild tree:

```bash
: "${OPENPFC_EASYBUILD_ROOT:?Set OPENPFC_EASYBUILD_ROOT to the shared EasyBuild directory}"
export EBU_USER_PREFIX="$OPENPFC_EASYBUILD_ROOT"
module load LUMI/25.09 partition/C
module load OpenPFC/0.5.0-cpeGNU-25.09
srun -p small -n 4 tungsten case.json
```

On LUMI-G load `partition/G` and `OpenPFC/0.5.0-cpeGNU-25.09-rocm`, which adds
the HIP applications and sets `MPICH_GPU_SUPPORT_ENABLED=1`. Installing a new
release, the GPU binding and the recipe limits are in
[`easybuild/README.md`](../../easybuild/README.md).

## Further reading

- [Installation](../../INSTALL.md) — general dependencies and package installation.
- [CPU and GPU build trees](build_cpu_gpu.md) — backend separation.
- [Troubleshooting](../troubleshooting.md) — compiler, MPI and FFT failures.
- [LUMI documentation](https://docs.lumi-supercomputer.eu/) — current queues, storage and programming environments.
