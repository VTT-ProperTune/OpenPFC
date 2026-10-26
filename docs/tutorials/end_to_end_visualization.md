<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: end-to-end run and artifacts

Start from a working compiler, MPI and HeFFTe stack, then run a small case and
inspect its output before increasing the grid or rank count. The
[installation guide](../../INSTALL.md) covers dependencies; the
[first-model walkthrough](../start_here_15_minutes.md) is the shorter route
when you already have an installed SDK.

## Build the project

From the repository root on Tohtori:

```bash
./scripts/build.sh --machine=tohtori --cpu
module load openmpi/5.0.10
```

The script builds and tests in `builds/release`. Its module changes apply only
inside its process, so load the matching MPI stack in the launch shell too.
Use `--machine=portable` on a workstation with dependencies installed. On
LUMI follow the [scheduled build and run guide](../hpc/INSTALL.LUMI.md).

The runs below assume an allocation providing the requested MPI ranks.
Keep generated files under `results/`, or under scratch for large HPC runs.

## Track A: PNG snapshots from Allen–Cahn

This small 2-D program takes command-line parameters rather than JSON:

```bash
mkdir -p results/allen-cahn
mpirun -n 1 ./builds/release/apps/allen_cahn/allen_cahn 256 256 5000 0.005 8.0 0.75 0.25 \
  results/allen-cahn/initial.png results/allen-cahn/final.png
```

A successful run exits zero and produces the initial and final grayscale
images, with a fixed scale from −1 to 1. Its internal growth check can reject
a case; use the [application README](../../apps/allen_cahn/README.md) for the
parameter meanings and expected behaviour. These pictures are a quick visual
check, not a convergence study.

## Track B: tungsten binary field dumps

Prepare the small single-seed case from the checkout root:

```bash
mkdir -p results/first-tungsten
cp apps/tungsten/inputs_json/tungsten_single_seed.json \
  results/first-tungsten/input.json
```

Edit the copied input's `fields[]` entries: set the `psi` data path to
`results/first-tungsten/psi_%d.bin` and `psiMF` to
`results/first-tungsten/psimf_%d.bin`. The shipped file contains absolute paths
from its original site. Retain a positive `saveat` in `timestepping`.

Run the CPU executable:

```bash
mpirun -n 1 ./builds/release/apps/tungsten/tungsten \
  results/first-tungsten/input.json
```

Check exit zero, the progress log and the expected binary frames. A `%d`
filename suffix is a frame ordinal. Retain the input's domain, field names and
save cadence with the data; raw arrays have no header. See the
[binary format](../reference/binary_field_io_spec.md) before decoding them.
A field dump alone is not a complete restart bundle; the
[checkpoint protocol](../development/checkpoint_publish.md) has a separate
accepted-state and metadata contract.

For VTK output choose `"writer": "vtk"` and a `.vti` data template for each
field. For a programmatic series that records geometry, steps and time,
continue with [VTK and ParaView](vtk_paraview_workflow.md) and the
[result-series lesson](https://vtt-propertune.github.io/OpenPFC/learn/output/vtk-series/).

## What to read next

| Goal | Document |
|---|---|
| Compare application inputs and backends | [Applications](../user_guide/applications.md) |
| Understand configuration and the run lifecycle | [Application pipeline](../user_guide/app_pipeline.md) |
| Follow runnable examples by topic | [Examples catalog](../reference/examples_catalog.md) |
| Understand which model produced a figure | [Showcase](../user_guide/showcase.md) |
