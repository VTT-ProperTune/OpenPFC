<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# HPC operator's guide

This page is the entry point for production-style OpenPFC runs: cluster modules, MPI launchers, batch jobs, file layout, profiling and site-specific notes. It assumes you have already completed a small local or interactive run through [`../start_here_15_minutes.md`](../start_here_15_minutes.md). If that first `mpirun` does not work, fix the basic toolchain before moving to Slurm.

## Build the right binary first

Most cluster failures begin before the job is submitted. The compiler, MPI and HeFFTe used by OpenPFC must be the same stack that your job launcher sees at runtime. The canonical install instructions are in [`INSTALL.md`](../../INSTALL.md), and the one-page summary of tested and optional dependencies is [`../reference/dependency_matrix.md`](../reference/dependency_matrix.md).

Keep CPU and GPU builds in separate directories. It is tempting to flip CUDA or HIP options in an existing build tree, but the resulting cache is easy to misunderstand. [`build_cpu_gpu.md`](build_cpu_gpu.md) describes the safer pattern. If you are not sure whether GPU support is worth enabling for your run, read [`gpu_path_decision.md`](gpu_path_decision.md) before you start compiling device-enabled HeFFTe.

## Move from interactive runs to batch jobs

After an interactive `mpirun` succeeds, the next step is a minimal Slurm job. [`../tutorials/hpc_slurm_day_one.md`](../tutorials/hpc_slurm_day_one.md) shows the generic `#SBATCH` shape and the choice between `srun` and `mpirun`. LUMI-specific job layouts live under [`../lumi_slurm/README.md`](../lumi_slurm/README.md), and the LUMI install notes are in [`INSTALL.LUMI.md`](INSTALL.LUMI.md).

GPU jobs add another layer: the binary must be built with CUDA or HIP support, the HeFFTe backend must match, the JSON or TOML configuration must request a compatible backend, and the site MPI must handle device buffers correctly. [`../tutorials/gpu_app_quickstart.md`](../tutorials/gpu_app_quickstart.md) walks the application side, while [`gpu_path_decision.md`](gpu_path_decision.md) explains when the GPU path is worth the complexity.

## Treat files and MPI-IO as part of the run

OpenPFC result files are not just incidental logs. Binary field output is written through MPI-IO and has a precise layout. Before a production run, read [`mpi_io_layout_checklist.md`](mpi_io_layout_checklist.md) so paths, collectives and restart assumptions are deliberate. The raw binary format itself is documented in [`../reference/binary_field_io_spec.md`](../reference/binary_field_io_spec.md), and [`../user_guide/postprocess_binary_fields.md`](../user_guide/postprocess_binary_fields.md) explains how to inspect those files outside OpenPFC.

This is also where filesystem choice matters. A path that works on a login node may not be visible or fast on compute nodes, and rank-local scratch can silently make outputs hard to collect. Make the output directory explicit in your config and decide ahead of time whether the run is meant to produce restartable data, visualization data or only timings.

## Measure before tuning

OpenPFC has profiling hooks for application-driven runs. [`performance_profiling.md`](performance_profiling.md) explains how to enable and interpret them, and [`profiling_export_schema.md`](profiling_export_schema.md) documents the export layout. Use those traces before changing FFT plan options, GPU backend choices or decomposition assumptions.

When a run fails, separate environment failures from simulation failures. Configure, MPI and HeFFTe problems belong in [`../troubleshooting.md`](../troubleshooting.md). Symptom-oriented runtime playbooks are in [`../reference/operator_playbooks.md`](../reference/operator_playbooks.md). If the failure touches halo exchange, finite differences or FFT assumptions, [`../concepts/halo_exchange.md`](../concepts/halo_exchange.md) and [`../concepts/architecture.md`](../concepts/architecture.md) provide the conceptual background.

## Shortcuts

If you are still learning the software, [`../recipes/README.md`](../recipes/README.md) has smaller task-shaped guides. If you are trying to understand the whole cluster-user route rather than fix one job, return to [`../learning_paths.md`](../learning_paths.md) and follow the simulation-running path.
