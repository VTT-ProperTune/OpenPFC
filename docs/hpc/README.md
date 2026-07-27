<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# HPC and cluster operation

This section covers production-style runs on MPI clusters. Complete a small
single-rank run before moving to scheduler, filesystem, or GPU-specific setup.

Start with the [HPC operator guide](operator_guide.md).

## Generic cluster guidance

| Topic | Document |
|------|----------|
| CPU or GPU path selection | [GPU path decision](gpu_path_decision.md) |
| Separate CPU, CUDA, and HIP build trees | [CPU and GPU build trees](build_cpu_gpu.md) |
| MPI-IO paths and collective layout | [MPI-IO layout checklist](mpi_io_layout_checklist.md) |
| Runtime instrumentation | [Performance profiling](performance_profiling.md) |
| Profiling output contract | [Profiling export schema](profiling_export_schema.md) |
| First batch submission | [Slurm day one](../tutorials/hpc_slurm_day_one.md) |

## Site-specific guidance

| Site | Document |
|------|----------|
| LUMI software and ROCm environment | [LUMI installation](INSTALL.LUMI.md) |
| LUMI scheduler examples | [LUMI Slurm guide](../lumi_slurm/README.md) |
| VTT Tohtori modules and paths | [Tohtori installation](INSTALL.tohtori.md) |

Site pages may name current modules and filesystem paths. Generic API, CMake,
and configuration contracts belong in the main install guide and reference
pages rather than being copied into site notes.

## Before a production run

Verify that:

- OpenPFC and HeFFTe use the same compiler and MPI stack;
- the launcher visible inside the job matches the linked MPI implementation;
- CPU and device-enabled binaries come from separate build trees;
- output paths are visible and appropriate on compute nodes;
- the selected FFT backend matches the OpenPFC and HeFFTe build;
- a small representative run succeeds before scaling to the target allocation.

Use [Troubleshooting](../troubleshooting.md) for build and runtime failures and
[Operator playbooks](../reference/operator_playbooks.md) for symptom-oriented
checks.
