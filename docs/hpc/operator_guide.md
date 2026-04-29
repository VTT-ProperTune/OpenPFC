<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# HPC operator’s guide (runbook index)

Use this page as a **single entry point** for production-style runs: batch systems, MPI launchers, I/O, profiling, and site-specific notes. It does not replace the linked guides — it **orders** them for operators.

## 1. Toolchain and build

| Topic | Document |
|--------|-----------|
| Modules, GCC, Open MPI, HeFFTe install | [`INSTALL.md`](../../INSTALL.md) |
| One-page dependency / CI tooling summary | [`dependency_matrix.md`](../reference/dependency_matrix.md) |
| CPU vs separate GPU build trees | [`build_cpu_gpu.md`](build_cpu_gpu.md) |

## 2. Job launcher and allocation

| Topic | Document |
|--------|-----------|
| Generic Slurm skeleton (`#SBATCH`, `srun` / `mpirun`) | [`tutorials/hpc_slurm_day_one.md`](../tutorials/hpc_slurm_day_one.md) |
| LUMI Slurm examples and job layout | [`lumi_slurm/README.md`](../lumi_slurm/README.md) |
| GPU jobs and JSON `backend` | [`gpu_path_decision.md`](gpu_path_decision.md), [`tutorials/gpu_app_quickstart.md`](../tutorials/gpu_app_quickstart.md) |

## 3. Filesystems, paths, and MPI-IO

| Topic | Document |
|--------|-----------|
| Paths, collectives, restart sanity | [`mpi_io_layout_checklist.md`](mpi_io_layout_checklist.md) |
| Binary field format (no header) | [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md) |
| Offline analysis | [`postprocess_binary_fields.md`](../user_guide/postprocess_binary_fields.md) |

## 4. Performance and debugging

| Topic | Document |
|--------|-----------|
| Profiling hooks and export | [`performance_profiling.md`](performance_profiling.md), [`profiling_export_schema.md`](profiling_export_schema.md) |
| Symptom → fix playbooks | [`operator_playbooks.md`](../reference/operator_playbooks.md) |
| Configure / MPI / HeFFTe failures | [`troubleshooting.md`](../troubleshooting.md) |
| Halo / FFT / FD interactions | [`halo_exchange.md`](../concepts/halo_exchange.md), [`architecture.md`](../concepts/architecture.md) |

## 5. Quick recipes

| Topic | Document |
|--------|-----------|
| First `mpirun` from clone | [`start_here_15_minutes.md`](../start_here_15_minutes.md) |
| Named how-tos | [`recipes/README.md`](../recipes/README.md) |

## See also

- [`personas.md`](../development/personas.md) — “I run on a cluster” track  
- [`learning_paths.md`](../learning_paths.md) — sequenced HPC user path  
