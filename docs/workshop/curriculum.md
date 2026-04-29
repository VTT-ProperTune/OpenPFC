<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Workshop curriculum (three half-days)

Assume participants have a **cluster login** or a strong laptop with MPI + HeFFTe available (or use a prepared container/site module stack). Adjust times for your audience.

## Half-day 1 — Build, environment, and first MPI run { #workshop-day1 }

**Outcomes:** configure Release build, run `examples/05_simulator`, explain world/decomposition at a high level.

| Segment | Activity | Primary doc |
|---------|----------|-------------|
| 0:00–0:20 | Motivation: PFC, FFT scaling, where OpenPFC sits | [`README.md`](../../README.md) (intro), [`spectral_stack.md`](../concepts/spectral_stack.md) |
| 0:20–1:10 | Modules, GCC/Open MPI, HeFFTe prefix, `cmake` | [`INSTALL.md`](../../INSTALL.md), [`dependency_matrix.md`](../reference/dependency_matrix.md) |
| 1:10–1:40 | Build, `mpirun` `05_simulator`, exit codes | [`start_here_15_minutes.md`](../start_here_15_minutes.md), [`quickstart.md`](../quickstart.md) §2A |
| 1:40–2:30 | Examples ladder: `02` → `03` → `05` (instructor picks depth) | [`examples_catalog.md`](../reference/examples_catalog.md), [`getting_started/01-basics/README.md`](../getting_started/01-basics/README.md) |
| 2:30–3:00 | Q&A; homework: read [`architecture.md`](../concepts/architecture.md) layers | — |

## Half-day 2 — JSON app and artifacts { #workshop-day2 }

**Outcomes:** run **tungsten** with a repo JSON; know where binary fields go; optional VTK path.

| Segment | Activity | Primary doc |
|---------|----------|-------------|
| 0:00–0:15 | Recap: spectral stack, `App` vs bare `examples/` | [`spectral_stack.md`](../concepts/spectral_stack.md), [`applications.md`](../user_guide/applications.md) |
| 0:15–1:15 | `mpirun` tungsten sample JSON, directory layout | [`recipes/recipe_spectral_app_json.md`](../recipes/recipe_spectral_app_json.md), [`apps/tungsten/inputs_json/README.md`](../../apps/tungsten/inputs_json/README.md) |
| 1:15–2:00 | Config sections: domain, fields, `plan_options` skim | [`spectral_app_config_reference.md`](../reference/spectral_app_config_reference.md), [`configuration.md`](../user_guide/configuration.md) |
| 2:00–2:45 | Binary I/O contract; optional VTK tutorial path | [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md), [`tutorials/vtk_paraview_workflow.md`](../tutorials/vtk_paraview_workflow.md) |
| 2:45–3:00 | Q&A | [`troubleshooting.md`](../troubleshooting.md) |

## Half-day 3 — HPC and profiling { #workshop-day3 }

**Outcomes:** submit a batch job skeleton; avoid common MPI-IO mistakes; read a profiling export.

| Segment | Activity | Primary doc |
|---------|----------|-------------|
| 0:00–0:30 | Slurm vs interactive; `srun`/`mpirun` | [`hpc_operator_guide.md`](../hpc/operator_guide.md), [`tutorials/hpc_slurm_day_one.md`](../tutorials/hpc_slurm_day_one.md) |
| 0:30–1:30 | MPI-IO checklist; restarts | [`mpi_io_layout_checklist.md`](../hpc/mpi_io_layout_checklist.md), [`operator_playbooks.md`](../reference/operator_playbooks.md) |
| 1:30–2:30 | Enable profiling in JSON; interpret export | [`performance_profiling.md`](../hpc/performance_profiling.md), [`profiling_export_schema.md`](../hpc/profiling_export_schema.md) |
| 2:30–3:00 | GPU path decision (if site has GPUs) | [`gpu_path_decision.md`](../hpc/gpu_path_decision.md), [`tutorials/gpu_app_quickstart.md`](../tutorials/gpu_app_quickstart.md) |

## See also

- [`recipes/README.md`](../recipes/README.md) — copy-paste how-tos  
- [`when_not_to_use_openpfc.md`](../when_not_to_use_openpfc.md) — set expectations  
