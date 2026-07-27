<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Scalability analysis plan

This page defines the minimum information needed for a reproducible OpenPFC
scalability study. It is a plan and reporting contract, not a claim that every
combination below has already been benchmarked.

Use [Performance profiling](performance_profiling.md) for instrumentation and
the [HPC operator guide](operator_guide.md) for cluster preparation.

## Questions

A scaling study should answer:

1. How does time per accepted simulation step change as resources increase for
   a fixed global grid?
2. How stable is time per step when the local grid size is held approximately
   constant?
3. Which fraction of time is spent in FFT, communication, model evaluation,
   output, and synchronization?
4. At what resource count do communication or memory constraints dominate?
5. Do CPU, CUDA, and HIP results compare equivalent algorithms, precision, and
   output settings?

## Required metadata

Record the following with every result set:

| Category | Required information |
|----------|----------------------|
| OpenPFC | Commit SHA or release tag; relevant CMake options |
| Toolchain | Compiler, MPI implementation, HeFFTe version and backend |
| Machine | System, partition, node type, CPU/GPU model, memory per node |
| Launch | Nodes, MPI ranks, threads per rank, affinity and binding settings |
| Problem | Global grid, spacing, model, time step, number of measured steps |
| FFT | Backend, decomposition and planner options, GPU-aware MPI setting |
| I/O | Writers enabled, output cadence, filesystem and output location |
| Measurement | Warm-up steps, timed steps, repetition count and aggregation |

Store the effective configuration file and batch script next to the results.

## Strong scaling

Keep the global grid and physical/numerical configuration fixed while
increasing the resource count.

Suggested procedure:

1. Select a problem large enough that the smallest allocation is not dominated
   by startup time.
2. Disable result output during the timed interval or report I/O separately.
3. Run enough warm-up steps to establish FFT plans, allocations, and caches.
4. Measure an identical number of accepted steps for every allocation.
5. Repeat each point at least three times when allocation cost permits.
6. Report median time per step and the spread between repetitions.

For a baseline resource count `p0`, define speedup and parallel efficiency as:

```text
speedup(p)    = time(p0) / time(p)
efficiency(p) = speedup(p) * p0 / p
```

State whether `p` means CPU cores, MPI ranks, GPUs, nodes, or another resource.
Do not mix these units within one curve.

## Weak scaling

Keep the local work per rank or accelerator approximately constant while the
global grid and resource count grow.

Document the local grid target and the actual decomposition at every point.
Distributed FFT decompositions may prevent perfectly identical local shapes, so
report the inbox and outbox ranges or their extrema rather than assuming an
ideal partition.

The primary weak-scaling metric is time per accepted step normalized to the
smallest allocation. Also report memory usage per rank or accelerator when
available.

## Measurement boundaries

Separate these phases when possible:

- process and device initialization;
- FFT plan construction;
- model initialization and operator setup;
- warm-up time steps;
- measured time steps;
- result output and checkpointing;
- final reductions and shutdown.

For long production simulations, time per step is usually the most useful
steady-state metric. For short workflows, startup and plan construction can be
material and should be reported rather than hidden.

## Profiling breakdown

Use OpenPFC profiling to report at least:

- total measured step time;
- FFT time;
- communication and halo-exchange time when used;
- model or gradient evaluation time;
- result-writer time;
- synchronization or barrier time introduced by measurement.

The export contract is documented in
[Profiling export schema](profiling_export_schema.md).

## Correctness checks

Performance points are valid only when the run remains scientifically and
numerically comparable. Verify:

- identical initial conditions or recorded random seeds;
- the same precision and model parameters;
- consistent accepted time-step sequence;
- equivalent FFT normalization and backend behavior;
- matching result checksums, norms, or tolerance-based observables at selected
  steps;
- no silent change in output, validation, or NaN-check settings.

## Reporting

For each figure or table include:

- the metadata listed above;
- raw timing samples, not only averages;
- the baseline used for speedup;
- error bars or min/median/max values;
- resource-hours per simulated step or per target physical time when useful;
- a short explanation of the observed scaling limit.

Keep raw measurements in a machine-readable format and generate plots from a
script. Do not treat values copied from an image or an old README as the
benchmark source of truth.

## See also

- [Performance profiling](performance_profiling.md)
- [GPU path decision](gpu_path_decision.md)
- [MPI-IO layout checklist](mpi_io_layout_checklist.md)
- [Documentation versioning](../development/documentation_versioning.md)
