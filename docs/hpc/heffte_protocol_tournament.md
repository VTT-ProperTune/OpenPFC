<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# HeFFTe reshape-algorithm tournament (issue #61)

Controlled LUMI-G weak-scaling campaign: which HeFFTe reshape algorithm
scales best at large MPI rank counts?

This page is the **recipe**. It does not replace the admitted 16-GCD
reshape A/B or the production `p2p_plined` weak curve in
[`lumi_gpu_scaling.md`](lumi_gpu_scaling.md). Do not mix those tables
with this campaign.

Issue: #61.

## Question

How do `p2p_plined`, `p2p`, `alltoallv`, and `alltoall` scale as the
rank count becomes large, and does another protocol overtake production
`p2p_plined`?

Do not assume a winner. Do not invent a crossover.

## Frozen factors

One `heat3d_spectral_hip` binary per comparison group. Hold source SHA,
binary SHA256, HeFFTe, rocFFT, Cray MPICH, one rank per GCD, GPU-aware
MPI, production `use_reorder`, slabs, min-reshape complex-outbox
planner, CCD bind on exclusive `standard-g`, physics, `dt=0.01`, I/O
off, warmup 1, 20 steps.

The only primary variable is `HEAT3D_RESHAPE_ALG`.

Do not change the OpenPFC production FFT default until this evidence
exists.

## Account and scratch

New jobs bill **`project_462001245`**. Refuse `project_462001519`.

```text
/scratch/project_462001245/juaho/openpfc-scaling/heffte-protocol-tournament/
```

Job 22166458 inherited a leaked login environment and was not an
algorithm A/B. The tournament sbatch always unsets `HEAT3D_*` /
`OPENPFC_FFT_*` knobs, then sets only `HEAT3D_RESHAPE_ALG`. Submit uses
`--export=NONE` plus an explicit whitelist. The resolved banner must
show the intended `reshape=` (and production `use_pencils=0`,
`use_reorder=1`, `gpu_aware=1`) or the timing is rejected.

## Families

Local inbox stays `N³` on every rank by growing only the split axis:

```text
Nx = N
Ny = N
Nz = N * P
```

| family | `N` | node ladder |
|--------|-----|-------------|
| main | 768 | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 |
| communication-heavy | 512 | 1, 2, 4, 8, 16, 32, 64, 128 (extend if useful) |

8 MPI ranks per node. Optional 1-GCD references. Do not mix weak
efficiencies across families.

Poor efficiency is a result. Stop a protocol only for an operational
failure (MPI/library/memory/wall-time/allocation). Record the failure.

## Same-allocation comparison

Each job allocates the node count once and runs, in order:

1. `p2p_plined`
2. `p2p`
3. `alltoallv`
4. `alltoall`

Independent repeats: at least two jobs at 32, 64, 128, 256, and 512
nodes (three if ranking is close or a crossover appears). Report the
median, not the fastest repeat.

## Submit

```bash
export HEAT3D_SPECTRAL_HIP_BIN=/path/to/heat3d_spectral_hip
./docs/lumi_slurm/submit_heffte_protocol_tournament.sh check
./docs/lumi_slurm/submit_heffte_protocol_tournament.sh 1gcd
./docs/lumi_slurm/submit_heffte_protocol_tournament.sh 768
# after the 768 family is established:
./docs/lumi_slurm/submit_heffte_protocol_tournament.sh 512
./docs/lumi_slurm/submit_heffte_protocol_tournament.sh collect
./docs/lumi_slurm/submit_heffte_protocol_tournament.sh analyze
```

Optional: `NODES=1,2,4`, `REPEATS=2`, `DRY_RUN=1`.

Collect rebuilds CSV from run directories. Do not commit raw profiles.

## Production decision

Only after the curves exist:

* keep universal `p2p_plined`; or
* switch the universal default; or
* add a simple scale/message-dependent rule, and only if a crossover is
  clear and reproducible on held-out points.

Do not hard-code a table of benchmark sizes.

Timing collection applies the warmup threshold to each frame's absolute `step`
scalar. The HIP exporter already removes earlier steps, so the collector must
not remove that many frames again by list position. Legacy profiles without
step labels use zero-based frame positions and are assumed untrimmed. The
reported within-run statistic pools retained rank/frame samples; it is not
the slowest-rank total elapsed time printed by the executable.

The [frozen evidence archive](heffte-protocol-evidence/README.md) retains the
completed observations, account/protocol exclusions and replayable reductions.
