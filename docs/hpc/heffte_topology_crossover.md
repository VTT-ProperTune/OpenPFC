<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# HeFFTe protocol crossovers and LUMI-G topology (issue #106)

Mechanism study: is the non-monotone HeFFTe ranking from #61
smooth rank/message scaling, associated with allocation/topology
regimes, or too unstable under independent allocations and randomized
protocol order?

This page is the **recipe**. It does not replace the frozen #61 archive
in [`heffte-protocol-evidence/`](heffte-protocol-evidence/README.md). Do
not mix those tables with this campaign. Do not change the production
reshape default here.

Issue: #106.

## Frozen numerical / FFT policy

Same Heat3D spectral constant-local-work construction as #61.

```text
Nx = 768
Ny = 768
Nz = 768 * P
```

Hold fixed: one rank/GCD, GPU-aware MPI, rocFFT reorder, real slabs,
min-reshape complex outbox, physics, `dt=0.01`, I/O off, warmup 1, 20
steps. The only primary variable is `HEAT3D_RESHAPE_ALG`.

## What is not assumed

The #61 768³/GCD ranking (`alltoall` at intermediate scale,
`p2p_plined` at 128–256 nodes, `alltoallv` at 512 nodes) is the
observation to explain. It is not treated as universal.

LUMI-G documentation gives 24 Slingshot electrical groups of 124 nodes
(last group 126). That size is a **candidate sampling point**, not a
verified allocation label. Jobs record hostnames, nids, and any exposed
xname/cname/CXI strings. Cabinet fields parsed from xname are not
electrical-group identifiers.

## Account and scratch

New jobs bill **`project_462001245`**. Refuse `project_462001519`.

```text
/scratch/project_462001245/juaho/openpfc-scaling/heffte-topology-106/
```

Do not write into the #61 tournament tree.

## Protocol order

Each allocation draws a deterministic permutation of

`p2p_plined`, `p2p`, `alltoallv`, `alltoall`

from seed `10676800000 + 768*1000 + nodes*10 + repeat`. The seed and
colon-separated order are stored in `run_meta.txt`.

## First wave

`standard-g` accepts 1–1024 nodes. Wave 1 samples both sides of the
documented 124-node group size and the previous ranking changes:

| neighborhood | nodes | repeats |
|--------------|-------|--------:|
| ~128 | 112, 124, 128, 136 | 3 |
| ~256 | 240, 248, 256, 264 | 3 |
| ~512 | 480, 496, 512, 528 | 3 |

Do not add 1024 nodes unless the preceding data pose a live mechanism
question.

## Submit

```bash
export HEAT3D_SPECTRAL_HIP_BIN=/path/to/heat3d_spectral_hip
./docs/lumi_slurm/submit_heffte_topology_crossover.sh check
./docs/lumi_slurm/submit_heffte_topology_crossover.sh wave1
./docs/lumi_slurm/submit_heffte_topology_crossover.sh collect
./docs/lumi_slurm/submit_heffte_topology_crossover.sh analyze
./docs/lumi_slurm/submit_heffte_topology_crossover.sh harvest
```

Optional: `NODES=124,128`, `REPEATS=3`, `DRY_RUN=1`.

Submit uses `--export=NONE` plus an explicit whitelist. The billed
account is printed and the job exits 2 if it is not
`project_462001245`.

## Analysis

Keep per-allocation wall/step, ranking, and protocol order. Pooled
medians are a summary. Local slope uses neighboring sampled node
counts, not only doublings. Close (<5%) winner/second pairs stay
inconclusive until more repeats exist.

Pre-execution communication descriptors (no wall times) live in
[`heffte_topology_crossover_descriptors.csv`](heffte_topology_crossover_descriptors.csv),
generated from the OpenPFC slab / min-reshape replica in
`apps/heat3d/scripts/heffte_comm_plan.py`. Candidate models are defined
there. Coefficients were not fit; the campaign stopped without a selector.

Observability of LUMI topology:
[what a job can actually see](heffte_topology_observability.md).
Frozen H1/H2/H3 decision tests:
[pre-analysis](heffte_topology_preanalysis.md).

```bash
./docs/lumi_slurm/submit_heffte_topology_crossover.sh descriptors
./docs/lumi_slurm/submit_heffte_topology_crossover.sh harvest
```

`harvest` writes `runs.csv` (including rejected and missing protocol
rows), `order.csv`, `allocations.csv`, `scaling.csv`, and `status.md`.
It does not modify raw run directories.

Do not fit a production selector in this issue.

## Harvested 768³/GCD rankings (2026-09-21)

Pooled median `wall_step` in seconds for completed protocols. Compact
table: [`heffte_topology_campaign_scaling.csv`](heffte_topology_campaign_scaling.csv).
Per-allocation winners stay in `status.md` on scratch. Close
winner/second pairs (<5%) are **not** a default change.

| nodes | ranks | n (best) | best | T_best (s) | second | T_2/T_1 | `p2p` / best |
|------:|------:|---------:|------|----------:|--------|--------:|-------------:|
| 124 | 992 | 3 | `alltoall` | 2.014 | `alltoallv` | 1.001 | 1.26× |
| 128 | 1024 | 3 | `p2p_plined` | 1.724 | `alltoall` | 1.145 | 2.52× |
| 136 | 1088 | 3 | `alltoall` | 1.481 | `alltoallv` | 1.012 | 1.72× |
| 264 | 2112 | 2 | `alltoall` | 1.871 | `alltoallv` | 1.018 | 2.25× |
| 512 | 4096 | 1 | `alltoallv` | 2.079 | — | — | incomplete |
| 528 | 4224 | 2–3 | `alltoallv` | 2.493 | `alltoall` | 1.001 | 3.09× |

Collectives beat blocking `p2p` at every completed scale. `alltoall`
versus `alltoallv` is unresolved at 124, 136, 264 and 528 nodes
(winner/second within 2%). All three 128-node allocations elect
`p2p_plined` (jobs 22186602, 22198879, 22213995; 1.717 / 1.740 /
1.724 s; 14% ahead of `alltoall`). Protocol order does not explain
it: r1 third, r2 third, r3 fourth. **H3 is rejected at 128 nodes.**

The 128-node complex grid is `4x256x1`; 124 is `31x32x1` and 136 is
`17x64x1`. That descriptor jump is the H1-shaped explanation of the
124↔128↔136 ranking change. **H2 is not supported:** a 128-node
electrical-group claim is forbidden by the frozen test. Do not
change the production reshape default. Do not build the #107
selector from this table (`alltoall` vs `alltoallv` stays
unresolved; 256/512 four-protocol rankings were not obtained).

240-node r1 (`22186616`) hung on the first `srun` for 2 h 7 min
with an empty `run.log` and was cancelled. Do not submit further
240+ node four-protocol sequential `srun`s. No #106 jobs remain
queued. This campaign can stop.
