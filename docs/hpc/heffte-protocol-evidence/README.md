<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Frozen HeFFTe tournament evidence

This archive preserves 137 protocol records: 136 runtime-admitted observations
and one failed record. Runtime admission, allocation compliance and matching
the frozen measurement protocol are separate conditions. Sixteen historical
records billed account519; eight more used 105 steps and five warmup steps.
These remain in `all-runs.csv` but are excluded from `campaign-runs.csv`.
The matched cohort contains 113 records: 112 admitted and one failed.
The archived campaign used `project_462001245` for its matched cohort.
These account labels describe historical records, not current submission
instructions; use the [LUMI guide](../INSTALL.LUMI.md) for new builds and runs.

The corrected analysis retains node and rank counts separately and applies
warmup to absolute timestep labels. The executable already removes its warmup
steps; the earlier collector erroneously dropped one additional frame. Both
reductions are retained in each compact timing record. Correcting this changes
individual medians by at most 1.39%, without changing the observed per-scale
winners. Longer references are not described as numerically invalid; their
measurement protocol differs from the frozen comparison.

## Bounded observations

The generated [scaling table](scaling.md) gives every matched point and the
range of allocation-level repetitions. For the 768³/GCD family at 512 nodes
(4096 GCDs), `alltoallv` has pooled rank/frame median 2.107624 s versus
2.419094 s for `p2p_plined`. Both compliant repeats preserve that ordering.
At 128 nodes, `p2p_plined` is faster (1.662606 s versus 1.999056 s).
At 32 nodes, `alltoall` is fastest in the 768 family, while `p2p_plined`
is fastest in the independent 512 family. These nonmonotone, workload-dependent
rankings do not support a universal node-count switch or a new production
default. No selector is fitted here.

One frozen binary is used throughout. The reported within-run value is the
median of pooled rank/timestep samples after warmup, followed by the median
of allocation repetitions. Rank samples are correlated, not independent
replicates. The printed MPI_MAX total elapsed time is a different statistic.
The single-GCD reference has only one matching allocation per family; do not
interpret derived weak efficiencies as high-precision universal constants.

The global domain is a slab: Nx=Ny equals the family edge, Nz equals that
edge times MPI ranks. A 768³/GCD workload means a local inbox of that size,
not a cubic global domain. Protocols execute in fixed sequential order within
an allocation; order effects have not been excluded. Two large-scale repeats
are useful evidence, not held-out production-policy qualification.

## Reproduction and omitted data

`records.json.gz` contains the original metadata, batch recipes, logs and
admission decisions, deduplicated by SHA256, plus exact reduced timing inputs.
Each omitted full timing profile has its byte count and SHA256. The extraction
independently verifies exported step ranges, all positive finite samples,
rank counts, central order statistics and scheduler account identities.
Original and corrected warmup reductions remain side by side.

```sh
python3.11 apps/heat3d/scripts/archive_protocol_evidence.py \
  --replay docs/hpc/heffte-protocol-evidence
```

This regenerates all CSVs and the scaling table using committed compact inputs.
It verifies archived text hashes and reduction arithmetic, but cannot prove
that stored central values occupy their claimed order without full profiles.
The original extraction performed that check against every full profile.
`provenance.json` supplies exact source/binary identities, runtime stack and
commands for new measurements; timing replay is scientifically equivalent,
not promised bitwise identical. The complete original build log is retained.
Failed runs and multiple-libSCI runtime warnings are preserved, not discarded.

A future policy experiment must freeze held-out scales and address protocol
order. This archive changes no production FFT default or inverse numerics.
