<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Finite-difference order × halo-overlap scaling (issue #108)

Mechanism study: as Heat3D FD order increases, do wider halos hurt
large-scale weak scaling more than extra stencil work helps two-stream
overlap, or vice versa?

This is not an accuracy comparison. Do not change production FD physics
from these jobs.

Issue: [#108](https://github.com/ahojukka5/OpenPFC/issues/108).

## Frozen weak-scaling protocol

Keep exactly `256³` owned interior cells / GCD. Production two-stream
halo overlap. One rank/GCD. I/O off. `dt=0.01`. Wall/step is 1--8 ms,
so cheap rungs run enough timed steps for a stable median; the JSON
profile is capped near the 1024-node 105-step size (~440 MB):

| nodes | steps | warmup | timed | clean repeats |
|------:|------:|-------:|------:|--------------:|
| 1 | 5005 | 50 | 4955 | 3 |
| 8 | 5005 | 50 | 4955 | 3 |
| 32 | 3005 | 50 | 2955 | 3 |
| 128 | 805 | 20 | 785 | 3 |
| 512 | 205 | 10 | 195 | 3 |
| 1024 | 105 | 5 | 100 | 3 |

Process grids follow the admitted FD-2 construction (double the
smallest axis, z then y then x on ties):

| nodes | ranks | proc grid | global |
|------:|------:|-----------|--------|
| 1 | 8 | `2x2x2` | `512³` |
| 8 | 64 | `4x4x4` | `1024³` |
| 32 | 256 | `4x8x8` | `1024×2048×2048` |
| 128 | 1024 | `8x8x16` | `2048×2048×4096` |
| 512 | 4096 | `16x16x16` | `4096³` |
| 1024 | 8192 | `16x16x32` | `4096×4096×8192` |

Grids are exported as `OPENPFC_FD_PROC_GRID=gx x gy x gz` because
`sbatch --export` splits on commas. The batch script then calls
`srun --export=ALL` so module-loaded `LD_LIBRARY_PATH` reaches the
ranks. Without that override, `--export=NONE` is inherited and the
binary dies 127 on `libfabric.so.1` (job 22186610).

## Orders

`2, 4, 8, 12, 20`. Halo width is `order/2`. Geometric packed-face bytes
per rank are `6 × 256² × width × 8`. If an order fails for a real
numerical or software reason, record the failure rather than substituting
another order.

If a higher order is unstable at `dt=0.01`, keep wall/step on the frozen
timestep and treat time-to-solution as a separate question.

## Account and scratch

The archived jobs billed `project_462001245`. The paths below describe that
study; new builds and runs follow the [current LUMI guide](INSTALL.LUMI.md).

```text
/scratch/project_462001245/juaho/openpfc-scaling/fd-order-108/
```

Do not mix with the #25 FD-2 tree or with issue #106.

## First wave

Clean production matrix: every declared order × the node ladder.
Independent repeats (3) at every node count — 1/8/32-node jobs are
milliseconds per step, so extra allocations are cheap. 1-node and 8-node
clean cells share one packed allocation each so the matrix does not
trip `AssocMaxSubmitJobLimit`. Packed export values use `:` not `,`
because `sbatch --export` splits on commas. On `dev-g` the packed job
does not apply the `standard-g` CPU mask. Component diagnostics
(`HEAT3D_DIAG_TIMING=1`) only at 8, 128, and 1024 nodes.

## Submit

```bash
export HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip
./docs/lumi_slurm/submit_fd_order_scaling.sh check
./docs/lumi_slurm/submit_fd_order_scaling.sh clean
./docs/lumi_slurm/submit_fd_order_scaling.sh diag
./docs/lumi_slurm/submit_fd_order_scaling.sh collect
./docs/lumi_slurm/submit_fd_order_scaling.sh analyze
```

Optional: `NODES=8,128`, `ORDERS=2,8`, `REPEATS=1`, `DRY_RUN=1`.

Clean jobs unset overlap/diagnostic knobs so the binary keeps the
production two-stream default. Diagnostic jobs set
`HEAT3D_KEEP_OVERRIDES=1` and `HEAT3D_DIAG_TIMING=1` only.

## Geometry and overlap model (pre-measurement)

Packed-face halo bytes follow the production `Connectivity::Faces` pack:
six faces of `256² × width` doubles. Corners ride more than one face.

The generated table is
[`fd_order_geometry.csv`](fd_order_geometry.csv). Interior work counts
are from `EvenCentralD2` (three axial applies, not a hardware FLOP
counter).

Production two-stream path (`heat3d_fd_hip` overlap mode 2):

```text
T_step ≈ T_post + max(T_inner, T_network_progress)
         + T_exposed_wait + T_border + T_update
```

| term | diagnostic field |
|------|------------------|
| `T_post` | `HEAT3D_OVERLAP post_s` |
| `T_inner` | `HEAT3D_OVERLAP inner_s` |
| `T_exposed_wait` | `HEAT3D_OVERLAP exposed_wait_s` |
| `T_border` | `HEAT3D_OVERLAP border_s` |
| `T_update` | `HEAT3D_DIAG update_s` |
| `T_network_progress` | not timed; hidden when `inner_s > exposed_wait_s` |
| `T_step` | `wall_step` |

Qualitative discrimination, no numerical winners:

- **H1** (width-dominated): large-scale weak efficiency degrades
  monotonically with order; `exposed_wait` tracks face bytes.
- **H2** (arithmetic/overlap-dominated): some higher order keeps weak
  efficiency at least as high as FD-2 because extra interior work hides
  the wait.
- **H3** (non-monotone): best large-scale efficiency sits strictly
  between FD-2 and FD-20.

```bash
./docs/lumi_slurm/submit_fd_order_scaling.sh geometry
./docs/lumi_slurm/submit_fd_order_scaling.sh harvest
```

## Analysis

Compare clean wall/step and weak efficiency across orders at the same
node count. Attribute with halo bytes/rank, exposed MPI wait, interior
and boundary kernels, and off-node faces. Look for monotone degradation,
improved scaling at moderate/high order, or a non-monotone optimum.

## Harvested production wall/step (2026-09-21)

Fail-closed harvest first returned zero rows because the batch script
copied `fd_placement.txt` onto itself under `set -e` and skipped
`admit.txt`. The GPU jobs had already written checksums and
`heat3d_profile.json`. Harvest job 22198421 recovered 24/25 rows from
those artifacts (`reason=recovered_missing_admit`). Compact table:
[`fd_order_campaign_scaling.csv`](fd_order_campaign_scaling.csv).

Clean production median `wall_step` after warmup, halo overlap mode 1,
frozen `dt=0.01`. Weak efficiency is versus that order's 1-node
5005-step median. Do not mix diagnostic-mode overlap timings with this
table. Do not change production FD physics from these jobs.

Wall/step (ms):

| order | 1 n=3 | 8 n=3 | 32 | 128 | 512 n=3 | 1024 n=3 |
|------:|------:|------:|---:|----:|--------:|---------:|
| 2 | 0.891 | 0.893 | 0.884 n=3 | 0.974 n=3 | 1.002 | 1.025 |
| 4 | 1.321 | 1.587 | 1.595 n=3 | 1.827 n=3 | 1.874 | 1.932 |
| 8 | 1.844 | 2.403 | 2.479 n=2 | 2.913 n=2 | 3.060 | 3.127 |
| 12 | 2.603 | 3.427 | 3.700 n=2 | 4.212 n=3 | 4.392 | 4.523 |
| 20 | 4.184 | 5.728 | 6.265 n=3 | 7.120 n=3 | 7.403 | 7.726 |

Weak efficiency:

| order | 1 | 8 | 32 | 128 | 512 | 1024 |
|------:|--:|--:|---:|----:|----:|-----:|
| 2 | 1.000 | 0.997 | 1.008 | 0.914 | 0.889 | 0.869 |
| 4 | 1.000 | 0.833 | 0.828 | 0.723 | 0.705 | 0.684 |
| 8 | 1.000 | 0.767 | 0.744 | 0.633 | 0.603 | 0.590 |
| 12 | 1.000 | 0.760 | 0.703 | 0.618 | 0.593 | 0.575 |
| 20 | 1.000 | 0.730 | 0.668 | 0.588 | 0.565 | 0.542 |

1024-node geometry is $4096\times4096\times8192$ on 8192 ranks
(`16\times16\times32`), local $256^3$. FD-20 512-node is n=3
(jobs 22198927--22198929, 7.403 ms, weff 0.565). FD-8 32/128 and
FD-12 32 are n=2. Wall/step weff is monotone in order at every
completed scale.

## Harvested diagnostic overlap (2026-09-21)

Admitted `HEAT3D_DIAG_TIMING=1` jobs at 8 and 1024 nodes
(22187785--22187789, 22187795--22187799). Compact table:
[`fd_order_overlap_diag.csv`](fd_order_overlap_diag.csv).
Times below are milliseconds per step.

| order | face B | 8-node wait | 8-node inner | 1024 wait | 1024 inner |
|------:|-------:|------------:|-------------:|----------:|-----------:|
| 2 | 3.15 MB | 0.296 | 0.438 | 0.388 | 0.425 |
| 4 | 6.29 MB | 0.410 | 0.567 | 0.739 | 0.568 |
| 8 | 12.6 MB | 0.781 | 0.903 | 1.428 | 0.904 |
| 12 | 18.9 MB | 1.164 | 1.328 | 2.323 | 1.331 |
| 20 | 31.5 MB | 2.224 | 2.177 | 3.851 | 2.183 |

At 1024 nodes `exposed_wait` tracks face bytes (ratios 1.00 / 1.90 /
3.68 / 5.99 / 9.92 versus bytes 1 / 2 / 4 / 6 / 10). Interior work is
almost independent of node count. Wait exceeds inner for FD-4 and
above at 1024 nodes; FD-2 still hides. **H1** (width-dominated) is
supported. **H2** is rejected: no higher order keeps weak efficiency
with FD-2. **H3** is rejected: weff is monotone in order. Production
`fd_order` is unchanged. 128-node diagnostics were never submitted;
the 8 vs 1024 endpoints already decide the gate.
