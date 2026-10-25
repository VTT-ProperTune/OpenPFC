<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# LUMI-G Heat3D FD weak scaling (issue #25)

Constant **256³ owned interior cells per GCD** for `heat3d_fd_hip` on LUMI-G.
This is not the fixed-global-grid strong curve in
[lumi_gpu_scaling.md](lumi_gpu_scaling.md).

## Frozen protocol

| Knob | Value |
|------|--------|
| App | `heat3d_fd_hip` |
| FD order | 2 (halo width 1) |
| I/O | off |
| Timed steps | 105 total, `HEAT3D_WARMUP=5` (100 admitted) |
| `dt` | 0.01 |
| Decomposition | `OPENPFC_FD_PROC_GRID` |
| Local gate | `HEAT3D_REQUIRE_INTERIOR=256x256x256` |
| Metric | median barriered `wall_step` |
| Scratch | `/scratch/project_462001519/juaho/openpfc-scaling/heat3d-fd-weak/` |

Do not set `HEAT3D_DIAG_TIMING` on admitted clean runs.

| Nodes | GCDs | proc grid | global grid |
|------:|-----:|-----------|-------------|
| 1 | 8 | `2x2x2` | `512x512x512` |
| 2 | 16 | `2x2x4` | `512x512x1024` |
| 4 | 32 | `2x4x4` | `512x1024x1024` |
| 8 | 64 | `4x4x4` | `1024x1024x1024` |
| 16 | 128 | `4x4x8` | `1024x1024x2048` |

## Submit (LUMI login)

```bash
python3 apps/heat3d/scripts/fd_weak_ladder.py --check
export HEAT3D_HIP_BIN=/flash/project_462001519/juaho/build/<tree>/apps/heat3d/heat3d_fd_hip
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh clean
# after the clean series, attribution (max rank-local median):
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh diag
export HEAT3D_HALO_BIN=/flash/project_462001519/juaho/build/<tree>/examples/23_halo_microtiming
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh halo
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh collect
```

`HEAT3D_DIAG` headline times are the **max across rank-local medians**,
matching the barriered clean `wall_step`. `halo_minmax` / `rhs_minmax` /
`update_minmax` are the spread.

## Admitted FD-2 result

Every admitted rank owns exactly `256^3`. `gpu_aware=1`, `contiguous=1`,
no field I/O. Clean production path: blocking halo → FD RHS → Euler →
device sync.

Clean barriered `wall_step` (100 admitted samples, SHA `d5b7e856`, dirty=0,
binary `d7a9338bdc431b8456c173da9d0f550ff3902933a315389b945bb0769d3362e5`):

| nodes | GCDs | proc grid | wall/step | weak efficiency | job |
| ----: | ---: | --------- | --------: | --------------: | ----: |
|     1 |    8 | 2x2x2     |  0.914 ms |           1.000 | 22151601 |
|     2 |   16 | 2x2x4     |  0.908 ms |           1.007 | 22151602 |
|     4 |   32 | 2x4x4     |  0.941 ms |           0.971 | 22151603 |
|     8 |   64 | 4x4x4     |  1.060 ms |           0.863 | 22151604 |
|    16 |  128 | 4x4x8     |  1.024 ms |           0.892 | 22151605 |

HIP-event attribution (`HEAT3D_DIAG_TIMING=1`, not admitted wall time;
`reduce=max_rank_median`; SHA `d12eeb4e`, binary
`b09d9b59be6234e25683b63fd7c2d55093522cc5392ce9b669f769c339d94913`)
and halo-only `23_halo_microtiming --hip` (binary
`636090f527f9707509c5ab9a2aa5c2292132918fd6518a104479c9e506e8c299`):

| nodes | clean | halo | RHS | update | halo-only | off-node/rank | jobs |
| ----: | ----: | ---: | --: | -----: | --------: | ------------: | ---- |
|     1 | 0.914 | 0.206 | 0.382 | 0.329 |     0.211 |             0 | 22151983 / 22151991 |
|     2 | 0.908 | 0.199 | 0.382 | 0.330 |     0.204 |             1 | 22151984 / 22151992 |
|     4 | 0.941 | 0.221 | 0.382 | 0.330 |     0.294 |             2 | 22151986 / 22151993 |
|     8 | 1.060 | 0.284 | 0.384 | 0.330 |     0.350 |             3 | 22151987 / 22151994 |
|    16 | 1.024 | 0.313 | 0.383 | 0.331 |     0.330 |             3 | 22151989 / 22151995 |

Times in milliseconds. Placement is uniform: every rank at a given node
count has the same off-node-face count. RHS and update stay flat
($+0.2\%$ and $+0.5\%$ from 1 to 16 nodes). Halo grows $+52\%$
($0.206\to 0.313$ ms) and accounts for the $+0.110$ ms clean-step
increase. The two-node Slingshot transition (one off-node face per rank)
does not increase wall time. Component sum / clean wall is 1.00, 1.00,
0.99, 0.94, 1.00; the diagnostic path is not the admitted timer.

Issue #25 H1 (halo/network) is supported. H2 is not primary: halo-only
time is not flat. H3 is not supported.

Issue #48 tests whether that blocking halo can be hidden. The admitted
#25 curve stays on `HEAT3D_HALO_OVERLAP=0`. Production `heat3d_fd_hip`
now defaults to two-stream overlap (`=1`) after the `standard-g` A/B
below. Mode `2` (`MPI_Testall`) is not the production default. Overlap
must not replace the clean barriered `wall_step`.

### Overlap A/B (not admitted; six-launch border)

Same protocol, SHA `2174c960`, binary
`cef9b6099bf910707f76b73c4afb72bd7eaac846b2b79d1cedc668aeb49209a2`.
`HEAT3D_HIP_CHECKSUM_HEX` bitwise identical on all jobs
(`sum_u=0x1.7785970621d7cp+3`, `sumsq_u=0x1.4ac44f14c882ep-1`).

Clean median `wall_step`:

| nodes | ov=0 | ov=1 | ov=2 | jobs |
| ----: | ---: | ---: | ---: | ---- |
|     2 | 0.907 | 0.945 | 0.960 | 22161584 / 86 / 87 |
|     4 | 0.937 | 0.950 | 0.961 | 22161588 / 89 / 90 |

Diagnostic `HEAT3D_OVERLAP` (ms, max rank-local median; jobs
22161679–84). Blocking halo/RHS/update match the admitted #25 split.
Mode 1 exposed wait ≈ inner kernel (H-progress-2: GPU-aware `Waitall`
does not complete while the default-stream interior kernel runs).
Mode 2 `MPI_Testall` drops exposed wait to ~0.060 ms, but six thin
border launches cost ~0.127 ms versus ~0.381 ms for the full RHS, so
clean wall stays worse than blocking.

### Fused one-launch border (not admitted)

SHA `3bbde34c`, binary
`662f3112c9948fe4f30f3f4ec94469ebaf6052c2b9827b184b1cbc84e2e7f3b5`.
Checksum HEX unchanged. Clean median `wall_step`:

| nodes | ov=0 | ov=1 | ov=2 | jobs |
| ----: | ---: | ---: | ---: | ---- |
|     2 | 0.904 | 0.910 | 0.923 | 22162094 / 95 / 96 |
|     4 | 1.001 | 0.915 | 0.926 | 22162097 / 98 / 99 |

4-node blocking 1.001 repeated as job 22162121; treat as jitter relative
to the #25 0.941 ms point, not as a fuse regression (blocking does not
run the border kernel).

Fused diagnostics (jobs 22162115–20), ms:

| nodes | mode | post | exposed wait | inner | border |
| ----: | ---- | ---: | -----------: | ----: | -----: |
|     2 | 1 | 0.076 | 0.419 | 0.364 | 0.094 |
|     2 | 2 | 0.076 | 0.061 | 0.362 | 0.093 |
|     4 | 1 | 0.078 | 0.418 | 0.363 | 0.094 |
|     4 | 2 | 0.079 | 0.061 | 0.362 | 0.094 |

Border fell 0.127 → 0.094 ms. Inner+border is still 0.456 vs 0.381 full
RHS. Mode 2 critical path remains post+inner+wait+border+update ≈
0.92 ms, so overlap is not a production default.

### Two-stream interior (not admitted; `dev-g`)

SHA `ec4d74ec`, binary
`6fe3b2804bf3acdabd026f894849d69df7a0a794c2ae6217f7c4a9198b680ecf`.
Interior launches first on `hipStreamNonBlocking`; pack/MPI stay on the
default stream. `standard-g` was Priority-blocked; these runs used
`dev-g` (no CCD `mask_cpu`). Checksum HEX unchanged.

Clean median `wall_step`:

| nodes | ov=0 | ov=1 | ov=2 | jobs |
| ----: | ---: | ---: | ---: | ---- |
|     2 | 0.948 | **0.865** | 0.904 | 22162296 / 97 / 22162310 |
|     4 | 1.703 | **0.898** | 0.901 | 22162334 / 35 / 22162356 |

2-node overlap-1 is 8.8% faster than blocking on the same partition.
4-node blocking 1.703 ms is uniform across ranks (not a single
straggler); overlap hides that allocation's communication. Do not treat
`dev-g` 4-node blocking as the #25 0.941 ms point.

Mode 1 Waitall exposed wait fell 0.419 → 0.241 ms once inner left the
default stream (job 22162311). Mode 2 `MPI_Testall` (job 22162357)
exposed wait 0.538 ms and is not the better 2-stream policy.

### Two-stream interior (`standard-g` CCD bind)

Same binary as the `dev-g` table
(`6fe3b2804bf3acdabd026f894849d69df7a0a794c2ae6217f7c4a9198b680ecf`).
Checksum HEX unchanged. Clean median `wall_step`:

| nodes | ov=0 | ov=1 | Δ | jobs |
| ----: | ---: | ---: | --: | ---- |
|     1 | 0.910 | **0.859** | −5.6% | 22162434 / 22162435 |
|     2 | 0.908 | **0.860** | −5.3% | 22162371 / 22162372 |
|     4 | 0.938 | **0.866** | −7.6% | 22162427 / 22162428 |
|     8 | 1.005 | **0.868** | −13.6% | 22162444 / 22162445 |
|    16 | 1.036 | **0.876** | −15.4% | 22162463 / 22162464 |
|    32 | 1.044 | **0.886** | −15.1% | 22162481 / 22162482 |
|    64 | 1.146 | **0.944** | −17.6% | 22167340 / 22167339 |
|   128 | 1.155 | **0.960** | −16.9% | 22167439 / 22167438 |

1–16 nodes used binary `6fe3b280…`. 32-node used a rebuild `c01edbba…`
of the same protocol (`4x8x8` / `1024x2048x2048`). 64- and 128-node
used production `aa757ee4…` with default `HEAT3D_HALO_OVERLAP=1`.
Blocking matches the #25 pins through 16 nodes. Overlap-1 is nearly
flat from 1 to 32 nodes (0.859–0.886 ms) while off-node faces/rank
stay at three. At 64 nodes every rank has **four** off-node faces:
the `8x8x8` x-line fills a node, so y and z faces all leave the node.
Overlap-1 is 0.944 ms (91% vs $T_1=0.859$ ms); blocking is 1.146 ms.
128-node `8x8x16` keeps four faces on all 1024 ranks: overlap-1 is
0.960 ms (89%); blocking is 1.155 ms. That 64→128 step is +16 µs at
fixed face count, so the remaining limit is off-node topology, not a
node-count drip. Weak efficiency vs overlap-1 $T_1$: 1.00 / 1.00 /
0.99 / 0.99 / 0.98 / 0.97 / 0.91 / 0.89. HEX checksum unchanged
(`sum_u=0x1.7785970621d7cp+3`). Diagnostic `HEAT3D_OVERLAP` on the
same production binary (not headline): 8-node three-face exposed wait
0.289 ms (job 22167570) vs 64-node four-face 0.322 ms (22167571);
post stays 0.149 ms. The extra face is exposed MPI, not interior
work. Do not submit 256 nodes. Do not replace the admitted blocking
table.

### Persistent MPI (A6; not implemented)

Device `HaloExchange` still rejects `persistent`. The remaining
overlap-1 1→32 node loss is +27 µs (0.859 → 0.886 ms) and tracks the
three off-node faces, not per-step `MPI_Isend`/`Irecv` setup. At 64
nodes the extra off-node face (3→4) accounts for the further 0.886 →
0.944 ms step. 128-node `8x8x16` keeps four faces and only adds 16 µs
(0.960 ms). Diagnostic `start()` post was 76 µs and already overlaps the
interior kernel on the two-stream path. Persistent requests would
only replace init+post with `MPI_Startall`; they do not shrink packed
bytes or Slingshot latency. Host persistent Faces is already in-tree
and is red on multi-rank LUMI. Do not add a device persistent path
without a 2-node A/B that beats overlap-1 on clean wall by more than
jitter.
