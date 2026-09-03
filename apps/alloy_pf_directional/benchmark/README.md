<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# alloy_pf_directional 2D cases

The **starting point** for this app is the locked full-length two-grain strip. The 40-step CI slice is not that product.

| Case | Role | How to run |
|------|------|------------|
| [`ly3.2_w10nm_bicrystal/`](ly3.2_w10nm_bicrystal/) | **LOCKED** starting point: \(12\times 3.2\,\mu\mathrm{m}\), \(W_0=10\,\mathrm{nm}\), noise off | `./apps/alloy_pf_directional/scripts/run_benchmark.sh` or `alloy_pf_directional_openmp` / `start` / `benchmark` |
| CLI `repro` | Last-bit OpenMP check (\(128\times 64\), 40 steps, \(F_0=10^{-3}\)) | `./apps/alloy_pf_directional/scripts/check_repro.sh` / `ctest -R alloy-pf-directional-repro` |

Raw fields live under `results/alloy_pf_directional/benchmark/` (gitignored). Do not commit `.raw` files.
