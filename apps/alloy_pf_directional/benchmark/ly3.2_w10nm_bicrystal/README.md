<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# LOCKED starting point: \(12\times 3.2\,\mu\mathrm{m}\) bicrystal, \(W_0=10\,\mathrm{nm}\)

This is the **alloy_pf_directional** starting point: a finished cellular–dendritic front in a two-grain Al–Cu FTA strip. **Locked** — do not retune FTA physics, Zhong \(\omega(T)\), or the two-grain IC without recording a new gold (figures + this file + `reference_meta.txt`). Last-bit identity is **not** this case; that is CLI `repro`.

LUMI-C reference job: `g3e6_V04_Ly3.2_bicrystal_pm30_w0_10nm` (OpenMP, Sep 2026). Fields are not in git. After a fetch or a new run they belong in

`results/alloy_pf_directional/benchmark/ly3.2_w10nm_bicrystal/`

(with a `reference/` copy of the LUMI `.raw` + `meta.txt`).

## Contract (frozen by default CLI / `start` / `benchmark`)

| Quantity | Value |
|---------|--------|
| \(W_0\) | \(10\,\mathrm{nm}\) |
| \(\Delta x/W_0\) | 1 |
| \(L_x\times L_y\) | \(12\times 3.2\,\mu\mathrm{m}\) → \(1200\times 320\) |
| \(N_z\), \(n_\mathrm{dim}\) | 1, 2 |
| \(G\) | \(3\times 10^6\,\mathrm{K/m}\) |
| \(V_p\) | \(0.4\,\mathrm{m/s}\) |
| \(\theta\) | \(\pm 30^\circ\) |
| \(\Delta t/\tau_0\) | 0.2 |
| Seed depth | \(0.20\,\mu\mathrm{m}\) (radius shrunk to keep \(16 W_0\) liquid gap) |
| Noise \(F_0\) | **0** (this gold; the noisy 2× ensemble is a separate campaign) |
| Stop | far-wall solute (`STOP_FAR_C`), not the right wall |
| \(t_\mathrm{end}\) cap | \(120\,\mu\mathrm{s}\) (LUMI stopped earlier on `wall_c`) |
| Periodic \(y\) | yes |
| Periodic \(z\) | no (\(N_z=1\)) |
| Glasner + Ji iso | on |
| Zhong \(\omega(T)\) | on |

`OPENPFC_ALCU_*` leftover from other campaigns cannot change this CLI (re-applied after env). `OPENPFC_ALCU_MAX_STEPS` still caps Euler steps (`QUICK=1` in `run_benchmark.sh`).

## LUMI reference outcome

See [`reference_meta.txt`](reference_meta.txt). In short: \(n=33457\), \(t=72.9\,\mu\mathrm{s}\), \(x_\mathrm{tip}=11.94\,\mu\mathrm{m}\), `abort_reason wall_c`. A new compiler should reproduce a **recognizable** two-grain cellular–dendritic strip that stops the same way; do not expect last-bit `sum_phi` vs Cray vs AppleClang.

## Run

```bash
# Full length (~33k steps; minutes on a laptop, production on LUMI-C)
./apps/alloy_pf_directional/scripts/run_benchmark.sh

# Laptop smoke of the same box (seeds only — not the morphology gold)
QUICK=1 ./apps/alloy_pf_directional/scripts/run_benchmark.sh

# Figures from existing fields (LUMI reference or a finished run)
./apps/alloy_pf_directional/scripts/run_benchmark.sh --plot-only
```

## What “locked” means

1. Visual match to the strip + front figures (two grains, mid-plane GB, primary trunks, secondary arms, Cu in grooves).
2. Same stop (`wall_c`), \(x_\mathrm{tip}\) within a few cells of \(11.94\,\mu\mathrm{m}\).
3. Unrelated: `check_repro.sh` last-bit on the tiny noisy slice; `check_bicrystal_repro.sh` last-bit on a **capped** noisy twin of this box.

Do not change FTA physics, IC, or the Zhong mapping until (1)–(2) still hold, or record a new gold with figures and `reference_meta.txt`.
