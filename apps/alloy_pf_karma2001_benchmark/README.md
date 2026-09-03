<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# alloy_pf_karma2001_benchmark

Frozen Karma (2001) alloy phase-field **present-model** benchmark. This tree
does not share headers or objects with other alloy phase-field apps.
Binary: `alloy_pf_karma2001_benchmark_openmp`.

Protocol of [Karma, PRL 87, 115701 (2001)](https://doi.org/10.1103/PhysRevLett.87.115701):
\(A=0\), \(\beta_0=0\), \(\varepsilon_k=0\), \(k=0.15\), \(\varepsilon_c=0.02\),
isothermal \(\Omega=0.55\), seed \(R=22\,d_0\). The box is \(L/d_0=1000\) so the
solute field stays off the far Neumann walls through \(t^*=10^4\). Capillary
anisotropy is cubic \(a_s(\mathbf{n})\); Glasner \(\psi\) and Ji isotropic
stencils are on unless a case turns them off.

Tip speed on Fig. 1 is a **centered** least-squares \(\mathrm{d}r/\mathrm{d}t\)
over \(\Delta t^*=80\) (not the causal 10-point / \(8\Delta x\) window stored in
the engine `V` column). The quantitative checks are **late \(V^*\) for \(t^*\ge 5000\)** (default run to \(t^*=10^4\))
and a **flat \(c_s/c_l^0\approx k=0.15\)** along \([100]\) in the grown solid (behind the
seed). Fig. 1 is clipped at \(V^*=0.08\) like the PRL. `QUICK=1` (\(t^*=80\)) is only a
pipeline check: the tip has barely left the seed, so there is no velocity steady
state and no \(c_s\) plateau.

Digitized paper curves: `data/karma2001_fig1_present.tsv` and
`data/karma2001_fig2_present.tsv` (from arXiv [cond-mat/0103289](https://arxiv.org/abs/cond-mat/0103289)
page 4). `data/karma2001_fig1_present_v1.tsv` is a superseded earlier trace.

## Default suite (all \([100]\))

| Role | \(d_0/W\) | \(\Delta x\) | \(\Delta t\) | Numerics |
|---|---:|---|---|---|
| Fast Glasner | 0.277 | \(W_0\) | \(0.09\,\tau_0\) | Glasner + Ji; max stable with local \(e^u\) |
| Thicker \(W\) | 0.544 | \(W_0\) | \(0.09\,\tau_0\) | same; late \(V^*\) closer to the paper \(\sim 0.0179\) |
| 2001-like mesh | 0.277 | \(0.4\,W_0\) | \(0.008\,\tau_0\) | no Glasner, 5-pt, \(\tau\) frozen at \(e^u=1\) |

History is dumped so \(\Delta t^*\lesssim 8\) between samples. The Fig. 1
estimator is a centered least-squares \(\mathrm{d}r/\mathrm{d}t\) over
\(\Delta t^*=80\). On \([100]\) with \(\Delta x=W_0\) the remaining \(V^*\)
wiggle is **grid pinning** (unchanged if \(\Delta t\) is halved or quartered).
Late smoothed \(\langle V^*\rangle\) (last 10% of each run) of the three cases
agrees to about 9% of the mean; the two smoother cases (\(d_0/W=0.544\) and
\(\Delta x=0.4\,W_0\)) agree to about 1%. The mean itself is already independent
of \(\Delta t/\tau_0\) from 0.02 to 0.09 on this scheme.

`OPENPFC_KARMA_DT` is \(\Delta t/\tau_0\) **at** \(\Delta x/W_0=1\), then
\(\Delta t=(\Delta t/\tau_0)\cdot(\Delta x/W_0)\cdot\tau_0\).

## Build

Always go through `./scripts/build.sh` at the repo root on Tohtori/LUMI (see
`AGENTS.md`). On this laptop the CPU tree is `builds/macos-cpu-release/`:

```bash
cmake --build builds/macos-cpu-release --target alloy_pf_karma2001_benchmark_openmp -j 8
```

`ctest -R alloy-pf-karma2001-benchmark-smoke` runs two capped `smoke` steps
(`OPENPFC_KARMA_MAX_STEPS=2`, PNG skipped).

## Run

```bash
# Paper suite — t* = 10000, L/d0 = 1000 (needed for late V* and cs ≈ 0.15)
./apps/alloy_pf_karma2001_benchmark/scripts/run_karma2001_benchmark.sh

# Pipeline check only (t* = 80): not a Fig. 1–2 comparison
QUICK=1 ./apps/alloy_pf_karma2001_benchmark/scripts/run_karma2001_benchmark.sh

# Optional: one 45° orientation check, or the old Δt family (not the paper set)
GRID=1 ./apps/alloy_pf_karma2001_benchmark/scripts/run_karma2001_benchmark.sh
DT_SCAN=1 ./apps/alloy_pf_karma2001_benchmark/scripts/run_karma2001_benchmark.sh

# Re-plot existing runs (Fig. 1 = the three positional dirs; pinning figure optional)
./apps/alloy_pf_karma2001_benchmark/scripts/run_karma2001_benchmark.sh \
  --plot-only results/alloy_pf_karma2001_benchmark/paper/d0W_0.277_th0_dx1.0_dt0.09 \
              results/alloy_pf_karma2001_benchmark/paper/d0W_0.544_th0_dx1.0_dt0.09 \
              results/alloy_pf_karma2001_benchmark/paper/d0W_0.277_th0_dx0.4_notauEU_paperlike \
  --dx-scan results/alloy_pf_karma2001_benchmark/paper/d0W_0.277_th0_dx0.4_notauEU_paperlike \
            results/alloy_pf_karma2001_benchmark/paper/d0W_0.277_th0_dx0.6_dt0.09 \
            results/alloy_pf_karma2001_benchmark/paper/d0W_0.277_th0_dx0.8_dt0.09 \
            results/alloy_pf_karma2001_benchmark/paper/d0W_0.277_th0_dx1.0_dt0.09
```

Outputs under `results/alloy_pf_karma2001_benchmark/paper/figures/` (QUICK writes
`.../quick/figures/`):

- `fig1_tip_velocity.png` — \(V d_0/D\) vs \(t D/d_0^2\) (Y clipped at 0.08); notes [100] pinning at \(\Delta x=W_0\) and late \(\langle V^*\rangle\) agreement
- `fig2_cs_growth_ray.png` — \(c_s/c_l^0\) along the growth ray
- `fig_dx_pinning.png` — extra \(d_0/W=0.277\) \(\Delta x\) scan (\(0.4\)–\(1.0\,W_0\)); skip with `DX_PINNING=0`
- `karma2001_metrics.tsv` — late \(V^*\) and RMSE vs the paper curves

Bare binary (present-model defaults; the driver is the usual entry):

```bash
alloy_pf_karma2001_benchmark_openmp glasner 0.277 0 outdir 8
alloy_pf_karma2001_benchmark_openmp fine 0.277 outdir 8   # 2001-like mesh
alloy_pf_karma2001_benchmark_openmp smoke
```

`glasner` is \(\Delta x=W_0\), Glasner \(\psi\), Ji 9-pt. `fine` is the third
table row. Env `OPENPFC_KARMA_DT`, `DX`, `GLASNER`, `ISO`, `TAU_EU` select the
other recipes. `--help` lists the rest.

## LUMI-C

Helpers in `scripts/` (`lumi_paths.sh`, `lumi_build_cpu.sh`, `sync_to_lumi.sh`).
Do **not** use root `scripts/build.sh` for this CPU OpenMP binary (that path
defaults to HIP / LUMI-G).

```bash
./apps/alloy_pf_karma2001_benchmark/scripts/lumi_submit_paper.sh
```

That syncs, builds `alloy_pf_karma2001_benchmark_openmp` if needed, and submits
`lumi_paper.sh` (the 3-case driver, 32 cores, 12 h).

## Extras (not the PRL product)

The engine still knows solute trapping and uniform AM cooling. They are **not**
the advertised benchmark.

| Script | Role |
|--------|------|
| `run_trapping_w0_convergence.sh` | isothermal \(A,\beta_0>0\) vs no-trap \(W_0\) scan |
| `run_am_w0_convergence.sh` | spatially isothermal \(\dot T(t)\) cooling |
| `lumi_submit_glasner_w0.sh` / `lumi_glasner_w0.sh` | LUMI trapping \(W_0\) family |
| `lumi_submit_am_w0.sh` / `lumi_am_w0.sh` | LUMI AM cooling |
| `run_early_transient.sh` | early \(V^*(t^*)\) diagnostics vs Fig. 1 |
| `digitize_karma2001_fig1.py` | regenerate the Fig. 1 TSV from a raster |

`scripts/plot_figures.py` is the extra AM/trapping plotter and the loader
library used by `compare_karma2001.py`.
