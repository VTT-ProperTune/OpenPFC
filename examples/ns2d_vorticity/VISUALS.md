<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# 2-D MHD conference visuals

Reproducible figure and animation package for the **accepted** 2-D
incompressible resistive-MHD line:

* #23 / PR #24 — verified Orszag–Tang prototype
* #26 / PR #27 — persistent-X reconnection for $t\le 0.80$
* #38 / PR #39 — frozen-window scaling, outcome (2)

This is a presentation/evidence package. It does **not** change the
solver, reopen scaling, or add physics. Scientific claims live in
[`RECONNECTION.md`](RECONNECTION.md) and [`SCALING.md`](SCALING.md).

## Evidence vs showcase

| Class | Window | Use |
| --- | --- | --- |
| **Evidence** | reconnection $t\in[0,0.80]$; scaling $t\in[0.314,0.70]$ | talks **and** the paper-style argument |
| **Showcase** | $t>0.80$ on the #24 $\eta=0.005$, $N=512$ peak-current dumps | qualitative dynamics only |

Do not mix those windows in one evidence asset. Peak-current /
plasmoid-looking frames are **not** support for the accepted scaling
claim.

Every finished figure or movie is labelled with case, $\eta$,
$\nu$, $N$, $t$, and the window class.

## Datasets

Override the experiment root with `OPENPFC_MHD_DATA` or `--data-root`.
Default:

```text
/scratch/project_462001519/juaho/mhd2d-23
```

| Asset | Run directory | $\eta=\nu$ | $N$ | CFL | Convergence in the claim |
| --- | --- | ---: | ---: | ---: | --- |
| Hero / flux | `ot512_nu0005_peak` | 0.005 | 512 | 0.4 | 256 vs 512; 256 CFL 0.4 vs 0.2 |
| Comparison left | `ot256_nu001_cfl04` | 0.01 | 256 | 0.4 | 128 vs 256; 256 CFL 0.4 vs 0.2 |
| Comparison right | `ot1024_nu00025_cfl02_t080` | 0.0025 | 1024 | 0.2 | 512 vs 1024; geometry/$F$ vs historical 512 CFL 0.4 |

The $\eta=0.005$ directory also contains $t>0.80$ dumps used
**only** for `figures/showcase/`. Evidence drivers stop at $t\le 0.80$.

Tracked X/O, $E_z$, and sheet geometry are read from
`sheet_scaling.json` / `island_flux.json` in those run directories.
Field frames are the native `a_*.bin` / `j_*.bin` dumps (`interpolation=
nearest`). Colour limits are frozen at $\lvert j\rvert\le 6.5$ for
all evidence field figures.

## Commands

From the repository root. Python needs numpy, matplotlib, and ffmpeg
(for MP4). On LUMI, `cray-python/3.11.7` provides matplotlib.

```bash
export OPENPFC_MHD_DATA=/scratch/project_462001519/juaho/mhd2d-23
PY=/opt/cray/pe/python/3.11.7/bin/python3.11
NS=examples/ns2d_vorticity
FIG=$NS/figures
```

Self-test (no dumps required):

```bash
python3 $NS/scripts/test_mhd_conference_vis.py
```

### Paper / slide stills (evidence)

```bash
$PY $NS/scripts/plot_mhd_flux_budget.py \
  --out $FIG/evidence/flux_budget_eta0005.png
$PY $NS/scripts/plot_mhd_scaling_summary.py \
  --out $FIG/evidence/scaling_R_vs_S.png
$PY $NS/scripts/plot_mhd_geometry_summary.py \
  --out $FIG/evidence/geometry_summary.png
$PY $NS/scripts/render_mhd_conference_frames.py --asset hero \
  --t 0.628 --dpi 220 \
  --out $FIG/evidence/stills/hero_sheet_eta0005_n512.png
$PY $NS/scripts/render_mhd_conference_frames.py --asset hero \
  --t 0.471 --dpi 220 \
  --out $FIG/evidence/stills/topology_overlay_eta0005.png
$PY $NS/scripts/render_mhd_conference_frames.py --asset comparison \
  --t 0.628 --dpi 220 \
  --out $FIG/evidence/stills/comparison_three_eta.png
```

### Hero animation (evidence, $t\le 0.80$)

```bash
$PY $NS/scripts/render_mhd_conference_frames.py --asset hero \
  --kind evidence --out-dir $FIG/generated/hero_evidence
$PY $NS/scripts/make_mhd_conference_animation.py \
  --frames-dir $FIG/generated/hero_evidence --kind evidence \
  --mp4 $FIG/evidence/hero_reconnection_eta0005_n512.mp4 \
  --gif $FIG/evidence/hero_reconnection_eta0005_n512.gif
```

### Synchronized 3-case comparison (evidence)

```bash
$PY $NS/scripts/render_mhd_conference_frames.py --asset comparison \
  --kind evidence --out-dir $FIG/generated/comparison_evidence
$PY $NS/scripts/make_mhd_conference_animation.py \
  --frames-dir $FIG/generated/comparison_evidence --kind evidence \
  --asset comparison \
  --mp4 $FIG/evidence/comparison_three_eta.mp4
```

### Optional late-time showcase (not evidence)

```bash
$PY $NS/scripts/render_mhd_conference_frames.py --asset hero \
  --kind showcase --out-dir $FIG/generated/hero_showcase
$PY $NS/scripts/make_mhd_conference_animation.py \
  --frames-dir $FIG/generated/hero_showcase --kind showcase \
  --mp4 $FIG/showcase/late_dynamics_eta0005_n512.mp4
```

One-shot (stills, movies, optional showcase):

```bash
$PY $NS/scripts/make_mhd_conference_package.py
# stills and XY figures only:
$PY $NS/scripts/make_mhd_conference_package.py --stills-only
# skip the late-time movie:
$PY $NS/scripts/make_mhd_conference_package.py --skip-showcase
```

Frame PNGs under `figures/generated/` are local build products and are
not committed. Finished MP4s, paper figures, and slide stills under
`figures/evidence/` and `figures/showcase/` are the talk/paper assets.

## What to use where

**Talks / slides**

* `evidence/stills/hero_sheet_eta0005_n512.png`
* `evidence/stills/comparison_three_eta.png`
* `evidence/stills/topology_overlay_eta0005.png`
* `evidence/scaling_R_vs_S.png` (also copied under `stills/`)
* `evidence/hero_reconnection_eta0005_n512.mp4`
* `evidence/hero_reconnection_eta0005_n512.gif` (short preview)
* `evidence/comparison_three_eta.mp4`

**Scientific evidence**

* flux-budget figure
* scaling $R(S_{\mathrm{local}})$
* geometry summary
* the two evidence movies (accepted window only)

**Showcase only**

* `showcase/late_dynamics_eta0005_n512.mp4`

The OLS slope printed on the scaling figure is a **formal standard
error** on three points. It is not a physical uncertainty interval.
Sweet–Parker $S^{-1/2}$ is drawn as a reference guide, not a fit.
