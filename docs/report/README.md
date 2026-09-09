<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Applications report (Quarto)

A report covering every application shipped with OpenPFC 0.2: the governing
equations, the discrete formulation (spectral symbol or finite-difference
stencil), what the test suite verifies, and measured scalability where it
exists.

## Build

```bash
quarto render docs/report            # HTML into docs/report/_output/
quarto render docs/report --to pdf   # PDF (needs a LaTeX toolchain)
```

The report has **no compute engine**: `engine: markdown` in
[`_quarto.yml`](_quarto.yml). It renders with `quarto` alone and needs no
Python, R, or Jupyter.

## Layout

| Path | Contents |
|---|---|
| `index.qmd`, `00_*`–`02_*` | preface, abstract, introduction, shared numerical methods |
| `03_*`–`15_*` | one chapter per application |
| `16_scalability.qmd` | measured LUMI-G strong-scaling curves |
| `17_conclusions.qmd` | what is demonstrated, and the gaps |
| `data/*.csv` | measured timings, transcribed from [`../hpc/lumi_gpu_scaling.md`](../hpc/lumi_gpu_scaling.md) |
| `figures/*.svg` | committed figures |
| `figures/make_figures.py` | regenerates the figures from `data/` |

## Updating the figures

The SVGs are committed so the report renders without a plotting stack. After
changing anything under `data/`, regenerate them:

```bash
python3 docs/report/figures/make_figures.py
```

This needs matplotlib. On LUMI, `module load cray-python` provides a suitable
Python but not matplotlib, so use a virtual environment.

## Adding an application chapter

Copy the structure of an existing chapter: physical setting, governing
equations, discrete formulation, solution strategy, verification, binaries and
inputs, scalability. Give the chapter a `{#sec-...}` label, and add it to the
`chapters:` list in [`_quarto.yml`](_quarto.yml).

Keep the *Verification* section tied to tests that actually exist, and say
"not measured" rather than estimating a number.

Sphinx does not build this directory (it is excluded in `docs/conf.py`), and
the Markdown link checker only scans `.md`, so `.qmd` files are invisible to
both.
