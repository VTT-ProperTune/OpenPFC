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

Build output (`_output/`, `.quarto/`) is **not** committed. It was, once, and
then sat stale for months because nothing regenerated it; render locally
instead.

### Rendering on LUMI

Nothing needed is available as a module, but all three pieces install into
`$HOME` without root:

```bash
# 1. Quarto itself
v=1.10.18
curl -sL -o /tmp/quarto.tar.gz \
  "https://github.com/quarto-dev/quarto-cli/releases/download/v${v}/quarto-${v}-linux-amd64.tar.gz"
mkdir -p ~/.local/opt ~/.local/bin && tar xzf /tmp/quarto.tar.gz -C ~/.local/opt/
ln -sf ~/.local/opt/quarto-${v}/bin/quarto ~/.local/bin/quarto

# 2. A LaTeX toolchain, for the PDF only
quarto install tinytex
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$HOME/.local/bin:$PATH"
```

The third piece is `rsvg-convert`, which Quarto's PDF path shells out to in
order to turn the committed SVG figures into PDF. LUMI has no librsvg, but
`cairosvg` does the same job; a shim on `PATH` accepting the flags Quarto
actually passes (`-f pdf -a -o OUT IN`) is enough. HTML needs neither TinyTeX
nor the shim.

**Render the PDF before believing the report is sound.** The markdown checkers
and the HTML render both tolerate math that LaTeX rejects — MathJax quietly
accepts `$R^\*$`, LuaTeX stops with `Missing { inserted`. That exact error has
now reached the repository twice, in two different chapters. Write `^{*}`.

The report has **no compute engine**: `engine: markdown` in
[`_quarto.yml`](_quarto.yml). It renders with `quarto` alone and needs no
Python, R, or Jupyter.

## Layout

| Path | Contents |
|---|---|
| `index.qmd`, `00_*`–`02_*` | preface, abstract, introduction, shared numerical methods |
| `03_*`–`15_*` | one chapter per continuum application (tungsten through kobayashi) |
| `16_alloy_dendrite.qmd` | the thermo-solutal-elastic dendrite: local FD physics and a global FFT solve in one coupled application |
| `17_vlasov_maxwell.qmd` | the 1D2V electromagnetic Vlasov–Maxwell kinetic capstone: a distribution on a 3-D phase space coupled to the full Maxwell system |
| `18_scalability.qmd` | measured LUMI-G strong-scaling curves (tungsten and Heat3D; Vlasov scaling is in chapter 17) |
| `19_conclusions.qmd` | what is demonstrated, and the gaps |
| `data/*.csv` | measured timings, transcribed from [`../hpc/lumi_gpu_scaling.md`](../hpc/lumi_gpu_scaling.md) |
| `figures/*.svg` | committed figures |
| `figures/make_figures.py` | regenerates the scalability figures from `data/` |
| `figures/field_io.py`, `figures/field_plots.py` | field-visualisation library: read `.vti`/`.bin`/grayscale-`.png` field output, render panels/montages/comparisons/line plots |
| `figures/make_field_figures.py` | regenerates the field-visualisation figures from a run of `run_field_demos.sh` |
| `figures/run_field_demos.sh` | run recipe: reproduces the raw `.vti`/`.bin`/`.png` output the field figures are rendered from |

## Updating the scalability figures

The SVGs are committed so the report renders without a plotting stack. After
changing anything under `data/`, regenerate them:

```bash
python3 docs/report/figures/make_figures.py
```

This needs matplotlib. On LUMI, `module load cray-python` provides a suitable
Python but not matplotlib, so use a virtual environment.

## Field-visualisation figures

Chapters `03_tungsten.qmd`, `05_cahn_hilliard.qmd`, `06_thin_film.qmd`,
`07_surface_diffusion.qmd`, and `09_ehd_film.qmd`
Chapters `03_tungsten.qmd`, `04_aluminum.qmd`, `05_cahn_hilliard.qmd`,
`06_thin_film.qmd`, `10_higher_order_pfc.qmd` and `11_gradient_elasticity.qmd`
show real simulation output — a "Visualisation" section near the end of
each — rendered by `figures/field_io.py` (readers) and `figures/field_plots.py`
(panels, montages, comparisons) from real runs, not synthesised data. As with
the scalability figures, only the SVGs are committed; the multi-megabyte raw
`.vti`/`.bin` output used to render them is not.
Several application chapters show real simulation output — a "Visualisation"
section near the end of each — rendered by `figures/field_io.py` (readers) and
`figures/field_plots.py` (panels, montages, comparisons, line plots) from real
runs, not synthesised data. As with the scalability figures, only the SVGs are
committed; the multi-megabyte raw `.vti`/`.bin`/`.png` output used to render
them is not.

Two of these figures are not 2-D images, and the reason is worth knowing
before adding another:

* `08_kawahara.qmd` is a **one-dimensional** application, so its figure is a
  line plot (`render_line_comparison`). A `512 x 1 x 1` snapshot drawn through
  `render_panel` would be a one-pixel stripe, and the thing being shown — a
  shed wave train at a few percent of the pulse height — is an amplitude,
  which a linear axis reads far better than a colour bar.
* `14_allen_cahn.qmd` and `15_kobayashi.qmd` are command-line applications
  that never touch the JSON session pipeline, so they write no `.vti` and no
  `.bin` at all. Their only field output is the 8-bit grayscale PNG that
  `pfc::io::write_mpi_scalar_field_png_xy` emits, with a fixed affine map and
  caller-supplied clip bounds. `field_io.read_gray_png` inverts that map, so
  the figure is rendered from the run's own data rather than from a picture
  of it — at the cost of 1/255-of-range quantisation and of losing anything
  the writer clipped. Both figure docstrings say what that costs for their
  run; the Allen–Cahn one is checkable, since the superlevel-set areas
  recovered from the PNGs are the same integers the program printed.

One reader-facing wart to know about before choosing axis units: `pfc::VTKWriter`
writes `Origin="0 0 0" Spacing="1 1 1"` into every `.vti` regardless of the
domain's real `origin` and `dx`, and names every array `Field`. A `.vti` can
therefore only be plotted in *grid cells*. Where a figure needs physical
coordinates — the tungsten and aluminium PFC panels, whose axes are in reduced
PFC length units — the run writes `.bin` instead and the figure supplies a
`field_io.GridSpec` out of band.

To regenerate them from scratch:

```bash
# Pick two directories of your own first; nothing below is shared state.
build_dir=/flash/project_462001519/juaho/build/report-figures
field_data_dir=/flash/project_462001519/juaho/tmp-shared/report-figures

# 1. Build (see AGENT_NOTES.md / the root README for the shared LUMI
#    allocation rules — do not call salloc/sbatch yourself):
./scripts/build.sh --machine=lumi --cpu --no-submit --no-test \
    --build-dir="$build_dir"

# 2. Run the demo applications (cahn_hilliard, thin_film, tungsten,
#    surface_diffusion, ehd_film) and write their .vti/.bin output
#    somewhere. On the shared LUMI allocation:
#    kawahara, wave2d, allen_cahn, kobayashi) and write their
#    .vti/.bin/.png output somewhere. On the shared LUMI allocation:
#    higher_order_pfc, gradient_elasticity, aluminumNew) and write their
#    .vti/.bin output somewhere. Every run is single rank; the two
#    expensive ones are tungsten (256^3) and aluminium (192^3, twice), a
#    few minutes each. On a LUMI login node, with no launcher at all:
RUNNER="" FIELD_DATA_DIR="$field_data_dir" \
    docs/report/figures/run_field_demos.sh "$build_dir"

#    Or on the shared LUMI allocation:
SLURM_JOB_ID=$(cat /flash/project_462001519/juaho/shared_job_id.txt) \
TMPDIR=/flash/project_462001519/juaho/tmp-shared \
RUNNER="srun --overlap -n 1" \
FIELD_DATA_DIR="$field_data_dir" \
    docs/report/figures/run_field_demos.sh "$build_dir"

# 3. Render the SVGs (needs matplotlib + numpy; see above for why
#    cray-python is not enough):
FIELD_DATA_DIR="$field_data_dir" \
    /flash/project_462001519/juaho/venv-pytest/bin/python \
    docs/report/figures/make_field_figures.py
```

`run_field_demos.sh` documents, next to each run, why its parameters were
chosen (in particular: the two `thin_film` runs deliberately extend the
shipped preset's `t1`/`saveat` past what the default JSON input uses,
while the `surface_diffusion` and `ehd_film` pairs deliberately do not,
and the tungsten run uses the 256³ preset rather than the 32³ one because
`SingleSeed`'s seed radius does not fit inside the smaller domain). Read
chosen (in particular: two of the thin-film runs deliberately extend the
shipped preset's `t1`/`saveat` past what the default JSON input uses; the
tungsten run uses the 256³ preset rather than the 32³ one because
`SingleSeed`'s seed radius does not fit inside the smaller domain; the
gradient-elasticity and aluminium pairs each vary exactly one parameter —
the internal length, the seed radius — against an otherwise identical run,
which is what makes them comparisons rather than two pictures). Read
those comments before changing a parameter — the values are not arbitrary,
several were chosen to stay just inside a numerical-stability boundary that
chosen. A few examples of what those comments contain: two thin-film runs
deliberately extend the shipped preset's `t1`/`saveat` past what the default
JSON input uses; the tungsten run uses the 256³ preset rather than the 32³
one because `SingleSeed`'s seed radius does not fit inside the smaller
domain; `wave2d` must run at `fd_order` 2 because a wider halo does not fit
in an `nz == 1` slab; and the Allen–Cahn and Kobayashi runs use larger grids
than their defaults so the growing phase does not meet its own periodic
image inside the run. Read those comments before changing a parameter — the
values are not arbitrary, several were chosen to stay just inside a
numerical-stability boundary or a domain-size boundary that
`make_field_figures.py`'s docstrings also explain.
several were chosen to stay just inside a numerical-stability boundary, or
outside a periodic-image artefact, that `make_field_figures.py`'s docstrings
also explain.

**Before committing a regenerated figure**, render a PNG copy and look at
it — do not commit an SVG you have not visually inspected:

```python
import matplotlib.pyplot as plt
import make_field_figures as mff  # after sys.path.insert(0, "docs/report/figures")
for make in mff.FIGURES:
    out, fig = make()
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
```

### Plugging in a new application

To add a field figure for another application:

1. **Produce output.** Either point one of its `fields[].data` entries at a
   `.vti` path (any app using the JSON `SpectralETDSession` pipeline gets
   this for free — an extension of `.vti`/`.vtk` selects the VTK writer) or
   read the raw `.bin` dump it already writes; add the run to
   `run_field_demos.sh` (or a similar recipe) so the figure is reproducible.
   An application with its own `main()` rather than a session — the science
   drivers of `surface_diffusion` and `ehd_film` — reads the same `fields[]`
   key through `apps/common/include/openpfc_apps/field_snapshots.hpp`, which
   is two calls: build the writer after the field exists, then write a
   snapshot wherever the driver already samples its diagnostics.
   read the raw `.bin` dump it already writes. A command-line application
   with no JSON path may write only grayscale PNGs; that is still real data
   (see step 2). Add the run to `run_field_demos.sh` (or a similar recipe) so
   the figure is reproducible.
2. **Read it.** `field_io.read_vti(path)` returns a `Field2D` directly. For a
   `.bin` dump, build a `field_io.GridSpec` from the run's JSON `domain`
   (`nx`/`ny`/`nz` are grid *point counts* — `domain.Lx` etc., not physical
   lengths — `dx`/`dy`/`dz`, and `origin`, `"corner"` or `"center"`) and call
   `field_io.read_bin` (3D array) or `field_io.slice_bin` (a 2D slice of a 3D
   field, e.g. for a PFC density — see the tungsten example). For a
   grayscale PNG, `field_io.read_gray_png` inverts the writer's affine map;
   you must pass the same `vmin`/`vmax` the application passed
   (`(-1, 1)` for Allen–Cahn, `(0, 1)` for Kobayashi) or the recovered
   values are wrong.

   **Beware the `.vti` extent.** `pfc::apply_writer_domain` calls only the
   three-argument `set_domain`, never `VTKWriter::set_spacing`/`set_origin`,
   so every `.vti` the JSON session writes carries `Spacing="1 1 1"`
   regardless of the run's `domain.dx`. That is invisible at `dx = 1` and
   wrong by a factor of four for, say, `apps/kawahara`'s `dx = 0.25` inputs.
   Pass the run's own `dx` through `field_io.with_spacing` when a figure's
   axis is in physical units.
3. **Decide sequential vs diverging.** One-sided quantities (a thickness, a
   magnitude) are `kind="sequential"`. Signed quantities around a physically
   meaningful midpoint (a composition around its mean, a PFC density around
   its baseline) are `kind="diverging"` with an explicit `center` — never a
   plain rainbow map.
4. **Render.** For a 2-D field: `field_plots.render_panel` (one field),
   `render_montage` (a time series, shared colour scale), or
   `render_comparison` (two fields, shared colour scale). For a 1-D field
   (`Lx x 1 x 1`): `render_line_panel` (several profiles on one axes) or
   `render_line_comparison` (two runs, shared x and y axes) — never an image
   renderer, which would draw a one-pixel stripe. Line colours come from the
   Okabe–Ito set and are paired with distinct dash patterns, so the curves
   survive both colour-blind vision and a greyscale print. See
   `make_field_figures.py` for worked examples of each, including how each
   one's caption explains what the reader is looking at and why, not just
   "field at t=100".
5. **Add a "Visualisation" section** to the application's chapter, between
   "Binaries and inputs" and "Scalability", following the existing three.

## Adding an application chapter

Copy the structure of an existing chapter: physical setting (ending in the
physical question and its observable), `Problem setup` table, governing
equations, discrete formulation, solution strategy, verification, binaries and
inputs, scalability. Give the chapter a `{#sec-...}` label, and add it to the
`chapters:` list in [`_quarto.yml`](_quarto.yml).

Every application chapter carries the physical-experiment contract from
issue `#112`:

1. a physical question and a **named** observable, stated *before* the
   equations;
2. a `Problem setup` table with rows for use case, domain, grid, boundary
   conditions, initial condition, key physical parameters, observable and model
   maturity;
3. **model maturity on three independent axes**, never collapsed into one word
   — numerical verification (analytical / manufactured / regression / none),
   physical completeness (canonical / reduced / extended), and calibration
   (none / representative / quantitative);
4. where an application ships both, an explicit statement of which input is the
   verification preset and that it is not a production science case;
5. a sentence saying what periodicity, or any other idealisation, means
   physically *for that problem* and what it excludes — written for that
   chapter, not copied between them;
6. physical parameters with units and a reference, or labelled illustrative /
   nondimensional.

The same block appears in each `apps/<name>/README.md`, so the app and the
report agree.

Keep the *Verification* section tied to tests that actually exist, and say
"not measured" rather than estimating a number. The same rule applies to the
setup table: describe the domain, boundary condition or observable the code
really has, and where something cannot be confirmed from the repository, say
so rather than supplying a plausible value.

Sphinx does not build this directory (it is excluded in `docs/conf.py`), and
the Markdown link checker only scans `.md`, so `.qmd` files are invisible to
both.
