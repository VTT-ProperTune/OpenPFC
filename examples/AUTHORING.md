<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Write an experiment worth understanding

The canonical teaching sources live at `examples/<theme>/<implementation>/`.
Keep `index.qmd`, `model.cpp`, and `input.json` together. A theme may own a
small shared header for genuinely identical input and observation code;
the numerical method should remain visible in the implementation.

Begin with a question, state the model and assumptions, derive the numerical
choice, and show actual source excerpts. Give an independently checkable
answer and an experiment with an expected observation. Explain the limits.
Prefer a short mathematical argument to a list of API calls. Preserve
attribution when refactoring existing examples; distinguish a book-inspired
experiment from a complete reproduction of a published program or study.

## Preview the chapters

Use Quarto 1.10.19, matching the website. No numerical runtime is needed:

```sh
quarto preview examples
quarto render examples
python3 scripts/check_example_book.py examples/_build/site
```

The output is `examples/_build/site/`. Quarto reads the committed SVGs and
source excerpts; it never runs a simulation. The maintained MIT-licensed
`include-code-files` extension is vendored at a pinned revision in
`_extensions/include-code-files/UPSTREAM.md`. Add markers such as
`// start snippet evolve` and `// end snippet evolve` to real C++ code,
then include that region with `include="model.cpp" snippet="evolve"`.
Do not copy a second implementation into the article.
Use letters, digits and underscores in snippet names: the upstream filter
interprets names as Lua patterns, so punctuation such as a hyphen can cause
selection to fall back to the entire file. The rendered-source checker
rejects excerpts containing leftover snippet markers.

Use labelled equations, figures and tables when the argument refers back to
them. Add checked primary references to `references.bib`, and cite only the
parts of a source actually consulted. Keep front matter first, followed by
SPDX comments. The catalogue reads title, description, order and categories
from each chapter's front matter.

## Build and verify the examples

```sh
./scripts/build.sh
```

The theme CMake files build the models. Ordinary CTest registration launches
`check_lessons.py`, `check_foundations.py`, `check_dynamics.py` and
`check_polynomials.py`, and `check_workflows.py`. They check complete heat fields,
global ownership, complex FFT coefficients, convergence, transport
characteristics, stability polynomials, interpolation, integrals, and
linear/nonlinear boundary problems.

The operator sequence additionally uses `check_spectra.py` and a small native
reference-eigensolver check. These compare inverse-iteration spectra,
independent Lagrange-product derivatives, complete manufactured boundary
fields, contour identities and exact finite-grid multigrid solutions.
The clamped beam is compared with an independently determined exact cubic
correction. `check_clamped.py` checks all 25 plate modes against directly
differentiated Lagrange products, including boundary slopes, symmetry,
eigenvector independence and the complete small-grid spectrum's trace.
`check_orr_sommerfeld.py` checks all four program-40 spectra, every
generalized eigenvector and the full trace against Lagrange-product
operators. Generalized inverse iteration checks the leading mode; a second
Reynolds number checks the change in growth sign. The native reference
eigensystem test also checks pivoting with an exact complex mass system.
`check_pseudospectra.py` compares complete coarse complex-plane grids at
degrees 12 and 70 with singular inverse iteration through shifted solves.
Even and odd reflection sectors avoid slow convergence of near-degenerate pairs.
It uses neither Jacobi rotations nor an explicit normal-equation matrix.
The native helper check includes complex triangular matrices with tiny
singular values and an exact zero matrix.
`check_disk.py` checks native disk eigenmode and Poisson entry points using
direct Lagrange-product derivatives, full Fourier sums and Bessel series.
The Poisson reference integrates each forcing Fourier term independently;
manufactured and radial-refinement cases distinguish equation residuals
from approximation errors.
`check_coordinates.py` checks the refactored polar-defined Cartesian field
against complex-polynomial formulas, including undefined origin angles,
finite scalar limits and complete ownership on one/two MPI ranks.
`check_evolution.py` checks the six wave/reaction models against direct
Lagrange derivatives, full Fourier sums and cyclic convolutions. It also
distinguishes first- and second-order wave startups and Euler convergence.
`check_everyday.py` runs the domain, initialization, JSON/TOML session,
implicit heat, driver hooks, binary/VTK snapshots and Cahn–Hilliard models.
It checks actual owned fields and output payloads, including padded-field
packing, physical geometry and time metadata. Direct Fourier sums verify
the nonlinear phase-field update and its nonzero nonlinear term at a
neutral linear mode. Both one- and two-rank registrations invoke the real
entry points.
Use independent analytic or basis formulas, not a second copy of the model's
implementation. MPI-capable chapters also have two-rank checks; the complete-domain
polynomial and mixed-coordinate references run on one process. This is a small numerical regression,
not a scientific campaign. Existing targets
`19_explicit_stepper_fd` and `20_explicit_stepper_spectral` are retained;
their implementations now live under `heat/`. Refactored geometry and Fourier
targets likewise retain their binary names. The remaining legacy programs stay
in the inventory until their own teaching chapters are ready.

For an installed SDK, follow each chapter's `openpfc run` command. The heat
examples require `--source-root ..` when invoked in their implementation
directory so that `../lesson.hpp` belongs to the source snapshot. Polynomial
chapters use the same option for `../collocation.hpp`; the periodic matrix
and smoothness experiments use it for `../profiles.hpp`. The new wave
chapters use `../lesson.hpp`, and the Allen–Cahn pair additionally uses
`../allen_cahn.hpp`; their commands also use `--source-root ..`.

## Keep coverage explicit

Each chapter declares `targets`, `capabilities`, and, where appropriate,
`book-programs` in its front matter. Add a precise scope note for any newly
covered book program in `coverage.json`, then run:

```sh
python3 scripts/audit_example_book.py --sync
python3 scripts/audit_example_book.py
```

The audit reconciles executable declarations and chapter metadata, preserves
all forty book entries, and regenerates `coverage.qmd`. It checks inventory
consistency, not scientific correctness or complete capability coverage.
Explain any change from the book's numerical method in the actual chapter.
Chebyshev reference operators in `polynomials/collocation.hpp` belong to the
examples; they do not establish a new public library API.

## Recreate the figures

From the repository root, after a CPU build, generate fresh tutorial output:

```sh
mkdir -p results/example-book/differentiation
(cd results/example-book/differentiation && ../../../builds/release/examples/periodic_differentiation)
mkdir -p results/example-book/fd
(cd results/example-book/fd && ../../../builds/release/examples/19_explicit_stepper_fd)
mkdir -p results/example-book/fourier
(cd results/example-book/fourier && ../../../builds/release/examples/20_explicit_stepper_spectral)
mkdir -p results/example-book/exponential
(cd results/example-book/exponential && ../../../builds/release/examples/heat_exponential)
uv run --with matplotlib==3.11.2 python examples/render_figures.py \
  --differentiation results/example-book/differentiation/results/convergence.csv \
  --fd results/example-book/fd/results/diagnostics.csv \
  --fourier results/example-book/fourier/results/diagnostics.csv \
  --exponential results/example-book/exponential/results/diagnostics.csv
```

Use your allocated compiler/MPI environment, and adjust `builds/release` if
the build has another name. The code runs with its documented defaults;
CSV files stay under ignored `results/`. Only the educational SVG figures
are committed. Curves show numerical output, not invented measurements or
performance claims. Heat figure defaults are N=16, D=1, dt=0.001, 100 steps;
derivative resolutions are 8, 16, 32, 64, 128. Platform roundoff can change
the last digits and the exponential error curve. Inspect figures and the
rendered mathematical argument; a successful render alone cannot judge them.

The clamped beam and plate figures use the actual native default runs:

```sh
example_binaries="$(pwd)/builds/release/examples"
for lesson in beam plate; do
  run_directory="results/example-book/clamped/$lesson"
  mkdir -p "$run_directory"
  (cd "$run_directory" && "$example_binaries/clamped_$lesson")
done
uv run --with matplotlib==3.11.2 python examples/render_clamped.py \
  --beam results/example-book/clamped/beam/results/beam.csv \
  --plate results/example-book/clamped/plate
```

The beam uses degree 15, and the plate uses degree 17 with 25 modes. Native
`chebyshev_derivative` supplies the one-dimensional derivative; the examples
own the weighted fourth-order composition and symmetry-block eigensolver.
The display uses constrained barycentric polynomials to retain the clamped
slopes, rather than an unconstrained image interpolation.

For the Orr–Sommerfeld figure, use the original four degrees and Reynolds
number from the default input:

```sh
mkdir -p results/example-book/orr-sommerfeld
(cd results/example-book/orr-sommerfeld && ../../../builds/release/examples/orr_sommerfeld)
uv run --with matplotlib==3.11.2 python examples/render_orr_sommerfeld.py \
  --run results/example-book/orr-sommerfeld
```

The plot retains the book's spectral window. The CSV files also contain
the modes outside that window and all complex eigenvectors. The mass solve
and shifted QR are example-owned reference implementations; native OpenPFC
Chebyshev differentiation supplies their derivative building blocks.

For the program-24 pseudospectrum, keep the original default spatial and
complex-plane grids:

```sh
mkdir -p results/example-book/pseudospectra
(cd results/example-book/pseudospectra && ../../../builds/release/examples/davies_pseudospectra)
uv run --with matplotlib==3.11.2 python examples/render_pseudospectra.py \
  --run results/example-book/pseudospectra
```

This computes 546 smallest singular values at degree 70. Increasing the
plane sampling adds independent dense Jacobi solves and can be slow; it
is separate from spatial refinement. Only the derived SVG belongs in Git.
To draw a modified run, give either stability plotting script the CLI's
attempt directory as `--run` and choose `--output results/plots`. They accept
changed degrees, Reynolds numbers and complex-plane sampling grids.

For the two disk figures, collect the original native defaults:

```sh
example_binaries="$(pwd)/builds/release/examples"
for entry in modes:disk_eigenmodes poisson:disk_poisson; do
  run_directory="results/example-book/disk/${entry%%:*}"
  mkdir -p "$run_directory"
  (cd "$run_directory" && "$example_binaries/${entry#*:}")
done
uv run --with matplotlib==3.11.2 python examples/render_disk.py \
  --runs results/example-book/disk
```

Defaults retain the book's N25/M20 eigenproblem and N31/M40 Poisson problem.
Dense plotting values reconstruct the signed radial polynomial and angular
modes; they add no PDE resolution. Native diagnostics compare the actual
collocation values with the exact separated solution. The smoother
manufactured field and N63 refinement have their own downloadable inputs.

For the coordinate-pattern figure, run `17_custom_coordinate_system` with
its chapter defaults under `results/example-book/polar-patterns/`, then use:

```sh
uv run --with matplotlib==3.11.2 python examples/render_coordinates.py \
  --run results/example-book/polar-patterns
```

The renderer assembles actual per-rank fields by global indices; the same
command accepts a one- or two-rank run directory.

For the adaptive and coupled-field plots, collect native outputs under
`results/example-book/workflows/<case>/results/`. The `adaptive` case runs
`21_adaptive_stepping`, `etd` runs `24_adaptive_etd`, and `oscillator` runs
`runtime_coupled_fields`, all with their chapter defaults. A second
`oscillator-coarse` run uses `{"dt":0.1,"steps":10}`; both oscillator runs
finish at time 1. Then draw the actual ledgers and field comparisons:

```sh
uv run --with matplotlib==3.11.2 python examples/render_workflows.py \
  --runs results/example-book/workflows
```

These are mathematical teaching figures. Timings printed by diagnostics
models are not frozen into figures or reported as performance results.

For the foundation and dynamics chapters, collect native outputs under
`results/example-book/foundations/<case>/results/` and use
`render_foundations.py --runs results/example-book/foundations`. Its case
names are `collocation`, `smoothness`, `cardinal`, `transform`, `transport`
and `stability`. The matching executables are `periodic_collocation`,
`spectral_smoothness`, `cardinal_interpolation`, `03_parallel_fft`,
`variable_transport`, and `stability_regions`. The displayed transport plot
uses `waves/variable-transport/trefethen-p6.json`.

To regenerate the seven polynomial figures, run in an allocated environment:

```sh
example_binaries="$(pwd)/builds/release/examples"
for lesson in interpolation nodal-polynomials differentiation regularity integration boundary-values nonlinear-boundary; do
  target="polynomial_${lesson//-/_}"
  run_directory="results/example-book/polynomials/$lesson"
  mkdir -p "$run_directory"
  (cd "$run_directory" && "$example_binaries/$target")
done
uv run --with matplotlib==3.11.2 python examples/render_polynomials.py \
  --runs results/example-book/polynomials
```

The shell loop uses Bash's string replacement. These defaults match the
book's degrees and profiles. The plotting script reads the C++ CSV files;
it does not run an independent Python numerical solver to supply the curves.

The operator and contour figures read the following default native runs:

```sh
example_binaries="$(pwd)/builds/release/examples"
for case in oscillator:harmonic_oscillator mathieu:mathieu_spectrum eigenmodes:chebyshev_eigenmodes airy:airy_eigenproblem membrane:membrane_eigenmodes poisson:square_poisson helmholtz:square_helmholtz laplace:square_laplace gamma:gamma_contour multigrid:multigrid_poisson; do
  run_directory="results/example-book/spectra/${case%%:*}"
  mkdir -p "$run_directory"
  (cd "$run_directory" && "$example_binaries/${case#*:}")
done
stiffness_input="$(pwd)/examples/polynomials/eigenmodes/trefethen-p26.json"
mkdir -p results/example-book/spectra/stiffness
(cd results/example-book/spectra/stiffness && "$example_binaries/chebyshev_eigenmodes" "$stiffness_input")
uv run --with matplotlib==3.11.2 python examples/render_spectra.py \
  --runs results/example-book/spectra
```

The wave and reaction figures likewise use native defaults, including the
book's original startup choices and output schedules:

```sh
for case in line:chebyshev_wave_line square:chebyshev_wave_square tank:mixed_wave_tank kdv:kdv_integrating_factor fixed:allen_cahn_fixed driven:allen_cahn_driven; do
  run_directory="results/example-book/evolution/${case%%:*}"
  mkdir -p "$run_directory"
  (cd "$run_directory" && "$example_binaries/${case#*:}")
done
uv run --with matplotlib==3.11.2 python examples/render_evolution.py \
  --runs results/example-book/evolution
```

Both loops run in an allocated environment after defining `example_binaries`.
The polynomial, mixed-coordinate and local-FFT models use one process.
The fixed Allen–Cahn figure includes a symmetry diagnostic because late
evolution amplifies floating-point asymmetries. Its tiny centre displacement
and the final digits of spectral or contour errors can vary by platform.
The Allen–Cahn operator-based timestep cap keeps the original degree-20
default intact while stabilizing tested finer grids; the numerical
regression includes degree 40 and rejection of its original oversized step.
The chapters distinguish finite-grid checks from continuum comparisons;
the routine Python numerical checks need only the standard library.

## Recreate the everyday workflow figures

The three everyday-workflow figures use the chapter inputs, including the
implicit heat lesson's larger demonstration timestep:

```sh
example_binaries="$(pwd)/builds/release/examples"
examples_source="$(pwd)/examples"
for entry in patterns:14_custom_field_initializer:initialization/patterns implicit:04_diffusion_model:heat/implicit-euler phase:12_cahn_hilliard:phase-fields/cahn-hilliard; do
  IFS=: read -r lesson_case lesson_target lesson_chapter <<< "$entry"
  run_directory="results/example-book/everyday/$lesson_case"
  mkdir -p "$run_directory"
  (cd "$run_directory" && "$example_binaries/$lesson_target" "$examples_source/$lesson_chapter/input.json")
done
uv run --with matplotlib==3.11.2 python examples/render_everyday.py \
  --runs results/example-book/everyday
```

Run the binaries in an allocated compiler/MPI environment. The renderer
only plots native CSV values. The JSON/TOML lesson's `.toml` companion is
staged as a downloadable input; source-based `openpfc run` currently
validates JSON, while its built executable can load either file format.
The phase-field chapter also supplies `refine-grid.json` and
`refine-time.json`. Both end at time 1; compare the finer grid at every
second node and compare both timesteps on their common grid. The default
box length 4 and gamma .04 keep the interfaces resolved on the small grid.

## Render within the scientific website

Use a disposable copy of `openpfc-pages`:

```sh
python3 scripts/stage_example_book.py /path/to/website-preview
cd /path/to/website-preview
quarto render --profile examples
```

The staging command imports the lesson sources, assets, download files and
the source-inclusion filter into `learn/`. A generated Quarto profile adds
them to the existing render list. The articles inherit the website's
`openpfc-html` format and use native Quarto citations and cross-references.
The profile does not execute C++ or Python numerical code. The production
documentation workflow performs this same import from its checked-out
OpenPFC revision before rendering the combined website. It publishes the
handbook at `docs/` and these lessons at `learn/`.

Review the combined website at its `/OpenPFC/` deployment prefix, checking
source excerpts, equation numbers, citations, downloads, plots, and mobile
width. Local preview does not establish that a future hosted build has
passed, and does not authorize a merge or publication.

## Physical Fourier symbols

Generate native output in a separate run directory, then render it:

```sh
example_binaries="$(pwd)/builds/release/examples"
mkdir -p results/example-book/symbols
(cd results/example-book/symbols && "$example_binaries/fft_kspace_helpers_example")
uv run --with matplotlib==3.11.2 python examples/render_utilities.py \
  --symbols results/example-book/symbols
```

The default parameters are in `fourier/symbols/input.json`. The checker
compares every spectral symbol with physical Fourier basis functions.
The derived figure makes no timing or performance claim.

## A periodic Gaussian pulse

Run the complete native heat model and plot its actual owned fields:

```sh
example_binaries="$(pwd)/builds/release/examples"
mkdir -p results/example-book/gaussian
(cd results/example-book/gaussian && "$example_binaries/diffusion_model_with_custom_initial_condition")
uv run --with matplotlib==3.11.2 python examples/render_utilities.py \
  --gaussian results/example-book/gaussian
```

The defaults are in `heat/gaussian-pulse/input.json`. Each renderer argument
is a run directory containing `results/`. Give `--output results/plots` for
exploratory figures. The checker compares complete Gaussian fields with
direct Fourier sums, including one/two-rank ownership and time refinement.
A continuous Fourier series checks the model's image-sum reference.
