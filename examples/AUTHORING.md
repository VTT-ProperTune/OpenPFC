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

## Build and verify

Use `./scripts/build.sh` from the repository root. Numerical checks belong
to the C++ build and test path; publication never runs a simulation.

Update coverage after adding a chapter:

```sh
python3 scripts/audit_example_book.py --sync
python3 scripts/audit_example_book.py
```

## Reproduce the comparison figures

Run each model from the repository root so runtime artifacts stay under
`results/`. Use separate result directories for differentiation and each
heat method. With Matplotlib 3.11.2, draw the committed figures from their
actual CSV outputs:

```sh
python3 examples/render_figures.py \
  --differentiation results/derivative/convergence.csv \
  --fd results/heat-fd/diagnostics.csv \
  --fourier results/heat-fourier/diagnostics.csv \
  --exponential results/heat-exponential/diagnostics.csv
```

The scripts use no numerical simulation during Quarto rendering.

## Reproduce the foundations figures

For the foundation and dynamics chapters, collect native outputs under
`results/example-book/foundations/<case>/results/` and use
`render_foundations.py --runs results/example-book/foundations`. Its case
names are `collocation`, `smoothness`, `cardinal`, `transform`, `transport`
and `stability`. The matching executables are `periodic_collocation`,
`spectral_smoothness`, `cardinal_interpolation`, `03_parallel_fft`,
`variable_transport`, and `stability_regions`. The displayed transport plot
uses `waves/variable-transport/trefethen-p6.json`.


## Reproduce the polynomials figures

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


## Reproduce the workflows figures

Draw the native CSV outputs with `python3 examples/render_workflows.py --runs results/example-book/workflows`. Use the default RK, ETD and coupled-field runs described in the chapters.


## Reproduce the everyday figures

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


## Reproduce the operators figures

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
