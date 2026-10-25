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
