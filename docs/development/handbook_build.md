<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Building a printable handbook (optional)

Some readers want a **single PDF** (offline reading on clusters, proposals, teaching). This repository does not require Pandoc for a normal build; handbook generation is **optional**.

## Prerequisites

- [`pandoc`](https://pandoc.org/) on `PATH`
- A LaTeX engine if you want PDF output (e.g. `pdflatex` via a TeX Live install)

## Generate

From the repository root:

```bash
./scripts/build_handbook.sh
```

The script concatenates the manifest order in [`handbook_manifest.txt`](../handbook_manifest.txt) and writes `build-handbook/openpfc-handbook.pdf` when LaTeX is available; otherwise it may emit Markdown only—see script output.

**Note:** CI does not fail if Pandoc is missing; this is a **maintainer / educator** convenience.

## See also

- [`README.md`](../README.md) — master index of all guides  
