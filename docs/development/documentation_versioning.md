<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Documentation vs releases

OpenPFC prose lives in this repository (`docs/`, root `README.md`, `INSTALL.md`). The [published API reference](https://vtt-propertune.github.io/OpenPFC/dev/) tracks **development** (`master` / `main`) unless your site pins a tag.

| You have | Read |
|----------|------|
| A **tagged release tarball** | Match [`CHANGELOG.md`](../../CHANGELOG.md) for that tag; treat `docs/` at that tag as the prose baseline. CMake options and JSON keys evolve—if in doubt, compare your tree to the tag. |
| A **git clone of `master`** | Expect docs to describe **upcoming** behavior; features may land before a release. |
| **Only** the HTML API site | Pair it with [`docs/README.md`](../README.md) and [`quickstart.md`](../quickstart.md)—tutorials are not fully duplicated in Doxygen. |

**Practical rule:** for reproducible papers or production jobs, record **OpenPFC commit hash or release tag**, **HeFFTe version**, and **MPI module versions** alongside your config files.

## See also

- [`CHANGELOG.md`](../../CHANGELOG.md) — user-visible changes by version  
- [`build_options.md`](../reference/build_options.md) — CMake flags that drift most often  
