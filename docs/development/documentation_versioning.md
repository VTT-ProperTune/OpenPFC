<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Documentation vs releases

OpenPFC documentation lives with the source repository. The published Sphinx
site combines Markdown prose from `docs/` with C++ declarations extracted from
the same commit by Doxygen and Breathe. The default published site tracks
development (`master` / `main`) unless a deployment is explicitly pinned to a
tag.

| You have | Read |
|----------|------|
| A **tagged release tarball** | Match [`CHANGELOG.md`](../../CHANGELOG.md) for that tag and read `docs/` from the same tree. CMake options, JSON keys, and API declarations evolve together. |
| A **git clone of `master`** | Expect the documentation to describe upcoming behavior; features may land before a release. |
| The **published HTML site** | Check the displayed OpenPFC version and source link before applying commands to an older checkout or release. |

**Practical rule:** for reproducible papers or production jobs, record the
**OpenPFC commit hash or release tag**, **HeFFTe version**, and **MPI module
versions** alongside configuration files.

## See also

- [`CHANGELOG.md`](../../CHANGELOG.md) — user-visible changes by version
- [`build_options.md`](../reference/build_options.md) — CMake flags that drift most often
- [`sphinx_preview.md`](sphinx_preview.md) — local and CI documentation builds
