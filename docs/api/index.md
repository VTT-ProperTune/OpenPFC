<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# C++ API reference

Doxygen extracts declarations and source comments from OpenPFC public headers.
Breathe renders the selected public contracts directly inside this Sphinx site.
The API section is organized by user-facing concepts rather than by the
physical include tree, so implementation helpers do not dominate navigation or
search results.

Use the [tour of main types](../reference/class_tour.md) for architectural
orientation and the pages below for exact members and declarations.

```{toctree}
:maxdepth: 2

Data and domains <data>
Decomposition and FFT <spectral>
Simulation contracts <simulation>
Application frontend <frontend>
Execution and fields <execution>
```

The complete installed source-level contract remains the public headers under
`include/openpfc/`. Add a type to this curated reference when it is a supported
entry point users are expected to call directly.
