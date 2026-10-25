<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Examples

Start with [OpenPFC by example](https://vtt-propertune.github.io/OpenPFC/learn/),
a sequence of Quarto lessons with runnable C++ models under thematic
directories: `geometry/`, `fourier/`, `differentiation/`, `heat/`,
`interpolation/`, `waves/`, `integration/`, `polynomials/`, `parallel/`,
`coupling/`, `output/`, `diagnostics/`, `elliptic/`, `quadrature/`,
`initialization/`, `configuration/`, and `phase-fields/`.
See [index.qmd](index.qmd) for the story and [AUTHORING.md](AUTHORING.md) for
local preview, numerical checks, and figure reproduction.
The [coverage inventory](coverage.qmd) keeps the existing targets and all
forty Trefethen programs visible, including work still pending.

Small programs that demonstrate the OpenPFC library. They are built when `OpenPFC_BUILD_EXAMPLES=ON` (CMake default).

## Build

From the repository root (after satisfying [`INSTALL.md`](../INSTALL.md)):

```bash
./scripts/build.sh
```

The default Tohtori CPU executables appear under `builds/release/examples/`.

## Suggested first runs

With an installed SDK and CLI, start from the repository root:

```bash
(cd examples/geometry/first-domain && openpfc run model.cpp input.json --launcher=none)
(cd examples/initialization/patterns && openpfc run model.cpp input.json --source-root .. --launcher=none)
(cd examples/heat/implicit-euler && openpfc run model.cpp input.json --source-root .. --ranks=2 --launcher=mpirun)
(cd examples/phase-fields/cahn-hilliard && openpfc run model.cpp input.json --launcher=none)
```

The CLI prints each isolated attempt directory. Its `results/` contains
that run's fields and diagnostics. The chapters explain the input parameters,
expected behaviour and checks. Native executables built by `scripts/build.sh`
can also be run from the repository root under `builds/release/examples/`;
their relative `results/` output then stays under the repository's ignored
output directory.

A **research prototype** (not part of the numbered teaching ladder) is
`ns2d_vorticity`: 2-D vorticity–streamfunction Navier–Stokes
([#21](https://github.com/ahojukka5/OpenPFC/issues/21)) and
incompressible MHD (`mhd2d`,
[#23](https://github.com/ahojukka5/OpenPFC/issues/23)). See
[`ns2d_vorticity/README.md`](ns2d_vorticity/README.md). Conference
figures for the accepted MHD windows:
[`ns2d_vorticity/VISUALS.md`](ns2d_vorticity/VISUALS.md). Bounded
prototype under `examples/`. Promotion to `apps/` follows the
[admission criteria](../apps/README.md#what-belongs-here).

## Full catalog and curriculum

Every target registered in `CMakeLists.txt` is listed in [`../docs/examples_catalog.md`](../docs/reference/examples_catalog.md), including **Tier 1–3** suggested order and a flowchart. Guided **spectral** sequence (`04_diffusion_model` → `05_simulator` → `12_cahn_hilliard`): [`../docs/tutorials/spectral_examples_sequence.md`](../docs/tutorials/spectral_examples_sequence.md).

## See also

- [`../docs/tutorials/README.md`](../docs/tutorials/README.md) — tutorials hub (VTK, HeFFTe, GPU, …)
- [`../docs/learning_paths.md`](../docs/learning_paths.md) — ordered tracks by role
- [`../docs/quickstart.md`](../docs/quickstart.md) — onboarding
- [`../docs/tutorials/end_to_end_visualization.md`](../docs/tutorials/end_to_end_visualization.md) — run once, inspect outputs
- [`../docs/api_examples_walkthrough.md`](../docs/reference/api_examples_walkthrough.md) — Doxygen `docs/api/examples` reading order
- [`../docs/showcase.md`](../docs/user_guide/showcase.md) — figures → runnable entry points
- [`../docs/class_tour.md`](../docs/reference/class_tour.md) — how examples map to `Domain`, stacks, `pfc::sim::run`, and ETD sessions
- [`../docs/extending_openpfc/README.md`](../docs/extending_openpfc/README.md) — turning patterns into your own model or `App`
- `fft_backend_selection.toml` — commented TOML for `[plan_options]` (FFT backend and HeFFTe knobs)
- [`../docs/configuration.md`](../docs/user_guide/configuration.md) — how config files map to the framework
- [`../docs/app_pipeline.md`](../docs/user_guide/app_pipeline.md) — declarative runs with JSON/TOML
