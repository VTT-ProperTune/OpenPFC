<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Application frontend

The frontend turns JSON or TOML settings into a simulation object graph. Read
the [application pipeline](../user_guide/app_pipeline.md), the
[minimal custom application tutorial](../tutorials/custom_app_minimal.md), and
[0.1 → 0.2 migration](../MIGRATION_0.1_to_0.2.md) before using these types
directly.

`pfc::ui::App` and `pfc::ui::SpectralSimulationSession` are deleted. JSON
drivers use `pfc::ui::make_simulation_session<Stack>` or an app-owned ETD
session, then `pfc::sim::run`.

## `pfc::sim::stacks::BasicSpectralCPUStack`

`SpectralCPUStack` is the double-precision instance
`BasicSpectralCPUStack<double>`; `BasicSpectralCPUStack<float>` is the
single-precision stack of ADR 0005.

```{doxygenclass} pfc::sim::stacks::BasicSpectralCPUStack
:project: OpenPFC
:members:
:protected-members:
```

```{doxygentypedef} pfc::sim::stacks::SpectralCPUStack
:project: OpenPFC
```
