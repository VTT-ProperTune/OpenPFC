<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Simulation contracts

These interfaces define the reusable simulation lifecycle. Read the
[architecture](../concepts/architecture.md),
[application pipeline](../user_guide/app_pipeline.md), and
[0.1 → 0.2 migration](../MIGRATION_0.1_to_0.2.md) for ownership and control
flow before using the exact declarations.

## `pfc::sim::SimulationDriver`

Thin time loop over `Time` plus a physics `step`. See
`include/openpfc/kernel/simulation/simulation_driver.hpp` (`pfc::sim::run`).

## `pfc::Time`

```{doxygenclass} pfc::Time
:project: OpenPFC
:members:
:protected-members:
:no-link:
```

## `pfc::FieldModifier`

```{doxygenclass} pfc::FieldModifier
:project: OpenPFC
:members:
:protected-members:
:no-link:
```

## `pfc::ResultsWriter`

```{doxygenclass} pfc::ResultsWriter
:project: OpenPFC
:members:
:protected-members:
:no-link:
```
