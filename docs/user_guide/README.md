<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# User guide

This section covers the operation of existing OpenPFC applications and the
configuration-driven frontend. It assumes OpenPFC has already been built and a
basic example runs successfully.

For installation use [`INSTALL.md`](../../INSTALL.md). For the first build and
run use [Start here](../start_here_15_minutes.md).

## Run applications

| Topic | Document |
|------|----------|
| Available binaries and sample inputs | [Applications](applications.md) |
| Configuration vocabulary | [Configuration](configuration.md) |
| JSON/TOML to runtime objects | [Application pipeline](app_pipeline.md) |
| Parameter validation and diagnostics | [Parameter validation](parameter_validation.md) |
| Custom time-stepper integration | [Custom stepper integration](custom_stepper_integration.md) |

Exact configuration keys belong in the
[Spectral App configuration reference](../reference/spectral_app_config_reference.md).

## Work with results

| Topic | Document |
|------|----------|
| Writers and result files | [Result files](io_results.md) |
| Read raw binary fields with external tools | [Post-process binary fields](postprocess_binary_fields.md) |
| Figures mapped to runnable programs | [Showcase](showcase.md) |

The exact binary layout is specified in
[Binary field file layout](../reference/binary_field_io_spec.md).

## Continue by goal

- For a full run-to-visualization walkthrough, use
  [End-to-end visualization](../tutorials/end_to_end_visualization.md).
- For a batch-system workflow, use the [HPC operator guide](../hpc/operator_guide.md).
- For an application in another repository, use
  [Minimal custom application](../tutorials/custom_app_minimal.md).
- For new models, modifiers, or writers, use
  [Extending OpenPFC](../extending_openpfc/README.md).
