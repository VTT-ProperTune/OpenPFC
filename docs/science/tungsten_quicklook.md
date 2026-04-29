<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Science note: Tungsten PFC runs (what you are simulating)

OpenPFC’s **tungsten** application solves a **phase field crystal (PFC)** formulation suited to **body-centered cubic (BCC)**-like order and elasticity on diffusive time scales. It is the main **production-style** demo for JSON-driven `App` runs, validated parameters, and large-scale HPC ([`applications.md`](../user_guide/applications.md), [`apps/tungsten/README.md`](../../apps/tungsten/README.md)).

## What a minimal run demonstrates

- **Solidification / ordering** from seeds or patterned initial data under **fixed** or **moving** boundary models (see sample JSON under [`apps/tungsten/inputs_json/`](../../apps/tungsten/inputs_json/README.md)).  
- **Multi-field output** (e.g. order parameter and auxiliary fields) written as **binary** slices when `saveat` and `fields` are set ([`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)).  
- **MPI + distributed FFT** through the same spectral stack described in [`app_pipeline.md`](../user_guide/app_pipeline.md).

## Where to start in the docs

| Goal | Document |
|------|----------|
| Run a stock input locally | [`quickstart.md`](../quickstart.md) §2B, [`tutorials/end_to_end_visualization.md`](../tutorials/end_to_end_visualization.md) |
| Understand JSON keys | [`spectral_app_config_reference.md`](../reference/spectral_app_config_reference.md), [`app_pipeline.md`](../user_guide/app_pipeline.md) |
| Slurm / cluster | [`tutorials/hpc_slurm_day_one.md`](../tutorials/hpc_slurm_day_one.md), [`lumi_slurm/README.md`](../lumi_slurm/README.md) |
| Figures / motivation | [`showcase.md`](../user_guide/showcase.md), root [`README.md`](../../README.md) |

## Literature

The root [`README.md`](../../README.md) cites the OpenPFC paper (MSMSE, 2024) for methodology and validation context.
