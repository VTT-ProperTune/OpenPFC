<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Getting started: library concepts

This folder contains the slower, code-oriented introduction to the OpenPFC
library. It is for readers who have already completed the short
[Start here](../start_here_15_minutes.md) build-and-run check and now want to
understand the objects behind that run.

For the complete documentation map, use the [documentation index](../index.md).
For a role-based sequence covering applications, extension work, or downstream
integration, use [Learning paths](../learning_paths.md).

## Tutorial sequence

1. [Library basics](01-basics/README.md)
   - create a domain;
   - decompose it across MPI ranks;
   - construct and use the FFT layer;
   - build a small out-of-tree CMake consumer.
2. [Functional field operations](functional_field_ops.md)
   - express initial and boundary operations without repeating nested index
     loops;
   - connect field operations to the extension style used elsewhere in the
     framework.

The runnable examples that accompany these ideas are catalogued in
[Examples catalog](../reference/examples_catalog.md). A guided sequence through
the spectral examples is available in
[Spectral examples sequence](../tutorials/spectral_examples_sequence.md).

## Choose the next document

| Next goal | Document |
|-----------|----------|
| Build a config-driven executable in another repository | [Minimal custom application](../tutorials/custom_app_minimal.md) |
| Understand the main package layers | [Architecture](../concepts/architecture.md) |
| Find the header for a type | [Tour of main types](../reference/class_tour.md) |
| Understand JSON/TOML wiring | [Application pipeline](../user_guide/app_pipeline.md) |
| Add models, modifiers, or writers | [Extending OpenPFC](../extending_openpfc/README.md) |
| Run an existing application | [Applications](../user_guide/applications.md) |

## Scope

This folder teaches library concepts. Installation details belong in
[`INSTALL.md`](../../INSTALL.md), short operational procedures belong in
[`recipes/`](../recipes/README.md), and exact options and formats belong in the
[reference index](../reference/README.md). Keeping those details in their
canonical documents prevents tutorial examples from drifting away from the
tested build and runtime interfaces.
