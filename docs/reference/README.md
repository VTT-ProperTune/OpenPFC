<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Reference

This section contains lookup material: exact options, configuration keys, file
formats, catalogs, and maps from common names to public headers. Read a tutorial
first when you are learning a workflow.

## Build and integration

| Reference | Purpose |
|-----------|---------|
| [CMake options](build_options.md) | Build switches, defaults, and optional features |
| [Dependency matrix](dependency_matrix.md) | Required and optional tools and libraries |
| [Tour of main types](class_tour.md) | Roles, layers, headers, and examples for common API types |
| [API examples walkthrough](api_examples_walkthrough.md) | Reading order for source-focused snippets |
| [Generated C++ API](../api/index.md) | Exact declarations, overloads, namespaces, and Doxygen comments |

## Runtime configuration and data

| Reference | Purpose |
|-----------|---------|
| [Spectral App configuration](spectral_app_config_reference.md) | Exact JSON/TOML keys and values |
| [Binary field file layout](binary_field_io_spec.md) | Raw field storage contract |
| [Profiling operator playbooks](operator_playbooks.md) | Symptom-oriented operational checks |
| [Example run output](example_run_output.md) | Typical log structure and success indicators |

## Inventories and terminology

| Reference | Purpose |
|-----------|---------|
| [Examples catalog](examples_catalog.md) | Runnable targets and teaching tiers |
| [Glossary](glossary.md) | Project terminology |

Public class and function declarations are extracted from headers by Doxygen
and rendered in the [integrated API reference](../api/index.md) through
Breathe. The prose reference pages should explain contracts and relationships
rather than copy complete declarations.
