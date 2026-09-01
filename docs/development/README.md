<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Development and maintenance

This section is for contributors and maintainers. User-facing build, run, and
configuration guidance belongs elsewhere in the documentation tree.

General contribution and commit rules are in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Development workflow

| Topic | Document |
|------|----------|
| Build and run tests | [Testing](testing.md) |
| Debugging techniques and build modes | [Debugging](debugging.md) |
| Public API and source style | [Style guide](styleguide.md) |
| Header dependency discipline | [Include hygiene](include_hygiene.md) |
| Documentation structure and checks | [Contributing to documentation](contributing-docs.md) |
| Local unified documentation preview | [Sphinx preview](sphinx_preview.md) |
| Documentation and release versions | [Documentation versioning](documentation_versioning.md) |

## Design and project history

| Topic | Document |
|------|----------|
| Recorded architecture decisions | [ADR index](../adr/README.md) |
| Refactoring direction | [Refactoring roadmap](refactoring_roadmap.md) |
| Relate the publication to runnable software | [From paper to run](from_paper_to_run.md) |
| Checkpoint publication contract | [Checkpoint publication](checkpoint_publish.md) (`CheckpointService` restart) |
| Checkpoint state capture | [Checkpoint state capture](checkpoint_state_capture.md) |
| External coupling | [External coupling](../extending_openpfc/external_coupling.md) |

## Documentation products

| Product | Source and build |
|---------|------------------|
| Unified web documentation | Markdown plus Doxygen XML, rendered by Sphinx/MyST/Breathe |
| C++ API extraction | Public headers and examples, parsed by Doxygen into XML |
| Printable handbook | [Handbook build](handbook_build.md) |
| Figures and visual briefs | [Image prompts](image-prompts.md) |

Before merging a user-visible change, update the canonical guide or reference
page that owns the changed contract and add a changelog entry when required.
Avoid placing temporary issue notes or implementation TODOs in introductory
user documentation.
