<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Archived 0.2 planning documents

These files planned and tracked the 0.1 → 0.2 architecture work. That work
shipped as OpenPFC 0.2.0. They are kept for history. They are not a description
of the current tree: they still talk about Gen-1 `Model`, `Simulator`, `App`,
and `World` as if those types were alive.

Do not start new work from them.

The four dumps are in this directory in git. They are not rendered on the
documentation site (Sphinx excludes them). Open them on GitHub:

| File | What it was |
|------|-------------|
| [Architecture audit](https://github.com/VTT-ProperTune/OpenPFC/blob/master/docs/archive/OPENPFC_ARCHITECTURE_AUDIT.md) | Snapshot of the 0.1 codebase that motivated the refactor |
| [Refactoring execution plan](https://github.com/VTT-ProperTune/OpenPFC/blob/master/docs/archive/OPENPFC_REFACTORING_EXECUTION_PLAN.md) | Milestone sequence M0–M12 that produced 0.2.0 |
| [0.1 → 0.2 migration map](https://github.com/VTT-ProperTune/OpenPFC/blob/master/docs/archive/0.2_migration_map.md) | Per-type replacement table used during the milestones |
| [Refactoring roadmap](https://github.com/VTT-ProperTune/OpenPFC/blob/master/docs/archive/refactoring_roadmap.md) | Earlier phased notes (communicator, Simulator split, JSON stacks) |

Current documents:

- [Architecture](../concepts/architecture.md)
- [Migrating from 0.1 to 0.2](../MIGRATION_0.1_to_0.2.md)
- [ADR index](../adr/README.md)
- Remaining product work: [GitHub milestone 0.2.1](https://github.com/VTT-ProperTune/OpenPFC/milestone/7)
