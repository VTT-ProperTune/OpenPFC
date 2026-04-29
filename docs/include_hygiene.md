<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Include hygiene (minimal headers)

OpenPFC is organized so **kernel**, **frontend**, and **runtime** layers stay
compilable with **narrow includes**. Prefer domain headers over umbrella pulls:

- **`openpfc_minimal.hpp`** — small entry when you only need core aliases and
  forward declarations (see that header’s Doxygen).
- **Kernel work** — include `openpfc/kernel/...` headers that declare what you
  use (`world.hpp`, `simulator.hpp`, `fft_fftw.hpp`, …) instead of shipping
  `openpfc.hpp` into every translation unit.
- **Frontend / JSON** — include `openpfc/frontend/ui/...` slices (`errors_config_format.hpp`,
  `simulation_wiring_context.hpp`, …) when a parser should not depend on
  unrelated UI modules.

## Tests and out-of-tree drivers

Unit tests under `tests/unit/kernel/` should stay on **kernel + Catch2**
includes where possible so missing dependencies surface at link time, not
hidden behind a mega-header.

## Optional CI check

[`scripts/check_minimal_includes.sh`](../scripts/check_minimal_includes.sh) is a
**lightweight guard**: it fails if kernel unit tests include the full umbrella
`openpfc/openpfc.hpp` (policy can be tightened over time). Run from the repo
root:

```bash
./scripts/check_minimal_includes.sh
```

Wire this into CI only if your pipeline already shells out to similar `rg` /
`grep` gates.

See also [`architecture.md`](architecture.md) (include audit) and
[`refactoring_roadmap.md`](refactoring_roadmap.md) backlog item on minimal
includes.
