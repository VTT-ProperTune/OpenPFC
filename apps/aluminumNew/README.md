<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# AluminumNew

Sample phase field application built on `pfc::ui::App<Aluminum>`. It loads JSON or TOML from the command line, registers custom field modifiers (`SeedGridFCC`, `SlabFCC`), and runs the standard spectral simulation session.

## Build

Built with the main OpenPFC tree when `OpenPFC_BUILD_APPS=ON` (default):

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

The executable is `build/apps/aluminumNew/aluminumNew` (path may vary with the generator).

## Run

Pass a configuration file as the first argument (working directory affects relative paths inside the file):

```bash
cd build
mpirun -n 4 ./apps/aluminumNew/aluminumNew ../apps/aluminumNew/aluminumNew.json
```

A matching `aluminumNew.toml` is provided for TOML workflows. Adjust `results`, `fields`, and paths under `model.params` for your machine.

## Source layout

| File | Role |
|------|------|
| `aluminumNew.cpp` | `main`: registers modifiers, constructs `App<Aluminum>` |
| `Aluminum.hpp` | Aluminum Model implementation |
| `SeedGridFCC.hpp`, `SlabFCC.hpp`, `SeedFCC.hpp` | Registered FieldModifier / IC helpers |

## See also

- [`../../docs/applications.md`](../../docs/user_guide/applications.md) — other shipped apps 
- [`../../docs/configuration.md`](../../docs/user_guide/configuration.md) — config file concepts 
- [`../../docs/quickstart.md`](../../docs/quickstart.md) — first-time setup 
