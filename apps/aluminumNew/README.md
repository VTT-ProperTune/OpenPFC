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
The M9 A/B CPU binary is `aluminum_etd` (Gen-1 `aluminumNew` remains).
GPU A/B binaries are `aluminum_etd_cuda` / `aluminum_etd_hip` when those backends are enabled.

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
| `aluminum_etd.cpp` | M9 A/B CPU session (`AluminumPhysics` + moving-frame ETD) |
| `aluminum_etd_cuda.cpp` / `aluminum_etd_hip.cpp` | GPU sessions on `GPUSpectralStack` |
| `include/aluminum/aluminum_physics.hpp` | Schema + moving-frame mean-field ETD descriptors |
| `Aluminum.hpp` | Gen-1 Aluminum Model (still built) |
| `SeedGridFCC.hpp`, `SlabFCC.hpp`, `SeedFCC.hpp` | Registered FieldModifier / IC helpers |

## See also

- [`../../docs/applications.md`](../../docs/user_guide/applications.md) — other shipped apps 
- [`../../docs/configuration.md`](../../docs/user_guide/configuration.md) — config file concepts 
- [`../../docs/quickstart.md`](../../docs/quickstart.md) — first-time setup 
