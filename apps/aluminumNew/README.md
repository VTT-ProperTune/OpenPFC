<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# AluminumNew

Production aluminum binary: JSON/TOML → `AluminumETDSession` (moving-frame mean-field ETD on `SimulationState`). Initial conditions include `constant` and `seed_grid_fcc`.

## Build

Built with the main OpenPFC tree when `OpenPFC_BUILD_APPS=ON` (default):

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

The executable is `build/apps/aluminumNew/aluminumNew` (path may vary with the generator).
`aluminum_etd` is an alias of `aluminumNew`. GPU binaries are `aluminum_etd_cuda` / `aluminum_etd_hip` when those backends are enabled.

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
| `aluminumNew.cpp` | `main`: JSON → `AluminumETDSession` |
| `aluminum_etd.cpp` | alias of `aluminumNew` |
| `aluminum_etd_cuda.cpp` / `aluminum_etd_hip.cpp` | GPU sessions on `GPUSpectralStack` |
| `include/aluminum/aluminum_physics.hpp` | Schema + moving-frame mean-field ETD descriptors |
| `include/aluminum/aluminum_field_modifiers.hpp` | Host-buffer ICs (`constant`, `seed_grid_fcc`) |
| `SeedFCC.hpp` | FCC seed helper used by the host-buffer IC |

## See also

- [`../../docs/applications.md`](../../docs/user_guide/applications.md) — other shipped apps 
- [`../../docs/configuration.md`](../../docs/user_guide/configuration.md) — config file concepts 
- [`../../docs/quickstart.md`](../../docs/quickstart.md) — first-time setup 
