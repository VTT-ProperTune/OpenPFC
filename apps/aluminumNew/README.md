<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# AluminumNew

Production aluminum binary: JSON/TOML → `pfc::ui::SpectralETDSession<AluminumPhysics, Stack>` (aliases `AluminumSession`, `AluminumCUDASession`, `AluminumHIPSession` in `include/aluminum/aluminum_session.hpp`); the moving-frame mean-field ETD runs on the shared `pfc::sim::SpectralETDSystem`. Initial conditions include `constant` and the app-registered `seed_grid_fcc`.

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
| `src/aluminum.cpp` | `main`: JSON → `AluminumSession` (`aluminumNew` and its alias `aluminum_etd`) |
| `src/aluminum_cuda.cpp` / `src/aluminum_hip.cpp` | `main` for the GPU sessions on `GPUSpectralStack` |
| `src/gpu/aluminum_pointwise.inc` (`.cu` / `.hip`) | Device instantiation of the pointwise nonlinearity |
| `include/aluminum/aluminum_physics.hpp` | Schema, k-space symbols (`linear_symbol`, `filter_mf`, `correlation_kernel`, `nonlinear_symbol`), `pointwise()` |
| `include/aluminum/aluminum_pointwise.hpp` | `OPENPFC_HD` functor: `N`, free-energy density, `temperature_variation(x, t)` |
| `include/aluminum/seed_grid_fcc.hpp` | `seed_grid_fcc` catalog `FieldModifier` (registered by `aluminum::register_catalog()`) |
| `include/aluminum/aluminum_session.hpp` | Session aliases + `register_catalog()` |
| `SeedFCC.hpp` | FCC seed helper used by `seed_grid_fcc` |

## See also

- [`../../docs/applications.md`](../../docs/user_guide/applications.md) — other shipped apps 
- [`../../docs/configuration.md`](../../docs/user_guide/configuration.md) — config file concepts 
- [`../../docs/quickstart.md`](../../docs/quickstart.md) — first-time setup 
