<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: minimal custom application (`App` + JSON)

This tutorial sketches a small out-of-tree program that links OpenPFC and runs `pfc::ui::App<MyModel>` with a `JSON` file. It complements `[`../getting_started/01-basics/README.md`](../getting_started/01-basics/README.md)` (library linking) and `[`../app_pipeline.md`](../app_pipeline.md)` (what the config contains).

Prerequisites: OpenPFC installed or visible to CMake (`[`INSTALL.md`](../../INSTALL.md)`), MPI and HeFFTe as for any OpenPFC build.

## 1. What you will implement

- `MyModel` — subclass `pfc::Model`, override `initialize`, `step`.
- `from_json` — optional `void from_json(const pfc::ui::json &, MyModel &)` so `App` can apply `model.params` after the session is built (same pattern as `apps/aluminumNew/Aluminum.hpp`).
- `main` — register any `FieldModifier` types, construct `App<MyModel>`, call `app.main()`.

Reference implementations:

- Smallest UI demo: `examples/10_ui_register_ic.cpp` (in-process JSON; good for tests).
- Production shape: `apps/aluminumNew/aluminumNew.cpp` + `Aluminum.hpp` (file-based config, real physics).

## 2. CMake (separate project)

Install OpenPFC, then point `CMAKE_PREFIX_PATH` at the install prefix (the directory that contains `lib/cmake/OpenPFC/`).

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_openpfc_app LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenPFC REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(my_app main.cpp my_model.cpp)
target_link_libraries(my_app PRIVATE OpenPFC nlohmann_json::nlohmann_json)
```

`nlohmann_json` is linked privately inside `openpfc`, so your target still needs `find_package(nlohmann_json)` and `nlohmann_json::nlohmann_json` when you include `App` / `pfc::ui::json` headers (those pull in `<nlohmann/json.hpp>`).

You can consolidate to a single `main.cpp` if you keep the model in the same file.

## 3. Minimal model header (sketch)

Your model must construct with `(pfc::FFT &fft, const pfc::World &world)` and may take an optional `MPI_Comm` (third argument, default `MPI_COMM_WORLD`) so rank-0 checks match the `App` session communicator — see `pfc::Model` in `openpfc/kernel/simulation/model.hpp`.

```cpp
// my_model.hpp
#pragma once
#include <openpfc/frontend/ui/json_helpers.hpp>
#include <openpfc/kernel/simulation/model.hpp>

class MyModel : public pfc::Model {
public:
  explicit MyModel(pfc::FFT &fft, const pfc::World &world,
                   MPI_Comm mpi_comm = MPI_COMM_WORLD)
      : pfc::Model(fft, world, mpi_comm) {}
  void initialize(double dt) override;
  void step(double t) override;
};

// Optional: called by App when "model.params" exists in JSON
void from_json(const pfc::ui::json &j, MyModel &m);
```

Implement `from_json` only if you read `model.params`. Use `void from_json(const pfc::ui::json &, MyModel &)` (or `nlohmann::json`) so ADL matches the call in `openpfc/frontend/ui/from_json.hpp`.

## 4. `main` with a config file

Use the same argv convention as shipped apps: first argument = path to `.json` or `.toml`.

```cpp
#include "my_model.hpp"
#include <iostream>
#include <openpfc/frontend/ui/app.hpp>

int main(int argc, char argv) {
  try {
    pfc::ui::App<MyModel> app(argc, argv);
    return app.main();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
```

Run with MPI (same toolchain you built against):

```bash
mpirun -n 4 ./my_app /path/to/settings.json
```

### In-memory JSON (tests and examples)

`pfc::ui::App<Model>` also has a constructor `App(pfc::ui::json settings, MPI_Comm comm)` that skips `argv[1]` and uses an in-memory object instead. `examples/10_ui_register_ic.cpp` builds JSON in C++ and passes it to `App`—useful for unit tests or CI where there is no config file on disk. Production binaries should prefer the `argc`/`argv` constructor so users pass a path.

## 5. Minimal JSON

Your file must include enough keys for `SpectralCpuStack`: grid (`Lx`, `Ly`, `Lz`, spacing, origin), time (`t0`, `t1`, `dt`), `plan_options` if you rely on non-default FFT settings, plus `model.name` / `model.params` as needed. Copy structure from `examples/fft_backend_selection.toml` or `apps/tungsten/inputs_json/tungsten_single_seed.json` and shrink fields you do not use.

When `saveat` is greater than zero, add `fields` with `name` and `data` paths so binary writers register ([`../io_results.md`](../io_results.md)).

## 6. Optional: custom initial condition

1. Subclass `pfc::FieldModifier`, implement `apply`, optionally `from_json` for parameters.
2. In `main`, before constructing `App`:

   ```cpp
   pfc::ui::register_field_modifier<MyIC>("my_ic_type");
   ```

3. Reference `"type": "my_ic_type"` under `initial_conditions` in JSON ([`../app_pipeline.md`](../app_pipeline.md)).

See `examples/10_ui_register_ic.cpp` end-to-end.

## 7. Debugging tips

- Start from `examples/10_ui_register_ic.cpp` with your model class swapped in.
- If `find_package(OpenPFC)` fails, set `OpenPFC_DIR` or `CMAKE_PREFIX_PATH` ([`../getting_started/01-basics/README.md`](../getting_started/01-basics/README.md)).
- Use [`../troubleshooting.md`](../troubleshooting.md) for MPI / HeFFTe mismatches.

## See also

- [`../class_tour.md`](../class_tour.md) — map of main types
- [`../parameter_validation.md`](../parameter_validation.md) — optional validated `model.params`
- [`../extending_openpfc/README.md`](../extending_openpfc/README.md)** — extension checklist
