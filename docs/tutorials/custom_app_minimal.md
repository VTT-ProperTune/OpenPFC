<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: minimal custom application (`App` + JSON)

## Why this tutorial exists (read this first)

OpenPFC ships **in-tree** programs (`apps/tungsten`, `apps/aluminumNew`, …) and **library examples** under `examples/` (often a single `main.cpp` with no config file). Many teams need a **third** shape:

- Your simulation code lives in **your own** Git repository and CMake project.  
- You still want the **same JSON/TOML → MPI + HeFFTe + time stepping + writers** pipeline that the shipped apps use.  
- You do **not** want to fork OpenPFC just to add a `main`.

This tutorial is about that **integration shell**: a small **out-of-tree** executable that links OpenPFC, subclasses `pfc::Model`, and runs `pfc::ui::App<MyModel>` so a **config file on disk** drives world size, `dt`, FFT options, optional binary output, IC/BC hooks, etc.

**Style:** keep `MyModel` a thin seam over your physics; prefer **`pfc::get_fft(*this)`**, **`pfc::get_world(*this)`**, **`pfc::initialize(*this, dt)`**-style free functions in `initialize` / `step` so readers see where work happens (see [`../styleguide.md`](../styleguide.md#api-shape-free-functions-and-data-centric-types)).

### What you are *not* building here

- **Not** a full new phase-field model with free energy, stencils, and validation — that is the job of your `Model::initialize` / `Model::step` (and possibly collaborators you add).  
- **Not** a duplicate of the tungsten science story — for physics-heavy context see [`science_tungsten_quicklook.md`](../science_tungsten_quicklook.md), the spectral example ladder [`spectral_examples_sequence.md`](spectral_examples_sequence.md), and [`../extending_openpfc/README.md`](../extending_openpfc/README.md).

### What you *are* building (concrete outcome)

At the end you have:

| Artifact | Role |
|----------|------|
| A **CMake project** outside OpenPFC | `find_package(OpenPFC)` + your sources. |
| A **`MyModel`** subclass | Where **your** physics will run (`initialize`, `step`). This tutorial keeps `step()` minimal on purpose so the file stays short; in a real project it hosts FFT + nonlinear splits, extra fields, etc. |
| A **`main`** that uses `App<MyModel>` | Same `argv[1]` → config path convention as shipped apps (`mpirun … ./my_app settings.json`). |
| A **JSON (or TOML)** file | Same *shape* as spectral apps: domain, time, `plan_options`, optional `fields` / `saveat`, optional `initial_conditions` — see [`../app_pipeline.md`](../app_pipeline.md) and [`../spectral_app_config_reference.md`](../spectral_app_config_reference.md). |

So the **point** of the tutorial: you leave knowing **where your code plugs in** (`Model`), **what the framework already does** (parse config, build FFT + decomposition + `Simulator`, call `step` in a loop), and **how to compile and run** your own binary against an installed OpenPFC — not how to derive a new PFC free energy.

### Mental model (one sentence)

**Config file** → **`App<MyModel>`** builds **world + FFT + time + `Simulator`**, then repeatedly calls **`MyModel::step(t)`** — your job is to make `step` physically meaningful; OpenPFC’s job is to make MPI, HeFFTe, and I/O consistent with that config.

If the diagram in [`../app_pipeline.md`](../app_pipeline.md) (*Big picture*) is still fuzzy, skim that section once, then continue here.

---

## Prerequisites

- OpenPFC **installed** or on `CMAKE_PREFIX_PATH` so `find_package(OpenPFC)` works ([`INSTALL.md`](../../INSTALL.md)).  
- Same MPI / HeFFTe expectations as any OpenPFC run.

---

## Part A — What you implement (three pieces)

| Piece | Responsibility |
|-------|------------------|
| **`MyModel`** | Subclass `pfc::Model`, override `initialize` and `step`. **All interesting physics belongs here** (fields, FFT usage, nonlinear terms). This tutorial uses a stub `step` so we can focus on wiring. |
| **Optional `from_json`** | If you use `model.params` in JSON, implement `void from_json(const pfc::ui::json &, MyModel &)` so `App` can push parameters into the model after construction (same idea as `apps/aluminumNew/Aluminum.hpp`). |
| **`main`** | Construct `pfc::ui::App<MyModel>(argc, argv)` and return `app.main()`. Optionally register custom `FieldModifier` types **before** constructing `App` (see Part E). |

**Reference code in this repository** (in increasing weight):

- Smallest UI + JSON in memory: `examples/10_ui_register_ic.cpp` (good for tests).  
- File-based config + real physics: `apps/aluminumNew/aluminumNew.cpp` + `Aluminum.hpp` (production shape).

---

## Part B — CMake (separate project)

**Purpose:** tell your project where OpenPFC is installed and link the same targets the in-tree apps use.

Install OpenPFC, then point `CMAKE_PREFIX_PATH` at the prefix that contains `lib/cmake/OpenPFC/`.

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_openpfc_app LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenPFC REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(my_app main.cpp my_model.cpp)
target_link_libraries(my_app PRIVATE OpenPFC nlohmann_json::nlohmann_json)
```

`nlohmann_json` is used inside OpenPFC’s UI headers; your target still needs `find_package(nlohmann_json)` and `nlohmann_json::nlohmann_json` when you include `App` / `pfc::ui::json` paths.

You may merge everything into one `main.cpp` if you prefer.

---

## Part C — Minimal model (skeleton)

**Purpose:** give `App` something that satisfies the `Model` interface. The constructor signature must match what `SpectralSimulationSession` expects when it builds your type.

Your model constructs with `(pfc::FFT &fft, const pfc::World &world)` and may take an optional third argument `MPI_Comm` (default `MPI_COMM_WORLD`) so rank-0 logging matches the session communicator — see `pfc::Model` in `include/openpfc/kernel/simulation/model.hpp`.

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

Implement `from_json` only if you read `model.params`. Use the signature above so ADL matches `openpfc/frontend/ui/from_json.hpp`.

**What goes inside `step` in a real app** — typically: forward FFTs, multiply by k-space operators, inverse FFTs, nonlinear terms in real space, coupling to auxiliary fields — the same semi-implicit spectral pattern described in `Model`’s Doxygen and in [`../spectral_stack.md`](../spectral_stack.md). This tutorial omits that body so the page stays a **wiring guide**, not a duplicate of the spectral examples.

---

## Part D — `main` and how you run it

**Purpose:** hand control to OpenPFC’s `App` so it loads `argv[1]`, builds the session, and runs the integrator.

```cpp
#include "my_model.hpp"
#include <iostream>
#include <openpfc/frontend/ui/app.hpp>

int main(int argc, char **argv) {
  try {
    pfc::ui::App<MyModel> app(argc, argv);
    return app.main();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
```

```bash
mpirun -n 4 ./my_app /path/to/settings.json
```

### In-memory JSON (tests only)

`App<Model>` also supports `App(pfc::ui::json settings, MPI_Comm comm)` — no `argv[1]`. `examples/10_ui_register_ic.cpp` builds JSON in C++ for tests or CI. Production binaries should use the file path constructor so operators can swap configs without recompiling.

If you bypass `App::main` but still call `from_json` paths manually, call `pfc::ui::configure_spectral_json_driver_hooks(comm, mpi_rank)` once so FFT parse logs and NaN-check defaults match the communicator (`App::main` does this for you).

---

## Part E — Minimal JSON (what the file must express)

**Purpose:** the same **declarative** surface as shipped spectral apps — grid, time, FFT planner, optional writers.

Your file must include enough keys for `SpectralCpuStack`: world (`Lx`, `Ly`, `Lz`, spacing, origin), time (`t0`, `t1`, `dt`), `plan_options` if you need non-default FFT behavior, plus `model.name` / `model.params` as needed. Copy structure from `examples/fft_backend_selection.toml` or `apps/tungsten/inputs_json/tungsten_single_seed.json` and delete sections you do not use.

When `saveat > 0`, add `fields` with `name` and `data` paths so binary writers register ([`../io_results.md`](../io_results.md)).

---

## Part F — Optional: custom initial condition

**Purpose:** register a named modifier so JSON `initial_conditions[].type` can construct your IC.

1. Subclass `pfc::FieldModifier`, implement `apply`, optionally `from_json` for parameters.  
2. Before constructing `App`:

   ```cpp
   pfc::ui::register_field_modifier<MyIC>("my_ic_type");
   ```

3. In JSON, use `"type": "my_ic_type"` under `initial_conditions` ([`../app_pipeline.md`](../app_pipeline.md)).

See `examples/10_ui_register_ic.cpp` end-to-end.

---

## Part G — Debugging tips

- Start from `examples/10_ui_register_ic.cpp` and swap in your model class.  
- If `find_package(OpenPFC)` fails, set `OpenPFC_DIR` or `CMAKE_PREFIX_PATH` ([`../getting_started/01-basics/README.md`](../getting_started/01-basics/README.md)).  
- MPI / HeFFTe mismatches: [`../troubleshooting.md`](../troubleshooting.md).

---

## Suggested reading order (if you felt lost)

1. [`../app_pipeline.md`](../app_pipeline.md) — **what** JSON does (`SpectralCpuStack` → session → `Simulator`).  
2. This page — **how** to host that pipeline in **your** CMake project.  
3. [`spectral_examples_sequence.md`](spectral_examples_sequence.md) — **physics-shaped** examples inside the OpenPFC tree.  
4. [`../parameter_validation.md`](../parameter_validation.md) — optional validated `model.params`.

## See also

- [`README.md`](README.md) — all tutorials in `docs/tutorials/`  
- [`../class_tour.md`](../class_tour.md) — map of main types  
- [`../extending_openpfc/README.md`](../extending_openpfc/README.md) — extension checklist  
