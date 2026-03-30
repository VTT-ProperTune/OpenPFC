<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# OpenPFC developer style guide

This document summarizes **how we organize code, name things, and shape APIs** (free functions, data-centric types) in OpenPFC so new contributors can align with existing practice. It complements [CONTRIBUTING.md](../CONTRIBUTING.md) (legal and contribution flow), [INSTALL.md](../INSTALL.md) (build and dependencies), and [architecture.md](architecture.md) (layers and dependency rules).

## Language and tooling

- **C++17** is required (`CMAKE_CXX_STANDARD` is 17; extensions off).
- **Formatting:** follow the repository [`.clang-format`](../.clang-format) at the root. When in doubt, run `clang-format` on touched files before submitting.
- **Static analysis:** optional `clang-tidy` via CMake option `USE_CLANG_TIDY` (see `cmake/CompilerSettings.cmake`).
- **MPI:** OpenPFC is built and tested with **OpenMPI** in typical workflows; use the same MPI stack for HeFFTe and OpenPFC (see INSTALL.md).

## License and file headers

- The project is **AGPL-3.0-or-later**. New source files should carry the standard **SPDX** lines used elsewhere, for example:

  ```cpp
  // SPDX-FileCopyrightText: YYYY VTT Technical Research Centre of Finland Ltd
  // SPDX-License-Identifier: AGPL-3.0-or-later
  ```

- Use the same pattern in **CMake** files that already use SPDX comments (`# SPDX-FileCopyrightText: ...`).
- Contributing implies the **copyright transfer** described in [CONTRIBUTING.md](../CONTRIBUTING.md); read that file before large changes.

## Repository layout (where things go)

| Area | Role |
|------|------|
| [`include/openpfc/`](../include/openpfc/) | **Public API** — headers only, mirroring the kernel / runtime / frontend split (see below). |
| [`src/openpfc/`](../src/openpfc/) | **Library implementation** — `.cpp` (and backend-specific sources) for symbols declared in `include/`. Paths mirror the include tree (e.g. `kernel/data/world.cpp`). |
| [`apps/`](../apps/) | **Full applications** (e.g. scenario-specific drivers, JSON inputs). Each app has its own `CMakeLists.txt` where appropriate. |
| [`examples/`](../examples/) | **Small, teaching executables** — prefer clear names; many tutorials use numeric prefixes (`01_…`, `02_…`). |
| [`tests/unit/`](../tests/unit/) | **Unit tests** (Catch2), grouped under `kernel/`, `runtime/`, `frontend/`, etc., mirroring library structure. |
| [`tests/integration/`](../tests/integration/) | **Integration tests** — multi-component or heavier scenarios. |
| [`tests/benchmarks/`](../tests/benchmarks/) | **Benchmarks** (optional). Sources are compiled only when **`OpenPFC_BUILD_BENCHMARKS=ON`**; see [tests/benchmarks/README.md](../tests/benchmarks/README.md). |
| [`cmake/`](../cmake/) | **Build logic** included from the root `CMakeLists.txt` — prefer adding options and target wiring here rather than inflating the root file. |
| [`docs/`](../docs/) | **Human-readable documentation** (architecture, guides, Doxygen-related assets). |

**HeFFTe** (and similar large third-party trees) must **not** live inside the OpenPFC clone; install to a prefix (e.g. under `$HOME/opt/heffte/...`) as described in INSTALL.md.

## Layers: kernel, runtime, frontend

The mental model is fixed: **kernel → runtime → frontend** in terms of allowed dependencies (frontend may use kernel + runtime; runtime uses kernel only; **kernel must not** include or depend on runtime or frontend). Details and diagrams are in [architecture.md](architecture.md).

Practical rules:

- **Kernel** (`include/openpfc/kernel/...`): backend-agnostic simulation core (data, decomposition, execution abstractions on CPU/host, field ops, FFT *interface*, simulation, MPI helpers, profiling). **Do not** add `#ifdef OpenPFC_ENABLE_CUDA` / HIP switches here; GPU code belongs in runtime.
- **Runtime** (`include/openpfc/runtime/...`): **cpu**, **cuda**, **hip**, and **common** (shared between backends). CUDA/HIP tags, device memory, device `parallel_for`, and backend FFT implementations live here.
- **Frontend** (`include/openpfc/frontend/...`): optional application-facing pieces (UI, JSON/TOML helpers, extra I/O, logging). Minimal simulations can avoid this layer entirely.

When adding a feature, choose the **lowest** layer that can express it without breaking the dependency graph.

## Naming conventions

### Files and directories

- Headers: **`snake_case.hpp`**, implementation: **`snake_case.cpp`**.
- Directory names: **`snake_case`**, describing content (e.g. `decomposition`, `initial_conditions`). Avoid vague catch-alls like **`core`** or **`all`**. **`common`** is reserved for code **shared by sibling components** (e.g. `runtime/common` for HeFFTe adapter code used by multiple backends).
- Unit tests: **`test_<topic>.cpp`** in the subdirectory that matches the component under test.

### C++ identifiers

- **Namespaces:** top-level **`pfc`**, with **nested namespaces** for areas (`pfc::world`, `pfc::decomposition`, …) matching headers and responsibility. Prefer narrow namespaces over dumping everything into `pfc`.
- **Classes / structs / enums:** **`PascalCase`** (e.g. `HaloExchanger`, `Decomposition`).
- **Functions and variables:** **`snake_case`** for most APIs; follow the style of the file you are editing.
- **Non-static data members:** **`m_` prefix** plus **`snake_case`** (e.g. `m_tic`, `m_lap_started`, `m_min_level`). Prefer this for new code and when editing a type for other reasons. Avoid trailing underscores on members (`tic_`, `duration_`) in new or heavily touched types—some older headers still use suffix style; migrate opportunistically rather than mass-renaming unrelated files.
- **Macros / compile-time flags:** **`OPENPFC_*`** or existing macro families (e.g. profiling macros in `kernel/profiling`); avoid introducing generic unprefixed macros.

### CMake

- **Targets:** the main library is **`openpfc`** (alias **`OpenPFC`**). Tests aggregate as **`openpfc-tests`**. Match existing executable names in `examples/` and `apps/`.
- **Options:** **`OpenPFC_*`** prefix for project-specific cache variables (see `cmake/ProjectSetup.cmake` and related modules).

## API shape: free functions and data-centric types

OpenPFC favors a **laboratory, not fortress** style: code should be easy to read, experiment with, and compose. Prefer exposing behavior as **free functions** in the appropriate namespace over **member functions** when the operation is a natural query or transformation on a value, rather than something that must stay tied to hidden invariants.

- **Examples already in the tree:** `pfc::world::get_size(world, …)` takes `World` as an argument; **`get_world`** is a **free function** for `Decomposition` and `Field` (see `get_world(const Decomposition&)` and `get_world(const Field<T>&)` in their headers). Prefer `get_world(decomp)` / `get_world(field)` over a member spelling when both exist.
- **Model and Simulator:** prefer **`pfc::get_world`**, **`pfc::get_fft`**, **`pfc::is_rank0`**, **`pfc::has_field` / `has_real_field` / `has_complex_field`**, **`pfc::get_real_field` / `get_complex_field`**, **`pfc::get_field`** (default named field), **`pfc::initialize(model, dt)`**, **`pfc::step(model, t)`**, plus **`pfc::get_model(sim)`**, **`pfc::get_time(sim)`**, **`pfc::get_field(sim)`**, **`pfc::is_rank0(sim)`**, **`pfc::get_world(sim)`**, **`pfc::get_fft(sim)`** (see `simulation/model.hpp` and `simulation/simulator.hpp`). Members remain for backward compatibility.
- **Name lookup inside `Model` subclasses:** unqualified **`get_fft(*this)`** can conflict with the inherited member `Model::get_fft()`; prefer **`pfc::get_fft(*this)`** and **`pfc::get_world(*this)`** in derived-class bodies.
- **Older APIs:** some types still expose members such as `model.get_world()`. When you add or refactor nearby access, **introduce** a free overload in the same module (e.g. `get_world(const Model&)`) and **call that** in new or touched code. Full migration can be incremental; the direction of travel is free functions at namespace scope.
- **Classes as data carriers:** types should **primarily hold state**. Prefer **`public` data members** when there is no concrete reason to hide them—in practice many types should read like **`struct`s**. Reserve **`private` members** (with the `m_` convention) for cases where hiding genuinely prevents invalid states or where encapsulation is clearly justified, not as a default habit.

This sits alongside the **layer rules** in [architecture.md](architecture.md): kernel/runtime/frontend boundaries still apply; openness is about how each type exposes its *own* fields and helpers, not about crossing forbidden includes.

## Includes and public API

- Prefer **explicit includes** with the full path under `openpfc/`:

  `#include <openpfc/kernel/data/world.hpp>`

- Umbrella headers [`openpfc/openpfc.hpp`](../include/openpfc/openpfc.hpp) and [`openpfc/openpfc_minimal.hpp`](../include/openpfc/openpfc_minimal.hpp) are convenient but pull more than needed; **prefer specific headers** in library and example code for compile times.
- Anything under `include/openpfc/` is treated as **public API**. Subdirectories named **`detail`** (or future **`internal`**) are **not** stability promises—do not rely on them from external projects.

See [architecture.md](architecture.md) for minimal-app include patterns and HeFFTe/runtime headers.

## Adding or moving library code

1. **Header** in `include/openpfc/<layer>/.../name.hpp` (or split headers if the module is large, as with `world_*`).
2. **Source** in `src/openpfc/<layer>/.../name.cpp` if not header-only.
3. **Register** new `.cpp` files on the `openpfc` target in [`cmake/LibraryConfiguration.cmake`](../cmake/LibraryConfiguration.cmake) (generator expressions for CUDA/HIP files follow existing examples).
4. **Tests:** add `test_*.cpp` under the matching `tests/unit/...` tree and list them in the nearest `CMakeLists.txt` via `target_sources(openpfc-tests PRIVATE ...)`.
5. **Examples** (optional): add `.cpp` under `examples/` and wire the executable in [`examples/CMakeLists.txt`](../examples/CMakeLists.txt).

After structural changes, update or add **Doxygen** on public types and functions where the rest of the module is documented (`@file`, `@brief`, `@param`, etc.).

## Documentation and cross-references

- **Design / physics / algorithms:** add or extend Markdown under `docs/` and link from related headers (as with `docs/halo_exchange.md`).
- **User-facing build and HPC notes:** INSTALL.md, `docs/build_cpu_gpu.md`, site-specific guides (e.g. `docs/INSTALL.LUMI.md`).
- **API reference:** built with Doxygen from configured inputs in [`docs/CMakeLists.txt`](CMakeLists.txt).

## Tests and quality gate

- **Framework:** Catch2 v3 (fetched by CMake for tests).
- Run the test target after changes, e.g. `cmake --build <build-dir> --target openpfc-tests` and execute the test binary (see [tests/README.md](../tests/README.md) for conventions).
- Prefer **deterministic** unit tests; use MPI tests only where the behavior under rank layout is what you are validating.

## Summary checklist for a typical change

1. Correct **layer** (kernel vs runtime vs frontend) and **no upward dependencies**.
2. **`snake_case` files**, **`PascalCase` types**, **`m_` data members** where members are private, **`pfc::` namespaces** consistent with neighbors; prefer **free functions** and **public-by-default** data where the API shape section applies.
3. **SPDX header** on new files.
4. **`LibraryConfiguration.cmake`** updated for new `.cpp` files.
5. **Unit tests** where behavior is non-trivial or regression-prone.
6. **`clang-format`** applied; builds cleanly at least in the configuration you use (Debug recommended during development).

For questions not covered here, use **[architecture.md](architecture.md)** and nearby code in the same subdirectory as the canonical reference.
