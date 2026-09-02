<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Build a minimal config-driven application

This tutorial creates an OpenPFC executable in a separate CMake project using
`pfc::ui::make_simulation_session` and `pfc::sim::run`. Shipped 0.2 apps
(tungsten, aluminumNew) drive JSON/TOML through ETD sessions instead. The JSON
keys (domain, time, `plan_options`, modifiers, writers) are the same.

Use this path when the simulation belongs in its own repository. Do not fork
OpenPFC merely to add an application `main`. Porting 0.1 `App<Model>` code:
[`MIGRATION_0.1_to_0.2.md`](../MIGRATION_0.1_to_0.2.md).

## Prerequisites

You need:

- an installed OpenPFC package;
- the same compatible compiler, MPI, and HeFFTe stack used to build OpenPFC;
- `CMAKE_PREFIX_PATH` or `OpenPFC_DIR` pointing to the installation;
- nlohmann-json available to the downstream project when frontend JSON headers
  are used.

Complete installation guidance is in [`INSTALL.md`](../../INSTALL.md).

## Project layout

Create a new directory with this layout:

```text
my-openpfc-app/
├── CMakeLists.txt
├── main.cpp
└── settings.json
```

## Configure the CMake project

The target and enabled languages below match the installed-package consumer
exercised by OpenPFC CI:

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_openpfc_app LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenPFC REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(
  my_app
  PRIVATE
    OpenPFC::openpfc
    nlohmann_json::nlohmann_json
)
```

The C language is enabled because the installed package resolves MPI components
that include an MPI C target. `OpenPFC::openpfc` is the supported installed
target; un-namespaced in-tree aliases are not a downstream contract.

## Drive a spectral session

Physics is a callable `step(t)`, not a `pfc::Model` subclass. Keep the driver
thin and place reusable mechanics in ordinary functions.

```cpp
// main.cpp
#include <exception>
#include <iostream>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/frontend/ui/from_json_simulation_session.hpp>
#include <openpfc/frontend/ui/settings_loader.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

int main(int argc, char **argv) {
  try {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (argc < 2) {
      if (rank == 0) {
        std::cerr << "usage: my_app settings.json\n";
      }
      MPI_Finalize();
      return 1;
    }

    const auto settings = pfc::ui::load_settings_file(argv[1]);
    auto session =
        pfc::ui::make_simulation_session<pfc::sim::stacks::SpectralCPUStack>(
            settings, rank, nproc);
    auto &psi = session.stack().u();
    for (auto &v : psi.vec()) {
      v = 0.0;
    }
    session.run([&](double /*t*/) {
      // Advance psi by one step (FFT / stepper / ETD).
    });
    MPI_Finalize();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
```

`make_simulation_session` reads domain, time, `method`/`backend`, and
`plan_options` from the document. `session.run` is `pfc::sim::run` over
`Time`. For a full spectral implicit-Euler example see
`examples/04_diffusion_model.cpp`. Production tungsten/aluminum sessions add
ICs, BCs, writers, and `CheckpointService` around the same loop.

## Add a configuration

Start from a shipped input, then reduce it to the fields required by your
physics. The exact supported keys belong in the
[Spectral App configuration reference](../reference/spectral_app_config_reference.md).
The conversion from configuration to runtime objects is described in
[Application pipeline](../user_guide/app_pipeline.md).

A document that consumes `model.params` should parse that subtree in the
driver (tungsten uses `apply_tungsten_json`). Optional
[`ParameterValidator`](../user_guide/parameter_validation.md) can run on the
same subtree before construction.

## Next steps

- Register JSON ICs with a catalog: `examples/10_ui_register_ic.cpp`.
- Attach writers on the `on_save` hook: `examples/11_write_results.cpp`.
- Extension checklist: [`extending_openpfc/README.md`](../extending_openpfc/README.md).
