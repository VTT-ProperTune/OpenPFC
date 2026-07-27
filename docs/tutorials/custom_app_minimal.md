<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Build a minimal config-driven application

This tutorial creates an OpenPFC executable in a separate CMake project. The
application uses `pfc::ui::App<MyModel>` so a JSON or TOML file drives the
domain, time integration, FFT setup, modifiers, and result writers.

Use this path when the simulation belongs in its own repository. Do not fork
OpenPFC merely to add an application `main`.

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
├── my_model.cpp
├── my_model.hpp
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

add_executable(my_app main.cpp my_model.cpp)
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

## Define the model seam

`MyModel` owns the application-specific fields and physics. Keep the subclass
thin and place reusable mechanics in ordinary functions and data types.

```cpp
// my_model.hpp
#pragma once

#include <mpi.h>
#include <openpfc/kernel/simulation/model.hpp>

class MyModel : public pfc::Model {
public:
  explicit MyModel(pfc::FFT &fft, const pfc::Domain &domain,
                   MPI_Comm comm = MPI_COMM_WORLD)
      : pfc::Model(fft, domain, comm) {}

  void initialize(double dt) override;
  void step(double time) override;
};
```

```cpp
// my_model.cpp
#include "my_model.hpp"

void MyModel::initialize(double dt) {
  (void)dt;
  // Allocate fields and precompute operators here.
}

void MyModel::step(double time) {
  (void)time;
  // Advance the model by one step here.
}
```

The empty bodies deliberately isolate the integration shell from the physics.
For spectral model implementations, read
[Spectral examples sequence](spectral_examples_sequence.md) and inspect
`examples/04_diffusion_model.cpp` and `examples/12_cahn_hilliard.cpp`.

## Hand control to `App`

```cpp
// main.cpp
#include "my_model.hpp"

#include <exception>
#include <iostream>
#include <openpfc/frontend/ui/app.hpp>

int main(int argc, char **argv) {
  try {
    pfc::ui::App<MyModel> app(argc, argv);
    return app.main();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
```

`App` reads the configuration path from `argv[1]`, initializes MPI-facing
runtime state, builds the spectral stack and simulator, and executes the time
loop.

## Add a configuration

Start from a shipped input that uses the same frontend path, then reduce it to
the fields required by your model. The exact supported keys and nesting belong
in the
[Spectral App configuration reference](../reference/spectral_app_config_reference.md).
The conversion from configuration to runtime objects is described in
[Application pipeline](../user_guide/app_pipeline.md).

A model that consumes `model.params` can provide an ADL-visible conversion
function:

```cpp
void from_json(const pfc::ui::json &input, MyModel &model);
```

Add that function only when the model has parameters to read. Parameter ranges,
required keys, units, and diagnostic reports are covered in
[Parameter validation](../user_guide/parameter_validation.md).

## Build and run

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/openpfc/install
cmake --build build -j"$(nproc)"
mpirun -n 1 ./build/my_app ./settings.json
```

Begin with one MPI rank. Increase the rank count only after the application
works and the launcher has the required slots or scheduler allocation.

## Optional: custom field modifiers

A configuration-selectable initial or boundary condition uses a
`FieldModifier` implementation registered in a catalog before the application
runs. The smallest example is `examples/10_ui_register_ic.cpp`; the extension
choices and catalog lifetime guidance are in
[Extending OpenPFC](../extending_openpfc/README.md).

Prefer an explicit local catalog in tests and reusable libraries. Process-wide
registration is appropriate only when shared mutable registration state is
acceptable for the executable.

## Verify the integration

Before adding substantial physics, verify that:

1. CMake finds the intended OpenPFC installation;
2. the consumer and OpenPFC use compatible MPI and HeFFTe installations;
3. the executable starts with one rank and reads `settings.json`;
4. invalid or missing configuration values fail before time integration;
5. a configured writer creates the expected output artifact.

For build and runtime failures, use [Troubleshooting](../troubleshooting.md).
For the roles of `Domain`, `Model`, `Simulator`, and `App`, use the
[Tour of main types](../reference/class_tour.md).
