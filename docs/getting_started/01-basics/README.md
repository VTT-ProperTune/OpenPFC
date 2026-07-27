<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Library basics: domain, decomposition, and FFT

This tutorial introduces the lowest-level path through OpenPFC:

```text
Domain → Decomposition → FFT → Model → Simulator
```

You will first build a minimal downstream consumer, then connect its concepts to
the repository's runnable MPI and FFT examples.

Complete [Start here](../../start_here_15_minutes.md) before this tutorial. It
verifies that OpenPFC itself builds and that the MPI and HeFFTe stack works.

## What each object owns

| Object | Responsibility |
|--------|----------------|
| `Domain` | Global grid size, spacing, origin, and periodicity |
| `Decomposition` | Partition of the global domain across MPI ranks |
| `FFT` | Distributed real-to-complex and complex-to-real transforms |
| `Model` | Application-specific fields and time-step physics |
| `Simulator` | Time loop, modifiers, and result writers |

The full package boundaries are described in
[Architecture](../../concepts/architecture.md). Exact declarations belong in
the generated API reference.

## 1. Create a downstream project

Create a new directory outside the OpenPFC source tree:

```text
openpfc-domain-check/
├── CMakeLists.txt
└── main.cpp
```

Use the installed package target exercised by the packaging smoke test:

```cmake
cmake_minimum_required(VERSION 3.21)
project(openpfc_domain_check LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenPFC REQUIRED)

add_executable(domain_check main.cpp)
target_link_libraries(domain_check PRIVATE OpenPFC::openpfc)
```

The project enables C as well as C++ because the installed package resolves MPI
components that include an MPI C target.

## 2. Construct a domain

```cpp
// main.cpp
#include <array>
#include <iostream>

#include <openpfc/kernel/data/domain.hpp>

int main() {
  const auto domain = pfc::domain::create({32, 16, 8});
  const auto size = pfc::domain::get_size(domain);

  std::cout << "grid = " << size[0] << " x " << size[1] << " x "
            << size[2] << '\n';
  return 0;
}
```

`domain::create` validates that every grid dimension and spacing component is
positive. Prefer the factory over aggregate initialization so invalid input
fails at the construction boundary.

Configure and run:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/openpfc/install
cmake --build build -j"$(nproc)"
./build/domain_check
```

Expected output:

```text
grid = 32 x 16 x 8
```

When CMake cannot find the package, set `CMAKE_PREFIX_PATH` to the installation
prefix or set `OpenPFC_DIR` to the directory containing
`OpenPFCConfig.cmake`. See [Troubleshooting](../../troubleshooting.md).

## 3. Add MPI decomposition

A `Domain` describes the global grid. `Decomposition` describes how that grid is
split across ranks for distributed transforms and communication.

The maintained example is:

```text
examples/02_domain_decomposition.cpp
```

It constructs the objects with the current factories:

```cpp
const auto domain = pfc::domain::create({32, 4, 4});
const auto decomposition = pfc::decomposition::create(domain, comm_size);
```

Build the examples in the OpenPFC tree and run:

```bash
cmake --build build --target 02_domain_decomposition
mpirun -n 2 ./build/examples/02_domain_decomposition
```

The decomposition is determined by the global grid and communicator size. Code
that owns MPI must initialize it before creating communication-dependent
objects and finalize it only after those objects are no longer used.

## 4. Perform a distributed FFT

The next maintained example is:

```text
examples/03_parallel_fft.cpp
```

Its essential construction sequence is:

```cpp
const auto domain = pfc::domain::create({8, 1, 1});
const auto decomposition = pfc::decomposition::create(domain, num_procs);
auto fft = pfc::fft::create(decomposition);

std::vector<double> input(fft.size_inbox());
std::vector<std::complex<double>> output(fft.size_outbox());
fft.forward(input, output);
```

Build and run it first with one rank:

```bash
cmake --build build --target 03_parallel_fft
mpirun -n 1 ./build/examples/03_parallel_fft
```

Then run with a rank count supported by the example dimensions and your
launcher allocation:

```bash
mpirun -n 2 ./build/examples/03_parallel_fft
```

The real-space inbox and Fourier-space outbox can have different local extents.
Always allocate buffers from `fft.size_inbox()` and `fft.size_outbox()` rather
than deriving local sizes from the global domain manually.

## 5. Move from transforms to a model

A model owns the fields and describes one time step. A typical spectral step
contains:

1. a forward transform from real space;
2. multiplication by linear operators in Fourier space;
3. optional inverse transforms for nonlinear terms;
4. an update of the accepted field state.

Do not build that logic from this page alone. Continue with the maintained
example sequence:

1. `examples/04_diffusion_model.cpp`;
2. `examples/05_simulator.cpp`;
3. `examples/12_cahn_hilliard.cpp`.

The guided reading is in
[Spectral examples sequence](../../tutorials/spectral_examples_sequence.md).

## What to read next

| Goal | Next document |
|------|---------------|
| Understand the data flow through a spectral run | [Spectral stack](../../concepts/spectral_stack.md) |
| Build a JSON-driven application in another repository | [Minimal custom application](../../tutorials/custom_app_minimal.md) |
| Find headers and type roles | [Tour of main types](../../reference/class_tour.md) |
| Select FFT planner and backend options | [HeFFTe plan options](../../tutorials/fft_heffte_plan_options.md) |
| Add field operations without manual nested loops | [Functional field operations](../functional_field_ops.md) |
| Run existing applications | [Applications](../../user_guide/applications.md) |

## Maintenance rule

The runnable files under `examples/` are the source of truth for code-level
examples. This tutorial explains their sequence and contracts; it should not
copy complete implementations that can drift independently from the build.
