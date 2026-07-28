<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Execution and fields

These types connect logical fields to host or device execution. Backend setup
and support constraints belong in the
[GPU path decision guide](../hpc/gpu_path_decision.md); halo layout choices are
explained in [Halo exchange](../concepts/halo_exchange.md).

## `pfc::View`

```{doxygenstruct} pfc::View
:project: OpenPFC
:members:
:no-link:
```

## `pfc::data::Field`

```{doxygenclass} pfc::data::Field
:project: OpenPFC
:members:
:protected-members:
:no-link:
```

## Field storage layouts

The canonical field container `pfc::data::Field<T, MemorySpace>` provides
unified storage for grid-structured data with explicit halo configuration.
Unlike the legacy field zoo (LocalField, PaddedBrick, DiscreteField), the
modern `Field` type supports both host and device memory spaces while maintaining
compatible access patterns.

### Creating fields with factory functions

Use the `field_from_subdomain*` factory functions to create fields from
decomposition geometry:

```cpp
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/domain/create.hpp>

using namespace pfc;
using namespace pfc::data;

// Create domain and decomposition
auto domain = pfc::domain::create({128, 128, 128});
auto decomp = pfc::decomposition::create(domain, 4);  // 4 subdomains
int rank = 0;  // Current process rank

// Create padded field (equivalent to legacy PaddedBrick)
auto my_field = field_from_subdomain<double>(decomp, rank, 2);  // halo width 2
// Explicit type:
Field<double, HostSpace> my_field = field_from_subdomain<double>(decomp, rank, 2);

// Create unpadded field with iteration halo (equivalent to legacy LocalField)
auto unpadded_field = field_from_subdomain_unpadded<double>(decomp, rank, 1);
```

### Field geometry and indexing

`Field` stores geometry by value and provides query methods compatible with
legacy access patterns:

```cpp
// Geometry queries
pfc::Int3 local_size = my_field.local_size();      // {nx, ny, nz} owned cells
pfc::Int3 global_size = my_field.global_size();    // {Nx, Ny, Nz} domain size
pfc::Int3 lower_global = my_field.lower_global();  // Global index of (0,0,0)
pfc::Real3 spacing = my_field.spacing();          // Grid spacing {dx, dy, dz}
pfc::Real3 origin = my_field.origin();            // Physical origin {ox, oy, oz}

// Indexing (same API as LocalField/PaddedBrick)
double value = my_field(i, j, k);                  // Access at local indices
my_field(i, j, k) = new_value;                     // Write at local indices

// Linear indexing
std::size_t linear_idx = my_field.idx(i, j, k);   // Convert to linear index
double linear_value = my_field.data()[linear_idx]; // Raw buffer access
```

### Memory space support

`Field` supports both host and device backends:

```cpp
// Host field (default)
Field<double, HostSpace> host_field = field_from_subdomain<double>(decomp, rank, halo);

// Device field (requires CUDA/HIP backend)
Field<double, CudaSpace> device_field = field_from_subdomain<double>(decomp, rank, halo);
Field<double, HipSpace> hip_field = field_from_subdomain<double>(decomp, rank, halo);

// Host-device synchronization for device fields
device_field.sync_to_device();              // Push host->device before device kernel
device_field.note_device_write();          // Mark device side as modified
device_field.with_host_view([&](double* ptr, std::size_t size) {
    // Device data pulled to host, access via ptr, then host marked modified
});
```

### Field iteration

Iterate over field regions using the provided methods:

```cpp
// Visit all owned cells (local indices [0, nx), [0, ny), [0, nz))
my_field.for_each_owned([&](int i, int j, int k) {
    double& value = my_field(i, j, k);
    // Computation on owned cells only
});

// Visit interior cells ( halo-width buffered regions)
my_field.for_each_interior([&](const pfc::Real3& coords, double value) {
    // coords: physical coordinates {x, y, z}
    // value: field value at this location
    // Computation on interior region (halo-excluded)
});

// Fill field by sampling a function
my_field.apply([&](double x, double y, double z) {
    return std::sin(x) * std::cos(y) * std::exp(-z);
});
```

## LEGACY: `pfc::field::LocalField`

<!-- LEGACY: This demonstrates historical LocalField usage. Current code should use pfc::data::field_from_subdomain or field_from_subdomain_unpadded instead. -->

```{doxygenclass} pfc::field::LocalField
:project: OpenPFC
:members:
:protected-members:
:no-link:
```

**Migration pattern**:
```cpp
// LEGACY (old):
// LocalField<double> my_field = LocalField<double>::from_subdomain(decomp, rank, halo);

// NEW (modern):
auto my_field = pfc::data::field_from_subdomain_unpadded<double>(decomp, rank, halo);
```

## LEGACY: `pfc::field::PaddedBrick`

<!-- LEGACY: This demonstrates historical PaddedBrick usage. Current code should use pfc::data::field_from_subdomain with halo parameter instead. -->

```{doxygenclass} pfc::field::PaddedBrick
:project: OpenPFC
:members:
:protected-members:
:no-link:
```

**Migration pattern**:
```cpp
// LEGACY (old):
// PaddedBrick<double> padded_field(decomp, rank, halo_width);

// NEW (modern):
auto padded_field = pfc::data::field_from_subdomain<double>(decomp, rank, halo_width);
```

## Boundary conditions with padding

For finite-difference methods requiring halo regions, use `field_from_subdomain`
with explicit halo width. The halo cells provide boundary storage for stencil
operations and MPI communication.

### Creating padded fields

```cpp
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;
using namespace pfc::data;

// Create field with 2-cell halo on each side
auto field = field_from_subdomain<double>(decomp, rank, 2);

// Geometry includes halo
int halo = field.halo_width();                    // 2
int total_width = field.local_size()[0] + 2 * halo; // nx + 4

// Halo indices are addressable in [-halo, n + halo) on every axis
double ghost_value = field(-1, j, k);             // Left halo cell
double interior_value = field(i, j, k);           // Owned cell
```

### Configuring storage vs iteration halos

For face-halo FD layouts (unpadded storage with iteration halo):

```cpp
// Storage is tightly packed, iteration halo is metadata only
auto face_halo_field = field_from_subdomain_unpadded<double>(decomp, rank, 2);

// storage_halo = 0 (no padding in buffer)
// iteration_halo = 2 (width used by for_each_interior and FdGradient)
int storage_halo = face_halo_field.storage_halo();      // 0
int iteration_halo = face_halo_field.halo_width();      // 2
```

### LEGACY: Boundary conditions with PaddedBrick

<!-- LEGACY: This demonstrates historical PaddedBrick halo usage. Current code should use pfc::data::field_from_subdomain with halo parameter. -->

```cpp
// LEGACY (old):
// PaddedBrick<LocalField<double>> padded_field(my_field, 2);

// NEW (modern):
auto padded_field = pfc::data::field_from_subdomain<double>(decomp, rank, 2);
```

## Discrete field operations

Binary and discrete operations use the standard `Field` type with integer element
types. The same iteration and access patterns apply to discrete fields as to
continuous fields.

### Creating discrete fields

```cpp
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;
using namespace pfc::data;

// Create integer field for masking/material regions
auto mask = field_from_subdomain<int>(decomp, rank, 1);

// Create boolean field for state tracking
auto state_field = field_from_subdomain<bool>(decomp, rank, 0);

// Apply initial conditions
mask.apply([&](double x, double y, double z) {
    if (x < 0.5) return 1;  // Material 1
    else return 2;           // Material 2
});
```

### Field transformations

```cpp
// Transform field by coordinate function
Field<double, HostSpace> u = field_from_subdomain<double>(decomp, rank, 1);

// Binary field from continuous field
auto mask = field_from_subdomain<int>(decomp, rank, 0);
mask.for_each_owned([&](int i, int j, int k) {
    mask(i, j, k) = (u(i, j, k) > 0.5) ? 1 : 0;
});

// Coordinate-based sampling
auto sampled = field_from_subdomain<double>(decomp, rank, 0);
sampled.apply([&](double x, double y, double z) {
    return std::exp(-(x*x + y*y + z*z));
});
```

### LEGACY: Discrete field operations

<!-- LEGACY: This demonstrates historical DiscreteField usage. Current code should use pfc::data::Field with appropriate element type (int, bool, etc.). -->

```cpp
// LEGACY (old):
// DiscreteField<int> mask(grid);
// mask.set_region(1, [&](int i, int j, int k) { /* condition */ });

// NEW (modern):
auto mask = pfc::data::field_from_subdomain<int>(decomp, rank, 0);
mask.for_each_owned([&](int i, int j, int k) {
    if (/* condition */) {
        mask(i, j, k) = 1;
    } else {
        mask(i, j, k) = 0;
    }
});
```