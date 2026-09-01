<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# External coupling

OpenPFC can run as a library inside a host solver (FEM, another PDE code, or a
custom orchestrator). The stable surface is small on purpose.

Runnable sketch: [`examples/22_external_coupling.cpp`](../../examples/22_external_coupling.cpp).

## What is stable

| Piece | Header | Role |
|-------|--------|------|
| `pfc::coupling::FieldHandle` | `kernel/simulation/coupling.hpp` | Non-owning host export: name, `FieldView<double>`, owned `Box3i`, spacing, origin, memory space (`"host"`) |
| `export_host_field(state, name)` | same | Build a handle from `SimulationState` |
| `pfc::sim::run` / `SimulationDriver` | `kernel/simulation/simulation_driver.hpp` | Free-function time loop the host can own or call step-by-step |
| `Time::clip_attempt_dt` / `begin_attempt` / `commit_attempt` | `kernel/simulation/time.hpp` | dt negotiation: the host proposes a step; OpenPFC clips to `t1` and the next `saveat` |
| `pfc::mpi::communicator::duplicate` | `kernel/mpi/communicator.hpp` | Opt-in `MPI_Comm_dup` so OpenPFC halo tags do not collide with the host on `MPI_COMM_WORLD` |
| `CheckpointService` | `kernel/simulation/checkpoint_service.hpp` | Restart owned by OpenPFC; coordinate checkpoints with the host clock via `restart_from` |

`FieldHandle` is a **read** export. Write a source or boundary back through
`SimulationState` (typically `get_field<double>(name).apply(...)`) or a
FieldModifier-shaped adapter `apply(SimulationState&, double)` as in the
example header. Do not treat `FieldView` as mutable.

`pfc::coupling::FieldHandle` is not `pfc::FieldHandle<T>` (the typed index
inside `SimulationState`).

## What is not stable

- Device-resident fields (`CUDASpace` / `HIPSpace`): this export is host-only.
- Gen-1 `Model` / `Simulator` accessors.
- Stage buffers, FFT plans, operator caches, stepper rollback scratch.
- Halo cells: the handle’s `owned_box` is the owned index box; ghost values
  need a halo exchange after the host writes owned cells.
- Changing the MPI rank map or global grid between checkpoint and restore.

## dt negotiation

The host owns the outer loop. Before an attempt:

```cpp
const double dt = time.clip_attempt_dt(host_proposed_dt);
time.begin_attempt(dt);
source.apply(state, time.get_accepted_time());
// ... host physics / OpenPFC step ...
time.commit_attempt();
```

`clip_attempt_dt` never lengthens the proposal. Reject with
`time.reject_attempt()` if the host step fails.

## Restart coordination

Use one OpenPFC bundle, not a second host-side dump of the same field:

```json
{
  "restart_from": "results/ckpt/step_10",
  "checkpoint": { "every": 10, "directory": "results/ckpt" }
}
```

`CheckpointService::restore_from_config` loads fields, accepted time,
increment, result counter, and method identity. The host must restart at the
same accepted time; mismatch of grid or method is a hard error. See
[`checkpoint_publish.md`](../development/checkpoint_publish.md).

## Communicator isolation

If the host already uses `MPI_COMM_WORLD` for its own halo or reductions,
duplicate before constructing OpenPFC exchanges:

```cpp
pfc::mpi::communicator world;
auto isolated = world.duplicate();
```

Pass `MPI_Comm(isolated)` into stacks, `CheckpointService`, and
`BinaryReader`. Destroying the wrapper frees the duplicate.
