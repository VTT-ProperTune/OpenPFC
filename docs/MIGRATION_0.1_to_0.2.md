<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Migrating from OpenPFC 0.1 to 0.2

This is the user-facing replacement list for the 0.2 architecture. The
historical milestone table used during the refactor is in
[`docs/archive/`](archive/README.md).
Use this page when porting an application or example off the 0.1 `Model` /
`Simulator` / `App` path.

0.2 production physics (tungsten, aluminumNew) already runs on the generic
`pfc::ui::SpectralETDSession<Physics, Stack>` (over `pfc::sim::SpectralETDSystem`)
and `pfc::sim::run`. There is no
`pfc::Model`, `pfc::Simulator`, or `pfc::ui::App`.

## Architecture in one paragraph

A 0.2 run owns a **domain** (`pfc::Domain` + `pfc::Box3i`), a **stack**
(`SpectralCPUStack`, `GPUSpectralStack`, `FDCPUStack`, or `FDGPUStack`), optional
**`SimulationState`** for named fields, a **`Time`** object, and a thin loop
(`pfc::sim::run` / `SimulationDriver`). Physics is a callable `step(t)` (or a
composed stepper). JSON still describes domain, time, `method`/`backend`,
`plan_options`, initial conditions, writers, and `checkpoint` /
`restart_from`.

## Removed symbols → replacements

| 0.1 | 0.2 |
|-----|-----|
| `pfc::Model` (virtual `step` / `initialize`) | Physics struct or lambda; for stiff spectral models a `SpectralETDPhysics` consumed by `pfc::sim::SpectralETDSystem<Physics, MemorySpace>`; `pfc::sim::run` |
| `pfc::Simulator` | `pfc::sim::run` / `pfc::sim::SimulationDriver` |
| `pfc::ui::App<Model>` / `JsonAppRun` / `SpectralSimulationSession` | `pfc::ui::make_simulation_session<Stack>` + session `run()`, or `pfc::ui::SpectralETDSession<Physics, Stack>` for spectral-ETD physics |
| `Model::add_real_field` / `get_real_field` | `pfc::data::Field<T>` on the stack (`stack.u()`) or `SimulationState::get_field<T>(name)` |
| `Simulator::add_initial_conditions` | `pfc::apply_field_modifier(modifier, field, t)` once at start (host or device `Field`), or the JSON `initial_conditions` catalog |
| `Simulator::add_boundary_conditions` | Apply modifiers in the `pfc::sim::run` `apply` hook, or stage-prep (`StagePreparationService`) |
| `Simulator::add_results_writer` | Writer on the `on_save` hook; JSON `parse_result_writers_from_json` |
| `Simulator::step` / `pfc::step(sim)` / `pfc::done(sim)` | `pfc::sim::run(time, step, on_start, apply, on_save)` |
| `pfc::compat::LegacyModelPhysics` (A1) | Deleted — wrap physics as `step(t)` directly |
| `Simulator::step_with_physics` (A2) | Deleted — call the physics stepper from `pfc::sim::run` |
| `FieldModifier::apply(Model&, double)` | `apply(field::FieldOutput<double>, const Domain&, const Box3i&, double)`; a `std::vector<double>` lvalue converts implicitly |
| `pfc::Field` / `RealField` (`std::vector<double>` aliases, `model_types.hpp`) | `pfc::Field<T, MemorySpace>` = `pfc::data::Field` (owning); `field::FieldView<T>` / `field::FieldOutput<T>` (non-owning) at modifier/writer/reader boundaries |
| `ResultsWriter::write(int, const RealField&)` | `write(int, field::FieldView<double>)` (`field.view()` or a `std::vector<double>`) |
| `checkpoint::PublishedFieldBrick` + `std::ofstream` bricks | `publish_checkpoint_directory(final_dir, meta, comm, write_fields)` — MPI-collective; bricks via `brick_io.hpp` |
| `from_json(json, Model&)` stub | App-local `from_json` into a params struct (`apply_tungsten_json`, …) |
| `restart_from` + `simulator.increment` / `result_counter` | `CheckpointService` only (`checkpoint.every`, `restart_from`); mixing the old keys is an error |
| `Box3D`, `csys.hpp`, `world_types.hpp` | `pfc::Box3i`, `pfc::Domain` |
| `DiscreteField` / `Array` / `LocalField` / `PaddedBrick` | `pfc::data::Field<T, MemorySpace>` |
| `IFFT` | `IHostFFT` / `IDeviceFFT<MemorySpace>` |
| `HaloExchanger` family | `comm::HaloExchange<MemorySpace>` / `comm::SparseExchange<MemorySpace>` |
| `pfc::World` / `world::create` / `from_json<World>` / stack `world()` | `pfc::Domain` + `pfc::Box3i` (`domain::create`, `decomposition::local_box`, `decomposition::domain`); JSON is `from_json<Domain>` only |

`from_json<Domain>` shares the `Lx`/`Ly`/`Lz`/`dx`/`dy`/`dz`/`origin` schema
with the old `from_json<World>`.

## Code shape

**0.1**

```cpp
class DiffusionModel : public pfc::Model {
  void initialize(double dt) override;
  void step(double t) override;
};
pfc::Simulator sim(model, time);
sim.add_initial_conditions(std::make_unique<Constant>(0.0));
while (!pfc::done(sim)) pfc::step(sim);
```

**0.2**

```cpp
pfc::Domain domain = pfc::ui::from_json<pfc::Domain>(settings);
pfc::Time time = pfc::ui::from_json<pfc::Time>(settings);
pfc::sim::stacks::SpectralCPUStack stack(std::move(domain), rank, nproc);
auto &psi = stack.u();

pfc::Constant ic(0.0);
ic.apply(psi.vec(), psi.domain(), psi.box(), 0.0);

pfc::sim::run(
    time,
    [&](double t) { physics.step(t); },
    /* on_start */ {},
    /* apply BCs */ {},
    /* on_save */ [&](const pfc::Time &clock) { writer.write(pfc::time::increment(clock), psi.vec()); });
```

JSON sessions:

```cpp
auto session = pfc::ui::make_simulation_session<pfc::sim::stacks::SpectralCPUStack>(
    settings, rank, nproc, comm);
session.run([&](double t) { /* step stack.u() */ });
```

Production tungsten/aluminum keep JSON ICs, BCs, writers, and
`checkpoint`/`restart_from` inside their ETD session types.

## JSON keys that still work

Unchanged: `Lx`/`Ly`/`Lz`, `dx`/`dy`/`dz`, `origin`/`origo`, `t0`/`t1`/`dt`/`saveat`,
`method` (`spectral`|`fd`), `backend` (`cpu`|`cuda`|`hip`), `plan_options`,
`fields[]`, `initial_conditions[]`, `boundary_conditions[]`, `checkpoint`,
`restart_from`.

Changed:

- There is no `pfc::ui::App`. Drivers construct a session and call `run()`.
- `simulator.integrator.method` still overlays `Time::method()`
  (`overlay_simulator_integrator_method` / `apply_simulator_section_from_json`).
- `simulator.increment` may overlay `Time::set_increment`.
- `simulator.result_counter` is ignored; dump indices come from `Time` or the
  checkpoint bundle.
- `restart_from` cannot be combined with `simulator.increment` or
  `simulator.result_counter`.

## Field modifiers

Implement one method:

```cpp
void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
           const pfc::Box3i &box, double time) override;
```

JSON catalogs (`default_field_modifier_catalog()`,
`parse_initial_conditions_from_json`) still resolve `type` strings. Apply the
returned modifiers on the host buffer that matches `box`.

## Checkpoints

`pfc::ui::make_checkpoint_service(settings, comm)` builds a
`CheckpointService`. Sessions call `restore_from_config(state, time)` after
declaring fields, and `maybe_save(state, time)` from the physics step. There is
no Gen-1 `restore_gen1_from_checkpoint`.

## Where to look

| Task | Header / example |
|------|------------------|
| Time loop | `include/openpfc/kernel/simulation/simulation_driver.hpp` |
| JSON → session | `include/openpfc/frontend/ui/from_json_simulation_session.hpp` |
| JSON FD heat | `include/openpfc/frontend/ui/json_fd_session.hpp` |
| JSON ICs / writers | `include/openpfc/frontend/ui/simulation_wiring_conditions.hpp`, `simulation_wiring_writers.hpp` |
| Spectral-ETD JSON session | `include/openpfc/frontend/ui/json_spectral_etd_session.hpp` |
| Tungsten production | `apps/tungsten/include/tungsten/tungsten_session.hpp`, `tungsten_physics.hpp` |
| Spectral example | `examples/04_diffusion_model.cpp`, `examples/05_simulator.cpp` |
| Custom JSON IC | `examples/10_ui_register_ic.cpp` |
| API catalog | `docs/api/examples/` |

## Adapters A0–A2

| ID | Status in 0.2 |
|----|----------------|
| A1 `LegacyModelPhysics` | **Deleted** |
| A2 `step_with_physics` | **Deleted** |
| A0 `pfc::World` | **Deleted** — use `Domain` + `Box3i` |
