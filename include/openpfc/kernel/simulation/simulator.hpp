// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulator.hpp
 * @brief Simulation orchestration and time integration loop
 *
 * @details
 * This file defines the Simulator class, which orchestrates the execution of
 * phase-field simulations in OpenPFC. The Simulator manages:
 * - Time integration loop (calling Model::step repeatedly)
 * - Initial condition application via FieldModifiers
 * - Boundary condition enforcement
 * - Results output scheduling via ResultsWriters
 * - Simulation checkpointing and restart
 *
 * The Simulator acts as the "main loop" that coordinates all simulation components:
 * @code
 * // Typical simulation setup
 * MyPhysicsModel model(fft, world);
 * pfc::Time time({0.0, 100.0, 0.1}, 1.0);  // t0, t1, dt, saveat
 * pfc::Simulator sim(model, time);  // optional: third arg MPI_Comm (default world)
 *
 * // Add initial conditions
 * sim.add_initial_condition(std::make_unique<pfc::Constant>(0.5));
 *
 * // Add results writer
 * sim.add_results_writer("output", std::make_unique<pfc::BinaryWriter>("data.bin"));
 *
 * // Run simulation
 * sim.run();
 * @endcode
 *
 * This file is part of the Simulation Control module, providing the main
 * execution framework for time-dependent simulations. Per-field result output is
 * delegated to **`simulator_results_dispatch.hpp`**
 * (`write_results_for_registered_fields`). IC/BC application uses
 * **`simulator_field_modifiers_dispatch.hpp`** (`apply_field_modifier_list`).
 *
 * @see model.hpp for physics model implementation
 * @see time.hpp for time state management
 * @see field_modifier.hpp for initial/boundary conditions
 * @see results_writer.hpp for output handling
 */

#ifndef PFC_SIMULATOR_HPP
#define PFC_SIMULATOR_HPP

#include <memory>
#include <mpi.h>
#include <string>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>
#include <openpfc/kernel/simulation/simulator_field_modifiers_dispatch.hpp>
#include <openpfc/kernel/simulation/simulator_modifier_registration.hpp>
#include <openpfc/kernel/simulation/simulator_results_dispatch.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <openpfc/kernel/utils/logging.hpp>
#include <unordered_map>
#include <utility>

namespace pfc {

/**
 * @brief The Simulator class is responsible for running the simulation of the
 * model.
 */
class Simulator {

private:
  Model &m_model;
  Time &m_time;

  std::unordered_map<std::string, std::unique_ptr<ResultsWriter>> m_result_writers;
  std::vector<std::unique_ptr<FieldModifier>> m_initial_conditions;
  std::vector<std::unique_ptr<FieldModifier>> m_boundary_conditions;
  int m_result_counter = 0;
  MPI_Comm m_mpi_comm{MPI_COMM_WORLD};
  bool m_is_rank0{};

  void warn_rank0_(std::string message) const {
    if (!m_is_rank0) {
      return;
    }
    const Logger lg{LogLevel::Warning, 0};
    log_warning(lg, message);
  }

public:
  /**
   * @brief Construct a new Simulator object
   *
   * @param model The model to simulate.
   * @param time The time object to use for simulation.
   * @param mpi_comm Communicator passed to field modifiers (MPI-IO collectives,
   * etc.)
   */
  Simulator(Model &model, Time &time, MPI_Comm mpi_comm = MPI_COMM_WORLD)
      : m_model(model), m_time(time), m_mpi_comm(mpi_comm),
        m_is_rank0(mpi_comm_rank_is_zero(mpi_comm)) {}

  void set_mpi_comm(MPI_Comm mpi_comm) noexcept {
    m_mpi_comm = mpi_comm;
    m_is_rank0 = mpi_comm_rank_is_zero(mpi_comm);
  }

  [[nodiscard]] MPI_Comm mpi_comm() const noexcept { return m_mpi_comm; }

  /**
   * @brief Get the model object
   *
   * @return Model&
   */
  Model &get_model() { return m_model; }

  /**
   * @brief Get the decomposition object
   *
   * @return const Decomposition&
   */
  /**
   * @brief Get the world object
   *
   * @return const World&
   */
  const World &get_world() { return pfc::get_world(m_model); }

  /**
   * @brief Get the FFT object
   *
   * @return fft::IFFT&
   */
  fft::IFFT &get_fft() { return pfc::get_fft(m_model); }

  /**
   * @brief Get the time object
   *
   * @return Time&
   */
  Time &get_time() { return m_time; }

  /**
   * @brief Get the default field object
   *
   * @return Field&
   */
  [[deprecated("Use get_real_field(get_model(sim), \"default\") or named real "
               "fields instead")]]
  Field &get_field() {
    return pfc::get_real_field(m_model, "default");
  }

  void initialize() { pfc::initialize(m_model, m_time.get_dt()); }

  /** @brief Rank 0 in mpi_comm() (same communicator passed to field modifiers) */
  [[nodiscard]] bool is_rank0() const noexcept { return m_is_rank0; }

  unsigned int get_increment() { return get_time().get_increment(); }

  /**
   * @brief Register a results writer for a specific field
   *
   * Associates a ResultsWriter with a named field for periodic output during
   * simulation. The writer will be called automatically when save points are
   * reached in the time integration loop.
   *
   * @param field_name Name of the field to write (must be registered with Model)
   * @param writer Unique pointer to ResultsWriter object (ownership transferred)
   * @return true if writer was successfully registered, false if field doesn't exist
   *
   * @note Writer is called automatically at intervals specified by
   * Time::set_saveat()
   * @note Multiple writers can write the same field to different formats
   * @note Domain decomposition info is automatically configured
   *
   * @warning Field must be registered with Model before calling this
   * @warning Writer takes ownership via std::unique_ptr - don't access after
   * transfer
   *
   * @example
   * ```cpp
   * // Set up simulation
   * MyModel model(fft, world);
   * model.initialize(0.01);
   * Time time({0.0, 100.0, 0.01}, 1.0);  // save every 1.0 time units
   * Simulator sim(model, time);
   *
   * // Register writers for different fields and formats
   * sim.add_results_writer("density",
   *     std::make_unique<BinaryWriter>("density"));
   * sim.add_results_writer("temperature",
   *     std::make_unique<VTKWriter>("temp.vtk"));
   *
   * // Both fields will be written automatically during sim.run()
   * ```
   *
   * @see add_results_writer(std::unique_ptr<ResultsWriter>) for "default" field
   * @see ResultsWriter for available output formats
   * @see Time::set_saveat() to configure output frequency
   */
  bool add_results_writer(const std::string &field_name,
                          std::unique_ptr<ResultsWriter> writer) {
    auto inbox = get_inbox(pfc::get_fft(m_model));
    const auto &world = pfc::get_world(m_model);
    writer->set_domain(get_size(world), inbox.size, inbox.low);

    Model &model = get_model();
    if (!pfc::has_field(model, field_name)) {
      warn_rank0_("Warning: tried to add writer for inexistent field " + field_name +
                  ", RESULTS ARE NOT WRITTEN!");
      return false;
    }
    m_result_writers.insert({field_name, std::move(writer)});
    return true;
  }

  bool add_results_writer(std::unique_ptr<ResultsWriter> writer) {
    warn_rank0_("Warning: adding result writer to write field 'default'");
    return add_results_writer("default", std::move(writer));
  }

  /**
   * @brief Register an initial condition modifier for a field
   *
   * Adds a FieldModifier that will be applied once at t=0 to set the initial
   * state of a field. Common initial conditions include uniform values, seeds,
   * noise, or data loaded from files.
   *
   * Initial conditions are applied in registration order before the first time step.
   *
   * @param modifier Unique pointer to FieldModifier (ownership transferred)
   * @return true if IC was successfully registered, false if target field doesn't
   * exist
   *
   * @note ICs are applied exactly once at simulation start
   * @note Multiple ICs can be composed for the same field (applied in order)
   * @note The modifier's get_field_name() determines which field is modified
   *
   * @warning Field must be registered with Model before adding IC
   * @warning Modifier takes ownership - don't access after transfer
   *
   * @example
   * ```cpp
   * MyModel model(fft, world);
   * model.initialize(0.01);
   * Time time({0.0, 100.0, 0.01}, 1.0);
   * Simulator sim(model, time);
   *
   * // Start with uniform density
   * sim.add_initial_conditions(
   *     std::make_unique<Constant>("density", 0.5));
   *
   * // Add random noise on top
   * sim.add_initial_conditions(
   *     std::make_unique<GaussianNoise>("density", 0.0, 0.01));
   *
   * // Add localized seed
   * sim.add_initial_conditions(
   *     std::make_unique<SphericalSeed>("density", center, radius, 1.0));
   *
   * sim.run();  // ICs applied before first step
   * ```
   *
   * @see add_boundary_conditions() for time-varying conditions
   * @see FieldModifier for creating custom initial conditions
   * @see initial_conditions/ for built-in IC types
   */
  bool add_initial_conditions(std::unique_ptr<FieldModifier> modifier) {
    return try_push_field_modifier_with_model_check(
        get_model(), m_initial_conditions, std::move(modifier),
        k_initial_condition_registration_msg,
        [this](std::string m) { warn_rank0_(std::move(m)); });
  }

  /**
   * @brief Gets the initial conditions of the simulation.
   *
   * This function returns a const reference to the vector of unique pointers to
   * FieldModifier objects that represent the initial conditions of the
   * simulation.
   *
   * @return A const reference to the vector of unique pointers to FieldModifier
   * objects.
   */
  const std::vector<std::unique_ptr<FieldModifier>> &get_initial_conditions() const {
    return m_initial_conditions;
  }

  /**
   * @brief Register a boundary condition modifier for a field
   *
   * Adds a FieldModifier that will be applied at every time step to enforce
   * boundary conditions. BCs are applied after each Model::step() call.
   *
   * Common boundary conditions include fixed values, periodic boundaries,
   * reflective boundaries, or moving boundaries.
   *
   * @param modifier Unique pointer to FieldModifier (ownership transferred)
   * @return true if BC was successfully registered, false if target field doesn't
   * exist
   *
   * @note BCs are applied after every time step
   * @note Multiple BCs can be composed for the same field (applied in order)
   * @note The modifier's get_field_name() determines which field is modified
   * @note For moving boundaries, the modifier can access current time via
   * apply(model, t)
   *
   * @warning Field must be registered with Model before adding BC
   * @warning Modifier takes ownership - don't access after transfer
   * @warning BCs that depend on time must implement time-varying logic internally
   *
   * @example
   * ```cpp
   * MyModel model(fft, world);
   * model.initialize(0.01);
   * Time time({0.0, 100.0, 0.01}, 1.0);
   * Simulator sim(model, time);
   *
   * // Fixed temperature at boundaries
   * sim.add_boundary_conditions(
   *     std::make_unique<FixedValueBC>("temperature",
   *         boundary_region, 300.0));
   *
   * // Moving solidification front
   * sim.add_boundary_conditions(
   *     std::make_unique<MovingFrontBC>("density",
   *         velocity, direction));
   *
   * sim.run();  // BCs applied after every step
   * ```
   *
   * @see add_initial_conditions() for one-time setup
   * @see FieldModifier::apply() for implementation interface
   * @see boundary_conditions/ for built-in BC types
   */
  bool add_boundary_conditions(std::unique_ptr<FieldModifier> modifier) {
    return try_push_field_modifier_with_model_check(
        get_model(), m_boundary_conditions, std::move(modifier),
        k_boundary_condition_registration_msg,
        [this](std::string m) { warn_rank0_(std::move(m)); });
  }

  /**
   * @brief Gets the boundary conditions of the simulation.
   *
   * This function returns a const reference to the vector of unique pointers to
   * FieldModifier objects that represent the boundary conditions of the
   * simulation.
   *
   * @return A const reference to the vector of unique pointers to FieldModifier
   * objects.
   */
  const std::vector<std::unique_ptr<FieldModifier>> &
  get_boundary_conditions() const {
    return m_boundary_conditions;
  }

  void set_result_counter(int result_counter) { m_result_counter = result_counter; }

  int get_result_counter() const { return m_result_counter; }

  void write_results() {
    const int file_num = get_result_counter();
    pfc::write_results_for_registered_fields(get_model(), m_result_writers,
                                             file_num);
    set_result_counter(file_num + 1);
  }

  void apply_initial_conditions() {
    Model &model = get_model();
    Time &time = get_time();
    const SimulationContext sim_ctx{m_mpi_comm};
    pfc::apply_field_modifier_list(sim_ctx, model, time.get_current(),
                                   m_initial_conditions);
  }

  void apply_boundary_conditions() {
    Model &model = get_model();
    Time &time = get_time();
    const SimulationContext sim_ctx{m_mpi_comm};
    pfc::apply_field_modifier_list(sim_ctx, model, time.get_current(),
                                   m_boundary_conditions);
  }

  /**
   * @brief Prologue of one integrator step (same ordering as step())
   *
   * When `Time::get_increment() == 0`, applies initial conditions, boundary
   * conditions, and optionally writes results if `Time::do_save()`. Then always
   * calls `Time::next()` and applies boundary conditions at the new time.
   *
   * Call this once per iteration before the physics update, then invoke
   * `Model::step` (or a model-specific `step(simulator, model)` overload), then
   * call `end_integrator_step()`.
   *
   * @see end_integrator_step()
   * @see step_with_physics()
   * @see step()
   */
  void begin_integrator_step() {
    Time &time = get_time();
    if (time.get_increment() == 0) {
      apply_initial_conditions();
      apply_boundary_conditions();
      if (time.do_save()) {
        write_results();
      }
    }
    time.next();
    apply_boundary_conditions();
  }

  /**
   * @brief Epilogue of one integrator step: write results if at a save point
   *
   * @see begin_integrator_step()
   */
  void end_integrator_step() {
    if (get_time().do_save()) {
      write_results();
    }
  }

  /**
   * @brief One full step with a custom physics body (same ordering as step())
   *
   * Equivalent to `begin_integrator_step(); physics_fn(); end_integrator_step();`
   * with `physics_fn` typically calling `pfc::step(model, time.get_current())` or
   * a model-specific overload such as `step(simulator, concrete_model)`.
   *
   * @tparam PhysicsFn nullary callable
   */
  template <class PhysicsFn> void step_with_physics(PhysicsFn &&physics_fn) {
    begin_integrator_step();
    std::forward<PhysicsFn>(physics_fn)();
    end_integrator_step();
  }

  /**
   * @brief Execute one time step of the simulation
   *
   * Advances the simulation by one time step, orchestrating all components:
   * 1. On first call (`Time::get_increment() == 0`): apply initial conditions,
   *    apply boundary conditions, optionally write results if `Time::do_save()`
   * 2. Call `Time::next()` (increment increases by one; current time becomes
   *    `t0 + increment * dt`, clamped to `t1`)
   * 3. Apply boundary conditions at the new time
   * 4. Call `Model::step()` with `Time::get_current()` (first physics step uses
   *    `t0 + dt`, not `t0`, because `next()` runs before `step()`)
   * 5. Optionally write results again if `Time::do_save()` at the new time
   *
   * This is the main simulation loop body. Call repeatedly until done() returns
   * true.
   *
   * @note Initial conditions run only when the increment is still zero at entry
   * @note Boundary conditions run after ICs (first call) and again after each
   *       `next()`, before every `Model::step()`
   * @note Results written when `Time::do_save()` is true (after IC path and/or
   *       after the physics step)
   * @note `Time::next()` runs on every call, including the first
   *
   * @example
   * ```cpp
   * // Typical simulation loop
   * MyModel model(fft, world);
   * model.initialize(0.01);
   * Time time({0.0, 100.0, 0.01}, 1.0);  // t0=0, t1=100, dt=0.01, saveat=1.0
   * Simulator sim(model, time);
   *
   * // Add initial conditions and results writers
   * sim.add_initial_conditions(std::make_unique<Seed>("density", center, radius));
   * sim.add_results_writer("density", std::make_unique<BinaryWriter>("data"));
   *
   * // Main simulation loop
   * while (!sim.done()) {
   *     sim.step();  // ICs on first iteration, BCs every iteration
   *
   *     // Optional: custom processing between steps
   *     if (sim.get_increment() % 100 == 0) {
   *         std::cout << "Progress: t = " << sim.get_time().get_current() << "\n";
   *     }
   * }
   * ```
   *
   * Workflow (actual order):
   * - First call (`increment == 0`): optional IC/BC/write, then **always**
   *   `time.next()`, BC, `model.step(time.get_current())`, optional write.
   * - Later calls: `time.next()`, BC, `model.step(...)`, optional write.
   *
   * @see done() to check if simulation is complete
   * @see get_time() to access current time state
   * @see get_increment() to get current iteration number
   */
  void step() {
    begin_integrator_step();
    pfc::step(get_model(), get_time().get_current());
    end_integrator_step();
  }

  /**
   * @brief Check if simulation has reached its end time
   *
   * Returns true when the simulation has completed all time steps up to
   * the final time specified in the Time object.
   *
   * @return true if t >= t_final, false otherwise
   *
   * @note This checks Time::done() internally
   * @note Use as loop condition: `while (!sim.done()) { sim.step(); }`
   *
   * @example
   * ```cpp
   * Time time({0.0, 10.0, 0.1}, 1.0);  // Simulate from 0 to 10
   * Simulator sim(model, time);
   *
   * // Run until completion
   * while (!sim.done()) {
   *     sim.step();
   * }
   *
   * std::cout << "Simulation completed at t = "
   *           << sim.get_time().get_current() << "\n";
   * ```
   *
   * @see step() to advance simulation
   * @see Time::done() for time completion logic
   * @see get_time() to access current time
   */
  bool done() {
    Time &time = get_time();
    return time.done();
  }
};

[[nodiscard]] inline Model &get_model(Simulator &sim) noexcept {
  return sim.get_model();
}

[[nodiscard]] inline Time &get_time(Simulator &sim) noexcept {
  return sim.get_time();
}

[[nodiscard]] inline const World &get_world(Simulator &sim) noexcept {
  return pfc::get_world(get_model(sim));
}

[[nodiscard]] inline fft::IFFT &get_fft(Simulator &sim) noexcept {
  return pfc::get_fft(get_model(sim));
}

[[nodiscard,
  deprecated("Use get_real_field(get_model(sim), \"default\") or named real fields "
             "instead")]]
inline Field &get_field(Simulator &sim) {
  return pfc::get_real_field(get_model(sim), "default");
}

[[nodiscard]] inline bool is_rank0(const Simulator &sim) noexcept {
  return sim.is_rank0();
}

inline void step(Simulator &s, Model &m) { pfc::step(m, get_time(s).get_current()); }

} // namespace pfc

#endif // PFC_SIMULATOR_HPP
