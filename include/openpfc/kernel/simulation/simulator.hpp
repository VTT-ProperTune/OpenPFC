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
 * pfc::Simulator sim(model, time);
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
 * execution framework for time-dependent simulations.
 *
 * @see model.hpp for physics model implementation
 * @see time.hpp for time state management
 * @see field_modifier.hpp for initial/boundary conditions
 * @see results_writer.hpp for output handling
 */

#ifndef PFC_SIMULATOR_HPP
#define PFC_SIMULATOR_HPP

#include <iostream>
#include <memory>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <unordered_map>

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

public:
  /**
   * @brief Construct a new Simulator object
   *
   * @param model The model to simulate.
   * @param time The time object to use for simulation.
   */
  Simulator(Model &model, Time &time) : m_model(model), m_time(time) {}

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
  /*
  const Decomposition &get_decomposition() {
    return get_model().get_decomposition();
  }
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
   * @return FFT&
   */
  FFT &get_fft() { return pfc::get_fft(m_model); }

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

  bool is_rank0() { return pfc::is_rank0(m_model); }

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
      std::cout << "Warning: tried to add writer for inexistent field " << field_name
                << ", RESULTS ARE NOT WRITTEN!" << '\n';
      return false;
    }
    m_result_writers.insert({field_name, std::move(writer)});
    return true;
  }

  bool add_results_writer(std::unique_ptr<ResultsWriter> writer) {
    std::cout << "Warning: adding result writer to write field 'default'" << '\n';
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
    Model &model = get_model();
    for (const std::string &field_name : modifier->get_field_names()) {
      if (field_name == "default") {
        std::cout << "Warning: adding initial condition to modify field 'default'"
                  << '\n';
      }
      if (!pfc::has_field(model, field_name)) {
        std::cout << "Warning: tried to add initial condition for inexistent field "
                  << field_name << ", INITIAL CONDITIONS ARE NOT APPLIED!" << '\n';
        return false;
      }
    }
    m_initial_conditions.push_back(std::move(modifier));
    return true;
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
    Model &model = get_model();
    for (const std::string &field_name : modifier->get_field_names()) {
      if (field_name == "default") {
        std::cout << "Warning: adding boundary condition to modify field 'default'"
                  << '\n';
      }
      if (!pfc::has_field(model, field_name)) {
        std::cout << "Warning: tried to add boundary condition for inexistent field "
                  << field_name << ", BOUNDARY CONDITIONS ARE NOT APPLIED!" << '\n';
        return false;
      }
    }
    m_boundary_conditions.push_back(std::move(modifier));
    return true;
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
    int file_num = get_result_counter();
    Model &model = get_model();
    for (const auto &[field_name, writer] : m_result_writers) {
      if (pfc::has_real_field(model, field_name)) {
        writer->write(file_num, pfc::get_real_field(model, field_name));
      }
      if (pfc::has_complex_field(model, field_name)) {
        writer->write(file_num, pfc::get_complex_field(model, field_name));
      }
    }
    set_result_counter(file_num + 1);
  }

  void apply_initial_conditions() {
    Model &model = get_model();
    Time &time = get_time();
    for (const auto &ic : m_initial_conditions) {
      ic->apply(model, time.get_current());
    }
  }

  void apply_boundary_conditions() {
    Model &model = get_model();
    Time &time = get_time();
    for (const auto &bc : m_boundary_conditions) {
      bc->apply(model, time.get_current());
    }
  }

  /**
   * @brief Execute one time step of the simulation
   *
   * Advances the simulation by one time step, orchestrating all components:
   * 1. On first call (increment==0): Apply initial conditions
   * 2. Apply boundary conditions
   * 3. Call Model::step() to evolve physics
   * 4. Write results if at save point
   * 5. Advance time counter
   *
   * This is the main simulation loop body. Call repeatedly until done() returns
   * true.
   *
   * @note Initial conditions applied only on first call
   * @note Boundary conditions applied before every Model::step()
   * @note Results written automatically at intervals set by Time::set_saveat()
   * @note Time state advanced automatically
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
   * Workflow per step:
   * - t=0: apply_initial_conditions() → apply_boundary_conditions() →
   * write_results() → next() → ...
   * - t>0: next() → apply_boundary_conditions() → model.step() → write_results() (if
   * save point)
   *
   * @see done() to check if simulation is complete
   * @see get_time() to access current time state
   * @see get_increment() to get current iteration number
   */
  void step() {
    Time &time = get_time();
    Model &model = get_model();
    if (time.get_increment() == 0) {
      apply_initial_conditions();
      apply_boundary_conditions();
      if (time.do_save()) {
        write_results();
      }
    }
    time.next();
    apply_boundary_conditions();
    pfc::step(model, time.get_current());
    if (time.do_save()) {
      write_results();
    }
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

[[nodiscard]] inline FFT &get_fft(Simulator &sim) noexcept {
  return pfc::get_fft(get_model(sim));
}

[[nodiscard,
  deprecated("Use get_real_field(get_model(sim), \"default\") or named real fields "
             "instead")]]
inline Field &get_field(Simulator &sim) {
  return pfc::get_real_field(get_model(sim), "default");
}

[[nodiscard]] inline bool is_rank0(Simulator &sim) noexcept {
  return pfc::is_rank0(get_model(sim));
}

inline void step(Simulator &s, Model &m) { pfc::step(m, get_time(s).get_current()); }

} // namespace pfc

#endif // PFC_SIMULATOR_HPP
