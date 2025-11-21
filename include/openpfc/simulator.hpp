// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_SIMULATOR_HPP
#define PFC_SIMULATOR_HPP

#include "core/world.hpp"
#include "field_modifier.hpp"
#include "model.hpp"
#include "results_writer.hpp"
#include "time.hpp"
#include <iostream>
#include <memory>
#include <unordered_map>

namespace pfc {

void step(class Simulator &s, Model &m);

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
  const World &get_world() { return get_model().get_world(); }

  /**
   * @brief Get the FFT object
   *
   * @return FFT&
   */
  FFT &get_fft() { return get_model().get_fft(); }

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
  Field &get_field() { return get_model().get_field(); }

  void initialize() { get_model().initialize(get_time().get_dt()); }

  bool is_rank0() { return get_model().rank0; }

  unsigned int get_increment() { return get_time().get_increment(); }

  bool add_results_writer(const std::string &field_name,
                          std::unique_ptr<ResultsWriter> writer) {
    auto inbox = get_inbox(get_fft());
    auto world = get_world();
    writer->set_domain(get_size(world), inbox.size, inbox.low);

    Model &model = get_model();
    if (model.has_field(field_name)) {
      m_result_writers.insert({field_name, std::move(writer)});
      return true;
    } else {
      std::cout << "Warning: tried to add writer for inexistent field " << field_name
                << ", RESULTS ARE NOT WRITTEN!" << std::endl;
      return false;
    }
  }

  bool add_results_writer(std::unique_ptr<ResultsWriter> writer) {
    std::cout << "Warning: adding result writer to write field 'default'"
              << std::endl;
    return add_results_writer("default", std::move(writer));
  }

  /**
   * @brief Adds initial conditions to the simulation.
   *
   * This function takes a unique pointer to a FieldModifier object, which is
   * used to modify the initial conditions of a field in the model. The field to
   * be modified is determined by the `get_field_name` method of the
   * FieldModifier object. If the field name is "default", a warning message is
   * printed. If the model has the field, the FieldModifier is added to the
   * initial conditions and the function returns true. If the model does not
   * have the field, a warning message is printed and the function returns
   * false.
   *
   * @param modifier A unique pointer to a FieldModifier object.
   * @return True if the FieldModifier was successfully added to the initial
   * conditions, false otherwise.
   */
  bool add_initial_conditions(std::unique_ptr<FieldModifier> modifier) {
    Model &model = get_model();
    std::string field_name = modifier->get_field_name();
    if (field_name == "default") {
      std::cout << "Warning: adding initial condition to modify field 'default'"
                << std::endl;
    }
    if (model.has_field(field_name)) {
      m_initial_conditions.push_back(std::move(modifier));
      return true;
    } else {
      std::cout << "Warning: tried to add initial condition for inexistent field "
                << field_name << ", INITIAL CONDITIONS ARE NOT APPLIED!"
                << std::endl;
      return false;
    }
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
   * @brief Adds boundary conditions to the simulation.
   *
   * This function takes a unique pointer to a FieldModifier object, which is
   * used to modify the boundary conditions of a field in the model. The field
   * to be modified is determined by the `get_field_name` method of the
   * FieldModifier object. If the field name is "default", a warning message is
   * printed. If the model has the field, the FieldModifier is added to the
   * boundary conditions and the function returns true. If the model does not
   * have the field, a warning message is printed and the function returns
   * false.
   *
   * @param modifier A unique pointer to a FieldModifier object.
   * @return True if the FieldModifier was successfully added to the boundary
   * conditions, false otherwise.
   */
  bool add_boundary_conditions(std::unique_ptr<FieldModifier> modifier) {
    Model &model = get_model();
    std::string field_name = modifier->get_field_name();
    if (field_name == "default") {
      std::cout << "Warning: adding boundary condition to modify field 'default'"
                << std::endl;
    }
    if (model.has_field(field_name)) {
      m_boundary_conditions.push_back(std::move(modifier));
      return true;
    } else {
      std::cout << "Warning: tried to add boundary condition for inexistent field "
                << field_name << ", BOUNDARY CONDITIONS ARE NOT APPLIED!"
                << std::endl;
      return false;
    }
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

  double get_result_counter() const { return m_result_counter; }

  void write_results() {
    int file_num = get_result_counter();
    Model &model = get_model();
    for (const auto &[field_name, writer] : m_result_writers) {
      if (model.has_real_field(field_name)) {
        writer->write(file_num, get_model().get_real_field(field_name));
      }
      if (model.has_complex_field(field_name)) {
        writer->write(file_num, get_model().get_complex_field(field_name));
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
    model.step(time.get_current());
    if (time.do_save()) {
      write_results();
    }
    return;
  }

  bool done() {
    Time &time = get_time();
    return time.done();
  }
};

void step(Simulator &s, Model &m) { m.step(s.get_time().get_current()); }

} // namespace pfc

#endif // PFC_SIMULATOR_HPP
