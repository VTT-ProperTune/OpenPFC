/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#ifndef PFC_SIMULATOR_HPP
#define PFC_SIMULATOR_HPP

#include "field_modifier.hpp"
#include "model.hpp"
#include "results_writer.hpp"
#include "time.hpp"
#include "world.hpp"
#include <iostream>
#include <memory>
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
  const Decomposition &get_decomposition() { return get_model().get_decomposition(); }

  /**
   * @brief Get the world object
   *
   * @return const World&
   */
  const World &get_world() { return get_decomposition().get_world(); }

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

  bool add_results_writer(const std::string &field_name, std::unique_ptr<ResultsWriter> writer) {
    const Decomposition &d = get_decomposition();
    writer->set_domain(d.get_world().get_size(), d.inbox.size, d.inbox.low);
    Model &model = get_model();
    if (model.has_field(field_name)) {
      m_result_writers.insert({field_name, std::move(writer)});
      return true;
    } else {
      std::cout << "Warning: tried to add writer for inexistent field " << field_name << ", RESULTS ARE NOT WRITTEN!"
                << std::endl;
      return false;
    }
  }

  bool add_results_writer(std::unique_ptr<ResultsWriter> writer) {
    std::cout << "Warning: adding result writer to write field 'default'" << std::endl;
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
      std::cout << "Warning: adding initial condition to modify field 'default'" << std::endl;
    }
    if (model.has_field(field_name)) {
      m_initial_conditions.push_back(std::move(modifier));
      return true;
    } else {
      std::cout << "Warning: tried to add initial condition for inexistent field " << field_name
                << ", INITIAL CONDITIONS ARE NOT APPLIED!" << std::endl;
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
  const std::vector<std::unique_ptr<FieldModifier>> &get_initial_conditions() const { return m_initial_conditions; }

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
      std::cout << "Warning: adding boundary condition to modify field 'default'" << std::endl;
    }
    if (model.has_field(field_name)) {
      m_boundary_conditions.push_back(std::move(modifier));
      return true;
    } else {
      std::cout << "Warning: tried to add boundary condition for inexistent field " << field_name
                << ", BOUNDARY CONDITIONS ARE NOT APPLIED!" << std::endl;
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
  const std::vector<std::unique_ptr<FieldModifier>> &get_boundary_conditions() const { return m_boundary_conditions; }

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

} // namespace pfc

#endif // PFC_SIMULATOR_HPP
