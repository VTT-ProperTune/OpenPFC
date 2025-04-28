// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_MODEL_HPP
#define PFC_MODEL_HPP

#include "core/world.hpp"
#include "decomposition.hpp"
#include "fft.hpp"
#include "openpfc/backends/heffte_adapter.hpp"
#include "types.hpp"
#include <iostream>
#include <memory>

namespace pfc {

/**
 * @brief The Model class represents the physics model for simulations in
 * OpenPFC.
 *
 * The Model class is responsible for introducing the physics to the simulation
 * model. Users can override the `initialize` and `step` functions to define
 * their own functionality and implement specific physics for their simulations.
 *
 * The `initialize` function is called at the beginning of the simulation and is
 * used to allocate arrays, pre-calculate operators used in time integration,
 * and perform other necessary initialization tasks.
 *
 * The `step` function is called sequentially during the time integration
 * process. OpenPFC currently uses a semi-implicit time integration scheme,
 * where the linear part is solved implicitly and the non-linear part is solved
 * explicitly.
 */
class Model {
private:
  FFT *m_fft = nullptr;             ///< Raw pointer to the FFT object used by the model
  RealFieldSet m_real_fields;       ///< Collection of real-valued fields associated
                                    ///< with the model
  ComplexFieldSet m_complex_fields; ///< Collection of complex-valued fields
                                    ///< associated with the model
  const World &m_world;             ///< Reference to the World object
  heffte::box3d<int> domain;        ///< Domain dimensions

public:
  bool rank0 = false; ///< Flag indicating if the current MPI rank is 0 (useful
                      ///< for rank-specific operations)

  /**
   * @brief Construct a new Model object.
   *
   * @param world Reference to the World object
   */
  Model(const World &world) : m_world(world), domain(to_heffte_box(world)) {}

  /**
   * @brief Destroy the Model object.
   */
  ~Model() {}

  /**
   * @brief Construct a new Model object.
   *
   * @param fft Reference to the FFT object used by the model
   * @param world Reference to the World object
   */
  Model(FFT &fft, const World &world)
      : m_fft(&fft), m_world(world), domain(to_heffte_box(world)), // Use to_heffte_box
        rank0(get_decomposition().get_rank() == 0) {}

  /**
   * @brief Disable copy constructor.
   */
  Model(const Model &) = delete;

  /**
   * @brief Disable copy assignment operator.
   */
  Model &operator=(const Model &) = delete;

  /**
   * @brief Return boolean flag indicating is this process rank 0.
   *
   * @return true/false
   */
  bool is_rank0() { return rank0; }

  /**
   * @brief Get the decomposition object associated with the model.
   *
   * @return Reference to the Decomposition object
   */
  const Decomposition &get_decomposition() { return get_fft().get_decomposition(); }

  /**
   * @brief Get the world object associated with the model.
   *
   * @return Reference to the World object
   */
  const World &get_world() { return m_world; }

  /**
   * @brief Set the FFT object for the model.
   *
   * @param fft Reference to the FFT object.
   */
  void set_fft(FFT &fft) {
    m_fft = &fft;
    rank0 = (get_decomposition().get_rank() == 0);
  }

  /**
   * @brief Get the FFT object associated with the model.
   *
   * @return Reference to the FFT object.
   * @throws std::runtime_error if the FFT object has not been set.
   */
  FFT &get_fft() {
    if (m_fft == nullptr) {
      std::cerr << "get_fft called on Model instance: " << this << ". FFT object is not set!" << std::endl;
      throw std::runtime_error("FFT object has not been set.");
    }
    return *m_fft;
  }

  /**
   * @brief Pure virtual function to be overridden by concrete implementations.
   *
   * The `step` function is responsible for performing the time integration step
   * for the model. Users should override this function to implement their own
   * time integration scheme and update the model state accordingly.
   *
   * @param t Current simulation time
   */
  virtual void step(double t) = 0;

  /**
   * @brief Pure virtual function to be overridden by concrete implementations.
   *
   * The `initialize` function is called at the beginning of the simulation and
   * is used to perform any necessary initialization tasks, such as allocating
   * arrays and pre-calculating operators used in time integration.
   *
   * @param dt Time step size for the simulation
   */
  virtual void initialize(double dt) = 0;

  /**
   * @brief Check if the model has a real-valued field with the given name.
   *
   * @param field_name Name of the field to check
   * @return True if the field exists, False otherwise
   */
  bool has_real_field(const std::string &field_name) { return m_real_fields.count(field_name) > 0; }

  /**
   * @brief Add a real-valued field to the model.
   *
   * @param name Name of the field
   * @param field Reference to the RealField object representing the field
   */
  void add_real_field(const std::string &name, RealField &field) { m_real_fields.insert({name, field}); }

  /**
   * @brief Check if the model has a complex-valued field with the given name.
   *
   * @param field_name Name of the field to check
   * @return True if the field exists, False otherwise
   */
  bool has_complex_field(const std::string &field_name) { return m_complex_fields.count(field_name) > 0; }

  /**
   * @brief Add a complex-valued field to the model.
   *
   * @param name Name of the field
   * @param field Reference to the ComplexField object representing the field
   */
  void add_complex_field(const std::string &name, ComplexField &field) { m_complex_fields.insert({name, field}); }

  /**
   * @brief Get a reference to the real-valued field with the given name.
   *
   * @param name Name of the field
   * @return Reference to the RealField object
   */
  RealField &get_real_field(const std::string &name) { return m_real_fields.find(name)->second; }

  /**
   * @brief Get a reference to the complex-valued field with the given name.
   *
   * @param name Name of the field
   * @return Reference to the ComplexField object
   */
  ComplexField &get_complex_field(const std::string &name) { return m_complex_fields.find(name)->second; }

  /**
   * @brief Add a field to the model.
   *
   * @param name Name of the field
   * @param field Reference to the RealField object representing the field
   */
  void add_field(const std::string &name, RealField &field) { add_real_field(name, field); }

  /**
   * @brief Add a field to the model.
   *
   * @param name Name of the field
   * @param field Reference to the ComplexField object representing the field
   */
  void add_field(const std::string &name, ComplexField &field) { add_complex_field(name, field); }

  /**
   * @brief Check if the model has a field with the given name (real or
   * complex).
   *
   * @param field_name Name of the field to check
   * @return True if the field exists, False otherwise
   */
  bool has_field(const std::string &field_name) { return has_real_field(field_name) || has_complex_field(field_name); }

  /**
   * @brief Get a reference to the default primary unknown field.
   *
   * This is deprecated function and will likely be removed in future.
   *
   * @return Reference to the RealField called "default"
   */
  virtual Field &get_field() {
    if (!has_real_field("default")) {
      throw std::runtime_error("'default' field is not defined.");
    }
    return get_real_field("default");
  };
};

} // namespace pfc

#endif
