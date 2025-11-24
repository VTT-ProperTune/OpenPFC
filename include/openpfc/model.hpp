// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file model.hpp
 * @brief Physics model abstraction for phase-field simulations
 *
 * @details
 * This file defines the Model class, which represents the physics model in OpenPFC
 * phase-field simulations. The Model class serves as the base class for implementing
 * specific physics models such as:
 * - Phase Field Crystal (PFC) models
 * - Cahn-Hilliard equation
 * - Allen-Cahn equation
 * - Coupled multi-field models (temperature, concentration, etc.)
 *
 * The Model class manages:
 * - Registration and storage of real and complex fields
 * - Access to FFT operations for spectral methods
 * - Virtual interface for physics-specific initialization and time stepping
 *
 * Users implement custom models by:
 * 1. Deriving from Model
 * 2. Overriding `initialize()` to set up initial conditions and operators
 * 3. Overriding `step(dt)` to define time evolution equations
 * 4. Registering fields via register_real_field() and register_complex_field()
 *
 * Example:
 * @code
 * class MyPhysicsModel : public pfc::Model {
 * public:
 *     MyPhysicsModel(pfc::FFT& fft, const pfc::World& world)
 *         : Model(fft, world) {}
 *
 *     void initialize() override {
 *         // Register fields
 *         register_real_field("density");
 *         register_complex_field("density_fourier");
 *
 *         // Set initial conditions
 *         // Precompute operators
 *     }
 *
 *     void step(double dt) override {
 *         // Implement time evolution equations
 *         // Use FFT for spectral derivatives
 *     }
 * };
 * @endcode
 *
 * This file is part of the Physics Models module, providing the core abstraction
 * for describing material behavior and evolution equations.
 *
 * @see simulator.hpp for how models are executed in time integration
 * @see fft.hpp for spectral operations interface
 * @see core/world.hpp for computational domain
 * @see field_modifier.hpp for initial/boundary conditions
 */

#ifndef PFC_MODEL_HPP
#define PFC_MODEL_HPP

#include "core/decomposition.hpp"
#include "core/world.hpp"
#include "fft.hpp"
#include "mpi.hpp"
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
  FFT *m_fft = nullptr;       ///< Raw pointer to the FFT object used by the model
  RealFieldSet m_real_fields; ///< Collection of real-valued fields associated
                              ///< with the model
  ComplexFieldSet m_complex_fields; ///< Collection of complex-valued fields
                                    ///< associated with the model
  const World &m_world;             ///< Reference to the World object
  heffte::box3d<int> domain;        ///< Domain dimensions
  bool m_rank0 = false;             ///< Flag indicating if the current MPI rank is 0

public:
  /**
   * @brief Construct a new Model object.
   *
   * @param world Reference to the World object
   */
  Model(const World &world)
      : m_world(world), domain(to_heffte_box(world)), m_rank0(mpi::get_rank() == 0) {
  }

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
      : m_fft(&fft), m_world(world),
        domain(to_heffte_box(world)), // Use to_heffte_box
        m_rank0(mpi::get_rank() == 0) {}

  /**
   * @brief Disable copy constructor.
   */
  Model(const Model &) = delete;

  /**
   * @brief Disable copy assignment operator.
   */
  Model &operator=(const Model &) = delete;

  /**
   * @brief Check if current MPI rank is 0
   *
   * Useful for conditional output (only rank 0 prints).
   *
   * @return true if rank is 0, false otherwise
   *
   * @note This function is thread-safe and has zero overhead (inline)
   *
   * @code
   * if (model.is_rank0()) {
   *     std::cout << "Status message\n";
   * }
   * @endcode
   */
  inline bool is_rank0() const { return m_rank0; }

  /**
   * @brief Get the decomposition object associated with the model.
   *
   * @return Reference to the Decomposition object
   */
  /*
  const Decomposition &get_decomposition() { return get_fft().get_decomposition(); }
  */

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
    m_rank0 = (mpi::get_rank() == 0);
  }

  /**
   * @brief Get the FFT object associated with the model.
   *
   * @return Reference to the FFT object.
   * @throws std::runtime_error if the FFT object has not been set.
   */
  FFT &get_fft() {
    if (m_fft == nullptr) {
      std::cerr << "get_fft called on Model instance: " << this
                << ". FFT object is not set!" << std::endl;
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
  bool has_real_field(const std::string &field_name) {
    return m_real_fields.count(field_name) > 0;
  }

  /**
   * @brief Add a real-valued field to the model.
   *
   * @param name Name of the field
   * @param field Reference to the RealField object representing the field
   */
  void add_real_field(const std::string &name, RealField &field) {
    m_real_fields.insert({name, field});
  }

  /**
   * @brief Check if the model has a complex-valued field with the given name.
   *
   * @param field_name Name of the field to check
   * @return True if the field exists, False otherwise
   */
  bool has_complex_field(const std::string &field_name) {
    return m_complex_fields.count(field_name) > 0;
  }

  /**
   * @brief Add a complex-valued field to the model.
   *
   * @param name Name of the field
   * @param field Reference to the ComplexField object representing the field
   */
  void add_complex_field(const std::string &name, ComplexField &field) {
    m_complex_fields.insert({name, field});
  }

  /**
   * @brief Get a reference to the real-valued field with the given name.
   *
   * @param name Name of the field
   * @return Reference to the RealField object
   */
  RealField &get_real_field(const std::string &name) {
    return m_real_fields.find(name)->second;
  }

  /**
   * @brief Get a reference to the complex-valued field with the given name.
   *
   * @param name Name of the field
   * @return Reference to the ComplexField object
   */
  ComplexField &get_complex_field(const std::string &name) {
    return m_complex_fields.find(name)->second;
  }

  /**
   * @brief Add a field to the model.
   *
   * @param name Name of the field
   * @param field Reference to the RealField object representing the field
   */
  void add_field(const std::string &name, RealField &field) {
    add_real_field(name, field);
  }

  /**
   * @brief Add a field to the model.
   *
   * @param name Name of the field
   * @param field Reference to the ComplexField object representing the field
   */
  void add_field(const std::string &name, ComplexField &field) {
    add_complex_field(name, field);
  }

  /**
   * @brief Check if the model has a field with the given name (real or
   * complex).
   *
   * @param field_name Name of the field to check
   * @return True if the field exists, False otherwise
   */
  bool has_field(const std::string &field_name) {
    return has_real_field(field_name) || has_complex_field(field_name);
  }

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
