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
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <string_view>
#include <vector>

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
  /// @brief Raw pointer to the FFT object used by the model
  ///
  /// @note **Ownership**: Model does NOT own the FFT object. The FFT must outlive
  ///       the Model instance. The FFT is passed by reference in constructors
  ///       and set_fft(), and Model stores a pointer for access.
  ///
  /// @warning **Lifetime**: Ensure the FFT object exists for the entire lifetime
  ///          of the Model. Do not destroy the FFT while Model is still in use.
  ///
  /// @see get_fft() for access
  /// @see set_fft() to associate FFT with model
  FFT *m_fft = nullptr;
  RealFieldSet m_real_fields;       ///< Collection of real-valued fields associated
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
  inline bool is_rank0() const noexcept { return m_rank0; }

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
  const World &get_world() const noexcept { return m_world; }

  /**
   * @brief Set the FFT object for the model
   *
   * Associates an FFT instance with this model, enabling spectral operations.
   * This is typically called during model setup before initialization.
   *
   * @param fft Reference to the FFT object to use for transforms
   *
   * @note The FFT object must outlive the Model (Model stores a pointer)
   * @note Calling set_fft() after initialization may invalidate precomputed
   * operators
   * @note This also updates the m_rank0 flag for MPI rank checking
   *
   * @warning The Model stores a raw pointer - ensure FFT lifetime exceeds Model
   * lifetime
   * @warning Changing FFT after initialize() may cause undefined behavior
   *
   * @example
   * ```cpp
   * // Typical usage pattern
   * auto world = world::create({256, 256, 256});
   * auto decomp = Decomposition(world, MPI_COMM_WORLD);
   * auto fft = FFT(decomp);
   *
   * MyModel model(world);  // Construct without FFT
   * model.set_fft(fft);    // Set FFT before initialization
   * model.initialize(0.01);
   * ```
   *
   * @see get_fft() to access the FFT object
   * @see Model(FFT&, const World&) constructor that sets FFT at construction
   */
  void set_fft(FFT &fft) {
    m_fft = &fft;
    m_rank0 = (mpi::get_rank() == 0);
  }

  /**
   * @brief Get the FFT object associated with the model
   *
   * Returns a reference to the FFT instance used for spectral operations
   * (forward and backward Fourier transforms). Use this to perform transforms
   * during time stepping.
   *
   * @pre FFT must be set via constructor or set_fft() before calling this
   * @post Returns valid reference to FFT object
   * @return Reference to the FFT object
   *
   * @throws std::runtime_error if the FFT object has not been set
   *
   * @note The returned reference is valid as long as the FFT object exists
   *
   * @warning Calling this before set_fft() or appropriate constructor throws
   *
   * @example
   * ```cpp
   * void step(double t) override {
   *     auto& density = get_real_field("density");
   *     auto& density_k = get_complex_field("density_k");
   *     auto& fft = get_fft();
   *
   *     // Real → Complex transform
   *     fft.forward(density.data(), density_k.data());
   *
   *     // Apply operators in k-space
   *     for (size_t k = 0; k < density_k.size(); ++k) {
   *         density_k[k] = operator_k[k] * density_k[k];
   *     }
   *
   *     // Complex → Real transform
   *     fft.backward(density_k.data(), density.data());
   * }
   * ```
   *
   * @see set_fft() to associate FFT with model
   * @see FFT::forward() for real-to-complex transforms
   * @see FFT::backward() for complex-to-real transforms
   */
  [[nodiscard]] FFT &get_fft() {
    if (m_fft == nullptr) {
      std::string msg =
          "FFT object has not been set for Model instance at " +
          std::to_string(reinterpret_cast<uintptr_t>(this)) +
          ". "
          "Call set_fft() or use Model(FFT&, const World&) constructor.";
      throw std::runtime_error(msg);
    }
    return *m_fft;
  }

  /**
   * @brief Advance the model by one time step
   *
   * This pure virtual function must be implemented by derived classes to
   * update the model's fields for one time step. This is where the physics
   * of your model is implemented.
   *
   * OpenPFC uses semi-implicit spectral methods:
   * - Linear terms: solved implicitly in Fourier space
   * - Nonlinear terms: evaluated explicitly in real space
   *
   * @param t Current simulation time
   *
   * @note This function is called repeatedly during time integration
   * @note For explicit schemes, dt is typically member variable
   * @note Access fields via get_real_field() and get_complex_field()
   *
   * @example
   * ```cpp
   * // Example: Phase Field Crystal
   * // ∂n/∂t = ∇²[(1 + ∇²)² n + n³]
   * void step(double t) override {
   *     auto& n = get_real_field("density");
   *     auto& n_k = get_complex_field("density_k");
   *     auto& fft = get_fft();
   *
   *     // Compute n³ in real space
   *     for (size_t i = 0; i < n.size(); ++i) {
   *         nonlinear[i] = n[i] * n[i] * n[i];
   *     }
   *
   *     // FFT to k-space and solve
   *     fft.forward(nonlinear.data(), nonlinear_k.data());
   *     for (size_t k = 0; k < n_k.size(); ++k) {
   *         n_k[k] = propagator[k] * (laplacian_op[k] * n_k[k] + nonlinear_k[k]);
   *     }
   *     fft.backward(n_k.data(), n.data());
   * }
   * ```
   *
   * Time complexity: O(N log N) for FFT operations
   *
   * @see initialize() for precomputing operators
   * @see get_fft() for FFT operations
   */
  virtual void step(double t) = 0;

  /**
   * @brief Initialize the model before time integration begins
   *
   * This pure virtual function must be implemented by derived classes to
   * set up the model's state before the simulation starts. Typical tasks include:
   * - Registering and allocating field storage
   * - Precomputing spectral operators (k-space derivatives, propagators)
   * - Setting up time integration coefficients
   * - Initializing auxiliary fields
   *
   * This function is called once before the first time step.
   *
   * @param dt Time step size for the simulation
   *
   * @note dt can be used to precompute time-step-dependent operators
   * @note For adaptive time stepping, this receives the initial dt
   *
   * @example
   * ```cpp
   * class MyPFCModel : public pfc::Model {
   * public:
   *     MyPFCModel(pfc::FFT& fft, const pfc::World& world)
   *         : Model(fft, world) {}
   *
   *     void initialize(double dt) override {
   *         // Register fields
   *         add_real_field("density", density_field);
   *         add_complex_field("density_k", density_k_field);
   *
   *         // Precompute spectral operators
   *         auto world = get_world();
   *         for (int k = 0; k < kspace_size; ++k) {
   *             double k2 = kx[k]*kx[k] + ky[k]*ky[k] + kz[k]*kz[k];
   *             laplacian_op[k] = -k2;
   *
   *             // Semi-implicit operator: (1 + dt*L)^-1
   *             propagator[k] = 1.0 / (1.0 + dt * (1 + k2)*(1 + k2));
   *         }
   *
   *         if (is_rank0()) {
   *             std::cout << "Model initialized with dt = " << dt << "\n";
   *         }
   *     }
   *
   *     // ... step() implementation ...
   * };
   * ```
   *
   * @see step() for time integration implementation
   * @see add_real_field() to register fields
   * @see get_fft() to access FFT operations
   */
  virtual void initialize(double dt) = 0;

  /**
   * @brief Check if the model has a real-valued field with the given name.
   *
   * @param field_name Name of the field to check
   * @return True if the field exists, False otherwise
   */
  [[nodiscard]] bool has_real_field(std::string_view field_name) const noexcept {
    return m_real_fields.count(std::string(field_name)) > 0;
  }

  /**
   * @brief Register a real-valued field with the model
   *
   * Adds a named real-valued field to the model's field registry.
   *
   * @param name Unique identifier for the field
   * @param field Reference to the RealField object
   *
   * @note Field names must be unique
   * @note Common names: "density", "temperature", "concentration"
   *
   * @example
   * ```cpp
   * void initialize(double dt) override {
   *     add_real_field("density", m_density);
   *     add_real_field("temperature", m_temperature);
   * }
   * ```
   *
   * @see get_real_field() to retrieve registered fields
   */
  void add_real_field(std::string_view name, RealField &field) {
    m_real_fields.emplace(std::string(name), field);
  }

  /**
   * @brief Check if the model has a complex-valued field with the given name
   *
   * Checks whether a complex field (typically in Fourier space) has been
   * registered with the model. Use this before accessing fields to avoid
   * runtime errors.
   *
   * @param field_name Name of the field to check
   * @return True if the field exists, False otherwise
   *
   * @note Complex fields are typically used for Fourier-space representations
   * @note Field names are case-sensitive
   *
   * @example
   * ```cpp
   * if (has_complex_field("density_k")) {
   *     auto& n_k = get_complex_field("density_k");
   *     // Perform k-space operations...
   * } else {
   *     std::cerr << "Field not registered!\n";
   * }
   * ```
   *
   * @see add_complex_field() to register fields
   * @see get_complex_field() to access registered fields
   */
  [[nodiscard]] bool has_complex_field(std::string_view field_name) const noexcept {
    return m_complex_fields.count(std::string(field_name)) > 0;
  }

  /**
   * @brief Register a complex-valued field with the model
   *
   * Adds a named complex-valued field (typically for Fourier-space data) to
   * the model's field registry. Complex fields store FFT-transformed data and
   * are used for spectral operations.
   *
   * @param name Unique identifier for the field
   * @param field Reference to the ComplexField object
   *
   * @note Field names must be unique across all field types
   * @note Complex fields have roughly half the size of real fields (N/2+1 in FFT
   * convention)
   * @note Common naming: append "_k" or "_fourier" to indicate k-space
   *
   * @example
   * ```cpp
   * void initialize(double dt) override {
   *     // Real space field
   *     add_real_field("density", m_density);
   *
   *     // Corresponding k-space field
   *     add_complex_field("density_k", m_density_k);
   *
   *     // Temperature field and its transform
   *     add_real_field("temperature", m_temp);
   *     add_complex_field("temperature_k", m_temp_k);
   * }
   * ```
   *
   * @see get_complex_field() to retrieve registered fields
   * @see has_complex_field() to check existence
   * @see add_real_field() for real-valued fields
   */
  void add_complex_field(std::string_view name, ComplexField &field) {
    m_complex_fields.emplace(std::string(name), field);
  }

  /**
   * @brief Retrieve a registered real-valued field by name
   *
   * Returns a reference to a previously registered field for reading or
   * modification.
   *
   * @param name Name of the field to retrieve
   * @return Reference to the RealField object
   *
   * @throws std::out_of_range if field name not registered
   *
   * @warning Accessing non-existent fields causes runtime error
   *
   * @example
   * ```cpp
   * void step(double t) override {
   *     auto& density = get_real_field("density");
   *     // Modify field...
   * }
   * ```
   *
   * @see add_real_field() to register fields
   * @see has_real_field() to check existence
   */
  [[nodiscard]] RealField &get_real_field(std::string_view name) {
    auto it = m_real_fields.find(std::string(name));
    if (it == m_real_fields.end()) {
      throw std::out_of_range("Real field '" + std::string(name) +
                              "' not found. "
                              "Available fields: " +
                              list_field_names());
    }
    return it->second;
  }

  /**
   * @brief Retrieve a registered real-valued field by name (const version)
   *
   * Returns a const reference to a previously registered field for reading.
   *
   * @param name Name of the field to retrieve
   * @return Const reference to the RealField object
   *
   * @throws std::out_of_range if field name not registered
   *
   * @see get_real_field() for non-const version
   */
  [[nodiscard]] const RealField &get_real_field(std::string_view name) const {
    auto it = m_real_fields.find(std::string(name));
    if (it == m_real_fields.end()) {
      throw std::out_of_range("Real field '" + std::string(name) +
                              "' not found. "
                              "Available fields: " +
                              list_field_names());
    }
    return it->second;
  }

  /**
   * @brief Retrieve a registered complex-valued field by name
   *
   * Returns a reference to a previously registered complex field (typically
   * Fourier-space data) for reading or modification during time stepping.
   *
   * @param name Name of the field to retrieve
   * @return Reference to the ComplexField object
   *
   * @throws std::out_of_range if field name not registered
   *
   * @warning Accessing non-existent fields causes runtime error
   * @note Complex fields contain FFT-transformed data (k-space representation)
   * @note Modifications to complex fields are typically followed by inverse FFT
   *
   * @example
   * ```cpp
   * void step(double t) override {
   *     auto& n = get_real_field("density");
   *     auto& n_k = get_complex_field("density_k");
   *     auto& fft = get_fft();
   *
   *     // Transform to k-space
   *     fft.forward(n.data(), n_k.data());
   *
   *     // Apply spectral operator
   *     for (size_t k = 0; k < n_k.size(); ++k) {
   *         n_k[k] *= propagator[k];
   *     }
   *
   *     // Transform back to real space
   *     fft.backward(n_k.data(), n.data());
   * }
   * ```
   *
   * @see add_complex_field() to register fields
   * @see has_complex_field() to check existence
   * @see get_fft() to access FFT operations
   */
  [[nodiscard]] ComplexField &get_complex_field(std::string_view name) {
    auto it = m_complex_fields.find(std::string(name));
    if (it == m_complex_fields.end()) {
      throw std::out_of_range("Complex field '" + std::string(name) +
                              "' not found. "
                              "Available fields: " +
                              list_field_names());
    }
    return it->second;
  }

  /**
   * @brief Retrieve a registered complex-valued field by name (const version)
   *
   * Returns a const reference to a previously registered complex field for reading.
   *
   * @param name Name of the field to retrieve
   * @return Const reference to the ComplexField object
   *
   * @throws std::out_of_range if field name not registered
   *
   * @see get_complex_field() for non-const version
   */
  [[nodiscard]] const ComplexField &get_complex_field(std::string_view name) const {
    auto it = m_complex_fields.find(std::string(name));
    if (it == m_complex_fields.end()) {
      throw std::out_of_range("Complex field '" + std::string(name) +
                              "' not found. "
                              "Available fields: " +
                              list_field_names());
    }
    return it->second;
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
   * @brief Check if the model has a field with the given name (real or complex)
   *
   * Convenience function that checks both real and complex field registries.
   * Useful when you don't know or care about the field type.
   *
   * @param field_name Name of the field to check
   * @return True if the field exists in either registry, False otherwise
   *
   * @note Checks both real-valued and complex-valued field registries
   * @note Field names are case-sensitive
   *
   * @example
   * ```cpp
   * void some_operation(const std::string& field_name) {
   *     if (!has_field(field_name)) {
   *         throw std::runtime_error("Field not found: " + field_name);
   *     }
   *
   *     // Determine type and access accordingly
   *     if (has_real_field(field_name)) {
   *         auto& field = get_real_field(field_name);
   *         // Process real field...
   *     } else {
   *         auto& field = get_complex_field(field_name);
   *         // Process complex field...
   *     }
   * }
   * ```
   *
   * @see has_real_field() to check only real fields
   * @see has_complex_field() to check only complex fields
   */
  [[nodiscard]] bool has_field(std::string_view field_name) const noexcept {
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

private:
  /**
   * @brief List all registered field names (real and complex)
   *
   * Helper method to generate helpful error messages when fields are not found.
   *
   * @return Comma-separated string of all registered field names, or "(none)" if
   * empty
   */
  std::string list_field_names() const {
    std::vector<std::string> names;
    for (const auto &[name, _] : m_real_fields) {
      names.push_back(name);
    }
    for (const auto &[name, _] : m_complex_fields) {
      names.push_back(name);
    }
    if (names.empty()) {
      return "(none)";
    }
    return std::accumulate(
        std::next(names.begin()), names.end(), names[0],
        [](const std::string &a, const std::string &b) { return a + ", " + b; });
  }
};

} // namespace pfc

#endif
