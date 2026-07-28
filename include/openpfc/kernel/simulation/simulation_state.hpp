// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef PFC_KERNEL_SIMULATION_SIMULATION_STATE_HPP
#define PFC_KERNEL_SIMULATION_SIMULATION_STATE_HPP

/**
 * @file simulation_state.hpp
 * @brief M2 owning simulation state: `openpfc::kernel::simulation::SimulationState`.
 *
 * `SimulationState` owns canonical fields by value, keyed two ways:
 *   - by **name** (a `std::string`) for I/O, checkpointing and wiring, and
 *   - by a **typed handle** (`FieldHandle<T>`) for hot-path lookup that skips
 *     the string hash.
 *
 * It can hold fields of different element types (e.g. `double` and
 * `std::complex<double>`) at the same time. Heterogeneous field values are
 * stored through a small type-erased layer: one `FieldHolder<T>` per concrete
 * type `T`, looked up by `std::type_index`. There is deliberately **no** new
 * `Field` type and **no** virtual field base -- the canonical `Field` is used
 * directly.
 *
 * Not wired to the Gen-1 `ModelFieldRegistry`; that stays untouched until M12.
 */

#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include <openpfc/kernel/data/field.hpp>

namespace openpfc {
namespace kernel {
namespace simulation {

/**
 * @brief Opaque, type-safe handle to a field owned by a `SimulationState`.
 *
 * A handle carries a raw pointer to the stored field and the element type `T` in its own type.
 * Retrieving the field uses the handle's pointer directly, avoiding string lookups.
 * A null handle (default-constructed) indicates an invalid/unset field reference.
 */
template <typename T> class FieldHandle {
public:
  FieldHandle() : m_field(nullptr) {}
  explicit FieldHandle(pfc::field::Field<T>* field_ptr) : m_field(field_ptr) {}

  // Access the underlying field
  pfc::field::Field<T>& get() noexcept { return *m_field; }
  const pfc::field::Field<T>& get() const noexcept { return *m_field; }

  // Check if handle is valid (points to a field)
  explicit operator bool() const noexcept { return m_field != nullptr; }

  bool operator==(const FieldHandle& other) const noexcept {
    return m_field == other.m_field;
  }
  bool operator!=(const FieldHandle& other) const noexcept {
    return !(*this == other);
  }

private:
  pfc::field::Field<T>* m_field;
};

/**
 * @brief Owns canonical field values keyed by name and typed handle.
 *
 * Field names are unique across the whole state (a name identifies exactly one
 * field, of exactly one type). Handles provide zero-allocation access to stored
 * fields by raw pointer, avoiding string hash lookups on hot paths.
 */
class SimulationState {
public:
  SimulationState() = default;

  // Non-copyable (owns unique fields by value)
  SimulationState(const SimulationState&) = delete;
  SimulationState& operator=(const SimulationState&) = delete;

  // Movable (transfer ownership of fields)
  SimulationState(SimulationState&&) noexcept = default;
  SimulationState& operator=(SimulationState&&) noexcept = default;

  ~SimulationState() = default;

  /**
   * @brief Take ownership of @p field under @p name.
   * @param name Unique identifier for the field
   * @param field Field to take ownership of; moved into storage
   * @throws std::runtime_error if @p name is already in use.
   */
  template <typename T>
  void insert_field(std::string name, pfc::field::Field<T>&& field);

  /**
   * @brief Handle for the field named @p name, for repeated hot-path access.
   * @param name Field identifier
   * @return FieldHandle for typed access (null handle if not found or wrong type)
   */
  template <typename T>
  FieldHandle<T> get_field(const std::string& name) noexcept;

  /**
   * @brief Check if a field exists by name (type-agnostic).
   * @param name Field identifier
   * @return true if any field with that name exists
   */
  bool has_field(const std::string& name) const noexcept {
    return m_name_to_index.count(name) != 0;
  }

  /**
   * @brief Remove field by name; returns false if not found or type mismatch.
   * @param name Field identifier
   * @return true if removal succeeded
   */
  template <typename T>
  bool remove_field(const std::string& name);

  /**
   * @brief Clear all stored fields.
   */
  void clear() noexcept;

  /**
   * @brief Get number of stored fields.
   */
  size_t size() const noexcept { return m_name_to_index.size(); }

private:
  // Type-erased base class for holding fields of any type
  struct FieldHolderBase {
    virtual ~FieldHolderBase() = default;
    virtual const std::type_info& type() const noexcept = 0;
  };

  // Concrete holder for fields of type T
  template <typename T>
  struct FieldHolder : FieldHolderBase {
    pfc::field::Field<T> field;

    explicit FieldHolder(pfc::field::Field<T>&& f) : field(std::move(f)) {}

    const std::type_info& type() const noexcept override { return typeid(T); }
  };

  // Find or create the store for type T
  template <typename T>
  std::unordered_map<size_t, std::unique_ptr<FieldHolderBase>>& store() noexcept;

  // Find the store for type T (const version)
  template <typename T>
  const std::unordered_map<size_t, std::unique_ptr<FieldHolderBase>>&
  store() const noexcept;

  std::unordered_map<std::type_index,
                     std::unordered_map<size_t, std::unique_ptr<FieldHolderBase>>>
      m_type_stores;
  std::unordered_map<std::string, size_t> m_name_to_index;
  size_t m_next_index = 0;
};

} // namespace simulation
} // namespace kernel
} // namespace openpfc

// Hash specialization so FieldHandle<T> can key unordered containers
namespace std {
template <typename T>
struct hash<openpfc::kernel::simulation::FieldHandle<T>> {
  size_t operator()(
      const openpfc::kernel::simulation::FieldHandle<T>& handle) const noexcept {
    return reinterpret_cast<size_t>(&handle.get());
  }
};
} // namespace std

#include <openpfc/kernel/simulation/simulation_state.ipp>

#endif // PFC_KERNEL_SIMULATION_SIMULATION_STATE_HPP