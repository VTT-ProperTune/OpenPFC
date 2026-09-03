// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef PFC_KERNEL_SIMULATION_SIMULATION_STATE_HPP
#define PFC_KERNEL_SIMULATION_SIMULATION_STATE_HPP

/**
 * @file simulation_state.hpp
 * @brief M2 owning simulation state: `pfc::SimulationState`.
 *
 * `SimulationState` owns the canonical `pfc::data::Field<T, MemorySpace>`
 * containers (from `kernel/data/grid_field.hpp`) **by value**, keyed two ways:
 *   - by **name** (a `std::string`) for I/O, checkpointing and wiring, and
 *   - by a **typed handle** (`FieldHandle<T>`) for hot-path lookup that skips
 *     the string hash.
 *
 * It can hold fields of different element types (e.g. `double` and
 * `std::complex<double>`) and different memory spaces (host or device) at the
 * same time. Heterogeneous field values are stored through a small type-erased
 * layer: one `TypedStore<T, MemorySpace>` per concrete `(T, MemorySpace)` pair,
 * looked up by `std::type_index`. There is deliberately **no** new `Field`
 * type and **no** virtual field base -- the canonical `pfc::data::Field` is
 * used directly (Audit §13.3; the previous attempt was rejected for inventing a
 * colliding duplicate `pfc::data::Field`).
 *
 * Owns named fields for 0.2 sessions (`pfc::ui::SpectralETDSession`, `SpectralETDSystem`).
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>

namespace pfc {

/**
 * @brief Opaque, type-safe handle to a field owned by a `SimulationState`.
 *
 * A handle carries an integer id (`invalid_id` is the sentinel for a
 * default-constructed handle) and the element type `T` in its own type.
 * Retrieving the field additionally needs the `MemorySpace` (supplied at the
 * call site), so a handle stays as small as a `std::size_t` while still
 * preventing a `double` handle from being used to fetch a
 * `std::complex<double>` field.
 */
template <typename T> class FieldHandle {
public:
  using value_type = T;
  static constexpr std::size_t invalid_id = static_cast<std::size_t>(-1);

  FieldHandle() noexcept : m_id(invalid_id) {}
  explicit FieldHandle(std::size_t id) noexcept : m_id(id) {}

  std::size_t id() const noexcept { return m_id; }
  bool is_valid() const noexcept { return m_id != invalid_id; }

  bool operator==(const FieldHandle &other) const noexcept {
    return m_id == other.m_id;
  }
  bool operator!=(const FieldHandle &other) const noexcept {
    return !(*this == other);
  }

private:
  std::size_t m_id;
};

/**
 * @brief Owns canonical `pfc::data::Field` values keyed by name and typed handle.
 *
 * Field names are unique across the whole state (a name identifies exactly one
 * field, of exactly one `(T, MemorySpace)`). Handle ids are unique across all
 * fields regardless of type, so a handle never aliases another field.
 */
class SimulationState {
public:
  SimulationState() = default;

  /**
   * @brief Take ownership of @p field under @p name.
   * @throws std::runtime_error if @p name is already in use.
   */
  template <typename T, typename MemorySpace = pfc::HostSpace>
  void add_field(const std::string &name, pfc::data::Field<T, MemorySpace> field);

  /**
   * @brief Reference to the field named @p name.
   * @throws std::runtime_error if no such field, or it is of another
   *         `(T, MemorySpace)` than requested.
   */
  template <typename T, typename MemorySpace = pfc::HostSpace>
  pfc::data::Field<T, MemorySpace> &get_field(const std::string &name);
  template <typename T, typename MemorySpace = pfc::HostSpace>
  const pfc::data::Field<T, MemorySpace> &get_field(const std::string &name) const;

  /**
   * @brief Handle for the field named @p name, for repeated hot-path access.
   * @throws std::runtime_error if no such field of the requested type exists.
   */
  template <typename T, typename MemorySpace = pfc::HostSpace>
  FieldHandle<T> get_field_handle(const std::string &name) const;

  /**
   * @brief Reference to the field a handle refers to.
   * @throws std::runtime_error if the handle is invalid or refers to a field of
   *         another `(T, MemorySpace)` than requested.
   */
  template <typename T, typename MemorySpace = pfc::HostSpace>
  pfc::data::Field<T, MemorySpace> &
  get_field_by_handle(const FieldHandle<T> &handle);
  template <typename T, typename MemorySpace = pfc::HostSpace>
  const pfc::data::Field<T, MemorySpace> &
  get_field_by_handle(const FieldHandle<T> &handle) const;

  /// True if a field named @p name exists (of any type/memory space).
  bool has_field(const std::string &name) const noexcept {
    return m_name_to_id.count(name) != 0;
  }

  /// True if @p name is a field of `(T, MemorySpace)`.
  template <typename T, typename MemorySpace = pfc::HostSpace>
  [[nodiscard]] bool has_typed_field(const std::string &name) const noexcept;

  /// Number of fields currently owned.
  std::size_t num_fields() const noexcept { return m_name_to_id.size(); }

  /// Sorted field names (I/O and checkpoint).
  [[nodiscard]] std::vector<std::string> field_names() const {
    std::vector<std::string> names;
    names.reserve(m_name_to_id.size());
    for (const auto &kv : m_name_to_id) {
      names.push_back(kv.first);
    }
    std::sort(names.begin(), names.end());
    return names;
  }

private:
  // One store per concrete (T, MemorySpace), held via a std::shared_ptr inside
  // a std::any and keyed by the field type's std::type_index. The shared_ptr
  // keeps std::any's copy-constructibility requirement off `Field` itself, so a
  // move-only device `Field` can still be owned here.
  template <typename T, typename MemorySpace> struct TypedStore {
    std::unordered_map<std::size_t, pfc::data::Field<T, MemorySpace>> by_id;
    std::unordered_map<std::string, std::size_t> name_to_id;
  };

  template <typename T, typename MemorySpace>
  static std::type_index store_key() noexcept {
    return std::type_index(typeid(pfc::data::Field<T, MemorySpace>));
  }

  // Get-or-create the store for (T, MemorySpace).
  template <typename T, typename MemorySpace> TypedStore<T, MemorySpace> &store();

  // Find the store for (T, MemorySpace); nullptr if none has been created.
  template <typename T, typename MemorySpace>
  TypedStore<T, MemorySpace> *find_store() noexcept;
  template <typename T, typename MemorySpace>
  const TypedStore<T, MemorySpace> *find_store() const noexcept;

  std::unordered_map<std::type_index, std::any> m_stores;
  std::unordered_map<std::string, std::size_t> m_name_to_id;
  std::size_t m_next_id = 0; // ids count up; invalid_id (size_t -1) is the sentinel
};

} // namespace pfc

namespace std {
/// Hash specialization so `FieldHandle<T>` can key unordered containers.
template <typename T> struct hash<pfc::FieldHandle<T>> {
  std::size_t operator()(const pfc::FieldHandle<T> &handle) const noexcept {
    return std::hash<std::size_t>{}(handle.id());
  }
};
} // namespace std

#include <openpfc/kernel/simulation/simulation_state.ipp>

#endif // PFC_KERNEL_SIMULATION_SIMULATION_STATE_HPP
