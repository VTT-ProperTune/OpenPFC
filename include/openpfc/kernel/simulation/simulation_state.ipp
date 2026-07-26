// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef PFC_KERNEL_SIMULATION_SIMULATION_STATE_IPP
#define PFC_KERNEL_SIMULATION_SIMULATION_STATE_IPP

/**
 * @file simulation_state.ipp
 * @brief Out-of-line template definitions for `pfc::SimulationState`.
 * Included at the end of `simulation_state.hpp`; not a standalone header.
 */

namespace pfc {

template <typename T, typename MemorySpace>
SimulationState::TypedStore<T, MemorySpace> &SimulationState::store() {
  using store_ptr = std::shared_ptr<TypedStore<T, MemorySpace>>;
  const auto key = store_key<T, MemorySpace>();
  auto it = m_stores.find(key);
  if (it == m_stores.end()) {
    it = m_stores
             .emplace(key, std::any(std::make_shared<TypedStore<T, MemorySpace>>()))
             .first;
  }
  return *std::any_cast<store_ptr &>(it->second);
}

template <typename T, typename MemorySpace>
SimulationState::TypedStore<T, MemorySpace> *SimulationState::find_store() noexcept {
  using store_ptr = std::shared_ptr<TypedStore<T, MemorySpace>>;
  auto it = m_stores.find(store_key<T, MemorySpace>());
  if (it == m_stores.end()) return nullptr;
  return std::any_cast<store_ptr &>(it->second).get();
}

template <typename T, typename MemorySpace>
const SimulationState::TypedStore<T, MemorySpace> *
SimulationState::find_store() const noexcept {
  using store_ptr = std::shared_ptr<TypedStore<T, MemorySpace>>;
  auto it = m_stores.find(store_key<T, MemorySpace>());
  if (it == m_stores.end()) return nullptr;
  return std::any_cast<const store_ptr &>(it->second).get();
}

template <typename T, typename MemorySpace>
void SimulationState::add_field(const std::string &name,
                                pfc::data::Field<T, MemorySpace> field) {
  if (m_name_to_id.count(name) != 0) {
    throw std::runtime_error("SimulationState::add_field: duplicate field name '" +
                             name + "'");
  }
  const std::size_t id = m_next_id++;
  auto &s = store<T, MemorySpace>();
  s.by_id.emplace(id, std::move(field));
  s.name_to_id.emplace(name, id);
  m_name_to_id.emplace(name, id);
}

template <typename T, typename MemorySpace>
pfc::data::Field<T, MemorySpace> &
SimulationState::get_field(const std::string &name) {
  const auto nit = m_name_to_id.find(name);
  if (nit == m_name_to_id.end()) {
    throw std::runtime_error("SimulationState::get_field: no field named '" + name +
                             "'");
  }
  auto *s = find_store<T, MemorySpace>();
  if (s != nullptr) {
    const auto fit = s->by_id.find(nit->second);
    if (fit != s->by_id.end()) return fit->second;
  }
  throw std::runtime_error("SimulationState::get_field: field '" + name +
                           "' is not of the requested type");
}

template <typename T, typename MemorySpace>
const pfc::data::Field<T, MemorySpace> &
SimulationState::get_field(const std::string &name) const {
  const auto nit = m_name_to_id.find(name);
  if (nit == m_name_to_id.end()) {
    throw std::runtime_error("SimulationState::get_field: no field named '" + name +
                             "'");
  }
  const auto *s = find_store<T, MemorySpace>();
  if (s != nullptr) {
    const auto fit = s->by_id.find(nit->second);
    if (fit != s->by_id.end()) return fit->second;
  }
  throw std::runtime_error("SimulationState::get_field: field '" + name +
                           "' is not of the requested type");
}

template <typename T, typename MemorySpace>
FieldHandle<T> SimulationState::get_field_handle(const std::string &name) const {
  const auto nit = m_name_to_id.find(name);
  if (nit == m_name_to_id.end()) {
    throw std::runtime_error("SimulationState::get_field_handle: no field named '" +
                             name + "'");
  }
  const auto *s = find_store<T, MemorySpace>();
  if (s == nullptr || s->by_id.count(nit->second) == 0) {
    throw std::runtime_error("SimulationState::get_field_handle: field '" + name +
                             "' is not of the requested type");
  }
  return FieldHandle<T>(nit->second);
}

template <typename T, typename MemorySpace>
pfc::data::Field<T, MemorySpace> &
SimulationState::get_field_by_handle(const FieldHandle<T> &handle) {
  auto *s = find_store<T, MemorySpace>();
  if (s != nullptr) {
    const auto fit = s->by_id.find(handle.id());
    if (fit != s->by_id.end()) return fit->second;
  }
  throw std::runtime_error(
      "SimulationState::get_field_by_handle: invalid handle or wrong "
      "(T, MemorySpace)");
}

template <typename T, typename MemorySpace>
const pfc::data::Field<T, MemorySpace> &
SimulationState::get_field_by_handle(const FieldHandle<T> &handle) const {
  const auto *s = find_store<T, MemorySpace>();
  if (s != nullptr) {
    const auto fit = s->by_id.find(handle.id());
    if (fit != s->by_id.end()) return fit->second;
  }
  throw std::runtime_error(
      "SimulationState::get_field_by_handle: invalid handle or wrong "
      "(T, MemorySpace)");
}

} // namespace pfc

#endif // PFC_KERNEL_SIMULATION_SIMULATION_STATE_IPP
