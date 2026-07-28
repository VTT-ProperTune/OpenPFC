// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef PFC_KERNEL_SIMULATION_SIMULATION_STATE_IPP
#define PFC_KERNEL_SIMULATION_SIMULATION_STATE_IPP

/**
 * @file simulation_state.ipp
 * @brief Out-of-line template definitions for `SimulationState`.
 * Included at the end of `simulation_state.hpp`; not a standalone header.
 */

namespace openpfc {
namespace kernel {
namespace simulation {

template <typename T>
std::unordered_map<size_t, std::unique_ptr<SimulationState::FieldHolderBase>>&
SimulationState::store() noexcept {
  const auto key = std::type_index(typeid(T));
  return m_type_stores[key];
}

template <typename T>
const std::unordered_map<size_t, std::unique_ptr<SimulationState::FieldHolderBase>>&
SimulationState::store() const noexcept {
  static const std::unordered_map<size_t,
                                   std::unique_ptr<FieldHolderBase>>
      empty_store;
  const auto key = std::type_index(typeid(T));
  auto it = m_type_stores.find(key);
  if (it == m_type_stores.end()) {
    return empty_store;
  }
  return it->second;
}

template <typename T>
void SimulationState::insert_field(std::string name,
                                   pfc::field::Field<T>&& field) {
  if (m_name_to_index.count(name) != 0) {
    throw std::runtime_error(
        "SimulationState::insert_field: duplicate field name '" + name + "'");
  }

  const size_t index = m_next_index++;
  auto holder = std::make_unique<FieldHolder<T>>(std::move(field));
  auto& s = store<T>();
  s.emplace(index, std::move(holder));
  m_name_to_index.emplace(std::move(name), index);
}

template <typename T>
FieldHandle<T> SimulationState::get_field(const std::string& name) noexcept {
  const auto nit = m_name_to_index.find(name);
  if (nit == m_name_to_index.end()) {
    return FieldHandle<T>(); // null handle
  }

  const auto& s = store<T>();
  const auto fit = s.find(nit->second);
  if (fit == s.end()) {
    return FieldHandle<T>(); // null handle (wrong type)
  }

  auto* holder = static_cast<FieldHolder<T>*>(fit->second.get());
  return FieldHandle<T>(&holder->field);
}

template <typename T>
bool SimulationState::remove_field(const std::string& name) {
  const auto nit = m_name_to_index.find(name);
  if (nit == m_name_to_index.end()) {
    return false; // not found
  }

  auto& s = store<T>();
  const auto fit = s.find(nit->second);
  if (fit == s.end()) {
    return false; // wrong type
  }

  s.erase(fit);
  m_name_to_index.erase(nit);
  return true;
}

inline void SimulationState::clear() noexcept {
  m_type_stores.clear();
  m_name_to_index.clear();
  m_next_index = 0;
}

} // namespace simulation
} // namespace kernel
} // namespace openpfc

#endif // PFC_KERNEL_SIMULATION_SIMULATION_STATE_IPP