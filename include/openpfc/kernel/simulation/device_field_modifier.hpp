// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file device_field_modifier.hpp
 * @brief Device appliers for `FieldModifier`s, registered by concrete type.
 *
 * @details
 * A `FieldModifier` works on a host buffer. For a device-resident field,
 * `apply_field_modifier` used to bracket every call with
 * `Field::with_host_view`: a full device-to-host copy, the host loop, and a
 * full host-to-device copy, on every boundary-condition step.
 *
 * A modifier that can run on the device registers an applier for its type:
 *
 * @code
 * // A host TU compiled only into GPU targets:
 * static const pfc::RegisterDeviceModifier<foo::FooBC> reg{
 *     [](pfc::FieldModifier &m, const pfc::DeviceFieldView &f,
 *        const pfc::Domain &d, const pfc::Box3i &b, double t) {
 *       static_cast<foo::FooBC &>(m).apply_on_device(f, d, b, t);
 *     }};
 * @endcode
 *
 * `apply_field_modifier` looks the modifier's dynamic type up here and runs
 * the applier on the device buffer; a type with no applier keeps the host
 * bracket. The registry is keyed by the exact type, so a subclass is not
 * served by its base's applier. Registration lives in a GPU-only TU, so CPU
 * targets never reference device code.
 */

#include <cstddef>
#include <functional>
#include <typeindex>
#include <unordered_map>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace pfc {

/// The owned box of a device field: x-fastest, storage halo 0.
struct DeviceFieldView {
  double *data = nullptr;
  std::size_t size = 0;
};

/// Applies a modifier to a device field in place.
using DeviceModifierApplier =
    std::function<void(FieldModifier &, const DeviceFieldView &, const Domain &,
                       const Box3i &, double)>;

namespace detail {
inline std::unordered_map<std::type_index, DeviceModifierApplier> &
device_modifier_registry() {
  static std::unordered_map<std::type_index, DeviceModifierApplier> registry;
  return registry;
}
} // namespace detail

/// Registers @p apply as the device applier of modifier type `M`.
template <class M> struct RegisterDeviceModifier {
  explicit RegisterDeviceModifier(DeviceModifierApplier apply) {
    detail::device_modifier_registry()[std::type_index(typeid(M))] =
        std::move(apply);
  }
};

/// True when the dynamic type of @p m has a device applier.
[[nodiscard]] inline bool has_device_applier(const FieldModifier &m) {
  return detail::device_modifier_registry().count(std::type_index(typeid(m))) != 0;
}

/**
 * @brief Run the device applier of @p m on @p field.
 * @return false when the type has none; the caller falls back to the host.
 */
inline bool try_apply_on_device(FieldModifier &m, const DeviceFieldView &field,
                                const Domain &domain, const Box3i &box,
                                double time) {
  auto &registry = detail::device_modifier_registry();
  const auto it = registry.find(std::type_index(typeid(m)));
  if (it == registry.end()) return false;
  it->second(m, field, domain, box, time);
  return true;
}

} // namespace pfc
