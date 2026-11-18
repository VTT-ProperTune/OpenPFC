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
 *
 * Device fields come in both precisions (ADR 0006): `DeviceFieldViewT<T>`
 * is the owned box of a `Field<T, DeviceSpace>`, and the registry holds one
 * applier per scalar. A generic callable that accepts both views registers
 * for both at once:
 *
 * @code
 * static const pfc::RegisterDeviceModifier<foo::FooBC> reg{
 *     [](pfc::FieldModifier &m, const auto &f, const pfc::Domain &d,
 *        const pfc::Box3i &b, double t) {
 *       // f.data is double* or float*; the rule launchers accept both.
 *       static_cast<foo::FooBC &>(m).apply_on_device(f, d, b, t);
 *     }};
 * @endcode
 *
 * A callable that only takes `const DeviceFieldView &` (double) registers for
 * double alone, and a float device field of that modifier keeps the host
 * bracket.
 */

#include <cstddef>
#include <functional>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace pfc {

/// The owned box of a device field of scalar `T`: x-fastest, storage halo 0.
template <class T> struct DeviceFieldViewT {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                "DeviceFieldViewT: device appliers serve real fields");
  T *data = nullptr;
  std::size_t size = 0;
};

/// The double-precision view, the form every shipped applier was written for.
using DeviceFieldView = DeviceFieldViewT<double>;
/// The single-precision view (ADR 0006).
using DeviceFieldViewF = DeviceFieldViewT<float>;

/// Applies a modifier to a device field of scalar `T` in place.
template <class T>
using DeviceModifierApplierT =
    std::function<void(FieldModifier &, const DeviceFieldViewT<T> &, const Domain &,
                       const Box3i &, double)>;
using DeviceModifierApplier = DeviceModifierApplierT<double>;

namespace detail {
template <class T>
inline std::unordered_map<std::type_index, DeviceModifierApplierT<T>> &
device_modifier_registry() {
  static std::unordered_map<std::type_index, DeviceModifierApplierT<T>> registry;
  return registry;
}

template <class F, class T>
inline constexpr bool is_device_applier_for_v =
    std::is_invocable_v<F &, FieldModifier &, const DeviceFieldViewT<T> &,
                        const Domain &, const Box3i &, double>;
} // namespace detail

/**
 * @brief Registers @p apply as the device applier of modifier type `M`.
 *
 * @p apply is stored for every scalar it can be called with: a generic
 * callable (`const auto &field`) serves double and float device fields, a
 * callable taking `const DeviceFieldView &` serves double only.
 */
template <class M> struct RegisterDeviceModifier {
  template <class F> explicit RegisterDeviceModifier(F apply) {
    static_assert(detail::is_device_applier_for_v<F, double>,
                  "RegisterDeviceModifier: the applier must accept a "
                  "DeviceFieldView (double) field");
    detail::device_modifier_registry<double>()[std::type_index(typeid(M))] = apply;
    if constexpr (detail::is_device_applier_for_v<F, float>) {
      detail::device_modifier_registry<float>()[std::type_index(typeid(M))] =
          std::move(apply);
    }
  }
};

/// True when the dynamic type of @p m has a device applier for scalar `T`.
template <class T = double>
[[nodiscard]] inline bool has_device_applier(const FieldModifier &m) {
  return detail::device_modifier_registry<T>().count(std::type_index(typeid(m))) !=
         0;
}

/**
 * @brief Run the device applier of @p m on @p field.
 * @return false when the type has none for this scalar; the caller falls
 *         back to the host.
 */
template <class T>
inline bool try_apply_on_device(FieldModifier &m, const DeviceFieldViewT<T> &field,
                                const Domain &domain, const Box3i &box,
                                double time) {
  auto &registry = detail::device_modifier_registry<T>();
  const auto it = registry.find(std::type_index(typeid(m)));
  if (it == registry.end()) return false;
  it->second(m, field, domain, box, time);
  return true;
}

} // namespace pfc
