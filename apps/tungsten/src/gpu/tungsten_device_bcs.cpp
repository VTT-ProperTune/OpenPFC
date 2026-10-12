// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_device_bcs.cpp
 * @brief Device appliers for the fixed and moving solidification BCs.
 *
 * Compiled only into GPU targets. Registering here makes
 * `apply_field_modifier` run these BCs on the device field instead of
 * copying the whole field to the host and back every step. The rule
 * launchers are instantiated in tungsten_pointwise.inc.
 */

#include <openpfc/kernel/simulation/device_field_modifier.hpp>
#include <openpfc/runtime/gpu/modifier_rule_gpu.hpp>
#include <tungsten/fixed_bc.hpp>
#include <tungsten/moving_bc.hpp>

namespace {

namespace gpu = pfc::sim::gpu;

const pfc::RegisterDeviceModifier<tungsten::FixedBC> fixed_bc{
    [](pfc::FieldModifier &m, const pfc::DeviceFieldView &field,
       const pfc::Domain &domain, const pfc::Box3i &box, double /*time*/) {
      const auto &bc = static_cast<tungsten::FixedBC &>(m);
      gpu::modifier_rule_apply(gpu::pointwise_geometry(domain, box), field.data,
                               bc.rule(domain));
    }};

const pfc::RegisterDeviceModifier<tungsten::MovingBC> moving_bc{
    [](pfc::FieldModifier &m, const pfc::DeviceFieldView &field,
       const pfc::Domain &domain, const pfc::Box3i &box, double /*time*/) {
      auto &bc = static_cast<tungsten::MovingBC &>(m);
      const auto g = gpu::pointwise_geometry(domain, box);
      // Only the column maxima leave the device; the front tracking and its
      // MPI reduction are the host code the CPU path runs.
      bc.track_front(gpu::column_max_x(g, field.data), domain, box);
      gpu::modifier_rule_apply(g, field.data, bc.band_rule(domain));
    }};

} // namespace
