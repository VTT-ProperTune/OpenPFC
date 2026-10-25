// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten/fixed_bc.hpp
 * @brief Directional-solidification fixed BC (tungsten / aluminum JSON App path)
 *
 * @details
 * This file defines the FixedBC class, which enforces fixed values at domain
 * boundaries with a smooth transition region. The boundary condition:
 * - Sets field values to specified low/high values at domain edges
 * - Uses smooth exponential transition to avoid sharp discontinuities
 * - Configurable transition width and strength
 *
 * Useful for:
 * - Dirichlet boundary conditions
 * - Fixed-wall simulations
 * - Controlled boundary reservoirs
 *
 * Usage:
 * @code
 * auto bc = std::make_unique<tungsten::FixedBC>(0.0, 1.0);  // low, high values
 * bc->set_field_name("density");
 * simulator.add_boundary_condition(std::move(bc));
 * @endcode
 *
 * @see field_modifier.hpp for base class
 * @see moving_bc.hpp for time-dependent boundary condition
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#pragma once
// Deliberate app-owned copy: these are solidification-model rules, not a
// physics-agnostic OpenPFC API. Independently buildable apps cannot include a
// sibling app. Share them only after extraction to their model's software owner;
// until then retain parity with the other solidification app's boundary rules.

#include <cmath>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <tungsten/solidification_bc_rules.hpp>

namespace tungsten {

class FixedBC : public pfc::FieldModifier {

private:
  double xwidth = 20.0;
  double alpha = 1.0;
  double m_rho_low = 0.0, m_rho_high = 0.0;
  std::string m_name = "FixedBC";

public:
  FixedBC() = default;

  FixedBC(double rho_low, double rho_high)
      : m_rho_low(rho_low), m_rho_high(rho_high) {}

  void set_rho_low(double rho_low) { m_rho_low = rho_low; }
  void set_rho_high(double rho_high) { m_rho_high = rho_high; }

  const std::string &get_modifier_name() const override { return m_name; }

  /// The band against the upper x limit of the box, wherever the box is.
  [[nodiscard]] FixedBandRule rule(const pfc::Domain &domain) const {
    return {.blend = {.alpha = alpha, .rho_low = m_rho_low, .rho_high = m_rho_high},
            .xpos = pfc::domain::upper(domain)[0] - xwidth,
            .xwidth = xwidth};
  }

  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double time = 0.0) override {
    (void)time;
    const FixedBandRule r = rule(domain);
    pfc::field::apply_inplace(field, domain, box,
                              [r](const pfc::Real3 &X, double current) {
                                return r(X[0], X[1], X[2], current);
                              });
  }
};

} // namespace tungsten
