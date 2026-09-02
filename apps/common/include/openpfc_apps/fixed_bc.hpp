// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file openpfc_apps/fixed_bc.hpp
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
 * auto bc = std::make_unique<pfc::FixedBC>(0.0, 1.0);  // low, high values
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

#include <cmath>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/model.hpp>

namespace pfc {

using pfc::types::Int3;

class FixedBC : public FieldModifier {

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

  void apply(RealField &field, const Domain &domain, const Box3i &box) const {
    const double Lx = pfc::domain::get_size(domain, 0);
    const double dx = pfc::domain::get_spacing(domain, 0);
    const double xpos = (Lx * dx) - xwidth;

    pfc::field::apply_inplace(
        field, domain, box, [=, this](const pfc::Real3 &X, double current) {
          const double x = X[0];
          if (std::abs(x - xpos) < xwidth) {
            const double S = 1.0 / (1.0 + std::exp(-alpha * (x - xpos)));
            return (m_rho_low * S) + (m_rho_high * (1.0 - S));
          }
          return current;
        });
  }

  void apply(Model &m, double time) override {
    (void)time;
    apply(pfc::get_real_field(m, get_field_name()), pfc::get_world(m).domain_,
          pfc::fft::get_inbox(pfc::get_fft(m)));
  }
};

} // namespace pfc
