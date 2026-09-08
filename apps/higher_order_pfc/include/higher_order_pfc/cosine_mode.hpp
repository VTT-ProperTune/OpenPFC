// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cosine_mode.hpp
 * @brief Reproducible single Fourier-mode initial condition for PFC.
 *
 * @details
 * Sets \f$\psi(\mathbf x)=\psi_0+A\cos(2\pi n_x x/L_x+2\pi n_y y/L_y
 * +2\pi n_z z/L_z)\f$. JSON `"type": "cosine_mode"`. Integer mode counts keep
 * the field periodic, which is what makes the measured growth rate comparable
 * to \f$L(k)\f$ to round-off.
 */

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_field_modifiers.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace higher_order_pfc {

class CosineMode : public pfc::FieldModifier {
public:
  void set_psi0(double psi0) { m_psi0 = psi0; }
  void set_amplitude(double amplitude) { m_amplitude = amplitude; }
  void set_nx(int nx) { m_nx = nx; }
  void set_ny(int ny) { m_ny = ny; }
  void set_nz(int nz) { m_nz = nz; }

  [[nodiscard]] double psi0() const { return m_psi0; }
  [[nodiscard]] double amplitude() const { return m_amplitude; }
  [[nodiscard]] int nx() const { return m_nx; }
  [[nodiscard]] int ny() const { return m_ny; }
  [[nodiscard]] int nz() const { return m_nz; }

  const std::string &get_modifier_name() const override {
    static const std::string k{"CosineMode"};
    return k;
  }

  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double /*time*/) override {
    const auto size = pfc::domain::get_size(domain);
    const auto spacing = pfc::domain::get_spacing(domain);
    const double Lx = spacing[0] * static_cast<double>(size[0]);
    const double Ly = spacing[1] * static_cast<double>(size[1]);
    const double Lz = spacing[2] * static_cast<double>(size[2]);
    const double twopi = 2.0 * std::numbers::pi;
    const double kx = (Lx > 0.0) ? twopi * static_cast<double>(m_nx) / Lx : 0.0;
    const double ky = (Ly > 0.0) ? twopi * static_cast<double>(m_ny) / Ly : 0.0;
    const double kz = (Lz > 0.0) ? twopi * static_cast<double>(m_nz) / Lz : 0.0;
    const double psi0 = m_psi0;
    const double amp = m_amplitude;
    pfc::field::apply(field, domain, box, [=](const pfc::Real3 &x) {
      return psi0 + amp * std::cos(kx * x[0] + ky * x[1] + kz * x[2]);
    });
  }

private:
  double m_psi0{0.0};
  double m_amplitude{0.01};
  int m_nx{1};
  int m_ny{0};
  int m_nz{0};
};

inline void from_json(const nlohmann::json &j, CosineMode &ic) {
  pfc::ui::detail::throw_unless_json_modifier_type(
      j, "cosine_mode", "Invalid JSON input: missing or incorrect 'type' field.");
  if (!j.contains("psi0") || !j["psi0"].is_number()) {
    throw std::invalid_argument("cosine_mode: missing or invalid 'psi0' field.");
  }
  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "cosine_mode: missing or invalid 'amplitude' field.");
  }
  ic.set_psi0(j["psi0"].get<double>());
  ic.set_amplitude(j["amplitude"].get<double>());
  auto mode = [&](const char *key, void (CosineMode::*set)(int)) {
    if (!j.contains(key)) {
      return;
    }
    if (!j[key].is_number()) {
      throw std::invalid_argument(std::string("cosine_mode: '") + key +
                                  "' must be an integer.");
    }
    (ic.*set)(j[key].get<int>());
  };
  mode("nx", &CosineMode::set_nx);
  mode("ny", &CosineMode::set_ny);
  mode("nz", &CosineMode::set_nz);
}

} // namespace higher_order_pfc
