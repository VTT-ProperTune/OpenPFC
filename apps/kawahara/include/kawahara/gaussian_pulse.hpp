// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file gaussian_pulse.hpp
 * @brief Localized 1D pulse \(u=u_0+A\exp(-(x-x_0)^2/(2\sigma^2))\).
 *
 * JSON `"type": "gaussian_pulse"`. Default \(x_0=L_x/2\).
 */

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_field_modifiers.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace kawahara {

class GaussianPulse : public pfc::FieldModifier {
public:
  void set_u0(double u0) { m_u0 = u0; }
  void set_amplitude(double amplitude) { m_amplitude = amplitude; }
  void set_sigma(double sigma) { m_sigma = sigma; }
  void set_x0(double x0) { m_x0 = x0; }

  [[nodiscard]] double u0() const { return m_u0; }
  [[nodiscard]] double amplitude() const { return m_amplitude; }
  [[nodiscard]] double sigma() const { return m_sigma; }
  [[nodiscard]] double x0() const { return m_x0; }

  const std::string &get_modifier_name() const override {
    static const std::string k{"GaussianPulse"};
    return k;
  }

  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double /*time*/) override {
    const auto size = pfc::domain::get_size(domain);
    const auto spacing = pfc::domain::get_spacing(domain);
    const double Lx = spacing[0] * static_cast<double>(size[0]);
    const double x0 = std::isfinite(m_x0) ? m_x0 : 0.5 * Lx;
    const double u0 = m_u0;
    const double amp = m_amplitude;
    const double sig = m_sigma;
    if (!(sig > 0.0)) {
      throw std::invalid_argument("gaussian_pulse: sigma must be > 0.");
    }
    const double inv = 1.0 / (2.0 * sig * sig);
    pfc::field::apply(field, domain, box, [=](const pfc::Real3 &x) {
      const double dx = x[0] - x0;
      return u0 + amp * std::exp(-dx * dx * inv);
    });
  }

private:
  double m_u0{0.0};
  double m_amplitude{0.4};
  double m_sigma{4.0};
  double m_x0{std::numeric_limits<double>::quiet_NaN()};
};

inline void from_json(const nlohmann::json &j, GaussianPulse &ic) {
  pfc::ui::detail::throw_unless_json_modifier_type(
      j, "gaussian_pulse", "Invalid JSON input: missing or incorrect 'type' field.");
  if (j.contains("u0")) {
    if (!j["u0"].is_number()) {
      throw std::invalid_argument("gaussian_pulse: 'u0' must be numeric.");
    }
    ic.set_u0(j["u0"].get<double>());
  }
  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "gaussian_pulse: missing or invalid 'amplitude' field.");
  }
  ic.set_amplitude(j["amplitude"].get<double>());
  if (!j.contains("sigma") || !j["sigma"].is_number()) {
    throw std::invalid_argument("gaussian_pulse: missing or invalid 'sigma' field.");
  }
  ic.set_sigma(j["sigma"].get<double>());
  if (j.contains("x0")) {
    if (!j["x0"].is_number()) {
      throw std::invalid_argument("gaussian_pulse: 'x0' must be numeric.");
    }
    ic.set_x0(j["x0"].get<double>());
  }
}

} // namespace kawahara
