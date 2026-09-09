// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file wave_packet.hpp
 * @brief Gaussian-envelope carrier wave packet, \(u=u_0+A\exp(-(x-x_0)^2/2\sigma^2)
 *        \cos(k_0(x-x_0)+\phi_0)\) (`#119`).
 *
 * JSON `"type": "wave_packet"`. This is the science-case A initial condition:
 * a narrow-band Gaussian-envelope carrier at wave number \(k_0\), used to
 * measure group velocity (envelope motion) against the carrier phase velocity
 * and against \(d\omega/dk\)/\(\omega/k\).
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

class WavePacket : public pfc::FieldModifier {
public:
  void set_u0(double u0) { m_u0 = u0; }
  void set_amplitude(double amplitude) { m_amplitude = amplitude; }
  void set_sigma(double sigma) { m_sigma = sigma; }
  void set_k0(double k0) { m_k0 = k0; }
  void set_x0(double x0) { m_x0 = x0; }
  void set_phase0(double phase0) { m_phase0 = phase0; }

  [[nodiscard]] double u0() const { return m_u0; }
  [[nodiscard]] double amplitude() const { return m_amplitude; }
  [[nodiscard]] double sigma() const { return m_sigma; }
  [[nodiscard]] double k0() const { return m_k0; }
  [[nodiscard]] double x0() const { return m_x0; }
  [[nodiscard]] double phase0() const { return m_phase0; }

  const std::string &get_modifier_name() const override {
    static const std::string k{"WavePacket"};
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
    const double k0 = m_k0;
    const double phase0 = m_phase0;
    if (!(sig > 0.0)) {
      throw std::invalid_argument("wave_packet: sigma must be > 0.");
    }
    const double inv = 1.0 / (2.0 * sig * sig);
    pfc::field::apply(field, domain, box, [=](const pfc::Real3 &x) {
      const double dx = x[0] - x0;
      const double envelope = amp * std::exp(-dx * dx * inv);
      return u0 + envelope * std::cos(k0 * dx + phase0);
    });
  }

private:
  double m_u0{0.0};
  double m_amplitude{0.05};
  double m_sigma{20.0};
  double m_k0{0.5};
  double m_phase0{0.0};
  double m_x0{std::numeric_limits<double>::quiet_NaN()};
};

inline void from_json(const nlohmann::json &j, WavePacket &ic) {
  pfc::ui::detail::throw_unless_json_modifier_type(
      j, "wave_packet", "Invalid JSON input: missing or incorrect 'type' field.");
  if (j.contains("u0")) {
    if (!j["u0"].is_number()) {
      throw std::invalid_argument("wave_packet: 'u0' must be numeric.");
    }
    ic.set_u0(j["u0"].get<double>());
  }
  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "wave_packet: missing or invalid 'amplitude' field.");
  }
  ic.set_amplitude(j["amplitude"].get<double>());
  if (!j.contains("sigma") || !j["sigma"].is_number()) {
    throw std::invalid_argument("wave_packet: missing or invalid 'sigma' field.");
  }
  ic.set_sigma(j["sigma"].get<double>());
  if (!j.contains("k0") || !j["k0"].is_number()) {
    throw std::invalid_argument("wave_packet: missing or invalid 'k0' field.");
  }
  ic.set_k0(j["k0"].get<double>());
  if (j.contains("phase0")) {
    if (!j["phase0"].is_number()) {
      throw std::invalid_argument("wave_packet: 'phase0' must be numeric.");
    }
    ic.set_phase0(j["phase0"].get<double>());
  }
  if (j.contains("x0")) {
    if (!j["x0"].is_number()) {
      throw std::invalid_argument("wave_packet: 'x0' must be numeric.");
    }
    ic.set_x0(j["x0"].get<double>());
  }
}

} // namespace kawahara
