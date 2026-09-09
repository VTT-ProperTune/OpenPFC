// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file circular_inclusion.hpp
 * @brief Smoothed flat-top circular inclusion:
 * \(g=g_0+A\cdot\tfrac12\bigl(1-\tanh\bigl(\tfrac{r-R}{w}\bigr)\bigr)\).
 *
 * @details
 * `g` is \(\approx A\) for \(r\ll R\), \(\approx 0\) for \(r\gg R\), and
 * transitions over an interface half-width \(w\) around \(r=R\) (a
 * `tanh`-smoothed indicator, not a sharp disk): a sharp disk has infinite
 * Fourier content and rings on a periodic grid; the smoothing keeps the
 * eigenstrain field well resolved at finite grid spacing while still
 * defining a clean radius `R` for the `#117` inclusion size sweep. Distance
 * `r` from the centre is measured with periodic (minimum-image) wrapping, so
 * this is well-defined even when `R` is a sizeable fraction of the box.
 *
 * This is the *dilatational eigenstrain amplitude* field `g`; the physics
 * eigenstrain is \(\varepsilon^*=\varepsilon_0 g I\) (`eps0` in
 * `model.params`), matching `cosine_mode`/`gaussian_inclusion`.
 *
 * JSON `"type": "circular_inclusion"`.
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

namespace gradient_elasticity {

class CircularInclusion : public pfc::FieldModifier {
public:
  void set_g0(double g0) { m_g0 = g0; }
  void set_amplitude(double amplitude) { m_amplitude = amplitude; }
  void set_radius(double radius) { m_radius = radius; }
  void set_interface_width(double w) { m_interface_width = w; }
  void set_x0(double x0) { m_x0 = x0; }
  void set_y0(double y0) { m_y0 = y0; }

  [[nodiscard]] double g0() const { return m_g0; }
  [[nodiscard]] double amplitude() const { return m_amplitude; }
  [[nodiscard]] double radius() const { return m_radius; }
  [[nodiscard]] double interface_width() const { return m_interface_width; }
  [[nodiscard]] double x0() const { return m_x0; }
  [[nodiscard]] double y0() const { return m_y0; }

  const std::string &get_modifier_name() const override {
    static const std::string k{"CircularInclusion"};
    return k;
  }

  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double /*time*/) override {
    const auto size = pfc::domain::get_size(domain);
    const auto spacing = pfc::domain::get_spacing(domain);
    const double Lx = spacing[0] * static_cast<double>(size[0]);
    const double Ly = spacing[1] * static_cast<double>(size[1]);
    const double x0 = std::isfinite(m_x0) ? m_x0 : 0.5 * Lx;
    const double y0 = std::isfinite(m_y0) ? m_y0 : 0.5 * Ly;
    const double g0 = m_g0;
    const double amp = m_amplitude;
    const double R = m_radius;
    const double w = m_interface_width;
    if (!(R > 0.0)) {
      throw std::invalid_argument("circular_inclusion: radius must be > 0.");
    }
    if (!(w > 0.0)) {
      throw std::invalid_argument("circular_inclusion: interface_width must be > 0.");
    }
    auto wrap = [](double d, double L) {
      if (!(L > 0.0)) {
        return d;
      }
      return d - L * std::nearbyint(d / L);
    };
    pfc::field::apply(field, domain, box, [=](const pfc::Real3 &x) {
      const double dx = wrap(x[0] - x0, Lx);
      const double dy = wrap(x[1] - y0, Ly);
      const double r = std::sqrt(dx * dx + dy * dy);
      return g0 + amp * 0.5 * (1.0 - std::tanh((r - R) / w));
    });
  }

private:
  double m_g0{0.0};
  double m_amplitude{1.0};
  double m_radius{8.0};
  double m_interface_width{1.0};
  double m_x0{std::numeric_limits<double>::quiet_NaN()};
  double m_y0{std::numeric_limits<double>::quiet_NaN()};
};

inline void from_json(const nlohmann::json &j, CircularInclusion &ic) {
  pfc::ui::detail::throw_unless_json_modifier_type(
      j, "circular_inclusion",
      "Invalid JSON input: missing or incorrect 'type' field.");
  if (j.contains("g0")) {
    if (!j["g0"].is_number()) {
      throw std::invalid_argument("circular_inclusion: 'g0' must be numeric.");
    }
    ic.set_g0(j["g0"].get<double>());
  }
  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "circular_inclusion: missing or invalid 'amplitude' field.");
  }
  ic.set_amplitude(j["amplitude"].get<double>());
  if (!j.contains("radius") || !j["radius"].is_number()) {
    throw std::invalid_argument(
        "circular_inclusion: missing or invalid 'radius' field.");
  }
  ic.set_radius(j["radius"].get<double>());
  if (j.contains("interface_width")) {
    if (!j["interface_width"].is_number()) {
      throw std::invalid_argument(
          "circular_inclusion: 'interface_width' must be numeric.");
    }
    ic.set_interface_width(j["interface_width"].get<double>());
  }
  if (j.contains("x0")) {
    if (!j["x0"].is_number()) {
      throw std::invalid_argument("circular_inclusion: 'x0' must be numeric.");
    }
    ic.set_x0(j["x0"].get<double>());
  }
  if (j.contains("y0")) {
    if (!j["y0"].is_number()) {
      throw std::invalid_argument("circular_inclusion: 'y0' must be numeric.");
    }
    ic.set_y0(j["y0"].get<double>());
  }
}

} // namespace gradient_elasticity
