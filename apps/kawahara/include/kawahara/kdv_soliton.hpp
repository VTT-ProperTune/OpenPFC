// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file kdv_soliton.hpp
 * @brief Exact KdV solitary wave, the control state for the fifth-order study.
 *
 * @details
 * JSON `"type": "kdv_soliton"`.
 *
 * With \f$\gamma=0\f$ this app's equation
 *
 * \f[
 *   u_t + \alpha u u_x - \beta u_{xxx} + \gamma u_{xxxxx} = 0
 * \f]
 *
 * is the Korteweg–de Vries equation with dispersion coefficient
 * \f$\delta=-\beta\f$, and it has the exact travelling solution
 *
 * \f[
 *   u(x,t) = A\,\operatorname{sech}^2\!\Bigl(\frac{x-x_0-ct}{W}\Bigr),
 *   \qquad
 *   c = \frac{\alpha A}{3},
 *   \qquad
 *   W = \sqrt{\frac{12\delta}{\alpha A}} = \sqrt{\frac{-12\beta}{\alpha A}}.
 * \f]
 *
 * ## Why this initial condition exists
 *
 * The question the science preset asks is *what does the fifth-order term
 * do?*, and answering it needs a control whose behaviour without that term is
 * known exactly rather than merely observed. A Gaussian bump is not a solution
 * of either equation: it steepens under \f$\alpha uu_x\f$, and near the
 * critical Bond number \f$\tau=1/3\f$ — the regime this app is about — the
 * third-order dispersion left to arrest that steepening is weak, so the
 * control run ends up integrating the numerical approach to a gradient
 * singularity. Its own behaviour then swamps the effect being measured, and
 * whether it survives at all becomes a property of the platform's rounding.
 *
 * The solitary wave removes that: at \f$\gamma=0\f$ it propagates unchanged
 * indefinitely, so *every* departure from a constant peak amplitude and an
 * empty tail is attributable to \f$\gamma\f$.
 *
 * ## Width is derived, not supplied
 *
 * \f$W\f$ follows from \f$(\alpha,\beta,A)\f$; supplying it independently
 * would let an input silently stop being a solution of the equation it is
 * run against. So the modifier takes \f$\alpha\f$ and \f$\beta\f$ — which
 * must match the `model.params` of the same session — and computes \f$W\f$.
 *
 * A real \f$W\f$ needs \f$-\beta/(\alpha A)>0\f$: with \f$\alpha>0\f$ and
 * \f$A>0\f$ that means \f$\beta<0\f$, i.e. a Bond number below the critical
 * \f$1/3\f$. Above it, KdV of this sign carries depression solitary waves
 * instead and \f$A\f$ must be negative. The modifier rejects the combination
 * rather than producing a NaN field.
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

/// Width of the KdV solitary wave of amplitude @p A. Non-finite when the
/// sign combination admits no such wave.
[[nodiscard]] inline double kdv_soliton_width(double alpha, double beta, double A) {
  const double arg = -12.0 * beta / (alpha * A);
  return (arg > 0.0) ? std::sqrt(arg) : std::numeric_limits<double>::quiet_NaN();
}

/// Speed of the KdV solitary wave of amplitude @p A.
[[nodiscard]] inline double kdv_soliton_speed(double alpha, double A) {
  return alpha * A / 3.0;
}

class KdVSoliton : public pfc::FieldModifier {
public:
  void set_u0(double u0) { m_u0 = u0; }
  void set_amplitude(double amplitude) { m_amplitude = amplitude; }
  void set_alpha(double alpha) { m_alpha = alpha; }
  void set_beta(double beta) { m_beta = beta; }
  void set_x0(double x0) { m_x0 = x0; }

  [[nodiscard]] double u0() const { return m_u0; }
  [[nodiscard]] double amplitude() const { return m_amplitude; }
  [[nodiscard]] double alpha() const { return m_alpha; }
  [[nodiscard]] double beta() const { return m_beta; }
  [[nodiscard]] double x0() const { return m_x0; }

  /// \f$W=\sqrt{-12\beta/(\alpha A)}\f$, from the coefficients as configured.
  [[nodiscard]] double width() const {
    return kdv_soliton_width(m_alpha, m_beta, m_amplitude);
  }
  /// \f$c=\alpha A/3\f$, the speed the wave should be observed to travel at.
  [[nodiscard]] double speed() const {
    return kdv_soliton_speed(m_alpha, m_amplitude);
  }

  const std::string &get_modifier_name() const override {
    static const std::string k{"KdVSoliton"};
    return k;
  }

  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double /*time*/) override {
    const auto size = pfc::domain::get_size(domain);
    const auto spacing = pfc::domain::get_spacing(domain);
    const double Lx = spacing[0] * static_cast<double>(size[0]);
    const double x0 = std::isfinite(m_x0) ? m_x0 : 0.5 * Lx;
    const double w = width();
    if (!std::isfinite(w)) {
      throw std::invalid_argument(
          "kdv_soliton: no sech^2 solitary wave exists for this sign "
          "combination; -12*beta/(alpha*amplitude) must be positive (for "
          "alpha > 0 and amplitude > 0 that means beta < 0, i.e. a Bond "
          "number below the critical 1/3).");
    }
    const double u0 = m_u0;
    const double amp = m_amplitude;
    // The profile is evaluated on the periodic image nearest x0, so a wave
    // placed near a boundary is still a single pulse rather than two halves.
    pfc::field::apply(field, domain, box, [=](const pfc::Real3 &x) {
      double d = x[0] - x0;
      if (d > 0.5 * Lx) d -= Lx;
      if (d < -0.5 * Lx) d += Lx;
      const double s = 1.0 / std::cosh(d / w);
      return u0 + amp * s * s;
    });
  }

private:
  double m_u0{0.0};
  double m_amplitude{0.05};
  double m_alpha{1.5};
  double m_beta{-1.0 / 60.0};
  double m_x0{std::numeric_limits<double>::quiet_NaN()};
};

inline void from_json(const nlohmann::json &j, KdVSoliton &ic) {
  pfc::ui::detail::throw_unless_json_modifier_type(
      j, "kdv_soliton", "Invalid JSON input: missing or incorrect 'type' field.");
  auto required = [&](const char *name) {
    if (!j.contains(name) || !j[name].is_number()) {
      throw std::invalid_argument(std::string("kdv_soliton: missing or invalid '") +
                                  name + "' field.");
    }
    return j[name].get<double>();
  };
  ic.set_amplitude(required("amplitude"));
  // alpha and beta are required rather than defaulted: the width they imply is
  // what makes the field a solution, so a session that forgets them would run
  // a plausible-looking pulse that is not one.
  ic.set_alpha(required("alpha"));
  ic.set_beta(required("beta"));
  if (j.contains("u0")) {
    if (!j["u0"].is_number()) {
      throw std::invalid_argument("kdv_soliton: 'u0' must be numeric.");
    }
    ic.set_u0(j["u0"].get<double>());
  }
  if (j.contains("x0")) {
    if (!j["x0"].is_number()) {
      throw std::invalid_argument("kdv_soliton: 'x0' must be numeric.");
    }
    ic.set_x0(j["x0"].get<double>());
  }
  if (!std::isfinite(ic.width())) {
    throw std::invalid_argument(
        "kdv_soliton: no sech^2 solitary wave exists for this sign "
        "combination; -12*beta/(alpha*amplitude) must be positive.");
  }
}

} // namespace kawahara
