// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cmath>
#include <openpfc/kernel/data/host_device.hpp>
#include <tuple>
namespace coupled {
// start snippet derivatives
// Only these members are requested from either gradient backend.
struct UGrads {
  double xx{}, yy{}, zz{}, xy{}, yz{};
};
struct VGrads {
  double value{}, xz{};
};
struct Local {
  UGrads u;
  VGrads v;
};
// end snippet derivatives

struct Parameters {
  double a{.3}, b{.04}, c{-.2}, d{2. / 15.};
};
// start snippet increments
struct Increments {
  double du{}, dv{};
  auto as_tuple() { return std::tie(du, dv); }
  auto as_tuple() const { return std::tie(du, dv); }
};
// end snippet increments

// start snippet equations
struct Model {
  Parameters parameters;
  OPENPFC_HD Increments rhs(double /*stage_time*/, const Local &g) const noexcept {
    const auto p = parameters;
    return {g.v.value + p.a * g.u.xy,
            p.b * (g.u.xx + g.u.yy + g.u.zz) + p.c * g.v.xz + p.d * g.u.yz};
  }
};
// end snippet equations

struct Position {
  double x{}, y{}, z{};
};
// Default coefficients admit this exact homogeneous periodic solution.
// u=cos(t)cos(x+2y+3z), v=(-sin(t)+.6cos(t))cos(x+2y+3z).
// The golden gradients are analytic, independent of any evaluator.
inline Local exact_local(double t, Position x) {
  const double mode = std::cos(x.x + 2 * x.y + 3 * x.z);
  const double u = std::cos(t) * mode;
  const double v = (-std::sin(t) + .6 * std::cos(t)) * mode;
  return {{-u, -4 * u, -9 * u, -2 * u, -6 * u}, {v, -3 * v}};
}
inline Increments exact_rhs(double t, Position x) {
  const double mode = std::cos(x.x + 2 * x.y + 3 * x.z);
  return {-std::sin(t) * mode, (-std::cos(t) - .6 * std::sin(t)) * mode};
}
inline double exact_u(double t, Position x) {
  return std::cos(t) * std::cos(x.x + 2 * x.y + 3 * x.z);
}
inline double exact_v(double t, Position x) {
  return (-std::sin(t) + .6 * std::cos(t)) * std::cos(x.x + 2 * x.y + 3 * x.z);
}
} // namespace coupled
