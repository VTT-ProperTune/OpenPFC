// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cmath>
#include <openpfc/kernel/data/host_device.hpp>
#include <tuple>
namespace nfield_rk {
struct Value {
  double value{};
};
struct Local3 {
  Value u, v, w;
};
struct Local4 {
  Value u, v, w, q;
};
struct Inc3 {
  double du{}, dv{}, dw{};
  auto as_tuple() { return std::tie(du, dv, dw); }
  auto as_tuple() const { return std::tie(du, dv, dw); }
};
struct Inc4 {
  double du{}, dv{}, dw{}, dq{};
  auto as_tuple() { return std::tie(du, dv, dw, dq); }
  auto as_tuple() const { return std::tie(du, dv, dw, dq); }
};
struct Model3 {
  OPENPFC_HD Inc3 rhs(double t, const Local3 &g) const noexcept {
    return {g.v.value + 0.1 * t, g.w.value, -g.u.value};
  }
};
struct Model4 {
  OPENPFC_HD Inc4 rhs(double t, const Local4 &g) const noexcept {
    return {g.v.value + 0.1 * t, g.w.value, g.q.value, -g.u.value};
  }
};
inline void rk4_triple(double t, double dt, double &u, double &v, double &w) {
  Model3 m;
  auto stage = [&](double ts, double uu, double vv, double ww) {
    Local3 g{{uu}, {vv}, {ww}};
    return m.rhs(ts, g);
  };
  const auto k1 = stage(t, u, v, w);
  const auto k2 = stage(t + 0.5 * dt, u + 0.5 * dt * k1.du, v + 0.5 * dt * k1.dv,
                        w + 0.5 * dt * k1.dw);
  const auto k3 = stage(t + 0.5 * dt, u + 0.5 * dt * k2.du, v + 0.5 * dt * k2.dv,
                        w + 0.5 * dt * k2.dw);
  const auto k4 = stage(t + dt, u + dt * k3.du, v + dt * k3.dv, w + dt * k3.dw);
  u += dt * (k1.du + 2 * k2.du + 2 * k3.du + k4.du) / 6.0;
  v += dt * (k1.dv + 2 * k2.dv + 2 * k3.dv + k4.dv) / 6.0;
  w += dt * (k1.dw + 2 * k2.dw + 2 * k3.dw + k4.dw) / 6.0;
}
inline void rk4_quad(double t, double dt, double &u, double &v, double &w,
                     double &q) {
  Model4 m;
  auto stage = [&](double ts, double uu, double vv, double ww, double qq) {
    Local4 g{{uu}, {vv}, {ww}, {qq}};
    return m.rhs(ts, g);
  };
  const auto k1 = stage(t, u, v, w, q);
  const auto k2 = stage(t + 0.5 * dt, u + 0.5 * dt * k1.du, v + 0.5 * dt * k1.dv,
                        w + 0.5 * dt * k1.dw, q + 0.5 * dt * k1.dq);
  const auto k3 = stage(t + 0.5 * dt, u + 0.5 * dt * k2.du, v + 0.5 * dt * k2.dv,
                        w + 0.5 * dt * k2.dw, q + 0.5 * dt * k2.dq);
  const auto k4 =
      stage(t + dt, u + dt * k3.du, v + dt * k3.dv, w + dt * k3.dw, q + dt * k3.dq);
  u += dt * (k1.du + 2 * k2.du + 2 * k3.du + k4.du) / 6.0;
  v += dt * (k1.dv + 2 * k2.dv + 2 * k3.dv + k4.dv) / 6.0;
  w += dt * (k1.dw + 2 * k2.dw + 2 * k3.dw + k4.dw) / 6.0;
  q += dt * (k1.dq + 2 * k2.dq + 2 * k3.dq + k4.dq) / 6.0;
}
} // namespace nfield_rk
