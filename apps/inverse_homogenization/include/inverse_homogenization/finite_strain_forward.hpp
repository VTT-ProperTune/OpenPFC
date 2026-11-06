// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file finite_strain_forward.hpp
 * @brief 2-D plane-strain finite-strain forward ladder (OpenPFC #55).
 *
 * Frozen kinematics for the forward gate:
 * - in-plane deformation gradient \f$F\f$ with \f$F_{33}=1\f$;
 * - first Piola--Kirchhoff stress \f$P=\partial W/\partial F\f$;
 * - logarithmic strain \f$\varepsilon_i=\ln F_{ii}\f$ on a principal path;
 * - tangent Poisson
 *   \f$\nu_t=-d\varepsilon_y/d\varepsilon_x=-\mathrm{d}\ln F_{22}/\mathrm{d}\ln
 * F_{11}\f$.
 *
 * Uniaxial loading prescribes \f$F_{11}\f$ and relaxes \f$F_{22}\f$ until
 * \f$P_{22}=0\f$. This header does **not** call the small-strain FFT
 * homogenizer or reuse \f$C_H\f$.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfc::apps::inverse::fs {

struct Lame {
  double lambda{0.0};
  double mu{0.0};
};

[[nodiscard]] inline Lame lame_from_young_poisson(double E, double nu) {
  if (!(E > 0.0) || nu >= 0.5 || nu <= -1.0) {
    throw std::invalid_argument("finite-strain Lamé: need E>0 and -1<nu<1/2");
  }
  const double den = (1.0 + nu) * (1.0 - 2.0 * nu);
  return Lame{E * nu / den, E / (2.0 * (1.0 + nu))};
}

[[nodiscard]] inline double plane_strain_small_nu(const Lame &lame) {
  return lame.lambda / (lame.lambda + 2.0 * lame.mu);
}

enum class Model { NeoHookean, StVenantKirchhoff };

[[nodiscard]] inline const char *model_name(Model m) {
  return m == Model::NeoHookean ? "neo-hookean" : "stvk";
}

struct Mat2 {
  double a11{1.0}, a12{0.0}, a21{0.0}, a22{1.0};
};

[[nodiscard]] inline Mat2 diag2(double a, double b) { return Mat2{a, 0.0, 0.0, b}; }

[[nodiscard]] inline double det(const Mat2 &A) {
  return A.a11 * A.a22 - A.a12 * A.a21;
}

[[nodiscard]] inline Mat2 inverse_t(const Mat2 &A) {
  const double J = det(A);
  if (!(std::abs(J) > 1e-18)) {
    throw std::runtime_error("finite-strain: singular F");
  }
  // F^{-T}
  return Mat2{A.a22 / J, -A.a21 / J, -A.a12 / J, A.a11 / J};
}

struct FirstPK {
  Mat2 P{};
  double W{0.0};
  double J{1.0};
  double I1{3.0};
};

[[nodiscard]] inline FirstPK neo_hookean_pk(const Lame &lame, const Mat2 &F) {
  const double J = det(F);
  if (!(J > 1e-18)) {
    throw std::runtime_error("neo-Hookean: J<=0");
  }
  const double I1 =
      F.a11 * F.a11 + F.a12 * F.a12 + F.a21 * F.a21 + F.a22 * F.a22 + 1.0;
  const double lnJ = std::log(J);
  const Mat2 FinvT = inverse_t(F);
  const double coef = lame.lambda * lnJ - lame.mu;
  FirstPK out;
  out.J = J;
  out.I1 = I1;
  out.W = 0.5 * lame.mu * (I1 - 3.0) - lame.mu * lnJ + 0.5 * lame.lambda * lnJ * lnJ;
  out.P.a11 = lame.mu * F.a11 + coef * FinvT.a11;
  out.P.a12 = lame.mu * F.a12 + coef * FinvT.a12;
  out.P.a21 = lame.mu * F.a21 + coef * FinvT.a21;
  out.P.a22 = lame.mu * F.a22 + coef * FinvT.a22;
  return out;
}

[[nodiscard]] inline FirstPK stvk_pk(const Lame &lame, const Mat2 &F) {
  // C = F^T F (in-plane); C33=1, E33=0.
  const double C11 = F.a11 * F.a11 + F.a21 * F.a21;
  const double C12 = F.a11 * F.a12 + F.a21 * F.a22;
  const double C22 = F.a12 * F.a12 + F.a22 * F.a22;
  const double E11 = 0.5 * (C11 - 1.0);
  const double E12 = 0.5 * C12;
  const double E22 = 0.5 * (C22 - 1.0);
  const double trE = E11 + E22;
  const double S11 = lame.lambda * trE + 2.0 * lame.mu * E11;
  const double S12 = 2.0 * lame.mu * E12;
  const double S22 = lame.lambda * trE + 2.0 * lame.mu * E22;
  FirstPK out;
  out.J = det(F);
  out.I1 = C11 + C22 + 1.0;
  out.W = 0.5 * lame.lambda * trE * trE +
          lame.mu * (E11 * E11 + 2.0 * E12 * E12 + E22 * E22);
  // P = F S
  out.P.a11 = F.a11 * S11 + F.a12 * S12;
  out.P.a12 = F.a11 * S12 + F.a12 * S22;
  out.P.a21 = F.a21 * S11 + F.a22 * S12;
  out.P.a22 = F.a21 * S12 + F.a22 * S22;
  return out;
}

[[nodiscard]] inline FirstPK first_pk(Model model, const Lame &lame, const Mat2 &F) {
  return model == Model::NeoHookean ? neo_hookean_pk(lame, F) : stvk_pk(lame, F);
}

struct UniaxialState {
  double F11{1.0};
  double F22{1.0};
  double P11{0.0};
  double P22{0.0};
  double J{1.0};
  double W{0.0};
  int newton_iters{0};
  double residual{0.0};
  bool converged{false};
  bool stable{false};
};

[[nodiscard]] inline UniaxialState relax_transverse(Model model, const Lame &lame,
                                                    double F11, int max_iter = 50,
                                                    double tol = 1e-12) {
  if (!(F11 > 1e-8)) {
    throw std::invalid_argument("relax_transverse: F11 must be positive");
  }
  UniaxialState s;
  s.F11 = F11;
  double F22 = 1.0 / std::sqrt(F11);
  const double fd = 1e-8;
  for (int it = 0; it < max_iter; ++it) {
    const FirstPK pk = first_pk(model, lame, diag2(F11, F22));
    const double r = pk.P.a22;
    s.newton_iters = it + 1;
    s.residual = std::abs(r);
    s.J = pk.J;
    s.W = pk.W;
    s.P11 = pk.P.a11;
    s.P22 = pk.P.a22;
    if (s.residual < tol && pk.J > 1e-12) {
      s.F22 = F22;
      s.converged = true;
      s.stable = pk.J > 0.0 && std::isfinite(pk.W) && std::isfinite(r);
      return s;
    }
    const FirstPK pk_fd =
        first_pk(model, lame, diag2(F11, F22 + fd * std::max(1.0, F22)));
    const double dP = (pk_fd.P.a22 - r) / (fd * std::max(1.0, F22));
    if (!(std::abs(dP) > 1e-18)) {
      break;
    }
    double step = r / dP;
    step = std::clamp(step, -0.5 * F22, 0.5 * F22);
    F22 -= step;
    if (!(F22 > 1e-8)) F22 = 1e-8;
  }
  const FirstPK pk = first_pk(model, lame, diag2(F11, F22));
  s.F22 = F22;
  s.P11 = pk.P.a11;
  s.P22 = pk.P.a22;
  s.J = pk.J;
  s.W = pk.W;
  s.residual = std::abs(pk.P.a22);
  s.converged = s.residual < tol && pk.J > 1e-12;
  s.stable = pk.J > 0.0 && std::isfinite(pk.W);
  return s;
}

[[nodiscard]] inline double tangent_poisson_log(double F11_a, double F22_a,
                                                double F11_b, double F22_b) {
  const double dEx = std::log(F11_b) - std::log(F11_a);
  if (!(std::abs(dEx) > 1e-18)) {
    throw std::invalid_argument("tangent_poisson_log: F11 samples coincide");
  }
  const double dEy = std::log(F22_b) - std::log(F22_a);
  return -dEy / dEx;
}

[[nodiscard]] inline double tangent_poisson_at(Model model, const Lame &lame,
                                               double F11, double rel_step = 1e-5) {
  const double F11m = F11 * (1.0 - rel_step);
  const double F11p = F11 * (1.0 + rel_step);
  const auto sm = relax_transverse(model, lame, F11m);
  const auto sp = relax_transverse(model, lame, F11p);
  if (!sm.converged || !sp.converged) {
    throw std::runtime_error("tangent_poisson_at: relaxation failed");
  }
  return tangent_poisson_log(sm.F11, sm.F22, sp.F11, sp.F22);
}

struct LaminateState {
  UniaxialState phase_a{};
  UniaxialState phase_b{};
  double volume_a{0.5};
  double F11{1.0};
  double F22{1.0};
  double P11{0.0};
  double P22{0.0};
  bool converged{false};
  bool stable{false};
};

[[nodiscard]] inline LaminateState laminate_uniaxial(Model model, const Lame &a,
                                                     const Lame &b, double volume_a,
                                                     double F11) {
  if (!(volume_a >= 0.0 && volume_a <= 1.0)) {
    throw std::invalid_argument("laminate_uniaxial: volume_a in [0,1]");
  }
  LaminateState s;
  s.volume_a = volume_a;
  s.F11 = F11;
  s.phase_a = relax_transverse(model, a, F11);
  s.phase_b = relax_transverse(model, b, F11);
  const double vb = 1.0 - volume_a;
  s.F22 = volume_a * s.phase_a.F22 + vb * s.phase_b.F22;
  s.P11 = volume_a * s.phase_a.P11 + vb * s.phase_b.P11;
  s.P22 = volume_a * s.phase_a.P22 + vb * s.phase_b.P22;
  s.converged = s.phase_a.converged && s.phase_b.converged;
  s.stable = s.phase_a.stable && s.phase_b.stable && s.F22 > 1e-12;
  return s;
}

struct LadderRung {
  std::string kind;
  Model model{Model::NeoHookean};
  double F11{1.0};
  double F22{1.0};
  double P11{0.0};
  double P22{0.0};
  double nu_t{0.0};
  double residual{0.0};
  int newton_iters{0};
  bool converged{false};
  bool stable{false};
};

struct LadderReport {
  Lame lame_stiff{};
  Lame lame_compliant{};
  double small_strain_nu{0.0};
  std::vector<LadderRung> rungs;
  bool all_converged{false};
  bool all_stable{false};
};

[[nodiscard]] inline LadderReport run_declared_ladder() {
  LadderReport report;
  report.lame_stiff = lame_from_young_poisson(1.0, 0.3);
  report.lame_compliant = lame_from_young_poisson(0.1, 0.1);
  report.small_strain_nu = plane_strain_small_nu(report.lame_stiff);
  const double F11s[] = {1.001, 1.05, 1.10, 1.20};
  const Model models[] = {Model::NeoHookean, Model::StVenantKirchhoff};
  report.all_converged = true;
  report.all_stable = true;
  auto push = [&](LadderRung r) {
    report.all_converged = report.all_converged && r.converged;
    report.all_stable = report.all_stable && r.stable;
    report.rungs.push_back(std::move(r));
  };
  for (Model model : models) {
    for (double F11 : F11s) {
      const auto u = relax_transverse(model, report.lame_stiff, F11);
      LadderRung r;
      r.kind = std::string("homogeneous-") + model_name(model);
      r.model = model;
      r.F11 = u.F11;
      r.F22 = u.F22;
      r.P11 = u.P11;
      r.P22 = u.P22;
      r.residual = u.residual;
      r.newton_iters = u.newton_iters;
      r.converged = u.converged;
      r.stable = u.stable;
      r.nu_t = tangent_poisson_at(model, report.lame_stiff, F11);
      push(std::move(r));
    }
    for (double F11 : F11s) {
      const auto lam = laminate_uniaxial(model, report.lame_stiff,
                                         report.lame_compliant, 0.5, F11);
      const auto lm = laminate_uniaxial(
          model, report.lame_stiff, report.lame_compliant, 0.5, F11 * (1.0 - 1e-5));
      const auto lp = laminate_uniaxial(
          model, report.lame_stiff, report.lame_compliant, 0.5, F11 * (1.0 + 1e-5));
      LadderRung r;
      r.kind = std::string("laminate-") + model_name(model);
      r.model = model;
      r.F11 = lam.F11;
      r.F22 = lam.F22;
      r.P11 = lam.P11;
      r.P22 = lam.P22;
      r.residual = std::max(lam.phase_a.residual, lam.phase_b.residual);
      r.newton_iters = std::max(lam.phase_a.newton_iters, lam.phase_b.newton_iters);
      r.converged = lam.converged && lm.converged && lp.converged;
      r.stable = lam.stable && lm.stable && lp.stable;
      r.nu_t = tangent_poisson_log(lm.F11, lm.F22, lp.F11, lp.F22);
      push(std::move(r));
    }
  }
  return report;
}

} // namespace pfc::apps::inverse::fs
