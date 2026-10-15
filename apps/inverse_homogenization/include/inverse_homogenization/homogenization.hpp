// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file homogenization.hpp
 * @brief Application view of periodic homogenization (issue #161), plus the
 *        analytic oracles the inverse-design driver and its tests use.
 *
 * @details
 * The homogenizer, its self-adjoint sensitivity, `Voigt6` and the unit
 * loads are library code in
 * `openpfc/solvers/microelasticity/homogenization.hpp`; this header names
 * them in `pfc::apps` and adds what only this application needs: SPD and
 * eigenvalue diagnostics of \f$C_H\f$, the Voigt and Reuss bounds, and the
 * exact binary laminate (Postma) used as test oracles.
 *
 * @see Postma, *Geophysics* **20**, 780 (1955)
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <openpfc/solvers/microelasticity/homogenization.hpp>
#include <openpfc/solvers/microelasticity/microelasticity.hpp>

namespace pfc::apps {

using pfc::solvers::EigenstrainMicroelasticity;
using pfc::solvers::engineering_unit_strain;
using pfc::solvers::frobenius_inner;
using pfc::solvers::hadamard;
using pfc::solvers::HomogenizationResult;
using pfc::solvers::invert_voigt;
using pfc::solvers::inverted;
using pfc::solvers::kVoigtDim;
using pfc::solvers::MicroelasticityParams;
using pfc::solvers::MicroelasticityReport;
using pfc::solvers::Stiffness;
using pfc::solvers::Sym3;
using pfc::solvers::tensor_mismatch;
using pfc::solvers::Voigt6;
using pfc::solvers::voigt_from_stiffness;

/// The host homogenizer used by the CPU drivers.
using PeriodicHomogenizer =
    pfc::solvers::PeriodicHomogenizer<pfc::solvers::EigenstrainMicroelasticity>;

/**
 * @brief Cholesky test that a symmetrized matrix is SPD.
 *
 * Used as a diagnostic on \f$C_H\f$, not as a solver.
 */
[[nodiscard]] inline bool is_spd(const Voigt6 &in, double abs_tol = 0.0) {
  Voigt6 a = in.symmetrized();
  for (int i = 0; i < kVoigtDim; ++i) {
    for (int j = 0; j <= i; ++j) {
      double s = a(i, j);
      for (int k = 0; k < j; ++k) s -= a(i, k) * a(j, k);
      if (i == j) {
        if (s <= abs_tol) return false;
        a(i, j) = std::sqrt(s);
      } else {
        a(i, j) = s / a(j, j);
      }
    }
  }
  return true;
}

/// Smallest eigenvalue of a symmetrized 6x6 matrix (Jacobi).
[[nodiscard]] inline double min_eigenvalue_symmetric(const Voigt6 &in) {
  Voigt6 a = in.symmetrized();
  constexpr int n = kVoigtDim;
  for (int iter = 0; iter < 64; ++iter) {
    int p = 0, q = 1;
    double maxv = 0.0;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        const double v = std::abs(a(i, j));
        if (v > maxv) {
          maxv = v;
          p = i;
          q = j;
        }
      }
    if (maxv < 1.0e-15) break;
    const double app = a(p, p);
    const double aqq = a(q, q);
    const double apq = a(p, q);
    const double tau = (aqq - app) / (2.0 * apq);
    const double t =
        std::copysign(1.0, tau) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    const double c = 1.0 / std::sqrt(1.0 + t * t);
    const double s = t * c;
    a(p, p) = app - t * apq;
    a(q, q) = aqq + t * apq;
    a(p, q) = a(q, p) = 0.0;
    for (int k = 0; k < n; ++k) {
      if (k == p || k == q) continue;
      const double aik = a(k, p);
      const double aiq = a(k, q);
      a(k, p) = a(p, k) = c * aik - s * aiq;
      a(k, q) = a(q, k) = s * aik + c * aiq;
    }
  }
  double m = a(0, 0);
  for (int i = 1; i < n; ++i) m = std::min(m, a(i, i));
  return m;
}

/// Compliance Poisson ratios and cubic/isotropy spreads of an engineering C.
struct StiffnessDiagnostics {
  Voigt6 C{};
  Voigt6 S{};
  bool spd{false};
  bool invertible{false};
  double min_eig{0.0};
  double rel_frobenius{0.0};
  /// Shortcut \f$C_{12}/(C_{11}+C_{12})\f$; not the qualification metric.
  double nu_shortcut{0.0};
  double nu_xy{0.0};
  double nu_xz{0.0};
  double nu_yx{0.0};
  double nu_yz{0.0};
  double nu_zx{0.0};
  double nu_zy{0.0};
  double spread_C11{0.0};
  double spread_C12{0.0};
  double spread_C44{0.0};
};

[[nodiscard]] inline StiffnessDiagnostics
diagnose_stiffness(const Voigt6 &C, const Voigt6 *C_target = nullptr) {
  StiffnessDiagnostics d;
  d.C = C.symmetrized();
  d.spd = is_spd(d.C);
  d.min_eig = min_eigenvalue_symmetric(d.C);
  Voigt6 S = d.C;
  d.invertible = invert_voigt(S);
  if (d.invertible) {
    d.S = S;
    auto ratio = [](double num, double den) {
      return (std::abs(den) > 1.0e-30) ? num / den : 0.0;
    };
    d.nu_xy = ratio(-S(0, 1), S(0, 0));
    d.nu_xz = ratio(-S(0, 2), S(0, 0));
    d.nu_yx = ratio(-S(1, 0), S(1, 1));
    d.nu_yz = ratio(-S(1, 2), S(1, 1));
    d.nu_zx = ratio(-S(2, 0), S(2, 2));
    d.nu_zy = ratio(-S(2, 1), S(2, 2));
  }
  const double den = d.C(0, 0) + d.C(0, 1);
  d.nu_shortcut = (std::abs(den) > 1.0e-30) ? d.C(0, 1) / den : 0.0;
  auto spread = [](double a, double b, double c) {
    return std::max(a, std::max(b, c)) - std::min(a, std::min(b, c));
  };
  d.spread_C11 = spread(d.C(0, 0), d.C(1, 1), d.C(2, 2));
  d.spread_C12 = spread(d.C(0, 1), d.C(0, 2), d.C(1, 2));
  d.spread_C44 = spread(d.C(3, 3), d.C(4, 4), d.C(5, 5));
  if (C_target != nullptr) {
    d.rel_frobenius = (d.C - *C_target).frobenius_norm() /
                      std::max(C_target->frobenius_norm(), 1.0e-30);
  }
  return d;
}

/**
 * @brief Voigt (arithmetic) bound \f$f C_1+(1-f)C_2\f$.
 *
 * An upper bound on the effective tensor in the sense of the elastic energy;
 * for a laminate it is exact on the iso-strain block.
 */
[[nodiscard]] inline Voigt6 voigt_bound(const Stiffness &c1, const Stiffness &c2,
                                        double f1) noexcept {
  return voigt_from_stiffness(Stiffness::blend(c1, f1, c2, 1.0 - f1));
}

/**
 * @brief Reuss (harmonic) bound \f$(f S_1+(1-f)S_2)^{-1}\f$.
 *
 * Exact on the iso-stress block of a laminate.
 */
[[nodiscard]] inline Voigt6 reuss_bound(const Stiffness &c1, const Stiffness &c2,
                                        double f1) {
  const Voigt6 s1 = inverted(voigt_from_stiffness(c1));
  const Voigt6 s2 = inverted(voigt_from_stiffness(c2));
  return inverted(s1 * f1 + s2 * (1.0 - f1));
}

/**
 * @brief Exact effective stiffness of a binary cubic laminate.
 *
 * @param c1,c2   cubic stiffnesses, crystal axes along the grid
 * @param f1      volume fraction of phase 1
 * @param axis    layer normal, 0/1/2 for \f$x/y/z\f$
 *
 * Formula: iso-strain in the interface plane, iso-stress on the normal
 * traction (Postma 1955 / Backus averages, written in cubic constants).
 * The in-plane shear modulus is the arithmetic mean of \f$c_{44}\f$; the two
 * out-of-plane shears are the harmonic mean.
 */
[[nodiscard]] inline Voigt6 exact_binary_laminate(const Stiffness &c1,
                                                  const Stiffness &c2, double f1,
                                                  int axis) {
  if (axis < 0 || axis > 2) {
    throw std::invalid_argument("exact_binary_laminate: axis must be 0, 1 or 2");
  }
  if (!(f1 >= 0.0 && f1 <= 1.0)) {
    throw std::invalid_argument(
        "exact_binary_laminate: volume fraction must lie in [0, 1]");
  }
  const double f2 = 1.0 - f1;
  const auto avg = [&](double a, double b) { return f1 * a + f2 * b; };
  const double inv_c11 = avg(1.0 / c1.c11, 1.0 / c2.c11);
  const double c12_over = avg(c1.c12 / c1.c11, c2.c12 / c2.c11);
  const double p11 =
      avg(c1.c11 - c1.c12 * c1.c12 / c1.c11, c2.c11 - c2.c12 * c2.c12 / c2.c11);
  const double p12 =
      avg(c1.c12 - c1.c12 * c1.c12 / c1.c11, c2.c12 - c2.c12 * c2.c12 / c2.c11);
  const double c44_a = avg(c1.c44, c2.c44);
  const double inv_c44 = avg(1.0 / c1.c44, 1.0 / c2.c44);
  const double Cnn = 1.0 / inv_c11;
  const double Cnp = c12_over / inv_c11;
  const double Cpp = p11 + c12_over * c12_over / inv_c11;
  const double Cpq = p12 + c12_over * c12_over / inv_c11;
  const double mu_n = 1.0 / inv_c44; // harmonic, out-of-plane shear
  const double mu_p = c44_a;         // arithmetic, in-plane shear

  Voigt6 C;
  const int n = axis;
  const int p = (axis + 1) % 3;
  const int q = (axis + 2) % 3;
  C(n, n) = Cnn;
  C(n, p) = C(p, n) = C(n, q) = C(q, n) = Cnp;
  C(p, p) = C(q, q) = Cpp;
  C(p, q) = C(q, p) = Cpq;
  // yz (3) is out-of-plane unless the normal is x; xz (4) unless y; xy (5)
  // unless z. Out-of-plane shears take the harmonic mean of c44.
  C(3, 3) = (axis == 0) ? mu_p : mu_n;
  C(4, 4) = (axis == 1) ? mu_p : mu_n;
  C(5, 5) = (axis == 2) ? mu_p : mu_n;
  return C;
}

/**
 * @brief Exact 1-D homogenization of a *graded* laminate
 * \f$h=h(x_{\mathrm{axis}})\f$.
 *
 * Same algebra as `exact_binary_laminate`, but the averages are taken over
 * the actual cellwise \f$\mathbf C(h_i)\f$ rather than two phases at
 * \f$\langle h\rangle\f$. Needed for a tanh interface, where
 * \f$\mathbf C(h)\f$ is a nonlinear function of \f$h\f$ in the combinations
 * \f$1/c_{11}\f$ and \f$c_{12}/c_{11}\f$.
 */
[[nodiscard]] inline Voigt6
exact_laminate_from_field(const pfc::data::Field<double> &h, const Stiffness &cs,
                          const Stiffness &cl, int axis, MPI_Comm comm) {
  if (axis < 0 || axis > 2) {
    throw std::invalid_argument("exact_laminate_from_field: axis must be 0, 1 or 2");
  }
  double loc[6] = {0, 0, 0, 0, 0, 0};
  const std::size_t n = h.size();
  const double *hp = h.data();
  for (std::size_t i = 0; i < n; ++i) {
    const Stiffness c = Stiffness::blend(cs, hp[i], cl, 1.0 - hp[i]);
    loc[0] += 1.0 / c.c11;
    loc[1] += c.c12 / c.c11;
    loc[2] += c.c11 - c.c12 * c.c12 / c.c11;
    loc[3] += c.c12 - c.c12 * c.c12 / c.c11;
    loc[4] += c.c44;
    loc[5] += 1.0 / c.c44;
  }
  double glo[6] = {0, 0, 0, 0, 0, 0};
  MPI_Allreduce(loc, glo, 6, MPI_DOUBLE, MPI_SUM, comm);
  const auto gs = h.global_size();
  const double N = static_cast<double>(gs[0]) * static_cast<double>(gs[1]) *
                   static_cast<double>(gs[2]);
  for (double &v : glo) v /= N;

  const double inv_c11 = glo[0];
  const double c12_over = glo[1];
  const double p11 = glo[2];
  const double p12 = glo[3];
  const double c44_a = glo[4];
  const double inv_c44 = glo[5];
  const double Cnn = 1.0 / inv_c11;
  const double Cnp = c12_over / inv_c11;
  const double Cpp = p11 + c12_over * c12_over / inv_c11;
  const double Cpq = p12 + c12_over * c12_over / inv_c11;
  const double mu_n = 1.0 / inv_c44;
  const double mu_p = c44_a;

  Voigt6 C;
  const int nn = axis;
  const int p = (axis + 1) % 3;
  const int q = (axis + 2) % 3;
  C(nn, nn) = Cnn;
  C(nn, p) = C(p, nn) = C(nn, q) = C(q, nn) = Cnp;
  C(p, p) = C(q, q) = Cpp;
  C(p, q) = C(q, p) = Cpq;
  C(3, 3) = (axis == 0) ? mu_p : mu_n;
  C(4, 4) = (axis == 1) ? mu_p : mu_n;
  C(5, 5) = (axis == 2) ? mu_p : mu_n;
  return C;
}

} // namespace pfc::apps
