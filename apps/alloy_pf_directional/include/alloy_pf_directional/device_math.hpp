// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file device_math.hpp
 * @brief Host/device FTA physics and Ji isotropic operators (2D and 3D).
 *
 * CPU backends include this for a single formula source; HIP kernels do too.
 * Classic AMR is not implemented here — see defaults.hpp and the app README.
 */

#include <cmath>

#ifndef ALCU_HD
#if defined(__HIPCC__) || defined(__CUDACC__)
#define ALCU_HD __host__ __device__
#else
#define ALCU_HD
#endif
#endif

namespace alloy_pf_directional::dmath {

ALCU_HD inline double antitrapping_a() noexcept { return 1.0 / (2.0 * std::sqrt(2.0)); }
ALCU_HD inline double f_prime(double phi) noexcept { return phi * phi * phi - phi; }
ALCU_HD inline double g_prime(double phi) noexcept {
  const double one_m_phi2 = 1.0 - phi * phi;
  return one_m_phi2 * one_m_phi2;
}
ALCU_HD inline double h_of(double phi) noexcept { return phi; }
ALCU_HD inline double q_of(double phi, double k) noexcept {
  const double h = h_of(phi);
  return (1.0 - phi) / (1.0 + k - (1.0 - k) * h);
}
ALCU_HD inline double denom_c(double phi, double k) noexcept {
  return 1.0 + k - (1.0 - k) * h_of(phi);
}
ALCU_HD inline double c_eq(double phi, double k, double clo) noexcept {
  return 0.5 * clo * denom_c(phi, k);
}
ALCU_HD inline double eu_from_phi_c(double phi, double c, double ke, double clo) noexcept {
  double ci = c;
  if (ci < 1.0e-16 && ci > -1.0e6) {
    ci = 1.0e-16;
  }
  const double den = denom_c(phi, ke);
  return (2.0 * ci / clo) / (den > 1.0e-12 ? den : 1.0e-12);
}
ALCU_HD inline double u_from_eu(double euv) noexcept {
  return std::log(euv > 1.0e-30 ? euv : 1.0e-30);
}
ALCU_HD inline double beta_glasner_from_phi(double ph) noexcept {
  const double p = ph < -0.999999 ? -0.999999 : (ph > 0.999999 ? 0.999999 : ph);
  return std::sqrt(2.0) * std::atanh(p);
}
ALCU_HD inline double a_at_nodal(double ph, double euv, double dte, double a_at,
                                 double A_trap, double W0, double clo, double ke,
                                 bool glasner) noexcept {
  const double pref = a_prime_trap(ph, a_at, A_trap) * W0 * clo * (1.0 - ke) * euv * dte;
  if (glasner) {
    return pref * W0;
  }
  const double om = (1.0 - ph * ph) > 1.0e-8 ? (1.0 - ph * ph) : 1.0e-8;
  return pref * std::sqrt(2.0) * W0 / om;
}
ALCU_HD inline double u_corr_from_eu(double eu) noexcept {
  return eu > 0.2 ? eu : 0.2;
}
ALCU_HD inline double phi_hat(double phi) noexcept { return 0.5 * (1.0 + phi); }
ALCU_HD inline double phi_eff_two(double p1, double p2) noexcept {
  const double s = p1 + p2 + 1.0;
  return s < -1.0 ? -1.0 : (s > 1.0 ? 1.0 : s);
}
ALCU_HD inline double grain_repulsion(double phi_self, double phi_other,
                                     double omega) noexcept {
  const double ho = phi_hat(phi_other);
  return -omega * phi_hat(phi_self) * ho * ho;
}
ALCU_HD inline double omega_zhong_fta(double lambda, double ke, double therm) noexcept {
  const double den = 1.0 - ke;
  if (!(den > 0.0) || !(therm > 0.0)) {
    return 0.0;
  }
  return (32.0 / 5.0) * (lambda / den) * therm;
}
ALCU_HD inline double omega_used(bool omega_zhong, double omega_const, double lambda,
                                double ke, double therm) noexcept {
  if (!omega_zhong) {
    return omega_const;
  }
  return omega_zhong_fta(lambda, ke, therm);
}
ALCU_HD inline double a_prime_trap(double phi, double a_at, double A_trap) noexcept {
  return a_at * (1.0 - A_trap * (1.0 - phi * phi));
}
ALCU_HD inline double phi_from_psi(double psi) noexcept {
  return std::tanh(psi / std::sqrt(2.0));
}
ALCU_HD inline double dphi_dpsi_from_phi(double phi) noexcept {
  return (1.0 - phi * phi) / std::sqrt(2.0);
}
ALCU_HD inline double grain_dpsi_dt(double grain, double tau, double phi, double dt) noexcept {
  const double dps = dphi_dpsi_from_phi(phi);
  const double den = (dps > 1.0e-12 ? dps : 1.0e-12) * tau;
  double g = grain / den;
  if (dt > 0.0) {
    const double cap = 2.0 / dt;
    g = g > cap ? cap : (g < -cap ? -cap : g);
  }
  return g;
}

ALCU_HD inline void cubic_aniso_from_grad(double gx, double gy, double gz, double eps_c,
                                          double eps_k, double W0, double tau0,
                                          const double R_c2l[9], double &jx, double &jy,
                                          double &jz, double &tau, double &A) noexcept {
  const double q2 = gx * gx + gy * gy + gz * gz;
  if (q2 < 1.0e-30) {
    jx = 0.0;
    jy = 0.0;
    jz = 0.0;
    tau = tau0;
    A = 1.0;
    return;
  }
  const double q = std::sqrt(q2);
  const double n0 = gx / q;
  const double n1 = gy / q;
  const double n2 = gz / q;
  // n_c = R^T n_lab
  const double nc0 = R_c2l[0] * n0 + R_c2l[3] * n1 + R_c2l[6] * n2;
  const double nc1 = R_c2l[1] * n0 + R_c2l[4] * n1 + R_c2l[7] * n2;
  const double nc2 = R_c2l[2] * n0 + R_c2l[5] * n1 + R_c2l[8] * n2;
  const double n4 = nc0 * nc0 * nc0 * nc0 + nc1 * nc1 * nc1 * nc1 + nc2 * nc2 * nc2 * nc2;
  A = 1.0 - 3.0 * eps_c + 4.0 * eps_c * n4;
  const double dAc0 = 16.0 * eps_c * nc0 * nc0 * nc0;
  const double dAc1 = 16.0 * eps_c * nc1 * nc1 * nc1;
  const double dAc2 = 16.0 * eps_c * nc2 * nc2 * nc2;
  const double dA0 = R_c2l[0] * dAc0 + R_c2l[1] * dAc1 + R_c2l[2] * dAc2;
  const double dA1 = R_c2l[3] * dAc0 + R_c2l[4] * dAc1 + R_c2l[5] * dAc2;
  const double dA2 = R_c2l[6] * dAc0 + R_c2l[7] * dAc1 + R_c2l[8] * dAc2;
  const double ndA = n0 * dA0 + n1 * dA1 + n2 * dA2;
  const double pref = W0 * W0 * A * q;
  jx = pref * (A * n0 + dA0 - n0 * ndA);
  jy = pref * (A * n1 + dA1 - n1 * ndA);
  jz = pref * (A * n2 + dA2 - n2 * ndA);
  const double a_k = 1.0 - 3.0 * eps_k + 4.0 * eps_k * n4;
  tau = tau0 * A * A * a_k;
}

ALCU_HD inline double tau_with_u_corr(double tau_as_U0, double tau_beta, double tau_a2,
                                      double tau0, double eu) noexcept {
  return tau_as_U0 * (tau_beta + tau_a2 * u_corr_from_eu(eu)) / tau0;
}

ALCU_HD inline double thermal_drive(double G, double mle, double clo, double x_tl,
                                    double Vp, double x, double t, double ke,
                                    double delta_iso) noexcept {
  return (1.0 - ke) * delta_iso + G * (x - x_tl - Vp * t) / (mle * clo);
}

// ---- Ji isotropic operators (same weights as isotropic_fd.hpp) ----

template <class F>
ALCU_HD inline double laplacian_std(const F &f, int i, int j, int k, double h,
                                    bool dim3) noexcept {
  const double invh2 = 1.0 / (h * h);
  double s = f(i + 1, j, k) + f(i - 1, j, k) + f(i, j + 1, k) + f(i, j - 1, k);
  if (dim3) {
    s += f(i, j, k + 1) + f(i, j, k - 1);
    return (s - 6.0 * f(i, j, k)) * invh2;
  }
  return (s - 4.0 * f(i, j, k)) * invh2;
}

template <class F>
ALCU_HD inline double laplacian_iso(const F &f, int i, int j, int k, double h,
                                    bool dim3) noexcept {
  const double invh2 = 1.0 / (h * h);
  const double c = f(i, j, k);
  const double nn = f(i + 1, j, k) + f(i - 1, j, k) + f(i, j + 1, k) + f(i, j - 1, k);
  if (!dim3) {
    const double L10 = (nn - 4.0 * c) * invh2;
    const double diag = f(i + 1, j + 1, k) + f(i - 1, j + 1, k) + f(i + 1, j - 1, k) +
                        f(i - 1, j - 1, k);
    const double L01 = 0.5 * invh2 * (diag - 4.0 * c);
    return (2.0 * L10 + L01) / 3.0;
  }
  const double L100 = (nn + f(i, j, k + 1) + f(i, j, k - 1) - 6.0 * c) * invh2;
  const double e110 = f(i + 1, j + 1, k) + f(i + 1, j - 1, k) + f(i - 1, j + 1, k) +
                      f(i - 1, j - 1, k) + f(i + 1, j, k + 1) + f(i + 1, j, k - 1) +
                      f(i - 1, j, k + 1) + f(i - 1, j, k - 1) + f(i, j + 1, k + 1) +
                      f(i, j + 1, k - 1) + f(i, j - 1, k + 1) + f(i, j - 1, k - 1);
  const double L110 = 0.25 * invh2 * (e110 - 12.0 * c);
  return (L100 + 2.0 * L110) / 3.0;
}

template <class F>
ALCU_HD inline double grad2_iso(const F &f, int i, int j, int k, double h,
                                bool dim3) noexcept {
  const double invh2 = 1.0 / (h * h);
  const double dx = f(i + 1, j, k) - f(i - 1, j, k);
  const double dy = f(i, j + 1, k) - f(i, j - 1, k);
  if (!dim3) {
    const double G10 = 0.25 * invh2 * (dx * dx + dy * dy);
    const double d1 = f(i + 1, j + 1, k) - f(i - 1, j - 1, k);
    const double d2 = f(i + 1, j - 1, k) - f(i - 1, j + 1, k);
    const double G01 = 0.125 * invh2 * (d1 * d1 + d2 * d2);
    return (2.0 * G10 + G01) / 3.0;
  }
  const double dz = f(i, j, k + 1) - f(i, j, k - 1);
  const double G100 = 0.25 * invh2 * (dx * dx + dy * dy + dz * dz);
  const double a = f(i + 1, j + 1, k) - f(i - 1, j - 1, k);
  const double b = f(i + 1, j - 1, k) - f(i - 1, j + 1, k);
  const double c = f(i + 1, j, k + 1) - f(i - 1, j, k - 1);
  const double d = f(i - 1, j, k + 1) - f(i + 1, j, k - 1);
  const double e = f(i, j + 1, k + 1) - f(i, j - 1, k - 1);
  const double g = f(i, j - 1, k + 1) - f(i, j + 1, k - 1);
  const double G110 = (1.0 / 16.0) * invh2 * (a * a + b * b + c * c + d * d + e * e + g * g);
  return (G100 + 2.0 * G110) / 3.0;
}

template <class A, class B>
ALCU_HD inline double div_alpha_grad_2d(const A &al, const B &be, int i, int j, int k,
                                        double h) noexcept {
  const double invh = 1.0 / h;
  const double invh4 = 0.25 * invh;
  const double a00 = al(i, j, k);
  const double a10 = al(i + 1, j, k);
  const double a01 = al(i, j + 1, k);
  const double a11 = al(i + 1, j + 1, k);
  const double am10 = al(i - 1, j, k);
  const double a0m1 = al(i, j - 1, k);
  const double am11 = al(i - 1, j + 1, k);
  const double a1m1 = al(i + 1, j - 1, k);
  const double am1m1 = al(i - 1, j - 1, k);
  const double b00 = be(i, j, k);
  const double bar_pp = 0.25 * (a11 + a10 + a01 + a00);
  const double bar_pm = 0.25 * (a10 + a1m1 + a00 + a0m1);
  const double bar_mp = 0.25 * (a01 + a00 + am11 + am10);
  const double bar_mm = 0.25 * (a00 + a0m1 + am10 + am1m1);
  const double Fxp = invh4 * (a10 + a00 + bar_pp + bar_pm) * (be(i + 1, j, k) - b00);
  const double Fxm = invh4 * (a00 + am10 + bar_mp + bar_mm) * (be(i - 1, j, k) - b00);
  const double Fyp = invh4 * (a01 + a00 + bar_pp + bar_mp) * (be(i, j + 1, k) - b00);
  const double Fym = invh4 * (a00 + a0m1 + bar_pm + bar_mm) * (be(i, j - 1, k) - b00);
  const double D10 = (Fxp + Fxm + Fyp + Fym) * invh;
  const double invs2 = 1.0 / std::sqrt(2.0);
  const double s2 = 0.25 * invh * invs2;
  const double Fpp = s2 * (a11 + a00 + a01 + a10) * (be(i + 1, j + 1, k) - b00);
  const double Fmp = s2 * (am11 + a00 + a01 + am10) * (be(i - 1, j + 1, k) - b00);
  const double Fpm = s2 * (a1m1 + a00 + a10 + a0m1) * (be(i + 1, j - 1, k) - b00);
  const double Fmm = s2 * (am1m1 + a00 + a0m1 + am10) * (be(i - 1, j - 1, k) - b00);
  const double D01 = (Fpp + Fmp + Fpm + Fmm) * (invh * invs2);
  return (2.0 * D10 + D01) / 3.0;
}

template <class A>
ALCU_HD inline double bar_alpha_100_x(const A &al, int i, int j, int k, int di) noexcept {
  const int ip = i + di;
  const double nearest = 0.25 * (al(ip, j, k) + al(i, j, k));
  const double nn = (al(ip, j + 1, k) + al(i, j + 1, k) + al(i, j, k + 1) + al(ip, j, k + 1) +
                     al(ip, j - 1, k) + al(i, j - 1, k) + al(i, j, k - 1) + al(ip, j, k - 1)) /
                    16.0;
  return nearest + nn;
}
template <class A>
ALCU_HD inline double bar_alpha_100_y(const A &al, int i, int j, int k, int dj) noexcept {
  const int jp = j + dj;
  const double nearest = 0.25 * (al(i, jp, k) + al(i, j, k));
  const double nn = (al(i + 1, jp, k) + al(i + 1, j, k) + al(i, j, k + 1) + al(i, jp, k + 1) +
                     al(i - 1, jp, k) + al(i - 1, j, k) + al(i, j, k - 1) + al(i, jp, k - 1)) /
                    16.0;
  return nearest + nn;
}
template <class A>
ALCU_HD inline double bar_alpha_100_z(const A &al, int i, int j, int k, int dk) noexcept {
  const int kp = k + dk;
  const double nearest = 0.25 * (al(i, j, kp) + al(i, j, k));
  const double nn = (al(i + 1, j, kp) + al(i + 1, j, k) + al(i, j + 1, k) + al(i, j + 1, kp) +
                     al(i - 1, j, kp) + al(i - 1, j, k) + al(i, j - 1, k) + al(i, j - 1, kp)) /
                    16.0;
  return nearest + nn;
}
template <class A>
ALCU_HD inline double bar_alpha_110_xy(const A &al, int i, int j, int k, int di,
                                       int dj) noexcept {
  const int ip = i + di;
  const int jp = j + dj;
  const double nearest = (3.0 / 16.0) * (al(ip, j, k) + al(ip, jp, k) + al(i, jp, k) + al(i, j, k));
  const double nn = (al(ip, jp, k + 1) + al(ip, j, k + 1) + al(i, jp, k + 1) + al(i, j, k + 1) +
                     al(ip, jp, k - 1) + al(ip, j, k - 1) + al(i, jp, k - 1) + al(i, j, k - 1)) /
                    32.0;
  return nearest + nn;
}
template <class A>
ALCU_HD inline double bar_alpha_110_xz(const A &al, int i, int j, int k, int di,
                                       int dk) noexcept {
  const int ip = i + di;
  const int kp = k + dk;
  const double nearest = (3.0 / 16.0) * (al(ip, j, k) + al(ip, j, kp) + al(i, j, kp) + al(i, j, k));
  const double nn = (al(ip, j + 1, kp) + al(ip, j + 1, k) + al(i, j + 1, kp) + al(i, j + 1, k) +
                     al(ip, j - 1, kp) + al(ip, j - 1, k) + al(i, j - 1, kp) + al(i, j - 1, k)) /
                    32.0;
  return nearest + nn;
}
template <class A>
ALCU_HD inline double bar_alpha_110_yz(const A &al, int i, int j, int k, int dj,
                                       int dk) noexcept {
  const int jp = j + dj;
  const int kp = k + dk;
  const double nearest = (3.0 / 16.0) * (al(i, jp, k) + al(i, jp, kp) + al(i, j, kp) + al(i, j, k));
  const double nn = (al(i + 1, jp, kp) + al(i + 1, jp, k) + al(i + 1, j, kp) + al(i + 1, j, k) +
                     al(i - 1, jp, kp) + al(i - 1, jp, k) + al(i - 1, j, kp) + al(i - 1, j, k)) /
                    32.0;
  return nearest + nn;
}

struct Cube27 {
  double v[3][3][3];
  int ci, cj, ck;
  ALCU_HD double operator()(int i, int j, int k) const noexcept {
    return v[i - ci + 1][j - cj + 1][k - ck + 1];
  }
};

template <class A, class B>
ALCU_HD inline double div_alpha_grad_3d(const A &al, const B &be, int i, int j, int k,
                                        double h) noexcept {
  Cube27 A3{}, B3{};
  A3.ci = B3.ci = i;
  A3.cj = B3.cj = j;
  A3.ck = B3.ck = k;
  for (int dk = 0; dk < 3; ++dk) {
    for (int dj = 0; dj < 3; ++dj) {
      for (int di = 0; di < 3; ++di) {
        A3.v[di][dj][dk] = al(i + di - 1, j + dj - 1, k + dk - 1);
        B3.v[di][dj][dk] = be(i + di - 1, j + dj - 1, k + dk - 1);
      }
    }
  }
  const double b00 = B3(i, j, k);
  const double invh = 1.0 / h;
  const double Fxp = bar_alpha_100_x(A3, i, j, k, +1) * (B3(i + 1, j, k) - b00) * invh;
  const double Fxm = bar_alpha_100_x(A3, i, j, k, -1) * (B3(i - 1, j, k) - b00) * invh;
  const double Fyp = bar_alpha_100_y(A3, i, j, k, +1) * (B3(i, j + 1, k) - b00) * invh;
  const double Fym = bar_alpha_100_y(A3, i, j, k, -1) * (B3(i, j - 1, k) - b00) * invh;
  const double Fzp = bar_alpha_100_z(A3, i, j, k, +1) * (B3(i, j, k + 1) - b00) * invh;
  const double Fzm = bar_alpha_100_z(A3, i, j, k, -1) * (B3(i, j, k - 1) - b00) * invh;
  const double D100 = (Fxp + Fxm + Fyp + Fym + Fzp + Fzm) * invh;
  const double invs2 = 1.0 / std::sqrt(2.0);
  const double invs2h = invh * invs2;
  double sum110 = 0.0;
  const int s[2] = {+1, -1};
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      const int di = s[a];
      const int dj = s[b];
      sum110 += bar_alpha_110_xy(A3, i, j, k, di, dj) * (B3(i + di, j + dj, k) - b00) * invs2h;
      sum110 += bar_alpha_110_xz(A3, i, j, k, di, dj) * (B3(i + di, j, k + dj) - b00) * invs2h;
      sum110 += bar_alpha_110_yz(A3, i, j, k, di, dj) * (B3(i, j + di, k + dj) - b00) * invs2h;
    }
  }
  const double D110 = sum110 * (0.5 * invs2h);
  return (D100 + 2.0 * D110) / 3.0;
}

template <class A, class B>
ALCU_HD inline double div_alpha_grad(const A &al, const B &be, int i, int j, int k,
                                     double h, bool dim3) noexcept {
  if (dim3) {
    return div_alpha_grad_3d(al, be, i, j, k, h);
  }
  return div_alpha_grad_2d(al, be, i, j, k, h);
}

} // namespace alloy_pf_directional::dmath
