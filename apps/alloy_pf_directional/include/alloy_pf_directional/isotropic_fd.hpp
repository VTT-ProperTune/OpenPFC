// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cmath>

/**
 * @file isotropic_fd.hpp
 * @brief Ji, Tabrizi & Karma (JCP 457, 111069, 2022) isotropic FD operators.
 *
 * 2D scheme \(\bar S_{2,1}\): 2/3 on ⟨10⟩ + 1/3 on ⟨11⟩.
 * 3D scheme \(\bar S_{1,2,0}\): 1/3 on ⟨100⟩ + 2/3 on ⟨110⟩ (no ⟨111⟩).
 * Divergence is \(\nabla\cdot(\alpha\nabla\beta)\) with the face-α averages of Secs. 3.2 / 5.2.
 *
 * Copy of the stencil also shipped in alloy_pf_karma2001_benchmark. This app
 * must not include that tree; keep weights in sync if the stencil changes.
 */
namespace alloy_pf_directional::iso {

template <class F>
inline double laplacian_std(const F &f, int i, int j, int k, double h, bool dim3) noexcept {
  const double invh2 = 1.0 / (h * h);
  double s = f(i + 1, j, k) + f(i - 1, j, k) + f(i, j + 1, k) + f(i, j - 1, k);
  if (dim3) {
    s += f(i, j, k + 1) + f(i, j, k - 1);
    return (s - 6.0 * f(i, j, k)) * invh2;
  }
  return (s - 4.0 * f(i, j, k)) * invh2;
}

template <class F>
inline double laplacian_iso(const F &f, int i, int j, int k, double h, bool dim3) noexcept {
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
inline double grad2_iso(const F &f, int i, int j, int k, double h, bool dim3) noexcept {
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

/** \(\tilde D_{2,1}\) in 2D: outward-flux form of Ji et al. eqs. (26)–(29). */
template <class A, class B>
inline double div_alpha_grad_2d(const A &al, const B &be, int i, int j, int k, double h) noexcept {
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
inline double bar_alpha_100_x(const A &al, int i, int j, int k, int di) noexcept {
  const int ip = i + di;
  const double nearest = 0.25 * (al(ip, j, k) + al(i, j, k));
  const double nn = (al(ip, j + 1, k) + al(i, j + 1, k) + al(i, j, k + 1) + al(ip, j, k + 1) +
                     al(ip, j - 1, k) + al(i, j - 1, k) + al(i, j, k - 1) + al(ip, j, k - 1)) /
                    16.0;
  return nearest + nn;
}

template <class A>
inline double bar_alpha_100_y(const A &al, int i, int j, int k, int dj) noexcept {
  const int jp = j + dj;
  const double nearest = 0.25 * (al(i, jp, k) + al(i, j, k));
  const double nn = (al(i + 1, jp, k) + al(i + 1, j, k) + al(i, j, k + 1) + al(i, jp, k + 1) +
                     al(i - 1, jp, k) + al(i - 1, j, k) + al(i, j, k - 1) + al(i, jp, k - 1)) /
                    16.0;
  return nearest + nn;
}

template <class A>
inline double bar_alpha_100_z(const A &al, int i, int j, int k, int dk) noexcept {
  const int kp = k + dk;
  const double nearest = 0.25 * (al(i, j, kp) + al(i, j, k));
  const double nn = (al(i + 1, j, kp) + al(i + 1, j, k) + al(i, j + 1, k) + al(i, j + 1, kp) +
                     al(i - 1, j, kp) + al(i - 1, j, k) + al(i, j - 1, k) + al(i, j - 1, kp)) /
                    16.0;
  return nearest + nn;
}

template <class A>
inline double bar_alpha_110_xy(const A &al, int i, int j, int k, int di, int dj) noexcept {
  const int ip = i + di;
  const int jp = j + dj;
  const double nearest =
      (3.0 / 16.0) * (al(ip, j, k) + al(ip, jp, k) + al(i, jp, k) + al(i, j, k));
  const double nn = (al(ip, jp, k + 1) + al(ip, j, k + 1) + al(i, jp, k + 1) + al(i, j, k + 1) +
                     al(ip, jp, k - 1) + al(ip, j, k - 1) + al(i, jp, k - 1) + al(i, j, k - 1)) /
                    32.0;
  return nearest + nn;
}

template <class A>
inline double bar_alpha_110_xz(const A &al, int i, int j, int k, int di, int dk) noexcept {
  const int ip = i + di;
  const int kp = k + dk;
  const double nearest =
      (3.0 / 16.0) * (al(ip, j, k) + al(ip, j, kp) + al(i, j, kp) + al(i, j, k));
  const double nn = (al(ip, j + 1, kp) + al(ip, j + 1, k) + al(i, j + 1, kp) + al(i, j + 1, k) +
                     al(ip, j - 1, kp) + al(ip, j - 1, k) + al(i, j - 1, kp) + al(i, j - 1, k)) /
                    32.0;
  return nearest + nn;
}

template <class A>
inline double bar_alpha_110_yz(const A &al, int i, int j, int k, int dj, int dk) noexcept {
  const int jp = j + dj;
  const int kp = k + dk;
  const double nearest =
      (3.0 / 16.0) * (al(i, jp, k) + al(i, jp, kp) + al(i, j, kp) + al(i, j, k));
  const double nn = (al(i + 1, jp, kp) + al(i + 1, jp, k) + al(i + 1, j, kp) + al(i + 1, j, k) +
                     al(i - 1, jp, kp) + al(i - 1, jp, k) + al(i - 1, j, kp) + al(i - 1, j, k)) /
                    32.0;
  return nearest + nn;
}

/** 3×3×3 gather so \(\bar\alpha\) rereads unique nodes, not capturing lambdas. */
struct Cube27 {
  double v[3][3][3];
  int ci = 0, cj = 0, ck = 0;
  double operator()(int i, int j, int k) const noexcept {
    return v[i - ci + 1][j - cj + 1][k - ck + 1];
  }
};

/** \(\tilde D_{1,2,0}\) in 3D: eqs. (47)–(52). */
template <class A, class B>
inline double div_alpha_grad_3d(const A &al, const B &be, int i, int j, int k, double h) noexcept {
  Cube27 A3{};
  Cube27 B3{};
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
inline double div_alpha_grad(const A &al, const B &be, int i, int j, int k, double h,
                             bool dim3) noexcept {
  if (dim3) {
    return div_alpha_grad_3d(al, be, i, j, k, h);
  }
  return div_alpha_grad_2d(al, be, i, j, k, h);
}

} // namespace alloy_pf_directional::iso
