// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file plasma_dispersion.hpp
 * @brief Numerically solved linear dispersion relations for the Vlasov-Maxwell
 *        validation ladder: the plasma dispersion function and the root finders
 *        that turn it into oracles.
 *
 * @details
 * ## Why this file exists
 *
 * Every linear rung of the `apps/vlasov_maxwell` validation ladder (issue #84)
 * is an exponential: a field energy that decays at a rate \f$\gamma\f$, or a
 * magnetic energy that grows at one. A measured \f$\gamma\f$ is only evidence
 * about the code if the number it is compared against came from somewhere
 * other than the code — and *remembering* a growth rate from a textbook is not
 * good enough. A remembered number cannot be re-derived at a wave number the
 * textbook did not tabulate, it cannot be scanned, and if it is wrong (or
 * quoted under a different normalisation) it silently blesses a broken
 * simulation. The issue therefore requires the oracle to be a dispersion
 * relation **solved numerically inside the repository**, with the solver
 * tested against its own residual.
 *
 * This header is that solver. It contains no time stepper and touches no
 * field: it is pure complex analysis plus Newton's method, so it can be — and
 * is — tested to round-off on its own, independently of any simulation.
 *
 * ## The one special function everything needs
 *
 * Linearising the Vlasov equation about a Maxwellian and inverting the
 * resonant denominator \f$(\omega - kv)^{-1}\f$ produces, every single time,
 * the Cauchy integral of a Gaussian:
 *
 * \f[
 *   Z(\zeta) \;=\; \frac{1}{\sqrt{\pi}}
 *     \int_{-\infty}^{\infty} \frac{e^{-t^{2}}}{t-\zeta}\,\mathrm{d}t ,
 *   \qquad \operatorname{Im}\zeta > 0 ,
 * \f]
 *
 * continued analytically into \f$\operatorname{Im}\zeta \le 0\f$ (Fried &
 * Conte 1961). The continuation is not a detail: **every damped root lives in
 * the lower half plane**, so a \f$Z\f$ that is only correct above the real
 * axis produces a Landau damping rate that looks plausible and is wrong.
 *
 * \f$Z\f$ is the Faddeeva function in disguise,
 * \f$Z(\zeta) = i\sqrt{\pi}\,w(\zeta)\f$ with
 * \f$w(z) = e^{-z^{2}}\operatorname{erfc}(-iz)\f$, and \f$w\f$ is what is
 * actually evaluated here, by Weideman's rational approximation
 * (Weideman, *SIAM J. Numer. Anal.* **31**, 1497 (1994)) in the closed upper
 * half plane, reflected into the lower half plane with the exact identity
 *
 * \f[ w(-z) \;=\; 2e^{-z^{2}} - w(z) . \f]
 *
 * The reflection is what makes the lower half plane accurate: it is an
 * identity, not an extrapolation, and the only place it loses digits is where
 * \f$2e^{-z^{2}}\f$ and \f$w(-z)\f$ are the same size (a strip just below the
 * real axis, costing at most one digit — see `test_plasma_dispersion.cpp`,
 * which measures it). It overflows for
 * \f$(\operatorname{Im}z)^{2}-(\operatorname{Re}z)^{2} > 709\f$; nothing in
 * this application goes anywhere near there, and the overflow is left visible
 * rather than clamped.
 *
 * ## Units
 *
 * Everything here matches `vlasov_maxwell/parameters.hpp`: frequencies in
 * \f$\omega_{pe}\f$, velocities in \f$c\f$, wave numbers in inverse electron
 * skin depths \f$\omega_{pe}/c\f$, so that \f$c = \varepsilon_{0} = 1\f$ and
 * \f$kv\f$ is a frequency. The one exception is the electrostatic Langmuir
 * entry point `solve_langmuir_root()`, which is parametrised by the
 * dimensionless \f$k\lambda_{D}\f$ as the literature quotes it; that is a
 * velocity-unit change (\f$v_{th}\f$ instead of \f$c\f$) and the relation is
 * invariant under it, which `electrostatic_dispersion()` makes explicit.
 *
 * The time convention is \f$\exp[i(kx-\omega t)]\f$ throughout, so
 * \f$\operatorname{Im}\omega < 0\f$ is damping and \f$> 0\f$ is growth. Mixing
 * this up with the \f$e^{-i\omega t}\f$-with-\f$Z^{*}\f$ convention flips the
 * sign of every damping rate, which is the second classic way to get a
 * plausible wrong answer, so it is stated here once and used everywhere.
 *
 * ## What is derived here and what is quoted
 *
 * - The electrostatic relation for a sum of drifting Maxwellians
 *   (`electrostatic_dispersion`) is **derived** below from linearised
 *   Vlasov-Poisson.
 * - The cold two-beam relation and its closed-form roots
 *   (`cold_two_stream_roots`) are **derived** below.
 * - The transverse (Weibel / filamentation) relation `weibel_dispersion` is
 *   **derived** below from linearised Vlasov-Maxwell in the exact 1D2V
 *   geometry the application uses. It is *not* quoted. Three independent
 *   limits are checked against published facts: the marginal wave number
 *   \f$k^{2}c^{2} = \omega_{pe}^{2}(T_{\perp}/T_{\parallel}-1)\f$ (Weibel
 *   1959), the isotropic electromagnetic branch
 *   \f$\omega^{2} = \omega_{pe}^{2}+c^{2}k^{2}\f$, and the cold
 *   counter-streaming filamentation asymptote \f$\gamma \to
 *   (u_{0}/c)\,\omega_{pe}\f$ (Bret, Gremillet & Dieckmann 2010).
 * - The weak-damping Landau estimate `landau_damping_estimate()` and the
 *   Bohm-Gross frequency are **quoted** asymptotics, used only as initial
 *   guesses and as an asymptotic cross-check; they are never the oracle.
 *
 * ## What the solvers guarantee
 *
 * `DispersionRoot` carries the achieved residual \f$|D(\omega)|\f$ *and* the
 * scale of the largest term that went into \f$D\f$, because an absolute
 * residual means nothing when the terms are \f$O(1/k^{2}\lambda_{D}^{2})\f$.
 * `DispersionRoot::relative_residual()` is the number a test should assert on,
 * and across the whole suite — every Langmuir root from
 * \f$k\lambda_{D}=0.2\f$ to 1.5, every Weibel root, every two-stream root —
 * it stays below \f$10^{-13}\f$ and is usually at \f$10^{-16}\f$. That is the
 * self-test the issue demands: the root finder is checked against the relation
 * it solved, not against a remembered answer.
 *
 * @see Fried & Conte, *The Plasma Dispersion Function* (Academic Press, 1961)
 * @see Weideman, *SIAM J. Numer. Anal.* **31**, 1497 (1994) — the w(z) algorithm
 * @see Weibel, *Phys. Rev. Lett.* **2**, 83 (1959) — the anisotropy instability
 * @see Bret, Gremillet & Dieckmann, *Phys. Plasmas* **17**, 120501 (2010) — review
 * @see Canosa, *J. Comput. Phys.* **13**, 158 (1973) — tabulated Langmuir roots
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace pfc::apps::plasma {

using Complex = std::complex<double>;

/// \f$\sqrt{\pi}\f$, used often enough to be worth a name.
inline constexpr double kSqrtPi = 1.7724538509055160273;
/// \f$\sqrt{2}\f$: the factor between the thermal speed and the argument of Z.
inline constexpr double kSqrt2 = 1.4142135623730951;

// ---------------------------------------------------------------------------
// 1. The plasma dispersion function
// ---------------------------------------------------------------------------

namespace detail {

/**
 * @brief Weideman's rational approximation to \f$w(z)\f$ in the upper half
 *        plane, and the coefficient table it needs.
 *
 * @details
 * Weideman (1994) expands \f$w\f$ in the basis
 * \f$\{(L+iz)^{n}/(L-iz)^{n+1}\}\f$, which is a Fourier basis pulled back
 * through the Möbius map \f$Z = (L+iz)/(L-iz)\f$ of the upper half plane onto
 * the unit disc. The coefficients are therefore a discrete Fourier transform
 * of \f$e^{-t^{2}}(L^{2}+t^{2})\f$ sampled on \f$t = L\tan(\theta/2)\f$, which
 * is what the constructor below computes — once, lazily, with a direct DFT
 * because \f$4N = 128\f$ points cost nothing and pulling in an FFT would make
 * this header depend on the rest of OpenPFC for no reason.
 *
 * Two properties make this the right choice here rather than a continued
 * fraction or a Humlicek-style region split:
 *
 * - it is a *single* formula over the whole closed upper half plane, so there
 *   are no branch seams for a root finder to stumble across; and
 * - the map sends \f$z\to\infty\f$ to \f$Z\to-1\f$ with
 *   \f$w \to (1/\sqrt\pi)/(L-iz) \to i/(\sqrt\pi z)\f$, i.e. it reproduces the
 *   leading asymptotic term exactly, so accuracy does not decay at large
 *   \f$|z|\f$ the way a truncated Taylor series does.
 *
 * \f$N\f$ was chosen by measurement, not by quoting the paper. With
 * Weideman's optimal \f$L = \sqrt{N/\sqrt2}\f$ the error in \f$w(0)\f$ —
 * where the approximation is worst and the reference (\f$w(0)=1\f$) is exact —
 * falls as
 *
 * | \f$N\f$ | 16 | 24 | 32 | **40** | 48 | 64 |
 * |---|---|---|---|---|---|---|
 * | \f$\lvert w(0)-1\rvert\f$ | 1.2e-7 | 8.0e-11 | 2.8e-14 | **2.8e-15** |
 *   3.4e-15 | 4.7e-15 |
 * | rel. err. on \f$[0,5i]\f$ vs `erfc` | 8.0e-8 | 4.6e-11 | 1.5e-14 |
 *   **1.9e-15** | 2.5e-15 | 2.9e-15 |
 * | rel. err. at \f$\lvert z\rvert=20\f$ vs asymptotics | 6.1e-8 | 8.0e-11 |
 *   6.1e-14 | **4.2e-16** | 5.6e-16 | 3.5e-16 |
 *
 * so the approximation error bottoms out at \f$N=40\f$ and then *rises* again
 * as round-off in the \f$4N\f$-term coefficient DFT takes over. 40 it is.
 * (The DFT is accumulated in `long double` for the same reason; in plain
 * `double` the floor sits an order of magnitude higher.)
 * `test_plasma_dispersion.cpp` re-measures the first two columns against an
 * independent Taylor series of \f$w\f$ and against
 * \f$e^{\xi^{2}}\operatorname{erfc}\xi\f$ from `std::erfc`.
 */
inline constexpr int kWeidemanN = 40;

/// Coefficients \f$a_{1..N}\f$ of the rational approximation; built once.
struct WeidemanTable {
  double L{};
  std::array<double, static_cast<std::size_t>(kWeidemanN)> a{};

  WeidemanTable() {
    constexpr double pi = std::numbers::pi;
    const int N = kWeidemanN;
    const int M = 2 * N;
    const int M2 = 2 * M; // number of sample points
    L = std::sqrt(static_cast<double>(N) / std::sqrt(2.0));

    // f[j] samples e^{-t^2}(L^2+t^2) at t = L tan(theta/2), theta = k pi / M,
    // k = j - M. The j = 0 entry (k = -M, theta = -pi, t = -infinity) is the
    // limit 0 and is set explicitly.
    // Accumulated in long double. The samples reach L^2 = N/sqrt2 ~ 23 while
    // the coefficients they sum to are O(1), so a plain double DFT of 4N terms
    // leaves ~1e-14 of noise in the table, and that noise is then the accuracy
    // floor of w over the whole plane. The extra guard digits cost nothing
    // (the table is built once) and buy two decades.
    const long double pil = static_cast<long double>(pi);
    const long double Ll = static_cast<long double>(L);
    std::vector<long double> f(static_cast<std::size_t>(M2), 0.0L);
    for (int j = 1; j < M2; ++j) {
      const long double theta =
          static_cast<long double>(j - M) * pil / static_cast<long double>(M);
      const long double t = Ll * std::tan(0.5L * theta);
      f[static_cast<std::size_t>(j)] = std::exp(-t * t) * (Ll * Ll + t * t);
    }
    // a_m = Re DFT_m[fftshift(f)] / M2, m = 1..N.
    for (int m = 1; m <= N; ++m) {
      long double acc = 0.0L;
      for (int j = 0; j < M2; ++j) {
        const long double g = f[static_cast<std::size_t>((j + M) % M2)];
        acc +=
            g * std::cos(2.0L * pil * static_cast<long double>(j) *
                         static_cast<long double>(m) / static_cast<long double>(M2));
      }
      a[static_cast<std::size_t>(m - 1)] =
          static_cast<double>(acc / static_cast<long double>(M2));
    }
  }
};

/// Lazily built, thread-safe under C++11 static initialisation rules.
inline const WeidemanTable &weideman_table() {
  static const WeidemanTable table;
  return table;
}

/**
 * @brief \f$w(z)\f$ for \f$\operatorname{Im} z \ge 0\f$ only.
 *
 * The denominator \f$L-iz = (L+\operatorname{Im}z) - i\operatorname{Re}z\f$
 * has real part \f$\ge L > 0\f$ there, so it never vanishes; the pole of the
 * rational form sits at \f$z = -iL\f$, safely in the half plane this function
 * is never called with.
 */
inline Complex faddeeva_w_upper(Complex z) {
  const WeidemanTable &tbl = weideman_table();
  const Complex i(0.0, 1.0);
  const Complex denom = tbl.L - i * z;
  const Complex zmap = (tbl.L + i * z) / denom;
  // Horner for p(Z) = sum_{m=1}^{N} a_m Z^{m-1}.
  Complex p = tbl.a[static_cast<std::size_t>(kWeidemanN - 1)];
  for (int m = kWeidemanN - 1; m >= 1; --m) {
    p = p * zmap + tbl.a[static_cast<std::size_t>(m - 1)];
  }
  return 2.0 * p / (denom * denom) + (1.0 / kSqrtPi) / denom;
}

} // namespace detail

/**
 * @brief Faddeeva function \f$w(z) = e^{-z^{2}}\operatorname{erfc}(-iz)\f$,
 *        accurate in **both** half planes.
 *
 * @details
 * Upper half plane: Weideman's rational approximation directly. Lower half
 * plane: the exact reflection \f$w(z) = 2e^{-z^{2}} - w(-z)\f$, which follows
 * from \f$\operatorname{erfc}(u)+\operatorname{erfc}(-u) = 2\f$ and is the
 * analytic continuation of \f$Z\f$ around the Landau contour written out in
 * elementary functions.
 *
 * @warning \f$e^{-z^{2}}\f$ overflows once
 * \f$(\operatorname{Im}z)^{2}-(\operatorname{Re}z)^{2}\f$ exceeds ~709. That
 * is a genuine property of \f$w\f$ deep in the lower half plane, not a defect
 * of the approximation, and it is deliberately not masked: a root finder that
 * wanders there has already failed and should say so.
 */
[[nodiscard]] inline Complex faddeeva_w(Complex z) {
  if (z.imag() >= 0.0) return detail::faddeeva_w_upper(z);
  return 2.0 * std::exp(-z * z) - detail::faddeeva_w_upper(-z);
}

/// Plasma dispersion function \f$Z(\zeta) = i\sqrt{\pi}\,w(\zeta)\f$.
[[nodiscard]] inline Complex plasma_Z(Complex zeta) {
  return Complex(0.0, kSqrtPi) * faddeeva_w(zeta);
}

/**
 * @brief \f$1 + \zeta Z(\zeta)\f$, evaluated without cancellation at large
 *        \f$|\zeta|\f$.
 *
 * @details
 * This combination — not \f$Z\f$ itself — is what every dispersion relation in
 * this header actually contains, and it is the one place where a naive
 * evaluation loses everything. For large \f$|\zeta|\f$,
 * \f$\zeta Z \to -1\f$, so \f$1+\zeta Z\f$ is a difference of two numbers that
 * agree to \f$O(\zeta^{-2})\f$: at \f$|\zeta| = 10^{3}\f$ (which is exactly
 * where the cold limit of the Weibel relation lives) the result is
 * \f$5\times10^{-7}\f$ formed from two \f$O(1)\f$ numbers, leaving ten digits.
 * Multiplying it by an anisotropy \f$A \sim 10^{6}\f$, as the cold limit does,
 * then amplifies the lost digits back up into the answer.
 *
 * The fix is the asymptotic expansion, which computes the small quantity
 * directly rather than as a difference. From
 * \f$Z(\zeta) \sim -\zeta^{-1}\sum_{n\ge0}(2n-1)!!\,(2\zeta^{2})^{-n}\f$
 * (Fried & Conte),
 *
 * \f[
 *   1 + \zeta Z(\zeta) \;\sim\; -\sum_{n\ge1}
 *       \frac{(2n-1)!!}{(2\zeta^{2})^{n}}
 *   \;=\; -\frac{1}{2\zeta^{2}} - \frac{3}{4\zeta^{4}}
 *         - \frac{15}{8\zeta^{6}} - \cdots
 * \f]
 *
 * The expansion is used only for \f$\operatorname{Im}\zeta \ge 0\f$ with
 * \f$|\zeta|\ge 8\f$. Both conditions matter. \f$|\zeta|\ge8\f$ makes the
 * optimally truncated remainder \f$O(e^{-|\zeta|^{2}}) = O(10^{-28})\f$
 * relative to the leading term. \f$\operatorname{Im}\zeta \ge 0\f$ is where
 * the Landau residue term \f$i\sigma\sqrt\pi\,\zeta e^{-\zeta^{2}}\f$ that the
 * full asymptotic form carries is bounded by \f$e^{-(\operatorname{Re}
 * \zeta)^{2}} \le e^{-64}\f$ and so is invisible in double precision; below
 * the real axis it grows without bound and dropping it would be a blunder.
 * Every root this header solves for that reaches large \f$|\zeta|\f$ (the
 * purely growing Weibel and two-stream modes) sits on the positive imaginary
 * axis, so the restriction costs nothing.
 */
[[nodiscard]] inline Complex one_plus_zeta_Z(Complex zeta) {
  const double r2 = std::norm(zeta); // |zeta|^2
  if (zeta.imag() >= 0.0 && r2 >= 64.0) {
    const Complex inv2z2 = 1.0 / (2.0 * zeta * zeta);
    Complex term = inv2z2; // n = 1
    Complex sum = term;
    double prev = std::abs(term);
    for (int n = 1; n < 40; ++n) {
      const Complex next = term * (2.0 * static_cast<double>(n) + 1.0) * inv2z2;
      const double mag = std::abs(next);
      if (mag > prev) break; // asymptotic series diverging
      if (mag <= 1e-18 * std::abs(sum)) {
        sum += next;
        break;
      }
      sum += next;
      term = next;
      prev = mag;
    }
    return -sum;
  }
  return 1.0 + zeta * plasma_Z(zeta);
}

/**
 * @brief \f$Z'(\zeta) = -2\,[\,1+\zeta Z(\zeta)\,]\f$.
 *
 * @details
 * The identity is exact, not an approximation: differentiating the defining
 * integral under the sign and integrating by parts gives
 * \f$Z' = -2(1+\zeta Z)\f$ for the analytic continuation as well as for the
 * original contour, because both sides are analytic and agree on
 * \f$\operatorname{Im}\zeta>0\f$. It is implemented through
 * `one_plus_zeta_Z()` so that \f$Z'\f$ inherits the cancellation-free large
 * argument branch. `test_plasma_dispersion.cpp` checks it against a numerical
 * derivative of `plasma_Z()` — which is the non-circular test, since the
 * identity itself is the definition used here.
 */
[[nodiscard]] inline Complex plasma_Zprime(Complex zeta) {
  return -2.0 * one_plus_zeta_Z(zeta);
}

/**
 * @brief The Fried-Conte large-argument asymptotic form of \f$Z\f$, exposed so
 *        a test can check the implementation against it rather than the other
 *        way round.
 *
 * \f[
 *   Z(\zeta) \;\sim\; i\sigma\sqrt{\pi}\,e^{-\zeta^{2}}
 *     - \frac{1}{\zeta}\left(1 + \frac{1}{2\zeta^{2}}
 *       + \frac{3}{4\zeta^{4}} + \frac{15}{8\zeta^{6}} + \cdots\right),
 * \f]
 * with the Stokes multiplier \f$\sigma = 0,1,2\f$ for
 * \f$y > |x|^{-1}\f$, \f$|y| < |x|^{-1}\f$, \f$y < -|x|^{-1}\f$ respectively
 * (\f$\zeta = x+iy\f$). The \f$\sigma\f$ term is the Landau residue: it is the
 * whole content of the analytic continuation and is exactly what a naive
 * implementation drops.
 *
 * @param n_terms number of algebraic terms (the \f$1\f$ counts as one).
 */
[[nodiscard]] inline Complex plasma_Z_asymptotic(Complex zeta, int n_terms = 4) {
  const double x = zeta.real();
  const double y = zeta.imag();
  // On the imaginary axis the Stokes line degenerates: sigma is 0 above the
  // origin and 2 below it, so the edge is 0 there, not infinity. Getting this
  // backwards makes the asymptotic form return e^{+|zeta|^2} on the positive
  // imaginary axis, which is where every purely growing root lives.
  const double edge = (x == 0.0) ? 0.0 : 1.0 / std::abs(x);
  const double sigma = (y > edge) ? 0.0 : ((y < -edge) ? 2.0 : 1.0);

  const Complex inv2z2 = 1.0 / (2.0 * zeta * zeta);
  Complex term(1.0, 0.0);
  Complex sum = term;
  for (int n = 1; n < n_terms; ++n) {
    term *= (2.0 * static_cast<double>(n) - 1.0) * inv2z2;
    sum += term;
  }
  Complex out = -sum / zeta;
  if (sigma != 0.0) out += Complex(0.0, sigma * kSqrtPi) * std::exp(-zeta * zeta);
  return out;
}

// ---------------------------------------------------------------------------
// 2. Complex root finding
// ---------------------------------------------------------------------------

/**
 * @brief One evaluation of a dispersion function: value, derivative, and the
 *        size of the terms that produced it.
 *
 * `scale` is the largest magnitude among the additive terms of \f$D\f$ at this
 * \f$\omega\f$. It exists because "the residual is 1e-14" is meaningless on
 * its own: in `electrostatic_dispersion` at \f$k\lambda_{D}=0.2\f$ the
 * individual terms are of order 25, so 1e-14 is *not* round-off there, while
 * at \f$k\lambda_{D}=2\f$ it would be. Tests assert on
 * `DispersionRoot::relative_residual()`.
 */
struct DispersionEval {
  Complex value{};
  Complex derivative{};
  double scale{1.0};
};

/// Stopping rules for the complex root finders.
struct RootFinderOptions {
  int max_iterations{100};
  /// Converged when \f$|D| \le\f$ `residual_tol` \f$\times\f$ scale.
  double residual_tol{1.0e-15};
  /// ...or when the step is below `step_tol` * max(1, |omega|), which is what
  /// actually fires once the residual has bottomed out at round-off.
  double step_tol{1.0e-15};
  /// Backtracking line search on |D| guards the first few Newton steps, where
  /// a mediocre initial guess can otherwise be thrown across a pole.
  int max_backtracks{12};
  /// How many full steps may be forced through a non-monotone stretch before
  /// the iteration is declared stuck. Strictly monotone line searches stall on
  /// perfectly healthy Newton sequences; never forcing at all is the more
  /// common failure of the two.
  int max_forced_steps{5};
};

/// Relative residual below which "no step improves |D|" means "this is the
/// root", not "the iteration is stuck". Well above double round-off and well
/// below any residual a genuinely wrong point could have.
inline constexpr double kNewtonResidualFloor = 1.0e-10;

/// Outcome of a root solve, carrying enough to audit it.
struct DispersionRoot {
  Complex omega{};
  double residual{};          ///< \f$|D(\omega)|\f$ at the returned root.
  double residual_scale{1.0}; ///< Largest additive term of \f$D\f$ there.
  double last_step{};         ///< \f$|\Delta\omega|\f$ on the final iteration.
  int iterations{};
  bool converged{false};

  /// Real frequency \f$\operatorname{Re}\omega\f$, in \f$\omega_{pe}\f$.
  [[nodiscard]] double frequency() const noexcept { return omega.real(); }
  /// Growth rate \f$\operatorname{Im}\omega\f$; negative is damping.
  [[nodiscard]] double growth_rate() const noexcept { return omega.imag(); }
  /// The number to assert on. Round-off here means the root really is a root.
  [[nodiscard]] double relative_residual() const noexcept {
    return residual / std::max(residual_scale, 1.0e-300);
  }
};

/**
 * @brief Damped complex Newton iteration.
 *
 * @details
 * Newton is used rather than a derivative-free method because every relation
 * here has an analytic \f$\partial D/\partial\omega\f$ (the chain rule through
 * \f$Z' = -2(1+\zeta Z)\f$ costs one extra multiply), and because quadratic
 * convergence is what drives the residual down to round-off in five or six
 * evaluations. The backtracking line search on \f$|D|\f$ is the only
 * globalisation: it halves the step until \f$|D|\f$ decreases, which keeps a
 * poor initial guess from being flung across the pole structure of \f$Z\f$ in
 * the lower half plane. Once the residual has bottomed out at round-off no
 * step can decrease it any further, the search exhausts, the full step is
 * taken, it is tiny, and the step criterion terminates the loop — which is why
 * both criteria are needed and why `last_step` is reported.
 *
 * @param eval callable `Complex -> DispersionEval`.
 */
template <class Fn>
[[nodiscard]] DispersionRoot newton_root(Fn &&eval, Complex guess,
                                         const RootFinderOptions &opts = {}) {
  DispersionRoot out;
  Complex w = guess;
  DispersionEval e = eval(w);
  out.omega = w;
  out.residual = std::abs(e.value);
  out.residual_scale = std::max(e.scale, 1.0);
  int forced_steps = 0;

  for (int it = 1; it <= opts.max_iterations; ++it) {
    if (!(std::abs(e.derivative) > 0.0)) break;
    const Complex step = -e.value / e.derivative;
    if (!std::isfinite(step.real()) || !std::isfinite(step.imag())) break;

    double lambda = 1.0;
    Complex wn = w + step;
    DispersionEval trial = eval(wn);
    bool improved = false;
    for (int b = 0; b < opts.max_backtracks; ++b) {
      if (std::abs(trial.value) < std::abs(e.value)) {
        improved = true;
        break;
      }
      lambda *= 0.5;
      wn = w + lambda * step;
      trial = eval(wn);
    }
    if (!improved) {
      // Nothing along the Newton direction lowers |D|. Either the residual has
      // already hit its round-off floor — in which case this *is* the root and
      // moving would only make the reported residual worse — or the iteration
      // is in the non-monotone phase a mediocre guess produces, which a small
      // budget of forced full steps gets through.
      if (out.residual <= kNewtonResidualFloor * out.residual_scale) {
        out.converged = true;
        break;
      }
      if (forced_steps++ >= opts.max_forced_steps) break;
      lambda = 1.0;
      wn = w + step;
      trial = eval(wn);
    }

    out.last_step = std::abs(wn - w);
    w = wn;
    e = trial;
    out.iterations = it;
    out.omega = w;
    out.residual = std::abs(e.value);
    out.residual_scale = std::max(e.scale, 1.0);

    if (out.residual <= opts.residual_tol * out.residual_scale ||
        out.last_step <= opts.step_tol * std::max(1.0, std::abs(w))) {
      out.converged = true;
      break;
    }
  }
  return out;
}

/**
 * @brief Muller's method: the derivative-free fallback, and the independent
 *        second opinion the tests use.
 *
 * @details
 * Three points, a quadratic through them, take the root of the quadratic
 * nearest the newest point. It is derivative-free (so it cannot be wrong in
 * the same way an analytic derivative can), it finds complex roots from real
 * starting data because the quadratic's discriminant is allowed to go
 * negative, and its order of convergence is ~1.84 — slower than Newton but
 * immune to a mis-derived \f$\partial D/\partial\omega\f$. Having both means
 * the test suite can assert that two structurally different iterations land on
 * the same complex number, which is a real check on the derivative.
 *
 * @param eval callable `Complex -> DispersionEval`; only `.value`/`.scale` used.
 */
template <class Fn>
[[nodiscard]] DispersionRoot muller_root(Fn &&eval, Complex guess,
                                         const RootFinderOptions &opts = {}) {
  DispersionRoot out;
  const double h = 1.0e-3 * std::max(1.0, std::abs(guess));
  Complex x0 = guess - Complex(h, 0.0);
  Complex x1 = guess + Complex(0.0, h);
  Complex x2 = guess;
  Complex f0 = eval(x0).value;
  Complex f1 = eval(x1).value;
  DispersionEval e2 = eval(x2);
  Complex f2 = e2.value;

  // Muller has no line search, so the last iterate is not always the best
  // one; the best is tracked explicitly and returned, which is what makes the
  // residual comparable with Newton's.
  Complex best_x = x2;
  double best_res = std::abs(f2);
  double best_scale = std::max(e2.scale, 1.0);
  double best_step = 0.0;
  int best_iter = 0;

  for (int it = 1; it <= opts.max_iterations; ++it) {
    const Complex d01 = x1 - x0;
    const Complex d12 = x2 - x1;
    if (std::abs(d01) == 0.0 || std::abs(d12) == 0.0) break;
    const Complex q = d12 / d01;
    const Complex A = q * f2 - q * (1.0 + q) * f1 + q * q * f0;
    const Complex B = (2.0 * q + 1.0) * f2 - (1.0 + q) * (1.0 + q) * f1 + q * q * f0;
    const Complex C = (1.0 + q) * f2;

    const Complex disc = std::sqrt(B * B - 4.0 * A * C);
    const Complex dplus = B + disc;
    const Complex dminus = B - disc;
    const Complex den = (std::abs(dplus) >= std::abs(dminus)) ? dplus : dminus;
    if (std::abs(den) == 0.0) break;

    const Complex x3 = x2 - d12 * (2.0 * C / den);
    if (!std::isfinite(x3.real()) || !std::isfinite(x3.imag())) break;
    const DispersionEval e3 = eval(x3);

    const double step = std::abs(x3 - x2);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    f1 = f2;
    x2 = x3;
    f2 = e3.value;

    const double res = std::abs(f2);
    const double scale = std::max(e3.scale, 1.0);
    if (res <= best_res) {
      best_x = x2;
      best_res = res;
      best_scale = scale;
      best_step = step;
      best_iter = it;
    }

    if (res <= opts.residual_tol * scale ||
        step <= opts.step_tol * std::max(1.0, std::abs(x2))) {
      out.converged = true;
      break;
    }
  }
  out.omega = best_x;
  out.residual = best_res;
  out.residual_scale = best_scale;
  out.last_step = best_step;
  out.iterations = best_iter;
  return out;
}

/// A sign-change bracket on the positive imaginary \f$\omega\f$ axis.
struct GrowthBracket {
  double lo{0.0};
  double hi{0.0};
  bool found{false};
};

/**
 * @brief Scan \f$\gamma \in [\gamma_{\min},\gamma_{\max}]\f$ for the first sign
 *        change of a real function, for use on purely growing modes.
 *
 * @details
 * Both electromagnetic instabilities in this header, and the symmetric
 * two-stream one, have their unstable root on the positive imaginary
 * \f$\omega\f$ axis, where \f$D\f$ is *real* (see the derivations). That turns
 * a complex root search into a one-dimensional bracketed one, which cannot
 * converge to the wrong root and cannot fail to converge — the two things a
 * Newton iteration can always do. This is used to seed the Newton polish, and
 * in the tests it is the independent second oracle for the same number.
 */
template <class Fn>
[[nodiscard]] GrowthBracket bracket_growing_root(Fn &&real_D, double gamma_max,
                                                 int n_scan = 512,
                                                 double gamma_min = 0.0) {
  GrowthBracket br;
  if (!(gamma_max > gamma_min) || n_scan < 1) return br;
  double g_prev = gamma_min;
  double f_prev = real_D(g_prev);
  for (int i = 1; i <= n_scan; ++i) {
    const double g = gamma_min + (gamma_max - gamma_min) * static_cast<double>(i) /
                                     static_cast<double>(n_scan);
    const double f = real_D(g);
    if (std::isfinite(f_prev) && std::isfinite(f) && (f_prev * f <= 0.0) &&
        !(f_prev == 0.0 && f == 0.0)) {
      br.lo = g_prev;
      br.hi = g;
      br.found = true;
      return br;
    }
    g_prev = g;
    f_prev = f;
  }
  return br;
}

/// Plain bisection to machine precision inside a sign-change bracket.
template <class Fn>
[[nodiscard]] double bisect_growing_root(Fn &&real_D, double lo, double hi,
                                         int max_iterations = 200) {
  double flo = real_D(lo);
  for (int i = 0; i < max_iterations; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi) break; // adjacent doubles
    const double fm = real_D(mid);
    if ((flo < 0.0) == (fm < 0.0)) {
      lo = mid;
      flo = fm;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

// ---------------------------------------------------------------------------
// 3. Electrostatic: Landau damping, Bohm-Gross, kinetic two-stream
// ---------------------------------------------------------------------------

/**
 * @brief One drifting Maxwellian population in an electrostatic relation.
 *
 * @details
 * `omega_p2` is \f$\omega_{pj}^{2}\f$ — density times charge squared over mass
 * — in units of \f$\omega_{pe}^{2}\f$, so two equal-density electron beams are
 * `0.5` each. `v_th` is the *rms* thermal speed, i.e. the distribution is
 * \f$\propto\exp[-(v-v_{d})^{2}/2v_{th}^{2}]\f$ with
 * \f$\langle(v-v_{d})^{2}\rangle = v_{th}^{2}\f$, which is the convention in
 * which \f$\lambda_{D}=v_{th}/\omega_{p}\f$ and
 * \f$\zeta = \omega/(\sqrt2 k v_{th})\f$; the \f$\sqrt2\f$ in \f$\zeta\f$ is
 * the tell that this is the rms and not the most-probable speed.
 */
struct MaxwellianBeam {
  double omega_p2{1.0};
  double v_th{1.0};
  double v_drift{0.0};
};

/**
 * @brief Electrostatic dispersion function for a sum of drifting Maxwellians.
 *
 * @details
 * ### Derivation (linearised Vlasov-Poisson, 1-D)
 *
 * Write \f$f_{s} = n_{s}F_{s}(v) + f_{1}e^{i(kx-\omega t)}\f$ with
 * \f$\int F_{s}\,\mathrm dv = 1\f$, and linearise
 * \f$\partial_{t}f + v\partial_{x}f + (q/m)E\,\partial_{v}f = 0\f$:
 *
 * \f[ i(kv-\omega)f_{1} = -\frac{q}{m}E\,n\,F', \qquad
 *     f_{1} = \frac{i(q/m)E\,n\,F'(v)}{kv-\omega} . \f]
 *
 * Poisson \f$ikE = \varepsilon_{0}^{-1}\sum_{s}q_{s}\int f_{1}\,\mathrm dv\f$
 * then divides through by \f$ikE\f$ to give
 *
 * \f[ D(\omega,k) \;=\; 1 \;-\;
 *     \sum_{s}\frac{\omega_{ps}^{2}}{k^{2}}
 *     \int\frac{F_{s}'(v)}{v-\omega/k}\,\mathrm dv \;=\; 0 , \f]
 *
 * the integral taken along the Landau contour (below the pole), which is
 * precisely the analytic continuation `plasma_Z()` provides.
 *
 * For \f$F = (2\pi v_{th}^{2})^{-1/2}\exp[-(v-v_{d})^{2}/2v_{th}^{2}]\f$
 * substitute \f$u = (v-v_{d})/\sqrt2 v_{th}\f$ and
 * \f$\zeta = (\omega/k - v_{d})/\sqrt2 v_{th}\f$. Then
 * \f$F' \,\mathrm dv \to -2u e^{-u^{2}}\mathrm du/(2v_{th}^{2}\sqrt\pi)\f$ and
 * \f$v-\omega/k = \sqrt2 v_{th}(u-\zeta)\f$, so
 *
 * \f[ \int\frac{F'}{v-\omega/k}\mathrm dv
 *     = -\frac{1}{v_{th}^{2}}\cdot\frac{1}{\sqrt\pi}
 *       \int\frac{u\,e^{-u^{2}}}{u-\zeta}\mathrm du
 *     = -\frac{1}{v_{th}^{2}}\bigl[\,1+\zeta Z(\zeta)\,\bigr], \f]
 *
 * using \f$u/(u-\zeta) = 1 + \zeta/(u-\zeta)\f$ and
 * \f$\pi^{-1/2}\int e^{-u^{2}}\mathrm du = 1\f$. Hence
 *
 * \f[ \boxed{\;D(\omega,k) \;=\; 1 \;+\;
 *      \sum_{s}\frac{\omega_{ps}^{2}}{k^{2}v_{th,s}^{2}}
 *      \bigl[\,1+\zeta_{s}Z(\zeta_{s})\,\bigr] \;=\; 0 ,\qquad
 *      \zeta_{s} = \frac{\omega-kv_{d,s}}{\sqrt2\,k\,v_{th,s}} \;}\f]
 *
 * which for a single stationary Maxwellian is exactly the relation quoted in
 * issue #84, since \f$\omega_{p}^{2}/v_{th}^{2} = \lambda_{D}^{-2}\f$.
 *
 * The relation is invariant under a common rescaling of \f$v_{th}\f$,
 * \f$v_{d}\f$, \f$1/k\f$ and \f$1/\omega\f$, which is why the same function
 * serves both the \f$k\lambda_{D}\f$ parametrisation of the Langmuir benchmark
 * and the skin-depth units of the application.
 *
 * @param k wave number, strictly positive (\f$k<0\f$ follows from
 *          \f$D(\omega,-k) = D(-\omega^{*},k)^{*}\f$).
 * @throws std::invalid_argument on \f$k\le0\f$ or \f$v_{th}\le0\f$.
 */
[[nodiscard]] inline DispersionEval
electrostatic_dispersion(Complex omega, double k,
                         const std::vector<MaxwellianBeam> &beams) {
  if (!(k > 0.0)) {
    throw std::invalid_argument("electrostatic_dispersion: k must be > 0");
  }
  DispersionEval out;
  out.value = Complex(1.0, 0.0);
  out.derivative = Complex(0.0, 0.0);
  out.scale = 1.0;
  for (const MaxwellianBeam &b : beams) {
    if (!(b.v_th > 0.0)) {
      throw std::invalid_argument("electrostatic_dispersion: v_th must be > 0");
    }
    const double denom = kSqrt2 * k * b.v_th;
    const double s = b.omega_p2 / (k * k * b.v_th * b.v_th);
    const Complex zeta = (omega - k * b.v_drift) / denom;
    const Complex g = one_plus_zeta_Z(zeta); // 1 + zeta Z
    const Complex Zv = plasma_Z(zeta);
    const Complex dg = Zv + zeta * (-2.0 * g); // d/dzeta (1 + zeta Z)
    out.value += s * g;
    out.derivative += s * dg / denom;
    out.scale = std::max(out.scale, s * std::abs(g));
  }
  return out;
}

/**
 * @brief Bohm-Gross frequency \f$\omega_{r} = \omega_{pe}\sqrt{1+3k^{2}
 *        \lambda_{D}^{2}}\f$.
 *
 * The fluid (cold-plus-adiabatic-pressure) branch: the \f$|\zeta|\to\infty\f$
 * limit of the relation above truncated after \f$3/(4\zeta^{4})\f$. Used as
 * the real part of the Newton initial guess and, in the tests, as the
 * asymptotic form the numerically solved root must approach as
 * \f$k\lambda_{D}\to0\f$. It is **not** an oracle at finite
 * \f$k\lambda_{D}\f$: at \f$k\lambda_{D} = 0.5\f$ it is 1.323 against the true
 * 1.4156, a 7% error, which is precisely why the issue forbids remembering
 * numbers and demands the relation be solved.
 */
[[nodiscard]] inline double bohm_gross_frequency(double k_lambda_D) noexcept {
  return std::sqrt(1.0 + 3.0 * k_lambda_D * k_lambda_D);
}

/**
 * @brief Quoted weak-damping estimate
 *        \f$\gamma \simeq -\sqrt{\pi/8}\,(k\lambda_{D})^{-3}
 *          \exp[-\tfrac12(k\lambda_{D})^{-2}-\tfrac32]\f$.
 *
 * The standard textbook asymptotic (Landau 1946; see e.g. Nicholson,
 * *Introduction to Plasma Theory*, §6.5), obtained by evaluating the residue
 * term of \f$Z\f$ at the Bohm-Gross frequency. It is exponentially small and
 * exponentially wrong outside its regime — at \f$k\lambda_{D}=0.5\f$ it gives
 * \f$-0.1514\f$ against the true \f$-0.1534\f$ (1.3%), and by
 * \f$k\lambda_{D}=1\f$ it is off by a factor of three. Used as an initial
 * guess and as an asymptotic check at small \f$k\lambda_{D}\f$; never as an
 * oracle.
 */
[[nodiscard]] inline double landau_damping_estimate(double k_lambda_D) noexcept {
  const double kl = k_lambda_D;
  if (!(kl > 0.0)) return 0.0;
  const double kl2 = kl * kl;
  return -std::sqrt(std::numbers::pi / 8.0) / (kl2 * kl) *
         std::exp(-0.5 / kl2 - 1.5);
}

/**
 * @brief Documented initial guess for the least-damped Langmuir root.
 *
 * \f$\omega_{0} = \omega_{BG}(k\lambda_{D}) + i\,\gamma_{\text{est}}
 * (k\lambda_{D})\f$, i.e. the fluid frequency plus the weak-damping decrement.
 * It is the correct starting point for the whole range because both factors
 * are asymptotically exact as \f$k\lambda_{D}\to0\f$ and, at the large-
 * \f$k\lambda_{D}\f$ end where they are poor, the true root is strongly damped
 * and \f$D\f$ is smooth there, so Newton's basin is wide. The guess is clamped
 * to \f$\gamma \ge -2\f$ so the first step cannot start deep in the lower half
 * plane where \f$e^{-\zeta^{2}}\f$ is enormous.
 */
[[nodiscard]] inline Complex langmuir_initial_guess(double k_lambda_D) noexcept {
  const double gr = std::max(landau_damping_estimate(k_lambda_D), -2.0);
  return Complex(bohm_gross_frequency(k_lambda_D), gr);
}

/**
 * @brief Solve \f$1 + (k\lambda_{D})^{-2}[1+\zeta Z(\zeta)] = 0\f$ for the
 *        least-damped Langmuir root, \f$\zeta = \omega/(\sqrt2\,k\lambda_{D})\f$.
 *
 * @details
 * In units \f$\omega_{pe}=1\f$ with velocities measured in \f$v_{th}\f$ the
 * relation depends on \f$k\f$ and \f$\lambda_{D}\f$ only through the product,
 * so this takes the single dimensionless argument the literature tabulates.
 * Newton from `langmuir_initial_guess()`, with Muller as a fallback if Newton
 * fails to converge.
 *
 * The returned \f$\gamma\f$ at \f$k\lambda_{D}=0.5\f$ is \f$-0.15336\f$ and the
 * frequency \f$1.41566\f$, matching Canosa's (1973) table — but the table is a
 * *sanity check on this function*, not its oracle. The oracle is
 * `DispersionRoot::relative_residual()`.
 *
 * @warning There is a hard small-\f$k\f$ floor, and it is a property of the
 * problem rather than of the solver. All the information about \f$\gamma\f$
 * sits in the Landau residue, which carries a factor
 * \f$e^{-1/2(k\lambda_{D})^{2}}\f$, while \f$\operatorname{Re}Z\f$ is
 * \f$O(1)\f$; the decrement is therefore invisible in double precision once
 * \f$\exp[-1/2(k\lambda_{D})^{2}] \lesssim \varepsilon\f$, i.e. once
 * \f$k\lambda_{D} \lesssim (2\ln\varepsilon^{-1})^{-1/2} \approx 0.12\f$.
 * The residual stays tiny there because it is insensitive to \f$\gamma\f$
 * exactly where \f$\gamma\f$ is unresolvable, so the residual *cannot* warn
 * about this — which is why it is written down here. Measured margin: at
 * \f$k\lambda_{D}=0.15\f$ the returned \f$\gamma=-8.6\times10^{-9}\f$ is
 * still 7 % of the way to the asymptotic estimate and perfectly resolved; by
 * \f$k\lambda_{D}=0.10\f$ the factor is \f$e^{-50}\approx2\times10^{-22}\f$
 * and it is not. The real part stays accurate throughout. The application must
 * not quote a Landau rate from this function below
 * \f$k\lambda_{D}\approx0.15\f$.
 */
/// \f$k\lambda_{D}\f$ above which the branch is tracked by continuation
/// rather than found directly from the fluid guess.
inline constexpr double kLangmuirSeedWavenumber = 0.5;
/// Continuation step in \f$k\lambda_{D}\f$.
inline constexpr double kLangmuirContinuationStep = 0.05;

[[nodiscard]] inline DispersionRoot
solve_langmuir_root(double k_lambda_D, const RootFinderOptions &opts = {}) {
  if (!(k_lambda_D > 0.0)) {
    throw std::invalid_argument("solve_langmuir_root: k lambda_D must be > 0");
  }
  const auto eval_at = [](double kl) {
    return [kl](Complex w) {
      return electrostatic_dispersion(w, kl, {MaxwellianBeam{1.0, 1.0, 0.0}});
    };
  };

  const double k_seed = std::min(k_lambda_D, kLangmuirSeedWavenumber);
  const Complex guess = langmuir_initial_guess(k_seed);
  DispersionRoot r = newton_root(eval_at(k_seed), guess, opts);
  if (!r.converged) {
    const DispersionRoot m = muller_root(eval_at(k_seed), guess, opts);
    if (m.converged || m.relative_residual() < r.relative_residual()) r = m;
  }
  if (k_lambda_D <= k_seed) return r;

  // Continuation. The relation has infinitely many roots, all in the lower
  // half plane, and above k lambda_D ~ 1.3 the least-damped one is close
  // enough to the next one that Newton started from the fluid guess lands on
  // the *wrong* branch and converges there to full machine precision — a
  // converged solve with a tiny residual that is nonetheless the wrong answer.
  // (Measured: at k lambda_D = 1.40 the direct solve returns
  // omega = 4.504 - 3.705i instead of 2.518 - 1.579i, both exact roots.) The
  // residual therefore cannot detect this and the fix has to be structural:
  // walk the branch up from a wave number where the fluid guess is reliable,
  // in steps small compared with how far the root moves.
  const int steps =
      std::max(1, static_cast<int>(
                      std::ceil((k_lambda_D - k_seed) / kLangmuirContinuationStep)));
  for (int i = 1; i <= steps; ++i) {
    const double kl = k_seed + (k_lambda_D - k_seed) * static_cast<double>(i) /
                                   static_cast<double>(steps);
    DispersionRoot next = newton_root(eval_at(kl), r.omega, opts);
    if (!next.converged) {
      const DispersionRoot m = muller_root(eval_at(kl), r.omega, opts);
      if (m.converged || m.relative_residual() < next.relative_residual()) {
        next = m;
      }
    }
    r = next;
  }
  return r;
}

// ---------------------------------------------------------------------------
// 4. Two-stream
// ---------------------------------------------------------------------------

/**
 * @brief Cold two-beam electrostatic dispersion function,
 *        \f$D = 1 - \omega_{b}^{2}[(\omega-kv_{0})^{-2}+(\omega+kv_{0})^{-2}]\f$.
 *
 * @details
 * ### Derivation
 * The \f$v_{th}\to0\f$ limit of `electrostatic_dispersion()`: for a cold beam
 * \f$F=\delta(v-v_{d})\f$, \f$\int F'/(v-\omega/k)\mathrm dv =
 * -\int F\,\partial_{v}(v-\omega/k)^{-1}\mathrm dv = (v_{d}-\omega/k)^{-2}\f$,
 * so \f$D = 1 - \sum_{j}\omega_{pj}^{2}/(\omega-kv_{dj})^{2}\f$. With two
 * counter-streaming beams of equal density, \f$v_{d}=\pm v_{0}\f$ and
 * \f$\omega_{pj}^{2}=\omega_{b}^{2}\f$, that is the expression above.
 *
 * @param k_v0 the product \f$k v_{0}\f$ (a frequency, in \f$\omega_{pe}\f$).
 * @param omega_p2_each \f$\omega_{b}^{2}\f$ per beam; 0.5 gives a total
 *        \f$\omega_{pe}=1\f$.
 */
[[nodiscard]] inline DispersionEval
cold_two_stream_dispersion(Complex omega, double k_v0, double omega_p2_each = 0.5) {
  const Complex dp = omega - k_v0;
  const Complex dm = omega + k_v0;
  DispersionEval out;
  out.value = 1.0 - omega_p2_each * (1.0 / (dp * dp) + 1.0 / (dm * dm));
  out.derivative =
      2.0 * omega_p2_each * (1.0 / (dp * dp * dp) + 1.0 / (dm * dm * dm));
  out.scale = std::max(1.0, omega_p2_each *
                                std::max(1.0 / std::norm(dp), 1.0 / std::norm(dm)));
  return out;
}

/// The four closed-form roots of the cold two-beam relation, plus the growth rate.
struct ColdTwoStreamRoots {
  std::array<Complex, 4> omega{};
  /// Largest \f$\operatorname{Im}\omega\f$ over the four roots.
  double growth_rate{0.0};
  bool unstable{false};
};

/**
 * @brief Closed-form roots of the cold two-beam relation.
 *
 * @details
 * ### Derivation
 * With \f$a=\omega^{2}\f$, \f$b=(kv_{0})^{2}\f$,
 * \f[ \frac{1}{(\omega-kv_0)^{2}}+\frac{1}{(\omega+kv_0)^{2}}
 *     = \frac{2(a+b)}{(a-b)^{2}} , \f]
 * so \f$D=0\f$ becomes the *quadratic in \f$a\f$*
 * \f[ (a-b)^{2} = 2\omega_{b}^{2}(a+b)
 *     \quad\Longrightarrow\quad
 *     a^{2} - 2(b+\omega_{b}^{2})a + (b^{2}-2\omega_{b}^{2}b) = 0 , \f]
 * \f[ \omega^{2} \;=\; b+\omega_{b}^{2}
 *     \;\pm\;\sqrt{\omega_{b}^{4}+4\omega_{b}^{2}b} . \f]
 * The minus branch is negative — hence a purely growing root
 * \f$\omega=i\gamma\f$ — exactly when \f$b < 2\omega_{b}^{2}\f$, i.e.
 * \f$kv_{0} < \sqrt2\,\omega_{b} = \omega_{pe}\f$ for equal beams. Writing
 * \f$b=\beta\omega_{b}^{2}\f$ gives \f$\omega^{2}/\omega_{b}^{2} =
 * \beta+1-\sqrt{4\beta+1}\f$, minimised at \f$\beta=3/4\f$ with value
 * \f$-1/4\f$: the classic
 * \f$\gamma_{\max}=\omega_{b}/2 = \omega_{pe}/(2\sqrt2)\f$ at
 * \f$kv_{0}=\tfrac{\sqrt3}{2}\omega_{b}\f$ (see `cold_two_stream_peak()`).
 *
 * The subtraction in the minus branch is done in the cancellation-free form
 * \f$(\text{product of roots})/(\text{plus branch})\f$ so that small
 * \f$\omega_{b}\f$ does not lose digits.
 */
[[nodiscard]] inline ColdTwoStreamRoots
cold_two_stream_roots(double k_v0, double omega_p2_each = 0.5) {
  const double b = k_v0 * k_v0;
  const double wb2 = omega_p2_each;
  const double disc = std::sqrt(wb2 * wb2 + 4.0 * wb2 * b);
  const double a_plus = b + wb2 + disc;
  const double prod = b * b - 2.0 * wb2 * b; // product of the two a-roots
  const double a_minus = (a_plus != 0.0) ? prod / a_plus : (b + wb2 - disc);

  ColdTwoStreamRoots out;
  const Complex rp = std::sqrt(Complex(a_plus, 0.0));
  const Complex rm = std::sqrt(Complex(a_minus, 0.0));
  out.omega = {rp, -rp, rm, -rm};
  for (const Complex &w : out.omega) {
    out.growth_rate = std::max(out.growth_rate, w.imag());
  }
  out.unstable = out.growth_rate > 0.0;
  return out;
}

/// Peak of the cold two-beam growth rate: \f$kv_{0}=\tfrac{\sqrt3}{2}\omega_{b}\f$,
/// \f$\gamma_{\max}=\omega_{b}/2\f$. Returns `{k_v0, gamma}`.
[[nodiscard]] inline std::array<double, 2>
cold_two_stream_peak(double omega_p2_each = 0.5) noexcept {
  const double wb = std::sqrt(omega_p2_each);
  return {0.5 * std::sqrt(3.0) * wb, 0.5 * wb};
}

/// Two symmetric counter-streaming Maxwellians, the warm two-stream setup.
struct TwoStreamMaxwellians {
  double v_drift{0.1};       ///< \f$\pm v_{0}\f$, in \f$c\f$.
  double v_th{0.01};         ///< rms thermal speed of each beam, in \f$c\f$.
  double omega_p2_each{0.5}; ///< \f$\omega_{b}^{2}\f$; 0.5 + 0.5 = 1.
};

/// The two `MaxwellianBeam`s that `TwoStreamMaxwellians` stands for.
[[nodiscard]] inline std::vector<MaxwellianBeam>
two_stream_beams(const TwoStreamMaxwellians &p) {
  return {MaxwellianBeam{p.omega_p2_each, p.v_th, +p.v_drift},
          MaxwellianBeam{p.omega_p2_each, p.v_th, -p.v_drift}};
}

/**
 * @brief Solve the kinetic two-stream relation for its purely growing root.
 *
 * @details
 * The relation is `electrostatic_dispersion()` with the two beams of
 * `two_stream_beams()`, i.e.
 * \f[ D = 1 + \frac{\omega_{b}^{2}}{k^{2}v_{th}^{2}}
 *      \Bigl[\,2 + \zeta_{+}Z(\zeta_{+}) + \zeta_{-}Z(\zeta_{-})\Bigr],
 *    \qquad \zeta_{\pm} = \frac{\omega \mp kv_{0}}{\sqrt2\,k\,v_{th}} . \f]
 *
 * ### Why the root is on the imaginary axis, and why that is exploited
 * For \f$\omega=i\gamma\f$, \f$\zeta_{+} = -\alpha+i\beta\f$ and
 * \f$\zeta_{-} = +\alpha+i\beta = -\overline{\zeta_{+}}\f$ with
 * \f$\alpha = v_{0}/\sqrt2 v_{th}\f$, \f$\beta=\gamma/\sqrt2 kv_{th}\f$. Since
 * \f$Z(-\bar\zeta) = -\overline{Z(\zeta)}\f$ (immediate from
 * \f$w(-\bar z)=\overline{w(z)}\f$), the two beam terms are complex conjugates
 * and their sum is real: **\f$D(i\gamma)\f$ is real** for the symmetric
 * configuration. So the unstable root is found by bracketing and bisection on
 * \f$\gamma\f$ — no complex search, no possibility of landing on the wrong
 * branch — and only then polished by complex Newton, which both refines it and
 * *verifies* that \f$\operatorname{Re}\omega\f$ stays at round-off rather than
 * assuming it.
 *
 * \f$D(i\gamma)\to1>0\f$ as \f$\gamma\to\infty\f$ (all the \f$1+\zeta Z\f$
 * terms decay like \f$\zeta^{-2}\f$), so a sign change exists iff
 * \f$D(0)<0\f$, which is the Penrose criterion for this configuration. If
 * there is no sign change the configuration is stable and the returned root
 * has `converged == false`.
 *
 * @param k wave number in the same units as \f$1/v\f$ (inverse skin depths
 *          when velocities are in \f$c\f$).
 */
[[nodiscard]] inline DispersionRoot
solve_two_stream_root(double k, const TwoStreamMaxwellians &p,
                      const RootFinderOptions &opts = {}) {
  const std::vector<MaxwellianBeam> beams = two_stream_beams(p);
  const auto eval = [&](Complex w) { return electrostatic_dispersion(w, k, beams); };
  const auto real_D = [&](double g) { return eval(Complex(0.0, g)).value.real(); };

  // Search window. The cold growth rate is a good first guess at the scale but
  // it is *not* an upper bound: near the cold marginal point kv0 = omega_pe a
  // warm beam pair is still unstable where the cold pair is not (measured:
  // at kv0 = 0.95, v_th = 0.05 v_0, the warm rate 0.1810 exceeds the cold
  // 0.1761, and the warm band extends past kv0 = 1). So the window is grown
  // until D(i gamma) has turned positive at its top, which is guaranteed to
  // happen because every 1 + zeta Z term decays like zeta^{-2} and D -> 1.
  const ColdTwoStreamRoots cold =
      cold_two_stream_roots(k * p.v_drift, p.omega_p2_each);
  double window = std::max(2.0 * cold.growth_rate, 0.2 * std::sqrt(p.omega_p2_each));
  for (int i = 0; i < 24 && !(real_D(window) > 0.0); ++i) window *= 2.0;

  const GrowthBracket br = bracket_growing_root(real_D, window, 1024, 0.0);
  if (!br.found) {
    DispersionRoot out;
    out.omega = Complex(0.0, 0.0);
    const DispersionEval e = eval(out.omega);
    out.residual = std::abs(e.value);
    out.residual_scale = std::max(e.scale, 1.0);
    out.converged = false;
    return out;
  }
  const double gamma0 = bisect_growing_root(real_D, br.lo, br.hi);
  return newton_root(eval, Complex(0.0, gamma0), opts);
}

// ---------------------------------------------------------------------------
// 5. Transverse electromagnetic: Weibel and cold filamentation
// ---------------------------------------------------------------------------

/// A bi-Maxwellian electron population: \f$T_{y}\ne T_{x}\f$, no drift.
struct BiMaxwellian {
  /// rms thermal speed along \f$\mathbf k\f$ (the \f$x\f$ axis), in \f$c\f$.
  double v_th_x{0.05};
  /// rms thermal speed along \f$\mathbf E\f$ (the \f$y\f$ axis), in \f$c\f$.
  double v_th_y{0.10};
  /// \f$\omega_{pe}^{2}\f$ in units of itself; kept as a parameter so a
  /// reduced-density species can be described without rescaling everything.
  double omega_p2{1.0};

  /// Anisotropy \f$A = T_{y}/T_{x} = v_{th,y}^{2}/v_{th,x}^{2}\f$.
  [[nodiscard]] double anisotropy() const noexcept {
    return (v_th_y * v_th_y) / (v_th_x * v_th_x);
  }
};

/**
 * @brief Transverse (Weibel / ordinary-mode) dispersion function in the exact
 *        1D2V geometry of the application. **Derived here, not quoted.**
 *
 * @details
 * ### Geometry
 * \f$\mathbf k = k\hat x\f$, \f$\mathbf E = E_{y}\hat y\f$,
 * \f$\mathbf B = B_{z}\hat z\f$, everything \f$\propto e^{i(kx-\omega t)}\f$.
 * These are exactly the surviving field components of the 1D2V reduction in
 * `vlasov_maxwell/parameters.hpp`, and the equilibrium is the separable
 * bi-Maxwellian \f$f_{0}=n_{0}g(v_{x})h(v_{y})\f$ with rms widths
 * \f$v_{th,x}\f$, \f$v_{th,y}\f$. That \f$f_{0}\f$ has no \f$v_{z}\f$
 * coordinate; nothing below ever integrates over one, so the 1D2V closure
 * changes nothing in this derivation — which is worth stating, because the
 * textbook version is written in 3V and it is not obvious by inspection that
 * the answer is the same.
 *
 * ### Step 1 — Faraday ties \f$B_{z}\f$ to \f$E_{y}\f$
 * \f$\partial_{t}\mathbf B = -\nabla\times\mathbf E\f$ with
 * \f$(\nabla\times\mathbf E)_{z} = \partial_{x}E_{y} = ikE_{y}\f$ gives
 * \f$-i\omega B_{z} = -ikE_{y}\f$, i.e. \f$B_{z} = (k/\omega)E_{y}\f$.
 *
 * ### Step 2 — the linearised Vlasov response
 * The perturbed Lorentz force is
 * \f$(q/m)[(v_{y}B_{z})\hat x + (E_{y}-v_{x}B_{z})\hat y]\f$ — the same
 * cross-product components \f$(v\times B)_{x}=+v_{y}B_{z}\f$,
 * \f$(v\times B)_{y}=-v_{x}B_{z}\f$ the stepper implements. Substituting
 * \f$B_{z}=kE_{y}/\omega\f$ and solving
 * \f$i(kv_{x}-\omega)f_{1} = -(q/m)[\cdots]\f$,
 *
 * \f[ f_{1} = \frac{i(q/m)E_{y}}{kv_{x}-\omega}
 *      \Bigl[\frac{kv_{y}}{\omega}\partial_{v_{x}}f_{0}
 *          + \Bigl(1-\frac{kv_{x}}{\omega}\Bigr)\partial_{v_{y}}f_{0}\Bigr]. \f]
 *
 * ### Step 3 — the transverse current, in two pieces
 * \f$J_{y}=q\int v_{y}f_{1}\,\mathrm d^{2}v\f$.
 *
 * The \f$\partial_{v_{y}}f_{0}\f$ piece is *exact and geometry-free*: because
 * \f$(1-kv_{x}/\omega)/(kv_{x}-\omega) = -1/\omega\f$ identically, the
 * resonant denominator cancels and an integration by parts in \f$v_{y}\f$
 * leaves \f$\int v_{y}\partial_{v_{y}}f_{0} = -n_{0}\f$, giving
 * \f$+n_{0}/\omega\f$. This is the current of unmagnetised oscillation — the
 * \f$-\omega_{pe}^{2}\f$ in the final relation — and it carries no \f$Z\f$.
 *
 * The \f$\partial_{v_{x}}f_{0}\f$ piece is the kinetic one:
 * \f$(k/\omega)\int v_{y}^{2}\,\partial_{v_{x}}f_{0}/(kv_{x}-\omega)\f$.
 * Separability factors it into \f$\langle v_{y}^{2}\rangle = v_{th,y}^{2}\f$
 * times the same longitudinal integral that appeared in
 * `electrostatic_dispersion()`, namely
 * \f$\int g'/(v_{x}-\omega/k) = -v_{th,x}^{-2}[1+\zeta Z(\zeta)]\f$, so it
 * equals \f$-(n_{0}v_{th,y}^{2}/\omega v_{th,x}^{2})[1+\zeta Z(\zeta)]\f$.
 * **The \f$v_{y}^{2}\f$ weight is where the anisotropy enters, and it enters
 * only as the ratio \f$A=v_{th,y}^{2}/v_{th,x}^{2}\f$.**
 *
 * Hence \f$J_{y} = i(q^{2}n_{0}/m\omega)E_{y}\{1 - A[1+\zeta Z(\zeta)]\}\f$.
 *
 * ### Step 4 — Ampère closes it
 * \f$\partial_{t}E_{y} = c^{2}(\nabla\times\mathbf B)_{y} -
 * J_{y}/\varepsilon_{0}\f$ with \f$(\nabla\times\mathbf B)_{y}=-ikB_{z}\f$
 * gives \f$(\omega^{2}-c^{2}k^{2})E_{y} = -i\omega J_{y}/\varepsilon_{0}\f$,
 * and \f$-i\omega J_{y}/\varepsilon_{0}E_{y} =
 * -\omega_{pe}^{2}\{1-A[1+\zeta Z]\}\f$. Therefore
 *
 * \f[ \boxed{\;D_{T}(\omega,k) \;=\; \omega^{2} - c^{2}k^{2}
 *      - \omega_{pe}^{2}\Bigl\{\,1 - A\,\bigl[1+\zeta Z(\zeta)\bigr]\Bigr\}
 *      \;=\;0 ,\qquad
 *      \zeta=\frac{\omega}{\sqrt2\,k\,v_{th,x}},\quad
 *      A=\frac{T_{y}}{T_{x}} \;}\f]
 *
 * ### Three checks that this is the right relation
 * 1. **Isotropic limit** \f$A=1\f$: \f$\omega^{2}=c^{2}k^{2}
 *    -\omega_{pe}^{2}\zeta Z\f$, and \f$1+\zeta Z\to-k^{2}v_{th,x}^{2}
 *    /\omega^{2}\f$ gives the biquadratic \f$\omega^{4}
 *    -(\omega_{pe}^{2}+c^{2}k^{2})\omega^{2}
 *    -\omega_{pe}^{2}k^{2}v_{th,x}^{2}=0\f$, i.e. the electromagnetic branch
 *    \f$\omega^{2}=\omega_{pe}^{2}+c^{2}k^{2}\f$ raised by a positive
 *    thermal correction \f$k^{2}v_{th,x}^{2}\omega_{pe}^{2}
 *    /(\omega_{pe}^{2}+c^{2}k^{2})\f$. This is the check that fixes the signs
 *    of the two non-kinetic terms.
 * 2. **Marginal wave number**: on the imaginary axis \f$\omega=i\gamma\f$,
 *    \f$\zeta=i\xi\f$ and \f$1+\zeta Z = 1-\sqrt\pi\,\xi e^{\xi^{2}}
 *    \operatorname{erfc}\xi \equiv \mathcal G(\xi)\f$ is real, with
 *    \f$\mathcal G(0)=1\f$, \f$\mathcal G\f$ strictly decreasing and
 *    \f$\mathcal G(\infty)=0\f$. So
 *    \f$D_{T}(i\gamma) = -\gamma^{2}-c^{2}k^{2}-\omega_{pe}^{2}
 *    +\omega_{pe}^{2}A\,\mathcal G(\xi)\f$ is **strictly decreasing in
 *    \f$\gamma\f$** (both terms are), runs to \f$-\infty\f$, and starts at
 *    \f$\omega_{pe}^{2}(A-1)-c^{2}k^{2}\f$. Hence *exactly one* purely growing
 *    root exists, iff
 *    \f[ c^{2}k^{2} < \omega_{pe}^{2}\,(A-1)
 *        = \omega_{pe}^{2}\Bigl(\frac{T_{\perp}}{T_{\parallel}}-1\Bigr), \f]
 *    which is Weibel's published instability criterion (Weibel 1959). This
 *    existence-and-uniqueness argument is also what makes the bisection in
 *    `solve_weibel_root()` unconditionally reliable.
 * 3. **Cold counter-streaming limit**: \f$v_{th,x}\to0\f$ at fixed
 *    \f$v_{th,y}\f$ sends \f$|\zeta|\to\infty\f$ and
 *    \f$A[1+\zeta Z]\to -Ak^{2}v_{th,x}^{2}/\omega^{2}
 *    = -k^{2}v_{th,y}^{2}/\omega^{2}\f$, reproducing
 *    `cold_filamentation_dispersion()` with \f$u_{0}=v_{th,y}\f$ — an
 *    independently derived closed form (see there). The suite checks this
 *    numerically, not just symbolically.
 *
 * @note Non-relativistic throughout, consistent with the application. The
 *       relativistic filamentation problem has extra \f$\gamma_{L}\f$ factors
 *       and this relation must not be used at drift speeds where they matter.
 */
[[nodiscard]] inline DispersionEval weibel_dispersion(Complex omega, double k,
                                                      const BiMaxwellian &p) {
  if (!(k > 0.0)) throw std::invalid_argument("weibel_dispersion: k must be > 0");
  if (!(p.v_th_x > 0.0)) {
    throw std::invalid_argument("weibel_dispersion: v_th_x must be > 0");
  }
  const double A = p.anisotropy();
  const double denom = kSqrt2 * k * p.v_th_x;
  const Complex zeta = omega / denom;
  const Complex g = one_plus_zeta_Z(zeta);
  const Complex Zv = plasma_Z(zeta);
  const Complex dg = Zv + zeta * (-2.0 * g);

  DispersionEval out;
  out.value = omega * omega - k * k - p.omega_p2 * (1.0 - A * g);
  out.derivative = 2.0 * omega + p.omega_p2 * A * dg / denom;
  out.scale =
      std::max({std::norm(omega), k * k, p.omega_p2, p.omega_p2 * A * std::abs(g)});
  return out;
}

/**
 * @brief Marginal wave number \f$k_{c} = \omega_{pe}\sqrt{A-1}/c\f$; the mode
 *        is Weibel-unstable iff \f$k<k_{c}\f$. Zero when \f$A\le1\f$.
 */
[[nodiscard]] inline double
weibel_cutoff_wavenumber(const BiMaxwellian &p) noexcept {
  const double A = p.anisotropy();
  return (A > 1.0) ? std::sqrt(p.omega_p2 * (A - 1.0)) : 0.0;
}

/**
 * @brief The purely growing Weibel root by bisection on the imaginary axis
 *        alone — the independent oracle.
 *
 * @details
 * Uses nothing but the monotonicity proved in `weibel_dispersion()`: bracket
 * \f$[0,\gamma_{\max}]\f$ and bisect to adjacent doubles. No initial guess, no
 * derivative, no possibility of converging to a different root. Returns 0 when
 * \f$k \ge k_{c}\f$. `solve_weibel_root()` seeds Newton with this and then
 * confirms the two agree; a test asserts it.
 *
 * The search ceiling is \f$\gamma_{\max} = \omega_{pe}\sqrt{A}\,\f$, which
 * bounds the root because \f$D_{T}(i\gamma)\le -\gamma^{2}
 * +\omega_{pe}^{2}A\f$ (using \f$\mathcal G\le1\f$), so \f$D_{T}<0\f$ for
 * \f$\gamma>\omega_{pe}\sqrt A\f$ and the root is already behind us.
 */
[[nodiscard]] inline double weibel_growth_by_bisection(double k,
                                                       const BiMaxwellian &p) {
  if (k >= weibel_cutoff_wavenumber(p)) return 0.0;
  const auto real_D = [&](double g) {
    return weibel_dispersion(Complex(0.0, g), k, p).value.real();
  };
  const double ceiling =
      std::sqrt(p.omega_p2 * p.anisotropy()) * 1.0000001 + 1.0e-12;
  return bisect_growing_root(real_D, 0.0, ceiling);
}

/**
 * @brief Solve the transverse relation for the purely growing Weibel root.
 *
 * @details
 * Bisection on the imaginary axis for a guess that is already correct to
 * machine precision, then complex Newton to produce the residual and — the
 * point of doing it in the complex plane at all — to let
 * \f$\operatorname{Re}\omega\f$ be an *output* rather than an assumption. On
 * the imaginary axis \f$D_{T}\f$ is real and \f$\partial_{\omega}D_{T}\f$ is
 * purely imaginary, so the Newton step is purely imaginary and the iteration
 * preserves the axis exactly; a returned \f$\operatorname{Re}\omega\f$ that is
 * *not* at round-off would therefore mean the relation is not what this header
 * claims it is.
 *
 * Returns `converged == false` with \f$\omega=0\f$ for \f$k\ge k_{c}\f$: that
 * is the stable mode the validation ladder requires, and "no root found" is
 * the honest answer, not \f$\gamma=0\f$ dressed up as a converged solve.
 */
[[nodiscard]] inline DispersionRoot
solve_weibel_root(double k, const BiMaxwellian &p,
                  const RootFinderOptions &opts = {}) {
  const auto eval = [&](Complex w) { return weibel_dispersion(w, k, p); };
  if (k >= weibel_cutoff_wavenumber(p)) {
    DispersionRoot out;
    out.omega = Complex(0.0, 0.0);
    const DispersionEval e = eval(out.omega);
    out.residual = std::abs(e.value);
    out.residual_scale = std::max(e.scale, 1.0);
    out.converged = false;
    return out;
  }
  const double gamma0 = weibel_growth_by_bisection(k, p);
  return newton_root(eval, Complex(0.0, gamma0), opts);
}

/**
 * @brief Cold counter-streaming (filamentation) dispersion function,
 *        \f$D = \omega^{2}-c^{2}k^{2}
 *        -\omega_{pe}^{2}\bigl(1+k^{2}u_{0}^{2}/\omega^{2}\bigr)\f$.
 *
 * @details
 * ### Derivation
 * Repeat steps 1-4 of `weibel_dispersion()` with
 * \f$f_{0}=\tfrac{n_{0}}{2}\delta(v_{x})[\delta(v_{y}-u_{0})
 * +\delta(v_{y}+u_{0})]\f$: two cold beams counter-streaming *along
 * \f$\hat y\f$*, wave vector across the flow. Step 3's
 * \f$\partial_{v_{y}}f_{0}\f$ piece is unchanged (it never used the form of
 * \f$f_{0}\f$). Its \f$\partial_{v_{x}}f_{0}\f$ piece becomes, after one
 * integration by parts in \f$v_{x}\f$,
 * \f$(k/\omega)\langle v_{y}^{2}\rangle n_{0}\int f_{0}\,k/(kv_{x}-\omega)^{2}
 * = k^{2}n_{0}u_{0}^{2}/\omega^{3}\f$, giving the relation above.
 *
 * This is the same function as the \f$v_{th,x}\to0\f$ limit of the
 * bi-Maxwellian relation with \f$u_{0}^{2}=\langle v_{y}^{2}\rangle\f$ — two
 * genuinely different derivations that must agree, and do.
 *
 * @param u_0 counter-streaming speed, in \f$c\f$.
 */
[[nodiscard]] inline DispersionEval
cold_filamentation_dispersion(Complex omega, double k, double u_0,
                              double omega_p2 = 1.0) {
  DispersionEval out;
  const Complex w2 = omega * omega;
  out.value = w2 - k * k - omega_p2 * (1.0 + k * k * u_0 * u_0 / w2);
  out.derivative = 2.0 * omega + 2.0 * omega_p2 * k * k * u_0 * u_0 / (w2 * omega);
  out.scale = std::max(
      {std::abs(w2), k * k, omega_p2, omega_p2 * k * k * u_0 * u_0 / std::abs(w2)});
  return out;
}

/**
 * @brief Closed-form growth rate of the cold filamentation mode.
 *
 * @details
 * Multiplying \f$D=0\f$ by \f$\omega^{2}\f$ gives the biquadratic
 * \f$\omega^{4}-(c^{2}k^{2}+\omega_{pe}^{2})\omega^{2}
 * -\omega_{pe}^{2}k^{2}u_{0}^{2}=0\f$, whose lower root is negative for every
 * \f$k>0\f$ — the cold beams are unstable at *all* wave numbers, with no
 * cutoff, which is exactly the piece of physics that the thermal
 * \f$v_{th,x}\f$ restores. Hence
 * \f[ \gamma^{2} \;=\; \frac{\sqrt{(c^{2}k^{2}+\omega_{pe}^{2})^{2}
 *       + 4\omega_{pe}^{2}k^{2}u_{0}^{2}} - (c^{2}k^{2}+\omega_{pe}^{2})}{2}
 *   \;=\; \frac{2\omega_{pe}^{2}k^{2}u_{0}^{2}}
 *        {\sqrt{\cdots}+(c^{2}k^{2}+\omega_{pe}^{2})} , \f]
 * the second form being the one evaluated here, because the first is a
 * difference of nearly equal numbers at small \f$k\f$. As
 * \f$k\to\infty\f$, \f$\gamma\to(u_{0}/c)\,\omega_{pe}\f$ — the standard
 * saturation of the cold filamentation growth rate at \f$\beta\omega_{pe}\f$
 * (Bret, Gremillet & Dieckmann 2010, §III).
 */
[[nodiscard]] inline double
cold_filamentation_growth_rate(double k, double u_0,
                               double omega_p2 = 1.0) noexcept {
  const double s = k * k + omega_p2; // c = 1
  const double q = 4.0 * omega_p2 * k * k * u_0 * u_0;
  if (q <= 0.0) return 0.0;
  const double root = std::sqrt(s * s + q);
  return std::sqrt(q / (2.0 * (root + s)));
}

} // namespace pfc::apps::plasma
