// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_plasma_dispersion.cpp
 * @brief Verification ladder for `openpfc_apps/plasma_dispersion.hpp`.
 *
 * @details
 * These are the *oracles* of the Vlasov-Maxwell validation ladder (issue #84),
 * so they are held to a higher standard than the code they will judge: a
 * dispersion solver that is quietly wrong does not fail, it certifies a broken
 * simulation. Nothing here compares the header against itself, and nothing
 * here asserts on a remembered number except where the comment says in so many
 * words that the assertion is a sanity check rather than an oracle.
 *
 * Every reference used below is independent of the implementation:
 *
 *  1. \f$w(0)=1\f$ and \f$Z(0)=i\sqrt\pi\f$ — exact, from the definition.
 *  2. The Taylor series \f$w(z)=\sum_{n}(iz)^{n}/\Gamma(n/2+1)\f$ — converges
 *     everywhere, so it is a reference in the **lower** half plane, which is
 *     the half the implementation reaches by a reflection identity and the
 *     half every damped root lives in.
 *  3. \f$Z(i\xi)=i\sqrt\pi\,e^{\xi^{2}}\operatorname{erfc}\xi\f$ from
 *     `std::erfc` — a completely different library function.
 *  4. A directly evaluated principal-value integral plus the Plemelj residue,
 *     on the real axis.
 *  5. A Cauchy contour derivative of \f$Z\f$, for \f$Z'\f$.
 *  6. The Fried-Conte asymptotic expansion, including the Stokes multiplier.
 *  7. For the root finders: the **residual** \f$|D(\omega,k)|\f$ relative to
 *     the largest term in \f$D\f$. This is the self-test the issue demands.
 *  8. Two structurally different iterations (damped Newton with an analytic
 *     derivative; derivative-free Muller) landing on the same complex number.
 *  9. For the purely growing modes, a bracketed bisection on the imaginary
 *     axis — no derivative, no guess, no wrong branch to converge to.
 * 10. Closed forms in the cold limits, reached by two independent derivations,
 *     approached at the documented order in the thermal spread.
 *
 * No MPI, no fields, no FFT: this translation unit is pure arithmetic and runs
 * in well under a second on one core.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include <openpfc_apps/plasma_dispersion.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

using pfc::apps::plasma::BiMaxwellian;
using pfc::apps::plasma::bohm_gross_frequency;
using pfc::apps::plasma::bracket_growing_root;
using pfc::apps::plasma::cold_filamentation_dispersion;
using pfc::apps::plasma::cold_filamentation_growth_rate;
using pfc::apps::plasma::cold_two_stream_dispersion;
using pfc::apps::plasma::cold_two_stream_peak;
using pfc::apps::plasma::cold_two_stream_roots;
using pfc::apps::plasma::Complex;
using pfc::apps::plasma::DispersionEval;
using pfc::apps::plasma::DispersionRoot;
using pfc::apps::plasma::electrostatic_dispersion;
using pfc::apps::plasma::faddeeva_w;
using pfc::apps::plasma::kSqrt2;
using pfc::apps::plasma::kSqrtPi;
using pfc::apps::plasma::landau_damping_estimate;
using pfc::apps::plasma::langmuir_initial_guess;
using pfc::apps::plasma::MaxwellianBeam;
using pfc::apps::plasma::muller_root;
using pfc::apps::plasma::newton_root;
using pfc::apps::plasma::one_plus_zeta_Z;
using pfc::apps::plasma::plasma_Z;
using pfc::apps::plasma::plasma_Z_asymptotic;
using pfc::apps::plasma::plasma_Zprime;
using pfc::apps::plasma::solve_langmuir_root;
using pfc::apps::plasma::solve_two_stream_root;
using pfc::apps::plasma::solve_weibel_root;
using pfc::apps::plasma::TwoStreamMaxwellians;
using pfc::apps::plasma::weibel_cutoff_wavenumber;
using pfc::apps::plasma::weibel_dispersion;
using pfc::apps::plasma::weibel_growth_by_bisection;

namespace {

constexpr double kPi = std::numbers::pi;

/// Format a diagnostic line at full precision (Catch's INFO default is 6 digits).
template <class... Ts> std::string precise(Ts &&...parts) {
  std::ostringstream oss;
  oss << std::setprecision(14);
  (oss << ... << parts);
  return oss.str();
}

std::string cstr(Complex z) { return precise("(", z.real(), ", ", z.imag(), ")"); }

double relerr(Complex value, Complex reference) {
  return std::abs(value - reference) / std::max(std::abs(reference), 1.0e-300);
}

/**
 * @brief Reference \f$w(z)\f$ from its Taylor series,
 *        \f$w(z)=\sum_{n\ge0}(iz)^{n}/\Gamma(n/2+1)\f$.
 *
 * @details
 * The series has infinite radius of convergence — it is *not* an asymptotic
 * form and it knows nothing about half planes — which is exactly what makes it
 * a legitimate reference below the real axis, where `faddeeva_w()` reaches its
 * answer through the reflection \f$w(z)=2e^{-z^{2}}-w(-z)\f$. Its own accuracy
 * is limited by cancellation: the largest term is \f$O(e^{|z|^{2}/2})\f$ while
 * the sum is \f$O(1)\f$, which costs about four digits at \f$|z|=2.5\f$ and is
 * why the tolerances against it are 1e-11 rather than 1e-14. (Measured
 * disagreement is 2.7e-13, and it does not improve when the Weideman order is
 * raised from 32 to 64, which identifies it as the *series'* error, not the
 * implementation's.)
 */
Complex w_taylor(Complex z, int n_terms = 90) {
  const Complex iz(-z.imag(), z.real());
  Complex term(1.0, 0.0);
  Complex sum(0.0, 0.0);
  for (int n = 0; n < n_terms; ++n) {
    sum += term / std::tgamma(0.5 * static_cast<double>(n) + 1.0);
    term *= iz;
  }
  return sum;
}

/**
 * @brief Reference \f$Z(x)\f$ on the real axis, from the Plemelj decomposition.
 *
 * @details
 * \f$\lim_{\epsilon\to0^{+}}(t-x-i\epsilon)^{-1}
 *    = \mathcal P(t-x)^{-1} + i\pi\delta(t-x)\f$, so
 * \f[ Z(x) = \frac{1}{\sqrt\pi}\,\mathcal P\!\!
 *      \int\frac{e^{-t^{2}}}{t-x}\mathrm dt \;+\; i\sqrt\pi\,e^{-x^{2}} . \f]
 * Folding the principal value about \f$t=x\f$ removes the singularity
 * analytically,
 * \f[ \mathcal P\!\!\int\frac{e^{-t^{2}}}{t-x}\mathrm dt
 *     = \int_{0}^{\infty}\frac{e^{-(x+s)^{2}}-e^{-(x-s)^{2}}}{s}\,\mathrm ds ,\f]
 * whose integrand is smooth, tends to \f$-4xe^{-x^{2}}\f$ at \f$s=0\f$ and
 * decays like \f$e^{-s^{2}}\f$. Composite Simpson on \f$[0,14+|x|]\f$ with
 * 200 000 panels is then good to ~1e-13, which is what limits this reference.
 */
Complex pv_Z_real_axis(double x) {
  const double span = 14.0 + std::abs(x);
  const int n_panels = 200000; // even
  const double h = span / static_cast<double>(n_panels);
  const auto integrand = [x](double s) {
    if (s == 0.0) return -4.0 * x * std::exp(-x * x);
    return (std::exp(-(x + s) * (x + s)) - std::exp(-(x - s) * (x - s))) / s;
  };
  double acc = integrand(0.0) + integrand(span);
  for (int i = 1; i < n_panels; ++i) {
    acc += ((i % 2) ? 4.0 : 2.0) * integrand(static_cast<double>(i) * h);
  }
  return Complex(acc * h / 3.0 / kSqrtPi, kSqrtPi * std::exp(-x * x));
}

/**
 * @brief Numerical derivative by the Cauchy integral formula.
 *
 * \f$f'(z)=\frac{1}{2\pi i}\oint\frac{f(s)}{(s-z)^{2}}\mathrm ds\f$ on a
 * circle of radius \f$r\f$, discretised by the trapezoidal rule, collapses to
 * \f$f'(z)\approx\frac{1}{nr}\sum_{j}f(z+re^{i\theta_{j}})e^{-i\theta_{j}}\f$.
 * For an entire function the trapezoidal rule on a circle converges
 * geometrically, so this is accurate to round-off (times \f$\max|f|/r\f$) —
 * unlike a finite difference, whose truncation error is set by \f$Z^{(5)}\f$
 * and which only reaches 5e-7 here.
 */
template <class F>
Complex cauchy_derivative(F &&f, Complex z, double r = 0.5, int n = 64) {
  Complex acc(0.0, 0.0);
  for (int j = 0; j < n; ++j) {
    const double theta = 2.0 * kPi * static_cast<double>(j) / static_cast<double>(n);
    const Complex e(std::cos(theta), std::sin(theta));
    acc += f(z + r * e) / e;
  }
  return acc / (static_cast<double>(n) * r);
}

/// The single-Maxwellian electrostatic relation, in `k lambda_D` units.
DispersionEval langmuir_eval(Complex omega, double k_lambda_D) {
  return electrostatic_dispersion(omega, k_lambda_D,
                                  {MaxwellianBeam{1.0, 1.0, 0.0}});
}

} // namespace

// ---------------------------------------------------------------------------
// 1. The plasma dispersion function
// ---------------------------------------------------------------------------

/**
 * Oracle: the definition itself. At \f$\zeta=0\f$ the Cauchy integral is
 * \f$\pi^{-1/2}\int e^{-t^{2}}/t\,\mathrm dt\f$ closed above the pole, whose
 * principal value vanishes by oddness and whose residue contributes
 * \f$i\pi\cdot\pi^{-1/2}e^{0} = i\sqrt\pi\f$. Equivalently
 * \f$w(0)=e^{0}\operatorname{erfc}(0)=1\f$. Both are exact.
 */
TEST_CASE("Z and w at the origin are exact", "[plasma][Z]") {
  const Complex w0 = faddeeva_w(Complex(0.0, 0.0));
  const Complex z0 = plasma_Z(Complex(0.0, 0.0));
  INFO(
      precise("w(0) = ", cstr(w0), "  Z(0) = ", cstr(z0), "  sqrt(pi) = ", kSqrtPi));
  CHECK_THAT(w0.real(), WithinAbs(1.0, 1.0e-14));
  CHECK_THAT(w0.imag(), WithinAbs(0.0, 1.0e-15));
  CHECK_THAT(z0.real(), WithinAbs(0.0, 1.0e-15));
  CHECK_THAT(z0.imag(), WithinAbs(kSqrtPi, 1.0e-14));
}

/**
 * Oracle: `std::erfc`, a different library entirely. On the positive imaginary
 * axis \f$w(i\xi)=e^{\xi^{2}}\operatorname{erfc}(\xi)\f$ is real and positive,
 * so \f$Z(i\xi)=i\sqrt\pi\,e^{\xi^{2}}\operatorname{erfc}\xi\f$ is purely
 * imaginary. This is the axis on which every purely growing (Weibel,
 * two-stream) root sits, so both the value and the exact vanishing of the real
 * part matter. \f$\xi\le5\f$ keeps \f$e^{\xi^{2}}\operatorname{erfc}\xi\f$ out
 * of the range where `erfc` itself underflows.
 */
TEST_CASE("Z on the imaginary axis matches e^{xi^2} erfc(xi)", "[plasma][Z]") {
  double worst = 0.0;
  double worst_xi = 0.0;
  for (double xi = 0.05; xi <= 5.0001; xi += 0.05) {
    const Complex value = plasma_Z(Complex(0.0, xi));
    const double reference = kSqrtPi * std::exp(xi * xi) * std::erfc(xi);
    REQUIRE(value.real() == 0.0); // exactly purely imaginary, not nearly so
    const double e = std::abs(value.imag() - reference) / reference;
    if (e > worst) {
      worst = e;
      worst_xi = xi;
    }
  }
  INFO(precise("worst relative error ", worst, " at xi = ", worst_xi));
  CHECK(worst < 1.0e-13);
}

/**
 * Oracle: the principal-value integral plus the residue, evaluated directly
 * (see `pv_Z_real_axis`). This checks the *real* axis, where the Landau
 * contour is pinched against the pole and where the imaginary part of \f$Z\f$
 * is the pure residue \f$\sqrt\pi e^{-x^{2}}\f$ — at \f$x=6\f$ that is
 * 4e-16, so the agreement there is an absolute statement about the
 * implementation's noise floor rather than a relative one.
 */
TEST_CASE("Z on the real axis matches the principal value plus the residue",
          "[plasma][Z]") {
  double worst = 0.0;
  for (double x : {0.0, 0.25, 0.75, 1.5, 2.5, 4.0}) {
    const Complex value = plasma_Z(Complex(x, 0.0));
    const Complex reference = pv_Z_real_axis(x);
    const double e = relerr(value, reference);
    INFO(precise("x = ", x, "  Z = ", cstr(value),
                 "  PV+residue = ", cstr(reference), "  rel = ", e));
    CHECK(e < 1.0e-11);
    worst = std::max(worst, e);
  }
  INFO(precise("worst ", worst));
  CHECK(worst < 1.0e-11);

  // Deep in the exponentially small tail the residue is all there is, and the
  // statement worth making is absolute, not relative.
  const double x = 6.0;
  CHECK_THAT(plasma_Z(Complex(x, 0.0)).imag(),
             WithinAbs(kSqrtPi * std::exp(-x * x), 1.0e-13));
}

/**
 * Oracle: the everywhere-convergent Taylor series of \f$w\f$ (see
 * `w_taylor`). **This is the lower-half-plane test**, the one the issue
 * singles out, and it is run on a disc that straddles the real axis plus a
 * thin strip just below it — the strip being where the reflection
 * \f$w(z)=2e^{-z^{2}}-w(-z)\f$ subtracts two numbers of comparable size and so
 * is the only place the identity costs accuracy.
 */
TEST_CASE("Z is accurate in the lower half plane", "[plasma][Z][continuation]") {
  double worst_upper = 0.0;
  double worst_lower = 0.0;
  Complex arg_lower;
  for (int ix = -25; ix <= 25; ++ix) {
    for (int iy = -25; iy <= 25; ++iy) {
      const Complex z(0.1 * ix, 0.1 * iy);
      if (std::abs(z) > 2.5 || std::abs(z) == 0.0) continue;
      const double e = relerr(faddeeva_w(z), w_taylor(z));
      if (z.imag() >= 0.0) {
        worst_upper = std::max(worst_upper, e);
      } else if (e > worst_lower) {
        worst_lower = e;
        arg_lower = z;
      }
    }
  }
  INFO(precise("upper half plane ", worst_upper, ", lower half plane ", worst_lower,
               " at ", cstr(arg_lower)));
  CHECK(worst_lower < 1.0e-11);
  CHECK(worst_upper < 1.0e-11);

  double worst_strip = 0.0;
  Complex arg_strip;
  for (int ix = -40; ix <= 40; ++ix) {
    for (int j = 1; j <= 10; ++j) {
      const Complex z(0.05 * ix, -0.01 * j);
      const double e = relerr(faddeeva_w(z), w_taylor(z));
      if (e > worst_strip) {
        worst_strip = e;
        arg_strip = z;
      }
    }
  }
  INFO(precise("strip below the real axis ", worst_strip, " at ", cstr(arg_strip)));
  CHECK(worst_strip < 1.0e-11);
}

/**
 * Oracle: the two exact functional equations of \f$Z\f$.
 *
 * \f$w(-z) = 2e^{-z^{2}} - w(z)\f$ (from
 * \f$\operatorname{erfc}u+\operatorname{erfc}(-u)=2\f$) becomes
 * \f$Z(-\zeta) = -Z(\zeta) + 2i\sqrt\pi e^{-\zeta^{2}}\f$: the
 * \f$2i\sqrt\pi e^{-\zeta^{2}}\f$ **is** the analytic continuation, the whole
 * difference between \f$Z\f$ and the naive odd reflection \f$-Z(\zeta)\f$.
 * \f$w(-\bar z)=\overline{w(z)}\f$ becomes
 * \f$Z(-\bar\zeta) = -\overline{Z(\zeta)}\f$, which is what makes
 * \f$D(i\gamma)\f$ real for the symmetric two-stream configuration.
 *
 * The last section makes the continuation term's *size* concrete: at the
 * physically important \f$k\lambda_{D}=0.5\f$ Landau root it is a 13 % term,
 * and an implementation that drops it does not produce a slightly wrong
 * damping rate — it produces a residual of order one half at the true root.
 */
TEST_CASE("The analytic continuation identities hold exactly",
          "[plasma][Z][continuation]") {
  double worst_reflect = 0.0;
  double worst_conj = 0.0;
  for (double x = -3.0; x <= 3.01; x += 0.25) {
    for (double y = 0.1; y <= 3.01; y += 0.25) {
      const Complex z(x, y);
      const Complex reflected =
          -plasma_Z(z) + Complex(0.0, 2.0 * kSqrtPi) * std::exp(-z * z);
      worst_reflect = std::max(worst_reflect, relerr(plasma_Z(-z), reflected));
      worst_conj = std::max(
          worst_conj, relerr(plasma_Z(-std::conj(z)), -std::conj(plasma_Z(z))));
    }
  }
  INFO(precise("reflection ", worst_reflect, ", conjugation ", worst_conj));
  CHECK(worst_reflect < 1.0e-14);
  CHECK(worst_conj < 1.0e-15);

  // How much the continuation is worth, at the root that matters.
  const DispersionRoot root = solve_langmuir_root(0.5);
  const Complex zeta = root.omega / (kSqrt2 * 0.5);
  const Complex jump = Complex(0.0, 2.0 * kSqrtPi) * std::exp(-zeta * zeta);
  const double fraction = std::abs(jump) / std::abs(plasma_Z(zeta));
  INFO(precise("zeta = ", cstr(zeta), "  |continuation|/|Z| = ", fraction));
  CHECK(fraction > 0.05);

  // Residual of the true root when Z is replaced by the naive odd reflection.
  const double s = 1.0 / (0.5 * 0.5);
  const Complex d_true = langmuir_eval(root.omega, 0.5).value;
  const Complex d_naive = d_true - s * zeta * jump;
  INFO(precise("|D| with the continuation ", std::abs(d_true), ", without it ",
               std::abs(d_naive)));
  CHECK(std::abs(d_true) < 1.0e-13);
  CHECK(std::abs(d_naive) > 0.1);
}

/**
 * Oracle: a Cauchy contour derivative of `plasma_Z` (see `cauchy_derivative`).
 *
 * `plasma_Zprime` is *defined* as \f$-2(1+\zeta Z)\f$, so checking it against
 * that expression would be circular. What is checked here is that
 * \f$-2(1+\zeta Z)\f$ really is the derivative of the function this header
 * computes — i.e. that the identity and the implementation are consistent
 * across both half planes, including where the reflection is used.
 */
TEST_CASE("Zprime is the derivative of Z", "[plasma][Z]") {
  double worst = 0.0;
  Complex arg;
  for (double x = -3.0; x <= 3.01; x += 0.5) {
    for (double y = -2.0; y <= 3.01; y += 0.5) {
      const Complex z(x, y);
      const Complex numeric =
          cauchy_derivative([](Complex s) { return plasma_Z(s); }, z, 0.5, 64);
      const double e =
          std::abs(numeric - plasma_Zprime(z)) / std::max(std::abs(numeric), 1.0);
      if (e > worst) {
        worst = e;
        arg = z;
      }
    }
  }
  INFO(precise("worst ", worst, " at ", cstr(arg)));
  CHECK(worst < 1.0e-12);

  // Below the |zeta| = 8 seam, Zprime and the literal identity must agree to
  // the last bit, because there they are the same arithmetic.
  for (Complex z : {Complex(0.3, 0.2), Complex(-1.5, -0.7), Complex(2.0, -0.25)}) {
    CHECK(relerr(plasma_Zprime(z), -2.0 * (1.0 + z * plasma_Z(z))) < 1.0e-15);
  }
}

/**
 * Oracle: the Fried-Conte asymptotic expansion
 * \f$Z\sim i\sigma\sqrt\pi e^{-\zeta^{2}} - \zeta^{-1}\sum_{n}
 * (2n-1)!!(2\zeta^{2})^{-n}\f$, checked in all three Stokes sectors.
 *
 * The \f$\sigma=2\f$ sector is the interesting one: there the exponential
 * term *dominates* (at \f$\zeta=3-4i\f$, \f$|Z|\approx3.9\times10^{3}\f$,
 * essentially all of it the residue), so this is a second, independent
 * confirmation that the lower-half-plane continuation is right — and it is a
 * confirmation at large \f$|\zeta|\f$, where the Taylor reference above cannot
 * reach.
 *
 * Tolerances are set by the truncation of the expansion, not by the
 * implementation: the first omitted term is \f$(2n-1)!!/(2\zeta^{2})^{n}\f$
 * relative to the leading one, which at \f$|\zeta|=20\f$ with 8 terms is
 * \f$\sim10^{-16}\f$ and at \f$|\zeta|=5\f$ with 8 terms is only \f$10^{-3}\f$
 * — hence the per-point tolerance below.
 */
TEST_CASE("Z matches its asymptotic expansion in all three Stokes sectors",
          "[plasma][Z]") {
  struct Point {
    Complex zeta;
    double tol;
    const char *sector;
  };
  const std::array<Point, 8> points{{
      {Complex(0.0, 14.0), 1.0e-13, "sigma=0, positive imaginary axis"},
      {Complex(14.0, 14.0), 1.0e-13, "sigma=0, upper half plane"},
      {Complex(-14.0, 14.0), 1.0e-13, "sigma=0, upper half plane"},
      {Complex(20.0, 0.0), 1.0e-13, "sigma=1, real axis"},
      {Complex(-20.0, 0.0), 1.0e-13, "sigma=1, real axis"},
      {Complex(3.0, -4.0), 1.0e-10, "sigma=2, exponential dominates"},
      {Complex(-3.0, -4.0), 1.0e-10, "sigma=2, exponential dominates"},
      {Complex(0.0, -14.0), 1.0e-13, "sigma=2, negative imaginary axis"},
  }};
  for (const Point &p : points) {
    const Complex exact = plasma_Z(p.zeta);
    const Complex asymptotic = plasma_Z_asymptotic(p.zeta, 8);
    const double e = relerr(asymptotic, exact);
    INFO(precise(p.sector, ": zeta = ", cstr(p.zeta), "  Z = ", cstr(exact),
                 "  asymptotic = ", cstr(asymptotic), "  rel = ", e));
    CHECK(e < p.tol);
  }
}

/**
 * Oracle: the closed-form asymptotic series for \f$1+\zeta Z\f$ itself.
 *
 * This is the combination every dispersion relation contains, and at large
 * \f$|\zeta|\f$ it is a difference of two numbers that agree to
 * \f$O(\zeta^{-2})\f$. The test states the damage quantitatively: at
 * \f$|\zeta|=10^{3}\f$ the value is \f$-5\times10^{-7}\f$, the naive
 * \f$1+\zeta Z\f$ carries the absolute noise of \f$Z\f$ (\f$\sim10^{-15}\f$)
 * and so is wrong in its ninth digit, while the asymptotic branch matches the
 * analytic series to round-off. This is not academic: the cold limit of the
 * Weibel relation is evaluated at exactly these arguments, and the
 * `[weibel][cold-limit]` case below converges to 1.5e-8 only because of it.
 */
TEST_CASE("one_plus_zeta_Z beats the cancellation in 1 + zeta Z", "[plasma][Z]") {
  for (double R : {100.0, 300.0, 1000.0}) {
    for (double frac : {0.0, 0.4, 0.9}) {
      const double y = frac * R;
      const Complex zeta(std::sqrt(R * R - y * y), y);
      const Complex z2 = zeta * zeta;
      // -1/(2z^2) - 3/(4z^4) - 15/(8z^6) - 105/(16 z^8). The first omitted
      // term is 945/(2 zeta^2)^5, i.e. 945/(2 zeta^2)^4 = 6e-15 of the total
      // at |zeta| = 100 and less beyond, so this reference is itself good to
      // the tolerance asserted.
      const Complex analytic =
          -(1.0 / (2.0 * z2) + 3.0 / (4.0 * z2 * z2) + 15.0 / (8.0 * z2 * z2 * z2) +
            105.0 / (16.0 * z2 * z2 * z2 * z2));
      const Complex stable = one_plus_zeta_Z(zeta);
      const Complex naive = 1.0 + zeta * plasma_Z(zeta);
      INFO(precise("|zeta| = ", R, " arg frac ", frac, ": stable ", cstr(stable),
                   " naive ", cstr(naive), " analytic ", cstr(analytic),
                   "  rel(stable) ", relerr(stable, analytic), "  rel(naive) ",
                   relerr(naive, analytic)));
      CHECK(relerr(stable, analytic) < 1.0e-13);
    }
  }
  // And the naive form really is the worse one, by orders of magnitude. The
  // gap grows like |zeta|^2 because the absolute noise of Z is fixed while the
  // quantity being resolved shrinks: at |zeta| = 1e3 it is a factor ~70, at
  // 1e4 a factor ~1e7. The two-term reference is good to 3.75/|zeta|^4 here.
  const Complex zeta(1.0e4, 0.0);
  const Complex z2 = zeta * zeta;
  const Complex analytic = -(1.0 / (2.0 * z2) + 3.0 / (4.0 * z2 * z2));
  const double naive_error = relerr(1.0 + zeta * plasma_Z(zeta), analytic);
  const double stable_error = relerr(one_plus_zeta_Z(zeta), analytic);
  INFO(precise("at |zeta| = 1e4: naive ", naive_error, ", stable ", stable_error));
  CHECK(stable_error < 1.0e-13);
  CHECK(naive_error > 1000.0 * stable_error);
}

// ---------------------------------------------------------------------------
// 2. Electrostatic roots: Landau damping
// ---------------------------------------------------------------------------

/**
 * **The oracle for every root in this file.** A returned \f$\omega\f$ is a
 * root of \f$D\f$ if and only if \f$|D(\omega)|\f$ is at the round-off of the
 * terms that make up \f$D\f$ — which is what `relative_residual()` measures,
 * and which is the self-test issue #84 asks for by name.
 *
 * The scan also fixes the *direction* of the physics: increasing
 * \f$k\lambda_{D}\f$ must raise the real frequency (shorter wavelengths are
 * stiffer) and must make the damping stronger (the phase velocity
 * \f$\omega/k\f$ falls toward the bulk of the distribution, so more particles
 * are resonant). Both are monotone over the whole range, and neither is a
 * remembered number.
 */
TEST_CASE("Langmuir roots: residual at round-off over a k lambda_D scan",
          "[plasma][roots][langmuir]") {
  double previous_omega = 0.0;
  double previous_gamma = 0.0;
  double worst_residual = 0.0;
  double worst_cross_check = 0.0;
  bool first = true;

  for (double kl = 0.20; kl <= 1.5001; kl += 0.05) {
    const DispersionRoot root = solve_langmuir_root(kl);
    INFO(precise("k lambda_D = ", kl, "  omega = ", cstr(root.omega),
                 "  relative residual = ", root.relative_residual(),
                 "  iterations = ", root.iterations));
    REQUIRE(root.converged);
    CHECK(root.relative_residual() < 1.0e-13);
    worst_residual = std::max(worst_residual, root.relative_residual());

    // Derivative-free second opinion on the same relation. Agreement between
    // a Newton iteration driven by an analytic dD/domega and a Muller
    // iteration that never sees one is a direct check of the derivative.
    const DispersionRoot cross =
        muller_root([kl](Complex w) { return langmuir_eval(w, kl); },
                    langmuir_initial_guess(kl));
    REQUIRE(cross.converged);
    const double diff = std::abs(cross.omega - root.omega);
    INFO(precise("Muller gives ", cstr(cross.omega), ", difference ", diff));
    CHECK(diff < 1.0e-10);
    worst_cross_check = std::max(worst_cross_check, diff);

    CHECK(root.growth_rate() < 0.0); // a Maxwellian cannot drive a Langmuir wave
    if (!first) {
      CHECK(root.frequency() > previous_omega);
      CHECK(root.growth_rate() < previous_gamma);
    }
    previous_omega = root.frequency();
    previous_gamma = root.growth_rate();
    first = false;
  }
  INFO(precise("worst relative residual ", worst_residual,
               ", worst Newton-Muller difference ", worst_cross_check));
  CHECK(worst_residual < 1.0e-13);
}

/**
 * **Sanity check, not an oracle.** The \f$k\lambda_{D}=0.5\f$ root of the
 * kinetic Langmuir relation is the most-tabulated number in linear plasma
 * theory (Canosa, *J. Comput. Phys.* **13**, 158 (1973), and every textbook
 * since): \f$\omega\approx1.4156\f$, \f$\gamma\approx-0.1534\f$. Landing on it
 * means the *relation* is the one the literature solves and the normalisation
 * convention matches. It does not certify the root — the residual above does
 * that — so the tolerance here is deliberately loose, at the precision the
 * literature quotes.
 */
TEST_CASE("Langmuir root at k lambda_D = 0.5 lands on the tabulated value",
          "[plasma][roots][langmuir]") {
  const DispersionRoot root = solve_langmuir_root(0.5);
  INFO(precise("omega = ", cstr(root.omega),
               "  relative residual = ", root.relative_residual()));
  REQUIRE(root.converged);
  CHECK_THAT(root.frequency(), WithinAbs(1.4156, 1.0e-3));
  CHECK_THAT(root.growth_rate(), WithinAbs(-0.1534, 1.0e-3));
  CHECK(root.relative_residual() < 1.0e-13);

  // Bohm-Gross is 7 % low here, which is the concrete reason the issue
  // forbids substituting an asymptotic formula for the solved root.
  CHECK(std::abs(bohm_gross_frequency(0.5) - root.frequency()) / root.frequency() >
        0.05);
}

/**
 * Oracle: the two quoted weak-damping asymptotics, used in the only regime
 * where they are asymptotics — approached, not evaluated at.
 *
 * As \f$k\lambda_{D}\to0\f$ the numerically solved root must tend to the
 * Bohm-Gross frequency \f$\sqrt{1+3k^{2}\lambda_{D}^{2}}\f$ (whose first
 * neglected term is \f$O(k^{4}\lambda_{D}^{4})\f$) and to the exponentially
 * small Landau decrement. The test asserts *convergence*, i.e. that both
 * errors shrink as the limit is approached, rather than asserting a value at
 * one point — the latter would just be another remembered number.
 *
 * The measured frequency error falls as 1.5e-2, 5.4e-3, 1.6e-3 at
 * \f$k\lambda_{D}=0.25,0.20,0.15\f$, i.e. like \f$(k\lambda_{D})^{4}\f$ as the
 * next Bohm-Gross term predicts; the damping ratio
 * \f$\gamma/\gamma_{\mathrm{est}}\f$ rises 0.72, 0.85, 0.93 toward 1.
 *
 * The scan stops at 0.15 on purpose. Below \f$k\lambda_{D}\approx0.12\f$ the
 * residue that carries all the information about \f$\gamma\f$ is smaller than
 * the round-off of \f$\operatorname{Re}Z\f$ and the returned \f$\gamma\f$
 * stops meaning anything, while the residual stays tiny and cannot warn about
 * it — a limitation documented on `solve_langmuir_root()` and re-stated here
 * so it is not rediscovered by the application.
 */
TEST_CASE("Weak-damping limit reproduces Bohm-Gross and the Landau estimate",
          "[plasma][roots][langmuir]") {
  const std::array<double, 3> scan{0.25, 0.20, 0.15};
  double previous_freq_error = 1.0;
  double previous_gamma_gap = 1.0;
  for (double kl : scan) {
    const DispersionRoot root = solve_langmuir_root(kl);
    REQUIRE(root.converged);
    const double freq_error =
        std::abs(root.frequency() - bohm_gross_frequency(kl)) / root.frequency();
    const double ratio = root.growth_rate() / landau_damping_estimate(kl);
    const double gamma_gap = std::abs(ratio - 1.0);
    INFO(precise("k lambda_D = ", kl, "  omega = ", root.frequency(),
                 " (Bohm-Gross ", bohm_gross_frequency(kl), ", rel ", freq_error,
                 ")  gamma = ", root.growth_rate(), " (estimate ",
                 landau_damping_estimate(kl), ", ratio ", ratio, ")"));
    CHECK(freq_error < previous_freq_error);
    CHECK(gamma_gap < previous_gamma_gap);
    previous_freq_error = freq_error;
    previous_gamma_gap = gamma_gap;
  }
  // ...and by k lambda_D = 0.15 both asymptotics are actually close.
  const DispersionRoot root = solve_langmuir_root(0.15);
  CHECK_THAT(root.frequency(), WithinRel(bohm_gross_frequency(0.15), 3.0e-3));
  CHECK_THAT(root.growth_rate(), WithinRel(landau_damping_estimate(0.15), 0.15));
}

/// Argument validation: a zero or negative wave number has no relation to solve.
TEST_CASE("Electrostatic relation rejects unusable arguments",
          "[plasma][roots][langmuir]") {
  CHECK_THROWS_AS(electrostatic_dispersion(Complex(1.0, 0.0), 0.0,
                                           {MaxwellianBeam{1.0, 1.0, 0.0}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(electrostatic_dispersion(Complex(1.0, 0.0), -0.5,
                                           {MaxwellianBeam{1.0, 1.0, 0.0}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(electrostatic_dispersion(Complex(1.0, 0.0), 0.5,
                                           {MaxwellianBeam{1.0, 0.0, 0.0}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(solve_langmuir_root(0.0), std::invalid_argument);
  CHECK_THROWS_AS(weibel_dispersion(Complex(0.0, 0.1), 0.0, BiMaxwellian{}),
                  std::invalid_argument);
}

// ---------------------------------------------------------------------------
// 3. Two-stream
// ---------------------------------------------------------------------------

/**
 * Oracle: the closed form derived in `cold_two_stream_roots()`, checked
 * against the relation it claims to solve (residual at round-off at all four
 * roots) and against the textbook extremum of that closed form,
 * \f$\gamma_{\max}=\omega_{b}/2\f$ at \f$kv_{0}=\tfrac{\sqrt3}{2}\omega_{b}\f$
 * — which is here re-derived by *scanning* the closed form rather than
 * quoted, so the two agree only if the algebra is right.
 *
 * The stability boundary \f$kv_{0}=\sqrt2\,\omega_{b}=\omega_{pe}\f$ is
 * checked from both sides.
 */
TEST_CASE("Cold two-stream: closed-form roots satisfy their own relation",
          "[plasma][roots][two-stream]") {
  for (double kv0 : {0.05, 0.2, 0.4, 0.6124, 0.8, 0.999, 1.2, 2.0}) {
    const auto roots = cold_two_stream_roots(kv0, 0.5);
    double worst = 0.0;
    for (const Complex &w : roots.omega) {
      const DispersionEval e = cold_two_stream_dispersion(w, kv0, 0.5);
      worst = std::max(worst, std::abs(e.value) / e.scale);
    }
    INFO(precise("k v0 = ", kv0, "  gamma = ", roots.growth_rate,
                 "  worst relative residual ", worst));
    CHECK(worst < 1.0e-14);
    CHECK(roots.unstable == (kv0 < 1.0));
  }

  // The peak, re-found by scanning the closed form.
  const auto peak = cold_two_stream_peak(0.5);
  double best_k = 0.0;
  double best_gamma = 0.0;
  for (int i = 1; i < 200000; ++i) {
    const double kv0 = 1.0 * static_cast<double>(i) / 200000.0;
    const double g = cold_two_stream_roots(kv0, 0.5).growth_rate;
    if (g > best_gamma) {
      best_gamma = g;
      best_k = kv0;
    }
  }
  INFO(precise("analytic peak (", peak[0], ", ", peak[1], "), scanned (", best_k,
               ", ", best_gamma, ")"));
  CHECK_THAT(best_k, WithinAbs(peak[0], 1.0e-4));
  CHECK_THAT(best_gamma, WithinAbs(peak[1], 1.0e-8));
  CHECK_THAT(peak[1],
             WithinRel(1.0 / (2.0 * kSqrt2), 1.0e-14)); // omega_pe/(2 sqrt 2)

  // A derivative-free solve of the same relation, started away from the root,
  // must reproduce the closed form.
  const DispersionRoot solved = muller_root(
      [](Complex w) { return cold_two_stream_dispersion(w, 0.6124, 0.5); },
      Complex(0.05, 0.25));
  INFO(precise("Muller ", cstr(solved.omega), " vs closed form gamma ",
               cold_two_stream_roots(0.6124, 0.5).growth_rate));
  REQUIRE(solved.converged);
  CHECK_THAT(solved.omega.imag(),
             WithinAbs(cold_two_stream_roots(0.6124, 0.5).growth_rate, 1.0e-12));
}

/**
 * Oracle: the residual, plus the cold closed form as a *limit*.
 *
 * Two symmetric counter-streaming Maxwellians. The unstable root must be
 * purely growing — which here is not merely asserted but comes out
 * \f$\operatorname{Re}\omega = 0\f$ bitwise, because the bisection runs on the
 * imaginary axis and the Newton polish provably keeps it there.
 *
 * The cold limit is the sharp test: the kinetic growth rate must approach the
 * closed form as \f$v_{th}\to0\f$, and it must approach it at *second* order,
 * because the leading thermal correction enters through
 * \f$1+\zeta Z \simeq -1/2\zeta^{2} - 3/4\zeta^{4}\f$ whose second term is
 * \f$O(v_{th}^{2})\f$ relative to the first. Measured deficits
 * \f$\gamma_{\mathrm{cold}}-\gamma\f$ are 3.98e-7, 3.98e-5 and 1.01e-3 at
 * \f$v_{th}/c = 10^{-4}, 10^{-3}, 5\times10^{-3}\f$: exactly a factor 100 per
 * decade. A first-order error, or a sign error in the drift, would not do
 * that.
 */
TEST_CASE("Kinetic two-stream: purely growing root and the cold limit",
          "[plasma][roots][two-stream]") {
  const double kv0_peak = 0.6123724357;
  std::vector<double> deficits;
  for (double vth : {1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2}) {
    const TwoStreamMaxwellians p{0.1, vth, 0.5};
    const double k = kv0_peak / p.v_drift;
    const DispersionRoot root = solve_two_stream_root(k, p);
    const double cold = cold_two_stream_roots(k * p.v_drift, 0.5).growth_rate;
    INFO(precise("v_th = ", vth, "  gamma = ", root.growth_rate(), " (cold ", cold,
                 ")  Re omega = ", root.frequency(),
                 "  relative residual = ", root.relative_residual()));
    REQUIRE(root.converged);
    // Purely growing. On the imaginary axis D is real and dD/domega is purely
    // imaginary, so the complex Newton step has an exactly zero real part and
    // this comes out bitwise 0; the tolerance is there only so the test does
    // not depend on that being true of every compiler.
    CHECK_THAT(root.frequency(), WithinAbs(0.0, 1.0e-15));
    CHECK(root.relative_residual() < 1.0e-13);
    CHECK(root.growth_rate() > 0.0);
    CHECK(root.growth_rate() < cold); // thermal spread only removes free energy
    deficits.push_back(cold - root.growth_rate());
  }
  // Second-order convergence: a factor of 10 in v_th is a factor of 100 here.
  INFO(precise("deficits ", deficits[0], ", ", deficits[1], ", ", deficits[2]));
  CHECK_THAT(deficits[1] / deficits[0], WithinRel(100.0, 0.05));
  CHECK(deficits[0] < 1.0e-6);

  // Well beyond the cold marginal point kv0 = omega_pe the configuration is
  // stable and "no root" is the honest return.
  const TwoStreamMaxwellians p{0.1, 5.0e-3, 0.5};
  for (double kv0 : {1.5, 2.0, 4.0}) {
    const DispersionRoot root = solve_two_stream_root(kv0 / p.v_drift, p);
    INFO(precise("k v0 = ", kv0, "  converged = ", root.converged));
    CHECK_FALSE(root.converged);
    CHECK(root.growth_rate() == 0.0);
  }

  // ...while inside the band a warm pair is unstable, and near the cold
  // boundary it is *more* unstable than the cold pair, which is why the
  // solver's search window is grown rather than taken from the cold rate.
  const DispersionRoot near_edge = solve_two_stream_root(0.95 / p.v_drift, p);
  REQUIRE(near_edge.converged);
  INFO(precise("k v0 = 0.95: warm ", near_edge.growth_rate(), ", cold ",
               cold_two_stream_roots(0.95, 0.5).growth_rate));
  CHECK(near_edge.growth_rate() > cold_two_stream_roots(0.95, 0.5).growth_rate);
  CHECK(near_edge.relative_residual() < 1.0e-13);
}

// ---------------------------------------------------------------------------
// 4. Transverse electromagnetic: Weibel and filamentation
// ---------------------------------------------------------------------------

/**
 * Oracle: the residual, plus a bracketed bisection that shares no code path
 * with the Newton polish.
 *
 * The structural claims of the derivation in `weibel_dispersion()` are all
 * tested here rather than assumed:
 *
 *  - \f$D_{T}(i\gamma)\f$ is **real** and **strictly decreasing** in
 *    \f$\gamma\f$ — the monotonicity is what proves the purely growing root is
 *    unique and what makes the bisection unconditionally reliable, so it is
 *    checked numerically on a fine grid;
 *  - the root is purely growing, \f$\operatorname{Re}\omega=0\f$, as an
 *    *output* of a complex-plane Newton iteration rather than an assumption;
 *  - it is unchanged when the Newton iteration is started off the axis;
 *  - bisection and Newton agree to twelve relative digits (near cutoff
 *    \f$D_{T}\f$ is nearly tangent, so last-bit relative agreement is not a
 *    well-posed demand under fused multiply-add);
 *  - the growth rate rises, peaks and falls to zero at
 *    \f$k_{c}=\omega_{pe}\sqrt{A-1}/c\f$ (Weibel's published criterion), and
 *    beyond \f$k_{c}\f$ there is no root at all.
 *
 * The scan covers eight wave numbers including the peak and two stable ones,
 * which is the "at least four wave numbers including the fastest-growing mode
 * and at least one stable mode" the validation ladder asks of the application.
 */
TEST_CASE("Weibel: purely growing root, cutoff, and bisection agreement",
          "[plasma][roots][weibel]") {
  const BiMaxwellian p{0.05, 0.15, 1.0}; // A = 9
  const double kc = weibel_cutoff_wavenumber(p);
  INFO(precise("A = ", p.anisotropy(), "  k_c = ", kc));
  CHECK_THAT(kc, WithinRel(std::sqrt(p.anisotropy() - 1.0), 1.0e-15));

  // D_T(i gamma) is real and strictly decreasing: the uniqueness proof.
  const double k_probe = 0.5 * kc;
  double previous = std::numeric_limits<double>::infinity();
  for (int i = 0; i <= 400; ++i) {
    const double gamma = 3.0 * static_cast<double>(i) / 400.0;
    const DispersionEval e = weibel_dispersion(Complex(0.0, gamma), k_probe, p);
    INFO(precise("gamma = ", gamma, "  D_T = ", cstr(e.value)));
    REQUIRE(e.value.imag() == 0.0);
    REQUIRE(e.value.real() < previous);
    previous = e.value.real();
  }

  double best_gamma = 0.0;
  double best_k = 0.0;
  double previous_gamma = -1.0;
  bool rising = true;
  for (double frac : {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99}) {
    const double k = frac * kc;
    const DispersionRoot root = solve_weibel_root(k, p);
    const double bisected = weibel_growth_by_bisection(k, p);
    INFO(precise("k/k_c = ", frac, "  gamma = ", root.growth_rate(),
                 "  Re omega = ", root.frequency(), "  bisection = ", bisected,
                 "  relative residual = ", root.relative_residual()));
    REQUIRE(root.converged);
    CHECK_THAT(root.frequency(), WithinAbs(0.0, 1.0e-15)); // purely growing
    CHECK(root.growth_rate() > 0.0);
    CHECK(root.relative_residual() < 1.0e-13);
    // Independent oracles: derivative-free bisection vs complex Newton.
    // At k/kc = 0.99, D_T is nearly tangent and gcc-13 -O3 FMA moves the
    // two evaluations at the 14th digit even when both residuals are at
    // round-off. Twelve matching digits is still far tighter than any
    // measured Vlasov growth rate.
    CHECK_THAT(root.growth_rate(), WithinRel(bisected, 1.0e-12));
    if (root.growth_rate() > best_gamma) {
      best_gamma = root.growth_rate();
      best_k = k;
    }
    if (previous_gamma >= 0.0 && root.growth_rate() < previous_gamma) {
      rising = false;
    } else {
      CHECK(rising); // one maximum, no oscillation
    }
    previous_gamma = root.growth_rate();
  }
  INFO(precise("peak gamma ", best_gamma, " at k = ", best_k, " = ", best_k / kc,
               " k_c"));
  CHECK(best_k > 0.1 * kc);
  CHECK(best_k < 0.9 * kc);

  // Beyond the cutoff: no root, and the solver says so rather than returning
  // a zero dressed up as a converged solve.
  for (double frac : {1.0, 1.2, 2.0}) {
    const DispersionRoot root = solve_weibel_root(frac * kc, p);
    INFO(precise("k/k_c = ", frac, "  converged = ", root.converged));
    CHECK_FALSE(root.converged);
    CHECK(root.growth_rate() == 0.0);
    CHECK(weibel_growth_by_bisection(frac * kc, p) == 0.0);
  }

  // Started off the imaginary axis, the complex Newton iteration comes back to
  // it: Re omega = 0 is a property of the root, not of the starting point.
  const double gamma_axis = weibel_growth_by_bisection(k_probe, p);
  for (double f : {0.02, 0.1, 0.3, 1.0}) {
    const DispersionRoot off =
        newton_root([&](Complex w) { return weibel_dispersion(w, k_probe, p); },
                    Complex(f * gamma_axis, (1.0 + f) * gamma_axis));
    INFO(precise("off-axis start f = ", f, ": omega = ", cstr(off.omega),
                 "  axis value ", gamma_axis));
    REQUIRE(off.converged);
    CHECK_THAT(off.omega.real(), WithinAbs(0.0, 1.0e-14));
    CHECK_THAT(off.omega.imag(), WithinRel(gamma_axis, 1.0e-13));
  }
}

/**
 * Oracle: the derivation's own marginal condition, approached continuously.
 *
 * The instability exists only because \f$T_{y}>T_{x}\f$, so the growth rate
 * must vanish as \f$A\to1\f$ and the mode must be stable at every wave number
 * for \f$A\le1\f$. The measured peak rates fall 7.3e-2, 3.2e-2, 3.6e-3,
 * 1.2e-4, 3.8e-6 for \f$A = 9, 4, 1.44, 1.0404, 1.004\f$, i.e. faster than
 * linearly in \f$A-1\f$ — the free energy *and* the band of unstable wave
 * numbers both shrink.
 *
 * The \f$T_{y}<T_{x}\f$ case is the other half of the statement and is the one
 * that would catch a transposed anisotropy: swapping the two thermal speeds
 * must turn the instability off completely, not merely reduce it.
 */
TEST_CASE("Weibel growth vanishes as the anisotropy does",
          "[plasma][roots][weibel]") {
  double previous = std::numeric_limits<double>::infinity();
  for (double vty : {0.150, 0.100, 0.060, 0.051, 0.0501}) {
    const BiMaxwellian p{0.05, vty, 1.0};
    const double kc = weibel_cutoff_wavenumber(p);
    REQUIRE(kc > 0.0);
    const DispersionRoot root = solve_weibel_root(0.5 * kc, p);
    INFO(precise("A = ", p.anisotropy(), "  k_c = ", kc,
                 "  gamma = ", root.growth_rate()));
    REQUIRE(root.converged);
    CHECK(root.growth_rate() > 0.0);
    CHECK(root.growth_rate() < previous);
    CHECK(root.relative_residual() < 1.0e-13);
    previous = root.growth_rate();
  }
  CHECK(previous < 1.0e-5);

  // Isotropic, and anti-anisotropic: no instability at any wave number.
  for (double vty : {0.050, 0.040, 0.010}) {
    const BiMaxwellian p{0.05, vty, 1.0};
    INFO(precise("A = ", p.anisotropy()));
    CHECK(weibel_cutoff_wavenumber(p) == 0.0);
    for (double k : {0.01, 0.1, 1.0, 10.0}) {
      CHECK_FALSE(solve_weibel_root(k, p).converged);
      CHECK(weibel_growth_by_bisection(k, p) == 0.0);
    }
  }
}

/**
 * Oracle: `cold_filamentation_growth_rate()`, an **independently derived**
 * closed form (cold counter-streaming beams, \f$\delta\f$-functions in
 * velocity) which the bi-Maxwellian relation must reproduce as
 * \f$v_{th,x}\to0\f$ with \f$\langle v_{y}^{2}\rangle\f$ held fixed.
 *
 * The two derivations share no algebra beyond the common first two steps, so
 * agreement is real evidence. And it must be agreement *at second order in*
 * \f$v_{th,x}\f$, because the first neglected term in
 * \f$A[1+\zeta Z]\f$ is \f$-3Ak^{4}v_{th,x}^{4}/\omega^{4}
 * = O(v_{th,x}^{2})\f$ relative to the retained one. Measured relative
 * differences at \f$k=0.5\f$: 1.85e-2, 1.87e-4, 1.88e-6, 1.88e-8 for
 * \f$v_{th,x}=10^{-2}\ldots10^{-5}\f$ — a clean factor of 100 per decade over
 * four decades, which also exercises `one_plus_zeta_Z()` out to
 * \f$|\zeta|\sim10^{5}\f$ where the naive difference has no digits left.
 */
TEST_CASE("Weibel reduces to the cold filamentation closed form",
          "[plasma][roots][weibel][cold-limit]") {
  const double u0 = 0.1;
  for (double k : {0.1, 0.5, 2.0}) {
    std::vector<double> errors;
    for (double vtx : {1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5}) {
      const BiMaxwellian p{vtx, u0, 1.0};
      const DispersionRoot root = solve_weibel_root(k, p);
      const double cold = cold_filamentation_growth_rate(k, u0, 1.0);
      const double e = std::abs(root.growth_rate() - cold) / cold;
      INFO(precise("k = ", k, "  v_th,x = ", vtx, "  kinetic ", root.growth_rate(),
                   "  cold ", cold, "  rel ", e));
      REQUIRE(root.converged);
      CHECK(root.relative_residual() < 1.0e-13);
      errors.push_back(e);
    }
    // Second order in v_th,x, over four decades.
    for (std::size_t i = 1; i < errors.size(); ++i) {
      INFO(precise("k = ", k, "  error ratio ", errors[i - 1] / errors[i]));
      CHECK_THAT(errors[i - 1] / errors[i], WithinRel(100.0, 0.12));
    }
    CHECK(errors.back() < 2.0e-7);
  }
}

/**
 * Oracle: the biquadratic the closed form solves, plus its two analytic
 * limits.
 *
 * The residual check is the same self-test as everywhere else. The limits are
 * the published facts the relation has to reproduce: \f$\gamma\to ku_{0}\f$ as
 * \f$k\to0\f$ (the beams stream freely; the field cannot keep up), and
 * \f$\gamma\to(u_{0}/c)\,\omega_{pe}\f$ as \f$k\to\infty\f$ — the saturation
 * of the cold filamentation growth rate at \f$\beta\omega_{pe}\f$ that is
 * quoted throughout the beam-plasma literature (Bret, Gremillet & Dieckmann,
 * *Phys. Plasmas* **17**, 120501 (2010)). The absence of a cutoff is the piece
 * of physics the thermal \f$v_{th,x}\f$ restores, and stating it here is what
 * keeps the cold form from being misused as an oracle at large \f$k\f$.
 */
TEST_CASE("Cold filamentation closed form: residual and both limits",
          "[plasma][roots][weibel][cold-limit]") {
  const double u0 = 0.1;
  for (double k : {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0}) {
    const double gamma = cold_filamentation_growth_rate(k, u0, 1.0);
    const DispersionEval e =
        cold_filamentation_dispersion(Complex(0.0, gamma), k, u0, 1.0);
    INFO(precise("k = ", k, "  gamma = ", gamma, "  relative residual ",
                 std::abs(e.value) / e.scale));
    CHECK(gamma > 0.0); // unstable at every k: no cutoff without v_th,x
    CHECK(std::abs(e.value) / e.scale < 1.0e-14);
    CHECK(e.value.imag() == 0.0);
  }
  CHECK_THAT(cold_filamentation_growth_rate(0.001, u0, 1.0),
             WithinRel(0.001 * u0, 1.0e-5));
  CHECK_THAT(cold_filamentation_growth_rate(1.0e4, u0, 1.0), WithinRel(u0, 1.0e-7));
}

/**
 * Oracle: the isotropic limit of the transverse relation must be the
 * electromagnetic wave, \f$\omega^{2}=\omega_{pe}^{2}+c^{2}k^{2}\f$ plus the
 * thermal correction \f$k^{2}v_{th,x}^{2}\f$.
 *
 * This is the third of the three checks quoted in the derivation and the one
 * that fixes the *signs* of the two non-kinetic terms: get either wrong and
 * the light wave comes out with the wrong cutoff or the wrong phase velocity,
 * which no amount of agreement on the Weibel branch would reveal. Stage 1 of
 * the application's validation ladder measures exactly this dispersion
 * relation in the vacuum limit, so it is also the rung this function will be
 * compared against.
 */
TEST_CASE("Transverse relation reproduces the electromagnetic branch",
          "[plasma][roots][weibel]") {
  const BiMaxwellian p{0.02, 0.02, 1.0}; // isotropic, cold-ish
  for (double k : {0.5, 1.0, 2.0, 5.0}) {
    const double fluid = std::sqrt(1.0 + k * k);
    const DispersionRoot root = newton_root(
        [&](Complex w) { return weibel_dispersion(w, k, p); }, Complex(fluid, 0.0));
    // Retaining only 1 + zeta Z = -k^2 v_th,x^2 / omega^2 turns the relation
    // into the biquadratic omega^4 - (omega_pe^2 + c^2k^2) omega^2
    // - omega_pe^2 k^2 v_th,x^2 = 0, whose upper root is this. Note the
    // correction is k^2 v_th,x^2 / omega^2, not k^2 v_th,x^2.
    const double s0 = 1.0 + k * k;
    const double thermal = std::sqrt(
        0.5 * (s0 + std::sqrt(s0 * s0 + 4.0 * k * k * p.v_th_x * p.v_th_x)));
    INFO(precise("k = ", k, "  omega = ", cstr(root.omega), "  cold fluid ", fluid,
                 "  with thermal correction ", thermal, "  relative residual ",
                 root.relative_residual()));
    REQUIRE(root.converged);
    CHECK(root.relative_residual() < 1.0e-13);
    CHECK_THAT(root.omega.imag(), WithinAbs(0.0, 1.0e-12)); // undamped
    CHECK_THAT(root.omega.real(), WithinRel(thermal, 1.0e-6));
    CHECK(root.omega.real() > fluid); // the thermal term is positive
  }
}

/**
 * Oracle: the bracketing primitive itself, on a function whose roots are
 * known in closed form.
 *
 * `bracket_growing_root()` underpins both purely growing solvers, so it is
 * tested on \f$\gamma^{2}-a^{2}\f$ (one sign change, at \f$a\f$) and on a
 * function with none, rather than only implicitly through the physics.
 */
TEST_CASE("Imaginary-axis bracketing finds the first sign change",
          "[plasma][roots]") {
  const auto quadratic = [](double g) { return 4.0 - g * g; };
  const auto bracket = bracket_growing_root(quadratic, 10.0, 512, 0.0);
  REQUIRE(bracket.found);
  CHECK(bracket.lo <= 2.0);
  CHECK(bracket.hi >= 2.0);
  CHECK_THAT(
      pfc::apps::plasma::bisect_growing_root(quadratic, bracket.lo, bracket.hi),
      WithinRel(2.0, 1.0e-14));

  const auto positive = [](double g) { return 1.0 + g * g; };
  CHECK_FALSE(bracket_growing_root(positive, 10.0, 512, 0.0).found);
}
