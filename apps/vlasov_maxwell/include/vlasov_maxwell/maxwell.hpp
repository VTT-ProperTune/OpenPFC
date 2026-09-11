// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file maxwell.hpp
 * @brief The 1-D field side of the 1D2V Vlasov-Maxwell system: the Gauss
 *        solve, the Ampere update, the *exactly* integrated transverse light
 *        wave, and the Gauss residual diagnostic.
 *
 * @details
 * ## The system this solves
 *
 * After the 1D2V reduction derived in `parameters.hpp`, three field components
 * survive, all functions of `x` alone and all periodic:
 *
 *     d_x E_x = rho                  (Gauss   -- a constraint)
 *     d_t E_x = -J_x                 (Ampere  -- the evolution form)
 *     d_t E_y = -d_x B_z - J_y       (transverse pair, exactly integrable)
 *     d_t B_z = -d_x E_y
 *
 * with `c = eps_0 = 1`. `B_x == 0` identically, so `div B = 0` holds as a
 * property of the geometry. **That is not a test and this header does not
 * offer one.** Reporting a geometric identity as a passed check would tell a
 * reader something about the 1-D reduction that they already knew and nothing
 * whatever about the code.
 *
 * ## Why everything here is a small serial spectral computation
 *
 * `E_x`, `E_y` and `B_z` are `N_x` doubles -- kilobytes -- and the phase space
 * is decomposed on `v_y` only, so every rank already holds the whole `x` axis.
 * Replicating the fields is therefore cheaper than distributing them and it
 * removes the field solve from the parallel critical path entirely: the only
 * communication in a step is the moment `MPI_Allreduce` in `moments.hpp`.
 *
 * The transforms are consequently a rank-local 1-D DFT, which is why this
 * header carries its own rather than calling HeFFTe. HeFFTe's plans are
 * distributed 3-D objects; asking one to transform an `N_x` vector that every
 * rank already owns would mean building a private communicator per rank and
 * would buy nothing. The implementation is a radix-2 Cooley-Tukey with
 * directly evaluated twiddles (no recurrence, so no phase drift), falling back
 * to a direct `O(N^2)` DFT when `N_x` is not a power of two. At the sizes this
 * object ever sees -- `N_x <= a few thousand` -- both are free next to one
 * sweep of `f`.
 *
 * ## The Nyquist mode, stated once
 *
 * On an even grid, the derivative of the Nyquist cosine is a Nyquist sine,
 * which vanishes at *every* grid point. The Nyquist mode therefore carries no
 * representable derivative, and every operator here -- `d_x`, its inverse, and
 * the transverse propagator -- uses `k = 0` for it. This is the standard
 * spectral projection and it is exact, not an approximation, but it has a
 * consequence worth stating: Nyquist content in `rho` cannot be matched by
 * `d_x E_x`, so it appears in the Gauss residual as a genuine and irreducible
 * error. A run whose `rho` has Nyquist content is under-resolved and the
 * residual is right to say so.
 *
 * ## Duhamel for the transverse pair
 *
 * In Fourier along `x` the source-free transverse system is
 *
 *     d/dt [Ey_hat; Bz_hat] = -i k S [Ey_hat; Bz_hat],   S = [[0,1],[1,0]]
 *
 * and `S^2 = I`, so the matrix exponential closes in elementary functions:
 *
 *     exp(-i k dt S) = cos(k dt) I - i sin(k dt) S.
 *
 * @ref advance_transverse_vacuum applies exactly that. It is **exact at any
 * `dt`**: the light wave carries no dispersion error and imposes no CFL
 * condition, which is the single strongest numerical claim the application
 * makes and the one validation stage 1 exists to falsify.
 *
 * With the current, write the source as `b(t) = [-Jy_hat(t); 0]`. Variation of
 * constants gives, exactly,
 *
 *     y(t+dt) = exp(-i k dt S) y(t)
 *               + int_0^dt exp(-i k (dt - s) S) b(t+s) ds.
 *
 * The integral is the only place an approximation enters, and the order of the
 * scheme is the order to which `b` is represented inside it.
 *
 * **ETD1** freezes `b` at `b_n`. Substituting `tau = dt - s`,
 *
 *     M1 = int_0^dt exp(-i k tau S) dtau
 *        = (sin(k dt)/k) I - i ((1 - cos(k dt))/k) S
 *        = dt [ sinc(theta) I - i ((1-cos theta)/theta) S ],   theta = k dt,
 *
 * with the `k -> 0` limit `M1 = dt I`. Exact for a constant current; local
 * error `O(dt^2 db/dt)`, so **first order** globally.
 *
 * **ETD2** represents `b` linearly across the step from `b_{n-1}` and `b_n`,
 * `b(t+s) = b_n + (s/dt)(b_n - b_{n-1}) + O(dt^2)`. The extra weight is
 *
 *     M2 = int_0^dt s exp(-i k (dt - s) S) ds
 *        = dt M1 - int_0^dt tau exp(-i k tau S) dtau
 *        = dt^2 [ ((1 - cos theta)/theta^2) I
 *                 - i ((theta - sin theta)/theta^2) S ],
 *
 * where the two scalar coefficients follow from
 * `int_0^dt tau cos(k tau) dtau = (cos theta - 1)/k^2 + dt sin(theta)/k` and
 * `int_0^dt tau sin(k tau) dtau = sin(theta)/k^2 - dt cos(theta)/k`, and both
 * tend to `dt^2/2` and `0` as `k -> 0`. The update is
 *
 *     y_{n+1} = exp(-i k dt S) y_n + M1 b_n + (M2/dt) (b_n - b_{n-1}),
 *
 * exact for a current linear in time; local error `O(dt^3)`, so **second
 * order** globally. That matches the Strang splitting around it, which is
 * also second order, so nothing is wasted and nothing is the bottleneck.
 *
 * Both coefficients are evaluated through numerically stable forms
 * (`(1-cos t)/t = 2 sin^2(t/2)/t`, and a series for `(t - sin t)/t^2` below
 * `|theta| = 0.1`) because a run with a long box and a short step puts
 * `theta` at `1e-6` for the lowest mode, where the naive quotients lose eight
 * digits.
 *
 * @see parameters.hpp for the reduction and the normalisation
 * @see moments.hpp for where `rho`, `J_x` and `J_y` come from
 * @see docs/report/02_numerical_methods.qmd for the ETD pattern OpenPFC uses
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <vlasov_maxwell/parameters.hpp>

namespace vlasov {

using Complex = std::complex<double>;

namespace detail {

inline constexpr double kPi = 3.14159265358979323846;

[[nodiscard]] inline bool is_power_of_two(int n) noexcept {
  return n > 0 && (n & (n - 1)) == 0;
}

/**
 * @brief In-place radix-2 Cooley-Tukey, unnormalised.
 *
 * Twiddles are evaluated directly with `std::polar` rather than accumulated by
 * a complex recurrence. The recurrence is faster and drifts: its phase error
 * grows like `sqrt(log N)` and shows up precisely in the round-off-level
 * assertions this application lives by (`omega = k` to `1e-12`). At these
 * sizes the trig calls do not matter.
 */
inline void fft_radix2(std::vector<Complex> &a, bool inverse) {
  const int n = static_cast<int>(a.size());
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; (j & bit) != 0; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[static_cast<std::size_t>(i)], a[static_cast<std::size_t>(j)]);
  }
  const double sign = inverse ? 1.0 : -1.0;
  for (int len = 2; len <= n; len <<= 1) {
    const int half = len / 2;
    const double base = sign * 2.0 * kPi / static_cast<double>(len);
    for (int i = 0; i < n; i += len) {
      for (int m = 0; m < half; ++m) {
        const Complex w = std::polar(1.0, base * static_cast<double>(m));
        const std::size_t lo = static_cast<std::size_t>(i + m);
        const std::size_t hi = static_cast<std::size_t>(i + m + half);
        const Complex u = a[lo];
        const Complex v = a[hi] * w;
        a[lo] = u + v;
        a[hi] = u - v;
      }
    }
  }
}

/// Direct DFT, unnormalised. Fallback for a non-power-of-two `N_x`.
inline void dft_direct(std::vector<Complex> &a, bool inverse) {
  const int n = static_cast<int>(a.size());
  std::vector<Complex> out(static_cast<std::size_t>(n), Complex{0.0, 0.0});
  const double sign = inverse ? 1.0 : -1.0;
  for (int m = 0; m < n; ++m) {
    Complex acc{0.0, 0.0};
    for (int i = 0; i < n; ++i) {
      const double ang =
          sign * 2.0 * kPi * static_cast<double>((static_cast<long long>(m) *
                                                  static_cast<long long>(i)) %
                                                 n) /
          static_cast<double>(n);
      acc += a[static_cast<std::size_t>(i)] * std::polar(1.0, ang);
    }
    out[static_cast<std::size_t>(m)] = acc;
  }
  a.swap(out);
}

inline void dft(std::vector<Complex> &a, bool inverse) {
  if (is_power_of_two(static_cast<int>(a.size()))) {
    fft_radix2(a, inverse);
  } else {
    dft_direct(a, inverse);
  }
  if (inverse) {
    const double s = 1.0 / static_cast<double>(a.size());
    for (Complex &z : a) z *= s;
  }
}

/// `sin(t)/t`, exact at `t = 0`.
[[nodiscard]] inline double sinc(double t) noexcept {
  if (std::fabs(t) < 1.0e-8) return 1.0 - t * t / 6.0;
  return std::sin(t) / t;
}

/// `(1 - cos t)/t`, written as `2 sin^2(t/2)/t` so no cancellation occurs.
[[nodiscard]] inline double versine_over_t(double t) noexcept {
  if (t == 0.0) return 0.0;
  const double s = std::sin(0.5 * t);
  return 2.0 * s * s / t;
}

/// `(1 - cos t)/t^2`, the ETD2 `I` coefficient. Tends to `1/2`.
[[nodiscard]] inline double versine_over_t2(double t) noexcept {
  if (t == 0.0) return 0.5;
  const double s = std::sin(0.5 * t);
  return 2.0 * s * s / (t * t);
}

/**
 * @brief `(t - sin t)/t^2`, the ETD2 `S` coefficient. Tends to `0` like `t/6`.
 *
 * `t - sin t` is `t^3/6` for small `t`, so the direct form loses all
 * significance below `t ~ 1e-4`; the series is used there instead.
 */
[[nodiscard]] inline double t_minus_sin_over_t2(double t) noexcept {
  if (std::fabs(t) < 0.1) {
    const double t2 = t * t;
    return t * (1.0 / 6.0 - t2 * (1.0 / 120.0 - t2 / 5040.0));
  }
  return (t - std::sin(t)) / (t * t);
}

} // namespace detail

/**
 * @brief The three surviving field components, replicated on every rank.
 */
struct FieldState {
  std::vector<double> Ex;
  std::vector<double> Ey;
  std::vector<double> Bz;

  [[nodiscard]] static FieldState zeros(int nx) {
    FieldState s;
    s.Ex.assign(static_cast<std::size_t>(nx), 0.0);
    s.Ey.assign(static_cast<std::size_t>(nx), 0.0);
    s.Bz.assign(static_cast<std::size_t>(nx), 0.0);
    return s;
  }
  [[nodiscard]] int size() const noexcept { return static_cast<int>(Ex.size()); }
};

/**
 * @brief A periodic 1-D spectral line in `x`: transforms, wavenumbers and the
 *        two derivative operators.
 *
 * Holds no plan and no scratch that would make it unsafe to share; it is a
 * value object carrying `N_x` and `L_x`.
 *
 * It is deliberately **agnostic to where the samples sit inside their cells**.
 * `SimParams::x_of` is cell-centred, `x_i = (i + 1/2) dx`, while a textbook DFT
 * assumes `x_i = i dx`. Shifting the sample grid by a constant multiplies every
 * Fourier coefficient by a phase `exp(-i k x_0)`, and every operator here is
 * *diagonal* in Fourier space, so that phase commutes straight back out: the
 * derivative, the Poisson solve and the transverse propagator are all exact on
 * either convention. Only a routine that read off an absolute position from a
 * coefficient would need to know, and none does.
 */
class SpectralLine1D {
public:
  SpectralLine1D(int nx, double Lx) : nx_(nx), Lx_(Lx) {
    if (nx < 2) throw std::invalid_argument("SpectralLine1D: need nx >= 2");
    if (!(Lx > 0.0)) throw std::invalid_argument("SpectralLine1D: need Lx > 0");
  }

  [[nodiscard]] int size() const noexcept { return nx_; }
  [[nodiscard]] double length() const noexcept { return Lx_; }
  [[nodiscard]] double dx() const noexcept {
    return Lx_ / static_cast<double>(nx_);
  }

  /// Signed wavenumber of spectral index `m`, as it labels the mode. The
  /// Nyquist index reports `+k_Nyq`; use @ref k_deriv in any operator.
  [[nodiscard]] double wavenumber(int m) const noexcept {
    const int mm = (m <= nx_ / 2) ? m : m - nx_;
    return 2.0 * detail::kPi * static_cast<double>(mm) / Lx_;
  }

  /**
   * @brief Wavenumber as the derivative operator sees it: zero at Nyquist.
   *
   * See the file comment. `d_x` of the Nyquist mode vanishes identically on
   * the grid, so the only self-consistent (and Hermitian-symmetry preserving)
   * choice is `k = 0` there.
   */
  [[nodiscard]] double k_deriv(int m) const noexcept {
    if ((nx_ % 2) == 0 && m == nx_ / 2) return 0.0;
    return wavenumber(m);
  }

  [[nodiscard]] std::vector<Complex> forward(const std::vector<double> &u) const {
    check(u, "forward");
    std::vector<Complex> a(static_cast<std::size_t>(nx_));
    for (int i = 0; i < nx_; ++i) {
      a[static_cast<std::size_t>(i)] = Complex{u[static_cast<std::size_t>(i)], 0.0};
    }
    detail::dft(a, false);
    return a;
  }

  [[nodiscard]] std::vector<double> inverse(std::vector<Complex> a) const {
    if (static_cast<int>(a.size()) != nx_) {
      throw std::invalid_argument("SpectralLine1D::inverse: wrong length");
    }
    detail::dft(a, true);
    std::vector<double> u(static_cast<std::size_t>(nx_));
    for (int i = 0; i < nx_; ++i) {
      u[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)].real();
    }
    return u;
  }

  /// Spectral `d_x`. Exact for every resolved mode; zero on Nyquist.
  [[nodiscard]] std::vector<double> derivative(const std::vector<double> &u) const {
    auto a = forward(u);
    for (int m = 0; m < nx_; ++m) {
      a[static_cast<std::size_t>(m)] *= Complex{0.0, k_deriv(m)};
    }
    return inverse(std::move(a));
  }

  /// Mean of a field over the period. The `k = 0` Fourier coefficient.
  [[nodiscard]] double mean(const std::vector<double> &u) const {
    check(u, "mean");
    double s = 0.0;
    for (double v : u) s += v;
    return s / static_cast<double>(nx_);
  }

private:
  void check(const std::vector<double> &u, const char *who) const {
    if (static_cast<int>(u.size()) != nx_) {
      throw std::invalid_argument(std::string("SpectralLine1D::") + who +
                                  ": wrong length");
    }
  }
  int nx_;
  double Lx_;
};

// ---------------------------------------------------------------------------
// 1. Gauss / Poisson
// ---------------------------------------------------------------------------

/**
 * @brief Result of the electrostatic field solve.
 */
struct PoissonResult {
  std::vector<double> Ex; ///< the zero-mean solution of `d_x E_x = rho - <rho>`
  /**
   * @brief `<rho>`, the `k = 0` Fourier coefficient of the charge density.
   *
   * A periodic solution of `d_x E_x = rho` exists **only** if this vanishes:
   * integrating over the period gives `0 = int rho dx`. It is returned rather
   * than quietly subtracted because a nonzero value means the neutralising
   * background is wrong, or a species has leaked particles through the
   * velocity boundary, and both are things the run needs to be told about.
   */
  double net_charge{0.0};
  bool neutral{true};
  double tolerance{0.0};
};

/**
 * @brief Solve `d_x E_x = rho` spectrally: `Ex_hat = rho_hat / (i k)`.
 *
 * `k = 0` is handled explicitly and is the whole subtlety. The equation
 * constrains only the derivative, so the mean of `E_x` is a free constant; the
 * zero-mean choice is taken, which is the unique solution with no net field
 * across the period. The mean of `rho` is *not* free: it must vanish, and if
 * it does not, the returned field solves the neutralised problem while
 * @ref PoissonResult::net_charge reports exactly how much charge was left over.
 * Nothing is silently zeroed.
 *
 * @param neutrality_tol absolute tolerance on `|<rho>|` below which the state
 *        is called neutral. The default is set for `rho` of order one.
 */
[[nodiscard]] inline PoissonResult solve_gauss(const SpectralLine1D &line,
                                               const std::vector<double> &rho,
                                               double neutrality_tol = 1.0e-12) {
  const int nx = line.size();
  auto a = line.forward(rho);
  PoissonResult r;
  r.net_charge = a[0].real() / static_cast<double>(nx);
  r.tolerance = neutrality_tol;
  r.neutral = std::fabs(r.net_charge) <= neutrality_tol;

  a[0] = Complex{0.0, 0.0};
  for (int m = 1; m < nx; ++m) {
    const double k = line.k_deriv(m);
    if (k == 0.0) {
      a[static_cast<std::size_t>(m)] = Complex{0.0, 0.0}; // Nyquist
    } else {
      // 1/(i k) = -i/k.
      a[static_cast<std::size_t>(m)] *= Complex{0.0, -1.0 / k};
    }
  }
  r.Ex = line.inverse(std::move(a));
  return r;
}

/// Throw unless the state is neutral. For call sites that want the run to stop
/// rather than to carry on reporting a growing net charge.
inline void require_neutral(const PoissonResult &r) {
  if (!r.neutral) {
    throw std::runtime_error("solve_gauss: net charge " +
                             std::to_string(r.net_charge) +
                             " exceeds the neutrality tolerance " +
                             std::to_string(r.tolerance) +
                             "; Gauss has no periodic solution");
  }
}

// ---------------------------------------------------------------------------
// 2. Ampere
// ---------------------------------------------------------------------------

/**
 * @brief `d_t E_x = -J_x`, advanced by `dt`.
 *
 * Deliberately the plainest possible update. Its accuracy is entirely the
 * accuracy of the `J_x` handed to it: with the step-averaged current this is
 * *exact*, and exactness here is what makes the Gauss constraint hold to
 * round-off (differentiate Gauss, substitute Ampere, use continuity). The
 * function is trivial; the property is not, and it is the one
 * `test_fields.cpp` asserts on.
 */
inline void ampere_ex(std::vector<double> &Ex, const std::vector<double> &Jx,
                      double dt) {
  if (Ex.size() != Jx.size()) {
    throw std::invalid_argument("ampere_ex: Ex and Jx have different lengths");
  }
  for (std::size_t i = 0; i < Ex.size(); ++i) Ex[i] -= dt * Jx[i];
}

// ---------------------------------------------------------------------------
// 3. The transverse pair, advanced exactly
// ---------------------------------------------------------------------------

namespace detail {

/**
 * @brief Core of the transverse update. See the file comment for the
 *        derivation of `M1` and `M2`.
 *
 * @param Jy      current at the start of the step, or `nullptr` for vacuum
 * @param Jy_prev current at the previous step, or `nullptr` for ETD1
 */
inline void advance_transverse_impl(const SpectralLine1D &line,
                                    std::vector<double> &Ey,
                                    std::vector<double> &Bz,
                                    const std::vector<double> *Jy,
                                    const std::vector<double> *Jy_prev,
                                    double dt) {
  const int nx = line.size();
  if (static_cast<int>(Ey.size()) != nx || static_cast<int>(Bz.size()) != nx) {
    throw std::invalid_argument("advance_transverse: Ey/Bz length mismatch");
  }
  if (Jy != nullptr && static_cast<int>(Jy->size()) != nx) {
    throw std::invalid_argument("advance_transverse: Jy length mismatch");
  }
  if (Jy_prev != nullptr && static_cast<int>(Jy_prev->size()) != nx) {
    throw std::invalid_argument("advance_transverse: Jy_prev length mismatch");
  }

  auto ey = line.forward(Ey);
  auto bz = line.forward(Bz);
  std::vector<Complex> jy;
  std::vector<Complex> jp;
  if (Jy != nullptr) jy = line.forward(*Jy);
  if (Jy_prev != nullptr) jp = line.forward(*Jy_prev);

  for (int m = 0; m < nx; ++m) {
    const std::size_t mm = static_cast<std::size_t>(m);
    const double theta = line.k_deriv(m) * dt;
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    // exp(-i k dt S) = cos I - i sin S, with S the component swap.
    const Complex e0 = ey[mm];
    const Complex b0 = bz[mm];
    Complex e1 = c * e0 - Complex{0.0, 1.0} * (s * b0);
    Complex b1 = c * b0 - Complex{0.0, 1.0} * (s * e0);

    if (Jy != nullptr) {
      // M1 b with b = (-Jy_hat, 0): alpha = dt sinc(theta), beta = dt (1-cos)/theta.
      const double a1 = dt * detail::sinc(theta);
      const double bb1 = dt * detail::versine_over_t(theta);
      const Complex J = jy[mm];
      e1 += -a1 * J;
      b1 += Complex{0.0, 1.0} * (bb1 * J);

      if (Jy_prev != nullptr) {
        // (M2/dt) (b_n - b_{n-1}); M2 = dt^2 [ a2 I - i b2 S ].
        const double a2 = dt * detail::versine_over_t2(theta);
        const double b2 = dt * detail::t_minus_sin_over_t2(theta);
        const Complex dJ = J - jp[mm];
        e1 += -a2 * dJ;
        b1 += Complex{0.0, 1.0} * (b2 * dJ);
      }
    }

    ey[mm] = e1;
    bz[mm] = b1;
  }

  Ey = line.inverse(std::move(ey));
  Bz = line.inverse(std::move(bz));
}

} // namespace detail

/**
 * @brief Source-free transverse update, `exp(-i k dt S)`, exact at any `dt`.
 *
 * This is validation stage 1. A single transverse mode initialised as
 * `E_y = B_z = A cos(k x)` is the right-travelling solution `A cos(k(x - t))`,
 * and this routine reproduces it to round-off for `dt = 1e-3` and for
 * `dt = 37.3` alike, because the propagator *is* the solution operator rather
 * than an approximation to it. If a measured `omega(k)` ever differs from `k`
 * by more than round-off, the integrator -- not the resolution -- is wrong.
 */
inline void advance_transverse_vacuum(const SpectralLine1D &line,
                                      std::vector<double> &Ey,
                                      std::vector<double> &Bz, double dt) {
  detail::advance_transverse_impl(line, Ey, Bz, nullptr, nullptr, dt);
}

/// Transverse update with the current held constant across the step (ETD1).
/// Exact for a constant `J_y`; first order otherwise.
inline void advance_transverse_etd1(const SpectralLine1D &line,
                                    std::vector<double> &Ey,
                                    std::vector<double> &Bz,
                                    const std::vector<double> &Jy, double dt) {
  detail::advance_transverse_impl(line, Ey, Bz, &Jy, nullptr, dt);
}

/// Transverse update with the current extrapolated linearly from the previous
/// step (ETD2). Exact for a `J_y` linear in time; second order otherwise,
/// which matches the Strang splitting around it.
inline void advance_transverse_etd2(const SpectralLine1D &line,
                                    std::vector<double> &Ey,
                                    std::vector<double> &Bz,
                                    const std::vector<double> &Jy,
                                    const std::vector<double> &Jy_prev,
                                    double dt) {
  detail::advance_transverse_impl(line, Ey, Bz, &Jy, &Jy_prev, dt);
}

// ---------------------------------------------------------------------------
// 4. The Gauss residual diagnostic
// ---------------------------------------------------------------------------

/**
 * @brief The sharpest diagnostic in the application.
 *
 * Gauss is a constraint, not an evolution equation: nothing in the time loop
 * enforces it. Differentiating it and substituting Ampere gives
 * `d_t(d_x E_x - rho) = -(d_x J_x + d_t rho)`, which vanishes by continuity.
 * So the residual measures, in one number, whether the discrete scheme
 * conserves charge -- which is to say whether the deposition and the advection
 * agree with each other. It costs two transforms of an `N_x` vector, so it is
 * computed every sample, always, in every run.
 */
struct GaussDiagnostic {
  /// `max|d_x E_x - rho| / max|rho|`, **before** any correction.
  double residual{0.0};
  /// `max|d_x E_x - rho|`. Kept because the relative form is meaningless when
  /// `rho` is itself at round-off, which is exactly the vacuum test case.
  double abs_residual{0.0};
  /// `<rho>`; see @ref PoissonResult::net_charge.
  double net_charge{0.0};
  /// Whether a divergence correction was applied after the measurement.
  bool corrected{false};
};

/// Measure the Gauss residual. Never modifies the field.
[[nodiscard]] inline GaussDiagnostic gauss_residual(const SpectralLine1D &line,
                                                    const std::vector<double> &Ex,
                                                    const std::vector<double> &rho) {
  if (Ex.size() != rho.size()) {
    throw std::invalid_argument("gauss_residual: Ex and rho have different lengths");
  }
  const auto dEx = line.derivative(Ex);
  GaussDiagnostic g;
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < rho.size(); ++i) {
    num = std::max(num, std::fabs(dEx[i] - rho[i]));
    den = std::max(den, std::fabs(rho[i]));
  }
  g.abs_residual = num;
  g.residual = num / std::max(den, kTiny);
  g.net_charge = line.mean(rho);
  return g;
}

/**
 * @brief Optional Poisson-based divergence correction.
 *
 * Off by default in @ref SimParams::gauss_correction, and this signature is
 * why: the function **measures first and corrects second**, and what it
 * returns is the *uncorrected* residual. A correction that ran silently would
 * hide the single number that says whether the kinetic solver is conserving
 * charge, which would be the opposite of useful.
 *
 * The correction replaces every non-constant mode of `E_x` with the one Gauss
 * demands, `Ex_hat = rho_hat/(i k)`, and leaves the mean of `E_x` alone --
 * the mean is set by the initial condition and by Ampere, and Gauss says
 * nothing about it.
 *
 * @param apply when false the field is untouched and only the measurement is
 *              returned, so a driver can call this unconditionally.
 */
inline GaussDiagnostic correct_divergence(const SpectralLine1D &line,
                                          std::vector<double> &Ex,
                                          const std::vector<double> &rho,
                                          bool apply) {
  GaussDiagnostic g = gauss_residual(line, Ex, rho);
  if (!apply) return g;

  const double mean_ex = line.mean(Ex);
  PoissonResult r = solve_gauss(line, rho);
  for (std::size_t i = 0; i < Ex.size(); ++i) Ex[i] = r.Ex[i] + mean_ex;
  g.corrected = true;
  return g;
}

// ---------------------------------------------------------------------------
// 5. Field energy and momentum
// ---------------------------------------------------------------------------

/**
 * @brief `(1/2) int (|E|^2 + |B|^2) dx` with `E_z = B_x = B_y = 0`.
 *
 * In vacuum this is conserved to round-off by @ref advance_transverse_vacuum,
 * because the propagator is a unitary rotation of each Fourier mode and
 * Parseval turns that into exact conservation of the sum.
 */
[[nodiscard]] inline double field_energy(const SpectralLine1D &line,
                                         const FieldState &f) {
  double s = 0.0;
  for (std::size_t i = 0; i < f.Ex.size(); ++i) {
    s += f.Ex[i] * f.Ex[i] + f.Ey[i] * f.Ey[i] + f.Bz[i] * f.Bz[i];
  }
  return 0.5 * s * line.dx();
}

/// Transverse-pair energy alone, `(1/2) int (E_y^2 + B_z^2) dx`. The quantity
/// validation stage 1 conserves and the one whose growth is the Weibel signal.
[[nodiscard]] inline double transverse_energy(const SpectralLine1D &line,
                                              const std::vector<double> &Ey,
                                              const std::vector<double> &Bz) {
  double s = 0.0;
  for (std::size_t i = 0; i < Ey.size(); ++i) s += Ey[i] * Ey[i] + Bz[i] * Bz[i];
  return 0.5 * s * line.dx();
}

/**
 * @brief `int (E x B)_x dx = int E_y B_z dx`, the field momentum along `x`.
 *
 * The only surviving component: `(E x B)_x = E_y B_z - E_z B_y` and
 * `E_z = B_y = 0` under the 1D2V closure. It is the field half of the total
 * momentum the conservation table tracks against
 * `sum_s mu_s int int int v f_s`.
 */
[[nodiscard]] inline double field_momentum_x(const SpectralLine1D &line,
                                             const FieldState &f) {
  double s = 0.0;
  for (std::size_t i = 0; i < f.Ey.size(); ++i) s += f.Ey[i] * f.Bz[i];
  return s * line.dx();
}

} // namespace vlasov
