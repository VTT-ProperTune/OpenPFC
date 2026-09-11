// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file advect.hpp
 * @brief The three constant-coefficient translations of the 1D2V Strang split.
 *
 * @details
 * ## Why this file is three translations and not one advection solver
 *
 * The 1D2V electromagnetic Vlasov equation in the normalisation of
 * @ref parameters.hpp is
 *
 *     d_t f + v_x d_x f
 *           + (sigma/mu) (E_x + v_y B_z) d_vx f
 *           + (sigma/mu) (E_y - v_x B_z) d_vy f  =  0 .
 *
 * The property the whole numerical strategy rests on is that **each of the
 * three advection speeds is independent of the coordinate it advects along**:
 *
 *  | step | equation                   | speed                | independent of |
 *  |------|----------------------------|----------------------|----------------|
 *  | A    | `d_t f + v_x d_x f = 0`    | `v_x`                | `x`            |
 *  | B    | `d_t f + a_x d_vx f = 0`   | `(s/m)(E_x+v_y B_z)` | `v_x`          |
 *  | C    | `d_t f + a_y d_vy f = 0`   | `(s/m)(E_y-v_x B_z)` | `v_y`          |
 *
 * (`s/m` is `sigma_s/mu_s`, abbreviated only to keep the table narrow.)
 *
 * So each split step is an *exact rigid translation* of every one-dimensional
 * line of the brick, by an amount that is constant along that line. There is
 * no Riemann problem, no flux limiter, no nonlinear stability question and no
 * CFL number in the usual sense. What is left is one question per step: how
 * accurately can the code translate an array by a non-integer number of
 * cells? This file answers it three times, and @ref test_transport.cpp
 * measures each answer against an oracle that is not the code.
 *
 * ## Step A: exact, by construction
 *
 * `x` is periodic, so translation is diagonal in Fourier space and the
 * discrete operator
 *
 *     fhat_m  <-  fhat_m * exp(-2 pi i m delta / N_x) ,    delta = v_x dt / dx
 *
 * is the *exact* translation of the trigonometric interpolant through the
 * samples, for any `delta`, integer or not. There is no time-step restriction
 * and no dispersion error: the only error is the FFT's own round-off and the
 * fact that the trigonometric interpolant is not the true `f`. That makes
 * this the one step in the whole application with a closed-form oracle at
 * machine precision, which is why the test suite leans on it so hard.
 *
 * Two details that are easy to get wrong and invisible when wrong:
 *
 *  - **The Nyquist mode.** For even `N_x` the coefficient at `m = N_x/2` is
 *    shared between the modes `+N_x/2` and `-N_x/2`, so `exp(-i pi delta)`
 *    is not a well-defined multiplier for it: applying it verbatim makes the
 *    result complex. The symmetric (and standard) choice is the real part,
 *    `cos(pi delta)`, which is applied here. For an integer `delta` that is
 *    exactly `(-1)^delta`, so integer shifts are unaffected.
 *  - **Phase wrap-around.** The multiplier argument is reduced with `fmod`
 *    before the trigonometry, so a shift of many periods does not lose
 *    precision in `cos`/`sin` argument reduction.
 *
 * ## Why FFTW directly, and not HeFFTe
 *
 * The `x` axis is **rank-local by construction** (see @ref phase_space.hpp):
 * the decomposition splits `v_y` and nothing else. HeFFTe is a *distributed*
 * FFT: asking it for a transform along one axis of a brick it believes it
 * owns a pencil of means either a 3-D transform we do not want, or a
 * single-rank communicator plus a box geometry contrived to make two of the
 * three transforms trivial. Both are more machinery, and more chances to be
 * silently wrong, than calling a serial 1-D real FFT on data that is already
 * serial and already in the right layout. So this file calls **FFTW's serial
 * `r2c`/`c2r` plans directly** on the local brick, which is the choice the
 * issue explicitly sanctions.
 *
 * Where `<fftw3.h>` is not on the include path (a build configured without
 * HeFFTe, which is where FFTW enters the dependency graph), @ref XShiftPlan
 * falls back to @ref detail::spectral_shift_reference, a self-contained
 * radix-2 / naive DFT, so that the application still *builds and is correct*
 * at a cost in speed that is irrelevant for the unit tests and unacceptable
 * for a production run. @ref XShiftPlan::uses_fftw reports which path a
 * binary was compiled with.
 *
 * The reference is compiled **unconditionally**, not only in the fallback
 * build. That is deliberate: an `#if`-guarded code path that no CI
 * configuration compiles is a liability, and having a second, independent
 * implementation of the same operator sitting next to the first is the
 * cheapest possible cross-check. The test suite holds both against the same
 * closed-form oracle *and* against each other, in every build.
 *
 * ## Steps B and C: semi-Lagrangian, and where the real error lives
 *
 * Both velocity axes are *not* periodic and are physically truncated, so a
 * spectral shift is not available and would be wrong if it were. Instead the
 * departure point `v - a dt` is located and `f` is reconstructed there by
 * Lagrange interpolation through `interp_order` points (see
 * @ref lagrange_weights). Three consequences, all of them tested:
 *
 *  - **Order.** A Lagrange interpolant through `p` points is the unique
 *    degree-`p-1` polynomial through them, so the translation is `O(dv^p)`.
 *    This is the dominant error of the whole scheme; the light wave is exact
 *    and the splitting is second order, but the velocity interpolation is
 *    where the accuracy actually goes.
 *  - **Exactness at integer shifts.** When the shift is a whole number of
 *    cells the fractional offset is exactly `0`, one Lagrange node coincides
 *    with the evaluation point, and the weights come out as exactly
 *    `{0,...,0,1,0,...,0}` in IEEE arithmetic -- not approximately. So an
 *    integer-cell translation is *bitwise* a copy.
 *  - **Mass.** Lagrange weights sum to exactly one, and the shift is the same
 *    for every cell along the advected line, so the gather telescopes and the
 *    line sum is preserved -- *except* for what the stencil reads from
 *    outside the velocity box, which is zero. That difference is not an
 *    error: it is the physical outflow through the truncated velocity
 *    boundary, and it is reported in @ref TransportReport::mass_lost rather
 *    than hidden. A run in which it grows is not a valid run.
 *
 * ## The halo width on `v_y`, and why exceeding it must throw
 *
 * Step C advects along the *distributed* axis, so the departure point can lie
 * on a neighbouring rank. The ghost ring supplies it, and the required width
 * is
 *
 *     hw >= ceil(max |a_y| dt / dv_y) + interp_order/2
 *
 * (@ref required_halo_width). Exceeding it does not produce a visibly wrong
 * answer -- it produces a *plausible* one, assembled from stale ghost values
 * or from whatever the allocator left there. @ref advect_vy therefore reduces
 * the required width over the communicator with `MPI_MAX` and throws on every
 * rank before it communicates. The reduction happens *before* the exchange
 * deliberately: a single rank throwing after its neighbours had posted their
 * sends is a hang, not an error message.
 *
 * ## Working copies
 *
 * Memory is a first-class result in a kinetic code. Per species this file
 * needs, beyond the distribution itself:
 *
 *  - step A: one `x` line (`N_x` doubles) inside @ref XShiftPlan -- nothing;
 *  - step B: one `(x, v_x)` plane, `N_x N_vx` doubles;
 *  - step C: **one full copy of the owned brick**, because the gather along
 *    the distributed axis reads cells that a same-brick write would already
 *    have destroyed.
 *
 * So the scheme's high-water mark is two bricks per species plus the fields,
 * and @ref TransportWorkspace is where the second one lives, explicitly,
 * rather than being allocated and freed invisibly inside a step.
 *
 * @see phase_space.hpp for the grid, the decomposition and the `v_y` exchange
 * @see parameters.hpp for the reduction and the normalisation
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <vlasov_maxwell/parameters.hpp>
#include <vlasov_maxwell/phase_space.hpp>

#ifndef VLASOV_MAXWELL_HAVE_FFTW
#if __has_include(<fftw3.h>)
#define VLASOV_MAXWELL_HAVE_FFTW 1
#else
#define VLASOV_MAXWELL_HAVE_FFTW 0
#endif
#endif

#if VLASOV_MAXWELL_HAVE_FFTW
#include <fftw3.h>
#endif

namespace vlasov {

/// Largest `interp_order` the fixed-size weight buffers admit. Matches the
/// bound `SimParams::validate` enforces.
inline constexpr int kMaxInterpOrder = 9;

/**
 * @brief What one split step did, per rank.
 *
 * The mass entries are **unweighted sums of owned cells**, not integrals:
 * multiplying by @ref PhaseSpace::cell_volume and reducing over ranks turns
 * them into particle number. Keeping them unweighted and rank-local means a
 * test can assert on a number that involved no collective and no rounding
 * beyond the operator's own.
 */
struct TransportReport {
  double mass_before{0.0};     ///< sum of owned cells before the step
  double mass_after{0.0};      ///< sum of owned cells after the step
  double max_shift_cells{0.0}; ///< largest `|a dt / dv|` (or `|v dt / dx|`)
  int required_halo{0};        ///< ghost width this step actually needed
  bool measured{false};        ///< false if mass measurement was switched off

  /// Mass that left through the truncated velocity boundary (plus round-off).
  /// Positive means loss. Zero inflow means this can never be negative by
  /// more than round-off.
  [[nodiscard]] double mass_lost() const noexcept {
    return mass_before - mass_after;
  }
  /// Relative mass change, safe at zero mass.
  [[nodiscard]] double relative_mass_change() const noexcept {
    const double denom = std::max(std::abs(mass_before), kTiny);
    return mass_lost() / denom;
  }
};

/**
 * @brief Lagrange interpolation weights for a fractional cell offset.
 *
 * Nodes are the integers `lagrange_first_offset(p) .. +p-1` relative to the
 * departure cell, and the evaluation point is `frac` in `[0, 1)` measured
 * from the same cell. `w[m] = prod_{l != m} (frac - t_l)/(t_m - t_l)`.
 *
 * At `frac == 0` the evaluation point coincides with node `t = 0`, every
 * factor of that node's product is a value divided by itself (exactly `1.0`
 * in IEEE), and every other weight contains the factor `(0 - 0)`. So the
 * weights are exactly `{0,...,1,...,0}` -- this is why an integer-cell shift
 * is bitwise a copy, and the test suite asserts bitwise, not approximately.
 *
 * @param p     Number of points, `1 <= p <= kMaxInterpOrder`.
 * @param frac  Offset of the evaluation point from the departure cell.
 * @param w     Output, at least `p` doubles.
 */
inline void lagrange_weights(int p, double frac, double *w) {
  const int first = lagrange_first_offset(p);
  for (int m = 0; m < p; ++m) {
    const double tm = static_cast<double>(first + m);
    double prod = 1.0;
    for (int l = 0; l < p; ++l) {
      if (l == m) continue;
      const double tl = static_cast<double>(first + l);
      prod *= (frac - tl) / (tm - tl);
    }
    w[m] = prod;
  }
}

namespace detail {

/**
 * @brief Unnormalised DFT of `a`, in place. Always compiled.
 *
 * Radix-2 Cooley-Tukey when `N` is a power of two, an honest `O(N^2)` sum
 * otherwise. Twiddle factors are *evaluated*, never accumulated by repeated
 * complex multiplication: the accumulated form drifts by `O(N eps)` and this
 * file makes round-off claims that would then be claims about the drift.
 *
 * This exists for two reasons. It is the fallback when `<fftw3.h>` is absent,
 * and -- more usefully -- it is an independent second implementation of the
 * same transform that @ref XShiftPlan can be checked against in *every*
 * build, including the ones where FFTW is present and is the path actually
 * taken.
 */
inline void dft_inplace(std::vector<std::complex<double>> &a, bool inverse) {
  const int n = static_cast<int>(a.size());
  if (n <= 1) return;
  const double two_pi = 2.0 * std::acos(-1.0);
  const double sgn = inverse ? 1.0 : -1.0;
  const bool pow2 = (n & (n - 1)) == 0;
  if (pow2) {
    for (int i = 1, j = 0; i < n; ++i) {
      int bit = n >> 1;
      for (; (j & bit) != 0; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        std::swap(a[static_cast<std::size_t>(i)], a[static_cast<std::size_t>(j)]);
      }
    }
    for (int len = 2; len <= n; len <<= 1) {
      for (int i = 0; i < n; i += len) {
        for (int k = 0; k < len / 2; ++k) {
          const std::complex<double> w = std::polar(
              1.0, sgn * two_pi * static_cast<double>(k) / static_cast<double>(len));
          const auto u = a[static_cast<std::size_t>(i + k)];
          const auto v = a[static_cast<std::size_t>(i + k + len / 2)] * w;
          a[static_cast<std::size_t>(i + k)] = u + v;
          a[static_cast<std::size_t>(i + k + len / 2)] = u - v;
        }
      }
    }
    return;
  }
  std::vector<std::complex<double>> out(static_cast<std::size_t>(n));
  for (int m = 0; m < n; ++m) {
    std::complex<double> acc{0.0, 0.0};
    for (int j = 0; j < n; ++j) {
      acc += a[static_cast<std::size_t>(j)] *
             std::polar(1.0, sgn * two_pi * static_cast<double>(m) *
                                 static_cast<double>(j) / static_cast<double>(n));
    }
    out[static_cast<std::size_t>(m)] = acc;
  }
  a.swap(out);
}

/**
 * @brief Reference spectral translation: `out(i) = in(i - delta)`, in place.
 *
 * The same operator @ref XShiftPlan implements, written the slow obvious way
 * through @ref dft_inplace. Both apply the identical Nyquist rule (real
 * multiplier `cos(pi delta)` for even `N`) and the identical `fmod` argument
 * reduction, so the two are expected to agree to round-off and not merely to
 * plotting accuracy; that is what makes the comparison a test rather than a
 * gesture.
 */
inline void spectral_shift_reference(double *line, int n, double delta) {
  if (n < 2) return;
  const double two_pi = 2.0 * std::acos(-1.0);
  const double nd = static_cast<double>(n);
  std::vector<std::complex<double>> buf(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    buf[static_cast<std::size_t>(i)] = std::complex<double>{line[i], 0.0};
  }
  dft_inplace(buf, /*inverse=*/false);
  for (int m = 0; m < n; ++m) {
    // Modes above N/2 are the negative frequencies; their multiplier is the
    // conjugate, which is exactly what keeps the result real.
    const int signed_mode = (m <= n / 2) ? m : m - n;
    const double arg = std::fmod(static_cast<double>(signed_mode) * delta, nd);
    const double theta = -two_pi * arg / nd;
    std::complex<double> factor{std::cos(theta), std::sin(theta)};
    if (m == 0) factor = std::complex<double>{1.0, 0.0};
    if (n % 2 == 0 && m == n / 2) {
      factor = std::complex<double>{std::cos(theta), 0.0};
    }
    buf[static_cast<std::size_t>(m)] *= factor;
  }
  dft_inplace(buf, /*inverse=*/true);
  const double inv = 1.0 / nd;
  for (int i = 0; i < n; ++i) {
    line[i] = buf[static_cast<std::size_t>(i)].real() * inv;
  }
}

} // namespace detail

/**
 * @brief Exact spectral translation of one periodic, rank-local `x` line.
 *
 * Holds the transform machinery and a phase table for one shift, so that the
 * `N_vx * N_vy` lines that share a shift (every line at the same `v_x`) pay
 * for the trigonometry once. Not copyable, not movable, and not thread-safe:
 * FFTW's planner is not re-entrant and the scratch buffers are shared between
 * calls.
 */
class XShiftPlan {
public:
  explicit XShiftPlan(int n) : m_n(n), m_nk(n / 2 + 1) {
    if (n < 2) {
      throw std::invalid_argument(
          "XShiftPlan: need at least two x cells to transform");
    }
    m_phase_re.assign(static_cast<std::size_t>(m_nk), 1.0);
    m_phase_im.assign(static_cast<std::size_t>(m_nk), 0.0);
#if VLASOV_MAXWELL_HAVE_FFTW
    m_in = static_cast<double *>(
        fftw_malloc(sizeof(double) * static_cast<std::size_t>(n)));
    m_out = static_cast<fftw_complex *>(
        fftw_malloc(sizeof(fftw_complex) * static_cast<std::size_t>(m_nk)));
    if (m_in == nullptr || m_out == nullptr) {
      release_();
      throw std::bad_alloc();
    }
    // FFTW_ESTIMATE, not MEASURE: MEASURE overwrites the buffers during
    // planning and, more importantly, makes the plan -- and therefore the
    // round-off pattern -- depend on the machine's timing noise. A test that
    // asserts agreement at round-off should not be at the mercy of that.
    m_fwd = fftw_plan_dft_r2c_1d(n, m_in, m_out, FFTW_ESTIMATE);
    m_bwd = fftw_plan_dft_c2r_1d(n, m_out, m_in, FFTW_ESTIMATE);
    if (m_fwd == nullptr || m_bwd == nullptr) {
      release_();
      throw std::runtime_error("XShiftPlan: FFTW refused to make a plan");
    }
#endif
  }

  ~XShiftPlan() { release_(); }

  XShiftPlan(const XShiftPlan &) = delete;
  XShiftPlan &operator=(const XShiftPlan &) = delete;
  XShiftPlan(XShiftPlan &&) = delete;
  XShiftPlan &operator=(XShiftPlan &&) = delete;

  [[nodiscard]] int size() const noexcept { return m_n; }

  /// True when this translation unit found `<fftw3.h>`. A production build
  /// must report `true`; the fallback is `O(N^2)` for non-power-of-two `N`
  /// and exists so the application still builds, and is still right, in a
  /// configuration without HeFFTe (which is where FFTW enters the graph).
  [[nodiscard]] static constexpr bool uses_fftw() noexcept {
    return VLASOV_MAXWELL_HAVE_FFTW != 0;
  }

  /**
   * @brief Cache the multipliers for a shift of `delta` cells.
   *
   * `delta` may be any real number, positive or negative, larger than the
   * period or not: the argument is reduced modulo `N` before the
   * trigonometry, so accuracy does not decay with the number of periods
   * travelled.
   */
  void prepare(double delta) {
    m_delta = delta;
    const double n = static_cast<double>(m_n);
    const double two_pi = 2.0 * std::acos(-1.0);
    for (int m = 0; m < m_nk; ++m) {
      const double arg = std::fmod(static_cast<double>(m) * delta, n);
      const double theta = -two_pi * arg / n;
      m_phase_re[static_cast<std::size_t>(m)] = std::cos(theta);
      m_phase_im[static_cast<std::size_t>(m)] = std::sin(theta);
    }
    // Mode 0 is the mass. exp(0) is exactly 1 and must stay exactly 1, or the
    // operator stops conserving particle number for reasons that have nothing
    // to do with physics.
    m_phase_re[0] = 1.0;
    m_phase_im[0] = 0.0;
    if (m_n % 2 == 0) {
      // Nyquist: see the file comment. Keep only the real part so the result
      // stays real; cos(pi delta) is exactly +/-1 at integer delta.
      m_phase_im[static_cast<std::size_t>(m_nk - 1)] = 0.0;
    }
    m_prepared = true;
  }

  /**
   * @brief Replace `line[0..N)` by its translate: `out(i) = in(i - delta)`,
   *        with `delta` the value most recently passed to @ref prepare.
   */
  void apply(double *line) {
    if (!m_prepared) {
      throw std::logic_error("XShiftPlan::apply: call prepare(delta) first");
    }
#if VLASOV_MAXWELL_HAVE_FFTW
    std::memcpy(m_in, line, sizeof(double) * static_cast<std::size_t>(m_n));
    fftw_execute(m_fwd);
    for (int m = 0; m < m_nk; ++m) {
      const double re = m_out[m][0];
      const double im = m_out[m][1];
      const double pr = m_phase_re[static_cast<std::size_t>(m)];
      const double pim = m_phase_im[static_cast<std::size_t>(m)];
      m_out[m][0] = re * pr - im * pim;
      m_out[m][1] = re * pim + im * pr;
    }
    // A real signal has a real DC and, for even N, a real Nyquist
    // coefficient. Force it: c2r assumes Hermitian input and silently ignores
    // anything else, so a stray imaginary part here would be a difference
    // between this path and the reference that no test could see.
    m_out[0][1] = 0.0;
    if (m_n % 2 == 0) m_out[m_nk - 1][1] = 0.0;
    fftw_execute(m_bwd);
    const double inv = 1.0 / static_cast<double>(m_n);
    for (int i = 0; i < m_n; ++i) line[i] = m_in[i] * inv;
#else
    detail::spectral_shift_reference(line, m_n, m_delta);
#endif
  }

  /// `prepare` + `apply`, for a one-off line.
  void shift(double *line, double delta) {
    prepare(delta);
    apply(line);
  }

private:
  void release_() noexcept {
#if VLASOV_MAXWELL_HAVE_FFTW
    if (m_fwd != nullptr) fftw_destroy_plan(m_fwd);
    if (m_bwd != nullptr) fftw_destroy_plan(m_bwd);
    if (m_in != nullptr) fftw_free(m_in);
    if (m_out != nullptr) fftw_free(m_out);
    m_fwd = nullptr;
    m_bwd = nullptr;
    m_in = nullptr;
    m_out = nullptr;
#endif
  }

  int m_n{0};
  int m_nk{0};
  bool m_prepared{false};
  double m_delta{0.0};
  std::vector<double> m_phase_re{};
  std::vector<double> m_phase_im{};
#if VLASOV_MAXWELL_HAVE_FFTW
  double *m_in{nullptr};
  fftw_complex *m_out{nullptr};
  fftw_plan m_fwd{nullptr};
  fftw_plan m_bwd{nullptr};
#endif
};

/**
 * @brief The scratch the velocity steps need, allocated once and named.
 *
 * See "Working copies" in the file comment: step B needs one `(x, v_x)`
 * plane, step C needs one whole owned brick. They live here so that the
 * application's memory high-water mark is a visible property of the program
 * rather than a transient allocation inside a hot loop.
 */
struct TransportWorkspace {
  std::vector<double> plane{}; ///< `N_x * N_vx`, step B
  std::vector<double> brick{}; ///< `N_x * N_vx * N_vy^local`, step C
  std::vector<int> base{};     ///< integer part of the departure offset
  std::vector<double> wts{};   ///< `kMaxInterpOrder` weights per line

  /// Number of doubles this workspace holds, for a memory report.
  [[nodiscard]] std::size_t doubles() const noexcept {
    return plane.size() + brick.size() + wts.size();
  }

  /// Size for step B (`N_x` lines of coefficients, one `(x,v_x)` plane).
  void resize_for_vx(const PhaseSpace &ps) {
    const std::size_t nx = static_cast<std::size_t>(ps.nx());
    plane.resize(nx * static_cast<std::size_t>(ps.nvx()));
    base.resize(std::max(base.size(), nx));
    wts.resize(std::max(wts.size(), nx * kMaxInterpOrder));
  }

  /// Size for step C (`N_x * N_vx` lines of coefficients, one owned brick).
  void resize_for_vy(const PhaseSpace &ps) {
    const std::size_t np =
        static_cast<std::size_t>(ps.nx()) * static_cast<std::size_t>(ps.nvx());
    brick.resize(np * static_cast<std::size_t>(ps.nvy_local()));
    base.resize(std::max(base.size(), np));
    wts.resize(std::max(wts.size(), np * kMaxInterpOrder));
  }

  /// Measure `sum f` before and after every step. On by default: particle
  /// number is a required diagnostic and the two extra streaming passes are
  /// small next to a `p`-point gather. Switch it off in a profiling run.
  bool measure_mass{true};
};

namespace detail {

/// Guard used by both velocity steps.
inline void check_interp_order(int p) {
  if (p < 1 || p > kMaxInterpOrder) {
    throw std::invalid_argument("advect: interp_order must be 1.." +
                                std::to_string(kMaxInterpOrder) + ", got " +
                                std::to_string(p));
  }
}

/// Guard on a 1-D field array supplied by the Maxwell solver.
inline void check_field_extent(std::span<const double> a, int nx, const char *name) {
  if (static_cast<int>(a.size()) != nx) {
    throw std::invalid_argument(
        std::string("advect: ") + name + " has " + std::to_string(a.size()) +
        " entries but x has " + std::to_string(nx) +
        " cells. The Maxwell fields are 1-D in x and replicated on every "
        "rank, because x is never decomposed.");
  }
}

} // namespace detail

/**
 * @brief Step A: `d_t f + v_x d_x f = 0`, exact spectral shift along `x`.
 *
 * Every line of constant `(v_x, v_y)` is translated by `v_x dt`, which is
 * `v_x dt / dx` cells and depends only on `v_x`. The loop is therefore
 * ordered with `v_x` outermost so the phase table is built `N_vx` times, not
 * `N_vx N_vy` times.
 *
 * `x` is periodic and the shift is exact, so nothing is lost at a boundary
 * and the reported mass change is pure FFT round-off: mode 0 is multiplied by
 * exactly `1.0` (see @ref XShiftPlan::prepare), so the operator conserves
 * `sum f` in exact arithmetic.
 *
 * @param ps    The phase-space stack (for geometry and `v_x` coordinates).
 * @param f     Distribution to advance, in place. Only owned cells are read
 *              or written; the ghost ring is untouched.
 * @param dt    Step over which to translate.
 * @param plan  Transform plan; must have `plan.size() == ps.nx()`.
 * @param work  Only consulted for @ref TransportWorkspace::measure_mass.
 */
inline TransportReport advect_x(const PhaseSpace &ps, PhaseField &f, double dt,
                                XShiftPlan &plan, const TransportWorkspace &work) {
  const int nx = ps.nx();
  const int nvx = ps.nvx();
  const int nvy = ps.nvy_local();
  if (plan.size() != nx) {
    throw std::invalid_argument("advect_x: plan is sized for " +
                                std::to_string(plan.size()) + " cells but x has " +
                                std::to_string(nx));
  }

  TransportReport rep;
  rep.measured = work.measure_mass;
  rep.required_halo = 0; // x is rank-local and periodic: no ghosts are read
  if (work.measure_mass) rep.mass_before = PhaseSpace::local_sum(f);

  const double inv_dx = 1.0 / ps.dx();
  for (int j = 0; j < nvx; ++j) {
    const double delta = ps.vx(j) * dt * inv_dx;
    rep.max_shift_cells = std::max(rep.max_shift_cells, std::abs(delta));
    plan.prepare(delta);
    for (int k = 0; k < nvy; ++k) {
      // In the padded, x-fastest layout the owned run of an x line is
      // contiguous, so the transform reads and writes the field in place
      // with no gather.
      plan.apply(&f(0, j, k));
    }
  }

  if (work.measure_mass) rep.mass_after = PhaseSpace::local_sum(f);
  return rep;
}

/**
 * @brief Step B: `d_t f + a_x d_vx f = 0` with `a_x = (sigma/mu)(E_x + v_y B_z)`.
 *
 * `a_x` does not depend on `v_x`, so each line of constant `(x, v_y)` is a
 * rigid translation. `v_x` is rank-local, so **no halo is involved**: a
 * departure point outside `[-v_max, v_max]` contributes literal zero, which
 * is the zero-inflow boundary condition, and there is consequently no limit
 * on how far this step may shift. Mass that leaves is gone, and the amount is
 * in @ref TransportReport::mass_lost.
 *
 * Loop order is `v_y` (plane), then output `v_x`, then stencil point, then
 * `x`. The innermost loop runs along `x`, which is the contiguous axis, and
 * the source row index varies with `x` only through `floor(-alpha(x))`, which
 * is piecewise constant for a smooth `E_x`; so the gather is nearly a
 * contiguous copy.
 *
 * @param ps           Phase-space stack.
 * @param f            Distribution, advanced in place.
 * @param qm           `sigma/mu` of this species (@ref Species::qm).
 * @param dt           Step.
 * @param Ex, Bz       1-D fields over `x`, replicated on every rank.
 * @param interp_order Number of Lagrange points; also the order of accuracy.
 * @param work         Scratch; resized as needed.
 */
inline TransportReport advect_vx(const PhaseSpace &ps, PhaseField &f, double qm,
                                 double dt, std::span<const double> Ex,
                                 std::span<const double> Bz, int interp_order,
                                 TransportWorkspace &work) {
  const int nx = ps.nx();
  const int nvx = ps.nvx();
  const int nvy = ps.nvy_local();
  detail::check_interp_order(interp_order);
  detail::check_field_extent(Ex, nx, "E_x");
  detail::check_field_extent(Bz, nx, "B_z");
  work.resize_for_vx(ps);

  const int p = interp_order;
  const int first = lagrange_first_offset(p);
  const double inv_dvx = 1.0 / ps.dvx();

  TransportReport rep;
  rep.measured = work.measure_mass;
  rep.required_halo = 0; // v_x is rank-local; the boundary is a literal zero
  if (work.measure_mass) rep.mass_before = PhaseSpace::local_sum(f);

  for (int k = 0; k < nvy; ++k) {
    const double vy_k = ps.vy(k);
    for (int i = 0; i < nx; ++i) {
      // a_x = (sigma/mu) (E_x + v_y B_z): the +v_y B_z half of the Lorentz
      // cross product (parameters.hpp). A sign error here is invisible until
      // a gyro-motion test measures the rotation sense.
      const double ax = qm * (Ex[static_cast<std::size_t>(i)] +
                              vy_k * Bz[static_cast<std::size_t>(i)]);
      const double alpha = ax * dt * inv_dvx;
      rep.max_shift_cells = std::max(rep.max_shift_cells, std::abs(alpha));
      const double departure = -alpha;
      const double cell = std::floor(departure);
      work.base[static_cast<std::size_t>(i)] = static_cast<int>(cell);
      lagrange_weights(p, departure - cell,
                       &work.wts[static_cast<std::size_t>(i) * kMaxInterpOrder]);
    }

    double *out = work.plane.data();
    for (int jo = 0; jo < nvx; ++jo) {
      double *row = out + static_cast<std::size_t>(jo) * nx;
      for (int i = 0; i < nx; ++i) row[i] = 0.0;
      for (int m = 0; m < p; ++m) {
        for (int i = 0; i < nx; ++i) {
          const int js = jo + work.base[static_cast<std::size_t>(i)] + first + m;
          if (js < 0 || js >= nvx) continue; // zero inflow
          row[i] += work.wts[static_cast<std::size_t>(i) * kMaxInterpOrder +
                             static_cast<std::size_t>(m)] *
                    f(i, js, k);
        }
      }
    }
    for (int jo = 0; jo < nvx; ++jo) {
      const double *row = out + static_cast<std::size_t>(jo) * nx;
      for (int i = 0; i < nx; ++i) f(i, jo, k) = row[i];
    }
  }

  if (work.measure_mass) rep.mass_after = PhaseSpace::local_sum(f);
  return rep;
}

/**
 * @brief Step C: `d_t f + a_y d_vy f = 0` with `a_y = (sigma/mu)(E_y - v_x B_z)`.
 *
 * Same translation as step B, along the **distributed** axis. Three things
 * are therefore different and all three matter:
 *
 *  1. the required ghost width is computed, reduced with `MPI_MAX` and
 *     checked *before* any communication, so an overrun is an exception on
 *     every rank rather than a hang on one and wrong numbers on the rest;
 *  2. @ref PhaseSpace::exchange_vy runs inside this function, because a
 *     caller who forgets it gets an answer built from last step's ghosts,
 *     which looks fine;
 *  3. the gather cannot be done in place, so it writes a whole owned brick of
 *     scratch and copies back.
 *
 * @param ps           Phase-space stack.
 * @param f            Distribution, advanced in place.
 * @param qm           `sigma/mu` of this species.
 * @param dt           Step.
 * @param Ey, Bz       1-D fields over `x`, replicated on every rank.
 * @param interp_order Number of Lagrange points; also the order of accuracy.
 * @param work         Scratch; resized as needed.
 *
 * @throws std::runtime_error if the shift plus the stencil reaches past the
 *         allocated `v_y` halo.
 */
inline TransportReport advect_vy(const PhaseSpace &ps, PhaseField &f, double qm,
                                 double dt, std::span<const double> Ey,
                                 std::span<const double> Bz, int interp_order,
                                 TransportWorkspace &work) {
  const int nx = ps.nx();
  const int nvx = ps.nvx();
  const int nvy = ps.nvy_local();
  detail::check_interp_order(interp_order);
  detail::check_field_extent(Ey, nx, "E_y");
  detail::check_field_extent(Bz, nx, "B_z");
  work.resize_for_vy(ps);

  const int p = interp_order;
  const int first = lagrange_first_offset(p);
  const double inv_dvy = 1.0 / ps.dvy();
  const std::size_t plane =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(nvx);

  TransportReport rep;
  rep.measured = work.measure_mass;

  double local_max = 0.0;
  for (int j = 0; j < nvx; ++j) {
    const double vx_j = ps.vx(j);
    for (int i = 0; i < nx; ++i) {
      // a_y = (sigma/mu) (E_y - v_x B_z): the -v_x B_z half of the cross
      // product. Together with the +v_y B_z in step B this is what rotates
      // velocity vectors in the (v_x, v_y) plane.
      const double ay = qm * (Ey[static_cast<std::size_t>(i)] -
                              vx_j * Bz[static_cast<std::size_t>(i)]);
      const double alpha = ay * dt * inv_dvy;
      const std::size_t line =
          static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(i);
      local_max = std::max(local_max, std::abs(alpha));
      const double departure = -alpha;
      const double cell = std::floor(departure);
      work.base[line] = static_cast<int>(cell);
      lagrange_weights(p, departure - cell, &work.wts[line * kMaxInterpOrder]);
    }
  }

  // Agree on the requirement before anyone communicates. `E` and `B` are
  // replicated, so in a correct run every rank computes the same maximum and
  // this reduction is a consistency check; when they are not replicated it is
  // the difference between a clean exception and a deadlock.
  double global_max = local_max;
  if (ps.comm_size() > 1) {
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, ps.comm());
  }
  rep.max_shift_cells = global_max;
  rep.required_halo = required_halo_width(global_max, p);
  if (rep.required_halo > ps.halo_width()) {
    throw std::runtime_error(
        "advect_vy: a shift of " + std::to_string(global_max) +
        " v_y cells with a " + std::to_string(p) +
        "-point stencil needs a halo of " + std::to_string(rep.required_halo) +
        " but only " + std::to_string(ps.halo_width()) +
        " is allocated. Reduce dt, coarsen v_y, or build the PhaseSpace with "
        "required_halo_width(max|a_y| dt/dv_y, interp_order). This is an "
        "error and not a clamp on purpose: reading past the ghost ring "
        "produces a plausible answer, which is worse than no answer.");
  }

  if (work.measure_mass) rep.mass_before = PhaseSpace::local_sum(f);
  ps.exchange_vy(f);

  double *out = work.brick.data();
  for (int ko = 0; ko < nvy; ++ko) {
    double *dst = out + static_cast<std::size_t>(ko) * plane;
    for (std::size_t n = 0; n < plane; ++n) dst[n] = 0.0;
    for (int m = 0; m < p; ++m) {
      for (int j = 0; j < nvx; ++j) {
        const std::size_t row =
            static_cast<std::size_t>(j) * static_cast<std::size_t>(nx);
        for (int i = 0; i < nx; ++i) {
          const std::size_t line = row + static_cast<std::size_t>(i);
          // Guaranteed inside [-hw, nvy+hw) by the check above; ghosts at the
          // global ends hold zeros, which is the zero-inflow condition.
          const int ks = ko + work.base[line] + first + m;
          dst[line] +=
              work.wts[line * kMaxInterpOrder + static_cast<std::size_t>(m)] *
              f(i, j, ks);
        }
      }
    }
  }
  for (int ko = 0; ko < nvy; ++ko) {
    const double *src = out + static_cast<std::size_t>(ko) * plane;
    for (int j = 0; j < nvx; ++j) {
      const std::size_t row =
          static_cast<std::size_t>(j) * static_cast<std::size_t>(nx);
      for (int i = 0; i < nx; ++i) {
        f(i, j, ko) = src[row + static_cast<std::size_t>(i)];
      }
    }
  }

  if (work.measure_mass) rep.mass_after = PhaseSpace::local_sum(f);
  return rep;
}

/**
 * @brief Upper bound on `|a_x| dt / dv_x` from the field extrema.
 *
 * `a_x = (sigma/mu)(E_x + v_y B_z)` and `|v_y| <= v_max`, so the bound is
 * closed-form and needs no pass over the brick. Step B does not need a halo,
 * so this is for the time-step controller and the diagnostics, not for a
 * guard.
 */
[[nodiscard]] inline double max_vx_shift_cells(const SimParams &p, double qm,
                                               double dt, std::span<const double> Ex,
                                               std::span<const double> Bz) {
  double worst = 0.0;
  for (std::size_t i = 0; i < Ex.size(); ++i) {
    worst = std::max(worst, std::abs(Ex[i]) + p.v_max * std::abs(Bz[i]));
  }
  return std::abs(qm) * worst * std::abs(dt) / p.dvx();
}

/**
 * @brief Upper bound on `|a_y| dt / dv_y` from the field extrema.
 *
 * The number to feed @ref required_halo_width when sizing the `v_y` halo at
 * start-up, before any field exists on the grid. `a_y = (sigma/mu)(E_y -
 * v_x B_z)` and `|v_x| <= v_max`.
 */
[[nodiscard]] inline double max_vy_shift_cells(const SimParams &p, double qm,
                                               double dt, std::span<const double> Ey,
                                               std::span<const double> Bz) {
  double worst = 0.0;
  for (std::size_t i = 0; i < Ey.size(); ++i) {
    worst = std::max(worst, std::abs(Ey[i]) + p.v_max * std::abs(Bz[i]));
  }
  return std::abs(qm) * worst * std::abs(dt) / p.dvy();
}

} // namespace vlasov
