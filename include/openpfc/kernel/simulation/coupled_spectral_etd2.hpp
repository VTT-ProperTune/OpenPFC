// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file coupled_spectral_etd2.hpp
 * @brief Second-order exponential time differencing for two coupled real
 *        fields with a 2 × 2 Fourier symbol.
 *
 * @details
 * Integrates `U_t = L U + N(t, U)` for `U = (u0, u1)`, where every Fourier
 * mode `k` carries a constant real 2 × 2 matrix `L(k)`. One step of size `h`
 * is the second-order exponential Runge–Kutta method (Cox & Matthews 2002,
 * "ETD2RK"; Hochbruck & Ostermann 2005, second-order scheme with
 * `c2 = 1`):
 *
 *   A       = exp(hL) U_n + h φ1(hL) N(t_n, U_n)
 *   U_{n+1} = A + h φ2(hL) [N(t_n + h, A) − N(t_n, U_n)]
 *
 * `A` is the ETD1 step. It is the embedded low-order solution, so
 * `U_{n+1} − A` is the local error estimate of the first-order method,
 * `O(h²)`. Use `error_order = 2` in @ref AdaptiveTimeController. The
 * accepted solution is the second-order `U_{n+1}` (local extrapolation).
 *
 * The stage callback is
 *
 *   stage(t, u0, u1, f0, f1)
 *
 * with `std::span<const double>` inputs and `std::span<double>` outputs of
 * `fft.size_inbox()` entries. With @ref CoupledStageOutput::full_rhs it
 * returns the whole right-hand side `F = L U + N` and the system forms
 * `N̂ = F̂ − L Û` in Fourier space; the split is then exact whenever `L(k)`
 * is the symbol of the operator the callback applies. With
 * @ref CoupledStageOutput::nonlinear it returns `N` itself.
 *
 * `L(k)` must be real and even in `k`, so that `L` maps real fields to real
 * fields on the half-complex outbox.
 *
 * `attempt` leaves the accepted fields and their transforms untouched.
 * `commit` swaps the candidate in; `reject` keeps everything. `N(t_n, U_n)`
 * is cached across rejected attempts from the same accepted state, so a
 * retry costs one stage evaluation. The coefficients `exp(hL)`, `h φ1(hL)`,
 * `h φ2(hL)` are rebuilt only when `h` or `L` changes.
 *
 * Cost of a first attempt: two stage evaluations, four forward and four
 * backward transforms. A retry: one stage, two forward, four backward.
 *
 * Host memory space and `IHostFFT` only. Every rank must pass the same `t`
 * and `h`.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/integrator/matrix_phi_2x2.hpp>
#include <openpfc/kernel/simulation/adaptive_controller.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace pfc::sim {

/// What the stage callback of @ref CoupledSpectralETD2 returns.
enum class CoupledStageOutput : std::uint8_t {
  full_rhs, ///< `F = L U + N`; the system subtracts `L Û`
  nonlinear ///< `N` only
};

/// `stage(t, u0, u1, f0, f1)` for @ref CoupledSpectralETD2.
template <class F>
concept CoupledStageFunction =
    std::invocable<F &, double, std::span<const double>, std::span<const double>,
                   std::span<double>, std::span<double>>;

/// Cumulative work since construction or `reset_statistics`.
struct CoupledETD2Statistics {
  long attempts{0};
  long stage_evaluations{0};
  long forward_transforms{0};
  long backward_transforms{0};
  long coefficient_builds{0};
  double coefficient_seconds{0.0}; ///< building exp(hL), hφ1, hφ2
  double stage_seconds{0.0};       ///< inside the stage callback
  double transform_seconds{0.0};   ///< forward and backward FFTs
  double combine_seconds{0.0};     ///< per-mode 2 × 2 updates
};

/// One `attempt`: interval and whether the candidate is finite.
struct CoupledETD2Attempt {
  double t0{0.0};
  double dt{0.0};
  double t1{0.0};
  bool finite{false};
};

/**
 * @brief Evaluate a 2 × 2 symbol on this rank's outbox.
 *
 * `symbol(kx, ky, kz)` uses the same wavenumbers and ordering as
 * @ref pfc::fft::kspace::for_each_kpoint.
 */
template <class Symbol>
[[nodiscard]] std::vector<integrator::Matrix2x2>
coupled_linear_symbol(const fft::IFFTQueries &fft, const Box3i &outbox,
                      const Domain &domain, Symbol &&symbol) {
  std::vector<integrator::Matrix2x2> out(fft.size_outbox());
  fft::kspace::for_each_kpoint(outbox, domain,
                               [&](std::size_t idx, double kx, double ky, double kz,
                                   int, int,
                                   int) { out[idx] = symbol(kx, ky, kz); });
  return out;
}

/**
 * @brief Coupled two-field ETD2 on the host FFT.
 *
 * @tparam Stage models @ref CoupledStageFunction.
 */
template <class Stage>
  requires CoupledStageFunction<Stage>
class CoupledSpectralETD2 {
public:
  using Complex = std::complex<double>;
  using Matrix = integrator::Matrix2x2;
  using Coefficients = integrator::ETD2Coefficients2x2;

  CoupledSpectralETD2(fft::IHostFFT &fft, std::vector<Matrix> symbol, Stage stage,
                      CoupledStageOutput output = CoupledStageOutput::full_rhs)
      : m_fft(fft), m_stage(std::move(stage)), m_output(output),
        m_n(fft.size_inbox()), m_m(fft.size_outbox()) {
    for (int f = 0; f < 2; ++f) {
      m_u[f].assign(m_n, 0.0);
      m_a[f].assign(m_n, 0.0);
      m_next[f].assign(m_n, 0.0);
      m_work[f].assign(m_n, 0.0);
      m_u_hat[f].assign(m_m, Complex{});
      m_a_hat[f].assign(m_m, Complex{});
      m_next_hat[f].assign(m_m, Complex{});
      m_n0_hat[f].assign(m_m, Complex{});
      m_n1_hat[f].assign(m_m, Complex{});
    }
    m_error.assign(2 * m_n, 0.0);
    m_accepted.assign(2 * m_n, 0.0);
    m_candidate.assign(2 * m_n, 0.0);
    set_linear_symbol(std::move(symbol));
  }

  CoupledSpectralETD2(const CoupledSpectralETD2 &) = delete;
  CoupledSpectralETD2 &operator=(const CoupledSpectralETD2 &) = delete;

  /**
   * @brief Replace `L(k)`. Drops the coefficients and the cached `N̂(U_n)`.
   * @throws std::invalid_argument on a size mismatch or a non-finite entry.
   */
  void set_linear_symbol(std::vector<Matrix> symbol) {
    if (symbol.size() != m_m) {
      throw std::invalid_argument(
          "CoupledSpectralETD2: symbol size != FFT outbox size");
    }
    for (const auto &s : symbol) {
      if (!std::isfinite(s.m00) || !std::isfinite(s.m01) || !std::isfinite(s.m10) ||
          !std::isfinite(s.m11)) {
        throw std::invalid_argument("CoupledSpectralETD2: non-finite symbol entry");
      }
    }
    m_symbol = std::move(symbol);
    m_coeff_dt = 0.0;
    invalidate_stage_cache();
  }

  /// Accepted real field `i` (0 or 1). Call @ref load_state after writing it.
  [[nodiscard]] std::span<double> field(std::size_t i) { return m_u[check(i)]; }
  [[nodiscard]] std::span<const double> field(std::size_t i) const {
    return m_u[check(i)];
  }
  /// Fourier transform of accepted field `i`.
  [[nodiscard]] std::span<const Complex> field_hat(std::size_t i) const {
    return m_u_hat[check(i)];
  }

  /// Transform the accepted fields after the caller wrote them.
  void load_state() {
    require_closed("load_state");
    for (int f = 0; f < 2; ++f) {
      forward(m_u[f], m_u_hat[f]);
    }
    invalidate_stage_cache();
  }

  /**
   * @brief Form the ETD2 candidate and its ETD1 predictor for `[t, t + dt]`.
   *
   * Does not modify the accepted state. Throws if an attempt is open.
   */
  CoupledETD2Attempt attempt(double t, double dt) {
    require_closed("attempt");
    if (!(dt > 0.0) || !std::isfinite(dt)) {
      throw std::invalid_argument(
          "CoupledSpectralETD2::attempt: dt must be finite and > 0");
    }
    m_open = true;
    ++m_stats.attempts;
    ensure_coefficients(dt);

    // N̂(t_n, U_n), reused by a retry from the same accepted state.
    if (!(m_cache_valid && m_cache_t == t)) {
      evaluate_stage(t, m_u, m_u_hat, m_n0_hat);
      m_cache_valid = true;
      m_cache_t = t;
    }

    // Â = exp(hL) Û + h φ1(hL) N̂0, then A in real space.
    {
      const auto t0 = clock::now();
      for (std::size_t i = 0; i < m_m; ++i) {
        const Coefficients &c = m_coeff[i];
        const Complex u0 = m_u_hat[0][i];
        const Complex u1 = m_u_hat[1][i];
        const Complex n0 = m_n0_hat[0][i];
        const Complex n1 = m_n0_hat[1][i];
        m_a_hat[0][i] = c.exp_hL.m00 * u0 + c.exp_hL.m01 * u1 + c.h_phi1.m00 * n0 +
                        c.h_phi1.m01 * n1;
        m_a_hat[1][i] = c.exp_hL.m10 * u0 + c.exp_hL.m11 * u1 + c.h_phi1.m10 * n0 +
                        c.h_phi1.m11 * n1;
      }
      m_stats.combine_seconds += seconds_since(t0);
    }
    for (int f = 0; f < 2; ++f) {
      backward(m_a_hat[f], m_a[f]);
    }

    // N̂(t_n + h, A), then Û⁺ = Â + h φ2(hL) (N̂1 − N̂0).
    evaluate_stage(t + dt, m_a, m_a_hat, m_n1_hat);
    {
      const auto t0 = clock::now();
      for (std::size_t i = 0; i < m_m; ++i) {
        const Matrix &p = m_coeff[i].h_phi2;
        const Complex d0 = m_n1_hat[0][i] - m_n0_hat[0][i];
        const Complex d1 = m_n1_hat[1][i] - m_n0_hat[1][i];
        m_next_hat[0][i] = m_a_hat[0][i] + p.m00 * d0 + p.m01 * d1;
        m_next_hat[1][i] = m_a_hat[1][i] + p.m10 * d0 + p.m11 * d1;
      }
      m_stats.combine_seconds += seconds_since(t0);
    }
    for (int f = 0; f < 2; ++f) {
      backward(m_next_hat[f], m_next[f]);
    }

    bool finite = true;
    for (int f = 0; f < 2; ++f) {
      const std::size_t off = static_cast<std::size_t>(f) * m_n;
      for (std::size_t i = 0; i < m_n; ++i) {
        const double y = m_next[f][i];
        finite = finite && std::isfinite(y);
        m_error[off + i] = y - m_a[f][i];
        m_accepted[off + i] = m_u[f][i];
        m_candidate[off + i] = y;
      }
    }
    m_last = CoupledETD2Attempt{.t0 = t, .dt = dt, .t1 = t + dt, .finite = finite};
    return m_last;
  }

  /// Accept the open attempt: the candidate becomes the accepted state.
  void commit() {
    require_open("commit");
    for (int f = 0; f < 2; ++f) {
      std::swap(m_u[f], m_next[f]);
      std::swap(m_u_hat[f], m_next_hat[f]);
    }
    invalidate_stage_cache();
    m_open = false;
  }

  /// Discard the open attempt. The accepted state was never written.
  void reject() {
    require_open("reject");
    m_open = false;
  }

  /// One fixed step: `attempt` then `commit`. Returns `t + dt`.
  double step(double t, double dt) {
    const auto a = attempt(t, dt);
    if (!a.finite) {
      reject();
      throw std::runtime_error("CoupledSpectralETD2::step: candidate is not finite");
    }
    commit();
    return a.t1;
  }

  [[nodiscard]] bool attempt_open() const noexcept { return m_open; }
  [[nodiscard]] const CoupledETD2Attempt &last_attempt() const noexcept {
    return m_last;
  }

  /// Second-order candidate of field `i`, real space. Valid until the next attempt.
  [[nodiscard]] std::span<const double> candidate(std::size_t i) const {
    return m_next[check(i)];
  }
  /// ETD1 predictor `A` of field `i`, real space.
  [[nodiscard]] std::span<const double> predictor(std::size_t i) const {
    return m_a[check(i)];
  }
  /// `U_{n+1} − A` for both fields, field 0 first. Feed the controller.
  [[nodiscard]] std::span<const double> embedded_error() const noexcept {
    return m_error;
  }
  /// Accepted state at the start of the last attempt, field 0 first.
  [[nodiscard]] std::span<const double> attempt_start() const noexcept {
    return m_accepted;
  }
  /// Candidate of the last attempt, field 0 first.
  [[nodiscard]] std::span<const double> attempt_candidate() const noexcept {
    return m_candidate;
  }

  [[nodiscard]] std::span<const Matrix> linear_symbol() const noexcept {
    return m_symbol;
  }
  /// Coefficients for the step size of the last attempt.
  [[nodiscard]] std::span<const Coefficients> coefficients() const noexcept {
    return m_coeff;
  }
  /// Step size the coefficients were built for, 0 when none are built.
  [[nodiscard]] double coefficient_dt() const noexcept { return m_coeff_dt; }

  [[nodiscard]] const CoupledETD2Statistics &statistics() const noexcept {
    return m_stats;
  }
  void reset_statistics() noexcept { m_stats = {}; }

  [[nodiscard]] Stage &stage() noexcept { return m_stage; }
  [[nodiscard]] std::size_t size() const noexcept { return m_n; }
  [[nodiscard]] std::size_t modes() const noexcept { return m_m; }

private:
  using clock = std::chrono::steady_clock;

  static double seconds_since(clock::time_point t0) {
    return std::chrono::duration<double>(clock::now() - t0).count();
  }

  static std::size_t check(std::size_t i) {
    if (i > 1) {
      throw std::out_of_range("CoupledSpectralETD2: field index must be 0 or 1");
    }
    return i;
  }

  void require_open(const char *what) const {
    if (!m_open) {
      throw std::logic_error(std::string("CoupledSpectralETD2::") + what +
                             ": no open attempt");
    }
  }
  void require_closed(const char *what) const {
    if (m_open) {
      throw std::logic_error(std::string("CoupledSpectralETD2::") + what +
                             ": an attempt is open");
    }
  }

  void invalidate_stage_cache() noexcept { m_cache_valid = false; }

  void ensure_coefficients(double dt) {
    if (m_coeff_dt == dt && m_coeff.size() == m_m) {
      return;
    }
    const auto t0 = clock::now();
    m_coeff.resize(m_m);
    for (std::size_t i = 0; i < m_m; ++i) {
      m_coeff[i] = integrator::etd2_coefficients(m_symbol[i], dt);
    }
    m_coeff_dt = dt;
    ++m_stats.coefficient_builds;
    m_stats.coefficient_seconds += seconds_since(t0);
  }

  void forward(std::vector<double> &in, std::vector<Complex> &out) {
    const auto t0 = clock::now();
    m_fft.forward(in, out);
    ++m_stats.forward_transforms;
    m_stats.transform_seconds += seconds_since(t0);
  }
  void backward(const std::vector<Complex> &in, std::vector<double> &out) {
    const auto t0 = clock::now();
    m_fft.backward(in, out);
    ++m_stats.backward_transforms;
    m_stats.transform_seconds += seconds_since(t0);
  }

  /// `n_hat = F̂ − L Û` (or `N̂`) for the stage at time `t` and state `u`.
  void evaluate_stage(double t, const std::vector<double> (&u)[2],
                      const std::vector<Complex> (&u_hat)[2],
                      std::vector<Complex> (&n_hat)[2]) {
    {
      const auto t0 = clock::now();
      m_stage(t, std::span<const double>(u[0]), std::span<const double>(u[1]),
              std::span<double>(m_work[0]), std::span<double>(m_work[1]));
      ++m_stats.stage_evaluations;
      m_stats.stage_seconds += seconds_since(t0);
    }
    for (int f = 0; f < 2; ++f) {
      forward(m_work[f], n_hat[f]);
    }
    if (m_output == CoupledStageOutput::full_rhs) {
      const auto t0 = clock::now();
      for (std::size_t i = 0; i < m_m; ++i) {
        const Matrix &l = m_symbol[i];
        const Complex u0 = u_hat[0][i];
        const Complex u1 = u_hat[1][i];
        n_hat[0][i] -= l.m00 * u0 + l.m01 * u1;
        n_hat[1][i] -= l.m10 * u0 + l.m11 * u1;
      }
      m_stats.combine_seconds += seconds_since(t0);
    }
  }

  fft::IHostFFT &m_fft;
  Stage m_stage;
  CoupledStageOutput m_output;
  std::size_t m_n{0};
  std::size_t m_m{0};

  std::vector<Matrix> m_symbol;
  std::vector<Coefficients> m_coeff;
  double m_coeff_dt{0.0};

  std::vector<double> m_u[2];
  std::vector<double> m_a[2];
  std::vector<double> m_next[2];
  std::vector<double> m_work[2];
  std::vector<Complex> m_u_hat[2];
  std::vector<Complex> m_a_hat[2];
  std::vector<Complex> m_next_hat[2];
  std::vector<Complex> m_n0_hat[2];
  std::vector<Complex> m_n1_hat[2];
  std::vector<double> m_error;
  std::vector<double> m_accepted;
  std::vector<double> m_candidate;

  bool m_open{false};
  bool m_cache_valid{false};
  double m_cache_t{0.0};
  CoupledETD2Attempt m_last{};
  CoupledETD2Statistics m_stats{};
};

/// Options of @ref run_coupled_etd2_adaptive.
struct CoupledETD2RunOptions {
  /// After an accepted step, keep the attempted `dt` when the controller
  /// proposes `next ∈ [dt, hold_band · dt)`. 1 disables the hold. A hold
  /// avoids rebuilding the coefficients for a marginal growth; a proposed
  /// shrink is always taken.
  double hold_band{1.0};
  MPI_Comm comm{MPI_COMM_SELF};
};

/// Accepted-step statistics of one adaptive run.
struct CoupledETD2RunSummary {
  int accepted{0};
  int rejected{0};
  double dt_min{std::numeric_limits<double>::infinity()};
  double dt_max{0.0};
  double dt_sum{0.0};
  [[nodiscard]] double dt_mean() const noexcept {
    return accepted > 0 ? dt_sum / static_cast<double>(accepted) : 0.0;
  }
};

/**
 * @brief Drive @p system to `time`'s end with embedded ETD2(1) control.
 *
 * Each pass opens a `Time` attempt (clipped to the end and save times),
 * forms the candidate, asks @p controller from `U_{n+1} − A` scaled by
 * `max(|U_n|, |U_{n+1}|)`, then commits or rejects both the system and
 * `Time` through `AdaptiveTimeController::apply`. A rejection leaves the
 * accepted fields and time unchanged. @p on_accepted is called as
 * `on_accepted(const Time &)` after each accepted step.
 *
 * Construct @p controller with `error_order = 2`.
 */
template <class Stage, class OnAccepted = NoopHook>
CoupledETD2RunSummary
run_coupled_etd2_adaptive(CoupledSpectralETD2<Stage> &system, Time &time,
                          AdaptiveTimeController &controller,
                          const CoupledETD2RunOptions &options = {},
                          OnAccepted &&on_accepted = {}) {
  if (!(options.hold_band >= 1.0)) {
    throw std::invalid_argument("run_coupled_etd2_adaptive: hold_band must be >= 1");
  }
  CoupledETD2RunSummary summary;
  while (!pfc::time::done(time)) {
    time.begin_attempt(time.get_dt());
    detail::AttemptGuard guard{time};
    const double t0 = time.get_accepted_time();
    const double h = time.get_attempted_dt();
    system.attempt(t0, h);
    AdaptiveDecision d = controller.decide_from_embedded_error(
        h, system.embedded_error(), system.attempt_start(),
        system.attempt_candidate(), options.comm);
    const bool accept = d.accepted && d.decision_available;
    if (accept) {
      system.commit();
      if (d.next_dt >= h && d.next_dt < options.hold_band * h) {
        d.next_dt = h;
      }
    } else {
      system.reject();
    }
    guard.dismiss();
    controller.apply(time, d);
    if (accept) {
      ++summary.accepted;
      summary.dt_min = std::min(summary.dt_min, h);
      summary.dt_max = std::max(summary.dt_max, h);
      summary.dt_sum += h;
      on_accepted(static_cast<const Time &>(time));
    } else {
      ++summary.rejected;
    }
  }
  return summary;
}

} // namespace pfc::sim
