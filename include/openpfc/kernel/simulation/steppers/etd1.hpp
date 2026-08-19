// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file etd1.hpp
 * @brief First-order exponential time-differencing (ETD1) step-attempt API.
 *
 * @details
 * Advances a diagonal spectral ODE via the classic ETD1 update
 *
 *     u_{n+1} = exp(dt*L) u_n + dt * phi_1(dt*L) N(u_n, t_n)
 *
 * with @c phi_1(z) = (exp(z)-1)/z and @c phi_1(0) = 1. Coefficient spans from
 * @ref pfc::integrator::fill_spectral_exp_coeffs /
 * @ref pfc::integrator::SpectralExpCoefficientCache already store
 * @c exp_Ldt = exp(L*dt) and @c phi1_L = (exp(L*dt)-1)/L (= @c dt * phi_1(L*dt)).
 * Therefore the method applies
 *
 *     candidate[i] = exp_Ldt[i] * u[i] + phi1_L[i] * N[i]
 *
 * and must **not** multiply @c phi1_L by @c dt again.
 *
 * The accepted solution buffer is never written. Nonlinear evaluation uses a
 * method-owned scratch copy so a misbehaving @c StageFunction cannot mutate
 * caller state. Transient coefficient / scratch caches are recomputable and
 * are **not** checkpointable method state.
 *
 * This header is frontend-free and does not hard-wire HeFFTe types.
 *
 * @see openpfc/kernel/integrator/spectral_exp_coefficients.hpp
 * @see docs/development/time_integration_architecture.md
 */

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/integrator/etd1_apply.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/state_concepts.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>

namespace pfc::sim::steppers {

namespace detail {

template <class T> [[nodiscard]] bool is_finite_scalar(const T &x) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::isfinite(x);
  } else {
    return std::isfinite(x.real()) && std::isfinite(x.imag());
  }
}

} // namespace detail

/**
 * @brief CPU ETD1 stepper with isolated candidate state.
 *
 * @tparam Rhs    Callable satisfying @ref StageFunctionFor for @p Scalar.
 * @tparam Scalar Field element type (`double` or `std::complex<double>`).
 *                Real diagonal coefficients (`exp_Ldt`, `phi1_L`) are applied
 *                as `Scalar(coeff) * value`.
 *
 * Coefficient ownership:
 * - **Caller-lent spans** via @ref set_coefficients(std::span, std::span):
 *   views must remain valid until the next @c set_coefficients or destruction.
 * - **Method-owned copy** via the @c SpectralExpCoefficientCache overload or
 *   @ref set_coefficients_owned: copies into internal vectors so the source
 *   may be dropped.
 *
 * Transient caches are not checkpointable method state.
 */
template <class Rhs, class Scalar = double>
  requires StageFunctionFor<Rhs, Scalar>
class Etd1Stepper {
public:
  using scalar_type = Scalar;
  using Attempt = StepAttempt<Scalar>;

  Etd1Stepper(double dt, std::size_t local_size, Rhs rhs)
      : m_dt(dt), m_local_size(local_size), m_du(local_size, Scalar{}),
        m_candidate(local_size, Scalar{}), m_u_scratch(local_size, Scalar{}),
        m_rhs(std::move(rhs)) {}

  /**
   * @brief Bind caller-lent coefficient spans.
   *
   * Requires @p exp_Ldt and @p phi1_L to have equal length. Matching against
   * @c local_size / @c u_accepted is deferred to @ref attempt so a
   * size-mismatch failure path remains reachable.
   *
   * Spans must outlive the next @ref attempt (or be replaced via a later
   * @c set_coefficients / owned copy).
   */
  void set_coefficients(std::span<const double> exp_Ldt,
                        std::span<const double> phi1_L) {
    if (exp_Ldt.size() != phi1_L.size()) {
      throw std::invalid_argument(
          "Etd1Stepper::set_coefficients: exp_Ldt.size() != phi1_L.size()");
    }
    m_owned_exp.clear();
    m_owned_phi1.clear();
    m_exp_Ldt = exp_Ldt;
    m_phi1_L = phi1_L;
  }

  /**
   * @brief Copy coefficients into method-owned storage.
   *
   * Prefer this when binding a cache that may rebuild, or when the source
   * spans would otherwise go out of scope before @ref attempt.
   */
  void set_coefficients_owned(std::span<const double> exp_Ldt,
                              std::span<const double> phi1_L) {
    if (exp_Ldt.size() != phi1_L.size()) {
      throw std::invalid_argument(
          "Etd1Stepper::set_coefficients_owned: exp_Ldt.size() != "
          "phi1_L.size()");
    }
    m_owned_exp.assign(exp_Ldt.begin(), exp_Ldt.end());
    m_owned_phi1.assign(phi1_L.begin(), phi1_L.end());
    m_exp_Ldt = m_owned_exp;
    m_phi1_L = m_owned_phi1;
  }

  /**
   * @brief Copy views from a @ref pfc::integrator::SpectralExpCoefficientCache.
   *
   * The cache may be rebuilt afterward; this overload owns independent copies.
   */
  void set_coefficients(
      const pfc::integrator::SpectralExpCoefficientCache<> &cache) {
    set_coefficients_owned(cache.exp_Ldt(), cache.phi1_L());
  }

  /**
   * @brief Form an isolated ETD1 candidate without mutating @p u_accepted.
   *
   * Algorithm: size-check → copy accepted into scratch → evaluate @c N on
   * scratch → @c candidate = exp_Ldt * u_accepted + phi1_L * N.
   */
  [[nodiscard]] Attempt attempt(double t,
                                const std::vector<Scalar> &u_accepted) {
    m_last_reason.clear();
    if (u_accepted.size() != m_local_size) {
      m_last_reason = "u_accepted.size() != local_size";
      return Attempt(t, m_dt, t, /*success=*/false, m_candidate);
    }
    if (m_exp_Ldt.size() != m_local_size || m_phi1_L.size() != m_local_size) {
      m_last_reason = "coefficient span size != local_size";
      return Attempt(t, m_dt, t, /*success=*/false, m_candidate);
    }

    m_u_scratch = u_accepted;
    m_rhs(t, m_u_scratch, m_du);

    pfc::integrator::apply_etd1_update(
        m_exp_Ldt, m_phi1_L, std::span<const Scalar>(u_accepted),
        std::span<const Scalar>(m_du), std::span<Scalar>(m_candidate));
    for (std::size_t i = 0; i < m_local_size; ++i) {
      if (!detail::is_finite_scalar(m_candidate[i])) {
        m_last_reason = "non-finite candidate value";
        return Attempt(t, m_dt, t, /*success=*/false, m_candidate);
      }
    }
    return Attempt(t, m_dt, t + m_dt, /*success=*/true, m_candidate);
  }

  /** Isolate a candidate from host field state (via `vec()`). */
  template <pfc::field::HostFieldState<Scalar> F>
  [[nodiscard]] Attempt attempt(double t, const F &u) {
    return attempt(t, u.vec());
  }

  [[nodiscard]] std::span<const Scalar> candidate() const noexcept {
    return m_candidate;
  }

  [[nodiscard]] const std::string &last_reason() const noexcept {
    return m_last_reason;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }

  [[nodiscard]] std::size_t local_size() const noexcept { return m_local_size; }

private:
  double m_dt{0.0};
  std::size_t m_local_size{0};
  std::vector<Scalar> m_du;
  std::vector<Scalar> m_candidate;
  std::vector<Scalar> m_u_scratch;
  std::vector<double> m_owned_exp;
  std::vector<double> m_owned_phi1;
  std::span<const double> m_exp_Ldt{};
  std::span<const double> m_phi1_L{};
  std::string m_last_reason{};
  Rhs m_rhs;
};

/**
 * @brief N-field ETD1 stepper with per-field isolated candidates.
 *
 * @tparam Rhs    Callable satisfying @ref MultiStageFunction with arity N
 *                and element type @p Scalar.
 * @tparam N      Field count (`N >= 1`).
 * @tparam Scalar Field element type (`double` or `std::complex<double>`).
 *                Real diagonal coefficients are applied as `Scalar(coeff) *
 *                value`.
 *
 * Each accepted field is copied into method-owned scratch before the
 * multi-field nonlinear evaluation so a misbehaving @c N cannot mutate
 * caller buffers.
 */
template <class Rhs, std::size_t N, class Scalar = double>
  requires(N >= 1) && MultiStageFunction<Rhs, N, Scalar>
class MultiEtd1Stepper {
public:
  using RhsType = Rhs;
  using scalar_type = Scalar;
  static constexpr std::size_t field_count = N;

  MultiEtd1Stepper(double dt, std::array<std::size_t, N> local_sizes, Rhs rhs)
      : m_dt(dt), m_local_sizes(local_sizes), m_rhs(std::move(rhs)) {
    for (std::size_t f = 0; f < N; ++f) {
      m_du[f].assign(local_sizes[f], Scalar{});
      m_candidate[f].assign(local_sizes[f], Scalar{});
      m_u_scratch[f].assign(local_sizes[f], Scalar{});
    }
  }

  /**
   * @brief Bind caller-lent per-field coefficient spans.
   *
   * Per field, @c exp and @c phi1 must have equal length. Matching against
   * each field's @c local_size is deferred to @ref attempt.
   */
  void set_coefficients(std::array<std::span<const double>, N> exp_Ldt,
                        std::array<std::span<const double>, N> phi1_L) {
    for (std::size_t f = 0; f < N; ++f) {
      if (exp_Ldt[f].size() != phi1_L[f].size()) {
        throw std::invalid_argument(
            "MultiEtd1Stepper::set_coefficients: per-field exp/phi1 size "
            "mismatch");
      }
      m_owned_exp[f].clear();
      m_owned_phi1[f].clear();
      m_exp_Ldt[f] = exp_Ldt[f];
      m_phi1_L[f] = phi1_L[f];
    }
  }

  void set_coefficients_owned(std::array<std::span<const double>, N> exp_Ldt,
                              std::array<std::span<const double>, N> phi1_L) {
    for (std::size_t f = 0; f < N; ++f) {
      if (exp_Ldt[f].size() != phi1_L[f].size()) {
        throw std::invalid_argument(
            "MultiEtd1Stepper::set_coefficients_owned: per-field exp/phi1 "
            "size mismatch");
      }
      m_owned_exp[f].assign(exp_Ldt[f].begin(), exp_Ldt[f].end());
      m_owned_phi1[f].assign(phi1_L[f].begin(), phi1_L[f].end());
      m_exp_Ldt[f] = m_owned_exp[f];
      m_phi1_L[f] = m_owned_phi1[f];
    }
  }

  /**
   * @brief Form isolated per-field candidates without mutating accepted inputs.
   */
  template <class... U>
  [[nodiscard]] MultiStepAttemptResult<N, Scalar>
  attempt(double t, const std::vector<U> &...u_accepted) {
    static_assert(sizeof...(U) == N,
                  "MultiEtd1Stepper::attempt: buffer count must match N");
    static_assert((std::is_same_v<U, Scalar> && ...),
                  "MultiEtd1Stepper requires std::vector<Scalar>");
    m_last_reason.clear();
    const std::array<const std::vector<Scalar> *, N> accepted{&u_accepted...};
    for (std::size_t f = 0; f < N; ++f) {
      if (accepted[f]->size() != m_local_sizes[f]) {
        m_last_reason = "accepted field size != local_size";
        return MultiStepAttemptResult<N, Scalar>(t, m_dt, t, /*success=*/false,
                                                 candidate_ptrs());
      }
      if (m_exp_Ldt[f].size() != m_local_sizes[f] ||
          m_phi1_L[f].size() != m_local_sizes[f]) {
        m_last_reason = "coefficient span size != local_size";
        return MultiStepAttemptResult<N, Scalar>(t, m_dt, t, /*success=*/false,
                                                 candidate_ptrs());
      }
    }

    copy_accepted_to_scratch(std::make_index_sequence<N>{}, u_accepted...);
    auto u_pack = make_scratch_tuple(std::make_index_sequence<N>{});
    auto du_pack = make_du_tuple(std::make_index_sequence<N>{});
    m_rhs(t, u_pack, du_pack);

    for (std::size_t f = 0; f < N; ++f) {
      const auto &u_acc = *accepted[f];
      pfc::integrator::apply_etd1_update(
          m_exp_Ldt[f], m_phi1_L[f], std::span<const Scalar>(u_acc),
          std::span<const Scalar>(m_du[f]), std::span<Scalar>(m_candidate[f]));
      for (std::size_t i = 0; i < m_local_sizes[f]; ++i) {
        if (!detail::is_finite_scalar(m_candidate[f][i])) {
          m_last_reason = "non-finite candidate value";
          return MultiStepAttemptResult<N, Scalar>(t, m_dt, t,
                                                   /*success=*/false,
                                                   candidate_ptrs());
        }
      }
    }
    return MultiStepAttemptResult<N, Scalar>(t, m_dt, t + m_dt,
                                             /*success=*/true,
                                             candidate_ptrs());
  }

  [[nodiscard]] std::span<const Scalar>
  candidate(std::size_t field_index) const noexcept {
    return m_candidate[field_index];
  }

  [[nodiscard]] const std::string &last_reason() const noexcept {
    return m_last_reason;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }

private:
  double m_dt{0.0};
  std::array<std::size_t, N> m_local_sizes{};
  std::array<std::vector<Scalar>, N> m_du{};
  std::array<std::vector<Scalar>, N> m_candidate{};
  std::array<std::vector<Scalar>, N> m_u_scratch{};
  std::array<std::vector<double>, N> m_owned_exp{};
  std::array<std::vector<double>, N> m_owned_phi1{};
  std::array<std::span<const double>, N> m_exp_Ldt{};
  std::array<std::span<const double>, N> m_phi1_L{};
  std::string m_last_reason{};
  Rhs m_rhs;

  template <std::size_t... I, class... U>
  void copy_accepted_to_scratch(std::index_sequence<I...>,
                                const std::vector<U> &...u_accepted) {
    ((m_u_scratch[I] = u_accepted), ...);
  }

  template <std::size_t... I>
  auto make_scratch_tuple(std::index_sequence<I...>) {
    return std::tie(m_u_scratch[I]...);
  }

  template <std::size_t... I> auto make_du_tuple(std::index_sequence<I...>) {
    return std::tie(m_du[I]...);
  }

  [[nodiscard]] std::array<const std::vector<Scalar> *, N>
  candidate_ptrs() const {
    return candidate_ptrs_impl(std::make_index_sequence<N>{});
  }

  template <std::size_t... I>
  [[nodiscard]] std::array<const std::vector<Scalar> *, N>
  candidate_ptrs_impl(std::index_sequence<I...>) const {
    return {&m_candidate[I]...};
  }
};

} // namespace pfc::sim::steppers
