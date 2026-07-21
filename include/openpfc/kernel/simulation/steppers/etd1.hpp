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
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Computational outcome of one ETD1 step attempt.
 *
 * @c success means the candidate was formed with finite values — not an
 * adaptive accept/reject decision (ETD1 has no embedded estimator).
 */
struct Etd1StepAttempt {
  bool success{false};
  double t_next{0.0};
  std::string reason{}; ///< empty on success; optional diagnostic on failure
};

[[nodiscard]] inline Etd1StepAttempt make_etd1_success(double t_next) {
  return Etd1StepAttempt{.success = true, .t_next = t_next, .reason = {}};
}

[[nodiscard]] inline Etd1StepAttempt
make_etd1_failure(std::string reason = {}) {
  return Etd1StepAttempt{.success = false,
                         .t_next = 0.0,
                         .reason = std::move(reason)};
}

/**
 * @brief CPU ETD1 stepper with isolated candidate state.
 *
 * @tparam Rhs Callable satisfying @ref StageFunction (`rhs(t, u, du)` fills
 *             @c du with the nonlinear term @c N).
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
template <class Rhs>
  requires StageFunction<Rhs>
class Etd1Stepper {
public:
  Etd1Stepper(double dt, std::size_t local_size, Rhs rhs)
      : m_dt(dt), m_local_size(local_size), m_du(local_size, 0.0),
        m_candidate(local_size, 0.0), m_u_scratch(local_size, 0.0),
        m_rhs(std::move(rhs)) {}

  /**
   * @brief Bind caller-lent coefficient spans.
   *
   * Requires @p exp_Ldt and @p phi1_L to have equal length. Matching against
   * @c local_size / @c u_accepted is deferred to @ref attempt_step so a
   * size-mismatch failure path remains reachable.
   *
   * Spans must outlive the next @ref attempt_step (or be replaced via a later
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
   * spans would otherwise go out of scope before @ref attempt_step.
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
      const pfc::integrator::SpectralExpCoefficientCache &cache) {
    set_coefficients_owned(cache.exp_Ldt(), cache.phi1_L());
  }

  /**
   * @brief Form an isolated ETD1 candidate without mutating @p u_accepted.
   *
   * Algorithm: size-check → copy accepted into scratch → evaluate @c N on
   * scratch → @c candidate = exp_Ldt * u_accepted + phi1_L * N.
   */
  [[nodiscard]] Etd1StepAttempt
  attempt_step(double t, const std::vector<double> &u_accepted) {
    if (u_accepted.size() != m_local_size) {
      return make_etd1_failure("u_accepted.size() != local_size");
    }
    if (m_exp_Ldt.size() != m_local_size || m_phi1_L.size() != m_local_size) {
      return make_etd1_failure("coefficient span size != local_size");
    }

    m_u_scratch = u_accepted;
    m_rhs(t, m_u_scratch, m_du);

    for (std::size_t i = 0; i < m_local_size; ++i) {
      const double c =
          m_exp_Ldt[i] * u_accepted[i] + m_phi1_L[i] * m_du[i];
      if (!std::isfinite(c)) {
        return make_etd1_failure("non-finite candidate value");
      }
      m_candidate[i] = c;
    }
    return make_etd1_success(t + m_dt);
  }

  [[nodiscard]] std::span<const double> candidate() const noexcept {
    return m_candidate;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }

  [[nodiscard]] std::size_t local_size() const noexcept { return m_local_size; }

private:
  double m_dt{0.0};
  std::size_t m_local_size{0};
  std::vector<double> m_du;
  std::vector<double> m_candidate;
  std::vector<double> m_u_scratch;
  std::vector<double> m_owned_exp;
  std::vector<double> m_owned_phi1;
  std::span<const double> m_exp_Ldt{};
  std::span<const double> m_phi1_L{};
  Rhs m_rhs;
};

/**
 * @brief Two-field ETD1 stepper with per-field isolated candidates.
 *
 * @tparam Rhs Callable satisfying @ref MultiStageFunction.
 * @tparam N   Field count (fixed at 2 for this slice).
 *
 * Each accepted field is copied into method-owned scratch before the multi-field
 * nonlinear evaluation so a misbehaving @c N cannot mutate caller buffers.
 */
template <class Rhs, std::size_t N>
  requires(N == 2) && MultiStageFunction<Rhs>
class MultiEtd1Stepper {
public:
  using RhsType = Rhs;
  static constexpr std::size_t field_count = N;

  MultiEtd1Stepper(double dt, std::array<std::size_t, N> local_sizes, Rhs rhs)
      : m_dt(dt), m_local_sizes(local_sizes), m_rhs(std::move(rhs)) {
    for (std::size_t f = 0; f < N; ++f) {
      m_du[f].assign(local_sizes[f], 0.0);
      m_candidate[f].assign(local_sizes[f], 0.0);
      m_u_scratch[f].assign(local_sizes[f], 0.0);
    }
  }

  /**
   * @brief Bind caller-lent per-field coefficient spans.
   *
   * Per field, @c exp and @c phi1 must have equal length. Matching against
   * each field's @c local_size is deferred to @ref attempt_step.
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
  [[nodiscard]] Etd1StepAttempt
  attempt_step(double t, const std::vector<double> &u0,
               const std::vector<double> &u1) {
    if (u0.size() != m_local_sizes[0] || u1.size() != m_local_sizes[1]) {
      return make_etd1_failure("accepted field size != local_size");
    }
    for (std::size_t f = 0; f < N; ++f) {
      if (m_exp_Ldt[f].size() != m_local_sizes[f] ||
          m_phi1_L[f].size() != m_local_sizes[f]) {
        return make_etd1_failure("coefficient span size != local_size");
      }
    }

    m_u_scratch[0] = u0;
    m_u_scratch[1] = u1;
    auto u_pack = std::tie(m_u_scratch[0], m_u_scratch[1]);
    auto du_pack = std::tie(m_du[0], m_du[1]);
    m_rhs(t, u_pack, du_pack);

    const std::array<const std::vector<double> *, N> accepted{&u0, &u1};
    for (std::size_t f = 0; f < N; ++f) {
      const auto &u_acc = *accepted[f];
      for (std::size_t i = 0; i < m_local_sizes[f]; ++i) {
        const double c =
            m_exp_Ldt[f][i] * u_acc[i] + m_phi1_L[f][i] * m_du[f][i];
        if (!std::isfinite(c)) {
          return make_etd1_failure("non-finite candidate value");
        }
        m_candidate[f][i] = c;
      }
    }
    return make_etd1_success(t + m_dt);
  }

  [[nodiscard]] std::span<const double>
  candidate(std::size_t field_index) const noexcept {
    return m_candidate[field_index];
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }

private:
  double m_dt{0.0};
  std::array<std::size_t, N> m_local_sizes{};
  std::array<std::vector<double>, N> m_du{};
  std::array<std::vector<double>, N> m_candidate{};
  std::array<std::vector<double>, N> m_u_scratch{};
  std::array<std::vector<double>, N> m_owned_exp{};
  std::array<std::vector<double>, N> m_owned_phi1{};
  std::array<std::span<const double>, N> m_exp_Ldt{};
  std::array<std::span<const double>, N> m_phi1_L{};
  Rhs m_rhs;
};

} // namespace pfc::sim::steppers
