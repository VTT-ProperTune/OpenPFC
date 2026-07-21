// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_exp_coefficients.hpp
 * @brief CPU diagonal spectral exponential-action coefficient builder
 *
 * @details
 * Builds per-mode coefficients for integrating-factor / ETD1-style updates from
 * already-formed diagonal spectral samples @c L[i] and a timestep @c dt:
 *
 * - @c exp_Ldt = exp(L*dt)
 * - @c phi1_L  = (exp(L*dt)-1)/L   (stable near L → 0 via expm1/L or Taylor)
 *
 * Ownership:
 * - **Caller-owned:** @ref fill_spectral_exp_coeffs writes into caller spans;
 *   the caller owns lifetime of those buffers.
 * - **Method-owned:** @ref SpectralExpCoefficientCache owns @c std::vector
 *   storage. @ref SpectralExpCoefficientCache::exp_Ldt and
 *   @ref SpectralExpCoefficientCache::phi1_L return non-owning views that
 *   remain valid until the next @c ensure that resizes, or until the cache is
 *   destroyed / moved-from.
 *
 * Coefficients are transient and recomputable; they are **not** part of
 * checkpointed simulation state. Rebuild when operator, @c dt, or
 * configuration identity tokens change (or when the number of modes changes).
 *
 * This API takes already-formed @c L. It is **not** Tungsten's physics-specific
 * @c opN = expm1(arg)/opCk construction in @c tungsten_spectral.hpp.
 *
 * @see docs/development/time_integration_architecture.md
 */

#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace pfc::integrator {

/**
 * @brief Per-mode spectral exponential-action coefficients.
 *
 * @c exp_Ldt is @c exp(L*dt). @c phi1_L is @c (exp(L*dt)-1)/L, evaluated in a
 * numerically stable form near @c L → 0.
 */
struct SpectralExpCoeffs {
  double exp_Ldt{}; ///< exp(L*dt)
  double phi1_L{};  ///< (exp(L*dt)-1)/L (stable near L→0)
};

/**
 * @brief Evaluate diagonal spectral exponential-action coefficients for one mode.
 *
 * @param L Diagonal spectral sample (already-formed linear symbol).
 * @param dt Timestep. @c dt == 0 yields @c exp_Ldt = 1 and @c phi1_L = 0.
 * @param abs_L_threshold Absolute |L| below which Taylor is used for @c phi1_L
 *        (default @c 1e-12, matching Tungsten's near-zero spirit).
 * @return Finite @ref SpectralExpCoeffs.
 *
 * When @c |L| >= abs_L_threshold: @c phi1_L = expm1(L*dt)/L.
 * When @c |L| < abs_L_threshold: Taylor @c phi1_L = dt + 0.5*L*dt*dt.
 */
[[nodiscard]] inline SpectralExpCoeffs
spectral_exp_coeffs(double L, double dt,
                    double abs_L_threshold = 1e-12) {
  SpectralExpCoeffs out{};
  const double arg = L * dt;
  out.exp_Ldt = std::exp(arg);

  if (std::abs(L) < abs_L_threshold) {
    // Second-order Taylor of (exp(L*dt)-1)/L as L → 0: dt + (1/2) L dt^2 + …
    out.phi1_L = dt + 0.5 * L * dt * dt;
  } else {
    out.phi1_L = std::expm1(arg) / L;
  }
  return out;
}

/**
 * @brief Fill caller-owned coefficient arrays from diagonal @c L samples.
 *
 * @param L Input spectral samples.
 * @param dt Timestep.
 * @param exp_Ldt Output span for @c exp(L*dt); must match @c L.size().
 * @param phi1_L Output span for stable @c (exp(L*dt)-1)/L; must match @c L.size().
 * @param abs_L_threshold Near-zero |L| threshold (default @c 1e-12).
 *
 * @throws std::invalid_argument if span sizes mismatch.
 *
 * Caller owns @p exp_Ldt and @p phi1_L lifetime.
 */
inline void fill_spectral_exp_coeffs(std::span<const double> L, double dt,
                                     std::span<double> exp_Ldt,
                                     std::span<double> phi1_L,
                                     double abs_L_threshold = 1e-12) {
  if (exp_Ldt.size() != L.size() || phi1_L.size() != L.size()) {
    throw std::invalid_argument(
        "fill_spectral_exp_coeffs: span sizes must match L.size()");
  }
  for (std::size_t i = 0; i < L.size(); ++i) {
    const SpectralExpCoeffs c =
        spectral_exp_coeffs(L[i], dt, abs_L_threshold);
    exp_Ldt[i] = c.exp_Ldt;
    phi1_L[i] = c.phi1_L;
  }
}

/** @brief Opaque operator-identity token for coefficient cache invalidation. */
struct SpectralExpOperatorId {
  std::uint64_t value{};
  friend constexpr bool operator==(SpectralExpOperatorId,
                                   SpectralExpOperatorId) = default;
};

/**
 * @brief Opaque timestep-identity token for coefficient cache invalidation.
 *
 * Prefer a generation counter from the caller. @ref from_bits packs the IEEE
 * bit pattern of @c dt when a counter is unavailable.
 */
struct SpectralExpDtId {
  std::uint64_t value{};
  friend constexpr bool operator==(SpectralExpDtId, SpectralExpDtId) = default;

  [[nodiscard]] static SpectralExpDtId from_bits(double dt) noexcept {
    return SpectralExpDtId{.value = std::bit_cast<std::uint64_t>(dt)};
  }
};

/** @brief Opaque configuration-identity token (scheme / threshold / flags). */
struct SpectralExpConfigId {
  std::uint64_t value{};
  friend constexpr bool operator==(SpectralExpConfigId,
                                   SpectralExpConfigId) = default;
};

/**
 * @brief Method-owned storage for diagonal spectral exponential-action coefficients.
 *
 * Owns @c std::vector buffers for @c exp(L*dt) and @c phi1_L. Views from
 * @ref exp_Ldt / @ref phi1_L remain valid until the next @ref ensure that
 * resizes storage, or until the cache is destroyed / moved-from.
 *
 * @ref ensure skips recompute only when the cache is valid and operator,
 * @c dt, and configuration identities match and @c L.size() equals the stored
 * length. Changing any identity or length forces a rebuild.
 * @ref rebuilt_last_call is a diagnostic only (cache hit/miss).
 */
class SpectralExpCoefficientCache {
public:
  /**
   * @brief Ensure coefficients match @p L, @p dt, and the given identities.
   *
   * Rebuilds iff not valid, any identity differs, or @p L.size() differs from
   * stored capacity. After success, @ref exp_Ldt and @ref phi1_L have size
   * @p L.size().
   */
  void ensure(std::span<const double> L, double dt,
              SpectralExpOperatorId op_id, SpectralExpDtId dt_id,
              SpectralExpConfigId config_id,
              double abs_L_threshold = 1e-12) {
    const bool same_ids = m_valid && op_id == m_op_id && dt_id == m_dt_id &&
                          config_id == m_config_id &&
                          L.size() == m_exp_Ldt.size();
    if (same_ids) {
      m_rebuilt_last = false;
      return;
    }

    m_exp_Ldt.resize(L.size());
    m_phi1_L.resize(L.size());
    fill_spectral_exp_coeffs(L, dt, m_exp_Ldt, m_phi1_L, abs_L_threshold);

    m_op_id = op_id;
    m_dt_id = dt_id;
    m_config_id = config_id;
    m_valid = true;
    m_rebuilt_last = true;
  }

  [[nodiscard]] std::span<const double> exp_Ldt() const noexcept {
    return m_exp_Ldt;
  }

  [[nodiscard]] std::span<const double> phi1_L() const noexcept {
    return m_phi1_L;
  }

  [[nodiscard]] bool valid() const noexcept { return m_valid; }

  /// True iff the last @ref ensure recomputed coefficients (diagnostic only).
  [[nodiscard]] bool rebuilt_last_call() const noexcept {
    return m_rebuilt_last;
  }

private:
  std::vector<double> m_exp_Ldt;
  std::vector<double> m_phi1_L;
  SpectralExpOperatorId m_op_id{};
  SpectralExpDtId m_dt_id{};
  SpectralExpConfigId m_config_id{};
  bool m_valid{false};
  bool m_rebuilt_last{false};
};

} // namespace pfc::integrator
