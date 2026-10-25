// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file etd_step_doubling.hpp
 * @brief ETD1 local-error evidence from one step and two half-steps.
 *
 * @details
 * From one accepted state and one macro step `dt` this forms
 *
 * - `y_full`: one ETD1 step of size `dt`;
 * - `y_half`: two ETD1 steps of size `dt/2`;
 *
 * and keeps the raw difference `y_half - y_full`. ETD1 is first order, so
 * that difference is `O(dt^2)`. The leading local error of the two-half-step
 * result is `(y_full - y_half) / (2^p - 1)` with `p = 1`, which is the raw
 * difference itself. The solution is not Richardson-extrapolated. On commit
 * the accepted field is `y_half`.
 *
 * The controller exponent is the order of this difference, `error_order = 2`,
 * the same convention as an embedded pair. Scale the difference with the
 * #208 rule `max(|y_start|, |y_half|)`, not against the constant 1.
 *
 * `set_dt(dt)` rebuilds coefficients before the full step, and the real
 * full-step field is copied before `set_dt(dt/2)`. Half-step attempts
 * therefore cannot see full-step `exp(L dt)` or `phi1`. Commit and reject
 * both finish by rebuilding the macro-step coefficients, so a later `step`
 * does not inherit `dt/2`.
 *
 * A rejection restores `psi` and `psi_hat` and does not touch `Time`. Output
 * that keys off accepted time stays put. Every rank must be given the same
 * `dt`; this type does not invent one.
 *
 * CPU only. One plain attempt (no mean-field filter and no correlation
 * kernel) costs 3 nonlinear evaluations and 9 transforms: 6 forwards and 3
 * backwards.
 */

#include <cmath>
#include <complex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>

namespace pfc::sim {

/**
 * @brief One ETD1 step-doubling attempt on a host `SpectralETDSystem`.
 */
template <class Physics> class ETD1StepDoubling {
public:
  using System = SpectralETDSystem<Physics, pfc::HostSpace>;
  using Complex = std::complex<double>;

  explicit ETD1StepDoubling(System &system) : m_sys(system) {}

  /**
   * @brief Build `y_full` and `y_half` from the current accepted state.
   *
   * Leaves `psi` equal to `y_half` until `commit` or `reject`. Throws if an
   * attempt is already open.
   */
  void attempt(double t, double dt) {
    if (m_open) {
      throw std::logic_error(
          "ETD1StepDoubling::attempt: an attempt is already open");
    }
    if (!(dt > 0.0) || !std::isfinite(dt)) {
      throw std::invalid_argument(
          "ETD1StepDoubling::attempt: dt must be finite and > 0");
    }
    m_open = true;
    m_dt = dt;

    auto &psi = m_sys.psi();
    auto &hat = m_sys.psi_hat();
    m_saved_psi = psi.vec();
    m_saved_hat = hat.vec();

    const int forwards0 = m_sys.forward_count();
    const int backwards0 = m_sys.backward_count();
    const int nonlinear0 = m_sys.nonlinear_count();

    // Full step. The real field is frozen before the half-step cache exists.
    m_sys.set_dt(dt);
    m_sys.attempt(t);
    m_sys.realize_candidate(m_full);

    m_sys.set_dt(dt * 0.5);
    m_sys.attempt(t);
    m_sys.commit();
    m_sys.attempt(t + dt * 0.5);
    m_sys.commit();

    m_half = m_sys.psi().vec();
    if (m_half.size() != m_full.size()) {
      throw std::logic_error(
          "ETD1StepDoubling: full and half fields differ in length");
    }
    m_raw.resize(m_half.size());
    for (std::size_t i = 0; i < m_half.size(); ++i) {
      m_raw[i] = m_half[i] - m_full[i];
    }

    m_forwards = m_sys.forward_count() - forwards0;
    m_backwards = m_sys.backward_count() - backwards0;
    m_nonlinear = m_sys.nonlinear_count() - nonlinear0;
  }

  /**
   * @brief Keep `y_half` and rebuild the macro-step coefficients.
   *
   * The field is already `y_half`. Rebuilding `dt` drops the half-step cache
   * so it cannot be reused as if it belonged to the accepted interval.
   */
  void commit() {
    require_open("commit");
    m_sys.set_dt(m_dt);
    m_open = false;
  }

  /**
   * @brief Restore the accepted field and the macro-step coefficients.
   *
   * `psi` and `psi_hat` return to the values saved at `attempt`. Scratch
   * nonlinearity fields are left dirty; the next attempt recomputes them.
   */
  void reject() {
    require_open("reject");
    auto &psi = m_sys.psi();
    psi.vec() = m_saved_psi;
    psi.note_host_write();
    auto &hat = m_sys.psi_hat();
    hat.vec() = m_saved_hat;
    hat.note_host_write();
    m_sys.set_dt(m_dt);
    m_open = false;
  }

  [[nodiscard]] bool attempt_open() const noexcept { return m_open; }
  [[nodiscard]] double dt() const noexcept { return m_dt; }

  /// `y_half - y_full` in real space. Valid until the next `attempt`.
  [[nodiscard]] std::span<const double> raw_difference() const noexcept {
    return m_raw;
  }
  /// One ETD1 step of `dt`, in real space.
  [[nodiscard]] std::span<const double> full_step() const noexcept { return m_full; }
  /// Two ETD1 steps of `dt/2`, in real space.
  [[nodiscard]] std::span<const double> half_steps() const noexcept {
    return m_half;
  }
  /// Accepted field at the start of the open attempt.
  [[nodiscard]] std::span<const double> saved_state() const noexcept {
    return m_saved_psi;
  }

  [[nodiscard]] int forward_count() const noexcept { return m_forwards; }
  [[nodiscard]] int backward_count() const noexcept { return m_backwards; }
  [[nodiscard]] int nonlinear_count() const noexcept { return m_nonlinear; }
  [[nodiscard]] int transform_count() const noexcept {
    return m_forwards + m_backwards;
  }

private:
  void require_open(const char *what) const {
    if (!m_open) {
      throw std::logic_error(std::string("ETD1StepDoubling::") + what +
                             ": no open attempt");
    }
  }

  System &m_sys;
  bool m_open{false};
  double m_dt{0.0};
  std::vector<double> m_saved_psi;
  std::vector<Complex> m_saved_hat;
  std::vector<double> m_full;
  std::vector<double> m_half;
  std::vector<double> m_raw;
  int m_forwards{0};
  int m_backwards{0};
  int m_nonlinear{0};
};

} // namespace pfc::sim
