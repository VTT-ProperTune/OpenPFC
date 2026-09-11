// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file phase_field_inverse.hpp
 * @brief Allen–Cahn-type gradient flow for inverse homogenization (#161 Stages 2–3).
 *
 * @details
 * Design variable \(h\in[0,1]\). After each six-load homogenization the
 * discrete tensor-mismatch gradient is converted to a variational density
 * (multiply by \(N\)) so a uniform mode has an \(O(1)\) rate, then an
 * Allen–Cahn step
 *
 * \f[
 *   \partial_t h = -M\Bigl(
 *     \frac{\delta J_{\mathrm{el}}}{\delta h}
 *     + \lambda_v\,2(\langle h\rangle-\bar h)
 *     + \lambda_r\bigl[-\varepsilon\Delta h + \varepsilon^{-1} W'(h)\bigr]
 *   \Bigr)
 * \f]
 *
 * is taken with \(W(h)=h^2(1-h)^2\). By default \(g\) is RMS-normalised
 * so \(\Delta t\) is the RMS change in \(h\), and \(|\Delta h|\) is capped
 * per cell; job 21949415 collapsed the volume because the raw gradient
 * RMS was \(\sim 4\). This is Takezawa-style PF-TO, not
 * Cahn–Hilliard (volume is a penalty, not a conserved mass) and not an
 * external MMA. Cahn–Hilliard is reserved for the process-constrained
 * family in Stage 6.
 *
 * The Laplacian is spectral on the same HeFFTe plan as the elasticity.
 * There is no second FFT stack and no general-purpose optimizer.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc_apps/homogenization.hpp>

namespace pfc::apps::inverse {

using RealField = pfc::data::Field<double>;
using ComplexField = pfc::data::Field<std::complex<double>>;

struct InverseSpec {
  Voigt6 C_target{};
  Voigt6 W = Voigt6::ones();
  double volume_target{0.5};
  double lambda_volume{1.0};
  double lambda_reg{0.05};
  double epsilon{2.0};
  double mobility{1.0};
  double dt{0.1};
  bool clip{true};
  /// RMS-normalise the *elastic* gradient only so `dt` sets that step
  /// size. Volume and the double well are then added in physical units;
  /// otherwise λ_r W' is crushed whenever ||g_el|| is large.
  bool normalize_grad{true};
  /// Hard cap on |Δh| per cell after the normalised step.
  double max_abs_delta{0.05};
  /// After the step, shift h by a constant so ⟨h⟩ matches volume_target
  /// (clip, then repeat a few times). Stronger than the quadratic penalty.
  bool project_volume{false};
  /// SIMP exponent: elasticity sees \(h^p\), sensitivity is chained by
  /// \(p h^{p-1}\). \(p=1\) is the linear interpolation; \(p=3\) is the
  /// usual solid/void penalization. Grey two-phase linear interpolation
  /// cannot realize auxetic \(C_{12}<0\).
  double simp_p{1.0};
};

struct InverseStepReport {
  double J{0.0};
  double J_tensor{0.0};
  double J_volume{0.0};
  double J_reg{0.0};
  double volume_fraction{0.0};
  double grad_rms{0.0};
  double step_rms{0.0};
  /// Fraction of cells with 0.1 < h < 0.9.
  double grey_fraction{0.0};
  /// sqrt(mean(-h Δh)), a specific-surface / length-scale proxy.
  double perimeter{0.0};
  double C11{0.0};
  double C12{0.0};
  bool elasticity_converged{false};
};

/// \(W(h)=h^2(1-h)^2\).
[[nodiscard]] inline double double_well(double h) noexcept {
  const double u = h * (1.0 - h);
  return u * u;
}
/// \(W'(h)=2h(1-h)(1-2h)\).
[[nodiscard]] inline double double_well_prime(double h) noexcept {
  return 2.0 * h * (1.0 - h) * (1.0 - 2.0 * h);
}

/**
 * @brief Spectral Laplacian \(\Delta h\) on the homogenizer's FFT plan.
 *
 * HeFFTe scales on the backward transform, so a forward / \((-k^2)\) /
 * backward round trip returns \(\Delta h\) in real space.
 */
inline void spectral_laplacian(const pfc::Domain &domain, pfc::fft::IHostFFT &fft,
                               const RealField &h, ComplexField &hat,
                               RealField &lap) {
  std::vector<double> tmp(h.vec());
  fft.forward(tmp, hat.vec());
  pfc::fft::kspace::for_each_kpoint(
      fft.get_outbox_bounds(), domain,
      [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
        const double L = pfc::fft::kspace::k_laplacian_value(kx, ky, kz);
        hat.data()[idx] *= L;
      });
  hat.note_host_write();
  fft.backward(hat.vec(), lap.vec());
  lap.note_host_write();
}

class PhaseFieldInverse {
public:
  PhaseFieldInverse(const pfc::Domain &domain, pfc::fft::IHostFFT &fft,
                    MicroelasticityParams params)
      : m_domain(domain), m_fft(fft), m_hom(domain, fft, std::move(params)),
        m_hat(domain, fft.get_outbox_bounds(), 0),
        m_lap(pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds())),
        m_dJdh(pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds())),
        m_g(pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds())),
        m_penalized(
            pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds())),
        m_n_local(m_dJdh.size()) {
    const auto gs = m_dJdh.global_size();
    m_n_global = static_cast<double>(gs[0]) * static_cast<double>(gs[1]) *
                 static_cast<double>(gs[2]);
  }

  [[nodiscard]] PeriodicHomogenizer &homogenizer() noexcept { return m_hom; }
  [[nodiscard]] const PeriodicHomogenizer &homogenizer() const noexcept {
    return m_hom;
  }

  /**
   * @brief One explicit Allen–Cahn step. Updates @p h in place.
   *
   * Gradient densities use the cell-average inner product on a unit-volume
   * cell: the discrete \(\partial J/\partial h_e\) from
   * `objective_sensitivity` is multiplied by \(N\).
   */
  InverseStepReport step(RealField &h, const InverseSpec &spec) {
    if (h.size() != m_n_local) {
      throw std::invalid_argument("PhaseFieldInverse::step: h has the wrong size");
    }
    if (spec.epsilon <= 0.0 || spec.dt <= 0.0 || spec.mobility < 0.0) {
      throw std::invalid_argument(
          "PhaseFieldInverse::step: epsilon, dt must be > 0 and mobility >= 0");
    }
    if (spec.simp_p < 1.0) {
      throw std::invalid_argument("PhaseFieldInverse::step: simp_p must be >= 1");
    }

    InverseStepReport out;
    const bool tensor_on = spec.W.max_abs() > 0.0;
    const double vf0 = mean_value(h);
    const double dv = vf0 - spec.volume_target;
    out.J_volume = spec.lambda_volume * dv * dv;

    if (tensor_on) {
      const RealField *h_el = &h;
      if (spec.simp_p != 1.0) {
        double *pp = m_penalized.data();
        const double *hd = h.data();
        for (std::size_t i = 0; i < m_n_local; ++i)
          pp[i] = std::pow(hd[i], spec.simp_p);
        m_penalized.note_host_write();
        h_el = &m_penalized;
      }
      const auto r = m_hom.compute(*h_el);
      out.elasticity_converged = r.all_converged();
      out.J_tensor = tensor_mismatch(r.stiffness, spec.C_target, spec.W);
      out.C11 = r.stiffness(0, 0);
      out.C12 = r.stiffness(0, 1);
      m_hom.objective_sensitivity(*h_el, spec.C_target, spec.W, m_dJdh);
      if (spec.simp_p != 1.0) {
        const double pexp = spec.simp_p;
        const double pm1 = pexp - 1.0;
        double *dj = m_dJdh.data();
        const double *hd = h.data();
        for (std::size_t i = 0; i < m_n_local; ++i)
          dj[i] *= pexp * std::pow(hd[i], pm1);
        m_dJdh.note_host_write();
      }
    } else {
      // Double-well / volume-only step (binarization oracle). Skip the six
      // elasticity solves; W=0 so they cannot change J.
      out.elasticity_converged = true;
      out.J_tensor = 0.0;
      std::fill(m_dJdh.vec().begin(), m_dJdh.vec().end(), 0.0);
      m_dJdh.note_host_write();
    }
    spectral_laplacian(m_domain, m_fft, h, m_hat, m_lap);

    double local_reg = 0.0;
    double local_el2 = 0.0;
    double local_grey = 0.0;
    double local_hLap = 0.0;
    const double *hp = h.data();
    const double *djel = m_dJdh.data();
    const double *lp = m_lap.data();
    double *gp = m_g.data();
    const double inv_eps = 1.0 / spec.epsilon;
    for (std::size_t i = 0; i < m_n_local; ++i) {
      const double well = double_well(hp[i]);
      // ∫ |grad h|^2 = -∫ h Δh  (periodic, spectral Δ).
      local_reg += 0.5 * spec.epsilon * (-hp[i] * lp[i]) + inv_eps * well;
      local_hLap += -hp[i] * lp[i];
      if (hp[i] > 0.1 && hp[i] < 0.9) local_grey += 1.0;
      const double g_el = m_n_global * djel[i];
      gp[i] = g_el;
      local_el2 += g_el * g_el;
    }

    double glo[4] = {0, 0, 0, 0};
    const double loc[4] = {local_reg, local_el2, local_grey, local_hLap};
    MPI_Allreduce(loc, glo, 4, MPI_DOUBLE, MPI_SUM, comm());
    out.J_reg = spec.lambda_reg * (glo[0] / m_n_global);
    out.J = out.J_tensor + out.J_volume + out.J_reg;
    const double el_rms = std::sqrt(glo[1] / m_n_global);
    out.grad_rms = el_rms;
    out.grey_fraction = glo[2] / m_n_global;
    out.perimeter = std::sqrt(std::max(0.0, glo[3] / m_n_global));

    // Normalise elasticity only. Job 21950094 stayed fully grey because
    // RMS-normalising the *total* g crushed λ_r W'(h) to a few percent of
    // each step. Volume and the double well keep physical units.
    const double el_scale =
        (spec.normalize_grad && el_rms > 1.0e-30) ? (1.0 / el_rms) : 1.0;
    double local_g2 = 0.0;
    for (std::size_t i = 0; i < m_n_local; ++i) {
      const double g_el = el_scale * gp[i];
      const double g_vol = spec.lambda_volume * 2.0 * dv;
      const double g_reg =
          spec.lambda_reg * (-spec.epsilon * lp[i] + inv_eps * double_well_prime(hp[i]));
      const double g = g_el + g_vol + g_reg;
      gp[i] = g;
      local_g2 += g * g;
    }
    m_g.note_host_write();
    double glo_g2 = 0.0;
    MPI_Allreduce(&local_g2, &glo_g2, 1, MPI_DOUBLE, MPI_SUM, comm());
    out.grad_rms = std::sqrt(glo_g2 / m_n_global);

    double scale = spec.dt * spec.mobility;
    const double cap = spec.max_abs_delta;
    double local_dh2 = 0.0;
    for (std::size_t i = 0; i < m_n_local; ++i) {
      double dh = -scale * gp[i];
      if (cap > 0.0) dh = std::min(cap, std::max(-cap, dh));
      local_dh2 += dh * dh;
      double hn = hp[i] + dh;
      if (spec.clip) hn = std::min(1.0, std::max(0.0, hn));
      h.data()[i] = hn;
    }
    h.note_host_write();
    double glo_dh2 = 0.0;
    MPI_Allreduce(&local_dh2, &glo_dh2, 1, MPI_DOUBLE, MPI_SUM, comm());
    out.step_rms = std::sqrt(glo_dh2 / m_n_global);
    if (spec.project_volume) project_mean(h, spec.volume_target, spec.clip);
    out.volume_fraction = mean_value(h);
    return out;
  }

  [[nodiscard]] double mean_value(const RealField &h) const {
    double local = 0.0;
    const double *p = h.data();
    for (std::size_t i = 0; i < m_n_local; ++i) local += p[i];
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm());
    return global / m_n_global;
  }

  void project_mean(RealField &h, double target, bool clip_flag) const {
    for (int it = 0; it < 6; ++it) {
      const double vf = mean_value(h);
      const double shift = target - vf;
      if (std::abs(shift) < 1.0e-12) break;
      double *p = h.data();
      for (std::size_t i = 0; i < m_n_local; ++i) {
        double hn = p[i] + shift;
        if (clip_flag) hn = std::min(1.0, std::max(0.0, hn));
        p[i] = hn;
      }
      h.note_host_write();
    }
  }

private:
  [[nodiscard]] MPI_Comm comm() const noexcept {
    return m_hom.solver().params().comm;
  }

  pfc::Domain m_domain;
  pfc::fft::IHostFFT &m_fft;
  PeriodicHomogenizer m_hom;
  ComplexField m_hat;
  RealField m_lap, m_dJdh, m_g, m_penalized;
  std::size_t m_n_local{0};
  double m_n_global{1.0};
};

} // namespace pfc::apps::inverse
