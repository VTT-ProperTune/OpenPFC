// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_flux.hpp
 * @brief Conservative flux nonlinearity \f$\nabla\cdot[M(u)\nabla p]\f$.
 *
 * @details
 * `SpectralETDSystem` splits the right-hand side into a pointwise remainder
 * \f$N(\psi)\f$ times a reciprocal-space multiplier \f$M_{\mathrm{nl}}(k)\f$.
 * That covers every application whose nonlinearity is *local*, but not one
 * whose mobility depends on the field inside a divergence:
 *
 * \f[
 *   \partial_t u = \nabla\cdot\bigl[M(u)\,\nabla p\bigr].
 * \f]
 *
 * A state-dependent \f$M\f$ cannot be moved outside the divergence, so it
 * cannot be written as a symbol, and freezing it at \f$M(u_0)\f$ is exactly
 * the linearisation the science cases are trying to escape. Lubrication films
 * (`thin_film`, `ehd_film`) need the honest form because \f$M\propto h^3\f$ is
 * what sets rupture and droplet selection.
 *
 * ## What this does
 *
 * Per active axis \f$d\f$, given \f$\hat p\f$ and the real field \f$u\f$:
 *
 * \f[
 *   \widehat{\partial_d p} = i k_d\,\hat p
 *   \;\to\; \partial_d p
 *   \;\to\; M(u)\,\partial_d p
 *   \;\to\; \widehat{M\partial_d p}
 *   \;\to\; \text{accumulate } i k_d\,\widehat{M\partial_d p}.
 * \f]
 *
 * Two transforms per axis, so four in 2-D on top of the transforms the step
 * already performs. That is the price of an honest flux and it is why the
 * constant-mobility verifier is kept: it costs two transforms and has an exact
 * answer to check against.
 *
 * ## Time stepping
 *
 * `FluxETD` integrates
 *
 * \f[
 *   \partial_t\hat u = L(k)\,\hat u + \hat N,
 *   \qquad
 *   \hat N = \widehat{\nabla\cdot[M(u)\nabla p]} - L(k)\,\hat u,
 * \f]
 *
 * with the same ETD1 update as the pointwise path. \f$L\f$ is the operator
 * linearised about the reference state, so \f$\hat N\f$ vanishes identically
 * when the mobility is constant and the field is a small perturbation — the
 * nonlinear solver then reproduces the linear verifier, which is asserted in
 * the application tests rather than assumed.
 *
 * ## Scope
 *
 * Host (CPU) only. The device `Ops` layer exposes real-coefficient multiplies,
 * and the gradient needs a complex \f$ik_d\f$, so a GPU flux path needs kernels
 * that do not exist yet. Applications keep their existing constant-mobility HIP
 * twins; the nonlinear science presets are CPU.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>

namespace pfc::apps {

/**
 * @brief Spectral evaluation of \f$\nabla\cdot[M(u)\nabla p]\f$ on the host.
 */
class SpectralFlux {
  using Ops = pfc::sim::SpectralETDOps<pfc::HostSpace>;

public:
  using Complex = Ops::Complex;
  using RealField = Ops::RealField;
  using ComplexField = Ops::ComplexField;
  using FFT = Ops::FFT;

  SpectralFlux(const pfc::Domain &domain, FFT &fft)
      : m_domain(domain), m_fft(fft),
        m_grad_hat(domain, fft.get_outbox_bounds(), 0),
        m_grad(domain, fft.get_inbox_bounds(), 0),
        m_flux_hat(domain, fft.get_outbox_bounds(), 0) {
    const auto size = pfc::domain::get_size(domain);
    for (int d = 0; d < 3; ++d) m_active[d] = size[d] > 1;
    const std::size_t n = fft.size_outbox();
    for (int d = 0; d < 3; ++d) m_k[d].assign(n, 0.0);
    // Orszag 2/3 mask. M(u) grad p is a strongly nonlinear product, so the
    // transform of the flux carries wave numbers the grid cannot represent.
    // Without this the aliased content folds back and the run diverges once
    // the profile steepens -- and it does so independently of the timestep,
    // which is how the omission shows itself.
    m_mask.assign(n, 1.0);
    const auto dx = pfc::domain::get_spacing(domain);
    double cut[3]{};
    for (int d = 0; d < 3; ++d)
      cut[d] = (size[d] > 1) ? (2.0 / 3.0) * (std::numbers::pi / dx[d]) : 0.0;
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          m_k[0][i] = kx;
          m_k[1][i] = ky;
          m_k[2][i] = kz;
          const double a[3]{std::abs(kx), std::abs(ky), std::abs(kz)};
          for (int d = 0; d < 3; ++d)
            if (m_active[d] && a[d] > cut[d]) m_mask[i] = 0.0;
        });
  }

  /**
   * @brief Accumulate \f$\nabla\cdot[M(u)\nabla p]\f$ into @p out_hat.
   *
   * @param p_hat    transform of the potential
   * @param u        real field the mobility depends on
   * @param mobility callable `double(double u)` returning \f$M(u)\f$
   * @param out_hat  overwritten with the divergence
   */
  template <class Mobility>
  void divergence(ComplexField &p_hat, RealField &u, Mobility &&mobility,
                  ComplexField &out_hat) {
    const std::size_t n_out = m_fft.size_outbox();
    out_hat.with_host_view([&](Complex *out, std::size_t) {
      for (std::size_t i = 0; i < n_out; ++i) out[i] = Complex{0.0, 0.0};
    });

    for (int d = 0; d < 3; ++d) {
      if (!m_active[d]) continue;

      // grad_hat = i k_d p_hat
      p_hat.with_host_view([&](Complex *p, std::size_t) {
        m_grad_hat.with_host_view([&](Complex *g, std::size_t) {
          for (std::size_t i = 0; i < n_out; ++i)
            g[i] = Complex{0.0, m_k[d][i]} * p[i];
        });
      });
      Ops::backward(m_fft, m_grad_hat, m_grad);

      // grad *= M(u), in real space, where the nonlinearity actually lives
      u.with_host_view([&](const double *uu, std::size_t count) {
        m_grad.with_host_view([&](double *g, std::size_t) {
          for (std::size_t i = 0; i < count; ++i) g[i] *= mobility(uu[i]);
        });
      });
      Ops::forward(m_fft, m_grad, m_flux_hat);

      // out_hat += i k_d * dealias(flux_hat)
      m_flux_hat.with_host_view([&](Complex *f, std::size_t) {
        out_hat.with_host_view([&](Complex *out, std::size_t) {
          for (std::size_t i = 0; i < n_out; ++i)
            out[i] += Complex{0.0, m_k[d][i]} * (m_mask[i] * f[i]);
        });
      });
    }
  }

  [[nodiscard]] const pfc::Domain &domain() const noexcept { return m_domain; }

private:
  pfc::Domain m_domain;
  FFT &m_fft;
  bool m_active[3]{};
  std::vector<double> m_k[3];
  std::vector<double> m_mask;
  ComplexField m_grad_hat;
  RealField m_grad;
  ComplexField m_flux_hat;
};

/**
 * @brief ETD1 stepper for a conserved equation with a flux nonlinearity.
 *
 * The caller supplies two callbacks, keeping all physics in the application:
 *
 * * `potential(u_hat, u, p_hat)` — fill @p p_hat with \f$\hat p\f$;
 * * `mobility(u)` — the state-dependent \f$M(u)\f$.
 *
 * plus the linear symbol \f$L(k)\f$ used for the exponential integrator.
 */
class FluxETD {
  using Ops = pfc::sim::SpectralETDOps<pfc::HostSpace>;

public:
  using Complex = Ops::Complex;
  using RealField = Ops::RealField;
  using ComplexField = Ops::ComplexField;
  using FFT = Ops::FFT;

  /**
   * @param domain grid geometry
   * @param fft    transform owned by the caller
   * @param dt     fixed step
   * @param L      linear symbol as a function of \f$k_{\mathrm{lap}}\f$
   */
  FluxETD(const pfc::Domain &domain, FFT &fft, double dt,
          const std::function<double(double)> &L)
      : m_fft(fft), m_dt(dt), m_flux(domain, fft),
        m_u_hat(domain, fft.get_outbox_bounds(), 0),
        m_p_hat(domain, fft.get_outbox_bounds(), 0),
        m_div_hat(domain, fft.get_outbox_bounds(), 0) {
    const std::size_t n = fft.size_outbox();
    m_expL.assign(n, 1.0);
    m_phi1.assign(n, dt);
    m_L.assign(n, 0.0);
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          const double k_lap = -(kx * kx + ky * ky + kz * kz);
          const double l = L(k_lap);
          m_L[i] = l;
          const double a = l * dt;
          m_expL[i] = std::exp(a);
          // (e^a - 1)/l, by series where the quotient loses precision.
          m_phi1[i] = (std::abs(a) < 1.0e-8) ? dt * (1.0 + 0.5 * a)
                                             : (m_expL[i] - 1.0) / l;
        });
  }

  /**
   * @brief Advance @p u by one step. Returns `t + dt`.
   */
  template <class Potential, class Mobility>
  double step(double t, RealField &u, Potential &&potential, Mobility &&mobility) {
    Ops::forward(m_fft, u, m_u_hat);
    potential(m_u_hat, u, m_p_hat);
    m_flux.divergence(m_p_hat, u, mobility, m_div_hat);

    const std::size_t n = m_fft.size_outbox();
    m_u_hat.with_host_view([&](Complex *uh, std::size_t) {
      m_div_hat.with_host_view([&](Complex *div, std::size_t) {
        for (std::size_t i = 0; i < n; ++i) {
          // N = full flux divergence minus the part already carried by L.
          const Complex nl = div[i] - m_L[i] * uh[i];
          uh[i] = m_expL[i] * uh[i] + m_phi1[i] * nl;
        }
      });
    });
    Ops::backward(m_fft, m_u_hat, u);
    return t + m_dt;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }

private:
  FFT &m_fft;
  double m_dt;
  SpectralFlux m_flux;
  ComplexField m_u_hat, m_p_hat, m_div_hat;
  std::vector<double> m_expL, m_phi1, m_L;
};

} // namespace pfc::apps
