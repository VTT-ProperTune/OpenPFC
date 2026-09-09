// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file anisotropic_flux.hpp
 * @brief Spectral ETD1 stepper for the anisotropic small-slope surface-
 *        diffusion equation (`#115`).
 *
 * @details
 * The isotropic Mullins model (`surface_diffusion_physics.hpp`) is the
 * divergence form
 * \f[
 *   \partial_t h = \nabla\cdot\bigl[B_0\,\nabla(\nabla^2 h)\bigr]
 *               = -B_0\nabla^4 h,
 * \f]
 * which is a pure reciprocal-space symbol because \f$B_0\f$ is constant. The
 * anisotropic generalisation replaces the constant kinetic coefficient with
 * the orientation-dependent stiffness \f$B(\theta)\f$ of `anisotropy.hpp`,
 * \f$\theta=\operatorname{atan2}(h_y,h_x)\f$ the local surface-gradient
 * orientation:
 * \f[
 *   \partial_t h = \nabla\cdot\bigl[B(\theta)\,\nabla(\nabla^2 h)\bigr].
 * \f]
 * \f$B(\theta)\f$ depends on the field itself (through its gradient), so the
 * operator cannot be written as a single reciprocal-space multiplier the way
 * `SurfaceDiffusionPhysics::linear_symbol` is; it is evaluated the same way
 * `openpfc_apps/spectral_flux.hpp` evaluates a state-dependent mobility
 * inside a divergence, generalised so the coefficient depends on the local
 * gradient *orientation* rather than on the field value. This header does not
 * reuse `pfc::apps::SpectralFlux` because that class's `mobility` callback is
 * a function of the transported field's value, not of a separately derived
 * orientation field; duplicating the small kernel here keeps the shared
 * `apps/common` helper's contract unchanged for the other applications that
 * already depend on it.
 *
 * ## Splitting for the exponential integrator
 *
 * \f[
 *   \partial_t\hat h = L_0(k)\,\hat h + \hat N,\qquad
 *   L_0(k) = -B_0\,k_{\mathrm{lap}}^2,\qquad
 *   \hat N = \widehat{\nabla\cdot[B(\theta)\nabla(\nabla^2 h)]} - L_0(k)\hat h,
 * \f]
 * i.e. the constant-\f$B_0\f$ part is integrated exactly (unconditionally
 * stable for the stiff fourth-order term) and \f$\hat N\f$ carries only the
 * orientation-dependent remainder. When `eps_a = 0`, `B(theta) == B0`
 * everywhere and the full divergence equals \f$L_0(k)\hat h\f$ up to FFT
 * round-off, so \f$\hat N\to 0\f$ and this stepper reproduces the isotropic
 * `SpectralETDSystem<SurfaceDiffusionPhysics>` trajectory to floating-point
 * tolerance -- asserted in `test_surface_diffusion.cpp` rather than assumed.
 *
 * ## Scope
 *
 * 2-D height fields only (`Lz == 1`): \f$\theta\f$ is defined from the
 * in-plane gradient `(h_x, h_y)`, matching the small-slope 2-D patterned-
 * surface preset this model was built for. Host (CPU) only, like the shared
 * flux helper it parallels; a directional mobility needs a complex `i*k_d`
 * multiply that the device `Ops` layer does not expose.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>

#include <surface_diffusion/anisotropy.hpp>

namespace surface_diffusion {

/// ETD1 stepper for `dh/dt = div[B(theta) grad(lap h)]` on a 2-D domain.
class AnisotropicSurfaceDiffusionETD {
  using Ops = pfc::sim::SpectralETDOps<pfc::HostSpace>;

public:
  using Complex = Ops::Complex;
  using RealField = Ops::RealField;
  using ComplexField = Ops::ComplexField;
  using FFT = Ops::FFT;

  /**
   * @param domain     grid geometry (must have `Lz == 1`)
   * @param fft        transform owned by the caller
   * @param dt         fixed step
   * @param stiffness  \f$B(\theta)\f$
   */
  AnisotropicSurfaceDiffusionETD(const pfc::Domain &domain, FFT &fft, double dt,
                                 SurfaceStiffness stiffness)
      : m_dt(dt), m_stiffness(stiffness), m_fft(fft),
        m_h_hat(domain, fft.get_outbox_bounds(), 0),
        m_p_hat(domain, fft.get_outbox_bounds(), 0),
        m_grad_hat(domain, fft.get_outbox_bounds(), 0),
        m_flux_hat(domain, fft.get_outbox_bounds(), 0),
        m_div_hat(domain, fft.get_outbox_bounds(), 0),
        m_grad_real(domain, fft.get_inbox_bounds(), 0),
        m_hx(domain, fft.get_inbox_bounds(), 0),
        m_hy(domain, fft.get_inbox_bounds(), 0),
        m_theta(domain, fft.get_inbox_bounds(), 0),
        m_Btheta(domain, fft.get_inbox_bounds(), 0) {
    const auto size = pfc::domain::get_size(domain);
    if (size[2] != 1) {
      throw std::invalid_argument(
          "AnisotropicSurfaceDiffusionETD: requires a 2-D domain (Lz == 1); "
          "theta = atan2(h_y, h_x) is only defined for an in-plane gradient.");
    }
    const std::size_t n = fft.size_outbox();
    m_kx.assign(n, 0.0);
    m_ky.assign(n, 0.0);
    m_k_lap.assign(n, 0.0);
    m_mask.assign(n, 1.0);
    m_expL.assign(n, 1.0);
    m_phi1.assign(n, dt);
    m_L0.assign(n, 0.0);

    const auto dx = pfc::domain::get_spacing(domain);
    const double cutx = (2.0 / 3.0) * (std::numbers::pi / dx[0]);
    const double cuty = (2.0 / 3.0) * (std::numbers::pi / dx[1]);

    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          m_kx[i] = kx;
          m_ky[i] = ky;
          m_k_lap[i] = -(kx * kx + ky * ky + kz * kz);
          // Orszag 2/3 dealiasing mask: B(theta)*grad(lap h) is a strongly
          // nonlinear product (theta itself is a ratio of gradients), so its
          // transform carries wavenumbers the grid cannot represent.
          if (std::abs(kx) > cutx || std::abs(ky) > cuty) m_mask[i] = 0.0;
          const double l0 = -stiffness.B0 * m_k_lap[i] * m_k_lap[i];
          m_L0[i] = l0;
          const double a = l0 * dt;
          m_expL[i] = std::exp(a);
          m_phi1[i] =
              (std::abs(a) < 1.0e-8) ? dt * (1.0 + 0.5 * a) : (m_expL[i] - 1.0) / l0;
        });
  }

  /**
   * @brief Advance @p h by one step. Returns `t + dt`.
   */
  double step(double t, RealField &h) {
    Ops::forward(m_fft, h, m_h_hat);

    // theta(x) = atan2(h_y, h_x), gradient evaluated spectrally.
    spectral_derivative(m_kx, m_h_hat, m_hx);
    spectral_derivative(m_ky, m_h_hat, m_hy);
    m_hx.with_host_view([&](const double *hx, std::size_t n) {
      m_hy.with_host_view([&](const double *hy, std::size_t) {
        m_theta.with_host_view([&](double *th, std::size_t) {
          m_Btheta.with_host_view([&](double *b, std::size_t) {
            for (std::size_t i = 0; i < n; ++i) {
              th[i] = std::atan2(hy[i], hx[i]);
              b[i] = m_stiffness(th[i]);
            }
          });
        });
      });
    });

    // p_hat = -k_lap * h_hat = FFT(-lap h)
    const std::size_t n_out = m_fft.size_outbox();
    m_h_hat.with_host_view([&](const Complex *hh, std::size_t) {
      m_p_hat.with_host_view([&](Complex *p, std::size_t) {
        for (std::size_t i = 0; i < n_out; ++i) p[i] = -m_k_lap[i] * hh[i];
      });
    });

    m_div_hat.with_host_view([&](Complex *d, std::size_t) {
      for (std::size_t i = 0; i < n_out; ++i) d[i] = Complex{0.0, 0.0};
    });

    const std::vector<double> *k_axes[2]{&m_kx, &m_ky};
    for (const auto *k : k_axes) {
      m_p_hat.with_host_view([&](const Complex *p, std::size_t) {
        m_grad_hat.with_host_view([&](Complex *g, std::size_t) {
          for (std::size_t i = 0; i < n_out; ++i) g[i] = Complex{0.0, (*k)[i]} * p[i];
        });
      });
      Ops::backward(m_fft, m_grad_hat, m_grad_real);

      m_Btheta.with_host_view([&](const double *b, std::size_t count) {
        m_grad_real.with_host_view([&](double *g, std::size_t) {
          for (std::size_t i = 0; i < count; ++i) g[i] *= b[i];
        });
      });
      Ops::forward(m_fft, m_grad_real, m_flux_hat);

      m_flux_hat.with_host_view([&](const Complex *f, std::size_t) {
        m_div_hat.with_host_view([&](Complex *d, std::size_t) {
          for (std::size_t i = 0; i < n_out; ++i)
            d[i] += Complex{0.0, (*k)[i]} * (m_mask[i] * f[i]);
        });
      });
    }

    // ETD1: h_hat_new = expL*h_hat + phi1*(div_hat - L0*h_hat).
    m_h_hat.with_host_view([&](Complex *hh, std::size_t) {
      m_div_hat.with_host_view([&](const Complex *d, std::size_t) {
        for (std::size_t i = 0; i < n_out; ++i) {
          const Complex nl = d[i] - m_L0[i] * hh[i];
          hh[i] = m_expL[i] * hh[i] + m_phi1[i] * nl;
        }
      });
    });
    Ops::backward(m_fft, m_h_hat, h);
    return t + m_dt;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }
  [[nodiscard]] const SurfaceStiffness &stiffness() const noexcept {
    return m_stiffness;
  }

  /// Local surface-gradient orientation from the most recent `step` (or the
  /// state the caller last wrote into `h` before calling `step`).
  [[nodiscard]] const RealField &theta() const noexcept { return m_theta; }

  /// Local |grad h| from the most recent `step`, for the max-slope observable.
  ///
  /// Not `const`: `Field::with_host_view` is a non-const accessor even for a
  /// read-only pass (it brackets host/device coherence), so this method
  /// cannot be `const` while it reads `m_hx`/`m_hy` through it.
  void grad_magnitude(RealField &out) {
    m_hx.with_host_view([&](const double *hx, std::size_t n) {
      m_hy.with_host_view([&](const double *hy, std::size_t) {
        out.with_host_view([&](double *g, std::size_t) {
          for (std::size_t i = 0; i < n; ++i)
            g[i] = std::sqrt(hx[i] * hx[i] + hy[i] * hy[i]);
        });
      });
    });
  }

private:
  void spectral_derivative(const std::vector<double> &k, ComplexField &f_hat,
                           RealField &out) {
    const std::size_t n_out = m_fft.size_outbox();
    f_hat.with_host_view([&](const Complex *f, std::size_t) {
      m_grad_hat.with_host_view([&](Complex *g, std::size_t) {
        for (std::size_t i = 0; i < n_out; ++i) g[i] = Complex{0.0, k[i]} * f[i];
      });
    });
    Ops::backward(m_fft, m_grad_hat, out);
  }

  double m_dt;
  SurfaceStiffness m_stiffness;
  FFT &m_fft;
  std::vector<double> m_kx, m_ky, m_k_lap, m_mask, m_expL, m_phi1, m_L0;
  ComplexField m_h_hat, m_p_hat, m_grad_hat, m_flux_hat, m_div_hat;
  RealField m_grad_real, m_hx, m_hy, m_theta, m_Btheta;
};

} // namespace surface_diffusion
