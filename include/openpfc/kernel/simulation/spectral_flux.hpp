// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_flux.hpp
 * @brief Conservative flux \f$\nabla\cdot[c\,\nabla p]\f$ and its ETD1 stepper.
 *
 * @details
 * `SpectralETDSystem` splits a right-hand side into a pointwise remainder
 * times a reciprocal-space multiplier. That covers a local nonlinearity, not
 * one whose coefficient sits inside a divergence:
 *
 * \f[
 *   \partial_t u = \nabla\cdot\bigl[c\,\nabla p\bigr].
 * \f]
 *
 * \f$c\f$ may be a mobility of the transported field, \f$M(u)\f$, a
 * coefficient the caller has already evaluated (for example
 * \f$B(\theta(\nabla u))\f$), or a field registered for another reason.
 * The operator does not know which. It owns the spectral derivatives, the
 * real-space product, the divergence, the 2/3 dealiasing mask from
 * `kernel/fft/dealias.hpp`, and the scratch those steps need. The
 * application owns the coefficient law and the potential.
 *
 * ## Evaluation
 *
 * Per active axis \f$d\f$, given \f$\hat p\f$ and a real coefficient field
 * \f$c\f$ (or \f$u\f$ together with \f$M\f$):
 *
 * \f[
 *   \widehat{\partial_d p} = i k_d\,\hat p
 *   \;\to\; \partial_d p
 *   \;\to\; c\,\partial_d p
 *   \;\to\; \widehat{c\,\partial_d p}
 *   \;\to\; \text{accumulate } i k_d\,\widehat{c\,\partial_d p}.
 * \f]
 *
 * Inactive axes (global size 1) are left out of the sum. Their spacing is
 * not used for the mask, so a 2-D grid does not divide by a degenerate
 * \f$\Delta z\f$. The mask is exactly `fill_two_thirds_mask` from
 * `dealias.hpp`: a mode is kept only when
 * \f$|k_d| < (2/3)\pi/\Delta x_d\f$ on every active axis. A mode that lands
 * on that cutoff is removed. The earlier app-local mask used a strict
 * greater-than and would have kept that one mode. On a grid whose size is
 * not divisible by 3, no stored wave number sits on the cutoff, so the two
 * rules agree on every mode. This operator follows the public dealiasing
 * contract; it does not restore the old inequality.
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
 * with the same ETD1 update as the pointwise path. \f$L\f$ is the symbol
 * the application linearises about. A coefficient that is not \f$M(u)\f$
 * uses `SpectralFlux::divergence` directly; the application keeps its own
 * exponential update, as anisotropic surface diffusion does.
 *
 * ## Device
 *
 * `SpectralFlux<MemorySpace>` and `FluxETD<MemorySpace>` are written against
 * `SpectralETDOps<MemorySpace>`. Host specialisation lives in
 * `spectral_etd_ops.hpp`. A device translation unit includes
 * `runtime/gpu/spectral_etd_ops_gpu.hpp` before it instantiates
 * `CUDASpace` or `HIPSpace` — this kernel header does not include runtime.
 * The mobility or pass-through functor must be trivially copyable and
 * `OPENPFC_HD`, and `MobilityGradPointwise<Mobility>` must be instantiated
 * in one device translation unit. `apps/thin_film/src/gpu/` is the worked
 * example.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace pfc::sim {

/**
 * @brief Device-capable adapter: `M(u) * grad` in one pointwise kernel.
 *
 * `SpectralCell` carries `psi`, `psi_mf`, and `p_star`. `SpectralFlux`
 * stores the real-space gradient in the mean-field slot, so the coefficient
 * and the multiply share one launch. `Mobility` must be trivially copyable
 * and expose `OPENPFC_HD double operator()(double) const`.
 */
template <class Mobility> struct MobilityGradPointwise {
  Mobility mobility{};

  [[nodiscard]] OPENPFC_HD double nonlinearity(const SpectralCell &cell) const {
    return mobility(cell.psi) * cell.psi_mf;
  }
};

/**
 * @brief Pass a caller-filled coefficient field through unchanged.
 *
 * Use with `divergence(p_hat, coefficient, out_hat)` when \f$c\f$ is already
 * a field: \f$M(u)\f$ evaluated by the application, \f$B(\theta)\f$, or any
 * other local factor. The stored value is the factor.
 */
struct AsStored {
  [[nodiscard]] OPENPFC_HD double operator()(double value) const { return value; }
};

/**
 * @brief Spectral evaluation of \f$\nabla\cdot[c\,\nabla p]\f$.
 *
 * @tparam MemorySpace `HostSpace` (default), `CUDASpace`, or `HIPSpace`.
 */
template <class MemorySpace = pfc::HostSpace> class SpectralFlux {
  using Ops = SpectralETDOps<MemorySpace>;

public:
  using Complex = typename Ops::Complex;
  using RealField = typename Ops::RealField;
  using ComplexField = typename Ops::ComplexField;
  using FFT = typename Ops::FFT;
  using real_coeffs = typename Ops::real_coeffs;
  using complex_scratch = typename Ops::complex_scratch;

  SpectralFlux(const pfc::Domain &domain, FFT &fft)
      : m_fft(fft), m_grad(domain, fft.get_inbox_bounds(), 0),
        m_flux_hat(domain, fft.get_outbox_bounds(), 0) {
    const auto size = pfc::domain::get_size(domain);
    for (int d = 0; d < 3; ++d) m_active[d] = size[d] > 1;
    const std::size_t n = fft.size_outbox();

    std::vector<double> k[3];
    for (int d = 0; d < 3; ++d) k[d].assign(n, 0.0);
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          k[0][i] = kx;
          k[1][i] = ky;
          k[2][i] = kz;
        });

    // Public dealias contract: keep |k_d| < (2/3) π/Δx_d. Equality is
    // removed. Inactive axes get a dummy positive spacing so the helper
    // does not divide by zero; their wave numbers are zero, so they do
    // not drop modes.
    std::vector<double> mask_host(n, 1.0);
    auto mask_spacing = pfc::domain::get_spacing(domain);
    for (int d = 0; d < 3; ++d) {
      if (!m_active[d] || !(mask_spacing[d] > 0.0)) mask_spacing[d] = 1.0;
    }
    pfc::fft::kspace::fill_two_thirds_mask(fft.get_outbox_bounds(), size,
                                           mask_spacing, mask_host.data(), n);

    m_mask = Ops::make_real(n);
    Ops::upload(m_mask, std::span<const double>(mask_host));

    std::vector<Complex> zero(n, Complex{0.0, 0.0});
    std::vector<Complex> ones(n, Complex{1.0, 0.0});
    m_zero_c = Ops::make_complex(n);
    m_ones_c = Ops::make_complex(n);
    Ops::upload(m_zero_c, std::span<const Complex>(zero));
    Ops::upload(m_ones_c, std::span<const Complex>(ones));

    for (int d = 0; d < 3; ++d) {
      if (!m_active[d]) continue;
      std::vector<Complex> ik(n);
      for (std::size_t i = 0; i < n; ++i) ik[i] = Complex{0.0, k[d][i]};
      m_ik[d] = Ops::make_complex(n);
      Ops::upload(m_ik[d], std::span<const Complex>(ik));
    }

    m_grad_hat_scratch = Ops::make_complex(n);
    m_flux_hat_masked = Ops::make_complex(n);
    m_out_accum = Ops::make_complex(n);
    m_geometry = geometry_of(m_grad);
  }

  /**
   * @brief Overwrite @p out_hat with \f$\nabla\cdot[M(u)\nabla p]\f$.
   *
   * @param p_hat    transform of the potential
   * @param u        real field the mobility reads
   * @param mobility callable `OPENPFC_HD double(double)` returning \f$M(u)\f$;
   *                 trivially copyable. A device build also needs
   *                 `MobilityGradPointwise<Mobility>` instantiated in one
   *                 device translation unit
   * @param out_hat  overwritten with the divergence
   */
  template <class Mobility>
  void divergence(ComplexField &p_hat, RealField &u, Mobility &&mobility,
                  ComplexField &out_hat) {
    using MobilityT = std::decay_t<Mobility>;
    static_assert(std::is_trivially_copyable_v<MobilityT>,
                  "SpectralFlux::divergence: Mobility must be trivially "
                  "copyable to run through the device pointwise launcher");
    const MobilityGradPointwise<MobilityT> functor{mobility};

    bool first = true;
    for (int d = 0; d < 3; ++d) {
      if (!m_active[d]) continue;

      Ops::combine(p_hat, p_hat, m_ik[d], m_zero_c, m_grad_hat_scratch);
      Ops::backward(m_fft, m_grad_hat_scratch, m_grad);

      Ops::pointwise(m_geometry, 0.0, u, &m_grad, nullptr, m_grad, nullptr, functor);

      Ops::forward(m_fft, m_grad, m_flux_hat);
      Ops::multiply(m_flux_hat, m_mask, m_flux_hat_masked);

      if (first) {
        Ops::combine(p_hat, m_flux_hat_masked, m_zero_c, m_ik[d], m_out_accum);
        first = false;
      } else {
        Ops::combine(out_hat, m_flux_hat_masked, m_ones_c, m_ik[d], m_out_accum);
      }
      Ops::swap(out_hat, m_out_accum);
    }
    if (first) {
      Ops::combine(out_hat, out_hat, m_zero_c, m_zero_c, m_out_accum);
      Ops::swap(out_hat, m_out_accum);
    }
  }

  /**
   * @brief Overwrite @p out_hat with \f$\nabla\cdot[c\,\nabla p]\f$.
   *
   * @p coefficient is the local factor at each cell. The application fills
   * it; this method does not evaluate a material law.
   */
  void divergence(ComplexField &p_hat, RealField &coefficient,
                  ComplexField &out_hat) {
    divergence(p_hat, coefficient, AsStored{}, out_hat);
  }

private:
  static PointwiseGeometry geometry_of(const RealField &f) {
    const auto &o = f.origin();
    const auto &s = f.spacing();
    const auto &box = f.box();
    return PointwiseGeometry{.nx = box.size[0],
                             .ny = box.size[1],
                             .nz = box.size[2],
                             .low_x = box.low[0],
                             .low_y = box.low[1],
                             .low_z = box.low[2],
                             .origin_x = o[0],
                             .origin_y = o[1],
                             .origin_z = o[2],
                             .dx = s[0],
                             .dy = s[1],
                             .dz = s[2]};
  }

  FFT &m_fft;
  bool m_active[3]{};
  real_coeffs m_mask;
  complex_scratch m_ik[3];
  complex_scratch m_zero_c, m_ones_c;
  RealField m_grad;
  ComplexField m_flux_hat;
  complex_scratch m_grad_hat_scratch, m_flux_hat_masked, m_out_accum;
  PointwiseGeometry m_geometry{};
};

/**
 * @brief ETD1 stepper for a conserved equation with mobility \f$M(u)\f$.
 *
 * The caller supplies:
 *
 * * `potential(u_hat, u, p_hat)` — fill \f$\hat p\f$;
 * * `mobility(u)` — the state-dependent \f$M(u)\f$;
 *
 * and the linear symbol \f$L(k)\f$ used for the exponential integrator.
 * A coefficient that is not a function of \f$u\f$ does not use this stepper;
 * call `SpectralFlux::divergence` with the prepared field instead.
 *
 * @tparam MemorySpace `HostSpace` (default), `CUDASpace`, or `HIPSpace`.
 */
template <class MemorySpace = pfc::HostSpace> class FluxETD {
  using Ops = SpectralETDOps<MemorySpace>;

public:
  using Complex = typename Ops::Complex;
  using RealField = typename Ops::RealField;
  using ComplexField = typename Ops::ComplexField;
  using FFT = typename Ops::FFT;
  using real_coeffs = typename Ops::real_coeffs;
  using complex_scratch = typename Ops::complex_scratch;

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
    std::vector<double> expL(n, 1.0), phi1(n, dt), negL(n, 0.0), ones(n, 1.0);
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          const double k_lap = -(kx * kx + ky * ky + kz * kz);
          const double l = L(k_lap);
          negL[i] = -l;
          const double a = l * dt;
          expL[i] = std::exp(a);
          phi1[i] =
              (std::abs(a) < 1.0e-8) ? dt * (1.0 + 0.5 * a) : (expL[i] - 1.0) / l;
        });

    m_expL = Ops::make_real(n);
    m_phi1 = Ops::make_real(n);
    m_negL = Ops::make_real(n);
    m_ones = Ops::make_real(n);
    Ops::upload(m_expL, std::span<const double>(expL));
    Ops::upload(m_phi1, std::span<const double>(phi1));
    Ops::upload(m_negL, std::span<const double>(negL));
    Ops::upload(m_ones, std::span<const double>(ones));
    m_nl_scratch = Ops::make_complex(n);
    m_candidate = Ops::make_complex(n);
  }

  /**
   * @brief Advance @p u by one step. Returns `t + dt`.
   */
  template <class Potential, class Mobility>
  double step(double t, RealField &u, Potential &&potential, Mobility &&mobility) {
    Ops::forward(m_fft, u, m_u_hat);
    potential(m_u_hat, u, m_p_hat);
    m_flux.divergence(m_p_hat, u, std::forward<Mobility>(mobility), m_div_hat);

    Ops::combine(m_u_hat, m_div_hat, m_negL, m_ones, m_nl_scratch);
    Ops::combine(m_u_hat, m_nl_scratch, m_expL, m_phi1, m_candidate);
    Ops::swap(m_u_hat, m_candidate);

    Ops::backward(m_fft, m_u_hat, u);
    return t + m_dt;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }

private:
  FFT &m_fft;
  double m_dt;
  SpectralFlux<MemorySpace> m_flux;
  ComplexField m_u_hat, m_p_hat, m_div_hat;
  real_coeffs m_expL, m_phi1, m_negL, m_ones;
  complex_scratch m_nl_scratch, m_candidate;
};

} // namespace pfc::sim
