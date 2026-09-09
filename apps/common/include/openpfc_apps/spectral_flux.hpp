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
 * `SpectralFlux<MemorySpace>` and `FluxETD<MemorySpace>` run on host *and*
 * device: every elementwise step (the complex \f$ik_d\f$ gradient, the
 * dealias mask, the ETD combine) goes through
 * `pfc::sim::SpectralETDOps<MemorySpace>`, which for `CUDASpace`/`HIPSpace`
 * dispatches to the real CUDA/HIP kernels behind `combine_raw` — including its
 * **complex**-coefficient overload (`combine_two_term_{cuda,hip}_impl` with
 * `const Complex *e, const Complex *w`), which is exactly what a complex
 * `i k_d` multiply needs. The one piece that is not a diagonal k-space
 * operator is the real-space mobility \f$M(u)\f$; it is evaluated with the
 * same device-capable pointwise mechanism the physics nonlinearities use
 * (`Ops::pointwise`, `OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE`), fused with the
 * multiply against the gradient in one kernel launch via `MobilityGradPointwise`
 * (see below). A caller wiring the mobility onto a device build must supply an
 * `OPENPFC_HD`, trivially-copyable `Mobility` and instantiate
 * `MobilityGradPointwise<Mobility>` in one `.cu`/`.hip` translation unit,
 * exactly as a physics functor does; `apps/thin_film/src/gpu/` is the worked
 * example (`CubicMobility` on `pfc::HIPSpace`).
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <numbers>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>
#include <openpfc/runtime/gpu/spectral_etd_ops_gpu.hpp>

namespace pfc::apps {

/**
 * @brief Device-capable adapter: `M(u) * grad` in one pointwise kernel.
 *
 * @details
 * `pfc::sim::SpectralCell` carries three real inputs per cell (`psi`,
 * `psi_mf`, `p_star`); `SpectralFlux` repurposes the mean-field slot to carry
 * the already-transformed real-space gradient, so the mobility evaluation and
 * its multiply against the gradient are one launch instead of two. `Mobility`
 * must be trivially copyable and expose `OPENPFC_HD double operator()(double)
 * const` (a small value struct such as `thin_film::CubicMobility`, or a
 * capture-by-value lambda for the host-only path).
 */
template <class Mobility> struct MobilityGradPointwise {
  Mobility mobility{};

  [[nodiscard]] OPENPFC_HD double
  nonlinearity(const pfc::sim::SpectralCell &cell) const {
    return mobility(cell.psi) * cell.psi_mf;
  }
};

/**
 * @brief Spectral evaluation of \f$\nabla\cdot[M(u)\nabla p]\f$.
 *
 * @tparam MemorySpace `HostSpace` (default), `CUDASpace`, or `HIPSpace`.
 */
template <class MemorySpace = pfc::HostSpace> class SpectralFlux {
  using Ops = pfc::sim::SpectralETDOps<MemorySpace>;

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
    // Orszag 2/3 mask. M(u) grad p is a strongly nonlinear product, so the
    // transform of the flux carries wave numbers the grid cannot represent.
    // Without this the aliased content folds back and the run diverges once
    // the profile steepens -- and it does so independently of the timestep,
    // which is how the omission shows itself.
    std::vector<double> mask_host(n, 1.0);
    const auto dx = pfc::domain::get_spacing(domain);
    double cut[3]{};
    for (int d = 0; d < 3; ++d)
      cut[d] = (size[d] > 1) ? (2.0 / 3.0) * (std::numbers::pi / dx[d]) : 0.0;
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          k[0][i] = kx;
          k[1][i] = ky;
          k[2][i] = kz;
          const double a[3]{std::abs(kx), std::abs(ky), std::abs(kz)};
          for (int d = 0; d < 3; ++d)
            if (m_active[d] && a[d] > cut[d]) mask_host[i] = 0.0;
        });

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
   * @param u        real field the mobility depends on
   * @param mobility callable `OPENPFC_HD double(double u)` returning
   *                 \f$M(u)\f$; must be trivially copyable (device builds
   *                 additionally need `MobilityGradPointwise<Mobility>`
   *                 explicitly instantiated in one device translation unit)
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

      // grad_hat = i k_d * p_hat  (second combine term zeroed)
      Ops::combine(p_hat, p_hat, m_ik[d], m_zero_c, m_grad_hat_scratch);
      Ops::backward(m_fft, m_grad_hat_scratch, m_grad);

      // grad *= M(u), in real space, where the nonlinearity actually lives.
      // Fused into one kernel: n_out = mobility(psi) * psi_mf, with psi_mf
      // repurposed to carry the gradient (see MobilityGradPointwise).
      Ops::pointwise(m_geometry, 0.0, u, &m_grad, nullptr, m_grad, nullptr,
                     functor);

      Ops::forward(m_fft, m_grad, m_flux_hat);
      Ops::multiply(m_flux_hat, m_mask, m_flux_hat_masked);

      // out_hat = i k_d * dealias(flux_hat)  [first active axis]
      // out_hat += i k_d * dealias(flux_hat) [subsequent axes]
      if (first) {
        Ops::combine(p_hat, m_flux_hat_masked, m_zero_c, m_ik[d], m_out_accum);
        first = false;
      } else {
        Ops::combine(out_hat, m_flux_hat_masked, m_ones_c, m_ik[d], m_out_accum);
      }
      Ops::swap(out_hat, m_out_accum);
    }
    if (first) {
      // No active axis (degenerate 1x1x1 domain): out_hat must still be
      // zeroed, matching every other path through this loop.
      Ops::combine(out_hat, out_hat, m_zero_c, m_zero_c, m_out_accum);
      Ops::swap(out_hat, m_out_accum);
    }
  }

private:
  static pfc::sim::PointwiseGeometry geometry_of(const RealField &f) {
    const auto &o = f.origin();
    const auto &s = f.spacing();
    const auto &box = f.box();
    return pfc::sim::PointwiseGeometry{.nx = box.size[0],
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
  pfc::sim::PointwiseGeometry m_geometry{};
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
 *
 * @tparam MemorySpace `HostSpace` (default), `CUDASpace`, or `HIPSpace`.
 */
template <class MemorySpace = pfc::HostSpace> class FluxETD {
  using Ops = pfc::sim::SpectralETDOps<MemorySpace>;

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
          // (e^a - 1)/l, by series where the quotient loses precision.
          phi1[i] = (std::abs(a) < 1.0e-8) ? dt * (1.0 + 0.5 * a)
                                          : (expL[i] - 1.0) / l;
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

    // N = full flux divergence minus the part already carried by L:
    //   nl_hat = div_hat - L * u_hat = (-L)*u_hat + 1*div_hat
    Ops::combine(m_u_hat, m_div_hat, m_negL, m_ones, m_nl_scratch);
    // candidate = exp(L dt) * u_hat + phi1(L dt) * nl_hat
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

} // namespace pfc::apps
