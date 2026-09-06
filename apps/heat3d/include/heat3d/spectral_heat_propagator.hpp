// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_heat_propagator.hpp
 * @brief Implicit-Euler-in-Fourier-space propagator for the heat equation.
 *
 * @details
 * One step of \f$\partial_t u = D\nabla^2 u\f$ in Fourier space with
 * implicit Euler is the elementwise multiply
 * \f$\hat u^{n+1}(\mathbf k) = \hat u^{n}(\mathbf k)\,/\,
 *    \bigl(1 - \Delta t\, D\, k_\mathrm{lap}(\mathbf k)\bigr)\f$
 * with \f$k_\mathrm{lap} = -(k_x^2 + k_y^2 + k_z^2)\f$.
 *
 * The wavenumber lookup-table \f$1/(1 - \Delta t\,D\,k_\mathrm{lap})\f$
 * is heat-specific (it embeds the operator's symbol), so this lives in
 * the heat3d application rather than in OpenPFC. The constructor builds
 * the table from the FFT layout once; `step` is a fwd FFT, an
 * elementwise multiply, and an inv FFT.
 *
 * Backend-agnostic: holds a reference to the abstract `pfc::fft::IHostFFT`,
 * so the same class works with CPU FFTW, cuFFT, ROCm — whatever
 * `fft::create` returns.
 *
 * Lifetime contract: the propagator borrows the FFT by reference, so the
 * `IHostFFT` instance must outlive the propagator (same pattern as
 * `pfc::field::SpectralGradient`).
 */

#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

namespace heat3d {

/**
 * @brief Fill the implicit-Euler Fourier multiplier
 *        \f$1/(1 - \Delta t\, D\, k_\mathrm{lap})\f$ for one rank's outbox.
 *
 * @details Host-only and FFT-backend-free: the same table is uploaded to the
 *          device by `SpectralHeatPropagatorHIP`. `k_lap = -(k_x^2+k_y^2+k_z^2)`
 *          with the usual periodic signed-wavenumber convention
 *          (`i <= N/2` → positive, else wrapped). Layout is k-outer, i-inner,
 *          matching HeFFTe's local outbox walk.
 *
 * @param opL      Destination, length equal to the outbox volume.
 * @param outbox   Local Fourier-space index box (inclusive corners).
 * @param size     Global grid `{Nx, Ny, Nz}`.
 * @param spacing  Grid spacing `{dx, dy, dz}`.
 * @param D        Diffusion coefficient.
 * @param dt       Time-step size.
 * @throws std::invalid_argument when `opL.size()` does not match the outbox
 *         volume implied by `outbox.low`/`outbox.high`.
 */
inline void fill_implicit_euler_symbol(std::span<double> opL,
                                       const pfc::Box3i &outbox, pfc::Int3 size,
                                       pfc::Real3 spacing, double D, double dt) {
  const auto axis_count = [](int lo, int hi) -> std::size_t {
    return static_cast<std::size_t>(hi - lo + 1);
  };
  const std::size_t expected = axis_count(outbox.low[0], outbox.high[0]) *
                               axis_count(outbox.low[1], outbox.high[1]) *
                               axis_count(outbox.low[2], outbox.high[2]);
  if (opL.size() != expected) {
    throw std::invalid_argument("fill_implicit_euler_symbol: opL size " +
                                std::to_string(opL.size()) + " != outbox volume " +
                                std::to_string(expected));
  }
  const double fx =
      2.0 * pfc::constants::pi / (spacing[0] * static_cast<double>(size[0]));
  const double fy =
      2.0 * pfc::constants::pi / (spacing[1] * static_cast<double>(size[1]));
  const double fz =
      2.0 * pfc::constants::pi / (spacing[2] * static_cast<double>(size[2]));
  const auto signed_k = [](int i, int n, double f) {
    return (i <= n / 2) ? static_cast<double>(i) * f
                        : static_cast<double>(i - n) * f;
  };
  std::size_t idx = 0;
  for (int k = outbox.low[2]; k <= outbox.high[2]; ++k) {
    for (int j = outbox.low[1]; j <= outbox.high[1]; ++j) {
      for (int i = outbox.low[0]; i <= outbox.high[0]; ++i) {
        const double ki = signed_k(i, size[0], fx);
        const double kj = signed_k(j, size[1], fy);
        const double kk = signed_k(k, size[2], fz);
        const double k_lap = -(ki * ki + kj * kj + kk * kk);
        opL[idx++] = 1.0 / (1.0 - dt * D * k_lap);
      }
    }
  }
}

/**
 * @brief Heat equation solver using implicit Euler time integration in Fourier space
 *
 * @details SpectralHeatPropagator solves the three-dimensional heat equation
 * ∂T/∂t = α * ∇²T on a doubly-periodic domain using an implicit Euler
 * time-integration scheme in Fourier space. The implementation is unconditionally
 * stable and efficient, requiring only forward and inverse FFTs per time step.
 *
 * @par Integrator method
 * Concrete integrator: implicit Euler (first-order, unconditionally stable).
 * Each advance() call computes the solution in spectral space using the exact
 * symbol of the Laplacian operator: T_next = T / (1 - dt * α * k_lap), where
 * k_lap = -(k_x² + k_y² + k_z²) is the Fourier symbol of the Laplacian.
 *
 * @par Lifecycle stage ownership
 * SpectralHeatPropagator implements the following lifecycle stages:
 * - Pre-step construction: builds the wavenumber lookup table from FFT layout
 *   and diffusion coefficient
 * - Time advancement: performs forward FFT of field, applies diagonal multiplier
 *   in Fourier space, then inverse FFT back to physical space
 * - No explicit boundary/halo exchange: periodic boundary conditions are
 *   implicit in Fourier representation
 *
 * @par Boundary/halo synchronization
 * Boundary conditions and halo exchanges occur at:
 * - Construction: the wavenumber table is precomputed based on domain geometry
 * - Runtime: periodic boundary conditions are automatically satisfied by the
 *   Fourier representation; no explicit halo exchange is needed
 * - The propagator assumes a fully periodic domain; Dirichlet/Neumann conditions
 *   would require additional modifications (not supported in this implementation)
 *
 * @par Application-specific constraints
 * - Spectral transforms: assumes the underlying FFT layout is compatible with
 *   the provided field geometry
 * - Time step stability: the implicit Euler scheme is unconditionally stable,
 *   allowing arbitrarily large dt without numerical instability
 * - Spectral accuracy: the method achieves spectral spatial accuracy for smooth
 *   solutions, but still only first-order accuracy in time
 * - Memory: requires storage for complex-valued spectral representation of the field
 *
 * @par Contract for substituting alternative integrators
 * To implement a different time-integration scheme (e.g., explicit Euler,
 * Runge-Kutta, Crank-Nicolson), subclasses must:
 * - Override the step() method to implement the desired algorithm
 * - Preserve the constructor interface (FFT, field, diffusion coefficient, dt)
 * - Document the new scheme's stability constraints and accuracy order
 * - For explicit schemes, implement appropriate time step restrictions
 * - For multi-stage schemes, manage intermediate storage appropriately
 *
 * @note This propagator does not inherit from Simulator; it is a standalone
 *   implementation designed for the Heat3D application family. The documentation
 *   here provides a concrete example of how time-integration contracts are
 *   fulfilled in practice, complementing the abstract contract described in
 *   the Simulator base class.
 *
 * @see Simulator for the base class contract on time-integration assumptions
 *   and how to substitute alternative integrators.
 */
class SpectralHeatPropagator {
public:
  /**
   * @param fft FFT plan to reuse (borrowed; must outlive the propagator).
   * @param u   Field that defines the global grid size + spacing used to
   *            build the wavenumber table. The propagator does not
   *            retain a reference to this field — only its geometry is
   *            sampled at construction.
   * @param D   Diffusion coefficient (heat-equation parameter).
   * @param dt  Time-step size.
   */
  SpectralHeatPropagator(pfc::fft::IHostFFT &fft, const pfc::data::Field<double> &u,
                         double D, double dt)
      : m_fft(fft), m_psi_F(fft.size_outbox()), m_opL(fft.size_outbox()) {
    fill_implicit_euler_symbol(m_opL, fft.get_outbox_bounds(), u.global_size(),
                               u.spacing(), D, dt);
  }

  /** Advance `u` by one implicit-Euler step (1 fwd FFT + 1 inv FFT). */
  void step(pfc::data::Field<double> &u) {
    m_fft.forward(u.vec(), m_psi_F);
    for (std::size_t k = 0; k < m_psi_F.size(); ++k) m_psi_F[k] *= m_opL[k];
    m_fft.backward(m_psi_F, u.vec());
  }

private:
  pfc::fft::IHostFFT &m_fft;
  std::vector<std::complex<double>> m_psi_F;
  std::vector<double> m_opL;
};

} // namespace heat3d
