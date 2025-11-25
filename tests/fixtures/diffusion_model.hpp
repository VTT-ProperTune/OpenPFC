// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file diffusion_model.hpp
 * @brief Test fixture for diffusion model integration tests
 *
 * This is a test-specific version of the diffusion model adapted from
 * examples/diffusion_model.hpp. It provides:
 * - Public member variables with m_ prefix (Laboratory philosophy)
 * - No verbose output (suitable for automated testing)
 * - Helper methods for test validation (e.g., get_midpoint_idx())
 * - Clear field access for numerical validation
 *
 * This model solves the diffusion equation:
 *   ∂u/∂t = D∇²u
 * using a spectral method with implicit time stepping.
 */

#ifndef OPENPFC_TESTS_DIFFUSION_MODEL_HPP
#define OPENPFC_TESTS_DIFFUSION_MODEL_HPP

#include <complex>
#include <openpfc/constants.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/model.hpp>
#include <vector>

namespace pfc {
namespace test {

/**
 * @brief Simple diffusion model for integration testing
 *
 * This class implements the diffusion equation using spectral methods.
 * All data is public for transparency and easy inspection in tests.
 *
 * LLM: Test-specific model - follows Laboratory philosophy with public fields
 * LLM: Adapted from examples/ but simplified for testing
 */
class DiffusionModel : public Model {
  using Model::Model;

public:
  // LLM: Public fields with m_ prefix - transparent, inspectable (Laboratory)
  std::vector<double> m_psi;                 ///< Real-space field
  std::vector<std::complex<double>> m_psi_F; ///< Fourier-space field
  std::vector<double> m_opL;                 ///< Laplacian operator in k-space
  int m_midpoint_idx = -1;                   ///< Index of domain center point
  double m_diffusion_coefficient = 1.0;      ///< Diffusion coefficient D

  /**
   * @brief Initialize diffusion model
   *
   * Sets up:
   * - Field storage arrays
   * - k-space Laplacian operator
   * - Initial Gaussian condition centered at origin
   *
   * LLM: Initialize allocates fields and computes spectral operators
   *
   * @param dt Time step size
   */
  void initialize(double dt) override {
    const World &w = get_world();
    FFT &fft = get_fft();

    // Allocate fields
    // LLM: FFT determines field sizes based on decomposition
    m_psi.resize(fft.size_inbox());
    m_psi_F.resize(fft.size_outbox());
    m_opL.resize(fft.size_outbox());

    // Get local domain bounds
    // LLM: Each MPI rank has subset of domain (inbox = real space, outbox = k-space)
    Vec3<int> i_low = get_inbox(fft).low;
    Vec3<int> i_high = get_inbox(fft).high;
    Vec3<int> o_low = get_outbox(fft).low;
    Vec3<int> o_high = get_outbox(fft).high;

    auto origin = get_origin(w);
    auto spacing = get_spacing(w);

    // Initialize field with Gaussian: u(x,0) = exp(-r²/(4D))
    // LLM: Initial condition - fundamental solution to diffusion equation
    int idx = 0;
    for (int k = i_low[2]; k <= i_high[2]; k++) {
      for (int j = i_low[1]; j <= i_high[1]; j++) {
        for (int i = i_low[0]; i <= i_high[0]; i++) {
          double x = origin[0] + i * spacing[0];
          double y = origin[1] + j * spacing[1];
          double z = origin[2] + k * spacing[2];
          double r2 = x * x + y * y + z * z;
          m_psi[idx] = std::exp(-r2 / (4.0 * m_diffusion_coefficient));

          // Track center point for validation
          // LLM: Midpoint tracking needed for 3D Gaussian test validation
          if (std::abs(x) < 1.0e-9 && std::abs(y) < 1.0e-9 && std::abs(z) < 1.0e-9) {
            m_midpoint_idx = idx;
          }
          idx += 1;
        }
      }
    }

    // Compute k-space Laplacian operator: -|k|²
    // LLM: Spectral method - differential operators become algebraic in Fourier
    // space
    idx = 0;
    const double fx = 2.0 * constants::pi / (spacing[0] * get_size(w, 0));
    const double fy = 2.0 * constants::pi / (spacing[1] * get_size(w, 1));
    const double fz = 2.0 * constants::pi / (spacing[2] * get_size(w, 2));

    for (int k = o_low[2]; k <= o_high[2]; k++) {
      for (int j = o_low[1]; j <= o_high[1]; j++) {
        for (int i = o_low[0]; i <= o_high[0]; i++) {
          // Handle FFT frequency wrapping
          // LLM: FFT convention - frequencies above Nyquist are negative
          const double ki =
              (i <= get_size(w, 0) / 2) ? i * fx : (i - get_size(w, 0)) * fx;
          const double kj =
              (j <= get_size(w, 1) / 2) ? j * fy : (j - get_size(w, 1)) * fy;
          const double kk =
              (k <= get_size(w, 2) / 2) ? k * fz : (k - get_size(w, 2)) * fz;

          const double k2 = ki * ki + kj * kj + kk * kk;

          // Implicit Euler: (1 - dt*D*∇²)u^{n+1} = u^n
          // In k-space: (1 + dt*D*k²)û^{n+1} = û^n
          // LLM: Implicit method - unconditionally stable for diffusion
          m_opL[idx] = 1.0 / (1.0 + dt * m_diffusion_coefficient * k2);
          idx++;
        }
      }
    }
  }

  /**
   * @brief Perform one time step of diffusion
   *
   * Implements spectral diffusion solve:
   * 1. Forward FFT: u → û
   * 2. Apply operator in k-space: û^{n+1} = (1 + dt*D*k²)^{-1} û^n
   * 3. Backward FFT: û → u
   *
   * LLM: Spectral method - O(N log N) complexity via FFT
   *
   * @param t Current time (unused in linear diffusion)
   */
  void step(double /* t */) override {
    FFT &fft = get_fft();

    // Transform to k-space
    // LLM: FFT is MPI-aware - handles distributed memory automatically
    fft.forward(m_psi, m_psi_F);

    // Apply operator in k-space
    // LLM: Hot loop - no allocations, fully vectorizable
    for (size_t k = 0, N = m_psi_F.size(); k < N; k++) {
      m_psi_F[k] = m_opL[k] * m_psi_F[k];
    }

    // Transform back to real space
    fft.backward(m_psi_F, m_psi);
  }

  /**
   * @brief Get reference to field for direct access
   *
   * LLM: Override provides direct field access for validation
   *
   * @return Reference to real-space field
   */
  Field &get_field() override { return m_psi; }

  /**
   * @brief Get index of center point in local domain
   *
   * Returns the local array index of the point at the origin (0,0,0),
   * or -1 if this rank doesn't own the center point.
   *
   * LLM: Needed for 3D Gaussian center point validation
   *
   * @return Local index of center point, or -1 if not on this rank
   */
  int get_midpoint_idx() const { return m_midpoint_idx; }

  /**
   * @brief Set diffusion coefficient
   *
   * @param D Diffusion coefficient (must be positive)
   */
  void set_diffusion_coefficient(double D) { m_diffusion_coefficient = D; }
};

} // namespace test
} // namespace pfc

#endif // OPENPFC_TESTS_DIFFUSION_MODEL_HPP
