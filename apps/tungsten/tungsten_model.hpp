// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_model.hpp
 * @brief Tungsten Phase Field Crystal (PFC) model implementation
 *
 * @details
 * This file implements the Tungsten PFC model for simulating solidification and
 * crystal growth in tungsten. The model uses a mean-field filtered density field
 * approach with a quasi-Gaussian correlation function to capture the crystal
 * structure.
 *
 * ## Mathematical Model
 *
 * The evolution equation for the density field ψ is:
 *
 * \f[
 * \frac{\partial \psi}{\partial t} = \mathcal{L}[\psi] + \mathcal{N}[\psi,
 * \psi_{MF}]
 * \f]
 *
 * where:
 * - \f$\mathcal{L}[\psi]\f$ is the linear operator in Fourier space
 * - \f$\mathcal{N}[\psi, \psi_{MF}]\f$ is the nonlinear operator
 * - \f$\psi_{MF}\f$ is the mean-field filtered density
 *
 * ### Linear Operator
 *
 * In Fourier space, the linear operator is:
 *
 * \f[
 * \mathcal{L}(k) = \text{stabP} + \bar{p}_2 - B_x e^{-T/T_0} g_f(k) + \bar{q}_2
 * \chi(k)
 * \f]
 *
 * where:
 * - \f$g_f(k)\f$ is a quasi-Gaussian peak function
 * - \f$\chi(k) = e^{-k^2/(2\lambda^2)}\f$ is the mean-field filter
 * - \f$k = |\mathbf{k}|\f$ is the wave number magnitude
 *
 * ### Nonlinear Operator
 *
 * The nonlinear term in real space is:
 *
 * \f[
 * \mathcal{N}[\psi, \psi_{MF}] = \bar{p}_3 \psi^2 + \bar{p}_4 \psi^3 + \bar{q}_3
 * \psi_{MF}^2 + \bar{q}_4 \psi_{MF}^3
 * \f]
 *
 * ### Mean-Field Filtering
 *
 * The mean-field density is computed via:
 *
 * \f[
 * \psi_{MF}(\mathbf{r}) = \mathcal{F}^{-1}[\chi(k) \mathcal{F}[\psi](\mathbf{k})]
 * \f]
 *
 * where \f$\mathcal{F}\f$ denotes the Fourier transform.
 *
 * ### Quasi-Gaussian Peak Function
 *
 * The peak function \f$g_f(k)\f$ is defined as:
 *
 * \f[
 * g_f(k) = \begin{cases}
 *   g_1(k) & \text{if } k < 0 \\
 *   g_2(k) & \text{if } k \geq 0
 * \end{cases}
 * \f]
 *
 * where:
 * - \f$g_1(k) = \exp\left(-\frac{k^2 + r_{\text{tol}} k^{n}}{2\alpha^2}\right)\f$
 * (quasi-Gaussian)
 * - \f$g_2(k) = 1 - \frac{k^2}{2\alpha^2}\f$ (Taylor expansion)
 * - \f$r_{\text{tol}} = -2\alpha^2 \ln(\alpha_{\text{farTol}}) - 1\f$
 *
 * ### Time Integration
 *
 * The model uses exponential time integration:
 *
 * \f[
 * \psi(t+\Delta t) = e^{\mathcal{L} \Delta t} \psi(t) + \frac{e^{\mathcal{L} \Delta
 * t} - 1}{\mathcal{L}} \mathcal{N}[\psi, \psi_{MF}]
 * \f]
 *
 * ## Implementation Details
 *
 * - Uses FFT for efficient spectral methods
 * - Operators are precomputed in `prepare_operators()`
 * - Mean-field filtering is applied in each time step
 * - Supports MPI parallelization via domain decomposition
 *
 * @see tungsten_params.hpp for model parameters
 * @see tungsten_input.hpp for JSON configuration parsing
 * @see Model base class for interface documentation
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef TUNGSTEN_MODEL_HPP
#define TUNGSTEN_MODEL_HPP

#include "tungsten_params.hpp"
#include <openpfc/constants.hpp>
#include <openpfc/fft/kspace.hpp>
#include <openpfc/openpfc.hpp>
#include <openpfc/utils/nancheck.hpp>

using namespace pfc;
using namespace pfc::fft::kspace;
using namespace pfc::utils;

/**
 * @brief Tungsten Phase Field Crystal model
 *
 * Implements the Tungsten PFC model with mean-field filtering and quasi-Gaussian
 * correlation functions. The model simulates solidification and crystal growth
 * processes in tungsten.
 *
 * @note All parameters are accessed via `params.get_*()` getters
 * @note Parameters are set via `params.set_*()` setters
 */
class Tungsten : public Model {
  using Model::Model;

private:
  std::vector<double> filterMF; ///< Mean-field filter in Fourier space
  std::vector<double> opL;      ///< Linear operator: exp(L·dt)
  std::vector<double> opN;      ///< Nonlinear operator: (exp(L·dt) - 1) / L
#ifdef MAHTI_HACK
  // in principle, we can reuse some of the arrays ...
  std::vector<double> psiMF, psi, &psiN = psiMF;
  std::vector<std::complex<double>> psiMF_F, psi_F, &psiN_F = psiMF_F;
#else
  std::vector<double> psiMF;                 ///< Mean-field filtered density
  std::vector<double> psi;                   ///< Density field
  std::vector<double> psiN;                  ///< Nonlinear term
  std::vector<std::complex<double>> psiMF_F; ///< Mean-field in Fourier space
  std::vector<std::complex<double>> psi_F;   ///< Density in Fourier space
  std::vector<std::complex<double>> psiN_F;  ///< Nonlinear term in Fourier space
#endif
  size_t mem_allocated = 0; ///< Memory allocated (for debugging)

public:
  /**
   * @brief Model parameters
   *
   * Access parameters using getters: `params.get_n0()`, `params.get_T()`, etc.
   * Set parameters using setters: `params.set_n0(value)`, `params.set_T(value)`,
   * etc.
   */
  TungstenParams params;

  /**
   * @brief Allocate memory for fields and operators
   *
   * Resizes all field arrays based on FFT inbox/outbox sizes and registers
   * fields with the Model base class.
   */
  void allocate() {
    FFT &fft = get_fft();
    auto size_inbox = fft.size_inbox();
    auto size_outbox = fft.size_outbox();

    // Operators are only half size due to the symmetry of Fourier space
    filterMF.resize(size_outbox);
    opL.resize(size_outbox);
    opN.resize(size_outbox);

    // Real-space fields
    psi.resize(size_inbox);
    psiMF.resize(size_inbox);
    psiN.resize(size_inbox);

    // Fourier-space fields (suffix F means in Fourier space)
    psi_F.resize(size_outbox);
    psiMF_F.resize(size_outbox);
    psiN_F.resize(size_outbox);

    // Register fields with Model base class
    add_real_field("psi", psi);
    add_real_field("default", psi); // for backward compatibility
    add_real_field("psiMF", psiMF);

    // Track memory usage
    mem_allocated = 0;
    mem_allocated += utils::sizeof_vec(filterMF);
    mem_allocated += utils::sizeof_vec(opL);
    mem_allocated += utils::sizeof_vec(opN);
    mem_allocated += utils::sizeof_vec(psi);
    mem_allocated += utils::sizeof_vec(psiMF);
    mem_allocated += utils::sizeof_vec(psiN);
    mem_allocated += utils::sizeof_vec(psi_F);
    mem_allocated += utils::sizeof_vec(psiMF_F);
    mem_allocated += utils::sizeof_vec(psiN_F);
  }

  /**
   * @brief Precompute time integration operators in k-space
   *
   * Computes the linear and nonlinear operators for exponential time integration:
   * - Linear operator: \f$L(k) = \exp(\mathcal{L}(k) \cdot \Delta t)\f$
   * - Nonlinear operator: \f$N(k) = [\exp(\mathcal{L}(k) \cdot \Delta t) - 1] /
   * \mathcal{L}(k)\f$
   *
   * The linear operator \f$\mathcal{L}(k)\f$ includes:
   * - Stabilization term: `stabP`
   * - Vapor model term: `p2_bar`
   * - Temperature-dependent peak: \f$B_x e^{-T/T_0} g_f(k)\f$
   * - Mean-field coupling: \f$\bar{q}_2 \chi(k)\f$
   *
   * @param dt Time step size
   */
  void prepare_operators(double dt) {
    auto &fft = get_fft();
    auto &world = get_world();
    [[maybe_unused]] auto [dx, dy, dz] = get_spacing(world);
    auto [Lx, Ly, Lz] = get_size(world);

    auto outbox = get_outbox(fft);
    auto low = outbox.low;
    auto high = outbox.high;

    // Get frequency scaling factors using helper function
    auto [fx, fy, fz] = k_frequency_scaling(world);

    // Get model parameters
    double alpha = params.get_alpha();
    double alpha2 = 2.0 * alpha * alpha;
    double lambda = params.get_lambda();
    double lambda2 = 2.0 * lambda * lambda;
    double alpha_farTol = params.get_alpha_farTol();
    int alpha_highOrd = params.get_alpha_highOrd();
    double Bx = params.get_Bx();
    double T = params.get_T();
    double T0 = params.get_T0();
    double stabP = params.get_stabP();
    double p2_bar = params.get_p2_bar();
    double q2_bar = params.get_q2_bar();

    int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {

          // Compute wave vector components using helper function
          double ki = k_component(i, Lx, fx);
          double kj = k_component(j, Ly, fy);
          double kk = k_component(k, Lz, fz);

          // Compute Laplacian operator -k² using helper function
          double kLap = k_laplacian_value(ki, kj, kk);

          // Mean-field filtering operator: χ(k) = exp(-k²/(2λ²))
          double fMF = exp(kLap / lambda2);
          filterMF[idx] = fMF;

          // Compute quasi-Gaussian peak function g_f(k)
          double k_val = sqrt(-kLap) - 1.0;
          double k2 = k_val * k_val;

          // Tolerance parameter for higher-order component
          double rTol = -alpha2 * log(alpha_farTol) - 1.0;

          double g1 = 0.0;
          if (alpha_highOrd == 0) {
            // Pure Gaussian peak
            g1 = exp(-k2 / alpha2);
          } else {
            // Quasi-Gaussian peak with higher-order component to make it decay
            // faster towards k=0
            g1 = exp(-(k2 + rTol * pow(k_val, alpha_highOrd)) / alpha2);
          }

          // Taylor expansion of Gaussian peak to order 2 (for k ≥ 0)
          double g2 = 1.0 - 1.0 / alpha2 * k2;

          // Splice the two sides of the peak
          double gf = (k_val < 0.0) ? g1 : g2;

          // Temperature-dependent peak contribution
          double opPeak = Bx * exp(-T / T0) * gf;

          // Linear operator: L(k) = stabP + p2_bar - opPeak + q2_bar * χ(k)
          double opCk = stabP + p2_bar - opPeak + q2_bar * fMF;

          // Exponential time integration operators
          opL[idx] = exp(kLap * opCk * dt);
          opN[idx] = (opCk == 0.0) ? kLap * dt : (opL[idx] - 1.0) / opCk;

          idx += 1;
        }
      }
    }

    CHECK_AND_ABORT_IF_NANS(opL);
    CHECK_AND_ABORT_IF_NANS(opN);
  }

  /**
   * @brief Initialize the model
   *
   * Allocates memory and precomputes operators. Must be called before time
   * stepping.
   *
   * @param dt Time step size
   */
  void initialize(double dt) override {
    allocate();
    prepare_operators(dt);
  }

  /**
   * @brief Perform one time step of the evolution equation
   *
   * Implements the exponential time integration scheme:
   * 1. Compute mean-field filtered density ψ_MF
   * 2. Calculate nonlinear term N[ψ, ψ_MF] in real space
   * 3. Transform nonlinear term to Fourier space
   * 4. Apply linear and nonlinear operators
   * 5. Transform back to real space
   *
   * @param t Current time (unused, kept for interface compatibility)
   */
  void step(double t) override {
    (void)t; // suppress compiler warning about unused parameter

    FFT &fft = get_fft();

    // Step 1: Calculate mean-field density n_MF
    // Forward FFT: ψ → ψ̂
    fft.forward(psi, psi_F);

    // Apply mean-field filter in Fourier space: ψ̂_MF = χ(k) · ψ̂
    for (size_t idx = 0, N = psiMF_F.size(); idx < N; idx++) {
      psiMF_F[idx] = filterMF[idx] * psi_F[idx];
    }

    // Inverse FFT: ψ̂_MF → ψ_MF
    fft.backward(psiMF_F, psiMF);

    // Step 2: Calculate nonlinear part in real space
    // N[ψ, ψ_MF] = p̄₃ψ² + p̄₄ψ³ + q̄₃ψ_MF² + q̄₄ψ_MF³
    double p3_bar = params.get_p3_bar();
    double p4_bar = params.get_p4_bar();
    double q3_bar = params.get_q3_bar();
    double q4_bar = params.get_q4_bar();

    for (size_t idx = 0, N = psiN.size(); idx < N; idx++) {
      double u = psi[idx];
      double v = psiMF[idx];
      double u2 = u * u;
      double u3 = u * u * u;
      double v2 = v * v;
      double v3 = v * v * v;
      psiN[idx] = p3_bar * u2 + p4_bar * u3 + q3_bar * v2 + q4_bar * v3;
    }

    // Step 3: Apply stabilization factor if given
    double stabP = params.get_stabP();
    if (stabP != 0.0) {
      for (size_t idx = 0, N = psiN.size(); idx < N; idx++) {
        psiN[idx] = psiN[idx] - stabP * psi[idx];
      }
    }

    // Step 4: Transform nonlinear term to Fourier space
    fft.forward(psiN, psiN_F);

    // Step 5: Apply exponential time integration in Fourier space
    // ψ̂(t+Δt) = L(k)·ψ̂(t) + N(k)·N̂[ψ, ψ_MF]
    for (size_t idx = 0, N = psi_F.size(); idx < N; idx++) {
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];
    }

    // Step 6: Transform back to real space
    fft.backward(psi_F, psi);

    // Check for NaNs (enabled in Debug mode)
    CHECK_AND_ABORT_IF_NANS(psi);
  }

  /**
   * @brief Constructs a Tungsten model with the given World object
   *
   * @param world The World object defining the simulation domain
   */
  explicit Tungsten(const World &world) : Model(world) {
    // Additional initialization if needed
  }
};

#endif // TUNGSTEN_MODEL_HPP
