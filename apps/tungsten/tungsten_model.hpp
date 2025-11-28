// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef TUNGSTEN_MODEL_HPP
#define TUNGSTEN_MODEL_HPP

#include "tungsten_params.hpp"
#include <openpfc/openpfc.hpp>
#include <openpfc/utils/nancheck.hpp>

namespace pfc {
namespace utils {} // namespace utils
} // namespace pfc

using namespace pfc;
using namespace pfc::utils;

/**
 * @brief Tungsten model class for OpenPFC. This class is used to define the
 * model.
 *
 */
class Tungsten : public Model {
  using Model::Model;

private:
  std::vector<double> filterMF, opL, opN;
#ifdef MAHTI_HACK
  // in principle, we can reuse some of the arrays ...
  std::vector<double> psiMF, psi, &psiN = psiMF;
  std::vector<std::complex<double>> psiMF_F, psi_F, &psiN_F = psiMF_F;
#else
  std::vector<double> psiMF, psi, psiN;
  std::vector<std::complex<double>> psiMF_F, psi_F, psiN_F;
#endif
  size_t mem_allocated = 0;

public:
  /**
   * @brief Model parameters, which can be overridden from json file
   *
   * Access parameters using getters: params.get_n0(), params.get_T(), etc.
   * Set parameters using setters: params.set_n0(value), params.set_T(value), etc.
   */
  TungstenParams params;

  void allocate() {
    FFT &fft = get_fft();
    auto size_inbox = fft.size_inbox();
    auto size_outbox = fft.size_outbox();

    // operators are only half size due to the symmetry of fourier space
    filterMF.resize(size_outbox);
    opL.resize(size_outbox);
    opN.resize(size_outbox);

    // psi, psiMF, psiN
    psi.resize(size_inbox);
    psiMF.resize(size_inbox);
    psiN.resize(size_inbox);

    // psi_F, psiMF_F, psiN_F, where suffix F means in fourier space
    psi_F.resize(size_outbox);
    psiMF_F.resize(size_outbox);
    psiN_F.resize(size_outbox);

    add_real_field("psi", psi);
    add_real_field("default", psi); // for backward compatibility
    add_real_field("psiMF", psiMF);

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

  void prepare_operators(double dt) {
    auto &fft = get_fft();
    auto &world = get_world();
    auto [dx, dy, dz] = get_spacing(world);
    auto [Lx, Ly, Lz] = get_size(world);

    auto low = get_outbox(fft).low;
    auto high = get_outbox(fft).high;

    int idx = 0;
    const double pi = std::atan(1.0) * 4.0;
    const double fx = 2.0 * pi / (dx * Lx);
    const double fy = 2.0 * pi / (dy * Ly);
    const double fz = 2.0 * pi / (dz * Lz);

    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {

          // laplacian operator -k^2
          double ki = (i <= Lx / 2) ? i * fx : (i - Lx) * fx;
          double kj = (j <= Ly / 2) ? j * fy : (j - Ly) * fy;
          double kk = (k <= Lz / 2) ? k * fz : (k - Lz) * fz;
          double kLap = -(ki * ki + kj * kj + kk * kk);

          // mean-field filtering operator (chi) make a C2 that's quasi-gaussian
          // on the left, and ken-style on the right
          double alpha = params.get_alpha();
          double alpha2 = 2.0 * alpha * alpha;
          double lambda = params.get_lambda();
          double lambda2 = 2.0 * lambda * lambda;
          double fMF = exp(kLap / lambda2);
          double k_val = sqrt(-kLap) - 1.0;
          double k2 = k_val * k_val;

          double alpha_farTol = params.get_alpha_farTol();
          double rTol = -alpha2 * log(alpha_farTol) - 1.0;
          double g1 = 0;
          int alpha_highOrd = params.get_alpha_highOrd();
          if (alpha_highOrd == 0) { // gaussian peak
            g1 = exp(-k2 / alpha2);
          } else { // quasi-gaussian peak with higher order component to make it
                   // decay faster towards k=0
            g1 = exp(-(k2 + rTol * pow(k_val, alpha_highOrd)) / alpha2);
          }

          // taylor expansion of gaussian peak to order 2
          double g2 = 1.0 - 1.0 / alpha2 * k2;
          // splice the two sides of the peak
          double gf = (k_val < 0.0) ? g1 : g2;
          // we separate this out because it is needed in the nonlinear
          // calculation when T is not constant in space
          double Bx = params.get_Bx();
          double T = params.get_T();
          double T0 = params.get_T0();
          double opPeak = Bx * exp(-T / T0) * gf;
          // includes the lowest order n_mf term since it is a linear term
          double stabP = params.get_stabP();
          double p2_bar = params.get_p2_bar();
          double q2_bar = params.get_q2_bar();
          double opCk = stabP + p2_bar - opPeak + q2_bar * fMF;

          filterMF[idx] = fMF;
          opL[idx] = exp(kLap * opCk * dt);
          opN[idx] = (opCk == 0.0) ? kLap * dt : (opL[idx] - 1.0) / opCk;
          idx += 1;
        }
      }
    }

    CHECK_AND_ABORT_IF_NANS(opL);
    CHECK_AND_ABORT_IF_NANS(opN);
  }

  void initialize(double dt) override {
    allocate();
    prepare_operators(dt);
  }

  void step(double t) override {

    (void)t; // suppress compiler warning about unused parameter

    FFT &fft = get_fft();

    // Calculate mean-field density n_mf
    fft.forward(psi, psi_F);
    for (size_t idx = 0, N = psiMF_F.size(); idx < N; idx++) {
      psiMF_F[idx] = filterMF[idx] * psi_F[idx];
    }
    fft.backward(psiMF_F, psiMF);

    // Calculate the nonlinear part of the evolution equation in a real space
    double p3_bar = params.get_p3_bar();
    double p4_bar = params.get_p4_bar();
    double q3_bar = params.get_q3_bar();
    double q4_bar = params.get_q4_bar();
    for (size_t idx = 0, N = psiN.size(); idx < N; idx++) {
      double u = psi[idx], v = psiMF[idx];
      double u2 = u * u, u3 = u * u * u, v2 = v * v, v3 = v * v * v;
      psiN[idx] = p3_bar * u2 + p4_bar * u3 + q3_bar * v2 + q4_bar * v3;
    }

    // Apply stabilization factor if given in parameters
    double stabP = params.get_stabP();
    if (stabP != 0.0)
      for (size_t idx = 0, N = psiN.size(); idx < N; idx++) {
        psiN[idx] = psiN[idx] - stabP * psi[idx];
      }

    // Fourier transform of the nonlinear part of the evolution equation
    fft.forward(psiN, psiN_F);

    // Apply one step of the evolution equation
    for (size_t idx = 0, N = psi_F.size(); idx < N; idx++) {
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];
    }

    // Inverse Fourier transform result back to real space
    fft.backward(psi_F, psi);

    // Check does psi has any NaNs and abort the calculation if NaNs are
    // detected. This macro is enabled with compile option 'NAN_CHECK_ENABLED',
    // which is enabled when build type is 'Debug', i.e.
    // -DCMAKE_BUILD_TYPE=Debug. In normal production mode, use
    // -DCMAKE_BUILD_TYPE=Release, which turns on all the optimizations and
    // disables NaN checks and other debug mode checks which may cause any
    // overhead to the actual simulation.
    CHECK_AND_ABORT_IF_NANS(psi);
  }

  /**
   * @brief Constructs a Tungsten model with the given World object.
   *
   * @param world The World object to initialize the model.
   */
  explicit Tungsten(const World &world) : Model(world) {
    // Additional initialization if needed
  }

}; // end of class

#endif // TUNGSTEN_MODEL_HPP
