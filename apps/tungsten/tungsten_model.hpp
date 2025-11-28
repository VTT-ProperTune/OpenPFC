// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef TUNGSTEN_MODEL_HPP
#define TUNGSTEN_MODEL_HPP

#include <openpfc/openpfc.hpp>
#include <openpfc/utils/nancheck.hpp>

namespace pfc {
namespace utils {
} // namespace utils
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
   */
  struct {
    // average density of the metastable fluid
    double n0;
    // Bulk densities at coexistence, obtained from phase diagram for chosen
    // temperature
    double n_sol, n_vap;
    // Effective temperature parameters. Temperature in K. Remember to change
    // n_sol and n_vap according to phase diagram when T is changed.
    double T, T0, Bx;
    // width of C2's peak
    double alpha;
    // how much we allow the k=1 peak to affect the k=0 value of the
    // correlation, by changing the higher order components of the Gaussian
    // function
    double alpha_farTol;
    // power of the higher order component of the gaussian function. Should be a
    // multiple of 2. Setting this to zero also disables the tolerance setting.
    int alpha_highOrd;
    // derived dimensionless values used in calculating vapor model parameters
    double tau;
    // Strength of the meanfield filter. Avoid values higher than ~0.28, to
    // avoid lattice-wavelength variations in the mean field
    double lambda;
    // numerical stability parameter for the exponential integrator method
    double stabP;
    // Vapor-model parameters
    double shift_u, shift_s;
    double p2, p3, p4, p2_bar, p3_bar, p4_bar;
    double q20, q21, q30, q31, q40;
    double q20_bar, q21_bar, q30_bar, q31_bar, q40_bar, q2_bar, q3_bar, q4_bar;
  } params;

  // Setters for parameters (useful for testing)
  void set_n0(double n0) { params.n0 = n0; }
  void set_n_sol(double n_sol) { params.n_sol = n_sol; }
  void set_n_vap(double n_vap) { params.n_vap = n_vap; }
  void set_T(double T) {
    params.T = T;
    params.tau = params.T / params.T0;
    // Recalculate derived parameters
    params.q2_bar = params.q21_bar * params.tau + params.q20_bar;
    params.q3_bar = params.q31_bar * params.tau + params.q30_bar;
  }
  void set_T0(double T0) {
    params.T0 = T0;
    params.tau = params.T / params.T0;
    // Recalculate derived parameters
    params.q2_bar = params.q21_bar * params.tau + params.q20_bar;
    params.q3_bar = params.q31_bar * params.tau + params.q30_bar;
  }
  void set_Bx(double Bx) { params.Bx = Bx; }
  void set_alpha(double alpha) { params.alpha = alpha; }
  void set_alpha_farTol(double alpha_farTol) { params.alpha_farTol = alpha_farTol; }
  void set_alpha_highOrd(int alpha_highOrd) { params.alpha_highOrd = alpha_highOrd; }
  void set_lambda(double lambda) { params.lambda = lambda; }
  void set_stabP(double stabP) { params.stabP = stabP; }
  void set_shift_u(double shift_u) {
    params.shift_u = shift_u;
    // Recalculate derived parameters
    params.p3_bar = params.shift_u * (params.p3 + 3 * params.shift_s * params.p4);
    params.p4_bar = pow(params.shift_u, 2) * params.p4;
    params.q30_bar =
        params.shift_u * (params.q30 + 3.0 * params.shift_s * params.q40);
    params.q31_bar = params.shift_u * params.q31;
    params.q40_bar = pow(params.shift_u, 2) * params.q40;
    params.q2_bar = params.q21_bar * params.tau + params.q20_bar;
    params.q3_bar = params.q31_bar * params.tau + params.q30_bar;
    params.q4_bar = params.q40_bar;
  }
  void set_shift_s(double shift_s) {
    params.shift_s = shift_s;
    // Recalculate derived parameters
    params.p2_bar = params.p2 + 2 * params.shift_s * params.p3 +
                    3 * pow(params.shift_s, 2) * params.p4;
    params.p3_bar = params.shift_u * (params.p3 + 3 * params.shift_s * params.p4);
    params.q20_bar = params.q20 + 2.0 * params.shift_s * params.q30 +
                     3.0 * pow(params.shift_s, 2) * params.q40;
    params.q21_bar = params.q21 + 2.0 * params.shift_s * params.q31;
    params.q30_bar =
        params.shift_u * (params.q30 + 3.0 * params.shift_s * params.q40);
    params.q2_bar = params.q21_bar * params.tau + params.q20_bar;
    params.q3_bar = params.q31_bar * params.tau + params.q30_bar;
  }
  void set_p2(double p2) {
    params.p2 = p2;
    params.p2_bar = params.p2 + 2 * params.shift_s * params.p3 +
                    3 * pow(params.shift_s, 2) * params.p4;
  }
  void set_p3(double p3) {
    params.p3 = p3;
    params.p2_bar = params.p2 + 2 * params.shift_s * params.p3 +
                    3 * pow(params.shift_s, 2) * params.p4;
    params.p3_bar = params.shift_u * (params.p3 + 3 * params.shift_s * params.p4);
  }
  void set_p4(double p4) {
    params.p4 = p4;
    params.p2_bar = params.p2 + 2 * params.shift_s * params.p3 +
                    3 * pow(params.shift_s, 2) * params.p4;
    params.p3_bar = params.shift_u * (params.p3 + 3 * params.shift_s * params.p4);
    params.p4_bar = pow(params.shift_u, 2) * params.p4;
  }
  void set_q20(double q20) {
    params.q20 = q20;
    params.q20_bar = params.q20 + 2.0 * params.shift_s * params.q30 +
                     3.0 * pow(params.shift_s, 2) * params.q40;
    params.q2_bar = params.q21_bar * params.tau + params.q20_bar;
  }
  void set_q21(double q21) {
    params.q21 = q21;
    params.q21_bar = params.q21 + 2.0 * params.shift_s * params.q31;
    params.q2_bar = params.q21_bar * params.tau + params.q20_bar;
  }
  void set_q30(double q30) {
    params.q30 = q30;
    params.q20_bar = params.q20 + 2.0 * params.shift_s * params.q30 +
                     3.0 * pow(params.shift_s, 2) * params.q40;
    params.q30_bar =
        params.shift_u * (params.q30 + 3.0 * params.shift_s * params.q40);
    params.q3_bar = params.q31_bar * params.tau + params.q30_bar;
  }
  void set_q31(double q31) {
    params.q31 = q31;
    params.q21_bar = params.q21 + 2.0 * params.shift_s * params.q31;
    params.q31_bar = params.shift_u * params.q31;
    params.q2_bar = params.q21_bar * params.tau + params.q20_bar;
    params.q3_bar = params.q31_bar * params.tau + params.q30_bar;
  }
  void set_q40(double q40) {
    params.q40 = q40;
    params.q20_bar = params.q20 + 2.0 * params.shift_s * params.q30 +
                     3.0 * pow(params.shift_s, 2) * params.q40;
    params.q30_bar =
        params.shift_u * (params.q30 + 3.0 * params.shift_s * params.q40);
    params.q40_bar = pow(params.shift_u, 2) * params.q40;
    params.q3_bar = params.q31_bar * params.tau + params.q30_bar;
    params.q4_bar = params.q40_bar;
  }
  // Setters for derived parameters (for direct testing)
  void set_p2_bar(double p2_bar) { params.p2_bar = p2_bar; }
  void set_p3_bar(double p3_bar) { params.p3_bar = p3_bar; }
  void set_p4_bar(double p4_bar) { params.p4_bar = p4_bar; }
  void set_q20_bar(double q20_bar) {
    params.q20_bar = q20_bar;
    params.q2_bar = params.q21_bar * params.tau + params.q20_bar;
  }
  void set_q21_bar(double q21_bar) {
    params.q21_bar = q21_bar;
    params.q2_bar = params.q21_bar * params.tau + params.q20_bar;
  }
  void set_q30_bar(double q30_bar) {
    params.q30_bar = q30_bar;
    params.q3_bar = params.q31_bar * params.tau + params.q30_bar;
  }
  void set_q31_bar(double q31_bar) {
    params.q31_bar = q31_bar;
    params.q2_bar = params.q21_bar * params.tau + params.q20_bar;
    params.q3_bar = params.q31_bar * params.tau + params.q30_bar;
  }
  void set_q40_bar(double q40_bar) {
    params.q40_bar = q40_bar;
    params.q4_bar = params.q40_bar;
  }
  void set_q2_bar(double q2_bar) { params.q2_bar = q2_bar; }
  void set_q3_bar(double q3_bar) { params.q3_bar = q3_bar; }
  void set_q4_bar(double q4_bar) { params.q4_bar = q4_bar; }

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
          double alpha2 = 2.0 * params.alpha * params.alpha;
          double lambda2 = 2.0 * params.lambda * params.lambda;
          double fMF = exp(kLap / lambda2);
          double k_val = sqrt(-kLap) - 1.0;
          double k2 = k_val * k_val;

          double rTol = -alpha2 * log(params.alpha_farTol) - 1.0;
          double g1 = 0;
          if (params.alpha_highOrd == 0) { // gaussian peak
            g1 = exp(-k2 / alpha2);
          } else { // quasi-gaussian peak with higher order component to make it
                   // decay faster towards k=0
            g1 = exp(-(k2 + rTol * pow(k_val, params.alpha_highOrd)) / alpha2);
          }

          // taylor expansion of gaussian peak to order 2
          double g2 = 1.0 - 1.0 / alpha2 * k2;
          // splice the two sides of the peak
          double gf = (k_val < 0.0) ? g1 : g2;
          // we separate this out because it is needed in the nonlinear
          // calculation when T is not constant in space
          double opPeak = params.Bx * exp(-params.T / params.T0) * gf;
          // includes the lowest order n_mf term since it is a linear term
          double opCk = params.stabP + params.p2_bar - opPeak + params.q2_bar * fMF;

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
    for (size_t idx = 0, N = psiN.size(); idx < N; idx++) {
      double u = psi[idx], v = psiMF[idx];
      double u2 = u * u, u3 = u * u * u, v2 = v * v, v3 = v * v * v;
      double p3 = params.p3_bar, p4 = params.p4_bar;
      double q3 = params.q3_bar, q4 = params.q4_bar;
      psiN[idx] = p3 * u2 + p4 * u3 + q3 * v2 + q4 * v3;
    }

    // Apply stabilization factor if given in parameters
    if (params.stabP != 0.0)
      for (size_t idx = 0, N = psiN.size(); idx < N; idx++) {
        psiN[idx] = psiN[idx] - params.stabP * psi[idx];
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

