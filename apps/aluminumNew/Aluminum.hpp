/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#ifndef ALUMINUM_HPP
#define ALUMINUM_HPP

#include "SeedGridFCC.hpp"
#include <openpfc/openpfc.hpp>

using namespace pfc;
using namespace pfc::ui;

/**
 * @brief Aluminum model class for OpenPFC. This class is used to define the model.
 *
 */
class Aluminum : public Model {

private:
  std::vector<double> filterMF, opL, opN, opEps, P_F;
  std::vector<double> psiMF, psi, psiN, P_star_psi, temperature, stress;
  std::vector<std::complex<double>> psiMF_F, psi_F, psiN_F, P_psi_F, temperature_F, stress_F;
  size_t mem_allocated = 0;
  bool m_first = true;

public:
  /**
   * @brief Model parameters, which can be overridden from json file
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
    double T_const, T_max, T_min;
    double G_grid, V_grid, x_initial;
    double m_xpos;
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
    double tau_const;
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
    double q2_bar_L;
  } params;

  // setters
  void set_n0(double n0) { params.n0 = n0; }
  void set_n_sol(double n_sol) { params.n_sol = n_sol; }
  void set_n_vap(double n_vap) { params.n_vap = n_vap; }
  void set_T(double T) { params.T = T; }
  void set_T0(double T0) { params.T0 = T0; }
  void set_Bx(double Bx) { params.Bx = Bx; }
  void set_T_const(double T_const) { params.T_const = T_const; }
  void set_T_max(double T_max) { params.T_max = T_max; }
  void set_T_min(double T_min) { params.T_min = T_min; }
  void set_G_grid(double G_grid) { params.G_grid = G_grid; }
  void set_V_grid(double V_grid) { params.V_grid = V_grid; }
  void set_x_initial(double x_initial) { params.x_initial = x_initial; }
  void set_m_xpos(double m_xpos) { params.m_xpos = m_xpos; }
  void set_alpha(double alpha) { params.alpha = alpha; }
  void set_alpha_farTol(double alpha_farTol) { params.alpha_farTol = alpha_farTol; }
  void set_alpha_highOrd(int alpha_highOrd) { params.alpha_highOrd = alpha_highOrd; }
  void set_tau_const(double tau_const) { params.tau_const = tau_const; }
  void set_lambda(double lambda) { params.lambda = lambda; }
  void set_stabP(double stabP) { params.stabP = stabP; }
  void set_shift_u(double shift_u) { params.shift_u = shift_u; }
  void set_shift_s(double shift_s) { params.shift_s = shift_s; }
  void set_p2(double p2) { params.p2 = p2; }
  void set_p3(double p3) { params.p3 = p3; }
  void set_p4(double p4) { params.p4 = p4; }
  void set_p2_bar(double p2_bar) { params.p2_bar = p2_bar; }
  void set_p3_bar(double p3_bar) { params.p3_bar = p3_bar; }
  void set_p4_bar(double p4_bar) { params.p4_bar = p4_bar; }
  void set_q20(double q20) { params.q20 = q20; }
  void set_q21(double q21) { params.q21 = q21; }
  void set_q30(double q30) { params.q30 = q30; }
  void set_q31(double q31) { params.q31 = q31; }
  void set_q40(double q40) { params.q40 = q40; }
  void set_q20_bar(double q20_bar) { params.q20_bar = q20_bar; }
  void set_q21_bar(double q21_bar) { params.q21_bar = q21_bar; }
  void set_q30_bar(double q30_bar) { params.q30_bar = q30_bar; }
  void set_q31_bar(double q31_bar) { params.q31_bar = q31_bar; }
  void set_q40_bar(double q40_bar) { params.q40_bar = q40_bar; }
  void set_q2_bar(double q2_bar) { params.q2_bar = q2_bar; }
  void set_q3_bar(double q3_bar) { params.q3_bar = q3_bar; }
  void set_q4_bar(double q4_bar) { params.q4_bar = q4_bar; }
  void set_q2_bar_L(double q2_bar_L) { params.q2_bar_L = q2_bar_L; }

  void allocate() {
    FFT &fft = get_fft();
    auto size_inbox = fft.size_inbox();
    auto size_outbox = fft.size_outbox();

    // operators are only half size due to the symmetry of fourier space
    filterMF.resize(size_outbox);
    opL.resize(size_outbox);
    opN.resize(size_outbox);
    opEps.resize(size_outbox);
    P_F.resize(size_outbox);

    // psi, psiMF, psiN
    psi.resize(size_inbox);
    psiMF.resize(size_inbox);
    psiN.resize(size_inbox);
    P_star_psi.resize(size_inbox);
    temperature.resize(size_inbox);
    stress.resize(size_inbox);

    // psi_F, psiMF_F, psiN_F, where suffix F means in fourier space
    psi_F.resize(size_outbox);
    psiMF_F.resize(size_outbox);
    psiN_F.resize(size_outbox);
    P_psi_F.resize(size_outbox);
    stress_F.resize(size_outbox);

    add_real_field("psi", psi);
    add_real_field("default", psi); // for backward compatibility
    add_real_field("psiMF", psiMF);
    add_real_field("psiN", psiN);
    add_real_field("P_star_psi", P_star_psi);
    add_real_field("temperature", temperature);
    add_real_field("stress", stress);

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
    mem_allocated += utils::sizeof_vec(P_F);
    mem_allocated += utils::sizeof_vec(P_psi_F);
    mem_allocated += utils::sizeof_vec(P_star_psi);
    mem_allocated += utils::sizeof_vec(temperature);
    mem_allocated += utils::sizeof_vec(stress);
    mem_allocated += utils::sizeof_vec(stress_F);
  }

  void prepare_operators(double dt) {
    World w = get_world();
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto Lx = w.Lx;
    auto Ly = w.Ly;
    auto Lz = w.Lz;

    const Decomposition &decomp = get_decomposition();
    std::array<int, 3> low = decomp.outbox.low;
    std::array<int, 3> high = decomp.outbox.high;

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
          double k = sqrt(-kLap) - 1.0;
          double k2 = k * k;

          double kp = sqrt(-kLap) - 2.0 / sqrt(3.0);
          double kp2 = kp * kp;

          double g1 = exp(-k2 / alpha2);
          double gp1 = exp(-kp2 / alpha2);
          double peak = (g1 > gp1) ? g1 : gp1;

          P_F[idx] = params.Bx * exp(-params.tau_const) * peak;

          double opCk = params.stabP + params.p2_bar - P_F[idx] + params.q2_bar_L * fMF;

          filterMF[idx] = fMF;
          opL[idx] = exp(kLap * opCk * dt);
          opN[idx] = (opCk == 0.0) ? kLap * dt : (opL[idx] - 1.0) / opCk;

          double alpha2new = 2.0 * params.alpha * params.alpha / 10.0;
          double g1new = exp(-k2 / alpha2new);
          double gp1new = exp(-kp2 / alpha2new);

          double peaknew = (g1new > gp1new) ? g1new : gp1new;

          opEps[idx] = peaknew;

          idx += 1;
        }
      }
    }
  }

  void initialize(double dt) override {
    allocate();
    prepare_operators(dt);
  }

  void step(double t) override {

    FFT &fft = get_fft();
    World w = get_world();
    double dx = w.dx;
    double x0 = w.x0;
    int Lx = w.Lx;
    const Decomposition &decomp = get_decomposition();
    std::array<int, 3> low = decomp.inbox.low;
    std::array<int, 3> high = decomp.inbox.high;

    // Calculate mean-field density n_mf
    fft.forward(psi, psi_F);
    for (size_t idx = 0, N = psiMF_F.size(); idx < N; idx++) {
      psiMF_F[idx] = filterMF[idx] * psi_F[idx];
    }
    fft.backward(psiMF_F, psiMF);

    for (size_t idx = 0, N = P_psi_F.size(); idx < N; idx++) {
      P_psi_F[idx] = P_F[idx] * psi_F[idx];
    }
    fft.backward(P_psi_F, P_star_psi);

    double l = Lx * dx;
    // double xpos = fmod(params.m_xpos, l);
    double fullruns = floor(params.m_xpos / l) * l;
    double steppoint = fmod(params.m_xpos, l);
    double local_FE = 0;

    size_t idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = x0 + i * dx;
          double dist = x + fullruns - (x > steppoint) * l;
          double T_var = params.G_grid * (dist - params.x_initial - params.V_grid * t);
          temperature[idx] = T_var;
          double q2_bar_N = params.q21_bar * T_var / params.T0;
          double q3_bar = params.q31_bar * (params.T_const + T_var) / params.T0 + params.q30_bar;
          double u = psi[idx];
          double v = psiMF[idx];
          double kernel_term_N = -(1.0 - exp(-T_var / params.T0)) * P_star_psi[idx];
          psiN[idx] = params.p3_bar * u * u + params.p4_bar * u * u * u + q2_bar_N * v + q3_bar * v * v +
                      params.q4_bar * v * v * v - kernel_term_N;
          local_FE += params.p3_bar * u * u * u / 3. + params.p4_bar * u * u * u * u / 4. + q2_bar_N * u * v / 2. +
                      q3_bar * u * v * v / 3. + params.q4_bar * u * v * v * v / 4. + -u * kernel_term_N * u / 2. +
                      -u * P_star_psi[idx] / 2. + params.p2_bar * u * u / 2. + params.q2_bar * u * v / 2.;
          idx++;
        }
      }
    }

    // Apply stabilization factor if given in parameters
    if (params.stabP != 0.0) {
      for (size_t idx = 0, N = psiN.size(); idx < N; idx++) {
        psiN[idx] = psiN[idx] - params.stabP * psi[idx];
      }
    }

    // Fourier transform of the nonlinear part of the evolution equation
    fft.forward(psiN, psiN_F);

    // Apply one step of the evolution equation
    for (size_t idx = 0, N = psi_F.size(); idx < N; idx++) {
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];
    }

    // Inverse Fourier transform result back to real space
    fft.backward(psi_F, psi);
  }

}; // end of class

/**
 * @brief Validate model configuration from json file.
 *
 * @param j json file
 */
void validate(const json &j) {
  if (!j.contains("n0") || !j.at("n0").is_number()) {
    throw std::runtime_error("Missing or invalid n0");
  }
  if (!j.contains("n_sol") || !j.at("n_sol").is_number()) {
    throw std::runtime_error("Missing or invalid n_sol");
  }
  if (!j.contains("n_vap") || !j.at("n_vap").is_number()) {
    throw std::runtime_error("Missing or invalid n_vap");
  }
  if (!j.contains("T0") || !j.at("T0").is_number()) {
    throw std::runtime_error("Missing or invalid T0");
  }
  if (!j.contains("Bx") || !j.at("Bx").is_number()) {
    throw std::runtime_error("Missing or invalid Bx");
  }
  if (!j.contains("G_grid") || !j.at("G_grid").is_number()) {
    throw std::runtime_error("Missing or invalid G_grid");
  }
  if (!j.contains("V_grid") || !j.at("V_grid").is_number()) {
    throw std::runtime_error("Missing or invalid V_grid");
  }
  if (!j.contains("x_initial") || !j.at("x_initial").is_number()) {
    throw std::runtime_error("Missing or invalid x_initial");
  }
  if (!j.contains("T_const") || !j.at("T_const").is_number()) {
    throw std::runtime_error("Missing or invalid T_const");
  }
  if (!j.contains("T_max") || !j.at("T_max").is_number()) {
    throw std::runtime_error("Missing or invalid T_max");
  }
  if (!j.contains("T_min") || !j.at("T_min").is_number()) {
    throw std::runtime_error("Missing or invalid T_min");
  }
  if (!j.contains("alpha") || !j.at("alpha").is_number()) {
    throw std::runtime_error("Missing or invalid alpha");
  }
  if (!j.contains("alpha_farTol") || !j.at("alpha_farTol").is_number()) {
    throw std::runtime_error("Missing or invalid alpha_farTol");
  }
  if (!j.contains("alpha_highOrd") || !j.at("alpha_highOrd").is_number()) {
    throw std::runtime_error("Missing or invalid alpha_highOrd");
  }
  if (!j.contains("lambda") || !j.at("lambda").is_number()) {
    throw std::runtime_error("Missing or invalid lambda");
  }
  if (!j.contains("stabP") || !j.at("stabP").is_number()) {
    throw std::runtime_error("Missing or invalid stabP");
  }
  if (!j.contains("shift_u") || !j.at("shift_u").is_number()) {
    throw std::runtime_error("Missing or invalid shift_u");
  }
  if (!j.contains("shift_s") || !j.at("shift_s").is_number()) {
    throw std::runtime_error("Missing or invalid shift_s");
  }
  if (!j.contains("p2_bar") || !j.at("p2_bar").is_number()) {
    throw std::runtime_error("Missing or invalid p2_bar");
  }
  if (!j.contains("p3_bar") || !j.at("p3_bar").is_number()) {
    throw std::runtime_error("Missing or invalid p3_bar");
  }
  if (!j.contains("p4_bar") || !j.at("p4_bar").is_number()) {
    throw std::runtime_error("Missing or invalid p4_bar");
  }
  if (!j.contains("q20_bar") || !j.at("q20_bar").is_number()) {
    throw std::runtime_error("Missing or invalid q20_bar");
  }
  if (!j.contains("q21_bar") || !j.at("q21_bar").is_number()) {
    throw std::runtime_error("Missing or invalid q21_bar");
  }
  if (!j.contains("q30_bar") || !j.at("q30_bar").is_number()) {
    throw std::runtime_error("Missing or invalid q30_bar");
  }
  if (!j.contains("q31_bar") || !j.at("q31_bar").is_number()) {
    throw std::runtime_error("Missing or invalid q31_bar");
  }
  if (!j.contains("q40_bar") || !j.at("q40_bar").is_number()) {
    throw std::runtime_error("Missing or invalid q40_bar");
  }
}

/**
 * @brief Read model configuration from json file, under model/params.
 *
 * @param j json file
 * @param m model
 */
void from_json(const json &j, Aluminum &m) {
  validate(j);
  auto &p = m.params;
  j.at("n0").get_to(p.n0);
  j.at("n_sol").get_to(p.n_sol);
  j.at("n_vap").get_to(p.n_vap);
  j.at("T0").get_to(p.T0);
  j.at("Bx").get_to(p.Bx);
  j.at("G_grid").get_to(p.G_grid);
  j.at("V_grid").get_to(p.V_grid);
  j.at("x_initial").get_to(p.x_initial);
  p.m_xpos = p.x_initial;
  j.at("T_const").get_to(p.T_const);
  j.at("T_max").get_to(p.T_max);
  j.at("T_min").get_to(p.T_min);
  j.at("alpha").get_to(p.alpha);
  j.at("alpha_farTol").get_to(p.alpha_farTol);
  j.at("alpha_highOrd").get_to(p.alpha_highOrd);
  p.tau_const = p.T_const / p.T0;
  j.at("lambda").get_to(p.lambda);
  j.at("stabP").get_to(p.stabP);
  j.at("shift_u").get_to(p.shift_u);
  j.at("shift_s").get_to(p.shift_s);
  j.at("p2_bar").get_to(p.p2_bar);
  j.at("p3_bar").get_to(p.p3_bar);
  j.at("p4_bar").get_to(p.p4_bar);
  /*
    p.p2_bar = p.p2 + 2 * p.shift_s * p.p3 + 3 * pow(p.shift_s, 2) * p.p4;
    p.p3_bar = p.shift_u * (p.p3 + 3 * p.shift_s * p.p4);
    p.p4_bar = pow(p.shift_u, 2) * p.p4;
  */
  j.at("q20_bar").get_to(p.q20_bar);
  j.at("q21_bar").get_to(p.q21_bar);
  j.at("q30_bar").get_to(p.q30_bar);
  j.at("q31_bar").get_to(p.q31_bar);
  j.at("q40_bar").get_to(p.q40_bar);
  /*
    p.q20_bar = p.q20 + 2.0 * p.shift_s * p.q30 + 3.0 * pow(p.shift_s, 2) * p.q40;
    p.q21_bar = p.q21 + 2.0 * p.shift_s * p.q31;
    p.q30_bar = p.shift_u * (p.q30 + 3.0 * p.shift_s * p.q40);
    p.q31_bar = p.shift_u * p.q31;
    p.q40_bar = pow(p.shift_u, 2) * p.q40;
  */
  p.q2_bar = p.q21_bar * p.tau_const + p.q20_bar;
  p.q2_bar_L = p.q2_bar;
  p.q3_bar = p.q31_bar * p.tau_const + p.q30_bar;
  p.q4_bar = p.q40_bar;
}

#endif // ALUMINUM_HPP
