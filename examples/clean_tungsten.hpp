#include <pfc/model.hpp>

using namespace pfc;

/*
Model parameters
*/
struct params {
  // average density of the metastable fluid
  double n0 = -0.4;

  // Bulk densities at coexistence, obtained from phase diagram for chosen
  // temperature
  double n_sol = -0.047;
  double n_vap = -0.464;

  // Effective temperature parameters. Temperature in K. Remember to change
  // n_sol and n_vap according to phase diagram when T is changed.
  double T = 3300.0;
  double T0 = 156000.0;
  double Bx = 0.8582;

  // parameters that affect elastic and interface energies

  // width of C2's peak
  double alpha = 0.50;

  // how much we allow the k=1 peak to affect the k=0 value of the
  // correlation, by changing the higher order components of the Gaussian
  // function
  double alpha_farTol = 1.0 / 1000.0;

  // power of the higher order component of the gaussian function. Should be a
  // multiple of 2. Setting this to zero also disables the tolerance setting.
  int alpha_highOrd = 4;

  // derived dimensionless values used in calculating vapor model parameters
  double tau = T / T0;

  // Strength of the meanfield filter. Avoid values higher than ~0.28, to
  // avoid lattice-wavelength variations in the mean field
  double lambda = 0.22;

  // numerical stability parameter for the exponential integrator method
  double stabP = 0.2;

  // Vapor-model parameters
  double shift_u = 0.3341;
  double shift_s = 0.1898;

  double p2 = 1.0;
  double p3 = -1.0 / 2.0;
  double p4 = 1.0 / 3.0;
  double p2_bar = p2 + 2 * shift_s * p3 + 3 * pow(shift_s, 2) * p4;
  double p3_bar = shift_u * (p3 + 3 * shift_s * p4);
  double p4_bar = pow(shift_u, 2) * p4;

  double q20 = -0.0037;
  double q21 = 1.0;
  double q30 = -12.4567;
  double q31 = 20.0;
  double q40 = 45.0;

  double q20_bar = q20 + 2.0 * shift_s * q30 + 3.0 * pow(shift_s, 2) * q40;
  double q21_bar = q21 + 2.0 * shift_s * q31;
  double q30_bar = shift_u * (q30 + 3.0 * shift_s * q40);
  double q31_bar = shift_u * q31;
  double q40_bar = pow(shift_u, 2) * q40;

  double q2_bar = q21_bar * tau + q20_bar;
  double q3_bar = q31_bar * tau + q30_bar;
  double q4_bar = q40_bar;
};

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
  std::array<double, 10> timing = {0};
  size_t mem_allocated = 0;
  bool m_first = true;
  params p;

public:
  void allocate(size_t size_inbox, size_t size_outbox) {

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

    // At the end, let's calculate how much did we allocate memory, should be
    // size_inbox*8*3     (psi, psiMF, psiN) are double vectors
    // size_outbox*16*3   (psi_F, psiMF_F, psiN_F) are complex vectors
    // size_outbox*8*3    (filterMF, opL, opN) are double vectors
    // size_inbox = Lx * Ly * Lz
    // size_outbox = (floor(Lx/2) + 1) * Ly * Lz

    mem_allocated =
        size_inbox * (8 * (3 - 1)) + size_outbox * (16 * (3 - 1) + 8 * 3);
  }

  /*
    This function is called after allocate(), used to fill operators.
  */
  void prepare_operators(double dt) {

    World w = get_world();
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto Lx = w.Lx;
    auto Ly = w.Ly;
    auto Lz = w.Lz;

    Decomposition &decomp = get_decomposition();
    std::array<int, 3> low = decomp.outbox.low;
    std::array<int, 3> high = decomp.outbox.high;

    // prepare the linear and non-linear operators

    int idx = 0;
    const double pi = std::atan(1.0) * 4.0;
    const double fx = 2.0 * pi / (dx * Lx);
    const double fy = 2.0 * pi / (dy * Ly);
    const double fz = 2.0 * pi / (dz * Lz);

    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {

          // laplacian operator -k^2
          const double ki = (i <= Lx / 2) ? i * fx : (i - Lx) * fx;
          const double kj = (j <= Ly / 2) ? j * fy : (j - Ly) * fy;
          const double kk = (k <= Lz / 2) ? k * fz : (k - Lz) * fz;
          const double kLap = -(ki * ki + kj * kj + kk * kk);

          // mean-field filtering operator (chi) make a C2 that's quasi-gaussian
          // on the left, and ken-style on the right
          const double alpha2 = 2.0 * p.alpha * p.alpha;
          const double lambda2 = 2.0 * p.lambda * p.lambda;
          const double fMF = exp(kLap / lambda2);
          const double k = sqrt(-kLap) - 1.0;
          const double k2 = k * k;

          double g1 = 0;
          if (p.alpha_highOrd == 0) {
            // gaussian peak
            g1 = exp(-k2 / alpha2);
          } else {
            // quasi-gaussian peak with higher order component
            // to make it decay faster towards k=0
            double rTol = -alpha2 * log(p.alpha_farTol) - 1.0;
            g1 = exp(-(k2 + rTol * pow(k, p.alpha_highOrd)) / alpha2);
          }
          // taylor expansion of gaussian peak to order 2
          const double g2 = 1.0 - 1.0 / alpha2 * k2;
          // splice the two sides of the peak
          const double gf = (k < 0.0) ? g1 : g2;

          // we separate this out because it is needed in the nonlinear
          // calculation when T is not constant in space
          const double opPeak = -p.Bx * exp(-p.T / p.T0) * gf;

          // includes the lowest order n_mf term since it is a linear term
          const double opCk = p.stabP + p.p2_bar + opPeak + p.q2_bar * fMF;

          filterMF[idx] = fMF;
          opL[idx] = exp(kLap * opCk * dt);
          opN[idx] = (opCk == 0.0) ? kLap * dt : (opL[idx] - 1.0) / opCk;

          idx += 1;
        }
      }
    }
  }

  /*
  Initial condition is defined here
  */
  void prepare_initial_condition() {

    World w = get_world();
    Decomposition &decomp = get_decomposition();

    std::array<int, 3> low = decomp.inbox.low;
    std::array<int, 3> high = decomp.inbox.high;
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto x0 = w.x0;
    auto y0 = w.y0;
    auto z0 = w.z0;

    // Calculating approx amplitude. This is related to the phase diagram
    // calculations.
    const double rho_seed = p.n_sol;
    const double A_phi = 135.0 * p.p4_bar;
    const double B_phi = 16.0 * p.p3_bar + 48.0 * p.p4_bar * rho_seed;
    const double C_phi = -6.0 * (p.Bx * exp(-p.T / p.T0)) + 6.0 * p.p2_bar +
                         12.0 * p.p3_bar * rho_seed +
                         18.0 * p.p4_bar * pow(rho_seed, 2);
    const double d = std::abs(9.0 * pow(B_phi, 2) - 32.0 * A_phi * C_phi);
    const double amp_eq = (-3.0 * B_phi + sqrt(d)) / (8.0 * A_phi);

    const double s = 1.0 / sqrt(2.0);
    const std::array<double, 3> q1 = {s, s, 0};
    const std::array<double, 3> q2 = {s, 0, s};
    const std::array<double, 3> q3 = {0, s, s};
    const std::array<double, 3> q4 = {s, 0, -s};
    const std::array<double, 3> q5 = {s, -s, 0};
    const std::array<double, 3> q6 = {0, s, -s};
    const std::array<std::array<double, 3>, 6> q = {q1, q2, q3, q4, q5, q6};

    long int idx = 0;
    // double r2 = pow(0.2 * (Lx * dx), 2);
    const double r2 = pow(64.0, 2);
    double u;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          const double x = x0 + i * dx;
          const double y = y0 + j * dy;
          const double z = z0 + k * dz;
          const bool seedmask = x * x + y * y + z * z < r2;
          if (!seedmask) {
            u = p.n0;
          } else {
            u = rho_seed;
            for (int i = 0; i < 6; i++) {
              u += 2.0 * amp_eq * cos(q[i][0] * x + q[i][1] * y + q[i][2] * z);
            }
          }
          psi[idx] = u;
          idx += 1;
        }
      }
    }
  }

  void initialize(double dt) override {
    FFT &fft = get_fft();
    allocate(fft.size_inbox(), fft.size_outbox());
    prepare_operators(dt);
    prepare_initial_condition();
  }

  /*
    PFC stepping function

    Timing:
    timing[0]   total time used in step
    timing[1]   time used in FFT
    timing[2]   time used in all other things

    size_inbox*8*3     (psi, psiMF, psiN) are double vectors
    size_outbox*16*3   (psi_F, psiMF_F, psiN_F) are complex vectors
    size_outbox*8*3    (filterMF, opL, opN) are double vectors

  */
  void step(double) override {

    FFT &fft = get_fft();

    if (m_first) { // First iteration, calculate psi_F = fft(psi)
      m_first = false;
      fft.forward(psi, psi_F);
    }

    for (int i = 0; i < 8; i++) {
      timing[i] = 0.0;
    }

    timing[0] -= MPI_Wtime();

    // Calculate mean-field density n_mf
    timing[2] -= MPI_Wtime();
    for (long int idx = 0, N = psiMF_F.size(); idx < N; idx++) {
      psiMF_F[idx] = filterMF[idx] * psi_F[idx];
    }
    timing[2] += MPI_Wtime();

    timing[1] -= MPI_Wtime();
    fft.backward(psiMF_F, psiMF);
    timing[1] += MPI_Wtime();

    // Calculate the nonlinear part of the evolution equation in a real space
    timing[2] -= MPI_Wtime();
    for (long int idx = 0, N = psiN.size(); idx < N; idx++) {
      const double u = psi[idx];
      const double v = psiMF[idx];
      const double u2 = u * u;
      const double u3 = u2 * u;
      const double v2 = v * v;
      const double v3 = v2 * v;
      psiN[idx] = p.p3_bar * u2 + p.p4_bar * u3 + p.q3_bar * v2 + p.q4_bar * v3;
    }
    timing[2] += MPI_Wtime();

    // Apply stabilization factor if given in parameters
    timing[2] -= MPI_Wtime();
    if (p.stabP != 0.0) {
      for (long int idx = 0, N = psiN.size(); idx < N; idx++) {
        psiN[idx] = psiN[idx] - p.stabP * psi[idx];
      }
    }
    timing[2] += MPI_Wtime();

    // Fourier transform of the nonlinear part of the evolution equation
    timing[1] -= MPI_Wtime();
    fft.forward(psiN, psiN_F);
    timing[1] += MPI_Wtime();

    // Apply one step of the evolution equation
    timing[2] -= MPI_Wtime();
    for (long int idx = 0, N = psi_F.size(); idx < N; idx++) {
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];
    }
    timing[2] += MPI_Wtime();

    // Inverse Fourier transform result back to real space
    timing[1] -= MPI_Wtime();
    fft.backward(psi_F, psi);
    timing[1] += MPI_Wtime();

    timing[0] += MPI_Wtime();
  }

  std::vector<double> &get_field() {
    return psi;
  }

}; // end of class
