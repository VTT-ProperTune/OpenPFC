// heFFTe implementation of pfc code

#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>
#include <heffte.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

const double pi = std::atan(1.0) * 4.0;

struct Simulation {

  int Lx;
  int Ly;
  int Lz;
  double dx;
  double dy;
  double dz;
  double x0;
  double y0;
  double z0;
  double t0;
  double t1;
  double dt;
  unsigned long int max_iters;

  std::vector<double> k2_;             // "Laplace" operator
  std::vector<double> L_;              // Linear part
  std::vector<double> u;               // Field in real space
  std::vector<double> psi;             // Field in real space
  std::vector<std::complex<double>> U; // Field in Fourier space
  std::vector<std::complex<double>> N; // Nonlinear part of u in Fourier space

  std::filesystem::path results_dir = ".";

  std::string exit_msg;

  Simulation() {
    set_domain({-64.0, -64.0, -64.0}, {1.0, 1.0, 1.0}, {128, 128, 128});
    set_time(0.0, 100.0, 1.0);
    max_iters = ULONG_MAX;
    exit_msg = "";
    create_params();
  }

  virtual void create_params() {
  }

  void set_domain(std::array<double, 3> o, std::array<double, 3> h,
                  std::array<int, 3> d) {
    x0 = o[0];
    y0 = o[1];
    z0 = o[2];
    dx = h[0];
    dy = h[1];
    dz = h[2];
    Lx = d[0];
    Ly = d[1];
    Lz = d[2];
  }

  void set_time(double t0_, double t1_, double dt_) {
    t0 = t0_;
    t1 = t1_;
    dt = dt_;
  }

  double get_dt() {
    return dt;
  }

  void set_dt(double dt_) {
    dt = dt_;
  }

  void set_max_iters(unsigned long int nmax) {
    max_iters = nmax;
  }

  void set_results_dir(std::string path) {
    results_dir = path;
  }

  std::filesystem::path get_results_dir() {
    return results_dir;
  }

  virtual bool done(unsigned long int n, double t) {
    if (n > max_iters) {
      exit_msg = "maximum number of iterations (" + std::to_string(max_iters) +
                 ") reached";
      return true;
    }
    if (t >= t1) {
      exit_msg = "simulated succesfully to time " + std::to_string(t1) + ", (" +
                 std::to_string(n) + " iterations)";
      return true;
    }
    return false;
  }

  double k2(double x, double y, double z) {
    auto fx = 2.0 * pi / (dx * Lx);
    auto fy = 2.0 * pi / (dy * Ly);
    auto fz = 2.0 * pi / (dz * Lz);
    auto kx = x < Lx / 2.0 ? x * fx : (x - Lx) * fx;
    auto ky = y < Ly / 2.0 ? y * fy : (y - Ly) * fy;
    auto kz = z < Lz / 2.0 ? z * fz : (z - Lz) * fz;
    return kx * kx + ky * ky + kz * kz;
  }

  virtual unsigned long int allocate(unsigned long int size_inbox,
                                     unsigned long int size_outbox,
                                     unsigned long int size_workspace) {
    unsigned long int size = 0;
    k2_.resize(size_outbox);
    L_.resize(size_outbox);
    u.resize(size_inbox);
    U.resize(size_outbox);
    N.resize(size_outbox);
    size += sizeof(double) * k2_.size();
    size += sizeof(double) * L_.size();
    size += sizeof(double) * u.size();
    size += sizeof(std::complex<double>) * U.size();
    size += sizeof(std::complex<double>) * N.size();
    return size;
  }

  virtual void fill_k2(std::array<int, 3> low, std::array<int, 3> high) {
    unsigned long int idx = 0;
    for (auto z = low[2]; z <= high[2]; z++) {
      for (auto y = low[1]; y <= high[1]; y++) {
        for (auto x = low[0]; x <= high[0]; x++) {
          k2_[idx] = k2(x, y, z);
          idx += 1;
        }
      }
    }
  }

  virtual void fill_L(std::array<int, 3> low, std::array<int, 3> high) {
    unsigned long int idx = 0;
    for (auto k = low[2]; k <= high[2]; k++) {
      for (auto j = low[1]; j <= high[1]; j++) {
        for (auto i = low[0]; i <= high[0]; i++) {
          L_[idx] = L(i, j, k);
          idx += 1;
        }
      }
    }
  }

  virtual void fill_u0(std::array<int, 3> low, std::array<int, 3> high) {
    unsigned long int idx = 0;
    for (auto k = low[2]; k <= high[2]; k++) {
      for (auto j = low[1]; j <= high[1]; j++) {
        for (auto i = low[0]; i <= high[0]; i++) {
          u[idx] = u0(x0 + i * dx, y0 + j * dy, z0 + k * dz);
          idx += 1;
        }
      }
    }
  }

  virtual void calculate_nonlinear_part() {
    for (unsigned long int i = 0; i < u.size(); i++) {
      u[i] = f(u[i]);
    }
  }

  virtual void integrate() {
    for (auto i = 0; i < U.size(); i++) {
      U[i] = 1.0 / (1.0 - dt * L_[i]) * (U[i] - k2_[i] * dt * N[i]);
    };
  }

  virtual void finalize_step(unsigned long int n, double t) {
  }

  virtual void finalize_master_step(unsigned long int n, double t) {
    if (ceil(t / t1 * 100.0) != ceil((t - dt) / t1 * 100.0)) {
      std::cout << "n = " << n << ", t = " << t << ", dt = " << dt << ", "
                << ceil(t / t1 * 100.0) << " percent done" << std::endl;
    }
  }

  virtual void tune_dt(unsigned long int n, double t) {
  }

  // in practice we are interested of replacing the things below with our
  // owns...

  virtual bool writeat(unsigned long int n, double t) {
    return true;
  }

  virtual std::filesystem::path get_result_file_name(unsigned long int n,
                                                     double t) {
    std::filesystem::path filename = "u" + std::to_string(n) + ".bin";
    return get_results_dir() / filename;
  }

  virtual double u0(double x, double y, double z) {
    return exp(-x * x / Lx) * exp(-y * y / Ly) * exp(-z * z / Lz);
  }

  virtual double L(double x, double y, double z) {
    return -k2(x, y, z);
  }

  virtual double f(double u) {
    return 0.0;
  }

  virtual void prepare_operators(std::array<int, 3> low,
                                 std::array<int, 3> high) {
    return;
  }

  virtual void step(unsigned long int n, double t,
                    heffte::fft3d_r2c<heffte::backend::fftw> &fft) {
    return;
  }
};

struct Tungsten : Simulation {

  int setNum, baseSeed, alpha_highOrd;
  std::array<int, 3> griddim;
  unsigned int maxStep, outStep, printStep, statusStep;
  unsigned long int N;
  bool dispOutput, write_fields, fullout_hdf5, makeMovie;
  std::string output_dir;
  std::array<int, 4> chunksize;
  double movieFramerate, a3D, a2D, a1D, n0, n_sol, n_vap, T, T0, Bx, alpha,
      alpha_farTol, tau, lambda, stabP, shift_u, shift_s, p2, p3, p4, p2_bar,
      p3_bar, p4_bar, q20, q21, q30, q31, q40, q20_bar, q21_bar, q30_bar,
      q31_bar, q40_bar, q2_bar, q3_bar, q4_bar, Bl;
  std::vector<double> kLap, filterMF, opL, opN, psiMF, psiN;
  std::vector<std::complex<double>> psi_F, psiMF_F, psiN_F, wrk;

  void create_params() {
    // The dataset number, also serves to name the set and determine rng seed
    setNum = 999;
    baseSeed = setNum * 1000;

    // width of grid, ideally should be power of 2
    Lx = 256;
    Ly = 256;
    Lz = 256;
    griddim = {Lx, Ly, Lz};

    // step parameters. maxStep should be a multiple of outStep.
    maxStep = 100000;
    outStep = 1000;
    printStep = outStep;
    statusStep = outStep / 2;

    // Whether to show a figure of the output while the simulation is running.
    // Remember to set this to false for remote runs
    dispOutput = true;

    // Output directory for parameter files and data files
    write_fields = true;
    output_dir = "/home/juajukka/dev/ppfc/results/clean_tungsten_3D_256";

    // Whether to output the full field at every outstep in hdf5
    fullout_hdf5 = true;
    chunksize = {Lx, Ly, Lz, 1};

    // Params for making a rough first-pass movie, won't look as good as a
    // properly formatted movie made with finished data NOTE: I HAVEN'T UPDATED
    // THIS FOR 3D
    makeMovie = false;
    movieFramerate = 5; // frames per second

    // 'lattice' constants for the three ordered phases that exist in 2D/3D PFC
    a3D = 2 * pi * sqrt(2);     // BCC
    a2D = 2 * pi * 2 / sqrt(3); // triangular
    a1D = 2 * pi;               // stripes

    // length and time steps
    dx = a3D / 8;
    dt = 1;

    // average density of the metastable fluid
    n0 = -0.4;

    // Bulk densities at coexistence, obtained from phase diagram for chosen
    // temperature
    n_sol = -0.047;
    n_vap = -0.464;

    // Effective temperature parameters. Temperature in K. Remember to change
    // n_sol and n_vap according to phase diagram when T is changed.
    T = 3300;
    T0 = 156000.0;
    Bx = 0.8582;

    // parameters that affect elastic and interface energies

    // width of C2's peak
    alpha = 0.50;

    // how much we allow the k=1 peak to affect the k=0 value of the
    // correlation, by changing the higher order components of the Gaussian
    // function
    alpha_farTol = 1.0 / 1000.0;

    // power of the higher order component of the gaussian function. Should be a
    // multiple of 2. Setting this to zero also disables the tolerance setting.
    alpha_highOrd = 4;

    // derived dimensionless values used in calculating vapor model parameters
    tau = T / T0;

    // Strength of the meanfield filter. Avoid values higher than ~0.28, to
    // avoid lattice-wavelength variations in the mean field
    lambda = 0.22;

    // numerical stability parameter for the exponential integrator method
    stabP = 0.2;

    // Vapor-model parameters

    shift_u = 0.3341;
    shift_s = 0.1898;

    p2 = 1.0;
    p3 = -1.0 / 2.0;
    p4 = 1.0 / 3.0;
    p2_bar = p2 + 2 * shift_s * p3 + 3 * pow(shift_s, 2) * p4;
    p3_bar = shift_u * (p3 + 3 * shift_s * p4);
    p4_bar = pow(shift_u, 2) * p4;

    q20 = -0.0037;
    q21 = 1.0;
    q30 = -12.4567;
    q31 = 20.0;
    q40 = 45.0;

    q20_bar = q20 + 2.0 * shift_s * q30 + 3.0 * pow(shift_s, 2) * q40;
    q21_bar = q21 + 2.0 * shift_s * q31;
    q30_bar = shift_u * (q30 + 3.0 * shift_s * q40);
    q31_bar = shift_u * q31;
    q40_bar = pow(shift_u, 2) * q40;

    q2_bar = q21_bar * tau + q20_bar;
    q3_bar = q31_bar * tau + q30_bar;
    q4_bar = q40_bar;
  }

  unsigned long int allocate(unsigned long int size_inbox,
                             unsigned long int size_outbox,
                             unsigned long int size_workspace) {
    unsigned long int size = 0;
    kLap.resize(size_outbox);
    filterMF.resize(size_outbox);
    opL.resize(size_outbox);
    opN.resize(size_outbox);
    psi.resize(size_inbox);
    psiMF.resize(size_inbox);
    psiN.resize(size_inbox);
    psi_F.resize(size_outbox);
    psiMF_F.resize(size_outbox);
    psiN_F.resize(size_outbox);
    wrk.resize(size_workspace);
    size += sizeof(double) * psi.size();
    /*
    size += sizeof(double) * k2_.size();
    size += sizeof(double) * L_.size();
    size += sizeof(std::complex<double>) * U.size();
    size += sizeof(std::complex<double>) * N.size();
    */
    return size;
  }

  void prepare_operators(std::array<int, 3> low, std::array<int, 3> high) {

    // prepare the linear and non-linear operators

    N = (high[0] - low[0] + 1) * (high[1] - low[1] + 1) *
        (high[2] - low[2] + 1);

    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    // laplacian operator
    double fx = 2.0 * pi / (dx * Lx);
    double fy = 2.0 * pi / (dy * Ly);
    double fz = 2.0 * pi / (dz * Lz);
    if (me == 0) {
      std::cout << "dx = " << dx << ", dy = " << dy << ", dz = " << dz
                << std::endl;
      std::cout << "Lx = " << Lx << ", Ly = " << Ly << ", Lz = " << Lz
                << std::endl;
      std::cout << "fx = " << fx << ", fy = " << fy << ", fz = " << fz
                << std::endl;
      std::cout << "low[0] = " << low[0] << ", low[1] = " << low[1]
                << ", low[2] = " << low[2] << std::endl;
      std::cout << "high[0] = " << high[0] << ", high[1] = " << high[1]
                << ", high[2] = " << high[2] << std::endl;
    }
    std::fill(kLap.begin(), kLap.end(), 0.0);
    unsigned long int idx = 0;
    double ki, kj, kk;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          /*
          if (i < Lx / 2) {
            ki = i * fx;
            std::cout << "i < Lx / 2, i = " << i << ", Lx = " << Lx
                      << ", fx = " << fx << ", ki = i * fx = " << ki
                      << std::endl;
          } else {
            ki = (i - Lx) * fx;
            std::cout << "i => Lx / 2, i = " << i << ", Lx = " << Lx
                      << ", fx = " << fx << ", ki = (i - Lx) * fx = " << ki
                      << std::endl;
          }
          */
          ki = i <= Lx / 2 ? i * fx : (i - Lx) * fx;
          kj = j <= Ly / 2 ? j * fy : (j - Ly) * fy;
          kk = k <= Lz / 2 ? k * fz : (k - Lz) * fz;
          /*
          std::cout << "ki = " << ki << ", kj = " << kj << ", kk = " << kk
                    << std::endl;
                    */
          kLap[idx] = -(ki * ki + kj * kj + kk * kk);
          /*
          if (kLap[idx] < -30.0) {
            std::cout << "i = " << i << ", j = " << j << ", k = " << k
                      << std::endl;
            std::cout << "ki = " << ki << ", kj = " << kj << ", kk = " << kk
                      << std::endl;
            throw "kLap too small";
          }
          */
          idx += 1;
        }
      }
    }

    for (idx = 0; idx < N; idx++) {
      // mean-field filtering operator (chi) make a C2 that's quasi-gaussian on
      // the left, and ken-style on the right
      double alpha2 = 2.0 * alpha * alpha;
      double lambda2 = 2.0 * lambda * lambda;
      filterMF[idx] = exp(kLap[idx] / lambda2);
      double k = sqrt(-kLap[idx]) - 1.0;
      double k2 = k * k;
      double g1, g2, gf, rTol;
      if (alpha_highOrd == 0) {
        g1 = exp(-k2 / alpha2); // gaussian peak
      } else {
        rTol = -alpha2 * log(alpha_farTol) - 1.0;
        // quasi-gaussian peak with higher order component to make it decay
        // faster towards k=0
        g1 = exp(-(k2 + rTol * pow(k, alpha_highOrd)) / alpha2);
      }

      // taylor expansion of gaussian peak to order 2
      g2 = 1.0 - 1.0 / alpha2 * k2;
      // splice the two sides of the peak
      gf = (k < 0.0) ? g1 : g2;

      // we separate this out because it is needed in the nonlinear
      // calculation when T is not constant in space
      double opPeak = -Bx * exp(-T / T0) * gf;

      // includes the lowest order n_mf term since it is a linear term
      double opCk = stabP + p2_bar + opPeak + q2_bar * filterMF[idx];

      opL[idx] = exp(kLap[idx] * opCk * dt);
      if (opCk == 0.0) {
        opN[idx] = kLap[idx] * dt;
      } else {
        opN[idx] = (exp(kLap[idx] * opCk * dt) - 1.0) / opCk;
      }

      bool has_nans = false;
      if (isnan(filterMF[idx])) {
        std::cout << "filterMF is NaN" << std::endl;
        has_nans = true;
      }
      if (isnan(k)) {
        std::cout << "k is NaN" << std::endl;
        has_nans = true;
      }
      if (isnan(rTol)) {
        std::cout << "rTol is NaN" << std::endl;
        has_nans = true;
      }
      if (isnan(g1)) {
        std::cout << "g1 is NaN" << std::endl;
        has_nans = true;
      }
      if (isnan(g2)) {
        std::cout << "g2 is NaN" << std::endl;
        has_nans = true;
      }
      if (isnan(gf)) {
        std::cout << "gf is NaN" << std::endl;
        has_nans = true;
      }
      if (isnan(opPeak)) {
        std::cout << "opPeak is NaN" << std::endl;
        has_nans = true;
      }
      if (isnan(opCk)) {
        std::cout << "opCk is NaN" << std::endl;
        has_nans = true;
      }
      if (isnan(opL[idx])) {
        std::cout << "opL[idx] is NaN" << std::endl;
        has_nans = true;
      }
      if (isnan(opN[idx])) {
        std::cout << "opN[idx] is NaN" << std::endl;
        has_nans = true;
      }
      if (has_nans) {
        std::cout << "k = " << k << std::endl;
        std::cout << "rTol = " << rTol << std::endl;
        std::cout << "alpha_highOrd = " << alpha_highOrd << std::endl;
        std::cout << "pow(k, 2) = " << pow(k, 2) << std::endl;
        std::cout << "pow(k, alpha_highOrd) = " << pow(k, alpha_highOrd)
                  << std::endl;
        // -2 * pow(alpha, 2) * log(alpha_farTol) - 1;
        std::cout << "pow(alpha, 2) = " << pow(alpha, 2) << std::endl;
        std::cout << "alpha_farTol = " << alpha_farTol << std::endl;
        std::cout << "log(alpha_farTol) = " << log(alpha_farTol) << std::endl;
        throw "Operators has NaN:s at index " + std::to_string(idx);
      }
    }

    if (me == 0) {
      std::cout << "operator summary on rank 0" << std::endl;
      {
        double min = std::numeric_limits<double>::max();
        double max = -min;
        double mean = 0.0;
        for (idx = 0; idx < kLap.size(); idx++) {
          min = std::min(min, kLap[idx]);
          max = std::max(max, kLap[idx]);
          mean += kLap[idx];
        }
        mean /= kLap.size();
        std::cout << "min(kLap) = " << min << std::endl;
        std::cout << "max(kLap) = " << max << std::endl;
        std::cout << "mean(kLap) = " << mean << std::endl;
      }
      {
        double min = std::numeric_limits<double>::max();
        double max = -min;
        double mean = 0.0;
        for (idx = 0; idx < opL.size(); idx++) {
          min = std::min(min, opL[idx]);
          max = std::max(max, opL[idx]);
          mean += opL[idx];
        }
        mean /= opL.size();
        std::cout << "min(opL) = " << min << std::endl;
        std::cout << "max(opL) = " << max << std::endl;
        std::cout << "mean(opL) = " << mean << std::endl;
      }
      {
        double min = std::numeric_limits<double>::max();
        double max = -min;
        double mean = 0.0;
        for (idx = 0; idx < opN.size(); idx++) {
          min = std::min(min, opN[idx]);
          max = std::max(max, opN[idx]);
          mean += opN[idx];
        }
        mean /= opN.size();
        std::cout << "min(opN) = " << min << std::endl;
        std::cout << "max(opN) = " << max << std::endl;
        std::cout << "mean(opN) = " << mean << std::endl;
      }
      {
        double min = std::numeric_limits<double>::max();
        double max = -min;
        double mean = 0.0;
        for (idx = 0; idx < filterMF.size(); idx++) {
          min = std::min(min, filterMF[idx]);
          max = std::max(max, filterMF[idx]);
          mean += filterMF[idx];
        }
        mean /= filterMF.size();
        std::cout << "min(filterMF) = " << min << std::endl;
        std::cout << "max(filterMF) = " << max << std::endl;
        std::cout << "mean(filterMF) = " << mean << std::endl;
      }
    }
  }

  void step(unsigned long int n, double t,
            heffte::fft3d_r2c<heffte::backend::fftw> &fft) {
    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    unsigned long int idx;
    if (n == 1) {
      if (me == 0) {
        std::cout << "First iteration, calculating initial value to psi_F"
                  << std::endl;
      }
      fft.forward(psi.data(), psi_F.data(), wrk.data());
    }

    // Mean-field density n_mf
    for (idx = 0; idx < psiMF_F.size(); idx++) {
      psiMF_F[idx] = psi_F[idx] * filterMF[idx];
    }
    fft.backward(psiMF_F.data(), psiMF.data(), wrk.data(), heffte::scale::full);

    /*
    if (me == 0) {
      double min = std::numeric_limits<double>::max();
      double max = -max;
      double mean = 0.0;
      for (idx = 0; idx < psiMF.size(); idx++) {
        min = std::min(min, psiMF[idx]);
        max = std::max(max, psiMF[idx]);
        mean += psiMF[idx];
      }
      mean /= psiMF.size();
      std::cout << "step = " << n << std::endl;
      std::cout << "psiMF max = " << max << std::endl;
      std::cout << "psiMF min = " << min << std::endl;
      std::cout << "psiMF mean = " << mean << std::endl;
    }
    */

    // Calculate the nonlinear part of the evolution equation in real space
    for (idx = 0; idx < psiN.size(); idx++) {
      double u = psi[idx];
      double v = psiMF[idx];
      psiN[idx] = p3_bar * u * u + p4_bar * u * u * u + q3_bar * v * v +
                  q4_bar * v * v * v;
      // Apply stabilization factor if given in parameters
      if (stabP != 0.0) {
        psiN[idx] = psiN[idx] - stabP * psi[idx];
      }
    }

    // Fourier transform of the nonlinear part of the evolution equation
    fft.forward(psiN.data(), psiN_F.data(), wrk.data());

    // Apply one step of the evolution equation
    for (idx = 0; idx < psi_F.size(); idx++) {
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];
    }

    // Apply inverse fourier transform to density field
    fft.backward(psi_F.data(), psi.data(), wrk.data(), heffte::scale::full);

    /*
    if (me == 0) {
      double min = std::numeric_limits<double>::max();
      double max = -max;
      double mean = 0.0;
      for (idx = 0; idx < psi.size(); idx++) {
        min = std::min(min, psi[idx]);
        max = std::max(max, psi[idx]);
        mean += psi[idx];
      }
      mean /= psi.size();
      std::cout << "step = " << n << std::endl;
      std::cout << "psi max = " << max << std::endl;
      std::cout << "psi min = " << min << std::endl;
      std::cout << "psi mean = " << mean << std::endl;
    }
    */
  }

  void fill_u0(std::array<int, 3> low, std::array<int, 3> high) {
    // calculating approx amplitude. This is related to the phase diagram
    // calculations.
    double rho_seed = n_sol;
    double A_phi = 135.0 * p4_bar;
    double B_phi = 16.0 * p3_bar + 48.0 * p4_bar * rho_seed;
    double C_phi = -6.0 * (Bx * exp(-T / T0)) + 6.0 * p2_bar +
                   12.0 * p3_bar * rho_seed + 18.0 * p4_bar * pow(rho_seed, 2);
    double d = 9.0 * pow(B_phi, 2) - 32.0 * A_phi * C_phi;
    if (d < 0) {
      d = -d;
    }
    double amp_eq = (-3.0 * B_phi + sqrt(d)) / (8.0 * A_phi);

    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    if (me == 0) {
      std::cout << "rho_seed = " << rho_seed << std::endl;
      std::cout << "A_phi = " << A_phi << std::endl;
      std::cout << "B_phi = " << B_phi << std::endl;
      std::cout << "C_phi = " << C_phi << std::endl;
      std::cout << "d = " << d << std::endl;
      std::cout << "amp_eq_cpx = " << amp_eq << std::endl;
    }

    dy = dx;
    dz = dx;

    double s = 1.0 / sqrt(2.0);
    std::array<double, 3> q1 = {s, s, 0};
    std::array<double, 3> q2 = {s, 0, s};
    std::array<double, 3> q3 = {0, s, s};
    std::array<double, 3> q4 = {s, 0, -s};
    std::array<double, 3> q5 = {s, -s, 0};
    std::array<double, 3> q6 = {0, s, -s};
    std::array<std::array<double, 3>, 6> q = {q1, q2, q3, q4, q5, q6};

    unsigned long int idx = 0;
    double r2 = pow(0.2 * (Lx * dx), 2);
    double u;
    for (auto k = low[2]; k <= high[2]; k++) {
      for (auto j = low[1]; j <= high[1]; j++) {
        for (auto i = low[0]; i <= high[0]; i++) {
          double x = x0 + i * dx;
          double y = y0 + j * dy;
          double z = z0 + k * dz;
          bool seedmask = x * x + y * y + z * z < r2;
          if (!seedmask) {
            u = n0;
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

    if (me == 0) {
      double min = std::numeric_limits<double>::max();
      double max = -max;
      double mean = 0.0;
      for (idx = 0; idx < psi.size(); idx++) {
        min = std::min(min, psi[idx]);
        max = std::max(max, psi[idx]);
        mean += psi[idx];
      }
      mean /= psi.size();
      std::cout << "min(psi) = " << min << std::endl;
      std::cout << "max(psi) = " << max << std::endl;
      std::cout << "mean(psi) = " << mean << std::endl;
    }
  }

  bool writeat(unsigned long int n, double t) {
    return (n % 1000 == 0);
  }
};

void MPI_Write_Data(std::string filename, MPI_Datatype &filetype,
                    std::vector<double> &u) {
  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_Offset filesize = 0;
  const unsigned int disp = 0;
  MPI_File_set_size(fh, filesize); // force overwriting existing data
  MPI_File_set_view(fh, disp, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
  MPI_File_write_all(fh, u.data(), u.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);
}

void MPI_Solve(Simulation *s) {

  // unpack simulation settings
  const int Lx = s->Lx;
  const int Ly = s->Ly;
  const int Lz = s->Lz;
  const double dx = s->dx;
  const double dy = s->dy;
  const double dz = s->dz;
  const double x0 = s->x0;
  const double y0 = s->y0;
  const double z0 = s->z0;
  const double t0 = s->t0;
  const double t1 = s->t1;
  const int max_iters = s->max_iters;

  MPI_Comm comm = MPI_COMM_WORLD;

  int me; // this process rank within the comm
  MPI_Comm_rank(comm, &me);

  int num_ranks; // total number of ranks in the comm
  MPI_Comm_size(comm, &num_ranks);

  /*
  If the input of an FFT transform consists of all real numbers,
   the output comes in conjugate pairs which can be exploited to reduce
   both the floating point operations and MPI communications.
   Given a global set of indexes, HeFFTe can compute the corresponding DFT
   and exploit the real-to-complex symmetry by selecting a dimension
   and reducing the indexes by roughly half (the exact formula is floor(n / 2)
  + 1).
   */
  const int Lx_c = floor(Lx / 2) + 1;
  // the dimension where the data will shrink
  const int r2c_direction = 0;
  // define real doman
  heffte::box3d<> real_indexes({0, 0, 0}, {Lx - 1, Ly - 1, Lz - 1});
  // define complex domain
  heffte::box3d<> complex_indexes({0, 0, 0}, {Lx_c - 1, Ly - 1, Lz - 1});

  // check if the complex indexes have correct dimension
  assert(real_indexes.r2c(r2c_direction) == complex_indexes);

  // report the indexes
  if (me == 0) {
    std::cout << "Number of ranks: " << num_ranks << std::endl;
    std::cout << "Domain size: " << Lx << " x " << Ly << " x " << Lz
              << std::endl;
    std::cout << "The global input contains " << real_indexes.count()
              << " real indexes." << std::endl;
    std::cout << "The global output contains " << complex_indexes.count()
              << " complex indexes." << std::endl;
  }

  // create a processor grid with minimum surface (measured in number of
  // indexes)
  auto proc_grid = heffte::proc_setup_min_surface(real_indexes, num_ranks);
  if (me == 0) {
    std::cout << "Minimum surface processor grid: [" << proc_grid[0] << ", "
              << proc_grid[1] << ", " << proc_grid[2] << "]" << std::endl;
  }

  // split all indexes across the processor grid, defines a set of boxes
  auto real_boxes = heffte::split_world(real_indexes, proc_grid);
  auto complex_boxes = heffte::split_world(complex_indexes, proc_grid);

  // pick the box corresponding to this rank
  heffte::box3d<> const inbox = real_boxes[me];
  heffte::box3d<> const outbox = complex_boxes[me];

  // define the heffte class and the input and output geometry
  heffte::fft3d_r2c<heffte::backend::fftw> fft(inbox, outbox, r2c_direction,
                                               comm);

  // vectors with the correct sizes to store the input and output data
  // taking the size of the input and output boxes
  std::cout << "Rank " << me << " input box: " << fft.size_inbox()
            << " indexes, indices x = [" << inbox.low[0] << ", "
            << inbox.high[0] << "], y = [" << inbox.low[1] << ", "
            << inbox.high[1] << "], "
            << "z = [" << inbox.low[2] << ", " << inbox.high[2]
            << "], outbox box: " << fft.size_outbox()
            << " indexes, indices x = [" << outbox.low[0] << ", "
            << outbox.high[0] << "], y = [" << outbox.low[1] << ", "
            << outbox.high[1] << "], "
            << "z = [" << outbox.low[2] << ", " << outbox.high[2] << "]"
            << std::endl;

  // Create and commit new data type
  MPI_Datatype filetype;
  const int size_array[] = {Lx, Ly, Lz};
  const int subsize_array[] = {inbox.high[0] - inbox.low[0] + 1,
                               inbox.high[1] - inbox.low[1] + 1,
                               inbox.high[2] - inbox.low[2] + 1};
  const int start_array[] = {inbox.low[0], inbox.low[1], inbox.low[2]};
  MPI_Type_create_subarray(3, size_array, subsize_array, start_array,
                           MPI_ORDER_FORTRAN, MPI_DOUBLE, &filetype);
  MPI_Type_commit(&filetype);

  if (me == 0) {
    std::cout << "Allocate arrays" << std::endl;
  }

  auto size =
      s->allocate(fft.size_inbox(), fft.size_outbox(), fft.size_workspace());
  if (me == 0) {
    double GB = 1.0 / (1024.0 * 1024.0 * 1024.0);
    double perdof = 1.0 * size / (Lx * Ly * Lz);
    std::cout << size * GB << " GB allocated on node 0 (" << perdof
              << "bytes / dof)" << std::endl;
  }

  if (me == 0) {
    std::cout << "Create parameters" << std::endl;
  }
  s->create_params();

  if (me == 0) {
    std::cout << "Prepare operators" << std::endl;
  }
  s->prepare_operators(outbox.low, outbox.high);

  /*
  if (me == 0) {
    std::cout << "Generate Laplace operator k2" << std::endl;
  }
  s->fill_k2(outbox.low, outbox.high);

  if (me == 0) {
    std::cout << "Generate linear operator L" << std::endl;
  }
  s->fill_L(outbox.low, outbox.high);
  */

  if (me == 0) {
    std::cout << "Generate initial condition u0" << std::endl;
  }
  s->fill_u0(inbox.low, inbox.high);

  if (me == 0) {
    std::cout << "Starting simulation" << std::endl;
  }

  unsigned long int n = 0;
  double t = t0;

  if (s->writeat(n, t)) {
    MPI_Write_Data(s->get_result_file_name(n, t), filetype, s->psi);
  }
  if (me == 0) {
    s->finalize_master_step(n, t);
  }

  /*
  double *u = s->u.data();
  std::complex<double> *U = s->U.data();
  std::complex<double> *N = s->N.data();
  std::complex<double> *wrk = workspace.data();
  */

  auto start = std::chrono::high_resolution_clock::now();
  while (!s->done(n, t)) {
    s->tune_dt(n, t);
    n += 1;
    t += s->get_dt();
    s->step(n, t, fft);
    if (s->writeat(n, t)) {
      MPI_Write_Data(s->get_result_file_name(n, t), filetype, s->psi);
    }
    if (me == 0) {
      s->finalize_master_step(n, t);
    }
  }

  /*
  while (!s->done(n, t)) {
    s->tune_dt(n, t);
    n += 1;
    t += s->get_dt();

    // FFT for linear part, U = fft(u),  O(n log n)
    auto t0 = std::chrono::high_resolution_clock::now();
    fft.forward(u, U, wrk);

    // calculate nonlinear part, u = f(u), O(n) (store in-place)
    auto t1 = std::chrono::high_resolution_clock::now();
    s->calculate_nonlinear_part();

    // FFT for nonlinear part, N = fft(u), O(n log n)
    auto t2 = std::chrono::high_resolution_clock::now();
    fft.forward(u, N, wrk);

    // Semi-implicit time integration U = 1 / (1 - dt * L) * (U - k2 * dt * N),
    // O(n)
    auto t3 = std::chrono::high_resolution_clock::now();
    s->integrate();

    // Back to real space, u = fft^-1(U), O(n log n)
    auto t4 = std::chrono::high_resolution_clock::now();
    fft.backward(U, u, wrk, heffte::scale::full);

    auto t5 = std::chrono::high_resolution_clock::now();
    if (s->writeat(n, t)) { // O(?)
      MPI_Write_Data(s->get_result_file_name(n, t), filetype, s->u);
    }

    auto t6 = std::chrono::high_resolution_clock::now();
    s->finalize_step(n, t);
    if (me == 0) {
      s->finalize_master_step(n, t);
    }

    if (me == 0) {
      std::cout << "Iteration " << n << " (time " << t << ") summary: ";
      auto dt1 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      auto dt2 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
      auto dt3 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
      auto dt4 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
      auto dt5 = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4);
      auto dt6 = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5);
      auto dt7 = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t0);
      std::cout << "U=fft(u) " << dt1.count() << " ms, ";
      std::cout << "n=f(u) " << dt2.count() << " ms, ";
      std::cout << "N=fft(n) " << dt3.count() << " ms, ";
      std::cout << "U=L(U, N) " << dt4.count() << " ms, ";
      std::cout << "u=FFT^-1(U) " << dt5.count() << " ms, ";
      std::cout << "W(u) " << dt6.count() << " ms, ";
      std::cout << "T " << dt7.count() << std::endl;
    }
  }
  */

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  if (me == 0) {
    std::cout << n << " iterations in " << duration.count() / 1000.0
              << " seconds (" << duration.count() / n << " ms / iteration)"
              << std::endl;
  }

  if (me == 0) {
    std::cout << "Simulation done. Exit message: " + s->exit_msg << std::endl;
  }
}

int main(int argc, char *argv[]) {

  argparse::ArgumentParser program("diffusion");

  program.add_argument("--verbose")
      .help("increase output verbosity")
      .default_value(true)
      .implicit_value(true);

  program.add_argument("--Lx")
      .help("Number of grid points in x direction")
      .scan<'i', int>()
      .default_value(256);

  program.add_argument("--Ly")
      .help("Number of grid points in y direction")
      .scan<'i', int>()
      .default_value(256);

  program.add_argument("--Lz")
      .help("Number of grid points in z direction")
      .scan<'i', int>()
      .default_value(256);

  program.add_argument("--results-dir")
      .help("Where to write results")
      .default_value("./results");

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  Simulation *s = new Tungsten();
  auto Lx = program.get<int>("--Lx");
  auto Ly = program.get<int>("--Ly");
  auto Lz = program.get<int>("--Lz");
  double dx = 2.0 * pi * sqrt(2) / 8.0;
  double dy = dx;
  double dz = dx;
  double x0 = -0.5 * Lx * dx;
  double y0 = -0.5 * Ly * dy;
  double z0 = -0.5 * Lz * dz;
  s->set_domain({x0, y0, z0}, {dx, dx, dx}, {Lx, Ly, Lz});
  s->set_time(0.0, 10000.0, 1.0);
  s->set_max_iters(10000);
  s->set_results_dir(program.get<std::string>("--results-dir"));
  MPI_Init(&argc, &argv);
  MPI_Solve(s);
  MPI_Finalize();

  return 0;
}
