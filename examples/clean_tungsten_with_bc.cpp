// heFFTe implementation of pfc code

#include <nlohmann/json.hpp>
#include <pfc/pfc.hpp>
#include <random>

using json = nlohmann::json;
using namespace pfc;

namespace pfc {

class seed {

private:
  typedef std::array<double, 3> vec3;
  typedef std::array<vec3, 3> mat3;
  typedef std::array<vec3, 6> vec36;
  typedef std::array<vec3, 2> vec32;

  const vec3 location_;
  const vec3 orientation_;
  const vec36 q_;
  const vec32 bbox_;
  const double rho_;
  const double radius_;
  const double amplitude_;

  mat3 yaw(double a) {
    double ca = cos(a);
    double sa = sin(a);
    return {vec3({ca, -sa, 0.0}), vec3({sa, ca, 0.0}), vec3({0.0, 0.0, 1.0})};
  }

  mat3 pitch(double b) {
    double cb = cos(b);
    double sb = sin(b);
    return {vec3({cb, 0.0, sb}), vec3({0.0, 1.0, 0.0}), vec3({-sb, 0.0, cb})};
  }

  mat3 roll(double c) {
    double cc = cos(c);
    double sc = sin(c);
    return {vec3({1.0, 0.0, 0.0}), vec3({0.0, cc, -sc}), vec3({0.0, sc, cc})};
  }

  mat3 mult3(const mat3 &A, const mat3 &B) {
    mat3 C = {0};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return C;
  }

  vec3 mult3(const mat3 &A, const vec3 &b) {
    vec3 c = {0};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        c[i] += A[i][j] * b[j];
      }
    }
    return c;
  }

  vec36 rotate(const vec3 &orientation) {
    const double s = 1.0 / sqrt(2.0);
    const vec3 q1 = {s, s, 0};
    const vec3 q2 = {s, 0, s};
    const vec3 q3 = {0, s, s};
    const vec3 q4 = {s, 0, -s};
    const vec3 q5 = {s, -s, 0};
    const vec3 q6 = {0, s, -s};
    mat3 Ra = yaw(orientation[0]);
    mat3 Rb = pitch(orientation[1]);
    mat3 Rc = roll(orientation[2]);
    mat3 R = mult3(Ra, mult3(Rb, Rc));
    const vec36 q = {mult3(R, q1), mult3(R, q2), mult3(R, q3),
                     mult3(R, q4), mult3(R, q5), mult3(R, q6)};
    return q;
  }

  vec32 bounding_box(const vec3 &location, double radius) {
    const vec3 low = {location[0] - radius, location[1] - radius,
                      location[2] - radius};
    const vec3 high = {location[0] + radius, location[1] + radius,
                       location[2] + radius};
    const vec32 bbox = {low, high};
    return bbox;
  }

  inline bool is_inside_bbox(const vec3 &location) const {
    const vec32 bbox = get_bbox();
    return (location[0] > bbox[0][0]) && (location[0] < bbox[1][0]) &&
           (location[1] > bbox[0][1]) && (location[1] < bbox[1][1]) &&
           (location[2] > bbox[0][2]) && (location[2] < bbox[1][2]);
  }

  double get_radius() const { return radius_; }
  double get_rho() const { return rho_; }
  double get_amplitude() const { return amplitude_; }
  vec3 get_location() const { return location_; }
  vec36 get_q() const { return q_; }
  vec32 get_bbox() const { return bbox_; }

public:
  seed(const vec3 &location, const vec3 &orientation, const double radius,
       const double rho, const double amplitude)
      : location_(location), orientation_(orientation), q_(rotate(orientation)),
        bbox_(bounding_box(location, radius)), rho_(rho), radius_(radius),
        amplitude_(amplitude) {}

  bool is_inside(const vec3 &X) const {
    /*
    if (!is_inside_bbox(X)) {
      return false;
    }
    */
    const vec3 Y = get_location();
    double x = X[0] - Y[0];
    double y = X[1] - Y[1];
    double z = X[2] - Y[2];
    double r = get_radius();

    return x * x + y * y + z * z < r * r;
  }

  double get_value(const vec3 &location) const {
    double x = location[0];
    double y = location[1];
    double z = location[2];
    double u = get_rho();
    double a = get_amplitude();
    vec36 q = get_q();
    for (int i = 0; i < 6; i++) {
      u += 2.0 * a * cos(q[i][0] * x + q[i][1] * y + q[i][2] * z);
    }
    return u;
  }
};

} // namespace pfc

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

  // calculating approx amplitude. This is related to the phase diagram
  // calculations.
  const double rho_seed = n_sol;
  const double A_phi = 135.0 * p4_bar;
  const double B_phi = 16.0 * p3_bar + 48.0 * p4_bar * rho_seed;
  const double C_phi = -6.0 * (Bx * exp(-T / T0)) + 6.0 * p2_bar +
                       12.0 * p3_bar * rho_seed +
                       18.0 * p4_bar * pow(rho_seed, 2);
  const double d = std::abs(9.0 * pow(B_phi, 2) - 32.0 * A_phi * C_phi);
  const double amp_eq = (-3.0 * B_phi + sqrt(d)) / (8.0 * A_phi);

  // for boundary condition
  double rho_low = n_vap;
  double rho_high = n0;
};

enum class initial_condition { single_crystal, random_uniform, regular_grid };

class Tungsten : public Simulation {
  using Simulation::Simulation;

private:
  initial_condition initial_condition_ = initial_condition::regular_grid;

public:
  params p;

  // we will allocate these arrays later on
  std::vector<double> filterMF, opL, opN;
  std::vector<double> psiMF, psi, psiN;
  std::vector<std::complex<double>> psiMF_F, psi_F, psiN_F;

  // used to measure execution time of step and write_results
  const std::array<double, 10> timers = {0};

  void set_initial_condition(std::string initial_condition) {
    if (initial_condition == "single_crystal") {
      initial_condition_ = initial_condition::single_crystal;
    } else if (initial_condition == "random_uniform") {
      initial_condition_ = initial_condition::random_uniform;
    } else if (initial_condition == "regular_grid") {
      initial_condition_ = initial_condition::regular_grid;
    } else {
      int me;
      MPI_Comm_rank(MPI_COMM_WORLD, &me);
      if (me == 0) {
        std::cerr << "Unknown initial condition " << initial_condition << "\n";
      }
    }
  }

  initial_condition get_initial_condition() { return initial_condition_; }

  /*
    This function is ran only one time during the initialization of solver. Used
    to allocate all necessary arrays.
   */
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

    mem_allocated = size_inbox * (8 * 3) + size_outbox * (16 * 3 + 8 * 3);
  }

  /*
    This function is called after allocate(), used to fill operators.
  */
  void prepare_operators(std::array<int, 3> low, std::array<int, 3> high) {

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

  void apply_bc(std::array<int, 3> low, std::array<int, 3> high) {
    double xwidth = 20.0;
    double alpha = 1.0;
    double xpos = Lx * dx - xwidth;
    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          const double x = x0 + i * dx;
          if (std::abs(x - xpos) < xwidth) {
            double S = 1.0 / (1.0 + exp(-alpha * (x - xpos)));
            psi[idx] = p.rho_low * S + p.rho_high * (1.0 - S);
          }
          idx += 1;
        }
      }
    }
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
  void step(int, double) {

    for (int i = 0; i < 8; i++) {
      timing[i] = 0.0;
    }

    timing[0] -= MPI_Wtime();

    // calculate psi_F = fft(psi)
    timing[1] -= MPI_Wtime();
    fft_r2c(psi, psi_F);
    timing[1] += MPI_Wtime();

    // Calculate mean-field density n_mf
    timing[2] -= MPI_Wtime();
    for (long int idx = 0, N = psiMF_F.size(); idx < N; idx++) {
      psiMF_F[idx] = filterMF[idx] * psi_F[idx];
    }
    timing[2] += MPI_Wtime();

    timing[1] -= MPI_Wtime();
    fft_c2r(psiMF_F, psiMF);
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
    fft_r2c(psiN, psiN_F);
    timing[1] += MPI_Wtime();

    // Apply one step of the evolution equation
    timing[2] -= MPI_Wtime();
    for (long int idx = 0, N = psi_F.size(); idx < N; idx++) {
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];
    }
    timing[2] += MPI_Wtime();

    // Inverse Fourier transform result back to real space
    timing[1] -= MPI_Wtime();
    fft_c2r(psi_F, psi);
    timing[1] += MPI_Wtime();

    timing[0] += MPI_Wtime();
  }

  void prepare_initial_condition(std::array<int, 3> low,
                                 std::array<int, 3> high) {
    switch (get_initial_condition()) {
    case initial_condition::random_uniform:
      prepare_initial_condition_random_uniform(low, high);
      break;
    case initial_condition::regular_grid:
      prepare_initial_condition_regular_grid(low, high);
      break;
    }
  }

  /*
  Initial condition is defined here
  */
  void prepare_initial_condition_random_uniform(std::array<int, 3> low,
                                                std::array<int, 3> high) {

    std::vector<pfc::seed> seeds;

    /*
    seeds.push_back(pfc::seed(radius, rho, amplitude).translate(T).rotate(R));
    */

    const int nseeds = (int)(dy * Ly * dz * Lz / 300.0);
    const double radius = 20.0;

    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    if (me == 0) {
      std::cout << "Generating " << nseeds << " random seeds with radius "
                << radius << "\n";
    }

    const double rho = p.rho_seed;
    const double amplitude = p.amp_eq;
    const double lower_x = radius;
    const double upper_x = 4 * radius;
    const double lower_y = 0.0;
    const double upper_y = Ly * dy;
    const double lower_z = 0.0;
    const double upper_z = Lz * dz;
    srand(42);
    std::uniform_real_distribution<double> rx(lower_x, upper_x);
    std::uniform_real_distribution<double> ry(lower_y, upper_y);
    std::uniform_real_distribution<double> rz(lower_z, upper_z);
    std::uniform_real_distribution<double> ro(0.0, 8.0 * atan(1.0));
    std::default_random_engine re;
    typedef std::array<double, 3> vec3;
    auto random_location = [&re, &rx, &ry, &rz]() {
      return vec3({rx(re), ry(re), rz(re)});
    };
    auto random_orientation = [&re, &ro]() {
      return vec3({ro(re), ro(re), ro(re)});
    };

    for (int i = 0; i < nseeds; i++) {
      const std::array<double, 3> location = random_location();
      const std::array<double, 3> orientation = random_orientation();
      const pfc::seed seed(location, orientation, radius, rho, amplitude);
      seeds.push_back(seed);
    }

    std::fill(psi.begin(), psi.end(), p.n0);
    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          const double x = x0 + i * dx;
          const double y = y0 + j * dy;
          const double z = z0 + k * dz;
          const std::array<double, 3> X = {x, y, z};
          for (const auto &seed : seeds) {
            if (seed.is_inside(X)) {
              psi[idx] = seed.get_value(X);
              break;
            }
          }
          idx += 1;
        }
      }
    }
  }

  void prepare_initial_condition_regular_grid(std::array<int, 3> low,
                                              std::array<int, 3> high) {
    std::vector<pfc::seed> seeds;
    const int Nx = 1;
    const int Ny = 6;
    const int Nz = 6;

    const double radius = 30.0;
    const double rho = p.rho_seed;
    const double amplitude = p.amp_eq;

    const double Dy = dy * Ly / Ny;
    const double Dz = dz * Lz / Nz;
    const double X0 = 3 * radius;
    const double Y0 = Dy / 2.0;
    const double Z0 = Dz / 2.0;
    const int nseeds = Nx * Ny * Nz;

    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    if (me == 0) {
      std::cout << "Generating " << nseeds << " regular seeds with radius "
                << radius << "\n";
    }

    srand(42);
    std::uniform_real_distribution<double> rt(-0.2 * radius, 0.2 * radius);
    std::uniform_real_distribution<double> rr(0.0, 8.0 * atan(1.0));
    std::default_random_engine re;

    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        const std::array<double, 3> location = {
            X0 + rt(re), Y0 + Dy * j + rt(re), Z0 + Dz * k + rt(re)};
        const std::array<double, 3> orientation = {rr(re), rr(re), rr(re)};
        const pfc::seed seed(location, orientation, radius, rho, amplitude);
        seeds.push_back(seed);
      }
    }

    std::fill(psi.begin(), psi.end(), p.n0);
    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          const double x = x0 + i * dx;
          const double y = y0 + j * dy;
          const double z = z0 + k * dz;
          const std::array<double, 3> X = {x, y, z};
          for (const auto &seed : seeds) {
            if (seed.is_inside(X)) {
              psi[idx] = seed.get_value(X);
              break;
            }
          }
          idx += 1;
        }
      }
    }
  }

  bool writeat(int n, double t) { return (n % 1000 == 0) || (t >= t1); }

  /*
 Results writing routine
 */
  void write_results(int n, double) {
    auto filename = results_dir / ("u" + std::to_string(n) + ".bin");
    if (me == 0) {
      std::cout << "Writing results to " << filename << std::endl;
    }
    MPI_Write_Data(filename, psi);
  };

}; // end of class

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

  {
    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    json settings;

    // read settings from file if given as a program argument, otherwise from
    // standard input
    if (argc > 1) {
      if (me == 0)
        std::cout << "Reading simulation settings from file " << argv[1]
                  << "\n\n";
      const std::filesystem::path settings_file(argv[1]);
      if (!std::filesystem::exists(settings_file)) {
        if (me == 0)
          std::cerr << "settings file " << settings_file
                    << " does not exist!\n";
        return 1;
      }
      std::ifstream input_file(settings_file);
      input_file >> settings;
    } else {
      if (me == 0)
        std::cout << "Reading simulation settings from standard input\n\n";
      std::cin >> settings;
    }
    if (me == 0) {
      std::cout << "Simulation settings:\n\n";
      std::cout << settings.dump(4) << "\n\n";
    }

    const std::string results_dir_ = settings["results_dir"];
    const std::filesystem::path results_dir(results_dir_);
    if (me == 0) {
      if (!std::filesystem::exists(results_dir)) {
        std::cout << "Results dir " << results_dir
                  << " does not exist, creating\n";
        std::filesystem::create_directories(results_dir);
      }
    }

    // Let's define simulation settings, that are kind of standard for all types
    // of simulations. At least we need to define the world size and time.
    // Even spaced grid is used, thus we have something like x = x0 + dx*i for
    // spatial coordinate and t = t0 + dt*n for time.

    Tungsten T({settings["Lx"], settings["Ly"], settings["Lz"]});

    T.set_dxdydz(settings["dx"], settings["dy"], settings["dz"]);
    T.set_origin(settings["x0"], settings["y0"], settings["z0"]);
    T.set_time(settings["t0"], settings["t1"], settings["dt"]);
    T.set_initial_condition(settings["initial_condition"]);

    // define where to store results
    T.set_results_dir(results_dir);
    T.set_saveat(settings["saveat"]);

    MPI_Solve(T);
  }

  MPI_Finalize();
  return 0;
}
