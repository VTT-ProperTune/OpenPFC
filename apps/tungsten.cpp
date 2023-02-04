#include <nlohmann/json.hpp>
#include <openpfc/openpfc.hpp>
#include <openpfc/utils/timeleft.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>

using namespace pfc;
using namespace pfc::utils;
using namespace std;

/*
Model parameters
*/
struct Params {
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
  double rho_seed = n_sol;
  double A_phi = 135.0 * p4_bar;
  double B_phi = 16.0 * p3_bar + 48.0 * p4_bar * rho_seed;
  double C_phi = -6.0 * (Bx * exp(-T / T0)) + 6.0 * p2_bar +
                 12.0 * p3_bar * rho_seed + 18.0 * p4_bar * pow(rho_seed, 2);
  double d = abs(9.0 * pow(B_phi, 2) - 32.0 * A_phi * C_phi);
  double amp_eq = (-3.0 * B_phi + sqrt(d)) / (8.0 * A_phi);
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
  Params p;

public:
  Params &get_params() { return p; }

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
          double alpha2 = 2.0 * p.alpha * p.alpha;
          double lambda2 = 2.0 * p.lambda * p.lambda;
          double fMF = exp(kLap / lambda2);
          double k = sqrt(-kLap) - 1.0;
          double k2 = k * k;

          double rTol = -alpha2 * log(p.alpha_farTol) - 1.0;
          double g1 = 0;
          if (p.alpha_highOrd == 0) { // gaussian peak
            g1 = exp(-k2 / alpha2);
          } else { // quasi-gaussian peak with higher order component to make it
                   // decay faster towards k=0
            g1 = exp(-(k2 + rTol * pow(k, p.alpha_highOrd)) / alpha2);
          }

          // taylor expansion of gaussian peak to order 2
          double g2 = 1.0 - 1.0 / alpha2 * k2;
          // splice the two sides of the peak
          double gf = (k < 0.0) ? g1 : g2;
          // we separate this out because it is needed in the nonlinear
          // calculation when T is not constant in space
          double opPeak = -p.Bx * exp(-p.T / p.T0) * gf;
          // includes the lowest order n_mf term since it is a linear term
          double opCk = p.stabP + p.p2_bar + opPeak + p.q2_bar * fMF;

          filterMF[idx] = fMF;
          opL[idx] = exp(kLap * opCk * dt);
          opN[idx] = (opCk == 0.0) ? kLap * dt : (opL[idx] - 1.0) / opCk;
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

    (void)t; // suppress compiler warning about unused parameter

    FFT &fft = get_fft();

    // Calculate mean-field density n_mf
    fft.forward(psi, psi_F);
    for (size_t idx = 0, N = psiMF_F.size(); idx < N; idx++)
      psiMF_F[idx] = filterMF[idx] * psi_F[idx];
    fft.backward(psiMF_F, psiMF);

    // Calculate the nonlinear part of the evolution equation in a real space
    for (size_t idx = 0, N = psiN.size(); idx < N; idx++) {
      double u = psi[idx], v = psiMF[idx];
      double u2 = u * u, u3 = u * u * u, v2 = v * v, v3 = v * v * v;
      double p3 = p.p3_bar, p4 = p.p4_bar, q3 = p.q3_bar, q4 = p.q4_bar;
      psiN[idx] = p3 * u2 + p4 * u3 + q3 * v2 + q4 * v3;
    }

    // Apply stabilization factor if given in parameters
    if (p.stabP != 0.0)
      for (size_t idx = 0, N = psiN.size(); idx < N; idx++)
        psiN[idx] = psiN[idx] - p.stabP * psi[idx];

    // Fourier transform of the nonlinear part of the evolution equation
    fft.forward(psiN, psiN_F);

    // Apply one step of the evolution equation
    for (size_t idx = 0, N = psi_F.size(); idx < N; idx++)
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];

    // Inverse Fourier transform result back to real space
    fft.backward(psi_F, psi);
  }

}; // end of class

class SingleSeed : public FieldModifier {

public:
  void apply(Model &m, double) override {
    Params &p = dynamic_cast<Tungsten &>(m).get_params();
    const World &w = m.get_world();
    const Decomposition &decomp = m.get_decomposition();
    Field &f = m.get_field();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto x0 = w.x0;
    auto y0 = w.y0;
    auto z0 = w.z0;

    double s = 1.0 / sqrt(2.0);
    std::array<double, 3> q1 = {s, s, 0};
    std::array<double, 3> q2 = {s, 0, s};
    std::array<double, 3> q3 = {0, s, s};
    std::array<double, 3> q4 = {s, 0, -s};
    std::array<double, 3> q5 = {s, -s, 0};
    std::array<double, 3> q6 = {0, s, -s};
    std::array<std::array<double, 3>, 6> q = {q1, q2, q3, q4, q5, q6};

    long int idx = 0;
    // double r2 = pow(0.2 * (Lx * dx), 2);
    double r2 = pow(64.0, 2);
    double amp_eq = p.amp_eq;
    double u;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = x0 + i * dx;
          double y = y0 + j * dy;
          double z = z0 + k * dz;
          bool seedmask = x * x + y * y + z * z < r2;
          if (!seedmask) {
            u = p.n0;
          } else {
            u = p.rho_seed;
            for (int i = 0; i < 6; i++) {
              u += 2.0 * amp_eq * cos(q[i][0] * x + q[i][1] * y + q[i][2] * z);
            }
          }
          f[idx] = u;
          idx += 1;
        }
      }
    }
  }
};

class RandomSeeds : public FieldModifier {

  void apply(Model &m, double) override {
    Params &p = dynamic_cast<Tungsten &>(m).get_params();
    const World &w = m.get_world();
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_field();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;

    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto x0 = w.x0;
    auto y0 = w.y0;
    auto z0 = w.z0;

    std::vector<Seed> seeds;

    const int nseeds = 150;
    const double radius = 20.0;
    const double rho = p.rho_seed;
    const double amplitude = p.amp_eq;
    const double lower_x = -128.0 + radius;
    const double upper_x = -128.0 + 3 * radius;
    const double lower_y = -128.0;
    const double upper_y = 128.0;
    const double lower_z = -128.0;
    const double upper_z = 128.0;
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
      const Seed seed(location, orientation, radius, rho, amplitude);
      seeds.push_back(seed);
    }

    std::fill(field.begin(), field.end(), p.n0);
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
              field[idx] = seed.get_value(X);
            }
          }
          idx += 1;
        }
      }
    }
  }
};

class SeedGrid : public FieldModifier {
private:
  int m_Nx, m_Ny, m_Nz;
  double m_X0, m_radius;

public:
  SeedGrid(int Ny, int Nz, double X0, double radius)
      : m_Nx(1), m_Ny(Ny), m_Nz(Nz), m_X0(X0), m_radius(radius) {}

  void apply(Model &m, double) override {
    Params &p = dynamic_cast<Tungsten &>(m).get_params();
    const World &w = m.get_world();
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_field();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;

    // auto Lx = w.Lx;
    auto Ly = w.Ly;
    auto Lz = w.Lz;
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto x0 = w.x0;
    auto y0 = w.y0;
    auto z0 = w.z0;

    std::vector<Seed> seeds;

    int Nx = m_Nx;
    int Ny = m_Ny;
    int Nz = m_Nz;
    double radius = m_radius;

    double rho = p.rho_seed;
    double amplitude = p.amp_eq;

    double Dy = dy * Ly / Ny;
    double Dz = dz * Lz / Nz;
    double X0 = m_X0;
    double Y0 = Dy / 2.0;
    double Z0 = Dz / 2.0;
    int nseeds = Nx * Ny * Nz;

    cout << "Generating " << nseeds << " regular seeds with radius " << radius
         << "\n";

    srand(42);
    std::uniform_real_distribution<double> rt(-0.2 * radius, 0.2 * radius);
    std::uniform_real_distribution<double> rr(0.0, 8.0 * atan(1.0));
    std::default_random_engine re;

    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        const std::array<double, 3> location = {
            X0 + rt(re), Y0 + Dy * j + rt(re), Z0 + Dz * k + rt(re)};
        const std::array<double, 3> orientation = {rr(re), rr(re), rr(re)};
        const Seed seed(location, orientation, radius, rho, amplitude);
        seeds.push_back(seed);
      }
    }

    std::fill(field.begin(), field.end(), p.n0);
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
              field[idx] = seed.get_value(X);
              break;
            }
          }
          idx += 1;
        }
      }
    }
  }
};

/*
Helper functions to construct objects from json file
*/

using json = nlohmann::json;

template <class T> T from_json(const json &settings);

template <> World from_json<World>(const json &settings) {
  int Lx = settings["Lx"];
  int Ly = settings["Ly"];
  int Lz = settings["Lz"];
  double dx = settings["dx"];
  double dy = settings["dy"];
  double dz = settings["dz"];
  double x0 = 0.0;
  double y0 = 0.0;
  double z0 = 0.0;
  string origo = settings["origo"];
  if (origo == "center") {
    x0 = -0.5 * dx * Lx;
    y0 = -0.5 * dy * Ly;
    z0 = -0.5 * dz * Lz;
  }
  World world({Lx, Ly, Lz}, {x0, y0, z0}, {dx, dy, dz});
  return world;
}

template <> Time from_json<Time>(const json &settings) {
  double t0 = settings["t0"];
  double t1 = settings["t1"];
  double dt = settings["dt"];
  double saveat = settings["saveat"];
  Time time({t0, t1, dt}, saveat);
  return time;
}

class MPI_Worker {
  MPI_Comm m_comm;
  int m_rank, m_num_procs;

public:
  MPI_Worker(int argc, char *argv[], MPI_Comm comm) : m_comm(comm) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &m_rank);
    MPI_Comm_size(comm, &m_num_procs);
    if (m_rank != 0) mute();
    cout << "MPI_Init(): initialized " << m_num_procs << " processes" << endl;
  }

  ~MPI_Worker() { MPI_Finalize(); }
  int get_rank() const { return m_rank; }
  int get_num_ranks() const { return m_num_procs; }
  void mute() { cout.setstate(ios::failbit); }
  void unmute() { cout.clear(); }
};

/*
The main application
*/

heffte::plan_options get_plan_options() {
  heffte::plan_options options =
      heffte::default_options<heffte::backend::fftw>();
  /*
  options.use_reorder = true;
  options.algorithm = reshape_algorithm::alltoall;
  options.algorithm = reshape_algorithm::alltoallv;
  options.algorithm = reshape_algorithm::p2p;
  options.use_pencils = true;
  options.use_gpu_aware = true;
  */
  options.algorithm = heffte::reshape_algorithm::p2p_plined;
  return options;
}

class App {
private:
  MPI_Comm m_comm;
  MPI_Worker m_worker;
  bool rank0;
  json m_settings;
  World m_world;
  Decomposition m_decomp;
  FFT m_fft;
  Time m_time;
  Tungsten m_model;
  Simulator m_simulator;
  double m_total_steptime = 0.0;
  double m_total_fft_time = 0.0;
  double m_steptime = 0.0;
  double m_fft_time = 0.0;
  double m_avg_steptime = 0.0;
  int m_steps_done = 0;

  // save detailed timing information for each mpi rank and step?
  bool m_detailed_timing = false;
  bool m_detailed_timing_print = false;
  bool m_detailed_timing_write = false;
  string m_detailed_timing_filename = "timing.bin";

  // read settings from file if or standard input
  json read_settings(int argc, char *argv[]) {
    json settings;
    if (argc > 1) {
      if (rank0) cout << "Reading input from file " << argv[1] << "\n\n";
      filesystem::path file(argv[1]);
      if (!filesystem::exists(file)) {
        if (rank0) cerr << "File " << file << " does not exist!\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      std::ifstream input_file(file);
      input_file >> settings;
    } else {
      if (rank0) std::cout << "Reading simulation settings from stdin\n\n";
      std::cin >> settings;
    }
    return settings;
  }

public:
  App(int argc, char *argv[], MPI_Comm comm = MPI_COMM_WORLD)
      : m_comm(comm), m_worker(MPI_Worker(argc, argv, comm)),
        rank0(m_worker.get_rank() == 0), m_settings(read_settings(argc, argv)),
        m_world(from_json<World>(m_settings)),
        m_decomp(Decomposition(m_world, comm)),
        m_fft(FFT(m_decomp, comm, get_plan_options())),
        m_time(from_json<Time>(m_settings)), m_model(Tungsten(m_fft)),
        m_simulator(Simulator(m_model, m_time)) {}

  bool create_results_dir(const string &output) {
    filesystem::path results_dir(output);
    if (results_dir.has_filename()) results_dir = results_dir.parent_path();
    if (!std::filesystem::exists(results_dir)) {
      cout << "Results dir " << results_dir << " does not exist, creating\n";
      filesystem::create_directories(results_dir);
      return true;
    } else {
      cout << "Warning: results dir " << results_dir << " already exists\n";
      return false;
    }
  }

  void read_model_parameters() {
    cout << "Reading model parameters from json file" << endl;
    Params &p = m_model.get_params();
    if (m_settings.contains("model")) {
      auto p2 = m_settings["model"]["params"];
      if (p2.contains("n0")) {
        double n0 = p2["n0"];
        cout << "Changing average density of metastable fluid to " << n0
             << endl;
        p.n0 = n0;
      }
    }
  }

  void read_detailed_timing_configuration() {
    if (m_settings.contains("detailed_timing")) {
      auto timing = m_settings["detailed_timing"];
      if (timing.contains("enabled")) m_detailed_timing = timing["enabled"];
      if (timing.contains("print")) m_detailed_timing_print = timing["print"];
      if (timing.contains("write")) m_detailed_timing_write = timing["write"];
      if (timing.contains("filename"))
        m_detailed_timing_filename = timing["filename"];
    }
  }

  void add_result_writers() {
    cout << "Adding results writers" << endl;
    if (m_settings.contains("saveat") && m_settings.contains("fields") &&
        m_settings["saveat"] > 0) {
      for (const auto &field : m_settings["fields"]) {
        string name = field["name"];
        string data = field["data"];
        if (rank0) create_results_dir(data);
        cout << "Writing field " << name << " to " << data << endl;
        m_simulator.add_results_writer(name, make_unique<BinaryWriter>(data));
      }
    } else {
      cout << "Warning: not writing results to anywhere." << endl;
      cout << "To write results, add ResultsWriter to model." << endl;
    }
  }

  void add_initial_conditions() {
    cout << "Adding initial conditions" << endl;
    auto ic = m_settings["initial_condition"];
    if (ic["type"] == "single_seed") {
      cout << "Adding single seed initial condition" << endl;
      m_simulator.add_initial_conditions(make_unique<SingleSeed>());
    } else if (ic["type"] == "random_seeds") {
      cout << "Adding randomized seeds initial condition" << endl;
      m_simulator.add_initial_conditions(make_unique<RandomSeeds>());
    } else if (ic["type"] == "seed_grid") {
      cout << "Adding seed grid initial condition" << endl;
      int Ny = ic["Ny"];
      int Nz = ic["Nz"];
      double X0 = ic["X0"];
      double radius = ic["radius"];
      cout << "Generating " << Ny << " seeds in y dir, " << Nz
           << " seeds in z dir, seed radius " << radius << endl;
      m_simulator.add_initial_conditions(
          make_unique<SeedGrid>(Ny, Nz, X0, radius));
    } else if (ic["type"] == "from_file") {
      cout << "Reading initial condition from file" << endl;
      string filename = ic["filename"];
      cout << "Reading from file: " << filename << endl;
      m_simulator.add_initial_conditions(make_unique<FileReader>(filename));
      int result_counter = ic["result_counter"];
      result_counter += 1;
      m_simulator.set_result_counter(result_counter);
      m_time.set_increment(ic["increment"]);
    } else {
      cout << "Warning: unknown initial condition " << ic["type"] << endl;
    }
  }

  void add_boundary_conditions() {
    cout << "Adding boundary conditions" << endl;
    Params &p = m_model.get_params();
    auto bc = m_settings["boundary_condition"];
    if (bc["type"] == "none") {
      cout << "Not using boundary condition" << endl;
    } else if (bc["type"] == "fixed") {
      cout << "Adding fixed bc" << endl;
      double rho_low = p.n_vap;
      double rho_high = p.n0;
      m_simulator.add_boundary_conditions(
          make_unique<FixedBC>(rho_low, rho_high));
    } else if (bc["type"] == "moving") {
      cout << "Applying moving boundary condition" << endl;
      double rho_low = p.n_vap;
      double rho_high = p.n0;
      unique_ptr<MovingBC> moving_bc = make_unique<MovingBC>(rho_low, rho_high);
      if (bc.contains("width")) {
        double width = bc["width"];
        cout << "Setting boudary condition (half) width to " << width << endl;
        moving_bc->set_xwidth(width);
      }
      if (bc.contains("alpha")) {
        double alpha = bc["alpha"];
        cout << "Setting boundary condition alpha to " << alpha << endl;
        moving_bc->set_alpha(alpha);
      }
      if (bc.contains("disp")) {
        double disp = bc["disp"];
        cout << "Settings boundary condition gap to " << disp << endl;
        moving_bc->set_disp(disp);
      }
      if (bc.contains("initial_position") && bc["initial_position"] == "end") {
        double x_pos = m_world.Lx * m_world.dx - moving_bc->get_xwidth();
        cout << "Setting boundary condition location to " << x_pos << endl;
        moving_bc->set_xpos(x_pos);
      }
      m_simulator.add_boundary_conditions(move(moving_bc));
    } else {
      cout << "Warning: unknown boundary condition " << bc["type"] << endl;
    }
  }

  int main() {
    cout << m_settings.dump(4) << "\n\n";
    cout << "World: " << m_world << endl;

    cout << "Initializing model... " << endl;
    m_model.initialize(m_time.get_dt());

    read_model_parameters();
    read_detailed_timing_configuration();
    add_result_writers();
    add_initial_conditions();
    add_boundary_conditions();

    cout << "Apply initial conditions" << endl;
    m_simulator.apply_initial_conditions();
    if (m_time.get_increment() == 0) {
      cout << "First increment: apply boundary conditions" << endl;
      m_simulator.apply_boundary_conditions();
      m_simulator.write_results();
    }

    while (!m_time.done()) {
      m_time.next(); // increase increment counter by 1
      m_simulator.apply_boundary_conditions();

      double l_steptime = 0.0; // l = local for this mpi process
      double l_fft_time = 0.0;
      MPI_Barrier(m_comm);
      l_steptime = -MPI_Wtime();
      m_model.step(m_time.get_current());
      MPI_Barrier(m_comm);
      l_steptime += MPI_Wtime();
      l_fft_time = m_fft.get_fft_time();

      if (m_detailed_timing) {
        double timing[2] = {l_steptime, l_fft_time};
        MPI_Send(timing, 2, MPI_DOUBLE, 0, 42, m_comm);
        if (m_worker.get_rank() == 0) {
          int num_ranks = m_worker.get_num_ranks();
          double timing[num_ranks][2];
          for (int rank = 0; rank < num_ranks; rank++) {
            MPI_Recv(timing[rank], 2, MPI_DOUBLE, rank, 42, m_comm,
                     MPI_STATUS_IGNORE);
          }
          auto inc = m_time.get_increment();
          if (m_detailed_timing_print) {
            auto old_precision = cout.precision(6);
            cout << "Timing information for all processes:" << endl;
            cout << "step;rank;step_time;fft_time" << endl;
            for (int rank = 0; rank < num_ranks; rank++) {
              cout << inc << ";" << rank << ";" << timing[rank][0] << ";"
                   << timing[rank][1] << endl;
            }
            cout.precision(old_precision);
          }
          if (m_detailed_timing_write) {
            // so we end up to a binary file, and opening with e.g. Python
            // np.fromfile("timing.bin").reshape(n_steps, n_procs, 2)
            ofstream outfile(m_detailed_timing_filename, ios::app);
            outfile.write((const char *)timing, sizeof(double) * 2 * num_ranks);
            outfile.close();
          }
        }
      }

      // max reduction over all mpi processes
      MPI_Reduce(&l_steptime, &m_steptime, 1, MPI_DOUBLE, MPI_MAX, 0, m_comm);
      MPI_Reduce(&l_fft_time, &m_fft_time, 1, MPI_DOUBLE, MPI_MAX, 0, m_comm);

      if (m_time.do_save()) {
        m_simulator.apply_boundary_conditions();
        m_simulator.write_results();
      }

      // Calculate eta from average step time.
      // Use exponential moving average when steps > 3.
      m_avg_steptime = m_steptime;
      if (m_steps_done > 3) {
        m_avg_steptime = 0.01 * m_steptime + 0.99 * m_avg_steptime;
      }
      int increment = m_time.get_increment();
      double t = m_time.get_current(), t1 = m_time.get_t1();
      double eta_i = (t1 - t) / m_time.get_dt();
      double eta_t = eta_i * m_avg_steptime;
      double other_time = m_steptime - m_fft_time;
      cout << "Step " << increment << " done in " << m_steptime << " s ";
      cout << "(" << m_fft_time << " s FFT, " << other_time << " s other). ";
      cout << "Simulation time: " << t << " / " << t1;
      cout << " (" << (t / t1 * 100) << " % done). ";
      cout << "ETA: " << TimeLeft(eta_t) << endl;

      m_total_steptime += m_steptime;
      m_total_fft_time += m_fft_time;
      m_steps_done += 1;
    }

    double avg_steptime = m_total_steptime / m_steps_done;
    double avg_fft_time = m_total_fft_time / m_steps_done;
    double avg_oth_time = avg_steptime - avg_fft_time;
    double p_fft = avg_fft_time / avg_steptime * 100.0;
    double p_oth = avg_oth_time / avg_steptime * 100.0;
    cout << "\nSimulated " << m_steps_done << " steps. Average times:" << endl;
    cout << "Step time:  " << avg_steptime << " s" << endl;
    cout << "FFT time:   " << avg_fft_time << " s / " << p_fft << " %" << endl;
    cout << "Other time: " << avg_oth_time << " s / " << p_oth << " %" << endl;

    return 0;
  }
};

int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(3);
  return App(argc, argv).main();
}
