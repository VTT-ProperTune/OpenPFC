#include <nlohmann/json.hpp>
#include <pfc/model.hpp>
#include <pfc/results_writer.hpp>
#include <pfc/simulator.hpp>
#include <pfc/time.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace pfc;
using namespace std;

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

    Decomposition &decomp = get_decomposition();
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
    double rho_seed = p.n_sol;
    double A_phi = 135.0 * p.p4_bar;
    double B_phi = 16.0 * p.p3_bar + 48.0 * p.p4_bar * rho_seed;
    double C_phi = -6.0 * (p.Bx * exp(-p.T / p.T0)) + 6.0 * p.p2_bar +
                   12.0 * p.p3_bar * rho_seed +
                   18.0 * p.p4_bar * pow(rho_seed, 2);
    double d = std::abs(9.0 * pow(B_phi, 2) - 32.0 * A_phi * C_phi);
    double amp_eq = (-3.0 * B_phi + sqrt(d)) / (8.0 * A_phi);

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
    allocate();
    prepare_operators(dt);
    // prepare_initial_condition();
  }

  void step(double) override {

    FFT &fft = get_fft();

    // Calculate mean-field density n_mf
    fft.forward(psi, psi_F);
    for (long int idx = 0, N = psiMF_F.size(); idx < N; idx++)
      psiMF_F[idx] = filterMF[idx] * psi_F[idx];
    fft.backward(psiMF_F, psiMF);

    // Calculate the nonlinear part of the evolution equation in a real space
    for (long int idx = 0, N = psiN.size(); idx < N; idx++) {
      double u = psi[idx];
      double v = psiMF[idx];
      psiN[idx] = p.p3_bar * u * u + p.p4_bar * u * u * u + p.q3_bar * v * v +
                  p.q4_bar * v * v * v;
    }

    // Apply stabilization factor if given in parameters
    if (p.stabP != 0.0)
      for (long int idx = 0, N = psiN.size(); idx < N; idx++)
        psiN[idx] = psiN[idx] - p.stabP * psi[idx];

    // Fourier transform of the nonlinear part of the evolution equation
    fft.forward(psiN, psiN_F);

    // Apply one step of the evolution equation
    for (long int idx = 0, N = psi_F.size(); idx < N; idx++)
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];

    // Inverse Fourier transform result back to real space
    fft.backward(psi_F, psi);
  }

  Field &get_field() {
    return psi;
  }

}; // end of class

class SingleSeed : public FieldModifier {

private:
  params p;

public:
  void apply(Model &m, double) override {
    World &w = m.get_world();
    Decomposition &decomp = m.get_decomposition();
    Field &f = m.get_field();

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
    double rho_seed = p.n_sol;
    double A_phi = 135.0 * p.p4_bar;
    double B_phi = 16.0 * p.p3_bar + 48.0 * p.p4_bar * rho_seed;
    double C_phi = -6.0 * (p.Bx * exp(-p.T / p.T0)) + 6.0 * p.p2_bar +
                   12.0 * p.p3_bar * rho_seed +
                   18.0 * p.p4_bar * pow(rho_seed, 2);
    double d = std::abs(9.0 * pow(B_phi, 2) - 32.0 * A_phi * C_phi);
    double amp_eq = (-3.0 * B_phi + sqrt(d)) / (8.0 * A_phi);

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
            u = rho_seed;
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

template <>
unique_ptr<BinaryWriter>
from_json<unique_ptr<BinaryWriter>>(const json &settings) {
  return make_unique<BinaryWriter>(settings["results"]);
}

/*
The main application
*/
class App {
private:
  nlohmann::json m_settings;
  bool rank0;

public:
  App(const json &settings)
      : m_settings(settings), rank0(mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {}

  void check_results_dir() {
    filesystem::path results_dir(m_settings["results"].get<string>());
    if (results_dir.has_filename()) results_dir = results_dir.parent_path();
    if (rank0 && !std::filesystem::exists(results_dir)) {
      cout << "Results dir " << results_dir << " does not exist, creating\n";
      filesystem::create_directories(results_dir);
    }
  }

  void run() {
    if (rank0) cout << m_settings.dump(4) << "\n\n";
    World world = from_json<World>(m_settings);
    cout << "World: " << world << endl;
    Decomposition decomposition(world, MPI_COMM_WORLD);
    FFT fft(decomposition, MPI_COMM_WORLD);
    Tungsten model(world, decomposition, fft);
    Time time = from_json<Time>(m_settings);
    Simulator simulator(world, decomposition, fft, model, time);

    simulator.add_results_writer(
        from_json<unique_ptr<BinaryWriter>>(m_settings));
    if (rank0) check_results_dir();

    simulator.add_initial_conditions(make_unique<SingleSeed>());

    while (!simulator.done()) {
      simulator.step();
      cout << "Step " << time.get_increment() << " done" << endl;
    }
  }
};

int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(3);
  MPI_Init(&argc, &argv);

  // read settings from file if or standard input
  int me = mpi::get_comm_rank(MPI_COMM_WORLD);
  json settings;
  if (argc > 1) {
    if (me == 0) cout << "Reading input from file " << argv[1] << "\n\n";
    filesystem::path file(argv[1]);
    if (!filesystem::exists(file)) {
      if (me == 0) cerr << "File " << file << " does not exist!\n";
      return 1;
    }
    std::ifstream input_file(file);
    input_file >> settings;
  } else {
    if (me == 0) std::cout << "Reading simulation settings from stdin\n\n";
    std::cin >> settings;
  }

  // construct application from settings and run
  App(settings).run();

  MPI_Finalize();
  return 0;
}
