#include <openpfc/openpfc.hpp>
#include <openpfc/ui.hpp>
#include <openpfc/utils/timeleft.hpp>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

using json = nlohmann::json;
using namespace pfc;
using namespace pfc::utils;
using namespace std;

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
  bool m_first = true;

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
          double alpha2 = 2.0 * params.alpha * params.alpha;
          double lambda2 = 2.0 * params.lambda * params.lambda;
          double fMF = exp(kLap / lambda2);
          double k = sqrt(-kLap) - 1.0;
          double k2 = k * k;

          double rTol = -alpha2 * log(params.alpha_farTol) - 1.0;
          double g1 = 0;
          if (params.alpha_highOrd == 0) { // gaussian peak
            g1 = exp(-k2 / alpha2);
          } else { // quasi-gaussian peak with higher order component to make it
                   // decay faster towards k=0
            g1 = exp(-(k2 + rTol * pow(k, params.alpha_highOrd)) / alpha2);
          }

          // taylor expansion of gaussian peak to order 2
          double g2 = 1.0 - 1.0 / alpha2 * k2;
          // splice the two sides of the peak
          double gf = (k < 0.0) ? g1 : g2;
          // we separate this out because it is needed in the nonlinear
          // calculation when T is not constant in space
          double opPeak = -params.Bx * exp(-params.T / params.T0) * gf;
          // includes the lowest order n_mf term since it is a linear term
          double opCk =
              params.stabP + params.p2_bar + opPeak + params.q2_bar * fMF;

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
      double p3 = params.p3_bar, p4 = params.p4_bar;
      double q3 = params.q3_bar, q4 = params.q4_bar;
      psiN[idx] = p3 * u2 + p4 * u3 + q3 * v2 + q4 * v3;
    }

    // Apply stabilization factor if given in parameters
    if (params.stabP != 0.0)
      for (size_t idx = 0, N = psiN.size(); idx < N; idx++)
        psiN[idx] = psiN[idx] - params.stabP * psi[idx];

    // Fourier transform of the nonlinear part of the evolution equation
    fft.forward(psiN, psiN_F);

    // Apply one step of the evolution equation
    for (size_t idx = 0, N = psi_F.size(); idx < N; idx++)
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];

    // Inverse Fourier transform result back to real space
    fft.backward(psi_F, psi);
  }

}; // end of class

/*
The main application
*/

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
        m_world(ui::from_json<World>(m_settings)),
        m_decomp(Decomposition(m_world, comm)),
        m_fft(FFT(
            m_decomp, comm,
            ui::from_json<heffte::plan_options>(m_settings["plan_options"]))),
        m_time(ui::from_json<Time>(m_settings)), m_model(Tungsten(m_fft)),
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
    if (!m_settings.contains("initial_conditions")) {
      std::cout << "WARNING: no initial conditions are set!" << std::endl;
      return;
    }
    std::cout << "Adding initial conditions" << std::endl;
    for (const json &ic : m_settings["initial_conditions"]) {
      m_simulator.add_initial_conditions(
          ui::from_json<ui::FieldModifier_p>(ic));
    }
  }

  void add_boundary_conditions() {
    if (!m_settings.contains("boundary_conditions")) {
      std::cout << "WARNING: no boundary conditions are set!" << std::endl;
      return;
    }
    std::cout << "Adding boundary conditions" << std::endl;
    for (const json &bc : m_settings["boundary_conditions"]) {
      m_simulator.add_boundary_conditions(
          ui::from_json<ui::FieldModifier_p>(bc));
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

    if (m_settings.contains("simulator")) {
      const json &j = m_settings["simulator"];
      if (j.contains("result_counter")) {
        if (!j["result_counter"].is_number_integer()) {
          throw std::invalid_argument(
              "Invalid JSON input: missing or invalid 'result_counter' field.");
        }
        int result_counter = (int)j["result_counter"] + 1;
        m_simulator.set_result_counter(result_counter);
      }
      if (j.contains("increment")) {
        if (!j["increment"].is_number_integer()) {
          throw std::invalid_argument(
              "Invalid JSON input: missing or invalid 'increment' field.");
        }
        int increment = j["increment"];
        m_time.set_increment(increment);
      }
    }

    cout << "Applying initial conditions" << endl;
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
