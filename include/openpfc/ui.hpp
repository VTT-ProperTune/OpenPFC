#pragma once

#include "boundary_conditions/fixed_bc.hpp"
#include "boundary_conditions/moving_bc.hpp"
#include "field_modifier.hpp"
#include "initial_conditions/constant.hpp"
#include "initial_conditions/file_reader.hpp"
#include "initial_conditions/random_seeds.hpp"
#include "initial_conditions/seed_grid.hpp"
#include "initial_conditions/single_seed.hpp"
#include "time.hpp"
#include "utils/timeleft.hpp"
#include "world.hpp"
#include "mpi.hpp"
#include "simulator.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace pfc {
namespace ui {

/*
Functions and classes to construct objects from json file, and other
"ui"-related things.
*/

using json = nlohmann::json;

template <class T> T from_json(const json &settings);

template <>
heffte::plan_options from_json<heffte::plan_options>(const json &j) {
  heffte::plan_options options =
      heffte::default_options<heffte::backend::fftw>();
  if (j.contains("use_reorder")) {
    options.use_reorder = j["use_reorder"];
  }
  if (j.contains("reshape_algorithm")) {
    if (j["reshape_algorithm"] == "alltoall") {
      options.algorithm = heffte::reshape_algorithm::alltoall;
    } else if (j["reshape_algorithm"] == "alltoallv") {
      options.algorithm = heffte::reshape_algorithm::alltoallv;
    } else if (j["reshape_algorithm"] == "p2p") {
      options.algorithm = heffte::reshape_algorithm::p2p;
    } else if (j["reshape_algorithm"] == "p2p_plined") {
      options.algorithm = heffte::reshape_algorithm::p2p_plined;
    } else {
      std::cerr << "Unknown communcation model " << j["reshape_algorithm"]
                << std::endl;
    }
  }
  if (j.contains("use_pencils")) {
    options.use_pencils = j["use_pencils"];
  }
  if (j.contains("use_gpu_aware")) {
    options.use_gpu_aware = j["use_gpu_aware"];
  }
  std::cout << "backend options: " << options << "\n\n";
  return options;
}

/**
 * Creates a World object from a JSON input.
 *
 * @param j A JSON object containing the following fields:
 *   - Lx (int): The number of grid points in the x direction.
 *   - Ly (int): The number of grid points in the y direction.
 *   - Lz (int): The number of grid points in the z direction.
 *   - dx (float): The grid spacing in the x direction.
 *   - dy (float): The grid spacing in the y direction.
 *   - dz (float): The grid spacing in the z direction.
 *   - origo (string): The origin of the coordinate system. Must be one of
 *     "center" or "corner".
 *
 * @return A World object.
 *
 * @throws std::invalid_argument if any of the required fields are missing
 *         or have an invalid value.
 */
template <> World from_json<World>(const json &j) {
  int Lx = 0, Ly = 0, Lz = 0;
  double dx = 0.0, dy = 0.0, dz = 0.0;
  double x0 = 0.0, y0 = 0.0, z0 = 0.0;
  std::string origo;

  if (!j.count("Lx") || !j["Lx"].is_number_integer()) {
    throw std::invalid_argument("Missing or invalid 'Lx' field in JSON input.");
  }
  Lx = j["Lx"];

  if (!j.count("Ly") || !j["Ly"].is_number_integer()) {
    throw std::invalid_argument("Missing or invalid 'Ly' field in JSON input.");
  }
  Ly = j["Ly"];

  if (!j.count("Lz") || !j["Lz"].is_number_integer()) {
    throw std::invalid_argument("Missing or invalid 'Lz' field in JSON input.");
  }
  Lz = j["Lz"];

  if (!j.count("dx") || !j["dx"].is_number_float()) {
    throw std::invalid_argument("Missing or invalid 'dx' field in JSON input.");
  }
  dx = j["dx"];

  if (!j.count("dy") || !j["dy"].is_number_float()) {
    throw std::invalid_argument("Missing or invalid 'dy' field in JSON input.");
  }
  dy = j["dy"];

  if (!j.count("dz") || !j["dz"].is_number_float()) {
    throw std::invalid_argument("Missing or invalid 'dz' field in JSON input.");
  }
  dz = j["dz"];

  if (!j.count("origo") || !j["origo"].is_string()) {
    throw std::invalid_argument(
        "Missing or invalid 'origo' field in JSON input.");
  }
  origo = j["origo"];

  if (origo != "center" && origo != "corner") {
    throw std::invalid_argument("Invalid 'origo' field in JSON input.");
  }

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

using Constant_p = std::unique_ptr<Constant>;

/**
 * A factory function that creates a `std::unique_ptr` to a `Constant` object
 * from a JSON input.
 *
 * The JSON input must have the following format:
 *
 * ```
 * {
 *   "type": "constant",
 *   "n0": <double>
 * }
 * ```
 *
 * where `<double>` is a floating-point number that represents the constant
 * value to set the field to.
 *
 * If the JSON input is invalid or missing required fields, an
 * `std::invalid_argument` exception is thrown.
 *
 * Example usage:
 *
 * ```
 * // Create a constant field modifier with value 1.0 from JSON input
 * json input = {
 *   {"type", "constant"},
 *   {"n0", 1.0}
 * };
 * Constant_p c = from_json<Constant_p>(input);
 * ```
 */
template <> Constant_p from_json<Constant_p>(const json &j) {

  // Check that the JSON input has the correct type field
  if (!j.contains("type") || j["type"] != "constant") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }

  // Check that the JSON input has the required 'n0' field
  if (!j.contains("n0") || !j["n0"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'n0' field.");
  }

  double n0 = j["n0"];
  return std::make_unique<Constant>(n0);
}

using SingleSeed_p = std::unique_ptr<SingleSeed>;

/**
 * Convert a JSON object to a unique_ptr to a SingleSeed object.
 *
 * This function converts a JSON object to a unique_ptr to a SingleSeed object.
 * The JSON object must contain the following fields:
 *   - type: A string with the value "single_seed".
 *   - amp_eq: A double with the amplitude of the equalizer.
 *   - rho_seed: A double with the seed value for the Rho function.
 *
 * If any of these fields are missing or have an invalid type, this function
 * throws a std::invalid_argument exception.
 *
 * @tparam T The type of the deleter object to use for the unique_ptr.
 * @param j The JSON object to convert.
 * @return A unique_ptr to a SingleSeed object.
 * @throws std::invalid_argument if any required fields are missing or have an
 * invalid type.
 */
template <> SingleSeed_p from_json<SingleSeed_p>(const json &j) {

  if (!j.count("type") || j["type"] != "single_seed") {
    throw std::invalid_argument(
        "JSON object does not contain a 'single_seed' type.");
  }

  if (!j.count("amp_eq")) {
    throw std::invalid_argument(
        "JSON object does not contain an 'amp_eq' key.");
  }

  if (!j.count("rho_seed")) {
    throw std::invalid_argument(
        "JSON object does not contain a 'rho_seed' key.");
  }

  std::unique_ptr<SingleSeed> seed = std::make_unique<SingleSeed>();
  seed->amp_eq = j["amp_eq"];
  seed->rho_seed = j["rho_seed"];
  return seed;
}

using RandomSeeds_p = std::unique_ptr<RandomSeeds>;

/**
 * Parses a JSON object to create a unique pointer to a `RandomSeeds` object.
 *
 * The input JSON object should have the following fields:
 * - "type": A string indicating the type of initial condition. Must be
 * "random_seeds".
 * - "amplitude": A number indicating the amplitude of the random seeds.
 * - "rho": A number indicating the radius of the random seeds.
 *
 * Throws an `invalid_argument` exception if the input JSON object does not have
 * the required fields or has incorrect data types.
 */
template <> RandomSeeds_p from_json<RandomSeeds_p>(const json &j) {

  // Check that the JSON input has the correct type field
  if (!j.contains("type") || j["type"] != "random_seeds") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }

  // Check that the JSON input has the required 'amplitude' field
  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'amplitude' field.");
  }

  // Check that the JSON input has the required 'rho' field
  if (!j.contains("rho") || !j["rho"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho' field.");
  }

  RandomSeeds_p ic = std::make_unique<RandomSeeds>();
  ic->amplitude = j["amplitude"];
  ic->rho = j["rho"];
  return ic;
}

using SeedGrid_p = std::unique_ptr<SeedGrid>;

template <> SeedGrid_p from_json<SeedGrid_p>(const json &j) {

  if (!j.contains("type") || j["type"] != "seed_grid") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("Ny") || !j["Ny"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'Ny' field.");
  }

  if (!j.contains("Nz") || !j["Nz"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'Nz' field.");
  }

  if (!j.contains("X0") || !j["X0"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'X0' field.");
  }

  if (!j.contains("radius") || !j["radius"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'radius' field.");
  }

  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'amplitude' field.");
  }

  if (!j.contains("rho") || !j["rho"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho' field.");
  }

  int Ny = j["Ny"];
  int Nz = j["Nz"];
  double X0 = j["X0"];
  double radius = j["radius"];
  SeedGrid_p ic = std::make_unique<SeedGrid>(Ny, Nz, X0, radius);
  ic->amplitude = j["amplitude"];
  ic->rho = j["rho"];
  return ic;
}

using FileReader_p = std::unique_ptr<FileReader>;

template <> FileReader_p from_json<FileReader_p>(const json &j) {

  if (!j.contains("type") || j["type"] != "from_file") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("filename") || !j["filename"].is_string()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'filename' field.");
  }

  std::string filename = j["filename"];
  FileReader_p ic = std::make_unique<FileReader>(filename);
  return ic;
}

using FixedBC_p = std::unique_ptr<FixedBC>;

template <> FixedBC_p from_json<FixedBC_p>(const json &j) {

  if (!j.contains("type") || j["type"] != "fixed") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("rho_low") || !j["rho_low"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_low' field.");
  }

  if (!j.contains("rho_high") || !j["rho_high"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_high' field.");
  }

  double rho_low = j["rho_low"];
  double rho_high = j["rho_high"];
  return std::make_unique<FixedBC>(rho_low, rho_high);
}

using MovingBC_p = std::unique_ptr<MovingBC>;

template <> MovingBC_p from_json<MovingBC_p>(const json &j) {

  if (!j.contains("type") || j["type"] != "moving") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("rho_low") || !j["rho_low"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_low' field.");
  }

  if (!j.contains("rho_high") || !j["rho_high"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_high' field.");
  }

  if (!j.contains("width") || !j["width"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'width' field.");
  }

  if (!j.contains("alpha") || !j["alpha"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'alpha' field.");
  }

  if (!j.contains("disp") || !j["disp"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'disp' field.");
  }

  if (!j.contains("xpos") || !j["xpos"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'xpos' field.");
  }

  double rho_low = j["rho_low"];
  double rho_high = j["rho_high"];
  double width = j["width"];
  double alpha = j["alpha"];
  double disp = j["disp"];
  double xpos = j["xpos"];
  MovingBC_p bc = std::make_unique<MovingBC>(rho_low, rho_high);
  bc->set_xwidth(width);
  bc->set_alpha(alpha);
  bc->set_disp(disp);
  bc->set_xpos(xpos);
  return bc;
}

using FieldModifier_p = std::unique_ptr<FieldModifier>;

template <> FieldModifier_p from_json<FieldModifier_p>(const json &j) {
  std::cout << "Creating FieldModifier from data " << j << std::endl;
  std::string type = j["type"];
  // Initial conditions
  if (type == "single_seed") {
    std::cout << "Creating SingleSeed <: FieldModifier" << std::endl;
    return from_json<SingleSeed_p>(j);
  }
  if (type == "constant") {
    std::cout << "Creating Constant <: FieldModifier" << std::endl;
    return from_json<Constant_p>(j);
  }
  if (type == "random_seeds") {
    std::cout << "Creating RandomSeeds <: FieldModifier" << std::endl;
    return from_json<RandomSeeds_p>(j);
  }
  if (type == "seed_grid") {
    std::cout << "Creating SeedGrid <: FieldModifier" << std::endl;
    return from_json<SeedGrid_p>(j);
  }
  if (type == "from_file") {
    std::cout << "Creating FileReader <: FieldModifier" << std::endl;
    return from_json<FileReader_p>(j);
  }
  // Boundary conditions
  if (type == "fixed") {
    std::cout << "Creating FixedBC <: FieldModifier" << std::endl;
    return from_json<FixedBC_p>(j);
  }
  if (type == "moving") {
    std::cout << "Creating MovingBC <: FieldModifier" << std::endl;
    return from_json<MovingBC_p>(j);
  }
  throw std::invalid_argument("Unknown FieldModifier type: " + type);
}

/**
 * @brief The main json-based application
 *
 */
template <class ConcreteModel> class App {
private:
  MPI_Comm m_comm;
  MPI_Worker m_worker;
  bool rank0;
  json m_settings;
  World m_world;
  Decomposition m_decomp;
  FFT m_fft;
  Time m_time;
  ConcreteModel m_model;
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
  std::string m_detailed_timing_filename = "timing.bin";

  // read settings from file if or standard input
  json read_settings(int argc, char *argv[]) {
    json settings;
    if (argc > 1) {
      if (rank0) std::cout << "Reading input from file " << argv[1] << "\n\n";
      std::filesystem::path file(argv[1]);
      if (!std::filesystem::exists(file)) {
        if (rank0) std::cerr << "File " << file << " does not exist!\n";
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
        m_time(ui::from_json<Time>(m_settings)), m_model(ConcreteModel(m_fft)),
        m_simulator(Simulator(m_model, m_time)) {}

  bool create_results_dir(const std::string &output) {
    std::filesystem::path results_dir(output);
    if (results_dir.has_filename()) results_dir = results_dir.parent_path();
    if (!std::filesystem::exists(results_dir)) {
      std::cout << "Results dir " << results_dir
                << " does not exist, creating\n";
      std::filesystem::create_directories(results_dir);
      return true;
    } else {
      std::cout << "Warning: results dir " << results_dir
                << " already exists\n";
      return false;
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
    std::cout << "Adding results writers" << std::endl;
    if (m_settings.contains("saveat") && m_settings.contains("fields") &&
        m_settings["saveat"] > 0) {
      for (const auto &field : m_settings["fields"]) {
        std::string name = field["name"];
        std::string data = field["data"];
        if (rank0) create_results_dir(data);
        std::cout << "Writing field " << name << " to " << data << std::endl;
        m_simulator.add_results_writer(name,
                                       std::make_unique<BinaryWriter>(data));
      }
    } else {
      std::cout << "Warning: not writing results to anywhere." << std::endl;
      std::cout << "To write results, add ResultsWriter to model." << std::endl;
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
    std::cout << m_settings.dump(4) << "\n\n";
    std::cout << "World: " << m_world << std::endl;

    std::cout << "Initializing model... " << std::endl;
    m_model.initialize(m_time.get_dt());

    if (m_settings.contains("model") &&
        m_settings["model"].contains("params")) {
      from_json(m_settings["model"]["params"], m_model);
    }
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

    std::cout << "Applying initial conditions" << std::endl;
    m_simulator.apply_initial_conditions();
    if (m_time.get_increment() == 0) {
      std::cout << "First increment: apply boundary conditions" << std::endl;
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
            auto old_precision = std::cout.precision(6);
            std::cout << "Timing information for all processes:" << std::endl;
            std::cout << "step;rank;step_time;fft_time" << std::endl;
            for (int rank = 0; rank < num_ranks; rank++) {
              std::cout << inc << ";" << rank << ";" << timing[rank][0] << ";"
                        << timing[rank][1] << std::endl;
            }
            std::cout.precision(old_precision);
          }
          if (m_detailed_timing_write) {
            // so we end up to a binary file, and opening with e.g. Python
            // np.fromfile("timing.bin").reshape(n_steps, n_procs, 2)
            std::ofstream outfile(m_detailed_timing_filename, std::ios::app);
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
      std::cout << "Step " << increment << " done in " << m_steptime << " s ";
      std::cout << "(" << m_fft_time << " s FFT, " << other_time
                << " s other). ";
      std::cout << "Simulation time: " << t << " / " << t1;
      std::cout << " (" << (t / t1 * 100) << " % done). ";
      std::cout << "ETA: " << pfc::utils::TimeLeft(eta_t) << std::endl;

      m_total_steptime += m_steptime;
      m_total_fft_time += m_fft_time;
      m_steps_done += 1;
    }

    double avg_steptime = m_total_steptime / m_steps_done;
    double avg_fft_time = m_total_fft_time / m_steps_done;
    double avg_oth_time = avg_steptime - avg_fft_time;
    double p_fft = avg_fft_time / avg_steptime * 100.0;
    double p_oth = avg_oth_time / avg_steptime * 100.0;
    std::cout << "\nSimulated " << m_steps_done
              << " steps. Average times:" << std::endl;
    std::cout << "Step time:  " << avg_steptime << " s" << std::endl;
    std::cout << "FFT time:   " << avg_fft_time << " s / " << p_fft << " %"
              << std::endl;
    std::cout << "Other time: " << avg_oth_time << " s / " << p_oth << " %"
              << std::endl;

    return 0;
  }
};

} // namespace ui
} // namespace pfc
