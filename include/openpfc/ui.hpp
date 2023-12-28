#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "boundary_conditions/fixed_bc.hpp"
#include "boundary_conditions/moving_bc.hpp"
#include "field_modifier.hpp"
#include "initial_conditions/constant.hpp"
#include "initial_conditions/file_reader.hpp"
#include "initial_conditions/random_seeds.hpp"
#include "initial_conditions/seed_grid.hpp"
#include "initial_conditions/single_seed.hpp"
#include "mpi.hpp"
#include "simulator.hpp"
#include "time.hpp"
#include "utils/timeleft.hpp"
#include "world.hpp"

namespace pfc {
namespace ui {

/*
Functions and classes to construct objects from json file, and other
"ui"-related things.
*/

using json = nlohmann::json;

template <class T> T from_json(const json &settings);

template <> heffte::plan_options from_json<heffte::plan_options>(const json &j) {
  heffte::plan_options options = heffte::default_options<heffte::backend::fftw>();
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
      std::cerr << "Unknown communcation model " << j["reshape_algorithm"] << std::endl;
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
    throw std::invalid_argument("Missing or invalid 'origo' field in JSON input.");
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

void from_json(const json &j, Constant &ic) {
  // Check that the JSON input has the correct type field
  if (!j.contains("type") || j["type"] != "constant") {
    throw std::invalid_argument("Invalid JSON input: missing or incorrect 'type' field.");
  }
  // Check that the JSON input has the required 'n0' field
  if (!j.contains("n0") || !j["n0"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'n0' field.");
  }
  ic.set_density(j["n0"]);
}

void from_json(const json &j, SingleSeed &seed) {
  if (!j.count("type") || j["type"] != "single_seed") {
    throw std::invalid_argument("JSON object does not contain a 'single_seed' type.");
  }

  if (!j.count("amp_eq")) {
    throw std::invalid_argument("JSON object does not contain an 'amp_eq' key.");
  }

  if (!j.count("rho_seed")) {
    throw std::invalid_argument("JSON object does not contain a 'rho_seed' key.");
  }

  seed.set_amplitude(j["amp_eq"]);
  seed.set_density(j["rho_seed"]);
}

void from_json(const json &j, RandomSeeds &ic) {
  // Check that the JSON input has the correct type field
  if (!j.contains("type") || j["type"] != "random_seeds") {
    throw std::invalid_argument("Invalid JSON input: missing or incorrect 'type' field.");
  }

  // Check that the JSON input has the required 'amplitude' field
  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'amplitude' field.");
  }

  // Check that the JSON input has the required 'rho' field
  if (!j.contains("rho") || !j["rho"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'rho' field.");
  }

  ic.set_amplitude(j["amplitude"]);
  ic.set_density(j["rho"]);
}

void from_json(const json &j, SeedGrid &ic) {
  if (!j.contains("type") || j["type"] != "seed_grid") {
    throw std::invalid_argument("Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("Ny") || !j["Ny"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'Ny' field.");
  }

  if (!j.contains("Nz") || !j["Nz"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'Nz' field.");
  }

  if (!j.contains("X0") || !j["X0"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'X0' field.");
  }

  if (!j.contains("radius") || !j["radius"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'radius' field.");
  }

  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'amplitude' field.");
  }

  if (!j.contains("rho") || !j["rho"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'rho' field.");
  }

  ic.set_Ny(j["Ny"]);
  ic.set_Nz(j["Nz"]);
  ic.set_X0(j["X0"]);
  ic.set_radius(j["radius"]);
  ic.set_amplitude(j["amplitude"]);
  ic.set_density(j["rho"]);
}

void from_json(const json &j, FileReader &ic) {
  if (!j.contains("type") || j["type"] != "from_file") {
    throw std::invalid_argument("Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("filename") || !j["filename"].is_string()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'filename' field.");
  }

  ic.set_filename(j["filename"]);
}

void from_json(const json &j, FixedBC &bc) {
  if (!j.contains("type") || j["type"] != "fixed") {
    throw std::invalid_argument("Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("rho_low") || !j["rho_low"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'rho_low' field.");
  }

  if (!j.contains("rho_high") || !j["rho_high"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'rho_high' field.");
  }

  bc.set_rho_low(j["rho_low"]);
  bc.set_rho_high(j["rho_high"]);
}

void from_json(const json &j, MovingBC &bc) {
  if (!j.contains("type") || j["type"] != "moving") {
    throw std::invalid_argument("Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("rho_low") || !j["rho_low"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'rho_low' field.");
  }

  if (!j.contains("rho_high") || !j["rho_high"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'rho_high' field.");
  }

  if (!j.contains("width") || !j["width"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'width' field.");
  }

  if (!j.contains("alpha") || !j["alpha"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'alpha' field.");
  }

  if (!j.contains("disp") || !j["disp"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'disp' field.");
  }

  if (!j.contains("xpos") || !j["xpos"].is_number()) {
    throw std::invalid_argument("Invalid JSON input: missing or invalid 'xpos' field.");
  }

  bc.set_rho_low(j["rho_low"]);
  bc.set_rho_high(j["rho_high"]);
  bc.set_xwidth(j["width"]);
  bc.set_alpha(j["alpha"]);
  bc.set_disp(j["disp"]);
  bc.set_xpos(j["xpos"]);
}

using FieldModifier_p = std::unique_ptr<FieldModifier>;

/**
 * @class FieldModifierRegistry
 * @brief A registry for field modifiers used in the application.
 *
 * The FieldModifierRegistry class provides a centralized registry for field
 * modifiers. It allows registration of field modifiers along with their
 * corresponding creator functions, and provides a way to create instances of
 * field modifiers based on their registered types.
 */
class FieldModifierRegistry {
public:
  using CreatorFunction = std::function<FieldModifier_p(const json &)>;

  /**
   * @brief Get the singleton instance of the FieldModifierRegistry.
   * @return Reference to the singleton instance of FieldModifierRegistry.
   */
  static FieldModifierRegistry &get_instance() {
    static FieldModifierRegistry instance;
    return instance;
  }

  /**
   * @brief Register a field modifier with its creator function.
   * @param type The type string associated with the field modifier.
   * @param creator The creator function that creates an instance of the field
   * modifier.
   */
  void register_modifier(const std::string &type, CreatorFunction creator) { modifiers[type] = creator; }

  /**
   * @brief Create an instance of a field modifier based on its registered type.
   * @param type The type string of the field modifier to create.
   * @param data A json object defining the field modifier parameters.
   * @return Pointer to the created field modifier instance.
   * @throw std::invalid_argument if the specified type is not registered.
   */
  FieldModifier_p create_modifier(const std::string &type, const json &data) {
    auto it = modifiers.find(type);
    if (it != modifiers.end()) {
      return it->second(data);
    }
    throw std::invalid_argument("Unknown FieldModifier type: " + type);
  }

private:
  /**
   * @brief Private constructor to enforce singleton pattern.
   */
  FieldModifierRegistry() {}

  std::unordered_map<std::string, CreatorFunction> modifiers; /**< Map storing the registered field modifiers and their
                                                                 creator functions. */
};

/*
void from_json(const json &, FieldModifier &) {
  std::cout
      << "Warning: This field modifier does not implement reading "
         "parameters from json file. In order to read parameter from json "
         "file, one needs to implement 'void from_json(const json &, FieldModifier &)'"
      << std::endl;
}
*/

void from_json(const json &, Model &) {
  std::cout << "Warning: This model does not implement reading parameters from "
               "json file. In order to read parameters from json file, one needs to "
               "implement 'void from_json(const json &, Model &)'"
            << std::endl;
}

/**
 * @brief Register a field modifier type with the FieldModifierRegistry.
 * @tparam T The type of the field modifier to register.
 * @param type The type string associated with the field modifier.
 *
 * This function registers a field modifier type with the FieldModifierRegistry.
 * It associates the specified type string with a creator function that creates
 * an instance of the field modifier.
 */
template <typename T> void register_field_modifier(const std::string &type) {
  FieldModifierRegistry::get_instance().register_modifier(type, [](const json &params) -> std::unique_ptr<T> {
    std::unique_ptr<T> modifier = std::make_unique<T>();
    from_json(params, *modifier);
    return modifier;
    // return from_json<std::unique_ptr<T>>(j);
  });
}

/**
 * @brief Create an instance of a field modifier based on its type.
 * @param type The type string of the field modifier to create.
 * @param params A json object describing the parameters for field modifier.
 * @return Pointer to the created field modifier instance.
 * @throw std::invalid_argument if the specified type is not registered.
 *
 * This function creates an instance of a field modifier based on its registered
 * type. It retrieves the registered creator function associated with the
 * specified type string from the FieldModifierRegistry and uses it to create
 * the field modifier instance.
 */
std::unique_ptr<FieldModifier> create_field_modifier(const std::string &type, const json &params) {
  return FieldModifierRegistry::get_instance().create_modifier(type, params);
}

/**
 * @struct FieldModifierInitializer
 * @brief Helper struct for registering field modifiers during static
 * initialization.
 *
 * The FieldModifierInitializer struct provides a convenient way to register
 * field modifiers during static initialization by utilizing its constructor.
 * Inside the constructor, various field modifiers can be registered using the
 * `register_field_modifier` function.
 */
struct FieldModifierInitializer {
  /**
   * @brief Constructor for FieldModifierInitializer.
   *
   * This constructor is automatically executed during static initialization.
   * It can be used to register field modifiers by calling the
   * `register_field_modifier` function for each desired field modifier type.
   */
  FieldModifierInitializer() {
    // Initial conditions
    register_field_modifier<Constant>("constant");
    register_field_modifier<SingleSeed>("single_seed");
    register_field_modifier<RandomSeeds>("random_seeds");
    register_field_modifier<SeedGrid>("seed_grid");
    register_field_modifier<FileReader>("from_file");
    // Boundary conditions
    register_field_modifier<FixedBC>("fixed");
    register_field_modifier<MovingBC>("moving");
    // Register other field modifiers here ...
  }
};

static FieldModifierInitializer fieldModifierInitializer; /**< Static instance of FieldModifierInitializer
                                                             to trigger field modifier registration during
                                                             static initialization. */

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
      : m_comm(comm), m_worker(MPI_Worker(argc, argv, comm)), rank0(m_worker.get_rank() == 0),
        m_settings(read_settings(argc, argv)) {}

  App(const json &settings, MPI_Comm comm = MPI_COMM_WORLD)
      : m_comm(comm), m_worker(MPI_Worker(0, nullptr, comm)), rank0(m_worker.get_rank() == 0), m_settings(settings) {}

  bool create_results_dir(const std::string &output) {
    std::filesystem::path results_dir(output);
    if (results_dir.has_filename()) results_dir = results_dir.parent_path();
    if (!std::filesystem::exists(results_dir)) {
      std::cout << "Results dir " << results_dir << " does not exist, creating\n";
      std::filesystem::create_directories(results_dir);
      return true;
    } else {
      std::cout << "Warning: results dir " << results_dir << " already exists\n";
      return false;
    }
  }

  void read_detailed_timing_configuration() {
    if (m_settings.contains("detailed_timing")) {
      auto timing = m_settings["detailed_timing"];
      if (timing.contains("enabled")) m_detailed_timing = timing["enabled"];
      if (timing.contains("print")) m_detailed_timing_print = timing["print"];
      if (timing.contains("write")) m_detailed_timing_write = timing["write"];
      if (timing.contains("filename")) m_detailed_timing_filename = timing["filename"];
    }
  }

  void add_result_writers(Simulator &sim) {
    std::cout << "Adding results writers" << std::endl;
    if (m_settings.contains("saveat") && m_settings.contains("fields") && m_settings["saveat"] > 0) {
      for (const auto &field : m_settings["fields"]) {
        std::string name = field["name"];
        std::string data = field["data"];
        if (rank0) create_results_dir(data);
        std::cout << "Writing field " << name << " to " << data << std::endl;
        sim.add_results_writer(name, std::make_unique<BinaryWriter>(data));
      }
    } else {
      std::cout << "Warning: not writing results to anywhere." << std::endl;
      std::cout << "To write results, add ResultsWriter to model." << std::endl;
    }
  }

  void add_initial_conditions(Simulator &sim) {
    if (!m_settings.contains("initial_conditions")) {
      std::cout << "WARNING: no initial conditions are set!" << std::endl;
      return;
    }
    std::cout << "Adding initial conditions" << std::endl;
    for (const json &params : m_settings["initial_conditions"]) {
      std::cout << "Creating initial condition from data " << params << std::endl;
      std::string type = params["type"];
      sim.add_initial_conditions(create_field_modifier(type, params));
    }
  }

  void add_boundary_conditions(Simulator &sim) {
    if (!m_settings.contains("boundary_conditions")) {
      std::cout << "WARNING: no boundary conditions are set!" << std::endl;
      return;
    }
    std::cout << "Adding boundary conditions" << std::endl;
    for (const json &params : m_settings["boundary_conditions"]) {
      std::cout << "Creating boundary condition from data " << params << std::endl;
      std::string type = params["type"];
      sim.add_boundary_conditions(create_field_modifier(type, params));
    }
  }

  int main() {
    std::cout << "Reading configuration from json file:" << std::endl;
    std::cout << m_settings.dump(4) << "\n\n";

    World world(ui::from_json<World>(m_settings));
    std::cout << "World: " << world << std::endl;

    Decomposition decomp(world, m_comm);
    auto plan_options = ui::from_json<heffte::plan_options>(m_settings["plan_options"]);
    FFT fft(decomp, m_comm, plan_options);
    Time time(ui::from_json<Time>(m_settings));
    ConcreteModel model;
    model.set_fft(fft);
    Simulator simulator(model, time);

    if (m_settings.contains("model") && m_settings["model"].contains("params")) {
      from_json(m_settings["model"]["params"], model);
    }
    read_detailed_timing_configuration();

    std::cout << "Initializing model... " << std::endl;
    model.initialize(time.get_dt());

    add_result_writers(simulator);
    add_initial_conditions(simulator);
    add_boundary_conditions(simulator);

    if (m_settings.contains("simulator")) {
      const json &j = m_settings["simulator"];
      if (j.contains("result_counter")) {
        if (!j["result_counter"].is_number_integer()) {
          throw std::invalid_argument("Invalid JSON input: missing or invalid 'result_counter' field.");
        }
        int result_counter = (int)j["result_counter"] + 1;
        simulator.set_result_counter(result_counter);
      }
      if (j.contains("increment")) {
        if (!j["increment"].is_number_integer()) {
          throw std::invalid_argument("Invalid JSON input: missing or invalid 'increment' field.");
        }
        int increment = j["increment"];
        time.set_increment(increment);
      }
    }

    std::cout << "Applying initial conditions" << std::endl;
    simulator.apply_initial_conditions();
    if (time.get_increment() == 0) {
      std::cout << "First increment: apply boundary conditions" << std::endl;
      simulator.apply_boundary_conditions();
      simulator.write_results();
    }

    while (!time.done()) {
      time.next(); // increase increment counter by 1
      simulator.apply_boundary_conditions();

      double l_steptime = 0.0; // l = local for this mpi process
      double l_fft_time = 0.0;
      MPI_Barrier(m_comm);
      l_steptime = -MPI_Wtime();
      model.step(time.get_current());
      MPI_Barrier(m_comm);
      l_steptime += MPI_Wtime();
      l_fft_time = fft.get_fft_time();

      if (m_detailed_timing) {
        double timing[2] = {l_steptime, l_fft_time};
        MPI_Send(timing, 2, MPI_DOUBLE, 0, 42, m_comm);
        if (m_worker.get_rank() == 0) {
          int num_ranks = m_worker.get_num_ranks();
          double timing[num_ranks][2];
          for (int rank = 0; rank < num_ranks; rank++) {
            MPI_Recv(timing[rank], 2, MPI_DOUBLE, rank, 42, m_comm, MPI_STATUS_IGNORE);
          }
          auto inc = time.get_increment();
          if (m_detailed_timing_print) {
            auto old_precision = std::cout.precision(6);
            std::cout << "Timing information for all processes:" << std::endl;
            std::cout << "step;rank;step_time;fft_time" << std::endl;
            for (int rank = 0; rank < num_ranks; rank++) {
              std::cout << inc << ";" << rank << ";" << timing[rank][0] << ";" << timing[rank][1] << std::endl;
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

      if (time.do_save()) {
        simulator.apply_boundary_conditions();
        simulator.write_results();
      }

      // Calculate eta from average step time.
      // Use exponential moving average when steps > 3.
      m_avg_steptime = m_steptime;
      if (m_steps_done > 3) {
        m_avg_steptime = 0.01 * m_steptime + 0.99 * m_avg_steptime;
      }
      int increment = time.get_increment();
      double t = time.get_current(), t1 = time.get_t1();
      double eta_i = (t1 - t) / time.get_dt();
      double eta_t = eta_i * m_avg_steptime;
      double other_time = m_steptime - m_fft_time;
      std::cout << "Step " << increment << " done in " << m_steptime << " s ";
      std::cout << "(" << m_fft_time << " s FFT, " << other_time << " s other). ";
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
    std::cout << "\nSimulated " << m_steps_done << " steps. Average times:" << std::endl;
    std::cout << "Step time:  " << avg_steptime << " s" << std::endl;
    std::cout << "FFT time:   " << avg_fft_time << " s / " << p_fft << " %" << std::endl;
    std::cout << "Other time: " << avg_oth_time << " s / " << p_oth << " %" << std::endl;

    return 0;
  }
};

} // namespace ui
} // namespace pfc
