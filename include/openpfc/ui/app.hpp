// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/app.hpp
 * @brief Application class for running simulations from JSON/TOML configuration
 *
 * @details
 * This header provides the App template class that orchestrates the entire
 * simulation workflow from configuration files. It handles:
 * - Reading JSON/TOML configuration files
 * - Creating and configuring the simulation world
 * - Setting up initial and boundary conditions
 * - Running the simulation loop
 * - Writing results
 * - Performance timing and reporting
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UI_APP_HPP
#define PFC_UI_APP_HPP

#include "field_modifier_registry.hpp"
#include "from_json.hpp"
#include "json_helpers.hpp"
#include "openpfc/openpfc.hpp"
#include "openpfc/utils/timeleft.hpp"
#include "openpfc/utils/toml_to_json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <toml++/toml.hpp>

namespace pfc {
namespace ui {

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

  // read settings from file (JSON or TOML format)
  json read_settings(int argc, char *argv[]) {
    if (argc <= 1) {
      if (rank0) {
        std::cerr << "Error: Configuration file required.\n";
        std::cerr << "Usage: " << argv[0] << " <config.json|config.toml>\n";
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::filesystem::path file(argv[1]);
    if (!std::filesystem::exists(file)) {
      if (rank0) std::cerr << "Error: File " << file << " does not exist!\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    auto ext = file.extension().string();
    json settings;

    if (ext == ".toml") {
      if (rank0) std::cout << "Reading TOML configuration from " << file << "\n\n";
      try {
        auto toml_data = toml::parse_file(file.string());
        settings = utils::toml_to_json(toml_data);
      } catch (const toml::parse_error &err) {
        if (rank0) {
          std::cerr << "Error parsing TOML file: " << err.description() << "\n";
          std::cerr << "  at line " << err.source().begin.line << ", column "
                    << err.source().begin.column << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    } else if (ext == ".json") {
      if (rank0) std::cout << "Reading JSON configuration from " << file << "\n\n";
      std::ifstream input_file(file);
      try {
        input_file >> settings;
      } catch (const nlohmann::json::parse_error &err) {
        if (rank0) {
          std::cerr << "Error parsing JSON file: " << err.what() << "\n";
          std::cerr << "  at byte position " << err.byte << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    } else {
      if (rank0) {
        std::cerr << "Error: Unsupported file format: " << ext << "\n";
        std::cerr << "Supported formats: .json, .toml\n";
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return settings;
  }

public:
  App(int argc, char *argv[], MPI_Comm comm = MPI_COMM_WORLD)
      : m_comm(comm), m_worker(MPI_Worker(argc, argv, comm)),
        rank0(m_worker.get_rank() == 0), m_settings(read_settings(argc, argv)) {}

  App(const json &settings, MPI_Comm comm = MPI_COMM_WORLD)
      : m_comm(comm), m_worker(MPI_Worker(0, nullptr, comm)),
        rank0(m_worker.get_rank() == 0), m_settings(settings) {}

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
      if (timing.contains("filename"))
        m_detailed_timing_filename = timing["filename"];
    }
  }

  void add_result_writers(Simulator &sim) {
    std::cout << "Adding results writers" << std::endl;
    if (m_settings.contains("saveat") && m_settings.contains("fields") &&
        m_settings["saveat"] > 0) {
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
      if (!params.contains("type")) {
        std::cout << "Warning: no type is set for initial condition!" << std::endl;
        continue;
      }
      std::string type = params["type"];
      auto field_modifier = create_field_modifier(type, params);
      if (!params.contains("target")) {
        std::cout << "Warning: no target is set for initial condition! Using "
                     "target 'default'"
                  << std::endl;
      } else {
        std::string target = params["target"];
        std::cout << "Setting initial condition target to " << target << std::endl;
        field_modifier->set_field_name(target);
      }
      sim.add_initial_conditions(std::move(field_modifier));
    }
  }

  void add_boundary_conditions(Simulator &sim) {
    if (!m_settings.contains("boundary_conditions")) {
      std::cout << "Warning: no boundary conditions are set!" << std::endl;
      return;
    }
    std::cout << "Adding boundary conditions" << std::endl;
    for (const json &params : m_settings["boundary_conditions"]) {
      std::cout << "Creating boundary condition from data " << params << std::endl;
      if (!params.contains("type")) {
        std::cout << "Warning: no type is set for initial condition!" << std::endl;
        continue;
      }
      std::string type = params["type"];
      auto field_modifier = create_field_modifier(type, params);
      if (!params.contains("target")) {
        std::cout << "Warning: no target is set for boundary condition! Using "
                     "target 'default'"
                  << std::endl;
      } else {
        std::string target = params["target"];
        std::cout << "Setting boundary condition target to " << target << std::endl;
        field_modifier->set_field_name(target);
      }
      sim.add_boundary_conditions(std::move(field_modifier));
    }
  }

  int main() {
    std::cout << "Reading configuration from json file:" << std::endl;
    std::cout << m_settings.dump(4) << "\n\n";

    World world(ui::from_json<World>(m_settings));
    std::cout << "World: " << world << std::endl;

    int num_ranks = m_worker.get_num_ranks();
    int rank_id = m_worker.get_rank();
    auto decomp = decomposition::create(world, num_ranks);
    auto options = ui::from_json<heffte::plan_options>(m_settings["plan_options"]);
    auto fft_layout = fft::layout::create(decomp, 0);
    auto fft = fft::create(fft_layout, rank_id, options);
    Time time(ui::from_json<Time>(m_settings));
    ConcreteModel model(world);
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
          throw std::invalid_argument(
              "Invalid JSON input: missing or invalid 'result_counter' field.");
        }
        int result_counter = (int)j["result_counter"] + 1;
        simulator.set_result_counter(result_counter);
      }
      if (j.contains("increment")) {
        if (!j["increment"].is_number_integer()) {
          throw std::invalid_argument(
              "Invalid JSON input: missing or invalid 'increment' field.");
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
      step(simulator, model);
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
            MPI_Recv(timing[rank], 2, MPI_DOUBLE, rank, 42, m_comm,
                     MPI_STATUS_IGNORE);
          }
          auto inc = time.get_increment();
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

#endif // PFC_UI_APP_HPP
