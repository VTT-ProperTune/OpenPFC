// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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
 * - Performance timing and reporting (kernel/profiling, see
 * docs/performance_profiling.md)
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UI_APP_HPP
#define PFC_UI_APP_HPP

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/frontend/ui/json_helpers.hpp>
#include <openpfc/frontend/utils/memory_reporter.hpp>
#include <openpfc/frontend/utils/timeleft.hpp>
#include <openpfc/frontend/utils/toml_to_json.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>
#include <openpfc/openpfc_minimal.hpp>
#include <toml++/toml.hpp>
#include <vector>

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

  bool m_prof_enabled = false;
  std::string m_prof_format = "json";
  std::string m_prof_output = "openpfc_profile";
  bool m_prof_memory_samples = false;
  bool m_prof_print_report = false;
  std::vector<std::string> m_prof_extra_regions;
  std::unique_ptr<pfc::profiling::ProfilingSession> m_profiler;

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

  void read_profiling_configuration() {
    if (!m_settings.contains("profiling")) return;
    const auto &p = m_settings["profiling"];
    if (p.contains("enabled")) m_prof_enabled = p["enabled"].get<bool>();
    if (p.contains("format") && p["format"].is_string())
      m_prof_format = p["format"].get<std::string>();
    if (p.contains("output") && p["output"].is_string())
      m_prof_output = p["output"].get<std::string>();
    if (p.contains("memory_samples"))
      m_prof_memory_samples = p["memory_samples"].get<bool>();
    if (p.contains("print_report"))
      m_prof_print_report = p["print_report"].get<bool>();
    m_prof_extra_regions.clear();
    if (p.contains("regions") && p["regions"].is_array()) {
      for (const auto &el : p["regions"]) {
        if (el.is_string()) m_prof_extra_regions.push_back(el.get<std::string>());
      }
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
        const auto &target = params["target"];
        if (target.is_array()) {
          std::vector<std::string> names;
          names.reserve(target.size());
          for (const auto &el : target) {
            names.push_back(el.get<std::string>());
          }
          field_modifier->set_field_names(std::move(names));
          std::cout << "Setting initial condition targets (multi-field)"
                    << std::endl;
        } else {
          std::string t = target.get<std::string>();
          std::cout << "Setting initial condition target to " << t << std::endl;
          field_modifier->set_field_name(t);
        }
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
        const auto &target = params["target"];
        if (target.is_array()) {
          std::vector<std::string> names;
          names.reserve(target.size());
          for (const auto &el : target) {
            names.push_back(el.get<std::string>());
          }
          field_modifier->set_field_names(std::move(names));
          std::cout << "Setting boundary condition targets (multi-field)"
                    << std::endl;
        } else {
          std::string t = target.get<std::string>();
          std::cout << "Setting boundary condition target to " << t << std::endl;
          field_modifier->set_field_name(t);
        }
      }
      sim.add_boundary_conditions(std::move(field_modifier));
    }
  }

  int main() {
#if defined(OpenPFC_ENABLE_HIP) && defined(OpenPFC_MPI_HIP_AWARE)
    if (rank0) {
      std::cout << "OpenPFC: GPU-aware MPI (HIP) is enabled at compile time.\n";
      const char *gpu_env = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
      if (gpu_env == nullptr || std::string(gpu_env) != "1") {
        std::cerr << "Warning: MPICH_GPU_SUPPORT_ENABLED is not set to 1; "
                     "Cray MPICH may not accept device pointers (set export "
                     "MPICH_GPU_SUPPORT_ENABLED=1 in your job).\n";
      }
    }
#endif
#if defined(OpenPFC_ENABLE_CUDA) && defined(OpenPFC_MPI_CUDA_AWARE)
    if (rank0) {
      std::cout << "OpenPFC: GPU-aware MPI (CUDA) is enabled at compile time.\n";
    }
#endif
    std::cout << "Reading configuration from json file:" << std::endl;
    std::cout << m_settings.dump(4) << "\n\n";

    World world(ui::from_json<World>(m_settings));
    std::cout << "World: " << world << std::endl;

    int num_ranks = m_worker.get_num_ranks();
    int rank_id = m_worker.get_rank();
    auto decomp = decomposition::create(world, num_ranks);

    // Create FFT with default FFTW backend for now
    // Note: Runtime backend selection via create_with_backend() can be added when
    // needed
    auto options =
        m_settings.contains("plan_options")
            ? ui::from_json<heffte::plan_options>(m_settings["plan_options"])
            : heffte::default_options<heffte::backend::fftw>();

    auto fft_layout = fft::layout::create(decomp, 0);
    auto fft = fft::create(fft_layout, rank_id, options);
    Time time(ui::from_json<Time>(m_settings));
    ConcreteModel model(fft, world);
    Simulator simulator(model, time);

    if (m_settings.contains("model") && m_settings["model"].contains("params")) {
      from_json(m_settings["model"]["params"], model);
    }
    read_profiling_configuration();
    if (m_prof_enabled) {
      m_profiler = std::make_unique<pfc::profiling::ProfilingSession>(
          pfc::profiling::ProfilingMetricCatalog::with_defaults_and_extras(
              m_prof_extra_regions));
    }

    std::cout << "Initializing model... " << std::endl;
    model.initialize(time.get_dt());

    // Report memory usage
    {
      size_t model_mem = model.get_allocated_memory_bytes();
      size_t fft_mem = fft.get_allocated_memory_bytes();
      pfc::utils::MemoryUsage usage{model_mem, fft_mem};
      pfc::Logger logger{pfc::LogLevel::Info, rank_id};
      pfc::utils::report_memory_usage(usage, world, logger, m_comm);
    }

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

      fft.reset_fft_time();
      const double barrier_step_s = pfc::profiling::measure_barriered(m_comm, [&] {
        if (m_profiler) m_profiler->begin_step_frame(time.get_increment(), rank_id);
        if (m_profiler) {
          pfc::profiling::ProfilingContextScope scope(m_profiler.get());
          step(simulator, model);
        } else {
          step(simulator, model);
        }
      });
      const double fft_meter_s = fft.get_fft_time();

      std::uint64_t rss = 0;
      std::uint64_t model_mem = 0;
      std::uint64_t fft_mem = 0;
      if (m_profiler && m_prof_memory_samples) {
        rss = pfc::profiling::try_read_process_rss_bytes();
        model_mem = model.get_allocated_memory_bytes();
        fft_mem = fft.get_allocated_memory_bytes();
      }
      if (m_profiler) {
        m_profiler->set_frame_wall_step(barrier_step_s);
        m_profiler->assign_recorded_time("fft", fft_meter_s);
        m_profiler->end_step_frame(rss, model_mem, fft_mem);
      }

      m_steptime = pfc::profiling::reduce_max_to_root(m_comm, barrier_step_s, 0);
      m_fft_time = pfc::profiling::reduce_max_to_root(m_comm, fft_meter_s, 0);

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

    if (m_profiler) {
      pfc::profiling::ProfilingExportOptions exp;
      std::string fmt = m_prof_format;
      std::transform(fmt.begin(), fmt.end(), fmt.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (fmt == "hdf5") {
        exp.write_json = false;
        exp.write_csv = false;
        exp.write_hdf5 = true;
        exp.hdf5_path = m_prof_output + ".h5";
      } else if (fmt == "csv") {
        exp.write_json = false;
        exp.write_csv = true;
        exp.csv_path = m_prof_output + ".csv";
      } else if (fmt == "csv_hdf5" || fmt == "csv+hdf5") {
        exp.write_json = false;
        exp.write_csv = true;
        exp.write_hdf5 = true;
        exp.csv_path = m_prof_output + ".csv";
        exp.hdf5_path = m_prof_output + ".h5";
      } else if (fmt == "both") {
        exp.write_json = true;
        exp.write_hdf5 = true;
        exp.json_path = m_prof_output + ".json";
        exp.hdf5_path = m_prof_output + ".h5";
      } else {
        if (fmt != "json" && rank0)
          std::cerr << "profiling.format unknown (\"" << m_prof_format
                    << "\"), using json\n";
        exp.write_json = true;
        exp.json_path = m_prof_output + ".json";
      }
      m_profiler->finalize_and_export(m_comm, exp);
      if (rank0)
        std::cout << "Profiling export written (see profiling.output / format).\n";
      if (rank0 && m_prof_print_report && m_profiler) {
        pfc::profiling::ProfilingPrintOptions popts;
        popts.title = "OpenPFC profiling (this rank)";
        pfc::profiling::print_profiling_timer(std::cout, *m_profiler, popts);
      }
    }

    return 0;
  }
};

} // namespace ui
} // namespace pfc

#endif // PFC_UI_APP_HPP
