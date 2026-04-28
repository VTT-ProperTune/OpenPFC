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
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/frontend/ui/json_helpers.hpp>
#include <openpfc/frontend/ui/settings_loader.hpp>
#include <openpfc/frontend/utils/logging.hpp>
#include <openpfc/frontend/utils/memory_reporter.hpp>
#include <openpfc/frontend/utils/timeleft.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>
#include <openpfc/openpfc_minimal.hpp>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pfc::ui {

/**
 * @brief Return profiling JSON keys that are not in the supported set (for tests
 * and diagnostics).
 */
[[nodiscard]] inline std::vector<std::string>
list_unknown_profiling_keys(const json &profiling) {
  std::vector<std::string> out;
  if (!profiling.is_object()) {
    return out;
  }
  static constexpr std::array<std::string_view, 8> known_keys = {
      "enabled",      "format",  "output", "memory_samples",
      "print_report", "regions", "run_id", "export_metadata"};

  for (const auto &[key, _] : profiling.items()) {
    if (std::find(known_keys.begin(), known_keys.end(), key) == known_keys.end()) {
      out.push_back(std::string("unknown profiling config key '") + key + "'");
    }
  }
  return out;
}

inline void warn_unknown_profiling_keys(const json &profiling,
                                        const pfc::Logger &lg) {
  for (const auto &msg : list_unknown_profiling_keys(profiling)) {
    pfc::log_warning(lg, msg);
  }
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
  /// Optional; if empty at export time, `SLURM_JOB_ID` / `OPENPFC_PROFILING_RUN_ID`
  /// may be used (see `read_profiling_configuration`).
  std::string m_prof_run_id;
  json m_prof_export_metadata;
  std::unique_ptr<pfc::profiling::ProfilingSession> m_profiler;

  // read settings from file (JSON or TOML format)
  json read_settings(int argc, char **argv) {
    const pfc::Logger lg{pfc::LogLevel::Info, m_worker.get_rank()};
    if (argc <= 1) {
      if (rank0) {
        pfc::log_error(lg, std::string("Configuration file required. Usage: ") +
                               argv[0] + " <config.json|config.toml>");
      }
      throw std::runtime_error("OpenPFC App: configuration file required (pass path "
                               "to JSON or TOML as first argument)");
    }

    std::filesystem::path file(argv[1]);
    auto ext = file.extension().string();

    if (rank0) {
      if (ext == ".toml") {
        pfc::log_info(lg, std::string("Reading TOML configuration from ") +
                              file.string());
      } else if (ext == ".json") {
        pfc::log_info(lg, std::string("Reading JSON configuration from ") +
                              file.string());
      }
    }

    try {
      return load_settings_file(file);
    } catch (const std::exception &err) {
      if (rank0) {
        pfc::log_error(lg, std::string("Failed to load settings: ") + err.what());
      }
      throw std::runtime_error(
          std::string("OpenPFC App: failed to load settings: ") + err.what());
    }
  }

public:
  App(int argc, char **argv, MPI_Comm comm = MPI_COMM_WORLD)
      : m_comm(comm), m_worker(MPI_Worker(argc, argv, comm)),
        rank0(m_worker.get_rank() == 0), m_settings(read_settings(argc, argv)) {}

  App(json settings, MPI_Comm comm = MPI_COMM_WORLD)
      : m_comm(comm), m_worker(MPI_Worker(0, nullptr, comm)),
        rank0(m_worker.get_rank() == 0), m_settings(std::move(settings)) {}

  bool create_results_dir(const std::string &output) {
    const pfc::Logger lg{pfc::LogLevel::Info, m_worker.get_rank()};
    std::filesystem::path results_dir(output);
    if (results_dir.has_filename()) {
      results_dir = results_dir.parent_path();
    }
    if (!std::filesystem::exists(results_dir)) {
      pfc::log_info(lg, std::string("Results dir ") + results_dir.string() +
                            " does not exist, creating");
      std::filesystem::create_directories(results_dir);
      return true;
    }
    pfc::log_warning(lg, std::string("results dir ") + results_dir.string() +
                             " already exists");
    return false;
  }

  void read_profiling_configuration() {
    if (!m_settings.contains("profiling")) {
      return;
    }
    const auto &p = m_settings["profiling"];
    if (rank0) {
      warn_unknown_profiling_keys(
          p, pfc::Logger{pfc::LogLevel::Info, m_worker.get_rank()});
    }
    if (p.contains("enabled")) {
      m_prof_enabled = p["enabled"].get<bool>();
    }
    if (p.contains("format") && p["format"].is_string()) {
      m_prof_format = p["format"].get<std::string>();
    }
    if (p.contains("output") && p["output"].is_string()) {
      m_prof_output = p["output"].get<std::string>();
    }
    if (p.contains("memory_samples")) {
      m_prof_memory_samples = p["memory_samples"].get<bool>();
    }
    if (p.contains("print_report")) {
      m_prof_print_report = p["print_report"].get<bool>();
    }
    m_prof_extra_regions.clear();
    if (p.contains("regions") && p["regions"].is_array()) {
      for (const auto &el : p["regions"]) {
        if (el.is_string()) {
          m_prof_extra_regions.push_back(el.get<std::string>());
        }
      }
    }
    m_prof_run_id.clear();
    if (p.contains("run_id") && p["run_id"].is_string()) {
      m_prof_run_id = p["run_id"].get<std::string>();
    }
    m_prof_export_metadata = json::object();
    if (p.contains("export_metadata") && p["export_metadata"].is_object()) {
      m_prof_export_metadata = p["export_metadata"];
    }
  }

  void
  apply_profiling_export_options(pfc::profiling::ProfilingExportOptions &exp) const {
    exp.run_id = m_prof_run_id;
    if (exp.run_id.empty()) {
      if (const char *e = std::getenv("SLURM_JOB_ID")) {
        exp.run_id = e;
      } else if (const char *e2 = std::getenv("OPENPFC_PROFILING_RUN_ID")) {
        exp.run_id = e2;
      }
    }
    exp.export_metadata = json::object();
    if (m_settings.contains("domain")) {
      const auto &d = m_settings["domain"];
      if (d.contains("Lx")) {
        exp.export_metadata["domain_lx"] = d["Lx"];
      }
      if (d.contains("Ly")) {
        exp.export_metadata["domain_ly"] = d["Ly"];
      }
      if (d.contains("Lz")) {
        exp.export_metadata["domain_lz"] = d["Lz"];
      }
    }
    if (const char *e = std::getenv("SLURM_JOB_ID")) {
      exp.export_metadata["slurm_job_id"] = std::string(e);
    }
    if (const char *e = std::getenv("SLURM_JOB_PARTITION")) {
      exp.export_metadata["slurm_partition"] = std::string(e);
    }
    if (const char *e = std::getenv("SLURM_NNODES")) {
      exp.export_metadata["slurm_nnodes"] = std::string(e);
    }
    if (const char *e = std::getenv("SLURM_NTASKS")) {
      exp.export_metadata["slurm_ntasks"] = std::string(e);
    }
    for (auto it = m_prof_export_metadata.begin();
         it != m_prof_export_metadata.end(); ++it) {
      exp.export_metadata[it.key()] = it.value();
    }
  }

  void add_result_writers(Simulator &sim) {
    const pfc::Logger lg{pfc::LogLevel::Info, m_worker.get_rank()};
    if (rank0) {
      pfc::log_info(lg, "Adding results writers");
    }
    if (m_settings.contains("saveat") && m_settings.contains("fields") &&
        m_settings["saveat"] > 0) {
      for (const auto &field : m_settings["fields"]) {
        std::string name = field["name"];
        std::string data = field["data"];
        if (rank0) {
          create_results_dir(data);
        }
        if (rank0) {
          pfc::log_info(lg, "Writing field " + name + " to " + data);
        }
        sim.add_results_writer(name, std::make_unique<BinaryWriter>(data));
      }
    } else {
      if (rank0) {
        pfc::log_warning(lg, "not writing results to anywhere.");
        pfc::log_info(lg, "To write results, add ResultsWriter to model.");
      }
    }
  }

  void add_initial_conditions(Simulator &sim) {
    const pfc::Logger lg{pfc::LogLevel::Info, m_worker.get_rank()};
    if (!m_settings.contains("initial_conditions")) {
      if (rank0) {
        pfc::log_warning(lg, "no initial conditions are set!");
      }
      return;
    }
    if (rank0) {
      pfc::log_info(lg, "Adding initial conditions");
    }
    for (const json &params : m_settings["initial_conditions"]) {
      if (rank0) {
        std::ostringstream ps;
        ps << params;
        pfc::log_info(lg, std::string("Creating initial condition from data ") +
                              ps.str());
      }
      if (!params.contains("type")) {
        if (rank0) {
          pfc::log_warning(lg, "no type is set for initial condition!");
        }
        continue;
      }
      std::string type = params["type"];
      auto field_modifier = create_field_modifier(type, params);
      if (!params.contains("target")) {
        if (rank0) {
          pfc::log_warning(
              lg, "no target is set for initial condition! Using target 'default'");
        }
      } else {
        const auto &target = params["target"];
        if (target.is_array()) {
          std::vector<std::string> names;
          names.reserve(target.size());
          for (const auto &el : target) {
            names.push_back(el.get<std::string>());
          }
          field_modifier->set_field_names(std::move(names));
          if (rank0) {
            pfc::log_info(lg, "Setting initial condition targets (multi-field)");
          }
        } else {
          auto t = target.get<std::string>();
          if (rank0) {
            pfc::log_info(lg,
                          std::string("Setting initial condition target to ") + t);
          }
          field_modifier->set_field_name(t);
        }
      }
      sim.add_initial_conditions(std::move(field_modifier));
    }
  }

  void add_boundary_conditions(Simulator &sim) {
    const pfc::Logger lg{pfc::LogLevel::Info, m_worker.get_rank()};
    if (!m_settings.contains("boundary_conditions")) {
      if (rank0) {
        pfc::log_warning(lg, "no boundary conditions are set!");
      }
      return;
    }
    if (rank0) {
      pfc::log_info(lg, "Adding boundary conditions");
    }
    for (const json &params : m_settings["boundary_conditions"]) {
      if (rank0) {
        std::ostringstream ps;
        ps << params;
        pfc::log_info(lg, std::string("Creating boundary condition from data ") +
                              ps.str());
      }
      if (!params.contains("type")) {
        if (rank0) {
          pfc::log_warning(lg, "no type is set for boundary condition!");
        }
        continue;
      }
      std::string type = params["type"];
      auto field_modifier = create_field_modifier(type, params);
      if (!params.contains("target")) {
        if (rank0) {
          pfc::log_warning(
              lg, "no target is set for boundary condition! Using target 'default'");
        }
      } else {
        const auto &target = params["target"];
        if (target.is_array()) {
          std::vector<std::string> names;
          names.reserve(target.size());
          for (const auto &el : target) {
            names.push_back(el.get<std::string>());
          }
          field_modifier->set_field_names(std::move(names));
          if (rank0) {
            pfc::log_info(lg, "Setting boundary condition targets (multi-field)");
          }
        } else {
          auto t = target.get<std::string>();
          if (rank0) {
            pfc::log_info(lg,
                          std::string("Setting boundary condition target to ") + t);
          }
          field_modifier->set_field_name(t);
        }
      }
      sim.add_boundary_conditions(std::move(field_modifier));
    }
  }

  int main() {
    const int rank_id = m_worker.get_rank();
    const pfc::Logger app_lg{pfc::LogLevel::Info, rank_id};
#if defined(OpenPFC_ENABLE_HIP) && defined(OpenPFC_MPI_HIP_AWARE)
    if (rank0) {
      pfc::log_info(app_lg,
                    "OpenPFC: GPU-aware MPI (HIP) is enabled at compile time.");
      const char *gpu_env = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
      if (gpu_env == nullptr || std::string(gpu_env) != "1") {
        pfc::log_warning(
            app_lg, "MPICH_GPU_SUPPORT_ENABLED is not set to 1; Cray MPICH may not "
                    "accept device pointers (set export MPICH_GPU_SUPPORT_ENABLED=1 "
                    "in your job).");
      }
    }
#endif
#if defined(OpenPFC_ENABLE_CUDA) && defined(OpenPFC_MPI_CUDA_AWARE)
    if (rank0) {
      pfc::log_info(app_lg,
                    "OpenPFC: GPU-aware MPI (CUDA) is enabled at compile time.");
    }
#endif
    if (rank0) {
      pfc::log_info(app_lg, "Reading configuration from json file:");
      pfc::log_info(app_lg, m_settings.dump(4));
    }

    World world(ui::from_json<World>(m_settings));
    if (rank0) {
      std::ostringstream woss;
      woss << world;
      pfc::log_info(app_lg, std::string("World: ") + woss.str());
    }

    int num_ranks = m_worker.get_num_ranks();
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
              m_prof_extra_regions),
          pfc::profiling::ProfilingSession::openpfc_default_frame_metrics());
    }

    if (rank0) {
      pfc::log_info(app_lg, "Initializing model...");
    }
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

    if (rank0) {
      pfc::log_info(app_lg, "Starting time integration (Simulator integrator API)");
    }

    while (!time.done()) {
      fft.reset_fft_time();
      simulator.begin_integrator_step();
      const double barrier_step_s = pfc::profiling::measure_barriered(m_comm, [&] {
        if (m_profiler) {
          pfc::profiling::openpfc_begin_frame_with_step_and_rank(
              *m_profiler, time.get_increment(), rank_id);
        }
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
        m_profiler->assign_recorded_time("fft", fft_meter_s);
        pfc::profiling::openpfc_end_frame_step_wall_and_memory(
            *m_profiler, barrier_step_s, rss, model_mem, fft_mem);
      }

      m_steptime = pfc::profiling::reduce_max_to_root(m_comm, barrier_step_s, 0);
      m_fft_time = pfc::profiling::reduce_max_to_root(m_comm, fft_meter_s, 0);

      simulator.end_integrator_step();

      // Calculate eta from average step time.
      // Use exponential moving average when steps > 3.
      m_avg_steptime = m_steptime;
      if (m_steps_done > 3) {
        m_avg_steptime = 0.01 * m_steptime + 0.99 * m_avg_steptime;
      }
      int increment = time.get_increment();
      double t = time.get_current();
      double t1 = time.get_t1();
      double eta_i = (t1 - t) / time.get_dt();
      double eta_t = eta_i * m_avg_steptime;
      double other_time = m_steptime - m_fft_time;
      if (rank0) {
        std::ostringstream steposs;
        steposs << "Step " << increment << " done in " << m_steptime << " s ("
                << m_fft_time << " s FFT, " << other_time
                << " s other). Simulation time: " << t << " / " << t1 << " ("
                << (t / t1 * 100)
                << " % done). ETA: " << pfc::utils::TimeLeft(eta_t);
        pfc::log_info(app_lg, steposs.str());
      }

      m_total_steptime += m_steptime;
      m_total_fft_time += m_fft_time;
      m_steps_done += 1;
    }

    if (m_steps_done > 0) {
      const double avg_steptime = m_total_steptime / m_steps_done;
      const double avg_fft_time = m_total_fft_time / m_steps_done;
      const double avg_oth_time = avg_steptime - avg_fft_time;
      const double p_fft = avg_fft_time / avg_steptime * 100.0;
      const double p_oth = avg_oth_time / avg_steptime * 100.0;
      if (rank0) {
        std::ostringstream sumoss;
        sumoss << "Simulated " << m_steps_done << " steps. Average times:\n"
               << "Step time:  " << avg_steptime << " s\n"
               << "FFT time:   " << avg_fft_time << " s / " << p_fft << " %\n"
               << "Other time: " << avg_oth_time << " s / " << p_oth << " %";
        pfc::log_info(app_lg, sumoss.str());
      }
    } else if (rank0) {
      pfc::log_info(
          app_lg,
          "No complete timesteps were executed; skipping average timing summary.");
    }

    if (m_profiler) {
      pfc::profiling::ProfilingExportOptions exp;
      std::string fmt = m_prof_format;
      std::transform(fmt.begin(), fmt.end(), fmt.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (fmt == "hdf5") {
        exp.write_json = false;
        exp.write_hdf5 = true;
        exp.hdf5_path = m_prof_output + ".h5";
      } else if (fmt == "csv" || fmt == "csv_hdf5" || fmt == "csv+hdf5") {
        if (rank0) {
          std::ostringstream pfoss;
          pfoss << "profiling.format \"" << m_prof_format
                << "\" is no longer supported (CSV export removed); using json";
          if (fmt == "csv_hdf5" || fmt == "csv+hdf5") {
            pfoss << " and hdf5";
          }
          pfoss << '.';
          pfc::log_warning(app_lg, pfoss.str());
        }
        if (fmt == "csv") {
          exp.write_json = true;
          exp.json_path = m_prof_output + ".json";
        } else {
          exp.write_json = true;
          exp.write_hdf5 = true;
          exp.json_path = m_prof_output + ".json";
          exp.hdf5_path = m_prof_output + ".h5";
        }
      } else if (fmt == "both") {
        exp.write_json = true;
        exp.write_hdf5 = true;
        exp.json_path = m_prof_output + ".json";
        exp.hdf5_path = m_prof_output + ".h5";
      } else {
        if (fmt != "json" && rank0) {
          pfc::log_warning(app_lg, std::string("profiling.format unknown (\"") +
                                       m_prof_format + "\"), using json");
        }
        exp.write_json = true;
        exp.json_path = m_prof_output + ".json";
      }
      apply_profiling_export_options(exp);
      m_profiler->finalize_and_export(m_comm, exp);
      if (rank0) {
        pfc::log_info(app_lg,
                      "Profiling export written (see profiling.output / format).");
      }
      if (m_prof_print_report && m_profiler) {
        pfc::profiling::ProfilingPrintOptions popts;
        popts.title = "OpenPFC profiling (MPI aggregate, mean)";
        popts.ascii_lines = true;
        popts.sort_by_time = true;
        popts.show_exclusive_column = true;
        popts.mpi_aggregate_stdout = true;
        std::ostringstream prof_out;
        pfc::profiling::print_profiling_timer(prof_out, m_comm, *m_profiler, popts);
        if (rank0 && !prof_out.str().empty()) {
          pfc::log_info(app_lg, prof_out.str());
        }
      }
    }

    return 0;
  }
};

} // namespace pfc::ui

#endif // PFC_UI_APP_HPP
