// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <sstream>

#include <openpfc/frontend/ui/app_profiling.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>

namespace pfc::ui {

void AppProfilingController::configure_from_root_settings(const json &root_settings,
                                                          int mpi_rank, bool rank0) {
  m_session.reset();
  m_enabled = false;
  m_format = "json";
  m_output = "openpfc_profile";
  m_memory_samples = false;
  m_print_report = false;
  m_extra_regions.clear();
  m_run_id.clear();
  m_export_metadata = json::object();

  if (!root_settings.contains("profiling")) {
    return;
  }
  const auto &p = root_settings["profiling"];
  if (rank0) {
    warn_unknown_profiling_keys(p, pfc::Logger{pfc::LogLevel::Info, mpi_rank});
  }
  if (p.contains("enabled")) {
    m_enabled = p["enabled"].get<bool>();
  }
  if (p.contains("format") && p["format"].is_string()) {
    m_format = p["format"].get<std::string>();
  }
  if (p.contains("output") && p["output"].is_string()) {
    m_output = p["output"].get<std::string>();
  }
  if (p.contains("memory_samples")) {
    m_memory_samples = p["memory_samples"].get<bool>();
  }
  if (p.contains("print_report")) {
    m_print_report = p["print_report"].get<bool>();
  }
  m_extra_regions.clear();
  if (p.contains("regions") && p["regions"].is_array()) {
    for (const auto &el : p["regions"]) {
      if (el.is_string()) {
        m_extra_regions.push_back(el.get<std::string>());
      }
    }
  }
  m_run_id.clear();
  if (p.contains("run_id") && p["run_id"].is_string()) {
    m_run_id = p["run_id"].get<std::string>();
  }
  m_export_metadata = json::object();
  if (p.contains("export_metadata") && p["export_metadata"].is_object()) {
    m_export_metadata = p["export_metadata"];
  }

  if (m_enabled) {
    m_session = std::make_unique<pfc::profiling::ProfilingSession>(
        pfc::profiling::ProfilingMetricCatalog::with_defaults_and_extras(
            m_extra_regions),
        pfc::profiling::ProfilingSession::openpfc_default_frame_metrics());
  }
}

void AppProfilingController::finalize_and_export_if_active(
    const json &root_settings, MPI_Comm comm, bool rank0,
    const pfc::Logger &app_lg) {
  if (!m_session) {
    return;
  }
  pfc::profiling::ProfilingExportOptions exp;
  std::string fmt = m_format;
  std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (fmt == "hdf5") {
    exp.write_json = false;
    exp.write_hdf5 = true;
    exp.hdf5_path = m_output + ".h5";
  } else if (fmt == "csv" || fmt == "csv_hdf5" || fmt == "csv+hdf5") {
    if (rank0) {
      std::ostringstream pfoss;
      pfoss << "profiling.format \"" << m_format
            << "\" is no longer supported (CSV export removed); using json";
      if (fmt == "csv_hdf5" || fmt == "csv+hdf5") {
        pfoss << " and hdf5";
      }
      pfoss << '.';
      pfc::log_warning(app_lg, pfoss.str());
    }
    if (fmt == "csv") {
      exp.write_json = true;
      exp.json_path = m_output + ".json";
    } else {
      exp.write_json = true;
      exp.write_hdf5 = true;
      exp.json_path = m_output + ".json";
      exp.hdf5_path = m_output + ".h5";
    }
  } else if (fmt == "both") {
    exp.write_json = true;
    exp.write_hdf5 = true;
    exp.json_path = m_output + ".json";
    exp.hdf5_path = m_output + ".h5";
  } else {
    if (fmt != "json" && rank0) {
      pfc::log_warning(app_lg, std::string("profiling.format unknown (\"") +
                                   m_format + "\"), using json");
    }
    exp.write_json = true;
    exp.json_path = m_output + ".json";
  }
  populate_export_metadata(root_settings, exp);
  m_session->finalize_and_export(comm, exp);
  if (rank0) {
    pfc::log_info(app_lg,
                  "Profiling export written (see profiling.output / format).");
  }
  if (m_print_report && m_session) {
    pfc::profiling::ProfilingPrintOptions popts;
    popts.title = "OpenPFC profiling (MPI aggregate, mean)";
    popts.ascii_lines = true;
    popts.sort_by_time = true;
    popts.show_exclusive_column = true;
    popts.mpi_aggregate_stdout = true;
    std::ostringstream prof_out;
    pfc::profiling::print_profiling_timer(prof_out, comm, *m_session, popts);
    if (rank0 && !prof_out.str().empty()) {
      pfc::log_info(app_lg, prof_out.str());
    }
  }
}

void AppProfilingController::populate_export_metadata(
    const json &root_settings, pfc::profiling::ProfilingExportOptions &exp) const {
  exp.run_id = m_run_id;
  if (exp.run_id.empty()) {
    if (const char *e = std::getenv("SLURM_JOB_ID")) {
      exp.run_id = e;
    } else if (const char *e2 = std::getenv("OPENPFC_PROFILING_RUN_ID")) {
      exp.run_id = e2;
    }
  }
  exp.export_metadata = json::object();
  if (root_settings.contains("domain")) {
    const auto &d = root_settings["domain"];
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
  for (auto it = m_export_metadata.begin(); it != m_export_metadata.end(); ++it) {
    exp.export_metadata[it.key()] = it.value();
  }
}

} // namespace pfc::ui
