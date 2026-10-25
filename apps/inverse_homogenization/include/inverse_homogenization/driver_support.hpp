// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/** @file driver_support.hpp
 * @brief Host orchestration shared by the CPU and HIP inverse drivers.
 * Numerical updates and field I/O remain with their backend adapters.
 */
#include <cmath>
#include <filesystem>
#include <fstream>
#include <inverse_homogenization/inverse_checkpoint.hpp>
#include <inverse_homogenization/inverse_convergence.hpp>
#include <inverse_homogenization/phase_field_inverse.hpp>
#include <iostream>
#include <mpi.h>
#include <string>
#include <utility>

namespace pfc::apps::inverse::driver {

// Common host-side parameter mapping; target loading stays with the driver CLI.
template <class Config>
InverseSpec configure_problem(const Config &cfg, const Voigt6 &target) {
  InverseSpec spec;
  spec.C_target = target;
  spec.volume_target = cfg.volume;
  spec.lambda_volume = cfg.lambda_volume;
  spec.lambda_reg = cfg.lambda_reg;
  spec.epsilon = cfg.epsilon;
  spec.dt = cfg.dt;
  spec.normalize_grad = cfg.normalize != 0;
  spec.max_abs_delta = cfg.max_delta;
  spec.project_volume = cfg.project_volume != 0;
  spec.simp_p = cfg.simp;
  if (cfg.no_tensor != 0) {
    spec.W = Voigt6{};
  } else if (cfg.w12 != 1.0 || cfg.target == "auxetic") {
    const double weight =
        (cfg.target == "auxetic" && cfg.w12 == 1.0) ? 4.0 : cfg.w12;
    spec.W(0, 1) = spec.W(1, 0) = weight;
    spec.W(0, 2) = spec.W(2, 0) = weight;
    spec.W(1, 2) = spec.W(2, 1) = weight;
  }
  return spec;
}

template <class Config>
pfc::solvers::MicroelasticityParams configure_elasticity(const Config &cfg) {
  pfc::solvers::MicroelasticityParams p;
  p.stiffness_at_one = pfc::solvers::Stiffness::isotropic(cfg.E_solid, cfg.nu_solid);
  p.stiffness_at_zero = pfc::solvers::Stiffness::isotropic(cfg.E_void, cfg.nu_void);
  p.relative_tolerance = 1.0e-8;
  p.max_iterations = cfg.n_el_iter;
  p.warm_start = false;
  p.comm = MPI_COMM_WORLD;
  return p;
}

inline ConvergenceMetrics diagnose_step(const InverseStepReport &last,
                                        const Voigt6 &previous_C, double previous_J,
                                        bool have_previous, double design_rms,
                                        double morph_fraction,
                                        const ConvergenceConfig &config) {
  double dC2 = 0.0;
  if (have_previous) {
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) {
        const double delta = last.C(i, j) - previous_C(i, j);
        dC2 += delta * delta;
      }
  }
  auto metrics = make_metrics(
      design_rms, last.J, have_previous ? previous_J : last.J, std::sqrt(dC2),
      have_previous ? previous_C.frobenius_norm() : last.C_fro, morph_fraction,
      config);
  if (!have_previous) metrics.quiet = false;
  return metrics;
}

template <class Config> ConvergenceTracker configure_tracker(const Config &cfg) {
  ConvergenceTracker tracker;
  tracker.cfg.continuation_steps = cfg.continuation_steps;
  tracker.cfg.max_steps = cfg.steps;
  tracker.cfg.conv_window = cfg.conv_window;
  tracker.cfg.verify_steps = cfg.verify_steps;
  tracker.cfg.tol_design = cfg.tol_design;
  tracker.cfg.tol_objective = cfg.tol_objective;
  tracker.cfg.tol_tensor = cfg.tol_tensor;
  return tracker;
}

template <class Config> std::ofstream open_csv(const Config &cfg, int rank) {
  std::ofstream csv;
  if (rank == 0 && !cfg.csv.empty()) {
    const bool resume = !cfg.restart_dir.empty() && std::filesystem::exists(cfg.csv);
    csv.open(cfg.csv, resume ? std::ios::app : std::ios::out);
    if (!resume) csv << kInverseCsvHeader << '\n';
  }
  return csv;
}

inline double effective_poisson_ratio(double c11, double c12) {
  const double den = c11 + c12;
  return std::abs(den) > 1.0e-30 ? c12 / den : 0.0;
}

// Emit the existing schema in its original order. Flushing is a caller choice.
template <class Metrics>
void write_step_csv(std::ostream &csv, int step, const InverseStepReport &last,
                    const InverseSpec &spec, const Metrics &metrics,
                    const ConvergenceTracker &tracker, TerminationReason reason,
                    double design_rms, double morph_frac, double ms, int frozen) {
  InverseCsvRow row;
  row.step = step;
  row.J = last.J;
  row.J_tensor = last.J_tensor;
  row.J_volume = last.J_volume;
  row.J_reg = last.J_reg;
  row.volume = last.volume_accepted;
  row.grey = last.grey_fraction;
  row.C11 = last.C11;
  row.C12 = last.C12;
  row.nu_eff = effective_poisson_ratio(last.C11, last.C12);
  row.C_fro = last.C_fro;
  row.design_rms = design_rms;
  row.dJ_rel = metrics.dJ_rel;
  row.dC_rel = metrics.dC_rel;
  row.morph_frac = morph_frac;
  row.step_rms = last.step_rms;
  row.grad_rms = last.grad_rms;
  row.simp_p = spec.simp_p;
  row.lambda_reg = spec.lambda_reg;
  row.frozen = frozen;
  row.conv_window = tracker.quiet_count;
  row.candidate = tracker.candidate ? 1 : 0;
  row.verified = tracker.verified ? 1 : 0;
  row.ms = ms;
  row.elasticity = last.elasticity_converged ? 1 : 0;
  row.termination = termination_name(reason);
  write_inverse_csv_row(csv, row);
}

inline void report_process_memory(std::ostream &out) {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmHWM:", 0) == 0 || line.rfind("VmRSS:", 0) == 0)
      out << line << '\n';
  }
}

template <class Config>
InverseCheckpoint capture_checkpoint(const Config &cfg, const InverseSpec &spec,
                                     const ConvergenceTracker &tracker,
                                     int next_step, bool have_prev, double J_prev,
                                     const Voigt6 &C_prev, TerminationReason why) {
  InverseCheckpoint ck;
  capture_problem(ck, cfg, spec.C_target, spec.W);
  ck.nx = cfg.nx;
  ck.ny = cfg.ny;
  ck.nz = cfg.nz;
  capture_tracker(ck, tracker, next_step);
  ck.have_prev = have_prev ? 1 : 0;
  ck.termination = static_cast<int>(why);
  ck.J_prev = J_prev;
  store_voigt6(ck.C_prev, C_prev);
  return ck;
}

// Collective staging order is identical for both backends. write_fields runs
// on every rank: CPU supplies a root-only writer; HIP supplies collective MPI I/O.
template <class WriteFields, class WriteExtra>
bool publish_checkpoint(const std::filesystem::path &root,
                        const InverseCheckpoint &ck, int rank,
                        WriteFields write_fields, WriteExtra write_extra) {
  const auto generation = checkpoint_generation_name(ck.next_step);
  int ready = 1;
  if (rank == 0) ready = prepare_checkpoint_staging(root) ? 1 : 0;
  MPI_Bcast(&ready, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!ready) {
    if (rank == 0) std::cerr << "checkpoint: failed to prepare staging\n";
    return false;
  }
  const auto staging = checkpoint_staging_dir(root).string();
  const bool fields_ready = write_fields(staging);
  if (rank == 0) {
    ready = fields_ready && checkpoint_field_sizes_match(staging, ck);
    ready = ready && write_checkpoint_file(staging + "/state.txt", ck);
    ready = ready && write_extra(staging);
    ready = ready && publish_checkpoint_generation(root, generation);
    if (!ready) std::cerr << "checkpoint: failed to publish " << generation << '\n';
  }
  MPI_Bcast(&ready, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ready != 0;
}

struct RestartBundle {
  InverseCheckpoint checkpoint;
  std::string path;
  bool ready = false;
};

// Validate only compact metadata on root, then publish the same decision and
// checkpoint to all ranks before either backend loads its field payload.
template <class Config, class ReadExtra>
RestartBundle inspect_restart(const Config &cfg, const InverseSpec &spec, int rank,
                              ReadExtra read_extra) {
  RestartBundle bundle;
  auto &ck = bundle.checkpoint;
  int ok = 1;
  if (rank == 0) {
    bundle.path = resolve_checkpoint_bundle(cfg.restart_dir).string();
    if (bundle.path.empty()) {
      ok = 0;
    } else {
      std::ifstream in(bundle.path + "/state.txt");
      if (!in || !read_checkpoint_text(in, ck) ||
          !checkpoint_matches_problem(ck, cfg, spec.C_target, spec.W) ||
          !checkpoint_is_restartable(ck, cfg.steps) || !read_extra(bundle.path, ck))
        ok = 0;
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int length = static_cast<int>(bundle.path.size());
  MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
  bundle.path.resize(static_cast<std::size_t>(length));
  if (length > 0) MPI_Bcast(bundle.path.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ck, static_cast<int>(sizeof(ck)), MPI_BYTE, 0, MPI_COMM_WORLD);
  bundle.ready = ok != 0;
  return bundle;
}

} // namespace pfc::apps::inverse::driver
