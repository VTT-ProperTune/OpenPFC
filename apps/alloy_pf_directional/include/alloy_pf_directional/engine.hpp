// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * OpenMP FTA engine: 2D (Nz=1) or 3D regular grid. Persistent fields are
 * φ/ψ, c, ∂tφ, and (default) e^u/u — see recompute.hpp. Anisotropy fluxes
 * are recomputed unless STORE_AUX. Classic AMR is deferred (README).
 */

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <string>

#include <alloy_pf_directional/cli.hpp>

namespace alloy_pf_directional::engine {

/** Per-stage wall times (timed window, or the whole loop if TIMED_STEPS=0). */
struct PerfTimes {
  double ghost_s = 0.0;
  double eu_s = 0.0;
  double flux_s = 0.0;
  double grain_s = 0.0;
  double euler_s = 0.0;
  double solute_s = 0.0;
  double reduce_s = 0.0;
  double io_s = 0.0;
  int store_eu = 1;
  int store_aux = 0;
  int n_persistent_fields = 0;
  std::size_t alloc_bytes = 0;
  std::size_t bytes_per_cell = 0;
};

struct RunResult {
  double wall_loop_s = 0.0;
  double halo_s = 0.0;
  double kernel_s = 0.0;
  double time_per_step_s = 0.0;
  PerfTimes perf;
  int nthreads = 1;
  int nproc = 1;
  double mass0 = 0.0;
  double mass1 = 0.0;
  double min_phi = 0.0;
  double max_phi = 0.0;
  double min_c = 0.0;
  double max_c = 0.0;
  double x_tip = 0.0;
  double sum_phi = 0.0;
  double sum_c = 0.0;
  int n_steps_done = 0;
  int n_timed = 0;
  int window_shift_cells = 0;
  bool hit_right = false;
  bool hit_far_c = false;
  bool blew_up = false;
  std::string abort_reason;
};

RunResult run(const RunConfig &cfg, bool skip_png, bool quiet);

inline void print_directional_perf(std::ostream &os, const RunResult &r, int Nx, int Ny, int Nz) {
  const auto &p = r.perf;
  const double staged = p.ghost_s + p.eu_s + p.flux_s + p.grain_s + p.euler_s + p.solute_s +
                        p.reduce_s + p.io_s;
  const double denom = staged > 0.0 ? staged : 1.0;
  const int n = r.n_timed > 0 ? r.n_timed : std::max(1, r.n_steps_done);
  auto frac = [&](double s) { return 100.0 * s / denom; };
  os << "ALCU_PERF backend=openmp nproc=1 store_eu=" << p.store_eu
     << " store_aux=" << p.store_aux << " nthreads=" << r.nthreads << " Nx=" << Nx
     << " Ny=" << Ny << " Nz=" << Nz << " n_timed=" << n
     << " time_per_step_s=" << r.time_per_step_s << " wall_loop_s=" << r.wall_loop_s
     << " ghost_s=" << p.ghost_s << " ghost_pct=" << frac(p.ghost_s) << " halo_s=" << p.ghost_s
     << " halo_pct=" << frac(p.ghost_s) << " eu_s=" << p.eu_s
     << " eu_pct=" << frac(p.eu_s) << " flux_s=" << p.flux_s << " flux_pct=" << frac(p.flux_s)
     << " grain_s=" << p.grain_s << " grain_pct=" << frac(p.grain_s) << " euler_s=" << p.euler_s
     << " euler_pct=" << frac(p.euler_s) << " solute_s=" << p.solute_s
     << " solute_pct=" << frac(p.solute_s) << " reduce_s=" << p.reduce_s
     << " reduce_pct=" << frac(p.reduce_s) << " io_s=" << p.io_s << " io_pct=" << frac(p.io_s)
     << " fields=" << p.n_persistent_fields << " bytes_per_cell=" << p.bytes_per_cell
     << " alloc_bytes=" << p.alloc_bytes << "\n";
}

/** MPI/HIP: halo vs kernel split (solute is inside kernel_s). */
inline void print_directional_perf_halo_kernel(std::ostream &os, const char *backend, int nproc,
                                        int Nx, int Ny, int Nz, int n_timed,
                                        double time_per_step_s, double wall_loop_s,
                                        double halo_s, double kernel_s) {
  const double denom = (halo_s + kernel_s) > 0.0 ? (halo_s + kernel_s) : 1.0;
  auto frac = [&](double s) { return 100.0 * s / denom; };
  os << "ALCU_PERF backend=" << backend << " nproc=" << nproc << " Nx=" << Nx << " Ny=" << Ny
     << " Nz=" << Nz << " n_timed=" << n_timed << " time_per_step_s=" << time_per_step_s
     << " wall_loop_s=" << wall_loop_s << " halo_s=" << halo_s << " halo_pct=" << frac(halo_s)
     << " ghost_s=" << halo_s << " ghost_pct=" << frac(halo_s) << " kernel_s=" << kernel_s
     << " kernel_pct=" << frac(kernel_s) << " solute_s=-1 solute_pct=-1\n";
}

} // namespace alloy_pf_directional::engine
