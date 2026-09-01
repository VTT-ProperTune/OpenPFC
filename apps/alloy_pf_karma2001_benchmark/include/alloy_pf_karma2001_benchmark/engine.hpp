// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <string>
#include <vector>

#include <alloy_pf_karma2001_benchmark/cli.hpp>

namespace alloy_pf_karma2001_benchmark::engine {

struct RunResult {
  std::vector<double> phi_xy;
  std::vector<double> c_xy;
  int Nx = 0;
  int Ny = 0;
  int Nz = 1;
  double wall_loop_s = 0.0;
  int nthreads = 1;
  double mass0 = 0.0;
  double mass1 = 0.0;
  double x_tip = 0.0;
  double rho_tip = 0.0;
  double min_phi = 0.0;
  double max_phi = 0.0;
  double min_c = 0.0;
  double max_c = 0.0;
};

RunResult run(const RunConfig &cfg, bool skip_png, bool quiet);

} // namespace alloy_pf_karma2001_benchmark::engine
