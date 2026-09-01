// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>

#include <alloy_pf_karma2001_benchmark/cli.hpp>
#include <alloy_pf_karma2001_benchmark/engine.hpp>

int main(int argc, char **argv) {
  const auto cfg = alloy_pf_karma2001_benchmark::parse_or_print_usage(argc, argv);
  if (!cfg) {
    return EXIT_FAILURE;
  }

  const bool skip_png = std::getenv("OPENPFC_KARMA_SKIP_PNG") != nullptr;
  const bool quiet = std::getenv("OPENPFC_KARMA_QUIET") != nullptr;

  std::filesystem::create_directories(cfg->output_dir);

  const auto &p = cfg->phys;
  std::cout << std::setprecision(10);
  std::cout << "KARMA2001 d0/W=" << p.d0_over_W << " d0=" << p.d0 << " m  W0=" << p.W0
            << " m  D=" << p.D << " m2/s  lambda=" << p.lambda << " A_trap=" << p.A_trap
            << " a2=" << p.a2 << " VD=" << p.VD_pf << " m/s  beta0=" << p.beta0
            << " s/m  tau0=" << p.tau0 << " s  Nx=" << cfg->Nx << " Ny=" << cfg->Ny
            << " steps=" << cfg->n_steps << " dx=" << p.dx << " dt=" << p.dt
            << " glasner=" << (cfg->use_glasner ? 1 : 0) << " iso=" << (cfg->use_isotropic ? 1 : 0)
            << " noise_F0=" << cfg->noise_F0 << " noise_seed=" << cfg->noise_seed
            << " Tdot=" << p.Tdot << " Omega=" << p.Omega << " dT_gt=" << p.dT_gt
            << " Nz=" << cfg->Nz << " phi1_deg="
            << (p.phi1 * 180.0 / std::acos(-1.0)) << " r_seed=" << p.r_seed
            << " out=" << cfg->output_dir << "\n";

  const auto res = alloy_pf_karma2001_benchmark::engine::run(*cfg, skip_png, quiet);

  const double rel_mass = (res.mass1 - res.mass0) / std::max(std::abs(res.mass0), 1.0e-30);
  std::cout << std::setprecision(17);
  std::cout << "KARMA_VERIFY wall_loop_s=" << res.wall_loop_s << " nthreads=" << res.nthreads
            << " mass0=" << res.mass0 << " mass1=" << res.mass1 << " rel_mass_err=" << rel_mass
            << " x_tip=" << res.x_tip << " rho=" << res.rho_tip << " min_phi=" << res.min_phi
            << " max_phi=" << res.max_phi << " min_c=" << res.min_c << " max_c=" << res.max_c
            << "\n";
  return EXIT_SUCCESS;
}
