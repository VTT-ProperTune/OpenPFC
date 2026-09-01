// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>

#include <alloy_pf_directional/cli.hpp>
#include <alloy_pf_directional/engine.hpp>

int main(int argc, char **argv) {
  const auto cfg = alloy_pf_directional::parse_or_print_usage(argc, argv);
  if (!cfg) {
    return EXIT_FAILURE;
  }

  const bool skip_png = alloy_pf_directional::env_on("OPENPFC_ALCU_SKIP_PNG", false);
  const bool quiet = alloy_pf_directional::env_on("OPENPFC_ALCU_QUIET", false);

  std::filesystem::create_directories(cfg->output_dir);

  const auto &p = cfg->phys;
  std::cout << std::setprecision(10);
  std::cout << "ALCU_FTA G=" << p.G << " Vp=" << p.Vp << " x_tl=" << p.x_tl
            << " omega_zhong=" << (p.omega_zhong ? 1 : 0)
            << (p.omega_zhong ? " omega_solidus=" : " omega=")
            << (p.omega_zhong ? alloy_pf_directional::omega_at_solidus(p) : p.omega)
            << " dx=" << p.dx
            << " dt=" << p.dt << " dt_cfl_c=" << p.dt_cfl_c << " dt_cfl_phi=" << p.dt_cfl_phi
            << " dt_cfl_iface=" << p.dt_cfl_iface << " dt_tau=" << p.dt_tau << " Nx=" << cfg->Nx << " Ny=" << cfg->Ny
            << " Nz=" << cfg->Nz << " n_dim=" << p.n_dim
            << " steps=" << cfg->n_steps << " nsave=" << cfg->nsave
            << " vtk_every=" << cfg->vtk_every << " n_hist=" << cfg->n_hist
            << " A_trap=" << p.A_trap << " a2=" << p.a2 << " alpha_drag=" << p.alpha_drag
            << " tau0=" << p.tau0 << " u_corr=1"
            << " glasner=" << (cfg->use_glasner ? 1 : 0)
            << " iso=" << (cfg->use_isotropic ? 1 : 0)
            << " store_eu=" << (cfg->store_eu ? 1 : 0)
            << " store_aux=" << (cfg->store_aux ? 1 : 0)
            << " periodic_y=" << (cfg->periodic_y ? 1 : 0)
            << " periodic_z=" << (cfg->periodic_z ? 1 : 0)
            << " n_grains=" << cfg->n_grains
            << " phi1_g1_deg=" << (p.phi1_g1 * 180.0 / std::acos(-1.0))
            << " phi1_g2_deg=" << (p.phi1_g2 * 180.0 / std::acos(-1.0))
            << " r_seed=" << p.r_seed
            << " noise_F0=" << cfg->noise_F0 << " noise_seed=" << cfg->noise_seed
            << " W0=" << p.W0 << " out=" << cfg->output_dir << "\n";

  const auto res = alloy_pf_directional::engine::run(*cfg, skip_png, quiet);
  const double rel_mass = (res.mass1 - res.mass0) / std::max(std::abs(res.mass0), 1.0e-30);
  std::cout << std::setprecision(17);
  std::cout << "ALCU_VERIFY wall_loop_s=" << res.wall_loop_s << " nthreads=" << res.nthreads
            << " mass0=" << res.mass0 << " mass1=" << res.mass1 << " rel_mass_err=" << rel_mass
            << " min_phi=" << res.min_phi << " max_phi=" << res.max_phi << " min_c=" << res.min_c
            << " max_c=" << res.max_c << " x_tip=" << res.x_tip
            << " n_steps_done=" << res.n_steps_done << " hit_right=" << (res.hit_right ? 1 : 0)
            << " hit_far_c=" << (res.hit_far_c ? 1 : 0)
            << " blew_up=" << (res.blew_up ? 1 : 0)
            << " sum_phi=" << res.sum_phi << " sum_c=" << res.sum_c
            << " time_per_step_s=" << res.time_per_step_s << "\n";
  std::cout << std::setprecision(6);
  alloy_pf_directional::engine::print_directional_perf(std::cout, res, cfg->Nx, cfg->Ny, cfg->Nz);
  if (!res.abort_reason.empty() && !res.blew_up) {
    std::cout << "ALCU_STOP " << res.abort_reason << "\n";
  }
  if (res.blew_up) {
    std::cerr << "ALCU_ABORT " << res.abort_reason << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
