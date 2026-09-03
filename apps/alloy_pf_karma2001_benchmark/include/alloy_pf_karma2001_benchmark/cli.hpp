// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include <alloy_pf_karma2001_benchmark/defaults.hpp>

namespace alloy_pf_karma2001_benchmark {

struct RunConfig {
  Physics phys = make_physics(0.544);
  int Nx = 0;
  int Ny = 0;
  int Nz = 1;
  int n_steps = 0;
  double t_star_max = 10000.0;
  std::string output_dir = "results/alloy_pf_karma2001_benchmark";
  int num_threads = 0;
  int nprint = 200;
  int nsave = 5000;
  int n_hist = 20;
  double L_over_d0 = 1000.0;
  /** Abort on any far-wall interaction (solid, φ, or c). 0 disables. */
  double stop_frac = 0.80;
  /** Number of φ=0 isoline dumps (including t=0). 0 disables. AM default 12. */
  int n_contour = 0;
  bool use_glasner = true;
  /** Ji et al. JCP 2022 isotropic FD (\(\bar S_{2,1}\) in 2D, \(\bar S_{1,2,0}\) in 3D). */
  bool use_isotropic = true;
  /** 2 = 5-point / Ji (2nd order). 4 = 4th-order Cartesian Laplacian on φ. */
  int fd_order = 2;
  /** Lab-frame origin of cell (1,1,1). Default (0,0): quarter seed, x≥0, y≥0. */
  double origin_x = 0.0;
  double origin_y = 0.0;
  double origin_z = 0.0;
  /** 1 = quarter (symmetry on x=0,y=0), 2 = x≥0 half-plane, 4 = full plane. */
  int n_halves = 1;
  /** FDT interface noise on φ. 0 disables. Default off; env OPENPFC_KARMA_NOISE. */
  double noise_F0 = 0.0;
  unsigned noise_seed = kNoiseSeed;
};

inline void print_usage(std::ostream &os, const char *exe) {
  os << "Karma 2001 present-model isothermal dendrite (OpenMP).\n"
     << "Independent OpenMP FD app. Paper suite:\n"
     << "  scripts/run_karma2001_benchmark.sh          # 3 [100] cases, t*=10^4\n"
     << "  QUICK=1 scripts/run_karma2001_benchmark.sh   # short t* pipeline check\n"
     << "Defaults: A=β0=εk=0, k=0.15, εc=0.02, Ω=0.55, seed 22 d0, L/d0=1000.\n"
     << "Usage:\n  " << exe << " glasner [d0_over_W] [phi1_deg] [output_dir] [nthreads]\n"
     << "  " << exe << " fine [d0_over_W] [output_dir] [nthreads]\n"
     << "  " << exe << " smoke [nthreads]\n"
     << "  " << exe << " am [W0_nm] [phi1_deg] [output_dir] [nthreads]   # extra: cooling\n"
     << "  " << exe << " smoke3d [nthreads]   # extra: tiny 3D brick\n"
     << "glasner: Δx=W0, Glasner ψ, Ji 9-pt (case 1–2). Default d0/W=0.277, φ1=0.\n"
     << "fine: 2001-like mesh — Δx=0.4 W0, no Glasner, 5-pt, τ frozen at e^u=1.\n"
     << "phi1_deg is Bunge φ1 ([100] is 0). Φ=φ2=0 unless OPENPFC_KARMA_PHI / PHI2.\n"
     << "Env (paper): OPENPFC_KARMA_DT=<dt/tau0>  OPENPFC_KARMA_DX=<dx/W0>\n"
     << "     OPENPFC_KARMA_TSTAR=<t D/d0^2>  OPENPFC_KARMA_LD0=<L/d0>\n"
     << "     OPENPFC_KARMA_GLASNER=0  OPENPFC_KARMA_ISO=0  OPENPFC_KARMA_TAU_EU=0\n"
     << "     OPENPFC_KARMA_SKIP_PNG=1  OPENPFC_KARMA_QUIET=1  OPENPFC_KARMA_MAX_STEPS=N\n"
     << "     OPENPFC_KARMA_HALVES=2|4  OPENPFC_KARMA_NHIST=<n>  OPENPFC_KARMA_FD=4\n"
     << "     OPENPFC_KARMA_SEED_D0=<R/d0>  OPENPFC_KARMA_STOP_FRAC=<0–1>\n"
     << "     OPENPFC_KARMA_EPSC  OPENPFC_KARMA_K  OPENPFC_KARMA_OMEGA\n"
     << "Extra trapping/AM: OPENPFC_KARMA_VD OPENPFC_KARMA_BETA0 OPENPFC_KARMA_EPSK\n"
     << "     OPENPFC_KARMA_TDOT OPENPFC_KARMA_TEND OPENPFC_KARMA_TDECAY OPENPFC_KARMA_L\n"
     << "     OPENPFC_KARMA_NCONTOUR OPENPFC_KARMA_DTEXTRA OPENPFC_KARMA_NOISE\n";
}

inline RunConfig sized_from_physics(const Physics &phys, double t_star_max, double L_over_d0,
                                    int Nz = 1) {
  RunConfig c;
  c.phys = phys;
  c.t_star_max = t_star_max;
  c.L_over_d0 = L_over_d0;
  c.Nz = Nz;
  const double L = L_over_d0 * phys.d0;
  c.Nx = static_cast<int>(std::ceil(L / phys.dx)) + 2;
  c.Ny = c.Nx;
  if (Nz > 1) {
    c.Nz = c.Nx;
  }
  const double t_end = t_star_max * phys.d0 * phys.d0 / phys.D;
  c.n_steps = std::max(1, static_cast<int>(std::ceil(t_end / phys.dt)));
  return c;
}

inline void apply_dx_env(Physics &phys) {
  if (const char *e = std::getenv("OPENPFC_KARMA_DX")) {
    const double dxW = std::atof(e);
    if (dxW > 0.0 && std::isfinite(dxW)) {
      set_dx_over_W(phys, dxW);
    }
  }
}

inline void apply_dt_env(Physics &phys) {
  if (const char *e = std::getenv("OPENPFC_KARMA_DT")) {
    const double dtt = std::atof(e);
    if (dtt > 0.0 && std::isfinite(dtt)) {
      set_dt_over_tau(phys, dtt);
    }
  }
}

inline void apply_kinetics_env(Physics &phys) {
  if (const char *e = std::getenv("OPENPFC_KARMA_VD")) {
    const double vd = std::atof(e);
    if (std::isfinite(vd) && vd >= 0.0) {
      phys.VD_pf = vd;
    }
  }
  if (const char *e = std::getenv("OPENPFC_KARMA_BETA0")) {
    const double b = std::atof(e);
    if (std::isfinite(b) && b >= 0.0) {
      phys.beta0 = b;
    }
  }
  if (const char *e = std::getenv("OPENPFC_KARMA_EPSC")) {
    const double v = std::atof(e);
    if (std::isfinite(v) && v >= 0.0) {
      phys.eps_c = v;
    }
  }
  if (const char *e = std::getenv("OPENPFC_KARMA_EPSK")) {
    const double v = std::atof(e);
    if (std::isfinite(v) && v >= 0.0) {
      phys.eps_k = v;
    }
  }
  if (const char *e = std::getenv("OPENPFC_KARMA_K")) {
    const double v = std::atof(e);
    if (std::isfinite(v) && v > 0.0 && v < 1.0) {
      phys.k = v;
    }
  }
  if (const char *e = std::getenv("OPENPFC_KARMA_OMEGA")) {
    const double v = std::atof(e);
    if (std::isfinite(v) && v > 0.0 && v < 1.0) {
      phys.Omega = v;
    }
  }
  refresh_derived(phys);
  set_dx_over_W(phys, phys.dx / phys.W0);
}

inline void apply_ld0_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_KARMA_LD0")) {
    const double ld0 = std::atof(e);
    if (ld0 > 0.0 && std::isfinite(ld0)) {
      c.L_over_d0 = ld0;
      const double L = ld0 * c.phys.d0;
      c.Nx = static_cast<int>(std::ceil(L / c.phys.dx)) + 2;
      c.Ny = c.Nx;
      if (c.Nz > 1) {
        c.Nz = c.Nx;
      }
    }
  }
}

inline void apply_stop_frac_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_KARMA_STOP_FRAC")) {
    const double v = std::atof(e);
    if (v >= 0.0 && v <= 1.0 && std::isfinite(v)) {
      c.stop_frac = v;
    }
  }
}

inline void apply_tau_eu_env(Physics &phys) {
  if (const char *e = std::getenv("OPENPFC_KARMA_TAU_EU")) {
    const std::string v(e);
    phys.tau_eu_local = !(v == "0" || v == "off" || v == "false" || v == "OFF" || v == "FALSE");
  }
}

inline void apply_seed_env(Physics &phys) {
  if (const char *e = std::getenv("OPENPFC_KARMA_SEED_D0")) {
    const double v = std::atof(e);
    if (v > 0.0 && std::isfinite(v)) {
      phys.r_seed = v * phys.d0;
      phys.dT_gt = (phys.r_seed > 0.0) ? (phys.Gamma / phys.r_seed) : 0.0;
    }
  }
}

inline void apply_ncontour_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_KARMA_NCONTOUR")) {
    const int v = std::atoi(e);
    if (v >= 0) {
      c.n_contour = v;
    }
  }
}

inline void apply_bunge_env(Physics &phys) {
  if (const char *e = std::getenv("OPENPFC_KARMA_PHI")) {
    phys.Phi = std::atof(e) * std::acos(-1.0) / 180.0;
  }
  if (const char *e = std::getenv("OPENPFC_KARMA_PHI2")) {
    phys.phi2 = std::atof(e) * std::acos(-1.0) / 180.0;
  }
}

inline std::optional<RunConfig> parse_args(int argc, char **argv) {
  if (argc >= 2 && std::string(argv[1]) == "--help") {
    return std::nullopt;
  }
  if (argc >= 2 && std::string(argv[1]) == "smoke") {
    Physics phys = make_physics(0.544);
    apply_kinetics_env(phys);
    RunConfig c = sized_from_physics(phys, 50.0, 80.0);
    c.use_glasner = true;
    c.output_dir = "results/alloy_pf_karma2001_benchmark/smoke";
    c.nprint = 50;
    c.nsave = 200;
    c.n_hist = 5;
    if (argc >= 3) {
      c.num_threads = std::atoi(argv[2]);
    }
    return c;
  }
  if (argc >= 2 && std::string(argv[1]) == "smoke3d") {
    Physics phys = make_physics(0.544);
    apply_kinetics_env(phys);
    RunConfig c = sized_from_physics(phys, 400.0, 70.0, /*Nz=*/2);
    c.use_glasner = true;
    c.output_dir = "results/alloy_pf_karma2001_benchmark/smoke3d";
    c.nprint = 50;
    c.nsave = 200;
    c.n_hist = 5;
    if (argc >= 3) {
      c.num_threads = std::atoi(argv[2]);
    }
    return c;
  }

  if (argc >= 2 && std::string(argv[1]) == "am") {
    double W0_nm = 20.0;
    double phi1_deg = 45.0;
    int arg0 = 2;
    if (argc >= arg0 + 1 &&
        std::string(argv[arg0]).find_first_not_of("+-0123456789.eE") == std::string::npos) {
      W0_nm = std::atof(argv[arg0]);
      ++arg0;
    }
    if (argc >= arg0 + 1 &&
        std::string(argv[arg0]).find_first_not_of("+-0123456789.eE") == std::string::npos) {
      phi1_deg = std::atof(argv[arg0]);
      ++arg0;
    }
    if (!(W0_nm > 0.0) || !std::isfinite(W0_nm) || !std::isfinite(phi1_deg)) {
      return std::nullopt;
    }
    Physics phys = make_physics_w0(W0_nm * 1.0e-9);
    phys.phi1 = phi1_deg * std::acos(-1.0) / 180.0;
    double Tdot = kTdotAm;
    double dT_extra = kDTExtra;
    double t_end = kTendAm;
    if (const char *e = std::getenv("OPENPFC_KARMA_TDOT")) {
      const double v = std::atof(e);
      if (v > 0.0 && std::isfinite(v)) {
        Tdot = v;
      }
    }
    if (const char *e = std::getenv("OPENPFC_KARMA_DTEXTRA")) {
      const double v = std::atof(e);
      if (v >= 0.0 && std::isfinite(v)) {
        dT_extra = v;
      }
    }
    if (const char *e = std::getenv("OPENPFC_KARMA_TEND")) {
      const double v = std::atof(e);
      if (v > 0.0 && std::isfinite(v)) {
        t_end = v;
      }
    }
    set_am_cooling(phys, Tdot, dT_extra);
    if (const char *e = std::getenv("OPENPFC_KARMA_TDECAY")) {
      const double v = std::atof(e);
      if (v >= 0.0 && std::isfinite(v)) {
        phys.t_decay = v;
      }
    }
    apply_kinetics_env(phys);
    apply_dx_env(phys);
    apply_dt_env(phys);
    apply_bunge_env(phys);
    double L_phys = kLAm;
    double stop_frac = kStopFrac;
    if (const char *e = std::getenv("OPENPFC_KARMA_L")) {
      const double v = std::atof(e);
      if (v > 0.0 && std::isfinite(v)) {
        L_phys = v;
      }
    }
    if (const char *e = std::getenv("OPENPFC_KARMA_STOP_FRAC")) {
      const double v = std::atof(e);
      if (v > 0.0 && v <= 1.0 && std::isfinite(v)) {
        stop_frac = v;
      }
    }
    const double t_star = t_end * phys.D / (phys.d0 * phys.d0);
    RunConfig c = sized_from_physics(phys, t_star, L_phys / phys.d0);
    c.stop_frac = stop_frac;
    c.n_contour = 12;
    apply_ncontour_env(c);
    c.use_glasner = true;
    c.noise_F0 = 0.0;
    c.output_dir = "results/alloy_pf_karma2001_benchmark/am_W" +
                   std::to_string(static_cast<int>(std::lround(W0_nm))) + "nm_th" +
                   std::to_string(static_cast<int>(std::lround(phi1_deg)));
    if (argc > arg0 && argv[arg0][0] != '\0' &&
        std::string(argv[arg0]).find_first_not_of("+-0123456789") != std::string::npos) {
      c.output_dir = argv[arg0];
      if (argc > arg0 + 1) {
        c.num_threads = std::atoi(argv[arg0 + 1]);
      }
    } else if (argc > arg0) {
      c.num_threads = std::atoi(argv[arg0]);
    }
    return c;
  }

  auto glasner_cfg = [&](double d0W, double phi1_deg) {
    Physics phys = make_physics(d0W);
    phys.phi1 = phi1_deg * std::acos(-1.0) / 180.0;
    apply_kinetics_env(phys);
    apply_dx_env(phys);
    apply_dt_env(phys);
    apply_bunge_env(phys);
    RunConfig c = sized_from_physics(phys, 10000.0, 1000.0);
    c.use_glasner = true;
    c.n_contour = 2; // t = 0 and final φ=0 isoline
    c.output_dir = "results/alloy_pf_karma2001_benchmark/d0W_" + std::to_string(d0W) + "_th" +
                   std::to_string(static_cast<int>(std::lround(phi1_deg)));
    return c;
  };

  const auto is_glasner = (argc < 2) || std::string(argv[1]) == "glasner" ||
                          (std::string(argv[1]).find_first_not_of("+-0123456789.eE") ==
                           std::string::npos);
  if (is_glasner && !(argc >= 2 && (std::string(argv[1]) == "smoke" ||
                                   std::string(argv[1]) == "smoke3d" ||
                                   std::string(argv[1]) == "fine" ||
                                   std::string(argv[1]) == "am"))) {
    double d0W = 0.277;
    double phi1_deg = 0.0;
    int arg0 = 1;
    if (argc >= 2 && std::string(argv[1]) == "glasner") {
      arg0 = 2;
    }
    if (argc >= arg0 + 1) {
      d0W = std::atof(argv[arg0]);
    }
    if (argc >= arg0 + 2) {
      const std::string a = argv[arg0 + 1];
      if (a.find_first_not_of("+-0123456789.eE") == std::string::npos) {
        phi1_deg = std::atof(a.c_str());
      }
    }
    if (!(d0W > 0.0) || !std::isfinite(d0W) || !std::isfinite(phi1_deg)) {
      return std::nullopt;
    }
    RunConfig c = glasner_cfg(d0W, phi1_deg);
    const int dir_arg = arg0 + 2;
    if (argc > dir_arg && argv[dir_arg][0] != '\0' &&
        std::string(argv[dir_arg]).find_first_not_of("+-0123456789") != std::string::npos) {
      c.output_dir = argv[dir_arg];
      if (argc > dir_arg + 1) {
        c.num_threads = std::atoi(argv[dir_arg + 1]);
      }
    } else if (argc > dir_arg) {
      c.num_threads = std::atoi(argv[dir_arg]);
    }
    return c;
  }

  if (std::string(argv[1]) == "fine") {
    double d0W = 0.277;
    int arg0 = 2;
    if (argc >= 3 &&
        std::string(argv[2]).find_first_not_of("+-0123456789.eE") == std::string::npos) {
      d0W = std::atof(argv[2]);
      ++arg0;
    }
    if (!(d0W > 0.0) || !std::isfinite(d0W)) {
      return std::nullopt;
    }
    Physics phys = make_physics(d0W);
    apply_kinetics_env(phys);
    set_dx_over_W(phys, kDx);
    set_dt_over_tau(phys, kDtGlasner);
    phys.tau_eu_local = false;
    RunConfig c = sized_from_physics(phys, 10000.0, 1000.0);
    c.use_glasner = false;
    c.use_isotropic = false;
    c.output_dir = "results/alloy_pf_karma2001_benchmark/d0W_" + std::to_string(d0W) + "_paperlike";
    if (argc > arg0 && argv[arg0][0] != '\0' &&
        std::string(argv[arg0]).find_first_not_of("+-0123456789") != std::string::npos) {
      c.output_dir = argv[arg0];
      if (argc > arg0 + 1) {
        c.num_threads = std::atoi(argv[arg0 + 1]);
      }
    } else if (argc > arg0) {
      c.num_threads = std::atoi(argv[arg0]);
    }
    return c;
  }

  return std::nullopt;
}

inline void apply_iso_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_KARMA_ISO")) {
    const std::string v(e);
    c.use_isotropic = !(v == "0" || v == "off" || v == "false" || v == "OFF" || v == "FALSE");
  }
}

inline void apply_fd_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_KARMA_FD")) {
    const int o = std::atoi(e);
    if (o == 4) {
      c.fd_order = 4;
      c.use_isotropic = false;
    } else if (o == 2) {
      c.fd_order = 2;
    }
  }
}

inline void apply_glasner_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_KARMA_GLASNER")) {
    const std::string v(e);
    c.use_glasner = !(v == "0" || v == "off" || v == "false" || v == "OFF" || v == "FALSE");
  }
}

inline void apply_nhist_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_KARMA_NHIST")) {
    const int v = std::atoi(e);
    if (v > 0) {
      c.n_hist = v;
    }
  }
}

inline void apply_halves_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_KARMA_HALVES")) {
    const int h = std::atoi(e);
    if (h == 2 || h == 4) {
      c.n_halves = h;
      const int n0 = c.Nx;
      if (h == 4) {
        c.Nx = 2 * n0;
        c.Ny = 2 * n0;
        c.origin_x = -0.5 * static_cast<double>(c.Nx) * c.phys.dx;
        c.origin_y = -0.5 * static_cast<double>(c.Ny) * c.phys.dx;
      } else {
        c.Ny = 2 * n0;
        c.origin_x = 0.0;
        c.origin_y = -0.5 * static_cast<double>(c.Ny) * c.phys.dx;
      }
    }
  }
}

inline void apply_step_cap(RunConfig &c) {
  if (const char *ms = std::getenv("OPENPFC_KARMA_MAX_STEPS")) {
    const int cap = std::atoi(ms);
    if (cap > 0) {
      c.n_steps = std::min(c.n_steps, cap);
    }
  }
}

inline void apply_noise_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_KARMA_NOISE")) {
    const std::string v(e);
    if (v == "1" || v == "on" || v == "true" || v == "ON" || v == "TRUE") {
      c.noise_F0 = kNoiseF0;
    } else if (v == "0" || v == "off" || v == "false" || v == "OFF" || v == "FALSE") {
      c.noise_F0 = 0.0;
    } else {
      const double f = std::atof(e);
      if (std::isfinite(f) && f >= 0.0) {
        c.noise_F0 = f;
      }
    }
  }
  if (const char *e = std::getenv("OPENPFC_KARMA_NOISE_SEED")) {
    c.noise_seed = static_cast<unsigned>(std::strtoul(e, nullptr, 10));
  }
}

inline void apply_tstar_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_KARMA_TSTAR")) {
    const double ts = std::atof(e);
    if (ts > 0.0 && std::isfinite(ts)) {
      c.t_star_max = ts;
      const double t_end = ts * c.phys.d0 * c.phys.d0 / c.phys.D;
      c.n_steps = std::max(1, static_cast<int>(std::ceil(t_end / c.phys.dt)));
    }
  }
}

inline std::optional<RunConfig> parse_or_print_usage(int argc, char **argv) {
  auto cfg = parse_args(argc, argv);
  if (!cfg) {
    print_usage(std::cerr, argv[0]);
    return cfg;
  }
  apply_iso_env(*cfg);
  apply_fd_env(*cfg);
  apply_glasner_env(*cfg);
  apply_dt_env(cfg->phys);
  apply_tau_eu_env(cfg->phys);
  apply_seed_env(cfg->phys);
  apply_ld0_env(*cfg);
  apply_halves_env(*cfg);
  if (std::getenv("OPENPFC_KARMA_NHIST") == nullptr && !(cfg->phys.Tdot > 0.0)) {
    cfg->n_hist = default_n_hist(cfg->phys);
  }
  apply_nhist_env(*cfg);
  apply_stop_frac_env(*cfg);
  {
    const double t_end = cfg->t_star_max * cfg->phys.d0 * cfg->phys.d0 / cfg->phys.D;
    cfg->n_steps = std::max(1, static_cast<int>(std::ceil(t_end / cfg->phys.dt)));
  }
  apply_tstar_env(*cfg);
  apply_ncontour_env(*cfg);
  apply_noise_env(*cfg);
  apply_step_cap(*cfg);
  if (cfg->phys.Tdot > 0.0 && std::getenv("OPENPFC_KARMA_NOISE") == nullptr) {
    cfg->noise_F0 = 0.0;
  }
  return cfg;
}

} // namespace alloy_pf_karma2001_benchmark
