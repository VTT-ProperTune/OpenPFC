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
  std::string output_dir = "results/karma2001";
  int num_threads = 0;
  int nprint = 200;
  int nsave = 5000;
  int n_hist = 20;
  double L_over_d0 = 460.0;
  /** Abort on any far-wall interaction (solid, φ, or c). 0 disables. */
  double stop_frac = 0.80;
  /** Number of φ=0 isoline dumps (including t=0). 0 disables. AM default 12. */
  int n_contour = 0;
  bool use_glasner = true;
  /** Ji et al. JCP 2022 isotropic FD (\(\bar S_{2,1}\) in 2D, \(\bar S_{1,2,0}\) in 3D). */
  bool use_isotropic = true;
  /** FDT interface noise on φ. 0 disables. Default off; env OPENPFC_KARMA_NOISE. */
  double noise_F0 = 0.0;
  unsigned noise_seed = kNoiseSeed;
};

inline void print_usage(std::ostream &os, const char *exe) {
  os << "Karma 2001 present-model isothermal dendrite (OpenMP).\n"
     << "Cubic a_s(n)=1-3ε_c+4ε_c∑n_i^4, a_k=1+3ε_k-4ε_k∑n_i^4; τ from Pinomaa (2020)\n"
     << "eq. (7) at W_s=W0 a_s and β_k=β0 a_k. A is the trapping parameter. Glasner Δx=W0.\n"
     << "Usage:\n  " << exe << " glasner [d0_over_W] [phi1_deg] [output_dir] [nthreads]\n"
     << "  " << exe << " am [W0_nm] [phi1_deg] [output_dir] [nthreads]\n"
     << "  " << exe << " smoke\n"
     << "  " << exe << " smoke3d [nthreads]\n"
     << "  " << exe << " fine [d0_over_W]   # Δx=0.4 W0, no Glasner (legacy)\n"
     << "phi1_deg is Bunge φ1 (2D rotation). Φ=φ2=0 unless set via env\n"
     << "OPENPFC_KARMA_PHI=deg OPENPFC_KARMA_PHI2=deg.\n"
     << "Env: OPENPFC_KARMA_SKIP_PNG=1  OPENPFC_KARMA_QUIET=1  OPENPFC_KARMA_MAX_STEPS=N\n"
     << "     OPENPFC_KARMA_DX=<dx/W0>  (Glasner; dt = 0.02·(dx/W0)·τ0)\n"
     << "     OPENPFC_KARMA_DT=<dt/tau0>  (default 0.02; scales with dx/W0)\n"
     << "     OPENPFC_KARMA_ISO=0       (disable Ji 2022 isotropic FD; default on)\n"
     << "     OPENPFC_KARMA_VD=<m/s>    OPENPFC_KARMA_BETA0=<s/m>\n"
     << "     OPENPFC_KARMA_TSTAR=<t D/d0^2>  (default 10000)\n"
     << "     OPENPFC_KARMA_TDOT=<K/s>  OPENPFC_KARMA_TEND=<s>  (am protocol)\n"
     << "     OPENPFC_KARMA_TDECAY=<s>  Ṫ(t)=Ṫ₀ e^{−t/τ}; 0 = linear Ṫ t (am)\n"
     << "     OPENPFC_KARMA_NCONTOUR=<n>  φ=0 isoline dumps (glasner 2, am 12)\n"
     << "     OPENPFC_KARMA_L=<m>       OPENPFC_KARMA_STOP_FRAC=<0–1>  (am box / far-wall abort)\n"
     << "     OPENPFC_KARMA_DTEXTRA=<K>  extra undercooling vs Γ/R (default 0.05)\n"
     << "     OPENPFC_KARMA_NOISE=<F0>  FDT φ-noise; 0 off, 1/on → 1e-3, or a value\n"
     << "     OPENPFC_KARMA_NOISE_SEED=<u>  (default 1)\n";
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
  refresh_derived(phys);
  set_dx_over_W(phys, phys.dx / phys.W0);
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
    c.output_dir = "results/karma2001_smoke";
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
    c.output_dir = "results/karma2001_smoke3d";
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
    c.output_dir = "results/karma2001_am_W" + std::to_string(static_cast<int>(std::lround(W0_nm))) +
                   "nm_th" + std::to_string(static_cast<int>(std::lround(phi1_deg)));
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
    RunConfig c = sized_from_physics(phys, 10000.0, 460.0);
    c.use_glasner = true;
    c.n_contour = 2; // t = 0 and final φ=0 isoline
    c.output_dir = "results/karma2001_trap_d0W_" + std::to_string(d0W) + "_th" +
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
    double d0W = 0.544;
    if (argc >= 3) {
      d0W = std::atof(argv[2]);
    }
    if (!(d0W > 0.0) || !std::isfinite(d0W)) {
      return std::nullopt;
    }
    Physics phys = make_physics(d0W);
    apply_kinetics_env(phys);
    set_dx_over_W(phys, kDx);
    RunConfig c = sized_from_physics(phys, 10000.0, 460.0);
    c.use_glasner = false;
    c.output_dir = "results/karma2001_d0W_" + std::to_string(d0W);
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
  apply_dt_env(cfg->phys);
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
