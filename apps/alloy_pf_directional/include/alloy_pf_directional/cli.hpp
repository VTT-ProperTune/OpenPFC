// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include <alloy_pf_directional/defaults.hpp>

namespace alloy_pf_directional {

struct RunConfig {
  Physics phys = make_physics();
  int Nx = 256;
  int Ny = 128;
  int Nz = 1;
  int n_steps = 0;
  double t_end = 2.0e-6;
  std::string output_dir = "results/alloy_pf_directional";
  int num_threads = 0;
  int nprint = 200;
  int nsave = kIoEverySnapshot;
  int n_hist = kIoEveryLog;
  int n_grains = 2;
  double seed_depth = kDsSeedDepth;
  double seed_bump = kDsSeedBump;
  double seed_bump_sigma = kDsSeedBumpSigma;
  bool use_glasner = true;
  bool use_isotropic = true;
  bool stop_on_right = false;
  /** Abort if any liquid-side far-face pixel leaves c_∞ (all pixels, not a mean). */
  bool stop_on_far_c = false;
  /** Transverse BCs. Directional (`ds`) defaults: periodic y, and periodic z in 3D. */
  bool periodic_y = false;
  bool periodic_z = false;
  bool skip_vtk = false;
  int vtk_every = kIoEverySnapshot;
  double noise_F0 = kNoiseF0;
  unsigned noise_seed = kNoiseSeed;
  /** Moving window: keep a slab around the front. 0 = off (full brick). */
  bool window_enable = false;
  int window_nx = 0;
  double window_margin_left = kWindowMarginLeft;
  double window_margin_right = kWindowMarginRight;
  double lab_Lx = 0.0;
  /** Block skip: 0 off; 16 or 32. */
  int block_skip = kBlockSkipDefault;
  double block_skip_tol_phi = kBlockSkipTolPhi;
  double block_skip_tol_c = kBlockSkipTolC;
  int block_skip_refresh = kBlockSkipRefresh;
  /** Scaling: warm-up then a timed window with I/O off. 0 = time the whole loop. */
  int warmup_steps = 0;
  int timed_steps = 0;
  /** Persist e^u and u (default). Anisotropy fluxes stay recomputed. */
  bool store_eu = true;
  /** 1: also persist jx/jy[/jz]. Implies store_eu. Default off (fluxes ~5% of 2D). */
  bool store_aux = false;
  /** CLI `repro`: 128×64, 40-step last-bit check (not the morphology product). */
  bool frozen_repro = false;
  /** LOCKED starting point: 12×3.2 µm W=10 nm two-grain strip. */
  bool frozen_benchmark = false;
};

inline void print_usage(std::ostream &os, const char *exe) {
  os << "Al-Cu FTA dilute alloy. T = Tl + G(x − xs − Vp t); xs = initial solidus (Bridgman).\n"
     << "Usage:\n  " << exe << " [output_dir] [nthreads]     LOCKED 12×3.2 µm W=10 nm bicrystal (start here)\n"
     << "  " << exe << " start|benchmark [output_dir] [nthreads]   same locked case\n"
     << "  " << exe << " smoke [output_dir] [nthreads]\n"
     << "  " << exe << " ds [output_dir] [nthreads] [--save-every N] [--log-every N]\n"
     << "  " << exe << " bicrystal [output_dir] [nthreads]   two-grain research CLI (env; not locked)\n"
     << "  " << exe << " repro [output_dir] [nthreads]       128×64, 40 steps, last-bit CI check\n"
     << "Env:\n"
     << "  OPENPFC_ALCU_SKIP_PNG=1  OPENPFC_ALCU_SKIP_VTK=1  OPENPFC_ALCU_QUIET=1\n"
     << "  OPENPFC_ALCU_MAX_STEPS=N  OPENPFC_ALCU_SAVE_EVERY=k  (PNG+VTK; ds default from W0)\n"
     << "  OPENPFC_ALCU_LOG_EVERY=k   (fields.log, default SAVE_EVERY/10)\n"
     << "  OPENPFC_ALCU_VTK_EVERY=k   (override VTK stride only)\n"
     << "  OPENPFC_ALCU_ISO=0 to disable Ji isotropic FD (default ON)\n"
     << "  ds only: OPENPFC_ALCU_W0  OPENPFC_ALCU_DXW  OPENPFC_ALCU_LX  OPENPFC_ALCU_LY\n"
     << "           OPENPFC_ALCU_LZ   (0 = 2D; >0 → 3D brick, n_dim=3)\n"
     << "           OPENPFC_ALCU_NZ / OPENPFC_ALCU_NDIM=3  (force 3D)\n"
     << "           OPENPFC_ALCU_NY=1  1D planar (set ISO=0; Ly is ignored if NY is set)\n"
     << "           OPENPFC_ALCU_G  OPENPFC_ALCU_VP  (Bridgman; defaults 5e6 K/m, 0.3 m/s)\n"
     << "           OPENPFC_ALCU_DELTA=Δ  uniform (Tl−T)/((1−ke)|ml|c0); use G=0 for 1D quench\n"
     << "           OPENPFC_ALCU_TEND  OPENPFC_ALCU_NGRANS  OPENPFC_ALCU_SEED\n"
     << "           OPENPFC_ALCU_THETA=deg  bicrystal: grains at ±θ (default 30)\n"
     << "           OPENPFC_ALCU_OMEGA=ω  constant grain coupling (default: Zhong ω(T))\n"
     << "           OPENPFC_ALCU_BUMP  OPENPFC_ALCU_BUMP_SIGMA\n"
     << "           OPENPFC_ALCU_STOP_RIGHT=0 to disable stop when φ>0 on the right wall\n"
     << "           OPENPFC_ALCU_STOP_FAR_C=0 to disable abort when far-face c leaves c_∞\n"
     << "           OPENPFC_ALCU_PERIODIC_Y=0 / PERIODIC_Z=0 to use no-flux instead\n"
     << "           (ds default: periodic y; periodic z when 3D)\n"
     << "  OPENPFC_ALCU_NOISE=F0     Langevin F/W0^d on φ (default 1e-3; 0 = off)\n"
     << "  OPENPFC_ALCU_NOISE_SEED=u  RNG seed for the noise (default 1)\n"
     << "  OPENPFC_ALCU_DT_OVER_TAU=r  dt = r τ0 if below Laplacian von Neumann and Δx/V_p\n"
     << "  OPENPFC_ALCU_WINDOW=1      moving slab around the front (regular grid)\n"
     << "  OPENPFC_ALCU_WINDOW_NX=N   slab cells in x (default 256)\n"
     << "  OPENPFC_ALCU_WINDOW_LEFT / WINDOW_RIGHT  margins in metres\n"
     << "  OPENPFC_ALCU_BLOCK_SKIP=16|32  skip idle 16³/32³ bricks (time, not memory)\n"
     << "  OPENPFC_ALCU_WARMUP=N  OPENPFC_ALCU_TIMED_STEPS=N  (I/O off in timed window)\n"
     << "  OPENPFC_ALCU_STORE_EU=0   recompute e^u/u (default stores them; no jx/jy)\n"
     << "  OPENPFC_ALCU_STORE_AUX=1  persist eu/u and jx/jy[/jz]\n";
}

inline bool env_on(const char *name, bool def) {
  if (const char *e = std::getenv(name)) {
    const std::string v(e);
    return !(v == "0" || v == "off" || v == "false" || v == "OFF" || v == "FALSE");
  }
  return def;
}

inline double env_d(const char *name, double def) {
  if (const char *e = std::getenv(name)) {
    return std::atof(e);
  }
  return def;
}

inline bool io_flag_takes_value(const std::string &a) {
  return a == "--save-every" || a == "--log-every" || a == "--vtk-every";
}

inline void apply_io_flags(RunConfig &c, int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (a == "--save-every" && i + 1 < argc) {
      const int k = std::atoi(argv[++i]);
      if (k > 0) {
        c.nsave = k;
        c.vtk_every = k;
        c.n_hist = std::max(1, k / 10);
      }
    } else if (a == "--log-every" && i + 1 < argc) {
      const int k = std::atoi(argv[++i]);
      if (k > 0) {
        c.n_hist = k;
      }
    } else if (a == "--vtk-every" && i + 1 < argc) {
      const int k = std::atoi(argv[++i]);
      if (k > 0) {
        c.vtk_every = k;
      }
    }
  }
}

inline void apply_vtk_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_ALCU_SKIP_VTK")) {
    const std::string v(e);
    c.skip_vtk = !(v == "0" || v == "off" || v == "false" || v == "OFF" || v == "FALSE");
  }
  if (const char *e = std::getenv("OPENPFC_ALCU_SAVE_EVERY")) {
    const int k = std::atoi(e);
    if (k > 0) {
      c.nsave = k;
      c.vtk_every = k;
      c.n_hist = std::max(1, k / 10);
    }
  }
  if (const char *e = std::getenv("OPENPFC_ALCU_LOG_EVERY")) {
    const int k = std::atoi(e);
    if (k > 0) {
      c.n_hist = k;
    }
  }
  if (const char *e = std::getenv("OPENPFC_ALCU_VTK_EVERY")) {
    const int k = std::atoi(e);
    if (k > 0) {
      c.vtk_every = k;
    }
  }
}

inline void apply_iso_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_ALCU_ISO")) {
    const std::string v(e);
    c.use_isotropic = !(v == "0" || v == "off" || v == "false" || v == "OFF" || v == "FALSE");
  }
}

inline void apply_step_cap(RunConfig &c) {
  if (const char *ms = std::getenv("OPENPFC_ALCU_MAX_STEPS")) {
    const int cap = std::atoi(ms);
    if (cap > 0) {
      c.n_steps = std::min(c.n_steps, cap);
    }
  }
}

inline void set_io_cadence(RunConfig &c) {
  const int snap = io_every_snapshot_for_w0(c.phys.W0);
  c.nsave = snap;
  c.vtk_every = snap;
  c.n_hist = std::max(1, snap / 10);
  c.nprint = snap;
}

inline void apply_pos_outdir_nthreads(RunConfig &c, int argc, char **argv, int start) {
  std::string pos[2];
  int npos = 0;
  for (int i = start; i < argc && npos < 2; ++i) {
    pos[npos++] = argv[i];
  }
  if (npos >= 1 && pos[0].find_first_not_of("+-0123456789") != std::string::npos) {
    c.output_dir = pos[0];
    if (npos >= 2) {
      c.num_threads = std::atoi(pos[1].c_str());
    }
  } else if (npos >= 1) {
    c.num_threads = std::atoi(pos[0].c_str());
  }
}

inline RunConfig sized(Physics phys, double t_end, double Lx, double Ly, double Lz = 0.0) {
  RunConfig c;
  c.phys = phys;
  c.t_end = t_end;
  c.lab_Lx = Lx;
  c.Nx = std::max(8, static_cast<int>(std::ceil(Lx / phys.dx)));
  c.Ny = std::max(1, static_cast<int>(std::ceil(Ly / phys.dx)));
  if (phys.n_dim >= 3 && Lz > 0.0) {
    c.Nz = std::max(8, static_cast<int>(std::ceil(Lz / phys.dx)));
  } else {
    c.Nz = 1;
  }
  c.n_steps = std::max(1, static_cast<int>(std::ceil(t_end / phys.dt)));
  return c;
}

/** LOCKED 2D gold. Physics is re-applied after env in parse_or_print_usage. */
inline RunConfig make_locked_bicrystal() {
  const double W0 = kBenchW0;
  Physics phys = make_physics(W0, 1.0, 2);
  phys.G = kBenchG;
  phys.Vp = kBenchVp;
  apply_dt_limits(phys);
  RunConfig c = sized(phys, kBenchTend, kBenchLx, kBenchLy);
  c.n_grains = 2;
  set_symmetric_misorientation(c.phys, kThetaDeg);
  c.noise_F0 = 0.0;
  c.noise_seed = kNoiseSeed;
  c.stop_on_right = false;
  c.stop_on_far_c = true;
  c.periodic_y = true;
  c.periodic_z = false;
  c.seed_depth = kDsSeedDepth;
  c.phys.x_tl = c.seed_depth;
  c.phys.r_seed = two_grain_seed_radius(c.seed_depth, c.Ny, c.phys.dx, c.phys.W0);
  c.output_dir = kBenchOutputDir;
  c.frozen_benchmark = true;
  set_io_cadence(c);
  return c;
}

inline void apply_ds_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_ALCU_NGRANS")) {
    c.n_grains = std::max(1, std::min(2, std::atoi(e)));
  }
  if (c.n_grains == 2) {
    set_symmetric_misorientation(c.phys, env_d("OPENPFC_ALCU_THETA", kThetaDeg));
  } else {
    c.phys.phi1_g1 = 0.0;
    c.phys.phi1_g2 = 0.0;
  }
  c.phys.G = env_d("OPENPFC_ALCU_G", c.phys.G);
  c.phys.Vp = env_d("OPENPFC_ALCU_VP", c.phys.Vp);
  c.phys.delta_iso = env_d("OPENPFC_ALCU_DELTA", c.phys.delta_iso);
  if (const char *e = std::getenv("OPENPFC_ALCU_NY")) {
    const int ny = std::atoi(e);
    if (ny >= 1) {
      c.Ny = ny;
    }
  }
  if (c.Ny == 1 && std::getenv("OPENPFC_ALCU_ISO") == nullptr) {
    c.use_isotropic = false;
  }
  c.seed_depth = env_d("OPENPFC_ALCU_SEED", c.seed_depth);
  c.seed_bump = env_d("OPENPFC_ALCU_BUMP", c.seed_bump);
  c.seed_bump_sigma = env_d("OPENPFC_ALCU_BUMP_SIGMA", c.seed_bump_sigma);
  if (const char *e = std::getenv("OPENPFC_ALCU_OMEGA")) {
    const double w = std::atof(e);
    if (std::isfinite(w) && w >= 0.0) {
      c.phys.omega_zhong = false;
      c.phys.omega = w;
    }
  }
  if (const char *e = std::getenv("OPENPFC_ALCU_STOP_RIGHT")) {
    const std::string v(e);
    c.stop_on_right = !(v == "0" || v == "off" || v == "false" || v == "OFF" || v == "FALSE");
  }
  if (const char *e = std::getenv("OPENPFC_ALCU_STOP_FAR_C")) {
    const std::string v(e);
    c.stop_on_far_c = !(v == "0" || v == "off" || v == "false" || v == "OFF" || v == "FALSE");
  }
  c.periodic_y = env_on("OPENPFC_ALCU_PERIODIC_Y", c.periodic_y);
  c.periodic_z = env_on("OPENPFC_ALCU_PERIODIC_Z", c.periodic_z);
}

inline void apply_window_env(RunConfig &c) {
  c.window_enable = env_on("OPENPFC_ALCU_WINDOW", c.window_enable);
  if (const char *e = std::getenv("OPENPFC_ALCU_WINDOW_NX")) {
    const int n = std::atoi(e);
    if (n >= 8) {
      c.window_nx = n;
      c.window_enable = true;
    }
  }
  c.window_margin_left = env_d("OPENPFC_ALCU_WINDOW_LEFT", c.window_margin_left);
  c.window_margin_right = env_d("OPENPFC_ALCU_WINDOW_RIGHT", c.window_margin_right);
  if (c.window_enable) {
    const int wnx = c.window_nx > 0 ? c.window_nx : kWindowNxDefault;
    c.window_nx = wnx;
    c.Nx = std::min(c.Nx, wnx);
    c.Nx = std::max(8, c.Nx);
  }
  if (const char *e = std::getenv("OPENPFC_ALCU_BLOCK_SKIP")) {
    const int bs = std::atoi(e);
    if (bs == 16 || bs == 32) {
      c.block_skip = bs;
    } else if (bs == 0) {
      c.block_skip = 0;
    }
  }
  c.block_skip_tol_phi = env_d("OPENPFC_ALCU_SKIP_TOL_PHI", c.block_skip_tol_phi);
  c.block_skip_tol_c = env_d("OPENPFC_ALCU_SKIP_TOL_C", c.block_skip_tol_c);
  if (const char *e = std::getenv("OPENPFC_ALCU_SKIP_REFRESH")) {
    const int r = std::atoi(e);
    if (r > 0) {
      c.block_skip_refresh = r;
    }
  }
}

inline void apply_scale_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_ALCU_WARMUP")) {
    const int n = std::atoi(e);
    if (n >= 0) {
      c.warmup_steps = n;
    }
  }
  if (const char *e = std::getenv("OPENPFC_ALCU_TIMED_STEPS")) {
    const int n = std::atoi(e);
    if (n > 0) {
      c.timed_steps = n;
    }
  }
}

inline void apply_noise_env(RunConfig &c) {
  if (const char *e = std::getenv("OPENPFC_ALCU_NOISE")) {
    c.noise_F0 = std::atof(e);
  }
  if (const char *e = std::getenv("OPENPFC_ALCU_NOISE_SEED")) {
    c.noise_seed = static_cast<unsigned>(std::strtoul(e, nullptr, 10));
  }
}

/** Request dt = r τ0. Allowed up to the explicit Laplacian von Neumann limit
 *  (twice the stored 0.5-safety CFL) and the full interface CFL Δx/V_p.
 *  Does not bypass a linearly unstable step. Recomputes limits after V_p is set. */
inline void apply_dt_over_tau_env(RunConfig &c) {
  auto &p = c.phys;
  apply_dt_limits(p);
  if (const char *e = std::getenv("OPENPFC_ALCU_DT_OVER_TAU")) {
    const double r = std::atof(e);
    if (r > 0.0 && std::isfinite(r)) {
      const double vn_lap = 2.0 * std::min(p.dt_cfl_phi, p.dt_cfl_c);
      const double vn_iface = (p.Vp > 0.0 && p.dx > 0.0) ? (p.dx / p.Vp) : vn_lap;
      const double vn = std::min(vn_lap, vn_iface);
      const double want = r * p.tau0;
      if (want > vn * (1.0 + 1.0e-12)) {
        std::cerr << "ALCU_DT r=" << r << " tau0 exceeds max "
                  << (vn / p.tau0) << " tau0 (Laplacian VN or Δx/V_p); leaving dt=" << p.dt
                  << "\n";
      } else {
        p.dt_tau = want;
        p.dt = want;
      }
    }
  }
  if (c.t_end > 0.0 && p.dt > 0.0) {
    c.n_steps = std::max(1, static_cast<int>(std::ceil(c.t_end / p.dt)));
  }
}

inline std::optional<RunConfig> parse_args(int argc, char **argv) {
  if (argc >= 2 && std::string(argv[1]) == "--help") {
    return std::nullopt;
  }
  if (argc >= 2 && std::string(argv[1]) == "smoke") {
    Physics phys = make_physics();
    RunConfig c = sized(phys, 0.2e-6, 80.0 * phys.W0, 48.0 * phys.W0);
    c.output_dir = "results/alloy_pf_directional_smoke";
    c.nprint = 50;
    c.nsave = 100;
    c.vtk_every = 100;
    c.n_hist = 10;
    std::string pos[2];
    int npos = 0;
    for (int i = 2; i < argc && npos < 2; ++i) {
      pos[npos++] = argv[i];
    }
    if (npos >= 1 && pos[0].find_first_not_of("+-0123456789") != std::string::npos) {
      c.output_dir = pos[0];
      if (npos >= 2) {
        c.num_threads = std::atoi(pos[1].c_str());
      }
    } else if (npos >= 1) {
      c.num_threads = std::atoi(pos[0].c_str());
    }
    return c;
  }
  if (argc >= 2 && std::string(argv[1]) == "repro") {
    const double W0 = kBenchW0;
    Physics phys = make_physics(W0, 1.0, 2);
    phys.G = kBenchG;
    phys.Vp = kBenchVp;
    apply_dt_limits(phys);
    RunConfig c = sized(phys, 1.0e-6, double(kReproNxW0) * W0, double(kReproNyW0) * W0);
    c.n_grains = 2;
    set_symmetric_misorientation(c.phys, kThetaDeg);
    c.noise_F0 = kNoiseF0;
    c.noise_seed = kNoiseSeed;
    c.stop_on_right = false;
    c.stop_on_far_c = false;
    c.periodic_y = true;
    c.periodic_z = true;
    c.seed_depth = kDsSeedDepth;
    c.phys.x_tl = c.seed_depth;
    c.phys.r_seed = two_grain_seed_radius(c.seed_depth, c.Ny, c.phys.dx, c.phys.W0);
    c.output_dir = "results/alloy_pf_directional/repro";
    c.frozen_repro = true;
    c.nprint = kReproSteps;
    c.nsave = 10000;
    c.vtk_every = 10000;
    c.n_hist = kReproSteps;
    apply_pos_outdir_nthreads(c, argc, argv, 2);
    return c;
  }
  if (argc >= 2 && (std::string(argv[1]) == "benchmark" || std::string(argv[1]) == "start")) {
    RunConfig c = make_locked_bicrystal();
    apply_pos_outdir_nthreads(c, argc, argv, 2);
    return c;
  }
  if (argc >= 2 && (std::string(argv[1]) == "ds" || std::string(argv[1]) == "bicrystal")) {
    const bool bicrystal = std::string(argv[1]) == "bicrystal";
    const double W0 = env_d("OPENPFC_ALCU_W0", kW0);
    const double dxw = env_d("OPENPFC_ALCU_DXW", kDxOverW0);
    const double Lx = env_d("OPENPFC_ALCU_LX", kDsLx);
    const double Ly = env_d("OPENPFC_ALCU_LY", kDsLy);
    const double Lz = env_d("OPENPFC_ALCU_LZ", kDsLz);
    const double t_end = env_d("OPENPFC_ALCU_TEND", kDsTend);
    int n_dim = kNDim;
    if (const char *e = std::getenv("OPENPFC_ALCU_NDIM")) {
      n_dim = std::max(1, std::min(3, std::atoi(e)));
    } else if (Lz > 0.0 || std::getenv("OPENPFC_ALCU_NZ")) {
      n_dim = 3;
    } else if (const char *e = std::getenv("OPENPFC_ALCU_NY")) {
      if (std::atoi(e) == 1) {
        n_dim = 1;
      }
    }
    Physics phys = make_physics(W0, dxw, n_dim);
    RunConfig c = sized(phys, t_end, Lx, Ly, Lz);
    if (const char *e = std::getenv("OPENPFC_ALCU_NZ")) {
      const int nz = std::atoi(e);
      if (nz >= 1) {
        c.Nz = nz;
        if (nz > 1) {
          c.phys.n_dim = 3;
          phys = make_physics(W0, dxw, 3);
          c.phys = phys;
          c.n_steps = std::max(1, static_cast<int>(std::ceil(t_end / phys.dt)));
        }
      }
    }
    c.n_grains = bicrystal ? 2 : 1;
    c.stop_on_right = true;
    c.stop_on_far_c = true;
    c.periodic_y = true;
    c.periodic_z = true;
    c.output_dir = bicrystal ? "results/alloy_pf_directional/bicrystal"
                             : "results/alloy_pf_directional/ds";
    apply_ds_env(c);
    c.phys.x_tl = c.seed_depth;
    if (c.n_grains == 2) {
      c.phys.r_seed = two_grain_seed_radius(c.seed_depth, c.Ny, c.phys.dx, c.phys.W0);
    }
    set_io_cadence(c);
    std::string pos[4];
    int npos = 0;
    for (int i = 2; i < argc && npos < 4; ++i) {
      const std::string a(argv[i]);
      if (io_flag_takes_value(a)) {
        if (i + 1 < argc) {
          ++i;
        }
        continue;
      }
      pos[npos++] = a;
    }
    if (npos >= 1 && pos[0].find_first_not_of("+-0123456789") != std::string::npos) {
      c.output_dir = pos[0];
      if (npos >= 2) {
        c.num_threads = std::atoi(pos[1].c_str());
      }
    } else if (npos >= 1) {
      c.num_threads = std::atoi(pos[0].c_str());
    }
    return c;
  }

  RunConfig c = make_locked_bicrystal();
  apply_pos_outdir_nthreads(c, argc, argv, 1);
  return c;
}

inline std::optional<RunConfig> parse_or_print_usage(int argc, char **argv) {
  auto cfg = parse_args(argc, argv);
  if (!cfg) {
    print_usage(std::cerr, argv[0]);
    return cfg;
  }
  apply_iso_env(*cfg);
  apply_vtk_env(*cfg);
  apply_io_flags(*cfg, argc, argv);
  apply_noise_env(*cfg);
  apply_ds_env(*cfg);
  if (cfg->n_grains == 2 && cfg->phys.x_tl == cfg->seed_depth) {
    cfg->phys.r_seed =
        two_grain_seed_radius(cfg->seed_depth, cfg->Ny, cfg->phys.dx, cfg->phys.W0);
  }
  apply_dt_over_tau_env(*cfg);
  apply_window_env(*cfg);
  apply_scale_env(*cfg);
  apply_step_cap(*cfg);
  cfg->store_aux = env_on("OPENPFC_ALCU_STORE_AUX", cfg->store_aux);
  cfg->store_eu = env_on("OPENPFC_ALCU_STORE_EU", cfg->store_eu);
  if (cfg->store_aux) {
    cfg->store_eu = true;
  }
  if (cfg->frozen_repro) {
    apply_dt_limits(cfg->phys);
    const double want = kBenchDtOverTau * cfg->phys.tau0;
    cfg->phys.dt_tau = want;
    cfg->phys.dt = want;
    cfg->phys.G = kBenchG;
    cfg->phys.Vp = kBenchVp;
    cfg->Nx = kReproNxW0;
    cfg->Ny = kReproNyW0;
    cfg->Nz = 1;
    cfg->phys.n_dim = 2;
    cfg->n_steps = kReproSteps;
    cfg->n_grains = 2;
    cfg->noise_F0 = kNoiseF0;
    cfg->noise_seed = kNoiseSeed;
    set_symmetric_misorientation(cfg->phys, kThetaDeg);
    cfg->stop_on_right = false;
    cfg->stop_on_far_c = false;
    cfg->window_enable = false;
    cfg->periodic_y = true;
    cfg->periodic_z = true;
    cfg->seed_depth = kDsSeedDepth;
    cfg->phys.x_tl = cfg->seed_depth;
    cfg->phys.r_seed =
        two_grain_seed_radius(cfg->seed_depth, cfg->Ny, cfg->phys.dx, cfg->phys.W0);
  }
  if (cfg->frozen_benchmark) {
    apply_dt_limits(cfg->phys);
    const double want = kBenchDtOverTau * cfg->phys.tau0;
    cfg->phys.dt_tau = want;
    cfg->phys.dt = want;
    cfg->phys.G = kBenchG;
    cfg->phys.Vp = kBenchVp;
    cfg->Nx = std::max(8, static_cast<int>(std::ceil(kBenchLx / cfg->phys.dx)));
    cfg->Ny = std::max(1, static_cast<int>(std::ceil(kBenchLy / cfg->phys.dx)));
    cfg->Nz = 1;
    cfg->phys.n_dim = 2;
    cfg->n_steps = std::max(1, static_cast<int>(std::ceil(kBenchTend / cfg->phys.dt)));
    cfg->n_grains = 2;
    cfg->noise_F0 = 0.0;
    cfg->noise_seed = kNoiseSeed;
    set_symmetric_misorientation(cfg->phys, kThetaDeg);
    cfg->stop_on_right = false;
    cfg->stop_on_far_c = true;
    cfg->window_enable = false;
    cfg->periodic_y = true;
    cfg->periodic_z = false;
    cfg->seed_depth = kDsSeedDepth;
    cfg->phys.x_tl = cfg->seed_depth;
    cfg->phys.r_seed =
        two_grain_seed_radius(cfg->seed_depth, cfg->Ny, cfg->phys.dx, cfg->phys.W0);
    apply_step_cap(*cfg);
  }
  return cfg;
}

} // namespace alloy_pf_directional
