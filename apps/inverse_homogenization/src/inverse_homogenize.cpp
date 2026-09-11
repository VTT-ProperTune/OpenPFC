// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file inverse_homogenize.cpp
 * @brief Phase-field inverse homogenization driver (issue #161 Stages 2–3).
 *
 * Allen–Cahn descent on
 * \(J=\tfrac12\lVert W\odot(C_H-C_\ast)\rVert_F^2\) plus volume and
 * perimeter. Not a black-box optimizer and not Cahn–Hilliard.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <inverse_homogenization/auxetic_geometry.hpp>
#include <inverse_homogenization/phase_field_inverse.hpp>
#include <inverse_homogenization/spinodal_generator.hpp>
#include <inverse_homogenization/manufacturability.hpp>
#include <openpfc_apps/homogenization.hpp>

namespace {

struct Config {
  int nx{16}, ny{16}, nz{16};
  double dx{1.0};
  double E_solid{2.0}, nu_solid{0.25};
  double E_void{0.5}, nu_void{0.25};
  std::string target{"isotropic"};
  double E_target{0.9}, nu_target{0.25};
  double C11{1.2}, C22{0.7}, C12{0.25}, C66{0.3};
  double volume{0.5};
  double lambda_volume{1.0};
  double lambda_reg{0.05};
  double epsilon{2.0};
  double dt{0.1};
  int steps{10};
  std::string init{"noise"};
  double init_volume{0.55};
  unsigned seed{1};
  std::string csv{};
  int normalize{1};
  double max_delta{0.05};
  int project_volume{0};
  double simp{1.0};
  double simp_end{-1.0};
  double lambda_reg_end{-1.0};
  double init_amp{0.25};
  double init_half{0.200};
  double init_angle{0.45};
  double init_thickness{0.035};
  double init_inset{0.30};
  int no_tensor{0};
  std::string dump_h{};
  std::string load_h{};
  double w12{1.0};
  int ch_steps{200};
  double ch_kappa{1.0};
  double ch_dt{0.2};
  double ch_aniso_y{1.0};
};

void usage(std::ostream &os, const char *exe) {
  os << "Usage: " << exe << " [--key=value]...\n"
     << "  Phase-field inverse homogenization (issue #161 Stages 2-3).\n"
     << "  Allen-Cahn descent on ||W odot (C_H - C_target)||_F^2.\n\n"
     << "  --nx --ny --nz --dx\n"
     << "  --E-solid --nu-solid --E-void --nu-void\n"
     << "  --target isotropic|auxetic|orthotropic\n"
     << "  --E-target --nu-target          (isotropic / auxetic)\n"
     << "  --C11 --C22 --C12 --C66         (orthotropic in-plane block)\n"
     << "  --volume --lambda-volume --lambda-reg --epsilon\n"
     << "  --dt --steps --init uniform|noise --init-volume --csv=PATH\n"
     << "  --normalize=0|1 --max-delta   (default 1 and 0.05; RMS-normalise g)\n"
     << "  --project-volume=0|1          shift h to hold --volume after each step\n"
     << "  --simp=P --simp-end=P         SIMP continuation (linear in step)\n"
     << "  --lambda-reg-end              perimeter continuation\n"
     << "  --init-amp                    noise amplitude (default 0.25)\n"
     << "  --init rotating-squares|reentrant|spinodal|noise|uniform\n"
     << "  --ch-steps --ch-kappa --ch-dt --ch-aniso-y   (Stage 6 CH family)\n"
     << "  --init-half --init-angle      rotating-square size/rotation\n"
     << "  --init-thickness --init-inset re-entrant wall geometry\n"
     << "  --no-tensor=1                 W=0 (binarization-only step)\n"
     << "  --dump-h=PATH --load-h=PATH   write/read h (single rank)\n"
     << "  --W-12                        extra weight on C12 (auxetic default 4)\n";
}

bool parse_double(std::string_view v, double &out) {
  try {
    std::size_t n = 0;
    out = std::stod(std::string(v), &n);
    return n == v.size();
  } catch (...) {
    return false;
  }
}
bool parse_int(std::string_view v, int &out) {
  try {
    std::size_t n = 0;
    out = std::stoi(std::string(v), &n);
    return n == v.size();
  } catch (...) {
    return false;
  }
}

bool parse_args(int argc, char **argv, Config &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view tok(argv[i]);
    if (tok == "--help" || tok == "-h") return false;
    const auto eq = tok.find('=');
    if (!tok.starts_with("--") || eq == std::string_view::npos) return false;
    const auto key = tok.substr(2, eq - 2);
    const auto val = tok.substr(eq + 1);
    bool ok = true;
    if (key == "nx") ok = parse_int(val, cfg.nx) && cfg.nx > 0;
    else if (key == "ny") ok = parse_int(val, cfg.ny) && cfg.ny > 0;
    else if (key == "nz") ok = parse_int(val, cfg.nz) && cfg.nz > 0;
    else if (key == "dx") ok = parse_double(val, cfg.dx) && cfg.dx > 0.0;
    else if (key == "E-solid") ok = parse_double(val, cfg.E_solid);
    else if (key == "nu-solid") ok = parse_double(val, cfg.nu_solid);
    else if (key == "E-void") ok = parse_double(val, cfg.E_void);
    else if (key == "nu-void") ok = parse_double(val, cfg.nu_void);
    else if (key == "target") cfg.target = std::string(val);
    else if (key == "E-target") ok = parse_double(val, cfg.E_target);
    else if (key == "nu-target") ok = parse_double(val, cfg.nu_target);
    else if (key == "C11") ok = parse_double(val, cfg.C11);
    else if (key == "C22") ok = parse_double(val, cfg.C22);
    else if (key == "C12") ok = parse_double(val, cfg.C12);
    else if (key == "C66") ok = parse_double(val, cfg.C66);
    else if (key == "volume") ok = parse_double(val, cfg.volume);
    else if (key == "lambda-volume") ok = parse_double(val, cfg.lambda_volume);
    else if (key == "lambda-reg") ok = parse_double(val, cfg.lambda_reg);
    else if (key == "epsilon") ok = parse_double(val, cfg.epsilon) && cfg.epsilon > 0.0;
    else if (key == "dt") ok = parse_double(val, cfg.dt) && cfg.dt > 0.0;
    else if (key == "steps") ok = parse_int(val, cfg.steps) && cfg.steps >= 0;
    else if (key == "init") cfg.init = std::string(val);
    else if (key == "init-volume") ok = parse_double(val, cfg.init_volume);
    else if (key == "seed") {
      int s = 1;
      ok = parse_int(val, s);
      cfg.seed = static_cast<unsigned>(s);
    } else if (key == "csv") {
      cfg.csv = std::string(val);
    } else if (key == "normalize") {
      ok = parse_int(val, cfg.normalize);
    } else if (key == "max-delta") {
      ok = parse_double(val, cfg.max_delta) && cfg.max_delta >= 0.0;
    } else if (key == "project-volume") {
      ok = parse_int(val, cfg.project_volume);
    } else if (key == "simp") {
      ok = parse_double(val, cfg.simp) && cfg.simp >= 1.0;
    } else if (key == "simp-end") {
      ok = parse_double(val, cfg.simp_end) && cfg.simp_end >= 1.0;
    } else if (key == "lambda-reg-end") {
      ok = parse_double(val, cfg.lambda_reg_end) && cfg.lambda_reg_end >= 0.0;
    } else if (key == "init-amp") {
      ok = parse_double(val, cfg.init_amp) && cfg.init_amp >= 0.0;
    } else if (key == "init-half") {
      ok = parse_double(val, cfg.init_half) && cfg.init_half > 0.0;
    } else if (key == "init-angle") {
      ok = parse_double(val, cfg.init_angle);
    } else if (key == "init-thickness") {
      ok = parse_double(val, cfg.init_thickness) && cfg.init_thickness > 0.0;
    } else if (key == "init-inset") {
      ok = parse_double(val, cfg.init_inset) && cfg.init_inset > 0.0;
    } else if (key == "no-tensor") {
      ok = parse_int(val, cfg.no_tensor);
    } else if (key == "dump-h") {
      cfg.dump_h = std::string(val);
    } else if (key == "load-h") {
      cfg.load_h = std::string(val);
    } else if (key == "W-12") {
      ok = parse_double(val, cfg.w12) && cfg.w12 >= 0.0;
    } else if (key == "ch-steps") {
      ok = parse_int(val, cfg.ch_steps) && cfg.ch_steps >= 0;
    } else if (key == "ch-kappa") {
      ok = parse_double(val, cfg.ch_kappa) && cfg.ch_kappa > 0.0;
    } else if (key == "ch-dt") {
      ok = parse_double(val, cfg.ch_dt) && cfg.ch_dt > 0.0;
    } else if (key == "ch-aniso-y") {
      ok = parse_double(val, cfg.ch_aniso_y) && cfg.ch_aniso_y > 0.0;
    } else {
      return false;
    }
    if (!ok) return false;
  }
  if (cfg.target != "isotropic" && cfg.target != "auxetic" &&
      cfg.target != "orthotropic")
    return false;
  if (cfg.init != "uniform" && cfg.init != "noise" &&
      cfg.init != "rotating-squares" && cfg.init != "reentrant" &&
      cfg.init != "spinodal")
    return false;
  return true;
}

pfc::apps::Voigt6 make_target(const Config &cfg) {
  using pfc::apps::Stiffness;
  using pfc::apps::voigt_from_stiffness;
  using pfc::apps::Voigt6;
  if (cfg.target == "orthotropic") {
    Voigt6 C;
    C(0, 0) = cfg.C11;
    C(1, 1) = cfg.C22;
    C(2, 2) = cfg.C22;
    C(0, 1) = C(1, 0) = C(0, 2) = C(2, 0) = cfg.C12;
    C(1, 2) = C(2, 1) = cfg.C12;
    C(3, 3) = C(4, 4) = C(5, 5) = cfg.C66;
    return C;
  }
  const double nu = (cfg.target == "auxetic") ? -std::abs(cfg.nu_target)
                                              : cfg.nu_target;
  return voigt_from_stiffness(Stiffness::isotropic(cfg.E_target, nu));
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  Config cfg;
  if (!parse_args(argc, argv, cfg)) {
    if (rank == 0) usage(std::cerr, argc >= 1 ? argv[0] : "openpfc_inverse_homogenize");
    MPI_Finalize();
    return 2;
  }

  int rc = 0;
  {
  const pfc::Domain domain = pfc::domain::create(
      pfc::GridSize({cfg.nx, cfg.ny, cfg.nz}),
      pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({cfg.dx, cfg.dx, cfg.dx}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, rank, nproc, MPI_COMM_WORLD);
  auto h = pfc::data::field_from_inbox<double>(domain,
                                               stack.fft().get_inbox_bounds());
  const auto n = h.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        double hv = cfg.init_volume;
        if (cfg.init == "noise") {
          const auto g = h.global(i, j, k);
          const double twopi = 2.0 * 3.141592653589793;
          const double nx = std::max(cfg.nx, 1);
          const double ny = std::max(cfg.ny, 1);
          const double nz = std::max(cfg.nz, 1);
          const double s = static_cast<double>(cfg.seed);
          const double n1 = std::sin(twopi * (g[0] + s) / nx);
          const double n2 = std::sin(twopi * (2.0 * g[1] + s) / ny);
          const double n3 = std::sin(twopi * (g[2] + 2.0 * s) / nz);
          const double n4 = std::sin(2.0 * twopi * g[0] / nx) *
                            std::sin(twopi * g[1] / ny);
          hv += cfg.init_amp * (0.6 * n1 * n2 + 0.3 * n3 + 0.4 * n4);
        }
        h(i, j, k) = std::min(1.0, std::max(0.0, hv));
      }
  if (cfg.init == "rotating-squares") {
    pfc::apps::inverse::fill_rotating_squares(h, cfg.nx, cfg.ny, cfg.init_half,
                                              cfg.init_angle);
  } else if (cfg.init == "reentrant") {
    pfc::apps::inverse::fill_reentrant_honeycomb(
        h, cfg.nx, cfg.ny, cfg.init_thickness, cfg.init_inset);
  } else if (cfg.init == "spinodal") {
    pfc::apps::inverse::SpinodalSpec ch;
    ch.c0 = cfg.init_volume;
    ch.kappa = cfg.ch_kappa;
    ch.dt = cfg.ch_dt;
    ch.steps = cfg.ch_steps;
    ch.noise = cfg.init_amp;
    ch.seed = cfg.seed;
    ch.ay = cfg.ch_aniso_y;
    pfc::apps::inverse::seed_spinodal_noise(h, cfg.nx, cfg.ny, cfg.nz, ch);
    pfc::apps::inverse::generate_spinodal(domain, stack.fft(), h, ch);
  }
  if (!cfg.load_h.empty()) {
    std::ifstream in(cfg.load_h);
    int nx = 0, ny = 0, nz = 0;
    in >> nx >> ny >> nz;
    if (!in || nx != cfg.nx || ny != cfg.ny || nz != cfg.nz) {
      if (rank == 0)
        std::cerr << "load-h: grid mismatch or unreadable " << cfg.load_h << '\n';
      rc = 2;
    }
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          double v = 0.0;
          in >> v;
          h(i, j, k) = std::min(1.0, std::max(0.0, v));
        }
  }
  h.note_host_write();

  pfc::apps::MicroelasticityParams p;
  p.c_solid = pfc::apps::Stiffness::isotropic(cfg.E_solid, cfg.nu_solid);
  p.c_liquid = pfc::apps::Stiffness::isotropic(cfg.E_void, cfg.nu_void);
  p.tol_el = 1.0e-8;
  p.n_el_iter = 200;
  p.warm_start = false;
  p.comm = MPI_COMM_WORLD;

  pfc::apps::inverse::InverseSpec spec;
  spec.C_target = make_target(cfg);
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
    spec.W = pfc::apps::Voigt6{};
  } else if (cfg.w12 != 1.0 || cfg.target == "auxetic") {
    const double w = (cfg.target == "auxetic" && cfg.w12 == 1.0) ? 4.0 : cfg.w12;
    spec.W(0, 1) = spec.W(1, 0) = w;
    spec.W(0, 2) = spec.W(2, 0) = w;
    spec.W(1, 2) = spec.W(2, 1) = w;
  }

  pfc::apps::inverse::PhaseFieldInverse inv(domain, stack.fft(), p);
  std::ofstream csv;
  if (rank == 0) {
    std::cout << "target " << cfg.target << " grid " << cfg.nx << 'x' << cfg.ny
              << 'x' << cfg.nz << " steps " << cfg.steps << '\n';
    std::cout << "backend cpu ranks " << nproc << " grid " << cfg.nx << 'x'
              << cfg.ny << 'x' << cfg.nz << " loads 6\n";
    std::cout << "step J J_tensor J_volume J_reg volume grad_rms step_rms grey perimeter C11 C12 ms\n";
    if (!cfg.csv.empty()) {
      csv.open(cfg.csv);
      csv << "step,J,J_tensor,J_volume,J_reg,volume,grad_rms,step_rms,grey,perimeter,C11,C12,ms\n";
    }
  }
  pfc::apps::inverse::InverseStepReport last{};
  const double simp0 = cfg.simp;
  const double simp1 = (cfg.simp_end > 0.0) ? cfg.simp_end : cfg.simp;
  const double lr0 = cfg.lambda_reg;
  const double lr1 = (cfg.lambda_reg_end >= 0.0) ? cfg.lambda_reg_end : cfg.lambda_reg;
  for (int s = 0; s < cfg.steps; ++s) {
    const double t =
        (cfg.steps > 1) ? static_cast<double>(s) / (cfg.steps - 1) : 1.0;
    spec.simp_p = simp0 + t * (simp1 - simp0);
    spec.lambda_reg = lr0 + t * (lr1 - lr0);
    const auto t0 = std::chrono::steady_clock::now();
    last = inv.step(h, spec);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (rank == 0) {
      std::cout << std::setprecision(8) << s << ' ' << last.J << ' '
                << last.J_tensor << ' ' << last.J_volume << ' ' << last.J_reg
                << ' ' << last.volume_fraction << ' ' << last.grad_rms << ' '
                << last.step_rms << ' ' << last.grey_fraction << ' '
                << last.perimeter << ' ' << last.C11 << ' ' << last.C12 << ' '
                << std::setprecision(3) << ms << '\n';
      if (csv.is_open()) {
        csv << s << ',' << last.J << ',' << last.J_tensor << ',' << last.J_volume
            << ',' << last.J_reg << ',' << last.volume_fraction << ','
            << last.grad_rms << ',' << last.step_rms << ',' << last.grey_fraction
            << ',' << last.perimeter << ',' << last.C11 << ',' << last.C12 << ','
            << ms << '\n';
      }
    }
    if (!last.elasticity_converged) {
      if (rank == 0) std::cerr << "elasticity did not converge at step " << s << '\n';
      rc = 1;
      break;
    }
  }
  if (rank == 0 && !cfg.dump_h.empty()) {
    std::ofstream hf(cfg.dump_h);
    hf << cfg.nx << ' ' << cfg.ny << ' ' << cfg.nz << '\n';
    const auto ln = h.local_size();
    for (int k = 0; k < ln[2]; ++k)
      for (int j = 0; j < ln[1]; ++j)
        for (int i = 0; i < ln[0]; ++i)
          hf << std::setprecision(8) << h(i, j, k) << '\n';
  }
  // Physical C_H of the final h (linear two-phase interpolation), even if
  // SIMP or W=0 was used during the loop.
  const auto final = inv.homogenizer().compute(h);
  auto hbin = h;
  {
    const auto ln2 = h.local_size();
    for (int k = 0; k < ln2[2]; ++k)
      for (int j = 0; j < ln2[1]; ++j)
        for (int i = 0; i < ln2[0]; ++i)
          hbin(i, j, k) = (h(i, j, k) > 0.5) ? 1.0 : 0.0;
    hbin.note_host_write();
  }
  const auto bin = inv.homogenizer().compute(hbin);
  if (rank == 0) {
    std::cout << std::setprecision(16) << "INVERSE_CHECKSUM " << last.J << '\n';
    const auto &C = final.stiffness;
    const auto &Ct = spec.C_target;
    std::cout << "C_target\n";
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (j) std::cout << ' ';
        std::cout << std::setprecision(8) << Ct(i, j);
      }
      std::cout << '\n';
    }
    std::cout << "C_H\n";
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (j) std::cout << ' ';
        std::cout << std::setprecision(8) << C(i, j);
      }
      std::cout << '\n';
    }
    const double rel =
        (C - Ct).frobenius_norm() / std::max(Ct.frobenius_norm(), 1.0e-30);
    std::cout << std::setprecision(8) << "rel_frobenius " << rel << '\n';
    const double den = C(0, 0) + C(0, 1);
    const double nu = (std::abs(den) > 1.0e-30) ? C(0, 1) / den : 0.0;
    std::cout << "C11 " << C(0, 0) << " C12 " << C(0, 1) << " nu_eff " << nu
              << " grey " << last.grey_fraction << '\n';
    const auto &Cb = bin.stiffness;
    const double denb = Cb(0, 0) + Cb(0, 1);
    const double nub = (std::abs(denb) > 1.0e-30) ? Cb(0, 1) / denb : 0.0;
    std::cout << "C_H_thresholded (h>0.5)\n";
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (j) std::cout << ' ';
        std::cout << std::setprecision(8) << Cb(i, j);
      }
      std::cout << '\n';
    }
    std::cout << "C11_bin " << Cb(0, 0) << " C12_bin " << Cb(0, 1)
              << " nu_bin " << nub << '\n';
    std::cout << "ranks " << nproc << " grid " << cfg.nx << 'x' << cfg.ny << 'x'
              << cfg.nz << " steps " << cfg.steps << " loads_per_step 6\n";
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
      if (line.rfind("VmHWM:", 0) == 0 || line.rfind("VmRSS:", 0) == 0)
        std::cout << line << '\n';
    }
    if (nproc == 1) {
      const auto man = pfc::apps::inverse::measure_manufacturability(
          h, cfg.nx, cfg.ny, cfg.nz);
      const auto manb = pfc::apps::inverse::measure_manufacturability(
          hbin, cfg.nx, cfg.ny, cfg.nz);
      std::cout << std::setprecision(6)
                << "manufacturability solid_comp " << manb.n_solid_components
                << " void_comp " << manb.n_void_components
                << " island_solid " << manb.island_solid_frac
                << " island_void " << manb.island_void_frac << '\n';
      std::cout << "percolate_solid x=" << manb.percolate_solid_x
                << " y=" << manb.percolate_solid_y << " z=" << manb.percolate_solid_z
                << " percolate_void x=" << manb.percolate_void_x
                << " y=" << manb.percolate_void_y << " z=" << manb.percolate_void_z
                << '\n';
      std::cout << "opening_loss_r1 " << manb.opening_loss_r1 << " opening_loss_r2 "
                << manb.opening_loss_r2 << " grey " << man.grey_fraction << '\n';
    }
  }
  } // SpectralCPUStack / HeFFTe before MPI_Finalize
  MPI_Finalize();
  return rc;
}
