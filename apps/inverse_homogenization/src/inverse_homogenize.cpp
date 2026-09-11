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
#include <inverse_homogenization/phase_field_inverse.hpp>
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
     << "  --normalize=0|1 --max-delta   (default 1 and 0.05; RMS-normalise g)\n";
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
    else if (key == "steps") ok = parse_int(val, cfg.steps) && cfg.steps > 0;
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
    } else {
      return false;
    }
    if (!ok) return false;
  }
  if (cfg.target != "isotropic" && cfg.target != "auxetic" &&
      cfg.target != "orthotropic")
    return false;
  if (cfg.init != "uniform" && cfg.init != "noise") return false;
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
          const double s = static_cast<double>(cfg.seed);
          hv += 0.08 * std::sin(2.0 * 3.141592653589793 *
                                ((g[0] + s) + 2.0 * g[1] + 3.0 * g[2]) /
                                std::max(cfg.nx, 1));
        }
        h(i, j, k) = std::min(1.0, std::max(0.0, hv));
      }
  h.note_host_write();

  pfc::apps::MicroelasticityParams p;
  p.c_solid = pfc::apps::Stiffness::isotropic(cfg.E_solid, cfg.nu_solid);
  p.c_liquid = pfc::apps::Stiffness::isotropic(cfg.E_void, cfg.nu_void);
  p.tol_el = 1.0e-8;
  p.n_el_iter = 80;
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

  pfc::apps::inverse::PhaseFieldInverse inv(domain, stack.fft(), p);
  std::ofstream csv;
  if (rank == 0) {
    std::cout << "target " << cfg.target << " grid " << cfg.nx << 'x' << cfg.ny
              << 'x' << cfg.nz << " steps " << cfg.steps << '\n';
    std::cout << "step J J_tensor J_volume J_reg volume grad_rms step_rms grey perimeter\n";
    if (!cfg.csv.empty()) {
      csv.open(cfg.csv);
      csv << "step,J,J_tensor,J_volume,J_reg,volume,grad_rms,step_rms,grey,perimeter\n";
    }
  }
  pfc::apps::inverse::InverseStepReport last{};
  for (int s = 0; s < cfg.steps; ++s) {
    last = inv.step(h, spec);
    if (rank == 0) {
      std::cout << std::setprecision(8) << s << ' ' << last.J << ' '
                << last.J_tensor << ' ' << last.J_volume << ' ' << last.J_reg
                << ' ' << last.volume_fraction << ' ' << last.grad_rms << ' '
                << last.step_rms << ' ' << last.grey_fraction << ' '
                << last.perimeter << '\n';
      if (csv.is_open()) {
        csv << s << ',' << last.J << ',' << last.J_tensor << ',' << last.J_volume
            << ',' << last.J_reg << ',' << last.volume_fraction << ','
            << last.grad_rms << ',' << last.step_rms << ',' << last.grey_fraction
            << ',' << last.perimeter << '\n';
      }
    }
    if (!last.elasticity_converged) {
      if (rank == 0) std::cerr << "elasticity did not converge at step " << s << '\n';
      MPI_Finalize();
      return 1;
    }
  }
  if (rank == 0) {
    std::cout << std::setprecision(16) << "INVERSE_CHECKSUM " << last.J << '\n';
    const auto &C = inv.homogenizer().last().stiffness;
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
  }
  MPI_Finalize();
  return 0;
}
