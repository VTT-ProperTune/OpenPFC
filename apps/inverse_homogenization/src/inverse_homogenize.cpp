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
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <mpi.h>

#include <inverse_homogenization/auxetic_geometry.hpp>
#include <inverse_homogenization/driver_support.hpp>
#include <inverse_homogenization/homogenization.hpp>
#include <inverse_homogenization/inverse_checkpoint.hpp>
#include <inverse_homogenization/inverse_convergence.hpp>
#include <inverse_homogenization/manufacturability.hpp>
#include <inverse_homogenization/phase_field_inverse.hpp>
#include <inverse_homogenization/spinodal_generator.hpp>
#include <inverse_homogenization/target_io.hpp>
#include <inverse_homogenization/yang_reentrant.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

namespace {
namespace driver = pfc::apps::inverse::driver;

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
  int steps{5000};
  int continuation_steps{300};
  int conv_window{20};
  int verify_steps{100};
  double tol_design{1e-4};
  double tol_objective{1e-6};
  double tol_tensor{1e-4};
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
  std::string dump_dir{};
  int dump_every{0};
  std::string load_h{};
  std::string load_bin{};
  std::string C_target_file{};
  std::string checkpoint_dir{};
  std::string restart_dir{};
  int stop_after{0};
  double w12{1.0};
  int n_el_iter{200};
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
     << "  --target isotropic|auxetic|orthotropic|file\n"
     << "  --E-target --nu-target          (isotropic / auxetic)\n"
     << "  --C11 --C22 --C12 --C66         (orthotropic in-plane block)\n"
     << "  --C-target-file=PATH            6x6 Voigt text (target=file)\n"
     << "  --volume --lambda-volume --lambda-reg --epsilon\n"
     << "  --dt --max-steps|--steps --continuation-steps --conv-window\n"
     << "  --verify-convergence-steps --tol-design --tol-objective --tol-tensor\n"
     << "  --init uniform|noise --init-volume --csv=PATH\n"
     << "  --normalize=0|1 --max-delta   (default 1 and 0.05; do not amplify\n"
     << "                                a small elastic residual)\n"
     << "  --project-volume=0|1          shift h to hold --volume after each step\n"
     << "  --simp=P --simp-end=P         SIMP continuation (linear in step)\n"
     << "  --lambda-reg-end              perimeter continuation\n"
     << "  --init-amp                    noise amplitude (default 0.25)\n"
     << "  --init rotating-squares|reentrant|spinodal|yang-a3|noise|uniform\n"
     << "  --load-bin=PATH               Fortran float64 brick (overrides init)\n"
     << "  --ch-steps --ch-kappa --ch-dt --ch-aniso-y   (Stage 6 CH family)\n"
     << "  --init-half --init-angle      rotating-square size/rotation\n"
     << "  --init-thickness --init-inset re-entrant wall geometry\n"
     << "  --no-tensor=1                 W=0 (binarization-only step)\n"
     << "  --dump-h=PATH --load-h=PATH   write/read h (rank-0 text)\n"
     << "  --dump-dir=DIR --dump-every=N gathered Fortran h bricks\n"
     << "  --checkpoint-dir --restart    continue the same frozen problem\n"
     << "  --stop-after=N                checkpoint running state and exit\n"
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
    if (key == "nx")
      ok = parse_int(val, cfg.nx) && cfg.nx > 0;
    else if (key == "ny")
      ok = parse_int(val, cfg.ny) && cfg.ny > 0;
    else if (key == "nz")
      ok = parse_int(val, cfg.nz) && cfg.nz > 0;
    else if (key == "dx")
      ok = parse_double(val, cfg.dx) && cfg.dx > 0.0;
    else if (key == "E-solid")
      ok = parse_double(val, cfg.E_solid);
    else if (key == "nu-solid")
      ok = parse_double(val, cfg.nu_solid);
    else if (key == "E-void")
      ok = parse_double(val, cfg.E_void);
    else if (key == "nu-void")
      ok = parse_double(val, cfg.nu_void);
    else if (key == "target")
      cfg.target = std::string(val);
    else if (key == "E-target")
      ok = parse_double(val, cfg.E_target);
    else if (key == "nu-target")
      ok = parse_double(val, cfg.nu_target);
    else if (key == "C11")
      ok = parse_double(val, cfg.C11);
    else if (key == "C22")
      ok = parse_double(val, cfg.C22);
    else if (key == "C12")
      ok = parse_double(val, cfg.C12);
    else if (key == "C66")
      ok = parse_double(val, cfg.C66);
    else if (key == "volume")
      ok = parse_double(val, cfg.volume);
    else if (key == "lambda-volume")
      ok = parse_double(val, cfg.lambda_volume);
    else if (key == "lambda-reg")
      ok = parse_double(val, cfg.lambda_reg);
    else if (key == "epsilon")
      ok = parse_double(val, cfg.epsilon) && cfg.epsilon > 0.0;
    else if (key == "dt")
      ok = parse_double(val, cfg.dt) && cfg.dt > 0.0;
    else if (key == "steps" || key == "max-steps")
      ok = parse_int(val, cfg.steps) && cfg.steps >= 0;
    else if (key == "continuation-steps")
      ok = parse_int(val, cfg.continuation_steps) && cfg.continuation_steps >= 0;
    else if (key == "conv-window")
      ok = parse_int(val, cfg.conv_window) && cfg.conv_window > 0;
    else if (key == "verify-convergence-steps")
      ok = parse_int(val, cfg.verify_steps) && cfg.verify_steps >= 0;
    else if (key == "tol-design")
      ok = parse_double(val, cfg.tol_design) && cfg.tol_design > 0.0;
    else if (key == "tol-objective")
      ok = parse_double(val, cfg.tol_objective) && cfg.tol_objective > 0.0;
    else if (key == "tol-tensor")
      ok = parse_double(val, cfg.tol_tensor) && cfg.tol_tensor > 0.0;
    else if (key == "init")
      cfg.init = std::string(val);
    else if (key == "init-volume")
      ok = parse_double(val, cfg.init_volume);
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
    } else if (key == "dump-dir") {
      cfg.dump_dir = std::string(val);
    } else if (key == "dump-every") {
      ok = parse_int(val, cfg.dump_every) && cfg.dump_every >= 0;
    } else if (key == "load-h") {
      cfg.load_h = std::string(val);
    } else if (key == "load-bin") {
      cfg.load_bin = std::string(val);
    } else if (key == "C-target-file") {
      cfg.C_target_file = std::string(val);
    } else if (key == "checkpoint-dir") {
      cfg.checkpoint_dir = std::string(val);
    } else if (key == "restart") {
      cfg.restart_dir = std::string(val);
    } else if (key == "stop-after") {
      ok = parse_int(val, cfg.stop_after) && cfg.stop_after >= 0;
    } else if (key == "W-12") {
      ok = parse_double(val, cfg.w12) && cfg.w12 >= 0.0;
    } else if (key == "n-el-iter") {
      ok = parse_int(val, cfg.n_el_iter) && cfg.n_el_iter > 0;
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
      cfg.target != "orthotropic" && cfg.target != "file")
    return false;
  if (cfg.target == "file" && cfg.C_target_file.empty()) return false;
  if (cfg.init != "uniform" && cfg.init != "noise" &&
      cfg.init != "rotating-squares" && cfg.init != "reentrant" &&
      cfg.init != "spinodal" && cfg.init != "yang-a3")
    return false;
  return true;
}

std::vector<double> gather_dense(const pfc::data::Field<double> &h, int nx, int ny,
                                 int nz) {
  auto local = pfc::apps::inverse::dense_from_field(h, nx, ny, nz);
  std::vector<double> g(local.size(), 0.0);
  MPI_Allreduce(local.data(), g.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);
  return g;
}

bool write_raw_bin(const std::string &path, const std::vector<double> &a) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(a.data()),
          static_cast<std::streamsize>(a.size() * sizeof(double)));
  f.close();
  return static_cast<bool>(f);
}

void write_xdmf_brick(const std::string &path, const std::string &bin, int nx,
                      int ny, int nz, double dx, const char *name) {
  std::ofstream f(path);
  f << "<?xml version=\"1.0\"?>\n<Xdmf Version=\"2.0\"><Domain>"
    << "<Grid Name=\"g\" GridType=\"Uniform\">\n"
    << "<Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"" << nz << ' ' << ny
    << ' ' << nx << "\"/>\n"
    << "<Geometry Type=\"ORIGIN_DXDYDZ\">\n"
    << "<DataItem Format=\"XML\" Dimensions=\"3\">0 0 0</DataItem>\n"
    << "<DataItem Format=\"XML\" Dimensions=\"3\">" << dx << ' ' << dx << ' ' << dx
    << "</DataItem>\n</Geometry>\n"
    << "<Attribute Name=\"" << name << "\" Center=\"Node\">\n"
    << "<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\"8\" "
    << "Endian=\"Little\" Dimensions=\"" << nz << ' ' << ny << ' ' << nx << "\">"
    << bin << "</DataItem>\n</Attribute>\n</Grid></Domain></Xdmf>\n";
}

void print_C(const char *label, const pfc::apps::Voigt6 &C) {
  std::cout << label << '\n';
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (j) std::cout << ' ';
      std::cout << std::setprecision(8) << C(i, j);
    }
    std::cout << '\n';
  }
}

void print_stiffness_report(const char *label, const pfc::apps::Voigt6 &C,
                            const pfc::apps::Voigt6 &Ct,
                            const pfc::apps::HomogenizationResult &hr) {
  using pfc::apps::diagnose_stiffness;
  const auto d = diagnose_stiffness(C, &Ct);
  print_C(label, d.C);
  std::cout << std::setprecision(8) << "rel_frobenius " << d.rel_frobenius << " spd "
            << (d.spd ? 1 : 0) << " invertible " << (d.invertible ? 1 : 0)
            << " min_eig " << d.min_eig << '\n';
  std::cout << "nu_shortcut " << d.nu_shortcut << " nu_xy " << d.nu_xy << " nu_xz "
            << d.nu_xz << " nu_yx " << d.nu_yx << " nu_yz " << d.nu_yz << " nu_zx "
            << d.nu_zx << " nu_zy " << d.nu_zy << '\n';
  std::cout << "spread_C11 " << d.spread_C11 << " spread_C12 " << d.spread_C12
            << " spread_C44 " << d.spread_C44 << '\n';
  int it_max = 0;
  double res_max = 0.0;
  bool conv = true;
  for (const auto &r : hr.reports) {
    it_max = std::max(it_max, r.iterations);
    res_max = std::max(res_max, r.residual);
    conv = conv && r.converged;
  }
  std::cout << "elasticity_converged " << (conv ? 1 : 0) << " el_iters_max "
            << it_max << " el_residual_max " << res_max << " volume "
            << hr.volume_fraction << '\n';
}

pfc::apps::Voigt6 make_target(const Config &cfg) {
  using pfc::apps::Voigt6;
  using pfc::apps::voigt_from_stiffness;
  using pfc::solvers::Stiffness;
  if (cfg.target == "file") {
    Voigt6 C;
    if (!pfc::apps::inverse::load_voigt6_file(cfg.C_target_file, C))
      throw std::runtime_error("C-target-file unreadable");
    return C;
  }
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
  const double nu =
      (cfg.target == "auxetic") ? -std::abs(cfg.nu_target) : cfg.nu_target;
  return voigt_from_stiffness(Stiffness::isotropic(cfg.E_target, nu));
}

} // namespace

int initialize_design(const Config &cfg, pfc::Field<double> &h,
                      const pfc::Domain &domain,
                      pfc::sim::stacks::SpectralCPUStack &stack, int rank, int &rc) {
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
          const double n4 =
              std::sin(2.0 * twopi * g[0] / nx) * std::sin(twopi * g[1] / ny);
          hv += cfg.init_amp * (0.6 * n1 * n2 + 0.3 * n3 + 0.4 * n4);
        }
        h(i, j, k) = std::min(1.0, std::max(0.0, hv));
      }
  if (cfg.init == "rotating-squares") {
    pfc::apps::inverse::fill_rotating_squares(h, cfg.nx, cfg.ny, cfg.init_half,
                                              cfg.init_angle);
  } else if (cfg.init == "reentrant") {
    pfc::apps::inverse::fill_reentrant_honeycomb(h, cfg.nx, cfg.ny,
                                                 cfg.init_thickness, cfg.init_inset);
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
  } else if (cfg.init == "yang-a3") {
    pfc::apps::inverse::fill_yang_a3(h, cfg.nx, cfg.ny, cfg.nz);
  }
  if (!cfg.load_bin.empty()) {
    if (!pfc::apps::inverse::load_fortran_bin(cfg.load_bin, cfg.nx, cfg.ny, cfg.nz,
                                              h)) {
      if (rank == 0)
        std::cerr << "load-bin: unreadable or wrong size " << cfg.load_bin << '\n';

      return 2;
    }
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
  return 0;
}

int run(int argc, char **argv, int rank, int nproc) {
  Config cfg;
  if (!parse_args(argc, argv, cfg)) {
    if (rank == 0)
      usage(std::cerr, argc >= 1 ? argv[0] : "openpfc_inverse_homogenize");
    return 2;
  }
  if (cfg.checkpoint_dir.empty() && !cfg.restart_dir.empty())
    cfg.checkpoint_dir = cfg.restart_dir;
  if (cfg.checkpoint_dir.empty() && !cfg.dump_dir.empty())
    cfg.checkpoint_dir = cfg.dump_dir + "/checkpoint";

  int rc = 0;
  {
    const pfc::Domain domain =
        pfc::domain::create(pfc::GridSize({cfg.nx, cfg.ny, cfg.nz}),
                            pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                            pfc::GridSpacing({cfg.dx, cfg.dx, cfg.dx}));
    pfc::sim::stacks::SpectralCPUStack stack(domain, rank, nproc, MPI_COMM_WORLD);
    auto h =
        pfc::data::field_from_inbox<double>(domain, stack.fft().get_inbox_bounds());
    if (const int status = initialize_design(cfg, h, domain, stack, rank, rc);
        status != 0)
      return status;

    const auto p = driver::configure_elasticity(cfg);

    pfc::apps::inverse::InverseSpec spec;
    try {
      spec = driver::configure_problem(cfg, make_target(cfg));
    } catch (const std::exception &e) {
      if (rank == 0) std::cerr << e.what() << '\n';
      return 2;
    }

    pfc::apps::inverse::PhaseFieldInverse inv(domain, stack.fft(), p);
    const auto init_h = inv.homogenizer().compute(h);
    if (rank == 0) {
      std::cout << "target " << cfg.target << " grid " << cfg.nx << 'x' << cfg.ny
                << 'x' << cfg.nz << " steps " << cfg.steps << " seed " << cfg.seed
                << '\n';
      std::cout << "backend cpu ranks " << nproc << " grid " << cfg.nx << 'x'
                << cfg.ny << 'x' << cfg.nz << " loads 6\n";
      print_stiffness_report("C_H_initial", init_h.stiffness, spec.C_target, init_h);
    }
    if (rank == 0 && !cfg.dump_dir.empty())
      std::filesystem::create_directories(cfg.dump_dir);
    if (rank == 0 && !cfg.checkpoint_dir.empty())
      std::filesystem::create_directories(cfg.checkpoint_dir);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!cfg.dump_dir.empty() && cfg.restart_dir.empty()) {
      const auto dense = gather_dense(h, cfg.nx, cfg.ny, cfg.nz);
      if (rank == 0) {
        write_raw_bin(cfg.dump_dir + "/h_init.bin", dense);
        write_xdmf_brick(cfg.dump_dir + "/h_init.xdmf", "h_init.bin", cfg.nx, cfg.ny,
                         cfg.nz, cfg.dx, "h");
      }
    }
    auto csv = driver::open_csv(cfg, rank);
    if (rank == 0) {
      std::cout << "step J J_tensor volume grey C11 C12 design_rms dJ_rel dC_rel "
                   "morph_frac step_rms termination\n";
    }
    pfc::apps::inverse::InverseStepReport last{};
    const double simp0 = cfg.simp;
    const double simp1 = (cfg.simp_end > 0.0) ? cfg.simp_end : cfg.simp;
    const double lr0 = cfg.lambda_reg;
    const double lr1 =
        (cfg.lambda_reg_end >= 0.0) ? cfg.lambda_reg_end : cfg.lambda_reg;
    auto tracker = driver::configure_tracker(cfg);
    auto h_prev = h;
    pfc::apps::Voigt6 C_prev{};
    double J_prev = 0.0;
    bool have_prev = false;
    int start_s = 0;
    auto write_ckpt = [&](int next_step,
                          pfc::apps::inverse::TerminationReason why =
                              pfc::apps::inverse::TerminationReason::Running) {
      if (cfg.checkpoint_dir.empty()) return true;
      const auto ck = driver::capture_checkpoint(cfg, spec, tracker, next_step,
                                                 have_prev, J_prev, C_prev, why);
      const auto dense_h = gather_dense(h, cfg.nx, cfg.ny, cfg.nz);
      const auto dense_p = gather_dense(h_prev, cfg.nx, cfg.ny, cfg.nz);
      return driver::publish_checkpoint(
          cfg.checkpoint_dir, ck, rank,
          [&](const std::string &staging) {
            if (rank != 0) return true;
            const bool ready = write_raw_bin(staging + "/h.bin", dense_h);
            return write_raw_bin(staging + "/h_prev.bin", dense_p) && ready;
          },
          [](const std::string &) { return true; });
    };
    if (!cfg.restart_dir.empty()) {
      const auto restart = driver::inspect_restart(
          cfg, spec, rank,
          [](const std::string &, const pfc::apps::inverse::InverseCheckpoint &) {
            return true;
          });
      const auto &ck = restart.checkpoint;
      const auto &bundle = restart.path;
      const bool ok = restart.ready;
      if (!ok) {
        if (rank == 0)
          std::cerr << "restart: unreadable or mismatched " << cfg.restart_dir
                    << '\n';
        rc = 2;
      } else if (!pfc::apps::inverse::load_fortran_bin(bundle + "/h.bin", cfg.nx,
                                                       cfg.ny, cfg.nz, h) ||
                 !pfc::apps::inverse::load_fortran_bin(
                     bundle + "/h_prev.bin", cfg.nx, cfg.ny, cfg.nz, h_prev)) {
        if (rank == 0) std::cerr << "restart: missing h.bin / h_prev.bin\n";
        rc = 2;
      } else {
        h.note_host_write();
        h_prev.note_host_write();
        pfc::apps::inverse::apply_tracker(ck, tracker);
        tracker.cfg.max_steps = cfg.steps;
        start_s = ck.next_step;
        have_prev = ck.have_prev != 0;
        J_prev = ck.J_prev;
        pfc::apps::inverse::fill_voigt6(C_prev, ck.C_prev);
        if (rank == 0)
          std::cout << "restart next_step " << start_s << " quiet "
                    << tracker.quiet_count << " candidate "
                    << (tracker.candidate ? 1 : 0) << '\n';
      }
    }
    auto reason = pfc::apps::inverse::TerminationReason::Running;
    const auto gs = h.global_size();
    const double n_global = static_cast<double>(gs[0]) * gs[1] * gs[2];
    int n_done = start_s;
    const auto evolve = [&] {
      for (int s = start_s; rc == 0 && s < cfg.steps; ++s) {
        const double t =
            pfc::apps::inverse::continuation_fraction(s, cfg.continuation_steps);
        spec.simp_p = simp0 + t * (simp1 - simp0);
        spec.lambda_reg = lr0 + t * (lr1 - lr0);
        double local_dh2 = 0.0, local_morph = 0.0;
        const double *hp = h.data();
        const double *hpp = h_prev.data();
        for (std::size_t i = 0; i < h.size(); ++i) {
          const double d = hp[i] - hpp[i];
          local_dh2 += d * d;
          if ((hp[i] > 0.5) != (hpp[i] > 0.5)) local_morph += 1.0;
        }
        double glo_dh2 = 0.0, glo_morph = 0.0;
        MPI_Allreduce(&local_dh2, &glo_dh2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_morph, &glo_morph, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        const double design_rms = std::sqrt(glo_dh2 / n_global);
        const double morph_frac = glo_morph / n_global;
        pfc::apps::inverse::copy_design_buffer(h.data(), h_prev.data(), h.size());
        h_prev.note_host_write();
        const auto t0 = std::chrono::steady_clock::now();
        last = inv.step(h, spec);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const auto metrics = driver::diagnose_step(
            last, C_prev, J_prev, have_prev, design_rms, morph_frac, tracker.cfg);
        reason = tracker.after_step(s, last.elasticity_converged, metrics);
        const int frozen =
            pfc::apps::inverse::params_frozen(s, cfg.continuation_steps) ? 1 : 0;
        if (rank == 0) {
          std::cout << std::setprecision(8) << s << ' ' << last.J << ' '
                    << last.J_tensor << ' ' << last.volume_accepted << ' '
                    << last.grey_fraction << ' ' << last.C11 << ' ' << last.C12
                    << ' ' << design_rms << ' ' << metrics.dJ_rel << ' '
                    << metrics.dC_rel << ' ' << morph_frac << ' ' << last.step_rms
                    << ' ' << pfc::apps::inverse::termination_name(reason) << '\n';
          if (csv.is_open()) {
            driver::write_step_csv(csv, s, last, spec, metrics, tracker, reason,
                                   design_rms, morph_frac, ms, frozen);
          }
        }
        n_done = s + 1;
        C_prev = last.C;
        J_prev = last.J;
        have_prev = true;
        if (reason != pfc::apps::inverse::TerminationReason::Running) {
          pfc::apps::inverse::copy_design_buffer(h_prev.data(), h.data(), h.size());
          h.note_host_write();
          if (!write_ckpt(n_done, reason)) {
            rc = 2;
            break;
          }
          if (reason == pfc::apps::inverse::TerminationReason::ElasticityFailure) {
            if (rank == 0)
              std::cerr << "elasticity did not converge at step " << s << '\n';
            rc = 1;
          }
          break;
        }
        if (!cfg.dump_dir.empty() && cfg.dump_every > 0 &&
            (s % cfg.dump_every == 0 || s + 1 == cfg.steps)) {
          const auto dense = gather_dense(h, cfg.nx, cfg.ny, cfg.nz);
          if (rank == 0) {
            char name[64];
            std::snprintf(name, sizeof(name), "/h_%04d.bin", s);
            write_raw_bin(cfg.dump_dir + name, dense);
          }
        }
        if (!write_ckpt(n_done)) {
          rc = 2;
          break;
        }
        if (cfg.stop_after > 0 && n_done >= cfg.stop_after) break;
      }
    };
    evolve();
    const auto finalize_output = [&] {
      if (rc != 2 && !(cfg.stop_after > 0 &&
                       reason == pfc::apps::inverse::TerminationReason::Running)) {
        const int certified_step = std::max(0, n_done - 1);
        if (!cfg.dump_dir.empty()) {
          const auto dense_final = gather_dense(h, cfg.nx, cfg.ny, cfg.nz);
          if (rank == 0) {
            char name[64];
            std::snprintf(name, sizeof(name), "/h_%04d.bin", certified_step);
            write_raw_bin(cfg.dump_dir + name, dense_final);
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
        if (rank == 0) {
          const auto &C = final.stiffness;
          const double den = C(0, 0) + C(0, 1);
          const double nu = (std::abs(den) > 1.0e-30) ? C(0, 1) / den : 0.0;
          const double Jt = pfc::apps::tensor_mismatch(C, spec.C_target, spec.W);
          std::cout << std::setprecision(16) << "FINAL_RECOMPUTE J_tensor " << Jt
                    << " C11 " << C(0, 0) << " C12 " << C(0, 1) << " nu_eff " << nu
                    << " C_fro " << C.symmetrized().frobenius_norm()
                    << " elasticity " << (final.all_converged() ? 1 : 0) << '\n';
          if (csv.is_open()) {
            csv << "# CERTIFIED_STEP " << std::max(0, n_done - 1) << " termination "
                << pfc::apps::inverse::termination_name(reason) << '\n';
            csv << "# FINAL_RECOMPUTE unpenalized C_H of certified accepted h; "
                   "not an iterate J_tensor="
                << Jt << " C11=" << C(0, 0) << " C12=" << C(0, 1) << " nu_eff=" << nu
                << " C_fro=" << C.symmetrized().frobenius_norm()
                << " elasticity=" << (final.all_converged() ? 1 : 0) << '\n';
            csv.flush();
          }
        }
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
        const auto dense = gather_dense(h, cfg.nx, cfg.ny, cfg.nz);
        const auto dense_bin = gather_dense(hbin, cfg.nx, cfg.ny, cfg.nz);
        if (rank == 0) {
          std::cout << std::setprecision(16) << "INVERSE_CHECKSUM " << last.J
                    << '\n';
          std::cout << "termination " << pfc::apps::inverse::termination_name(reason)
                    << " steps_done " << n_done << " certified_step "
                    << std::max(0, n_done - 1) << '\n';
          print_C("C_target", spec.C_target);
          print_stiffness_report("C_H_final", final.stiffness, spec.C_target, final);
          std::cout << "grey " << last.grey_fraction << '\n';
          print_stiffness_report("C_H_thresholded (h>0.5)", bin.stiffness,
                                 spec.C_target, bin);
          const auto man = pfc::apps::inverse::measure_manufacturability(
              dense, cfg.nx, cfg.ny, cfg.nz);
          const auto manb = pfc::apps::inverse::measure_manufacturability(
              dense_bin, cfg.nx, cfg.ny, cfg.nz);
          std::cout << std::setprecision(6)
                    << "manufacturability_physical solid_comp "
                    << man.n_solid_components << " void_comp "
                    << man.n_void_components << " island_solid "
                    << man.island_solid_frac << " island_void "
                    << man.island_void_frac << " grey " << man.grey_fraction << '\n';
          std::cout << "percolate_physical_solid x=" << man.percolate_solid_x
                    << " y=" << man.percolate_solid_y
                    << " z=" << man.percolate_solid_z << '\n';
          std::cout << "manufacturability_thresholded solid_comp "
                    << manb.n_solid_components << " void_comp "
                    << manb.n_void_components << " island_solid "
                    << manb.island_solid_frac << " island_void "
                    << manb.island_void_frac << '\n';
          std::cout << "percolate_thresholded_solid x=" << manb.percolate_solid_x
                    << " y=" << manb.percolate_solid_y
                    << " z=" << manb.percolate_solid_z
                    << " percolate_void x=" << manb.percolate_void_x
                    << " y=" << manb.percolate_void_y
                    << " z=" << manb.percolate_void_z << '\n';
          std::cout << "opening_loss_r1 " << manb.opening_loss_r1
                    << " opening_loss_r2 " << manb.opening_loss_r2 << '\n';
          std::cout << "ranks " << nproc << " grid " << cfg.nx << 'x' << cfg.ny
                    << 'x' << cfg.nz << " steps " << cfg.steps
                    << " loads_per_step 6\n";
          driver::report_process_memory(std::cout);
          if (!cfg.dump_dir.empty()) {
            write_raw_bin(cfg.dump_dir + "/h_final.bin", dense);
            write_raw_bin(cfg.dump_dir + "/h_thresh.bin", dense_bin);
            write_xdmf_brick(cfg.dump_dir + "/h_final.xdmf", "h_final.bin", cfg.nx,
                             cfg.ny, cfg.nz, cfg.dx, "h");
            write_xdmf_brick(cfg.dump_dir + "/h_thresh.xdmf", "h_thresh.bin", cfg.nx,
                             cfg.ny, cfg.nz, cfg.dx, "h");
          }
        }
      }
    };
    finalize_output();
  } // SpectralCPUStack / HeFFTe before MPI_Finalize
  return rc;
}

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        return run(app_argc, app_argv, rank, nproc);
      });
}
