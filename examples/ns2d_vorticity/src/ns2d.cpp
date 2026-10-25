// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ns2d.cpp
 * @brief CLI driver for the 2-D vorticity–streamfunction prototype (#21).
 *
 * Verification (`--case taylor_green --verify`) and the Kelvin–Helmholtz
 * showcase (`--case shear`) are separate. A roll-up movie is not a pass.
 */

#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#include <mpi.h>

#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

#include <ns2d/cases.hpp>
#include <ns2d/vorticity_stream.hpp>

namespace {

struct Cli {
  ns2d::Case cse{ns2d::Case::taylor_green};
  int n{64};
  int steps{100};
  int dump_every{0};
  double dt{0.01};
  double nu{0.1};
  double rho{30.0};
  double eps{0.05};
  double cfl{-1.0};
  bool verify{false};
  std::string outdir;
};

void print_usage(std::ostream &os, const char *exe) {
  os << "Usage: " << exe
     << " [--case taylor_green|shear|two_mode] [--N N] [--steps N]\n"
     << "       [--dt DT] [--cfl CFL] [--nu NU] [--rho RHO] [--eps EPS]\n"
     << "       [--dump EVERY] [--outdir DIR] [--verify]\n"
     << "\n"
     << "Periodic 2-D vorticity–streamfunction Navier–Stokes (CPU, #21).\n"
     << "  taylor_green  quantitative decaying vortex on [0,2pi]^2\n"
     << "  shear         Minion-Brown unit-square double shear (rho=30)\n"
     << "  two_mode      nonlinear IC for timestep-refinement tests\n"
     << "  --cfl         sets dt = CFL * dx (overrides --dt); dx = L/N\n"
     << "  --verify      Taylor-Green Linf check; ignore --case\n";
}

std::optional<std::string_view> take_value(int &i, int argc, char **argv) {
  if (i + 1 >= argc) return std::nullopt;
  return std::string_view(argv[++i]);
}

std::optional<Cli> parse_cli(int argc, char **argv) {
  Cli c;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a(argv[i]);
    auto need = [&](const char *flag) -> std::optional<std::string_view> {
      auto v = take_value(i, argc, argv);
      if (!v) std::cerr << "missing value for " << flag << "\n";
      return v;
    };
    if (a == "-h" || a == "--help") {
      print_usage(std::cout, argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (a == "--verify") {
      c.verify = true;
    } else if (a == "--case") {
      auto v = need("--case");
      if (!v) return std::nullopt;
      try {
        c.cse = ns2d::parse_case(*v);
      } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return std::nullopt;
      }
    } else if (a == "--N" || a == "-N") {
      auto v = need("--N");
      if (!v) return std::nullopt;
      c.n = std::atoi(std::string(*v).c_str());
    } else if (a == "--steps") {
      auto v = need("--steps");
      if (!v) return std::nullopt;
      c.steps = std::atoi(std::string(*v).c_str());
    } else if (a == "--dt") {
      auto v = need("--dt");
      if (!v) return std::nullopt;
      c.dt = std::atof(std::string(*v).c_str());
    } else if (a == "--cfl") {
      auto v = need("--cfl");
      if (!v) return std::nullopt;
      c.cfl = std::atof(std::string(*v).c_str());
    } else if (a == "--nu") {
      auto v = need("--nu");
      if (!v) return std::nullopt;
      c.nu = std::atof(std::string(*v).c_str());
    } else if (a == "--rho") {
      auto v = need("--rho");
      if (!v) return std::nullopt;
      c.rho = std::atof(std::string(*v).c_str());
    } else if (a == "--eps") {
      auto v = need("--eps");
      if (!v) return std::nullopt;
      c.eps = std::atof(std::string(*v).c_str());
    } else if (a == "--dump") {
      auto v = need("--dump");
      if (!v) return std::nullopt;
      c.dump_every = std::atoi(std::string(*v).c_str());
    } else if (a == "--outdir") {
      auto v = need("--outdir");
      if (!v) return std::nullopt;
      c.outdir = std::string(*v);
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      return std::nullopt;
    }
  }
  if (c.n < 4 || c.steps < 0 || c.dt <= 0.0 || c.nu < 0.0) return std::nullopt;
  if (c.verify) c.cse = ns2d::Case::taylor_green;
  return c;
}

void write_csv_header(std::ostream &os) {
  os << "step,time,ke,enstrophy,max_abs_omega,max_speed,div_linf,div_l2,cfl,"
        "mean_omega,wall_step_s,ke_exact,ens_exact,omega_linf\n";
}

void dump_fields(pfc::VTKWriter &vtk, pfc::BinaryWriter *bin, int increment,
                 pfc::data::Field<double> &omega) {
  vtk.write(increment, omega.view());
  if (bin) bin->write(increment, omega.view());
}

int run(const Cli &cli, int rank, int nproc) {
  const bool unit = (cli.cse == ns2d::Case::shear);
  const double length = unit ? 1.0 : 2.0 * pfc::pi;
  double dt = cli.dt;
  const double dx = length / static_cast<double>(cli.n);
  if (cli.cfl > 0.0) dt = cli.cfl * dx;

  if (cli.cse == ns2d::Case::shear && rank == 0) {
    const double cells = ns2d::shear_cells_per_thickness(cli.n, cli.rho);
    const double cells23 = ns2d::shear_effective_cells_per_thickness(cli.n, cli.rho);
    std::cout << "Minion-Brown unit-square shear rho=" << cli.rho
              << " thickness=1/rho=" << ns2d::shear_thickness(cli.rho)
              << " dx=" << dx << " cells/thickness=" << cells
              << " 2/3-effective=" << cells23 << "\n";
    std::cout << "integrator=IFRK4  dt=" << dt << " nu=" << cli.nu << "\n";
  }

  pfc::sim::stacks::SpectralCPUStack stack(ns2d::make_slab(cli.n, length), rank,
                                           nproc, MPI_COMM_WORLD);
  ns2d::VorticityStreamCPU solver(stack, ns2d::Params{cli.nu, dt});

  if (cli.cse == ns2d::Case::taylor_green) {
    solver.initialize_omega([&](double x, double y, double) {
      return ns2d::taylor_green_omega(x, y, cli.nu, 0.0);
    });
  } else if (cli.cse == ns2d::Case::shear) {
    solver.initialize_omega([&](double x, double y, double) {
      return ns2d::double_shear_omega(x, y, cli.rho, cli.eps);
    });
  } else {
    solver.initialize_omega(
        [&](double x, double y, double) { return ns2d::two_mode_omega(x, y); });
  }

  std::optional<pfc::VTKWriter> vtk;
  std::optional<pfc::BinaryWriter> bin;
  std::ofstream csv;
  if (!cli.outdir.empty()) {
    if (rank == 0) std::filesystem::create_directories(cli.outdir);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto inbox = stack.fft().get_inbox_bounds();
    vtk.emplace(cli.outdir + "/omega_%04d.vti");
    vtk->set_domain(stack.u().global_size(), inbox.size, inbox.low);
    vtk->set_geometry(stack.u().origin(), stack.u().spacing());
    vtk->set_field_name("omega");
    bin.emplace(cli.outdir + "/omega_%04d.bin");
    bin->set_domain(stack.u().global_size(), inbox.size, inbox.low);
    if (rank == 0) {
      csv.open(cli.outdir + "/diagnostics.csv");
      write_csv_header(csv);
      std::ofstream meta(cli.outdir + "/run.json");
      meta << "{\n"
           << "  \"issue\": 21,\n"
           << "  \"case\": \"" << ns2d::case_name(cli.cse) << "\",\n"
           << "  \"box\": \"" << (unit ? "unit" : "twopi") << "\",\n"
           << "  \"N\": " << cli.n << ",\n"
           << "  \"steps\": " << cli.steps << ",\n"
           << "  \"dt\": " << dt << ",\n"
           << "  \"nu\": " << cli.nu << ",\n"
           << "  \"rho\": " << cli.rho << ",\n"
           << "  \"eps\": " << cli.eps << ",\n"
           << "  \"nproc\": " << nproc << ",\n"
           << "  \"integrator\": \"IFRK4 (exp viscous + RK4 Jacobian)\",\n"
           << "  \"dealias\": \"Orszag 2/3 on state and N_hat\",\n"
           << "  \"zero_mode\": \"psi_hat(0)=0; N_hat(0)=0; mean omega conserved\"\n"
           << "}\n";
    }
  }

  auto emit = [&](int step, const ns2d::Diagnostics &d, double wall_step,
                  double omega_linf) {
    const double ke_ex = (cli.cse == ns2d::Case::taylor_green)
                             ? ns2d::taylor_green_ke(cli.nu, d.time)
                             : -1.0;
    const double ens_ex = (cli.cse == ns2d::Case::taylor_green)
                              ? ns2d::taylor_green_enstrophy(cli.nu, d.time)
                              : -1.0;
    if (rank == 0) {
      std::cout << std::scientific << std::setprecision(6) << "step=" << step
                << " t=" << d.time << " ke=" << d.ke << " Z=" << d.enstrophy
                << " max|w|=" << d.max_abs_omega << " div_inf=" << d.div_linf
                << " cfl=" << d.cfl << " mean_w=" << d.mean_omega;
      if (cli.cse == ns2d::Case::taylor_green) {
        std::cout << " ke_err=" << std::abs(d.ke - ke_ex)
                  << " w_linf=" << omega_linf;
      }
      std::cout << " wall_step_s=" << wall_step << "\n";
      if (csv) {
        csv << step << "," << d.time << "," << d.ke << "," << d.enstrophy << ","
            << d.max_abs_omega << "," << d.max_speed << "," << d.div_linf << ","
            << d.div_l2 << "," << d.cfl << "," << d.mean_omega << "," << wall_step
            << "," << ke_ex << "," << ens_ex << "," << omega_linf << "\n";
      }
    }
  };

  auto omega_err = [&]() {
    if (cli.cse != ns2d::Case::taylor_green) return -1.0;
    const double t = solver.time();
    const double nu = cli.nu;
    return solver.linf_omega_error(MPI_COMM_WORLD, [nu, t](double x, double y) {
      return ns2d::taylor_green_omega(x, y, nu, t);
    });
  };

  auto d0 = solver.diagnostics(MPI_COMM_WORLD);
  emit(0, d0, 0.0, omega_err());
  if (vtk) dump_fields(*vtk, bin ? &*bin : nullptr, 0, solver.omega());

  double wall_acc = 0.0;
  int timed = 0;
  for (int step = 1; step <= cli.steps; ++step) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    solver.step();
    MPI_Barrier(MPI_COMM_WORLD);
    const double wall = MPI_Wtime() - t0;
    wall_acc += wall;
    ++timed;
    const bool dump =
        (cli.dump_every > 0 && step % cli.dump_every == 0) || step == cli.steps;
    if (dump) {
      auto d = solver.diagnostics(MPI_COMM_WORLD);
      emit(step, d, wall, omega_err());
      if (vtk) dump_fields(*vtk, bin ? &*bin : nullptr, step, solver.omega());
      if (!std::isfinite(d.ke) || !std::isfinite(d.enstrophy) ||
          !std::isfinite(d.max_abs_omega)) {
        if (rank == 0) {
          std::cerr << "ns2d: non-finite diagnostic at step " << step
                    << " (under-resolved cascade or CFL blow-up)\n";
        }
        return EXIT_FAILURE;
      }
      if (d.cfl > 2.0) {
        if (rank == 0) {
          std::cerr << "ns2d: CFL=" << d.cfl << " > 2 at step " << step
                    << "; refusing to continue a fixed-dt run\n";
        }
        return EXIT_FAILURE;
      }
    }
  }

  if (cli.verify) {
    const double linf = omega_err();
    constexpr double tol = 1.0e-9;
    const bool pass = linf < tol;
    if (rank == 0) {
      std::cout << (pass ? "VERIFY PASS" : "VERIFY FAIL")
                << " taylor_green Linf=" << std::scientific << linf << " tol=" << tol
                << "\n";
    }
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  if (rank == 0 && timed > 0) {
    std::cout << "median-like mean wall_step_s=" << (wall_acc / timed) << "\n";
  }
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cli = parse_cli(app_argc, app_argv);
        if (!cli) {
          if (rank == 0) print_usage(std::cerr, app_argv[0]);
          return EXIT_FAILURE;
        }
        return run(*cli, rank, nproc);
      });
}
