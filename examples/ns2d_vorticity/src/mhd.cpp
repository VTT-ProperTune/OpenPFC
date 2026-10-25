// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file mhd.cpp
 * @brief CLI for 2-D incompressible visco-resistive MHD (#23, #113).
 *
 * Verification cases and the Orszag–Tang showcase are separate. A current
 * sheet in VTK is not a reconnection claim.
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

#include <ns2d/mhd.hpp>
#include <ns2d/mhd_cases.hpp>
#include <ns2d/spectral.hpp>

namespace {

struct Cli {
  ns2d::MHDCase cse{ns2d::MHDCase::orszag_tang};
  int n{64};
  int steps{100};
  int dump_every{0};
  int diag_every{0};
  double dt{0.01};
  double nu{0.02};
  double eta{0.02};
  double cfl{-1.0};
  // Orszag-Tang drive amplitude: scales the initial vorticity while the
  // flux function is held fixed, weakening the externally imposed flow
  // relative to the magnetic field. 1.0 is the admitted case ().
  double drive{1.0};
  bool verify{false};
  std::string outdir;
};

void print_usage(std::ostream &os, const char *exe) {
  os << "Usage: " << exe
     << " [--case orszag_tang|hydro_control|force_free|alfven|"
        "island_coalescence]\n"
     << "       [--N N] [--steps N] [--dt DT] [--cfl CFL]\n"
     << "       [--nu NU] [--eta ETA] [--drive LAMBDA]\n"
     << "       [--dump EVERY] [--diag EVERY]\n"
     << "       [--outdir DIR] [--verify]\n"
     << "\n"
     << "2-D incompressible visco-resistive MHD (CPU, #23/#113).\n"
     << "  orszag_tang         incompressible OT vortex on [0,2pi]^2\n"
     << "  hydro_control       same velocity, a=0\n"
     << "  force_free          magnetic eigenmode, u=0\n"
     << "  alfven              u=B two-mode aligned state\n"
     << "  island_coalescence  Ng flux a=0.4 sin x sin y plus frozen\n"
     << "                      streamfunction phi=0.01 (cos x - cos y)\n"
     << "  --cfl          nominal selector dt = CFL * dx (overrides --dt).\n"
     << "                 This is NOT the measured MHD CFL.\n"
     << "  measured CFL   fail-closed: dt * max(|zx|/dx + |zy|/dy) over zpm.\n"
     << "                 Also logs component-max and Euclidean |zpm| CFL.\n"
     << "                 The driver aborts if the sum bound exceeds 2.\n"
     << "  --dump EVERY   VTK/binary field dumps\n"
     << "  --diag EVERY   CSV/stdout diagnostics (default: same as --dump)\n"
     << "  --verify       force-free Linf check\n";
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
      c.cse = ns2d::MHDCase::force_free;
    } else if (a == "--case") {
      auto v = need("--case");
      if (!v) return std::nullopt;
      try {
        c.cse = ns2d::parse_mhd_case(*v);
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
    } else if (a == "--eta") {
      auto v = need("--eta");
      if (!v) return std::nullopt;
      c.eta = std::atof(std::string(*v).c_str());
    } else if (a == "--drive") {
      auto v = need("--drive");
      if (!v) return std::nullopt;
      c.drive = std::atof(std::string(*v).c_str());
    } else if (a == "--dump") {
      auto v = need("--dump");
      if (!v) return std::nullopt;
      c.dump_every = std::atoi(std::string(*v).c_str());
    } else if (a == "--diag") {
      auto v = need("--diag");
      if (!v) return std::nullopt;
      c.diag_every = std::atoi(std::string(*v).c_str());
    } else if (a == "--outdir") {
      auto v = need("--outdir");
      if (!v) return std::nullopt;
      c.outdir = std::string(*v);
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      return std::nullopt;
    }
  }
  if (c.n < 4 || c.steps < 0 || c.dt <= 0.0 || c.nu < 0.0 || c.eta < 0.0) {
    return std::nullopt;
  }
  return c;
}

void write_csv_header(std::ostream &os) {
  os << "step,time,ke,me,energy,cross_helicity,a2,enstrophy,mean_sq_j,"
        "max_abs_omega,max_abs_j,max_speed,max_b,div_u,div_b,dissipation,"
        "budget_residual,cfl_nominal,cfl_ub,cfl_elsasser,cfl_elsasser_mag,"
        "cfl_elsasser_sum,max_z_inf,max_z_mag,max_z_l1,mean_omega,mean_a,"
        "wall_step_s\n";
}

struct Writers {
  std::optional<pfc::VTKWriter> omega, a, j;
  std::optional<pfc::BinaryWriter> omega_bin, a_bin, j_bin;
};

void dump_all(Writers &w, int inc, ns2d::MHDSolver &s) {
  if (w.omega) w.omega->write(inc, s.omega().view());
  if (w.a) w.a->write(inc, s.a().view());
  if (w.j) w.j->write(inc, s.j().view());
  if (w.omega_bin) w.omega_bin->write(inc, s.omega().view());
  if (w.a_bin) w.a_bin->write(inc, s.a().view());
  if (w.j_bin) w.j_bin->write(inc, s.j().view());
}

int run(const Cli &cli, int rank, int nproc) {
  const double length = 2.0 * pfc::pi;
  double dt = cli.dt;
  const double dx = length / static_cast<double>(cli.n);
  if (cli.cfl > 0.0) dt = cli.cfl * dx;

  if (rank == 0) {
    std::cout << "mhd2d case=" << ns2d::mhd_case_name(cli.cse) << " N=" << cli.n
              << " nu=" << cli.nu << " eta=" << cli.eta
              << " Pm=" << (cli.eta > 0.0 ? cli.nu / cli.eta : 0.0) << " dt=" << dt
              << " cfl_nominal=" << (dt / dx) << " integrator=IFRK4\n";
    std::cout << "signs: omega=-lap phi, j=-lap a, Lorentz=+B.grad j "
                 "(not Strauss RMHD)\n";
    std::cout << "CFL: nominal=dt/dx; fail-closed Elsasser sum="
                 "dt*max(|zx|/dx+|zy|/dy); abort if that > 2\n";
  }

  pfc::sim::stacks::SpectralCPUStack stack(ns2d::make_twopi_slab(cli.n), rank, nproc,
                                           MPI_COMM_WORLD);
  ns2d::MHDSolver solver(stack, ns2d::MHDParams{cli.nu, cli.eta, dt, +1.0});

  if (cli.cse == ns2d::MHDCase::orszag_tang) {
    const double drive = cli.drive;
    solver.initialize(
        [drive](double x, double y, double) { return drive * ns2d::ot_omega(x, y); },
        [](double x, double y, double) { return ns2d::ot_a(x, y); });
  } else if (cli.cse == ns2d::MHDCase::hydro_control) {
    solver.initialize(
        [](double x, double y, double) { return ns2d::ot_omega(x, y); },
        [](double, double, double) { return 0.0; });
  } else if (cli.cse == ns2d::MHDCase::force_free) {
    solver.initialize(
        [](double, double, double) { return 0.0; },
        [](double x, double y, double) { return ns2d::force_free_a(x, y); });
  } else if (cli.cse == ns2d::MHDCase::island_coalescence) {
    solver.initialize(
        [](double x, double y, double) { return ns2d::coalescence_omega(x, y); },
        [](double x, double y, double) { return ns2d::coalescence_a(x, y); });
  } else {
    solver.initialize(
        [](double x, double y, double) { return ns2d::alfven_omega(x, y); },
        [](double x, double y, double) { return ns2d::alfven_phi(x, y); });
  }

  Writers w;
  std::ofstream csv;
  if (!cli.outdir.empty()) {
    if (rank == 0) std::filesystem::create_directories(cli.outdir);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto inbox = stack.fft().get_inbox_bounds();
    auto setup = [&](std::optional<pfc::VTKWriter> &vtk, const char *pat,
                     const char *name) {
      vtk.emplace(std::string(cli.outdir) + "/" + pat);
      vtk->set_domain(stack.u().global_size(), inbox.size, inbox.low);
      vtk->set_geometry(stack.u().origin(), stack.u().spacing());
      vtk->set_field_name(name);
    };
    setup(w.omega, "omega_%04d.vti", "omega");
    setup(w.a, "a_%04d.vti", "a");
    setup(w.j, "j_%04d.vti", "j");
    w.omega_bin.emplace(cli.outdir + "/omega_%04d.bin");
    w.omega_bin->set_domain(stack.u().global_size(), inbox.size, inbox.low);
    w.a_bin.emplace(cli.outdir + "/a_%04d.bin");
    w.a_bin->set_domain(stack.u().global_size(), inbox.size, inbox.low);
    w.j_bin.emplace(cli.outdir + "/j_%04d.bin");
    w.j_bin->set_domain(stack.u().global_size(), inbox.size, inbox.low);
    if (rank == 0) {
      csv.open(cli.outdir + "/diagnostics.csv");
      write_csv_header(csv);
      std::ofstream meta(cli.outdir + "/run.json");
      const int issue = cli.cse == ns2d::MHDCase::island_coalescence ? 113 : 23;
      meta << "{\n"
           << "  \"issue\": " << issue << ",\n"
           << "  \"case\": \"" << ns2d::mhd_case_name(cli.cse) << "\",\n"
           << "  \"N\": " << cli.n << ",\n"
           << "  \"steps\": " << cli.steps << ",\n"
           << "  \"dt\": " << dt << ",\n"
           << "  \"nu\": " << cli.nu << ",\n"
           << "  \"eta\": " << cli.eta << ",\n"
           << "  \"drive\": " << cli.drive << ",\n"
           << "  \"drive_definition\": \"orszag_tang initial vorticity is "
              "scaled by this factor; the flux function is unchanged. 1.0 is "
              "the admitted case ().\",\n"
           << "  \"nproc\": " << nproc << ",\n"
           << "  \"cfl_nominal\": " << (dt / dx) << ",\n"
           << "  \"cfl_nominal_definition\": \"dt/dx; --cfl sets dt=CFL*dx\",\n"
           << "  \"cfl_elsasser_definition\": "
              "\"dt * max_i |zpm_i| / dx, zpm = u +/- B\",\n"
           << "  \"cfl_elsasser_sum_definition\": "
              "\"dt * max(|zx|/dx + |zy|/dy) over z+ and z-\",\n"
           << "  \"cfl_abort\": \"measured cfl_elsasser_sum > 2\",\n"
           << "  \"a2_definition\": \"0.5 * <(a-<a>)^2> with a_hat(0)=0\",\n"
           << "  \"budget_definition\": "
              "\"(E_n-E_{n-1})/dt_diag + 0.5*(D_n+D_{n-1}), "
              "D=nu<w^2>+eta<j^2>\",\n"
           << "  \"integrator\": \"IFRK4 pair\",\n"
           << "  \"dealias\": \"Orszag 2/3 on omega, a, and all N_hat\",\n"
           << "  \"lorentz\": \"+B.grad j\",\n"
           << "  \"model\": \"2-D incompressible visco-resistive MHD\"";
      if (cli.cse == ns2d::MHDCase::island_coalescence) {
        meta << ",\n"
             << "  \"a0\": \"0.4 * sin(x) * sin(y)\",\n"
             << "  \"phi0\": \"0.01 * (cos(x) - cos(y))\",\n"
             << "  \"omega0\": \"0.01 * (cos(x) - cos(y))\",\n"
             << "  \"coalescence_abar\": " << ns2d::coalescence_abar << ",\n"
             << "  \"coalescence_eps\": " << ns2d::coalescence_eps << ",\n"
             << "  \"perturbation\": \"frozen; do not retune with eta\"\n";
      } else {
        meta << "\n";
      }
      meta << "}\n";
    }
  }

  auto emit = [&](int step, const ns2d::MHDDiagnostics &d, double wall) {
    if (rank != 0) return;
    std::cout << std::scientific << std::setprecision(6) << "step=" << step
              << " t=" << d.time << " E=" << d.energy << " ke=" << d.ke
              << " me=" << d.me << " max|j|=" << d.max_abs_j
              << " max|w|=" << d.max_abs_omega << " divu=" << d.div_u_linf
              << " divb=" << d.div_b_linf << " budget=" << d.energy_budget_residual
              << " cfl_nom=" << d.cfl_nominal << " cfl_sum=" << d.cfl_elsasser_sum
              << " wall_step_s=" << wall << "\n";
    if (csv) {
      csv << std::setprecision(16) << step << "," << d.time << "," << d.ke << ","
          << d.me << "," << d.energy << "," << d.cross_helicity << "," << d.a2 << ","
          << d.enstrophy << "," << d.mean_sq_j << "," << d.max_abs_omega << ","
          << d.max_abs_j << "," << d.max_speed << "," << d.max_b << ","
          << d.div_u_linf << "," << d.div_b_linf << "," << d.dissipation << ","
          << d.energy_budget_residual << "," << d.cfl_nominal << "," << d.cfl_ub
          << "," << d.cfl_elsasser << "," << d.cfl_elsasser_mag << ","
          << d.cfl_elsasser_sum << "," << d.max_z_inf << "," << d.max_z_mag << ","
          << d.max_z_l1 << "," << d.mean_omega << "," << d.mean_a << "," << wall
          << "\n";
    }
  };

  auto fail_closed = [&](int step, const ns2d::MHDDiagnostics &d) -> int {
    if (!std::isfinite(d.energy) || !std::isfinite(d.max_abs_j) ||
        !std::isfinite(d.cfl_elsasser_sum)) {
      if (rank == 0) {
        std::cerr << "mhd2d: non-finite diagnostic at step " << step << "\n";
      }
      return EXIT_FAILURE;
    }
    if (d.cfl_elsasser_sum > 2.0) {
      if (rank == 0) {
        std::cerr << "mhd2d: measured Elsasser sum CFL=" << d.cfl_elsasser_sum
                  << " > 2 at step " << step << " (nominal=" << d.cfl_nominal
                  << ", component=" << d.cfl_elsasser << ")\n";
      }
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  };

  auto d0 = solver.diagnostics(MPI_COMM_WORLD);
  emit(0, d0, 0.0);
  if (fail_closed(0, d0) != EXIT_SUCCESS) return EXIT_FAILURE;
  if (w.omega) dump_all(w, 0, solver);

  const int diag_every = cli.diag_every > 0
                             ? cli.diag_every
                             : (cli.dump_every > 0 ? cli.dump_every : cli.steps);

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
    const bool want_diag =
        (diag_every > 0 && step % diag_every == 0) || step == cli.steps;
    const bool want_dump =
        static_cast<bool>(w.omega) &&
        ((cli.dump_every > 0 && step % cli.dump_every == 0) || step == cli.steps);
    if (want_diag) {
      auto d = solver.diagnostics(MPI_COMM_WORLD);
      emit(step, d, wall);
      if (fail_closed(step, d) != EXIT_SUCCESS) return EXIT_FAILURE;
    }
    if (want_dump) {
      if (!want_diag) solver.recover_fields();
      dump_all(w, step, solver);
    }
  }

  if (cli.verify) {
    const double T = solver.time();
    const double eta = cli.eta;
    const double linf =
        solver.linf_a_error(MPI_COMM_WORLD, [eta, T](double x, double y) {
          return ns2d::force_free_a_exact(x, y, eta, T);
        });
    const bool pass = linf < 1.0e-9;
    if (rank == 0) {
      std::cout << (pass ? "VERIFY PASS" : "VERIFY FAIL")
                << " force_free Linf=" << std::scientific << linf << "\n";
    }
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  if (rank == 0 && timed > 0) {
    std::cout << "mean wall_step_s=" << (wall_acc / timed) << "\n";
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
