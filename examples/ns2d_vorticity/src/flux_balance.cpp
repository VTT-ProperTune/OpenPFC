// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file flux_balance.cpp
 * @brief Divergence and closed-box flux of the 2-D spectral velocity (#226).
 *
 * The step list is fixed before the run. Shear uses N=128, 256, and 512
 * at dt=0.4/512, and N=256 again at dt=0.4/1024, through t=1.2. Taylor–Green
 * uses N=32 and 64, nu=0.1, dt=0.05, t=0 and t=1. Two-mode uses N=64,
 * nu=0.05, dt=0.01, t=0 and t=0.2. Boxes are the full period, a half box,
 * a shifted quarter, one cell, and one off-node rectangle.
 */

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

#include <ns2d/cases.hpp>
#include <ns2d/flux_balance.hpp>
#include <ns2d/vorticity_stream.hpp>

namespace {

enum class Kind { Shear, TwoMode, Taylor };

struct Options {
  std::string which{"all"};
  int n{0};
  double dt{0.0};
  std::vector<double> times;
  std::optional<double> nu;
  double rho{30.0};
  double eps{0.05};
  std::string outdir{"results/ns2d-226"};
  std::string revision{"unspecified"};
  bool set_n{false};
  bool set_dt{false};
  bool set_times{false};
  bool set_nu{false};
  bool set_rho{false};
  bool set_eps{false};
};

struct Spec {
  std::string name;
  Kind kind{Kind::Shear};
  int n{0};
  double nu{0.0};
  double dt{0.0};
  double rho{30.0};
  double eps{0.05};
  std::vector<double> times;
};

struct Held {
  std::unique_ptr<pfc::sim::stacks::SpectralCPUStack> stack;
  std::unique_ptr<ns2d::VorticityStreamCPU> solver;
};

struct NamedBox {
  std::string name;
  ns2d::AxisBox box;
};

bool field_is_finite(ns2d::VorticityStreamCPU &solver) {
  bool ok = true;
  solver.omega().for_each_owned([&](int i, int j, int k) {
    if (!std::isfinite(solver.omega()(i, j, k))) ok = false;
  });
  int local = ok ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return global == 1;
}

Held make_solver(Kind kind, int n, double dt, double nu, int rank, int nproc) {
  const auto domain =
      (kind == Kind::Shear) ? ns2d::make_unit_slab(n) : ns2d::make_twopi_slab(n);
  Held h;
  h.stack = std::make_unique<pfc::sim::stacks::SpectralCPUStack>(domain, rank, nproc,
                                                                 MPI_COMM_WORLD);
  h.solver =
      std::make_unique<ns2d::VorticityStreamCPU>(*h.stack, ns2d::Params{nu, dt});
  return h;
}

std::vector<long long> save_steps(const std::vector<double> &times, double dt) {
  if (dt <= 0.0) throw std::invalid_argument("ns2d: dt must be positive");
  std::vector<long long> steps;
  for (double t : times) {
    if (t < 0.0) throw std::invalid_argument("ns2d: time must be >= 0");
    const double q = t / dt;
    const long long s = std::llround(q);
    if (std::abs(q - static_cast<double>(s)) > 1.0e-8) {
      throw std::invalid_argument("ns2d: sample time is not an integer step");
    }
    steps.push_back(s);
  }
  for (std::size_t i = 1; i < steps.size(); ++i) {
    if (steps[i] < steps[i - 1]) {
      throw std::invalid_argument("ns2d: times must be nondecreasing");
    }
  }
  return steps;
}

std::vector<NamedBox> control_volumes(double length, int n) {
  const double dx = length / static_cast<double>(n);
  return {
      {"full", {0.0, length, 0.0, length}},
      {"half", {0.0, 0.5 * length, 0.0, 0.5 * length}},
      {"quarter", {0.125 * length, 0.375 * length, 0.125 * length, 0.375 * length}},
      {"cell", {0.0, dx, 0.0, dx}},
      {"offset", {0.2 * length, 0.7 * length, 0.15 * length, 0.45 * length}},
  };
}

void write_hats(ns2d::VorticityStreamCPU &solver,
                std::vector<ns2d::SpectralPlane::Complex> &u_hat,
                std::vector<ns2d::SpectralPlane::Complex> &v_hat) {
  u_hat.resize(solver.plane().out_n());
  v_hat.resize(solver.plane().out_n());
  solver.plane().fft().forward(solver.u(), u_hat);
  solver.plane().fft().forward(solver.v(), v_hat);
}

std::string csv_trap(const std::optional<double> &trap) {
  if (!trap) return {};
  std::ostringstream os;
  os << std::scientific << std::setprecision(16) << *trap;
  return os.str();
}

bool run_spec(const Spec &spec, std::ostream &csv, int rank, int nproc) {
  const double length = (spec.kind == Kind::Shear) ? 1.0 : 2.0 * pfc::pi;
  const auto steps = save_steps(spec.times, spec.dt);
  const auto boxes = control_volumes(length, spec.n);
  if (rank == 0) {
    std::cout << "flux case=" << spec.name << " N=" << spec.n
              << " dt=" << std::scientific << std::setprecision(8) << spec.dt
              << std::endl;
  }
  auto held = make_solver(spec.kind, spec.n, spec.dt, spec.nu, rank, nproc);
  auto &solver = *held.solver;
  if (spec.kind == Kind::Shear) {
    solver.initialize_omega([&](double x, double y, double) {
      return ns2d::double_shear_omega(x, y, spec.rho, spec.eps);
    });
  } else if (spec.kind == Kind::TwoMode) {
    solver.initialize_omega(
        [](double x, double y, double) { return ns2d::two_mode_omega(x, y); });
  } else {
    solver.initialize_omega([&](double x, double y, double) {
      return ns2d::taylor_green_omega(x, y, spec.nu, 0.0);
    });
  }
  const double mean0 = solver.diagnostics(MPI_COMM_WORLD).mean_omega;
  const long long last = steps.empty() ? 0 : steps.back();
  std::size_t next = 0;
  bool ok = true;

  auto capture = [&](long long step) {
    const double time = static_cast<double>(step) * spec.dt;
    std::string status = "ok";
    ns2d::Diagnostics diag{};
    std::vector<ns2d::SpectralPlane::Complex> u_hat, v_hat;
    double modal = 0.0;
    if (!field_is_finite(solver)) {
      status = "nonfinite";
      ok = false;
    } else {
      diag = solver.diagnostics(MPI_COMM_WORLD);
      if (diag.cfl > 2.0) {
        status = "cfl";
        ok = false;
      }
      write_hats(solver, u_hat, v_hat);
      modal = ns2d::modal_div_amplitude(solver.plane(), u_hat, v_hat);
    }
    const double domain_div =
        (status == "nonfinite")
            ? 0.0
            : ns2d::domain_divergence(solver.plane(), u_hat, v_hat);
    if (rank == 0) {
      for (const auto &named : boxes) {
        ns2d::FaceFlux flux{};
        std::optional<double> trap;
        if (status != "nonfinite") {
          flux = ns2d::face_flux(solver.plane(), u_hat, v_hat, named.box);
          trap = ns2d::trapezoid_flux(solver.plane(), solver.u(), solver.v(),
                                      named.box);
        }
        csv << spec.name << ',' << spec.n << ',' << std::scientific
            << std::setprecision(16) << spec.dt << ',' << step << ',' << time << ','
            << named.name << ',' << named.box.x0 << ',' << named.box.x1 << ','
            << named.box.y0 << ',' << named.box.y1 << ',' << flux.net << ','
            << flux.imag << ',' << flux.abs_faces << ',' << flux.relative << ','
            << csv_trap(trap) << ',' << diag.div_linf << ',' << diag.div_l2 << ','
            << modal << ',' << domain_div << ',' << diag.ke << ',' << diag.enstrophy
            << ',' << diag.mean_omega << ','
            << (status == "nonfinite" ? 0.0 : diag.mean_omega - mean0) << ','
            << diag.cfl << ',' << status << '\n';
      }
    }
    return status == "ok";
  };

  if (!steps.empty() && steps[0] == 0) {
    if (!capture(0)) return false;
    ++next;
  }
  for (long long step = 1; step <= last && ok; ++step) {
    solver.step();
    if (next < steps.size() && step == steps[next]) {
      if (!capture(step)) return false;
      ++next;
    }
  }
  return ok && next == steps.size();
}

std::vector<double> parse_doubles(std::string_view text) {
  std::vector<double> out;
  std::stringstream ss{std::string(text)};
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    out.push_back(std::atof(item.c_str()));
  }
  return out;
}

Spec base_spec(Kind kind, const Options &opt) {
  Spec s;
  s.kind = kind;
  if (kind == Kind::Shear) {
    s.name = "shear";
    s.n = opt.set_n ? opt.n : 0;
    s.nu = opt.set_nu ? *opt.nu : 1.0e-4;
    s.dt = opt.set_dt ? opt.dt : 0.4 / 512.0;
    s.rho = opt.rho;
    s.eps = opt.eps;
    s.times = opt.set_times ? opt.times : std::vector<double>{0.4, 0.8, 1.2};
  } else if (kind == Kind::TwoMode) {
    s.name = "two-mode";
    s.n = opt.set_n ? opt.n : 64;
    s.nu = opt.set_nu ? *opt.nu : 0.05;
    s.dt = opt.set_dt ? opt.dt : 0.01;
    s.times = opt.set_times ? opt.times : std::vector<double>{0.0, 0.2};
  } else {
    s.name = "taylor";
    s.n = opt.set_n ? opt.n : 32;
    s.nu = opt.set_nu ? *opt.nu : 0.1;
    s.dt = opt.set_dt ? opt.dt : 0.05;
    s.times = opt.set_times ? opt.times : std::vector<double>{0.0, 1.0};
  }
  return s;
}

std::vector<Spec> expand(Kind kind, const Options &opt) {
  if (opt.set_n || opt.set_dt || opt.set_times || opt.set_nu) {
    auto s = base_spec(kind, opt);
    if (s.n < 4) throw std::invalid_argument("ns2d: N >= 4 is required");
    return {s};
  }
  if (kind == Kind::Shear) {
    std::vector<Spec> runs;
    for (int n : {128, 256, 512}) {
      Spec s = base_spec(kind, opt);
      s.n = n;
      s.dt = 0.4 / 512.0;
      runs.push_back(s);
    }
    Spec fine = base_spec(kind, opt);
    fine.n = 256;
    fine.dt = 0.4 / 1024.0;
    runs.push_back(fine);
    return runs;
  }
  if (kind == Kind::Taylor) {
    Spec a = base_spec(kind, opt);
    a.n = 32;
    Spec b = a;
    b.n = 64;
    return {a, b};
  }
  return {base_spec(kind, opt)};
}

void usage(std::ostream &os) {
  os << "Usage: ns2d_flux_balance [--case shear|two-mode|taylor|all]\n"
        "       [--n N] [--dt DT] [--times t1,t2,...] [--nu NU]\n"
        "       [--rho RHO] [--eps EPS] [--outdir DIR] [--revision SHA]\n"
        "\n"
        "Default shear: N=128,256,512 at dt=0.4/512, plus N=256 at\n"
        "dt=0.4/1024, t=0.4,0.8,1.2. Taylor-Green: N=32,64, nu=0.1,\n"
        "dt=0.05, t=0,1. Two-mode: N=64, nu=0.05, dt=0.01, t=0,0.2.\n"
        "Overrides apply to a single --case. One MPI rank.\n";
}

std::optional<std::string_view> take_value(int &i, int argc, char **argv) {
  if (i + 1 >= argc) return std::nullopt;
  return std::string_view(argv[++i]);
}

bool parse(int argc, char **argv, Options &opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view a(argv[i]);
    auto need = [&](const char *flag) -> std::optional<std::string_view> {
      auto v = take_value(i, argc, argv);
      if (!v) std::cerr << "missing value for " << flag << "\n";
      return v;
    };
    if (a == "-h" || a == "--help") {
      usage(std::cout);
      std::exit(EXIT_SUCCESS);
    } else if (a == "--case") {
      auto v = need("--case");
      if (!v) return false;
      opt.which = std::string(*v);
    } else if (a == "--n") {
      auto v = need("--n");
      if (!v) return false;
      opt.n = std::atoi(std::string(*v).c_str());
      opt.set_n = true;
    } else if (a == "--dt") {
      auto v = need("--dt");
      if (!v) return false;
      opt.dt = std::atof(std::string(*v).c_str());
      opt.set_dt = true;
    } else if (a == "--times") {
      auto v = need("--times");
      if (!v) return false;
      opt.times = parse_doubles(*v);
      opt.set_times = true;
    } else if (a == "--nu") {
      auto v = need("--nu");
      if (!v) return false;
      opt.nu = std::atof(std::string(*v).c_str());
      opt.set_nu = true;
    } else if (a == "--rho") {
      auto v = need("--rho");
      if (!v) return false;
      opt.rho = std::atof(std::string(*v).c_str());
      opt.set_rho = true;
    } else if (a == "--eps") {
      auto v = need("--eps");
      if (!v) return false;
      opt.eps = std::atof(std::string(*v).c_str());
      opt.set_eps = true;
    } else if (a == "--outdir") {
      auto v = need("--outdir");
      if (!v) return false;
      opt.outdir = std::string(*v);
    } else if (a == "--revision") {
      auto v = need("--revision");
      if (!v) return false;
      opt.revision = std::string(*v);
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      return false;
    }
  }
  if (opt.which != "shear" && opt.which != "two-mode" && opt.which != "taylor" &&
      opt.which != "all") {
    return false;
  }
  if (opt.which == "all" && (opt.set_n || opt.set_dt || opt.set_times ||
                             opt.set_nu || opt.set_rho || opt.set_eps)) {
    std::cerr << "ns2d_flux_balance: overrides require a single --case\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        Options opt;
        if (!parse(app_argc, app_argv, opt)) {
          if (rank == 0) usage(std::cerr);
          return EXIT_FAILURE;
        }
        if (nproc != 1) {
          if (rank == 0) {
            std::cerr << "ns2d_flux_balance: one rank is required so every "
                         "Fourier mode is owned locally\n";
          }
          return EXIT_FAILURE;
        }
        std::vector<Spec> specs;
        try {
          if (opt.which == "taylor" || opt.which == "all") {
            auto part = expand(Kind::Taylor, opt);
            specs.insert(specs.end(), part.begin(), part.end());
          }
          if (opt.which == "two-mode" || opt.which == "all") {
            auto part = expand(Kind::TwoMode, opt);
            specs.insert(specs.end(), part.begin(), part.end());
          }
          if (opt.which == "shear" || opt.which == "all") {
            auto part = expand(Kind::Shear, opt);
            specs.insert(specs.end(), part.begin(), part.end());
          }
        } catch (const std::exception &ex) {
          if (rank == 0) std::cerr << ex.what() << "\n";
          return EXIT_FAILURE;
        }

        if (rank == 0) std::filesystem::create_directories(opt.outdir);
        std::ofstream csv;
        if (rank == 0) {
          csv.open(opt.outdir + "/flux_balance.csv");
          csv << "case,N,dt,step,time,box,x0,x1,y0,y1,net,imag,abs_faces,"
                 "relative,trap_net,div_linf,div_l2,modal_div,domain_div,ke,"
                 "enstrophy,mean_omega,mean_drift,cfl,status\n";
          std::ofstream meta(opt.outdir + "/run.json");
          meta << "{\n"
               << "  \"issue\": 226,\n"
               << "  \"revision\": \"" << opt.revision << "\",\n"
               << "  \"flux\": \"r2c trigonometric interpolant, conjugate "
                  "completed\",\n"
               << "  \"trapezoid\": \"grid nodes only; not a property of the "
                  "scheme\",\n"
               << "  \"shear\": \"N=128,256,512 dt=0.4/512 plus N=256 "
                  "dt=0.4/1024, t=0.4,0.8,1.2\",\n"
               << "  \"taylor\": \"N=32,64 nu=0.1 dt=0.05 t=0,1\",\n"
               << "  \"two_mode\": \"N=64 nu=0.05 dt=0.01 t=0,0.2\"\n"
               << "}\n";
        }

        bool ok = true;
        for (const auto &spec : specs) {
          try {
            if (!run_spec(spec, csv, rank, nproc)) ok = false;
          } catch (const std::exception &ex) {
            if (rank == 0) std::cerr << ex.what() << "\n";
            ok = false;
          }
        }
        if (rank == 0) {
          csv.close();
          std::cout << "flux_balance rows written to " << opt.outdir
                    << "/flux_balance.csv\n";
          std::cout << "flux_balance status=" << (ok ? "ok" : "failed") << std::endl;
        }
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
      });
}
