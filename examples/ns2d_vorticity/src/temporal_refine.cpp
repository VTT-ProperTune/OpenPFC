// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file temporal_refine.cpp
 * @brief Fixed-grid timestep ladder for IFRK4 (#225).
 *
 * The step list is chosen before the run. Shear uses N=512, the finest
 * grid that finished the fixed-dt spatial ladder. Two-mode uses N=64:
 * the initial field is modes 1 and 2 on [0, 2π]², and the quadratic
 * interactions stay inside the 2/3 mask. Taylor-Green is not a case.
 * A step that hits the CFL guard or becomes non-finite is recorded and
 * the rest of the ladder still runs. Order uses absolute vorticity L2
 * against the finest step, and only above the floor.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
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

#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

#include <ns2d/band_error.hpp>
#include <ns2d/cases.hpp>
#include <ns2d/temporal_order.hpp>
#include <ns2d/vorticity_stream.hpp>

namespace {

enum class CaseKind { Shear, TwoMode };

struct Options {
  std::string which{"both"};
  int n{0};
  std::vector<double> dts;
  std::vector<double> times;
  std::optional<double> nu;
  double rho{30.0};
  double eps{0.05};
  double order_floor{1.0e-11};
  std::string outdir{"results/ns2d-225"};
  std::string revision{"unspecified"};
  bool set_n{false};
  bool set_dts{false};
  bool set_times{false};
  bool set_nu{false};
  bool set_rho{false};
  bool set_eps{false};
};

struct Sample {
  long long step{0};
  double time{0.0};
  ns2d::Diagnostics diag{};
  ns2d::CommonBandError err{};
  double mean0{0.0};
  double outer{0.0};
  std::string status{"ok"};
  std::vector<ns2d::SpectralPlane::Complex> hat;
  std::string order_omega;
  std::string order_velocity;
};

struct Held {
  std::unique_ptr<pfc::sim::stacks::SpectralCPUStack> stack;
  std::unique_ptr<ns2d::VorticityStreamCPU> solver;
};

struct Protocol {
  std::string name;
  CaseKind kind{CaseKind::Shear};
  int n{0};
  double nu{0.0};
  double rho{30.0};
  double eps{0.05};
  std::vector<double> times;
  std::vector<double> dts;
  double dt_ref{0.0};
};

Held make_solver(CaseKind kind, int n, double dt, double nu, int rank, int nproc) {
  const auto domain =
      (kind == CaseKind::Shear) ? ns2d::make_unit_slab(n) : ns2d::make_twopi_slab(n);
  Held h;
  h.stack = std::make_unique<pfc::sim::stacks::SpectralCPUStack>(domain, rank, nproc,
                                                                 MPI_COMM_WORLD);
  h.solver =
      std::make_unique<ns2d::VorticityStreamCPU>(*h.stack, ns2d::Params{nu, dt});
  return h;
}

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

std::vector<long long> reference_steps(const std::vector<double> &times,
                                       double dt_ref) {
  std::vector<long long> steps;
  steps.reserve(times.size());
  for (double t : times) {
    if (t < 0.0) throw std::invalid_argument("ns2d: time must be >= 0");
    steps.push_back(std::llround(t / dt_ref));
  }
  for (std::size_t i = 1; i < steps.size(); ++i) {
    if (steps[i] < steps[i - 1]) {
      throw std::invalid_argument("ns2d: times must be nondecreasing");
    }
  }
  return steps;
}

std::vector<long long> coarse_steps(const std::vector<long long> &ref_steps,
                                    long long sub) {
  if (sub < 1) throw std::invalid_argument("ns2d: dt must be >= reference dt");
  std::vector<long long> steps;
  steps.reserve(ref_steps.size());
  for (long long s : ref_steps) {
    if (s % sub != 0) {
      throw std::invalid_argument(
          "ns2d: a sample time is not an integer number of coarse steps");
    }
    steps.push_back(s / sub);
  }
  return steps;
}

long long subdivision(double dt, double dt_ref) {
  const long long sub = std::llround(dt / dt_ref);
  if (sub < 1) throw std::invalid_argument("ns2d: dt is below the reference");
  const double snapped = static_cast<double>(sub) * dt_ref;
  if (std::abs(dt - snapped) > 1.0e-8 * dt_ref) {
    throw std::invalid_argument(
        "ns2d: every dt must be an integer multiple of the finest dt");
  }
  return sub;
}

std::vector<Sample> integrate(
    CaseKind kind, int n, double dt, double nu, double rho, double eps,
    const std::vector<long long> &save_steps, ns2d::SpectralPlane *reference_plane,
    const std::vector<std::vector<ns2d::SpectralPlane::Complex>> *reference_hats,
    bool keep_hats, int rank, int nproc) {
  auto held = make_solver(kind, n, dt, nu, rank, nproc);
  auto &solver = *held.solver;
  if (kind == CaseKind::Shear) {
    solver.initialize_omega([&](double x, double y, double) {
      return ns2d::double_shear_omega(x, y, rho, eps);
    });
  } else {
    solver.initialize_omega(
        [](double x, double y, double) { return ns2d::two_mode_omega(x, y); });
  }
  const double mean0 = solver.diagnostics(MPI_COMM_WORLD).mean_omega;

  std::vector<Sample> samples;
  samples.reserve(save_steps.size());
  std::size_t next = 0;
  const long long last = save_steps.empty() ? 0 : save_steps.back();

  auto capture = [&](long long step) {
    Sample s;
    s.step = step;
    s.time = static_cast<double>(step) * dt;
    s.mean0 = mean0;
    if (!field_is_finite(solver)) {
      s.status = "nonfinite";
      samples.push_back(std::move(s));
      return false;
    }
    s.diag = solver.diagnostics(MPI_COMM_WORLD);
    s.outer = ns2d::outer_band_rms_fraction(solver, MPI_COMM_WORLD);
    if (s.diag.cfl > 2.0) s.status = "cfl";
    if (reference_plane != nullptr && reference_hats != nullptr) {
      s.err = ns2d::common_band_error_from_hat(
          *reference_plane, (*reference_hats)[next], solver, MPI_COMM_WORLD);
    }
    if (keep_hats) solver.copy_omega_hat(s.hat);
    const bool stop = s.status != "ok";
    samples.push_back(std::move(s));
    return !stop;
  };

  if (!save_steps.empty() && save_steps[0] == 0) {
    if (!capture(0)) return samples;
    ++next;
  }
  for (long long step = 1; step <= last; ++step) {
    solver.step();
    if (next < save_steps.size() && step == save_steps[next]) {
      if (!capture(step)) return samples;
      ++next;
    }
  }
  return samples;
}

std::string format_order(const std::optional<double> &p) {
  if (!p) return "roundoff";
  std::ostringstream os;
  os << std::scientific << std::setprecision(8) << *p;
  return os.str();
}

void assign_orders(std::vector<std::vector<Sample>> &by_dt, double dt_ref,
                   const std::vector<double> &dts, double floor) {
  if (by_dt.size() != dts.size()) return;
  for (std::size_t i = 0; i + 1 < dts.size(); ++i) {
    if (dts[i + 1] == dt_ref) continue;
    const std::size_t ntime = std::min(by_dt[i].size(), by_dt[i + 1].size());
    for (std::size_t t = 0; t < ntime; ++t) {
      auto &coarse = by_dt[i][t];
      const auto &fine = by_dt[i + 1][t];
      if (coarse.status != "ok" || fine.status != "ok") continue;
      coarse.order_omega = format_order(
          ns2d::observed_order(coarse.err.omega_l2, fine.err.omega_l2, floor));
      coarse.order_velocity = format_order(
          ns2d::observed_order(coarse.err.velocity_l2, fine.err.velocity_l2, floor));
    }
  }
}

void write_row(std::ostream &os, const Protocol &proto, double dt, const Sample *ref,
               const Sample &s) {
  const double ke_diff = (ref != nullptr) ? s.diag.ke - ref->diag.ke : 0.0;
  const double z_diff =
      (ref != nullptr) ? s.diag.enstrophy - ref->diag.enstrophy : 0.0;
  os << proto.name << ',' << proto.n << ',' << std::scientific
     << std::setprecision(16) << dt << ',' << proto.dt_ref << ',' << s.step << ','
     << s.time << ',' << s.err.omega_l2 << ',' << s.err.omega_l2_rel << ','
     << s.err.omega_linf << ',' << s.err.velocity_l2 << ',' << s.err.velocity_l2_rel
     << ',' << s.diag.ke << ',' << ke_diff << ',' << s.diag.enstrophy << ','
     << z_diff << ',' << s.diag.max_abs_omega << ',' << s.diag.div_linf << ','
     << s.diag.div_l2 << ',' << s.diag.cfl << ',' << s.diag.mean_omega << ','
     << (s.diag.mean_omega - s.mean0) << ',' << s.outer << ',' << s.order_omega
     << ',' << s.order_velocity << ',' << s.status << '\n';
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

std::vector<double> shear_dts() {
  const double dt_ref = 0.4 / 2048.0;
  return {dt_ref * 16.0, dt_ref * 8.0, dt_ref * 4.0, dt_ref * 2.0, dt_ref};
}

std::vector<double> two_mode_dts() {
  const double dt_ref = 0.2 / 160.0;
  return {dt_ref * 32.0, dt_ref * 16.0, dt_ref * 8.0,
          dt_ref * 4.0,  dt_ref * 2.0,  dt_ref};
}

Protocol make_protocol(CaseKind kind, const Options &opt) {
  Protocol p;
  p.kind = kind;
  if (kind == CaseKind::Shear) {
    p.name = "shear";
    p.n = opt.set_n ? opt.n : 512;
    p.nu = opt.set_nu ? *opt.nu : 1.0e-4;
    p.rho = opt.rho;
    p.eps = opt.eps;
    p.times = opt.set_times ? opt.times : std::vector<double>{0.4, 0.8, 1.2};
    p.dts = opt.set_dts ? opt.dts : shear_dts();
  } else {
    p.name = "two-mode";
    p.n = opt.set_n ? opt.n : 64;
    p.nu = opt.set_nu ? *opt.nu : 0.05;
    p.times = opt.set_times ? opt.times : std::vector<double>{0.2};
    p.dts = opt.set_dts ? opt.dts : two_mode_dts();
  }
  if (p.dts.empty() || p.times.empty() || p.n < 4 || p.nu < 0.0 || p.rho <= 0.0) {
    throw std::invalid_argument("ns2d: incomplete temporal protocol");
  }
  p.dt_ref = p.dts.front();
  for (double dt : p.dts) {
    if (dt <= 0.0) throw std::invalid_argument("ns2d: dt must be positive");
    if (dt < p.dt_ref) p.dt_ref = dt;
  }
  std::sort(p.dts.begin(), p.dts.end(), std::greater<double>());
  const auto ref_steps = reference_steps(p.times, p.dt_ref);
  for (double dt : p.dts) {
    coarse_steps(ref_steps, subdivision(dt, p.dt_ref));
  }
  return p;
}

void usage(std::ostream &os) {
  os << "Usage: ns2d_temporal_refine [--case shear|two-mode|both]\n"
        "       [--n N] [--dts d1,d2,...] [--times t1,t2,...]\n"
        "       [--nu NU] [--rho RHO] [--eps EPS]\n"
        "       [--order-floor 1e-11] [--outdir DIR] [--revision SHA]\n"
        "\n"
        "Fixed grid, several timesteps, common-band error against the\n"
        "finest dt. Default shear: N=512, dt = (0.4/2048)*{16,8,4,2,1},\n"
        "t=0.4,0.8,1.2. Default two-mode: N=64 on [0,2π]², nu=0.05,\n"
        "t=0.2, dt = (0.2/160)*{32,16,8,4,2,1}. One MPI rank.\n"
        "Overrides apply to a single --case, not to both.\n";
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
    } else if (a == "--dts") {
      auto v = need("--dts");
      if (!v) return false;
      opt.dts = parse_doubles(*v);
      opt.set_dts = true;
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
    } else if (a == "--order-floor") {
      auto v = need("--order-floor");
      if (!v) return false;
      opt.order_floor = std::atof(std::string(*v).c_str());
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
  if (opt.which != "shear" && opt.which != "two-mode" && opt.which != "both") {
    return false;
  }
  if (opt.which == "both" && (opt.set_n || opt.set_dts || opt.set_times ||
                              opt.set_nu || opt.set_rho || opt.set_eps)) {
    std::cerr << "ns2d_temporal_refine: overrides require a single --case\n";
    return false;
  }
  if (opt.order_floor < 0.0) return false;
  return true;
}

bool run_protocol(const Protocol &proto, const Options &opt, std::ostream &csv,
                  int rank, int nproc) {
  const auto ref_steps = reference_steps(proto.times, proto.dt_ref);
  if (rank == 0) {
    std::cout << "temporal ladder case=" << proto.name << " N=" << proto.n
              << " nu=" << std::scientific << std::setprecision(8) << proto.nu
              << " dt_ref=" << proto.dt_ref << std::endl;
  }

  const long long ref_sub = subdivision(proto.dt_ref, proto.dt_ref);
  auto ref_samples = integrate(
      proto.kind, proto.n, proto.dt_ref, proto.nu, proto.rho, proto.eps,
      coarse_steps(ref_steps, ref_sub), nullptr, nullptr, true, rank, nproc);
  bool ref_ok = ref_samples.size() == ref_steps.size();
  for (const auto &s : ref_samples) {
    if (s.status != "ok") ref_ok = false;
  }

  std::vector<std::vector<ns2d::SpectralPlane::Complex>> ref_hats;
  for (const auto &s : ref_samples) ref_hats.push_back(s.hat);
  std::optional<Held> reference;
  if (ref_ok) {
    reference =
        make_solver(proto.kind, proto.n, proto.dt_ref, proto.nu, rank, nproc);
  }

  std::vector<double> dts = proto.dts;
  std::vector<std::vector<Sample>> by_dt;
  by_dt.reserve(dts.size());
  for (double dt : dts) {
    const long long sub = subdivision(dt, proto.dt_ref);
    const double dt_use = static_cast<double>(sub) * proto.dt_ref;
    if (rank == 0) {
      std::cout << "temporal case=" << proto.name << " dt=" << std::scientific
                << std::setprecision(8) << dt_use << std::endl;
    }
    std::vector<Sample> samples;
    if (dt_use == proto.dt_ref) {
      samples = std::move(ref_samples);
    } else if (!ref_ok) {
      continue;
    } else {
      samples =
          integrate(proto.kind, proto.n, dt_use, proto.nu, proto.rho, proto.eps,
                    coarse_steps(ref_steps, sub), &reference->solver->plane(),
                    &ref_hats, false, rank, nproc);
    }
    by_dt.push_back(std::move(samples));
  }

  std::vector<double> ran_dts;
  ran_dts.reserve(by_dt.size());
  for (double dt : dts) {
    if (ran_dts.size() == by_dt.size()) break;
    if (!ref_ok && dt != proto.dt_ref) continue;
    ran_dts.push_back(static_cast<double>(subdivision(dt, proto.dt_ref)) *
                      proto.dt_ref);
  }
  if (!ran_dts.empty()) {
    assign_orders(by_dt, proto.dt_ref, ran_dts, opt.order_floor);
  }

  if (rank == 0) {
    for (std::size_t i = 0; i < by_dt.size(); ++i) {
      const double dt = ran_dts[i];
      const Sample *ref_row = nullptr;
      std::size_t ref_index = by_dt.size();
      for (std::size_t j = 0; j < ran_dts.size(); ++j) {
        if (ran_dts[j] == proto.dt_ref) ref_index = j;
      }
      for (std::size_t t = 0; t < by_dt[i].size(); ++t) {
        ref_row = nullptr;
        if (dt != proto.dt_ref && ref_index < by_dt.size() &&
            t < by_dt[ref_index].size() && by_dt[ref_index][t].status == "ok") {
          ref_row = &by_dt[ref_index][t];
        }
        write_row(csv, proto, dt, ref_row, by_dt[i][t]);
      }
    }
  }
  return ref_ok;
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
            std::cerr << "ns2d_temporal_refine: one rank is required so every "
                         "Fourier mode is owned locally\n";
          }
          return EXIT_FAILURE;
        }

        std::vector<Protocol> protocols;
        try {
          if (opt.which == "shear" || opt.which == "both") {
            protocols.push_back(make_protocol(CaseKind::Shear, opt));
          }
          if (opt.which == "two-mode" || opt.which == "both") {
            protocols.push_back(make_protocol(CaseKind::TwoMode, opt));
          }
        } catch (const std::exception &ex) {
          if (rank == 0) std::cerr << ex.what() << "\n";
          return EXIT_FAILURE;
        }

        if (rank == 0) std::filesystem::create_directories(opt.outdir);
        std::ofstream csv;
        if (rank == 0) {
          csv.open(opt.outdir + "/temporal_errors.csv");
          csv << "case,N,dt,reference_dt,step,time,omega_l2,omega_l2_rel,"
                 "omega_linf,velocity_l2,velocity_l2_rel,ke,ke_diff,enstrophy,"
                 "enstrophy_diff,max_abs_omega,div_linf,div_l2,cfl,mean_omega,"
                 "mean_drift,outer_rms_fraction,order_omega_l2,"
                 "order_velocity_l2,status\n";
          std::ofstream meta(opt.outdir + "/run.json");
          meta << "{\n"
               << "  \"issue\": 225,\n"
               << "  \"revision\": \"" << opt.revision << "\",\n"
               << "  \"order_floor\": " << std::scientific << std::setprecision(16)
               << opt.order_floor << ",\n"
               << "  \"order\": \"log2(e(dt)/e(dt/2)) on absolute L2 against "
                  "the finest dt; withheld at or below the floor and on the "
                  "finest measured step\",\n"
               << "  \"comparison\": \"restrict_hat_by_k at fixed N\",\n"
               << "  \"shear\": \"N=512 unit square, dt=(0.4/2048)*{16,8,4,2,1},"
                  " t=0.4,0.8,1.2\",\n"
               << "  \"two_mode\": \"N=64 on [0,2pi]^2, nu=0.05, t=0.2, "
                  "dt=(0.2/160)*{32,16,8,4,2,1}\"\n"
               << "}\n";
        }

        bool ok = true;
        for (const auto &proto : protocols) {
          try {
            if (!run_protocol(proto, opt, csv, rank, nproc)) ok = false;
          } catch (const std::exception &ex) {
            if (rank == 0) std::cerr << ex.what() << "\n";
            ok = false;
          }
        }
        if (rank == 0) {
          csv.close();
          std::cout << "temporal_refine rows written to " << opt.outdir
                    << "/temporal_errors.csv\n";
          std::cout << "temporal_refine status=" << (ok ? "ok" : "failed")
                    << std::endl;
        }
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
      });
}
