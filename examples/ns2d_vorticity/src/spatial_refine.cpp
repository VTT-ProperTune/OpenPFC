// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file spatial_refine.cpp
 * @brief Fixed-dt spatial ladder for the unit-square double shear (#224).
 *
 * Every resolution uses the same dt. Field error is the common-band
 * difference against the finest grid, not a collocation interpolant.
 * One extra resolution is repeated at dt/2 so a timestep-dominated
 * comparison is visible. Taylor-Green is not this gate.
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
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

#include <ns2d/band_error.hpp>
#include <ns2d/cases.hpp>
#include <ns2d/vorticity_stream.hpp>

namespace {

struct Options {
  std::vector<int> resolutions{64, 128, 256, 512};
  int reference{512};
  double dt{0.4 / 512.0};
  std::vector<double> times{0.4, 0.8, 1.2};
  double nu{1.0e-4};
  double rho{30.0};
  double eps{0.05};
  int control_n{256};
  double control_factor{0.5};
  std::string outdir{"results/ns2d-224"};
  std::string revision{"unspecified"};
};

struct Sample {
  long long step{0};
  double time{0.0};
  ns2d::Diagnostics diag{};
  ns2d::CommonBandError err{};
  double outer{0.0};
  std::string status{"ok"};
  std::vector<ns2d::SpectralPlane::Complex> hat;
};

struct Held {
  std::unique_ptr<pfc::sim::stacks::SpectralCPUStack> stack;
  std::unique_ptr<ns2d::VorticityStreamCPU> solver;
};

Held make_solver(int n, double dt, double nu, int rank, int nproc) {
  Held h;
  h.stack = std::make_unique<pfc::sim::stacks::SpectralCPUStack>(
      ns2d::make_unit_slab(n), rank, nproc, MPI_COMM_WORLD);
  h.solver =
      std::make_unique<ns2d::VorticityStreamCPU>(*h.stack, ns2d::Params{nu, dt});
  return h;
}

std::vector<long long> steps_for(const std::vector<double> &times, double dt) {
  std::vector<long long> steps;
  steps.reserve(times.size());
  for (double t : times) {
    if (t < 0.0) throw std::invalid_argument("ns2d: time must be >= 0");
    steps.push_back(std::llround(t / dt));
  }
  for (std::size_t i = 1; i < steps.size(); ++i) {
    if (steps[i] < steps[i - 1]) {
      throw std::invalid_argument("ns2d: times must be nondecreasing");
    }
  }
  return steps;
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

std::vector<Sample> integrate(
    int n, double dt, double nu, double rho, double eps,
    const std::vector<long long> &save_steps, ns2d::SpectralPlane *reference_plane,
    const std::vector<std::vector<ns2d::SpectralPlane::Complex>> *reference_hats,
    bool keep_hats, int rank, int nproc) {
  auto held = make_solver(n, dt, nu, rank, nproc);
  auto &solver = *held.solver;
  solver.initialize_omega([&](double x, double y, double) {
    return ns2d::double_shear_omega(x, y, rho, eps);
  });

  std::vector<Sample> samples;
  samples.reserve(save_steps.size());
  std::size_t next = 0;
  const long long last = save_steps.empty() ? 0 : save_steps.back();

  auto capture = [&](long long step) {
    Sample s;
    s.step = step;
    s.time = static_cast<double>(step) * dt;
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

void write_row(std::ostream &os, const std::string &kind, int n, int reference,
               double dt, const Sample &s) {
  os << kind << ',' << n << ',' << reference << ',' << std::scientific
     << std::setprecision(16) << dt << ',' << s.step << ',' << s.time << ','
     << s.err.omega_l2 << ',' << s.err.omega_l2_rel << ',' << s.err.omega_linf << ','
     << s.err.velocity_l2 << ',' << s.err.velocity_l2_rel << ',' << s.diag.ke << ','
     << s.diag.enstrophy << ',' << s.diag.max_abs_omega << ',' << s.diag.div_linf
     << ',' << s.diag.div_l2 << ',' << s.diag.cfl << ',' << s.diag.mean_omega << ','
     << s.outer << ',' << s.status << '\n';
}

std::vector<int> parse_ints(std::string_view text) {
  std::vector<int> out;
  std::stringstream ss{std::string(text)};
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    out.push_back(std::atoi(item.c_str()));
  }
  return out;
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

void usage(std::ostream &os) {
  os << "Usage: ns2d_spatial_refine [--resolutions 64,128,256,512]\n"
        "       [--reference 512] [--dt 0.00078125] [--times 0.4,0.8,1.2]\n"
        "       [--nu 1e-4] [--rho 30] [--eps 0.05]\n"
        "       [--control-n 256] [--control-dt-factor 0.5]\n"
        "       [--outdir DIR] [--revision SHA]\n"
        "\n"
        "Unit-square Minion-Brown shear at one fixed dt. Field errors are\n"
        "common-band differences against --reference. --control-n 0 skips\n"
        "the dt/2 repeat. One MPI rank.\n";
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
    } else if (a == "--resolutions") {
      auto v = need("--resolutions");
      if (!v) return false;
      opt.resolutions = parse_ints(*v);
    } else if (a == "--reference") {
      auto v = need("--reference");
      if (!v) return false;
      opt.reference = std::atoi(std::string(*v).c_str());
    } else if (a == "--dt") {
      auto v = need("--dt");
      if (!v) return false;
      opt.dt = std::atof(std::string(*v).c_str());
    } else if (a == "--times") {
      auto v = need("--times");
      if (!v) return false;
      opt.times = parse_doubles(*v);
    } else if (a == "--nu") {
      auto v = need("--nu");
      if (!v) return false;
      opt.nu = std::atof(std::string(*v).c_str());
    } else if (a == "--rho") {
      auto v = need("--rho");
      if (!v) return false;
      opt.rho = std::atof(std::string(*v).c_str());
    } else if (a == "--eps") {
      auto v = need("--eps");
      if (!v) return false;
      opt.eps = std::atof(std::string(*v).c_str());
    } else if (a == "--control-n") {
      auto v = need("--control-n");
      if (!v) return false;
      opt.control_n = std::atoi(std::string(*v).c_str());
    } else if (a == "--control-dt-factor") {
      auto v = need("--control-dt-factor");
      if (!v) return false;
      opt.control_factor = std::atof(std::string(*v).c_str());
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
  if (opt.resolutions.empty() || opt.times.empty() || opt.dt <= 0.0 ||
      opt.nu < 0.0 || opt.rho <= 0.0) {
    return false;
  }
  bool found = false;
  for (int n : opt.resolutions) {
    if (n < 4) return false;
    if (n == opt.reference) found = true;
    if (n > opt.reference) return false;
  }
  return found;
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
            std::cerr << "ns2d_spatial_refine: one rank is required so every "
                         "Fourier mode is owned locally\n";
          }
          return EXIT_FAILURE;
        }

        const auto save_steps = steps_for(opt.times, opt.dt);
        if (rank == 0) {
          std::filesystem::create_directories(opt.outdir);
          std::cout << "spatial ladder reference N=" << opt.reference
                    << " dt=" << std::scientific << std::setprecision(8) << opt.dt
                    << " nu=" << opt.nu << " rho=" << opt.rho << " eps=" << opt.eps
                    << "\n";
        }

        auto ref_samples =
            integrate(opt.reference, opt.dt, opt.nu, opt.rho, opt.eps, save_steps,
                      nullptr, nullptr, true, rank, nproc);
        // Hats are indexed in the 1-rank r2c order of this N. A second,
        // unstepped plane of the same N has that order and is what
        // restrict_hat_by_k uses to name wave numbers.
        auto reference = make_solver(opt.reference, opt.dt, opt.nu, rank, nproc);

        std::vector<std::vector<ns2d::SpectralPlane::Complex>> ref_hats;
        ref_hats.reserve(ref_samples.size());
        bool ok = ref_samples.size() == save_steps.size();
        for (const auto &s : ref_samples) {
          if (s.status != "ok") ok = false;
          ref_hats.push_back(s.hat);
        }

        std::ofstream csv;
        if (rank == 0) {
          csv.open(opt.outdir + "/spatial_errors.csv");
          csv << "kind,N,reference_N,dt,step,time,omega_l2,omega_l2_rel,"
                 "omega_linf,velocity_l2,velocity_l2_rel,ke,enstrophy,"
                 "max_abs_omega,div_linf,div_l2,cfl,mean_omega,"
                 "outer_rms_fraction,status\n";
          std::ofstream meta(opt.outdir + "/run.json");
          meta << "{\n"
               << "  \"issue\": 224,\n"
               << "  \"case\": \"shear\",\n"
               << "  \"box\": \"unit\",\n"
               << "  \"revision\": \"" << opt.revision << "\",\n"
               << "  \"reference_N\": " << opt.reference << ",\n"
               << "  \"dt\": " << std::scientific << std::setprecision(16) << opt.dt
               << ",\n"
               << "  \"nu\": " << opt.nu << ",\n"
               << "  \"rho\": " << opt.rho << ",\n"
               << "  \"eps\": " << opt.eps << ",\n"
               << "  \"comparison\": \"restrict_hat_by_k common band\",\n"
               << "  \"norms\": \"RMS = sqrt(mean(error^2)); relative to the "
                  "restricted reference\",\n"
               << "  \"outer_band\": \"max(|ki|,|kj|) > N/6 over full omega "
                  "RMS\"\n"
               << "}\n";
          for (const auto &s : ref_samples) {
            write_row(csv, "reference", opt.reference, opt.reference, opt.dt, s);
          }
        }

        if (ok) {
          for (int n : opt.resolutions) {
            if (n == opt.reference) continue;
            const auto samples =
                integrate(n, opt.dt, opt.nu, opt.rho, opt.eps, save_steps,
                          &reference.solver->plane(), &ref_hats, false, rank, nproc);
            if (samples.size() != save_steps.size()) ok = false;
            for (const auto &s : samples) {
              if (s.status != "ok") ok = false;
              if (rank == 0) {
                write_row(csv, "spatial", n, opt.reference, opt.dt, s);
              }
            }
          }
        }

        if (ok && opt.control_n > 0 && opt.control_factor > 0.0 &&
            opt.control_factor < 1.0) {
          const long long sub = std::llround(1.0 / opt.control_factor);
          const double dt_fine = opt.dt / static_cast<double>(sub);
          std::vector<long long> fine_steps;
          fine_steps.reserve(save_steps.size());
          for (long long s : save_steps) fine_steps.push_back(s * sub);
          auto base = integrate(opt.control_n, opt.dt, opt.nu, opt.rho, opt.eps,
                                save_steps, nullptr, nullptr, true, rank, nproc);
          if (base.size() != save_steps.size()) ok = false;
          std::vector<std::vector<ns2d::SpectralPlane::Complex>> base_hats;
          bool base_ok = base.size() == save_steps.size();
          for (const auto &s : base) {
            if (s.status != "ok") {
              ok = false;
              base_ok = false;
            }
            base_hats.push_back(s.hat);
          }
          if (base_ok) {
            auto control_plane =
                make_solver(opt.control_n, opt.dt, opt.nu, rank, nproc);
            const auto halved = integrate(
                opt.control_n, dt_fine, opt.nu, opt.rho, opt.eps, fine_steps,
                &control_plane.solver->plane(), &base_hats, false, rank, nproc);
            if (halved.size() != fine_steps.size()) ok = false;
            for (const auto &s : halved) {
              if (s.status != "ok") ok = false;
              if (rank == 0) {
                write_row(csv, "temporal", opt.control_n, opt.control_n, dt_fine, s);
              }
            }
          }
        }

        if (rank == 0) {
          csv.close();
          std::cout << "spatial_refine rows written to " << opt.outdir
                    << "/spatial_errors.csv\n";
          std::cout << "spatial_refine status=" << (ok ? "ok" : "failed") << "\n";
        }
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
      });
}
