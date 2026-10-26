// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file model.cpp
 * @brief Locked 3-D Taylor–Green ladder (issue #229).
 *
 * The sample times are 0, 1/2, and 1. Every step below divides both
 * nonzero times in exact binary, so a sample is an integer step.
 * Spatial: N = 32, 64, 128, 256 at dt = 1/256, plus N = 64 at dt/2.
 * Temporal: N = 64 at dt = 1/32, 1/64, 1/128, 1/256, 1/512.
 * Viscosity is 0.05. No Reynolds number is assigned. The comparison
 * reference is the finest series that reaches every sample with a finite
 * field and speed CFL at most 2. A series that stops is kept. Field
 * error is the common-band difference, scaled by Nc³/Nf³. Order is
 * log2 of successive absolute L2 errors and is withheld at or below
 * 1e-11. These choices are the protocol. They are not fitted afterward.
 */

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <openpfc/runtime/common/mpi_main.hpp>

#include "../tg3d/verify.hpp"

namespace {

constexpr double kNu = 0.05;
constexpr double kTimeMid = 0.5;
constexpr double kTimeEnd = 1.0;
constexpr double kDtSpatial = 1.0 / 256.0;
constexpr int kControlN = 64;
constexpr double kOrderFloor = 1.0e-11;
constexpr double kCflLimit = 2.0;
constexpr int kSmokeN = 16;
constexpr double kSmokeDt = 1.0 / 64.0;
constexpr double kSmokeTime = 1.0 / 32.0;

struct Sample {
  long long step{0};
  double time{0.0};
  tg3d::Diagnostics diag{};
  tg3d::FieldError err{};
  bool has_diag{false};
  bool has_error{false};
  std::string status{"ok"};
  std::optional<tg3d::Hats> hats;
  std::string order_velocity;
  std::string order_vorticity;
};

struct Series {
  std::string kind;
  int n{0};
  double dt{0.0};
  int reference_n{-1};
  double reference_dt{0.0};
  std::vector<Sample> samples;
};

struct Saved {
  int n{0};
  double dt{0.0};
  std::vector<tg3d::Hats> hats;
};

struct Options {
  std::string which;
  std::string outdir;
  std::string revision{"unspecified"};
};

std::vector<long long> steps_for(const std::vector<double> &times, double dt) {
  std::vector<long long> steps;
  steps.reserve(times.size());
  for (double time : times) {
    if (time < 0.0) {
      throw std::invalid_argument("taylor_green3d: time must be >= 0");
    }
    const double ratio = time / dt;
    const auto step = std::llround(ratio);
    if (std::abs(ratio - static_cast<double>(step)) > 1.0e-8) {
      throw std::invalid_argument("taylor_green3d: time is not an integer step");
    }
    steps.push_back(step);
  }
  for (std::size_t i = 1; i < steps.size(); ++i) {
    if (steps[i] < steps[i - 1]) {
      throw std::invalid_argument("taylor_green3d: times must be nondecreasing");
    }
  }
  return steps;
}

bool complete(const Series &series, std::size_t ntimes) {
  if (series.samples.size() != ntimes) return false;
  for (const auto &sample : series.samples) {
    if (sample.status != "ok") return false;
  }
  return true;
}

Series integrate(int n, double dt, const std::vector<double> &times, bool keep,
                 const Saved *reference, int rank, int nproc) {
  const auto steps = steps_for(times, dt);
  Series series;
  series.n = n;
  series.dt = dt;
  if (rank == 0) {
    std::cout << "taylor_green3d start N=" << n << " dt=" << std::scientific
              << std::setprecision(16) << dt << std::endl;
  }
  auto flow = tg3d::make_flow(n, kNu, dt, rank, nproc);
  tg3d::initialize_taylor_green(flow);

  std::size_t next = 0;
  const long long last = steps.empty() ? 0 : steps.back();
  auto capture = [&](long long step) {
    Sample sample;
    sample.step = step;
    sample.time = static_cast<double>(step) * dt;
    if (!tg3d::state_finite(flow)) {
      sample.status = "nonfinite";
      series.samples.push_back(std::move(sample));
      return false;
    }
    sample.diag = tg3d::diagnose(flow);
    sample.has_diag = true;
    if (!std::isfinite(sample.diag.ke) || !std::isfinite(sample.diag.enstrophy)) {
      sample.status = "nonfinite";
      series.samples.push_back(std::move(sample));
      return false;
    }
    if (sample.diag.cfl > kCflLimit) sample.status = "cfl";
    if (reference != nullptr && next < reference->hats.size()) {
      sample.err = tg3d::field_error(flow, reference->hats[next]);
      sample.has_error = true;
    }
    if (keep) sample.hats = tg3d::copy_hats(flow);
    if (rank == 0) {
      std::cout << "taylor_green3d sample N=" << n << " t=" << std::scientific
                << std::setprecision(8) << sample.time << " status=" << sample.status
                << " ke=" << sample.diag.ke << " w_l2=" << sample.diag.w_l2
                << std::endl;
    }
    const bool stop = sample.status != "ok";
    series.samples.push_back(std::move(sample));
    return !stop;
  };

  if (!steps.empty() && steps[0] == 0) {
    if (!capture(0)) return series;
    ++next;
  }
  for (long long step = 1; step <= last; ++step) {
    tg3d::step(flow);
    if (next < steps.size() && step == steps[next]) {
      if (!capture(step)) return series;
      ++next;
    }
  }
  return series;
}

Saved save_hats(Series &series) {
  Saved saved;
  saved.n = series.n;
  saved.dt = series.dt;
  saved.hats.reserve(series.samples.size());
  for (auto &sample : series.samples) {
    if (!sample.hats) continue;
    saved.hats.push_back(std::move(*sample.hats));
    sample.hats.reset();
  }
  return saved;
}

void drop_hats(Series &series) {
  for (auto &sample : series.samples) sample.hats.reset();
}

std::optional<double> observed_order(double coarse, double fine) {
  if (!(coarse > kOrderFloor && fine > kOrderFloor)) return std::nullopt;
  return std::log2(coarse / fine);
}

std::string format_order(const std::optional<double> &order) {
  if (!order) return "roundoff";
  std::ostringstream os;
  os << std::scientific << std::setprecision(8) << *order;
  return os.str();
}

void write_meas(std::ostream &os, bool present, double value) {
  if (present) os << std::scientific << std::setprecision(16) << value;
  os << ',';
}

void write_row(std::ostream &os, const Series &series, const Sample &sample,
               bool with_order) {
  os << series.kind << ',' << series.n << ',';
  if (series.reference_n >= 0) os << series.reference_n;
  os << ',' << std::scientific << std::setprecision(16) << series.dt << ',';
  if (series.reference_n >= 0) os << series.reference_dt;
  os << ',' << sample.step << ',' << sample.time << ',';
  write_meas(os, sample.has_error, sample.err.velocity_l2);
  write_meas(os, sample.has_error, sample.err.velocity_l2_rel);
  write_meas(os, sample.has_error, sample.err.vorticity_l2);
  write_meas(os, sample.has_error, sample.err.vorticity_l2_rel);
  write_meas(os, sample.has_error, sample.err.vorticity_linf);
  write_meas(os, sample.has_diag, sample.diag.ke);
  write_meas(os, sample.has_diag, tg3d::linear_shell_energy(kNu, sample.time));
  write_meas(os, sample.has_diag, sample.diag.enstrophy);
  write_meas(os, sample.has_diag, sample.diag.dissipation);
  write_meas(os, sample.has_diag, sample.diag.max_vorticity);
  write_meas(os, sample.has_diag, sample.diag.div_l2);
  write_meas(os, sample.has_diag, sample.diag.div_linf);
  write_meas(os, sample.has_diag, sample.diag.modal_div_max);
  write_meas(os, sample.has_diag, sample.diag.cfl);
  write_meas(os, sample.has_diag, sample.diag.mean_u);
  write_meas(os, sample.has_diag, sample.diag.mean_v);
  write_meas(os, sample.has_diag, sample.diag.mean_w);
  write_meas(os, sample.has_diag, sample.diag.w_l2);
  write_meas(os, sample.has_diag, sample.diag.max_abs_w);
  write_meas(os, sample.has_diag, sample.diag.outer_ke_fraction);
  if (with_order)
    os << sample.order_velocity << ',' << sample.order_vorticity << ',';
  os << sample.status << '\n';
}

void write_spectrum(std::ostream &os, const Series &series, const Sample &sample) {
  if (!sample.has_diag) return;
  for (const auto &shell : sample.diag.spectrum) {
    os << series.kind << ',' << series.n << ',' << std::scientific
       << std::setprecision(16) << series.dt << ',' << sample.time << ',' << shell.k2
       << ',' << shell.ke << '\n';
  }
}

const char *kHeader =
    "kind,N,reference_N,dt,reference_dt,step,time,velocity_l2,velocity_l2_rel,"
    "vorticity_l2,vorticity_l2_rel,vorticity_linf,ke,ke_linear,enstrophy,"
    "dissipation,max_vorticity,div_l2,div_linf,modal_div_max,cfl,mean_u,mean_v,"
    "mean_w,w_l2,max_abs_w,outer_ke_fraction,";

const char *kOrderHeader = "order_velocity_l2,order_vorticity_l2,";

void write_protocol(std::ostream &os, const Options &opt,
                    const std::vector<double> &times, std::string_view extra) {
  os << std::scientific << std::setprecision(16);
  os << "{\n"
     << "  \"issue\": 229,\n"
     << "  \"case\": \"" << opt.which << "\",\n"
     << "  \"ic\": \"u=sin(x)cos(y)cos(z), v=-cos(x)sin(y)cos(z), w=0\",\n"
     << "  \"box\": \"[0, 2pi]^3\",\n"
     << "  \"nu\": " << kNu << ",\n"
     << "  \"viscosity_note\": \"kinematic viscosity; no Reynolds number is "
        "assigned\",\n"
     << "  \"times\": [" << times[0] << ", " << times[1];
  if (times.size() > 2) os << ", " << times[2];
  os << "],\n"
     << "  \"reference_rule\": \"finest series that reaches every sample with a "
        "finite field and speed CFL <= 2\",\n"
     << "  \"comparison\": \"common-band restriction; unnormalized hats scale by "
        "Nc^3/Nf^3\",\n"
     << "  \"norms\": \"RMS = sqrt(mean(|difference|^2)) on the coarse grid\",\n"
     << "  \"outer_band\": \"max(|ki|,|kj|,|kk|) > N/6 as a fraction of kinetic "
        "energy\",\n"
     << "  \"ke_linear\": \"0.125*exp(-6*nu*t); viscous decay of the initial "
        "shell, not a later oracle\",\n"
     << "  \"cfl\": \"dt*max(|u|,|v|,|w|)/dx; above 2 stops that series\",\n"
     << "  \"dissipation\": \"nu * mean(|omega|^2)\",\n"
     << "  \"order_floor\": " << kOrderFloor << ",\n"
     << "  \"revision\": \"" << opt.revision << "\",\n"
     << extra << "}\n";
}

void usage(std::ostream &os) {
  os << "Usage: taylor_green3d --case spatial|temporal|smoke\n"
        "       [--outdir DIR] [--revision SHA]\n"
        "\n"
        "3-D Taylor-Green on [0, 2pi]^3 with nu=0.05. spatial is\n"
        "N=32,64,128,256 at dt=1/256, plus N=64 at dt/2. temporal is\n"
        "N=64 at dt=1/32,1/64,1/128,1/256,1/512. Times are 0, 1/2, 1.\n"
        "smoke is N=16 for two steps and is the regression command.\n"
        "One MPI rank. Numeric knobs are not accepted.\n";
}

bool parse(int argc, char **argv, Options &opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view flag(argv[i]);
    auto need = [&](const char *name) -> std::optional<std::string_view> {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        return std::nullopt;
      }
      return std::string_view(argv[++i]);
    };
    if (flag == "-h" || flag == "--help") {
      usage(std::cout);
      std::exit(EXIT_SUCCESS);
    } else if (flag == "--case") {
      auto value = need("--case");
      if (!value) return false;
      opt.which = std::string(*value);
    } else if (flag == "--outdir") {
      auto value = need("--outdir");
      if (!value) return false;
      opt.outdir = std::string(*value);
    } else if (flag == "--revision") {
      auto value = need("--revision");
      if (!value) return false;
      opt.revision = std::string(*value);
    } else {
      std::cerr << "unknown argument: " << flag << "\n";
      return false;
    }
  }
  if (opt.which != "spatial" && opt.which != "temporal" && opt.which != "smoke") {
    return false;
  }
  if (opt.outdir.empty()) opt.outdir = "results/taylor-green3d-" + opt.which;
  return true;
}

bool run_smoke(const Options &opt, int rank, int nproc) {
  const std::vector<double> times{0.0, kSmokeTime};
  std::filesystem::create_directories(opt.outdir);
  {
    std::ofstream meta(opt.outdir + "/run.json");
    write_protocol(meta, opt, times,
                   "  \"smoke_N\": 16,\n  \"smoke_dt\": 0.015625\n");
  }
  std::ofstream csv(opt.outdir + "/errors.csv");
  std::ofstream spectrum(opt.outdir + "/spectrum.csv");
  csv << kHeader << "status\n";
  spectrum << "kind,N,dt,time,k2,ke\n";
  auto series = integrate(kSmokeN, kSmokeDt, times, false, nullptr, rank, nproc);
  series.kind = "smoke";
  series.reference_n = kSmokeN;
  series.reference_dt = kSmokeDt;
  for (const auto &sample : series.samples) {
    write_row(csv, series, sample, false);
    write_spectrum(spectrum, series, sample);
  }
  csv.flush();
  const bool ok = complete(series, times.size());
  if (rank == 0) {
    std::cout << "taylor_green3d status=" << (ok ? "ok" : "failed") << std::endl;
  }
  return ok;
}

bool run_spatial(const Options &opt, int rank, int nproc) {
  const std::vector<double> times{0.0, kTimeMid, kTimeEnd};
  steps_for(times, kDtSpatial);
  steps_for(times, kDtSpatial / 2.0);
  std::filesystem::create_directories(opt.outdir);
  {
    std::ofstream meta(opt.outdir + "/run.json");
    write_protocol(
        meta, opt, times,
        "  \"dt\": 0.00390625,\n"
        "  \"resolutions\": [32, 64, 128, 256],\n"
        "  \"run_order\": \"finest first, so a stopped grid can fall back\",\n"
        "  \"control\": \"N=64 at dt/2\"\n");
  }
  std::ofstream csv(opt.outdir + "/spatial_errors.csv");
  std::ofstream spectrum(opt.outdir + "/spectrum.csv");
  csv << kHeader << "status\n";
  spectrum << "kind,N,dt,time,k2,ke\n";

  const int descending[] = {256, 128, 64, 32};
  std::optional<Saved> reference;
  std::optional<Saved> control_base;
  bool ok = true;
  for (int n : descending) {
    const bool keep = !reference.has_value() || n == kControlN;
    auto series = integrate(n, kDtSpatial, times, keep,
                            reference ? &*reference : nullptr, rank, nproc);
    if (!reference && complete(series, times.size())) {
      series.kind = "reference";
      series.reference_n = n;
      series.reference_dt = kDtSpatial;
      reference = save_hats(series);
    } else if (!reference) {
      series.kind = "spatial";
      drop_hats(series);
      ok = false;
    } else {
      series.kind = "spatial";
      series.reference_n = reference->n;
      series.reference_dt = reference->dt;
      if (!complete(series, times.size())) ok = false;
    }
    if (n == kControlN && complete(series, times.size())) {
      if (series.kind == "reference") {
        control_base = reference;
      } else {
        control_base = save_hats(series);
      }
    }
    drop_hats(series);
    if (!complete(series, times.size())) ok = false;
    for (const auto &sample : series.samples) {
      write_row(csv, series, sample, false);
      write_spectrum(spectrum, series, sample);
    }
    csv.flush();
    spectrum.flush();
  }

  auto control = integrate(kControlN, kDtSpatial / 2.0, times, false,
                           control_base ? &*control_base : nullptr, rank, nproc);
  control.kind = "control";
  if (control_base) {
    control.reference_n = control_base->n;
    control.reference_dt = control_base->dt;
  }
  if (!complete(control, times.size())) ok = false;
  for (const auto &sample : control.samples) {
    write_row(csv, control, sample, false);
    write_spectrum(spectrum, control, sample);
  }
  csv.flush();
  if (!reference) ok = false;
  if (rank == 0) {
    std::cout << "taylor_green3d status=" << (ok ? "ok" : "failed") << std::endl;
  }
  return ok;
}

bool run_temporal(const Options &opt, int rank, int nproc) {
  const std::vector<double> times{0.0, kTimeMid, kTimeEnd};
  const double dts[] = {1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0, 1.0 / 256.0,
                        1.0 / 512.0};
  for (double dt : dts) steps_for(times, dt);
  std::filesystem::create_directories(opt.outdir);
  {
    std::ofstream meta(opt.outdir + "/run.json");
    write_protocol(
        meta, opt, times,
        "  \"N\": 64,\n"
        "  \"dts\": [0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125],\n"
        "  \"order\": \"log2(e(dt)/e(dt/2)); blank when the finer step is the "
        "reference\"\n");
  }
  std::ofstream csv(opt.outdir + "/temporal_errors.csv");
  std::ofstream spectrum(opt.outdir + "/spectrum.csv");
  csv << kHeader << kOrderHeader << "status\n";
  spectrum << "kind,N,dt,time,k2,ke\n";

  constexpr int ndt = 5;
  std::optional<Saved> reference;
  std::vector<Series> ran(ndt);
  for (int i = ndt - 1; i >= 0; --i) {
    const bool keep = !reference.has_value();
    auto series = integrate(kControlN, dts[i], times, keep,
                            reference ? &*reference : nullptr, rank, nproc);
    if (!reference && complete(series, times.size())) {
      series.kind = "reference";
      series.reference_n = kControlN;
      series.reference_dt = dts[i];
      reference = save_hats(series);
    } else if (reference) {
      series.kind = "temporal";
      series.reference_n = reference->n;
      series.reference_dt = reference->dt;
      drop_hats(series);
    } else {
      series.kind = "temporal";
      drop_hats(series);
    }
    ran[static_cast<std::size_t>(i)] = std::move(series);
  }

  if (reference) {
    for (int i = 0; i + 1 < ndt; ++i) {
      if (dts[i + 1] == reference->dt) continue;
      auto &coarse = ran[static_cast<std::size_t>(i)];
      auto &fine = ran[static_cast<std::size_t>(i + 1)];
      if (!complete(coarse, times.size()) || !complete(fine, times.size())) continue;
      for (std::size_t t = 0; t < times.size(); ++t) {
        if (!coarse.samples[t].has_error || !fine.samples[t].has_error) continue;
        coarse.samples[t].order_velocity = format_order(observed_order(
            coarse.samples[t].err.velocity_l2, fine.samples[t].err.velocity_l2));
        coarse.samples[t].order_vorticity = format_order(observed_order(
            coarse.samples[t].err.vorticity_l2, fine.samples[t].err.vorticity_l2));
      }
    }
  }

  bool ok = reference.has_value();
  for (const auto &series : ran) {
    if (!complete(series, times.size())) ok = false;
    for (const auto &sample : series.samples) {
      write_row(csv, series, sample, true);
      write_spectrum(spectrum, series, sample);
    }
  }
  csv.flush();
  if (rank == 0) {
    std::cout << "taylor_green3d status=" << (ok ? "ok" : "failed") << std::endl;
  }
  return ok;
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
            std::cerr << "taylor_green3d: one rank is required so every Fourier "
                         "mode is owned locally\n";
          }
          return EXIT_FAILURE;
        }
        try {
          const bool ok = (opt.which == "spatial") ? run_spatial(opt, rank, nproc)
                          : (opt.which == "temporal")
                              ? run_temporal(opt, rank, nproc)
                              : run_smoke(opt, rank, nproc);
          return ok ? EXIT_SUCCESS : EXIT_FAILURE;
        } catch (const std::exception &ex) {
          std::cerr << ex.what() << std::endl;
          return EXIT_FAILURE;
        }
      });
}
