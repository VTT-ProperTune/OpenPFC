// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file main.cpp
 * @brief Periodic incompressible flow.
 *
 * Taylor–Green, one decaying homogeneous-isotropic realization, and one
 * constant-power low-mode forcing. The box length is [0, 2π]^3 and is
 * not a flag. Dealiasing stays on. Walls and a 2-D streamfunction case
 * are not this executable.
 */

#include <chrono>
#include <cstdint>
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
#include <tuple>

#include <openpfc/runtime/common/mpi_main.hpp>

#include <flow/decaying_hit.hpp>
#include <flow/forced_hit.hpp>
#include <flow/taylor_green.hpp>
#if defined(OPENPFC_FLOW_DEVICE)
#include <flow/device_session.hpp>
#endif

namespace {

#if defined(OPENPFC_FLOW_DEVICE)
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
constexpr const char *backend_name = "hip-heffte";
constexpr const char *executable_name = "incompressible_flow_hip";
#else
constexpr const char *backend_name = "cuda-heffte";
constexpr const char *executable_name = "incompressible_flow_cuda";
#endif
#endif

struct Options {
  std::string which{"taylor-green"};
  std::optional<int> n;
  std::optional<double> nu;
  std::optional<double> dt;
  std::optional<double> time;
  std::optional<std::uint64_t> seed;
  std::optional<double> power;
  std::optional<double> sample_dt;
  std::optional<double> volume_dt;
  bool profile{false};
  std::string series;
  std::string outdir{"results/incompressible-flow"};
};

void usage(std::ostream &os) {
#if defined(OPENPFC_FLOW_DEVICE)
  os << "Usage: " << executable_name
     << "\n"
        "       --case taylor-green|decaying-hit\n"
        "       [--n N] [--nu NU] [--dt DT] [--time T] [--seed SEED]\n"
        "       [--sample-dt DT] [--volume-dt DT] [--profile] [--outdir DIR]\n"
        "\n"
        "Periodic 3-D Navier-Stokes on [0, 2pi]^3. The step uses the\n"
        "selected GPU HeFFTe backend. Diagnostics copy the hats back\n"
        "at a sample. --sample-dt and --volume-dt are Taylor-Green\n"
        "only. Forced HIT and the locked series stay on\n"
        "incompressible_flow.\n";
#else
  os << "Usage: incompressible_flow\n"
        "       --case taylor-green|decaying-hit|forced-hit\n"
        "       [--n N] [--nu NU] [--dt DT] [--time T] [--seed SEED]\n"
        "       [--power EPS] [--sample-dt DT] [--volume-dt DT]\n"
        "       [--profile] [--series NAME] [--outdir DIR]\n"
        "\n"
        "Periodic 3-D Navier-Stokes on [0, 2pi]^3. N is even and at\n"
        "least 8. time is an integer number of steps. The 2/3 mask\n"
        "stays on. Ranks share the HeFFTe pencil. --series runs a\n"
        "locked one-rank ladder and ignores n, nu, dt, time, seed,\n"
        "and power.\n";
#endif
}

std::optional<int> parse_int(std::string_view text) {
  try {
    std::size_t used = 0;
    const int value = std::stoi(std::string(text), &used);
    if (used != text.size()) return std::nullopt;
    return value;
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

std::optional<double> parse_double(std::string_view text) {
  try {
    std::size_t used = 0;
    const double value = std::stod(std::string(text), &used);
    if (used != text.size()) return std::nullopt;
    return value;
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

std::optional<std::uint64_t> parse_u64(std::string_view text) {
  if (text.empty() || text.front() == '-') return std::nullopt;
  try {
    std::size_t used = 0;
    const unsigned long long value = std::stoull(std::string(text), &used, 10);
    if (used != text.size()) return std::nullopt;
    return static_cast<std::uint64_t>(value);
  } catch (const std::exception &) {
    return std::nullopt;
  }
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
    } else if (flag == "--n") {
      auto value = need("--n");
      if (!value) return false;
      auto parsed = parse_int(*value);
      if (!parsed) return false;
      opt.n = *parsed;
    } else if (flag == "--nu") {
      auto value = need("--nu");
      if (!value) return false;
      auto parsed = parse_double(*value);
      if (!parsed) return false;
      opt.nu = *parsed;
    } else if (flag == "--dt") {
      auto value = need("--dt");
      if (!value) return false;
      auto parsed = parse_double(*value);
      if (!parsed) return false;
      opt.dt = *parsed;
    } else if (flag == "--time") {
      auto value = need("--time");
      if (!value) return false;
      auto parsed = parse_double(*value);
      if (!parsed) return false;
      opt.time = *parsed;
    } else if (flag == "--seed") {
      auto value = need("--seed");
      if (!value) return false;
      auto parsed = parse_u64(*value);
      if (!parsed) return false;
      opt.seed = *parsed;
    } else if (flag == "--power") {
      auto value = need("--power");
      if (!value) return false;
      auto parsed = parse_double(*value);
      if (!parsed || !(*parsed > 0.0)) return false;
      opt.power = *parsed;
    } else if (flag == "--series") {
      auto value = need("--series");
      if (!value) return false;
      opt.series = std::string(*value);
    } else if (flag == "--sample-dt") {
      auto value = need("--sample-dt");
      if (!value) return false;
      auto parsed = parse_double(*value);
      if (!parsed || !(*parsed > 0.0)) return false;
      opt.sample_dt = *parsed;
    } else if (flag == "--profile") {
      opt.profile = true;
    } else if (flag == "--volume-dt") {
      auto value = need("--volume-dt");
      if (!value) return false;
      auto parsed = parse_double(*value);
      if (!parsed || !(*parsed > 0.0)) return false;
      opt.volume_dt = *parsed;
    } else if (flag == "--outdir") {
      auto value = need("--outdir");
      if (!value) return false;
      opt.outdir = std::string(*value);
    } else {
      std::cerr << "unknown argument: " << flag << "\n";
      return false;
    }
  }
  if (opt.which != "taylor-green" && opt.which != "decaying-hit" &&
      opt.which != "forced-hit") {
    std::cerr << "unknown case: " << opt.which << "\n";
    return false;
  }
  if (opt.which == "decaying-hit" && !opt.series.empty() &&
      opt.series != "spatial" && opt.series != "temporal") {
    std::cerr << "decaying-hit series must be spatial or temporal\n";
    return false;
  }
  if (opt.which == "forced-hit" && !opt.series.empty() &&
      opt.series != "stationary") {
    std::cerr << "forced-hit series must be stationary\n";
    return false;
  }
  if (opt.which != "forced-hit" && opt.power) {
    std::cerr << opt.which << " does not take --power\n";
    return false;
  }
  if (opt.which == "taylor-green" && (opt.seed || !opt.series.empty())) {
    std::cerr << "taylor-green does not take --seed or --series\n";
    return false;
  }
  if (opt.which != "taylor-green" && (opt.sample_dt || opt.volume_dt)) {
    std::cerr << opt.which << " does not take --sample-dt or --volume-dt\n";
    return false;
  }
  if (opt.which != "taylor-green" && opt.profile) {
    std::cerr << opt.which << " does not take --profile\n";
    return false;
  }
  if (!opt.series.empty() && (opt.n || opt.nu || opt.dt || opt.time || opt.seed ||
                              opt.power || opt.sample_dt || opt.volume_dt)) {
    std::cerr << "the locked series does not take n, nu, dt, time, seed, or power\n";
    return false;
  }
  return true;
}

void write_taylor_row(std::ostream &os, double time, const flow::Diagnostics &diag,
                      std::string_view status) {
  os << std::scientific << std::setprecision(16) << time << ',' << diag.ke << ','
     << diag.enstrophy << ',' << diag.dissipation << ',' << diag.div_l2 << ','
     << diag.div_linf << ',' << diag.modal_div_max << ',' << diag.w_l2 << ','
     << diag.max_abs_w << ',' << diag.cfl << ',' << status << '\n';
}

long long interval_stride(const std::optional<double> &interval, double dt,
                          long long steps, const char *flag) {
  if (!interval) return 0;
  long long stride = 0;
  try {
    stride = flow::steps_for(*interval, dt);
  } catch (const std::invalid_argument &) {
    throw std::invalid_argument(std::string("incompressible_flow: ") + flag +
                                " must be a positive integer number of steps");
  }
  if (steps % stride != 0) {
    throw std::invalid_argument(std::string("incompressible_flow: ") + flag +
                                " must divide the run");
  }
  return stride;
}

std::string sample_name(long long step) {
  std::ostringstream os;
  os << "s" << std::setw(8) << std::setfill('0') << step << ".bin";
  return os.str();
}

void write_volume_frame(flow::State &state, const std::string &outdir,
                        long long step, double time, int rank) {
  const std::string dir = outdir + "/vorticity";
  if (rank == 0) std::filesystem::create_directories(dir);
  MPI_Barrier(MPI_COMM_WORLD);
  const std::string path = dir + "/" + sample_name(step);
  const auto sample = flow::write_vorticity_magnitude(state, path);
  if (rank == 0) {
    std::ofstream index(dir + "/index.csv", std::ios::app);
    index << std::scientific << std::setprecision(16) << time << ',' << path << ','
          << sample.points << '\n';
    std::cout << "incompressible_flow volume=" << path << " points=" << sample.points
              << " mean_square=" << std::scientific << std::setprecision(8)
              << sample.mean_square << std::endl;
  }
}

int run_taylor(const Options &opt, int rank, int nproc) {
  const int n = opt.n.value_or(32);
  const double nu = opt.nu.value_or(0.05);
  const double dt = opt.dt.value_or(0.01);
  const double time = opt.time.value_or(0.1);
  const long long steps = flow::steps_for(time, dt);
  const long long sample_stride =
      interval_stride(opt.sample_dt, dt, steps, "--sample-dt");
  const long long volume_stride =
      interval_stride(opt.volume_dt, dt, steps, "--volume-dt");
  auto state = flow::make_state(n, nu, dt, rank, nproc);
  flow::initialize_taylor_green(state);
  std::ofstream csv;
  std::ofstream spectrum;
  std::ofstream scales_csv;
  if (rank == 0) {
    std::filesystem::create_directories(opt.outdir);
    csv.open(opt.outdir + "/diagnostics.csv");
    csv << "time,ke,enstrophy,dissipation,div_l2,div_linf,modal_div_max,"
           "w_l2,max_abs_w,cfl,status\n";
    spectrum.open(opt.outdir + "/spectrum.csv");
    spectrum << "time,shell,shell_ke\n";
    scales_csv.open(opt.outdir + "/scales.csv");
    scales_csv << "time,k_max,eta,k_max_eta\n";
    if (volume_stride > 0) {
      std::filesystem::create_directories(opt.outdir + "/vorticity");
      std::ofstream index(opt.outdir + "/vorticity/index.csv");
      index << "time,path,points\n";
    }
  }

  double ke0 = 0.0;
  double last_ke = 0.0;
  double integrated_dissipation = 0.0;
  double previous_time = 0.0;
  double previous_dissipation = 0.0;
  bool have_sample = false;

  auto report = [&](double sample, const flow::Diagnostics &diag,
                    std::string_view status) {
    const auto shells = flow::shell_energies(state);
    const int k_max = flow::retained_k_max(state.n[0]);
    const double eta =
        diag.dissipation > 0.0
            ? std::pow(state.nu * state.nu * state.nu / diag.dissipation, 0.25)
            : 0.0;
    if (have_sample) {
      integrated_dissipation +=
          0.5 * (previous_dissipation + diag.dissipation) * (sample - previous_time);
    } else {
      ke0 = diag.ke;
    }
    last_ke = diag.ke;
    previous_time = sample;
    previous_dissipation = diag.dissipation;
    have_sample = true;
    if (rank == 0) {
      write_taylor_row(csv, sample, diag, status);
      csv.flush();
      spectrum << std::scientific << std::setprecision(16);
      for (const auto &shell : shells) {
        spectrum << sample << ',' << shell.index << ',' << shell.ke << '\n';
      }
      spectrum.flush();
      scales_csv << std::scientific << std::setprecision(16) << sample << ','
                 << k_max << ',' << eta << ',' << static_cast<double>(k_max) * eta
                 << '\n';
      scales_csv.flush();
      std::cout << "incompressible_flow sample t=" << std::scientific
                << std::setprecision(8) << sample << " status=" << status
                << " ke=" << diag.ke << " dissipation=" << diag.dissipation
                << " cfl=" << diag.cfl << std::endl;
    }
  };

  auto stop_for = [](const flow::Diagnostics &diag) -> std::string {
    if (!diag.finite) return "nonfinite";
    if (diag.cfl > flow::cfl_limit) return "cfl";
    return "ok";
  };

  auto initial = flow::diagnose(state);
  std::string status = stop_for(initial);
  report(0.0, initial, status);
  if (status == "ok" && volume_stride > 0) {
    write_volume_frame(state, opt.outdir, 0, 0.0, rank);
  }
  if (status != "ok") {
    if (rank == 0) std::cout << "incompressible_flow status=" << status << std::endl;
    return EXIT_FAILURE;
  }

#if defined(OPENPFC_FLOW_DEVICE)
  auto device = flow::start_device_session(state, rank, nproc);
  flow::DeviceProfile profile;
#else
  double host_step_s = 0.0;
#endif
  for (long long step = 1; step <= steps; ++step) {
#if defined(OPENPFC_FLOW_DEVICE)
    flow::step_device_session(*device, opt.profile ? &profile : nullptr);
#else
    const auto host_began = std::chrono::steady_clock::now();
    flow::step(state);
    if (opt.profile) {
      host_step_s += std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                   host_began)
                         .count();
    }
#endif
    const bool sample = flow::record_at(step, steps, sample_stride);
    const bool volume = volume_stride > 0 && step % volume_stride == 0;
    if (!sample && !volume) continue;
#if defined(OPENPFC_FLOW_DEVICE)
    flow::finish_device_session(*device, state);
#endif
    const double now = static_cast<double>(step) * dt;
    auto diag = flow::diagnose(state);
    status = stop_for(diag);
    report(now, diag, status);
    if (volume && status == "ok") {
      write_volume_frame(state, opt.outdir, step, now, rank);
    }
    if (status != "ok") break;
  }
  if (rank == 0) {
    const double budget =
        have_sample ? (last_ke - ke0) + integrated_dissipation : 0.0;
    std::ofstream meta(opt.outdir + "/metadata.txt");
    meta << std::setprecision(16);
    meta << "case=taylor-green\n"
#if defined(OPENPFC_FLOW_DEVICE)
         << "backend=" << backend_name << "\n"
#else
         << "backend=cpu-heffte\n"
#endif
         << "n=" << n << "\n"
         << "nu=" << std::scientific << nu << "\n"
         << "dt=" << dt << "\n"
         << "time=" << time << "\n"
         << "steps=" << steps << "\n"
         << "k_max=" << flow::retained_k_max(n) << "\n"
         << "ke0=" << ke0 << "\n"
         << "ke_final=" << last_ke << "\n"
         << "integrated_dissipation=" << integrated_dissipation << "\n"
         << "budget=sample-trapezoid\n"
         << "budget_residual=" << budget << "\n";
    if (opt.profile) {
#if defined(OPENPFC_FLOW_DEVICE)
      meta << "profile_steps=" << profile.steps << "\n"
           << "profile_spectral_s=" << profile.spectral_s << "\n"
           << "profile_fft_s=" << profile.fft_s << "\n"
           << "profile_nonlinear_s=" << profile.nonlinear_s << "\n";
#else
      meta << "profile_host_step_s=" << host_step_s << "\n";
#endif
    }
    std::cout << "incompressible_flow status=" << status << std::endl;
  }
  return status == "ok" ? EXIT_SUCCESS : EXIT_FAILURE;
}

void write_hit_row(std::ostream &os, double time, const flow::Diagnostics &diag,
                   const flow::Scales &scales, std::string_view status) {
  os << std::scientific << std::setprecision(16) << time << ',' << diag.ke << ','
     << diag.enstrophy << ',' << scales.omega_rms << ',' << diag.dissipation << ','
     << scales.u_rms << ',' << scales.lambda << ',' << scales.re_lambda << ','
     << scales.integral_scale << ',' << scales.eta << ',' << scales.k_max << ','
     << scales.k_max_eta << ',' << diag.div_l2 << ',' << diag.div_linf << ','
     << diag.modal_div_max << ',' << diag.cfl << ',' << scales.outer_ke_fraction
     << ',' << status << '\n';
}

int run_hit(const Options &opt, int rank, int nproc) {
  const int n = opt.n.value_or(32);
  const double nu = opt.nu.value_or(flow::protocol_nu);
  const double dt = opt.dt.value_or(1.0 / 64.0);
  const double time = opt.time.value_or(0.0625);
  const std::uint64_t seed = opt.seed.value_or(flow::protocol_seed);
  const long long steps = flow::steps_for(time, dt);
  auto state = flow::make_state(n, nu, dt, rank, nproc);
  flow::initialize_decaying_hit(state, seed);
  std::ofstream csv;
  std::ofstream spectrum;
  if (rank == 0) {
    std::filesystem::create_directories(opt.outdir);
    csv.open(opt.outdir + "/diagnostics.csv");
    csv << "time,ke,enstrophy,omega_rms,dissipation,u_rms,lambda,re_lambda,"
           "integral_scale,eta,k_max,k_max_eta,div_l2,div_linf,modal_div_max,cfl,"
           "outer_ke_fraction,status\n";
    spectrum.open(opt.outdir + "/spectrum.csv");
    spectrum << "time,shell,shell_ke\n";
  }

  auto report = [&](double sample, const flow::Diagnostics &diag,
                    const flow::Scales &scales, std::string_view status) {
    const auto shells = flow::shell_energies(state);
    if (rank == 0) {
      write_hit_row(csv, sample, diag, scales, status);
      csv.flush();
      spectrum << std::scientific << std::setprecision(16);
      for (const auto &shell : shells) {
        spectrum << sample << ',' << shell.index << ',' << shell.ke << '\n';
      }
      spectrum.flush();
    }
    if (rank == 0) {
      std::cout << "incompressible_flow sample t=" << std::scientific
                << std::setprecision(8) << sample << " status=" << status
                << " ke=" << diag.ke << " re_lambda=" << scales.re_lambda
                << std::endl;
    }
  };

  auto initial = flow::diagnose(state);
  auto initial_scales = flow::measure_scales(state, initial);
  std::string status = "ok";
  if (!initial.finite)
    status = "nonfinite";
  else if (initial.cfl > flow::cfl_limit)
    status = "cfl";
  report(0.0, initial, initial_scales, status);
  if (status != "ok") {
    if (rank == 0) std::cout << "incompressible_flow status=" << status << std::endl;
    return EXIT_FAILURE;
  }

#if defined(OPENPFC_FLOW_DEVICE)
  auto device = flow::start_device_session(state, rank, nproc);
#endif
  flow::Diagnostics final_diag = initial;
  flow::Scales final_scales = initial_scales;
  for (long long step = 1; step <= steps; ++step) {
#if defined(OPENPFC_FLOW_DEVICE)
    flow::step_device_session(*device);
#else
    flow::step(state);
#endif
    if (step != steps) continue;
#if defined(OPENPFC_FLOW_DEVICE)
    flow::finish_device_session(*device, state);
#endif
    final_diag = flow::diagnose(state);
    final_scales = flow::measure_scales(state, final_diag);
    if (!final_diag.finite)
      status = "nonfinite";
    else if (final_diag.cfl > flow::cfl_limit)
      status = "cfl";
    report(time, final_diag, final_scales, status);
  }
  const double budget = (final_diag.ke - initial.ke) +
                        0.5 * (initial.dissipation + final_diag.dissipation) * time;
  if (rank == 0) {
    std::ofstream meta(opt.outdir + "/metadata.txt");
    meta << std::setprecision(16);
    meta << "case=decaying-hit\n"
#if defined(OPENPFC_FLOW_DEVICE)
         << "backend=" << backend_name << "\n"
#endif
         << "spectrum=yoffe-mccomb-2018-eq10\n"
         << "citation=arXiv:1805.01238\n"
         << "c=" << flow::spectrum_c << "\n"
         << "k0=" << flow::spectrum_k0 << "\n"
         << "seed=" << seed << "\n"
         << "n=" << n << "\n"
         << "nu=" << nu << "\n"
         << "dt=" << std::scientific << dt << "\n"
         << "time=" << time << "\n"
         << "steps=" << steps << "\n"
         << "budget_residual=" << budget << "\n"
         << "assignment=E(|k|)/(4*pi*|k|^2) on 2/3-retained modes\n";
    std::cout << "incompressible_flow status=" << status << std::endl;
  }
  return status == "ok" ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run_forced(const Options &opt, int rank, int nproc) {
  const int n = opt.n.value_or(32);
  const double nu = opt.nu.value_or(flow::forced_nu);
  const double dt = opt.dt.value_or(1.0 / 64.0);
  const double time = opt.time.value_or(0.0625);
  const double power = opt.power.value_or(flow::forced_power);
  const std::uint64_t seed = opt.seed.value_or(flow::protocol_seed);
  const long long steps = flow::steps_for(time, dt);
  auto state = flow::make_state(n, nu, dt, rank, nproc);
  flow::initialize_decaying_hit(state, seed);
  std::ofstream csv;
  if (rank == 0) {
    std::filesystem::create_directories(opt.outdir);
    csv.open(opt.outdir + "/diagnostics.csv");
    csv << "time,ke,enstrophy,dissipation,injection,band_ke,u_rms,re_lambda,"
           "k_max_eta,div_l2,modal_div_max,cfl,status\n";
  }

  auto measure = [&](double sample) {
    const auto outbox = state.stack->fft().get_outbox_bounds();
    std::vector<flow::Complex> fu, fv, fw;
    const auto force =
        flow::write_band_force(outbox, state.n, state.spacing, state.u, state.v,
                               state.w, fu, fv, fw, power);
    const auto diag = flow::diagnose(state);
    const auto scales = flow::measure_scales(state, diag);
    std::string status = "ok";
    if (!diag.finite)
      status = "nonfinite";
    else if (diag.cfl > flow::cfl_limit)
      status = "cfl";
    if (rank == 0) {
      csv << std::scientific << std::setprecision(16) << sample << ',' << diag.ke
          << ',' << diag.enstrophy << ',' << diag.dissipation << ','
          << force.injection << ',' << force.band_ke << ',' << scales.u_rms << ','
          << scales.re_lambda << ',' << scales.k_max_eta << ',' << diag.div_l2 << ','
          << diag.modal_div_max << ',' << diag.cfl << ',' << status << '\n';
      csv.flush();
    }
    if (rank == 0) {
      std::cout << "incompressible_flow sample t=" << std::scientific
                << std::setprecision(8) << sample << " status=" << status
                << " ke=" << diag.ke << " injection=" << force.injection
                << std::endl;
    }
    return std::tuple<std::string, flow::Diagnostics, flow::BandForce>{status, diag,
                                                                       force};
  };

  auto [status, initial, initial_force] = measure(0.0);
  if (status != "ok") {
    if (rank == 0) std::cout << "incompressible_flow status=" << status << std::endl;
    return EXIT_FAILURE;
  }
  flow::Diagnostics final_diag = initial;
  flow::BandForce final_force = initial_force;
  for (long long taken = 1; taken <= steps; ++taken) {
    flow::step_forced(state, power);
    if (taken != steps) continue;
    auto result = measure(time);
    status = std::get<0>(result);
    final_diag = std::get<1>(result);
    final_force = std::get<2>(result);
  }
  const double budget = (final_diag.ke - initial.ke) -
                        0.5 *
                            (initial_force.injection + final_force.injection -
                             initial.dissipation - final_diag.dissipation) *
                            time;
  if (rank == 0) {
    std::ofstream meta(opt.outdir + "/metadata.txt");
    meta << std::setprecision(16);
    meta << "case=forced-hit\n"
         << "forcing=doering-petrov-2004-eq3\n"
         << "citation=arXiv:physics/0404049\n"
         << "band=|k|=1\n"
         << "power=" << power << "\n"
         << "seed=" << seed << "\n"
         << "n=" << n << "\n"
         << "nu=" << nu << "\n"
         << "dt=" << std::scientific << dt << "\n"
         << "time=" << time << "\n"
         << "steps=" << steps << "\n"
         << "budget_residual=" << budget << "\n";
    std::cout << "incompressible_flow status=" << status << std::endl;
  }
  return status == "ok" ? EXIT_SUCCESS : EXIT_FAILURE;
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
        if (!opt.series.empty() && nproc != 1) {
          if (rank == 0) {
            std::cerr << "incompressible_flow: the locked series is the "
                         "one-rank protocol\n";
          }
          return EXIT_FAILURE;
        }
#if defined(OPENPFC_FLOW_DEVICE)
        if (!opt.series.empty() || opt.which == "forced-hit") {
          if (rank == 0) {
            std::cerr << "incompressible_flow: forced HIT and the locked series "
                         "stay on incompressible_flow\n";
          }
          return EXIT_FAILURE;
        }
        if (rank == 0) {
          std::cout << "incompressible_flow backend=" << backend_name << std::endl;
        }
#endif
        try {
          if (!opt.series.empty() && opt.which == "forced-hit")
            return flow::run_forced_series(opt.series, opt.outdir, rank);
          if (!opt.series.empty())
            return flow::run_series(opt.series, opt.outdir, rank);
          if (opt.which == "taylor-green") return run_taylor(opt, rank, nproc);
          if (opt.which == "forced-hit") return run_forced(opt, rank, nproc);
          return run_hit(opt, rank, nproc);
        } catch (const std::exception &ex) {
          std::cerr << ex.what() << std::endl;
          return EXIT_FAILURE;
        }
      });
}
