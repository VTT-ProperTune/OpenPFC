// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file incompressible_device_parity.cpp
 * @brief One host step against one device step on the same HeFFTe pencil.
 *
 * Taylor–Green and one decaying-HIT seed are the gates. `--bench` also
 * times a short Taylor–Green ladder and one profiled step. The profiled
 * step synchronizes between spectral work, FFTs, and the real-space
 * product; the timed steps do not.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL) && !defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
#error "incompressible device parity requires a spectral device backend"
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <openpfc/runtime/gpu/incompressible_device.hpp>

#include <flow/decaying_hit.hpp>
#include <flow/device_session.hpp>
#include <flow/taylor_green.hpp>

namespace {

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
using Space = pfc::HIPSpace;
#else
using Space = pfc::CUDASpace;
#endif

using Clock = std::chrono::steady_clock;

struct Options {
  int n{16};
  bool bench{false};
};

[[nodiscard]] Options parse(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string_view tok(argv[i]);
    if (tok == "--bench") {
      opt.bench = true;
      continue;
    }
    const auto eq = tok.find('=');
    if (!tok.starts_with("--") || eq == std::string_view::npos) continue;
    const auto key = tok.substr(2, eq - 2);
    if (key == "n") opt.n = std::stoi(std::string(tok.substr(eq + 1)));
  }
  return opt;
}

[[nodiscard]] double reduce_max(double x) {
  double out = 0.0;
  MPI_Allreduce(&x, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return out;
}

[[nodiscard]] int reduce_max_int(int x) {
  int out = 0;
  MPI_Allreduce(&x, &out, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return out;
}

void compare_hats(const std::vector<flow::Complex> &host,
                  const std::vector<flow::Complex> &device, double &local) {
  if (host.size() != device.size()) {
    local = 1.0e300;
    return;
  }
  for (std::size_t i = 0; i < host.size(); ++i) {
    local = std::max(local, std::abs(host[i] - device[i]));
  }
}

[[nodiscard]] int one_step(const char *name, int n, double nu, double dt, bool hit,
                           int rank, int nproc, bool profile) {
  auto host = flow::make_state(n, nu, dt, rank, nproc);
  if (hit) {
    flow::initialize_decaying_hit(host, 1);
  } else {
    flow::initialize_taylor_green(host);
  }
  const auto u0 = host.u;
  const auto v0 = host.v;
  const auto w0 = host.w;
  flow::step(host);

  const double h = pfc::two_pi / static_cast<double>(n);
  auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({h, h, h}));
  pfc::sim::stacks::GPUSpectralStack<Space> device(std::move(domain), rank, nproc,
                                                   MPI_COMM_WORLD);
  pfc::incompressible::DeviceVelocity<Space> vel;
  pfc::incompressible::prepare_device_velocity(vel, device.fft(), {n, n, n},
                                               {h, h, h}, nu, dt);
  pfc::incompressible::upload_device_velocity(vel, u0, v0, w0);
  GPU_CHECK(pfc::gpuDeviceSynchronize());
  pfc::incompressible::step_device_velocity(vel, device.fft(), true);
  GPU_CHECK(pfc::gpuDeviceSynchronize());

  std::vector<flow::Complex> du;
  std::vector<flow::Complex> dv;
  std::vector<flow::Complex> dw;
  const auto copy_begin = Clock::now();
  pfc::incompressible::download_device_velocity(vel, du, dv, dw);
  const double copy_s =
      std::chrono::duration<double>(Clock::now() - copy_begin).count();

  double local = 0.0;
  compare_hats(host.u, du, local);
  compare_hats(host.v, dv, local);
  compare_hats(host.w, dw, local);
  const double max_abs = reduce_max(local);
  const auto bytes = pfc::incompressible::device_velocity_bytes(vel);
  unsigned long long bytes_sum = bytes;
  MPI_Allreduce(MPI_IN_PLACE, &bytes_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  const double cells =
      static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);

  int bad = max_abs > 1.0e-8 || !std::isfinite(max_abs);
  bad = reduce_max_int(bad);
  if (rank == 0) {
    std::cout << "flow_device_parity case=" << name << " n=" << n
              << " ranks=" << nproc << " max_abs=" << max_abs
              << " bytes=" << bytes_sum
              << " bytes_per_cell=" << (static_cast<double>(bytes_sum) / cells)
              << " download_s=" << copy_s << "\n";
  }

  if (profile && !bad) {
    constexpr int kTimed = 4;
    GPU_CHECK(pfc::gpuDeviceSynchronize());
    const auto begin = Clock::now();
    for (int step = 0; step < kTimed; ++step) {
      pfc::incompressible::step_device_velocity(vel, device.fft(), true);
    }
    GPU_CHECK(pfc::gpuDeviceSynchronize());
    const double step_s =
        std::chrono::duration<double>(Clock::now() - begin).count() / kTimed;
    pfc::incompressible::StepProfile prof;
    pfc::incompressible::step_device_velocity(vel, device.fft(), true, &prof);
    if (rank == 0) {
      std::cout << "flow_device_bench case=" << name << " n=" << n
                << " ranks=" << nproc << " step_s=" << step_s
                << " spectral_s=" << prof.spectral_s << " fft_s=" << prof.fft_s
                << " nonlinear_s=" << prof.nonlinear_s << "\n";
    }
  }
  return bad;
}

int session_steps(int n, bool hit, int rank, int nproc) {
  auto reference = flow::make_state(n, .02, .00390625, rank, nproc);
  if (hit)
    flow::initialize_decaying_hit(reference, 1);
  else
    flow::initialize_taylor_green(reference);
  auto initial_u = reference.u, initial_v = reference.v, initial_w = reference.w;
  for (int k = 0; k < 8; ++k) flow::step(reference);
  int bad = 0;
  for (auto choice :
       {flow::TransformChoice::Distributed, flow::TransformChoice::Local}) {
    auto actual = flow::make_state(n, .02, .00390625, rank, nproc);
    actual.u = initial_u;
    actual.v = initial_v;
    actual.w = initial_w;
    if (choice == flow::TransformChoice::Local && nproc > 1) {
      bool rejected = false;
      try {
        auto session = flow::start_device_session(actual, rank, nproc, choice);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      bad |= !rejected;
      continue;
    }
    auto session = flow::start_device_session(actual, rank, nproc, choice);
    for (int k = 0; k < 8; ++k) flow::step_device_session(*session);
    flow::finish_device_session(*session, actual);
    double error = 0;
    compare_hats(reference.u, actual.u, error);
    compare_hats(reference.v, actual.v, error);
    compare_hats(reference.w, actual.w, error);
    error = reduce_max(error);
    bad |= !std::isfinite(error) || error > 1e-8;
    if (rank == 0)
      std::cout << "flow_session_parity case=" << (hit ? "hit" : "tgv")
                << " local=" << (choice == flow::TransformChoice::Local)
                << " max_abs=" << error << "\n";
  }
  return reduce_max_int(bad);
}

} // namespace

int main(int argc, char **argv) {
  pfc::runtime::gpu::bind_local_device_before_mpi();
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int rc = 0;
  try {
    const Options opt = parse(argc, argv);
    rc |= one_step("taylor-green", opt.n, 0.05, 0.01, false, rank, nproc, false);
    rc |= one_step("decaying-hit", opt.n, 0.02, 0.01, true, rank, nproc, false);
    rc |= session_steps(opt.n, false, rank, nproc);
    rc |= session_steps(opt.n, true, rank, nproc);
    if (opt.bench) {
      for (int n : {32, 64, 128}) {
        rc |= one_step("taylor-green", n, 0.05, 0.01, false, rank, nproc, true);
      }
    }
  } catch (const std::exception &err) {
    if (rank == 0) std::cerr << "flow_device_parity error: " << err.what() << "\n";
    rc = 1;
  }
  rc = reduce_max_int(rc);
  if (rank == 0 && rc == 0) std::cout << "FLOW_DEVICE_PARITY_PASS\n";
  MPI_Finalize();
  return rc == 0 ? 0 : 1;
}
