// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Explicit-stream SpectralETDSystem::step_n replays one captured step.
 * Plain stepping on that stream matches the legacy stream bitwise, and
 * graph replay matches plain stepping bitwise. Time-dependent coefficients
 * are refused before capture.
 */

#include "../../../fixtures/session.hpp"

#if !defined(OPENPFC_TEST_SPECTRAL_GRAPH_HIP) &&                                    \
    !defined(OPENPFC_TEST_SPECTRAL_GRAPH_CUDA)

int main(int argc, char *argv[]) { return pfc::test::run(argc, argv); }

#else

#include "test_helpers.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <fixtures/spectral_etd_toys.hpp>
#include <fixtures/swift_hohenberg.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/runtime/gpu/fft_plan_defaults.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <openpfc/runtime/gpu/spectral_etd_ops_gpu.hpp>

#if defined(OPENPFC_TEST_SPECTRAL_GRAPH_HIP)
#include <openpfc/runtime/hip/fft_hip.hpp>
using Space = pfc::HIPSpace;
#else
#include <openpfc/runtime/cuda/fft_cuda.hpp>
using Space = pfc::CUDASpace;
#endif

using pfc::SimulationState;
using pfc::data::Field;
using pfc::sim::SpectralETDOptions;
using pfc::sim::SpectralETDSystem;
using pfc::test::MovingFrameToy;
using pfc::test::SwiftHohenbergT;

namespace {

bool device_available() {
#if defined(OPENPFC_TEST_SPECTRAL_GRAPH_HIP)
  return pfc::gpu::test::is_hip_available();
#else
  return pfc::gpu::test::is_cuda_available();
#endif
}

std::string device_name() {
#if defined(OPENPFC_TEST_SPECTRAL_GRAPH_HIP)
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    return "unknown";
  }
  return prop.name;
#else
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    return "unknown";
  }
  return prop.name;
#endif
}

struct World {
  pfc::Domain domain;
  pfc::decomposition::Decomposition decomp;
  int rank;
};

World make_world() {
  constexpr int n = 8;
  pfc::Domain domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                           pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                           pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  auto decomp = pfc::decomposition::create(domain, mpi_size);
  return World{std::move(domain), std::move(decomp), rank};
}

void fill_cosine(Field<double> &psi) {
  const auto n = pfc::domain::get_size(psi.domain());
  const auto dx = pfc::domain::get_spacing(psi.domain());
  const double lx = static_cast<double>(n[0]) * dx[0];
  psi.apply([&](double x, double, double) {
    return 0.05 + 0.01 * std::cos(2.0 * pfc::pi * x / lx);
  });
}

std::vector<double> initial_state(const World &w, const pfc::Box3i &box) {
  Field<double> ic(w.domain, box, 0);
  fill_cosine(ic);
  return std::vector<double>(ic.data(), ic.data() + ic.size());
}

template <class FFT>
SpectralETDSystem<SwiftHohenbergT<Space>, Space>
make_system(const World &w, FFT &fft, SimulationState &state,
            const std::vector<double> &ic, SpectralETDOptions opt = {}) {
  SwiftHohenbergT<Space> physics{};
  physics.domain = w.domain;
  physics.box = fft.get_inbox_bounds();
  physics.declare_fields(state);
  auto &psi = state.template get_field<double, Space>("psi");
  psi.with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == ic.size());
    std::copy(ic.begin(), ic.end(), d);
  });
  return SpectralETDSystem<SwiftHohenbergT<Space>, Space>(physics, fft, state, 0.05,
                                                          std::move(opt));
}

std::vector<double> download(Field<double, Space> &psi) {
  std::vector<double> out;
  psi.with_host_view([&](double *d, std::size_t n) { out.assign(d, d + n); });
  return out;
}

pfc::gpuStream_t make_stream() {
  pfc::gpuStream_t stream{};
  GPU_CHECK(pfc::gpuStreamCreate(&stream));
  return stream;
}

} // namespace

TEST_CASE("graph replay matches plain spectral ETD stepping bitwise",
          "[gpu][spectral_etd][graph]") {
  if (!device_available()) {
    SKIP("GPU not available");
  }
  REQUIRE(MPI_COMM_WORLD != MPI_COMM_NULL);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    SKIP("one rank");
  }

  const World world = make_world();
#if defined(OPENPFC_TEST_SPECTRAL_GRAPH_HIP)
  auto legacy_fft = pfc::fft::create_hip(world.decomp, world.rank, MPI_COMM_WORLD);
  const auto options =
      pfc::fft::default_device_plan_options<heffte::backend::rocfft>();
#else
  auto legacy_fft = pfc::fft::create_cuda(world.decomp, world.rank, MPI_COMM_WORLD);
  const auto options =
      pfc::fft::default_device_plan_options<heffte::backend::cufft>();
#endif
  const std::vector<double> ic = initial_state(world, legacy_fft.get_inbox_bounds());

  SimulationState legacy_state;
  auto legacy = make_system(world, legacy_fft, legacy_state, ic);
  double t = 0.0;
  constexpr int steps = 4;
  for (int s = 0; s < steps; ++s) {
    t = legacy.step(t);
  }
  const std::vector<double> legacy_psi =
      download(legacy_state.template get_field<double, Space>("psi"));

  pfc::gpuStream_t stream = make_stream();
  {
#if defined(OPENPFC_TEST_SPECTRAL_GRAPH_HIP)
    auto plain_fft = pfc::fft::create_hip(world.decomp, world.rank, MPI_COMM_WORLD,
                                          0, options, stream);
    auto graph_fft = pfc::fft::create_hip(world.decomp, world.rank, MPI_COMM_WORLD,
                                          0, options, stream);
#else
    auto plain_fft = pfc::fft::create_cuda(world.decomp, world.rank, MPI_COMM_WORLD,
                                           0, options, stream);
    auto graph_fft = pfc::fft::create_cuda(world.decomp, world.rank, MPI_COMM_WORLD,
                                           0, options, stream);
#endif

    SimulationState plain_state;
    auto plain = make_system(world, plain_fft, plain_state, ic);
    plain.set_stream(stream);
    t = 0.0;
    for (int s = 0; s < steps; ++s) {
      t = plain.step(t);
    }
    const std::vector<double> plain_psi =
        download(plain_state.template get_field<double, Space>("psi"));
    REQUIRE(plain_psi.size() == legacy_psi.size());
    REQUIRE(std::memcmp(plain_psi.data(), legacy_psi.data(),
                        plain_psi.size() * sizeof(double)) == 0);

    SimulationState graph_state;
    auto graph = make_system(world, graph_fft, graph_state, ic);
    graph.set_stream(stream);
    const double t_end = graph.step_n(0.0, steps);
    REQUIRE(t_end == t);
    const std::vector<double> graph_psi =
        download(graph_state.template get_field<double, Space>("psi"));
    REQUIRE(graph_psi.size() == plain_psi.size());
    REQUIRE(std::memcmp(graph_psi.data(), plain_psi.data(),
                        graph_psi.size() * sizeof(double)) == 0);

    REQUIRE(graph.captured_op_launches() >= 5);
    REQUIRE(graph.graph_kernel_nodes() >= 2);
    REQUIRE(graph.graph_node_count() >=
            static_cast<std::size_t>(graph.captured_op_launches()));
    std::cout << "GRAPH_STEP nodes=" << graph.graph_node_count()
              << " kernels=" << graph.graph_kernel_nodes()
              << " ops=" << graph.captured_op_launches()
              << " mismatches=0 device=" << device_name() << '\n';
  }
  GPU_CHECK(pfc::gpuStreamDestroy(stream));
}

TEST_CASE("step_n refuses time-dependent coefficients and the legacy stream",
          "[gpu][spectral_etd][graph]") {
  if (!device_available()) {
    SKIP("GPU not available");
  }
  const World world = make_world();
#if defined(OPENPFC_TEST_SPECTRAL_GRAPH_HIP)
  auto fft = pfc::fft::create_hip(world.decomp, world.rank, MPI_COMM_WORLD);
#else
  auto fft = pfc::fft::create_cuda(world.decomp, world.rank, MPI_COMM_WORLD);
#endif
  const std::vector<double> ic = initial_state(world, fft.get_inbox_bounds());

  SpectralETDOptions timed;
  timed.time_dependent_coefficients = true;
  SimulationState timed_state;
  auto timed_sys = make_system(world, fft, timed_state, ic, timed);
  REQUIRE_THROWS_AS(timed_sys.step_n(0.0, 2), std::invalid_argument);

  MovingFrameToy<Space> moving{};
  moving.domain = world.domain;
  moving.box = fft.get_inbox_bounds();
  SimulationState moving_state;
  moving.declare_fields(moving_state);
  auto &psi = moving_state.template get_field<double, Space>("psi");
  psi.with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == ic.size());
    std::copy(ic.begin(), ic.end(), d);
  });
  SpectralETDSystem<MovingFrameToy<Space>, Space> moving_sys(moving, fft,
                                                             moving_state, 0.05);
  REQUIRE_THROWS_AS(moving_sys.step_n(0.0, 2), std::invalid_argument);

  SimulationState legacy_state;
  auto legacy = make_system(world, fft, legacy_state, ic);
  REQUIRE_THROWS_AS(legacy.step_n(0.0, 2), std::invalid_argument);
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  const int result = pfc::test::run(argc, argv);
  int mpi_finalized = 0;
  MPI_Finalized(&mpi_finalized);
  if (mpi_finalized == 0) {
    MPI_Finalize();
  }
  return result;
}

#endif
