// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Device `SpectralETDSystem<Physics, CUDASpace|HIPSpace>` vs the host system
 * for the three toy physics shapes (plain, mean-field, moving-frame). The
 * pointwise functors are instantiated for the device in
 * spectral_etd_toys_pointwise.cu / .hip (same target).
 */

#if !defined(OPENPFC_TEST_SPECTRAL_ETD_HIP) && !defined(OPENPFC_TEST_SPECTRAL_ETD_CUDA)

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

#else

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <fixtures/spectral_etd_toys.hpp>
#include <fixtures/swift_hohenberg.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <openpfc/runtime/gpu/spectral_etd_ops_gpu.hpp>

#if defined(OPENPFC_TEST_SPECTRAL_ETD_HIP)
#include <openpfc/runtime/hip/fft_hip.hpp>
using Space = pfc::HIPSpace;
#else
#include <openpfc/runtime/cuda/fft_cuda.hpp>
using Space = pfc::CUDASpace;
#endif

using Catch::Approx;
using pfc::SimulationState;
using pfc::data::Field;
using pfc::sim::SpectralETDSystem;

namespace {

bool device_available() {
#if defined(OPENPFC_TEST_SPECTRAL_ETD_HIP)
  return pfc::gpu::test::is_hip_available();
#else
  return pfc::gpu::test::is_cuda_available();
#endif
}

void fill_cosine(Field<double> &psi, double mean) {
  const auto n = pfc::domain::get_size(psi.domain());
  const auto dx = pfc::domain::get_spacing(psi.domain());
  const double lx = static_cast<double>(n[0]) * dx[0];
  psi.apply([&](double x, double, double) {
    return mean + 0.01 * std::cos(2.0 * pfc::pi * x / lx);
  });
}

double max_abs_diff(const std::vector<double> &a, const double *b, std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

/// Run `steps` ETD steps on host and device for the same physics; return
/// max |psi_host - psi_device| and both systems' free energies.
template <template <class> class PhysicsT>
struct ParityResult {
  double max_diff{};
  double fe_host{};
  double fe_device{};
};

template <template <class> class PhysicsT>
ParityResult<PhysicsT> run_parity(double mean, int steps, double dt) {
  constexpr int N = 8;
  auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  auto decomp = pfc::decomposition::create(domain, mpi_size);
  auto cpu_fft = pfc::fft::create(decomp);
#if defined(OPENPFC_TEST_SPECTRAL_ETD_HIP)
  auto gpu_fft = pfc::fft::create_hip(decomp, rank, MPI_COMM_WORLD);
#else
  auto gpu_fft = pfc::fft::create_cuda(decomp, rank, MPI_COMM_WORLD);
#endif

  PhysicsT<pfc::HostSpace> host_phys{};
  host_phys.domain = domain;
  host_phys.box = cpu_fft.get_inbox_bounds();
  SimulationState host_state;
  host_phys.declare_fields(host_state);
  fill_cosine(host_state.get_field<double>("psi"), mean);
  SpectralETDSystem<PhysicsT<pfc::HostSpace>> host_sys(host_phys, cpu_fft,
                                                       host_state, dt);

  PhysicsT<Space> dev_phys{};
  dev_phys.domain = domain;
  dev_phys.box = gpu_fft.get_inbox_bounds();
  SimulationState dev_state;
  dev_phys.declare_fields(dev_state);
  {
    Field<double> ic(domain, cpu_fft.get_inbox_bounds(), 0);
    fill_cosine(ic, mean);
    auto &psi_d = dev_state.get_field<double, Space>("psi");
    psi_d.with_host_view([&](double *d, std::size_t n) {
      REQUIRE(n == ic.size());
      std::copy(ic.data(), ic.data() + n, d);
    });
  }
  SpectralETDSystem<PhysicsT<Space>, Space> dev_sys(dev_phys, gpu_fft, dev_state,
                                                    dt);
  REQUIRE(dev_sys.linear_symbol().size() == gpu_fft.size_outbox());
  REQUIRE(dev_sys.linear_symbol()[0] == Approx(host_sys.linear_symbol()[0]));

  double t = 0.0;
  for (int s = 0; s < steps; ++s) {
    host_sys.step(t);
    t = dev_sys.step(t);
  }

  ParityResult<PhysicsT> r{};
  const auto &psi_h = host_state.get_field<double>("psi").vec();
  auto &psi_dev = dev_state.get_field<double, Space>("psi");
  psi_dev.with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == psi_h.size());
    r.max_diff = max_abs_diff(psi_h, d, n);
  });
  r.fe_host = host_sys.last_free_energy();
  r.fe_device = dev_sys.last_free_energy();
  return r;
}

} // namespace

TEST_CASE("device SpectralETDSystem (Swift-Hohenberg) matches host within 1e-10",
          "[gpu][spectral_etd]") {
  if (!device_available()) {
    SKIP("GPU not available");
  }
  const auto r = run_parity<pfc::test::SwiftHohenbergT>(0.05, 3, 0.01);
  REQUIRE(r.max_diff < 1e-10);
}

TEST_CASE("device SpectralETDSystem (mean-field) matches host within 1e-10",
          "[gpu][spectral_etd][mean_field]") {
  if (!device_available()) {
    SKIP("GPU not available");
  }
  const auto r = run_parity<pfc::test::MeanFieldToy>(-0.10, 3, 0.01);
  REQUIRE(r.max_diff < 1e-10);
}

TEST_CASE("device SpectralETDSystem (moving-frame) matches host within 1e-10 "
          "and reduces the same free energy",
          "[gpu][spectral_etd][moving_frame]") {
  if (!device_available()) {
    SKIP("GPU not available");
  }
  const auto r = run_parity<pfc::test::MovingFrameToy>(-0.006, 3, 0.01);
  REQUIRE(r.max_diff < 1e-10);
  REQUIRE(r.fe_device == Approx(r.fe_host).margin(1e-10));
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  const int result = Catch::Session().run(argc, argv);
  int mpi_finalized = 0;
  MPI_Finalized(&mpi_finalized);
  if (mpi_finalized == 0) {
    MPI_Finalize();
  }
  return result;
}

#endif
