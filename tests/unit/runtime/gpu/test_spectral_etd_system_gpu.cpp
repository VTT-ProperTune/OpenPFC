// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OPENPFC_TEST_SPECTRAL_ETD_HIP) &&                                 \
    !defined(OPENPFC_TEST_SPECTRAL_ETD_CUDA)

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

#else

#include "test_helpers.hpp"

#include <algorithm>
#include <vector>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <openpfc/runtime/gpu/spectral_etd_system_gpu.hpp>

#if defined(OPENPFC_TEST_SPECTRAL_ETD_HIP)
#include <openpfc/runtime/hip/fft_hip.hpp>
using Space = pfc::HipSpace;
#elif defined(OPENPFC_TEST_SPECTRAL_ETD_CUDA)
#include <openpfc/runtime/cuda/fft_cuda.hpp>
using Space = pfc::CudaSpace;
#endif

using Catch::Approx;
using pfc::SimulationState;
using pfc::sim::DeviceSpectralEtdSystem;
using pfc::sim::SpectralEtdPhysics;
using pfc::sim::SpectralEtdSystem;

namespace {

struct SwiftHohenberg {
  using parameters_type = double;
  pfc::Domain domain{};
  pfc::Box3i box{};
  double epsilon{0.25};

  void declare_fields(SimulationState &state) const {
    pfc::sim::add_declared_field<double>(state, "psi", domain, box, 0);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    const double one_plus_lap = 1.0 + k_laplacian;
    return epsilon - one_plus_lap * one_plus_lap;
  }

  [[nodiscard]] double nonlinearity(double psi) const {
    return -psi * psi * psi;
  }
};

} // namespace

static_assert(SpectralEtdPhysics<SwiftHohenberg>);

TEST_CASE("device SpectralEtdSystem uniform field matches host within 1e-10",
          "[gpu][spectral_etd]") {
#if defined(OPENPFC_TEST_SPECTRAL_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
#endif

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }

  constexpr int N = 8;
  constexpr double dt = 0.02;
  constexpr double c = 0.5;
  auto domain = pfc::domain::create(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  auto decomp = pfc::decomposition::create(domain, mpi_size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto cpu_fft = pfc::fft::create(decomp);
#if defined(OPENPFC_TEST_SPECTRAL_ETD_HIP)
  auto gpu_fft = pfc::fft::create_hip(decomp, rank, MPI_COMM_WORLD);
#else
  auto gpu_fft = pfc::fft::create_cuda(decomp, rank, MPI_COMM_WORLD);
#endif

  SwiftHohenberg host_phys{};
  host_phys.domain = domain;
  host_phys.box = cpu_fft.get_inbox_bounds();
  SimulationState host_state;
  host_phys.declare_fields(host_state);
  SpectralEtdSystem<SwiftHohenberg> host_sys(host_phys, cpu_fft, host_state,
                                             dt);
  auto &psi_h = host_state.get_field<double>("psi").vec();
  std::fill(psi_h.begin(), psi_h.end(), c);
  host_sys.step(0.0);

  SwiftHohenberg dev_phys{};
  dev_phys.domain = domain;
  dev_phys.box = gpu_fft.get_inbox_bounds();
  SimulationState dev_state;
  pfc::data::Field<double, Space> psi_d(domain, gpu_fft.get_inbox_bounds(), 0);
  psi_d.with_host_view([&](double *d, std::size_t n) {
    std::fill(d, d + n, c);
  });
  dev_state.add_field<double, Space>("psi", std::move(psi_d));
  DeviceSpectralEtdSystem<SwiftHohenberg, Space> dev_sys(dev_phys, gpu_fft,
                                                         dev_state, dt);
  REQUIRE(dev_sys.linear_symbol().size() == gpu_fft.size_outbox());
  REQUIRE(dev_sys.linear_symbol()[0] ==
          Approx(host_sys.linear_symbol()[0]));
  dev_sys.step(0.0);

  const double L0 = host_phys.linear_symbol(0.0);
  const double N0 = host_phys.nonlinearity(c);
  const auto coeff = pfc::integrator::spectral_exp_coeffs(L0, dt);
  const double expected = coeff.exp_Ldt * c + coeff.phi1_L * N0;

  auto &psi_dev = dev_state.get_field<double, Space>("psi");
  psi_dev.with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == psi_h.size());
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(d[i] == Approx(psi_h[i]).margin(1e-10));
      REQUIRE(d[i] == Approx(expected).margin(1e-10));
    }
  });
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  const int result = Catch::Session().run(argc, argv);
  return result;
}

#endif
