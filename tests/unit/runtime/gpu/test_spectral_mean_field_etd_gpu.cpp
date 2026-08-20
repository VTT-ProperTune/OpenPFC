// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OPENPFC_TEST_MEAN_FIELD_ETD_HIP) &&                               \
    !defined(OPENPFC_TEST_MEAN_FIELD_ETD_CUDA)

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

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_mean_field_etd.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <openpfc/runtime/gpu/spectral_mean_field_etd_gpu.hpp>

#if defined(OPENPFC_TEST_MEAN_FIELD_ETD_HIP)
#include <openpfc/runtime/hip/fft_hip.hpp>
using Space = pfc::HIPSpace;
#elif defined(OPENPFC_TEST_MEAN_FIELD_ETD_CUDA)
#include <openpfc/runtime/cuda/fft_cuda.hpp>
using Space = pfc::CUDASpace;
#endif

using Catch::Approx;
using pfc::SimulationState;
using pfc::data::Field;
using pfc::sim::DeviceSpectralMeanFieldETDSystem;
using pfc::sim::MeanFieldETDPhysics;
using pfc::sim::SpectralMeanFieldETDSystem;

namespace {

struct MeanFieldToy {
  pfc::Domain domain{};
  pfc::Box3i box{};
  double c0{0.85};
  double lambda2{0.0968};
  double p3{-0.5};
  double q3{0.1};
  double stab{0.2};

  void declare_fields(SimulationState &state) const {
    pfc::sim::add_declared_field<double>(state, "psi", domain, box, 0);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    return k_laplacian * c0;
  }

  [[nodiscard]] double filter_mf(double k_laplacian) const {
    return std::exp(k_laplacian / lambda2);
  }

  [[nodiscard]] double nonlinearity(double psi, double psi_mf) const {
    return p3 * psi * psi + q3 * psi_mf * psi_mf - stab * psi;
  }
};

void fill_cosine(Field<double> &psi) {
  const auto n = pfc::domain::get_size(psi.domain());
  const auto dx = pfc::domain::get_spacing(psi.domain());
  const double lx = static_cast<double>(n[0]) * dx[0];
  psi.apply([&](double x, double, double) {
    return -0.10 + 0.01 * std::cos(2.0 * pfc::pi * x / lx);
  });
}

double max_abs_diff(const std::vector<double> &a, const double *b,
                    std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

} // namespace

static_assert(MeanFieldETDPhysics<MeanFieldToy>);

TEST_CASE("device SpectralMeanFieldETDSystem matches host within 1e-10",
          "[gpu][mean_field_etd]") {
#if defined(OPENPFC_TEST_MEAN_FIELD_ETD_HIP)
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
  constexpr double dt = 0.01;
  auto domain = pfc::domain::create(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  auto decomp = pfc::decomposition::create(domain, mpi_size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto cpu_fft = pfc::fft::create(decomp);
#if defined(OPENPFC_TEST_MEAN_FIELD_ETD_HIP)
  auto gpu_fft = pfc::fft::create_hip(decomp, rank, MPI_COMM_WORLD);
#else
  auto gpu_fft = pfc::fft::create_cuda(decomp, rank, MPI_COMM_WORLD);
#endif

  MeanFieldToy host_phys{};
  host_phys.domain = domain;
  host_phys.box = cpu_fft.get_inbox_bounds();
  SimulationState host_state;
  host_phys.declare_fields(host_state);
  fill_cosine(host_state.get_field<double>("psi"));
  SpectralMeanFieldETDSystem<MeanFieldToy> host_sys(host_phys, cpu_fft,
                                                    host_state, dt);
  host_sys.step(0.0);

  MeanFieldToy dev_phys{};
  dev_phys.domain = domain;
  dev_phys.box = gpu_fft.get_inbox_bounds();
  SimulationState dev_state;
  Field<double, Space> psi_d(domain, gpu_fft.get_inbox_bounds(), 0);
  Field<double> ic(domain, cpu_fft.get_inbox_bounds(), 0);
  fill_cosine(ic);
  psi_d.with_host_view([&](double *d, std::size_t n) {
    const auto &src = ic.vec();
    REQUIRE(n == src.size());
    std::copy(src.begin(), src.end(), d);
  });
  dev_state.add_field<double, Space>("psi", std::move(psi_d));
  DeviceSpectralMeanFieldETDSystem<MeanFieldToy, Space> dev_sys(
      dev_phys, gpu_fft, dev_state, dt);
  REQUIRE(dev_sys.linear_symbol().size() == gpu_fft.size_outbox());
  REQUIRE(dev_sys.filter_mf().size() == gpu_fft.size_outbox());
  REQUIRE(dev_sys.linear_symbol()[0] == Approx(host_sys.linear_symbol()[0]));
  REQUIRE(dev_sys.filter_mf()[0] == Approx(host_sys.filter_mf()[0]));
  dev_sys.step(0.0);

  const auto &psi_h = host_state.get_field<double>("psi").vec();
  auto &psi_dev = dev_state.get_field<double, Space>("psi");
  psi_dev.with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == psi_h.size());
    REQUIRE(max_abs_diff(psi_h, d, n) < 1e-10);
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
