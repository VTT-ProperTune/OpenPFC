// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Device `SpectralFlux` against the host operator on one resolved Fourier
 * mode. `FluxScale` and `AsStored` are instantiated in
 * `spectral_flux_pointwise.cu` / `.hip`.
 */

#if !defined(OPENPFC_TEST_SPECTRAL_FLUX_HIP) &&                                     \
    !defined(OPENPFC_TEST_SPECTRAL_FLUX_CUDA)

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

#else

#include "spectral_flux_mobilities.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/kernel/simulation/spectral_flux.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <openpfc/runtime/gpu/spectral_etd_ops_gpu.hpp>

#if defined(OPENPFC_TEST_SPECTRAL_FLUX_HIP)
#include <openpfc/runtime/hip/fft_hip.hpp>
using Space = pfc::HIPSpace;
#else
#include <openpfc/runtime/cuda/fft_cuda.hpp>
using Space = pfc::CUDASpace;
#endif

using pfc::data::Field;
using pfc::sim::AsStored;
using pfc::sim::SpectralETDOps;
using pfc::sim::SpectralFlux;

namespace {

bool device_available() {
#if defined(OPENPFC_TEST_SPECTRAL_FLUX_HIP)
  return pfc::gpu::test::is_hip_available();
#else
  return pfc::gpu::test::is_cuda_available();
#endif
}

constexpr int N = 16;

pfc::Domain unit_domain() {
  return pfc::domain::create(pfc::GridSize({N, N, 1}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
}

template <class FieldT> void copy_real(const Field<double> &src, FieldT &dst) {
  dst.with_host_view([&](double *data, std::size_t n) {
    REQUIRE(n == src.size());
    std::copy(src.data(), src.data() + n, data);
  });
}

template <class ResultField>
double max_abs_diff(const Field<double> &host, ResultField &device) {
  double m = 0.0;
  const auto &h = host.vec();
  device.with_host_read([&](const double *data, std::size_t n) {
    REQUIRE(n == h.size());
    for (std::size_t i = 0; i < n; ++i) {
      m = std::max(m, std::abs(h[i] - data[i]));
    }
  });
  return m;
}

} // namespace

TEST_CASE("device spectral flux matches host for a constant coefficient",
          "[gpu][spectral_flux]") {
  if (!device_available()) {
    SKIP("GPU not available");
  }
  auto domain = unit_domain();
  int mpi_size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  auto decomp = pfc::decomposition::create(domain, mpi_size);
  auto cpu_fft = pfc::fft::create(decomp);
#if defined(OPENPFC_TEST_SPECTRAL_FLUX_HIP)
  auto gpu_fft = pfc::fft::create_hip(decomp, rank, MPI_COMM_WORLD);
#else
  auto gpu_fft = pfc::fft::create_cuda(decomp, rank, MPI_COMM_WORLD);
#endif

  const double k = 2.0 * pfc::pi / static_cast<double>(N);
  Field<double> potential(domain, cpu_fft.get_inbox_bounds(), 0);
  potential.apply([k](double x, double, double) { return std::cos(k * x); });
  Field<double> state(domain, cpu_fft.get_inbox_bounds(), 0);
  state.apply([](double, double, double) { return 1.0; });

  using HostOps = SpectralETDOps<pfc::HostSpace>;
  Field<std::complex<double>> p_hat(domain, cpu_fft.get_outbox_bounds(), 0);
  Field<std::complex<double>> out_hat(domain, cpu_fft.get_outbox_bounds(), 0);
  HostOps::forward(cpu_fft, potential, p_hat);
  SpectralFlux<> host_flux(domain, cpu_fft);
  host_flux.divergence(p_hat, state, FluxScale{2.5}, out_hat);
  Field<double> host_real(domain, cpu_fft.get_inbox_bounds(), 0);
  HostOps::backward(cpu_fft, out_hat, host_real);

  using DevOps = SpectralETDOps<Space>;
  typename DevOps::RealField pot_d(domain, gpu_fft.get_inbox_bounds(), 0);
  typename DevOps::RealField state_d(domain, gpu_fft.get_inbox_bounds(), 0);
  typename DevOps::ComplexField p_hat_d(domain, gpu_fft.get_outbox_bounds(), 0);
  typename DevOps::ComplexField out_hat_d(domain, gpu_fft.get_outbox_bounds(), 0);
  copy_real(potential, pot_d);
  copy_real(state, state_d);
  DevOps::forward(gpu_fft, pot_d, p_hat_d);
  SpectralFlux<Space> dev_flux(domain, gpu_fft);
  dev_flux.divergence(p_hat_d, state_d, FluxScale{2.5}, out_hat_d);
  typename DevOps::RealField dev_real(domain, gpu_fft.get_inbox_bounds(), 0);
  DevOps::backward(gpu_fft, out_hat_d, dev_real);

  REQUIRE(max_abs_diff(host_real, dev_real) < 1.0e-10);
}

TEST_CASE("device spectral flux matches host for a stored coefficient field",
          "[gpu][spectral_flux]") {
  if (!device_available()) {
    SKIP("GPU not available");
  }
  auto domain = unit_domain();
  int mpi_size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  auto decomp = pfc::decomposition::create(domain, mpi_size);
  auto cpu_fft = pfc::fft::create(decomp);
#if defined(OPENPFC_TEST_SPECTRAL_FLUX_HIP)
  auto gpu_fft = pfc::fft::create_hip(decomp, rank, MPI_COMM_WORLD);
#else
  auto gpu_fft = pfc::fft::create_cuda(decomp, rank, MPI_COMM_WORLD);
#endif

  const double k = 2.0 * pfc::pi / static_cast<double>(N);
  Field<double> potential(domain, cpu_fft.get_inbox_bounds(), 0);
  potential.apply([k](double x, double, double) { return std::cos(k * x); });
  Field<double> coefficient(domain, cpu_fft.get_inbox_bounds(), 0);
  coefficient.apply(
      [k](double x, double, double) { return 1.0 + 0.25 * std::cos(k * x); });

  using HostOps = SpectralETDOps<pfc::HostSpace>;
  Field<std::complex<double>> p_hat(domain, cpu_fft.get_outbox_bounds(), 0);
  Field<std::complex<double>> out_hat(domain, cpu_fft.get_outbox_bounds(), 0);
  HostOps::forward(cpu_fft, potential, p_hat);
  SpectralFlux<> host_flux(domain, cpu_fft);
  host_flux.divergence(p_hat, coefficient, out_hat);
  Field<double> host_real(domain, cpu_fft.get_inbox_bounds(), 0);
  HostOps::backward(cpu_fft, out_hat, host_real);

  using DevOps = SpectralETDOps<Space>;
  typename DevOps::RealField pot_d(domain, gpu_fft.get_inbox_bounds(), 0);
  typename DevOps::RealField coeff_d(domain, gpu_fft.get_inbox_bounds(), 0);
  typename DevOps::ComplexField p_hat_d(domain, gpu_fft.get_outbox_bounds(), 0);
  typename DevOps::ComplexField out_hat_d(domain, gpu_fft.get_outbox_bounds(), 0);
  copy_real(potential, pot_d);
  copy_real(coefficient, coeff_d);
  DevOps::forward(gpu_fft, pot_d, p_hat_d);
  SpectralFlux<Space> dev_flux(domain, gpu_fft);
  dev_flux.divergence(p_hat_d, coeff_d, out_hat_d);
  typename DevOps::RealField dev_real(domain, gpu_fft.get_inbox_bounds(), 0);
  DevOps::backward(gpu_fft, out_hat_d, dev_real);

  REQUIRE(max_abs_diff(host_real, dev_real) < 1.0e-10);
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
