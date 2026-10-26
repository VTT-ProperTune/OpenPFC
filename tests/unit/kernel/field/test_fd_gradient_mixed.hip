// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "../../../fixtures/session.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <hip/hip_runtime.h>
#include <iostream>
#include <numbers>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <openpfc/runtime/hip/fd_gradient_device.hpp>
#include <vector>

namespace {
struct Mixed {
  double xy{}, xz{}, yz{};
};
__global__ void evaluate(pfc::gpu::FDGradientDevicePOD e, Mixed *out, int nx, int ny,
                         int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < nx * ny * nz)
    out[c] =
        pfc::gpu::evaluate_fd_grad<Mixed>(e, c % nx, (c / nx) % ny, c / (nx * ny));
}
} // namespace

TEST_CASE("Native HIP mixed FD matches host Full-halo evaluation",
          "[fd_mixed][hip][MPI]") {
  int rank = 0, size = 0, devices = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  REQUIRE((size == 1 || size == 2 || size == 4 || size == 8));
  REQUIRE(hipGetDeviceCount(&devices) == hipSuccess);
  REQUIRE(devices > 0); // Hardware absence is an error, never a silent skip.
  REQUIRE(hipSetDevice(rank % devices) == hipSuccess);
  hipDeviceProp_t props{};
  REQUIRE(hipGetDeviceProperties(&props, rank % devices) == hipSuccess);
  std::cout << "mixed_fd native HIP rank=" << rank << " device=" << props.name
            << " visible=" << devices << " selected=" << rank % devices
            << " pci_bus=" << props.pciBusID << " pci_device=" << props.pciDeviceID
            << '\n';
  constexpr double pi = std::numbers::pi;
  const auto domain = pfc::domain::with_spacing(
      {24, 32, 40}, {2 * pi / 24, 3 * pi / 32, 5 * pi / 40});
  const auto dec = pfc::decomposition::create(
      domain, pfc::Int3{size >= 2 ? 2 : 1, size >= 4 ? 2 : 1, size >= 8 ? 2 : 1});
  const auto box = pfc::decomposition::local_box(dec, rank);
  const auto h = domain.spacing;
  const auto n = box.size;
  for (int order : {2, 4, 6}) {
    pfc::Field<double> host(domain, box, 3);
    pfc::Field<double, pfc::HIPSpace> device(domain, box, 3);
    host.for_each_owned([&](int i, int j, int k) {
      host(i, j, k) = std::sin((box.low[0] + i) * h[0]) *
                      std::cos(2 * (box.low[1] + j) * h[1] / 3) *
                      std::sin(2 * (box.low[2] + k) * h[2] / 5);
    });
    device.with_host_view([&](double *storage, size_t) {
      std::copy(host.vec().begin(), host.vec().end(), storage);
    });
    device.sync_to_device();
    pfc::comm::HaloExchangeOptions opt;
    opt.connectivity = pfc::comm::HaloConnectivity::Full;
    pfc::comm::HaloExchange<pfc::HostSpace, double> hh(host, dec, rank,
                                                       MPI_COMM_WORLD, opt);
    pfc::comm::HaloExchange<pfc::HIPSpace, double> dh(device, dec, rank,
                                                      MPI_COMM_WORLD, opt);
    hh.exchange();
    dh.exchange();
    const auto hg = pfc::field::create<Mixed>(host, order, opt.connectivity);
    const auto dg = pfc::hip::create<Mixed>(device, order);
    const size_t count = static_cast<size_t>(n[0]) * n[1] * n[2];
    Mixed *out = nullptr;
    REQUIRE(hipMalloc(&out, count * sizeof(Mixed)) == hipSuccess);
    hipLaunchKernelGGL(evaluate, dim3((count + 255) / 256), dim3(256), 0, 0,
                       dg.pod(), out, n[0], n[1], n[2]);
    REQUIRE(hipGetLastError() == hipSuccess);
    REQUIRE(hipDeviceSynchronize() == hipSuccess);
    std::vector<Mixed> got(count);
    REQUIRE(hipMemcpy(got.data(), out, count * sizeof(Mixed),
                      hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipFree(out) == hipSuccess);
    double error = 0;
    host.for_each_owned([&](int i, int j, int k) {
      const auto a = hg(i, j, k), b = got[i + n[0] * (j + n[1] * k)];
      REQUIRE(std::isfinite(b.xy));
      REQUIRE(std::isfinite(b.xz));
      REQUIRE(std::isfinite(b.yz));
      error = std::max({error, std::abs(a.xy - b.xy), std::abs(a.xz - b.xz),
                        std::abs(a.yz - b.yz)});
    });
    double maxerror = 0;
    MPI_Allreduce(&error, &maxerror, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
      std::cout << "mixed_fd HIP parity ranks=" << size << " order=" << order
                << " max_error=" << maxerror << '\n';
    REQUIRE(maxerror < 2e-12);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  const int status = pfc::test::run(argc, argv);
  MPI_Finalize();
  return status;
}
