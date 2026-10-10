// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// SingleDeviceFFT (issue #382) against the heFFTe device backend on one rank:
// forward and backward agree to rounding for even, odd and prime lengths;
// the backward input is preserved; GPUSpectralStack exposes it on request.
// Built once per vendor (OPENPFC_TEST_SPACE selects CUDASpace or HIPSpace).

#if !(defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL))

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

#else

#include "test_helpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/runtime/gpu/fft_gpu.hpp>
#include <openpfc/runtime/gpu/fft_single_device.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>

namespace {

#if defined(OPENPFC_TEST_HIP)
using Space = pfc::HIPSpace;
bool available() { return pfc::gpu::test::is_hip_available(); }
auto make_heffte(const pfc::decomposition::Decomposition &d) { return pfc::fft::create_hip(d, 0); }
#else
using Space = pfc::CUDASpace;
bool available() { return pfc::gpu::test::is_cuda_available(); }
auto make_heffte(const pfc::decomposition::Decomposition &d) { return pfc::fft::create_cuda(d, 0); }
#endif

using Real = typename pfc::fft::IDeviceFFT<Space>::RealBuffer;
using Cplx = typename pfc::fft::IDeviceFFT<Space>::ComplexBuffer;

void ensure_mpi() {
  int init = 0;
  MPI_Initialized(&init);
  if (!init) MPI_Init(nullptr, nullptr);
}

double max_rel(const std::vector<std::complex<double>> &a, const std::vector<std::complex<double>> &b) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num = std::max(num, std::abs(a[i] - b[i]));
    den = std::max(den, std::abs(b[i]));
  }
  return num / den;
}

double max_abs(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

void compare(const std::array<int, 3> &n) {
  auto domain = pfc::domain::create(pfc::GridSize({n[0], n[1], n[2]}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto ref = make_heffte(decomp);
  pfc::fft::SingleDeviceFFT<Space> one(n);
  REQUIRE(one.size_inbox() == ref.size_inbox());
  REQUIRE(one.size_outbox() == ref.size_outbox());
  REQUIRE(one.get_inbox_bounds() == ref.get_inbox_bounds());
  REQUIRE(one.get_outbox_bounds() == ref.get_outbox_bounds());

  std::mt19937_64 rng(12345);
  std::normal_distribution<double> nd;
  std::vector<double> h(one.size_inbox());
  for (double &v : h) v = nd(rng);
  Real in(h.size()), out_a(h.size()), out_b(h.size());
  in.copy_from_host(h);
  Cplx ka(one.size_outbox()), kb(one.size_outbox());
  ref.forward(in, ka);
  one.forward(in, kb);
  const auto hka = ka.to_host(), hkb = kb.to_host();
  INFO("n = " << n[0] << " x " << n[1] << " x " << n[2]);
  CHECK(max_rel(hkb, hka) < 1e-13);

  ref.backward(ka, out_a);
  one.backward(kb, out_b);
  const auto ha = out_a.to_host(), hb = out_b.to_host();
  CHECK(max_abs(hb, ha) < 1e-12);
  CHECK(max_abs(hb, h) < 1e-12);                 // round trip
  CHECK(max_rel(kb.to_host(), hkb) == 0.0);      // backward input preserved
}

} // namespace

TEST_CASE("SingleDeviceFFT agrees with the heFFTe backend", "[gpu][fft][single_device]") {
  if (!available()) SKIP("no GPU");
  ensure_mpi();
  for (const auto &n : {std::array<int, 3>{32, 32, 32}, std::array<int, 3>{43, 819, 16},
                        std::array<int, 3>{15, 9, 7}, std::array<int, 3>{13, 17, 11}, std::array<int, 3>{1, 24, 20}})
    compare(n);
}

TEST_CASE("GPUSpectralStack exposes the single-device FFT on request", "[gpu][fft][single_device]") {
  if (!available()) SKIP("no GPU");
  ensure_mpi();
  auto domain = pfc::domain::create(pfc::GridSize({20, 18, 12}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({0.5, 0.5, 0.5}));
  pfc::sim::stacks::GPUSpectralStack<Space> heffte(domain, 0, 1, MPI_COMM_SELF);
  pfc::sim::stacks::GPUSpectralStack<Space> single(domain, 0, 1, MPI_COMM_SELF,
                                                   pfc::sim::stacks::DeviceFFTChoice::single_device);
  CHECK(heffte.fft_choice() == pfc::sim::stacks::DeviceFFTChoice::heffte);
  CHECK(single.fft_choice() == pfc::sim::stacks::DeviceFFTChoice::single_device);
  CHECK(dynamic_cast<pfc::fft::SingleDeviceFFT<Space> *>(&single.fft()) != nullptr);
  CHECK(single.fft().get_outbox_bounds() == heffte.fft().get_outbox_bounds());
}

int main(int argc, char *argv[]) {
  ensure_mpi();
  const int rc = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return rc;
}

#endif
