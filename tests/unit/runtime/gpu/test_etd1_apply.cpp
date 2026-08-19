// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <span>
#include <vector>

#include <openpfc/kernel/integrator/etd1_apply.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/etd1_apply_gpu.hpp>

using Catch::Approx;
using Complex = std::complex<double>;

namespace {

void host_reference(const std::vector<double> &exp_Ldt,
                    const std::vector<double> &phi1_L,
                    const std::vector<double> &u, const std::vector<double> &nlin,
                    std::vector<double> &out) {
  pfc::integrator::apply_etd1_update(
      std::span<const double>{exp_Ldt}, std::span<const double>{phi1_L},
      std::span<const double>{u}, std::span<const double>{nlin},
      std::span<double>{out});
}

void host_reference(const std::vector<double> &exp_Ldt,
                    const std::vector<double> &phi1_L,
                    const std::vector<Complex> &u,
                    const std::vector<Complex> &nlin, std::vector<Complex> &out) {
  pfc::integrator::apply_etd1_update(
      std::span<const double>{exp_Ldt}, std::span<const double>{phi1_L},
      std::span<const Complex>{u}, std::span<const Complex>{nlin},
      std::span<Complex>{out});
}

} // namespace

#if defined(OPENPFC_TEST_ETD1_APPLY_CUDA)
TEST_CASE("device ETD1 apply CUDA matches host (real and complex)",
          "[gpu][etd1_apply][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  const std::vector<double> exp_Ldt{0.5, 2.0, 1.0};
  const std::vector<double> phi1_L{0.25, -0.1, 0.0};
  const std::vector<double> u{1.0, -2.0, 3.0};
  const std::vector<double> nlin{4.0, 8.0, -1.0};
  std::vector<double> host_out(3);
  host_reference(exp_Ldt, phi1_L, u, nlin, host_out);

  pfc::core::DataBuffer<pfc::backend::CudaTag, double> d_exp(3), d_phi(3), d_u(3),
      d_n(3), d_out(3);
  d_exp.copy_from_host(exp_Ldt);
  d_phi.copy_from_host(phi1_L);
  d_u.copy_from_host(u);
  d_n.copy_from_host(nlin);
  pfc::integrator::apply_etd1_update_cuda(d_u.data(), d_n.data(), d_exp.data(),
                                          d_phi.data(), d_out.data(), 3);
  const auto got = d_out.to_host();
  for (std::size_t i = 0; i < 3; ++i) {
    REQUIRE(got[i] == Approx(host_out[i]));
  }

  const std::vector<Complex> uc{Complex{1.0, 0.5}, Complex{0.0, -1.0},
                                Complex{2.0, 2.0}};
  const std::vector<Complex> nc{Complex{0.2, -0.1}, Complex{1.0, 0.0},
                                Complex{-0.5, 0.25}};
  std::vector<Complex> host_c(3);
  host_reference(exp_Ldt, phi1_L, uc, nc, host_c);
  pfc::core::DataBuffer<pfc::backend::CudaTag, Complex> d_uc(3), d_nc(3),
      d_oc(3);
  d_uc.copy_from_host(uc);
  d_nc.copy_from_host(nc);
  pfc::integrator::apply_etd1_update_cuda(d_uc.data(), d_nc.data(), d_exp.data(),
                                          d_phi.data(), d_oc.data(), 3);
  const auto got_c = d_oc.to_host();
  for (std::size_t i = 0; i < 3; ++i) {
    REQUIRE(got_c[i].real() == Approx(host_c[i].real()));
    REQUIRE(got_c[i].imag() == Approx(host_c[i].imag()));
  }
}
#endif

#if defined(OPENPFC_TEST_ETD1_APPLY_HIP)
TEST_CASE("device ETD1 apply HIP matches host (real and complex)",
          "[gpu][etd1_apply][hip]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  const std::vector<double> exp_Ldt{0.5, 2.0, 1.0};
  const std::vector<double> phi1_L{0.25, -0.1, 0.0};
  const std::vector<double> u{1.0, -2.0, 3.0};
  const std::vector<double> nlin{4.0, 8.0, -1.0};
  std::vector<double> host_out(3);
  host_reference(exp_Ldt, phi1_L, u, nlin, host_out);

  pfc::core::DataBuffer<pfc::backend::HipTag, double> d_exp(3), d_phi(3), d_u(3),
      d_n(3), d_out(3);
  d_exp.copy_from_host(exp_Ldt);
  d_phi.copy_from_host(phi1_L);
  d_u.copy_from_host(u);
  d_n.copy_from_host(nlin);
  pfc::integrator::apply_etd1_update_hip(d_u.data(), d_n.data(), d_exp.data(),
                                         d_phi.data(), d_out.data(), 3);
  const auto got = d_out.to_host();
  for (std::size_t i = 0; i < 3; ++i) {
    REQUIRE(got[i] == Approx(host_out[i]));
  }

  const std::vector<Complex> uc{Complex{1.0, 0.5}, Complex{0.0, -1.0},
                                Complex{2.0, 2.0}};
  const std::vector<Complex> nc{Complex{0.2, -0.1}, Complex{1.0, 0.0},
                                Complex{-0.5, 0.25}};
  std::vector<Complex> host_c(3);
  host_reference(exp_Ldt, phi1_L, uc, nc, host_c);
  pfc::core::DataBuffer<pfc::backend::HipTag, Complex> d_uc(3), d_nc(3), d_oc(3);
  d_uc.copy_from_host(uc);
  d_nc.copy_from_host(nc);
  pfc::integrator::apply_etd1_update_hip(d_uc.data(), d_nc.data(), d_exp.data(),
                                         d_phi.data(), d_oc.data(), 3);
  const auto got_c = d_oc.to_host();
  for (std::size_t i = 0; i < 3; ++i) {
    REQUIRE(got_c[i].real() == Approx(host_c[i].real()));
    REQUIRE(got_c[i].imag() == Approx(host_c[i].imag()));
  }
}
#endif

int main(int argc, char **argv) { return Catch::Session().run(argc, argv); }
