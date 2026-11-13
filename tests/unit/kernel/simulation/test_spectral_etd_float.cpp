// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_spectral_etd_float.cpp
 * @brief The spectral ETD chain in single precision (ADR 0006).
 *
 * A physics with `real_type = float` declares float fields; the system then
 * runs float FFTs, float coefficient tables and a float candidate while the
 * k-space symbols are still prepared in double. These checks run the same
 * toy physics in both precisions from the same initial field and bound the
 * difference, so the float path is validated against the double reference
 * rather than claimed bitwise.
 */

#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <fixtures/spectral_etd_toys.hpp>
#include <fixtures/swift_hohenberg.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>

using Catch::Approx;
using pfc::SimulationState;
using pfc::sim::SpectralETDSystem;

namespace {

constexpr int N = 8;

using SHFloat = pfc::test::SwiftHohenbergT<pfc::HostSpace, float>;
using SHDouble = pfc::test::SwiftHohenbergT<pfc::HostSpace, double>;
using MeanFieldFloat = pfc::test::MeanFieldToy<pfc::HostSpace, float>;
using MeanFieldDouble = pfc::test::MeanFieldToy<pfc::HostSpace, double>;
using MovingFrameFloat = pfc::test::MovingFrameToy<pfc::HostSpace, float>;
using MovingFrameDouble = pfc::test::MovingFrameToy<pfc::HostSpace, double>;

static_assert(std::is_same_v<pfc::sim::physics_real_t<SHFloat>, float>);
static_assert(std::is_same_v<pfc::sim::physics_real_t<SHDouble>, double>);
static_assert(std::is_same_v<SpectralETDSystem<SHFloat>::Real, float>);
static_assert(
    std::is_same_v<SpectralETDSystem<SHFloat>::Complex, std::complex<float>>);
static_assert(std::is_same_v<SpectralETDSystem<SHDouble>::Real, double>);
static_assert(pfc::sim::SpectralETDPhysics<SHFloat>);
static_assert(pfc::sim::SpectralETDPhysics<MeanFieldFloat>);
static_assert(pfc::sim::SpectralETDPhysics<MovingFrameFloat>);

pfc::Domain unit_domain() {
  return pfc::domain::create(pfc::GridSize({N, N, N}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
}

template <class Physics>
Physics make_physics(const pfc::Domain &domain, const pfc::fft::IHostFFT &fft) {
  Physics p{};
  p.domain = domain;
  p.box = fft.get_inbox_bounds();
  return p;
}

template <class Real> void fill_cosine(pfc::data::Field<Real> &psi, double mean) {
  const auto n = pfc::domain::get_size(psi.domain());
  const auto dx = pfc::domain::get_spacing(psi.domain());
  const double lx = static_cast<double>(n[0]) * dx[0];
  psi.apply([&](double x, double, double) {
    return static_cast<Real>(mean + 0.01 * std::cos(2.0 * pfc::pi * x / lx));
  });
}

struct Parity {
  double max_diff{};
  double fe_float{};
  double fe_double{};
};

/// Same toy, same initial field, `steps` ETD1 steps in float and in double.
template <class PhysF, class PhysD>
Parity run_parity(double mean, int steps, double dt) {
  auto domain = unit_domain();
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp = pfc::decomposition::create(domain, size);
  auto fft = pfc::fft::create(decomp);

  auto phys_d = make_physics<PhysD>(domain, fft);
  SimulationState state_d;
  phys_d.declare_fields(state_d);
  fill_cosine(state_d.template get_field<double>("psi"), mean);
  SpectralETDSystem<PhysD> sys_d(phys_d, fft, state_d, dt);

  auto phys_f = make_physics<PhysF>(domain, fft);
  SimulationState state_f;
  phys_f.declare_fields(state_f);
  fill_cosine(state_f.template get_field<float>("psi"), mean);
  SpectralETDSystem<PhysF> sys_f(phys_f, fft, state_f, dt);

  REQUIRE(sys_f.linear_symbol().size() == sys_d.linear_symbol().size());
  REQUIRE(sys_f.linear_symbol()[0] == Approx(sys_d.linear_symbol()[0]));

  double t_d = 0.0;
  double t_f = 0.0;
  for (int s = 0; s < steps; ++s) {
    t_d = sys_d.step(t_d);
    t_f = sys_f.step(t_f);
  }
  REQUIRE(t_f == Approx(t_d));

  Parity r{};
  const auto &psi_d = state_d.get_field<double>("psi").vec();
  const auto &psi_f = state_f.get_field<float>("psi").vec();
  REQUIRE(psi_f.size() == psi_d.size());
  for (std::size_t i = 0; i < psi_d.size(); ++i) {
    r.max_diff =
        std::max(r.max_diff, std::abs(static_cast<double>(psi_f[i]) - psi_d[i]));
  }
  r.fe_float = sys_f.last_free_energy();
  r.fe_double = sys_d.last_free_energy();
  return r;
}

} // namespace

TEST_CASE("float SpectralETDSystem declares float fields and scratch",
          "[spectral_etd][float]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  auto physics = make_physics<MovingFrameFloat>(domain, fft);
  SimulationState state;
  physics.declare_fields(state);
  SpectralETDSystem<MovingFrameFloat> sys(physics, fft, state, 0.01);

  REQUIRE(state.has_typed_field<float>("psi"));
  REQUIRE(state.has_typed_field<std::complex<float>>("psi_hat"));
  REQUIRE(state.has_typed_field<float>("N"));
  REQUIRE(state.has_typed_field<float>("psi_mf"));
  REQUIRE(state.has_typed_field<float>("P_star_psi"));
  REQUIRE(state.has_typed_field<float>("fe_density"));
  REQUIRE_FALSE(state.has_typed_field<double>("psi"));
  // The k-space tables stay double whatever the field scalar is.
  REQUIRE(sys.linear_symbol().size() == fft.size_outbox());
  REQUIRE(sys.filter_mf().size() == fft.size_outbox());
  REQUIRE(sys.correlation_kernel().size() == fft.size_outbox());
}

TEST_CASE("float SpectralETDSystem zero field stays zero", "[spectral_etd][float]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  auto physics = make_physics<SHFloat>(domain, fft);
  SimulationState state;
  physics.declare_fields(state);
  SpectralETDSystem<SHFloat> sys(physics, fft, state, 0.01);

  auto &psi = state.get_field<float>("psi").vec();
  for (float &v : psi) {
    v = 0.0f;
  }
  const double t1 = sys.step(0.0);
  REQUIRE(t1 == Approx(0.01));
  for (float v : psi) {
    REQUIRE(v == Approx(0.0).margin(1e-7));
  }
}

TEST_CASE("float SpectralETDSystem uniform field matches zero-mode ETD1",
          "[spectral_etd][float]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  auto physics = make_physics<SHFloat>(domain, fft);
  SimulationState state;
  physics.declare_fields(state);
  const double dt = 0.01;
  SpectralETDSystem<SHFloat> sys(physics, fft, state, dt);

  const double u0 = 0.3;
  auto &psi = state.get_field<float>("psi").vec();
  for (float &v : psi) {
    v = static_cast<float>(u0);
  }
  sys.step(0.0);
  // k = 0: L = epsilon - 1, N = -u0^3, u1 = e^{L dt} u0 + phi1(L dt) dt N.
  const double L = physics.linear_symbol(0.0);
  const double expected =
      std::exp(L * dt) * u0 + (std::exp(L * dt) - 1.0) / L * (-u0 * u0 * u0);
  for (float v : psi) {
    REQUIRE(static_cast<double>(v) == Approx(expected).margin(1e-6));
  }
}

TEST_CASE("float Swift-Hohenberg tracks the double reference",
          "[spectral_etd][float]") {
  const auto r = run_parity<SHFloat, SHDouble>(0.05, 10, 0.01);
  REQUIRE(r.max_diff < 1e-5);
}

TEST_CASE("float mean-field toy tracks the double reference",
          "[spectral_etd][float][mean_field]") {
  const auto r = run_parity<MeanFieldFloat, MeanFieldDouble>(-0.10, 10, 0.01);
  REQUIRE(r.max_diff < 1e-5);
}

TEST_CASE("float moving-frame toy tracks the double reference and its "
          "free energy",
          "[spectral_etd][float][moving_frame]") {
  const auto r = run_parity<MovingFrameFloat, MovingFrameDouble>(-0.006, 10, 0.01);
  REQUIRE(r.max_diff < 1e-5);
  REQUIRE(r.fe_float == Approx(r.fe_double).epsilon(1e-5).margin(1e-7));
}

TEST_CASE("float SpectralETDSystem set_dt rebuilds float coefficients",
          "[spectral_etd][float]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  auto physics = make_physics<SHFloat>(domain, fft);
  SimulationState a;
  SimulationState b;
  physics.declare_fields(a);
  physics.declare_fields(b);
  fill_cosine(a.get_field<float>("psi"), 0.05);
  fill_cosine(b.get_field<float>("psi"), 0.05);
  SpectralETDSystem<SHFloat> sys_a(physics, fft, a, 0.01);
  SpectralETDSystem<SHFloat> sys_b(physics, fft, b, 0.02);
  sys_a.set_dt(0.02);
  sys_a.step(0.0);
  sys_b.step(0.0);
  const auto &pa = a.get_field<float>("psi").vec();
  const auto &pb = b.get_field<float>("psi").vec();
  for (std::size_t i = 0; i < pa.size(); ++i) {
    REQUIRE(pa[i] == Approx(pb[i]).margin(1e-7));
  }
}
