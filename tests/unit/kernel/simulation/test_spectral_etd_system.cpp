// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Host `SpectralETDSystem` over the three toy physics shapes (plain,
 * mean-field, moving-frame). The device twin is
 * tests/unit/runtime/gpu/test_spectral_etd_system_gpu.cpp.
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
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>

using Catch::Approx;
using pfc::SimulationState;
using pfc::sim::SpectralETDOptions;
using pfc::sim::SpectralETDSystem;
using pfc::test::MeanFieldToy;
using pfc::test::MovingFrameToy;
using pfc::test::SwiftHohenberg;

namespace {

constexpr int N = 8;

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

void fill_cosine(pfc::data::Field<double> &psi, double mean) {
  const auto n = pfc::domain::get_size(psi.domain());
  const auto dx = pfc::domain::get_spacing(psi.domain());
  const double lx = static_cast<double>(n[0]) * dx[0];
  psi.apply([&](double x, double, double) {
    return mean + 0.01 * std::cos(2.0 * pfc::pi * x / lx);
  });
}

} // namespace

TEST_CASE("SpectralETDSystem zero field stays zero", "[spectral_etd][unit]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  auto physics = make_physics<SwiftHohenberg>(domain, fft);
  SimulationState state;
  physics.declare_fields(state);
  SpectralETDSystem<SwiftHohenberg> sys(physics, fft, state, 0.01);

  REQUIRE(sys.linear_symbol().size() == fft.size_outbox());
  REQUIRE(sys.linear_symbol()[0] == Approx(physics.linear_symbol(0.0)));
  REQUIRE(sys.filter_mf().empty());
  REQUIRE(sys.correlation_kernel().empty());
  REQUIRE_FALSE(state.has_field("psi_mf"));
  REQUIRE_FALSE(state.has_field("fe_density"));

  auto &psi = state.get_field<double>("psi").vec();
  for (double &v : psi) {
    v = 0.0;
  }
  const double t1 = sys.step(0.0);
  REQUIRE(t1 == Approx(0.01));
  for (double v : psi) {
    REQUIRE(v == Approx(0.0).margin(1e-14));
  }
}

TEST_CASE("SpectralETDSystem uniform field matches zero-mode ETD1",
          "[spectral_etd][unit]") {
  constexpr double dt = 0.02;
  constexpr double c = 0.5;
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  auto physics = make_physics<SwiftHohenberg>(domain, fft);
  physics.params.epsilon = 0.25;
  SimulationState state;
  physics.declare_fields(state);
  SpectralETDSystem<SwiftHohenberg> sys(physics, fft, state, dt);

  auto &psi = state.get_field<double>("psi").vec();
  for (double &v : psi) {
    v = c;
  }
  sys.step(0.0);

  const double L0 = physics.linear_symbol(0.0);
  const double N0 = physics.nonlinearity(c);
  const auto coeff = pfc::integrator::spectral_exp_coeffs(L0, dt);
  const double expected = coeff.exp_Ldt * c + coeff.phi1_L * N0;
  for (double v : psi) {
    REQUIRE(v == Approx(expected).margin(1e-10));
  }
}

TEST_CASE("SpectralETDSystem attempt does not modify psi; reject is free; "
          "commit equals step",
          "[spectral_etd][unit][protocol]") {
  constexpr double dt = 0.01;
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  auto physics = make_physics<SwiftHohenberg>(domain, fft);

  SimulationState a;
  physics.declare_fields(a);
  fill_cosine(a.get_field<double>("psi"), 0.05);
  SpectralETDSystem<SwiftHohenberg> sys_a(physics, fft, a, dt);

  SimulationState b;
  physics.declare_fields(b);
  fill_cosine(b.get_field<double>("psi"), 0.05);
  SpectralETDSystem<SwiftHohenberg> sys_b(physics, fft, b, dt);

  const std::vector<double> before = a.get_field<double>("psi").vec();
  const auto att = sys_a.attempt(0.0);
  REQUIRE(att.success);
  REQUIRE(att.t1 == Approx(dt));
  REQUIRE(a.get_field<double>("psi").vec() == before);
  sys_a.reject();
  REQUIRE(a.get_field<double>("psi").vec() == before);

  (void)sys_a.attempt(0.0);
  sys_a.commit();
  sys_b.step(0.0);
  REQUIRE(a.get_field<double>("psi").vec() == b.get_field<double>("psi").vec());
}

TEST_CASE("SpectralETDSystem set_dt rebuilds coefficients",
          "[spectral_etd][unit][protocol]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  auto physics = make_physics<SwiftHohenberg>(domain, fft);

  SimulationState a;
  physics.declare_fields(a);
  fill_cosine(a.get_field<double>("psi"), 0.05);
  SpectralETDSystem<SwiftHohenberg> sys_a(physics, fft, a, 0.01);
  sys_a.set_dt(0.02);
  REQUIRE(sys_a.dt() == Approx(0.02));
  sys_a.step(0.0);

  SimulationState b;
  physics.declare_fields(b);
  fill_cosine(b.get_field<double>("psi"), 0.05);
  SpectralETDSystem<SwiftHohenberg> sys_b(physics, fft, b, 0.02);
  sys_b.step(0.0);

  REQUIRE(a.get_field<double>("psi").vec() == b.get_field<double>("psi").vec());
}

TEST_CASE("SpectralETDSystem optional dealias mask does not throw",
          "[spectral_etd][unit][dealias]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  auto physics = make_physics<SwiftHohenberg>(domain, fft);
  SimulationState state;
  physics.declare_fields(state);
  SpectralETDOptions opt{};
  opt.dealias = true;
  SpectralETDSystem<SwiftHohenberg> sys(std::move(physics), fft, state, 0.01, opt);
  auto &psi = state.get_field<double>("psi").vec();
  psi[0] = 0.1;
  REQUIRE_NOTHROW(sys.step(0.0));
}

TEST_CASE("SpectralETDSystem mean-field physics allocates filter fields and "
          "uses k_lap * phi1 as the N weight",
          "[spectral_etd][unit][mean_field]") {
  constexpr double dt = 0.01;
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  auto physics = make_physics<MeanFieldToy<>>(domain, fft);
  SimulationState state;
  physics.declare_fields(state);
  fill_cosine(state.get_field<double>("psi"), -0.10);
  SpectralETDSystem<MeanFieldToy<>> sys(physics, fft, state, dt);

  REQUIRE(state.has_field("psi_mf"));
  REQUIRE(state.has_field("psi_mf_hat"));
  REQUIRE_FALSE(state.has_field("P_star_psi"));
  REQUIRE_FALSE(state.has_field("fe_density"));
  REQUIRE(sys.filter_mf().size() == fft.size_outbox());
  REQUIRE(sys.filter_mf()[0] == Approx(physics.filter_mf(0.0)));

  // Zero mode: L = 0 and M = k_lap = 0, so the mean is exactly conserved.
  const auto &L = sys.linear_symbol();
  const auto &w = sys.nonlinear_weight();
  REQUIRE(L[0] == Approx(0.0).margin(1e-15));
  REQUIRE(w[0] == Approx(0.0).margin(1e-15));

  auto &psi = state.get_field<double>("psi").vec();
  double mean0 = 0.0;
  for (double v : psi) {
    mean0 += v;
  }
  sys.step(0.0);
  double mean1 = 0.0;
  for (double v : psi) {
    mean1 += v;
  }
  REQUIRE(mean1 == Approx(mean0).margin(1e-10));
}

namespace {

struct AdvectionToy {
  pfc::Domain domain{};
  pfc::Box3i box{};

  void declare_fields(SimulationState &state) const {
    pfc::sim::add_declared_field<double>(state, "psi", domain, box, 0);
  }

  [[nodiscard]] std::complex<double> linear_symbol(double kx, double, double) const {
    return {0.0, -kx};
  }

  struct PW {
    [[nodiscard]] double nonlinearity(const pfc::sim::SpectralCell &) const {
      return 0.0;
    }
  };
  [[nodiscard]] PW pointwise() const { return {}; }
};

static_assert(pfc::sim::HasComplexLinearSymbol<AdvectionToy>);
static_assert(pfc::sim::SpectralETDPhysics<AdvectionToy>);

double cosine_phase(const pfc::data::Field<double> &u, int nx) {
  const auto n = u.local_size();
  const auto sp = u.spacing();
  const double Lx = static_cast<double>(n[0]) * sp[0];
  const double twopi = 2.0 * pfc::pi;
  double c = 0.0;
  double s = 0.0;
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = u.coords(i, j, k);
        const double th = twopi * static_cast<double>(nx) * x[0] / Lx;
        c += u(i, j, k) * std::cos(th);
        s += u(i, j, k) * std::sin(th);
      }
    }
  }
  return std::atan2(s, c);
}

} // namespace

TEST_CASE("SpectralETDSystem complex L advects a cosine without damping",
          "[spectral_etd][unit][complex]") {
  constexpr int N = 32;
  constexpr int nx = 2;
  constexpr double dt = 0.05;
  constexpr int n_steps = 8;
  const auto domain = pfc::domain::create(pfc::GridSize({N, 1, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  auto physics = make_physics<AdvectionToy>(domain, fft);
  SimulationState state;
  physics.declare_fields(state);
  auto &psi = state.get_field<double>("psi");
  const double Lx = static_cast<double>(N);
  const double k = 2.0 * pfc::pi * static_cast<double>(nx) / Lx;
  psi.apply([&](double x, double, double) { return std::cos(k * x); });

  SpectralETDSystem<AdvectionToy> sys(physics, fft, state, dt);
  REQUIRE(sys.linear_symbol().empty());
  REQUIRE(sys.linear_symbol_complex().size() == fft.size_outbox());

  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    t = sys.step(t);
  }
  const double amp = cosine_phase(psi, nx);
  REQUIRE(amp == Approx(k * t).margin(1e-6));
  double energy = 0.0;
  const auto n = psi.local_size();
  for (int kk = 0; kk < n[2]; ++kk) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        energy += psi(i, j, kk) * psi(i, j, kk);
      }
    }
  }
  REQUIRE(energy == Approx(0.5 * static_cast<double>(N)).margin(1e-8));
}

TEST_CASE("SpectralETDSystem moving-frame physics zero field stays zero and "
          "reduces the free energy",
          "[spectral_etd][unit][moving_frame]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  auto physics = make_physics<MovingFrameToy<>>(domain, fft);
  physics.nl.g = 0.0; // no source term
  SimulationState state;
  physics.declare_fields(state);
  SpectralETDOptions opt{};
  opt.comm = MPI_COMM_WORLD;
  SpectralETDSystem<MovingFrameToy<>> sys(physics, fft, state, 0.01, opt);

  REQUIRE(state.has_field("P_star_psi"));
  REQUIRE(state.has_field("P_hat"));
  REQUIRE(state.has_field("fe_density"));
  REQUIRE(sys.correlation_kernel().size() == fft.size_outbox());
  REQUIRE(sys.correlation_kernel()[0] == Approx(physics.correlation_kernel(0.0)));

  auto &psi = state.get_field<double>("psi").vec();
  for (double &v : psi) {
    v = 0.0;
  }
  const double t1 = sys.step(0.0);
  REQUIRE(t1 == Approx(0.01));
  for (double v : psi) {
    REQUIRE(v == Approx(0.0).margin(1e-14));
  }
  REQUIRE(sys.last_free_energy_sum() == Approx(0.0).margin(1e-14));
  REQUIRE(sys.last_free_energy() == Approx(0.0).margin(1e-14));
}

TEST_CASE("SpectralETDSystem moving-frame free energy equals sum of 0.5 psi^2 "
          "times the cell volume",
          "[spectral_etd][unit][moving_frame]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  auto physics = make_physics<MovingFrameToy<>>(domain, fft);
  SimulationState state;
  physics.declare_fields(state);
  auto &psi_field = state.get_field<double>("psi");
  fill_cosine(psi_field, -0.006);
  const std::vector<double> psi0 = psi_field.vec();
  SpectralETDSystem<MovingFrameToy<>> sys(physics, fft, state, 0.01);

  (void)sys.attempt(0.0);
  double expected = 0.0;
  for (double v : psi0) {
    expected += 0.5 * v * v;
  }
  REQUIRE(sys.last_free_energy_sum() == Approx(expected).margin(1e-12));
  REQUIRE(sys.last_free_energy() ==
          Approx(expected * pfc::sim::cell_volume(domain)).margin(1e-12));
}
