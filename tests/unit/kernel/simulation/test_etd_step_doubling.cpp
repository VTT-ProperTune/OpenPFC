// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * ETD1 step doubling: linear exactness, estimator order, tolerance sweep,
 * rejection, and a rank-consistent decision.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mpi.h>

#include <fixtures/swift_hohenberg.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/adaptive_controller.hpp>
#include <openpfc/kernel/simulation/etd_step_doubling.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using Catch::Matchers::WithinAbs;
using pfc::SimulationState;
using pfc::sim::AdaptiveControlConfig;
using pfc::sim::AdaptiveControlMode;
using pfc::sim::AdaptiveTimeController;
using pfc::sim::ETD1StepDoubling;
using pfc::sim::StepController;
using pfc::test::SwiftHohenberg;

namespace {

struct ZeroNonlinearity {
  double nonlinearity(const pfc::sim::SpectralCell &) const { return 0.0; }
};

/// ∂t ψ = ∇² ψ. One Fourier mode is exp(-k² t) times its initial amplitude.
struct LinearHeat {
  pfc::Domain domain{};
  pfc::Box3i box{};

  void declare_fields(SimulationState &state) const {
    pfc::sim::add_declared_field<double>(state, "psi", domain, box, 0);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    return k_laplacian;
  }

  [[nodiscard]] ZeroNonlinearity pointwise() const { return {}; }
};

static_assert(pfc::sim::SpectralETDPhysics<LinearHeat>);

constexpr int N = 8;

pfc::Domain unit_domain(int n = N) {
  return pfc::domain::create(pfc::GridSize({n, n, n}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
}

double mode_k2(int n = N) {
  const double k = 2.0 * pfc::pi / static_cast<double>(n);
  return k * k;
}

void fill_cosine(pfc::data::Field<double> &psi, double amplitude) {
  const auto n = pfc::domain::get_size(psi.domain());
  const auto dx = pfc::domain::get_spacing(psi.domain());
  const double lx = static_cast<double>(n[0]) * dx[0];
  psi.apply([&](double x, double, double) {
    return amplitude * std::cos(2.0 * pfc::pi * x / lx);
  });
}

double linf_diff(std::span<const double> a, std::span<const double> b) {
  double m = 0.0;
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

double linf_of(std::span<const double> a) {
  double m = 0.0;
  for (double v : a) {
    m = std::max(m, std::abs(v));
  }
  return m;
}

} // namespace

TEST_CASE("ETD1 step doubling is exact on a diagonal heat mode", "[etd_doubling]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  LinearHeat physics;
  physics.domain = domain;
  physics.box = fft.get_inbox_bounds();
  SimulationState state;
  physics.declare_fields(state);
  constexpr double dt = 0.05;
  pfc::sim::SpectralETDSystem<LinearHeat> sys(physics, fft, state, dt);
  auto &psi_field = state.get_field<double>("psi");
  fill_cosine(psi_field, 1.0);
  auto &psi = psi_field.vec();

  ETD1StepDoubling<LinearHeat> doubling(sys);
  doubling.attempt(0.0, dt);
  REQUIRE(doubling.nonlinear_count() == 3);
  REQUIRE(doubling.forward_count() == 6);
  REQUIRE(doubling.backward_count() == 3);
  REQUIRE(doubling.transform_count() == 9);
  REQUIRE(sys.dt() == dt * 0.5);

  const double factor = std::exp(-mode_k2() * dt);
  const auto saved = doubling.saved_state();
  std::vector<double> exact(saved.begin(), saved.end());
  for (double &v : exact) {
    v *= factor;
  }
  REQUIRE(linf_diff(doubling.full_step(), exact) < 1e-10);
  REQUIRE(linf_diff(doubling.half_steps(), exact) < 1e-10);
  REQUIRE(linf_of(doubling.raw_difference()) < 1e-10);

  // Half-step coefficients are installed during the open attempt, and they
  // match a system that was built at dt/2. They are not the full-step cache.
  SimulationState half_state;
  physics.declare_fields(half_state);
  pfc::sim::SpectralETDSystem<LinearHeat> half_sys(physics, fft, half_state,
                                                   dt * 0.5);
  REQUIRE(sys.nonlinear_weight() == half_sys.nonlinear_weight());
  SimulationState full_state;
  physics.declare_fields(full_state);
  pfc::sim::SpectralETDSystem<LinearHeat> full_sys(physics, fft, full_state, dt);
  REQUIRE(sys.nonlinear_weight() != full_sys.nonlinear_weight());

  doubling.commit();
  REQUIRE(sys.dt() == dt);
  REQUIRE(sys.nonlinear_weight() == full_sys.nonlinear_weight());
  REQUIRE(linf_diff(psi, exact) < 1e-10);

  // A rejected attempt restores the field and does not leave dt/2 installed.
  const std::vector<double> accepted = psi;
  doubling.attempt(dt, dt);
  REQUIRE(psi != accepted);
  doubling.reject();
  REQUIRE(psi == accepted);
  REQUIRE(sys.dt() == dt);
  REQUIRE(sys.nonlinear_weight() == full_sys.nonlinear_weight());
  REQUIRE_FALSE(doubling.attempt_open());
}

TEST_CASE("ETD1 doubling difference is second order on Swift-Hohenberg",
          "[etd_doubling]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  const auto difference = [&](double dt) {
    SwiftHohenberg physics;
    physics.domain = domain;
    physics.box = fft.get_inbox_bounds();
    physics.params.epsilon = 0.25;
    SimulationState state;
    physics.declare_fields(state);
    pfc::sim::SpectralETDSystem<SwiftHohenberg> sys(physics, fft, state, dt);
    fill_cosine(state.get_field<double>("psi"), 0.2);
    ETD1StepDoubling<SwiftHohenberg> doubling(sys);
    doubling.attempt(0.0, dt);
    return linf_of(doubling.raw_difference());
  };

  const double d1 = difference(0.04);
  const double d2 = difference(0.02);
  const double d3 = difference(0.01);
  const double d4 = difference(0.005);
  std::cout << "etd order differences " << d1 << ' ' << d2 << ' ' << d3 << ' ' << d4
            << "\n";
  REQUIRE(d1 > d2);
  REQUIRE(d2 > d3);
  REQUIRE(d3 > d4);
  // Local difference of a first-order step is O(dt^2): halving dt divides it by 4.
  REQUIRE_THAT(d1 / d2, WithinAbs(4.0, 0.5));
  REQUIRE_THAT(d2 / d3, WithinAbs(4.0, 0.5));
  REQUIRE_THAT(d3 / d4, WithinAbs(4.0, 0.35));
}

namespace {

struct AdaptiveRecord {
  int accepted{0};
  int rejected{0};
  int nonlinear{0};
  int transforms{0};
  double error{0.0};
  double first_metric{0.0};
  std::vector<double> dt;
  std::vector<double> final_state;
};

AdaptiveRecord run_sh(StepController kind, double rtol, double t_end,
                      const std::vector<double> &reference) {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  SwiftHohenberg physics;
  physics.domain = domain;
  physics.box = fft.get_inbox_bounds();
  physics.params.epsilon = 0.25;
  SimulationState state;
  physics.declare_fields(state);
  // The local difference on this 8³ mode is about 4e-8 at dt = 0.01, and the
  // state scale is about 0.2. rtol near 1e-6 is where that difference meets
  // the tolerance. Looser values all sit on the growth limit and take the
  // same steps, so they cannot show a tolerance effect.
  pfc::sim::SpectralETDSystem<SwiftHohenberg> sys(physics, fft, state, 0.02);
  fill_cosine(state.get_field<double>("psi"), 0.2);

  AdaptiveControlConfig cfg;
  cfg.mode = AdaptiveControlMode::adaptive;
  cfg.controller = kind;
  cfg.atol = 1e-12;
  cfg.rtol = rtol;
  cfg.safety_factor = 0.9;
  cfg.growth_max = 2.0;
  cfg.shrink_max = 0.5;
  cfg.min_dt = 1e-6;
  cfg.max_dt = 0.05;
  cfg.max_sequential_rejections = 30;
  AdaptiveTimeController controller(cfg, /*error_order=*/2);
  pfc::Time time({0.0, t_end, 0.02}, 0.0);
  ETD1StepDoubling<SwiftHohenberg> doubling(sys);

  AdaptiveRecord rec;
  int guard = 0;
  while (!time.done() && guard < 4000) {
    time.begin_attempt(time.get_dt());
    const double dt = time.get_attempted_dt();
    doubling.attempt(time.get_accepted_time(), dt);
    const auto decision = controller.decide_from_embedded_error(
        dt, doubling.raw_difference(), doubling.saved_state(),
        doubling.half_steps());
    if (guard == 0) {
      rec.first_metric = decision.metric;
    }
    if (decision.accepted) {
      doubling.commit();
      rec.dt.push_back(dt);
    } else {
      doubling.reject();
    }
    rec.nonlinear += doubling.nonlinear_count();
    rec.transforms += doubling.transform_count();
    controller.apply(time, decision);
    ++guard;
  }
  REQUIRE(time.done());
  rec.accepted = controller.accepted_count();
  rec.rejected = controller.rejected_count();
  rec.final_state = state.get_field<double>("psi").vec();
  rec.error = linf_diff(rec.final_state, reference);
  return rec;
}

std::vector<double> sh_reference(double t_end, double dt) {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  SwiftHohenberg physics;
  physics.domain = domain;
  physics.box = fft.get_inbox_bounds();
  physics.params.epsilon = 0.25;
  SimulationState state;
  physics.declare_fields(state);
  pfc::sim::SpectralETDSystem<SwiftHohenberg> sys(physics, fft, state, dt);
  fill_cosine(state.get_field<double>("psi"), 0.2);
  double t = 0.0;
  while (t + dt * 0.5 < t_end) {
    t = sys.step(t);
  }
  // Land on t_end with one clipped step when the grid does not divide t_end.
  if (t < t_end) {
    sys.set_dt(t_end - t);
    t = sys.step(t);
  }
  REQUIRE_THAT(t, WithinAbs(t_end, 1e-12));
  return state.get_field<double>("psi").vec();
}

} // namespace

TEST_CASE("tighter ETD tolerances reduce Swift-Hohenberg global error",
          "[etd_doubling]") {
  constexpr double t_end = 0.4;
  const auto reference = sh_reference(t_end, 2.5e-4);
  const auto loose = run_sh(StepController::memoryless, 1e-6, t_end, reference);
  const auto mid = run_sh(StepController::memoryless, 2.5e-7, t_end, reference);
  const auto tight = run_sh(StepController::memoryless, 6e-8, t_end, reference);
  const auto pi = run_sh(StepController::pi, 2.5e-7, t_end, reference);
  const auto again = run_sh(StepController::memoryless, 2.5e-7, t_end, reference);

  const auto report = [](const char *label, const AdaptiveRecord &rec) {
    std::cout << label << " accepted=" << rec.accepted
              << " rejected=" << rec.rejected << " nonlinear=" << rec.nonlinear
              << " transforms=" << rec.transforms << " error=" << rec.error
              << " first_metric=" << rec.first_metric << " dt";
    for (double h : rec.dt) {
      std::cout << ' ' << h;
    }
    std::cout << "\n";
  };
  report("etd memoryless rtol=1e-6", loose);
  report("etd memoryless rtol=2.5e-7", mid);
  report("etd memoryless rtol=6e-8", tight);
  report("etd pi rtol=2.5e-7", pi);

  REQUIRE(loose.error > mid.error);
  REQUIRE(mid.error > tight.error);
  REQUIRE(tight.error < 1e-5);
  REQUIRE(again.dt == mid.dt);
  REQUIRE(again.accepted == mid.accepted);
  REQUIRE(again.rejected == mid.rejected);
  // PI is a second controller at the middle tolerance, not a claim that it wins.
  REQUIRE(pi.accepted > 0);
  REQUIRE(std::isfinite(pi.error));
}

TEST_CASE("a rejected ETD attempt does not advance time or the accepted field",
          "[etd_doubling]") {
  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  SwiftHohenberg physics;
  physics.domain = domain;
  physics.box = fft.get_inbox_bounds();
  SimulationState state;
  physics.declare_fields(state);
  pfc::sim::SpectralETDSystem<SwiftHohenberg> sys(physics, fft, state, 0.2);
  fill_cosine(state.get_field<double>("psi"), 0.2);
  const std::vector<double> initial = state.get_field<double>("psi").vec();

  AdaptiveControlConfig cfg;
  cfg.mode = AdaptiveControlMode::adaptive;
  cfg.atol = 1e-14;
  cfg.rtol = 1e-14;
  cfg.min_dt = 1e-8;
  cfg.max_dt = 0.2;
  cfg.shrink_max = 0.5;
  cfg.growth_max = 2.0;
  cfg.max_sequential_rejections = 4;
  AdaptiveTimeController controller(cfg, 2);
  pfc::Time time({0.0, 1.0, 0.2}, 0.0);
  ETD1StepDoubling<SwiftHohenberg> doubling(sys);

  time.begin_attempt(time.get_dt());
  const double attempted = time.get_attempted_dt();
  doubling.attempt(0.0, attempted);
  const auto decision = controller.decide_from_embedded_error(
      attempted, doubling.raw_difference(), doubling.saved_state(),
      doubling.half_steps());
  REQUIRE_FALSE(decision.accepted);
  doubling.reject();
  controller.apply(time, decision);
  REQUIRE(time.get_accepted_time() == 0.0);
  REQUIRE(state.get_field<double>("psi").vec() == initial);
  REQUIRE(controller.rejected_count() == 1);
  REQUIRE(time.get_dt() <= attempted * 0.5 + 1e-15);
  REQUIRE(time.get_dt() >= cfg.min_dt);
}

TEST_CASE("MPI ranks share one ETD step size", "[etd_doubling][MPI]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    return;
  }

  auto domain = unit_domain();
  auto decomp = pfc::decomposition::create(domain, size);
  auto fft = pfc::fft::create(decomp);
  SwiftHohenberg physics;
  physics.domain = domain;
  physics.box = fft.get_inbox_bounds();
  SimulationState state;
  physics.declare_fields(state);
  constexpr double dt = 0.02;
  pfc::sim::SpectralETDSystem<SwiftHohenberg> sys(physics, fft, state, dt);
  auto &psi = state.get_field<double>("psi");
  psi.apply([&](double x, double, double) {
    return 0.2 * std::cos(2.0 * pfc::pi * x / static_cast<double>(N));
  });

  ETD1StepDoubling<SwiftHohenberg> doubling(sys);
  doubling.attempt(0.0, dt);
  AdaptiveControlConfig cfg;
  cfg.mode = AdaptiveControlMode::adaptive;
  cfg.atol = 1e-8;
  cfg.rtol = 1e-3;
  cfg.max_dt = 0.05;
  cfg.min_dt = 1e-8;
  AdaptiveTimeController controller(cfg, 2);
  const auto decision = controller.decide_from_embedded_error(
      dt, doubling.raw_difference(), doubling.saved_state(), doubling.half_steps());
  doubling.reject();

  double dt_min = decision.next_dt;
  double dt_max = decision.next_dt;
  MPI_Allreduce(MPI_IN_PLACE, &dt_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &dt_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(dt_min == dt_max);
  REQUIRE(std::isfinite(decision.next_dt));
  REQUIRE(decision.next_dt > 0.0);
  // The rejected attempt restored the local field.
  double local = 0.0;
  for (double v : psi.vec()) {
    local = std::max(local, std::abs(v));
  }
  REQUIRE_THAT(local, WithinAbs(0.2, 1e-12));
}
