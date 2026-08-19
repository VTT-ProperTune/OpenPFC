// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <vector>
#include <nlohmann/json.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <mpi.h>
#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/spectral_mean_field_etd.hpp>
#include <tungsten/cpu/tungsten_model.hpp>
#include <tungsten/tungsten_etd_session.hpp>
#include <tungsten/tungsten_physics.hpp>

using Catch::Approx;
using nlohmann::json;

namespace {

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

void fill_ic(pfc::data::Field<double> &psi) {
  const auto n = pfc::domain::get_size(psi.domain());
  const auto dx = pfc::domain::get_spacing(psi.domain());
  const double lx = static_cast<double>(n[0]) * dx[0];
  psi.apply([&](double x, double, double) {
    return -0.10 + 0.01 * std::cos(2.0 * pfc::pi * x / lx);
  });
}

} // namespace

TEST_CASE("TungstenPhysics schema round-trips JSON params",
          "[tungsten][physics][schema]") {
  auto schema = tungsten::TungstenPhysics<>::schema();
  json j = {{"n0", -0.12},
            {"n_sol", -0.047},
            {"n_vap", -0.464},
            {"T", 3300.0},
            {"T0", 156000.0},
            {"Bx", 0.8582},
            {"alpha", 0.50},
            {"alpha_farTol", 0.001},
            {"alpha_highOrd", 4},
            {"lambda", 0.22},
            {"stabP", 0.2},
            {"shift_u", 0.3341},
            {"shift_s", 0.1898},
            {"p2", 1.0},
            {"p3", -0.5},
            {"p4", 0.333333333},
            {"q20", -0.0037},
            {"q21", 1.0},
            {"q30", -12.4567},
            {"q31", 20.0},
            {"q40", 45.0}};
  const auto vals = schema.parse(j);
  REQUIRE(vals.n0 == Approx(-0.12));
  TungstenParams p;
  tungsten::apply_schema_values(vals, p);
  REQUIRE(p.get_n0() == Approx(-0.12));
  REQUIRE(p.get_alpha_highOrd() == 4);
}

TEST_CASE("SpectralMeanFieldEtdSystem matches Gen-1 Tungsten one step",
          "[tungsten][physics][etd]") {
  constexpr int N = 8;
  constexpr double dt = 0.01;
  auto domain = pfc::domain::create(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft_legacy = pfc::fft::create(decomp);
  auto fft_new = pfc::fft::create(decomp);

  Tungsten legacy(fft_legacy, domain);
  pfc::initialize(legacy, dt);
  pfc::data::Field<double> ic(domain, fft_legacy.get_inbox_bounds(), 0);
  fill_ic(ic);
  legacy.get_real_field("psi") = ic.vec();

  tungsten::TungstenPhysics<> phys;
  phys.domain = domain;
  phys.box = fft_new.get_inbox_bounds();
  phys.params = legacy.params;
  pfc::SimulationState state;
  phys.declare_fields(state);
  state.get_field<double>("psi").vec() = ic.vec();
  pfc::sim::SpectralMeanFieldEtdSystem<tungsten::TungstenPhysics<>> sys(
      phys, fft_new, state, dt);

  REQUIRE(sys.linear_symbol().size() == fft_new.size_outbox());
  REQUIRE(sys.filter_mf().size() == fft_new.size_outbox());

  legacy.step(0.0);
  sys.step(0.0);
  REQUIRE(max_abs_diff(legacy.get_real_field("psi"),
                       state.get_field<double>("psi").vec()) < 1e-10);
}

TEST_CASE("SpectralMeanFieldEtdSystem matches Gen-1 Tungsten for 10 steps",
          "[tungsten][physics][etd][multistep]") {
  constexpr int N = 8;
  constexpr double dt = 0.01;
  auto domain = pfc::domain::create(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft_legacy = pfc::fft::create(decomp);
  auto fft_new = pfc::fft::create(decomp);

  Tungsten legacy(fft_legacy, domain);
  pfc::initialize(legacy, dt);
  pfc::data::Field<double> ic(domain, fft_legacy.get_inbox_bounds(), 0);
  fill_ic(ic);
  legacy.get_real_field("psi") = ic.vec();

  tungsten::TungstenPhysics<> phys;
  phys.domain = domain;
  phys.box = fft_new.get_inbox_bounds();
  phys.params = legacy.params;
  pfc::SimulationState state;
  phys.declare_fields(state);
  state.get_field<double>("psi").vec() = ic.vec();
  pfc::sim::SpectralMeanFieldEtdSystem<tungsten::TungstenPhysics<>> sys(
      phys, fft_new, state, dt);

  double t = 0.0;
  for (int step = 0; step < 10; ++step) {
    legacy.step(t);
    t = sys.step(t);
  }
  REQUIRE(max_abs_diff(legacy.get_real_field("psi"),
                       state.get_field<double>("psi").vec()) < 1e-10);
}

TEST_CASE("TungstenEtdSession JSON constant IC matches Gen-1 two steps",
          "[tungsten][physics][session]") {
  json settings = {
      {"model",
       {{"name", "tungsten"},
        {"params",
         {{"n0", -0.10},
          {"n_sol", -0.047},
          {"n_vap", -0.464},
          {"T", 3300.0},
          {"T0", 156000.0},
          {"Bx", 0.8582},
          {"alpha", 0.50},
          {"alpha_farTol", 0.001},
          {"alpha_highOrd", 4},
          {"lambda", 0.22},
          {"stabP", 0.2},
          {"shift_u", 0.3341},
          {"shift_s", 0.1898},
          {"p2", 1.0},
          {"p3", -0.5},
          {"p4", 0.333333333},
          {"q20", -0.0037},
          {"q21", 1.0},
          {"q30", -12.4567},
          {"q31", 20.0},
          {"q40", 45.0}}}}},
      {"domain",
       {{"Lx", 8},
        {"Ly", 8},
        {"Lz", 8},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping",
       {{"t0", 0.0}, {"t1", 0.02}, {"dt", 0.01}, {"saveat", 0.01}}},
      {"initial_conditions",
       {{{"target", "psi"}, {"type", "constant"}, {"n0", -0.10}}}}};

  tungsten::TungstenEtdSession session(settings, 0, 1, MPI_COMM_WORLD);

  auto domain = pfc::domain::create(
      pfc::GridSize({8, 8, 8}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  Tungsten legacy(fft, domain);
  tungsten::apply_tungsten_json(settings["model"]["params"], legacy.params);
  pfc::initialize(legacy, 0.01);
  std::fill(legacy.get_real_field("psi").begin(),
            legacy.get_real_field("psi").end(), -0.10);

  session.run();
  legacy.step(0.0);
  legacy.step(0.01);
  REQUIRE(max_abs_diff(legacy.get_real_field("psi"), session.psi().vec()) <
          1e-10);
}
