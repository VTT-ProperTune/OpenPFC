// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <aluminum/aluminum_session.hpp>
#include <aluminum/aluminum_physics.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/openpfc.hpp>

using namespace Catch::Matchers;
using json = nlohmann::json;

namespace {
pfc::MPI_Worker g_mpi(0, nullptr, MPI_COMM_WORLD, false);

json aluminum_params_json() {
  return {{"n0", -0.0060},           {"alpha", 0.20},
          {"n_sol", -0.036},         {"n_vap", -1.297},
          {"T_const", 980.0},        {"T_min", 780.0},
          {"T_max", 1280.0},         {"T0", 89285.0},
          {"Bx", 0.817900686921996}, {"G_grid", 0.0},
          {"V_grid", 0.0},           {"x_initial", 130.0},
          {"alpha_farTol", 0.001},   {"alpha_highOrd", 0},
          {"lambda", 0.22},          {"stabP", 0.0},
          {"shift_u", 1.0},          {"shift_s", 0.0},
          {"p2_bar", 0.8286531831},  {"p3_bar", -0.04204863},
          {"p4_bar", 0.007533},      {"q20_bar", 0.016531729105214},
          {"q21_bar", 5.467},        {"q30_bar", 1.7152418049986},
          {"q31_bar", 0.45},         {"q40_bar", 0.787482}};
}

double sumsq(const std::vector<double> &v) {
  double s = 0.0;
  for (double x : v) {
    s += x * x;
  }
  return s;
}

json golden_settings(int n, double t1, double dt) {
  return {{"model", {{"name", "aluminum"}, {"params", aluminum_params_json()}}},
          {"domain",
           {{"Lx", n},
            {"Ly", n},
            {"Lz", n},
            {"dx", 1.0},
            {"dy", 1.0},
            {"dz", 1.0},
            {"origin", "corner"}}},
          {"timestepping", {{"t0", 0.0}, {"t1", t1}, {"dt", dt}, {"saveat", dt}}},
          {"initial_conditions",
           {{{"target", "psi"}, {"type", "constant"}, {"n0", -0.0060}}}}};
}

} // namespace

TEST_CASE("SpectralETDSystem<AluminumPhysics> 8^3 two steps is finite",
          "[aluminum][physics][etd]") {
  constexpr int N = 8;
  constexpr double dt = 0.01;
  auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  aluminum::AluminumPhysics<> phys;
  phys.domain = domain;
  phys.box = fft.get_inbox_bounds();
  aluminum::apply_aluminum_json(aluminum_params_json(), phys.params);
  pfc::SimulationState state;
  phys.declare_fields(state);
  std::fill(state.get_field<double>("psi").vec().begin(),
            state.get_field<double>("psi").vec().end(), -0.0060);
  pfc::sim::SpectralETDSystem<aluminum::AluminumPhysics<>> sys(
      phys, fft, state, dt);
  REQUIRE(sys.linear_symbol().size() == fft.size_outbox());
  sys.step(0.0);
  sys.step(dt);
  REQUIRE(std::isfinite(sys.last_free_energy_sum()));
  REQUIRE(sumsq(state.get_field<double>("psi").vec()) > 0.0);
}

TEST_CASE("AluminumSession JSON constant IC two steps",
          "[aluminum][physics][session]") {
  const json settings = golden_settings(8, 0.02, 0.01);
  aluminum::AluminumSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(session.psi().vec().size() == 512);
  REQUIRE(std::isfinite(sumsq(session.psi().vec())));
}

TEST_CASE("AluminumSession 4-rank 16^3/20-step run", "[aluminum][golden][MPI]") {
  int nproc = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (nproc != 4) {
    SKIP("requires exactly 4 MPI ranks");
  }
  const json settings = golden_settings(16, 0.20, 0.01);
  aluminum::AluminumSession session(settings, rank, nproc, MPI_COMM_WORLD);
  session.run();
  double s = sumsq(session.psi().vec());
  double g = 0.0;
  MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(std::isfinite(g));
  REQUIRE(g > 0.0);
}

TEST_CASE("AluminumSession seed_grid_fcc 5-step CPU checksum",
          "[aluminum][etd][seed_grid_fcc][session]") {
  constexpr int N = 32;
  constexpr double dt = 0.01;
  json settings = {
      {"model", {{"name", "aluminum"}, {"params", aluminum_params_json()}}},
      {"domain",
       {{"Lx", N},
        {"Ly", N},
        {"Lz", N},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping", {{"t0", 0.0}, {"t1", 0.05}, {"dt", dt}, {"saveat", 0.05}}},
      {"initial_conditions",
       json::array({{{"target", "psi"}, {"type", "constant"}, {"n0", -0.0060}},
                    {{"target", "psi"},
                     {"type", "seed_grid_fcc"},
                     {"X0", 8.0},
                     {"Ny", 2},
                     {"Nz", 2},
                     {"radius", 4.0},
                     {"amplitude", 0.4},
                     {"rho", -0.036},
                     {"rseed", 42}}})}};
  aluminum::register_catalog();
  aluminum::AluminumSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  double sum = 0.0;
  double sumsq_v = 0.0;
  for (double x : session.psi().vec()) {
    sum += x;
    sumsq_v += x * x;
  }
  std::cout << std::setprecision(17)
            << "CPU_GOLDEN aluminum_etd n=" << session.psi().vec().size()
            << " sum=" << sum << " sumsq=" << sumsq_v << '\n';
  REQUIRE(session.psi().vec().size() == 32768);
  REQUIRE_THAT(sum, WithinRel(-263.63079658808601, 1e-10));
  REQUIRE_THAT(sumsq_v, WithinRel(1111.6016268617182, 1e-10));
}
