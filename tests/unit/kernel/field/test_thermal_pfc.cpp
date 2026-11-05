// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/field/thermal_pfc.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <vector>

using Catch::Approx;
namespace coupling = pfc::field::coupling;
using pfc::field::fd::BoundaryQuantity;
using pfc::field::fd::DiffusionAxis;
using pfc::field::fd::FaceCondition;
using pfc::field::fd::FluxLedger;

namespace {

double pi() { return std::acos(-1.0); }

double weighted_mass(const DiffusionAxis &axis, const std::vector<double> &field) {
  double mass = 0.0;
  for (int i = 0; i < axis.size(); ++i)
    mass += axis.weight(i) * field[static_cast<std::size_t>(i)];
  return mass;
}

void report(const std::string &label, int n, double a, double b, double c,
            double d) {
  std::ostringstream out;
  out.setf(std::ios::scientific);
  out.precision(17);
  out << "THERMAL_PFC " << label << " " << n << " " << a << " " << b << " " << c
      << " " << d;
  std::cout << out.str() << '\n';
}

int diffusion_steps(int intervals, double horizon) {
  return static_cast<int>(
      std::ceil(horizon * static_cast<double>(intervals) * intervals * 20.0));
}

coupling::SubstrateDensityRequest density_request() {
  coupling::SubstrateDensityRequest request;
  request.boundary = coupling::SolidificationBoundary::Substrate;
  request.mobility = 0.4;
  request.wave_number = pi();
  request.cubic = 0.7;
  request.temperature_scale = 1.25;
  request.melting_level = 0.5;
  return request;
}

std::vector<double> cosine_line(int n, double h, double offset, double amplitude) {
  std::vector<double> field(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    field[static_cast<std::size_t>(i)] = offset + amplitude * std::cos(pi() * i * h);
  return field;
}

} // namespace

TEST_CASE("Solved heat on a substrate axis matches the mode and the wall flux",
          "[unit][field][thermal_pfc]") {
  const double horizon = 0.02;
  double previous_error = 0.0;
  for (int intervals : {32, 64}) {
    const int n = intervals + 1;
    const double h = 1.0 / static_cast<double>(intervals);
    DiffusionAxis axis(n, h);
    std::vector<double> field = cosine_line(n, h, 0.0, 1.0);
    const std::vector<double> initial = field;
    const int steps = diffusion_steps(intervals, horizon);
    const double dt = horizon / static_cast<double>(steps);
    coupling::SolvedHeatRequest request;
    request.diffusivity = 1.0;
    FluxLedger ledger;
    double reported = 0.0;
    for (int step = 0; step < steps; ++step) {
      const coupling::HeatStep advanced =
          coupling::advance_solved_heat(axis, field, request, step * dt, dt, ledger);
      reported += advanced.inventory_change;
    }
    const double mass_gap =
        weighted_mass(axis, field) - weighted_mass(axis, initial);
    double error = 0.0;
    double idle = 0.0;
    const double decay = std::exp(-pi() * pi() * horizon);
    for (int i = 0; i < n; ++i) {
      const double x = i * h;
      const double exact = std::cos(pi() * x) * decay;
      const double weight = axis.weight(i);
      const double miss = field[static_cast<std::size_t>(i)] - exact;
      const double untouched = initial[static_cast<std::size_t>(i)] - exact;
      error += weight * miss * miss;
      idle += weight * untouched * untouched;
    }
    error = std::sqrt(error);
    idle = std::sqrt(idle);
    report("cosine", n, error, idle, mass_gap, ledger.accepted());
    REQUIRE(error < 1.0e-5);
    REQUIRE(error < 0.01 * idle);
    REQUIRE(std::abs(mass_gap) < 1.0e-12);
    REQUIRE(std::abs(reported - mass_gap) < 1.0e-12);
    REQUIRE(ledger.accepted() == 0.0);
    if (previous_error != 0.0) {
      const double order = std::log2(previous_error / error);
      REQUIRE(order >= 3.5);
    }
    previous_error = error;
  }

  DiffusionAxis axis(33, 1.0 / 32.0);
  for (const double far : {0.0, 0.15}) {
    std::vector<double> field(33, far == 0.0 ? 0.4 : 0.5);
    const double mass_before = weighted_mass(axis, field);
    coupling::SolvedHeatRequest request;
    request.diffusivity = 0.8;
    request.substrate_flux = 0.15;
    request.far_flux = far;
    const double spatial_flux = -(request.substrate_flux + request.far_flux);
    FaceCondition substrate{BoundaryQuantity::ConstitutiveFlux,
                            [](double) { return 0.15; }};
    FaceCondition opposite{BoundaryQuantity::ConstitutiveFlux,
                           [far](double) { return far; }};
    const auto sample = [&](int j) { return field[static_cast<std::size_t>(j)]; };
    const auto material = [&](int) { return request.diffusivity; };
    double spatial = 0.0;
    for (int i = 0; i < axis.size(); ++i) {
      spatial += axis.weight(i) *
                 axis.apply(i, sample, material, 0.0, &substrate, &opposite);
    }
    REQUIRE(spatial == Approx(spatial_flux).margin(1.0e-12));
    const int steps = diffusion_steps(32, horizon);
    const double dt = horizon / static_cast<double>(steps);
    const double imposed = (request.substrate_flux + request.far_flux) * dt * steps;
    FluxLedger ledger;
    double reported = 0.0;
    for (int step = 0; step < steps; ++step) {
      const coupling::HeatStep advanced =
          coupling::advance_solved_heat(axis, field, request, step * dt, dt, ledger);
      reported += advanced.inventory_change;
    }
    const double mass_gap = weighted_mass(axis, field) - mass_before;
    report(far == 0.0 ? "substrate_flux" : "both_flux", 33, mass_gap,
           ledger.accepted(), imposed, reported);
    REQUIRE(ledger.accepted() == Approx(imposed).margin(1.0e-12));
    REQUIRE(mass_gap == Approx(-imposed).margin(1.0e-11));
    REQUIRE(reported == Approx(mass_gap).margin(1.0e-12));
  }

  std::vector<double> flat(33, 1.5);
  const std::vector<double> flat_initial = flat;
  coupling::SolvedHeatRequest still;
  still.diffusivity = 1.0;
  FluxLedger still_ledger;
  double still_reported = 0.0;
  const int still_steps = diffusion_steps(32, horizon);
  const double still_dt = horizon / static_cast<double>(still_steps);
  double still_drift = 0.0;
  for (int step = 0; step < still_steps; ++step) {
    still_reported += coupling::advance_solved_heat(
                          axis, flat, still, step * still_dt, still_dt, still_ledger)
                          .inventory_change;
  }
  for (int i = 0; i < axis.size(); ++i) {
    still_drift =
        std::max(still_drift, std::abs(flat[static_cast<std::size_t>(i)] -
                                       flat_initial[static_cast<std::size_t>(i)]));
  }
  report("stationary", 33, still_drift, still_reported, still_ledger.accepted(),
         0.0);
  REQUIRE(still_drift < 1.0e-12);
  REQUIRE(still_ledger.accepted() == 0.0);
  REQUIRE(std::abs(still_reported) < 1.0e-12);

  std::vector<double> rejected(33, 0.2);
  const std::vector<double> rejected_initial = rejected;
  FluxLedger bad_ledger;
  coupling::SolvedHeatRequest bad;
  bad.diffusivity = 0.0;
  REQUIRE_THROWS_AS(
      coupling::advance_solved_heat(axis, rejected, bad, 0.0, 1.0e-3, bad_ledger),
      std::invalid_argument);
  bad.diffusivity = -1.0;
  REQUIRE_THROWS_AS(
      coupling::advance_solved_heat(axis, rejected, bad, 0.0, 1.0e-3, bad_ledger),
      std::invalid_argument);
  bad.diffusivity = 1.0;
  REQUIRE_THROWS_AS(
      coupling::advance_solved_heat(axis, rejected, bad, 0.0, 0.0, bad_ledger),
      std::invalid_argument);
  std::vector<double> short_line(16, 0.2);
  REQUIRE_THROWS_AS(
      coupling::advance_solved_heat(axis, short_line, bad, 0.0, 1.0e-3, bad_ledger),
      std::invalid_argument);
  rejected[4] = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(
      coupling::advance_solved_heat(axis, rejected, bad, 0.0, 1.0e-3, bad_ledger),
      std::domain_error);
  REQUIRE(std::isnan(rejected[4]));
  for (int i = 0; i < axis.size(); ++i) {
    if (i == 4) continue;
    REQUIRE(rejected[static_cast<std::size_t>(i)] ==
            rejected_initial[static_cast<std::size_t>(i)]);
  }
  REQUIRE_FALSE(bad_ledger.active());
  REQUIRE(bad_ledger.accepted() == 0.0);
  REQUIRE(short_line[0] == 0.2);

  DiffusionAxis periodic(33, 1.0 / 32.0, true);
  std::vector<double> periodic_field(33, 0.3);
  const std::vector<double> periodic_initial = periodic_field;
  FluxLedger periodic_ledger;
  REQUIRE_THROWS_AS(coupling::advance_solved_heat(periodic, periodic_field, still,
                                                  0.0, 1.0e-3, periodic_ledger),
                    std::invalid_argument);
  REQUIRE(periodic_field == periodic_initial);
  REQUIRE_FALSE(periodic_ledger.active());
  REQUIRE(periodic_ledger.accepted() == 0.0);
}

TEST_CASE("Substrate PFC consumes the solved temperature",
          "[unit][field][thermal_pfc]") {
  const coupling::BoundaryMeaning periodic =
      coupling::boundary_meaning(coupling::SolidificationBoundary::Periodic);
  const coupling::BoundaryMeaning reservoir =
      coupling::boundary_meaning(coupling::SolidificationBoundary::Reservoir);
  const coupling::BoundaryMeaning substrate =
      coupling::boundary_meaning(coupling::SolidificationBoundary::Substrate);
  REQUIRE(periodic.periodic);
  REQUIRE_FALSE(periodic.density_reservoir);
  REQUIRE_FALSE(periodic.physical_wall);
  REQUIRE(reservoir.density_reservoir);
  REQUIRE_FALSE(reservoir.periodic);
  REQUIRE_FALSE(reservoir.physical_wall);
  REQUIRE(substrate.physical_wall);
  REQUIRE_FALSE(substrate.periodic);
  REQUIRE_FALSE(substrate.density_reservoir);
  REQUIRE_THROWS_AS(coupling::reject_free_surface(), std::invalid_argument);

  constexpr int intervals = 32;
  constexpr int n = intervals + 1;
  constexpr double h = 1.0 / intervals;
  constexpr double horizon = 0.02;
  DiffusionAxis axis(n, h);
  const coupling::SubstrateDensityRequest model = density_request();
  std::vector<double> cold = cosine_line(n, h, 0.2, 0.25);
  std::vector<double> hot = cosine_line(n, h, 1.4, 0.1);
  const std::vector<double> cold_initial = cold;
  const std::vector<double> hot_initial = hot;
  const std::vector<double> density = cosine_line(n, h, 0.2, 0.05);
  coupling::SolvedHeatRequest heat;
  heat.diffusivity = 1.0;
  const int steps = diffusion_steps(intervals, horizon);
  const double heat_dt = horizon / static_cast<double>(steps);
  FluxLedger cold_ledger;
  FluxLedger hot_ledger;
  double cold_heat = 0.0;
  double hot_heat = 0.0;
  for (int step = 0; step < steps; ++step) {
    cold_heat += coupling::advance_solved_heat(axis, cold, heat, step * heat_dt,
                                               heat_dt, cold_ledger)
                     .inventory_change;
    hot_heat += coupling::advance_solved_heat(axis, hot, heat, step * heat_dt,
                                              heat_dt, hot_ledger)
                    .inventory_change;
  }
  REQUIRE(std::abs(cold_heat) < 1.0e-12);
  REQUIRE(std::abs(hot_heat) < 1.0e-12);
  double cold_moved = 0.0;
  double hot_moved = 0.0;
  for (int i = 0; i < n; ++i) {
    cold_moved =
        std::max(cold_moved, std::abs(cold[static_cast<std::size_t>(i)] -
                                      cold_initial[static_cast<std::size_t>(i)]));
    hot_moved =
        std::max(hot_moved, std::abs(hot[static_cast<std::size_t>(i)] -
                                     hot_initial[static_cast<std::size_t>(i)]));
  }
  REQUIRE(cold_moved > 1.0e-3);
  REQUIRE(hot_moved > 1.0e-3);

  const pfc::field::fd::FaceCondition closed{BoundaryQuantity::NormalDerivative,
                                             [](double) { return 0.0; }};
  const auto one = [](int) { return 1.0; };
  std::vector<double> response(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    response[static_cast<std::size_t>(i)] =
        model.temperature_scale *
        (cold[static_cast<std::size_t>(i)] - hot[static_cast<std::size_t>(i)]) *
        density[static_cast<std::size_t>(i)];
  }
  const auto response_at = [&](int j) {
    return response[static_cast<std::size_t>(j)];
  };
  double max_potential_gap = 0.0;
  double max_rate_gap = 0.0;
  for (int i = 0; i < n; ++i) {
    const double cold_potential =
        coupling::substrate_chemical_potential(axis, i, density, cold, model, 0.0);
    const double hot_potential =
        coupling::substrate_chemical_potential(axis, i, density, hot, model, 0.0);
    const double expected_potential =
        model.temperature_scale *
        (cold[static_cast<std::size_t>(i)] - hot[static_cast<std::size_t>(i)]) *
        density[static_cast<std::size_t>(i)];
    max_potential_gap =
        std::max(max_potential_gap,
                 std::abs(cold_potential - hot_potential - expected_potential));
    const double cold_rate =
        coupling::substrate_pfc_rate(axis, i, density, cold, model, 0.0);
    const double hot_rate =
        coupling::substrate_pfc_rate(axis, i, density, hot, model, 0.0);
    const double expected_rate =
        model.mobility * axis.apply(i, response_at, one, 0.0, &closed, &closed);
    max_rate_gap =
        std::max(max_rate_gap, std::abs(cold_rate - hot_rate - expected_rate));
  }
  report("identity", n, max_potential_gap, max_rate_gap, 0.0, 0.0);
  REQUIRE(max_potential_gap < 1.0e-13);
  REQUIRE(max_rate_gap < 1.0e-10);

  constexpr double euler_dt = 1.0e-4;
  std::vector<double> cold_density = density;
  std::vector<double> hot_density = density;
  const coupling::SubstrateDensityStep cold_step = coupling::advance_substrate_pfc(
      axis, cold_density, cold, model, 0.0, euler_dt);
  const coupling::SubstrateDensityStep hot_step =
      coupling::advance_substrate_pfc(axis, hot_density, hot, model, 0.0, euler_dt);
  double max_euler_gap = 0.0;
  double density_gap = 0.0;
  for (int i = 0; i < n; ++i) {
    const double quotient = (cold_density[static_cast<std::size_t>(i)] -
                             hot_density[static_cast<std::size_t>(i)]) /
                            euler_dt;
    const double expected_rate =
        model.mobility * axis.apply(i, response_at, one, 0.0, &closed, &closed);
    max_euler_gap = std::max(max_euler_gap, std::abs(quotient - expected_rate));
    const double rate =
        coupling::substrate_pfc_rate(axis, i, density, cold, model, 0.0);
    REQUIRE(cold_density[static_cast<std::size_t>(i)] ==
            Approx(density[static_cast<std::size_t>(i)] + euler_dt * rate)
                .margin(1.0e-12));
    density_gap += axis.weight(i) * (cold_density[static_cast<std::size_t>(i)] -
                                     density[static_cast<std::size_t>(i)]);
  }
  report("euler", n, max_euler_gap, density_gap, cold_step.inventory_change,
         hot_step.inventory_change);
  REQUIRE(max_euler_gap < 1.0e-10);
  REQUIRE(std::abs(density_gap) < 1.0e-12);
  REQUIRE(std::abs(cold_step.inventory_change - density_gap) < 1.0e-12);
  REQUIRE(std::abs(hot_step.inventory_change) < 1.0e-12);

  std::vector<double> uniform_temperature(static_cast<std::size_t>(n), 1.7);
  FluxLedger uniform_ledger;
  const coupling::HeatStep uniform_step = coupling::advance_solved_heat(
      axis, uniform_temperature, heat, 0.0, 1.0e-3, uniform_ledger);
  REQUIRE(std::abs(uniform_step.inventory_change) < 1.0e-14);
  for (double sample : uniform_temperature)
    REQUIRE(sample == Approx(1.7).margin(1.0e-14));
  coupling::SubstrateDensityRequest constant_model = model;
  constant_model.temperature_scale = 0.4;
  constant_model.melting_level = 1.2;
  constant_model.cubic = 1.5;
  const std::vector<double> constant_density(static_cast<std::size_t>(n), 0.3);
  const double wave_sq = constant_model.wave_number * constant_model.wave_number;
  const double undercooling =
      constant_model.temperature_scale *
      (uniform_temperature[0] - constant_model.melting_level);
  const double expected_potential = (undercooling + wave_sq * wave_sq) * 0.3 +
                                    constant_model.cubic * 0.3 * 0.3 * 0.3;
  for (int i = 0; i < n; ++i) {
    const double potential = coupling::substrate_chemical_potential(
        axis, i, constant_density, uniform_temperature, constant_model, 0.0);
    REQUIRE(potential == Approx(expected_potential).margin(1.0e-12));
  }

  std::vector<double> temperature(static_cast<std::size_t>(n), 1.0);
  std::vector<double> coupled_density = cosine_line(n, h, 0.2, 0.05);
  const std::vector<double> temperature_before = temperature;
  const std::vector<double> density_before = coupled_density;
  coupling::SolvedHeatRequest wall_heat;
  wall_heat.diffusivity = 1.0;
  wall_heat.substrate_flux = 0.2;
  constexpr double couple_dt = 1.0e-4;
  std::vector<double> temperature_after = temperature;
  FluxLedger heat_only;
  const coupling::HeatStep heat_step = coupling::advance_solved_heat(
      axis, temperature_after, wall_heat, 0.0, couple_dt, heat_only);
  double max_temperature_change = 0.0;
  for (int i = 0; i < n; ++i) {
    max_temperature_change =
        std::max(max_temperature_change,
                 std::abs(temperature_after[static_cast<std::size_t>(i)] -
                          temperature_before[static_cast<std::size_t>(i)]));
  }
  REQUIRE(max_temperature_change > 1.0e-4);
  FluxLedger coupled_ledger;
  const coupling::OneWayStep coupled = coupling::advance_one_way_substrate(
      axis, temperature, coupled_density, wall_heat, model, 0.0, couple_dt,
      coupled_ledger);
  double max_pre_gap = 0.0;
  for (int i = 0; i < n; ++i) {
    const double post_rate = coupling::substrate_pfc_rate(
        axis, i, density_before, temperature_after, model, 0.0);
    const double pre_rate = coupling::substrate_pfc_rate(
        axis, i, density_before, temperature_before, model, 0.0);
    REQUIRE(temperature[static_cast<std::size_t>(i)] ==
            temperature_after[static_cast<std::size_t>(i)]);
    REQUIRE(
        coupled_density[static_cast<std::size_t>(i)] ==
        Approx(density_before[static_cast<std::size_t>(i)] + couple_dt * post_rate)
            .margin(1.0e-12));
    max_pre_gap =
        std::max(max_pre_gap, std::abs(couple_dt * (post_rate - pre_rate)));
  }
  const double coupled_mass =
      weighted_mass(axis, coupled_density) - weighted_mass(axis, density_before);
  report("one_way", n, max_pre_gap, coupled_mass, coupled.heat_inventory_change,
         coupled_ledger.accepted());
  REQUIRE(max_pre_gap > 1.0e-8);
  REQUIRE(coupled_ledger.accepted() ==
          Approx(wall_heat.substrate_flux * couple_dt).margin(1.0e-12));
  REQUIRE(coupled.heat_inventory_change ==
          Approx(-wall_heat.substrate_flux * couple_dt).margin(1.0e-12));
  REQUIRE(heat_step.inventory_change ==
          Approx(coupled.heat_inventory_change).margin(1.0e-12));
  REQUIRE(std::abs(coupled_mass) < 1.0e-12);
  REQUIRE(std::abs(coupled.density_inventory_change - coupled_mass) < 1.0e-12);

  std::vector<double> held_temperature = temperature_before;
  std::vector<double> held_density = density_before;
  FluxLedger held_ledger;
  coupling::SubstrateDensityRequest reservoir_model = model;
  reservoir_model.boundary = coupling::SolidificationBoundary::Reservoir;
  REQUIRE_THROWS_AS(coupling::advance_one_way_substrate(
                        axis, held_temperature, held_density, wall_heat,
                        reservoir_model, 0.0, couple_dt, held_ledger),
                    std::invalid_argument);
  reservoir_model.boundary = coupling::SolidificationBoundary::Periodic;
  REQUIRE_THROWS_AS(coupling::advance_one_way_substrate(
                        axis, held_temperature, held_density, wall_heat,
                        reservoir_model, 0.0, couple_dt, held_ledger),
                    std::invalid_argument);
  REQUIRE(held_temperature == temperature_before);
  REQUIRE(held_density == density_before);
  REQUIRE_FALSE(held_ledger.active());
  REQUIRE(held_ledger.accepted() == 0.0);

  coupling::SubstrateDensityRequest refused = model;
  refused.mobility = 0.0;
  std::vector<double> unchanged = density;
  REQUIRE_THROWS_AS(
      coupling::advance_substrate_pfc(axis, unchanged, cold, refused, 0.0, euler_dt),
      std::invalid_argument);
  refused = model;
  REQUIRE_THROWS_AS(
      coupling::advance_substrate_pfc(axis, unchanged, cold, refused, 0.0, 0.0),
      std::invalid_argument);
  REQUIRE(unchanged == density);
  DiffusionAxis periodic_axis(n, h, true);
  REQUIRE_THROWS_AS(coupling::advance_substrate_pfc(periodic_axis, unchanged, cold,
                                                    model, 0.0, euler_dt),
                    std::invalid_argument);
  REQUIRE(unchanged == density);
  held_ledger.begin();
  REQUIRE_THROWS_AS(coupling::advance_one_way_substrate(
                        axis, held_temperature, held_density, wall_heat, model, 0.0,
                        couple_dt, held_ledger),
                    std::logic_error);
  held_ledger.reject();
  REQUIRE(held_temperature == temperature_before);
  REQUIRE(held_density == density_before);
  REQUIRE_THROWS_AS(coupling::advance_one_way_substrate(
                        axis, held_temperature, held_temperature, wall_heat, model,
                        0.0, couple_dt, held_ledger),
                    std::invalid_argument);
  REQUIRE(held_temperature == temperature_before);
}

TEST_CASE("partially overlapping heat and substrate density are rejected",
          "[unit][field][thermal_pfc]") {
  DiffusionAxis axis(33, 1.0 / 32.0);
  std::vector<double> storage(34, 1.0);
  const auto before = storage;
  // T and n are shifted views of the same allocation.
  std::span<double> t(storage.data(), 33);
  std::span<double> n(storage.data() + 1, 33);
  coupling::SolvedHeatRequest h;
  coupling::SubstrateDensityRequest p;
  FluxLedger ledger;
  const auto advance = [&] {
    (void)coupling::advance_one_way_substrate(axis, t, n, h, p, 0.0, 1e-5, ledger);
  };
  REQUIRE_THROWS_AS(advance(), std::invalid_argument);
  REQUIRE(storage == before);
  REQUIRE_FALSE(ledger.active());
  REQUIRE(ledger.accepted() == 0.0);
}

TEST_CASE("Rejected solved-heat step preserves thermal field and flux ledger",
          "[unit][field][thermal_pfc]") {
  DiffusionAxis axis(33, 1.0 / 32.0);
  const double h = 1.0 / 32.0;
  std::vector<double> temperature = cosine_line(33, h, 0.5, 0.25);
  const auto before = temperature;
  coupling::SolvedHeatRequest request;
  request.diffusivity = 1.0e308;
  FluxLedger ledger;
  const auto unstable = [&] {
    (void)coupling::advance_solved_heat(axis, temperature, request, 0.0, 1.0,
                                        ledger);
  };
  REQUIRE_THROWS(unstable());
  REQUIRE(temperature == before);
  REQUIRE_FALSE(ledger.active());
  REQUIRE(ledger.accepted() == 0.0);
}
