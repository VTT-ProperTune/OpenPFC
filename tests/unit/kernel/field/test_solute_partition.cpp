// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/field/solute_partition.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace coupling = pfc::field::coupling;
using pfc::field::fd::BoundaryQuantity;
using pfc::field::fd::DiffusionAxis;
using pfc::field::fd::FaceCondition;
using pfc::field::fd::FluxLedger;

namespace {

void report(const std::string &label, int n, double a, double b, double c,
            double d) {
  std::ostringstream out;
  out.setf(std::ios::scientific);
  out.precision(17);
  out << "SOLUTE_PARTITION " << label << " " << n << " " << a << " " << b << " " << c
      << " " << d;
  std::cout << out.str() << '\n';
}

double weighted_mass(const DiffusionAxis &axis, const std::vector<double> &field) {
  double mass = 0.0;
  for (int i = 0; i < axis.size(); ++i)
    mass += axis.weight(i) * field[static_cast<std::size_t>(i)];
  return mass;
}

double weight_sum(const DiffusionAxis &axis) {
  double sum = 0.0;
  for (int i = 0; i < axis.size(); ++i) sum += axis.weight(i);
  return sum;
}

/// Effective diffusivity on the solid side is mobility / k = 4. The heat
/// line used 20 stages per unit diffusivity, so this line uses 80.
int diffusion_steps(int intervals, double horizon) {
  return static_cast<int>(
      std::ceil(horizon * static_cast<double>(intervals) * intervals * 80.0));
}

std::vector<double> step_partition(const DiffusionAxis &axis, double h,
                                   double solid_factor) {
  std::vector<double> factor(static_cast<std::size_t>(axis.size()));
  for (int i = 0; i < axis.size(); ++i) {
    const double x = static_cast<double>(i) * h;
    factor[static_cast<std::size_t>(i)] = x <= 0.5 ? solid_factor : 1.0;
  }
  return factor;
}

std::vector<double> equilibrium_composition(const DiffusionAxis &axis,
                                            const std::vector<double> &partition,
                                            double mass) {
  double weighted_factor = 0.0;
  for (int i = 0; i < axis.size(); ++i)
    weighted_factor += axis.weight(i) * partition[static_cast<std::size_t>(i)];
  const double level = mass / weighted_factor;
  std::vector<double> composition(partition.size());
  for (std::size_t i = 0; i < partition.size(); ++i)
    composition[i] = level * partition[i];
  return composition;
}

double energy(const DiffusionAxis &axis, const std::vector<double> &composition,
              const std::vector<double> &partition) {
  double value = 0.0;
  for (int i = 0; i < axis.size(); ++i) {
    const std::size_t index = static_cast<std::size_t>(i);
    value +=
        axis.weight(i) * composition[index] * composition[index] / partition[index];
  }
  return 0.5 * value;
}

double region_mean(const DiffusionAxis &axis, const std::vector<double> &field,
                   double h, bool solid) {
  double mass = 0.0;
  double volume = 0.0;
  for (int i = 0; i < axis.size(); ++i) {
    const bool in_solid = static_cast<double>(i) * h <= 0.5;
    if (in_solid != solid) continue;
    const double weight = axis.weight(i);
    mass += weight * field[static_cast<std::size_t>(i)];
    volume += weight;
  }
  return mass / volume;
}

} // namespace

TEST_CASE("A constant solute potential is a stationary partition equilibrium",
          "[unit][field][solute_partition]") {
  const double solid_factor = 0.25;
  const double c0 = 0.8;
  const double horizon = 0.02;
  for (int intervals : {32, 64}) {
    const int n = intervals + 1;
    const double h = 1.0 / static_cast<double>(intervals);
    DiffusionAxis axis(n, h);
    const double volume = weight_sum(axis);
    REQUIRE(std::abs(volume - 1.0) < 1.0e-12);
    const std::vector<double> partition = step_partition(axis, h, solid_factor);
    const double mass = c0 * volume;
    std::vector<double> field = equilibrium_composition(axis, partition, mass);
    const std::vector<double> initial = field;
    coupling::SolutePartitionRequest request;
    request.mobility = 1.0;
    double max_rate = 0.0;
    double dissipation = 0.0;
    for (int i = 0; i < n; ++i) {
      const double potential = coupling::solute_chemical_potential(
          field[static_cast<std::size_t>(i)],
          partition[static_cast<std::size_t>(i)]);
      const double rate =
          coupling::solute_partition_rate(axis, i, field, partition, request, 0.0);
      const FaceCondition closed{BoundaryQuantity::ConstitutiveFlux,
                                 [](double) { return 0.0; }};
      const auto mobility = [&](int) { return request.mobility; };
      const auto potential_at = [&](int j) {
        return coupling::solute_chemical_potential(
            field[static_cast<std::size_t>(j)],
            partition[static_cast<std::size_t>(j)]);
      };
      const double applied =
          axis.apply(i, potential_at, mobility, 0.0, &closed, &closed);
      REQUIRE(std::abs(rate - applied) < 1.0e-12);
      max_rate = std::max(max_rate, std::abs(rate));
      dissipation += axis.weight(i) * potential * rate;
    }
    const int steps = diffusion_steps(intervals, horizon);
    const double dt = horizon / static_cast<double>(steps);
    FluxLedger ledger;
    double reported = 0.0;
    for (int step = 0; step < steps; ++step) {
      const coupling::SoluteStep advanced = coupling::advance_solute_partition(
          axis, field, partition, request, step * dt, dt, ledger);
      reported += advanced.inventory_change;
    }
    double drift = 0.0;
    for (int i = 0; i < n; ++i) {
      drift = std::max(drift, std::abs(field[static_cast<std::size_t>(i)] -
                                       initial[static_cast<std::size_t>(i)]));
    }
    const double mass_gap =
        weighted_mass(axis, field) - weighted_mass(axis, initial);
    report("equilibrium", n, drift, max_rate, dissipation, mass_gap);
    REQUIRE(drift < 1.0e-12);
    REQUIRE(max_rate < 1.0e-12);
    REQUIRE(std::abs(dissipation) < 1.0e-12);
    REQUIRE(std::abs(mass_gap) < 1.0e-12);
    REQUIRE(std::abs(reported - mass_gap) < 1.0e-12);
    REQUIRE(ledger.accepted() == 0.0);
  }

  DiffusionAxis axis(33, 1.0 / 32.0);
  const std::vector<double> partition =
      step_partition(axis, 1.0 / 32.0, solid_factor);
  std::vector<double> field(partition.size(), c0);
  const std::vector<double> saved = field;
  coupling::SolutePartitionRequest request;
  request.mobility = 1.0;
  FluxLedger ledger;
  std::vector<double> rejected = partition;
  rejected.front() = 0.0;
  REQUIRE_THROWS_AS(coupling::advance_solute_partition(axis, field, rejected,
                                                       request, 0.0, 1.0e-4, ledger),
                    std::invalid_argument);
  REQUIRE(field == saved);
  REQUIRE(!ledger.active());
  DiffusionAxis periodic(33, 1.0 / 32.0, true);
  REQUIRE_THROWS_AS(coupling::advance_solute_partition(periodic, field, partition,
                                                       request, 0.0, 1.0e-4, ledger),
                    std::invalid_argument);
  REQUIRE(field == saved);
  REQUIRE(!ledger.active());
  REQUIRE(ledger.accepted() == 0.0);
}

TEST_CASE("Uniform solute relaxes to the prescribed partition",
          "[unit][field][solute_partition]") {
  const double solid_factor = 0.25;
  const double c0 = 0.8;
  const double horizon = 1.0;
  for (int intervals : {32, 64}) {
    const int n = intervals + 1;
    const double h = 1.0 / static_cast<double>(intervals);
    DiffusionAxis axis(n, h);
    const std::vector<double> partition = step_partition(axis, h, solid_factor);
    std::vector<double> field(static_cast<std::size_t>(n), c0);
    const double mass = weighted_mass(axis, field);
    const std::vector<double> target =
        equilibrium_composition(axis, partition, mass);
    const double initial_energy = energy(axis, field, partition);
    const double target_energy = energy(axis, target, partition);
    coupling::SolutePartitionRequest request;
    request.mobility = 1.0;
    double initial_dissipation = 0.0;
    for (int i = 0; i < n; ++i) {
      const double potential = coupling::solute_chemical_potential(
          field[static_cast<std::size_t>(i)],
          partition[static_cast<std::size_t>(i)]);
      const double rate =
          coupling::solute_partition_rate(axis, i, field, partition, request, 0.0);
      initial_dissipation += axis.weight(i) * potential * rate;
    }
    REQUIRE(initial_dissipation < 0.0);
    REQUIRE(initial_energy > target_energy);
    const int steps = diffusion_steps(intervals, horizon);
    const double dt = horizon / static_cast<double>(steps);
    FluxLedger ledger;
    double reported = 0.0;
    for (int step = 0; step < steps; ++step) {
      const coupling::SoluteStep advanced = coupling::advance_solute_partition(
          axis, field, partition, request, step * dt, dt, ledger);
      reported += advanced.inventory_change;
    }
    double error = 0.0;
    for (int i = 0; i < n; ++i) {
      const double miss =
          field[static_cast<std::size_t>(i)] - target[static_cast<std::size_t>(i)];
      error += axis.weight(i) * miss * miss;
    }
    error = std::sqrt(error);
    const double mass_gap = weighted_mass(axis, field) - mass;
    const double solid_mean = region_mean(axis, field, h, true);
    const double liquid_mean = region_mean(axis, field, h, false);
    const double ratio = solid_mean / liquid_mean;
    const double final_energy = energy(axis, field, partition);
    report("relax", n, error, ratio, mass_gap, final_energy - target_energy);
    REQUIRE(error < 1.0e-8);
    REQUIRE(solid_mean < c0);
    REQUIRE(liquid_mean > c0);
    REQUIRE(std::abs(ratio - solid_factor) < 1.0e-8);
    REQUIRE(std::abs(mass_gap) < 1.0e-12);
    REQUIRE(std::abs(reported - mass_gap) < 1.0e-12);
    REQUIRE(ledger.accepted() == 0.0);
    REQUIRE(final_energy < initial_energy);
    REQUIRE(std::abs(final_energy - target_energy) < 1.0e-10);
  }
}

TEST_CASE("solute composition and fixed partition views must not overlap",
          "[unit][field][solute_partition]") {
  DiffusionAxis axis(33, 1.0 / 32.0);
  std::vector<double> storage(34, 1.0);
  const auto before = storage;
  std::span<double> c(storage.data(), 33);
  std::span<const double> k(storage.data() + 1, 33);
  coupling::SolutePartitionRequest p;
  FluxLedger ledger;
  const auto advance = [&] {
    (void)coupling::advance_solute_partition(axis, c, k, p, 0.0, 1e-5, ledger);
  };
  REQUIRE_THROWS_AS(advance(), std::invalid_argument);
  REQUIRE(storage == before);
  REQUIRE_FALSE(ledger.active());
  REQUIRE(ledger.accepted() == 0.0);
}

TEST_CASE("Rejected solute update does not publish composition or flux",
          "[unit][field][solute_partition]") {
  DiffusionAxis axis(33, 1.0 / 32.0);
  std::vector<double> composition(33);
  for (int i = 0; i < 33; ++i) {
    const double phase = std::acos(-1.0) * static_cast<double>(i) / 32.0;
    composition[static_cast<std::size_t>(i)] = 0.5 + 0.1 * std::cos(phase);
  }
  const auto before = composition;
  const std::vector<double> partition(33, 1.0);
  coupling::SolutePartitionRequest request;
  request.mobility = 1.0e308;
  FluxLedger ledger;
  const auto unstable = [&] {
    (void)coupling::advance_solute_partition(axis, composition, partition, request,
                                             0.0, 1.0, ledger);
  };
  REQUIRE_THROWS(unstable());
  REQUIRE(composition == before);
  REQUIRE_FALSE(ledger.active());
  REQUIRE(ledger.accepted() == 0.0);
}
