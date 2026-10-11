// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_crystal_plasticity.cpp
 * @brief Material-point ladder for the FCC slip update.
 *
 * Closed-form stresses are the cubic Saint-Venant response at zero
 * plastic velocity. They are written out here so a shared helper cannot
 * hide a transcription error in the constitutive update.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/mechanics/constitutive/crystal_plasticity.hpp>

#include "aluminum_slip.hpp"

#include <cmath>

using Catch::Matchers::WithinAbs;
using pfc::finite_strain::aluminum_slip_parameters;
using pfc::finite_strain::axpy;
using pfc::finite_strain::fcc_identity_state;
using pfc::finite_strain::fcc_tangent_action;
using pfc::finite_strain::FccSlipMaterial;
using pfc::finite_strain::FccSlipParameters;
using pfc::finite_strain::FccSlipState;
using pfc::finite_strain::FccSlipUpdate;
using pfc::finite_strain::frobenius_norm;
using pfc::finite_strain::identity2;
using pfc::finite_strain::integrate_fcc;
using pfc::finite_strain::matmul;
using pfc::finite_strain::scaled;
using pfc::finite_strain::Tensor2;
using pfc::finite_strain::transposed;

namespace {

const FccSlipParameters kAluminum = aluminum_slip_parameters();

Tensor2 shear(double gamma) {
  Tensor2 value = identity2();
  value(0, 1) = gamma;
  return value;
}

Tensor2 rotation_z(double cosine, double sine) {
  Tensor2 value{};
  value(0, 0) = cosine;
  value(0, 1) = sine;
  value(1, 0) = -sine;
  value(1, 1) = cosine;
  value(2, 2) = 1.0;
  return value;
}

Tensor2 cube_quarter_turn() { return rotation_z(0.0, 1.0); }

Tensor2 active_quarter_turn() { return rotation_z(0.0, -1.0); }

double relative_difference(const Tensor2 &value, const Tensor2 &reference) {
  return frobenius_norm(axpy(-1.0, reference, value)) /
         std::max(frobenius_norm(reference), 1.0);
}

bool same_state(const FccSlipState &left, const FccSlipState &right) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (left.plastic(i, j) != right.plastic(i, j) ||
          left.velocity(i, j) != right.velocity(i, j)) {
        return false;
      }
    }
  }
  for (int system = 0; system < 12; ++system) {
    if (left.resistance[system] != right.resistance[system] ||
        left.shear[system] != right.shear[system]) {
      return false;
    }
  }
  return true;
}

Tensor2 independent_cubic_piola(const Tensor2 &deformation) {
  Tensor2 metric = matmul(transposed(deformation), deformation);
  metric(0, 0) -= 1.0;
  metric(1, 1) -= 1.0;
  metric(2, 2) -= 1.0;
  const Tensor2 strain = scaled(metric, 0.5);
  Tensor2 second;
  second(0, 0) =
      kAluminum.c11 * strain(0, 0) + kAluminum.c12 * (strain(1, 1) + strain(2, 2));
  second(1, 1) = kAluminum.c12 * strain(0, 0) + kAluminum.c11 * strain(1, 1) +
                 kAluminum.c12 * strain(2, 2);
  second(2, 2) =
      kAluminum.c12 * (strain(0, 0) + strain(1, 1)) + kAluminum.c11 * strain(2, 2);
  second(0, 1) = 2.0 * kAluminum.c44 * strain(0, 1);
  second(1, 0) = second(0, 1);
  second(0, 2) = 2.0 * kAluminum.c44 * strain(0, 2);
  second(2, 0) = second(0, 2);
  second(1, 2) = 2.0 * kAluminum.c44 * strain(1, 2);
  second(2, 1) = second(1, 2);
  return matmul(deformation, second);
}

Tensor2 first_schmid() {
  const double direction_scale = std::sqrt(2.0);
  const double normal_scale = std::sqrt(3.0);
  const double direction[3] = {0.0, 1.0 / direction_scale, -1.0 / direction_scale};
  const double normal[3] = {1.0 / normal_scale, 1.0 / normal_scale,
                            1.0 / normal_scale};
  Tensor2 schmid{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      schmid(i, j) = direction[i] * normal[j];
    }
  }
  return schmid;
}

double maximum_rate(const FccSlipUpdate &update) {
  double largest = 0.0;
  for (double rate : update.shear_rate) {
    largest = std::max(largest, std::abs(rate));
  }
  return largest;
}

double accumulated_shear(const FccSlipState &state) {
  double sum = 0.0;
  for (double shear_value : state.shear) {
    sum += shear_value;
  }
  return sum;
}

Tensor2 finite_difference(const FccSlipState &state, const Tensor2 &deformation,
                          const Tensor2 &direction) {
  constexpr double step = 1e-7;
  const Tensor2 forward = integrate_fcc(kAluminum, identity2(), state,
                                        axpy(step, direction, deformation))
                              .piola;
  const Tensor2 backward = integrate_fcc(kAluminum, identity2(), state,
                                         axpy(-step, direction, deformation))
                               .piola;
  return scaled(axpy(-1.0, backward, forward), 1.0 / (2.0 * step));
}

} // namespace

TEST_CASE("identity deformation stores zero FCC stress",
          "[finite_strain][crystal]") {
  const FccSlipState initial = fcc_identity_state(kAluminum);
  const auto update = integrate_fcc(kAluminum, identity2(), initial, identity2());
  REQUIRE_THAT(frobenius_norm(update.piola), WithinAbs(0.0, 0.0));
  REQUIRE(same_state(update.state, initial));
  for (int system = 0; system < 12; ++system) {
    REQUIRE_THAT(update.tau[system], WithinAbs(0.0, 0.0));
    REQUIRE_THAT(update.shear_rate[system], WithinAbs(0.0, 0.0));
  }
}

TEST_CASE("a small shear matches the cubic Piola stress",
          "[finite_strain][crystal]") {
  const Tensor2 deformation = shear(1e-5);
  const auto update = integrate_fcc(kAluminum, identity2(),
                                    fcc_identity_state(kAluminum), deformation);
  REQUIRE(relative_difference(update.piola, independent_cubic_piola(deformation)) <
          1e-8);
  REQUIRE(maximum_rate(update) < 1e-12);
  REQUIRE(accumulated_shear(update.state) < 1e-12);
}

TEST_CASE("system 1 leads when its Schmid tensor is extended",
          "[finite_strain][crystal]") {
  const Tensor2 schmid = first_schmid();
  const Tensor2 deformation = axpy(1e-4, schmid, identity2());
  const auto update = integrate_fcc(kAluminum, identity2(),
                                    fcc_identity_state(kAluminum), deformation);
  const Tensor2 elastic = independent_cubic_piola(deformation);
  REQUIRE(relative_difference(update.piola, elastic) < 1e-8);
  int leader = 0;
  for (int system = 1; system < 12; ++system) {
    if (std::abs(update.tau[system]) > std::abs(update.tau[leader])) {
      leader = system;
    }
  }
  REQUIRE(leader == 0);
  for (int system = 1; system < 12; ++system) {
    REQUIRE(std::abs(update.tau[0]) > std::abs(update.tau[system]));
  }
  Tensor2 metric = matmul(transposed(deformation), deformation);
  metric(0, 0) -= 1.0;
  metric(1, 1) -= 1.0;
  metric(2, 2) -= 1.0;
  const Tensor2 strain = scaled(metric, 0.5);
  Tensor2 second;
  second(0, 0) =
      kAluminum.c11 * strain(0, 0) + kAluminum.c12 * (strain(1, 1) + strain(2, 2));
  second(1, 1) = kAluminum.c12 * strain(0, 0) + kAluminum.c11 * strain(1, 1) +
                 kAluminum.c12 * strain(2, 2);
  second(2, 2) =
      kAluminum.c12 * (strain(0, 0) + strain(1, 1)) + kAluminum.c11 * strain(2, 2);
  second(0, 1) = 2.0 * kAluminum.c44 * strain(0, 1);
  second(1, 0) = second(0, 1);
  second(0, 2) = 2.0 * kAluminum.c44 * strain(0, 2);
  second(2, 0) = second(0, 2);
  second(1, 2) = 2.0 * kAluminum.c44 * strain(1, 2);
  second(2, 1) = second(1, 2);
  double resolved = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      resolved += schmid(i, j) * second(i, j);
    }
  }
  REQUIRE(relative_difference(update.piola, elastic) < 1e-8);
  REQUIRE(std::abs(update.tau[0] - resolved) / std::abs(resolved) < 1e-8);
}

TEST_CASE("a cube rotation leaves the sample stress unchanged",
          "[finite_strain][crystal]") {
  const Tensor2 deformation = shear(0.004);
  const FccSlipState initial = fcc_identity_state(kAluminum);
  const auto aligned = integrate_fcc(kAluminum, identity2(), initial, deformation);
  const auto turned =
      integrate_fcc(kAluminum, cube_quarter_turn(), initial, deformation);
  REQUIRE(relative_difference(turned.piola, aligned.piola) < 1e-8);

  const double half = std::sqrt(0.5);
  const Tensor2 forty_five = rotation_z(half, half);
  const auto rotated = integrate_fcc(kAluminum, forty_five, initial, deformation);
  REQUIRE(relative_difference(rotated.piola, aligned.piola) > 1e-2);
}

TEST_CASE("a spatial rotation of a slipped state rotates the stress",
          "[finite_strain][crystal]") {
  const FccSlipState initial = fcc_identity_state(kAluminum);
  const Tensor2 deformation = shear(0.004);
  const auto update = integrate_fcc(kAluminum, identity2(), initial, deformation);
  const Tensor2 rotation = active_quarter_turn();
  const auto rotated =
      integrate_fcc(kAluminum, identity2(), initial, matmul(rotation, deformation));
  REQUIRE(relative_difference(rotated.piola, matmul(rotation, update.piola)) < 1e-8);
  double dissipation = 0.0;
  for (int system = 0; system < 12; ++system) {
    dissipation += update.tau[system] * update.shear_rate[system];
  }
  REQUIRE(dissipation > 0.0);
}

TEST_CASE("monotonic shear hardens and unloading does not reverse slip",
          "[finite_strain][crystal]") {
  FccSlipState state = fcc_identity_state(kAluminum);
  double previous_shear = 0.0;
  double previous_resistance = kAluminum.initial_resistance;
  Tensor2 loaded{};
  for (int step = 1; step <= 6; ++step) {
    const auto update =
        integrate_fcc(kAluminum, identity2(), state, shear(0.001 * step));
    const double shear_sum = accumulated_shear(update.state);
    double resistance = update.state.resistance[0];
    for (int system = 1; system < 12; ++system) {
      resistance = std::max(resistance, update.state.resistance[system]);
    }
    REQUIRE(shear_sum + 1e-15 >= previous_shear);
    REQUIRE(resistance + 1e-6 >= previous_resistance);
    previous_shear = shear_sum;
    previous_resistance = resistance;
    state = update.state;
    loaded = update.piola;
  }
  REQUIRE(previous_shear > 0.0);
  REQUIRE(previous_resistance > kAluminum.initial_resistance);

  const auto unloaded = integrate_fcc(kAluminum, identity2(), state, shear(0.003));
  REQUIRE(frobenius_norm(unloaded.piola) < frobenius_norm(loaded));
  for (int system = 0; system < 12; ++system) {
    REQUIRE(unloaded.state.shear[system] + 1e-15 >= state.shear[system]);
  }
}

TEST_CASE("reject restores FCC history and a later reject keeps the accept",
          "[finite_strain][crystal]") {
  FccSlipMaterial material(1, kAluminum);
  const FccSlipState original = material.committed_state(0);
  material.begin_trial();
  material.stage(0, shear(0.004));
  REQUIRE(same_state(material.committed_state(0), original));
  const auto during_trial = material.stress(0, identity2());
  REQUIRE_THAT(frobenius_norm(during_trial), WithinAbs(0.0, 0.0));
  material.reject();
  REQUIRE(same_state(material.committed_state(0), original));

  material.begin_trial();
  material.stage(0, shear(0.004));
  material.accept();
  const FccSlipState accepted = material.committed_state(0);
  REQUIRE(accumulated_shear(accepted) > 0.0);
  REQUIRE_FALSE(same_state(accepted, original));

  material.begin_trial();
  material.stage(0, shear(0.008));
  material.reject();
  REQUIRE(same_state(material.committed_state(0), accepted));
}

TEST_CASE("the FCC tangent matches a central difference",
          "[finite_strain][crystal]") {
  const FccSlipState initial = fcc_identity_state(kAluminum);
  Tensor2 direction{};
  direction(0, 1) = 1.0;
  const Tensor2 elastic = shear(1e-5);
  const Tensor2 slipping = shear(0.004);
  const Tensor2 elastic_action =
      fcc_tangent_action(kAluminum, identity2(), initial, elastic, direction);
  const Tensor2 slipping_action =
      fcc_tangent_action(kAluminum, identity2(), initial, slipping, direction);
  REQUIRE(relative_difference(elastic_action, finite_difference(initial, elastic,
                                                                direction)) < 1e-5);
  REQUIRE(relative_difference(slipping_action, finite_difference(initial, slipping,
                                                                 direction)) < 1e-5);

  FccSlipMaterial material(1, kAluminum);
  REQUIRE(relative_difference(material.tangent_action(0, slipping, direction),
                              slipping_action) < 1e-12);
  material.begin_trial();
  material.stage(0, slipping);
  material.accept();
  REQUIRE(relative_difference(material.tangent_action(0, slipping, direction),
                              slipping_action) < 1e-12);
}

namespace {
template <int N> void check_recorded_elimination() {
  using namespace pfc::finite_strain::crystal;
  DenseFactors<N> factors;
  double original[N * N];
  for (int row = 0; row < N; ++row)
    for (int column = 0; column < N; ++column) {
      const int permuted = (row + 3) % N;
      original[row * N + column] = factors.entries[row * N + column] =
          (permuted == column ? 5.0 : 0.0) +
          .03 * std::sin(2.0 * permuted + .7 * column);
    }
  REQUIRE(factors.factor());
  int swaps = 0;
  for (int column = 0; column < N; ++column)
    swaps += factors.pivots[column] != column;
  REQUIRE(swaps > 1);
  for (int trial = 0; trial < N + 2; ++trial) {
    double rhs[N]{}, reference[N], matrix[N * N], expected[N];
    for (int column = 0; column < N; ++column)
      expected[column] = std::cos(.3 * column + trial);
    for (int row = 0; row < N; ++row) {
      for (int column = 0; column < N; ++column)
        rhs[row] += original[row * N + column] * expected[column];
      reference[row] = rhs[row];
    }
    std::copy(original, original + N * N, matrix);
    REQUIRE(solve_dense<N>(matrix, reference));
    factors.solve(rhs);
    for (int row = 0; row < N; ++row) {
      REQUIRE(rhs[row] == reference[row]);
      REQUIRE_THAT(rhs[row], WithinAbs(expected[row], 2e-14));
    }
  }
  std::fill(factors.entries, factors.entries + N * N, 0.0);
  REQUIRE_FALSE(factors.factor());
  factors.entries[0] = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_FALSE(factors.factor());
}
} // namespace

TEST_CASE("FCC tangent elimination replays pivoted nonsymmetric solves",
          "[finite_strain][fcc][factorization]") {
  check_recorded_elimination<9>();
  check_recorded_elimination<12>();
}
