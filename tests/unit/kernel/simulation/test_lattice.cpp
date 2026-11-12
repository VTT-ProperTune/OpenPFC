// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cmath>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/simulation/initial_conditions/seed.hpp>
#include <openpfc/kernel/simulation/lattice.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("Lattice names round-trip and reject anything else", "[lattice]") {
  REQUIRE(pfc::lattice_from_string("bcc") == pfc::Lattice::bcc);
  REQUIRE(pfc::lattice_from_string("fcc") == pfc::Lattice::fcc);
  REQUIRE(pfc::to_string(pfc::Lattice::bcc) == "bcc");
  REQUIRE(pfc::to_string(pfc::Lattice::fcc) == "fcc");
  REQUIRE_THROWS_AS(pfc::lattice_from_string("hcp"), std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::lattice_from_string("BCC"), std::invalid_argument);
}

TEST_CASE("Lattice reciprocal vectors are the unit {110} and <111> sets",
          "[lattice]") {
  const auto bcc = pfc::reciprocal_vectors(pfc::Lattice::bcc);
  const auto fcc = pfc::reciprocal_vectors(pfc::Lattice::fcc);
  REQUIRE(bcc.size() == 6);
  REQUIRE(fcc.size() == 4);
  REQUIRE(bcc.size() <= pfc::kMaxReciprocalVectors);
  for (const auto &q : bcc) {
    REQUIRE_THAT(std::hypot(q[0], q[1], q[2]), WithinAbs(1.0, 1e-15));
    // {110}: exactly one zero component.
    int zeros = 0;
    for (double c : q) zeros += (c == 0.0) ? 1 : 0;
    REQUIRE(zeros == 1);
  }
  for (const auto &q : fcc) {
    REQUIRE_THAT(std::hypot(q[0], q[1], q[2]), WithinAbs(1.0, 1e-15));
    // <111>: all components equal in magnitude.
    REQUIRE_THAT(std::abs(q[0]), WithinAbs(std::abs(q[1]), 1e-15));
    REQUIRE_THAT(std::abs(q[1]), WithinAbs(std::abs(q[2]), 1e-15));
  }
  // The historical order of the BCC table is part of the pinned checksums.
  const double s = 1.0 / std::sqrt(2.0);
  REQUIRE_THAT(bcc[0][0], WithinAbs(s, 1e-15));
  REQUIRE_THAT(bcc[0][1], WithinAbs(s, 1e-15));
  REQUIRE_THAT(bcc[0][2], WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(bcc[5][2], WithinAbs(-s, 1e-15));
}

TEST_CASE("Lattice correlation shells: BCC one, FCC adds 2/sqrt(3)", "[lattice]") {
  const auto bcc = pfc::correlation_shells(pfc::Lattice::bcc);
  const auto fcc = pfc::correlation_shells(pfc::Lattice::fcc);
  REQUIRE(bcc.size() == 1);
  REQUIRE_THAT(bcc[0], WithinAbs(1.0, 1e-15));
  REQUIRE(fcc.size() == 2);
  REQUIRE_THAT(fcc[0], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(fcc[1], WithinAbs(2.0 / std::sqrt(3.0), 1e-15));
}

TEST_CASE("one_mode_density sums 2A cos(q.X) over the lattice vectors",
          "[lattice]") {
  const std::array<double, 3> origin{0.0, 0.0, 0.0};
  REQUIRE_THAT(pfc::one_mode_density(pfc::reciprocal_vectors(pfc::Lattice::bcc),
                                     origin, -0.4, 0.2),
               WithinAbs(-0.4 + 12.0 * 0.2, 1e-14));
  REQUIRE_THAT(pfc::one_mode_density(pfc::reciprocal_vectors(pfc::Lattice::fcc),
                                     origin, -0.4, 0.2),
               WithinAbs(-0.4 + 8.0 * 0.2, 1e-14));
}

TEST_CASE("Seed carries its lattice through the rotation", "[lattice]") {
  const std::array<double, 3> at{0.0, 0.0, 0.0};
  const std::array<double, 3> euler{0.3, -0.7, 1.1};
  const pfc::Seed bcc(at, euler, 10.0, -0.1, 0.25);
  const pfc::Seed fcc(at, euler, 10.0, -0.1, 0.25, pfc::Lattice::fcc);
  REQUIRE(bcc.get_lattice() == pfc::Lattice::bcc);
  REQUIRE(fcc.get_lattice() == pfc::Lattice::fcc);
  REQUIRE(bcc.reciprocal_vectors().size() == 6);
  REQUIRE(fcc.reciprocal_vectors().size() == 4);
  for (const auto &q : fcc.reciprocal_vectors()) {
    REQUIRE_THAT(std::hypot(q[0], q[1], q[2]), WithinAbs(1.0, 1e-14));
  }
  // At the seed centre every cosine is one, whatever the orientation.
  REQUIRE_THAT(bcc.get_value(at), WithinAbs(-0.1 + 12.0 * 0.25, 1e-13));
  REQUIRE_THAT(fcc.get_value(at), WithinAbs(-0.1 + 8.0 * 0.25, 1e-13));
}
