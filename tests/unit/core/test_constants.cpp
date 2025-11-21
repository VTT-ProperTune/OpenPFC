// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <complex>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openpfc/constants.hpp"

using namespace pfc;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("constants - pi has correct value", "[constants][unit]") {
  SECTION("pi matches std::acos(-1.0)") {
    const double pi_reference = std::acos(-1.0);
    REQUIRE_THAT(constants::pi, WithinRel(pi_reference, 1e-15));
  }

  SECTION("pi accessible via pfc namespace") {
    REQUIRE_THAT(pfc::pi, WithinRel(constants::pi));
  }

  SECTION("pi has high precision (16+ digits)") {
    // Verify precision by checking that the constant differs from a less precise
    // value
    REQUIRE(constants::pi != 3.14159);
    REQUIRE(constants::pi != 3.141592653);
    // Verify it's between two precise bounds
    const double pi_lower = 3.14159265358979323;
    const double pi_upper = 3.14159265358979324;
    REQUIRE(constants::pi >= pi_lower);
    REQUIRE(constants::pi <= pi_upper);
  }
}

TEST_CASE("constants - derived pi constants", "[constants][unit]") {
  SECTION("two_pi equals 2*pi") {
    REQUIRE_THAT(constants::two_pi, WithinRel(2.0 * constants::pi, 1e-15));
  }

  SECTION("pi_2 equals pi/2") {
    REQUIRE_THAT(constants::pi_2, WithinRel(constants::pi / 2.0, 1e-15));
  }

  SECTION("pi_4 equals pi/4") {
    REQUIRE_THAT(constants::pi_4, WithinRel(constants::pi / 4.0, 1e-15));
  }

  SECTION("inv_pi equals 1/pi") {
    REQUIRE_THAT(constants::inv_pi, WithinRel(1.0 / constants::pi, 1e-15));
  }

  SECTION("sqrt_pi matches std::sqrt(pi)") {
    const double sqrt_pi_computed = std::sqrt(constants::pi);
    REQUIRE_THAT(constants::sqrt_pi, WithinRel(sqrt_pi_computed, 1e-15));
  }
}

TEST_CASE("constants - sqrt constants match std library", "[constants][unit]") {
  SECTION("sqrt2 matches std::sqrt(2.0)") {
    const double sqrt2_computed = std::sqrt(2.0);
    REQUIRE_THAT(constants::sqrt2, WithinRel(sqrt2_computed, 1e-15));
  }

  SECTION("sqrt3 matches std::sqrt(3.0)") {
    const double sqrt3_computed = std::sqrt(3.0);
    REQUIRE_THAT(constants::sqrt3, WithinRel(sqrt3_computed, 1e-15));
  }

  SECTION("sqrt2 accessible via pfc namespace") {
    REQUIRE_THAT(pfc::sqrt2, WithinRel(constants::sqrt2));
  }

  SECTION("sqrt3 accessible via pfc namespace") {
    REQUIRE_THAT(pfc::sqrt3, WithinRel(constants::sqrt3));
  }
}

TEST_CASE("constants - Euler's number", "[constants][unit]") {
  SECTION("e matches std::exp(1.0)") {
    const double e_computed = std::exp(1.0);
    REQUIRE_THAT(constants::e, WithinRel(e_computed, 1e-15));
  }

  SECTION("e accessible via pfc namespace") {
    REQUIRE_THAT(pfc::e, WithinRel(constants::e));
  }
}

TEST_CASE("constants - logarithmic constants", "[constants][unit]") {
  SECTION("ln2 matches std::log(2.0)") {
    const double ln2_computed = std::log(2.0);
    REQUIRE_THAT(constants::ln2, WithinRel(ln2_computed, 1e-15));
  }

  SECTION("ln10 matches std::log(10.0)") {
    const double ln10_computed = std::log(10.0);
    REQUIRE_THAT(constants::ln10, WithinRel(ln10_computed, 1e-15));
  }
}

TEST_CASE("constants - golden ratio", "[constants][unit]") {
  SECTION("phi equals (1 + sqrt(5)) / 2") {
    const double phi_computed = (1.0 + std::sqrt(5.0)) / 2.0;
    REQUIRE_THAT(constants::phi, WithinRel(phi_computed, 1e-15));
  }
}

TEST_CASE("constants - PFC lattice constants", "[constants][unit]") {
  SECTION("a1D equals 2*pi") {
    REQUIRE_THAT(constants::a1D, WithinRel(2.0 * constants::pi, 1e-15));
  }

  SECTION("a2D equals 4*pi/sqrt(3)") {
    const double a2D_computed = 4.0 * constants::pi / constants::sqrt3;
    REQUIRE_THAT(constants::a2D, WithinRel(a2D_computed, 1e-15));
  }

  SECTION("a3D equals 2*pi*sqrt(2)") {
    const double a3D_computed = 2.0 * constants::pi * constants::sqrt2;
    REQUIRE_THAT(constants::a3D, WithinRel(a3D_computed, 1e-15));
  }
}

TEST_CASE("constants - r2c_direction", "[constants][unit]") {
  SECTION("r2c_direction is zero") { REQUIRE(constants::r2c_direction == 0); }
}

TEST_CASE("constants - compile-time evaluation", "[constants][unit]") {
  SECTION("constants work in constexpr context") {
    // This must compile - proves constexpr works
    constexpr double compile_time_value = pfc::pi * 2.0;
    REQUIRE_THAT(compile_time_value, WithinRel(pfc::two_pi, 1e-15));
  }

  SECTION("can use constants in array size") {
    // This must compile - proves constexpr works at compile time
    constexpr int size = static_cast<int>(pfc::pi * 10);
    double array[size];
    REQUIRE(size == 31);
    (void)array; // Suppress unused variable warning
  }
}

TEST_CASE("constants - usage in FFT wave number calculation",
          "[constants][integration]") {
  SECTION("wave number formula k = 2π/L") {
    const double domain_length = 10.0;
    const double k = pfc::two_pi / domain_length;

    REQUIRE_THAT(k, WithinRel(2.0 * pfc::pi / domain_length, 1e-15));
    REQUIRE_THAT(k, WithinRel(0.628318530717958647692, 1e-14));
  }
}

TEST_CASE("constants - usage in Fourier phase calculation",
          "[constants][integration]") {
  SECTION("e^(2πi k·r) where k·r = 1/2 gives e^(πi) = -1") {
    const double k_dot_r = 0.5;
    const std::complex<double> phase =
        std::exp(std::complex<double>(0, pfc::two_pi * k_dot_r));

    // e^(πi) = -1 (Euler's identity)
    REQUIRE_THAT(phase.real(), WithinRel(-1.0, 1e-10));
    REQUIRE_THAT(phase.imag(), WithinAbs(0.0, 1e-10));
  }

  SECTION("e^(2πi) = 1 (full rotation)") {
    const std::complex<double> phase =
        std::exp(std::complex<double>(0, pfc::two_pi));

    REQUIRE_THAT(phase.real(), WithinRel(1.0, 1e-10));
    REQUIRE_THAT(phase.imag(), WithinAbs(0.0, 1e-10));
  }
}

TEST_CASE("constants - usage in crystal geometry", "[constants][integration]") {
  SECTION("FCC nearest neighbor distance") {
    const double lattice_constant = 4.0;
    const double nearest = lattice_constant / pfc::sqrt2;

    REQUIRE_THAT(nearest, WithinRel(2.82842712474619009760, 1e-14));
  }

  SECTION("hexagonal lattice height") {
    const double side_length = 2.0;
    const double height = side_length * pfc::sqrt3 / 2.0;

    REQUIRE_THAT(height, WithinRel(1.73205080756887729352, 1e-14));
  }
}

TEST_CASE("constants - namespace access patterns", "[constants][unit]") {
  SECTION("access via pfc::constants::pi") {
    REQUIRE_THAT(pfc::constants::pi, WithinRel(3.14159265358979323846, 1e-15));
  }

  SECTION("access via pfc::pi (using declaration)") {
    REQUIRE_THAT(pfc::pi, WithinRel(3.14159265358979323846, 1e-15));
  }

  SECTION("access via using namespace") {
    using namespace pfc::constants;
    REQUIRE_THAT(pi, WithinRel(3.14159265358979323846, 1e-15));
    REQUIRE_THAT(sqrt2, WithinRel(1.41421356237309504880, 1e-15));
    REQUIRE_THAT(sqrt3, WithinRel(1.73205080756887729352, 1e-15));
  }

  SECTION("access via individual using") {
    using pfc::pi;
    using pfc::sqrt2;
    REQUIRE_THAT(pi, WithinRel(3.14159265358979323846, 1e-15));
    REQUIRE_THAT(sqrt2, WithinRel(1.41421356237309504880, 1e-15));
  }
}
