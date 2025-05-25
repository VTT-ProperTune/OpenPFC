// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <pfc/diffop.hpp>
#include <pfc/fft.hpp>
#include <pfc/field.hpp>
#include <pfc/world.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace pfc;

TEST_CASE("Differentiate in all directions", "[SpectralDifferentiator]") {
  // Setup
  auto lo{-M_PI, -M_PI, -M_PI};
  auto hi{M_PI, M_PI, M_PI};
  auto size{128, 128, 128};
  auto world = world::create(lo, hi, size);
  auto fft = fft::create(world);
  auto diff = diffop::create(world, fft); // Creates spectral differentiator

  // Input function
  auto f = [](auto r) {
    return exp(sin(r[0])) * exp(sin(2 * r[1])) * exp(sin(3 * r[2]));
  };
  auto df_dx = [](auto r) {
    return exp(sin(r[0]) + sin(2 * r[1]) + sin(3 * r[2])) * cos(r[0]);
  };

  auto u = field::create(world, f);
  auto eps = 1e-9; // Tolerance

  // Apply spectral differentiation in x direction
  SECTION("Differentiation in x direction") {
    fft(diff, u);                                // forward FFT: F
    auto dx_op = get_operator(diff, diffop::dx); // ∂/∂x operator
    auto F = get_fft(diff);                      // F = forward FFT of u
    auto F_buf = get_fft_buffer(diff);           // F_buf = buffer for FFT results
    apply_operator(F_buf, F, dx_op);             // F_buf = ∂/∂x F
    auto du = similar(u);                        // du = ∂/∂x u
    ifft(diff, du, F_buf);                       // inverse FFT: du = ∂/∂x u
    auto true_du = field::create(world, df_dx);
    REQUIRE(isapprox(du, true_du, eps));
  }

  // Apply spectral differentiation in y direction using high-level interface
  SECTION("Differentiation in y direction") {
    fft(diff, u);                                // forward FFT: F
    auto dy_op = get_operator(diff, diffop::dy); // ∂/∂y operator
    auto du = differentiate(diff, dy_op);        // assumes F is already computed
    auto true_du = field::create(world, df_dy);
    REQUIRE(isapprox(du, true_du, eps));
  }

  SECTION("Differentiation in z direction") {
    auto du = differentiate(diff, u, diffop::dz); // calculates fft internally
    auto true_du = field::create(world, df_dz);
    REQUIRE(isapprox(du, true_du, eps));
  }

  SECTION("Second derivatives") {
    auto diffresult = differentiate(diff, u); // calculates fft internally + all ops
    REQUIRE(isapprox(dx2(diffresult), field::create(world, d2f_dx2)));
    REQUIRE(isapprox(dy2(diffresult), field::create(world, d2f_dy2)));
    REQUIRE(isapprox(dz2(diffresult), field::create(world, d2f_dz2)));
    REQUIRE(isapprox(laplace(diffresult), field::create(world, lap_f)));
  }

  SECTION("Differentiation in xy direction") {
    fft(diff, u);                                // forward FFT: F
    auto dx_op = get_operator(diff, diffop::dx); // ∂/∂x operator
    auto dy_op = get_operator(diff, diffop::dy); // ∂/∂y operator
    auto du = differentiate(diff, dx_op, dy_op); // assumes F is already computed
    auto true_du = field::create(world, d2f_dxdy);
    REQUIRE(isapprox(du, true_du));
  }
}
