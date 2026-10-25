// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_decaying_hit.cpp
 * @brief Seed, solenoidality, and spectrum assignment on a small grid.
 *
 * The spatial and temporal ladders are not this test.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <optional>

#include <mpi.h>

#include <openpfc/kernel/fft/kspace_iterator.hpp>

#include <flow/decaying_hit.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

std::optional<flow::Complex> coefficient(const flow::State &state, int si, int sj,
                                         int sk, int component) {
  const auto outbox = state.stack->fft().get_outbox_bounds();
  std::optional<flow::Complex> found;
  pfc::fft::kspace::for_each_kpoint(
      outbox, state.n, state.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        if (flow::signed_index(i, state.n[0]) != si) return;
        if (flow::signed_index(j, state.n[1]) != sj) return;
        if (flow::signed_index(k, state.n[2]) != sk) return;
        if (component == 0) found = state.u[idx];
        if (component == 1) found = state.v[idx];
        if (component == 2) found = state.w[idx];
      });
  return found;
}

flow::Complex owned_mode(const flow::State &state, int si, int sj, int sk,
                         int component, int &owners) {
  const auto local = coefficient(state, si, sj, sk, component);
  owners = local.has_value() ? 1 : 0;
  double packed[2] = {local ? local->real() : 0.0, local ? local->imag() : 0.0};
  MPI_Allreduce(MPI_IN_PLACE, &owners, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, packed, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return flow::Complex(packed[0], packed[1]);
}

} // namespace

TEST_CASE("decaying HIT is a repeatable divergence-free spectrum", "[flow][hit]") {
  if (world_size() != 1) SKIP("one rank owns every Fourier mode");
  constexpr int n = 16;
  constexpr double nu = 0.02;
  constexpr double dt = 0.01;
  auto first = flow::make_state(n, nu, dt, 0, 1);
  auto second = flow::make_state(n, nu, dt, 0, 1);
  auto other = flow::make_state(n, nu, dt, 0, 1);
  flow::initialize_decaying_hit(first, 1);
  flow::initialize_decaying_hit(second, 1);
  flow::initialize_decaying_hit(other, 2);

  REQUIRE(first.u == second.u);
  REQUIRE(first.v == second.v);
  REQUIRE(first.w == second.w);
  REQUIRE(first.u != other.u);

  // k = (1, 0, 0) is longitudinal in the first component, so Leray zeros it.
  const auto mode = coefficient(first, 1, 0, 0, 1);
  const auto repeat = coefficient(second, 1, 0, 0, 1);
  const auto changed = coefficient(other, 1, 0, 0, 1);
  REQUIRE(mode.has_value());
  REQUIRE(std::abs(*mode) > 0.0);
  REQUIRE(*mode == *repeat);
  REQUIRE(*mode != *changed);

  const auto positive = coefficient(first, 0, 1, 0, 0);
  const auto negative = coefficient(first, 0, -1, 0, 0);
  REQUIRE(positive.has_value());
  REQUIRE(negative.has_value());
  REQUIRE(std::abs(*positive) > 0.0);
  REQUIRE(std::abs(*negative - std::conj(*positive)) < 1.0e-14);

  const auto removed = coefficient(first, 6, 0, 0, 0);
  REQUIRE(removed.has_value());
  REQUIRE(*removed == flow::Complex{});
  REQUIRE(*coefficient(first, 0, 0, 0, 0) == flow::Complex{});

  const auto initial = flow::diagnose(first);
  const auto scales = flow::measure_scales(first, initial);
  const double modal = flow::modal_kinetic_energy(first);
  REQUIRE(initial.finite);
  REQUIRE(initial.modal_div_max < 1.0e-8);
  REQUIRE(initial.div_l2 < 1.0e-10);
  REQUIRE(std::abs(initial.mean_u) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_v) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_w) < 1.0e-12);
  REQUIRE_THAT(modal, WithinRel(initial.ke, 1.0e-8));
  REQUIRE_THAT(scales.modal_ke, WithinRel(modal, 1.0e-14));
  REQUIRE(modal > 0.4);
  REQUIRE(modal < 0.7);
  REQUIRE(scales.k_max == 5);
  REQUIRE(scales.re_lambda > 0.0);
  REQUIRE(scales.k_max_eta > 0.0);

  auto fine = flow::make_state(32, nu, dt, 0, 1);
  flow::initialize_decaying_hit(fine, 1);
  const double n16 = static_cast<double>(n * n * n);
  const double n32 = static_cast<double>(32 * 32 * 32);
  const auto coarse_mode = coefficient(first, 1, 0, 0, 1);
  const auto fine_mode = coefficient(fine, 1, 0, 0, 1);
  REQUIRE(coarse_mode.has_value());
  REQUIRE(fine_mode.has_value());
  REQUIRE_THAT((*coarse_mode / n16).real(),
               WithinAbs((*fine_mode / n32).real(), 1.0e-12));
  REQUIRE_THAT((*coarse_mode / n16).imag(),
               WithinAbs((*fine_mode / n32).imag(), 1.0e-12));

  flow::step(first);
  const auto stepped = flow::diagnose(first);
  REQUIRE(stepped.finite);
  REQUIRE(stepped.modal_div_max < 1.0e-8);
  REQUIRE(stepped.ke < initial.ke - 1.0e-4);
  REQUIRE(flow::advance(first, 0) == "ok");
}

TEST_CASE("decaying HIT spectrum is the same on every pencil", "[flow][hit][mpi]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  constexpr int n = 16;
  constexpr double nu = 0.02;
  constexpr double dt = 0.01;
  auto state = flow::make_state(n, nu, dt, rank, nproc);
  flow::initialize_decaying_hit(state, 1);

  int longitudinal_owners = 0;
  const auto longitudinal = owned_mode(state, 1, 0, 0, 0, longitudinal_owners);
  REQUIRE(longitudinal_owners == 1);
  REQUIRE(longitudinal == flow::Complex{});

  int transverse_owners = 0;
  const auto transverse = owned_mode(state, 1, 0, 0, 1, transverse_owners);
  REQUIRE(transverse_owners == 1);
  REQUIRE(std::abs(transverse) > 0.0);

  int positive_owners = 0;
  int negative_owners = 0;
  const auto positive = owned_mode(state, 0, 1, 0, 0, positive_owners);
  const auto negative = owned_mode(state, 0, -1, 0, 0, negative_owners);
  REQUIRE(positive_owners == 1);
  REQUIRE(negative_owners == 1);
  REQUIRE(std::abs(negative - std::conj(positive)) < 1.0e-14);

  const auto initial = flow::diagnose(state);
  const auto scales = flow::measure_scales(state, initial);
  const double modal = flow::modal_kinetic_energy(state);
  REQUIRE(initial.finite);
  REQUIRE(initial.modal_div_max < 1.0e-8);
  REQUIRE(initial.div_l2 < 1.0e-10);
  REQUIRE_THAT(modal, WithinRel(initial.ke, 1.0e-8));
  REQUIRE(modal > 0.4);
  REQUIRE(modal < 0.7);
  REQUIRE(scales.k_max == 5);

  flow::step(state);
  const auto stepped = flow::diagnose(state);
  REQUIRE(stepped.finite);
  REQUIRE(stepped.modal_div_max < 1.0e-8);
  REQUIRE(stepped.ke < initial.ke);
}

TEST_CASE("modal divergence is normalized by N^3", "[flow][hit]") {
  if (world_size() != 1) SKIP("one rank owns every Fourier mode");
  constexpr int n = 16;
  constexpr double nu = 0.02;
  constexpr double dt = 0.01;
  auto state = flow::make_state(n, nu, dt, 0, 1);
  const auto outbox = state.stack->fft().get_outbox_bounds();
  // Amplitude of an energetic shell at N=1024. Raw |k·û| then sits
  // above 1e-8 after a double-precision Leray projection.
  constexpr double amplitude = 1.0e9;
  pfc::fft::kspace::for_each_kpoint(
      outbox, state.n, state.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        if (flow::signed_index(i, n) != 2) return;
        if (flow::signed_index(j, n) != 3) return;
        if (flow::signed_index(k, n) != 1) return;
        state.u[idx] = flow::Complex(amplitude, 0.3 * amplitude);
        state.v[idx] = flow::Complex(-0.4 * amplitude, 0.2 * amplitude);
        state.w[idx] = flow::Complex(0.7 * amplitude, -0.5 * amplitude);
      });
  const auto nhat = state.u.size();
  pfc::field::leray_project(outbox, state.n, state.spacing, state.u.data(),
                            state.v.data(), state.w.data(), nhat);
  const double raw = pfc::field::max_modal_divergence(outbox, state.n, state.spacing,
                                                      state.u.data(), state.v.data(),
                                                      state.w.data(), nhat);
  const auto diag = flow::diagnose(state);
  const double ncells =
      static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
  REQUIRE(raw > 1.0e-8);
  REQUIRE(diag.finite);
  REQUIRE(diag.modal_div_max < 1.0e-8);
  REQUIRE(diag.modal_div_max == raw / ncells);
}
