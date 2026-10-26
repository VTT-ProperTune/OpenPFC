// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_taylor_green.cpp
 * @brief Small-grid regression for the maintained Taylor–Green case.
 *
 * Thresholds match the analytic t = 0 cell averages. N = 16 does not
 * repeat the spatial or temporal ladder.
 */

#include "../../../tests/fixtures/session.hpp"

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>

#include <flow/decaying_hit.hpp>
#include <flow/taylor_green.hpp>

using Catch::Matchers::WithinAbs;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

void world(int &rank, int &nproc) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
}

} // namespace

TEST_CASE("Taylor-Green starts at the analytic energy and grows a third component",
          "[flow][taylor]") {
  if (world_size() != 1) SKIP("one rank owns every Fourier mode");
  constexpr double nu = 0.05;
  constexpr double dt = 0.01;
  constexpr int n = 16;
  auto state = flow::make_state(n, nu, dt, 0, 1);
  flow::initialize_taylor_green(state);
  const auto initial = flow::diagnose(state);

  REQUIRE(initial.finite);
  REQUIRE(initial.modal_div_max < 1.0e-8);
  REQUIRE(initial.div_linf < 1.0e-10);
  REQUIRE(initial.div_l2 < 1.0e-10);
  REQUIRE(initial.w_l2 < 1.0e-12);
  REQUIRE(initial.max_abs_w < 1.0e-12);
  REQUIRE_THAT(initial.ke, WithinAbs(flow::kinetic_energy_0, 1.0e-12));
  REQUIRE_THAT(initial.enstrophy, WithinAbs(flow::enstrophy_0, 1.0e-10));
  REQUIRE(std::abs(initial.mean_u) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_v) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_w) < 1.0e-12);
  REQUIRE_THAT(initial.cfl,
               WithinAbs(dt * static_cast<double>(n) / pfc::two_pi, 1.0e-12));

  flow::step(state);
  const auto stepped = flow::diagnose(state);
  REQUIRE(stepped.finite);
  REQUIRE(stepped.modal_div_max < 1.0e-8);
  REQUIRE(stepped.div_linf < 1.0e-8);
  REQUIRE(stepped.ke < flow::kinetic_energy_0 - 1.0e-8);
  REQUIRE(stepped.w_l2 > 1.0e-8);
  REQUIRE(flow::advance(state, 0) == "ok");
}

TEST_CASE("Taylor-Green energy is the same on every pencil", "[flow][taylor][mpi]") {
  int rank = 0;
  int nproc = 1;
  world(rank, nproc);
  constexpr double nu = 0.05;
  constexpr double dt = 0.01;
  constexpr int n = 16;
  auto state = flow::make_state(n, nu, dt, rank, nproc);
  flow::initialize_taylor_green(state);
  const auto initial = flow::diagnose(state);
  REQUIRE(initial.finite);
  REQUIRE(initial.modal_div_max < 1.0e-8);
  REQUIRE(initial.div_l2 < 1.0e-10);
  REQUIRE(initial.w_l2 < 1.0e-12);
  REQUIRE_THAT(initial.ke, WithinAbs(flow::kinetic_energy_0, 1.0e-12));
  REQUIRE_THAT(initial.enstrophy, WithinAbs(flow::enstrophy_0, 1.0e-10));
  REQUIRE(std::abs(initial.mean_u) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_v) < 1.0e-12);
  REQUIRE(std::abs(initial.mean_w) < 1.0e-12);

  flow::step(state);
  const auto stepped = flow::diagnose(state);
  REQUIRE(stepped.finite);
  REQUIRE(stepped.modal_div_max < 1.0e-8);
  REQUIRE(stepped.ke < flow::kinetic_energy_0 - 1.0e-8);
  REQUIRE(stepped.w_l2 > 1.0e-8);
}

TEST_CASE("Taylor-Green t=0 energy sits in shell 2", "[flow][taylor]") {
  if (world_size() != 1) SKIP("one rank owns every Fourier mode");
  constexpr double nu = 0.05;
  constexpr double dt = 0.01;
  constexpr int n = 16;
  auto state = flow::make_state(n, nu, dt, 0, 1);
  flow::initialize_taylor_green(state);
  const auto initial = flow::diagnose(state);
  const auto shells = flow::shell_energies(state);
  // The analytic modes sit at |k|=sqrt(3), so shell 2. A real-space
  // transform leaves roundoff in the rest of the retained band, and on
  // N=16 that band contains nine nonempty shells.
  const flow::Shell *shell2 = nullptr;
  double sum = 0.0;
  double other = 0.0;
  for (const auto &shell : shells) {
    sum += shell.ke;
    if (shell.index == 2) {
      shell2 = &shell;
    } else {
      other += shell.ke;
    }
  }
  REQUIRE(shell2 != nullptr);
  REQUIRE_THAT(shell2->ke, WithinAbs(flow::kinetic_energy_0, 1.0e-12));
  REQUIRE_THAT(sum, WithinAbs(initial.ke, 1.0e-12));
  REQUIRE(other < 1.0e-12);
  REQUIRE(flow::retained_k_max(n) == 5);
  REQUIRE_THAT(initial.dissipation, WithinAbs(nu * flow::enstrophy_0, 1.0e-12));
}

TEST_CASE("Taylor-Green sample stride records the final step", "[flow][taylor]") {
  REQUIRE(flow::record_at(10, 10, 0));
  REQUIRE_FALSE(flow::record_at(9, 10, 0));
  REQUIRE(flow::record_at(4, 10, 2));
  REQUIRE_FALSE(flow::record_at(3, 10, 2));
  REQUIRE(flow::record_at(10, 10, 2));
  REQUIRE(flow::steps_for(0.1, 0.001) == 100);
  REQUIRE_THROWS_AS(flow::steps_for(0.0003, 0.001), std::invalid_argument);
}

TEST_CASE("Taylor-Green t=0 vorticity cube matches the enstrophy",
          "[flow][taylor]") {
  int rank = 0;
  int nproc = 1;
  world(rank, nproc);
  constexpr double nu = 0.05;
  constexpr double dt = 0.01;
  constexpr int n = 16;
  auto state = flow::make_state(n, nu, dt, rank, nproc);
  flow::initialize_taylor_green(state);
  const auto path =
      std::filesystem::temp_directory_path() / "openpfc-tgv-vorticity.bin";
  if (rank == 0) std::filesystem::remove(path);
  MPI_Barrier(MPI_COMM_WORLD);
  const auto sample = flow::write_vorticity_magnitude(state, path.string());
  REQUIRE(sample.points == static_cast<long long>(n) * n * n);
  REQUIRE_THAT(sample.mean_square, WithinAbs(flow::enstrophy_0, 1.0e-8));
  if (rank == 0) {
    REQUIRE(std::filesystem::file_size(path) ==
            static_cast<std::uintmax_t>(sample.points) * sizeof(float));
    std::ifstream in(path, std::ios::binary);
    std::vector<float> cube(static_cast<std::size_t>(sample.points));
    in.read(reinterpret_cast<char *>(cube.data()),
            static_cast<std::streamsize>(cube.size() * sizeof(float)));
    REQUIRE(in);
    double sum = 0.0;
    for (float value : cube) sum += static_cast<double>(value) * value;
    REQUIRE_THAT(sum / static_cast<double>(sample.points),
                 WithinAbs(flow::enstrophy_0, 1.0e-6));
    std::filesystem::remove(path);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = pfc::test::run(argc, argv);
  MPI_Finalize();
  return result;
}
