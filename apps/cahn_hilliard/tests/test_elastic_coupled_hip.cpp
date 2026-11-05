// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_elastic_coupled_hip.cpp
 * @brief CPU vs HIP parity for the coherent elastic Cahn–Hilliard step.
 */

#include "../../../tests/fixtures/session.hpp"

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <cahn_hilliard/elastic_coupled_host.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>

namespace {

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  if (a.size() != b.size()) {
    m = std::max(m, 1.0);
  }
  return m;
}

} // namespace

TEST_CASE("HIP coherent elastic Cahn-Hilliard matches the host solve",
          "[cahn_hilliard][elastic_coupled][hip]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    SKIP("single-rank spectral check");
  }

  const pfc::Domain domain = pfc::domain::create(
      pfc::GridSize({8, 8, 8}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  cahn_hilliard::CahnHilliardParams chemistry;
  chemistry.c0 = 0.4;
  chemistry.recompute_derived();
  const auto initial =
      cahn_hilliard::cosine_composition(stack.u(), chemistry.c0, 0.02, 1);
  const auto stiffness = pfc::solvers::Stiffness::isotropic(1.5, 0.25);
  constexpr double eps0 = 0.05;
  constexpr double dt = 0.02;
  constexpr int steps = 2;

  const auto host = cahn_hilliard::advance_coherent_elastic_host(
      domain, chemistry, stiffness, eps0, chemistry.c0, initial, dt, steps, 0, 1,
      MPI_COMM_WORLD);
  const auto hip = cahn_hilliard::advance_coherent_elastic_hip(
      domain, chemistry, stiffness, eps0, chemistry.c0, initial, dt, steps, 0, 1,
      MPI_COMM_WORLD);

  const double mismatch = max_abs_diff(host.composition, hip.composition);
  std::printf("ELASTIC_CH_HIP mass0=%.17g mass=%.17g energy=%.17g mu=%.17g "
              "iterations=%d residual=%.17g host_iterations=%d mismatches=%.17g\n",
              hip.mass0, hip.mass, hip.elastic_energy, hip.max_abs_mu_el,
              hip.iterations, hip.residual, host.iterations, mismatch);

  REQUIRE(host.converged);
  REQUIRE(hip.converged);
  REQUIRE(hip.iterations == host.iterations);
  REQUIRE(std::abs(hip.mass - hip.mass0) < 1.0e-9);
  REQUIRE(std::abs(host.mass - hip.mass) < 1.0e-9);
  REQUIRE(mismatch < 1.0e-8);
  const double energy_scale = std::max(1.0, std::abs(host.elastic_energy));
  REQUIRE(std::abs(hip.elastic_energy - host.elastic_energy) <
          1.0e-8 * energy_scale);
  const double mu_scale = std::max(1.0, std::abs(host.max_abs_mu_el));
  REQUIRE(std::abs(hip.max_abs_mu_el - host.max_abs_mu_el) < 1.0e-8 * mu_scale);
  REQUIRE(std::abs(hip.residual - host.residual) < 1.0e-8);
}

int main(int argc, char *argv[]) {
  pfc::runtime::gpu::bind_local_device_before_mpi();
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  const int rc = pfc::test::run(argc, argv);
  MPI_Finalize();
  return rc;
}
