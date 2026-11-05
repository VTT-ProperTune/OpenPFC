// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_elastic_coupled_host.cpp
 * @brief Host coherent elastic step conserves composition and stores energy.
 */

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <cahn_hilliard/elastic_coupled_host.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

TEST_CASE("coherent elastic Cahn-Hilliard conserves mass on one periodic cell",
          "[cahn_hilliard][elastic_coupled]") {
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
  const auto report = cahn_hilliard::advance_coherent_elastic_host(
      domain, chemistry, stiffness, 0.05, chemistry.c0, initial, 0.02, 2, 0, 1,
      MPI_COMM_WORLD);

  REQUIRE(report.converged);
  REQUIRE(report.iterations >= 1);
  REQUIRE(report.composition.size() == initial.size());
  REQUIRE(std::abs(report.mass - report.mass0) < 1.0e-9);
  REQUIRE(std::isfinite(report.elastic_energy));
  REQUIRE(report.elastic_energy > 0.0);
  REQUIRE(std::isfinite(report.max_abs_mu_el));
  REQUIRE(report.max_abs_mu_el > 0.0);
}
