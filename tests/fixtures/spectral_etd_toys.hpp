// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_etd_toys.hpp
 * @brief Toy spectral-ETD physics exercising each optional capability of
 *        `SpectralETDSystem`: plain (Swift–Hohenberg, see swift_hohenberg.hpp),
 *        mean-field (`filter_mf` + `nonlinear_symbol`), and moving-frame
 *        (`correlation_kernel` + coordinate/time dependence + free energy).
 *
 * Host and device tests share these so CPU-vs-GPU parity pins the same
 * physics on every backend.
 */

#include <cmath>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

#include <fixtures/spectral_etd_toys_pointwise.hpp>

namespace pfc::test {

/// PFC-like mean-field physics: \f$\partial_t\hat\psi = k(c_0\hat\psi + \hat N)\f$.
template <class MemorySpace = pfc::HostSpace> struct MeanFieldToy {
  pfc::Domain domain{};
  pfc::Box3i box{};
  double c0{0.85};
  double lambda2{0.0968};
  MeanFieldToyPointwise nl{};

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<double, MemorySpace>(state, "psi", domain, box, 0);
  }
  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    return k_laplacian * c0;
  }
  [[nodiscard]] double nonlinear_symbol(double k_laplacian) const {
    return k_laplacian;
  }
  [[nodiscard]] double filter_mf(double k_laplacian) const {
    return std::exp(k_laplacian / lambda2);
  }
  [[nodiscard]] MeanFieldToyPointwise pointwise() const { return nl; }
};

/// Moving-frame mean-field physics with a correlation kernel and observable.
template <class MemorySpace = pfc::HostSpace> struct MovingFrameToy {
  pfc::Domain domain{};
  pfc::Box3i box{};
  double c0{0.85};
  double lambda2{0.0968};
  double p_amp{0.2};
  MovingFrameToyPointwise nl{};

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<double, MemorySpace>(state, "psi", domain, box, 0);
  }
  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    return k_laplacian * c0;
  }
  [[nodiscard]] double nonlinear_symbol(double k_laplacian) const {
    return k_laplacian;
  }
  [[nodiscard]] double filter_mf(double k_laplacian) const {
    return std::exp(k_laplacian / lambda2);
  }
  [[nodiscard]] double correlation_kernel(double k_laplacian) const {
    return p_amp * std::exp(k_laplacian);
  }
  [[nodiscard]] MovingFrameToyPointwise pointwise() const { return nl; }
};

static_assert(pfc::sim::SpectralETDPhysics<MeanFieldToy<>>);
static_assert(pfc::sim::HasMeanFieldFilter<MeanFieldToy<>>);
static_assert(pfc::sim::HasNonlinearSymbol<MeanFieldToy<>>);
static_assert(!pfc::sim::HasCorrelationKernel<MeanFieldToy<>>);
static_assert(pfc::sim::SpectralETDPhysics<MovingFrameToy<>>);
static_assert(pfc::sim::HasCorrelationKernel<MovingFrameToy<>>);
static_assert(pfc::sim::HasFreeEnergyDensity<MovingFrameToyPointwise>);
static_assert(!pfc::sim::HasFreeEnergyDensity<MeanFieldToyPointwise>);

} // namespace pfc::test
