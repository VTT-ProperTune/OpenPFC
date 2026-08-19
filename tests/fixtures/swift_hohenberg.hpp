// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file swift_hohenberg.hpp
 * @brief Toy Swift–Hohenberg physics: one header, all backends.
 *
 * ∂t ψ = [ε − (1+∇²)²] ψ − ψ³
 *
 * Point-wise: `rhs(t, SHGrads)` with value / lap / biharm.
 * Spectral ETD: `linear_symbol(k_lap)` and `nonlinearity(psi)`.
 * Parameters: `ParameterSchema`. No k-loops, no backend classes.
 */

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

namespace pfc::test {

struct SHParams {
  double epsilon{0.25};
};

struct SHGrads {
  double value{};
  double lap{};
  double biharm{};
};

struct SwiftHohenberg {
  using parameters_type = SHParams;

  Domain domain{};
  Box3i box{};
  SHParams params{};

  static pfc::sim::ParameterSchema<SHParams> schema() {
    pfc::sim::ParameterSchema<SHParams> s;
    s.model_name("SwiftHohenberg")
        .real(&SHParams::epsilon,
              pfc::sim::SchemaSpec{.name = "epsilon",
                                   .description = "reduced undercooling",
                                   .required = true,
                                   .min = 0.0,
                                   .max = 1.0,
                                   .typical = 0.25});
    return s;
  }

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<double>(state, "psi", domain, box, 0);
  }

  [[nodiscard]] double rhs(double /*t*/, const SHGrads &g) const {
    return (params.epsilon - 1.0) * g.value - 2.0 * g.lap - g.biharm +
           nonlinearity(g.value);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    const double one_plus_lap = 1.0 + k_laplacian;
    return params.epsilon - one_plus_lap * one_plus_lap;
  }

  [[nodiscard]] double nonlinearity(double psi) const {
    return -psi * psi * psi;
  }
};

static_assert(pfc::sim::HasParameters<SwiftHohenberg>);
static_assert(pfc::sim::DeclaresFields<SwiftHohenberg>);
static_assert(pfc::sim::PointwiseRhs<SwiftHohenberg, SHGrads>);
static_assert(pfc::sim::SpectralETDPhysics<SwiftHohenberg>);
static_assert(pfc::sim::HasParameterSchema<SwiftHohenberg>);

} // namespace pfc::test
