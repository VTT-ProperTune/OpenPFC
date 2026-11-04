// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file bounded_axis.hpp
 * @brief One non-periodic linear Cahn–Hilliard axis on the public SBP faces.
 *
 * The line is a `DiffusionAxis` that is not periodic. The composition
 * gradient is a zero outward derivative. The composition flux is a
 * `reaction_face` on a caller-owned `FluxLedger`. Positive outward flux
 * removes inventory. The chemical potential on this path is the linear law
 * `α (c − c_eq) − κ Δc` with constant mobility. It is not the Fe–Cr
 * regular solution, not a grain boundary, and not a material free surface.
 * The periodic spectral session is a different function and is not called
 * from here.
 */

#include <openpfc/kernel/field/reaction_flux.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cahn_hilliard {

/// Coefficients of one explicit linear step. `outward_sink` and `exchange`
/// form the outward composition flux `outward_sink + exchange (c − c_eq)`.
struct BoundedAxisRequest {
  double mobility{1.0};
  double kappa{1.0e-4};
  double curvature{1.0};
  double outward_sink{0.0};
  double exchange{0.0};
  double equilibrium{0.0};
};

/// Owned-node change of the weighted composition over one accepted step.
struct BoundedAxisStep {
  double owned_inventory_change{0.0};
};

namespace detail {

template <class Read>
double closed_laplacian(const pfc::field::fd::DiffusionAxis &axis, int index,
                        Read read, double time,
                        const pfc::field::fd::FaceCondition &closed) {
  return axis.apply(index, read, [](int) { return 1.0; }, time, &closed, &closed);
}

inline void validate_bounded_axis(const pfc::field::fd::DiffusionAxis &axis,
                                  int owned_begin, int owned_end,
                                  const BoundedAxisRequest &request, double time,
                                  double dt) {
  if (owned_begin < 0 || owned_end < owned_begin || owned_end > axis.size())
    throw std::invalid_argument("bounded axis owns an index outside the grid");
  if (!(request.mobility > 0.0) || !std::isfinite(request.mobility))
    throw std::invalid_argument("bounded axis mobility must be positive");
  if (!(request.kappa >= 0.0) || !std::isfinite(request.kappa))
    throw std::invalid_argument(
        "bounded axis gradient coefficient must be non-negative");
  if (!std::isfinite(request.curvature) || !std::isfinite(request.outward_sink) ||
      !std::isfinite(request.exchange) || !std::isfinite(request.equilibrium))
    throw std::invalid_argument("bounded axis coefficient is non-finite");
  if (!(dt > 0.0) || !std::isfinite(dt) || !std::isfinite(time))
    throw std::invalid_argument("bounded axis time step must be positive");
}

} // namespace detail

/// Outward-derivative-free Laplacian at one node. The face data are the
/// public zero normal derivative, not an application-local ghost.
template <class Read>
double bounded_axis_laplacian(const pfc::field::fd::DiffusionAxis &axis, int index,
                              Read read, double time) {
  if (index < 0 || index >= axis.size())
    throw std::out_of_range("bounded axis laplacian index");
  if (!std::isfinite(time))
    throw std::invalid_argument("bounded axis laplacian time is non-finite");
  const pfc::field::fd::FaceCondition closed{
      pfc::field::fd::BoundaryQuantity::NormalDerivative,
      [](double) { return 0.0; }};
  return detail::closed_laplacian(axis, index, read, time, closed);
}

/// One RK4 step. `read` and `write` address the caller array. `exchange`
/// publishes owned samples before the residual reads a neighbour. Only
/// `[owned_begin, owned_end)` is written. Endpoint owners alone evaluate
/// and record the composition flux. A failed trial rejects the ledger.
template <class Read, class Write, class Exchange>
BoundedAxisStep advance_bounded_axis(const pfc::field::fd::DiffusionAxis &axis,
                                     int owned_begin, int owned_end, Read read,
                                     Write write, Exchange exchange,
                                     const BoundedAxisRequest &request, double time,
                                     double dt, pfc::field::fd::FluxLedger &ledger) {
  detail::validate_bounded_axis(axis, owned_begin, owned_end, request, time, dt);
  const int n = axis.size();
  std::vector<char> need_mu(static_cast<std::size_t>(n), 0);
  std::vector<char> need_c(static_cast<std::size_t>(n), 0);
  for (int i = owned_begin; i < owned_end; ++i) {
    need_mu[static_cast<std::size_t>(i)] = 1;
    axis.coefficients(
        i, [&](int j, int, double) { need_mu[static_cast<std::size_t>(j)] = 1; });
  }
  for (int i = 0; i < n; ++i) {
    if (need_mu[static_cast<std::size_t>(i)] == 0) continue;
    need_c[static_cast<std::size_t>(i)] = 1;
    axis.coefficients(
        i, [&](int j, int, double) { need_c[static_cast<std::size_t>(j)] = 1; });
  }

  std::vector<double> base(static_cast<std::size_t>(n), 0.0);
  for (int i = owned_begin; i < owned_end; ++i)
    base[static_cast<std::size_t>(i)] = read(i);

  struct Stage {
    double time_fraction;
    double prior[3];
    double flux_weight;
  };
  // Input state is base + dt * prior · (k1, k2, k3). Flux weights are the
  // RK4 quadrature of a boundary sample, and they sum to one.
  const std::array<Stage, 4> stages{{
      {0.0, {0.0, 0.0, 0.0}, 1.0 / 6.0},
      {0.5, {0.5, 0.0, 0.0}, 1.0 / 3.0},
      {0.5, {0.0, 0.5, 0.0}, 1.0 / 3.0},
      {1.0, {0.0, 0.0, 1.0}, 1.0 / 6.0},
  }};
  std::array<std::vector<double>, 4> slope;
  for (auto &column : slope) column.assign(static_cast<std::size_t>(n), 0.0);

  const pfc::field::fd::FaceCondition closed{
      pfc::field::fd::BoundaryQuantity::NormalDerivative,
      [](double) { return 0.0; }};
  auto admissible = [](const pfc::field::fd::ReactionPoint &,
                       const BoundedAxisRequest &) { return true; };
  auto law = [](const pfc::field::fd::ReactionPoint &point,
                const BoundedAxisRequest &model) {
    const double flux =
        model.outward_sink + model.exchange * (point.state - model.equilibrium);
    return pfc::field::fd::ReactionRate{flux, true};
  };

  ledger.begin();
  try {
    for (int stage = 0; stage < 4; ++stage) {
      const Stage &spec = stages[static_cast<std::size_t>(stage)];
      const double stage_time = time + spec.time_fraction * dt;
      for (int i = owned_begin; i < owned_end; ++i) {
        double value = base[static_cast<std::size_t>(i)];
        for (int previous = 0; previous < stage; ++previous)
          value +=
              dt * spec.prior[previous] *
              slope[static_cast<std::size_t>(previous)][static_cast<std::size_t>(i)];
        if (!std::isfinite(value))
          throw std::domain_error("bounded axis RK stage is non-finite");
        write(i, value);
      }
      exchange();
      std::vector<double> sample(static_cast<std::size_t>(n), 0.0);
      for (int i = 0; i < n; ++i) {
        if (need_c[static_cast<std::size_t>(i)] == 0) continue;
        sample[static_cast<std::size_t>(i)] = read(i);
      }
      auto composition = [&](int j) { return sample[static_cast<std::size_t>(j)]; };
      std::vector<double> chemical(static_cast<std::size_t>(n), 0.0);
      for (int i = 0; i < n; ++i) {
        if (need_mu[static_cast<std::size_t>(i)] == 0) continue;
        const double lap =
            detail::closed_laplacian(axis, i, composition, stage_time, closed);
        chemical[static_cast<std::size_t>(i)] =
            request.curvature * (composition(i) - request.equilibrium) -
            request.kappa * lap;
      }
      auto potential = [&](int j) { return chemical[static_cast<std::size_t>(j)]; };
      auto mobility = [&](int) { return request.mobility; };
      pfc::field::fd::FaceCondition left_face;
      pfc::field::fd::FaceCondition right_face;
      const pfc::field::fd::FaceCondition *left = nullptr;
      const pfc::field::fd::FaceCondition *right = nullptr;
      if (owned_begin == 0) {
        const pfc::field::fd::ReactionPoint point{
            composition(0), {}, {0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, stage_time};
        left_face =
            pfc::field::fd::reaction_face(point, request, law, admissible, ledger);
        left = &left_face;
        ledger.stage(left->flux(stage_time, request.mobility), 1.0,
                     spec.flux_weight * dt);
      }
      if (owned_end == n) {
        // Interior weights equal the grid spacing. N >= 17 puts n/2 outside
        // the four-node boundary closure.
        const double spacing = axis.weight(n / 2);
        const pfc::field::fd::ReactionPoint point{
            composition(n - 1),
            {},
            {static_cast<double>(n - 1) * spacing, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            stage_time};
        right_face =
            pfc::field::fd::reaction_face(point, request, law, admissible, ledger);
        right = &right_face;
        ledger.stage(right->flux(stage_time, request.mobility), 1.0,
                     spec.flux_weight * dt);
      }
      for (int i = owned_begin; i < owned_end; ++i) {
        slope[static_cast<std::size_t>(stage)][static_cast<std::size_t>(i)] =
            axis.apply(i, potential, mobility, stage_time, left, right);
      }
    }
    BoundedAxisStep step;
    for (int i = owned_begin; i < owned_end; ++i) {
      const double increment = dt *
                               (slope[0][static_cast<std::size_t>(i)] +
                                2.0 * slope[1][static_cast<std::size_t>(i)] +
                                2.0 * slope[2][static_cast<std::size_t>(i)] +
                                slope[3][static_cast<std::size_t>(i)]) /
                               6.0;
      const double value = base[static_cast<std::size_t>(i)] + increment;
      if (!std::isfinite(value))
        throw std::domain_error("bounded axis RK candidate is non-finite");
      write(i, value);
      step.owned_inventory_change +=
          axis.weight(i) * (value - base[static_cast<std::size_t>(i)]);
    }
    ledger.accept();
    return step;
  } catch (...) {
    // RK stages temporarily publish owned samples for the halo exchange.
    // Rejected trials must restore the *field* as well as the flux ledger.
    // Halo caches are caller-owned and refreshed by the next exchange.
    if (ledger.active()) ledger.reject();
    try {
      for (int i = owned_begin; i < owned_end; ++i)
        write(i, base[static_cast<std::size_t>(i)]);
    } catch (...) {
      throw std::runtime_error("bounded axis could not roll back owned state");
    }
    throw;
  }
}

} // namespace cahn_hilliard
