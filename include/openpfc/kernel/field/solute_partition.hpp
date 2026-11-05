// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file solute_partition.hpp
 * @brief Conserved solute on one bounded axis with a prescribed partition.
 *
 * The contract is `docs/design/solute_partition.md`. The caller owns the
 * composition samples and a positive partition factor at each node of one
 * non-periodic `DiffusionAxis`. The factor is stationary. Coefficients are
 * code units. This header does not read the aluminum application.
 */

#include <openpfc/kernel/field/sbp_diffusion.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace pfc::field::coupling {

/// Constant mobility of one solute line. The partition factor is a field.
struct SolutePartitionRequest {
  double mobility{1.0};
};

/// Weighted composition change of one accepted step.
struct SoluteStep {
  double inventory_change{0.0};
};

/// μ = c / K. Both arguments are code units.
[[nodiscard]] inline double solute_chemical_potential(double composition,
                                                      double partition) {
  if (!std::isfinite(composition) || !(partition > 0.0) || !std::isfinite(partition))
    throw std::invalid_argument("solute potential needs a finite composition and "
                                "a positive partition factor");
  const double potential = composition / partition;
  if (!std::isfinite(potential))
    throw std::domain_error("nonfinite solute chemical potential");
  return potential;
}

namespace detail {

/// A fixed partition factor cannot alias the composition updated in-place.
inline bool overlapping_solute_partition(std::span<const double> a,
                                         std::span<const double> b) {
  if (a.empty() || b.empty()) return false;
  const auto less = std::less<const double *>{};
  const bool a_starts_before_b_end = less(a.data(), b.data() + b.size());
  const bool b_starts_before_a_end = less(b.data(), a.data() + a.size());
  return a_starts_before_b_end && b_starts_before_a_end;
}

inline pfc::field::fd::FaceCondition zero_solute_flux() {
  return {pfc::field::fd::BoundaryQuantity::ConstitutiveFlux,
          [](double) { return 0.0; }};
}

inline void validate_solute_partition(const pfc::field::fd::DiffusionAxis &axis,
                                      std::span<const double> composition,
                                      std::span<const double> partition,
                                      const SolutePartitionRequest &request,
                                      double time, double dt) {
  if (composition.size() != static_cast<std::size_t>(axis.size()) ||
      partition.size() != static_cast<std::size_t>(axis.size()))
    throw std::invalid_argument("solute line length does not match the axis");
  if (!(request.mobility > 0.0) || !std::isfinite(request.mobility))
    throw std::invalid_argument("solute mobility must be positive");
  if (!(dt > 0.0) || !std::isfinite(dt) || !std::isfinite(time))
    throw std::invalid_argument("solute time step must be positive");
  for (double sample : composition) {
    if (!std::isfinite(sample)) throw std::domain_error("nonfinite solute sample");
  }
  for (double factor : partition) {
    if (!(factor > 0.0) || !std::isfinite(factor))
      throw std::invalid_argument("solute partition factor must be positive");
  }
}

inline std::vector<double>
solute_potential_field(std::span<const double> composition,
                       std::span<const double> partition) {
  std::vector<double> potential(composition.size());
  for (std::size_t i = 0; i < composition.size(); ++i)
    potential[i] = solute_chemical_potential(composition[i], partition[i]);
  return potential;
}

} // namespace detail

/// Mobility times `DiffusionAxis::apply` of μ = c / K. Both ends are a zero
/// outward constitutive flux, so solute does not cross the ends.
[[nodiscard]] inline double
solute_partition_rate(const pfc::field::fd::DiffusionAxis &axis, int index,
                      std::span<const double> composition,
                      std::span<const double> partition,
                      const SolutePartitionRequest &request, double time) {
  if (index < 0 || index >= axis.size())
    throw std::out_of_range("solute partition rate index");
  detail::validate_solute_partition(axis, composition, partition, request, time,
                                    /*dt=*/1.0);
  const std::vector<double> potential =
      detail::solute_potential_field(composition, partition);
  const pfc::field::fd::FaceCondition closed = detail::zero_solute_flux();
  const auto mobility = [&](int) { return request.mobility; };
  const auto potential_at = [&](int j) {
    return potential[static_cast<std::size_t>(j)];
  };
  return axis.apply(index, potential_at, mobility, time, &closed, &closed);
}

/// RK4 for ∂t c = ∂x(M ∂x (c / K)) on a non-periodic axis. K is read and not
/// written. A failed trial rejects the ledger and leaves `composition`
/// unchanged. Validation throws before `ledger.begin()`.
[[nodiscard]] inline SoluteStep advance_solute_partition(
    const pfc::field::fd::DiffusionAxis &axis, std::span<double> composition,
    std::span<const double> partition, const SolutePartitionRequest &request,
    double time, double dt, pfc::field::fd::FluxLedger &ledger) {
  if (detail::overlapping_solute_partition(composition, partition))
    throw std::invalid_argument("composition and fixed partition must not overlap");
  detail::validate_solute_partition(axis, composition, partition, request, time, dt);
  const int n = axis.size();
  const pfc::field::fd::FaceCondition closed = detail::zero_solute_flux();
  const auto mobility = [&](int) { return request.mobility; };
  struct Stage {
    double time_fraction;
    double prior[3];
    double flux_weight;
  };
  const std::array<Stage, 4> stages{{
      {0.0, {0.0, 0.0, 0.0}, 1.0 / 6.0},
      {0.5, {0.5, 0.0, 0.0}, 1.0 / 3.0},
      {0.5, {0.0, 0.5, 0.0}, 1.0 / 3.0},
      {1.0, {0.0, 0.0, 1.0}, 1.0 / 6.0},
  }};
  std::vector<double> base(composition.begin(), composition.end());
  std::array<std::vector<double>, 4> slope;
  for (std::vector<double> &column : slope)
    column.assign(static_cast<std::size_t>(n), 0.0);

  ledger.begin();
  try {
    for (int stage = 0; stage < 4; ++stage) {
      const Stage &spec = stages[static_cast<std::size_t>(stage)];
      const double stage_time = time + spec.time_fraction * dt;
      std::vector<double> state = base;
      for (int i = 0; i < n; ++i) {
        double value = base[static_cast<std::size_t>(i)];
        for (int previous = 0; previous < stage; ++previous) {
          value +=
              dt * spec.prior[previous] *
              slope[static_cast<std::size_t>(previous)][static_cast<std::size_t>(i)];
        }
        state[static_cast<std::size_t>(i)] = value;
      }
      std::vector<double> potential(static_cast<std::size_t>(n));
      for (int i = 0; i < n; ++i) {
        potential[static_cast<std::size_t>(i)] =
            solute_chemical_potential(state[static_cast<std::size_t>(i)],
                                      partition[static_cast<std::size_t>(i)]);
      }
      const auto potential_at = [&](int j) {
        return potential[static_cast<std::size_t>(j)];
      };
      ledger.stage(closed.flux(stage_time, request.mobility), 1.0,
                   spec.flux_weight * dt);
      ledger.stage(closed.flux(stage_time, request.mobility), 1.0,
                   spec.flux_weight * dt);
      for (int i = 0; i < n; ++i) {
        slope[static_cast<std::size_t>(stage)][static_cast<std::size_t>(i)] =
            axis.apply(i, potential_at, mobility, stage_time, &closed, &closed);
      }
    }
    SoluteStep step;
    std::vector<double> next = base;
    for (int i = 0; i < n; ++i) {
      const std::size_t index = static_cast<std::size_t>(i);
      const double increment = dt *
                               (slope[0][index] + 2.0 * slope[1][index] +
                                2.0 * slope[2][index] + slope[3][index]) /
                               6.0;
      next[index] = base[index] + increment;
      if (!std::isfinite(next[index]))
        throw std::domain_error("nonfinite solute update");
      step.inventory_change += axis.weight(i) * increment;
    }
    ledger.accept();
    std::copy(next.begin(), next.end(), composition.begin());
    return step;
  } catch (...) {
    if (ledger.active()) ledger.reject();
    throw;
  }
}

} // namespace pfc::field::coupling
