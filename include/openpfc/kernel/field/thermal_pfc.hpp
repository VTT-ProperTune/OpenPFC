// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file thermal_pfc.hpp
 * @brief One-way solved heat on a substrate wall, read by a conserved PFC line.
 *
 * The contract is `docs/design/thermal_pfc.md`. The caller owns one temperature
 * sample and one density sample per node of a non-periodic `DiffusionAxis`.
 * Heat is advanced first, with the same step and no subcycling. The density
 * step reads that temperature array and does not write it. Coefficients are
 * the caller's code-unit numbers. This header has no latent heat and no alloy
 * data.
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

/// Periodic, reservoir, and substrate are different boundary meanings.
/// A free surface is not a value of this enum; call @ref reject_free_surface.
enum class SolidificationBoundary { Periodic, Reservoir, Substrate };

struct BoundaryMeaning {
  bool periodic{false};
  bool density_reservoir{false};
  bool physical_wall{false};
};

struct SolvedHeatRequest {
  double diffusivity{1.0};
  /// Outward constitutive heat flux at the first node. Positive removes heat.
  double substrate_flux{0.0};
  /// Outward constitutive heat flux at the last node.
  double far_flux{0.0};
};

struct HeatStep {
  double inventory_change{0.0};
};

struct SubstrateDensityRequest {
  SolidificationBoundary boundary{SolidificationBoundary::Substrate};
  double mobility{1.0};
  double wave_number{0.0};
  double cubic{0.0};
  double temperature_scale{0.0};
  double melting_level{0.0};
};

struct SubstrateDensityStep {
  double inventory_change{0.0};
};

struct OneWayStep {
  double heat_inventory_change{0.0};
  double density_inventory_change{0.0};
};

[[nodiscard]] inline BoundaryMeaning boundary_meaning(SolidificationBoundary kind) {
  switch (kind) {
  case SolidificationBoundary::Periodic: return {true, false, false};
  case SolidificationBoundary::Reservoir: return {false, true, false};
  case SolidificationBoundary::Substrate: return {false, false, true};
  }
  throw std::invalid_argument("unknown solidification boundary");
}

/// A free surface is not a periodic box, a density reservoir, or a substrate
/// wall. There is no free-surface enumerator to select.
[[noreturn]] inline void reject_free_surface() {
  throw std::invalid_argument(
      "a free surface is not a periodic box, a density reservoir, or a "
      "substrate wall");
}

namespace detail {

/// Two writable coupled fields must not overlap, even by a shifted view.
inline bool overlapping_heat_density(std::span<const double> a,
                                     std::span<const double> b) {
  if (a.empty() || b.empty()) return false;
  const auto less = std::less<const double *>{};
  const bool a_starts_before_b_end = less(a.data(), b.data() + b.size());
  const bool b_starts_before_a_end = less(b.data(), a.data() + a.size());
  return a_starts_before_b_end && b_starts_before_a_end;
}

inline pfc::field::fd::FaceCondition zero_derivative() {
  return {pfc::field::fd::BoundaryQuantity::NormalDerivative,
          [](double) { return 0.0; }};
}

inline pfc::field::fd::FaceCondition constitutive_flux(double flux) {
  return {pfc::field::fd::BoundaryQuantity::ConstitutiveFlux,
          [flux](double) { return flux; }};
}

inline void validate_solved_heat(const pfc::field::fd::DiffusionAxis &axis,
                                 std::span<const double> temperature,
                                 const SolvedHeatRequest &request, double time,
                                 double dt) {
  if (temperature.size() != static_cast<std::size_t>(axis.size()))
    throw std::invalid_argument("heat line length does not match the axis");
  if (!(request.diffusivity > 0.0) || !std::isfinite(request.diffusivity))
    throw std::invalid_argument("heat diffusivity must be positive");
  if (!std::isfinite(request.substrate_flux) || !std::isfinite(request.far_flux))
    throw std::invalid_argument("heat flux is non-finite");
  if (!(dt > 0.0) || !std::isfinite(dt) || !std::isfinite(time))
    throw std::invalid_argument("heat time step must be positive");
  for (double sample : temperature) {
    if (!std::isfinite(sample)) throw std::domain_error("nonfinite heat sample");
  }
}

inline void validate_substrate_fields(const pfc::field::fd::DiffusionAxis &axis,
                                      std::span<const double> density,
                                      std::span<const double> temperature,
                                      const SubstrateDensityRequest &request,
                                      double time) {
  if (request.boundary != SolidificationBoundary::Substrate)
    throw std::invalid_argument("substrate evolution requires a substrate wall");
  if (density.size() != static_cast<std::size_t>(axis.size()) ||
      temperature.size() != static_cast<std::size_t>(axis.size()))
    throw std::invalid_argument("substrate line length does not match the axis");
  if (!(request.mobility > 0.0) || !std::isfinite(request.mobility))
    throw std::invalid_argument("substrate mobility must be positive");
  if (!std::isfinite(request.wave_number) || !std::isfinite(request.cubic) ||
      !std::isfinite(request.temperature_scale) ||
      !std::isfinite(request.melting_level) || !std::isfinite(time))
    throw std::invalid_argument("substrate coefficient is non-finite");
  for (double sample : density) {
    if (!std::isfinite(sample))
      throw std::domain_error("nonfinite substrate density");
  }
  for (double sample : temperature) {
    if (!std::isfinite(sample))
      throw std::domain_error("nonfinite substrate temperature");
  }
}

inline void validate_substrate_step(const pfc::field::fd::DiffusionAxis &axis,
                                    std::span<const double> density,
                                    std::span<const double> temperature,
                                    const SubstrateDensityRequest &request,
                                    double time, double dt) {
  validate_substrate_fields(axis, density, temperature, request, time);
  if (!(dt > 0.0) || !std::isfinite(dt))
    throw std::invalid_argument("substrate time step must be positive");
}

/// μ = r(T) n + (q0² + ∂xx)² n + cubic n³, with r = scale (T − melting).
/// Both Laplacians use a zero outward derivative on the public axis.
inline std::vector<double>
chemical_potential_field(const pfc::field::fd::DiffusionAxis &axis,
                         std::span<const double> density,
                         std::span<const double> temperature,
                         const SubstrateDensityRequest &request, double time) {
  validate_substrate_fields(axis, density, temperature, request, time);
  const int n = axis.size();
  const pfc::field::fd::FaceCondition closed = zero_derivative();
  const auto one = [](int) { return 1.0; };
  const auto density_at = [&](int j) {
    return density[static_cast<std::size_t>(j)];
  };
  std::vector<double> first(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    first[static_cast<std::size_t>(i)] =
        axis.apply(i, density_at, one, time, &closed, &closed);
  }
  const auto first_at = [&](int j) { return first[static_cast<std::size_t>(j)]; };
  const double wave_sq = request.wave_number * request.wave_number;
  const double wave_fourth = wave_sq * wave_sq;
  if (!std::isfinite(wave_fourth))
    throw std::invalid_argument("substrate wave number overflows");
  std::vector<double> potential(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double second = axis.apply(i, first_at, one, time, &closed, &closed);
    const std::size_t index = static_cast<std::size_t>(i);
    const double value = density[index];
    const double response =
        request.temperature_scale * (temperature[index] - request.melting_level);
    const double mu = (response + wave_fourth) * value +
                      2.0 * wave_sq * first[index] + second +
                      request.cubic * value * value * value;
    if (!std::isfinite(mu))
      throw std::domain_error("nonfinite substrate chemical potential");
    potential[index] = mu;
  }
  return potential;
}

} // namespace detail

[[nodiscard]] inline double
substrate_chemical_potential(const pfc::field::fd::DiffusionAxis &axis, int index,
                             std::span<const double> density,
                             std::span<const double> temperature,
                             const SubstrateDensityRequest &request, double time) {
  if (index < 0 || index >= axis.size())
    throw std::out_of_range("substrate chemical potential index");
  const std::vector<double> potential =
      detail::chemical_potential_field(axis, density, temperature, request, time);
  return potential[static_cast<std::size_t>(index)];
}

/// Mobility times `DiffusionAxis::apply` of the chemical potential. The faces
/// are a zero outward derivative, so this is not a density reservoir.
[[nodiscard]] inline double
substrate_pfc_rate(const pfc::field::fd::DiffusionAxis &axis, int index,
                   std::span<const double> density,
                   std::span<const double> temperature,
                   const SubstrateDensityRequest &request, double time) {
  if (index < 0 || index >= axis.size())
    throw std::out_of_range("substrate rate index");
  const std::vector<double> potential =
      detail::chemical_potential_field(axis, density, temperature, request, time);
  const pfc::field::fd::FaceCondition closed = detail::zero_derivative();
  const auto one = [](int) { return 1.0; };
  const auto potential_at = [&](int j) {
    return potential[static_cast<std::size_t>(j)];
  };
  return request.mobility *
         axis.apply(index, potential_at, one, time, &closed, &closed);
}

/// RK4 for ∂t T = ∂x(α ∂x T) on a non-periodic axis. The substrate flux and the
/// far flux are constitutive `FaceCondition` values. A failed trial rejects
/// the ledger and leaves `temperature` unchanged. Validation throws before
/// `ledger.begin()`.
[[nodiscard]] inline HeatStep
advance_solved_heat(const pfc::field::fd::DiffusionAxis &axis,
                    std::span<double> temperature, const SolvedHeatRequest &request,
                    double time, double dt, pfc::field::fd::FluxLedger &ledger) {
  detail::validate_solved_heat(axis, temperature, request, time, dt);
  const int n = axis.size();
  const pfc::field::fd::FaceCondition substrate =
      detail::constitutive_flux(request.substrate_flux);
  const pfc::field::fd::FaceCondition far =
      detail::constitutive_flux(request.far_flux);
  const auto material = [&](int) { return request.diffusivity; };
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
  std::vector<double> base(temperature.begin(), temperature.end());
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
      const auto state_at = [&](int j) {
        return state[static_cast<std::size_t>(j)];
      };
      ledger.stage(substrate.flux(stage_time, request.diffusivity), 1.0,
                   spec.flux_weight * dt);
      ledger.stage(far.flux(stage_time, request.diffusivity), 1.0,
                   spec.flux_weight * dt);
      for (int i = 0; i < n; ++i) {
        slope[static_cast<std::size_t>(stage)][static_cast<std::size_t>(i)] =
            axis.apply(i, state_at, material, stage_time, &substrate, &far);
      }
    }
    HeatStep step;
    std::vector<double> next = base;
    for (int i = 0; i < n; ++i) {
      const std::size_t index = static_cast<std::size_t>(i);
      const double increment = dt *
                               (slope[0][index] + 2.0 * slope[1][index] +
                                2.0 * slope[2][index] + slope[3][index]) /
                               6.0;
      next[index] = base[index] + increment;
      if (!std::isfinite(next[index]))
        throw std::domain_error("nonfinite heat update");
      step.inventory_change += axis.weight(i) * increment;
    }
    ledger.accept();
    std::copy(next.begin(), next.end(), temperature.begin());
    return step;
  } catch (...) {
    if (ledger.active()) ledger.reject();
    throw;
  }
}

/// Forward Euler of the substrate density rate. The temperature array is read
/// and not written. A periodic axis is rejected by `DiffusionAxis` when the
/// wall condition is supplied. The span is written only after the update is
/// finite.
[[nodiscard]] inline SubstrateDensityStep
advance_substrate_pfc(const pfc::field::fd::DiffusionAxis &axis,
                      std::span<double> density, std::span<const double> temperature,
                      const SubstrateDensityRequest &request, double time,
                      double dt) {
  detail::validate_substrate_step(axis, density, temperature, request, time, dt);
  const std::vector<double> potential =
      detail::chemical_potential_field(axis, density, temperature, request, time);
  const int n = axis.size();
  const pfc::field::fd::FaceCondition closed = detail::zero_derivative();
  const auto one = [](int) { return 1.0; };
  const auto potential_at = [&](int j) {
    return potential[static_cast<std::size_t>(j)];
  };
  std::vector<double> next(density.begin(), density.end());
  SubstrateDensityStep step;
  for (int i = 0; i < n; ++i) {
    const double rate =
        request.mobility * axis.apply(i, potential_at, one, time, &closed, &closed);
    const std::size_t index = static_cast<std::size_t>(i);
    const double updated = density[index] + dt * rate;
    if (!std::isfinite(rate) || !std::isfinite(updated))
      throw std::domain_error("nonfinite substrate update");
    next[index] = updated;
    step.inventory_change += axis.weight(i) * (updated - density[index]);
  }
  std::copy(next.begin(), next.end(), density.begin());
  return step;
}

/// Heat first, then one density step on the updated temperature. The caller
/// ledger receives the heat trial's outward integral only after both updates
/// exist. A failure leaves the caller's arrays and ledger unchanged.
[[nodiscard]] inline OneWayStep advance_one_way_substrate(
    const pfc::field::fd::DiffusionAxis &axis, std::span<double> temperature,
    std::span<double> density, const SolvedHeatRequest &heat_request,
    const SubstrateDensityRequest &density_request, double time, double dt,
    pfc::field::fd::FluxLedger &ledger) {
  if (ledger.active()) throw std::logic_error("heat ledger is already active");
  if (detail::overlapping_heat_density(temperature, density))
    throw std::invalid_argument("temperature and density views must not overlap");
  detail::validate_solved_heat(axis, temperature, heat_request, time, dt);
  detail::validate_substrate_step(axis, density, temperature, density_request, time,
                                  dt);
  std::vector<double> next_temperature(temperature.begin(), temperature.end());
  std::vector<double> next_density(density.begin(), density.end());
  pfc::field::fd::FluxLedger scratch;
  const HeatStep heat =
      advance_solved_heat(axis, next_temperature, heat_request, time, dt, scratch);
  const SubstrateDensityStep evolved = advance_substrate_pfc(
      axis, next_density, next_temperature, density_request, time, dt);
  ledger.begin();
  try {
    ledger.stage(scratch.accepted(), 1.0, 1.0);
    ledger.accept();
  } catch (...) {
    if (ledger.active()) ledger.reject();
    throw;
  }
  std::copy(next_temperature.begin(), next_temperature.end(), temperature.begin());
  std::copy(next_density.begin(), next_density.end(), density.begin());
  return {heat.inventory_change, evolved.inventory_change};
}

} // namespace pfc::field::coupling
