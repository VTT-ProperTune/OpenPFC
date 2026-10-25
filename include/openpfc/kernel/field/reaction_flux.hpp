// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/kernel/field/sbp_diffusion.hpp>
#include <span>

namespace pfc::field::fd {
struct ReactionPoint {
  double state;
  std::span<const double> other_fields;
  std::array<double, 3> position;
  std::array<double, 3> outward_normal;
  double stage_time;
};
struct ReactionRate {
  double outward_flux;
  bool converged = true;
};

namespace detail {
template <class Params, class Law, class Admissible>
double evaluate_reaction(const ReactionPoint &point, const Params &params, Law law,
                         Admissible admissible, FluxLedger &ledger, bool cartesian) {
  if (!ledger.active()) throw std::logic_error("reaction needs active flux trial");
  try {
    if (!std::isfinite(point.state) || !std::isfinite(point.stage_time))
      throw std::domain_error("nonfinite reaction state or time");
    for (double v : point.other_fields)
      if (!std::isfinite(v)) throw std::domain_error("nonfinite reaction field");
    int normal_axes = 0;
    double norm_squared = 0;
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(point.position[axis]))
        throw std::domain_error("nonfinite reaction position");
      const double n = point.outward_normal[axis];
      if (!std::isfinite(n)) throw std::domain_error("nonfinite normal");
      norm_squared += n * n;
      if (n == 1 || n == -1)
        ++normal_axes;
      else if (cartesian && n != 0)
        throw std::domain_error("non-Cartesian face normal");
    }
    if ((cartesian && normal_axes != 1) || std::abs(norm_squared - 1) > 1e-12 ||
        !admissible(point, params))
      throw std::domain_error("inadmissible reaction point");
    const ReactionRate result = law(point, params);
    if (!result.converged || !std::isfinite(result.outward_flux))
      throw std::domain_error("reaction solve failed or rate is nonfinite");
    return result.outward_flux;
  } catch (...) {
    ledger.invalidate();
    throw;
  }
}
} // namespace detail

/// Evaluate once on the physical owner after preparing stage fields. Neither
/// law nor admissible may mutate accepted state or perform global collectives.
/// Caller records the returned flux once with actual stage weight and area.
template <class Params, class Law, class Admissible>
FaceCondition reaction_face(const ReactionPoint &point, const Params &params,
                            Law law, Admissible admissible, FluxLedger &ledger) {
  const double flux =
      detail::evaluate_reaction(point, params, law, admissible, ledger, true);
  return {BoundaryQuantity::ConstitutiveFlux, [flux](double) { return flux; }};
}

/// Local volumetric source from an application-supplied surface density.
/// Positive outward flux removes inventory. Call once per owned volume node
/// and RK stage, never on ghosts. volume_weight is the inventory quadrature;
/// time_weight includes dt and the RK weight. This function records the flux:
/// do not also call ledger.stage for it. No geometry or collective is hidden.
/// The application qualifies its surface density and normal extension.
template <class Params, class Law, class Admissible>
double reaction_source(const ReactionPoint &point, double surface_density,
                       double volume_weight, double time_weight,
                       const Params &params, Law law, Admissible admissible,
                       FluxLedger &ledger) {
  if (!ledger.active()) throw std::logic_error("reaction needs active flux trial");
  try {
    if (!std::isfinite(surface_density) || surface_density < 0 ||
        !std::isfinite(volume_weight) || volume_weight <= 0 ||
        !std::isfinite(time_weight))
      throw std::domain_error("invalid surface source quadrature");
    const double flux =
        detail::evaluate_reaction(point, params, law, admissible, ledger, false);
    const double source = -surface_density * flux;
    if (!std::isfinite(source)) throw std::overflow_error("source overflow");
    ledger.stage(flux, surface_density * volume_weight, time_weight);
    return source;
  } catch (...) {
    ledger.invalidate();
    throw;
  }
}
} // namespace pfc::field::fd
